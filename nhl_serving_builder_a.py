#!/usr/bin/env python3
"""
NHL_SERVING_BUILDER_A

NHL's first live-serving market in this repo: real team-level moneyline
("who wins the game" outright), mirroring cfb_serving_builder_a.py's
MoneylineEngine design (frozen champion model + weekly growing-pool Platt
recalibration) with one deliberate simplification: NHL plays near-daily
(unlike CFB/NFL's one-game-a-week cadence), so this builder targets a
real DATE's real scheduled slate, not a "week's" slate -- and, more
importantly, a future game is scored off each team's TRUE CURRENT
cumulative record (whatever it is right now), not a week-bucketed
asof lookup keyed to an exact week number.

That distinction matters: cfb_moneyline_clean_baseline_a.py's asof
tracker only records a team's state at the exact week number a completed
game of theirs falls on. For CFB/NFL that's harmless because the served
week is always a whole week in the future, uniform across every team.
For NHL, "the next real game" can land on any day, and by the time it's
being served, the two teams involved will usually have played a
DIFFERENT number of games so far this week (or none at all, checking
mid-week) -- an exact week-number match would silently drop most real
games from the board. MoneylineEngine here instead tracks each team's
final (most-recent) cumulative state directly (final_state), valid for
scoring a game on ANY future date regardless of what week bucket it
falls in. The offline training/validation scripts (nhl_moneyline_clean_
baseline_a.py, nhl_moneyline_champion_gate_a.py, nhl_moneyline_
walkforward_stability_a.py) still use week-bucketed asof rows for
DEV/VAL/HOLDOUT splitting and week-block bootstrap -- that's a
real, already-happened-history replay, where the week-exact convention
this repo already uses for CFB/NFL is fine; only the FUTURE-scoring path
needed the fix.

Two markets, both team-level:
  moneyline               in-season model, once a team has >= 3 real
                           current-season games (nhl_moneyline_
                           walkforward_stability_a.py, champion-gated +
                           walkforward-stable)
  moneyline_early_season   prior-season-informed bootstrap for the first
                           3 weeks of a new season (nhl_prior_season_
                           moneyline_gate_a.py) -- the ONLY NHL market
                           this repo can serve until the 2026-27 season
                           (starting October) has actually begun, since
                           there is no current-season data yet.

Predictions-first: no real book odds wired for this yet -- just the
calibrated win probability, matching every other market's stated design
in this repo.

Run
---
python -u nhl_serving_builder_a.py --selftest        # offline sanity checks
python -u nhl_serving_builder_a.py                   # build today's (or next real slate's) board
python -u nhl_serving_builder_a.py --date 2026-10-08  # explicit target date
"""
import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

REPO = Path(__file__).resolve().parent
DB_DEFAULT = REPO / "nhl_models" / "nhl_model.sqlite"
DOCS = REPO / "docs"

import nhl_moneyline_champion_gate_a as gate_mod  # metrics/auc/NAN
from cfb_rushing_yards_champion_gate_b import fit_platt, apply_platt

MIN_PRIOR_GAMES = 3
PRIOR_SEASON_MAX_WEEK = 3

MONEYLINE_FEATURES = gate_mod.FEATURES
MONEYLINE_MODEL_DIR = REPO / "nhl_models" / "nhl_moneyline_walkforward_stability_a_work"

MONEYLINE_PRIOR_SEASON_FEATURES = [
    "prior_net_margin", "prior_win_rate", "prior_avg_goals_for", "prior_avg_goals_against",
    "prior_games", "opp_prior_net_margin", "opp_prior_win_rate", "opp_prior_avg_goals_for",
    "opp_prior_avg_goals_against", "opp_prior_games", "prior_projected_margin",
    "is_home",
]
MONEYLINE_PRIOR_SEASON_MODEL_DIR = REPO / "nhl_models" / "nhl_prior_season_moneyline_gate_a_work"
MIN_PRIOR_SEASON_GAMES = 10  # NHL plays 82 games/season -- a fuller floor than CFB's

# Same append-only picks-log ledger pattern as every other sport in this
# repo (cfb_picks_log.jsonl / nfl_picks_log.jsonl / tennis_picks_log.jsonl
# / mlb_picks_log.jsonl) -- source of truth for future grading, never
# rewritten once logged. player_id is always None here (team-level, no
# player), so the key falls back to team -- same fix CFB's moneyline log
# needed after a real dedup collision was found there.
PICKS_LOG_PATH = DOCS / "nhl_picks_log.jsonl"


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_logged_pick_keys(path):
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            keys.add((r.get("season"), r.get("game_date"), r.get("market"),
                       r.get("player_id") or r.get("team")))
    return keys


def append_new_picks_to_log(path, keys, picks):
    new_lines = []
    for p in picks:
        k = (p["season"], p["game_date"], p["market"], p["player_id"] or p["team"])
        if k in keys:
            continue
        keys.add(k)
        new_lines.append(json.dumps({**p, "logged_at": now_utc()}))
    if new_lines:
        with path.open("a") as f:
            f.write("\n".join(new_lines) + "\n")
    return len(new_lines)


class MoneylineEngine:
    """Team-level engine for nhl_moneyline. See module docstring for why
    asof_future() scores off final_state (each team's true current
    cumulative record) rather than a week-exact lookup."""

    def __init__(self, con, season):
        self.season = season
        self.con = con
        self.games = con.execute("""
            SELECT game_id, week, home_team, away_team, home_score, away_score
            FROM games WHERE season = ? AND home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY week, game_date
        """, (season,)).fetchall()
        self.state_asof, self.final_state = self._build_state()

    def _build_state(self):
        by_week = {}
        for g in self.games:
            by_week.setdefault(g[1], []).append(g)
        team_state = {}  # team -> [wins, goals_for, goals_against, games]
        state_asof = {}
        for w in sorted(by_week):
            for (gid, week, home, away, hs, aws) in by_week[w]:
                for team in (home, away):
                    st = team_state.get(team, [0, 0, 0, 0])
                    n = st[3]
                    state_asof[(team, week)] = {
                        "games_played": n,
                        "win_rate": (st[0] / n) if n > 0 else None,
                        "net_margin": ((st[1] - st[2]) / n) if n > 0 else None,
                        "avg_goals_for": (st[1] / n) if n > 0 else None,
                        "avg_goals_against": (st[2] / n) if n > 0 else None,
                    }
            for (gid, week, home, away, hs, aws) in by_week[w]:
                hst = team_state.setdefault(home, [0, 0, 0, 0])
                hst[0] += 1 if hs > aws else 0
                hst[1] += hs; hst[2] += aws; hst[3] += 1
                ast = team_state.setdefault(away, [0, 0, 0, 0])
                ast[0] += 1 if aws > hs else 0
                ast[1] += aws; ast[2] += hs; ast[3] += 1
        final_state = {}
        for team, (wins, gf, ga, n) in team_state.items():
            if n == 0:
                continue
            final_state[team] = {
                "games_played": n, "win_rate": wins / n, "net_margin": (gf - ga) / n,
                "avg_goals_for": gf / n, "avg_goals_against": ga / n,
            }
        return state_asof, final_state

    @staticmethod
    def _feat(own_st, opp_st, is_home):
        return {
            "team_net_margin": own_st["net_margin"],
            "team_win_rate": own_st["win_rate"],
            "team_avg_goals_for": own_st["avg_goals_for"],
            "team_avg_goals_against": own_st["avg_goals_against"],
            "opp_net_margin": opp_st["net_margin"],
            "opp_win_rate": opp_st["win_rate"],
            "opp_avg_goals_for": opp_st["avg_goals_for"],
            "opp_avg_goals_against": opp_st["avg_goals_against"],
            "projected_margin": own_st["net_margin"] - opp_st["net_margin"],
            "is_home": 1.0 if is_home else 0.0,
            "team_games_played": own_st["games_played"],
            "opp_games_played": opp_st["games_played"],
        }

    def replay(self):
        """Yield (team, opp, week, feat, team_won) for every real
        completed game this season, two rows per game -- for grading/
        calibration warmup. Uses the exact-week asof tracker (a real
        historical replay, not a future projection), matching the
        offline baseline builder exactly."""
        out = []
        for (gid, week, home, away, hs, aws) in self.games:
            home_st = self.state_asof.get((home, week))
            away_st = self.state_asof.get((away, week))
            if not home_st or not away_st:
                continue
            if home_st["games_played"] < MIN_PRIOR_GAMES or away_st["games_played"] < MIN_PRIOR_GAMES:
                continue
            home_won = 1 if hs > aws else 0
            out.append((home, away, week, self._feat(home_st, away_st, True), home_won))
            out.append((away, home, week, self._feat(away_st, home_st, False), 1 - home_won))
        return out

    def asof_future(self, schedule):
        """schedule: list of (home, away) real, already-scheduled games
        on the target date. Returns one row per team's perspective, only
        for games where BOTH teams already clear MIN_PRIOR_GAMES real
        games so far THIS season, scored off each team's true current
        cumulative record (see module docstring)."""
        out = []
        for home, away in schedule:
            home_st = self.final_state.get(home)
            away_st = self.final_state.get(away)
            if not home_st or not away_st:
                continue
            if home_st["games_played"] < MIN_PRIOR_GAMES or away_st["games_played"] < MIN_PRIOR_GAMES:
                continue
            out.append((home, away, self._feat(home_st, away_st, True)))
            out.append((away, home, self._feat(away_st, home_st, False)))
        return out


def score(bst, feats_order, feat_dicts, xgb):
    X = np.array([[fd.get(c) if fd.get(c) is not None else gate_mod.NAN for c in feats_order]
                  for fd in feat_dicts], dtype=np.float32)
    itr = (0, bst.best_iteration + 1)
    return np.asarray(
        bst.predict(xgb.DMatrix(X, feature_names=feats_order), iteration_range=itr),
        dtype=float)


def fit_serving_platt_moneyline(con, bst, xgb, serving_season):
    """Mirrors cfb_serving_builder_a.py's fit_serving_platt_moneyline:
    TWO separate Platt maps (home, away), same reason validated there
    (a single pooled map failed the away slice's own calibration test on
    CFB's identical model shape). Simplified relative to CFB: no
    target_week parameter needed -- MoneylineEngine.replay() already
    only returns COMPLETED games, so 'this season's calibration
    additions so far' is just the whole replay, unconditionally (there's
    no future/unplayed row to accidentally include)."""
    seasons = [r[0] for r in con.execute(
        "SELECT DISTINCT season FROM games WHERE season < ? ORDER BY season DESC",
        (serving_season,))]
    if not seasons:
        raise RuntimeError(f"no completed season before {serving_season} in db")
    warm_season = seasons[0]

    warm_engine = MoneylineEngine(con, warm_season)
    warm = warm_engine.replay()
    wk_counts = Counter(row[2] for row in warm)
    weeks_sorted = sorted(wk_counts)
    target_n = max(60, int(len(warm) * 0.2))
    cum = 0; cut = weeks_sorted[-1] if weeks_sorted else 0
    for w in reversed(weeks_sorted):
        cum += wk_counts[w]; cut = w
        if cum >= target_n:
            break
    warm_slice = [row for row in warm if row[2] >= cut]

    cur_engine = MoneylineEngine(con, serving_season)
    cur_seen = cur_engine.replay()

    def fit_side(is_home):
        is_home_val = 1.0 if is_home else 0.0
        w_slice = [row for row in warm_slice if row[3]["is_home"] == is_home_val]
        warm_raw = score(bst, MONEYLINE_FEATURES, [row[3] for row in w_slice], xgb)
        warm_y = np.array([float(row[4]) for row in w_slice])
        cur_slice = [row for row in cur_seen if row[3]["is_home"] == is_home_val]
        if cur_slice:
            cur_raw = score(bst, MONEYLINE_FEATURES, [row[3] for row in cur_slice], xgb)
            cur_y = np.array([float(row[4]) for row in cur_slice])
            pool_raw = np.concatenate([warm_raw, cur_raw])
            pool_y = np.concatenate([warm_y, cur_y])
        else:
            pool_raw, pool_y = warm_raw, warm_y
        a, b = fit_platt(pool_raw, pool_y)
        if a <= 0:
            a, b = 1.0, 0.0
        return a, b, len(w_slice), int(len(pool_y))

    a_home, b_home, warm_n_home, pool_n_home = fit_side(True)
    a_away, b_away, warm_n_away, pool_n_away = fit_side(False)

    return ((a_home, b_home), (a_away, b_away)), cur_engine, {
        "policy": "growing", "warmup_season": warm_season, "warmup_cut_week": int(cut),
        "warmup_n_home": warm_n_home, "warmup_n_away": warm_n_away,
        "current_season_n": len(cur_seen), "pool_n_home": pool_n_home, "pool_n_away": pool_n_away,
    }


def load_full_season_team_stats(conn, season):
    games = conn.execute("""
        SELECT home_team, away_team, home_score, away_score
        FROM games WHERE season = ? AND home_score IS NOT NULL AND away_score IS NOT NULL
    """, (season,)).fetchall()
    state = {}
    for home, away, hs, aws in games:
        hst = state.setdefault(home, [0, 0, 0, 0])
        hst[0] += 1 if hs > aws else 0
        hst[1] += hs; hst[2] += aws; hst[3] += 1
        ast = state.setdefault(away, [0, 0, 0, 0])
        ast[0] += 1 if aws > hs else 0
        ast[1] += aws; ast[2] += hs; ast[3] += 1
    out = {}
    for team, (wins, gf, ga, n) in state.items():
        if n == 0:
            continue
        out[team] = {"games": n, "win_rate": wins / n, "net_margin": (gf - ga) / n,
                     "avg_goals_for": gf / n, "avg_goals_against": ga / n}
    return out


def build_moneyline_prior_season_picks(con, season, game_date, schedule_with_week, xgb):
    """schedule_with_week: list of (home, away, week) for the target
    date's real scheduled games -- week <= PRIOR_SEASON_MAX_WEEK gates
    which games get a bootstrap pick, same convention as CFB's version.
    Real team names match directly here (NHL team names are stable
    year-to-year from this same live foundation script, no ESPN-vs-
    cfbfastR crosswalk needed the way CFB's version required)."""
    model_path = MONEYLINE_PRIOR_SEASON_MODEL_DIR / "nhl_prior_season_moneyline.json"
    cols_path = MONEYLINE_PRIOR_SEASON_MODEL_DIR / "nhl_prior_season_moneyline_columns.json"
    if not model_path.exists():
        return [], {"eligible": 0, "reason": "prior-season model not present"}
    feat_cols = json.loads(cols_path.read_text())
    assert feat_cols == MONEYLINE_PRIOR_SEASON_FEATURES

    early = [(h, a) for h, a, wk in schedule_with_week if wk <= PRIOR_SEASON_MAX_WEEK]
    if not early:
        return [], {"eligible": 0, "reason": f"no games with week <= {PRIOR_SEASON_MAX_WEEK}"}

    prior_season = season - 1
    prior_stats = load_full_season_team_stats(con, prior_season)
    bst = xgb.Booster(); bst.load_model(str(model_path))

    picks = []
    n_no_history = 0
    for home, away in early:
        home_st = prior_stats.get(home)
        away_st = prior_stats.get(away)
        if not home_st or not away_st:
            n_no_history += 1
            continue
        if home_st["games"] < MIN_PRIOR_SEASON_GAMES or away_st["games"] < MIN_PRIOR_SEASON_GAMES:
            n_no_history += 1
            continue

        rows = []
        for team_st, opp_st, is_home in ((home_st, away_st, True), (away_st, home_st, False)):
            rows.append([
                team_st["net_margin"], team_st["win_rate"], team_st["avg_goals_for"], team_st["avg_goals_against"],
                team_st["games"], opp_st["net_margin"], opp_st["win_rate"], opp_st["avg_goals_for"],
                opp_st["avg_goals_against"], opp_st["games"], team_st["net_margin"] - opp_st["net_margin"],
                1.0 if is_home else 0.0,
            ])
        dm = xgb.DMatrix(np.array(rows, dtype=np.float32), feature_names=feat_cols)
        probs = bst.predict(dm)
        home_prob, away_prob = float(probs[0]), float(probs[1])
        if home_prob >= away_prob:
            team, opp, cp = home, away, home_prob
        else:
            team, opp, cp = away, home, away_prob

        picks.append({
            "market": "moneyline_early_season", "player_id": None, "player": team,
            "team": team, "opponent": opp, "season": season, "game_date": game_date,
            "pick": f"{team} ML",
            "model_prob": round(cp, 4),
            "is_home": 1.0 if team == home else 0.0,
            "prior_season": prior_season,
            "model_source": "prior_season_informed",
        })

    return picks, {"eligible": len(picks), "prior_season": prior_season,
                    "scheduled_early_games": len(early), "no_prior_season_history": n_no_history}


def infer_target_date(con, today_str):
    r = con.execute(
        "SELECT MIN(game_date) FROM games WHERE game_date >= ?", (today_str,)).fetchone()
    return r[0] if r and r[0] else None


def selftest(con, xgb):
    print("SELFTEST: NHL moneyline engine sanity checks")
    ok = True

    engine = MoneylineEngine(con, 2024)
    rows = engine.replay()
    print(f"  2024 replay: {len(rows)} rows ({len(rows)//2} games), "
          f"win_rate={np.mean([r[4] for r in rows]):.4f} (should be exactly 0.5 by symmetric construction)")
    if abs(np.mean([r[4] for r in rows]) - 0.5) > 1e-9:
        print("  FAIL: symmetric construction violated")
        ok = False

    model_path = MONEYLINE_MODEL_DIR / "nhl_moneyline.json"
    if model_path.exists():
        bst = xgb.Booster(); bst.load_model(str(model_path))
        feat_cols = json.loads((MONEYLINE_MODEL_DIR / "nhl_moneyline_columns.json").read_text())
        assert feat_cols == MONEYLINE_FEATURES
        raw = score(bst, MONEYLINE_FEATURES, [r[3] for r in rows[:200]], xgb)
        print(f"  scored 200 real 2024 rows, mean raw prob={np.mean(raw):.3f} (sanity, not a formal test)")
    else:
        print("  (no champion model artifact yet -- skipping scoring smoke test)")

    print(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_DEFAULT))
    ap.add_argument("--date", default=None, help="YYYY-MM-DD, default: soonest real date with an upcoming game")
    ap.add_argument("--out", default=str(DOCS / "nhl_predictions.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    import xgboost as xgb

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    print("NHL_SERVING_BUILDER_A\n=====================")

    if args.selftest:
        ok = selftest(con, xgb)
        con.close()
        return 0 if ok else 1

    target_date = args.date or infer_target_date(con, date.today().isoformat())
    if target_date is None:
        print("no upcoming games found in the schedule -- writing empty board")
        payload = {"generated_at_utc": now_utc(), "season": None, "game_date": None,
                   "picks": [], "note": "no upcoming games in foundation schedule; "
                   "refresh the foundation db"}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        return 0

    row = con.execute(
        "SELECT DISTINCT season FROM games WHERE game_date = ?", (target_date,)).fetchone()
    season = row[0] if row else None
    if season is None:
        print(f"no season found for game_date={target_date} -- writing empty board")
        payload = {"generated_at_utc": now_utc(), "season": None, "game_date": target_date,
                   "picks": [], "note": "target date has no games in foundation schedule"}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        return 0

    print(f"target: game_date {target_date} (season {season}-{season+1})")
    schedule_rows = con.execute(
        "SELECT home_team, away_team, week, home_score, away_score FROM games WHERE game_date=?",
        (target_date,)).fetchall()
    schedule_all = [(h, a) for h, a, wk, hs, aws in schedule_rows]
    schedule_with_week = [(h, a, wk) for h, a, wk, hs, aws in schedule_rows]
    print(f"scheduled games: {len(schedule_all)}")

    finished_matchups = set()
    for h, a, wk, hs, aws in schedule_rows:
        if hs is not None and aws is not None:
            finished_matchups.add((h, a)); finished_matchups.add((a, h))
    if finished_matchups:
        print(f"  {len(finished_matchups) // 2} of {len(schedule_all)} scheduled games already final "
              f"(picks for these excluded from the live board)")

    picks = []
    all_picks_for_log = []
    market_meta = {}

    moneyline_model_path = MONEYLINE_MODEL_DIR / "nhl_moneyline.json"
    if not moneyline_model_path.exists():
        print("  moneyline: no model artifact found -- skipping")
        market_meta["moneyline"] = {"eligible": 0, "reason": "model not built yet"}
    else:
        moneyline_bst = xgb.Booster()
        moneyline_bst.load_model(str(moneyline_model_path))
        moneyline_cols = json.loads((MONEYLINE_MODEL_DIR / "nhl_moneyline_columns.json").read_text())
        assert moneyline_cols == MONEYLINE_FEATURES
        try:
            (platt_home, platt_away), moneyline_engine, pool_info = fit_serving_platt_moneyline(
                con, moneyline_bst, xgb, season)
            cand = moneyline_engine.asof_future(schedule_all)
            if cand:
                raw = score(moneyline_bst, MONEYLINE_FEATURES, [c[2] for c in cand], xgb)
                cal = np.empty(len(cand))
                for i, c in enumerate(cand):
                    a, b = platt_home if c[2]["is_home"] == 1.0 else platt_away
                    cal[i] = apply_platt(raw[i:i + 1], a, b)[0]

                by_game = {}
                for c, rp, cp in zip(cand, raw, cal):
                    team, opp, feat = c
                    key = frozenset((team, opp))
                    entry = by_game.setdefault(key, {})
                    entry[team] = (opp, feat, float(rp), float(cp))

                n_eligible_games = 0
                for key, sides in by_game.items():
                    if len(sides) != 2:
                        continue
                    n_eligible_games += 1
                    team = max(sides, key=lambda t: sides[t][3])
                    opp, feat, rp, cp = sides[team]
                    pick = {
                        "market": "moneyline", "player_id": None, "player": team,
                        "team": team, "opponent": opp, "season": season, "game_date": target_date,
                        "pick": f"{team} ML",
                        "model_prob": round(cp, 4),
                        "raw_prob": round(rp, 4),
                        "is_home": feat["is_home"],
                        "team_games_played": feat["team_games_played"],
                        "opp_games_played": feat["opp_games_played"],
                    }
                    all_picks_for_log.append(pick)
                    if (team, opp) in finished_matchups:
                        continue
                    picks.append(pick)

                print(f"  moneyline: {n_eligible_games} eligible games  "
                      f"platt_home a={platt_home[0]:.3f} b={platt_home[1]:+.3f}  "
                      f"platt_away a={platt_away[0]:.3f} b={platt_away[1]:+.3f}  pool={pool_info}")
                market_meta["moneyline"] = {
                    "eligible": n_eligible_games,
                    "platt_home": {"a": platt_home[0], "b": platt_home[1]},
                    "platt_away": {"a": platt_away[0], "b": platt_away[1]},
                    "calibration_pool": pool_info,
                }
            else:
                print("  moneyline: no eligible games yet (expected before both teams have "
                      f"{MIN_PRIOR_GAMES} real current-season games)")
                market_meta["moneyline"] = {"eligible": 0}
        except RuntimeError as e:
            print(f"  moneyline: {e}")
            market_meta["moneyline"] = {"eligible": 0, "reason": str(e)}

    early_picks, early_meta = build_moneyline_prior_season_picks(
        con, season, target_date, schedule_with_week, xgb)
    all_picks_for_log.extend(early_picks)
    n_before = len(early_picks)
    early_picks = [p for p in early_picks if (p["team"], p["opponent"]) not in finished_matchups]
    early_meta["dropped_game_final"] = n_before - len(early_picks)
    if early_picks:
        print(f"  moneyline_early_season: {len(early_picks)} eligible "
              f"(prior-season-informed, weeks 1-{PRIOR_SEASON_MAX_WEEK} only)")
    picks.extend(early_picks)
    market_meta["moneyline_early_season"] = early_meta

    logged_keys = load_logged_pick_keys(PICKS_LOG_PATH)
    n_new_logged = append_new_picks_to_log(PICKS_LOG_PATH, logged_keys, all_picks_for_log)
    print(f"  picks log: {n_new_logged} new entries appended ({len(logged_keys)} total)")

    picks.sort(key=lambda p: -p["model_prob"])
    payload = {
        "generated_at_utc": now_utc(), "season": season, "game_date": target_date,
        "builder": "NHL_SERVING_BUILDER_A",
        "design": "frozen champion + growing-pool Platt (validated on real 2018-2024 seasons)",
        "markets": market_meta,
        "note": "predictions-first: no odds. Eligibility is stats-based and cannot see "
                "injuries/scratches/goalie starters.",
        "picks": picks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    hist = out.parent / f"nhl_predictions_{season}_{target_date}.json"
    hist.write_text(json.dumps(payload, indent=2))
    print(f"\n{len(picks)} picks written to {out} (+ {hist.name})")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
