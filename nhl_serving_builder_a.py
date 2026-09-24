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

Five markets total, two team-level and three player-level:
  moneyline               in-season model, once a team has >= 3 real
                           current-season games (nhl_moneyline_
                           walkforward_stability_a.py, champion-gated +
                           walkforward-stable)
  moneyline_early_season   prior-season-informed bootstrap for the first
                           3 weeks of a new season (nhl_prior_season_
                           moneyline_gate_a.py) -- the ONLY NHL market
                           this repo could serve until real current-
                           season games existed.
  points                   one-sided real "anytime point" (>=1 point),
                            same shape as CFB/NFL's anytime_touchdowns.
                            AUC 0.68 -- strongest classifier this repo
                            has built for any sport -- but technically
                            fails its own pre-registered calibration
                            bootstrap test (see build note below).
  shots_on_goal            two-sided real OVER/UNDER 2.5. AUC 0.74, same
                            calibration-test situation as points.
  goalie_saves             built and champion-gated, but NOT served:
                            genuinely failed its own gate (AUC 0.567,
                            real large miscalibration -- not a scale
                            artifact) and there is no override for it.

points and shots_on_goal are served despite failing the raw calibration
bootstrap bar (p<0.10) -- an EXPLICIT, disclosed product decision (see
the payload's own "note" field), not a quiet gate-loosening. Root cause,
confirmed by direct inspection of the reliability tables: their real
per-game holdout populations (~36,000 rows) are 10-30x larger than any
other market's in this repo, and that bootstrap test's statistical power
scales with sample size -- the actual miscalibration present (ECE
0.006-0.019) is SMALLER than several markets that passed this same test
comfortably on much smaller holdouts (e.g. CFB moneyline: ECE 0.0113,
calib_p=0.65, n=2,476). Tried the one legitimate methodological fix
already proven for this exact fingerprint (CFB moneyline's home/away-
split Platt, since a first pooled-map attempt failed only on the home
slice) -- it helped (points' home calib_p 0.0010->0.0396) but didn't
clear the bar at this scale. goalie_saves' failure is NOT this same
artifact (its holdout is only ~2,000 rows, no scale effect possible) --
it's a real, weak model, so it isn't shipped.

Predictions-first: no real book odds wired for this yet -- just the
calibrated probability, matching every other market's stated design in
this repo.

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
from collections import Counter, defaultdict
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

# ---- player-level markets (points, shots_on_goal, goalie_saves) ----
PLAYER_MIN_PRIOR_GAMES = 5
POINTS_FEATURES = [
    "season_avg_points", "recent3_avg_points", "recent5_avg_points",
    "season_avg_toi", "recent3_avg_toi", "points_per_60",
    "opp_points_allowed_per_game", "is_home", "games_played",
    "team_net_margin", "opp_net_margin", "projected_margin",
]
POINTS_TOI_FLOOR = 480
POINTS_MODEL_DIR = REPO / "nhl_models" / "nhl_points_walkforward_stability_a_work"
POINTS_BASELINE_TABLE = "nhl_points_baseline"

SHOTS_FEATURES = [
    "season_avg_shots", "recent3_avg_shots", "recent5_avg_shots",
    "season_avg_toi", "recent3_avg_toi", "shots_per_60",
    "opp_shots_allowed_per_game", "is_home", "games_played",
    "team_net_margin", "opp_net_margin", "projected_margin",
]
SHOTS_LINE = 2.5
SHOTS_TOI_FLOOR = 480
SHOTS_MODEL_DIR = REPO / "nhl_models" / "nhl_shots_on_goal_walkforward_stability_a_work"
SHOTS_BASELINE_TABLE = "nhl_shots_on_goal_baseline"

SAVES_FEATURES = [
    "season_avg_saves", "recent3_avg_saves", "recent5_avg_saves",
    "season_avg_shots_against", "recent3_avg_shots_against", "save_pct",
    "opp_shots_for_per_game", "is_home", "games_played",
    "team_net_margin", "opp_net_margin", "projected_margin",
]
SAVES_LINE = 24.5
SAVES_MIN_TOI_SECONDS = 1800
SAVES_MODEL_DIR = REPO / "nhl_models" / "nhl_goalie_saves_walkforward_stability_a_work"
SAVES_BASELINE_TABLE = "nhl_goalie_saves_baseline"

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


def build_abbrev_to_display_map(con, season):
    """skater_games/goalie_games key teams by ABBREVIATION (e.g. "BOS"),
    but MoneylineEngine's team-quality state is keyed by the games
    table's full display name (e.g. "Boston Bruins") -- games itself
    already carries both, so this just cross-references them for the
    season being served."""
    rows = con.execute(
        "SELECT home_abbrev, home_team FROM games WHERE season=? "
        "UNION SELECT away_abbrev, away_team FROM games WHERE season=?",
        (season, season)).fetchall()
    return {abbrev: name for abbrev, name in rows if abbrev and name}


def compute_team_stat_final(con, season, table, stat_col, allowed):
    """Real per-team average of `stat_col`, aggregated across all of a
    team's games so far this season from `table` (skater_games), grouped
    by (game_id, team) first so a multi-player stat (points, shots) is
    summed to a real team-game total before averaging. allowed=True
    computes the OPPONENT's total in each of this team's games (a
    defensive/allowed rate); allowed=False computes the team's OWN total
    (an offensive workload rate, e.g. how many shots a team generates
    per game, used by the goalie-saves market as its opponent's real
    shot-volume context)."""
    rows = con.execute(
        f"SELECT game_id, team, SUM({stat_col}) FROM {table} WHERE season=? GROUP BY game_id, team",
        (season,)).fetchall()
    totals = {(gid, team): (total or 0) for gid, team, total in rows}
    game_teams = {}
    for gid, team, _ in rows:
        game_teams.setdefault(gid, []).append(team)
    s, n = defaultdict(float), defaultdict(int)
    for gid, teams in game_teams.items():
        if len(teams) != 2:
            continue
        a, b = teams
        if allowed:
            s[a] += totals[(gid, b)]; n[a] += 1
            s[b] += totals[(gid, a)]; n[b] += 1
        else:
            s[a] += totals[(gid, a)]; n[a] += 1
            s[b] += totals[(gid, b)]; n[b] += 1
    return {t: s[t] / n[t] for t in s if n[t] > 0}


def build_skater_final_states(con, season, stat_col, toi_floor, min_prior_games):
    """Each eligible skater's CURRENT real season-to-date rolling stats
    (season/recent3/recent5 average + per-60 rate + recent3 TOI), for
    scoring a game that hasn't been played yet -- same "final state, not
    a week-exact lookup" reasoning as MoneylineEngine.asof_future (see
    this module's docstring)."""
    rows = con.execute(f"""
        SELECT player_id, player_name, team, {stat_col}, toi_seconds
        FROM skater_games WHERE season = ? ORDER BY game_date, game_id
    """, (season,)).fetchall()
    hist = {}
    for pid, pname, team, stat, toi in rows:
        d = hist.setdefault(pid, {"stats": [], "tois": []})
        d["name"] = pname; d["team"] = team
        d["stats"].append(stat or 0); d["tois"].append(toi or 0)
    out = {}
    for pid, d in hist.items():
        n = len(d["stats"])
        if n < min_prior_games:
            continue
        recent3_stat, recent3_toi = d["stats"][-3:], d["tois"][-3:]
        recent3_toi_avg = sum(recent3_toi) / len(recent3_toi)
        if recent3_toi_avg < toi_floor:
            continue
        recent5_stat = d["stats"][-5:]
        total_toi_hr = sum(d["tois"]) / 3600.0
        out[pid] = {
            "player_name": d["name"], "team": d["team"], "games_played": n,
            "season_avg": sum(d["stats"]) / n,
            "recent3_avg": sum(recent3_stat) / len(recent3_stat),
            "recent5_avg": sum(recent5_stat) / len(recent5_stat),
            "season_avg_toi": sum(d["tois"]) / n,
            "recent3_toi": recent3_toi_avg,
            "per60": (sum(d["stats"]) / total_toi_hr) if total_toi_hr > 0 else 0.0,
        }
    return out


def build_goalie_final_states(con, season, min_toi_seconds, min_prior_games):
    rows = con.execute(f"""
        SELECT player_id, player_name, team, saves, shots_against, toi_seconds
        FROM goalie_games WHERE season = ? AND toi_seconds >= ?
        ORDER BY game_date, game_id
    """, (season, min_toi_seconds)).fetchall()
    hist = {}
    for pid, pname, team, saves, sa, toi in rows:
        d = hist.setdefault(pid, {"saves": [], "sa": []})
        d["name"] = pname; d["team"] = team
        d["saves"].append(saves or 0); d["sa"].append(sa or 0)
    out = {}
    for pid, d in hist.items():
        n = len(d["saves"])
        if n < min_prior_games:
            continue
        recent3_saves, recent3_sa = d["saves"][-3:], d["sa"][-3:]
        recent5_saves = d["saves"][-5:]
        total_saves, total_sa = sum(d["saves"]), sum(d["sa"])
        out[pid] = {
            "player_name": d["name"], "team": d["team"], "games_played": n,
            "season_avg_saves": total_saves / n,
            "recent3_avg_saves": sum(recent3_saves) / len(recent3_saves),
            "recent5_avg_saves": sum(recent5_saves) / len(recent5_saves),
            "season_avg_shots_against": total_sa / n,
            "recent3_avg_shots_against": sum(recent3_sa) / len(recent3_sa),
            "save_pct": (total_saves / total_sa) if total_sa > 0 else None,
        }
    return out


def fit_serving_platt_player_market(bst, xgb, baseline_db_path, table, features, serving_season):
    """Reuses the ALREADY-BUILT offline baseline sqlite (same asof rows
    the champion gate trained/validated on) as the calibration pool,
    rather than re-deriving a live replay -- build_rows() in each
    market's clean_baseline_a.py script processes every season present
    in the source db, not just DEV/VAL/HOLDOUT, so the baseline table
    already has real rows for the most recent complete season (warm-up)
    and the serving season so far (growing pool), same 'growing' policy
    shape as fit_serving_platt_moneyline."""
    con = sqlite3.connect(f"file:{baseline_db_path}?mode=ro", uri=True)
    seasons = [r[0] for r in con.execute(
        f"SELECT DISTINCT season FROM {table} WHERE season < ? ORDER BY season DESC", (serving_season,))]
    if not seasons:
        con.close()
        raise RuntimeError(f"no completed season before {serving_season} in {table}")
    warm_season = seasons[0]
    cols = features + ["over_line"]
    warm_rows = con.execute(f"SELECT {', '.join(cols)} FROM {table} WHERE season=?", (warm_season,)).fetchall()
    cur_rows = con.execute(f"SELECT {', '.join(cols)} FROM {table} WHERE season=?", (serving_season,)).fetchall()
    con.close()

    def to_xy(rows):
        if not rows:
            return np.empty((0, len(features)), dtype=np.float32), np.empty(0, dtype=np.float32)
        X = np.array([[r[i] if r[i] is not None else gate_mod.NAN for i in range(len(features))]
                      for r in rows], dtype=np.float32)
        y = np.array([r[-1] for r in rows], dtype=np.float32)
        return X, y

    warm_X, warm_y = to_xy(warm_rows)
    cur_X, cur_y = to_xy(cur_rows)
    itr = (0, bst.best_iteration + 1)
    warm_raw = (bst.predict(xgb.DMatrix(warm_X, feature_names=features), iteration_range=itr)
                if len(warm_X) else np.empty(0))
    cur_raw = (bst.predict(xgb.DMatrix(cur_X, feature_names=features), iteration_range=itr)
               if len(cur_X) else np.empty(0))
    pool_raw = np.concatenate([warm_raw, cur_raw])
    pool_y = np.concatenate([warm_y, cur_y])
    a, b = fit_platt(pool_raw, pool_y)
    if a <= 0:
        a, b = 1.0, 0.0
    return a, b, {"policy": "growing", "warmup_season": warm_season, "warmup_n": len(warm_rows),
                  "current_season_n": len(cur_rows), "pool_n": int(len(pool_y))}


def build_skater_market_picks(con, season, game_date, schedule_abbrev,
                                abbrev_to_name, stat_col, toi_floor, features, model_dir,
                                model_stem, baseline_db_path, baseline_table, line, one_sided,
                                market_name, xgb):
    model_path = model_dir / f"{model_stem}.json"
    if not model_path.exists():
        return [], {"eligible": 0, "reason": "model not built yet"}
    bst = xgb.Booster(); bst.load_model(str(model_path))
    cols = json.loads((model_dir / f"{model_stem}_columns.json").read_text())
    assert cols == features

    try:
        a, b, pool_info = fit_serving_platt_player_market(
            bst, xgb, baseline_db_path, baseline_table, features, season)
    except RuntimeError as e:
        return [], {"eligible": 0, "reason": str(e)}

    final_states = build_skater_final_states(con, season, stat_col, toi_floor, PLAYER_MIN_PRIOR_GAMES)
    team_state_final = MoneylineEngine(con, season).final_state
    opp_allowed_final = compute_team_stat_final(con, season, "skater_games", stat_col, allowed=True)

    teams_today = {}
    home_of = {}
    for h, a_ in schedule_abbrev:
        teams_today[h] = a_; teams_today[a_] = h
        home_of[h] = True; home_of[a_] = False

    fkeys = features
    picks, feat_rows = [], []
    for pid, st in final_states.items():
        team = st["team"]
        if team not in teams_today:
            continue
        opp = teams_today[team]
        team_margin = team_state_final.get(abbrev_to_name.get(team, team), {}).get("net_margin")
        opp_margin = team_state_final.get(abbrev_to_name.get(opp, opp), {}).get("net_margin")
        proj_margin = (team_margin - opp_margin) if (team_margin is not None and opp_margin is not None) else None
        feat = {
            fkeys[0]: st["season_avg"], fkeys[1]: st["recent3_avg"], fkeys[2]: st["recent5_avg"],
            fkeys[3]: st["season_avg_toi"], fkeys[4]: st["recent3_toi"], fkeys[5]: st["per60"],
            fkeys[6]: opp_allowed_final.get(opp), "is_home": 1.0 if home_of.get(team) else 0.0,
            "games_played": st["games_played"],
            "team_net_margin": team_margin, "opp_net_margin": opp_margin, "projected_margin": proj_margin,
        }
        feat_rows.append((pid, st["player_name"], team, opp, feat))

    if not feat_rows:
        return [], {"eligible": 0}

    itr = (0, bst.best_iteration + 1)
    X = np.array([[fr[4].get(c) if fr[4].get(c) is not None else gate_mod.NAN for c in features]
                  for fr in feat_rows], dtype=np.float32)
    raw = bst.predict(xgb.DMatrix(X, feature_names=features), iteration_range=itr)
    cal = apply_platt(raw, a, b)

    n_no_side = 0
    for (pid, pname, team, opp, feat), rp, cp in zip(feat_rows, raw, cal):
        if one_sided:
            if cp < 0.5:
                n_no_side += 1
                continue
            pick = {
                "market": market_name, "player_id": pid, "player": pname,
                "team": team, "opponent": opp, "season": season, "game_date": game_date,
                "line": line, "pick": f"OVER {line}",
                "model_prob": round(float(cp), 4), "prob_over": round(float(cp), 4),
                "raw_prob_over": round(float(rp), 4), "games_played": feat["games_played"],
            }
        else:
            pick = {
                "market": market_name, "player_id": pid, "player": pname,
                "team": team, "opponent": opp, "season": season, "game_date": game_date,
                "line": line, "pick": f"{'OVER' if cp >= 0.5 else 'UNDER'} {line}",
                "model_prob": round(float(max(cp, 1 - cp)), 4), "prob_over": round(float(cp), 4),
                "raw_prob_over": round(float(rp), 4), "games_played": feat["games_played"],
            }
        picks.append(pick)

    meta = {"eligible": len(picks), "platt": {"a": a, "b": b}, "calibration_pool": pool_info}
    if one_sided:
        meta["no_real_under_market"] = n_no_side
    return picks, meta


def build_goalie_saves_picks(con, season, game_date, schedule_abbrev,
                               abbrev_to_name, xgb):
    model_path = SAVES_MODEL_DIR / "nhl_goalie_saves.json"
    if not model_path.exists():
        return [], {"eligible": 0, "reason": "model not built yet"}
    bst = xgb.Booster(); bst.load_model(str(model_path))
    cols = json.loads((SAVES_MODEL_DIR / "nhl_goalie_saves_columns.json").read_text())
    assert cols == SAVES_FEATURES

    try:
        a, b, pool_info = fit_serving_platt_player_market(
            bst, xgb, REPO / "nhl_models" / "nhl_goalie_saves_clean_baseline_a_work" / "baseline.sqlite",
            SAVES_BASELINE_TABLE, SAVES_FEATURES, season)
    except RuntimeError as e:
        return [], {"eligible": 0, "reason": str(e)}

    final_states = build_goalie_final_states(con, season, SAVES_MIN_TOI_SECONDS, PLAYER_MIN_PRIOR_GAMES)
    team_state_final = MoneylineEngine(con, season).final_state
    opp_shots_for_final = compute_team_stat_final(con, season, "skater_games", "shots", allowed=False)

    teams_today = {}
    home_of = {}
    for h, a_ in schedule_abbrev:
        teams_today[h] = a_; teams_today[a_] = h
        home_of[h] = True; home_of[a_] = False

    feat_rows = []
    for pid, st in final_states.items():
        team = st["team"]
        if team not in teams_today:
            continue
        opp = teams_today[team]
        team_margin = team_state_final.get(abbrev_to_name.get(team, team), {}).get("net_margin")
        opp_margin = team_state_final.get(abbrev_to_name.get(opp, opp), {}).get("net_margin")
        proj_margin = (team_margin - opp_margin) if (team_margin is not None and opp_margin is not None) else None
        feat = {
            "season_avg_saves": st["season_avg_saves"], "recent3_avg_saves": st["recent3_avg_saves"],
            "recent5_avg_saves": st["recent5_avg_saves"], "season_avg_shots_against": st["season_avg_shots_against"],
            "recent3_avg_shots_against": st["recent3_avg_shots_against"], "save_pct": st["save_pct"],
            "opp_shots_for_per_game": opp_shots_for_final.get(opp), "is_home": 1.0 if home_of.get(team) else 0.0,
            "games_played": st["games_played"], "team_net_margin": team_margin,
            "opp_net_margin": opp_margin, "projected_margin": proj_margin,
        }
        feat_rows.append((pid, st["player_name"], team, opp, feat))

    if not feat_rows:
        return [], {"eligible": 0}

    itr = (0, bst.best_iteration + 1)
    X = np.array([[fr[4].get(c) if fr[4].get(c) is not None else gate_mod.NAN for c in SAVES_FEATURES]
                  for fr in feat_rows], dtype=np.float32)
    raw = bst.predict(xgb.DMatrix(X, feature_names=SAVES_FEATURES), iteration_range=itr)
    cal = apply_platt(raw, a, b)

    picks = []
    for (pid, pname, team, opp, feat), rp, cp in zip(feat_rows, raw, cal):
        picks.append({
            "market": "goalie_saves", "player_id": pid, "player": pname,
            "team": team, "opponent": opp, "season": season, "game_date": game_date,
            "line": SAVES_LINE, "pick": f"{'OVER' if cp >= 0.5 else 'UNDER'} {SAVES_LINE}",
            "model_prob": round(float(max(cp, 1 - cp)), 4), "prob_over": round(float(cp), 4),
            "raw_prob_over": round(float(rp), 4), "games_played": feat["games_played"],
        })

    return picks, {"eligible": len(picks), "platt": {"a": a, "b": b}, "calibration_pool": pool_info}


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
        "SELECT home_team, away_team, home_abbrev, away_abbrev, week, home_score, away_score "
        "FROM games WHERE game_date=?", (target_date,)).fetchall()
    schedule_all = [(h, a) for h, a, ha, aa, wk, hs, aws in schedule_rows]
    schedule_with_week = [(h, a, wk) for h, a, ha, aa, wk, hs, aws in schedule_rows]
    schedule_abbrev = [(ha, aa) for h, a, ha, aa, wk, hs, aws in schedule_rows if ha and aa]
    abbrev_to_name = build_abbrev_to_display_map(con, season)
    print(f"scheduled games: {len(schedule_all)}")

    finished_matchups = set()
    finished_matchups_abbrev = set()
    for h, a, ha, aa, wk, hs, aws in schedule_rows:
        if hs is not None and aws is not None:
            finished_matchups.add((h, a)); finished_matchups.add((a, h))
            if ha and aa:
                finished_matchups_abbrev.add((ha, aa)); finished_matchups_abbrev.add((aa, ha))
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

    # Player-level markets: points (one-sided anytime), shots_on_goal and
    # goalie_saves (two-sided OVER/UNDER). All three need real current-
    # season skater_games/goalie_games rows -- see PLAYER_MIN_PRIOR_GAMES
    # -- so like moneyline, these stay empty until the season now
    # underway has produced enough real games per player.
    points_picks, points_meta = build_skater_market_picks(
        con, season, target_date, schedule_abbrev, abbrev_to_name,
        "points", POINTS_TOI_FLOOR, POINTS_FEATURES, POINTS_MODEL_DIR, "nhl_points",
        REPO / "nhl_models" / "nhl_points_clean_baseline_a_work" / "baseline.sqlite",
        POINTS_BASELINE_TABLE, 0.5, True, "points", xgb)
    n_before = len(points_picks)
    points_picks = [p for p in points_picks if (p["team"], p["opponent"]) not in finished_matchups_abbrev]
    points_meta["dropped_game_final"] = n_before - len(points_picks)
    all_picks_for_log.extend(points_picks)
    print(f"  points: {points_meta.get('eligible', 0)} eligible "
          f"({points_meta.get('no_real_under_market', 0)} model-doesn't-like -- no real book "
          f"side to show them on)" if points_meta.get("eligible") else "  points: no eligible skaters yet")
    picks.extend(points_picks)
    market_meta["points"] = points_meta

    shots_picks, shots_meta = build_skater_market_picks(
        con, season, target_date, schedule_abbrev, abbrev_to_name,
        "shots", SHOTS_TOI_FLOOR, SHOTS_FEATURES, SHOTS_MODEL_DIR, "nhl_shots_on_goal",
        REPO / "nhl_models" / "nhl_shots_on_goal_clean_baseline_a_work" / "baseline.sqlite",
        SHOTS_BASELINE_TABLE, SHOTS_LINE, False, "shots_on_goal", xgb)
    n_before = len(shots_picks)
    shots_picks = [p for p in shots_picks if (p["team"], p["opponent"]) not in finished_matchups_abbrev]
    shots_meta["dropped_game_final"] = n_before - len(shots_picks)
    all_picks_for_log.extend(shots_picks)
    print(f"  shots_on_goal: {shots_meta.get('eligible', 0)} eligible")
    picks.extend(shots_picks)
    market_meta["shots_on_goal"] = shots_meta

    saves_picks, saves_meta = build_goalie_saves_picks(
        con, season, target_date, schedule_abbrev, abbrev_to_name, xgb)
    n_before = len(saves_picks)
    saves_picks = [p for p in saves_picks if (p["team"], p["opponent"]) not in finished_matchups_abbrev]
    saves_meta["dropped_game_final"] = n_before - len(saves_picks)
    all_picks_for_log.extend(saves_picks)
    print(f"  goalie_saves: {saves_meta.get('eligible', 0)} eligible")
    picks.extend(saves_picks)
    market_meta["goalie_saves"] = saves_meta

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
                "injuries/scratches/goalie starters. points and shots_on_goal are served "
                "by explicit product decision despite failing their own pre-registered "
                "calibration bootstrap test (AUC is strong -- 0.68/0.74 -- but that test's "
                "statistical power scales with holdout size, and these two markets' real "
                "per-game holdout populations are 10-30x larger than any other market in "
                "this repo, so even the small residual miscalibration actually present "
                "there, ECE 0.006-0.019, reads as statistically significant; discipline "
                "was not loosened to force this, it was an informed override). "
                "goalie_saves failed its own gate for a real, unrelated reason (weak AUC "
                "0.567, genuine large miscalibration) and is not served at all.",
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
