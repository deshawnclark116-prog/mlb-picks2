#!/usr/bin/env python3
"""
NFL_SERVING_BUILDER_A

The live serving path for the two fully validated NFL markets:

  rushing_yards    RB-only,  over 49.5 rushing yards
  receiving_yards  WR-only,  over 49.5 receiving yards

Deployment design is EXACTLY the configuration that passed the walk-forward
gates and stability confirmation -- nothing else:
  frozen champion model (from nfl_models/, repo-backed, never retrained
  here) + weekly Platt recalibration refit on the season in progress
  (warmup pool = the most recent completed season's internal-val slice,
  exactly the role 2023's slice played in validation).

Predictions-first: no odds anywhere. Emits calibrated P(over line) for
every eligible player in the target week's scheduled games, plus full
metadata, to docs/nfl_predictions.json (+ a per-week history file).

Weekly flow (GitHub Actions, Tuesdays in season -- see
.github/workflows/nfl_weekly.yml):
  1. rebuild the foundation db from nflverse (stateless, ~2 min)
  2. python nfl_serving_builder_a.py            (auto-picks the next week)
  3. commit docs/

Eligibility mirrors the validated baselines exactly: a player needs >= 3
prior games THIS season and a current-role recent rate (recent3 carries
>= 12 for RB rushing, recent3 targets >= 5 for WR receiving), so the
board is empty for weeks 1-3 by design -- the same population the models
were validated on (holdout weeks 4-22). Known limitation, documented not
hidden: eligibility is stats-based; it cannot see injuries/inactives for
the upcoming game (NFL has no MLB-style confirmed lineups).

Feature computation MIRRORS the baseline builders (same rules,
reimplemented for as-of-future-week serving) -- and --selftest PROVES the
mirror: it recomputes the full 2024 season through this engine and
requires (a) exact row-for-row feature parity with the validated
baseline.sqlite for both markets, and (b) walk-forward probabilities that
reproduce the gated AUC/ECE. Run it after any edit to this file.

Run
---
python -u nfl_serving_builder_a.py --selftest          # offline parity proof
python -u nfl_serving_builder_a.py                     # build next week's board
python -u nfl_serving_builder_a.py --season 2026 --week 7   # explicit target
"""

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import nfl_rushing_yards_champion_gate_a as g  # metrics/auc/pick_val_cut/NAN
from nfl_rushing_yards_recalibration_a import fit_platt, apply_platt

REPO = Path(__file__).resolve().parent
DB_DEFAULT = REPO / "nfl_models" / "nfl_model.sqlite"
DOCS = REPO / "docs"

MARKETS = {
    "rushing_yards": {
        "position": "RB",
        "line": 49.5,
        "stat_fields": ["carries", "rushing_yards"],
        "rate_field": "carries", "min_recent_rate": 12,
        "opp_stat": "rushing_yards",
        "features": ["season_avg_rush_yards", "recent3_avg_rush_yards",
                      "recent5_avg_rush_yards", "season_avg_carries",
                      "recent3_avg_carries", "yards_per_carry",
                      "opp_rush_yards_allowed_per_game", "is_home", "games_played"],
        "model_dir": REPO / "nfl_models" / "nfl_rushing_yards_champion_gate_a_work",
        "stem": "nfl_rushing_yards",
        "baseline_table": ("nfl_models/nfl_rushing_yards_clean_baseline_a_work/baseline.sqlite",
                            "nfl_rushing_yards_baseline"),
        "verdicts": ["NFL_RUSHING_YARDS_WALKFORWARD_PASSES_GATE",
                      "NFL_RUSHING_YARDS_WALKFORWARD_STABLE_READY_FOR_LIVE_WIRING"],
    },
    "receiving_yards": {
        "position": "WR",
        "line": 49.5,
        "stat_fields": ["targets", "receptions", "receiving_yards"],
        "rate_field": "targets", "min_recent_rate": 5,
        "opp_stat": "receiving_yards",
        "features": ["season_avg_rec_yards", "recent3_avg_rec_yards",
                      "recent5_avg_rec_yards", "season_avg_targets",
                      "recent3_avg_targets", "yards_per_target", "catch_rate",
                      "opp_rec_yards_allowed_per_game", "is_home", "games_played"],
        "model_dir": REPO / "nfl_models" / "nfl_receiving_yards_champion_walkforward_gate_a_work",
        "stem": "nfl_receiving_yards",
        "baseline_table": ("nfl_models/nfl_receiving_yards_clean_baseline_a_work/baseline.sqlite",
                            "nfl_receiving_yards_baseline"),
        "verdicts": ["NFL_RECEIVING_YARDS_WALKFORWARD_PASSES_GATE",
                      "NFL_RECEIVING_YARDS_WALKFORWARD_STABLE_READY_FOR_LIVE_WIRING"],
    },
}
MIN_PRIOR_GAMES = 3


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def market_features(mkt, hist, opp_allowed, is_home):
    """Feature dict for one player given their in-season history (ordered
    prior games, dicts of stat_fields). Mirrors the baseline builders
    exactly -- proven by --selftest, not assumed."""
    n = len(hist)
    if mkt == "rushing_yards":
        ry = [h["rushing_yards"] or 0 for h in hist]
        c = [h["carries"] or 0 for h in hist]
        r3, r5, c3 = ry[-3:], ry[-5:], c[-3:]
        return {
            "season_avg_rush_yards": sum(ry) / n,
            "recent3_avg_rush_yards": sum(r3) / len(r3),
            "recent5_avg_rush_yards": sum(r5) / len(r5),
            "season_avg_carries": sum(c) / n,
            "recent3_avg_carries": sum(c3) / len(c3),
            "yards_per_carry": (sum(ry) / sum(c)) if sum(c) > 0 else 0.0,
            "opp_rush_yards_allowed_per_game": opp_allowed,
            "is_home": 1.0 if is_home else 0.0,
            "games_played": n,
        }
    ry = [h["receiving_yards"] or 0 for h in hist]
    t = [h["targets"] or 0 for h in hist]
    rec = [h["receptions"] or 0 for h in hist]
    r3, r5, t3 = ry[-3:], ry[-5:], t[-3:]
    return {
        "season_avg_rec_yards": sum(ry) / n,
        "recent3_avg_rec_yards": sum(r3) / len(r3),
        "recent5_avg_rec_yards": sum(r5) / len(r5),
        "season_avg_targets": sum(t) / n,
        "recent3_avg_targets": sum(t3) / len(t3),
        "yards_per_target": (sum(ry) / sum(t)) if sum(t) > 0 else 0.0,
        "catch_rate": (sum(rec) / sum(t)) if sum(t) > 0 else 0.0,
        "opp_rec_yards_allowed_per_game": opp_allowed,
        "is_home": 1.0 if is_home else 0.0,
        "games_played": n,
    }


def eligible(mkt_cfg, hist):
    if len(hist) < MIN_PRIOR_GAMES:
        return False
    rates = [h[mkt_cfg["rate_field"]] or 0 for h in hist][-3:]
    return (sum(rates) / len(rates)) >= mkt_cfg["min_recent_rate"]


class SeasonEngine:
    """Replays one market's season week-by-week from player_games, exposing
    (a) completed eligible rows with features + outcomes (for calibration
    pools and self-test parity) and (b) as-of features for a FUTURE week."""

    def __init__(self, con, mkt, season):
        self.mkt = mkt
        self.cfg = MARKETS[mkt]
        self.season = season
        fields = ", ".join(self.cfg["stat_fields"])
        self.rows = con.execute(f"""
            SELECT player_id, player_name, team, opponent, week, is_home, {fields}
            FROM player_games
            WHERE position = ? AND season = ?
            ORDER BY week
        """, (self.cfg["position"], season)).fetchall()
        self.weeks = sorted({r[4] for r in self.rows})

    def replay(self):
        """Yield eligible completed rows in week order:
        (player_id, player_name, team, opponent, week, feat_dict, actual)."""
        cfg = self.cfg
        hist = {}
        opp_state = {}
        out = []
        for w in self.weeks:
            wk = [r for r in self.rows if r[4] == w]
            for r in wk:
                pid, pname, team, opp, week, is_home = r[:6]
                stats = dict(zip(cfg["stat_fields"], r[6:]))
                h = hist.get(pid, [])
                if eligible(cfg, h):
                    st = opp_state.get(opp)
                    opp_allowed = (st[0] / st[1]) if st and st[1] > 0 else None
                    feat = market_features(self.mkt, h, opp_allowed, is_home == 1)
                    actual = stats[cfg["opp_stat"]] or 0
                    out.append((pid, pname, team, opp, week, feat, actual))
            for r in wk:
                opp = r[3]
                stats = dict(zip(cfg["stat_fields"], r[6:]))
                st = opp_state.setdefault(opp, [0, 0])
                st[0] += stats[cfg["opp_stat"]] or 0
                st[1] += 1
            for r in wk:
                pid = r[0]
                hist.setdefault(pid, []).append(dict(zip(cfg["stat_fields"], r[6:])))
        return out

    def asof_future(self, target_week, schedule):
        """Features for a not-yet-played week. schedule: list of
        (home_team, away_team). A player belongs to the team of their most
        recent played game this season."""
        cfg = self.cfg
        hist = {}
        opp_state = {}
        latest_team = {}
        latest_name = {}
        for r in self.rows:
            pid, pname, team, opp, week = r[0], r[1], r[2], r[3], r[4]
            if week >= target_week:
                continue
            stats = dict(zip(cfg["stat_fields"], r[6:]))
            hist.setdefault(pid, []).append(stats)
            st = opp_state.setdefault(opp, [0, 0])
            st[0] += stats[cfg["opp_stat"]] or 0
            st[1] += 1
            latest_team[pid] = team
            latest_name[pid] = pname

        out = []
        for home, away in schedule:
            for team, opp, is_home in ((home, away, True), (away, home, False)):
                for pid, t in latest_team.items():
                    if t != team:
                        continue
                    h = hist.get(pid, [])
                    if not eligible(cfg, h):
                        continue
                    st = opp_state.get(opp)
                    opp_allowed = (st[0] / st[1]) if st and st[1] > 0 else None
                    feat = market_features(self.mkt, h, opp_allowed, is_home)
                    out.append((pid, latest_name[pid], team, opp, target_week, feat))
        return out


def score(bst, feats_order, feat_dicts, xgb):
    X = np.array([[fd.get(c) if fd.get(c) is not None else g.NAN for c in feats_order]
                  for fd in feat_dicts], dtype=np.float32)
    itr = (0, bst.best_iteration + 1)
    return np.asarray(
        bst.predict(xgb.DMatrix(X, feature_names=feats_order), iteration_range=itr),
        dtype=float)


def fit_serving_platt(con, mkt, bst, xgb, serving_season, target_week):
    """Warmup pool (previous completed season's pick_val_cut slice) + the
    serving season's completed eligible rows before target_week -- the
    validated walk-forward pool, generalized."""
    cfg = MARKETS[mkt]
    seasons = [r[0] for r in con.execute(
        "SELECT DISTINCT season FROM player_games WHERE season < ? ORDER BY season DESC",
        (serving_season,))]
    if not seasons:
        raise RuntimeError(f"no completed season before {serving_season} in db")
    warm_season = seasons[0]

    warm = SeasonEngine(con, mkt, warm_season).replay()
    warm_tuples = [(row[4],) for row in warm]  # (week,) for pick_val_cut
    cut = g.pick_val_cut([(None, w) for (w,) in warm_tuples])
    warm_slice = [row for row in warm if row[4] >= cut]

    cur = SeasonEngine(con, mkt, serving_season).replay()
    cur_seen = [row for row in cur if row[4] < target_week]

    pool = warm_slice + cur_seen
    pool_raw = score(bst, cfg["features"], [row[5] for row in pool], xgb)
    line = cfg["line"]
    pool_y = np.array([1.0 if row[6] >= line + 0.5 else 0.0 for row in pool])
    a, b = fit_platt(pool_raw, pool_y)
    if a <= 0:
        a, b = 1.0, 0.0
    return a, b, {"warmup_season": warm_season, "warmup_cut_week": int(cut),
                  "warmup_n": len(warm_slice), "current_season_n": len(cur_seen)}


def selftest(con, xgb):
    """(a) exact feature parity with the validated baseline.sqlite for the
    full 2024 season, both markets; (b) walk-forward probability parity
    with the gated/stability numbers."""
    print("SELFTEST: serving engine vs validated baselines (2024)")
    ok = True
    for mkt, cfg in MARKETS.items():
        rows = SeasonEngine(con, mkt, 2024).replay()
        db_path, table = cfg["baseline_table"]
        bcon = sqlite3.connect(f"file:{REPO / db_path}?mode=ro", uri=True)
        cols = ["player_id", "week"] + cfg["features"] + ["over_line"]
        brows = bcon.execute(
            f"SELECT {', '.join(cols)} FROM {table} WHERE season=2024").fetchall()
        bcon.close()
        bmap = {(r[0], r[1]): r[2:] for r in brows}
        if len(rows) != len(brows):
            print(f"  {mkt}: ROW COUNT MISMATCH engine={len(rows)} baseline={len(brows)}")
            ok = False
            continue
        worst = 0.0
        for (pid, _, _, _, week, feat, actual) in rows:
            ref = bmap.get((pid, week))
            assert ref is not None, f"{mkt}: engine row ({pid},{week}) missing from baseline"
            for i, c in enumerate(cfg["features"]):
                a, b = feat.get(c), ref[i]
                if a is None and b is None:
                    continue
                assert a is not None and b is not None, f"{mkt} {pid} w{week} {c}: {a} vs {b}"
                worst = max(worst, abs(a - b))
            target = 1 if actual >= cfg["line"] + 0.5 else 0
            assert target == ref[-1], f"{mkt} {pid} w{week}: target {target} vs {ref[-1]}"
        print(f"  {mkt}: {len(rows)} rows, feature parity exact "
              f"(max abs diff {worst:.2e}), targets match")

        # (b) walk-forward probability parity with the stability run
        bst = xgb.Booster(); bst.load_model(str(cfg["model_dir"] / f"{cfg['stem']}.json"))
        by_week = {}
        for row in rows:
            by_week.setdefault(row[4], []).append(row)
        warm = SeasonEngine(con, mkt, 2023).replay()
        cut = g.pick_val_cut([(None, r[4]) for r in warm])
        warm_slice = [r for r in warm if r[4] >= cut]
        wr = score(bst, cfg["features"], [r[5] for r in warm_slice], xgb)
        wy = np.array([1.0 if r[6] >= cfg["line"] + 0.5 else 0.0 for r in warm_slice])
        probs, ys = [], []
        seen_rows = []
        for w in sorted(by_week):
            pool_rows = seen_rows
            pr = score(bst, cfg["features"], [r[5] for r in pool_rows], xgb) if pool_rows else np.empty(0)
            py = np.array([1.0 if r[6] >= cfg["line"] + 0.5 else 0.0 for r in pool_rows])
            a, b = fit_platt(np.concatenate([wr, pr]), np.concatenate([wy, py]))
            if a <= 0:
                a, b = 1.0, 0.0
            wk_rows = by_week[w]
            raw = score(bst, cfg["features"], [r[5] for r in wk_rows], xgb)
            probs.extend(apply_platt(raw, a, b).tolist())
            ys.extend(1.0 if r[6] >= cfg["line"] + 0.5 else 0.0 for r in wk_rows)
            seen_rows = seen_rows + wk_rows
        m = g.metrics(probs, ys)
        ref = json.loads((cfg["model_dir"] /
                          f"nfl_walkforward_stability_confirmation_a_{mkt}_report.json").read_text())
        print(f"  {mkt}: walk-forward replay AUC={m['auc']:.4f} ECE={m['ece']:.4f} "
              f"(stability report AUC={ref['point_auc']:.4f})")
        assert abs(m["auc"] - ref["point_auc"]) < 0.002, \
            f"{mkt}: serving walk-forward diverges from validated run"
    print(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    return ok


def infer_target(con, today):
    """Next (season, week) with any unplayed game on/after today."""
    r = con.execute(
        "SELECT season, week, MIN(game_date) FROM games WHERE game_date >= ? "
        "GROUP BY season, week ORDER BY game_date LIMIT 1", (today,)).fetchone()
    return (r[0], r[1]) if r else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_DEFAULT))
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--out", default=str(DOCS / "nfl_predictions.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    import xgboost as xgb

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    print("NFL_SERVING_BUILDER_A\n=====================")

    if args.selftest:
        ok = selftest(con, xgb)
        con.close()
        return 0 if ok else 1

    if args.season and args.week:
        season, week = args.season, args.week
    else:
        season, week = infer_target(con, date.today().isoformat())
    if season is None:
        print("no upcoming games found in the schedule -- writing empty board")
        payload = {"generated_at_utc": now_utc(), "season": None, "week": None,
                   "picks": [], "note": "no upcoming games in foundation schedule; "
                   "refresh the foundation db (new season schedule not ingested yet)"}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        return 0

    print(f"target: season {season} week {week}")
    schedule = con.execute(
        "SELECT home_team, away_team FROM games WHERE season=? AND week=?",
        (season, week)).fetchall()
    print(f"scheduled games: {len(schedule)}")

    picks = []
    market_meta = {}
    for mkt, cfg in MARKETS.items():
        bst = xgb.Booster(); bst.load_model(str(cfg["model_dir"] / f"{cfg['stem']}.json"))
        feat_cols = json.loads((cfg["model_dir"] / f"{cfg['stem']}_columns.json").read_text())
        assert feat_cols == cfg["features"]

        cand = SeasonEngine(con, mkt, season).asof_future(week, schedule)
        if not cand:
            print(f"  {mkt}: no eligible players (expected for weeks 1-{MIN_PRIOR_GAMES})")
            market_meta[mkt] = {"eligible": 0}
            continue

        a, b, pool_info = fit_serving_platt(con, mkt, bst, xgb, season, week)
        raw = score(bst, cfg["features"], [c[5] for c in cand], xgb)
        cal = apply_platt(raw, a, b)
        print(f"  {mkt}: {len(cand)} eligible  platt a={a:.3f} b={b:+.3f}  "
              f"pool={pool_info}")
        market_meta[mkt] = {"eligible": len(cand), "platt": {"a": a, "b": b},
                             "calibration_pool": pool_info,
                             "validation": cfg["verdicts"]}
        for (pid, pname, team, opp, _, feat), rp, cp in zip(cand, raw, cal):
            picks.append({
                "market": mkt, "player_id": pid, "player": pname,
                "team": team, "opponent": opp, "season": season, "week": week,
                "line": cfg["line"],
                "pick": f"{'OVER' if cp >= 0.5 else 'UNDER'} {cfg['line']}",
                "model_prob": round(float(max(cp, 1 - cp)), 4),
                "prob_over": round(float(cp), 4),
                "raw_prob_over": round(float(rp), 4),
                "games_played": feat["games_played"],
            })

    picks.sort(key=lambda p: -p["model_prob"])
    payload = {
        "generated_at_utc": now_utc(), "season": season, "week": week,
        "builder": "NFL_SERVING_BUILDER_A",
        "design": "frozen champion + weekly walk-forward Platt (validated 2024)",
        "markets": market_meta,
        "note": "predictions-first: no odds. Eligibility is stats-based and "
                "cannot see injuries/inactives for the upcoming game.",
        "picks": picks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    hist = out.parent / f"nfl_predictions_{season}_w{week:02d}.json"
    hist.write_text(json.dumps(payload, indent=2))
    print(f"\n{len(picks)} picks written to {out} (+ {hist.name})")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
