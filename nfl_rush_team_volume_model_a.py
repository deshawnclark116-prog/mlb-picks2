#!/usr/bin/env python3
"""
NFL_RUSH_TEAM_VOLUME_MODEL_A

Stage A of the rushing_yards volume/efficiency rebuild (see
nfl_rushing_volume_premise_check_a.py for the premises this design rests
on: game script weakly-but-really predicts team rush volume, R^2=0.0665
with just spread+total; carry share is more stable than raw carries).

This stage predicts TEAM rush volume (total team carries in a game) --
not any individual player. Two arms, evaluated on the untouched 2024
holdout (2023 = train):

  constant    the team's own rolling season-to-date average carries
              (a persistence baseline -- deliberately not the population
              mean, since "this team runs this much" is already a strong
              prior before adding game-script context)
  challenger  XGBoost regression on constant's features + game script
              (own_spread, total_line) + opponent run-funnel context

Pre-registered pass (written before this script has ever been run):
  1. challenger RMSE beats constant RMSE (game script adds real signal
     beyond "how much does this team usually run")
  2. challenger R^2 on holdout >= 0.10 (some real explanatory power,
     not just noise -- premise check's raw 2-feature OLS already hit
     0.0665, so a fuller model should clear a modestly higher bar)

Strict D-1: every feature for a team-game uses only that team's (and
that opponent's) games with a strictly earlier week in the SAME season
(no cross-season carryover, matching every other script in this repo).

Feature set
-----------
  season_avg_team_carries, recent3_avg_team_carries   (own rolling volume)
  own_spread            (game-script: negative = this team favored,
                          sign convention verified empirically via
                          moneyline, never assumed)
  total_line
  opp_carries_allowed_per_game   (as-of: mean carries allowed by this
                                   opponent across all teams they've
                                   faced so far this season)
  is_home, games_played

Target: team_carries (actual total rush attempts by this team in this game)

Read-only on nfl_model.sqlite. Writes only its own work dir (model +
report), no production state changed.

Run (GitHub Actions runner; nflverse data is not reachable from this
sandbox directly, same as every other NFL script here)
------------------------------------------------------
python -u nfl_player_games_foundation_a.py --season 2023 2024 2025 --db nfl_models/nfl_model.sqlite
python -u nfl_rush_team_volume_model_a.py
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

REPO = Path(__file__).resolve().parent
SOURCE_DEFAULT = REPO / "nfl_models" / "nfl_model.sqlite"
WORKDIR_DEFAULT = REPO / "nfl_models" / "nfl_rush_team_volume_model_a_work"

FEATURES = [
    "season_avg_team_carries", "recent3_avg_team_carries",
    "own_spread", "total_line", "opp_carries_allowed_per_game",
    "is_home", "games_played",
]
PARAMS = {"objective": "reg:squarederror", "eval_metric": "rmse", "max_depth": 3,
          "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
          "min_child_weight": 5, "seed": 13}
GATE = {"min_r2": 0.10}
DEV_SEASON = 2023
HOLDOUT_SEASON = 2024


def determine_spread_sign(conn):
    """Empirically derive whether spread_line is home-signed or away-signed
    from moneyline (unambiguous: negative = favorite). Never assume this --
    the premise check caught an early run getting it backwards from memory."""
    rows = conn.execute("""
        SELECT spread_line, home_moneyline, away_moneyline FROM games
        WHERE spread_line IS NOT NULL AND home_moneyline IS NOT NULL
              AND away_moneyline IS NOT NULL AND home_moneyline != away_moneyline
        LIMIT 1000
    """).fetchall()
    neg_home_fav = neg_n = pos_home_fav = pos_n = 0
    for spread, hml, aml in rows:
        home_favored = hml < aml
        if spread < 0:
            neg_n += 1; neg_home_fav += int(home_favored)
        elif spread > 0:
            pos_n += 1; pos_home_fav += int(home_favored)
    home_signed = (neg_home_fav / max(neg_n, 1)) > (pos_home_fav / max(pos_n, 1))
    return home_signed


def build_rows(conn, home_signed):
    team_games = conn.execute("""
        SELECT pg.game_id, pg.team, pg.opponent, pg.season, pg.week, pg.is_home,
               SUM(pg.carries) as team_carries,
               g.home_team, g.spread_line, g.total_line
        FROM player_games pg
        JOIN games g ON pg.game_id = g.game_id
        WHERE pg.season_type = 'REG' AND pg.carries IS NOT NULL
              AND g.spread_line IS NOT NULL AND g.total_line IS NOT NULL
        GROUP BY pg.game_id, pg.team
        ORDER BY pg.season, pg.week
    """).fetchall()

    # opponent as-of context: mean carries allowed per game, built up
    # week-by-week across all teams the opponent has faced so far this
    # season (strictly earlier weeks only) -- same as-of pattern as the
    # existing rushing_yards clean baseline's opp_rush_yards_allowed_per_game.
    by_season_week = {}
    for r in team_games:
        by_season_week.setdefault((r[3], r[4]), []).append(r)

    opp_state = {}
    opp_asof = {}
    for (season, week) in sorted(by_season_week):
        wk_rows = by_season_week[(season, week)]
        for r in wk_rows:
            gid, team, opp = r[0], r[1], r[2]
            key = (gid, team)
            st = opp_state.get((season, opp))
            opp_asof[key] = (st[0] / st[1]) if st and st[1] > 0 else None
        for r in wk_rows:
            opp, tc = r[2], r[6]
            st = opp_state.setdefault((season, opp), [0, 0])
            st[0] += tc
            st[1] += 1

    # own rolling volume, reset each season (matches every other script
    # here -- last season's carries don't inform this season's cold start)
    team_hist = {}
    out = []
    for r in team_games:
        gid, team, opp, season, week, is_home, tc, home_team, spread, total = r
        key = (team, season)
        hist = team_hist.setdefault(key, [])
        n_prior = len(hist)
        if n_prior >= 1:
            recent3 = hist[-3:]
            own_spread = spread if (team == home_team) == home_signed else -spread
            feat = {
                "season_avg_team_carries": sum(hist) / n_prior,
                "recent3_avg_team_carries": sum(recent3) / len(recent3),
                "own_spread": own_spread,
                "total_line": total,
                "opp_carries_allowed_per_game": opp_asof.get((gid, team)),
                "is_home": 1.0 if is_home else 0.0,
                "games_played": float(n_prior),
            }
            out.append({
                "game_id": gid, "team": team, "opponent": opp,
                "season": season, "week": week,
                **feat, "team_carries": float(tc),
            })
        hist.append(tc)
    return out


def r2_score(y, pred):
    y = np.asarray(y, dtype=float); pred = np.asarray(pred, dtype=float)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def rmse(y, pred):
    y = np.asarray(y, dtype=float); pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(SOURCE_DEFAULT))
    ap.add_argument("--workdir", default=str(WORKDIR_DEFAULT))
    args = ap.parse_args()

    import xgboost as xgb

    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{args.source}?mode=ro", uri=True)

    print("NFL_RUSH_TEAM_VOLUME_MODEL_A\n=============================")
    home_signed = determine_spread_sign(conn)
    print(f"spread_line sign convention: {'HOME' if home_signed else 'AWAY'}-signed")

    rows = build_rows(conn, home_signed)
    conn.close()
    print(f"total eligible team-game rows: {len(rows)}")

    by_season = {}
    for r in rows:
        by_season.setdefault(r["season"], []).append(r)
    for s in sorted(by_season):
        print(f"  season {s}: {len(by_season[s])} rows")

    dev = by_season.get(DEV_SEASON, [])
    hol = by_season.get(HOLDOUT_SEASON, [])
    if not dev or not hol:
        print(f"ERROR: missing dev({DEV_SEASON})={len(dev)} or holdout({HOLDOUT_SEASON})={len(hol)} rows")
        return 1

    # internal val split: last 20% of dev weeks, for early stopping only
    dev_weeks = sorted({(r["season"], r["week"]) for r in dev})
    cut = dev_weeks[int(len(dev_weeks) * 0.8)]
    tr = [r for r in dev if (r["season"], r["week"]) < cut]
    va = [r for r in dev if (r["season"], r["week"]) >= cut]
    print(f"train: {len(tr)}  internal_val: {len(va)}  holdout({HOLDOUT_SEASON}): {len(hol)}")

    def mat(rowset):
        X = np.array([[r[f] for f in FEATURES] for r in rowset], dtype=np.float32)
        y = np.array([r["team_carries"] for r in rowset], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=FEATURES), y

    dtr, ytr = mat(tr)
    dva, yva = mat(va)
    dhol, yhol = mat(hol)

    booster = xgb.train(PARAMS, dtr, num_boost_round=400, evals=[(dva, "val")],
                        early_stopping_rounds=30, verbose_eval=False)
    pred_hol = booster.predict(dhol, iteration_range=(0, booster.best_iteration + 1))

    # constant arm: predict the team's own rolling season-to-date average
    # (already one of the challenger's own features, so this is a fair,
    # strong persistence baseline -- not a weak strawman)
    const_pred = np.array([r["season_avg_team_carries"] for r in hol], dtype=float)

    r2_challenger = r2_score(yhol, pred_hol)
    r2_constant = r2_score(yhol, const_pred)
    rmse_challenger = rmse(yhol, pred_hol)
    rmse_constant = rmse(yhol, const_pred)

    print(f"\nHOLDOUT ({HOLDOUT_SEASON}), n={len(hol)}")
    print(f"  {'arm':12s} {'R^2':>8s} {'RMSE':>8s}")
    print(f"  {'constant':12s} {r2_constant:8.4f} {rmse_constant:8.4f}")
    print(f"  {'challenger':12s} {r2_challenger:8.4f} {rmse_challenger:8.4f}")

    beats_rmse = rmse_challenger < rmse_constant
    clears_r2 = r2_challenger >= GATE["min_r2"]
    passed = beats_rmse and clears_r2
    print(f"\nGATE: beats_constant_rmse={beats_rmse}  r2>={GATE['min_r2']}: {clears_r2}  -> {'PASS' if passed else 'FAIL'}")

    imp = booster.get_score(importance_type="gain")
    print("\nfeature importance (gain):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"    {k:28s} {v:9.2f}")

    booster.save_model(str(work / "nfl_rush_team_volume.json"))
    (work / "nfl_rush_team_volume_columns.json").write_text(json.dumps(FEATURES))
    report = {
        "script": "NFL_RUSH_TEAM_VOLUME_MODEL_A",
        "features": FEATURES,
        "dev_season": DEV_SEASON, "holdout_season": HOLDOUT_SEASON,
        "n_train": len(tr), "n_val": len(va), "n_holdout": len(hol),
        "best_iteration": booster.best_iteration,
        "holdout": {
            "constant": {"r2": r2_constant, "rmse": rmse_constant},
            "challenger": {"r2": r2_challenger, "rmse": rmse_challenger},
        },
        "gate": GATE,
        "beats_constant_rmse": bool(beats_rmse),
        "clears_min_r2": bool(clears_r2),
        "passed": bool(passed),
        "feature_importance_gain": imp,
        "spread_sign_convention": "HOME" if home_signed else "AWAY",
    }
    (work / "nfl_rush_team_volume_model_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {work / 'nfl_rush_team_volume_model_a_report.json'}")
    print("Read-only on nfl_model.sqlite. No production state changed.")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
