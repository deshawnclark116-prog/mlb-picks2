#!/usr/bin/env python3
"""
NFL_PER_PLAYER_LINE_GATE_A

Real question raised by a user looking at the live board: rushing_yards
and receiving_yards (in-season AND both early-season variants) all judge
every player against the exact same fixed 49.5-yard line -- true, checked
directly in nfl_serving_builder_a.py, and true because this repo has no
live NFL odds source at all ("predictions-first: no odds anywhere"), so
there's no real per-player market line to grab. 49.5 was chosen as a
fixed backtest threshold when the binary classifier was built, not
derived from anything player-specific.

This tests whether a genuinely PER-PLAYER line -- each player's own
recent-average yards, refined by a real regression model using their
full feature set -- is something worth building, using the exact same
regression-style methodology this repo already prefers for continuous
targets (tennis_total_games_champion_gate_a.py, CFB's Monte Carlo
simulators): MAE + paired-error bootstrap, not a binary hit-rate.

Two arms, scored on the SAME real, already-validated 2024 holdout the
shipped classifiers use (nfl_rushing_yards_champion_gate_a.py /
nfl_receiving_yards_champion_walkforward_gate_a_work's own dev=2023/
holdout=2024 split -- reusing nfl_models/nfl_*_clean_baseline_a_work/
baseline.sqlite directly, same real engineered features, same real
population, apples-to-apples with what's already shipped):

  naive        each player's own recent3_avg_<stat> feature, unmodified
               -- the simplest possible "per-player line" (no model,
               just their raw recent average). This is the honest
               floor: what you get for free without building anything.
  challenger   XGBoost regression (reg:squarederror) on the full
               existing feature set (season_avg, recent3, recent5,
               volume, matchup, home/away, games_played), predicting
               actual yards directly.

Pre-registered pass bar (written before this script has ever been run):
  1. Model HOLDOUT MAE beats naive HOLDOUT MAE by >= 10% (relative) --
     set higher than tennis's 5% bar since single-game yardage is
     noisier than a match total; a real product change should clear a
     real margin, not a marginal one.
  2. Model confidently beats naive: paired bootstrap over per-player-
     week (|naive_error| - |model_error|), P(model better) >= 0.90.
  3. n >= 200 in the holdout (sample size floor).

A pass means: a per-player regression line is real, validated signal
beyond what a player's own raw recent average already gives for free --
worth building into a served per-player line. A fail means: the extra
model complexity isn't earning its keep over "just use their recent
average," which would still be a real, useful finding (a per-player
line derived from recent3_avg alone, no model needed, if the naive arm
alone already looks reasonable).

Read-only against nfl_models/nfl_*_clean_baseline_a_work/baseline.sqlite.
Writes only its own report file.

Run
---
python -u nfl_per_player_line_gate_a.py
"""
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
OUT_PATH = REPO / "nfl_per_player_line_gate_a_report.json"
SEED = 13
NAN = float("nan")

MARKETS = {
    "rushing_yards": {
        "db": REPO / "nfl_models" / "nfl_rushing_yards_clean_baseline_a_work" / "baseline.sqlite",
        "table": "nfl_rushing_yards_baseline",
        "actual_col": "actual_rushing_yards",
        "naive_col": "recent3_avg_rush_yards",
        "features": ["season_avg_rush_yards", "recent3_avg_rush_yards", "recent5_avg_rush_yards",
                     "season_avg_carries", "recent3_avg_carries", "yards_per_carry",
                     "opp_rush_yards_allowed_per_game", "is_home", "games_played"],
    },
    "receiving_yards": {
        "db": REPO / "nfl_models" / "nfl_receiving_yards_clean_baseline_a_work" / "baseline.sqlite",
        "table": "nfl_receiving_yards_baseline",
        "actual_col": "actual_receiving_yards",
        "naive_col": "recent3_avg_rec_yards",
        "features": ["season_avg_rec_yards", "recent3_avg_rec_yards", "recent5_avg_rec_yards",
                     "season_avg_targets", "recent3_avg_targets", "yards_per_target", "catch_rate",
                     "opp_rec_yards_allowed_per_game", "is_home", "games_played"],
    },
}
DEV_SEASON = 2023
HOLDOUT_SEASON = 2024


def load_rows(cfg):
    con = sqlite3.connect(f"file:{cfg['db']}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(f"SELECT * FROM {cfg['table']} ORDER BY season, week").fetchall()
    con.close()
    return [dict(r) for r in rows]


def mat(rows, feature_cols, actual_col, xgb):
    X = np.array([[r.get(c, NAN) for c in feature_cols] for r in rows], dtype=np.float32)
    y = np.array([r[actual_col] for r in rows], dtype=np.float32)
    return xgb.DMatrix(X, label=y, feature_names=feature_cols)


def bootstrap_p_paired_worse(err_a, err_b, seed=SEED, b=5000):
    """P(mean(err_a) >= mean(err_b)) via paired bootstrap over matched
    (naive_error, model_error) pairs -- same per-observation pairing as
    tennis_total_games_champion_gate_a.py, avoids the pseudo-replication
    trap of resampling errors independently."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(err_a) - np.asarray(err_b)
    n = len(diffs)
    means = rng.choice(diffs, size=(b, n), replace=True).mean(axis=1)
    return float(np.mean(means <= 0))


def run_market(mkt_key, cfg):
    import xgboost as xgb
    print(f"\n{'='*70}\n{mkt_key}\n{'='*70}")

    rows = load_rows(cfg)
    dev_rows = [r for r in rows if r["season"] == DEV_SEASON]
    hol_rows = [r for r in rows if r["season"] == HOLDOUT_SEASON]
    print(f"dev {DEV_SEASON}: {len(dev_rows)} rows   holdout {HOLDOUT_SEASON}: {len(hol_rows)} rows")

    n = len(dev_rows)
    cut = int(n * 0.85)
    tr, va = dev_rows[:cut], dev_rows[cut:]
    print(f"  train={len(tr)}  internal val={len(va)} (early stopping only)")

    feat_cols, actual_col, naive_col = cfg["features"], cfg["actual_col"], cfg["naive_col"]
    params = {"objective": "reg:squarederror", "eval_metric": "mae", "max_depth": 3,
              "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 5, "seed": SEED}
    bst = xgb.train(params, mat(tr, feat_cols, actual_col, xgb), num_boost_round=800,
                     evals=[(mat(va, feat_cols, actual_col, xgb), "val")],
                     early_stopping_rounds=40, verbose_eval=False)
    itr = (0, bst.best_iteration + 1)

    model_preds = bst.predict(mat(hol_rows, feat_cols, actual_col, xgb), iteration_range=itr)
    actuals = np.array([r[actual_col] for r in hol_rows], dtype=np.float64)
    naive_preds = np.array([r.get(naive_col) or 0.0 for r in hol_rows], dtype=np.float64)

    model_err = np.abs(actuals - np.asarray(model_preds, dtype=np.float64))
    naive_err = np.abs(actuals - naive_preds)
    model_mae, naive_mae = float(model_err.mean()), float(naive_err.mean())
    mae_improvement_pct = (naive_mae - model_mae) / naive_mae if naive_mae else None
    p_model_worse = bootstrap_p_paired_worse(naive_err, model_err)

    print(f"  naive (recent3 avg) HOLDOUT MAE: {naive_mae:.2f}")
    print(f"  model (XGBoost)     HOLDOUT MAE: {model_mae:.2f}")
    print(f"  MAE improvement: {mae_improvement_pct*100:.2f}%" if mae_improvement_pct is not None else "  MAE improvement: n/a")
    print(f"  P(model worse than naive): {p_model_worse:.4f}  (pass needs <= 0.10)")

    imp = bst.get_score(importance_type="gain")
    print("  feature importance (gain):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"    {k:32s} {v:9.2f}")

    check1 = mae_improvement_pct is not None and mae_improvement_pct >= 0.10
    check2 = p_model_worse <= 0.10
    check3 = len(hol_rows) >= 200
    passed = check1 and check2 and check3
    verdict = f"{mkt_key.upper()}_PER_PLAYER_LINE_{'PASSES_GATE' if passed else 'DOES_NOT_CLEAR_GATE'}"
    print(f"\n  GATE: MAE improvement>=10% -> {check1}   "
          f"P(worse)<=0.10 -> {check2}   n>=200 -> {check3}")
    print(f"  VERDICT: {verdict}")

    return {
        "n_dev": len(dev_rows), "n_holdout": len(hol_rows),
        "naive_holdout_mae": round(naive_mae, 3), "model_holdout_mae": round(model_mae, 3),
        "mae_improvement_pct_vs_naive_holdout": round(mae_improvement_pct, 4) if mae_improvement_pct is not None else None,
        "p_model_worse_than_naive_holdout": round(p_model_worse, 4),
        "importance": imp, "passed": passed, "verdict": verdict,
    }


def main():
    print("NFL_PER_PLAYER_LINE_GATE_A\n==========================")
    report = {"script": "NFL_PER_PLAYER_LINE_GATE_A", "dev_season": DEV_SEASON,
               "holdout_season": HOLDOUT_SEASON, "markets": {}}
    for mkt_key, cfg in MARKETS.items():
        report["markets"][mkt_key] = run_market(mkt_key, cfg)

    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\n\n{'#'*70}\nSUMMARY\n{'#'*70}")
    for mkt, r in report["markets"].items():
        print(f"  {mkt:18s} MAE improvement={r['mae_improvement_pct_vs_naive_holdout']}  "
              f"P(worse)={r['p_model_worse_than_naive_holdout']}  {r['verdict']}")
    print(f"\nreport: {OUT_PATH}")

    any_pass = any(r["passed"] for r in report["markets"].values())
    return 0 if any_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
