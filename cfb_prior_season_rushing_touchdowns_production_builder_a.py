#!/usr/bin/env python3
"""
CFB_PRIOR_SEASON_RUSHING_TOUCHDOWNS_PRODUCTION_BUILDER_A

Final production model for rushing_touchdowns' weeks-1-3 bootstrap,
validated in cfb_prior_season_rushing_touchdowns_gate_a.py (AUC 0.6201
on the 2024 holdout). Retrains on ALL available non-2025 scored seasons
(2019-2024, using 2018-2023 as their respective prior seasons) rather
than the dev-only split used for validation -- the holdout already
proved this generalizes.

2025 excluded from training too, not just as a holdout: cfbfastR's
touchdown_player_id field is measurably incomplete for that season
specifically (documented in cfb_rushing_touchdowns_clean_baseline_a.py),
so its rushing-TD labels aren't trustworthy to train on either.

Writes cfb_models/cfb_prior_season_rushing_touchdowns.json +
cfb_models/cfb_prior_season_rushing_touchdowns_columns.json for
cfb_serving_builder_a.py to load.

Run
---
python -u cfb_prior_season_rushing_touchdowns_production_builder_a.py
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

DB_DEFAULT = "cfb_models/cfb_model.sqlite"
MODEL_DIR = Path("cfb_models")
MAX_WEEK = 3
LINE = 0.5
ALL_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
NAN = float("nan")

POSITION = "RB"
STAT_FIELD = "rushing_touchdowns"
RATE_FIELD = "carries"
FEATURE_COLS = ["prior_season_avg_stat", "prior_season_games", "prior_season_avg_rate"]
VALIDATED_HOLDOUT_AUC = 0.6201


def load_weeks123(conn, season, position):
    return conn.execute("""
        SELECT player_id, carries, rushing_touchdowns
        FROM player_games WHERE season = ? AND week <= ? AND position = ?
    """, (season, MAX_WEEK, position)).fetchall()


def load_full_season_by_player(conn, season, position):
    rows = conn.execute("""
        SELECT player_id, carries, rushing_touchdowns
        FROM player_games WHERE season = ? AND position = ?
    """, (season, position)).fetchall()
    by_pid = {}
    for pid, carries, td in rows:
        by_pid.setdefault(pid, []).append({"carries": carries or 0, "rushing_touchdowns": td or 0})
    return by_pid


def build_rows(cur_rows, prior_by_pid):
    out = []
    for pid, carries, td in cur_rows:
        actual = td if td is not None else 0
        games = prior_by_pid.get(pid)
        if games:
            n = len(games)
            feat = {
                "prior_season_avg_stat": sum(g[STAT_FIELD] for g in games) / n,
                "prior_season_games": n,
                "prior_season_avg_rate": sum(g[RATE_FIELD] for g in games) / n,
            }
        else:
            feat = {c: NAN for c in FEATURE_COLS}
        out.append({**feat, "over_line": 1 if actual >= (LINE + 0.5) else 0})
    return out


def mat(rows, xgb):
    X = np.array([[r.get(c, NAN) for c in FEATURE_COLS] for r in rows], dtype=np.float32)
    y = np.array([r["over_line"] for r in rows], dtype=np.float32)
    return xgb.DMatrix(X, label=y, feature_names=FEATURE_COLS)


def main():
    import xgboost as xgb
    print("CFB_PRIOR_SEASON_RUSHING_TOUCHDOWNS_PRODUCTION_BUILDER_A\n"
          "=========================================================")
    conn = sqlite3.connect(f"file:{DB_DEFAULT}?mode=ro", uri=True)
    MODEL_DIR.mkdir(exist_ok=True)

    rows = []
    for season in ALL_SEASONS:
        cur = load_weeks123(conn, season, POSITION)
        prior = load_full_season_by_player(conn, season - 1, POSITION)
        rows += build_rows(cur, prior)
    conn.close()

    matched = sum(1 for r in rows if not np.isnan(r["prior_season_avg_stat"]))
    print(f"rushing_touchdowns: {len(rows)} rows (all {ALL_SEASONS}, weeks 1-{MAX_WEEK}, {POSITION}) "
          f"-- {matched} matched to prior season ({matched/len(rows)*100:.1f}%)")

    n = len(rows)
    cut = int(n * 0.85)
    tr, va = rows[:cut], rows[cut:]
    print(f"  train={len(tr)}  internal val={len(va)} (early stopping only, not a real holdout -- "
          f"generalization already proven in cfb_prior_season_rushing_touchdowns_gate_a.py)")

    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3,
              "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 5, "seed": 13}
    bst = xgb.train(params, mat(tr, xgb), num_boost_round=800, evals=[(mat(va, xgb), "val")],
                     early_stopping_rounds=40, verbose_eval=False)
    print(f"  best_iteration={bst.best_iteration}")

    bst.save_model(str(MODEL_DIR / "cfb_prior_season_rushing_touchdowns.json"))
    (MODEL_DIR / "cfb_prior_season_rushing_touchdowns_columns.json").write_text(json.dumps(FEATURE_COLS))
    manifest = {
        "script": "CFB_PRIOR_SEASON_RUSHING_TOUCHDOWNS_PRODUCTION_BUILDER_A",
        "market": "rushing_touchdowns", "position": POSITION, "line": LINE,
        "trained_on_seasons": ALL_SEASONS, "max_week": MAX_WEEK,
        "feature_cols": FEATURE_COLS, "best_iteration": bst.best_iteration,
        "validated_holdout_auc": VALIDATED_HOLDOUT_AUC,
        "validation_script": "cfb_prior_season_rushing_touchdowns_gate_a.py",
    }
    (MODEL_DIR / "cfb_prior_season_rushing_touchdowns_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  saved: {MODEL_DIR / 'cfb_prior_season_rushing_touchdowns.json'}")
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
