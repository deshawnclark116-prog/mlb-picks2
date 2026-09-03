#!/usr/bin/env python3
"""
CFB_PRIOR_SEASON_PRODUCTION_BUILDER_A

Final production models for all four markets validated in
cfb_prior_season_early_gate_a.py -- unlike the NFL analog (where only
rushing_yards cleared the bar), all four CFB markets passed cleanly on the
2025 holdout: rushing_yards AUC 0.6364, receiving_yards 0.6804,
passing_yards 0.6476, passing_touchdowns 0.6245.

Retrains on ALL available seasons (2023-2025 scored population, using
2022-2024 as their respective prior seasons) rather than the dev-only
split used for validation -- the holdout already proved this generalizes;
a shipped model should use every real data point available. Same features,
same eligibility (position-gated, no current-season game requirement --
that's the whole point) as the validated gate script.

Writes cfb_models/cfb_prior_season_<market>.json +
cfb_models/cfb_prior_season_<market>_columns.json for
cfb_serving_builder_a.py to load.

Run
---
python -u cfb_prior_season_production_builder_a.py
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
LINES = {
    "rushing_yards": 69.5,
    "receiving_yards": 59.5,
    "passing_yards": 214.5,
    "passing_touchdowns": 1.5,
}
ALL_SEASONS = [2023, 2024, 2025]
NAN = float("nan")

MARKETS = {
    "rushing_yards": {"position": "RB", "stat": "rushing_yards", "rate": "carries"},
    "receiving_yards": {"position": "WR", "stat": "receiving_yards", "rate": "receptions"},
    "passing_yards": {"position": "QB", "stat": "passing_yards", "rate": "pass_attempts"},
    "passing_touchdowns": {"position": "QB", "stat": "passing_touchdowns", "rate": "pass_attempts"},
}
FEATURE_COLS = ["prior_season_avg_stat", "prior_season_games", "prior_season_avg_rate"]
VALIDATED_HOLDOUT_AUC = {
    "rushing_yards": 0.6364, "receiving_yards": 0.6804,
    "passing_yards": 0.6476, "passing_touchdowns": 0.6245,
}


def load_weeks123(conn, season, position):
    return conn.execute("""
        SELECT player_id, week, carries, rushing_yards, receptions, receiving_yards,
               pass_attempts, passing_yards, passing_touchdowns
        FROM player_games WHERE season = ? AND week <= ? AND position = ?
    """, (season, MAX_WEEK, position)).fetchall()


def load_full_season_by_player(conn, season, position):
    rows = conn.execute("""
        SELECT player_id, carries, rushing_yards, receptions, receiving_yards,
               pass_attempts, passing_yards, passing_touchdowns
        FROM player_games WHERE season = ? AND position = ?
    """, (season, position)).fetchall()
    by_pid = {}
    for r in rows:
        pid = r[0]
        by_pid.setdefault(pid, []).append({
            "carries": r[1] or 0, "rushing_yards": r[2] or 0,
            "receptions": r[3] or 0, "receiving_yards": r[4] or 0,
            "pass_attempts": r[5] or 0, "passing_yards": r[6] or 0,
            "passing_touchdowns": r[7] or 0,
        })
    return by_pid


def build_rows(cur_rows, prior_by_pid, cfg, line):
    stat_field, rate_field = cfg["stat"], cfg["rate"]
    idx = {"carries": 2, "rushing_yards": 3, "receptions": 4, "receiving_yards": 5,
           "pass_attempts": 6, "passing_yards": 7, "passing_touchdowns": 8}
    out = []
    for r in cur_rows:
        pid = r[0]
        actual = r[idx[stat_field]]
        actual = actual if actual is not None else 0
        games = prior_by_pid.get(pid)
        if games:
            n = len(games)
            feat = {
                "prior_season_avg_stat": sum(g[stat_field] for g in games) / n,
                "prior_season_games": n,
                "prior_season_avg_rate": sum(g[rate_field] for g in games) / n,
            }
        else:
            feat = {c: NAN for c in FEATURE_COLS}
        out.append({**feat, "over_line": 1 if actual >= (line + 0.5) else 0})
    return out


def mat(rows, xgb):
    X = np.array([[r.get(c, NAN) for c in FEATURE_COLS] for r in rows], dtype=np.float32)
    y = np.array([r["over_line"] for r in rows], dtype=np.float32)
    return xgb.DMatrix(X, label=y, feature_names=FEATURE_COLS)


def main():
    import xgboost as xgb
    print("CFB_PRIOR_SEASON_PRODUCTION_BUILDER_A\n======================================")
    conn = sqlite3.connect(f"file:{DB_DEFAULT}?mode=ro", uri=True)
    MODEL_DIR.mkdir(exist_ok=True)

    for mkt_key, cfg in MARKETS.items():
        line = LINES[mkt_key]
        rows = []
        for season in ALL_SEASONS:
            cur = load_weeks123(conn, season, cfg["position"])
            prior = load_full_season_by_player(conn, season - 1, cfg["position"])
            rows += build_rows(cur, prior, cfg, line)
        matched = sum(1 for r in rows if not np.isnan(r["prior_season_avg_stat"]))
        print(f"\n{mkt_key}: {len(rows)} rows (all {ALL_SEASONS}, weeks 1-{MAX_WEEK}, "
              f"{cfg['position']}) -- {matched} matched to prior season ({matched/len(rows)*100:.1f}%)")

        n = len(rows)
        cut = int(n * 0.85)
        tr, va = rows[:cut], rows[cut:]
        print(f"  train={len(tr)}  internal val={len(va)} (early stopping only, not a real holdout -- "
              f"generalization already proven in cfb_prior_season_early_gate_a.py)")

        params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3,
                  "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
                  "min_child_weight": 5, "seed": 13}
        bst = xgb.train(params, mat(tr, xgb), num_boost_round=800, evals=[(mat(va, xgb), "val")],
                         early_stopping_rounds=40, verbose_eval=False)
        print(f"  best_iteration={bst.best_iteration}")

        bst.save_model(str(MODEL_DIR / f"cfb_prior_season_{mkt_key}.json"))
        (MODEL_DIR / f"cfb_prior_season_{mkt_key}_columns.json").write_text(json.dumps(FEATURE_COLS))
        manifest = {
            "script": "CFB_PRIOR_SEASON_PRODUCTION_BUILDER_A",
            "market": mkt_key, "position": cfg["position"], "line": line,
            "trained_on_seasons": ALL_SEASONS, "max_week": MAX_WEEK,
            "feature_cols": FEATURE_COLS, "best_iteration": bst.best_iteration,
            "validated_holdout_auc": VALIDATED_HOLDOUT_AUC[mkt_key],
            "validation_script": "cfb_prior_season_early_gate_a.py",
        }
        (MODEL_DIR / f"cfb_prior_season_{mkt_key}_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"  saved: {MODEL_DIR / f'cfb_prior_season_{mkt_key}.json'}")

    conn.close()
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
