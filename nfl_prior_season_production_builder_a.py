#!/usr/bin/env python3
"""
NFL_PRIOR_SEASON_PRODUCTION_BUILDER_A

Final production models for both markets validated in
nfl_prior_season_early_gate_a.py: rushing_yards (holdout AUC 0.7702) and
receiving_yards (holdout AUC 0.7778), both cleanly clearing the 0.58 bar.

Retrains on ALL available seasons (2023-2025 scored population, using
2022-2024 as their respective prior seasons) rather than the dev-only
split used for validation -- the holdout already proved this generalizes;
a shipped model should use every real data point available. Same
features, same eligibility (position-gated, no current-season game
requirement -- that's the whole point) as the validated gate script.

Writes nfl_models/nfl_prior_season_<market>.json +
nfl_models/nfl_prior_season_<market>_columns.json +
nfl_models/nfl_prior_season_<market>_manifest.json for
nfl_serving_builder_a.py to load. Frozen at build time (this repo's
established "frozen champion" pattern, same as the preseason-informed
rushing model) -- not retrained live in the workflow.

Run
---
python -u nfl_prior_season_production_builder_a.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

CSV_DIR = Path("/data/nflverse_csv")
MODEL_DIR = Path("nfl_models")
MAX_WEEK = 3
LINE = 49.5
ALL_SEASONS = [2023, 2024, 2025]
NAN = float("nan")

MARKETS = {
    "rushing_yards": {"position": "RB", "stat": "rushing_yards", "rate": "carries"},
    "receiving_yards": {"position": "WR", "stat": "receiving_yards", "rate": "targets"},
}
FEATURE_COLS = ["prior_season_avg_yards", "prior_season_games_played", "prior_season_avg_rate"]
VALIDATED_HOLDOUT_AUC = {"rushing_yards": 0.7702, "receiving_yards": 0.7778}


def _f(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def load_season_csv(season):
    path = CSV_DIR / f"stats_player_week_{season}.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_weeks123(rows, position):
    out = []
    for r in rows:
        if r.get("season_type") != "REG" or r.get("position") != position:
            continue
        try:
            week = int(r["week"])
        except Exception:
            continue
        if week > MAX_WEEK:
            continue
        out.append(r)
    return out


def load_full_season_by_player(rows, position):
    by_pid = {}
    for r in rows:
        if r.get("season_type") != "REG" or r.get("position") != position:
            continue
        pid = r.get("player_id")
        if not pid:
            continue
        by_pid.setdefault(pid, []).append({
            "carries": _f(r.get("carries")), "rushing_yards": _f(r.get("rushing_yards")),
            "targets": _f(r.get("targets")), "receiving_yards": _f(r.get("receiving_yards")),
        })
    return by_pid


def build_rows(cur_rows, prior_by_pid, cfg):
    stat_field, rate_field = cfg["stat"], cfg["rate"]
    out = []
    for r in cur_rows:
        pid = r.get("player_id")
        actual = _f(r.get(stat_field))
        games = prior_by_pid.get(pid)
        if games:
            n = len(games)
            feat = {
                "prior_season_avg_yards": sum(g[stat_field] for g in games) / n,
                "prior_season_games_played": n,
                "prior_season_avg_rate": sum(g[rate_field] for g in games) / n,
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
    print("NFL_PRIOR_SEASON_PRODUCTION_BUILDER_A\n======================================")
    MODEL_DIR.mkdir(exist_ok=True)

    needed_seasons = sorted({s for s in ALL_SEASONS} | {s - 1 for s in ALL_SEASONS})
    seasons_raw = {s: load_season_csv(s) for s in needed_seasons}

    for mkt_key, cfg in MARKETS.items():
        rows = []
        for season in ALL_SEASONS:
            cur = load_weeks123(seasons_raw[season], cfg["position"])
            prior = load_full_season_by_player(seasons_raw[season - 1], cfg["position"])
            rows += build_rows(cur, prior, cfg)
        matched = sum(1 for r in rows if not np.isnan(r["prior_season_avg_yards"]))
        print(f"\n{mkt_key}: {len(rows)} rows (all {ALL_SEASONS}, weeks 1-{MAX_WEEK}, "
              f"{cfg['position']}) -- {matched} matched to prior season ({matched/len(rows)*100:.1f}%)")

        n = len(rows)
        cut = int(n * 0.85)
        tr, va = rows[:cut], rows[cut:]
        print(f"  train={len(tr)}  internal val={len(va)} (early stopping only, not a real holdout -- "
              f"generalization already proven in nfl_prior_season_early_gate_a.py)")

        params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3,
                  "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
                  "min_child_weight": 5, "seed": 13}
        bst = xgb.train(params, mat(tr, xgb), num_boost_round=800, evals=[(mat(va, xgb), "val")],
                         early_stopping_rounds=40, verbose_eval=False)
        print(f"  best_iteration={bst.best_iteration}")

        bst.save_model(str(MODEL_DIR / f"nfl_prior_season_{mkt_key}.json"))
        (MODEL_DIR / f"nfl_prior_season_{mkt_key}_columns.json").write_text(json.dumps(FEATURE_COLS))
        manifest = {
            "script": "NFL_PRIOR_SEASON_PRODUCTION_BUILDER_A",
            "market": mkt_key, "position": cfg["position"], "line": LINE,
            "trained_on_seasons": ALL_SEASONS, "max_week": MAX_WEEK,
            "feature_cols": FEATURE_COLS, "best_iteration": bst.best_iteration,
            "validated_holdout_auc": VALIDATED_HOLDOUT_AUC[mkt_key],
            "validation_script": "nfl_prior_season_early_gate_a.py",
        }
        (MODEL_DIR / f"nfl_prior_season_{mkt_key}_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"  saved: {MODEL_DIR / f'nfl_prior_season_{mkt_key}.json'}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
