"""
train_extra.py - Trains batter prop models ONE SEASON PER RUN to stay under
the 512MB memory ceiling. Each run is a fresh process: loads the saved model
(if it exists), trains one season on top of it, saves, and exits — releasing
all memory between seasons.

Usage (run each, waiting for "Saved" before the next):
    python train_extra.py tb 2026
    python train_extra.py tb 2025
    python train_extra.py tb 2024
  then repeat for rbi and runs:
    python train_extra.py rbi 2026   (etc.)
    python train_extra.py runs 2026  (etc.)
"""
import json, gc, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path("/data")
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET_FIELD = {"tb": "tb", "rbi": "rbi", "runs": "runs"}
MODEL_NAME = {"tb": "batter_total_bases", "rbi": "batter_rbi", "runs": "batter_runs"}


def load_season_batters(season):
    fp = DATA_DIR / f"season_{season}.jsonl"
    if not fp.exists():
        return []
    rows = []
    with open(fp) as f:
        for line in f:
            try:
                r = json.loads(line)
            except:
                continue
            if r.get("type") == "batter":
                rows.append(r)
    return rows


def build_features(rows, target_field):
    by_player = defaultdict(list)
    for r in rows:
        by_player[r["player_id"]].append(r)

    feats = []
    for pid, games in by_player.items():
        games.sort(key=lambda r: r["date"])
        cum_ab = cum_h = cum_pa = cum_tb = cum_rbi = cum_runs = cum_hr = cum_bb = cum_so = 0
        recent_target = []
        n = 0
        for g in games:
            ab = g.get("ab", 0) or 0
            pa = g.get("pa", 0) or 0
            h = g.get("h", 0) or 0
            tb = g.get("tb", 0) or 0
            rbi = g.get("rbi", 0) or 0
            runs = g.get("runs", 0) or 0
            y = g.get(target_field, 0) or 0

            if cum_ab >= 20 and n >= 5:
                feats.append({
                    "season_avg": cum_h / cum_ab if cum_ab else 0,
                    "tb_per_pa": cum_tb / cum_pa if cum_pa else 0,
                    "rbi_per_pa": cum_rbi / cum_pa if cum_pa else 0,
                    "runs_per_pa": cum_runs / cum_pa if cum_pa else 0,
                    "hr_rate": cum_hr / cum_pa if cum_pa else 0,
                    "bb_rate": cum_bb / cum_pa if cum_pa else 0,
                    "so_rate": cum_so / cum_pa if cum_pa else 0,
                    "recent5_target": sum(recent_target[-5:]) / len(recent_target[-5:]),
                    "recent15_target": sum(recent_target[-15:]) / len(recent_target[-15:]),
                    "batting_order": g.get("batting_order") or 9,
                    "games_played": n,
                    "y": y,
                })

            cum_ab += ab; cum_h += h; cum_pa += pa
            cum_tb += tb; cum_rbi += rbi; cum_runs += runs
            cum_hr += g.get("hr", 0) or 0
            cum_bb += g.get("bb", 0) or 0
            cum_so += g.get("so", 0) or 0
            recent_target.append(y)
            n += 1

    return pd.DataFrame(feats)


def main():
    if len(sys.argv) < 3:
        print("Usage: python train_extra.py <tb|rbi|runs> <season>")
        print("Example: python train_extra.py tb 2026")
        return

    prop = sys.argv[1]
    season = sys.argv[2]
    if prop not in TARGET_FIELD:
        print(f"Unknown prop '{prop}'. Use tb, rbi, or runs."); return

    target_field = TARGET_FIELD[prop]
    model_name = MODEL_NAME[prop]
    model_path = MODEL_DIR / f"{model_name}.json"
    cols_path = MODEL_DIR / f"{model_name}_columns.json"

    print(f"=== {model_name}: training season {season} ===")

    rows = load_season_batters(season)
    if not rows:
        print(f"  No batter rows for {season}"); return
    df = build_features(rows, target_field)
    del rows; gc.collect()
    if df.empty:
        print(f"  No usable samples for {season}"); return

    feature_cols = [c for c in df.columns if c != "y"]

    # load existing booster if present (continue training on top of it)
    booster = None
    if model_path.exists():
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        print("  Loaded existing model, continuing training")

    params = {
        "objective": "count:poisson", "learning_rate": 0.05, "max_depth": 5,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
        "tree_method": "hist", "nthread": 1,
    }

    CHUNK = 10000
    for start in range(0, len(df), CHUNK):
        part = df.iloc[start:start + CHUNK]
        X = part[feature_cols].fillna(0).to_numpy(dtype=np.float32)
        y = part["y"].to_numpy(dtype=np.float32)
        dtrain = xgb.DMatrix(X, label=y)
        booster = xgb.train(params, dtrain, num_boost_round=20, xgb_model=booster)
        del X, y, dtrain, part; gc.collect()

    booster.save_model(str(model_path))
    cols_path.write_text(json.dumps(feature_cols))
    print(f"  Saved {model_name}.json — trained {len(df):,} samples from {season}")
    print("  DONE (memory released on exit)")


if __name__ == "__main__":
    main()
