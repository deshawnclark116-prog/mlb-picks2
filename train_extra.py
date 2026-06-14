"""
train_extra.py - Trains additional batter prop models on existing season data:
total bases (tb), RBI (rbi), runs (runs). Low-memory CHUNKED incremental training
so it survives Render's memory limit. Train one at a time:
    python train_extra.py tb
    python train_extra.py rbi
    python train_extra.py runs
"""
import json, glob, gc, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path("/data")
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def iter_season_files():
    return sorted(glob.glob(str(DATA_DIR / "season_*.jsonl")))


def load_one_season(fp):
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


def train_incremental(target_field, model_name):
    print(f"\n=== Training {model_name} (target={target_field}) ===")
    booster = None
    feature_cols = None
    total = 0

    params = {
        "objective": "count:poisson",
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "tree_method": "hist",
        "nthread": 1,
    }

    for fp in iter_season_files():
        season = Path(fp).stem
        rows = load_one_season(fp)
        df = build_features(rows, target_field)
        del rows; gc.collect()
        if df.empty:
            print(f"  {season}: no samples"); continue
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != "y"]

        # train in chunks of 15k rows to keep memory flat
        CHUNK = 15000
        for start in range(0, len(df), CHUNK):
            part = df.iloc[start:start + CHUNK]
            X = part[feature_cols].fillna(0).to_numpy(dtype=np.float32)
            y = part["y"].to_numpy(dtype=np.float32)
            dtrain = xgb.DMatrix(X, label=y)
            booster = xgb.train(params, dtrain, num_boost_round=20, xgb_model=booster)
            del X, y, dtrain, part; gc.collect()
        total += len(df)
        print(f"  {season}: trained on {len(df):,} (total {total:,})")
        del df; gc.collect()

    if booster is None:
        print(f"  No data for {model_name}"); return
    booster.save_model(str(MODEL_DIR / f"{model_name}.json"))
    (MODEL_DIR / f"{model_name}_columns.json").write_text(json.dumps(feature_cols))
    print(f"  Saved {model_name}.json ({total:,} samples)")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("=" * 55)
    print(f"TRAINING EXTRA PROPS — {target}")
    print("=" * 55)
    if target in ("tb", "all"):
        train_incremental("tb", "batter_total_bases")
    if target in ("rbi", "all"):
        train_incremental("rbi", "batter_rbi")
    if target in ("runs", "all"):
        train_incremental("runs", "batter_runs")
    print("\n" + "=" * 55)
    print("DONE.")
    print("=" * 55)


if __name__ == "__main__":
    main()
