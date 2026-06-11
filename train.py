"""
train.py - Low-memory incremental trainer.
Trains projection models on historical data from /data WITHOUT loading
everything into memory at once. Processes one season at a time and trains
the model incrementally, so it survives Render's memory limits.

Run from the Render shell:
    python train.py

Trains two core props: batter hits and pitcher strikeouts.
"""
import json, glob, gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path("/data")
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def ip_to_outs(ip):
    try:
        whole = int(float(ip))
        frac = round((float(ip) - whole) * 10)
        return whole * 3 + frac
    except:
        return 0


def iter_season_files():
    return sorted(glob.glob(str(DATA_DIR / "season_*.jsonl")))


def load_one_season(fp, player_type):
    """Load only one season's rows of one player type. Keeps memory small."""
    rows = []
    with open(fp) as f:
        for line in f:
            try:
                r = json.loads(line)
            except:
                continue
            if r.get("type") == player_type:
                rows.append(r)
    return rows


# ── Feature builders (same logic, per-season) ─────────────────────────────────

def build_batter_features(rows):
    by_player = defaultdict(list)
    for r in rows:
        by_player[r["player_id"]].append(r)

    feats = []
    for pid, games in by_player.items():
        games.sort(key=lambda r: r["date"])
        cum_h = cum_ab = cum_pa = cum_hr = cum_bb = cum_so = 0
        recent = []
        for g in games:
            ab = g.get("ab", 0) or 0
            pa = g.get("pa", 0) or 0
            h = g.get("h", 0) or 0
            if cum_ab >= 20 and len(recent) >= 5:
                feats.append({
                    "season_avg": cum_h / cum_ab if cum_ab else 0,
                    "recent15_avg": sum(recent[-15:]) / len(recent[-15:]),
                    "recent5_avg": sum(recent[-5:]) / len(recent[-5:]),
                    "hr_rate": cum_hr / cum_pa if cum_pa else 0,
                    "bb_rate": cum_bb / cum_pa if cum_pa else 0,
                    "so_rate": cum_so / cum_pa if cum_pa else 0,
                    "batting_order": g.get("batting_order") or 9,
                    "games_played": len(recent),
                    "y_hits": h,
                })
            cum_h += h; cum_ab += ab; cum_pa += pa
            cum_hr += g.get("hr", 0) or 0
            cum_bb += g.get("bb", 0) or 0
            cum_so += g.get("so", 0) or 0
            recent.append(h)
    return pd.DataFrame(feats)


def build_pitcher_features(rows):
    by_player = defaultdict(list)
    for r in rows:
        by_player[r["player_id"]].append(r)

    feats = []
    for pid, games in by_player.items():
        games.sort(key=lambda r: r["date"])
        cum_bf = cum_so = cum_outs = cum_bb = 0
        recent_k = []; recent_bf = []; n_starts = 0
        for g in games:
            bf = g.get("bf", 0) or 0
            so = g.get("so", 0) or 0
            outs = g.get("outs", 0) or ip_to_outs(g.get("ip", "0.0"))
            is_start = bf >= 12
            if n_starts >= 3 and is_start:
                feats.append({
                    "k_per_bf": cum_so / cum_bf if cum_bf else 0,
                    "avg_bf": sum(recent_bf[-10:]) / len(recent_bf[-10:]),
                    "recent_k_avg": sum(recent_k[-5:]) / len(recent_k[-5:]),
                    "bb_rate": cum_bb / cum_bf if cum_bf else 0,
                    "outs_per_start": cum_outs / n_starts if n_starts else 0,
                    "starts": n_starts,
                    "y_so": so,
                })
            if is_start:
                cum_bf += bf; cum_so += so; cum_outs += outs
                cum_bb += g.get("bb_allowed", 0) or 0
                recent_k.append(so); recent_bf.append(bf); n_starts += 1
    return pd.DataFrame(feats)


# ── Incremental training ──────────────────────────────────────────────────────

def train_incremental(player_type, feature_builder, target_col, model_name):
    print(f"\n=== Training {model_name} (incremental) ===")
    booster = None
    feature_cols = None
    total_samples = 0

    params = {
        "objective": "count:poisson",
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "tree_method": "hist",
        "nthread": 2,
    }

    for fp in iter_season_files():
        season = Path(fp).stem
        rows = load_one_season(fp, player_type)
        df = feature_builder(rows)
        del rows; gc.collect()

        if df.empty:
            print(f"  {season}: no usable samples")
            continue

        if feature_cols is None:
            feature_cols = [c for c in df.columns if not c.startswith("y_")]

        X = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
        y = df[target_col].to_numpy(dtype=np.float32)
        total_samples += len(df)
        dtrain = xgb.DMatrix(X, label=y)

        # continue training from existing booster (incremental)
        booster = xgb.train(
            params, dtrain, num_boost_round=60,
            xgb_model=booster,
        )
        print(f"  {season}: trained on {len(df):,} samples "
              f"(running total {total_samples:,})")
        del df, X, y, dtrain; gc.collect()

    if booster is None:
        print(f"  No data at all for {model_name}")
        return

    # save model + columns
    booster.save_model(str(MODEL_DIR / f"{model_name}.json"))
    (MODEL_DIR / f"{model_name}_columns.json").write_text(json.dumps(feature_cols))
    print(f"  Saved {model_name}.json  (trained on {total_samples:,} total samples)")


def main():
    print("=" * 50)
    print("INCREMENTAL TRAINING — hits + strikeouts")
    print("=" * 50)
    train_incremental("batter", build_batter_features, "y_hits", "batter_hits")
    train_incremental("pitcher", build_pitcher_features, "y_so", "pitcher_strikeouts")
    print("\n" + "=" * 50)
    print("DONE. Models saved to /data/models/")
    print("=" * 50)


if __name__ == "__main__":
    main()
