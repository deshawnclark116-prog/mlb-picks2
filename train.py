"""
train.py - Trains projection models on historical season data from /data.
Reads season_YYYY.jsonl files, builds features, trains XGBoost models,
saves them to /data so they persist across redeploys.

Run from the Render shell:
    python train.py

Trains two core props: batter hits and pitcher strikeouts.
Memory-safe: processes one prop at a time.
"""
import json, glob, gc, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path("/data")
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_all_rows(player_type):
    """Stream all season files, yield rows of the requested type only."""
    rows = []
    for fp in sorted(glob.glob(str(DATA_DIR / "season_*.jsonl"))):
        with open(fp) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except:
                    continue
                if r.get("type") == player_type:
                    rows.append(r)
    return rows


def ip_to_outs(ip):
    """Convert '6.2' innings notation to total outs (6 innings 2 outs = 20)."""
    try:
        whole = int(float(ip))
        frac = round((float(ip) - whole) * 10)
        return whole * 3 + frac
    except:
        return 0


# ── Batter hits model ─────────────────────────────────────────────────────────

def build_batter_features(rows):
    """For each batter-game, build features from their PRIOR games only."""
    # group by player, sort by date
    by_player = defaultdict(list)
    for r in rows:
        by_player[r["player_id"]].append(r)

    feats = []
    for pid, games in by_player.items():
        games.sort(key=lambda r: r["date"])
        # running totals
        cum_h = cum_ab = cum_pa = cum_hr = cum_tb = cum_bb = cum_so = 0
        recent = []  # last 15 games of hits
        for g in games:
            ab = g.get("ab", 0) or 0
            pa = g.get("pa", 0) or 0
            h = g.get("h", 0) or 0
            n_prior = len([x for x in recent])

            # only build a training row if we have enough prior history
            if cum_ab >= 20 and n_prior >= 5:
                season_avg = cum_h / cum_ab if cum_ab else 0
                recent15 = recent[-15:]
                recent_avg = sum(recent15) / len(recent15) if recent15 else 0
                recent5 = recent[-5:]
                recent5_avg = sum(recent5) / len(recent5) if recent5 else 0
                hr_rate = cum_hr / cum_pa if cum_pa else 0
                bb_rate = cum_bb / cum_pa if cum_pa else 0
                so_rate = cum_so / cum_pa if cum_pa else 0
                order = g.get("batting_order") or 9

                feats.append({
                    "season_avg": season_avg,
                    "recent15_avg": recent_avg,
                    "recent5_avg": recent5_avg,
                    "hr_rate": hr_rate,
                    "bb_rate": bb_rate,
                    "so_rate": so_rate,
                    "batting_order": order,
                    "games_played": len(recent),
                    "y_hits": h,          # target: actual hits this game
                })

            # update running state AFTER building the row (no leakage)
            cum_h += h
            cum_ab += ab
            cum_pa += pa
            cum_hr += g.get("hr", 0) or 0
            cum_tb += g.get("tb", 0) or 0
            cum_bb += g.get("bb", 0) or 0
            cum_so += g.get("so", 0) or 0
            recent.append(h)

    return pd.DataFrame(feats)


# ── Pitcher strikeouts model ──────────────────────────────────────────────────

def build_pitcher_features(rows):
    """For each pitcher-start, build features from PRIOR starts only.
    Models strikeouts via batters-faced and per-batter K rate."""
    by_player = defaultdict(list)
    for r in rows:
        by_player[r["player_id"]].append(r)

    feats = []
    for pid, games in by_player.items():
        games.sort(key=lambda r: r["date"])
        cum_bf = cum_so = cum_outs = cum_bb = cum_h = 0
        recent_k = []   # last starts' strikeouts
        recent_bf = []  # last starts' batters faced
        n_starts = 0
        for g in games:
            bf = g.get("bf", 0) or 0
            so = g.get("so", 0) or 0
            outs = g.get("outs", 0) or ip_to_outs(g.get("ip", "0.0"))

            # only starters: require meaningful batters faced
            is_start = bf >= 12
            if n_starts >= 3 and is_start:
                k_per_bf = cum_so / cum_bf if cum_bf else 0
                avg_bf = sum(recent_bf[-10:]) / len(recent_bf[-10:]) if recent_bf else 0
                recent_k_avg = sum(recent_k[-5:]) / len(recent_k[-5:]) if recent_k else 0
                bb_rate = cum_bb / cum_bf if cum_bf else 0
                outs_per_start = cum_outs / n_starts if n_starts else 0

                feats.append({
                    "k_per_bf": k_per_bf,
                    "avg_bf": avg_bf,
                    "recent_k_avg": recent_k_avg,
                    "bb_rate": bb_rate,
                    "outs_per_start": outs_per_start,
                    "starts": n_starts,
                    "y_so": so,         # target: actual strikeouts this start
                })

            if is_start:
                cum_bf += bf
                cum_so += so
                cum_outs += outs
                cum_bb += g.get("bb_allowed", 0) or 0
                cum_h += g.get("h_allowed", 0) or 0
                recent_k.append(so)
                recent_bf.append(bf)
                n_starts += 1

    return pd.DataFrame(feats)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(df, target_col, model_name):
    if df.empty or target_col not in df.columns:
        print(f"  No data for {model_name}")
        return
    feature_cols = [c for c in df.columns if not c.startswith("y_")]
    X = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)

    print(f"  Training {model_name} on {len(df):,} samples, "
          f"{len(feature_cols)} features")

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="count:poisson",   # right objective for count outcomes
        tree_method="hist",
        n_jobs=2,
        random_state=42,
    )
    model.fit(X, y)

    # save model + feature column order
    model_path = MODEL_DIR / f"{model_name}.json"
    model.save_model(str(model_path))
    cols_path = MODEL_DIR / f"{model_name}_columns.json"
    cols_path.write_text(json.dumps(feature_cols))

    # quick sanity: mean predicted vs mean actual
    preds = model.predict(X)
    print(f"  {model_name}: mean predicted={preds.mean():.3f}, "
          f"mean actual={y.mean():.3f}  -> saved to {model_path}")


def main():
    print("=" * 50)
    print("TRAINING — batter hits + pitcher strikeouts")
    print("=" * 50)

    print("\n[1/2] Batter hits")
    batter_rows = load_all_rows("batter")
    print(f"  Loaded {len(batter_rows):,} batter-game rows")
    bdf = build_batter_features(batter_rows)
    del batter_rows; gc.collect()
    train_model(bdf, "y_hits", "batter_hits")
    del bdf; gc.collect()

    print("\n[2/2] Pitcher strikeouts")
    pitcher_rows = load_all_rows("pitcher")
    print(f"  Loaded {len(pitcher_rows):,} pitcher-game rows")
    pdf = build_pitcher_features(pitcher_rows)
    del pitcher_rows; gc.collect()
    train_model(pdf, "y_so", "pitcher_strikeouts")
    del pdf; gc.collect()

    print("\n" + "=" * 50)
    print("DONE. Models saved to /data/models/")
    print("=" * 50)


if __name__ == "__main__":
    main()
