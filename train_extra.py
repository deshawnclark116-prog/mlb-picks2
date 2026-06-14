"""
train_extra.py - Trains batter prop models in SMALL date-range batches to stay
under 512MB. Each run trains one slice, saves, exits (releasing memory).

Usage — prop, season, and optional month range (start end, inclusive):
    python train_extra.py tb 2026
    python train_extra.py tb 2025 1 6      (first half of 2025)
    python train_extra.py tb 2025 7 12     (second half of 2025)
Repeat for rbi and runs.
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


def load_season_batters(season, m_start=None, m_end=None):
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
            if r.get("type") != "batter":
                continue
            if m_start is not None:
                # date is "YYYY-MM-DD"
                try:
                    mo = int(r["date"][5:7])
                except:
                    continue
                if mo < m_start or mo > m_end:
                    continue
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
        print("Usage: python train_extra.py <tb|rbi|runs> <season> [m_start m_end]")
        return

    prop = sys.argv[1]
    season = sys.argv[2]
    m_start = int(sys.argv[3]) if len(sys.argv) > 3 else None
    m_end = int(sys.argv[4]) if len(sys.argv) > 4 else None

    if prop not in TARGET_FIELD:
        print(f"Unknown prop '{prop}'. Use tb, rbi, or runs."); return

    target_field = TARGET_FIELD[prop]
    model_name = MODEL_NAME[prop]
    model_path = MODEL_DIR / f"{model_name}.json"
    cols_path = MODEL_DIR / f"{model_name}_columns.json"

    rng = f" months {m_start}-{m_end}" if m_start else ""
    print(f"=== {model_name}: season {season}{rng} ===")

    rows = load_season_batters(season, m_start, m_end)
    if not rows:
        print(f"  No batter rows for {season}{rng}"); return
    df = build_features(rows, target_field)
    del rows; gc.collect()
    if df.empty:
        print(f"  No usable samples"); return

    feature_cols = [c for c in df.columns if c != "y"]

    booster = None
    if model_path.exists():
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        print("  Loaded existing model, continuing")

    params = {
        "objective": "count:poisson", "learning_rate": 0.05, "max_depth": 5,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
        "tree_method": "hist", "nthread": 1,
    }

    CHUNK = 8000
    for start in range(0, len(df), CHUNK):
        part = df.iloc[start:start + CHUNK]
        X = part[feature_cols].fillna(0).to_numpy(dtype=np.float32)
        y = part["y"].to_numpy(dtype=np.float32)
        dtrain = xgb.DMatrix(X, label=y)
        booster = xgb.train(params, dtrain, num_boost_round=20, xgb_model=booster)
        del X, y, dtrain, part; gc.collect()

    booster.save_model(str(model_path))
    cols_path.write_text(json.dumps(feature_cols))
    print(f"  Saved {model_name}.json — {len(df):,} samples from {season}{rng}")
    print("  DONE (memory released on exit)")


if __name__ == "__main__":
    main()
