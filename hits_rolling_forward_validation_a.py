#!/usr/bin/env python3
"""Memory-safe rolling forward validation for the locked batter-hits incumbent.

Folds:
  Train 2019-2021 -> Test 2022
  Train 2019-2022 -> Test 2023
  Train 2019-2023 -> Test 2024
  Train 2019-2024 -> Test 2025

2026 is intentionally excluded as a clean holdout because the incumbent was
trained on partial 2026 data.

Run:
  python -u hits_rolling_forward_validation_a.py 2>&1 | tee /data/hr_model/hits_rolling_forward_validation_a.log
"""

import argparse, glob, json, math, os, re, subprocess, sys
from pathlib import Path

FEATURES = [
    "season_avg", "recent15_avg", "recent5_avg", "hr_rate",
    "bb_rate", "so_rate", "batting_order", "games_played",
]
PARAMS = {
    "objective": "count:poisson",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    "nthread": 2,
    "seed": 42,
}
OFFICIAL_MIN = 0.630
WATCHLIST_MIN = 0.606
WORK = Path("/data/hr_model/hits_rolling_forward_validation_a_work")
OUT = Path("/data/hr_model/hits_rolling_forward_validation_a_results.json")


def discover_files():
    out = {}
    for pattern in ("/data/season_*.jsonl", "/data/season_*.json"):
        for p in glob.glob(pattern):
            m = re.search(r"season_(20\d{2})\.(?:jsonl|json)$", Path(p).name)
            if m:
                y = int(m.group(1))
                if y not in out or p.endswith(".jsonl"):
                    out[y] = p
    return dict(sorted(out.items()))


def target_col(df):
    for c in ("y_hits", "actual_hits", "hits", "target"):
        if c in df.columns:
            return c
    raise KeyError(f"No hits target column. Columns: {list(df.columns)}")


def date_col(df):
    for c in ("game_date", "date"):
        if c in df.columns:
            return c
    return None


def worker_train(args):
    import gc, numpy as np, xgboost as xgb, train
    print(f"season={args.season}")
    rows = train.load_one_season(args.season, "batter")
    print(f"raw_batter_rows={len(rows):,}")
    df = train.build_batter_features(rows)
    del rows; gc.collect()
    tcol = target_col(df)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    X = df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    y = df[tcol].to_numpy(dtype=np.float32, copy=True)
    print(f"eligible_samples={len(y):,}")
    del df; gc.collect()
    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURES)
    del X, y; gc.collect()
    prev = None
    prior_rounds = 0
    if args.prev and Path(args.prev).exists():
        prev = xgb.Booster(); prev.load_model(args.prev)
        prior_rounds = int(prev.num_boosted_rounds())
    booster = xgb.train(PARAMS, dtrain, num_boost_round=60, xgb_model=prev)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(args.model_out)
    print(f"prior_rounds={prior_rounds}")
    print("added_rounds=60")
    print(f"total_rounds={booster.num_boosted_rounds()}")


def worker_score(args):
    import gc, numpy as np, xgboost as xgb, train
    print(f"season={args.season}")
    rows = train.load_one_season(args.season, "batter")
    print(f"raw_batter_rows={len(rows):,}")
    df = train.build_batter_features(rows)
    del rows; gc.collect()
    tcol = target_col(df); dcol = date_col(df)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    X = df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    y_hits = df[tcol].to_numpy(dtype=np.float32, copy=True)
    y = (y_hits >= 1).astype(np.int8)
    dates = df[dcol].astype(str).to_numpy(dtype="U16", copy=True) if dcol else np.full(len(df), "", dtype="U16")
    del df; gc.collect()
    booster = xgb.Booster(); booster.load_model(args.model)
    pred_lambda = booster.predict(xgb.DMatrix(X, feature_names=FEATURES)).astype(np.float64)
    p = 1.0 - np.exp(-np.clip(pred_lambda, 1e-8, 50.0))
    np.savez_compressed(args.pred_out, y=y, y_hits=y_hits, p=p.astype(np.float32), dates=dates)
    print(f"scored_rows={len(y):,}")
    print(f"actual_hit_rate={y.mean():.6f}")
    print(f"mean_probability={p.mean():.6f}")
    print(f"model_rounds={booster.num_boosted_rounds()}")


def run_child(extra, label):
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS":"2", "OPENBLAS_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "NUMEXPR_NUM_THREADS":"1"})
    print(f"\n{label}\n{'-'*len(label)}", flush=True)
    cp = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve())] + extra,
                        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(cp.stdout, end="" if cp.stdout.endswith("\n") else "\n", flush=True)
    if cp.returncode:
        raise RuntimeError(f"{label} failed with exit code {cp.returncode}")


def auc_rank(y, p):
    import numpy as np
    y = np.asarray(y, dtype=np.int8); p = np.asarray(p, dtype=np.float64)
    n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0: return float("nan")
    order = np.argsort(p, kind="mergesort"); sp = p[order]
    ranks = np.empty(len(p), dtype=np.float64)
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and sp[j] == sp[i]: j += 1
        ranks[order[i:j]] = ((i + 1) + j) / 2.0
        i = j
    return float((ranks[y == 1].sum() - n1*(n1+1)/2.0) / (n1*n0))


def metrics(y, p):
    import numpy as np
    y = np.asarray(y, dtype=np.int8); p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1-1e-8)
    out = {
        "rows": int(len(y)), "actual_hit_rate": float(y.mean()), "mean_probability": float(p.mean()),
        "brier": float(np.mean((p-y)**2)),
        "logloss": float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p))),
        "auc": auc_rank(y, p),
    }
    for pct, key in ((0.05,"top5"),(0.10,"top10")):
        k = max(1, int(math.ceil(len(y)*pct))); idx = np.argpartition(p, -k)[-k:]
        out[f"{key}_rows"] = int(k); out[f"{key}_hits"] = int(y[idx].sum())
        out[f"{key}_actual"] = float(y[idx].mean()); out[f"{key}_mean_prob"] = float(p[idx].mean())
    return out


def thresholds(y, p):
    import numpy as np
    y = np.asarray(y, dtype=np.int8); p = np.asarray(p, dtype=np.float64)
    def one(mask):
        n = int(mask.sum())
        return {"rows":n, "hits":int(y[mask].sum()) if n else 0,
                "actual_hit_rate":float(y[mask].mean()) if n else None,
                "mean_probability":float(p[mask].mean()) if n else None}
    return {
        "official_ge_0.630": one(p >= OFFICIAL_MIN),
        "watchlist_0.606_to_0.630": one((p >= WATCHLIST_MIN) & (p < OFFICIAL_MIN)),
        "official_plus_watchlist_ge_0.606": one(p >= WATCHLIST_MIN),
    }


def print_metrics(label, m):
    print(f"{label}: n={m['rows']:,} actual={m['actual_hit_rate']:.4f} pred={m['mean_probability']:.4f} "
          f"brier={m['brier']:.8f} logloss={m['logloss']:.8f} auc={m['auc']:.6f} "
          f"top5={m['top5_actual']:.4f} top10={m['top10_actual']:.4f}")


def parent():
    import numpy as np
    files = discover_files(); WORK.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    print("HITS_ROLLING_FORWARD_VALIDATION_A\n=================================")
    print("\nPREFLIGHT\n---------")
    if not files:
        print("No /data/season_*.jsonl or /data/season_*.json files found.")
        return 2
    for y,p in files.items(): print(f"{y}: {p} | {Path(p).stat().st_size:,} bytes")
    missing = [y for y in range(2019, 2026) if y not in files]
    if missing:
        print(f"Missing required years: {missing}"); return 2
    if 2026 in files:
        print("2026 detected but excluded from clean historical folds because incumbent training already used partial 2026.")

    current = None; folds = []; all_y=[]; all_p=[]
    for train_year in range(2019, 2025):
        model_out = WORK / f"booster_through_{train_year}.json"
        extra = ["--worker-train", "--season", files[train_year], "--model-out", str(model_out)]
        if current: extra += ["--prev", str(current)]
        run_child(extra, f"TRAIN THROUGH {train_year}")
        current = model_out
        test_year = train_year + 1
        if 2022 <= test_year <= 2025:
            pred_out = WORK / f"predictions_{test_year}.npz"
            run_child(["--worker-score", "--season", files[test_year], "--model", str(current), "--pred-out", str(pred_out)],
                      f"SCORE FORWARD HOLDOUT {test_year}")
            z = np.load(pred_out, allow_pickle=False); y=z["y"].astype(np.int8); p=z["p"].astype(np.float64)
            m=metrics(y,p); t=thresholds(y,p)
            folds.append({"train_years":list(range(2019,train_year+1)),"test_year":test_year,"model_rounds":60*(train_year-2018),"metrics":m,"thresholds":t})
            all_y.append(y); all_p.append(p)

    y_all=np.concatenate(all_y); p_all=np.concatenate(all_p)
    agg=metrics(y_all,p_all); agg_t=thresholds(y_all,p_all)

    print("\nROLLING FORWARD RESULTS\n-----------------------")
    for f in folds: print_metrics(f"TRAIN 2019-{max(f['train_years'])} -> TEST {f['test_year']}", f["metrics"])
    print("\nAGGREGATE BASELINE\n------------------"); print_metrics("ALL FORWARD FOLDS", agg)
    print("\nPRODUCTION THRESHOLD READ\n-------------------------")
    for name,b in agg_t.items():
        if b["rows"]:
            print(f"{name}: n={b['rows']:,} hits={b['hits']:,} actual_hit_rate={b['actual_hit_rate']:.4f} mean_probability={b['mean_probability']:.4f}")
        else: print(f"{name}: n=0")
    print("\nBASELINE STATUS\n---------------")
    print("incumbent_reconstruction: COMPLETE")
    print("rolling_forward_baseline: COMPLETE")
    print("2026_clean_holdout: NO — burned by original incumbent training")
    print("challenger_features_authorized: NOT YET — first lock hits-specific promotion gates from this baseline")
    print("probability_note: exact Poisson Over-0.5 probability 1-exp(-lambda), equivalent to production Poisson MC without sampling noise")

    result={"script":"HITS_ROLLING_FORWARD_VALIDATION_A","market":"batter_hits_over_0.5",
            "incumbent":{"model_name":"batter_hits","features":FEATURES,"params":PARAMS,"rounds_per_season_file":60,
                         "official_min_prob":OFFICIAL_MIN,"watchlist_min_prob":WATCHLIST_MIN,"2026_holdout_status":"burned"},
            "folds":folds,"aggregate":agg,"aggregate_thresholds":agg_t,"season_files":files}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nresults_json={OUT}")
    return 0


def parser():
    ap=argparse.ArgumentParser(); ap.add_argument("--worker-train", action="store_true"); ap.add_argument("--worker-score", action="store_true")
    ap.add_argument("--season"); ap.add_argument("--prev"); ap.add_argument("--model-out"); ap.add_argument("--model"); ap.add_argument("--pred-out")
    return ap


def main():
    args=parser().parse_args()
    if args.worker_train: worker_train(args); return 0
    if args.worker_score: worker_score(args); return 0
    return parent()

if __name__ == "__main__": raise SystemExit(main())
