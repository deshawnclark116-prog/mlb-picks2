#!/usr/bin/env python3
"""
HR_MODEL_TRAIN_REDUCED_B

Safer reduced logistic HR baseline.

Fixes REDUCED_A overconfidence:
- raw rate stats from tiny samples can explode probabilities
- this version shrinks rate features toward TRAIN-only priors
- uses stronger L2 regularization
- avoids actual postgame plate_appearances

Run:
    python hr_model_train_reduced_b.py

Try stronger shrink:
    python hr_model_train_reduced_b.py --shrink-k 75 --c 0.03
"""

import argparse, math, os, sqlite3
from pathlib import Path

RAW_FEATURES = [
    "expected_pa_v1", "temp_f",
    "batter_hr_per_air_60d",
    "batter_fly_ball_rate_30d",
    "batter_barrel_rate_30d",
    "batter_max_ev_60d",
    "pitcher_pull_air_allowed_rate_60d",
    "pitcher_hard_hit_allowed_rate_7d",
    "pitcher_hrfb_rate_30d",
    "batter_bbe_60d",
    "pitcher_bbe_allowed_60d",
]

RATE_TO_SAMPLE = {
    "batter_hr_per_air_60d": "batter_bbe_60d",
    "batter_fly_ball_rate_30d": "batter_bbe_60d",
    "batter_barrel_rate_30d": "batter_bbe_60d",
    "pitcher_pull_air_allowed_rate_60d": "pitcher_bbe_allowed_60d",
    "pitcher_hard_hit_allowed_rate_7d": "pitcher_bbe_allowed_60d",
    "pitcher_hrfb_rate_30d": "pitcher_bbe_allowed_60d",
}

MODEL_FEATURES = [
    "expected_pa_v1", "temp_f", "batter_max_ev_60d",
    "log_batter_bbe_60d", "log_pitcher_bbe_allowed_60d",
] + [f"{c}_shrunk" for c in RATE_TO_SAMPLE]


def default_db_path():
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    return p / "hr_model.sqlite" if p.parent.exists() else Path("./hr_model/hr_model.sqlite")


def imports():
    try:
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return pd, ColumnTransformer, SimpleImputer, LogisticRegression, brier_score_loss, log_loss, roc_auc_score, Pipeline, StandardScaler
    except Exception as e:
        print("Install missing deps with: pip install pandas scikit-learn numpy")
        raise SystemExit(f"{type(e).__name__}: {e}")


def cols(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def expr(conn, f):
    bg, bf, pf = set(cols(conn, "batter_games")), set(cols(conn, "batter_game_features")), set(cols(conn, "pitcher_game_features"))
    if f in bg: return f"bg.{f} AS {f}"
    if f in bf: return f"bf.{f} AS {f}"
    if f in pf: return f"pf.{f} AS {f}"
    return f"NULL AS {f}"


def load(conn, pd):
    select = [
        "bg.game_date AS game_date", "bg.game_id AS game_id", "bg.batter_id AS batter_id",
        "bg.batter_name AS batter_name", "bg.lineup_spot AS lineup_spot", "bg.actual_hr AS actual_hr",
    ] + [expr(conn, f) for f in RAW_FEATURES]
    sql = f"""
    SELECT {", ".join(select)}
    FROM batter_games bg
    LEFT JOIN batter_game_features bf ON bg.game_id=bf.game_id AND bg.batter_id=bf.batter_id
    LEFT JOIN pitcher_game_features pf ON bg.game_id=pf.game_id AND bg.batter_id=pf.batter_id
    WHERE bg.actual_hr IS NOT NULL
    ORDER BY bg.game_date, bg.game_id, bg.lineup_spot
    """
    return pd.read_sql_query(sql, conn)


def add_safe(train, test, shrink_k):
    for df in (train, test):
        df["log_batter_bbe_60d"] = df["batter_bbe_60d"].fillna(0).clip(lower=0).map(lambda x: math.log1p(x))
        df["log_pitcher_bbe_allowed_60d"] = df["pitcher_bbe_allowed_60d"].fillna(0).clip(lower=0).map(lambda x: math.log1p(x))

    priors = {}
    for rate_col, n_col in RATE_TO_SAMPLE.items():
        prior = float(train[rate_col].dropna().median()) if train[rate_col].notna().sum() else 0.0
        priors[rate_col] = prior
        for df in (train, test):
            raw = df[rate_col].fillna(prior).clip(lower=0, upper=1)
            n = df[n_col].fillna(0).clip(lower=0)
            w = n / (n + shrink_k)
            df[f"{rate_col}_shrunk"] = prior + (raw - prior) * w
    return priors


def safe_metric(fn, y, p):
    try: return float(fn(y, p))
    except Exception: return None


def show_metrics(label, y, p, brier, logloss, auc):
    base = float(sum(y) / len(y)) if len(y) else 0.0
    bp = [base] * len(y)
    print(f"\n{label} METRICS")
    print("-" * (len(label) + 8))
    print(f"rows: {len(y)}")
    print(f"hr_hits: {int(sum(y))}")
    print(f"actual_rate: {base:.3%}")
    print(f"baseline_brier: {safe_metric(brier, y, bp)}")
    print(f"model_brier:    {safe_metric(brier, y, p)}")
    print(f"baseline_logloss: {safe_metric(logloss, y, bp)}")
    print(f"model_logloss:    {safe_metric(logloss, y, p)}")
    print(f"model_auc: {safe_metric(auc, y, p)}")


def buckets(df):
    print("\nPROBABILITY BUCKET REPORT")
    print("-------------------------")
    d = df.sort_values("model_prob", ascending=False).copy()
    d["bucket"] = range(len(d))
    d["bucket"] = (d["bucket"] * 10 / max(1, len(d))).astype(int) + 1
    for b in sorted(d.bucket.unique()):
        s = d[d.bucket == b]
        print(f"bucket_{b:02d} n={len(s):5d} avg_prob={s.model_prob.mean():.3%} actual_hr={s.actual_hr.mean():.3%} hits={int(s.actual_hr.sum())}")


def top(df, n):
    print(f"\nTOP {n} TEST PREDICTIONS")
    print("------------------------")
    cols = ["game_date","batter_name","lineup_spot","model_prob","actual_hr","expected_pa_v1","temp_f",
            "batter_hr_per_air_60d","batter_hr_per_air_60d_shrunk","batter_bbe_60d",
            "batter_fly_ball_rate_30d","batter_barrel_rate_30d","batter_max_ev_60d",
            "pitcher_hrfb_rate_30d","pitcher_bbe_allowed_60d"]
    cols = [c for c in cols if c in df.columns]
    for _, r in df.sort_values("model_prob", ascending=False).head(n).iterrows():
        parts = []
        for c in cols:
            v = r.get(c)
            if hasattr(v, "date"):
                try: v = v.date()
                except Exception: pass
            parts.append(f"{c}={v:.4f}" if isinstance(v, float) else f"{c}={v}")
        print(" | ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--test-date-frac", type=float, default=0.25)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--c", type=float, default=0.05)
    ap.add_argument("--shrink-k", type=float, default=50.0)
    args = ap.parse_args()

    pd, ColumnTransformer, SimpleImputer, LogisticRegression, brier, logloss, auc, Pipeline, StandardScaler = imports()
    conn = sqlite3.connect(args.db)
    df = load(conn, pd)
    conn.close()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["actual_hr"] = df["actual_hr"].astype(int)
    for f in RAW_FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    dates = sorted(df.game_date.dropna().unique())
    test_n = max(1, int(math.ceil(len(dates) * args.test_date_frac)))
    train_dates, test_dates = dates[:-test_n], dates[-test_n:]
    train = df[df.game_date.isin(train_dates)].copy()
    test = df[df.game_date.isin(test_dates)].copy()
    priors = add_safe(train, test, args.shrink_k)

    print("HR_MODEL_TRAIN_REDUCED_B")
    print("========================")
    print(f"db: {args.db}")
    print(f"rows_total: {len(df)}")
    print(f"hr_hits_total: {int(df.actual_hr.sum())}")
    print(f"overall_hr_rate: {df.actual_hr.mean():.3%}")
    print(f"date_count: {len(dates)}")
    print(f"train_dates: {train.game_date.min().date()} to {train.game_date.max().date()} ({len(train)} rows)")
    print(f"test_dates:  {test.game_date.min().date()} to {test.game_date.max().date()} ({len(test)} rows)")
    print(f"features: {MODEL_FEATURES}")
    print(f"logistic_C: {args.c}")
    print(f"shrink_k: {args.shrink_k}")
    print("\nTRAIN PRIORS USED FOR SHRINKAGE")
    print("--------------------------------")
    for k, v in priors.items():
        print(f"{k}: {v:.5f}")

    pre = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), MODEL_FEATURES)])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", C=args.c))])
    model.fit(train[MODEL_FEATURES], train.actual_hr)
    p_train = model.predict_proba(train[MODEL_FEATURES])[:, 1]
    p_test = model.predict_proba(test[MODEL_FEATURES])[:, 1]

    show_metrics("TRAIN", list(train.actual_hr), list(p_train), brier, logloss, auc)
    show_metrics("TEST", list(test.actual_hr), list(p_test), brier, logloss, auc)

    out = test.copy()
    out["model_prob"] = p_test
    buckets(out)
    top(out, args.top)

    print("\nREAD")
    print("----")
    print("This version shrinks tiny-sample rate stats toward train-only priors.")
    print("Goal: lower overconfidence and better forward Brier/logloss, not flashy top probabilities.")
    print("Still not a betting model until calibrated on a true forward holdout and joined to FanDuel HR odds.")


if __name__ == "__main__":
    main()
