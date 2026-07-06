#!/usr/bin/env python3
"""
HR Model Train A — first diagnostic model for the SQLite HR dataset.

This is NOT a production betting model yet.
It is a leakage-aware baseline trainer to answer:
    "Do the rolling Statcast features predict game-level HR better than baseline?"

Reads:
    /data/hr_model/hr_model.sqlite

Uses:
    one row = one starter-game
    target = actual_hr
    time split by game_date, never random split

Runs:
    python hr_model_train_a.py

More conservative:
    python hr_model_train_a.py --min-bbe 5 --test-date-frac 0.25
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, List, Optional, Tuple


def default_db_path() -> Path:
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    if p.parent.exists():
        return p / "hr_model.sqlite"
    return Path("./hr_model/hr_model.sqlite")


def require_imports():
    try:
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        return pd, ColumnTransformer, SimpleImputer, LogisticRegression, brier_score_loss, log_loss, roc_auc_score, Pipeline, OneHotEncoder, StandardScaler
    except Exception as e:
        print("Missing dependency for model training.")
        print("Install with:")
        print("  pip install pandas scikit-learn numpy")
        print(f"Original error: {type(e).__name__}: {e}")
        raise SystemExit(1)


def table_cols(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def load_dataset(conn: sqlite3.Connection, pd):
    bf_cols = [c for c in table_cols(conn, "batter_game_features") if c not in {"game_id", "batter_id"}]
    pf_cols = [c for c in table_cols(conn, "pitcher_game_features") if c not in {"game_id", "batter_id", "opposing_pitcher_id"}]

    select_parts = [
        "bg.game_date AS game_date",
        "bg.game_id AS game_id",
        "bg.batter_id AS batter_id",
        "bg.batter_name AS batter_name",
        "bg.lineup_spot AS lineup_spot",
        "bg.batter_hand AS batter_hand",
        "bg.pitcher_hand AS pitcher_hand",
        "bg.platoon_bucket AS platoon_bucket",
        "bg.park_bucket AS park_bucket",
        "bg.hitterish_park AS hitterish_park",
        "bg.pitcherish_park AS pitcherish_park",
        "bg.temp_f AS temp_f",
        "bg.wind_speed_mph AS wind_speed_mph",
        "bg.wind_toward_pull_field AS wind_toward_pull_field",
        "bg.actual_hr AS actual_hr",
    ]
    select_parts += [f"bf.{c} AS {c}" for c in bf_cols]
    select_parts += [f"pf.{c} AS {c}" for c in pf_cols]

    sql = f"""
    SELECT {", ".join(select_parts)}
    FROM batter_games bg
    LEFT JOIN batter_game_features bf
      ON bg.game_id = bf.game_id
     AND bg.batter_id = bf.batter_id
    LEFT JOIN pitcher_game_features pf
      ON bg.game_id = pf.game_id
     AND bg.batter_id = pf.batter_id
    ORDER BY bg.game_date, bg.game_id, bg.lineup_spot
    """
    return pd.read_sql_query(sql, conn)


def add_manual_bucket_features(df):
    def ge(col, val):
        if col in df.columns:
            return (df[col].astype(float).fillna(-999) >= val).astype(int)
        return 0

    for suffix in ("7d", "15d", "30d"):
        if f"batter_max_ev_{suffix}" in df.columns and f"batter_fly_ball_rate_{suffix}" in df.columns:
            df[f"bucket_maxev105_fly35_{suffix}"] = (
                ge(f"batter_max_ev_{suffix}", 105) & ge(f"batter_fly_ball_rate_{suffix}", 0.35)
            ).astype(int)
        if f"batter_barrel_rate_{suffix}" in df.columns and f"batter_la_20_35_rate_{suffix}" in df.columns:
            df[f"bucket_barrel10_la25_{suffix}"] = (
                ge(f"batter_barrel_rate_{suffix}", 0.10) & ge(f"batter_la_20_35_rate_{suffix}", 0.25)
            ).astype(int)
        if f"batter_hard_hit_rate_{suffix}" in df.columns and f"batter_pull_air_rate_{suffix}" in df.columns:
            df[f"bucket_hard45_pull30_{suffix}"] = (
                ge(f"batter_hard_hit_rate_{suffix}", 0.45) & ge(f"batter_pull_air_rate_{suffix}", 0.30)
            ).astype(int)
    return df


def choose_features(df) -> Tuple[List[str], List[str]]:
    numeric = []
    for suffix in ("7d", "15d", "30d", "60d"):
        for col in [
            f"batter_bbe_{suffix}",
            f"batter_barrel_rate_{suffix}",
            f"batter_hard_hit_rate_{suffix}",
            f"batter_la_20_35_rate_{suffix}",
            f"batter_fly_ball_rate_{suffix}",
            f"batter_pull_air_rate_{suffix}",
            f"batter_avg_ev_{suffix}",
            f"batter_max_ev_{suffix}",
            f"pitcher_bbe_allowed_{suffix}",
            f"pitcher_barrel_allowed_rate_{suffix}",
            f"pitcher_hard_hit_allowed_rate_{suffix}",
            f"pitcher_la_20_35_allowed_rate_{suffix}",
            f"pitcher_fly_ball_rate_{suffix}",
            f"pitcher_pull_air_allowed_rate_{suffix}",
            f"pitcher_hrfb_rate_{suffix}",
            f"pitcher_avg_ev_allowed_{suffix}",
            f"pitcher_max_ev_allowed_{suffix}",
        ]:
            if col in df.columns:
                numeric.append(col)

    for col in ["lineup_spot", "hitterish_park", "pitcherish_park", "temp_f", "wind_speed_mph", "wind_toward_pull_field"]:
        if col in df.columns:
            numeric.append(col)

    bucket_cols = [c for c in df.columns if c.startswith("bucket_")]
    numeric += bucket_cols
    numeric = list(dict.fromkeys(numeric))

    categorical = [c for c in ["batter_hand", "pitcher_hand", "platoon_bucket", "park_bucket"] if c in df.columns]
    return numeric, categorical


def safe_metric(fn, y, p) -> Optional[float]:
    try:
        return float(fn(y, p))
    except Exception:
        return None


def print_metrics(label: str, y, p, brier_score_loss, log_loss, roc_auc_score) -> None:
    baseline = float(sum(y) / len(y)) if len(y) else 0.0
    baseline_pred = [baseline] * len(y)
    print(f"\n{label} METRICS")
    print("-" * (len(label) + 8))
    print(f"rows: {len(y)}")
    print(f"hr_hits: {int(sum(y))}")
    print(f"actual_rate: {baseline:.3%}")
    print(f"baseline_brier: {safe_metric(brier_score_loss, y, baseline_pred)}")
    print(f"model_brier:    {safe_metric(brier_score_loss, y, p)}")
    print(f"baseline_logloss: {safe_metric(log_loss, y, baseline_pred)}")
    print(f"model_logloss:    {safe_metric(log_loss, y, p)}")
    print(f"model_auc: {safe_metric(roc_auc_score, y, p)}")


def bucket_report(df, prob_col: str, y_col: str = "actual_hr", n_buckets: int = 10) -> None:
    print("\nPROBABILITY BUCKET REPORT")
    print("-------------------------")
    if len(df) == 0:
        return
    d = df.sort_values(prob_col, ascending=False).copy()
    d["bucket"] = range(len(d))
    d["bucket"] = (d["bucket"] * n_buckets / max(1, len(d))).astype(int) + 1
    for b in sorted(d["bucket"].unique()):
        sub = d[d["bucket"] == b]
        print(
            f"bucket_{b:02d} n={len(sub):4d} "
            f"avg_prob={sub[prob_col].mean():.3%} "
            f"actual_hr={sub[y_col].mean():.3%} "
            f"hits={int(sub[y_col].sum())}"
        )


def top_predictions(df, prob_col: str, n: int = 25) -> None:
    print(f"\nTOP {n} TEST PREDICTIONS")
    print("------------------------")
    cols = [
        "game_date", "batter_name", "lineup_spot", "batter_hand", "pitcher_hand",
        "platoon_bucket", "park_bucket", prob_col, "actual_hr",
        "batter_avg_ev_30d", "batter_barrel_rate_30d", "batter_hard_hit_rate_30d",
        "batter_fly_ball_rate_7d", "batter_pull_air_rate_30d",
    ]
    cols = [c for c in cols if c in df.columns]
    for _, r in df.sort_values(prob_col, ascending=False).head(n).iterrows():
        parts = []
        for c in cols:
            v = r.get(c)
            if hasattr(v, "date"):
                try:
                    v = v.date()
                except Exception:
                    pass
            if isinstance(v, float):
                parts.append(f"{c}={v:.4f}")
            else:
                parts.append(f"{c}={v}")
        print(" | ".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--test-date-frac", type=float, default=0.25)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    pd, ColumnTransformer, SimpleImputer, LogisticRegression, brier_score_loss, log_loss, roc_auc_score, Pipeline, OneHotEncoder, StandardScaler = require_imports()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    conn = sqlite3.connect(db)
    df = load_dataset(conn, pd)
    conn.close()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["actual_hr"] = df["actual_hr"].astype(int)
    df = add_manual_bucket_features(df)

    numeric, categorical = choose_features(df)
    dates = sorted(df["game_date"].dropna().unique())
    if len(dates) < 3:
        raise SystemExit("Need at least 3 unique dates for time split.")

    test_n_dates = max(1, int(math.ceil(len(dates) * args.test_date_frac)))
    train_dates = dates[:-test_n_dates]
    test_dates = dates[-test_n_dates:]

    train = df[df["game_date"].isin(train_dates)].copy()
    test = df[df["game_date"].isin(test_dates)].copy()

    print("HR MODEL TRAIN A")
    print("================")
    print(f"db: {db}")
    print(f"rows_total: {len(df)}")
    print(f"hr_hits_total: {int(df['actual_hr'].sum())}")
    print(f"overall_hr_rate: {df['actual_hr'].mean():.3%}")
    print(f"date_count: {len(dates)}")
    print(f"train_dates: {str(train['game_date'].min().date())} to {str(train['game_date'].max().date())} ({len(train)} rows)")
    print(f"test_dates:  {str(test['game_date'].min().date())} to {str(test['game_date'].max().date())} ({len(test)} rows)")
    print(f"numeric_features: {len(numeric)}")
    print(f"categorical_features: {len(categorical)}")

    if len(train) < 1000 or len(test) < 300:
        print("\nWARNING: sample is still small. Treat this as a pipeline/model sanity test only.")

    X_train = train[numeric + categorical]
    y_train = train["actual_hr"].astype(int)
    X_test = test[numeric + categorical]
    y_test = test["actual_hr"].astype(int)

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )

    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", C=0.5)),
    ])

    model.fit(X_train, y_train)
    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    print_metrics("TRAIN", list(y_train), list(p_train), brier_score_loss, log_loss, roc_auc_score)
    print_metrics("TEST", list(y_test), list(p_test), brier_score_loss, log_loss, roc_auc_score)

    test_out = test.copy()
    test_out["model_prob"] = p_test
    bucket_report(test_out, "model_prob")
    top_predictions(test_out, "model_prob", args.top)

    print("\nREAD")
    print("----")
    print("This is only the first leakage-aware diagnostic model.")
    print("Trust direction, not exact probabilities, until we have 10k+ rows and more dates.")
    print("A useful model should beat baseline Brier/logloss on the forward-time test set.")
    print("Next production step after more rows: calibrated model using validation dates, then FanDuel HR odds EV layer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
