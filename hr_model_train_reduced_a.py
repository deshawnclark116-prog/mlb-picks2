#!/usr/bin/env python3
"""
HR_MODEL_TRAIN_REDUCED_A

Reduced, leakage-aware logistic baseline for the HR SQLite dataset.

Important:
- Does NOT use actual postgame plate_appearances as a model feature.
- Uses expected_pa_v1 instead.
- Loads only a small selected feature set, not the full wide dataset.

Run:
    python hr_model_train_reduced_a.py
"""

import argparse
import math
import os
import sqlite3
from pathlib import Path

FEATURES = [
    "expected_pa_v1",
    "temp_f",
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


def default_db_path():
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    return p / "hr_model.sqlite" if p.parent.exists() else Path("./hr_model/hr_model.sqlite")


def require_imports():
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
        print("Missing dependency. Install with:")
        print("  pip install pandas scikit-learn numpy")
        raise SystemExit(f"{type(e).__name__}: {e}")


def table_cols(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def col_expr(conn, feature):
    bg = set(table_cols(conn, "batter_games"))
    bf = set(table_cols(conn, "batter_game_features"))
    pf = set(table_cols(conn, "pitcher_game_features"))
    if feature in bg:
        return f"bg.{feature} AS {feature}"
    if feature in bf:
        return f"bf.{feature} AS {feature}"
    if feature in pf:
        return f"pf.{feature} AS {feature}"
    return f"NULL AS {feature}"


def load_dataset(conn, pd):
    select = [
        "bg.game_date AS game_date",
        "bg.game_id AS game_id",
        "bg.batter_id AS batter_id",
        "bg.batter_name AS batter_name",
        "bg.lineup_spot AS lineup_spot",
        "bg.actual_hr AS actual_hr",
    ] + [col_expr(conn, f) for f in FEATURES]

    sql = f"""
    SELECT {", ".join(select)}
    FROM batter_games bg
    LEFT JOIN batter_game_features bf
      ON bg.game_id = bf.game_id
     AND bg.batter_id = bf.batter_id
    LEFT JOIN pitcher_game_features pf
      ON bg.game_id = pf.game_id
     AND bg.batter_id = pf.batter_id
    WHERE bg.actual_hr IS NOT NULL
    ORDER BY bg.game_date, bg.game_id, bg.lineup_spot
    """
    return pd.read_sql_query(sql, conn)


def metric(fn, y, p):
    try:
        return float(fn(y, p))
    except Exception:
        return None


def print_metrics(label, y, p, brier_score_loss, log_loss, roc_auc_score):
    baseline = float(sum(y) / len(y)) if len(y) else 0.0
    bp = [baseline] * len(y)
    print(f"\n{label} METRICS")
    print("-" * (len(label) + 8))
    print(f"rows: {len(y)}")
    print(f"hr_hits: {int(sum(y))}")
    print(f"actual_rate: {baseline:.3%}")
    print(f"baseline_brier: {metric(brier_score_loss, y, bp)}")
    print(f"model_brier:    {metric(brier_score_loss, y, p)}")
    print(f"baseline_logloss: {metric(log_loss, y, bp)}")
    print(f"model_logloss:    {metric(log_loss, y, p)}")
    print(f"model_auc: {metric(roc_auc_score, y, p)}")


def bucket_report(df, prob_col="model_prob", n_buckets=10):
    print("\nPROBABILITY BUCKET REPORT")
    print("-------------------------")
    d = df.sort_values(prob_col, ascending=False).copy()
    d["bucket"] = range(len(d))
    d["bucket"] = (d["bucket"] * n_buckets / max(1, len(d))).astype(int) + 1
    for b in sorted(d["bucket"].unique()):
        sub = d[d["bucket"] == b]
        print(
            f"bucket_{b:02d} n={len(sub):5d} "
            f"avg_prob={sub[prob_col].mean():.3%} "
            f"actual_hr={sub['actual_hr'].mean():.3%} "
            f"hits={int(sub['actual_hr'].sum())}"
        )


def top_predictions(df, n=25):
    print(f"\nTOP {n} TEST PREDICTIONS")
    print("------------------------")
    cols = ["game_date", "batter_name", "lineup_spot", "model_prob", "actual_hr"] + FEATURES
    cols = [c for c in cols if c in df.columns]
    for _, r in df.sort_values("model_prob", ascending=False).head(n).iterrows():
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--test-date-frac", type=float, default=0.25)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--c", type=float, default=0.15, help="Lower C = stronger L2 regularization.")
    args = ap.parse_args()

    pd, ColumnTransformer, SimpleImputer, LogisticRegression, brier_score_loss, log_loss, roc_auc_score, Pipeline, StandardScaler = require_imports()

    conn = sqlite3.connect(args.db)
    df = load_dataset(conn, pd)
    conn.close()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["actual_hr"] = df["actual_hr"].astype(int)
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    dates = sorted(df["game_date"].dropna().unique())
    test_n_dates = max(1, int(math.ceil(len(dates) * args.test_date_frac)))
    train_dates = dates[:-test_n_dates]
    test_dates = dates[-test_n_dates:]

    train = df[df["game_date"].isin(train_dates)].copy()
    test = df[df["game_date"].isin(test_dates)].copy()

    print("HR_MODEL_TRAIN_REDUCED_A")
    print("========================")
    print(f"db: {args.db}")
    print(f"rows_total: {len(df)}")
    print(f"hr_hits_total: {int(df['actual_hr'].sum())}")
    print(f"overall_hr_rate: {df['actual_hr'].mean():.3%}")
    print(f"date_count: {len(dates)}")
    print(f"train_dates: {train['game_date'].min().date()} to {train['game_date'].max().date()} ({len(train)} rows)")
    print(f"test_dates:  {test['game_date'].min().date()} to {test['game_date'].max().date()} ({len(test)} rows)")
    print(f"features: {FEATURES}")
    print(f"logistic_C: {args.c}")

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), FEATURES),
        ],
        remainder="drop",
    )

    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", C=args.c)),
    ])

    X_train = train[FEATURES]
    y_train = train["actual_hr"]
    X_test = test[FEATURES]
    y_test = test["actual_hr"]

    model.fit(X_train, y_train)
    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    print_metrics("TRAIN", list(y_train), list(p_train), brier_score_loss, log_loss, roc_auc_score)
    print_metrics("TEST", list(y_test), list(p_test), brier_score_loss, log_loss, roc_auc_score)

    out = test.copy()
    out["model_prob"] = p_test
    bucket_report(out)
    top_predictions(out, args.top)

    print("\nREAD")
    print("----")
    print("This reduced model avoids actual plate_appearances because that is postgame leakage.")
    print("Use expected_pa_v1 for pregame opportunity.")
    print("This is still not a betting model. We need forward-holdout calibration and FanDuel HR odds before EV decisions.")


if __name__ == "__main__":
    main()
