#!/usr/bin/env python3
"""
HR_MODEL_2026_HOLDOUT_REPORT_A

Pure season holdout diagnostic:
- Train base HR model on 2025 only.
- Test on 2026 only.
- Uses locked REDUCED_B architecture:
    C = 0.03
    shrink_k = 75
    expected_pa_v1 instead of actual plate_appearances
    small-sample rate shrinkage using train-only priors
- Also evaluates optional Platt calibration using blocked 2025 temporal folds.
- Does not use shuffled KFold.
- Does not use 2026 to fit model, priors, imputer/scaler, or calibrator.

Run:
    python hr_model_2026_holdout_report_a.py
"""

import argparse
import math
import os
import sqlite3
from pathlib import Path


RAW_FEATURES = [
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


RATE_TO_SAMPLE = {
    "batter_hr_per_air_60d": "batter_bbe_60d",
    "batter_fly_ball_rate_30d": "batter_bbe_60d",
    "batter_barrel_rate_30d": "batter_bbe_60d",
    "pitcher_pull_air_allowed_rate_60d": "pitcher_bbe_allowed_60d",
    "pitcher_hard_hit_allowed_rate_7d": "pitcher_bbe_allowed_60d",
    "pitcher_hrfb_rate_30d": "pitcher_bbe_allowed_60d",
}


MODEL_FEATURES = [
    "expected_pa_v1",
    "temp_f",
    "batter_max_ev_60d",
    "log_batter_bbe_60d",
    "log_pitcher_bbe_allowed_60d",
] + [f"{c}_shrunk" for c in RATE_TO_SAMPLE]


AUDIT_COLS = [
    "game_date",
    "batter_name",
    "batter_id",
    "team",
    "opponent",
    "venue",
    "lineup_spot",
    "expected_pa_v1",
    "temp_f",
    "raw_prob",
    "platt_prob",
    "actual_hr",
    "batter_hr_per_air_60d",
    "batter_hr_per_air_60d_shrunk",
    "batter_bbe_60d",
    "batter_fly_ball_rate_30d",
    "batter_barrel_rate_30d",
    "batter_max_ev_60d",
    "pitcher_hrfb_rate_30d",
    "pitcher_bbe_allowed_60d",
]


def default_db_path():
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    return p / "hr_model.sqlite" if p.parent.exists() else Path("./hr_model/hr_model.sqlite")


def require_imports():
    try:
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return np, pd, ColumnTransformer, SimpleImputer, LogisticRegression, brier_score_loss, log_loss, roc_auc_score, Pipeline, StandardScaler
    except Exception as e:
        print("Missing dependency. Install with:")
        print("  pip install pandas numpy scikit-learn")
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


def optional_col_expr(conn, col):
    bg = set(table_cols(conn, "batter_games"))
    if col in bg:
        return f"bg.{col} AS {col}"
    return f"NULL AS {col}"


def load_dataset(conn, pd):
    select = [
        "bg.game_date AS game_date",
        "bg.game_id AS game_id",
        "bg.batter_id AS batter_id",
        "bg.batter_name AS batter_name",
        optional_col_expr(conn, "team"),
        optional_col_expr(conn, "opponent"),
        optional_col_expr(conn, "venue"),
        "bg.lineup_spot AS lineup_spot",
        "bg.actual_hr AS actual_hr",
    ] + [col_expr(conn, f) for f in RAW_FEATURES]

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


def add_safe_features(train, test, shrink_k):
    for df in (train, test):
        df["log_batter_bbe_60d"] = df["batter_bbe_60d"].fillna(0).clip(lower=0).map(lambda x: math.log1p(x))
        df["log_pitcher_bbe_allowed_60d"] = df["pitcher_bbe_allowed_60d"].fillna(0).clip(lower=0).map(lambda x: math.log1p(x))

    priors = {}
    for rate_col, sample_col in RATE_TO_SAMPLE.items():
        prior = float(train[rate_col].dropna().median()) if train[rate_col].notna().sum() else 0.0
        priors[rate_col] = prior
        for df in (train, test):
            raw = df[rate_col].fillna(prior).clip(lower=0, upper=1)
            n = df[sample_col].fillna(0).clip(lower=0)
            weight = n / (n + shrink_k)
            df[f"{rate_col}_shrunk"] = prior + (raw - prior) * weight

    return priors


def make_model(C, ColumnTransformer, SimpleImputer, LogisticRegression, Pipeline, StandardScaler):
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), MODEL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", C=C)),
    ])


def safe_metric(fn, y, p):
    try:
        return float(fn(y, p))
    except Exception:
        return None


def fmt(x):
    if x is None:
        return "None"
    try:
        return f"{float(x):.8f}"
    except Exception:
        return str(x)


def print_overall(label, y, p, brier, logloss, auc):
    base = float(sum(y) / len(y)) if len(y) else 0.0
    bp = [base] * len(y)
    print(f"\n{label}")
    print("=" * len(label))
    print(f"rows: {len(y)}")
    print(f"hr_hits: {int(sum(y))}")
    print(f"actual_hr_rate: {base:.5f}")
    print(f"baseline_brier: {fmt(safe_metric(brier, y, bp))}")
    print(f"model_brier:    {fmt(safe_metric(brier, y, p))}")
    print(f"baseline_logloss: {fmt(safe_metric(logloss, y, bp))}")
    print(f"model_logloss:    {fmt(safe_metric(logloss, y, p))}")
    print(f"model_auc: {fmt(safe_metric(auc, y, p))}")


def logit_from_prob(np, p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_platt_blocked(train, args, np, ColumnTransformer, SimpleImputer, LogisticRegression, Pipeline, StandardScaler):
    splits = [
        ("2025-06-15", "2025-06-16", "2025-07-31"),
        ("2025-07-31", "2025-08-01", "2025-10-15"),
    ]

    oof_preds = []
    oof_y = []

    print("\nBLOCKED 2025 PLATT OOF")
    print("----------------------")

    for train_end, val_start, val_end in splits:
        idx_tr = train["game_date"] <= train_end
        idx_val = (train["game_date"] >= val_start) & (train["game_date"] <= val_end)

        n_tr = int(idx_tr.sum())
        n_val = int(idx_val.sum())
        print(f"fold train<= {train_end} rows={n_tr} | val {val_start}..{val_end} rows={n_val}")

        if n_tr < 1000 or n_val < 500:
            print("  skipped: insufficient rows")
            continue

        fold_train = train.loc[idx_tr].copy()
        fold_val = train.loc[idx_val].copy()

        add_safe_features(fold_train, fold_val, args.shrink_k)

        fold_model = make_model(args.c, ColumnTransformer, SimpleImputer, LogisticRegression, Pipeline, StandardScaler)
        fold_model.fit(fold_train[MODEL_FEATURES], fold_train["actual_hr"].astype(int))
        preds = fold_model.predict_proba(fold_val[MODEL_FEATURES])[:, 1]

        oof_preds.extend(preds.tolist())
        oof_y.extend(fold_val["actual_hr"].astype(int).tolist())

    if len(oof_preds) < 1000 or len(set(oof_y)) < 2:
        print("Platt skipped: not enough OOF calibration rows.")
        return None, None, None

    oof_preds = np.array(oof_preds)
    oof_y = np.array(oof_y)
    oof_logits = logit_from_prob(np, oof_preds).reshape(-1, 1)

    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    platt.fit(oof_logits, oof_y)

    print(f"OOF rows: {len(oof_y)}")
    print(f"OOF actual_rate: {oof_y.mean():.5f}")
    print(f"OOF raw_pred_mean: {oof_preds.mean():.5f}")
    return platt, oof_preds, oof_y


def monthly_report(df, prob_col, brier, logloss, auc):
    print(f"\nMONTHLY PERFORMANCE AND SPREAD: {prob_col}")
    print("--------------------------------" + "-" * len(prob_col))
    d = df.copy()
    d["month"] = d["game_date"].dt.strftime("%Y-%m")
    for month, g in d.groupby("month"):
        y = g["actual_hr"].astype(int).tolist()
        p = g[prob_col].tolist()
        print(
            f"{month} rows={len(g):5d} actual={g['actual_hr'].mean():.4f} "
            f"pred_mean={g[prob_col].mean():.4f} "
            f"brier={fmt(safe_metric(brier, y, p))} "
            f"logloss={fmt(safe_metric(logloss, y, p))} "
            f"auc={fmt(safe_metric(auc, y, p))} "
            f"spread[min/p25/med/p75/max]="
            f"{g[prob_col].min():.4f}/{g[prob_col].quantile(.25):.4f}/"
            f"{g[prob_col].median():.4f}/{g[prob_col].quantile(.75):.4f}/"
            f"{g[prob_col].max():.4f} "
            f"avg_batter_bbe60={g['batter_bbe_60d'].mean():.1f} "
            f"avg_pitcher_bbe60={g['pitcher_bbe_allowed_60d'].mean():.1f}"
        )


def bucket_report(df, prob_col, q=10):
    print(f"\n10-BUCKET CALIBRATION: {prob_col}")
    print("----------------------" + "-" * len(prob_col))
    d = df.sort_values(prob_col, ascending=False).copy()
    d["bucket"] = range(len(d))
    d["bucket"] = (d["bucket"] * q / max(1, len(d))).astype(int) + 1
    for b in sorted(d["bucket"].unique()):
        g = d[d["bucket"] == b]
        print(
            f"bucket_{b:02d} n={len(g):5d} "
            f"range={g[prob_col].min():.4f}-{g[prob_col].max():.4f} "
            f"mean_pred={g[prob_col].mean():.4f} "
            f"actual_hr={g['actual_hr'].mean():.4f} "
            f"hits={int(g['actual_hr'].sum())}"
        )


def lineup_report(df, prob_col):
    print(f"\nLINEUP SLOT VALIDATION: {prob_col}")
    print("-----------------------" + "-" * len(prob_col))
    for slot, g in df.groupby("lineup_spot"):
        print(
            f"slot={slot} n={len(g):5d} "
            f"avg_expected_pa={g['expected_pa_v1'].mean():.3f} "
            f"avg_pred={g[prob_col].mean():.4f} "
            f"actual_hr={g['actual_hr'].mean():.4f}"
        )


def top_audit(df, prob_col, top_pct=0.05, top_n=40):
    print(f"\nTOP {int(top_pct*100)}% AUDIT: {prob_col}")
    print("----------------" + "-" * len(prob_col))
    cutoff = df[prob_col].quantile(1 - top_pct)
    top = df[df[prob_col] >= cutoff].sort_values(prob_col, ascending=False).copy()
    print(f"cutoff: {cutoff:.5f}")
    print(f"rows: {len(top)}")
    print(f"actual_hr_rate: {top['actual_hr'].mean():.5f}")
    print(f"mean_pred: {top[prob_col].mean():.5f}")

    cols = [c for c in AUDIT_COLS if c in top.columns]
    print("\nSAMPLE:")
    for _, r in top.head(top_n).iterrows():
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
    ap.add_argument("--c", type=float, default=0.03)
    ap.add_argument("--shrink-k", type=float, default=75.0)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    np, pd, ColumnTransformer, SimpleImputer, LogisticRegression, brier, logloss, auc, Pipeline, StandardScaler = require_imports()

    conn = sqlite3.connect(args.db)
    df = load_dataset(conn, pd)
    conn.close()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["actual_hr"] = df["actual_hr"].astype(int)

    for f in RAW_FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    train = df[df["game_date"].dt.year == 2025].copy().sort_values("game_date")
    test = df[df["game_date"].dt.year == 2026].copy().sort_values("game_date")

    print("HR_MODEL_2026_HOLDOUT_REPORT_A")
    print("==============================")
    print(f"db: {args.db}")
    print(f"rows_total: {len(df)}")
    print(f"train_2025_rows: {len(train)}")
    print(f"test_2026_rows: {len(test)}")
    print(f"train_dates: {train['game_date'].min().date()} to {train['game_date'].max().date()}")
    print(f"test_dates:  {test['game_date'].min().date()} to {test['game_date'].max().date()}")
    print(f"base_model_C: {args.c}")
    print(f"shrink_k: {args.shrink_k}")
    print(f"features: {MODEL_FEATURES}")

    if len(train) < 1000 or len(test) < 1000:
        raise SystemExit("Not enough train/test rows for 2025->2026 holdout.")

    priors = add_safe_features(train, test, args.shrink_k)

    print("\nTRAIN-ONLY PRIORS USED FOR FINAL 2026 SHRINKAGE")
    print("-----------------------------------------------")
    for k, v in priors.items():
        print(f"{k}: {v:.5f}")

    base_model = make_model(args.c, ColumnTransformer, SimpleImputer, LogisticRegression, Pipeline, StandardScaler)
    base_model.fit(train[MODEL_FEATURES], train["actual_hr"].astype(int))

    raw_test_preds = base_model.predict_proba(test[MODEL_FEATURES])[:, 1]
    test["raw_prob"] = raw_test_preds

    print_overall("2026 HOLDOUT RAW LOCKED MODEL", test["actual_hr"].astype(int).tolist(), raw_test_preds.tolist(), brier, logloss, auc)

    platt, oof_preds, oof_y = fit_platt_blocked(train.copy(), args, np, ColumnTransformer, SimpleImputer, LogisticRegression, Pipeline, StandardScaler)

    if platt is not None:
        raw_logits = logit_from_prob(np, raw_test_preds).reshape(-1, 1)
        platt_preds = platt.predict_proba(raw_logits)[:, 1]
        test["platt_prob"] = platt_preds
        print_overall("2026 HOLDOUT PLATT-CALIBRATED MODEL", test["actual_hr"].astype(int).tolist(), platt_preds.tolist(), brier, logloss, auc)
    else:
        test["platt_prob"] = test["raw_prob"]

    monthly_report(test, "raw_prob", brier, logloss, auc)
    bucket_report(test, "raw_prob")
    lineup_report(test, "raw_prob")
    top_audit(test, "raw_prob", top_pct=0.05, top_n=args.top)

    if platt is not None:
        monthly_report(test, "platt_prob", brier, logloss, auc)
        bucket_report(test, "platt_prob")
        lineup_report(test, "platt_prob")
        top_audit(test, "platt_prob", top_pct=0.05, top_n=args.top)

    print("\nREAD")
    print("----")
    print("Raw model is the locked REDUCED_B baseline: C=0.03, shrink_k=75.")
    print("Platt model is a comparison layer fit only on blocked 2025 OOF predictions.")
    print("2026 is untouched holdout data for both model selection and calibration.")


if __name__ == "__main__":
    main()
