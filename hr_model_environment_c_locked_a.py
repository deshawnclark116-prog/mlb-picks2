#!/usr/bin/env python3
"""
HR_MODEL_ENVIRONMENT_C_LOCKED_A

Clean locked validation report for the current best HR model:

C_PLUS_WEATHER_HINGES =
    REDUCED_B baseline
    + lagged league environment features
    + weather hinge features
    + wind/hot interaction

Excludes:
    - RAD features, because Report B did not earn them.
    - Platt production calibration, because Report B raw probabilities were better.

Pure validation split:
    Train: 2025 only
    Test:  2026 only

Run:
    python hr_model_environment_c_locked_a.py
"""

import argparse
import json
import math
import os
import sqlite3
from pathlib import Path


RATE_TO_SAMPLE = {
    "batter_hr_per_air_60d": "batter_bbe_60d",
    "batter_fly_ball_rate_30d": "batter_bbe_60d",
    "batter_barrel_rate_30d": "batter_bbe_60d",
    "pitcher_pull_air_allowed_rate_60d": "pitcher_bbe_allowed_60d",
    "pitcher_hard_hit_allowed_rate_7d": "pitcher_bbe_allowed_60d",
    "pitcher_hrfb_rate_30d": "pitcher_bbe_allowed_60d",
}


BASE_FEATURES = [
    "expected_pa_v1",
    "temp_f",
    "batter_max_ev_60d",
    "log_batter_bbe_60d",
    "log_pitcher_bbe_allowed_60d",
] + [f"{c}_shrunk" for c in RATE_TO_SAMPLE]


LEAGUE_FEATURES = [
    "league_hr_per_bbe_10d_lag",
    "league_hr_per_air_10d_lag",
    "league_barrel_rate_10d_lag",
    "league_avg_ev_10d_lag",
    "log_league_bbe_10d",
]


WEATHER_FEATURES = [
    "temp_over_75",
    "temp_over_85",
    "wind_out_component",
    "wind_out_and_hot",
]


MODEL_FEATURES = BASE_FEATURES + LEAGUE_FEATURES + WEATHER_FEATURES


RAW_COLS = [
    "expected_pa_v1",
    "temp_f",
    "wind_toward_pull_field",
    "weather_wind_mph",
    "batter_hr_per_air_60d",
    "batter_fly_ball_rate_30d",
    "batter_barrel_rate_30d",
    "batter_max_ev_60d",
    "pitcher_pull_air_allowed_rate_60d",
    "pitcher_hard_hit_allowed_rate_7d",
    "pitcher_hrfb_rate_30d",
    "batter_bbe_60d",
    "pitcher_bbe_allowed_60d",
    "league_hr_per_bbe_10d_lag",
    "league_hr_per_air_10d_lag",
    "league_barrel_rate_10d_lag",
    "league_avg_ev_10d_lag",
    "league_bbe_10d",
    "league_air_10d",
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
        print("  pip install pandas numpy scikit-learn")
        raise SystemExit(f"{type(e).__name__}: {e}")


def table_cols(conn, table):
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    except Exception:
        return []


def expr_from_tables(conn, alias, candidates):
    bg = set(table_cols(conn, "batter_games"))
    bf = set(table_cols(conn, "batter_game_features"))
    pf = set(table_cols(conn, "pitcher_game_features"))
    le = set(table_cols(conn, "league_env_lag_features"))

    for c in candidates:
        if c in bg:
            return f"bg.{c} AS {alias}"
        if c in bf:
            return f"bf.{c} AS {alias}"
        if c in pf:
            return f"pf.{c} AS {alias}"
        if c in le:
            return f"le.{c} AS {alias}"
    return f"NULL AS {alias}"


def load_dataset(conn, pd):
    candidate_map = {
        "expected_pa_v1": ["expected_pa_v1"],
        "temp_f": ["temp_f", "weather_temp_f"],
        "wind_toward_pull_field": ["wind_toward_pull_field"],
        "weather_wind_mph": ["weather_wind_mph", "wind_mph"],
        "batter_hr_per_air_60d": ["batter_hr_per_air_60d"],
        "batter_fly_ball_rate_30d": ["batter_fly_ball_rate_30d"],
        "batter_barrel_rate_30d": ["batter_barrel_rate_30d"],
        "batter_max_ev_60d": ["batter_max_ev_60d"],
        "pitcher_pull_air_allowed_rate_60d": ["pitcher_pull_air_allowed_rate_60d"],
        "pitcher_hard_hit_allowed_rate_7d": ["pitcher_hard_hit_allowed_rate_7d"],
        "pitcher_hrfb_rate_30d": ["pitcher_hrfb_rate_30d"],
        "batter_bbe_60d": ["batter_bbe_60d"],
        "pitcher_bbe_allowed_60d": ["pitcher_bbe_allowed_60d"],
        "league_hr_per_bbe_10d_lag": ["league_hr_per_bbe_10d_lag"],
        "league_hr_per_air_10d_lag": ["league_hr_per_air_10d_lag"],
        "league_barrel_rate_10d_lag": ["league_barrel_rate_10d_lag"],
        "league_avg_ev_10d_lag": ["league_avg_ev_10d_lag"],
        "league_bbe_10d": ["league_bbe_10d"],
        "league_air_10d": ["league_air_10d"],
    }

    bg = set(table_cols(conn, "batter_games"))
    def opt_bg(col):
        return f"bg.{col} AS {col}" if col in bg else f"NULL AS {col}"

    select = [
        "bg.game_date AS game_date",
        "bg.game_id AS game_id",
        "bg.batter_id AS batter_id",
        "bg.batter_name AS batter_name",
        opt_bg("team"),
        opt_bg("opponent"),
        opt_bg("venue"),
        "bg.lineup_spot AS lineup_spot",
        "bg.actual_hr AS actual_hr",
    ]

    for alias, candidates in candidate_map.items():
        select.append(expr_from_tables(conn, alias, candidates))

    sql = f"""
    SELECT {", ".join(select)}
    FROM batter_games bg
    LEFT JOIN batter_game_features bf
      ON bg.game_id = bf.game_id
     AND bg.batter_id = bf.batter_id
    LEFT JOIN pitcher_game_features pf
      ON bg.game_id = pf.game_id
     AND bg.batter_id = pf.batter_id
    LEFT JOIN league_env_lag_features le
      ON bg.game_date = le.game_date
    WHERE bg.actual_hr IS NOT NULL
    ORDER BY bg.game_date, bg.game_id, bg.lineup_spot
    """
    return pd.read_sql_query(sql, conn)


def boolish_to_float(v):
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        try:
            return 1.0 if float(v) > 0 else 0.0
        except Exception:
            return 0.0
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "out", "toward", "pull", "wind_out", "wind toward pull"):
        return 1.0
    return 0.0


def add_safe_features(train, test, shrink_k):
    for df in (train, test):
        df["log_batter_bbe_60d"] = df["batter_bbe_60d"].fillna(0).clip(lower=0).map(lambda x: math.log1p(x))
        df["log_pitcher_bbe_allowed_60d"] = df["pitcher_bbe_allowed_60d"].fillna(0).clip(lower=0).map(lambda x: math.log1p(x))
        df["log_league_bbe_10d"] = df["league_bbe_10d"].fillna(0).clip(lower=0).map(lambda x: math.log1p(x))

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


def add_weather_hinges(df):
    df["temp_over_75"] = (df["temp_f"].fillna(70) - 75).clip(lower=0)
    df["temp_over_85"] = (df["temp_f"].fillna(70) - 85).clip(lower=0)

    wind_flag = df["wind_toward_pull_field"].map(boolish_to_float)
    wind_mph = df["weather_wind_mph"].fillna(0).clip(lower=0, upper=60)
    df["wind_out_component"] = wind_flag * wind_mph
    df["wind_out_and_hot"] = df["wind_out_component"] * df["temp_over_75"]
    return df


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
    return f"{float(x):.8f}"


def overall_metrics(y, p, brier, logloss, auc):
    base = float(sum(y) / len(y))
    bp = [base] * len(y)
    return {
        "rows": len(y),
        "hr_hits": int(sum(y)),
        "actual_rate": base,
        "baseline_brier": safe_metric(brier, y, bp),
        "model_brier": safe_metric(brier, y, p),
        "baseline_logloss": safe_metric(logloss, y, bp),
        "model_logloss": safe_metric(logloss, y, p),
        "model_auc": safe_metric(auc, y, p),
    }


def print_metrics(metrics):
    print(f"rows: {metrics['rows']}")
    print(f"hr_hits: {metrics['hr_hits']}")
    print(f"actual_rate: {metrics['actual_rate']:.5f}")
    print(f"baseline_brier: {fmt(metrics['baseline_brier'])}")
    print(f"model_brier:    {fmt(metrics['model_brier'])}")
    print(f"baseline_logloss: {fmt(metrics['baseline_logloss'])}")
    print(f"model_logloss:    {fmt(metrics['model_logloss'])}")
    print(f"model_auc: {fmt(metrics['model_auc'])}")


def monthly_report(df):
    print("\nMONTHLY CALIBRATION")
    print("-------------------")
    d = df.copy()
    d["month"] = d["game_date"].dt.strftime("%Y-%m")
    for month, g in d.groupby("month"):
        print(
            f"{month} rows={len(g):5d} "
            f"actual={g['actual_hr'].mean():.4f} "
            f"pred={g['model_prob'].mean():.4f} "
            f"miss={g['model_prob'].mean() - g['actual_hr'].mean():+.4f} "
            f"spread[min/p25/med/p75/max]="
            f"{g['model_prob'].min():.4f}/{g['model_prob'].quantile(.25):.4f}/"
            f"{g['model_prob'].median():.4f}/{g['model_prob'].quantile(.75):.4f}/"
            f"{g['model_prob'].max():.4f} "
            f"league_hr_bbe={g['league_hr_per_bbe_10d_lag'].mean():.4f} "
            f"league_hr_air={g['league_hr_per_air_10d_lag'].mean():.4f}"
        )


def bucket_report(df, q=10):
    print("\n10-BUCKET CALIBRATION")
    print("---------------------")
    d = df.sort_values("model_prob", ascending=False).copy()
    d["bucket"] = range(len(d))
    d["bucket"] = (d["bucket"] * q / max(1, len(d))).astype(int) + 1
    for b in sorted(d["bucket"].unique()):
        g = d[d["bucket"] == b]
        print(
            f"bucket_{b:02d} n={len(g):5d} "
            f"mean_pred={g['model_prob'].mean():.4f} "
            f"actual={g['actual_hr'].mean():.4f} "
            f"range={g['model_prob'].min():.4f}-{g['model_prob'].max():.4f} "
            f"hits={int(g['actual_hr'].sum())}"
        )


def top5_audit(df, n=40):
    print("\nTOP 5% AUDIT")
    print("------------")
    cutoff = df["model_prob"].quantile(0.95)
    top = df[df["model_prob"] >= cutoff].sort_values("model_prob", ascending=False)
    print(f"cutoff: {cutoff:.5f}")
    print(f"rows: {len(top)}")
    print(f"mean_pred: {top['model_prob'].mean():.5f}")
    print(f"actual_hr: {top['actual_hr'].mean():.5f}")
    print(f"hits: {int(top['actual_hr'].sum())}")

    cols = [
        "game_date", "batter_name", "team", "opponent", "venue", "lineup_spot",
        "model_prob", "actual_hr", "expected_pa_v1", "temp_f",
        "league_hr_per_bbe_10d_lag", "league_hr_per_air_10d_lag",
        "batter_hr_per_air_60d_shrunk", "batter_bbe_60d",
        "batter_fly_ball_rate_30d", "batter_barrel_rate_30d", "batter_max_ev_60d",
        "pitcher_hrfb_rate_30d"
    ]
    cols = [c for c in cols if c in top.columns]

    print("\nSAMPLE:")
    for _, r in top.head(n).iterrows():
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


def coefficient_report(model):
    print("\nSTANDARDIZED COEFFICIENTS")
    print("-------------------------")
    try:
        coefs = model.named_steps["clf"].coef_[0]
        pairs = sorted(zip(MODEL_FEATURES, coefs), key=lambda x: abs(x[1]), reverse=True)
        for name, coef in pairs:
            print(f"{name}: {coef:+.5f}")
    except Exception as e:
        print("Could not extract coefficients:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--c", type=float, default=0.03)
    ap.add_argument("--shrink-k", type=float, default=75.0)
    ap.add_argument("--out-json", default="/data/hr_model/hr_environment_c_locked_report.json")
    args = ap.parse_args()

    pd, ColumnTransformer, SimpleImputer, LogisticRegression, brier, logloss, auc, Pipeline, StandardScaler = require_imports()

    conn = sqlite3.connect(args.db)
    df = load_dataset(conn, pd)
    conn.close()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["actual_hr"] = df["actual_hr"].astype(int)

    for c in RAW_COLS:
        if c != "wind_toward_pull_field":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    train = df[df["game_date"].dt.year == 2025].copy().sort_values("game_date")
    test = df[df["game_date"].dt.year == 2026].copy().sort_values("game_date")

    add_safe_features(train, test, args.shrink_k)
    add_weather_hinges(train)
    add_weather_hinges(test)

    print("HR_MODEL_ENVIRONMENT_C_LOCKED_A")
    print("===============================")
    print(f"db: {args.db}")
    print(f"train_2025_rows: {len(train)}")
    print(f"test_2026_rows: {len(test)}")
    print(f"C: {args.c}")
    print(f"shrink_k: {args.shrink_k}")
    print(f"features_n: {len(MODEL_FEATURES)}")
    print(f"missing_2026_league_lag_rows: {int(test['league_hr_per_bbe_10d_lag'].isna().sum())}")

    model = make_model(args.c, ColumnTransformer, SimpleImputer, LogisticRegression, Pipeline, StandardScaler)
    model.fit(train[MODEL_FEATURES], train["actual_hr"])

    pred = model.predict_proba(test[MODEL_FEATURES])[:, 1]
    test["model_prob"] = pred

    metrics = overall_metrics(test["actual_hr"].tolist(), pred.tolist(), brier, logloss, auc)

    print("\n2026 HOLDOUT METRICS")
    print("--------------------")
    print_metrics(metrics)

    monthly_report(test)
    bucket_report(test)
    top5_audit(test)
    coefficient_report(model)

    top_cutoff = float(test["model_prob"].quantile(0.95))
    top = test[test["model_prob"] >= top_cutoff]
    report = {
        "model_name": "ENVIRONMENT_C_LOCKED_A",
        "train": "2025 only",
        "test": "2026 only",
        "C": args.c,
        "shrink_k": args.shrink_k,
        "features": MODEL_FEATURES,
        "metrics": metrics,
        "top5": {
            "cutoff": top_cutoff,
            "rows": int(len(top)),
            "mean_pred": float(top["model_prob"].mean()),
            "actual_hr": float(top["actual_hr"].mean()),
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nLOCK READ")
    print("---------")
    print("This is the current best validated HR prediction model.")
    print("Keeps: REDUCED_B + league environment lag + weather hinges.")
    print("Excludes: RAD features and Platt calibration.")
    print(f"Wrote JSON report: {out_path}")
    print("Next accuracy layer: pitch-level Statcast ingestion + Bayesian pitch-type/zone overlap.")


if __name__ == "__main__":
    main()
