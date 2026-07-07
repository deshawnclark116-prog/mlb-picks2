#!/usr/bin/env python3
"""
HR_MODEL_2026_HOLDOUT_REPORT_B

A/B/C/D holdout report for the HR model after adding league environment features.

Pure split:
- Train: 2025 only
- Test:  2026 only

Compares:
A = REDUCED_B baseline
B = A + league environment lag features
C = B + weather hinge features
D = C + RAD / air-density features

Still uses the locked REDUCED_B mechanics:
- expected_pa_v1, not actual plate_appearances
- small-sample rate shrinkage with train-only priors
- C=0.03
- shrink_k=75
- optional blocked-2025 Platt calibration as comparison only

Run:
    python hr_model_2026_holdout_report_b.py
"""

import argparse
import math
import os
import sqlite3
from pathlib import Path


RAW_FEATURES = [
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


RAD_FEATURES = [
    "rad_inverse",
    "rad_x_air_event",
]


CONFIGS = {
    "A_BASELINE_REDUCED_B": BASE_FEATURES,
    "B_PLUS_LEAGUE_ENV": BASE_FEATURES + LEAGUE_FEATURES,
    "C_PLUS_WEATHER_HINGES": BASE_FEATURES + LEAGUE_FEATURES + WEATHER_FEATURES,
    "D_PLUS_RAD_FEATURES": BASE_FEATURES + LEAGUE_FEATURES + WEATHER_FEATURES + RAD_FEATURES,
}


# Approximate stadium altitude map. Unknown venues default to 0 ft.
# This is intentionally conservative; exact park metadata can replace this later.
ALTITUDE_BY_KEYWORD = {
    "coors": 5200,
    "denver": 5200,
    "chase": 1080,
    "phoenix": 1080,
    "truist": 1050,
    "atlanta": 1050,
    "globe life": 550,
    "arlington": 550,
    "kauffman": 910,
    "kansas city": 910,
    "target field": 840,
    "minnesota": 840,
    "great american": 550,
    "cincinnati": 550,
    "busch": 450,
    "st. louis": 450,
    "progressive": 650,
    "cleveland": 650,
    "american family": 635,
    "milwaukee": 635,
    "guaranteed rate": 590,
    "rate field": 590,
    "wrigley": 590,
    "chicago": 590,
    "comerica": 600,
    "detroit": 600,
    "pnc": 730,
    "pittsburgh": 730,
}


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
        "temp_f": ["temp_f", "weather_temp_f"],
        "wind_toward_pull_field": ["wind_toward_pull_field"],
        "weather_wind_mph": ["weather_wind_mph", "wind_mph"],
        "expected_pa_v1": ["expected_pa_v1"],
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

    optional_bg = set(table_cols(conn, "batter_games"))
    def opt_bg(col):
        return f"bg.{col} AS {col}" if col in optional_bg else f"NULL AS {col}"

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


def as_float_or_nan(pd, x):
    return pd.to_numeric(x, errors="coerce")


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


def altitude_from_venue(venue):
    if venue is None:
        return 0.0
    s = str(venue).lower()
    for key, alt in ALTITUDE_BY_KEYWORD.items():
        if key in s:
            return float(alt)
    return 0.0


def calculate_rad_inverse(temp_f, altitude_ft):
    # Relative to standard sea-level dry-air density approximation.
    # temp_r standard is 59F = 518.67R. Pressure decays with altitude.
    if temp_f is None or math.isnan(temp_f):
        temp_f = 70.0
    temp_r = float(temp_f) + 459.67
    if temp_r <= 0:
        temp_r = 529.67
    temp_ratio = 518.67 / temp_r
    pressure_ratio = math.exp(-float(altitude_ft) / 27300.0)
    rad = temp_ratio * pressure_ratio
    if rad <= 0:
        return 1.0
    return 1.0 / rad


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
            w = n / (n + shrink_k)
            df[f"{rate_col}_shrunk"] = prior + (raw - prior) * w

    return priors


def add_environment_derived_features(df):
    df["temp_over_75"] = (df["temp_f"].fillna(70) - 75).clip(lower=0)
    df["temp_over_85"] = (df["temp_f"].fillna(70) - 85).clip(lower=0)

    wind_flag = df["wind_toward_pull_field"].map(boolish_to_float)
    wind_mph = df["weather_wind_mph"].fillna(0).clip(lower=0, upper=60)
    df["wind_out_component"] = wind_flag * wind_mph
    df["wind_out_and_hot"] = df["wind_out_component"] * df["temp_over_75"]

    alt = df["venue"].map(altitude_from_venue) if "venue" in df.columns else 0
    df["park_altitude_ft"] = alt
    df["rad_inverse"] = [
        calculate_rad_inverse(t, a)
        for t, a in zip(df["temp_f"].fillna(70), df["park_altitude_ft"])
    ]
    df["rad_x_air_event"] = df["rad_inverse"] * df["league_hr_per_air_10d_lag"].fillna(df["league_hr_per_air_10d_lag"].median())

    return df


def make_model(C, features, ColumnTransformer, SimpleImputer, LogisticRegression, Pipeline, StandardScaler):
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), features),
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


def logit_from_prob(np, p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_platt_blocked(train_base, features, args, np, ColumnTransformer, SimpleImputer, LogisticRegression, Pipeline, StandardScaler):
    splits = [
        ("2025-06-15", "2025-06-16", "2025-07-31"),
        ("2025-07-31", "2025-08-01", "2025-10-15"),
    ]
    oof_preds = []
    oof_y = []

    for train_end, val_start, val_end in splits:
        idx_tr = train_base["game_date"] <= train_end
        idx_val = (train_base["game_date"] >= val_start) & (train_base["game_date"] <= val_end)

        if int(idx_tr.sum()) < 1000 or int(idx_val.sum()) < 500:
            continue

        fold_train = train_base.loc[idx_tr].copy()
        fold_val = train_base.loc[idx_val].copy()

        add_safe_features(fold_train, fold_val, args.shrink_k)
        add_environment_derived_features(fold_train)
        add_environment_derived_features(fold_val)

        fold_model = make_model(args.c, features, ColumnTransformer, SimpleImputer, LogisticRegression, Pipeline, StandardScaler)
        fold_model.fit(fold_train[features], fold_train["actual_hr"].astype(int))
        preds = fold_model.predict_proba(fold_val[features])[:, 1]

        oof_preds.extend(preds.tolist())
        oof_y.extend(fold_val["actual_hr"].astype(int).tolist())

    if len(oof_preds) < 1000 or len(set(oof_y)) < 2:
        return None

    oof_preds = np.array(oof_preds)
    oof_y = np.array(oof_y)
    oof_logits = logit_from_prob(np, oof_preds).reshape(-1, 1)

    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    platt.fit(oof_logits, oof_y)
    return platt


def evaluate_config(name, features, train_base, test_base, args, np, ColumnTransformer, SimpleImputer, LogisticRegression, brier, logloss, auc, Pipeline, StandardScaler):
    train = train_base.copy()
    test = test_base.copy()

    add_safe_features(train, test, args.shrink_k)
    add_environment_derived_features(train)
    add_environment_derived_features(test)

    model = make_model(args.c, features, ColumnTransformer, SimpleImputer, LogisticRegression, Pipeline, StandardScaler)
    model.fit(train[features], train["actual_hr"].astype(int))
    raw = model.predict_proba(test[features])[:, 1]

    platt = fit_platt_blocked(train_base.copy(), features, args, np, ColumnTransformer, SimpleImputer, LogisticRegression, Pipeline, StandardScaler)
    if platt is not None:
        platt_pred = platt.predict_proba(logit_from_prob(np, raw).reshape(-1, 1))[:, 1]
    else:
        platt_pred = raw

    out = test.copy()
    out["raw_prob"] = raw
    out["platt_prob"] = platt_pred

    y = out["actual_hr"].astype(int).tolist()
    base_rate = float(sum(y) / len(y))
    base_pred = [base_rate] * len(y)

    summary = {
        "name": name,
        "rows": len(out),
        "actual_rate": base_rate,
        "baseline_brier": safe_metric(brier, y, base_pred),
        "raw_brier": safe_metric(brier, y, raw.tolist()),
        "platt_brier": safe_metric(brier, y, platt_pred.tolist()),
        "baseline_logloss": safe_metric(logloss, y, base_pred),
        "raw_logloss": safe_metric(logloss, y, raw.tolist()),
        "platt_logloss": safe_metric(logloss, y, platt_pred.tolist()),
        "raw_auc": safe_metric(auc, y, raw.tolist()),
        "platt_auc": safe_metric(auc, y, platt_pred.tolist()),
        "features": features,
    }

    return summary, out


def monthly_report(out, prob_col):
    lines = []
    d = out.copy()
    d["month"] = d["game_date"].dt.strftime("%Y-%m")
    for month, g in d.groupby("month"):
        lines.append(
            f"{month} rows={len(g):5d} "
            f"actual={g['actual_hr'].mean():.4f} "
            f"pred_mean={g[prob_col].mean():.4f} "
            f"spread[min/p25/med/p75/max]="
            f"{g[prob_col].min():.4f}/{g[prob_col].quantile(.25):.4f}/"
            f"{g[prob_col].median():.4f}/{g[prob_col].quantile(.75):.4f}/"
            f"{g[prob_col].max():.4f} "
            f"league_hr_bbe={g['league_hr_per_bbe_10d_lag'].mean():.4f} "
            f"league_hr_air={g['league_hr_per_air_10d_lag'].mean():.4f}"
        )
    return lines


def bucket_report(out, prob_col, q=10):
    lines = []
    d = out.sort_values(prob_col, ascending=False).copy()
    d["bucket"] = range(len(d))
    d["bucket"] = (d["bucket"] * q / max(1, len(d))).astype(int) + 1
    for b in sorted(d["bucket"].unique()):
        g = d[d["bucket"] == b]
        lines.append(
            f"bucket_{b:02d} n={len(g):5d} "
            f"mean_pred={g[prob_col].mean():.4f} "
            f"actual={g['actual_hr'].mean():.4f} "
            f"range={g[prob_col].min():.4f}-{g[prob_col].max():.4f} "
            f"hits={int(g['actual_hr'].sum())}"
        )
    return lines


def top5_line(out, prob_col):
    cutoff = out[prob_col].quantile(0.95)
    top = out[out[prob_col] >= cutoff]
    return f"top5 cutoff={cutoff:.5f} rows={len(top)} mean_pred={top[prob_col].mean():.5f} actual={top['actual_hr'].mean():.5f} hits={int(top['actual_hr'].sum())}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--c", type=float, default=0.03)
    ap.add_argument("--shrink-k", type=float, default=75.0)
    args = ap.parse_args()

    np, pd, ColumnTransformer, SimpleImputer, LogisticRegression, brier, logloss, auc, Pipeline, StandardScaler = require_imports()

    conn = sqlite3.connect(args.db)
    df = load_dataset(conn, pd)
    conn.close()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["actual_hr"] = df["actual_hr"].astype(int)

    numeric_cols = [c for c in RAW_FEATURES if c not in ("wind_toward_pull_field",)]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    train = df[df["game_date"].dt.year == 2025].copy().sort_values("game_date")
    test = df[df["game_date"].dt.year == 2026].copy().sort_values("game_date")

    print("HR_MODEL_2026_HOLDOUT_REPORT_B")
    print("==============================")
    print(f"db: {args.db}")
    print(f"train_rows_2025: {len(train)}")
    print(f"test_rows_2026:  {len(test)}")
    print(f"C: {args.c}")
    print(f"shrink_k: {args.shrink_k}")

    missing_league = int(test["league_hr_per_bbe_10d_lag"].isna().sum())
    print(f"test_missing_league_lag_rows: {missing_league}")

    results = []
    outputs = {}

    for name, features in CONFIGS.items():
        print(f"\nRUNNING {name}")
        print("-" * (8 + len(name)))
        summary, out = evaluate_config(
            name, features, train, test, args, np,
            ColumnTransformer, SimpleImputer, LogisticRegression,
            brier, logloss, auc, Pipeline, StandardScaler
        )
        results.append(summary)
        outputs[name] = out
        print(f"features_n: {len(features)}")
        print(f"raw_brier: {fmt(summary['raw_brier'])} | platt_brier: {fmt(summary['platt_brier'])} | baseline: {fmt(summary['baseline_brier'])}")
        print(f"raw_logloss: {fmt(summary['raw_logloss'])} | platt_logloss: {fmt(summary['platt_logloss'])} | baseline: {fmt(summary['baseline_logloss'])}")
        print(f"raw_auc: {fmt(summary['raw_auc'])} | platt_auc: {fmt(summary['platt_auc'])}")
        print("raw", top5_line(out, "raw_prob"))
        print("platt", top5_line(out, "platt_prob"))

    print("\nSUMMARY COMPARISON")
    print("==================")
    print("name | raw_brier | platt_brier | raw_logloss | platt_logloss | raw_auc | platt_auc | top5_raw_actual | top5_platt_actual")
    for r in results:
        out = outputs[r["name"]]
        top_raw = out[out["raw_prob"] >= out["raw_prob"].quantile(.95)]
        top_platt = out[out["platt_prob"] >= out["platt_prob"].quantile(.95)]
        print(
            f"{r['name']} | "
            f"{fmt(r['raw_brier'])} | {fmt(r['platt_brier'])} | "
            f"{fmt(r['raw_logloss'])} | {fmt(r['platt_logloss'])} | "
            f"{fmt(r['raw_auc'])} | {fmt(r['platt_auc'])} | "
            f"{top_raw['actual_hr'].mean():.5f} | {top_platt['actual_hr'].mean():.5f}"
        )

    for name in CONFIGS:
        out = outputs[name]
        print(f"\nMONTHLY RAW: {name}")
        print("----------------" + "-" * len(name))
        for line in monthly_report(out, "raw_prob"):
            print(line)

        print(f"\nBUCKETS RAW: {name}")
        print("-------------" + "-" * len(name))
        for line in bucket_report(out, "raw_prob"):
            print(line)

    print("\nREAD")
    print("----")
    print("A is the locked REDUCED_B baseline.")
    print("B adds lagged league environment features.")
    print("C adds weather hinges and wind/hot interaction.")
    print("D adds RAD-inspired air-density features.")
    print("Only dethrone A if June/July calibration improves without hurting overall Brier/logloss, April calibration, AUC, or top-5 calibration.")


if __name__ == "__main__":
    main()
