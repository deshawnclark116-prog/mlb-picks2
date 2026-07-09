#!/usr/bin/env python3
"""
HR_HAND_PITCHER_VECTOR_LITE_B

Memory-lite handedness-conditioned pitcher gate for a 512 MB Render instance.

What it does:
- Reuses the already-built table: pitcher_hand_features_a
- Does NOT rebuild raw feature tables
- Loads only exact columns needed for ENVIRONMENT_C_LOCKED_A + handedness vectors
- Uses the exact locked baseline feature construction
- Fits one logistic model at a time
- Flushes every print immediately
- Writes completed results to JSON after every model
- Evaluates both all-2026 and traditional-starter subset from the same predictions
  (no duplicate refits)

Run:
    python -u hr_hand_pitcher_vector_lite_b.py 2>&1 | tee /data/hr_model/hr_hand_pitcher_vector_lite_b.log

Paste back:
    HAND FEATURE COVERAGE
    HAND HR/BBE DECILES
    SUMMARY COMPARISON - ALL 2026
    DELTAS VS A0 - ALL 2026
    SUMMARY COMPARISON - TRADITIONAL STARTER EVAL 2026
    DELTAS VS A0 - TRADITIONAL STARTER EVAL 2026
    HAND FEATURE COEFFICIENTS
    READ
"""

import argparse
import gc
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

# Make stdout line-buffered so tee/log output survives as much as possible.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


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

LOCKED_FEATURES = BASE_FEATURES + LEAGUE_FEATURES + WEATHER_FEATURES

HAND_H1 = [
    "opp_hand_hr_per_bbe_365d_shrunk",
]

HAND_H2 = [
    "opp_hand_hr_per_bbe_365d_shrunk",
    "opp_hand_hr_per_pa_365d_shrunk",
    "opp_hand_k_per_pa_365d_shrunk",
    "opp_hand_gb_per_bbe_365d_shrunk",
    "opp_hand_barrel_per_bbe_365d_shrunk",
]

HAND_H3 = HAND_H2 + [
    "log_opp_hand_bbe_365d",
    "log_opp_hand_pa_365d",
]

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

HAND_LOAD_COLS = [
    "first_pitcher_id",
    "batter_stand",
    "p_throws",
    "first_pitcher_role",
    "opp_hand_pitches_365d",
    "opp_hand_pa_365d",
    "opp_hand_bbe_365d",
    "opp_hand_hr_365d",
    "opp_hand_hr_per_bbe_365d_raw",
    "opp_hand_hr_per_pa_365d_raw",
    "opp_hand_k_per_pa_365d_raw",
    "opp_hand_gb_per_bbe_365d_raw",
    "opp_hand_barrel_per_bbe_365d_raw",
    "opp_hand_hr_per_bbe_365d_shrunk",
    "opp_hand_hr_per_pa_365d_shrunk",
    "opp_hand_k_per_pa_365d_shrunk",
    "opp_hand_gb_per_bbe_365d_shrunk",
    "opp_hand_barrel_per_bbe_365d_shrunk",
    "log_opp_hand_bbe_365d",
    "log_opp_hand_pa_365d",
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
        return (
            np, pd, ColumnTransformer, SimpleImputer, LogisticRegression,
            brier_score_loss, log_loss, roc_auc_score, Pipeline, StandardScaler
        )
    except Exception as e:
        raise SystemExit("Missing dependency. Install pandas numpy scikit-learn.\n" + repr(e))


def table_cols(conn, table):
    try:
        return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
    except Exception:
        return []


def table_exists(conn, table):
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (table,),
    ).fetchone() is not None


def expr_from_tables(conn, alias, candidates):
    bg = set(table_cols(conn, "batter_games"))
    bf = set(table_cols(conn, "batter_game_features"))
    pf = set(table_cols(conn, "pitcher_game_features"))
    le = set(table_cols(conn, "league_env_lag_features"))

    for c in candidates:
        if c in bg:
            return f'bg."{c}" AS "{alias}"'
        if c in bf:
            return f'bf."{c}" AS "{alias}"'
        if c in pf:
            return f'pf."{c}" AS "{alias}"'
        if c in le:
            return f'le."{c}" AS "{alias}"'
    return f'NULL AS "{alias}"'


def load_dataset(conn, pd):
    if not table_exists(conn, "pitcher_hand_features_a"):
        raise SystemExit(
            "Missing pitcher_hand_features_a. The expensive handedness feature build did not survive."
        )

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

    bg_cols = set(table_cols(conn, "batter_games"))
    hand_cols = set(table_cols(conn, "pitcher_hand_features_a"))

    select = [
        "bg.game_date AS game_date",
        "CAST(bg.game_id AS TEXT) AS game_id",
        "bg.batter_id AS batter_id",
        "bg.actual_hr AS actual_hr",
    ]

    for alias, candidates in candidate_map.items():
        select.append(expr_from_tables(conn, alias, candidates))

    for c in HAND_LOAD_COLS:
        if c in hand_cols:
            select.append(f'h."{c}" AS "{c}"')
        else:
            select.append(f'NULL AS "{c}"')

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
    LEFT JOIN pitcher_hand_features_a h
      ON CAST(bg.game_id AS TEXT) = h.game_id
     AND bg.batter_id = h.batter_id
    WHERE bg.actual_hr IS NOT NULL
    ORDER BY bg.game_date, bg.game_id, bg.batter_id
    """

    print("loading only required columns...", flush=True)
    df = pd.read_sql_query(sql, conn)
    print(f"loaded rows={len(df):,} cols={len(df.columns)}", flush=True)
    return df


def boolish_to_float(v):
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        try:
            return 1.0 if float(v) > 0 else 0.0
        except Exception:
            return 0.0
    s = str(v).strip().lower()
    if s in (
        "1", "true", "t", "yes", "y", "out", "toward",
        "pull", "wind_out", "wind toward pull"
    ):
        return 1.0
    return 0.0


def add_safe_features(train, test, shrink_k):
    for df in (train, test):
        df["log_batter_bbe_60d"] = (
            df["batter_bbe_60d"].fillna(0).clip(lower=0).map(math.log1p)
        )
        df["log_pitcher_bbe_allowed_60d"] = (
            df["pitcher_bbe_allowed_60d"].fillna(0).clip(lower=0).map(math.log1p)
        )
        df["log_league_bbe_10d"] = (
            df["league_bbe_10d"].fillna(0).clip(lower=0).map(math.log1p)
        )

    for rate_col, sample_col in RATE_TO_SAMPLE.items():
        prior = (
            float(train[rate_col].dropna().median())
            if train[rate_col].notna().sum()
            else 0.0
        )
        for df in (train, test):
            raw = df[rate_col].fillna(prior).clip(lower=0, upper=1)
            n = df[sample_col].fillna(0).clip(lower=0)
            weight = n / (n + shrink_k)
            df[f"{rate_col}_shrunk"] = prior + (raw - prior) * weight


def add_weather_hinges(df):
    df["temp_over_75"] = (df["temp_f"].fillna(70) - 75).clip(lower=0)
    df["temp_over_85"] = (df["temp_f"].fillna(70) - 85).clip(lower=0)

    wind_flag = df["wind_toward_pull_field"].map(boolish_to_float)
    wind_mph = df["weather_wind_mph"].fillna(0).clip(lower=0, upper=60)
    df["wind_out_component"] = wind_flag * wind_mph
    df["wind_out_and_hot"] = df["wind_out_component"] * df["temp_over_75"]


def make_model(features, C, ColumnTransformer, SimpleImputer, LogisticRegression, Pipeline, StandardScaler):
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                features,
            )
        ],
        remainder="drop",
    )
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", C=C)),
    ])


def metric_bundle(np, y, p, brier, logloss, auc):
    y_arr = np.asarray(y, dtype=np.int8)
    p_arr = np.asarray(p, dtype=np.float64)

    out = {
        "rows": int(len(y_arr)),
        "hr_hits": int(y_arr.sum()),
        "actual_rate": float(y_arr.mean()),
        "brier": float(brier(y_arr, p_arr)),
        "logloss": float(logloss(y_arr, p_arr, labels=[0, 1])),
        "auc": float(auc(y_arr, p_arr)),
    }

    n = len(y_arr)
    k = max(1, int(math.ceil(n * 0.05)))
    top_idx = np.argpartition(p_arr, -k)[-k:]
    out["top5_rows"] = int(k)
    out["top5_hits"] = int(y_arr[top_idx].sum())
    out["top5_actual"] = float(y_arr[top_idx].mean())
    out["top5_pred"] = float(p_arr[top_idx].mean())
    return out


def decile_report(pd, d, feature, sample_col, threshold):
    x = d[
        d[feature].notna()
        & (d[sample_col].fillna(0) >= threshold)
    ][[feature, "actual_hr"]].copy()

    if len(x) < 200:
        print(
            f"HAND HR/BBE DECILES: {feature} min_{sample_col}={threshold} "
            f"not enough rows n={len(x)}",
            flush=True,
        )
        return

    try:
        x["decile"] = pd.qcut(x[feature], 10, labels=False, duplicates="drop") + 1
    except Exception as e:
        print(f"decile error: {e}", flush=True)
        return

    print(
        f"\nHAND HR/BBE DECILES: {feature} min_{sample_col}={threshold}",
        flush=True,
    )
    print("-" * 84, flush=True)

    for dec, g in x.groupby("decile"):
        print(
            f"decile={int(dec):02d} n={len(g):5d} "
            f"mean={g[feature].mean():.5f} "
            f"actual_hr={g['actual_hr'].mean():.5f} "
            f"hits={int(g['actual_hr'].sum())}",
            flush=True,
        )


def write_results(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_summary(title, rows):
    print(f"\n{title}", flush=True)
    print("-" * len(title), flush=True)
    for r in rows:
        print(
            f"{r['label']}: "
            f"brier={r['brier']:.8f} "
            f"logloss={r['logloss']:.8f} "
            f"auc={r['auc']:.8f} "
            f"top5_actual={r['top5_actual']:.5f} "
            f"top5_pred={r['top5_pred']:.5f} "
            f"top5_hits={r['top5_hits']}/{r['top5_rows']} "
            f"rows={r['rows']}",
            flush=True,
        )


def print_deltas(title, base, rows):
    print(f"\n{title}", flush=True)
    print("-" * len(title), flush=True)
    for r in rows:
        print(
            f"{r['label']} vs {base['label']}: "
            f"brier_delta={r['brier'] - base['brier']:+.8f} "
            f"logloss_delta={r['logloss'] - base['logloss']:+.8f} "
            f"auc_delta={r['auc'] - base['auc']:+.8f} "
            f"top5_actual_delta={r['top5_actual'] - base['top5_actual']:+.5f}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--c", type=float, default=0.03)
    ap.add_argument("--shrink-k", type=float, default=75.0)
    ap.add_argument(
        "--out-json",
        default="/data/hr_model/hr_hand_pitcher_vector_lite_b_results.json",
    )
    args = ap.parse_args()

    (
        np, pd, ColumnTransformer, SimpleImputer, LogisticRegression,
        brier, logloss, auc, Pipeline, StandardScaler
    ) = require_imports()

    print("HR_HAND_PITCHER_VECTOR_LITE_B", flush=True)
    print("=============================", flush=True)
    print(f"db: {args.db}", flush=True)
    print("mode: reuse existing handedness tables; no rebuild", flush=True)

    conn = sqlite3.connect(args.db)
    df = load_dataset(conn, pd)
    conn.close()

    print("parsing and compacting data...", flush=True)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["actual_hr"] = pd.to_numeric(df["actual_hr"], errors="coerce").fillna(0).astype("int8")

    numeric_cols = [
        c for c in RAW_COLS + HAND_LOAD_COLS
        if c in df.columns and c not in ("batter_stand", "p_throws", "first_pitcher_role")
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    train = df[df["game_date"].dt.year == 2025].copy()
    test = df[df["game_date"].dt.year == 2026].copy()
    del df
    gc.collect()

    add_safe_features(train, test, args.shrink_k)
    add_weather_hinges(train)
    add_weather_hinges(test)

    # Compact derived numeric columns too.
    for frame in (train, test):
        for c in LOCKED_FEATURES + HAND_H3:
            if c in frame.columns:
                frame[c] = pd.to_numeric(frame[c], errors="coerce").astype("float32")

    print("\nHAND FEATURE COVERAGE", flush=True)
    print("---------------------", flush=True)
    print(f"train_2025_rows: {len(train):,}", flush=True)
    print(f"test_2026_rows: {len(test):,}", flush=True)
    print(
        f"mapped_2026: {int(test['first_pitcher_id'].notna().sum()):,}",
        flush=True,
    )
    print(
        f"zero_hand_bbe_2025: {int((train['opp_hand_bbe_365d'].fillna(0) <= 0).sum()):,}",
        flush=True,
    )
    print(
        f"zero_hand_bbe_2026: {int((test['opp_hand_bbe_365d'].fillna(0) <= 0).sum()):,}",
        flush=True,
    )
    print(
        f"median_hand_bbe_2025: {float(train['opp_hand_bbe_365d'].median()):.1f}",
        flush=True,
    )
    print(
        f"median_hand_bbe_2026: {float(test['opp_hand_bbe_365d'].median()):.1f}",
        flush=True,
    )

    for threshold in [100, 250, 500]:
        decile_report(
            pd, test,
            "opp_hand_hr_per_bbe_365d_raw",
            "opp_hand_bbe_365d",
            threshold,
        )
    for threshold in [100, 250, 500]:
        decile_report(
            pd, test,
            "opp_hand_hr_per_bbe_365d_shrunk",
            "opp_hand_bbe_365d",
            threshold,
        )

    model_specs = [
        ("A0_ENVIRONMENT_C_LOCKED_A", LOCKED_FEATURES),
        ("H1_HAND_HR_BBE_ONLY", LOCKED_FEATURES + HAND_H1),
        ("H2_HAND_PACKAGE", LOCKED_FEATURES + HAND_H2),
        ("H3_HAND_PACKAGE_PLUS_SAMPLE", LOCKED_FEATURES + HAND_H3),
    ]

    y_train = train["actual_hr"].to_numpy(dtype=np.int8, copy=False)
    y_test = test["actual_hr"].to_numpy(dtype=np.int8, copy=False)
    traditional_mask = (
        test["first_pitcher_role"].fillna("").astype(str).to_numpy() == "traditional"
    )

    results = {
        "script": "HR_HAND_PITCHER_VECTOR_LITE_B",
        "db": args.db,
        "C": args.c,
        "shrink_k": args.shrink_k,
        "all_2026": [],
        "traditional_2026": [],
        "coefficients": {},
    }

    for label, features in model_specs:
        print(f"\nFITTING {label}", flush=True)
        print("-" * (8 + len(label)), flush=True)
        print(f"features_n: {len(features)}", flush=True)

        missing = [c for c in features if c not in train.columns]
        if missing:
            raise SystemExit(f"{label} missing columns: {missing}")

        model = make_model(
            features, args.c,
            ColumnTransformer, SimpleImputer, LogisticRegression,
            Pipeline, StandardScaler,
        )

        # Use only exact model matrices, not wide dataframes.
        X_train = train[features]
        X_test = test[features]

        model.fit(X_train, y_train)
        pred = model.predict_proba(X_test)[:, 1]

        all_metrics = metric_bundle(
            np, y_test, pred, brier, logloss, auc
        )
        all_metrics["label"] = label

        trad_metrics = metric_bundle(
            np,
            y_test[traditional_mask],
            pred[traditional_mask],
            brier,
            logloss,
            auc,
        )
        trad_metrics["label"] = label

        results["all_2026"].append(all_metrics)
        results["traditional_2026"].append(trad_metrics)

        # Save coefficients before deleting model.
        try:
            coefs = model.named_steps["clf"].coef_[0]
            results["coefficients"][label] = {
                name: float(coef) for name, coef in zip(features, coefs)
            }
        except Exception as e:
            results["coefficients"][label] = {"error": repr(e)}

        write_results(args.out_json, results)

        print(
            f"completed {label}: "
            f"brier={all_metrics['brier']:.8f} "
            f"logloss={all_metrics['logloss']:.8f} "
            f"auc={all_metrics['auc']:.8f} "
            f"top5_actual={all_metrics['top5_actual']:.5f}",
            flush=True,
        )
        print(f"saved partial results: {args.out_json}", flush=True)

        del X_train, X_test, pred, model
        gc.collect()

    print_summary("SUMMARY COMPARISON - ALL 2026", results["all_2026"])
    print_deltas(
        "DELTAS VS A0 - ALL 2026",
        results["all_2026"][0],
        results["all_2026"][1:],
    )

    print_summary(
        "SUMMARY COMPARISON - TRADITIONAL STARTER EVAL 2026",
        results["traditional_2026"],
    )
    print_deltas(
        "DELTAS VS A0 - TRADITIONAL STARTER EVAL 2026",
        results["traditional_2026"][0],
        results["traditional_2026"][1:],
    )

    print("\nHAND FEATURE COEFFICIENTS - H2_HAND_PACKAGE", flush=True)
    print("-------------------------------------------", flush=True)
    h2_coef = results["coefficients"].get("H2_HAND_PACKAGE", {})
    for f in HAND_H2:
        value = h2_coef.get(f)
        if isinstance(value, (int, float)):
            print(f"{f}: {value:+.6f}", flush=True)
        else:
            print(f"{f}: {value}", flush=True)

    print("\nREAD", flush=True)
    print("----", flush=True)
    print(
        "Gate is Brier-led, not Brier-only.",
        flush=True,
    )
    print(
        "Target unlock: roughly -0.0008 to -0.0010 Brier improvement "
        "with no unacceptable logloss/AUC/top-bucket damage.",
        flush=True,
    )
    print(
        "Traditional-starter evaluation uses the same fitted model and only "
        "subsets 2026 predictions; no post-game role information enters training.",
        flush=True,
    )
    print(
        "Bayesian pitch-zone overlap remains blocked until the cleaned "
        "handedness vector earns promotion.",
        flush=True,
    )
    print(f"final JSON: {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
