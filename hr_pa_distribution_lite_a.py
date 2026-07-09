#!/usr/bin/env python3
"""
HR_PA_DISTRIBUTION_LITE_A

Memory-lite PA exposure gate for a 512 MB Render instance.

Purpose
-------
Gate PA-1:
    Can a pregame PA-distribution model beat a lineup-spot-only baseline?

Gate PA-2:
    Is the full PA distribution calibrated, especially the right tail P(PA >= 5)?

Data split
----------
Train: 2025
Forward holdout: 2026

No sportsbook-implied totals.
No postgame features.
No actual same-game PA inside predictors.

Target buckets
--------------
1, 2, 3, 4, 5, 6+ PA

Run
---
    python -u hr_pa_distribution_lite_a.py 2>&1 | tee /data/hr_model/hr_pa_distribution_lite_a.log

Outputs
-------
    /data/hr_model/hr_pa_distribution_lite_a_results.json

Paste back
----------
    SCHEMA RESOLUTION
    FEATURE SET
    PA-1 SUMMARY
    PA-2 FULL DISTRIBUTION CALIBRATION
    PA>=5 RELIABILITY
    STRICT GATE READ
"""

import argparse
import gc
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


ACTUAL_PA_CANDIDATES = [
    "actual_pa",
    "pa",
    "plate_appearances",
    "plate_appearance_count",
    "game_pa",
]

LINEUP_CANDIDATES = [
    "lineup_spot",
    "batting_order",
    "batting_order_position",
]

HOME_CANDIDATES = [
    "is_home",
    "home_flag",
    "home",
]

PARK_CANDIDATES = [
    "park_id",
    "venue_id",
    "ballpark_id",
]

TEAM_CANDIDATES = [
    "team_id",
    "batting_team_id",
    "team",
]

TEAM_OFFENSE_CANDIDATES = [
    "team_runs_per_game_10d_lag",
    "team_runs_per_game_20d_lag",
    "team_offense_score",
    "team_woba_10d_lag",
    "team_woba_20d_lag",
    "team_ops_10d_lag",
    "team_ops_20d_lag",
]

PROJECTED_RUNS_CANDIDATES = [
    "projected_team_runs",
    "team_projected_runs",
    "internal_projected_runs",
]

PITCHER_HAND_CANDIDATES = [
    "p_throws",
    "pitcher_hand",
    "opp_pitcher_hand",
]

PITCHER_QUALITY_CANDIDATES = [
    "opp_hand_k_per_pa_365d_shrunk",
    "opp_hand_gb_per_bbe_365d_shrunk",
    "opp_hand_hr_per_pa_365d_shrunk",
    "pitcher_k_per_pa_365d",
    "pitcher_quality_score",
]

PREGAME_LINEUP_QUALITY_CANDIDATES = [
    "lineup_quality_score",
    "pregame_lineup_quality",
    "confirmed_lineup_quality",
]

GAME_ENV_CANDIDATES = [
    "league_hr_per_bbe_10d_lag",
    "league_hr_per_air_10d_lag",
    "league_barrel_rate_10d_lag",
    "league_avg_ev_10d_lag",
    "temp_f",
]

FORBIDDEN_EXACT = {
    "actual_hr",
    "actual_pa",
    "pa",
    "plate_appearances",
    "plate_appearance_count",
    "game_pa",
    "hits",
    "home_runs",
    "rbi",
    "runs",
    "total_bases",
    "ab",
}

FORBIDDEN_SUBSTRINGS = [
    "actual_",
    "post_",
    "future_",
    "result",
    "outcome",
    "same_game",
    "after_",
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
        from sklearn.metrics import (
            accuracy_score,
            brier_score_loss,
            log_loss,
            mean_absolute_error,
            mean_squared_error,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        return (
            np, pd, ColumnTransformer, SimpleImputer, LogisticRegression,
            accuracy_score, brier_score_loss, log_loss, mean_absolute_error,
            mean_squared_error, Pipeline, OneHotEncoder, StandardScaler
        )
    except Exception as e:
        raise SystemExit(
            "Missing dependency. Install pandas numpy scikit-learn.\n" + repr(e)
        )


def table_exists(conn, table):
    return conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type IN ('table','view') AND name=?",
        (table,),
    ).fetchone() is not None


def table_cols(conn, table):
    if not table_exists(conn, table):
        return []
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def find_source(conn, candidates):
    for table in [
        "batter_games",
        "batter_game_features",
        "pitcher_game_features",
        "league_env_lag_features",
        "pitcher_hand_features_a",
    ]:
        cols = set(table_cols(conn, table))
        for c in candidates:
            if c in cols:
                return table, c
    return None, None


def alias_for_table(table):
    return {
        "batter_games": "bg",
        "batter_game_features": "bf",
        "pitcher_game_features": "pf",
        "league_env_lag_features": "le",
        "pitcher_hand_features_a": "ph",
    }[table]


def source_expr(conn, alias, candidates):
    table, col = find_source(conn, candidates)
    if not table:
        return f'NULL AS "{alias}"', None
    return f'{alias_for_table(table)}."{col}" AS "{alias}"', f"{table}.{col}"


def resolve_schema(conn):
    actual_pa_table, actual_pa_col = find_source(conn, ACTUAL_PA_CANDIDATES)
    lineup_table, lineup_col = find_source(conn, LINEUP_CANDIDATES)

    if not actual_pa_table:
        raise SystemExit(
            "Could not find actual PA target. Checked: "
            + ", ".join(ACTUAL_PA_CANDIDATES)
        )
    if not lineup_table:
        raise SystemExit(
            "Could not find lineup spot. Checked: "
            + ", ".join(LINEUP_CANDIDATES)
        )

    return {
        "actual_pa": (actual_pa_table, actual_pa_col),
        "lineup_spot": (lineup_table, lineup_col),
    }


def build_query(conn):
    schema = resolve_schema(conn)
    pa_table, pa_col = schema["actual_pa"]
    lineup_table, lineup_col = schema["lineup_spot"]

    select = [
        "bg.game_date AS game_date",
        "CAST(bg.game_id AS TEXT) AS game_id",
        "bg.batter_id AS batter_id",
        f'{alias_for_table(pa_table)}."{pa_col}" AS actual_pa',
        f'{alias_for_table(lineup_table)}."{lineup_col}" AS lineup_spot',
    ]

    provenance = {
        "actual_pa": f"{pa_table}.{pa_col}",
        "lineup_spot": f"{lineup_table}.{lineup_col}",
    }

    candidate_groups = {
        "is_home": HOME_CANDIDATES,
        "park_id": PARK_CANDIDATES,
        "team_id": TEAM_CANDIDATES,
        "projected_team_runs": PROJECTED_RUNS_CANDIDATES,
        "team_offense_lag": TEAM_OFFENSE_CANDIDATES,
        "pitcher_hand": PITCHER_HAND_CANDIDATES,
        "pitcher_quality_1": PITCHER_QUALITY_CANDIDATES,
        "pitcher_quality_2": PITCHER_QUALITY_CANDIDATES[1:],
        "lineup_quality": PREGAME_LINEUP_QUALITY_CANDIDATES,
        "league_hr_per_bbe_10d_lag": ["league_hr_per_bbe_10d_lag"],
        "league_hr_per_air_10d_lag": ["league_hr_per_air_10d_lag"],
        "league_barrel_rate_10d_lag": ["league_barrel_rate_10d_lag"],
        "league_avg_ev_10d_lag": ["league_avg_ev_10d_lag"],
        "temp_f": ["temp_f", "weather_temp_f"],
    }

    used_sources = set()
    for alias, candidates in candidate_groups.items():
        expr, src = source_expr(conn, alias, candidates)
        if src and src in used_sources:
            select.append(f'NULL AS "{alias}"')
            provenance[alias] = None
        else:
            select.append(expr)
            provenance[alias] = src
            if src:
                used_sources.add(src)

    joins = [
        """
        LEFT JOIN batter_game_features bf
          ON bg.game_id = bf.game_id
         AND bg.batter_id = bf.batter_id
        """,
        """
        LEFT JOIN pitcher_game_features pf
          ON bg.game_id = pf.game_id
         AND bg.batter_id = pf.batter_id
        """,
        """
        LEFT JOIN league_env_lag_features le
          ON bg.game_date = le.game_date
        """,
    ]

    if table_exists(conn, "pitcher_hand_features_a"):
        joins.append("""
        LEFT JOIN pitcher_hand_features_a ph
          ON CAST(bg.game_id AS TEXT)=ph.game_id
         AND bg.batter_id=ph.batter_id
        """)

    sql = f"""
    SELECT {", ".join(select)}
    FROM batter_games bg
    {" ".join(joins)}
    WHERE bg.game_date IS NOT NULL
    ORDER BY bg.game_date, bg.game_id, bg.batter_id
    """

    return sql, provenance


def bucket_pa(x):
    try:
        n = int(round(float(x)))
    except Exception:
        return None
    if n <= 0:
        return None
    return min(n, 6)


def bucket_label(n):
    return "6+" if int(n) >= 6 else str(int(n))


def empirical_baseline(train, test, pd):
    global_dist = (
        train["pa_bucket"]
        .value_counts(normalize=True)
        .reindex([1, 2, 3, 4, 5, 6], fill_value=0.0)
        .to_numpy(dtype="float64")
    )

    by_spot = {}
    for spot, g in train.groupby("lineup_spot"):
        if len(g) < 50:
            continue
        by_spot[int(spot)] = (
            g["pa_bucket"]
            .value_counts(normalize=True)
            .reindex([1, 2, 3, 4, 5, 6], fill_value=0.0)
            .to_numpy(dtype="float64")
        )

    probs = []
    for spot in test["lineup_spot"].astype(int):
        probs.append(by_spot.get(int(spot), global_dist))

    return probs


def expected_pa_from_probs(np, probs):
    values = np.array([1, 2, 3, 4, 5, 6], dtype="float64")
    return np.asarray(probs).dot(values)


def multiclass_metrics(
    np, y_true, probs, accuracy_score, log_loss, mean_absolute_error, mean_squared_error
):
    y = np.asarray(y_true, dtype="int64")
    p = np.asarray(probs, dtype="float64")
    pred_bucket = np.argmax(p, axis=1) + 1
    exp_pa = expected_pa_from_probs(np, p)

    return {
        "rows": int(len(y)),
        "multiclass_logloss": float(log_loss(y, p, labels=[1, 2, 3, 4, 5, 6])),
        "bucket_accuracy": float(accuracy_score(y, pred_bucket)),
        "expected_pa_mae": float(mean_absolute_error(y, exp_pa)),
        "expected_pa_rmse": float(math.sqrt(mean_squared_error(y, exp_pa))),
        "mean_actual_pa_bucket": float(y.mean()),
        "mean_predicted_expected_pa": float(exp_pa.mean()),
    }


def tail_metrics(np, y_true, probs, brier_score_loss, log_loss):
    y = np.asarray(y_true, dtype="int64")
    p = np.asarray(probs, dtype="float64")
    p_ge5 = p[:, 4] + p[:, 5]
    y_ge5 = (y >= 5).astype("int8")

    return {
        "p_ge5_brier": float(brier_score_loss(y_ge5, p_ge5)),
        "p_ge5_logloss": float(log_loss(y_ge5, p_ge5, labels=[0, 1])),
        "p_ge5_actual_rate": float(y_ge5.mean()),
        "p_ge5_mean_pred": float(p_ge5.mean()),
    }


def reliability_rows(np, y_true, probs, bins=10):
    y = np.asarray(y_true, dtype="int64")
    p = np.asarray(probs, dtype="float64")
    p_ge5 = p[:, 4] + p[:, 5]
    y_ge5 = (y >= 5).astype("int8")

    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            mask = (p_ge5 >= lo) & (p_ge5 <= hi)
        else:
            mask = (p_ge5 >= lo) & (p_ge5 < hi)

        if mask.sum() == 0:
            continue

        rows.append({
            "bin": i + 1,
            "lo": float(lo),
            "hi": float(hi),
            "n": int(mask.sum()),
            "mean_pred": float(p_ge5[mask].mean()),
            "actual_rate": float(y_ge5[mask].mean()),
            "gap": float(y_ge5[mask].mean() - p_ge5[mask].mean()),
        })
    return rows


def class_calibration_rows(np, y_true, probs):
    y = np.asarray(y_true, dtype="int64")
    p = np.asarray(probs, dtype="float64")
    rows = []
    for cls in [1, 2, 3, 4, 5, 6]:
        idx = cls - 1
        actual = float((y == cls).mean())
        pred = float(p[:, idx].mean())
        rows.append({
            "pa_bucket": bucket_label(cls),
            "actual_rate": actual,
            "mean_pred": pred,
            "gap": actual - pred,
        })
    return rows


def print_metric_block(title, rows):
    print(f"\n{title}", flush=True)
    print("-" * len(title), flush=True)
    for label, m in rows:
        print(
            f"{label}: "
            f"logloss={m['multiclass_logloss']:.8f} "
            f"bucket_acc={m['bucket_accuracy']:.5f} "
            f"expected_pa_mae={m['expected_pa_mae']:.5f} "
            f"expected_pa_rmse={m['expected_pa_rmse']:.5f} "
            f"mean_actual={m['mean_actual_pa_bucket']:.5f} "
            f"mean_pred={m['mean_predicted_expected_pa']:.5f}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument(
        "--out-json",
        default="/data/hr_model/hr_pa_distribution_lite_a_results.json",
    )
    args = ap.parse_args()

    (
        np, pd, ColumnTransformer, SimpleImputer, LogisticRegression,
        accuracy_score, brier_score_loss, log_loss, mean_absolute_error,
        mean_squared_error, Pipeline, OneHotEncoder, StandardScaler
    ) = require_imports()

    print("HR_PA_DISTRIBUTION_LITE_A", flush=True)
    print("=========================", flush=True)
    print(f"db: {args.db}", flush=True)

    conn = sqlite3.connect(args.db)
    sql, provenance = build_query(conn)

    print("\nSCHEMA RESOLUTION", flush=True)
    print("-----------------", flush=True)
    for k, v in provenance.items():
        print(f"{k}: {v}", flush=True)

    print("\nloading required rows...", flush=True)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(f"loaded rows={len(df):,} cols={len(df.columns)}", flush=True)

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["actual_pa"] = pd.to_numeric(df["actual_pa"], errors="coerce")
    df["lineup_spot"] = pd.to_numeric(df["lineup_spot"], errors="coerce")

    df["pa_bucket"] = df["actual_pa"].map(bucket_pa)
    df = df[
        df["game_date"].notna()
        & df["pa_bucket"].notna()
        & df["lineup_spot"].between(1, 9)
    ].copy()

    df["pa_bucket"] = df["pa_bucket"].astype("int8")
    df["lineup_spot"] = df["lineup_spot"].astype("int8")

    train = df[df["game_date"].dt.year == 2025].copy()
    test = df[df["game_date"].dt.year == 2026].copy()
    del df
    gc.collect()

    print("\nDATA SPLIT", flush=True)
    print("----------", flush=True)
    print(f"train_2025_rows: {len(train):,}", flush=True)
    print(f"test_2026_rows: {len(test):,}", flush=True)
    print(
        f"train_range: {train['game_date'].min()} to {train['game_date'].max()}",
        flush=True,
    )
    print(
        f"test_range: {test['game_date'].min()} to {test['game_date'].max()}",
        flush=True,
    )

    candidate_features = [
        "lineup_spot",
        "is_home",
        "park_id",
        "team_id",
        "projected_team_runs",
        "team_offense_lag",
        "pitcher_hand",
        "pitcher_quality_1",
        "pitcher_quality_2",
        "lineup_quality",
        "league_hr_per_bbe_10d_lag",
        "league_hr_per_air_10d_lag",
        "league_barrel_rate_10d_lag",
        "league_avg_ev_10d_lag",
        "temp_f",
    ]

    usable = []
    for c in candidate_features:
        if c not in train.columns:
            continue
        nonnull = train[c].notna().mean()
        nunique = train[c].nunique(dropna=True)
        if nonnull >= 0.20 and nunique >= 2:
            usable.append(c)

    categorical = []
    numeric = []
    for c in usable:
        if c in ["park_id", "team_id", "pitcher_hand"]:
            categorical.append(c)
        elif c == "lineup_spot":
            categorical.append(c)
        else:
            # Treat obvious strings as categorical; numeric-like values as numeric.
            converted = pd.to_numeric(train[c], errors="coerce")
            if converted.notna().mean() >= 0.80:
                train[c] = converted.astype("float32")
                test[c] = pd.to_numeric(test[c], errors="coerce").astype("float32")
                numeric.append(c)
            else:
                categorical.append(c)

    print("\nFEATURE SET", flush=True)
    print("-----------", flush=True)
    print("categorical:", categorical, flush=True)
    print("numeric:", numeric, flush=True)

    baseline_probs = empirical_baseline(train, test, pd)

    baseline_metrics = multiclass_metrics(
        np, test["pa_bucket"], baseline_probs,
        accuracy_score, log_loss, mean_absolute_error, mean_squared_error
    )
    baseline_tail = tail_metrics(
        np, test["pa_bucket"], baseline_probs,
        brier_score_loss, log_loss
    )

    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))

    if not transformers:
        raise SystemExit("No usable pregame features found.")

    pre = ColumnTransformer(transformers, remainder="drop")

    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            C=0.10,
        )),
    ])

    X_train = train[usable]
    y_train = train["pa_bucket"].astype("int8")
    X_test = test[usable]
    y_test = test["pa_bucket"].astype("int8")

    print("\nfitting PA distribution model...", flush=True)
    model.fit(X_train, y_train)
    model_probs = model.predict_proba(X_test)

    # Align class columns to [1,2,3,4,5,6].
    aligned = np.zeros((len(test), 6), dtype="float64")
    classes = list(model.named_steps["clf"].classes_)
    for j, cls in enumerate(classes):
        aligned[:, int(cls) - 1] = model_probs[:, j]
    model_probs = aligned

    model_metrics = multiclass_metrics(
        np, y_test, model_probs,
        accuracy_score, log_loss, mean_absolute_error, mean_squared_error
    )
    model_tail = tail_metrics(
        np, y_test, model_probs,
        brier_score_loss, log_loss
    )

    print_metric_block(
        "PA-1 SUMMARY",
        [
            ("LINEUP_SPOT_BASELINE", baseline_metrics),
            ("PA_DISTRIBUTION_MODEL", model_metrics),
        ],
    )

    print("\nPA-1 DELTAS VS LINEUP-SPOT BASELINE", flush=True)
    print("------------------------------------", flush=True)
    print(
        f"logloss_delta={model_metrics['multiclass_logloss'] - baseline_metrics['multiclass_logloss']:+.8f}",
        flush=True,
    )
    print(
        f"expected_pa_mae_delta={model_metrics['expected_pa_mae'] - baseline_metrics['expected_pa_mae']:+.8f}",
        flush=True,
    )
    print(
        f"expected_pa_rmse_delta={model_metrics['expected_pa_rmse'] - baseline_metrics['expected_pa_rmse']:+.8f}",
        flush=True,
    )

    class_cal = class_calibration_rows(np, y_test, model_probs)

    print("\nPA-2 FULL DISTRIBUTION CALIBRATION", flush=True)
    print("----------------------------------", flush=True)
    for r in class_cal:
        print(
            f"PA={r['pa_bucket']} "
            f"actual={r['actual_rate']:.5f} "
            f"pred={r['mean_pred']:.5f} "
            f"gap={r['gap']:+.5f}",
            flush=True,
        )

    reliability = reliability_rows(np, y_test, model_probs, bins=10)

    print("\nPA>=5 RELIABILITY", flush=True)
    print("-----------------", flush=True)
    print(
        f"baseline_tail_brier={baseline_tail['p_ge5_brier']:.8f} "
        f"model_tail_brier={model_tail['p_ge5_brier']:.8f} "
        f"delta={model_tail['p_ge5_brier'] - baseline_tail['p_ge5_brier']:+.8f}",
        flush=True,
    )
    print(
        f"baseline_tail_logloss={baseline_tail['p_ge5_logloss']:.8f} "
        f"model_tail_logloss={model_tail['p_ge5_logloss']:.8f} "
        f"delta={model_tail['p_ge5_logloss'] - baseline_tail['p_ge5_logloss']:+.8f}",
        flush=True,
    )
    print(
        f"actual_ge5={model_tail['p_ge5_actual_rate']:.5f} "
        f"mean_pred_ge5={model_tail['p_ge5_mean_pred']:.5f}",
        flush=True,
    )

    for r in reliability:
        print(
            f"bin={r['bin']:02d} n={r['n']:5d} "
            f"pred={r['mean_pred']:.5f} actual={r['actual_rate']:.5f} "
            f"gap={r['gap']:+.5f}",
            flush=True,
        )

    pa1_pass = (
        model_metrics["multiclass_logloss"] < baseline_metrics["multiclass_logloss"]
        and model_metrics["expected_pa_mae"] < baseline_metrics["expected_pa_mae"]
        and model_metrics["expected_pa_rmse"] < baseline_metrics["expected_pa_rmse"]
    )

    pa2_pass = (
        model_tail["p_ge5_brier"] < baseline_tail["p_ge5_brier"]
        and model_tail["p_ge5_logloss"] < baseline_tail["p_ge5_logloss"]
        and abs(model_tail["p_ge5_mean_pred"] - model_tail["p_ge5_actual_rate"]) <= 0.03
    )

    if pa1_pass and pa2_pass:
        verdict = "PA1_PA2_PASS_READY_FOR_PA3_HR_INTEGRATION"
    elif pa1_pass:
        verdict = "PA1_PASS_PA2_FAIL_FIX_DISTRIBUTION_CALIBRATION"
    else:
        verdict = "PA1_FAIL_DO_NOT_INTEGRATE_INTO_HR_MODEL"

    results = {
        "script": "HR_PA_DISTRIBUTION_LITE_A",
        "db": args.db,
        "schema_resolution": provenance,
        "feature_set": {
            "categorical": categorical,
            "numeric": numeric,
        },
        "baseline_metrics": baseline_metrics,
        "model_metrics": model_metrics,
        "baseline_tail": baseline_tail,
        "model_tail": model_tail,
        "class_calibration": class_cal,
        "p_ge5_reliability": reliability,
        "pa1_pass": pa1_pass,
        "pa2_pass": pa2_pass,
        "verdict": verdict,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nSTRICT GATE READ", flush=True)
    print("----------------", flush=True)
    print(f"PA1_pass: {pa1_pass}", flush=True)
    print(f"PA2_pass: {pa2_pass}", flush=True)
    print(f"verdict: {verdict}", flush=True)
    print(
        "PA-3 remains blocked until PA-1 and PA-2 pass.",
        flush=True,
    )
    print(f"final JSON: {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
