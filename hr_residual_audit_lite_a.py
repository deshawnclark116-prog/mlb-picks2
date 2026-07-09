#!/usr/bin/env python3
"""
HR_RESIDUAL_AUDIT_LITE_A

Memory-lite residual audit for the locked HR champion:
    ENVIRONMENT_C_LOCKED_A

Purpose
-------
Do NOT add another feature yet.

Map exactly where the locked model is wrong on the 2026 forward holdout:

1. Calibration by predicted-probability decile.
2. Monthly base-rate drift.
3. Batter archetype residuals.
4. Interaction residuals.
5. Calibration intercept/slope and top-tail compression diagnostics.

This script is diagnostic only.
It does NOT promote:
- recalibration
- base-rate adjustment
- archetype corrections
- interaction features

Those decisions come only after the residual pattern is visible.

Run
---
python -u hr_residual_audit_lite_a.py 2>&1 | tee /data/hr_model/hr_residual_audit_lite_a.log

Output
------
/data/hr_model/hr_residual_audit_lite_a_results.json
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

SOURCE_MAP = {
    "expected_pa_v1": ["expected_pa_v1"],
    "temp_f": ["temp_f", "weather_temp_f"],
    "wind_toward_pull_field": ["wind_toward_pull_field"],
    "weather_wind_mph": ["weather_wind_mph", "wind_mph"],

    "batter_hr_per_air_60d": ["batter_hr_per_air_60d"],
    "batter_fly_ball_rate_30d": ["batter_fly_ball_rate_30d"],
    "batter_barrel_rate_30d": ["batter_barrel_rate_30d"],
    "batter_max_ev_60d": ["batter_max_ev_60d"],
    "batter_bbe_60d": ["batter_bbe_60d"],

    "pitcher_pull_air_allowed_rate_60d": ["pitcher_pull_air_allowed_rate_60d"],
    "pitcher_hard_hit_allowed_rate_7d": ["pitcher_hard_hit_allowed_rate_7d"],
    "pitcher_hrfb_rate_30d": ["pitcher_hrfb_rate_30d"],
    "pitcher_bbe_allowed_60d": ["pitcher_bbe_allowed_60d"],

    "league_hr_per_bbe_10d_lag": ["league_hr_per_bbe_10d_lag"],
    "league_hr_per_air_10d_lag": ["league_hr_per_air_10d_lag"],
    "league_barrel_rate_10d_lag": ["league_barrel_rate_10d_lag"],
    "league_avg_ev_10d_lag": ["league_avg_ev_10d_lag"],
    "league_bbe_10d": ["league_bbe_10d"],

    "lineup_spot": ["lineup_spot", "batting_order", "batting_order_position"],
    "pitcher_hand_diag": ["pitcher_hand", "p_throws", "opp_pitcher_hand"],
    "team_diag": ["team_id", "batting_team_id", "team"],
}


def default_db_path():
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    return p / "hr_model.sqlite" if p.parent.exists() else Path("./hr_model/hr_model.sqlite")


def require_imports():
    try:
        import numpy as np
        import pandas as pd
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        from sklearn.preprocessing import StandardScaler
        return (
            np, pd, SimpleImputer, LogisticRegression,
            brier_score_loss, log_loss, roc_auc_score, StandardScaler
        )
    except Exception as e:
        raise SystemExit(
            "Missing dependency. Install pandas numpy scikit-learn.\n" + repr(e)
        )


def table_exists(conn, table):
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (table,),
    ).fetchone() is not None


def table_cols(conn, table):
    if not table_exists(conn, table):
        return []
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def alias_for_table(table):
    return {
        "batter_games": "bg",
        "batter_game_features": "bf",
        "pitcher_game_features": "pf",
        "league_env_lag_features": "le",
    }[table]


def find_source(conn, candidates, preferred=None):
    tables = preferred or [
        "batter_game_features",
        "batter_games",
        "pitcher_game_features",
        "league_env_lag_features",
    ]
    for table in tables:
        cols = set(table_cols(conn, table))
        for col in candidates:
            if col in cols:
                return table, col
    return None, None


def source_expr(conn, out_alias, candidates, preferred=None):
    table, col = find_source(conn, candidates, preferred)
    if not table:
        return f'NULL AS "{out_alias}"', None
    return (
        f'{alias_for_table(table)}."{col}" AS "{out_alias}"',
        f"{table}.{col}",
    )


def load_dataset(conn, pd):
    select = [
        "bg.game_date AS game_date",
        "CAST(bg.game_id AS TEXT) AS game_id",
        "bg.batter_id AS batter_id",
        "bg.actual_hr AS actual_hr",
    ]
    provenance = {}

    for alias, candidates in SOURCE_MAP.items():
        expr, src = source_expr(conn, alias, candidates)
        select.append(expr)
        provenance[alias] = src

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
    ORDER BY bg.game_date, bg.game_id, bg.batter_id
    """

    print("loading exact required columns...", flush=True)
    df = pd.read_sql_query(sql, conn)
    print(f"loaded rows={len(df):,} cols={len(df.columns)}", flush=True)
    return df, provenance


def boolish_to_float(v):
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        try:
            return 1.0 if float(v) > 0 else 0.0
        except Exception:
            return 0.0
    return 1.0 if str(v).strip().lower() in {
        "1", "true", "t", "yes", "y", "out", "toward", "pull", "wind_out"
    } else 0.0


def add_locked_features(train, test, shrink_k):
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

    for df in (train, test):
        df["temp_over_75"] = (df["temp_f"].fillna(70) - 75).clip(lower=0)
        df["temp_over_85"] = (df["temp_f"].fillna(70) - 85).clip(lower=0)
        wind_flag = df["wind_toward_pull_field"].map(boolish_to_float)
        wind_mph = df["weather_wind_mph"].fillna(0).clip(lower=0, upper=60)
        df["wind_out_component"] = wind_flag * wind_mph
        df["wind_out_and_hot"] = df["wind_out_component"] * df["temp_over_75"]


def model_metrics(np, y, p, brier, logloss, auc):
    y_arr = np.asarray(y, dtype=np.int8)
    p_arr = np.asarray(p, dtype=np.float64)

    k5 = max(1, int(math.ceil(len(y_arr) * 0.05)))
    idx5 = np.argpartition(p_arr, -k5)[-k5:]

    k1 = max(1, int(math.ceil(len(y_arr) * 0.01)))
    idx1 = np.argpartition(p_arr, -k1)[-k1:]

    return {
        "rows": int(len(y_arr)),
        "actual_rate": float(y_arr.mean()),
        "mean_pred": float(p_arr.mean()),
        "brier": float(brier(y_arr, p_arr)),
        "logloss": float(logloss(y_arr, p_arr, labels=[0, 1])),
        "auc": float(auc(y_arr, p_arr)),
        "top5_rows": int(k5),
        "top5_hits": int(y_arr[idx5].sum()),
        "top5_actual": float(y_arr[idx5].mean()),
        "top5_pred": float(p_arr[idx5].mean()),
        "top1_rows": int(k1),
        "top1_hits": int(y_arr[idx1].sum()),
        "top1_actual": float(y_arr[idx1].mean()),
        "top1_pred": float(p_arr[idx1].mean()),
    }


def quantile_bins_from_train(pd, train_series, q=3):
    clean = train_series.dropna()
    if clean.nunique() < q:
        return None
    try:
        _, edges = pd.qcut(clean, q=q, retbins=True, duplicates="drop")
        edges = list(edges)
        edges[0] = -float("inf")
        edges[-1] = float("inf")
        return edges
    except Exception:
        return None


def assign_bin(pd, series, edges, labels):
    if edges is None or len(edges) - 1 != len(labels):
        return None
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True)


def group_residual_rows(df, group_col, min_n=100):
    rows = []
    for value, g in df.groupby(group_col, dropna=False):
        if len(g) < min_n:
            continue
        actual = float(g["actual_hr"].mean())
        pred = float(g["pred"].mean())
        rows.append({
            "group": str(value),
            "n": int(len(g)),
            "actual": actual,
            "pred": pred,
            "residual": actual - pred,
            "hits": int(g["actual_hr"].sum()),
        })
    return rows


def two_way_residual_rows(df, col1, col2, min_n=100):
    rows = []
    for (v1, v2), g in df.groupby([col1, col2], dropna=False):
        if len(g) < min_n:
            continue
        actual = float(g["actual_hr"].mean())
        pred = float(g["pred"].mean())
        rows.append({
            col1: str(v1),
            col2: str(v2),
            "n": int(len(g)),
            "actual": actual,
            "pred": pred,
            "residual": actual - pred,
            "hits": int(g["actual_hr"].sum()),
        })
    return rows


def calibration_by_decile(pd, test):
    d = test.copy()
    d["pred_decile"] = pd.qcut(
        d["pred"], q=10, labels=False, duplicates="drop"
    ) + 1

    rows = []
    for dec, g in d.groupby("pred_decile"):
        actual = float(g["actual_hr"].mean())
        pred = float(g["pred"].mean())
        rows.append({
            "decile": int(dec),
            "n": int(len(g)),
            "mean_pred": pred,
            "actual_hr": actual,
            "residual": actual - pred,
            "hits": int(g["actual_hr"].sum()),
        })
    return rows


def expected_calibration_error(decile_rows):
    total = sum(r["n"] for r in decile_rows)
    if total <= 0:
        return float("nan")
    return sum(
        (r["n"] / total) * abs(r["actual_hr"] - r["mean_pred"])
        for r in decile_rows
    )


def calibration_intercept_slope(np, LogisticRegression, y, p):
    eps = 1e-6
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    y_arr = np.asarray(y, dtype=np.int8)

    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
    )
    model.fit(logit_p, y_arr)

    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0][0]),
    }


def save_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def print_rows(rows, key_order):
    for r in rows:
        parts = []
        for k in key_order:
            if k not in r:
                continue
            v = r[k]
            if isinstance(v, float):
                parts.append(f"{k}={v:+.5f}" if k == "residual" else f"{k}={v:.5f}")
            else:
                parts.append(f"{k}={v}")
        print(" ".join(parts), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--c", type=float, default=0.03)
    ap.add_argument("--shrink-k", type=float, default=75.0)
    ap.add_argument(
        "--out-json",
        default="/data/hr_model/hr_residual_audit_lite_a_results.json",
    )
    args = ap.parse_args()

    (
        np, pd, SimpleImputer, LogisticRegression,
        brier, logloss, auc, StandardScaler
    ) = require_imports()

    print("HR_RESIDUAL_AUDIT_LITE_A", flush=True)
    print("========================", flush=True)
    print(f"db: {args.db}", flush=True)

    conn = sqlite3.connect(args.db)
    df, provenance = load_dataset(conn, pd)
    conn.close()

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["actual_hr"] = (
        pd.to_numeric(df["actual_hr"], errors="coerce")
        .fillna(0)
        .astype("int8")
    )

    for c in df.columns:
        if c in ("game_date", "game_id", "pitcher_hand_diag", "team_diag"):
            continue
        if c == "actual_hr":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    train = df[df["game_date"].dt.year == 2025].copy()
    test = df[df["game_date"].dt.year == 2026].copy()
    del df
    gc.collect()

    add_locked_features(train, test, args.shrink_k)

    for frame in (train, test):
        for c in LOCKED_FEATURES:
            frame[c] = pd.to_numeric(frame[c], errors="coerce").astype("float32")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        imputer.fit_transform(train[LOCKED_FEATURES].to_numpy(dtype="float64"))
    )
    X_test = scaler.transform(
        imputer.transform(test[LOCKED_FEATURES].to_numpy(dtype="float64"))
    )

    y_train = train["actual_hr"].to_numpy(dtype="int8", copy=False)
    y_test = test["actual_hr"].to_numpy(dtype="int8", copy=False)

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=args.c,
    )
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]

    test["pred"] = pred
    test["residual"] = test["actual_hr"] - test["pred"]

    locked_metrics = model_metrics(
        np, y_test, pred, brier, logloss, auc
    )

    print("\nLOCKED MODEL CHECK", flush=True)
    print("------------------", flush=True)
    print(
        f"brier={locked_metrics['brier']:.8f} "
        f"logloss={locked_metrics['logloss']:.8f} "
        f"auc={locked_metrics['auc']:.8f} "
        f"actual_rate={locked_metrics['actual_rate']:.5f} "
        f"mean_pred={locked_metrics['mean_pred']:.5f}",
        flush=True,
    )
    print(
        f"top5_actual={locked_metrics['top5_actual']:.5f} "
        f"top5_pred={locked_metrics['top5_pred']:.5f} "
        f"top5_hits={locked_metrics['top5_hits']}/{locked_metrics['top5_rows']}",
        flush=True,
    )
    print(
        f"top1_actual={locked_metrics['top1_actual']:.5f} "
        f"top1_pred={locked_metrics['top1_pred']:.5f} "
        f"top1_hits={locked_metrics['top1_hits']}/{locked_metrics['top1_rows']}",
        flush=True,
    )

    decile_rows = calibration_by_decile(pd, test)
    ece = expected_calibration_error(decile_rows)

    print("\nCALIBRATION BY PREDICTED DECILE", flush=True)
    print("--------------------------------", flush=True)
    print_rows(
        decile_rows,
        ["decile", "n", "mean_pred", "actual_hr", "residual", "hits"],
    )
    print(f"ece_10bin={ece:.6f}", flush=True)

    test["month"] = test["game_date"].dt.strftime("%Y-%m")
    monthly_rows = group_residual_rows(test, "month", min_n=1)

    print("\nMONTHLY BASE-RATE DRIFT", flush=True)
    print("-----------------------", flush=True)
    print_rows(
        monthly_rows,
        ["group", "n", "pred", "actual", "residual", "hits"],
    )

    barrel_edges = quantile_bins_from_train(
        pd, train["batter_barrel_rate_30d_shrunk"], q=3
    )
    ev_edges = quantile_bins_from_train(
        pd, train["batter_max_ev_60d"], q=3
    )
    hr_air_edges = quantile_bins_from_train(
        pd, train["batter_hr_per_air_60d_shrunk"], q=3
    )

    labels = ["LOW", "MID", "HIGH"]

    test["barrel_tier"] = assign_bin(
        pd, test["batter_barrel_rate_30d_shrunk"], barrel_edges, labels
    )
    test["max_ev_tier"] = assign_bin(
        pd, test["batter_max_ev_60d"], ev_edges, labels
    )
    test["hr_air_tier"] = assign_bin(
        pd, test["batter_hr_per_air_60d_shrunk"], hr_air_edges, labels
    )

    barrel_rows = group_residual_rows(test, "barrel_tier", min_n=100)
    ev_rows = group_residual_rows(test, "max_ev_tier", min_n=100)
    hr_air_rows = group_residual_rows(test, "hr_air_tier", min_n=100)
    archetype_2way = two_way_residual_rows(
        test, "barrel_tier", "max_ev_tier", min_n=100
    )

    print("\nBATTER ARCHETYPE RESIDUALS", flush=True)
    print("--------------------------", flush=True)
    print("BARREL TIER", flush=True)
    print_rows(barrel_rows, ["group", "n", "pred", "actual", "residual", "hits"])
    print("MAX EV TIER", flush=True)
    print_rows(ev_rows, ["group", "n", "pred", "actual", "residual", "hits"])
    print("HR/AIR TIER", flush=True)
    print_rows(hr_air_rows, ["group", "n", "pred", "actual", "residual", "hits"])
    print("BARREL x MAX EV", flush=True)
    print_rows(
        archetype_2way,
        ["barrel_tier", "max_ev_tier", "n", "pred", "actual", "residual", "hits"],
    )

    env_edges = quantile_bins_from_train(
        pd, train["league_hr_per_bbe_10d_lag"], q=3
    )
    temp_edges = [-float("inf"), 70.0, 80.0, float("inf")]

    test["league_env_tier"] = assign_bin(
        pd, test["league_hr_per_bbe_10d_lag"], env_edges, labels
    )
    test["temp_regime"] = assign_bin(
        pd, test["temp_f"], temp_edges, ["COOL", "WARM", "HOT"]
    )

    barrel_env_rows = two_way_residual_rows(
        test, "barrel_tier", "league_env_tier", min_n=100
    )
    barrel_temp_rows = two_way_residual_rows(
        test, "barrel_tier", "temp_regime", min_n=100
    )

    pitcher_hand_rows = []
    if (
        "pitcher_hand_diag" in test.columns
        and test["pitcher_hand_diag"].notna().mean() >= 0.20
    ):
        pitcher_hand_rows = two_way_residual_rows(
            test, "barrel_tier", "pitcher_hand_diag", min_n=100
        )

    print("\nINTERACTION RESIDUALS", flush=True)
    print("---------------------", flush=True)
    print("BARREL TIER x LEAGUE ENVIRONMENT", flush=True)
    print_rows(
        barrel_env_rows,
        ["barrel_tier", "league_env_tier", "n", "pred", "actual", "residual", "hits"],
    )
    print("BARREL TIER x TEMPERATURE REGIME", flush=True)
    print_rows(
        barrel_temp_rows,
        ["barrel_tier", "temp_regime", "n", "pred", "actual", "residual", "hits"],
    )
    if pitcher_hand_rows:
        print("BARREL TIER x PITCHER HAND", flush=True)
        print_rows(
            pitcher_hand_rows,
            ["barrel_tier", "pitcher_hand_diag", "n", "pred", "actual", "residual", "hits"],
        )

    cal_diag = calibration_intercept_slope(
        np, LogisticRegression, y_test, pred
    )

    top_decile = decile_rows[-1]
    top_tail_gap = locked_metrics["top5_actual"] - locked_metrics["top5_pred"]
    top1_gap = locked_metrics["top1_actual"] - locked_metrics["top1_pred"]

    print("\nCALIBRATION DIAGNOSTICS", flush=True)
    print("-----------------------", flush=True)
    print(f"calibration_intercept={cal_diag['intercept']:+.6f}", flush=True)
    print(f"calibration_slope={cal_diag['slope']:.6f}", flush=True)
    print(
        f"top_decile_pred={top_decile['mean_pred']:.5f} "
        f"top_decile_actual={top_decile['actual_hr']:.5f} "
        f"gap={top_decile['residual']:+.5f}",
        flush=True,
    )
    print(
        f"top5_pred={locked_metrics['top5_pred']:.5f} "
        f"top5_actual={locked_metrics['top5_actual']:.5f} "
        f"gap={top_tail_gap:+.5f}",
        flush=True,
    )
    print(
        f"top1_pred={locked_metrics['top1_pred']:.5f} "
        f"top1_actual={locked_metrics['top1_actual']:.5f} "
        f"gap={top1_gap:+.5f}",
        flush=True,
    )

    monthly_abs = max(
        [abs(r["residual"]) for r in monthly_rows], default=0.0
    )
    archetype_abs = max(
        [abs(r["residual"]) for r in barrel_rows + ev_rows + hr_air_rows],
        default=0.0,
    )
    interaction_abs = max(
        [abs(r["residual"]) for r in barrel_env_rows + barrel_temp_rows + pitcher_hand_rows],
        default=0.0,
    )

    top_compressed = (
        top_decile["residual"] > 0.01 and cal_diag["slope"] > 1.05
    )
    broad_shift = abs(
        locked_metrics["actual_rate"] - locked_metrics["mean_pred"]
    ) > 0.01

    if top_compressed:
        primary_read = "TOP_TAIL_COMPRESSION_OR_UNDERDISPERSION"
        next_question = (
            "Test a leak-safe calibration/dispersion correction using 2025-only selection "
            "before any new structural feature."
        )
    elif monthly_abs >= max(archetype_abs, interaction_abs, 0.015):
        primary_read = "MONTHLY_BASE_RATE_DRIFT_DOMINATES"
        next_question = (
            "Audit a dynamic league/base-rate offset before adding more batter features."
        )
    elif archetype_abs >= max(monthly_abs, interaction_abs, 0.015):
        primary_read = "BATTER_ARCHETYPE_MISCALIBRATION_DOMINATES"
        next_question = (
            "Test an archetype-specific base-rate refinement, not another generic feature stack."
        )
    elif interaction_abs >= max(monthly_abs, archetype_abs, 0.015):
        primary_read = "INTERACTION_RESIDUAL_STRUCTURE_DOMINATES"
        next_question = (
            "Choose one targeted interaction suggested by residuals and run one narrow gate."
        )
    elif broad_shift:
        primary_read = "BROAD_BASE_RATE_SHIFT"
        next_question = (
            "Test a single leak-safe base-rate offset learned without touching the 2026 holdout."
        )
    else:
        primary_read = "NO_SINGLE_DOMINANT_RESIDUAL_FAILURE_MODE"
        next_question = (
            "Do not force another branch. Review the highest stable subgroup gap and define one narrow diagnostic."
        )

    print("\nDIAGNOSTIC READ", flush=True)
    print("---------------", flush=True)
    print(f"primary_read: {primary_read}", flush=True)
    print(f"next_question: {next_question}", flush=True)
    print(
        "This audit promotes nothing. It only identifies the next hypothesis.",
        flush=True,
    )
    print(
        "Pitcher branch remains frozen. PA branch remains frozen. "
        "Bayesian overlap remains blocked. Batter-damage scalar remains not promoted.",
        flush=True,
    )

    results = {
        "script": "HR_RESIDUAL_AUDIT_LITE_A",
        "db": args.db,
        "schema_resolution": provenance,
        "locked_metrics": locked_metrics,
        "calibration_by_decile": decile_rows,
        "ece_10bin": ece,
        "monthly_base_rate_drift": monthly_rows,
        "batter_archetypes": {
            "barrel_tier": barrel_rows,
            "max_ev_tier": ev_rows,
            "hr_air_tier": hr_air_rows,
            "barrel_x_max_ev": archetype_2way,
            "train_only_cutpoints": {
                "barrel": barrel_edges,
                "max_ev": ev_edges,
                "hr_air": hr_air_edges,
            },
        },
        "interaction_residuals": {
            "barrel_x_league_env": barrel_env_rows,
            "barrel_x_temp_regime": barrel_temp_rows,
            "barrel_x_pitcher_hand": pitcher_hand_rows,
        },
        "calibration_diagnostics": {
            **cal_diag,
            "top_decile_gap": top_decile["residual"],
            "top5_gap": top_tail_gap,
            "top1_gap": top1_gap,
        },
        "diagnostic_summary": {
            "monthly_max_abs_residual": monthly_abs,
            "archetype_max_abs_residual": archetype_abs,
            "interaction_max_abs_residual": interaction_abs,
            "broad_shift": broad_shift,
            "top_compressed": top_compressed,
            "primary_read": primary_read,
            "next_question": next_question,
        },
    }

    save_json(args.out_json, results)

    print(f"final JSON: {args.out_json}", flush=True)

    del X_train, X_test, pred, model
    del train, test
    gc.collect()


if __name__ == "__main__":
    main()
