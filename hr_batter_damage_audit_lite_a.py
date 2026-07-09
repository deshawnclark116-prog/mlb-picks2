#!/usr/bin/env python3
"""
HR_BATTER_DAMAGE_AUDIT_LITE_A

Memory-lite batter-side damage calibration audit for a 512 MB Render instance.

Purpose
-------
This is the first batter-side calibration gate after freezing:
- pitcher branch
- Bayesian pitch-zone overlap
- expected-PA branch

Doctrine
--------
1. Raw directional audit first.
2. Use only strictly pregame / lagged batter features.
3. Select the best candidate on late-2025 internal validation.
4. Run ONE formal scalar gate on the 2026 forward holdout.
5. Brier-led, top-bucket-protected.
6. No stacking of correlated batter features yet.

Primary candidate family
------------------------
- barrel rate
- pull-air rate
- barrel x pull-air interaction
- max EV x pull-air interaction
- hard-hit-air rate, if already available
- recent-vs-longer power trends, if already available

Important
---------
Some raw barrel / EV features are already inside ENVIRONMENT_C_LOCKED_A.
They are still shown in raw deciles for directional sanity, but duplicate baseline
features are NOT treated as new incremental candidates.

Formal gate
-----------
Survival target:
- 2026 Brier delta <= -0.0005
- no top-5% degradation
- no logloss degradation
- no AUC degradation

If the selected scalar fails:
- do NOT stack correlated batter features
- do NOT launch deeper feature engineering yet
- review whether a direct base-rate refinement is the better next branch

Run
---
python -u hr_batter_damage_audit_lite_a.py 2>&1 | tee /data/hr_model/hr_batter_damage_audit_lite_a.log

Outputs
-------
/data/hr_model/hr_batter_damage_audit_lite_a_results.json

Paste back
----------
SCHEMA / CANDIDATE RESOLUTION
RAW DECILES - 2026
INTERNAL 2025 SELECTION
FORMAL 2026 SCALAR GATE
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

# Dynamic source candidates. The first existing column wins.
SOURCE_CANDIDATES = {
    "batter_pull_air_rate_short": [
        "batter_pull_air_rate_30d",
        "batter_pull_air_rate_14d",
        "batter_pull_air_rate_60d",
    ],
    "batter_pull_air_rate_long": [
        "batter_pull_air_rate_60d",
        "batter_pull_air_rate_90d",
    ],
    "batter_barrel_rate_short": [
        "batter_barrel_rate_14d",
        "batter_barrel_rate_30d",
    ],
    "batter_barrel_rate_long": [
        "batter_barrel_rate_60d",
        "batter_barrel_rate_90d",
        "batter_barrel_rate_30d",
    ],
    "batter_hr_per_air_short": [
        "batter_hr_per_air_14d",
        "batter_hr_per_air_30d",
    ],
    "batter_hr_per_air_long": [
        "batter_hr_per_air_60d",
        "batter_hr_per_air_90d",
    ],
    "batter_max_ev_short": [
        "batter_max_ev_30d",
        "batter_max_ev_14d",
        "batter_max_ev_60d",
    ],
    "batter_max_ev_long": [
        "batter_max_ev_60d",
        "batter_max_ev_90d",
    ],
    "batter_hard_hit_air_rate": [
        "batter_hard_hit_air_rate_30d",
        "batter_hard_hit_air_rate_60d",
        "batter_air_hard_hit_rate_30d",
        "batter_air_hard_hit_rate_60d",
    ],
    "batter_bbe_short": [
        "batter_bbe_30d",
        "batter_bbe_14d",
        "batter_bbe_60d",
    ],
    "batter_bbe_long": [
        "batter_bbe_60d",
        "batter_bbe_90d",
    ],
}

RAW_BASELINE_SOURCE_MAP = {
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
        "SELECT name FROM sqlite_master "
        "WHERE type IN ('table','view') AND name=?",
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
        for c in candidates:
            if c in cols:
                return table, c
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

    for alias, candidates in RAW_BASELINE_SOURCE_MAP.items():
        expr, src = source_expr(conn, alias, candidates)
        select.append(expr)
        provenance[alias] = src

    for alias, candidates in SOURCE_CANDIDATES.items():
        expr, src = source_expr(
            conn,
            alias,
            candidates,
            preferred=["batter_game_features", "batter_games"],
        )
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

    print("loading only required columns...", flush=True)
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


def add_safe_baseline_features(train, test, shrink_k):
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


def shrink_rate(train, test, raw_col, sample_col, out_col, k=75.0):
    prior = (
        float(train[raw_col].dropna().median())
        if train[raw_col].notna().sum()
        else 0.0
    )
    for df in (train, test):
        raw = df[raw_col].fillna(prior).clip(lower=0, upper=1)
        n = df[sample_col].fillna(0).clip(lower=0)
        weight = n / (n + k)
        df[out_col] = prior + (raw - prior) * weight


def add_damage_candidates(train, test):
    candidates = []
    raw_decile_features = []

    # Barrel raw directional audit.
    if "batter_barrel_rate_short" in train.columns and train["batter_barrel_rate_short"].notna().mean() >= 0.20:
        raw_decile_features.append("batter_barrel_rate_short")

    # Pull-air raw + shrunk candidate.
    if (
        "batter_pull_air_rate_short" in train.columns
        and train["batter_pull_air_rate_short"].notna().mean() >= 0.20
    ):
        raw_decile_features.append("batter_pull_air_rate_short")

        sample_col = "batter_bbe_short"
        if sample_col in train.columns and train[sample_col].notna().mean() >= 0.20:
            out = "damage_pull_air_short_shrunk"
            shrink_rate(
                train, test,
                "batter_pull_air_rate_short",
                sample_col,
                out,
                k=75.0,
            )
            candidates.append(out)
        else:
            candidates.append("batter_pull_air_rate_short")

    # Hard-hit air candidate if already available.
    if (
        "batter_hard_hit_air_rate" in train.columns
        and train["batter_hard_hit_air_rate"].notna().mean() >= 0.20
    ):
        raw_decile_features.append("batter_hard_hit_air_rate")
        candidates.append("batter_hard_hit_air_rate")

    # Barrel x pull-air interaction.
    if (
        "batter_barrel_rate_short" in train.columns
        and "batter_pull_air_rate_short" in train.columns
        and train["batter_barrel_rate_short"].notna().mean() >= 0.20
        and train["batter_pull_air_rate_short"].notna().mean() >= 0.20
    ):
        for df in (train, test):
            df["damage_barrel_x_pull_air"] = (
                df["batter_barrel_rate_short"].fillna(0)
                * df["batter_pull_air_rate_short"].fillna(0)
            )
        candidates.append("damage_barrel_x_pull_air")
        raw_decile_features.append("damage_barrel_x_pull_air")

    # Max EV x pull-air interaction.
    if (
        "batter_max_ev_short" in train.columns
        and "batter_pull_air_rate_short" in train.columns
        and train["batter_max_ev_short"].notna().mean() >= 0.20
        and train["batter_pull_air_rate_short"].notna().mean() >= 0.20
    ):
        train_ev_center = float(train["batter_max_ev_short"].median())
        for df in (train, test):
            ev_excess = (df["batter_max_ev_short"].fillna(train_ev_center) - 95.0).clip(lower=0)
            df["damage_ev_excess_x_pull_air"] = (
                ev_excess * df["batter_pull_air_rate_short"].fillna(0)
            )
        candidates.append("damage_ev_excess_x_pull_air")
        raw_decile_features.append("damage_ev_excess_x_pull_air")

    # Barrel trend.
    if (
        "batter_barrel_rate_short" in train.columns
        and "batter_barrel_rate_long" in train.columns
        and train["batter_barrel_rate_short"].notna().mean() >= 0.20
        and train["batter_barrel_rate_long"].notna().mean() >= 0.20
    ):
        for df in (train, test):
            df["damage_barrel_trend_short_minus_long"] = (
                df["batter_barrel_rate_short"]
                - df["batter_barrel_rate_long"]
            )
        # Only useful if it is not identically zero because both aliases resolved to same source.
        if train["damage_barrel_trend_short_minus_long"].abs().sum() > 0:
            candidates.append("damage_barrel_trend_short_minus_long")
            raw_decile_features.append("damage_barrel_trend_short_minus_long")

    # HR/air trend.
    if (
        "batter_hr_per_air_short" in train.columns
        and "batter_hr_per_air_long" in train.columns
        and train["batter_hr_per_air_short"].notna().mean() >= 0.20
        and train["batter_hr_per_air_long"].notna().mean() >= 0.20
    ):
        for df in (train, test):
            df["damage_hr_per_air_trend_short_minus_long"] = (
                df["batter_hr_per_air_short"]
                - df["batter_hr_per_air_long"]
            )
        if train["damage_hr_per_air_trend_short_minus_long"].abs().sum() > 0:
            candidates.append("damage_hr_per_air_trend_short_minus_long")
            raw_decile_features.append("damage_hr_per_air_trend_short_minus_long")

    # Deduplicate while preserving order.
    candidates = list(dict.fromkeys(candidates))
    raw_decile_features = list(dict.fromkeys(raw_decile_features))
    return candidates, raw_decile_features


def decile_report(pd, df, feature, label):
    d = df[df[feature].notna()][[feature, "actual_hr"]].copy()
    if len(d) < 500 or d[feature].nunique() < 5:
        print(f"{label} {feature}: insufficient usable variation n={len(d)}", flush=True)
        return []

    try:
        d["decile"] = pd.qcut(d[feature], 10, labels=False, duplicates="drop") + 1
    except Exception as e:
        print(f"{label} {feature}: qcut error {e}", flush=True)
        return []

    rows = []
    print(f"\n{label}: {feature}", flush=True)
    print("-" * (len(label) + len(feature) + 2), flush=True)
    for dec, g in d.groupby("decile"):
        row = {
            "decile": int(dec),
            "n": int(len(g)),
            "mean_feature": float(g[feature].mean()),
            "actual_hr": float(g["actual_hr"].mean()),
            "hits": int(g["actual_hr"].sum()),
        }
        rows.append(row)
        print(
            f"decile={int(dec):02d} n={len(g):5d} "
            f"mean={g[feature].mean():.6f} "
            f"actual_hr={g['actual_hr'].mean():.5f} "
            f"hits={int(g['actual_hr'].sum())}",
            flush=True,
        )
    return rows


def metric_bundle(np, y, p, brier, logloss, auc):
    y_arr = np.asarray(y, dtype=np.int8)
    p_arr = np.asarray(p, dtype=np.float64)
    out = {
        "rows": int(len(y_arr)),
        "actual_rate": float(y_arr.mean()),
        "brier": float(brier(y_arr, p_arr)),
        "logloss": float(logloss(y_arr, p_arr, labels=[0, 1])),
        "auc": float(auc(y_arr, p_arr)),
    }
    k = max(1, int(math.ceil(len(y_arr) * 0.05)))
    top_idx = np.argpartition(p_arr, -k)[-k:]
    out["top5_rows"] = int(k)
    out["top5_hits"] = int(y_arr[top_idx].sum())
    out["top5_actual"] = float(y_arr[top_idx].mean())
    out["top5_pred"] = float(p_arr[top_idx].mean())
    return out


def fit_eval(
    np,
    LogisticRegression,
    Xtr_all,
    ytr,
    Xev_all,
    yev,
    feature_index,
    features,
    C,
    brier,
    logloss,
    auc,
):
    idx = [feature_index[f] for f in features]
    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=C,
    )
    model.fit(Xtr_all[:, idx], ytr)
    pred = model.predict_proba(Xev_all[:, idx])[:, 1]
    m = metric_bundle(np, yev, pred, brier, logloss, auc)
    coef = float(model.coef_[0][-1]) if len(features) else float("nan")
    del model, pred
    gc.collect()
    return m, coef


def print_result(r):
    print(
        f"{r['label']}: "
        f"brier={r['brier']:.8f} "
        f"logloss={r['logloss']:.8f} "
        f"auc={r['auc']:.8f} "
        f"top5_actual={r['top5_actual']:.5f} "
        f"top5_hits={r['top5_hits']}/{r['top5_rows']} "
        f"rows={r['rows']}",
        flush=True,
    )


def delta_row(result, base):
    return {
        "label": result["label"],
        "brier_delta": result["brier"] - base["brier"],
        "logloss_delta": result["logloss"] - base["logloss"],
        "auc_delta": result["auc"] - base["auc"],
        "top5_actual_delta": result["top5_actual"] - base["top5_actual"],
    }


def print_delta(d):
    print(
        f"{d['label']}: "
        f"brier_delta={d['brier_delta']:+.8f} "
        f"logloss_delta={d['logloss_delta']:+.8f} "
        f"auc_delta={d['auc_delta']:+.8f} "
        f"top5_actual_delta={d['top5_actual_delta']:+.5f}",
        flush=True,
    )


def save_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--c", type=float, default=0.03)
    ap.add_argument("--shrink-k", type=float, default=75.0)
    ap.add_argument("--selection-split", default="2025-08-01")
    ap.add_argument(
        "--out-json",
        default="/data/hr_model/hr_batter_damage_audit_lite_a_results.json",
    )
    args = ap.parse_args()

    (
        np, pd, SimpleImputer, LogisticRegression,
        brier, logloss, auc, StandardScaler
    ) = require_imports()

    print("HR_BATTER_DAMAGE_AUDIT_LITE_A", flush=True)
    print("==============================", flush=True)
    print(f"db: {args.db}", flush=True)
    print(f"selection_split: {args.selection_split}", flush=True)

    conn = sqlite3.connect(args.db)
    df, provenance = load_dataset(conn, pd)
    conn.close()

    print("\nSCHEMA / CANDIDATE RESOLUTION", flush=True)
    print("-----------------------------", flush=True)
    for k, v in provenance.items():
        print(f"{k}: {v}", flush=True)

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["actual_hr"] = (
        pd.to_numeric(df["actual_hr"], errors="coerce")
        .fillna(0)
        .astype("int8")
    )

    for c in df.columns:
        if c in ("game_date", "game_id"):
            continue
        if c == "actual_hr":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    train = df[df["game_date"].dt.year == 2025].copy()
    test = df[df["game_date"].dt.year == 2026].copy()
    del df
    gc.collect()

    add_safe_baseline_features(train, test, args.shrink_k)
    add_weather_hinges(train)
    add_weather_hinges(test)

    candidates, raw_decile_features = add_damage_candidates(train, test)

    if not candidates:
        raise SystemExit(
            "No incremental batter-damage candidates were available. "
            "Need to build additional leak-safe batter BBE features first."
        )

    print("\nRESOLVED INCREMENTAL CANDIDATES", flush=True)
    print("--------------------------------", flush=True)
    for c in candidates:
        print(c, flush=True)

    print("\nRAW DECILES - 2026", flush=True)
    print("------------------", flush=True)

    decile_results = {}
    for feature in raw_decile_features:
        decile_results[feature] = decile_report(
            pd, test, feature, "RAW DECILES 2026"
        )

    all_features = list(dict.fromkeys(LOCKED_FEATURES + candidates))
    for frame in (train, test):
        for c in all_features:
            frame[c] = pd.to_numeric(frame[c], errors="coerce").astype("float32")

    split_ts = pd.Timestamp(args.selection_split)
    sel_train = train[train["game_date"] < split_ts]
    sel_val = train[train["game_date"] >= split_ts]

    if len(sel_train) < 5000 or len(sel_val) < 2000:
        raise SystemExit(
            f"Internal split too small: train={len(sel_train)} val={len(sel_val)}"
        )

    print("\nINTERNAL 2025 SELECTION", flush=True)
    print("-----------------------", flush=True)
    print(
        f"selection_train_rows={len(sel_train):,} "
        f"selection_val_rows={len(sel_val):,}",
        flush=True,
    )
    print("2026 is not used to choose the formal candidate.", flush=True)

    imputer_sel = SimpleImputer(strategy="median")
    scaler_sel = StandardScaler()

    Xsel_train = scaler_sel.fit_transform(
        imputer_sel.fit_transform(
            sel_train[all_features].to_numpy(dtype="float64")
        )
    )
    Xsel_val = scaler_sel.transform(
        imputer_sel.transform(
            sel_val[all_features].to_numpy(dtype="float64")
        )
    )
    ysel_train = sel_train["actual_hr"].to_numpy(dtype="int8", copy=False)
    ysel_val = sel_val["actual_hr"].to_numpy(dtype="int8", copy=False)

    feature_index = {
        name: i for i, name in enumerate(all_features)
    }

    internal_results = []
    for candidate in candidates:
        m, coef = fit_eval(
            np, LogisticRegression,
            Xsel_train, ysel_train,
            Xsel_val, ysel_val,
            feature_index,
            LOCKED_FEATURES + [candidate],
            args.c,
            brier, logloss, auc,
        )
        m["label"] = candidate
        m["coefficient"] = coef
        internal_results.append(m)
        print_result(m)
        print(f"coefficient={coef:+.6f}", flush=True)

    # Brier-led internal selection, then top bucket, logloss, AUC.
    selected = min(
        internal_results,
        key=lambda r: (
            r["brier"],
            -r["top5_actual"],
            r["logloss"],
            -r["auc"],
        ),
    )

    print("\nSELECTED FORMAL CANDIDATE", flush=True)
    print("-------------------------", flush=True)
    print(
        f"selected={selected['label']} "
        f"internal_brier={selected['brier']:.8f} "
        f"internal_top5={selected['top5_actual']:.5f}",
        flush=True,
    )

    del Xsel_train, Xsel_val, ysel_train, ysel_val
    del imputer_sel, scaler_sel, sel_train, sel_val
    gc.collect()

    # Formal 2025 -> 2026 gate.
    formal_features = list(dict.fromkeys(
        LOCKED_FEATURES + [selected["label"]]
    ))

    imputer_full = SimpleImputer(strategy="median")
    scaler_full = StandardScaler()

    Xtrain = scaler_full.fit_transform(
        imputer_full.fit_transform(
            train[all_features].to_numpy(dtype="float64")
        )
    )
    Xtest = scaler_full.transform(
        imputer_full.transform(
            test[all_features].to_numpy(dtype="float64")
        )
    )
    ytrain = train["actual_hr"].to_numpy(dtype="int8", copy=False)
    ytest = test["actual_hr"].to_numpy(dtype="int8", copy=False)

    a0, _ = fit_eval(
        np, LogisticRegression,
        Xtrain, ytrain,
        Xtest, ytest,
        feature_index,
        LOCKED_FEATURES,
        args.c,
        brier, logloss, auc,
    )
    a0["label"] = "A0_ENVIRONMENT_C_LOCKED_A"

    formal, coef = fit_eval(
        np, LogisticRegression,
        Xtrain, ytrain,
        Xtest, ytest,
        feature_index,
        formal_features,
        args.c,
        brier, logloss, auc,
    )
    formal["label"] = f"DAMAGE_SCALAR__{selected['label']}"
    formal["candidate"] = selected["label"]
    formal["coefficient"] = coef

    delta = delta_row(formal, a0)

    print("\nFORMAL 2026 SCALAR GATE", flush=True)
    print("-----------------------", flush=True)
    print_result(a0)
    print_result(formal)
    print(f"candidate_coefficient={coef:+.6f}", flush=True)

    print("\nFORMAL DELTA VS A0", flush=True)
    print("------------------", flush=True)
    print_delta(delta)

    survival_pass = (
        delta["brier_delta"] <= -0.0005
        and delta["top5_actual_delta"] >= 0.0
        and delta["logloss_delta"] <= 0.0
        and delta["auc_delta"] >= 0.0
    )

    if survival_pass:
        verdict = "BATTER_DAMAGE_SCALAR_SURVIVES_READY_FOR_NARROW_NEXT_GATE"
        next_step = (
            "Keep only this isolated batter-damage scalar. "
            "Do not stack correlated features yet. "
            "Design one narrow follow-up gate."
        )
    else:
        verdict = "BATTER_DAMAGE_SCALAR_FAILS_SURVIVAL_GATE"
        next_step = (
            "Do not stack correlated batter features. "
            "Pivot to direct base-rate refinement / alternative batter-side calibration."
        )

    results = {
        "script": "HR_BATTER_DAMAGE_AUDIT_LITE_A",
        "db": args.db,
        "selection_split": args.selection_split,
        "schema_resolution": provenance,
        "candidates": candidates,
        "raw_deciles_2026": decile_results,
        "internal_2025_selection": internal_results,
        "selected_formal_candidate": selected,
        "a0_2026": a0,
        "formal_candidate_2026": formal,
        "delta_vs_a0": delta,
        "survival_gate": {
            "brier_delta_max": -0.0005,
            "top5_actual_delta_min": 0.0,
            "logloss_delta_max": 0.0,
            "auc_delta_min": 0.0,
        },
        "survival_pass": survival_pass,
        "verdict": verdict,
        "next_step": next_step,
    }
    save_json(args.out_json, results)

    print("\nSTRICT GATE READ", flush=True)
    print("----------------", flush=True)
    print(f"selected_candidate: {selected['label']}", flush=True)
    print_delta(delta)
    print(f"survival_pass: {survival_pass}", flush=True)
    print(f"verdict: {verdict}", flush=True)
    print(f"next_step: {next_step}", flush=True)
    print(
        "Pitcher branch remains frozen. PA branch remains frozen. "
        "Bayesian overlap remains blocked.",
        flush=True,
    )
    print(f"final JSON: {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
