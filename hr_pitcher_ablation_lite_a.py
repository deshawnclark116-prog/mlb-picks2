#!/usr/bin/env python3
"""
HR_PITCHER_ABLATION_LITE_A

Memory-lite, two-track handedness pitcher ablation for a 512 MB Render instance.

Doctrine:
- No feature promotion by intuition.
- Strictly pre-game, leak-safe features only.
- 2025 internal temporal validation selects the best pair/triple.
- 2026 remains the forward holdout for the gate decision.
- Bayesian pitch-zone overlap stays blocked unless the cleaned pitcher vector earns it.

Track A: new handedness features added to locked ENVIRONMENT_C_LOCKED_A.
    P1 = + handedness HR/BBE
    P2 = + handedness HR/PA
    P3 = + handedness K/PA
    P4 = + handedness GB/BBE
    P5 = + handedness Barrel/BBE
    P6 = + best two-feature combination selected on late-2025 validation
    P7 = + best three-feature combination selected on late-2025 validation

Track B: remove old pitcher-contact family from A0.
    Q0 = bare batter/environment baseline
    Q1 = Q0 + best single
    Q2 = Q0 + selected best pair
    Q3 = Q0 + selected best triple

Important:
- Reuses already-built pitcher_hand_features_a.
- Does NOT rebuild pitch data or handedness feature tables.
- Preprocesses all numeric features once, then fits lightweight logistic models.
- Saves partial JSON after every completed model.
- Flushes output immediately.

Run:
    python -u hr_pitcher_ablation_lite_a.py 2>&1 | tee /data/hr_model/hr_pitcher_ablation_lite_a.log

Results JSON:
    /data/hr_model/hr_pitcher_ablation_lite_a_results.json
"""

import argparse
import gc
import itertools
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

OLD_PITCHER_CONTACT_FEATURES = [
    "log_pitcher_bbe_allowed_60d",
    "pitcher_pull_air_allowed_rate_60d_shrunk",
    "pitcher_hard_hit_allowed_rate_7d_shrunk",
    "pitcher_hrfb_rate_30d_shrunk",
]

BARE_FEATURES = [
    c for c in LOCKED_FEATURES
    if c not in OLD_PITCHER_CONTACT_FEATURES
]

HAND_FEATURES = {
    "P1_HAND_HR_BBE": "opp_hand_hr_per_bbe_365d_shrunk",
    "P2_HAND_HR_PA": "opp_hand_hr_per_pa_365d_shrunk",
    "P3_HAND_K_PA": "opp_hand_k_per_pa_365d_shrunk",
    "P4_HAND_GB_BBE": "opp_hand_gb_per_bbe_365d_shrunk",
    "P5_HAND_BARREL_BBE": "opp_hand_barrel_per_bbe_365d_shrunk",
}

HAND_FEATURE_ORDER = list(HAND_FEATURES.values())

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


def table_cols(conn, table):
    try:
        return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
    except Exception:
        return []


def table_exists(conn, table):
    return conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type IN ('table','view') AND name=?",
        (table,),
    ).fetchone() is not None


def source_expr(conn, alias, candidates):
    sources = {
        "bg": set(table_cols(conn, "batter_games")),
        "bf": set(table_cols(conn, "batter_game_features")),
        "pf": set(table_cols(conn, "pitcher_game_features")),
        "le": set(table_cols(conn, "league_env_lag_features")),
    }
    for candidate in candidates:
        for table_alias in ("bg", "bf", "pf", "le"):
            if candidate in sources[table_alias]:
                return f'{table_alias}."{candidate}" AS "{alias}"'
    return f'NULL AS "{alias}"'


def load_dataset(conn, pd):
    if not table_exists(conn, "pitcher_hand_features_a"):
        raise SystemExit(
            "Missing pitcher_hand_features_a. "
            "The expensive handedness build must exist before running this script."
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

    hand_cols = set(table_cols(conn, "pitcher_hand_features_a"))

    select = [
        "bg.game_date AS game_date",
        "CAST(bg.game_id AS TEXT) AS game_id",
        "bg.batter_id AS batter_id",
        "bg.actual_hr AS actual_hr",
    ]

    for alias, candidates in candidate_map.items():
        select.append(source_expr(conn, alias, candidates))

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

    print("loading exact required columns...", flush=True)
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
    return 1.0 if s in {
        "1", "true", "t", "yes", "y", "out", "toward",
        "pull", "wind_out", "wind toward pull"
    } else 0.0


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


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fit_eval(
    np,
    LogisticRegression,
    X_train_all,
    y_train,
    X_eval_all,
    y_eval,
    feature_index,
    feature_names,
    C,
    brier,
    logloss,
    auc,
):
    idx = [feature_index[f] for f in feature_names]
    Xtr = X_train_all[:, idx]
    Xev = X_eval_all[:, idx]

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=C,
        penalty="l2",
    )
    model.fit(Xtr, y_train)
    pred = model.predict_proba(Xev)[:, 1]

    metrics = metric_bundle(np, y_eval, pred, brier, logloss, auc)
    coefs = {
        name: float(value)
        for name, value in zip(feature_names, model.coef_[0])
    }

    del Xtr, Xev, pred, model
    gc.collect()
    return metrics, coefs


def delta_row(result, base):
    return {
        "label": result["label"],
        "brier_delta": result["brier"] - base["brier"],
        "logloss_delta": result["logloss"] - base["logloss"],
        "auc_delta": result["auc"] - base["auc"],
        "top5_actual_delta": result["top5_actual"] - base["top5_actual"],
    }


def print_result(r):
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


def print_delta(d):
    print(
        f"{d['label']}: "
        f"brier_delta={d['brier_delta']:+.8f} "
        f"logloss_delta={d['logloss_delta']:+.8f} "
        f"auc_delta={d['auc_delta']:+.8f} "
        f"top5_actual_delta={d['top5_actual_delta']:+.5f}",
        flush=True,
    )


def select_best(results):
    # Brier-led. Ties are broken by better top bucket, logloss, then AUC.
    return min(
        results,
        key=lambda r: (
            r["brier"],
            -r["top5_actual"],
            r["logloss"],
            -r["auc"],
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--c", type=float, default=0.03)
    ap.add_argument("--shrink-k", type=float, default=75.0)
    ap.add_argument("--selection-split", default="2025-08-01")
    ap.add_argument(
        "--out-json",
        default="/data/hr_model/hr_pitcher_ablation_lite_a_results.json",
    )
    args = ap.parse_args()

    (
        np, pd, SimpleImputer, LogisticRegression,
        brier, logloss, auc, StandardScaler
    ) = require_imports()

    print("HR_PITCHER_ABLATION_LITE_A", flush=True)
    print("==========================", flush=True)
    print(f"db: {args.db}", flush=True)
    print("mode: reuse existing handedness tables; no rebuild", flush=True)
    print(f"selection_split: {args.selection_split}", flush=True)

    conn = sqlite3.connect(args.db)
    df = load_dataset(conn, pd)
    conn.close()

    print("parsing and compacting data...", flush=True)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["actual_hr"] = (
        pd.to_numeric(df["actual_hr"], errors="coerce")
        .fillna(0)
        .astype("int8")
    )

    numeric_cols = [
        c for c in RAW_COLS + HAND_LOAD_COLS
        if c in df.columns
        and c not in ("batter_stand", "p_throws", "first_pitcher_role")
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    train_2025 = df[df["game_date"].dt.year == 2025].copy()
    test_2026 = df[df["game_date"].dt.year == 2026].copy()
    del df
    gc.collect()

    add_safe_features(train_2025, test_2026, args.shrink_k)
    add_weather_hinges(train_2025)
    add_weather_hinges(test_2026)

    all_model_features = list(dict.fromkeys(
        LOCKED_FEATURES + HAND_FEATURE_ORDER
    ))

    for frame in (train_2025, test_2026):
        for c in all_model_features:
            frame[c] = pd.to_numeric(frame[c], errors="coerce").astype("float32")

    print("\nFEATURE COVERAGE", flush=True)
    print("----------------", flush=True)
    print(f"train_2025_rows: {len(train_2025):,}", flush=True)
    print(f"test_2026_rows: {len(test_2026):,}", flush=True)
    print(
        f"zero_hand_bbe_2025: "
        f"{int((train_2025['opp_hand_bbe_365d'].fillna(0) <= 0).sum()):,}",
        flush=True,
    )
    print(
        f"zero_hand_bbe_2026: "
        f"{int((test_2026['opp_hand_bbe_365d'].fillna(0) <= 0).sum()):,}",
        flush=True,
    )

    split_ts = pd.Timestamp(args.selection_split)
    select_train = train_2025[train_2025["game_date"] < split_ts].copy()
    select_val = train_2025[train_2025["game_date"] >= split_ts].copy()

    if len(select_train) < 5000 or len(select_val) < 2000:
        raise SystemExit(
            f"Internal selection split too small: "
            f"train={len(select_train)} val={len(select_val)}"
        )

    print("\nINTERNAL TEMPORAL SELECTION", flush=True)
    print("---------------------------", flush=True)
    print(
        f"selection_train: {select_train['game_date'].min().date()} "
        f"to {select_train['game_date'].max().date()} "
        f"rows={len(select_train):,}",
        flush=True,
    )
    print(
        f"selection_val: {select_val['game_date'].min().date()} "
        f"to {select_val['game_date'].max().date()} "
        f"rows={len(select_val):,}",
        flush=True,
    )
    print(
        "2026 holdout is not used to choose the best pair/triple.",
        flush=True,
    )

    # Preprocess internal-selection matrices once.
    imputer_sel = SimpleImputer(strategy="median")
    scaler_sel = StandardScaler()

    Xsel_train_raw = select_train[all_model_features].to_numpy(dtype=np.float64)
    Xsel_val_raw = select_val[all_model_features].to_numpy(dtype=np.float64)

    Xsel_train = scaler_sel.fit_transform(
        imputer_sel.fit_transform(Xsel_train_raw)
    )
    Xsel_val = scaler_sel.transform(
        imputer_sel.transform(Xsel_val_raw)
    )

    ysel_train = select_train["actual_hr"].to_numpy(dtype=np.int8, copy=False)
    ysel_val = select_val["actual_hr"].to_numpy(dtype=np.int8, copy=False)

    feature_index = {
        name: i for i, name in enumerate(all_model_features)
    }

    del Xsel_train_raw, Xsel_val_raw
    del select_train, select_val
    gc.collect()

    results = {
        "script": "HR_PITCHER_ABLATION_LITE_A",
        "db": args.db,
        "C": args.c,
        "shrink_k": args.shrink_k,
        "selection_split": args.selection_split,
        "internal_selection": {
            "baseline": None,
            "singles": [],
            "pairs": [],
            "triples": [],
            "selected_best_single": None,
            "selected_best_pair": None,
            "selected_best_triple": None,
        },
        "track_a_2026": [],
        "track_b_2026": [],
        "coefficients": {},
        "verdict": {},
    }

    print("\nINTERNAL BASELINE", flush=True)
    print("-----------------", flush=True)
    m, _ = fit_eval(
        np, LogisticRegression,
        Xsel_train, ysel_train,
        Xsel_val, ysel_val,
        feature_index, LOCKED_FEATURES, args.c,
        brier, logloss, auc,
    )
    m["label"] = "A0_INTERNAL"
    results["internal_selection"]["baseline"] = m
    print_result(m)
    write_json(args.out_json, results)

    print("\nP1-P5 INTERNAL SINGLE-FEATURE ABLATION", flush=True)
    print("--------------------------------------", flush=True)
    single_results = []

    for label, feature in HAND_FEATURES.items():
        features = LOCKED_FEATURES + [feature]
        m, _ = fit_eval(
            np, LogisticRegression,
            Xsel_train, ysel_train,
            Xsel_val, ysel_val,
            feature_index, features, args.c,
            brier, logloss, auc,
        )
        m["label"] = label
        m["hand_features"] = [feature]
        single_results.append(m)
        results["internal_selection"]["singles"].append(m)
        print_result(m)
        write_json(args.out_json, results)

    best_single_internal = select_best(single_results)
    results["internal_selection"]["selected_best_single"] = best_single_internal

    print("\nINTERNAL PAIR SEARCH - ALL 10 COMBINATIONS", flush=True)
    print("------------------------------------------", flush=True)
    pair_results = []

    for combo in itertools.combinations(HAND_FEATURE_ORDER, 2):
        label = "PAIR__" + "__".join(
            c.replace("opp_hand_", "").replace("_365d_shrunk", "")
            for c in combo
        )
        features = LOCKED_FEATURES + list(combo)
        m, _ = fit_eval(
            np, LogisticRegression,
            Xsel_train, ysel_train,
            Xsel_val, ysel_val,
            feature_index, features, args.c,
            brier, logloss, auc,
        )
        m["label"] = label
        m["hand_features"] = list(combo)
        pair_results.append(m)
        results["internal_selection"]["pairs"].append(m)
        print_result(m)
        write_json(args.out_json, results)

    best_pair_internal = select_best(pair_results)
    results["internal_selection"]["selected_best_pair"] = best_pair_internal

    print("\nINTERNAL TRIPLE SEARCH - ALL 10 COMBINATIONS", flush=True)
    print("--------------------------------------------", flush=True)
    triple_results = []

    for combo in itertools.combinations(HAND_FEATURE_ORDER, 3):
        label = "TRIPLE__" + "__".join(
            c.replace("opp_hand_", "").replace("_365d_shrunk", "")
            for c in combo
        )
        features = LOCKED_FEATURES + list(combo)
        m, _ = fit_eval(
            np, LogisticRegression,
            Xsel_train, ysel_train,
            Xsel_val, ysel_val,
            feature_index, features, args.c,
            brier, logloss, auc,
        )
        m["label"] = label
        m["hand_features"] = list(combo)
        triple_results.append(m)
        results["internal_selection"]["triples"].append(m)
        print_result(m)
        write_json(args.out_json, results)

    best_triple_internal = select_best(triple_results)
    results["internal_selection"]["selected_best_triple"] = best_triple_internal

    print("\nSELECTED BY LATE-2025 INTERNAL VALIDATION", flush=True)
    print("-----------------------------------------", flush=True)
    print(
        "best_single:",
        best_single_internal["hand_features"],
        flush=True,
    )
    print(
        "best_pair:",
        best_pair_internal["hand_features"],
        flush=True,
    )
    print(
        "best_triple:",
        best_triple_internal["hand_features"],
        flush=True,
    )
    write_json(args.out_json, results)

    del Xsel_train, Xsel_val, ysel_train, ysel_val
    del imputer_sel, scaler_sel
    gc.collect()

    # Preprocess full 2025 train / 2026 holdout once.
    print("\nPREPARING FULL 2025 -> 2026 HOLDOUT MATRICES", flush=True)
    print("--------------------------------------------", flush=True)

    imputer_full = SimpleImputer(strategy="median")
    scaler_full = StandardScaler()

    Xtrain_raw = train_2025[all_model_features].to_numpy(dtype=np.float64)
    Xtest_raw = test_2026[all_model_features].to_numpy(dtype=np.float64)

    Xtrain = scaler_full.fit_transform(
        imputer_full.fit_transform(Xtrain_raw)
    )
    Xtest = scaler_full.transform(
        imputer_full.transform(Xtest_raw)
    )

    ytrain = train_2025["actual_hr"].to_numpy(dtype=np.int8, copy=False)
    ytest = test_2026["actual_hr"].to_numpy(dtype=np.int8, copy=False)

    del Xtrain_raw, Xtest_raw
    gc.collect()

    # Track A: locked A0 plus singles + selected pair/triple.
    print("\nTRACK A - 2026 FORWARD HOLDOUT", flush=True)
    print("------------------------------", flush=True)

    track_a_specs = [
        ("A0_ENVIRONMENT_C_LOCKED_A", LOCKED_FEATURES, []),
    ]

    for label, feature in HAND_FEATURES.items():
        track_a_specs.append(
            (label, LOCKED_FEATURES + [feature], [feature])
        )

    selected_pair = list(best_pair_internal["hand_features"])
    selected_triple = list(best_triple_internal["hand_features"])

    track_a_specs.extend([
        (
            "P6_SELECTED_BEST_PAIR",
            LOCKED_FEATURES + selected_pair,
            selected_pair,
        ),
        (
            "P7_SELECTED_BEST_TRIPLE",
            LOCKED_FEATURES + selected_triple,
            selected_triple,
        ),
    ])

    track_a = []

    for label, features, hand_feats in track_a_specs:
        print(f"\nFITTING {label}", flush=True)
        m, coefs = fit_eval(
            np, LogisticRegression,
            Xtrain, ytrain,
            Xtest, ytest,
            feature_index, features, args.c,
            brier, logloss, auc,
        )
        m["label"] = label
        m["hand_features"] = hand_feats
        track_a.append(m)
        results["track_a_2026"].append(m)
        results["coefficients"][label] = coefs
        print_result(m)
        write_json(args.out_json, results)

    a0 = track_a[0]

    print("\nTRACK A SUMMARY - 2026 HOLDOUT", flush=True)
    print("--------------------------------", flush=True)
    for r in track_a:
        print_result(r)

    print("\nTRACK A DELTAS VS A0", flush=True)
    print("---------------------", flush=True)
    track_a_deltas = []
    for r in track_a[1:]:
        d = delta_row(r, a0)
        track_a_deltas.append(d)
        print_delta(d)

    # Track B: remove old pitcher-contact family.
    best_single_feature = list(best_single_internal["hand_features"])
    print("\nTRACK B - OLD PITCHER-CONTACT FAMILY REMOVED", flush=True)
    print("--------------------------------------------", flush=True)
    print("removed:", ", ".join(OLD_PITCHER_CONTACT_FEATURES), flush=True)

    track_b_specs = [
        ("Q0_BARE_ENV_BATTER", BARE_FEATURES, []),
        (
            "Q1_BARE_PLUS_BEST_SINGLE",
            BARE_FEATURES + best_single_feature,
            best_single_feature,
        ),
        (
            "Q2_BARE_PLUS_SELECTED_PAIR",
            BARE_FEATURES + selected_pair,
            selected_pair,
        ),
        (
            "Q3_BARE_PLUS_SELECTED_TRIPLE",
            BARE_FEATURES + selected_triple,
            selected_triple,
        ),
    ]

    track_b = []

    for label, features, hand_feats in track_b_specs:
        print(f"\nFITTING {label}", flush=True)
        m, coefs = fit_eval(
            np, LogisticRegression,
            Xtrain, ytrain,
            Xtest, ytest,
            feature_index, features, args.c,
            brier, logloss, auc,
        )
        m["label"] = label
        m["hand_features"] = hand_feats
        track_b.append(m)
        results["track_b_2026"].append(m)
        results["coefficients"][label] = coefs
        print_result(m)
        write_json(args.out_json, results)

    q0 = track_b[0]

    print("\nTRACK B SUMMARY - 2026 HOLDOUT", flush=True)
    print("--------------------------------", flush=True)
    for r in track_b:
        print_result(r)

    print("\nTRACK B DELTAS VS Q0", flush=True)
    print("---------------------", flush=True)
    track_b_deltas = []
    for r in track_b[1:]:
        d = delta_row(r, q0)
        track_b_deltas.append(d)
        print_delta(d)

    # Gate decisions.
    single_holdout = [
        r for r in track_a
        if r["label"] in HAND_FEATURES
    ]
    best_single_holdout = select_best(single_holdout)
    best_single_delta = delta_row(best_single_holdout, a0)

    survival_pass = (
        best_single_delta["brier_delta"] <= -0.0005
        and best_single_delta["top5_actual_delta"] >= 0.0
    )

    overlap_candidates = [
        r for r in track_a
        if r["label"] in (
            "P6_SELECTED_BEST_PAIR",
            "P7_SELECTED_BEST_TRIPLE",
        )
    ]

    overlap_passers = []
    for r in overlap_candidates:
        d = delta_row(r, a0)
        strict_pass = (
            d["brier_delta"] <= -0.0008
            and d["logloss_delta"] <= 0.0
            and d["auc_delta"] >= 0.0
            and d["top5_actual_delta"] >= 0.0
        )
        if strict_pass:
            overlap_passers.append((r, d))

    overlap_unlock = len(overlap_passers) > 0

    if overlap_unlock:
        verdict = (
            "OVERLAP_UNLOCKED: at least one selected cleaned pitcher combination "
            "cleared the strict holdout gate."
        )
    elif survival_pass:
        verdict = (
            "PITCHER_BRANCH_SURVIVES_BUT_OVERLAP_BLOCKED: best isolated handedness "
            "feature cleared the ~0.0005 survival gate without top-bucket damage, "
            "but selected combinations did not clear the strict overlap gate."
        )
    else:
        verdict = (
            "FREEZE_PITCHER_BRANCH_AND_PIVOT_EXPECTED_PA: best isolated handedness "
            "feature did not improve Brier by at least ~0.0005 without top-bucket "
            "degradation. No further pitcher tinkering until exposure is properly modeled."
        )

    results["verdict"] = {
        "best_single_holdout": best_single_holdout,
        "best_single_delta_vs_a0": best_single_delta,
        "survival_pass": survival_pass,
        "overlap_unlock": overlap_unlock,
        "overlap_passers": [
            {"result": r, "delta": d}
            for r, d in overlap_passers
        ],
        "verdict": verdict,
    }
    write_json(args.out_json, results)

    print("\nSTRICT GATE READ", flush=True)
    print("----------------", flush=True)
    print(
        "best_isolated_hand_feature:",
        best_single_holdout["label"],
        best_single_holdout["hand_features"],
        flush=True,
    )
    print_delta(best_single_delta)
    print(f"survival_pass: {survival_pass}", flush=True)
    print(f"overlap_unlock: {overlap_unlock}", flush=True)
    print(f"verdict: {verdict}", flush=True)

    print("\nLOCKED RULE", flush=True)
    print("-----------", flush=True)
    print(
        "If the best isolated handedness feature cannot improve Brier by at least "
        "~0.0005 without degrading the top bucket, freeze the pitcher branch "
        "immediately and pivot to expected PA distribution. No further pitcher "
        "tinkering until exposure is properly modeled.",
        flush=True,
    )
    print(
        "Bayesian pitch-zone overlap unlock target: roughly -0.0008 to -0.0010 "
        "Brier improvement with no unacceptable damage to logloss, AUC, or top bucket.",
        flush=True,
    )

    print(f"\nfinal JSON: {args.out_json}", flush=True)

    del Xtrain, Xtest, ytrain, ytest
    del train_2025, test_2026
    gc.collect()


if __name__ == "__main__":
    main()
