#!/usr/bin/env python3
"""
HR_ARCHETYPE_CALIBRATION_GATE_A

One narrow, leak-safe archetype calibration test for the locked HR champion:
    ENVIRONMENT_C_LOCKED_A

Locked state
------------
- Pitcher branch: FROZEN
- Expected-PA branch: FROZEN
- Batter-damage scalar: NOT PROMOTED
- Bayesian pitch-zone overlap: BLOCKED

This script tests ONE hypothesis only:

    Can a small, training-only, aggressively shrunken
    max-EV-tier x barrel-tier calibration layer correct the
    low-power overprediction / high-power underprediction
    without worsening the already-overconfident extreme top 1%?

Safeguards
----------
1. Archetype cut points come from 2025 only.
2. 2025 archetype offsets are learned from expanding-window,
   strictly forward out-of-fold predictions, not in-sample predictions.
3. Bin offsets are aggressively shrunk toward zero.
4. Logit offsets are capped.
5. Positive boosts taper above 20% baseline probability and become zero at 25%.
6. 2026 is used only once for the final forward gate.
7. No feature stacking and no 2026 tuning.

Default fixed correction settings
---------------------------------
- Archetype grid: 3 max-EV tiers x 3 barrel-rate tiers
- Shrink strength: n / (n + 1500)
- Logit offset cap: +/- 0.25
- Positive-boost taper begins at p=0.20
- Positive boost becomes zero at p=0.25

Final promotion gate
--------------------
PASS requires ALL:
- Brier delta <= -0.0001
- Logloss does not worsen
- AUC does not worsen by more than 0.0005
- Top-5% actual rate does not worsen
- Top-1% actual-vs-predicted gap does not become more negative
  than the locked baseline gap

Run
---
python -u hr_archetype_calibration_gate_a.py 2>&1 | tee /data/hr_model/hr_archetype_calibration_gate_a.log

Output
------
/data/hr_model/hr_archetype_calibration_gate_a_results.json

Paste back
----------
OOF COVERAGE
ARCHETYPE CUT POINTS
LEARNED SHRUNKEN OFFSETS
FINAL 2026 GATE
DELTA VS LOCKED BASELINE
TOP-TAIL SAFETY CHECK
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


def find_source(conn, candidates):
    for table in [
        "batter_game_features",
        "batter_games",
        "pitcher_game_features",
        "league_env_lag_features",
    ]:
        cols = set(table_cols(conn, table))
        for col in candidates:
            if col in cols:
                return table, col
    return None, None


def source_expr(conn, out_alias, candidates):
    table, col = find_source(conn, candidates)
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


def fit_locked_predict(
    np,
    SimpleImputer,
    StandardScaler,
    LogisticRegression,
    train_df,
    pred_df,
    c_value,
):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        imputer.fit_transform(
            train_df[LOCKED_FEATURES].to_numpy(dtype="float64")
        )
    )
    X_pred = scaler.transform(
        imputer.transform(
            pred_df[LOCKED_FEATURES].to_numpy(dtype="float64")
        )
    )

    y_train = train_df["actual_hr"].to_numpy(dtype="int8", copy=False)

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=c_value,
    )
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_pred)[:, 1]

    del X_train, X_pred, y_train, imputer, scaler, model
    gc.collect()
    return pred


def build_oof_predictions(
    np,
    pd,
    SimpleImputer,
    StandardScaler,
    LogisticRegression,
    train_2025,
    c_value,
):
    folds = [
        ("2025-04-01", "2025-05-31", "2025-06-01", "2025-06-30"),
        ("2025-04-01", "2025-06-30", "2025-07-01", "2025-07-31"),
        ("2025-04-01", "2025-07-31", "2025-08-01", "2025-08-31"),
        ("2025-04-01", "2025-08-31", "2025-09-01", "2025-10-31"),
    ]

    parts = []

    for fold_no, (tr_start, tr_end, va_start, va_end) in enumerate(folds, 1):
        tr = train_2025[
            (train_2025["game_date"] >= pd.Timestamp(tr_start))
            & (train_2025["game_date"] <= pd.Timestamp(tr_end))
        ].copy()

        va = train_2025[
            (train_2025["game_date"] >= pd.Timestamp(va_start))
            & (train_2025["game_date"] <= pd.Timestamp(va_end))
        ].copy()

        if len(tr) < 3000 or len(va) < 500:
            print(
                f"fold={fold_no} skipped train_rows={len(tr)} val_rows={len(va)}",
                flush=True,
            )
            continue

        pred = fit_locked_predict(
            np,
            SimpleImputer,
            StandardScaler,
            LogisticRegression,
            tr,
            va,
            c_value,
        )

        out = va[
            [
                "game_date",
                "game_id",
                "batter_id",
                "actual_hr",
                "batter_max_ev_60d",
                "batter_barrel_rate_30d_shrunk",
            ]
        ].copy()
        out["oof_pred"] = pred
        out["fold"] = fold_no
        parts.append(out)

        print(
            f"fold={fold_no} train={tr_start}..{tr_end} "
            f"val={va_start}..{va_end} "
            f"train_rows={len(tr):,} val_rows={len(va):,}",
            flush=True,
        )

        del tr, va, out, pred
        gc.collect()

    if not parts:
        raise SystemExit("No usable temporal OOF folds were produced.")

    return pd.concat(parts, ignore_index=True)


def train_tertile_edges(pd, series):
    clean = series.dropna()
    if clean.nunique() < 3:
        raise SystemExit("Not enough unique values for tertile archetype bins.")

    _, edges = pd.qcut(
        clean,
        q=3,
        retbins=True,
        duplicates="drop",
    )
    edges = list(edges)

    if len(edges) != 4:
        raise SystemExit(
            f"Expected 4 tertile edges, got {len(edges)}: {edges}"
        )

    edges[0] = -float("inf")
    edges[-1] = float("inf")
    return edges


def assign_archetypes(pd, df, ev_edges, barrel_edges):
    labels = ["LOW", "MID", "HIGH"]

    out = df.copy()
    out["max_ev_tier"] = pd.cut(
        out["batter_max_ev_60d"],
        bins=ev_edges,
        labels=labels,
        include_lowest=True,
    )
    out["barrel_tier"] = pd.cut(
        out["batter_barrel_rate_30d_shrunk"],
        bins=barrel_edges,
        labels=labels,
        include_lowest=True,
    )
    out["archetype"] = (
        out["max_ev_tier"].astype(str)
        + "__"
        + out["barrel_tier"].astype(str)
    )
    return out


def safe_logit(np, p):
    eps = 1e-6
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def learn_offsets(
    np,
    oof,
    shrink_n,
    offset_cap,
):
    overall_actual = float(oof["actual_hr"].mean())
    overall_pred = float(oof["oof_pred"].mean())

    rows = []
    offsets = {}

    for archetype, g in oof.groupby("archetype", dropna=False):
        n = int(len(g))
        hits = int(g["actual_hr"].sum())
        actual = float(g["actual_hr"].mean())
        pred = float(g["oof_pred"].mean())

        raw_offset = float(
            safe_logit(np, [actual])[0]
            - safe_logit(np, [pred])[0]
        )

        shrink_weight = n / (n + shrink_n)
        shrunk_offset = raw_offset * shrink_weight
        capped_offset = float(
            np.clip(shrunk_offset, -offset_cap, offset_cap)
        )

        row = {
            "archetype": str(archetype),
            "n": n,
            "hits": hits,
            "actual_rate": actual,
            "mean_pred": pred,
            "raw_residual": actual - pred,
            "raw_logit_offset": raw_offset,
            "shrink_weight": shrink_weight,
            "shrunk_logit_offset": shrunk_offset,
            "capped_logit_offset": capped_offset,
        }
        rows.append(row)
        offsets[str(archetype)] = capped_offset

    summary = {
        "overall_oof_actual_rate": overall_actual,
        "overall_oof_mean_pred": overall_pred,
        "overall_oof_residual": overall_actual - overall_pred,
        "shrink_n": shrink_n,
        "offset_cap": offset_cap,
    }

    return offsets, rows, summary


def positive_taper(np, baseline_p, taper_start, taper_zero):
    p = np.asarray(baseline_p, dtype=np.float64)
    factor = np.ones_like(p)

    above_start = p > taper_start
    factor[above_start] = (
        (taper_zero - p[above_start])
        / (taper_zero - taper_start)
    )
    factor[p >= taper_zero] = 0.0

    return np.clip(factor, 0.0, 1.0)


def apply_archetype_correction(
    np,
    test,
    offsets,
    taper_start,
    taper_zero,
):
    baseline = test["baseline_pred"].to_numpy(dtype="float64")
    base_logit = safe_logit(np, baseline)

    raw_offsets = np.array(
        [
            float(offsets.get(str(a), 0.0))
            for a in test["archetype"]
        ],
        dtype="float64",
    )

    taper = positive_taper(
        np,
        baseline,
        taper_start,
        taper_zero,
    )

    effective_offset = raw_offsets.copy()
    pos = effective_offset > 0
    effective_offset[pos] *= taper[pos]

    corrected_logit = base_logit + effective_offset
    corrected = 1.0 / (1.0 + np.exp(-corrected_logit))

    return corrected, raw_offsets, effective_offset, taper


def metrics(np, y, p, brier, logloss, auc):
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
        "top5_gap": float(
            y_arr[idx5].mean() - p_arr[idx5].mean()
        ),
        "top1_rows": int(k1),
        "top1_hits": int(y_arr[idx1].sum()),
        "top1_actual": float(y_arr[idx1].mean()),
        "top1_pred": float(p_arr[idx1].mean()),
        "top1_gap": float(
            y_arr[idx1].mean() - p_arr[idx1].mean()
        ),
    }


def calibration_deciles(pd, df, pred_col):
    d = df[["actual_hr", pred_col]].copy()
    d["decile"] = pd.qcut(
        d[pred_col],
        q=10,
        labels=False,
        duplicates="drop",
    ) + 1

    rows = []
    for decile, g in d.groupby("decile"):
        actual = float(g["actual_hr"].mean())
        pred = float(g[pred_col].mean())
        rows.append({
            "decile": int(decile),
            "n": int(len(g)),
            "actual": actual,
            "pred": pred,
            "gap": actual - pred,
            "hits": int(g["actual_hr"].sum()),
        })
    return rows


def print_metrics(label, m):
    print(
        f"{label}: "
        f"brier={m['brier']:.8f} "
        f"logloss={m['logloss']:.8f} "
        f"auc={m['auc']:.8f} "
        f"top5_actual={m['top5_actual']:.5f} "
        f"top5_pred={m['top5_pred']:.5f} "
        f"top1_actual={m['top1_actual']:.5f} "
        f"top1_pred={m['top1_pred']:.5f}",
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
    ap.add_argument("--archetype-shrink-n", type=float, default=1500.0)
    ap.add_argument("--offset-cap", type=float, default=0.25)
    ap.add_argument("--taper-start", type=float, default=0.20)
    ap.add_argument("--taper-zero", type=float, default=0.25)
    ap.add_argument(
        "--out-json",
        default="/data/hr_model/hr_archetype_calibration_gate_a_results.json",
    )
    args = ap.parse_args()

    (
        np, pd, SimpleImputer, LogisticRegression,
        brier, logloss, auc, StandardScaler
    ) = require_imports()

    print("HR_ARCHETYPE_CALIBRATION_GATE_A", flush=True)
    print("================================", flush=True)
    print(f"db: {args.db}", flush=True)
    print(
        f"fixed_settings: shrink_n={args.archetype_shrink_n} "
        f"offset_cap={args.offset_cap} "
        f"taper_start={args.taper_start} "
        f"taper_zero={args.taper_zero}",
        flush=True,
    )

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
        if c in ("game_date", "game_id"):
            continue
        if c == "actual_hr":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    train_2025 = df[df["game_date"].dt.year == 2025].copy()
    test_2026 = df[df["game_date"].dt.year == 2026].copy()
    del df
    gc.collect()

    add_locked_features(
        train_2025,
        test_2026,
        args.shrink_k,
    )

    for frame in (train_2025, test_2026):
        for c in LOCKED_FEATURES:
            frame[c] = (
                pd.to_numeric(frame[c], errors="coerce")
                .astype("float32")
            )

    # 2025-only archetype cut points.
    ev_edges = train_tertile_edges(
        pd,
        train_2025["batter_max_ev_60d"],
    )
    barrel_edges = train_tertile_edges(
        pd,
        train_2025["batter_barrel_rate_30d_shrunk"],
    )

    train_2025 = assign_archetypes(
        pd,
        train_2025,
        ev_edges,
        barrel_edges,
    )
    test_2026 = assign_archetypes(
        pd,
        test_2026,
        ev_edges,
        barrel_edges,
    )

    print("\nARCHETYPE CUT POINTS", flush=True)
    print("--------------------", flush=True)
    print(f"max_ev_edges={ev_edges}", flush=True)
    print(f"barrel_edges={barrel_edges}", flush=True)

    # Strict forward OOF predictions inside 2025.
    print("\nOOF COVERAGE", flush=True)
    print("------------", flush=True)

    oof = build_oof_predictions(
        np,
        pd,
        SimpleImputer,
        StandardScaler,
        LogisticRegression,
        train_2025,
        args.c,
    )

    # Assign the same 2025-only archetype bins to OOF rows.
    oof = assign_archetypes(
        pd,
        oof,
        ev_edges,
        barrel_edges,
    )

    print(
        f"oof_rows={len(oof):,} "
        f"oof_date_min={oof['game_date'].min()} "
        f"oof_date_max={oof['game_date'].max()} "
        f"oof_actual_rate={oof['actual_hr'].mean():.5f} "
        f"oof_mean_pred={oof['oof_pred'].mean():.5f}",
        flush=True,
    )

    offsets, offset_rows, offset_summary = learn_offsets(
        np,
        oof,
        args.archetype_shrink_n,
        args.offset_cap,
    )

    print("\nLEARNED SHRUNKEN OFFSETS", flush=True)
    print("------------------------", flush=True)
    print(
        f"oof_actual={offset_summary['overall_oof_actual_rate']:.5f} "
        f"oof_pred={offset_summary['overall_oof_mean_pred']:.5f} "
        f"oof_residual={offset_summary['overall_oof_residual']:+.5f}",
        flush=True,
    )
    for r in sorted(offset_rows, key=lambda x: x["archetype"]):
        print(
            f"archetype={r['archetype']} "
            f"n={r['n']:5d} "
            f"actual={r['actual_rate']:.5f} "
            f"pred={r['mean_pred']:.5f} "
            f"residual={r['raw_residual']:+.5f} "
            f"raw_logit={r['raw_logit_offset']:+.5f} "
            f"shrink_w={r['shrink_weight']:.4f} "
            f"final_offset={r['capped_logit_offset']:+.5f}",
            flush=True,
        )

    # Fit locked champion on full 2025 and predict untouched 2026.
    baseline_pred = fit_locked_predict(
        np,
        SimpleImputer,
        StandardScaler,
        LogisticRegression,
        train_2025,
        test_2026,
        args.c,
    )
    test_2026["baseline_pred"] = baseline_pred

    corrected_pred, raw_offsets, effective_offsets, taper = (
        apply_archetype_correction(
            np,
            test_2026,
            offsets,
            args.taper_start,
            args.taper_zero,
        )
    )

    test_2026["corrected_pred"] = corrected_pred
    test_2026["raw_offset"] = raw_offsets
    test_2026["effective_offset"] = effective_offsets
    test_2026["positive_taper"] = taper

    y_test = test_2026["actual_hr"].to_numpy(
        dtype="int8",
        copy=False,
    )

    baseline_metrics = metrics(
        np, y_test, baseline_pred, brier, logloss, auc
    )
    candidate_metrics = metrics(
        np, y_test, corrected_pred, brier, logloss, auc
    )

    delta = {
        "brier_delta": (
            candidate_metrics["brier"]
            - baseline_metrics["brier"]
        ),
        "logloss_delta": (
            candidate_metrics["logloss"]
            - baseline_metrics["logloss"]
        ),
        "auc_delta": (
            candidate_metrics["auc"]
            - baseline_metrics["auc"]
        ),
        "top5_actual_delta": (
            candidate_metrics["top5_actual"]
            - baseline_metrics["top5_actual"]
        ),
        "top1_actual_delta": (
            candidate_metrics["top1_actual"]
            - baseline_metrics["top1_actual"]
        ),
        "top1_gap_delta": (
            candidate_metrics["top1_gap"]
            - baseline_metrics["top1_gap"]
        ),
    }

    print("\nFINAL 2026 GATE", flush=True)
    print("---------------", flush=True)
    print_metrics("LOCKED_BASELINE", baseline_metrics)
    print_metrics("ARCHETYPE_CALIBRATION", candidate_metrics)

    print("\nDELTA VS LOCKED BASELINE", flush=True)
    print("------------------------", flush=True)
    for k, v in delta.items():
        print(f"{k}={v:+.8f}", flush=True)

    print("\nTOP-TAIL SAFETY CHECK", flush=True)
    print("---------------------", flush=True)
    print(
        f"baseline_top5_gap={baseline_metrics['top5_gap']:+.5f} "
        f"candidate_top5_gap={candidate_metrics['top5_gap']:+.5f}",
        flush=True,
    )
    print(
        f"baseline_top1_gap={baseline_metrics['top1_gap']:+.5f} "
        f"candidate_top1_gap={candidate_metrics['top1_gap']:+.5f} "
        f"gap_delta={delta['top1_gap_delta']:+.5f}",
        flush=True,
    )
    print(
        f"positive_offsets_tapered_rows="
        f"{int(((test_2026['raw_offset'] > 0) & (test_2026['positive_taper'] < 1)).sum()):,}",
        flush=True,
    )
    print(
        f"positive_offsets_zeroed_rows="
        f"{int(((test_2026['raw_offset'] > 0) & (test_2026['positive_taper'] <= 0)).sum()):,}",
        flush=True,
    )

    baseline_deciles = calibration_deciles(
        pd,
        test_2026,
        "baseline_pred",
    )
    candidate_deciles = calibration_deciles(
        pd,
        test_2026,
        "corrected_pred",
    )

    brier_pass = delta["brier_delta"] <= -0.0001
    logloss_pass = delta["logloss_delta"] <= 0.0
    auc_pass = delta["auc_delta"] >= -0.0005
    top5_pass = delta["top5_actual_delta"] >= 0.0
    top1_safety_pass = delta["top1_gap_delta"] >= 0.0

    overall_pass = all(
        [
            brier_pass,
            logloss_pass,
            auc_pass,
            top5_pass,
            top1_safety_pass,
        ]
    )

    if overall_pass:
        verdict = "ARCHETYPE_CALIBRATION_PASSES_PROMOTION_GATE"
        next_step = (
            "Lock the bounded shrunken archetype calibration layer, "
            "then run one full holdout confirmation report before production integration."
        )
    else:
        verdict = "ARCHETYPE_CALIBRATION_FAILS_PROMOTION_GATE"
        next_step = (
            "Freeze archetype calibration. Do not rescue-tune this layer. "
            "Consider target reframing or accept the current model ceiling before opening another branch."
        )

    print("\nSTRICT GATE READ", flush=True)
    print("----------------", flush=True)
    print(f"brier_pass: {brier_pass}", flush=True)
    print(f"logloss_pass: {logloss_pass}", flush=True)
    print(f"auc_pass: {auc_pass}", flush=True)
    print(f"top5_pass: {top5_pass}", flush=True)
    print(f"top1_safety_pass: {top1_safety_pass}", flush=True)
    print(f"overall_pass: {overall_pass}", flush=True)
    print(f"verdict: {verdict}", flush=True)
    print(f"next_step: {next_step}", flush=True)
    print(
        "Pitcher branch remains frozen. PA branch remains frozen. "
        "Batter-damage scalar remains not promoted. Bayesian overlap remains blocked.",
        flush=True,
    )

    results = {
        "script": "HR_ARCHETYPE_CALIBRATION_GATE_A",
        "db": args.db,
        "schema_resolution": provenance,
        "fixed_settings": {
            "c": args.c,
            "baseline_shrink_k": args.shrink_k,
            "archetype_shrink_n": args.archetype_shrink_n,
            "offset_cap": args.offset_cap,
            "taper_start": args.taper_start,
            "taper_zero": args.taper_zero,
        },
        "archetype_cut_points_2025": {
            "max_ev_edges": ev_edges,
            "barrel_edges": barrel_edges,
        },
        "oof_summary": {
            "rows": int(len(oof)),
            "date_min": str(oof["game_date"].min()),
            "date_max": str(oof["game_date"].max()),
            **offset_summary,
        },
        "learned_offsets": offset_rows,
        "baseline_metrics_2026": baseline_metrics,
        "candidate_metrics_2026": candidate_metrics,
        "delta_vs_baseline": delta,
        "baseline_calibration_deciles": baseline_deciles,
        "candidate_calibration_deciles": candidate_deciles,
        "gate_checks": {
            "brier_pass": brier_pass,
            "logloss_pass": logloss_pass,
            "auc_pass": auc_pass,
            "top5_pass": top5_pass,
            "top1_safety_pass": top1_safety_pass,
            "overall_pass": overall_pass,
        },
        "gate_thresholds": {
            "brier_delta_max": -0.0001,
            "logloss_delta_max": 0.0,
            "auc_delta_min": -0.0005,
            "top5_actual_delta_min": 0.0,
            "top1_gap_delta_min": 0.0,
        },
        "verdict": verdict,
        "next_step": next_step,
    }

    save_json(args.out_json, results)

    print(f"final JSON: {args.out_json}", flush=True)

    del train_2025, test_2026, oof
    gc.collect()


if __name__ == "__main__":
    main()
