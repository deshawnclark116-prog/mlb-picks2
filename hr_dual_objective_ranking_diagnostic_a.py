#!/usr/bin/env python3
"""
HR_DUAL_OBJECTIVE_RANKING_DIAGNOSTIC_A

Diagnostic question
-------------------
Would separating calibrated probability from elite-candidate ranking resolve
the repeated top-bucket conflict?

Task A:
    Keep ENVIRONMENT_C_LOCKED_A as the probability model.
    Optimize / report Brier, logloss, AUC, calibration.

Task B:
    Train a lightweight pairwise ranking head on the SAME pregame features,
    with a ranking objective designed to order HR outcomes above non-HR outcomes.

Critical safeguards
-------------------
- No new feature branch.
- No 2026 outcome leakage.
- 2026 is forward holdout only.
- Ranking-head weight alpha is selected on late-2025 validation only.
- Final ranking head is refit on all 2025 and applied once to 2026.
- The ranking head does NOT alter the baseline probabilities, so Brier/logloss
  remain those of ENVIRONMENT_C_LOCKED_A.
- Top-1% selected-set calibration safety is checked using the unchanged baseline
  probabilities of the rows selected by the ranking head.

Architecture
------------
Final ranking score:
    hybrid_score = logit(locked_probability) + alpha * z(pairwise_ranker_score)

Where:
- locked_probability comes from ENVIRONMENT_C_LOCKED_A
- pairwise_ranker_score comes from pairwise logistic ranking
- alpha is selected on late-2025 validation from a fixed grid

Pair construction
-----------------
Within each game date:
- every positive HR row may be paired with up to N negative rows
- both directions are added:
      positive - negative -> label 1
      negative - positive -> label 0

This creates a lightweight ranking objective without requiring LightGBM/XGBoost.

Final ranking promotion gate
----------------------------
PASS requires ALL:
- Global top-5% actual HR rate improves by at least +0.005 absolute
  (roughly 6 extra hits in a 1,143-row top-5% bucket)
- Global top-1% actual HR rate does not worsen
- Top-1% selected-set calibration gap does not become more negative
- Credible-set top-5% actual HR rate does not worsen
- Probability Brier/logloss remain unchanged by construction

If it fails:
    Freeze the ranking-head idea and accept the current single-logistic ceiling
    before opening another branch.

Run
---
python -u hr_dual_objective_ranking_diagnostic_a.py 2>&1 | tee /data/hr_model/hr_dual_objective_ranking_diagnostic_a.log

Output
------
/data/hr_model/hr_dual_objective_ranking_diagnostic_a_results.json

Paste back
----------
LOCKED PROBABILITY MODEL CHECK
LATE-2025 ALPHA SELECTION
2026 GLOBAL TOP-BUCKET COMPARISON
2026 CREDIBLE-SET COMPARISON
2026 DAILY TOP-3 COMPARISON
TOP-1 CALIBRATION SAFETY
STRICT RANKING GATE READ
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
            np,
            pd,
            SimpleImputer,
            LogisticRegression,
            brier_score_loss,
            log_loss,
            roc_auc_score,
            StandardScaler,
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
        "1", "true", "t", "yes", "y",
        "out", "toward", "pull", "wind_out",
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
        df["wind_out_and_hot"] = (
            df["wind_out_component"] * df["temp_over_75"]
        )


def fit_preprocessor(
    SimpleImputer,
    StandardScaler,
    train_df,
    pred_df,
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

    return X_train, X_pred, imputer, scaler


def fit_locked_probability(
    np,
    SimpleImputer,
    StandardScaler,
    LogisticRegression,
    train_df,
    pred_df,
    c_value,
):
    X_train, X_pred, imputer, scaler = fit_preprocessor(
        SimpleImputer,
        StandardScaler,
        train_df,
        pred_df,
    )

    y_train = train_df["actual_hr"].to_numpy(dtype="int8", copy=False)

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=c_value,
    )
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_pred)[:, 1]

    del X_train, X_pred, y_train, model, imputer, scaler
    gc.collect()

    return pred


def safe_logit(np, p):
    p = np.asarray(p, dtype="float64")
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def make_pairwise_dataset(
    np,
    dates,
    y,
    X,
    negatives_per_positive,
    rng,
):
    dates = np.asarray(dates)
    y = np.asarray(y, dtype="int8")
    X = np.asarray(X, dtype="float32")

    pair_x = []
    pair_y = []

    unique_dates = np.unique(dates)

    for d in unique_dates:
        idx = np.where(dates == d)[0]
        pos_idx = idx[y[idx] == 1]
        neg_idx = idx[y[idx] == 0]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        for p_idx in pos_idx:
            sample_n = min(negatives_per_positive, len(neg_idx))
            sampled_neg = rng.choice(
                neg_idx,
                size=sample_n,
                replace=False,
            )

            pos_vec = X[p_idx]

            for n_idx in sampled_neg:
                diff = pos_vec - X[n_idx]
                pair_x.append(diff)
                pair_y.append(1)

                pair_x.append(-diff)
                pair_y.append(0)

    if not pair_x:
        raise RuntimeError("No pairwise training examples were created.")

    X_pair = np.asarray(pair_x, dtype="float32")
    y_pair = np.asarray(pair_y, dtype="int8")

    return X_pair, y_pair


def fit_pairwise_ranker(
    np,
    LogisticRegression,
    dates,
    y,
    X,
    negatives_per_positive,
    seed,
    rank_c,
):
    rng = np.random.default_rng(seed)

    X_pair, y_pair = make_pairwise_dataset(
        np,
        dates,
        y,
        X,
        negatives_per_positive,
        rng,
    )

    print(
        f"pairwise_examples={len(y_pair):,} "
        f"pair_features={X_pair.shape[1]}",
        flush=True,
    )

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=rank_c,
    )
    model.fit(X_pair, y_pair)

    del X_pair, y_pair
    gc.collect()

    return model


def ranking_metrics(np, y, score, baseline_prob):
    y = np.asarray(y, dtype="int8")
    score = np.asarray(score, dtype="float64")
    baseline_prob = np.asarray(baseline_prob, dtype="float64")

    k5 = max(1, int(math.ceil(len(y) * 0.05)))
    idx5 = np.argpartition(score, -k5)[-k5:]

    k1 = max(1, int(math.ceil(len(y) * 0.01)))
    idx1 = np.argpartition(score, -k1)[-k1:]

    return {
        "rows": int(len(y)),
        "top5_rows": int(k5),
        "top5_hits": int(y[idx5].sum()),
        "top5_actual": float(y[idx5].mean()),
        "top5_baseline_pred": float(baseline_prob[idx5].mean()),

        "top1_rows": int(k1),
        "top1_hits": int(y[idx1].sum()),
        "top1_actual": float(y[idx1].mean()),
        "top1_baseline_pred": float(baseline_prob[idx1].mean()),
        "top1_gap": float(
            y[idx1].mean() - baseline_prob[idx1].mean()
        ),
    }


def candidate_set_metrics(
    np,
    y,
    baseline_prob,
    baseline_score,
    candidate_score,
    threshold,
):
    mask = baseline_prob >= threshold

    if mask.sum() < 100:
        raise RuntimeError(
            f"Credible candidate set too small: n={int(mask.sum())}"
        )

    y_sub = y[mask]
    p_sub = baseline_prob[mask]
    base_sub = baseline_score[mask]
    cand_sub = candidate_score[mask]

    base = ranking_metrics(
        np,
        y_sub,
        base_sub,
        p_sub,
    )
    cand = ranking_metrics(
        np,
        y_sub,
        cand_sub,
        p_sub,
    )

    base["credible_rows"] = int(mask.sum())
    cand["credible_rows"] = int(mask.sum())

    return base, cand


def daily_topk_metrics(pd, df, score_col, k=3):
    total_rows = 0
    total_hits = 0
    days = 0
    days_with_hr = 0

    for _, g in df.groupby("game_date"):
        take = min(k, len(g))
        top = g.nlargest(take, score_col)

        hits = int(top["actual_hr"].sum())

        total_rows += len(top)
        total_hits += hits
        days += 1
        days_with_hr += int(hits > 0)

    return {
        "days": int(days),
        "rows_selected": int(total_rows),
        "hits": int(total_hits),
        "hit_rate": (
            float(total_hits / total_rows)
            if total_rows else float("nan")
        ),
        "days_with_at_least_one_hr": int(days_with_hr),
        "day_success_rate": (
            float(days_with_hr / days)
            if days else float("nan")
        ),
    }


def probability_metrics(np, y, p, brier, logloss, auc):
    y = np.asarray(y, dtype="int8")
    p = np.asarray(p, dtype="float64")

    return {
        "rows": int(len(y)),
        "actual_rate": float(y.mean()),
        "mean_pred": float(p.mean()),
        "brier": float(brier(y, p)),
        "logloss": float(logloss(y, p, labels=[0, 1])),
        "auc": float(auc(y, p)),
    }


def choose_alpha(
    np,
    y_val,
    baseline_prob,
    rank_score_z,
    credible_threshold,
    alpha_grid,
):
    baseline_logit = safe_logit(np, baseline_prob)

    baseline_global = ranking_metrics(
        np,
        y_val,
        baseline_prob,
        baseline_prob,
    )

    rows = []

    for alpha in alpha_grid:
        hybrid = baseline_logit + alpha * rank_score_z

        global_m = ranking_metrics(
            np,
            y_val,
            hybrid,
            baseline_prob,
        )

        credible_base, credible_candidate = candidate_set_metrics(
            np,
            y_val,
            baseline_prob,
            baseline_prob,
            hybrid,
            credible_threshold,
        )

        row = {
            "alpha": float(alpha),
            "global_top5_actual": global_m["top5_actual"],
            "global_top5_hits": global_m["top5_hits"],
            "global_top1_actual": global_m["top1_actual"],
            "global_top1_gap": global_m["top1_gap"],
            "credible_top5_actual": credible_candidate["top5_actual"],
            "credible_top5_hits": credible_candidate["top5_hits"],
            "baseline_global_top5_actual": baseline_global["top5_actual"],
            "baseline_global_top1_actual": baseline_global["top1_actual"],
            "baseline_global_top1_gap": baseline_global["top1_gap"],
            "credible_baseline_top5_actual": credible_base["top5_actual"],
        }
        rows.append(row)

    # Select on 2025 validation only.
    # Primary: global top-5 actual.
    # Tie breakers: global top-1 actual, credible-set top-5 actual, smaller alpha.
    best = max(
        rows,
        key=lambda r: (
            r["global_top5_actual"],
            r["global_top1_actual"],
            r["credible_top5_actual"],
            -r["alpha"],
        ),
    )

    return best, rows


def save_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def print_rank_block(label, m):
    print(
        f"{label}: "
        f"top5_actual={m['top5_actual']:.5f} "
        f"top5_hits={m['top5_hits']}/{m['top5_rows']} "
        f"top1_actual={m['top1_actual']:.5f} "
        f"top1_hits={m['top1_hits']}/{m['top1_rows']} "
        f"top1_gap={m['top1_gap']:+.5f}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--baseline-c", type=float, default=0.03)
    ap.add_argument("--rank-c", type=float, default=0.10)
    ap.add_argument("--shrink-k", type=float, default=75.0)
    ap.add_argument("--selection-split", default="2025-08-01")

    ap.add_argument(
        "--negatives-per-positive",
        type=int,
        default=8,
    )
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--credible-quantile",
        type=float,
        default=0.75,
    )

    ap.add_argument(
        "--alpha-grid",
        default="0,0.10,0.25,0.50,0.75,1.0,1.5,2.0",
    )

    ap.add_argument(
        "--out-json",
        default="/data/hr_model/hr_dual_objective_ranking_diagnostic_a_results.json",
    )

    args = ap.parse_args()

    (
        np,
        pd,
        SimpleImputer,
        LogisticRegression,
        brier,
        logloss,
        auc,
        StandardScaler,
    ) = require_imports()

    alpha_grid = [
        float(x.strip())
        for x in args.alpha_grid.split(",")
        if x.strip()
    ]

    print("HR_DUAL_OBJECTIVE_RANKING_DIAGNOSTIC_A", flush=True)
    print("======================================", flush=True)
    print(f"db: {args.db}", flush=True)
    print(f"selection_split: {args.selection_split}", flush=True)
    print(f"alpha_grid: {alpha_grid}", flush=True)

    conn = sqlite3.connect(args.db)
    df, provenance = load_dataset(conn, pd)
    conn.close()

    df["game_date"] = pd.to_datetime(
        df["game_date"],
        errors="coerce",
    )
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

    split_ts = pd.Timestamp(args.selection_split)

    sel_train = train_2025[
        train_2025["game_date"] < split_ts
    ].copy()

    sel_val = train_2025[
        train_2025["game_date"] >= split_ts
    ].copy()

    print("\nSELECTION SPLIT", flush=True)
    print("---------------", flush=True)
    print(
        f"sel_train_rows={len(sel_train):,} "
        f"sel_train_range={sel_train['game_date'].min()}..{sel_train['game_date'].max()}",
        flush=True,
    )
    print(
        f"sel_val_rows={len(sel_val):,} "
        f"sel_val_range={sel_val['game_date'].min()}..{sel_val['game_date'].max()}",
        flush=True,
    )
    print("2026 holdout is untouched during alpha selection.", flush=True)

    # ----- 2025 internal alpha selection -----
    sel_imputer = SimpleImputer(strategy="median")
    sel_scaler = StandardScaler()

    X_sel_train = sel_scaler.fit_transform(
        sel_imputer.fit_transform(
            sel_train[LOCKED_FEATURES].to_numpy(dtype="float64")
        )
    ).astype("float32")

    X_sel_val = sel_scaler.transform(
        sel_imputer.transform(
            sel_val[LOCKED_FEATURES].to_numpy(dtype="float64")
        )
    ).astype("float32")

    y_sel_train = sel_train["actual_hr"].to_numpy(
        dtype="int8",
        copy=False,
    )
    y_sel_val = sel_val["actual_hr"].to_numpy(
        dtype="int8",
        copy=False,
    )

    # Baseline probability model for late-2025 validation.
    baseline_sel_model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=args.baseline_c,
    )
    baseline_sel_model.fit(X_sel_train, y_sel_train)
    baseline_sel_val_pred = baseline_sel_model.predict_proba(
        X_sel_val
    )[:, 1]

    credible_threshold = float(
        np.quantile(
            baseline_sel_val_pred,
            args.credible_quantile,
        )
    )

    print("\n2025 CREDIBLE-SET THRESHOLD", flush=True)
    print("---------------------------", flush=True)
    print(
        f"credible_quantile={args.credible_quantile:.2f} "
        f"threshold_from_late_2025={credible_threshold:.6f}",
        flush=True,
    )

    print("\nTRAINING 2025 SELECTION RANKING HEAD", flush=True)
    print("------------------------------------", flush=True)

    rank_sel_model = fit_pairwise_ranker(
        np,
        LogisticRegression,
        sel_train["game_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        y_sel_train,
        X_sel_train,
        args.negatives_per_positive,
        args.seed,
        args.rank_c,
    )

    rank_train_score = rank_sel_model.decision_function(
        X_sel_train
    )
    rank_val_score = rank_sel_model.decision_function(
        X_sel_val
    )

    rank_mean = float(rank_train_score.mean())
    rank_std = float(rank_train_score.std())
    if rank_std <= 1e-12:
        rank_std = 1.0

    rank_val_z = (rank_val_score - rank_mean) / rank_std

    best_alpha_row, alpha_rows = choose_alpha(
        np,
        y_sel_val,
        baseline_sel_val_pred,
        rank_val_z,
        credible_threshold,
        alpha_grid,
    )

    selected_alpha = float(best_alpha_row["alpha"])

    print("\nLATE-2025 ALPHA SELECTION", flush=True)
    print("-------------------------", flush=True)

    for r in alpha_rows:
        print(
            f"alpha={r['alpha']:.2f} "
            f"global_top5={r['global_top5_actual']:.5f} "
            f"global_top5_hits={r['global_top5_hits']} "
            f"global_top1={r['global_top1_actual']:.5f} "
            f"global_top1_gap={r['global_top1_gap']:+.5f} "
            f"credible_top5={r['credible_top5_actual']:.5f}",
            flush=True,
        )

    print(
        f"selected_alpha={selected_alpha:.2f} "
        f"selected_on_late_2025_only",
        flush=True,
    )

    del (
        X_sel_train,
        X_sel_val,
        y_sel_train,
        y_sel_val,
        baseline_sel_model,
        baseline_sel_val_pred,
        rank_sel_model,
        rank_train_score,
        rank_val_score,
        rank_val_z,
        sel_imputer,
        sel_scaler,
        sel_train,
        sel_val,
    )
    gc.collect()

    # ----- Final full-2025 -> 2026 forward evaluation -----
    full_imputer = SimpleImputer(strategy="median")
    full_scaler = StandardScaler()

    X_train = full_scaler.fit_transform(
        full_imputer.fit_transform(
            train_2025[LOCKED_FEATURES].to_numpy(dtype="float64")
        )
    ).astype("float32")

    X_test = full_scaler.transform(
        full_imputer.transform(
            test_2026[LOCKED_FEATURES].to_numpy(dtype="float64")
        )
    ).astype("float32")

    y_train = train_2025["actual_hr"].to_numpy(
        dtype="int8",
        copy=False,
    )
    y_test = test_2026["actual_hr"].to_numpy(
        dtype="int8",
        copy=False,
    )

    baseline_model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=args.baseline_c,
    )
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict_proba(
        X_test
    )[:, 1]

    probability_m = probability_metrics(
        np,
        y_test,
        baseline_pred,
        brier,
        logloss,
        auc,
    )

    print("\nLOCKED PROBABILITY MODEL CHECK", flush=True)
    print("------------------------------", flush=True)
    print(
        f"brier={probability_m['brier']:.8f} "
        f"logloss={probability_m['logloss']:.8f} "
        f"auc={probability_m['auc']:.8f} "
        f"actual_rate={probability_m['actual_rate']:.5f} "
        f"mean_pred={probability_m['mean_pred']:.5f}",
        flush=True,
    )
    print(
        "These probabilities are frozen and are not altered by the ranking head.",
        flush=True,
    )

    print("\nTRAINING FINAL 2025 RANKING HEAD", flush=True)
    print("--------------------------------", flush=True)

    final_rank_model = fit_pairwise_ranker(
        np,
        LogisticRegression,
        train_2025["game_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        y_train,
        X_train,
        args.negatives_per_positive,
        args.seed,
        args.rank_c,
    )

    train_rank_score = final_rank_model.decision_function(X_train)
    test_rank_score = final_rank_model.decision_function(X_test)

    final_rank_mean = float(train_rank_score.mean())
    final_rank_std = float(train_rank_score.std())
    if final_rank_std <= 1e-12:
        final_rank_std = 1.0

    test_rank_z = (
        test_rank_score - final_rank_mean
    ) / final_rank_std

    baseline_logit = safe_logit(np, baseline_pred)

    hybrid_score = (
        baseline_logit
        + selected_alpha * test_rank_z
    )

    baseline_global = ranking_metrics(
        np,
        y_test,
        baseline_pred,
        baseline_pred,
    )

    hybrid_global = ranking_metrics(
        np,
        y_test,
        hybrid_score,
        baseline_pred,
    )

    print("\n2026 GLOBAL TOP-BUCKET COMPARISON", flush=True)
    print("---------------------------------", flush=True)
    print_rank_block("LOCKED_BASELINE", baseline_global)
    print_rank_block("HYBRID_RANKING_HEAD", hybrid_global)

    global_top5_delta = (
        hybrid_global["top5_actual"]
        - baseline_global["top5_actual"]
    )
    global_top1_delta = (
        hybrid_global["top1_actual"]
        - baseline_global["top1_actual"]
    )
    top1_gap_delta = (
        hybrid_global["top1_gap"]
        - baseline_global["top1_gap"]
    )

    print(
        f"global_top5_actual_delta={global_top5_delta:+.5f} "
        f"global_top5_hit_delta="
        f"{hybrid_global['top5_hits'] - baseline_global['top5_hits']:+d}",
        flush=True,
    )
    print(
        f"global_top1_actual_delta={global_top1_delta:+.5f} "
        f"top1_gap_delta={top1_gap_delta:+.5f}",
        flush=True,
    )

    credible_base, credible_hybrid = candidate_set_metrics(
        np,
        y_test,
        baseline_pred,
        baseline_pred,
        hybrid_score,
        credible_threshold,
    )

    print("\n2026 CREDIBLE-SET COMPARISON", flush=True)
    print("----------------------------", flush=True)
    print(
        f"credible_threshold={credible_threshold:.6f} "
        f"credible_rows={credible_base['credible_rows']:,}",
        flush=True,
    )
    print_rank_block("CREDIBLE_BASELINE", credible_base)
    print_rank_block("CREDIBLE_HYBRID", credible_hybrid)

    credible_top5_delta = (
        credible_hybrid["top5_actual"]
        - credible_base["top5_actual"]
    )

    print(
        f"credible_top5_actual_delta={credible_top5_delta:+.5f} "
        f"credible_top5_hit_delta="
        f"{credible_hybrid['top5_hits'] - credible_base['top5_hits']:+d}",
        flush=True,
    )

    eval_df = test_2026[
        [
            "game_date",
            "game_id",
            "batter_id",
            "actual_hr",
        ]
    ].copy()

    eval_df["baseline_score"] = baseline_pred
    eval_df["hybrid_score"] = hybrid_score

    daily_base = daily_topk_metrics(
        pd,
        eval_df,
        "baseline_score",
        k=3,
    )

    daily_hybrid = daily_topk_metrics(
        pd,
        eval_df,
        "hybrid_score",
        k=3,
    )

    print("\n2026 DAILY TOP-3 COMPARISON", flush=True)
    print("---------------------------", flush=True)
    print(
        f"BASELINE: days={daily_base['days']} "
        f"hits={daily_base['hits']}/{daily_base['rows_selected']} "
        f"hit_rate={daily_base['hit_rate']:.5f} "
        f"day_success_rate={daily_base['day_success_rate']:.5f}",
        flush=True,
    )
    print(
        f"HYBRID: days={daily_hybrid['days']} "
        f"hits={daily_hybrid['hits']}/{daily_hybrid['rows_selected']} "
        f"hit_rate={daily_hybrid['hit_rate']:.5f} "
        f"day_success_rate={daily_hybrid['day_success_rate']:.5f}",
        flush=True,
    )

    print("\nTOP-1 CALIBRATION SAFETY", flush=True)
    print("------------------------", flush=True)
    print(
        f"baseline_top1_actual={baseline_global['top1_actual']:.5f} "
        f"baseline_top1_pred={baseline_global['top1_baseline_pred']:.5f} "
        f"baseline_top1_gap={baseline_global['top1_gap']:+.5f}",
        flush=True,
    )
    print(
        f"hybrid_top1_actual={hybrid_global['top1_actual']:.5f} "
        f"hybrid_selected_rows_mean_baseline_pred="
        f"{hybrid_global['top1_baseline_pred']:.5f} "
        f"hybrid_top1_gap={hybrid_global['top1_gap']:+.5f} "
        f"gap_delta={top1_gap_delta:+.5f}",
        flush=True,
    )

    top5_pass = global_top5_delta >= 0.005
    top1_pass = global_top1_delta >= 0.0
    top1_safety_pass = top1_gap_delta >= 0.0
    credible_top5_pass = credible_top5_delta >= 0.0

    overall_pass = all(
        [
            top5_pass,
            top1_pass,
            top1_safety_pass,
            credible_top5_pass,
        ]
    )

    if overall_pass:
        verdict = "DUAL_OBJECTIVE_RANKING_HEAD_PASSES_DIAGNOSTIC"
        next_step = (
            "Architecture split is validated: keep locked probabilities unchanged "
            "and run one confirmation report for the elite-ranking head before production use."
        )
    else:
        verdict = "DUAL_OBJECTIVE_RANKING_HEAD_FAILS_DIAGNOSTIC"
        next_step = (
            "Freeze the ranking-head idea. Do not rescue-tune it on 2026. "
            "Accept the current single-logistic ceiling before opening another branch."
        )

    print("\nSTRICT RANKING GATE READ", flush=True)
    print("------------------------", flush=True)
    print(f"selected_alpha_2025_only: {selected_alpha:.2f}", flush=True)
    print(f"top5_pass: {top5_pass}", flush=True)
    print(f"top1_pass: {top1_pass}", flush=True)
    print(f"top1_safety_pass: {top1_safety_pass}", flush=True)
    print(f"credible_top5_pass: {credible_top5_pass}", flush=True)
    print(f"overall_pass: {overall_pass}", flush=True)
    print(f"verdict: {verdict}", flush=True)
    print(f"next_step: {next_step}", flush=True)
    print(
        "Probability Brier/logloss remain those of ENVIRONMENT_C_LOCKED_A "
        "because the ranking head changes ordering only, not probabilities.",
        flush=True,
    )
    print(
        "Pitcher branch remains frozen. PA branch remains frozen. "
        "Batter-damage scalar remains not promoted. Bayesian overlap remains blocked.",
        flush=True,
    )

    results = {
        "script": "HR_DUAL_OBJECTIVE_RANKING_DIAGNOSTIC_A",
        "db": args.db,
        "schema_resolution": provenance,

        "settings": {
            "baseline_c": args.baseline_c,
            "rank_c": args.rank_c,
            "shrink_k": args.shrink_k,
            "selection_split": args.selection_split,
            "negatives_per_positive": args.negatives_per_positive,
            "seed": args.seed,
            "credible_quantile": args.credible_quantile,
            "credible_threshold_late_2025": credible_threshold,
            "alpha_grid": alpha_grid,
            "selected_alpha_late_2025": selected_alpha,
        },

        "late_2025_alpha_selection": {
            "selected": best_alpha_row,
            "all_candidates": alpha_rows,
        },

        "locked_probability_metrics_2026": probability_m,

        "global_ranking_2026": {
            "baseline": baseline_global,
            "hybrid": hybrid_global,
            "top5_actual_delta": global_top5_delta,
            "top1_actual_delta": global_top1_delta,
            "top1_gap_delta": top1_gap_delta,
        },

        "credible_set_2026": {
            "threshold": credible_threshold,
            "baseline": credible_base,
            "hybrid": credible_hybrid,
            "top5_actual_delta": credible_top5_delta,
        },

        "daily_top3_2026": {
            "baseline": daily_base,
            "hybrid": daily_hybrid,
        },

        "gate_thresholds": {
            "global_top5_actual_delta_min": 0.005,
            "global_top1_actual_delta_min": 0.0,
            "top1_gap_delta_min": 0.0,
            "credible_top5_actual_delta_min": 0.0,
        },

        "gate_checks": {
            "top5_pass": top5_pass,
            "top1_pass": top1_pass,
            "top1_safety_pass": top1_safety_pass,
            "credible_top5_pass": credible_top5_pass,
            "overall_pass": overall_pass,
        },

        "verdict": verdict,
        "next_step": next_step,
    }

    save_json(args.out_json, results)

    print(f"final JSON: {args.out_json}", flush=True)

    del (
        X_train,
        X_test,
        baseline_pred,
        baseline_model,
        final_rank_model,
        train_rank_score,
        test_rank_score,
        test_rank_z,
        hybrid_score,
        train_2025,
        test_2026,
        eval_df,
    )
    gc.collect()


if __name__ == "__main__":
    main()
