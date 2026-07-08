#!/usr/bin/env python3
"""
HR_HAND_PITCHER_VECTOR_A

Builds handedness-conditioned opposing pitcher vectors and reruns the blunt
opponent gate against the current leak-safe HR baseline.

Run:
    python hr_hand_pitcher_vector_a.py --rebuild-features

Paste:
    HAND FEATURE COVERAGE
    HAND HR/BBE DECILES
    SUMMARY COMPARISON
    DELTAS VS A0
    TOP COEFFICIENTS
    READ
"""

import argparse
import math
import os
import sqlite3
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

DB_DEFAULT = "/data/hr_model/hr_model.sqlite"

HAND_FEATURES_H1 = ["opp_hand_hr_per_bbe_365d_shrunk"]

HAND_FEATURES_H2 = [
    "opp_hand_hr_per_bbe_365d_shrunk",
    "opp_hand_hr_per_pa_365d_shrunk",
    "opp_hand_k_per_pa_365d_shrunk",
    "opp_hand_gb_per_bbe_365d_shrunk",
    "opp_hand_barrel_per_bbe_365d_shrunk",
]

HAND_FEATURES_H3 = HAND_FEATURES_H2 + [
    "log_opp_hand_bbe_365d",
    "log_opp_hand_pa_365d",
]

PREFERRED_BASE_FEATURES = [
    "expected_pa_v1",
    "lineup_spot",

    "batter_hr_rate_30d_shrunk",
    "batter_hr_rate_60d_shrunk",
    "batter_barrel_rate_30d_shrunk",
    "batter_barrel_rate_60d_shrunk",
    "batter_hard_hit_rate_30d_shrunk",
    "batter_hard_hit_rate_60d_shrunk",
    "batter_pull_air_rate_30d_shrunk",
    "batter_pull_air_rate_60d_shrunk",
    "batter_air_rate_30d_shrunk",
    "batter_air_rate_60d_shrunk",
    "batter_bbe_30d",
    "batter_bbe_60d",

    "pitcher_pull_air_allowed_rate_60d",
    "pitcher_hard_hit_allowed_rate_7d",
    "pitcher_hrfb_rate_30d",
    "pitcher_bbe_allowed_60d",

    "league_hr_per_bbe_10d_lag",
    "league_hr_per_air_10d_lag",
    "league_barrel_rate_10d_lag",
    "league_avg_ev_10d_lag",
    "temp_over_75",
    "temp_over_85",
    "wind_out_component",
    "wind_out_and_hot",
]


def imports():
    try:
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        return (
            np, pd, ColumnTransformer, SimpleImputer, LogisticRegression,
            brier_score_loss, log_loss, roc_auc_score, Pipeline,
            OneHotEncoder, StandardScaler
        )
    except Exception as e:
        raise SystemExit("Missing dependency. Try: pip install pandas numpy scikit-learn\n" + repr(e))


def db_path():
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    return str(p / "hr_model.sqlite") if p.parent.exists() else DB_DEFAULT


def exists(conn, name):
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE name=?",
        (name,),
    ).fetchone() is not None


def build_features(conn):
    with conn:
        for t in [
            "pitcher_hand_daily_a",
            "league_hand_daily_a",
            "first_pitcher_hand_map_a",
            "game_pitcher_workload_hand_a",
            "pitcher_hand_roll_365_a",
            "league_hand_roll_365_a",
            "pitcher_hand_features_a",
        ]:
            conn.execute(f'DROP TABLE IF EXISTS "{t}"')

    print("building pitcher_hand_daily_a...")
    with conn:
        conn.execute("""
        CREATE TABLE pitcher_hand_daily_a AS
        SELECT
            game_date,
            pitcher_id,
            stand AS batter_stand,
            p_throws,
            COUNT(*) AS pitches,
            SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END) AS pa,
            SUM(CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) AS k,
            SUM(CASE WHEN is_bbe=1 THEN 1 ELSE 0 END) AS bbe,
            SUM(CASE WHEN is_hr=1 THEN 1 ELSE 0 END) AS hr,
            SUM(CASE WHEN is_barrel=1 THEN 1 ELSE 0 END) AS barrel,
            SUM(CASE WHEN launch_speed >= 95 THEN 1 ELSE 0 END) AS hardhit,
            SUM(CASE WHEN bb_type='ground_ball' THEN 1 ELSE 0 END) AS gb,
            SUM(CASE WHEN bb_type IN ('fly_ball','line_drive','popup') THEN 1 ELSE 0 END) AS air
        FROM statcast_pitches
        WHERE game_date IS NOT NULL
          AND pitcher_id IS NOT NULL
          AND stand IN ('L','R')
        GROUP BY game_date, pitcher_id, stand, p_throws
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_phda_pitcher_stand_date ON pitcher_hand_daily_a(pitcher_id, batter_stand, game_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_phda_stand_date ON pitcher_hand_daily_a(batter_stand, game_date)")

    print("building league_hand_daily_a...")
    with conn:
        conn.execute("""
        CREATE TABLE league_hand_daily_a AS
        SELECT
            game_date,
            batter_stand,
            SUM(pitches) AS pitches,
            SUM(pa) AS pa,
            SUM(k) AS k,
            SUM(bbe) AS bbe,
            SUM(hr) AS hr,
            SUM(barrel) AS barrel,
            SUM(hardhit) AS hardhit,
            SUM(gb) AS gb,
            SUM(air) AS air
        FROM pitcher_hand_daily_a
        GROUP BY game_date, batter_stand
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lhda_stand_date ON league_hand_daily_a(batter_stand, game_date)")

    print("building first pitcher / workload map...")
    with conn:
        conn.execute("""
        CREATE TABLE game_pitcher_workload_hand_a AS
        SELECT
            CAST(game_pk AS TEXT) AS game_id,
            game_date,
            pitcher_id,
            COUNT(*) AS game_pitches,
            COUNT(DISTINCT at_bat_number) AS game_batters_approx
        FROM statcast_pitches
        WHERE game_pk IS NOT NULL
          AND game_date IS NOT NULL
          AND pitcher_id IS NOT NULL
        GROUP BY CAST(game_pk AS TEXT), game_date, pitcher_id
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gpwh_game_pitcher ON game_pitcher_workload_hand_a(game_id, pitcher_id)")

        conn.execute("""
        CREATE TABLE first_pitcher_hand_map_a AS
        WITH ranked AS (
            SELECT
                CAST(game_pk AS TEXT) AS game_id,
                game_date,
                batter_id,
                pitcher_id,
                stand AS batter_stand,
                p_throws,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(game_pk AS TEXT), batter_id
                    ORDER BY COALESCE(at_bat_number, 9999), COALESCE(pitch_number, 9999)
                ) AS rn
            FROM statcast_pitches
            WHERE game_pk IS NOT NULL
              AND game_date IS NOT NULL
              AND batter_id IS NOT NULL
              AND pitcher_id IS NOT NULL
              AND stand IN ('L','R')
        )
        SELECT
            r.game_id,
            r.game_date,
            r.batter_id,
            r.pitcher_id AS first_pitcher_id,
            r.batter_stand,
            r.p_throws,
            w.game_pitches AS first_pitcher_game_pitches,
            w.game_batters_approx AS first_pitcher_game_batters,
            CASE
                WHEN w.game_pitches >= 60 OR w.game_batters_approx >= 18 THEN 'traditional'
                WHEN w.game_pitches <= 40 OR w.game_batters_approx <= 9 THEN 'opener'
                ELSE 'ambiguous'
            END AS first_pitcher_role
        FROM ranked r
        LEFT JOIN game_pitcher_workload_hand_a w
          ON r.game_id=w.game_id
         AND r.pitcher_id=w.pitcher_id
        WHERE rn=1
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fphm_game_batter ON first_pitcher_hand_map_a(game_id, batter_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fphm_pitcher_stand_date ON first_pitcher_hand_map_a(first_pitcher_id, batter_stand, game_date)")

    print("building pitcher_hand_roll_365_a...")
    with conn:
        conn.execute("""
        CREATE TABLE pitcher_hand_roll_365_a AS
        SELECT
            CAST(bg.game_id AS TEXT) AS game_id,
            bg.batter_id,
            bg.game_date,
            m.first_pitcher_id,
            m.batter_stand,
            m.p_throws,
            m.first_pitcher_role,
            COALESCE(SUM(d.pitches),0) AS opp_hand_pitches_365d,
            COALESCE(SUM(d.pa),0) AS opp_hand_pa_365d,
            COALESCE(SUM(d.k),0) AS opp_hand_k_365d,
            COALESCE(SUM(d.bbe),0) AS opp_hand_bbe_365d,
            COALESCE(SUM(d.hr),0) AS opp_hand_hr_365d,
            COALESCE(SUM(d.barrel),0) AS opp_hand_barrel_365d,
            COALESCE(SUM(d.hardhit),0) AS opp_hand_hardhit_365d,
            COALESCE(SUM(d.gb),0) AS opp_hand_gb_365d,
            COALESCE(SUM(d.air),0) AS opp_hand_air_365d
        FROM batter_games bg
        LEFT JOIN first_pitcher_hand_map_a m
          ON CAST(bg.game_id AS TEXT)=m.game_id
         AND bg.batter_id=m.batter_id
        LEFT JOIN pitcher_hand_daily_a d
          ON d.pitcher_id=m.first_pitcher_id
         AND d.batter_stand=m.batter_stand
         AND d.game_date < bg.game_date
         AND d.game_date >= DATE(bg.game_date, '-365 day')
        WHERE bg.actual_hr IS NOT NULL
        GROUP BY CAST(bg.game_id AS TEXT), bg.batter_id
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_phr_game_batter ON pitcher_hand_roll_365_a(game_id, batter_id)")

    print("building league_hand_roll_365_a...")
    with conn:
        conn.execute("""
        CREATE TABLE league_hand_roll_365_a AS
        SELECT
            CAST(bg.game_id AS TEXT) AS game_id,
            bg.batter_id,
            m.batter_stand,
            COALESCE(SUM(d.pitches),0) AS lg_hand_pitches_365d,
            COALESCE(SUM(d.pa),0) AS lg_hand_pa_365d,
            COALESCE(SUM(d.k),0) AS lg_hand_k_365d,
            COALESCE(SUM(d.bbe),0) AS lg_hand_bbe_365d,
            COALESCE(SUM(d.hr),0) AS lg_hand_hr_365d,
            COALESCE(SUM(d.barrel),0) AS lg_hand_barrel_365d,
            COALESCE(SUM(d.hardhit),0) AS lg_hand_hardhit_365d,
            COALESCE(SUM(d.gb),0) AS lg_hand_gb_365d,
            COALESCE(SUM(d.air),0) AS lg_hand_air_365d
        FROM batter_games bg
        LEFT JOIN first_pitcher_hand_map_a m
          ON CAST(bg.game_id AS TEXT)=m.game_id
         AND bg.batter_id=m.batter_id
        LEFT JOIN league_hand_daily_a d
          ON d.batter_stand=m.batter_stand
         AND d.game_date < bg.game_date
         AND d.game_date >= DATE(bg.game_date, '-365 day')
        WHERE bg.actual_hr IS NOT NULL
        GROUP BY CAST(bg.game_id AS TEXT), bg.batter_id
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lhr_game_batter ON league_hand_roll_365_a(game_id, batter_id)")

    print("building pitcher_hand_features_a...")
    with conn:
        conn.execute("""
        CREATE TABLE pitcher_hand_features_a AS
        SELECT
            p.game_id,
            p.batter_id,
            p.game_date,
            p.first_pitcher_id,
            p.batter_stand,
            p.p_throws,
            p.first_pitcher_role,

            p.opp_hand_pitches_365d,
            p.opp_hand_pa_365d,
            p.opp_hand_bbe_365d,
            p.opp_hand_hr_365d,
            p.opp_hand_k_365d,
            p.opp_hand_gb_365d,
            p.opp_hand_barrel_365d,

            CASE WHEN p.opp_hand_bbe_365d > 0 THEN 1.0*p.opp_hand_hr_365d/p.opp_hand_bbe_365d END AS opp_hand_hr_per_bbe_365d_raw,
            CASE WHEN p.opp_hand_pa_365d > 0 THEN 1.0*p.opp_hand_hr_365d/p.opp_hand_pa_365d END AS opp_hand_hr_per_pa_365d_raw,
            CASE WHEN p.opp_hand_pa_365d > 0 THEN 1.0*p.opp_hand_k_365d/p.opp_hand_pa_365d END AS opp_hand_k_per_pa_365d_raw,
            CASE WHEN p.opp_hand_bbe_365d > 0 THEN 1.0*p.opp_hand_gb_365d/p.opp_hand_bbe_365d END AS opp_hand_gb_per_bbe_365d_raw,
            CASE WHEN p.opp_hand_bbe_365d > 0 THEN 1.0*p.opp_hand_barrel_365d/p.opp_hand_bbe_365d END AS opp_hand_barrel_per_bbe_365d_raw,

            CASE WHEN l.lg_hand_bbe_365d > 0 THEN 1.0*l.lg_hand_hr_365d/l.lg_hand_bbe_365d ELSE 0.045 END AS lg_hand_hr_per_bbe_365d,
            CASE WHEN l.lg_hand_pa_365d > 0 THEN 1.0*l.lg_hand_hr_365d/l.lg_hand_pa_365d ELSE 0.030 END AS lg_hand_hr_per_pa_365d,
            CASE WHEN l.lg_hand_pa_365d > 0 THEN 1.0*l.lg_hand_k_365d/l.lg_hand_pa_365d ELSE 0.220 END AS lg_hand_k_per_pa_365d,
            CASE WHEN l.lg_hand_bbe_365d > 0 THEN 1.0*l.lg_hand_gb_365d/l.lg_hand_bbe_365d ELSE 0.420 END AS lg_hand_gb_per_bbe_365d,
            CASE WHEN l.lg_hand_bbe_365d > 0 THEN 1.0*l.lg_hand_barrel_365d/l.lg_hand_bbe_365d ELSE 0.080 END AS lg_hand_barrel_per_bbe_365d,

            (COALESCE(p.opp_hand_hr_365d,0) + 200.0*(CASE WHEN l.lg_hand_bbe_365d > 0 THEN 1.0*l.lg_hand_hr_365d/l.lg_hand_bbe_365d ELSE 0.045 END))
              / NULLIF(COALESCE(p.opp_hand_bbe_365d,0) + 200.0, 0) AS opp_hand_hr_per_bbe_365d_shrunk,

            (COALESCE(p.opp_hand_hr_365d,0) + 300.0*(CASE WHEN l.lg_hand_pa_365d > 0 THEN 1.0*l.lg_hand_hr_365d/l.lg_hand_pa_365d ELSE 0.030 END))
              / NULLIF(COALESCE(p.opp_hand_pa_365d,0) + 300.0, 0) AS opp_hand_hr_per_pa_365d_shrunk,

            (COALESCE(p.opp_hand_k_365d,0) + 300.0*(CASE WHEN l.lg_hand_pa_365d > 0 THEN 1.0*l.lg_hand_k_365d/l.lg_hand_pa_365d ELSE 0.220 END))
              / NULLIF(COALESCE(p.opp_hand_pa_365d,0) + 300.0, 0) AS opp_hand_k_per_pa_365d_shrunk,

            (COALESCE(p.opp_hand_gb_365d,0) + 200.0*(CASE WHEN l.lg_hand_bbe_365d > 0 THEN 1.0*l.lg_hand_gb_365d/l.lg_hand_bbe_365d ELSE 0.420 END))
              / NULLIF(COALESCE(p.opp_hand_bbe_365d,0) + 200.0, 0) AS opp_hand_gb_per_bbe_365d_shrunk,

            (COALESCE(p.opp_hand_barrel_365d,0) + 200.0*(CASE WHEN l.lg_hand_bbe_365d > 0 THEN 1.0*l.lg_hand_barrel_365d/l.lg_hand_bbe_365d ELSE 0.080 END))
              / NULLIF(COALESCE(p.opp_hand_bbe_365d,0) + 200.0, 0) AS opp_hand_barrel_per_bbe_365d_shrunk,

            LOG(COALESCE(p.opp_hand_bbe_365d,0) + 1.0) AS log_opp_hand_bbe_365d,
            LOG(COALESCE(p.opp_hand_pa_365d,0) + 1.0) AS log_opp_hand_pa_365d
        FROM pitcher_hand_roll_365_a p
        LEFT JOIN league_hand_roll_365_a l
          ON p.game_id=l.game_id
         AND p.batter_id=l.batter_id
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_phf_game_batter ON pitcher_hand_features_a(game_id, batter_id)")


def load_dataset(pd, conn):
    if not exists(conn, "hr_model_dataset_view"):
        raise SystemExit("Missing hr_model_dataset_view. Re-run the locked HR feature builder first.")

    df = pd.read_sql_query("""
    SELECT
        v.*,
        h.first_pitcher_id,
        h.batter_stand,
        h.p_throws,
        h.first_pitcher_role,
        h.opp_hand_pitches_365d,
        h.opp_hand_pa_365d,
        h.opp_hand_bbe_365d,
        h.opp_hand_hr_365d,
        h.opp_hand_k_365d,
        h.opp_hand_gb_365d,
        h.opp_hand_barrel_365d,
        h.opp_hand_hr_per_bbe_365d_raw,
        h.opp_hand_hr_per_pa_365d_raw,
        h.opp_hand_k_per_pa_365d_raw,
        h.opp_hand_gb_per_bbe_365d_raw,
        h.opp_hand_barrel_per_bbe_365d_raw,
        h.lg_hand_hr_per_bbe_365d,
        h.lg_hand_hr_per_pa_365d,
        h.lg_hand_k_per_pa_365d,
        h.lg_hand_gb_per_bbe_365d,
        h.lg_hand_barrel_per_bbe_365d,
        h.opp_hand_hr_per_bbe_365d_shrunk,
        h.opp_hand_hr_per_pa_365d_shrunk,
        h.opp_hand_k_per_pa_365d_shrunk,
        h.opp_hand_gb_per_bbe_365d_shrunk,
        h.opp_hand_barrel_per_bbe_365d_shrunk,
        h.log_opp_hand_bbe_365d,
        h.log_opp_hand_pa_365d
    FROM hr_model_dataset_view v
    LEFT JOIN pitcher_hand_features_a h
      ON CAST(v.game_id AS TEXT)=h.game_id
     AND v.batter_id=h.batter_id
    WHERE v.actual_hr IS NOT NULL
    ORDER BY v.game_date, v.game_id, v.batter_id
    """, conn)

    df["game_date"] = pd.to_datetime(df["game_date"].astype(str), errors="coerce")
    return df


def pd_to_numeric(s):
    import pandas as pd
    return pd.to_numeric(s, errors="coerce")


def maybe_add_weather_hinges(df):
    if "temp_over_75" not in df.columns:
        temp_col = None
        for c in ["temperature", "temp", "game_temp", "weather_temp"]:
            if c in df.columns:
                temp_col = c
                break
        if temp_col:
            temp = pd_to_numeric(df[temp_col])
            df["temp_over_75"] = (temp - 75).clip(lower=0)
            df["temp_over_85"] = (temp - 85).clip(lower=0)

    if "wind_out_component" not in df.columns:
        df["wind_out_component"] = 0.0
    if "wind_out_and_hot" not in df.columns:
        if "temp_over_75" in df.columns and "wind_out_component" in df.columns:
            df["wind_out_and_hot"] = df["temp_over_75"].fillna(0) * df["wind_out_component"].fillna(0)
        else:
            df["wind_out_and_hot"] = 0.0
    return df


def pick_base_features(df):
    cols = set(df.columns)
    selected = [c for c in PREFERRED_BASE_FEATURES if c in cols]

    if len(selected) < 8:
        exclude_exact = {
            "actual_hr", "game_id", "game_pk", "batter_id", "pitcher_id",
            "first_pitcher_id", "game_date", "batter_name", "player_name",
            "team", "opponent", "home_team", "away_team",
            "pa", "actual_pa", "plate_appearances", "ab", "h", "hits",
            "home_runs", "hr", "rbi", "runs", "total_bases",
        }
        exclude_substrings = [
            "actual_", "post_", "future_", "result", "outcome",
            "same_game", "after_", "label",
        ]
        selected = []
        for c in df.columns:
            lc = c.lower()
            if lc in exclude_exact:
                continue
            if any(x in lc for x in exclude_substrings):
                continue
            if c in HAND_FEATURES_H3:
                continue
            if c in ["first_pitcher_role", "batter_stand", "p_throws"]:
                continue
            try:
                vals = pd_to_numeric(df[c])
                if vals.notna().mean() > 0.70:
                    selected.append(c)
            except Exception:
                pass

    return selected


def metrics_for_predictions(y, p, brier_score_loss, log_loss, roc_auc_score):
    eps = 1e-6
    p = [min(max(float(x), eps), 1 - eps) for x in p]
    out = {}
    out["brier"] = float(brier_score_loss(y, p))
    out["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except Exception:
        out["auc"] = float("nan")

    n = len(y)
    k = max(1, int(math.ceil(n * 0.05)))
    order = sorted(range(n), key=lambda i: p[i], reverse=True)[:k]
    top_y = [y.iloc[i] if hasattr(y, "iloc") else y[i] for i in order]
    top_p = [p[i] for i in order]
    out["top5_rows"] = k
    out["top5_mean_pred"] = float(sum(top_p) / len(top_p))
    out["top5_actual"] = float(sum(top_y) / len(top_y))
    out["top5_hits"] = int(sum(top_y))
    out["n"] = n
    out["actual_rate"] = float(sum(y) / len(y))
    return out


def run_model(np, pd, df, features, cat_features, label, ColumnTransformer, SimpleImputer, LogisticRegression, brier_score_loss, log_loss, roc_auc_score, Pipeline, OneHotEncoder, StandardScaler, eval_mask=None):
    d = df.copy()
    for f in features:
        d[f] = pd.to_numeric(d[f], errors="coerce")

    train = d[d["game_date"].dt.year == 2025].copy()
    test = d[d["game_date"].dt.year == 2026].copy()
    if eval_mask is not None:
        test = test[eval_mask.loc[test.index]].copy()

    X_train = train[features + cat_features]
    y_train = train["actual_hr"].astype(int)
    X_test = test[features + cat_features]
    y_test = test["actual_hr"].astype(int)

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    if cat_features:
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        pre = ColumnTransformer([
            ("num", numeric_transformer, features),
            ("cat", categorical_transformer, cat_features),
        ])
    else:
        pre = ColumnTransformer([
            ("num", numeric_transformer, features),
        ])

    model = Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(C=0.03, penalty="l2", solver="lbfgs", max_iter=3000)),
    ])

    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    m = metrics_for_predictions(y_test.reset_index(drop=True), list(p), brier_score_loss, log_loss, roc_auc_score)
    m["label"] = label
    m["train_n"] = int(len(train))
    m["test_n"] = int(len(test))
    m["features_n"] = int(len(features) + len(cat_features))
    return model, m


def coefficient_report(model, features, cat_features, top_n=25):
    try:
        pre = model.named_steps["pre"]
        names = []
        if features:
            names.extend(features)
        if cat_features:
            cat_pipe = pre.named_transformers_["cat"]
            onehot = cat_pipe.named_steps["onehot"]
            names.extend(list(onehot.get_feature_names_out(cat_features)))
        coefs = model.named_steps["lr"].coef_[0]
        rows = list(zip(names, coefs))
        rows = sorted(rows, key=lambda x: abs(x[1]), reverse=True)
        return rows[:top_n]
    except Exception as e:
        return [("coefficient_report_error", repr(e))]


def print_metrics(title, metrics):
    print("\n" + title)
    print("-" * len(title))
    for m in metrics:
        print(
            f"{m['label']}: "
            f"train_n={m['train_n']} test_n={m['test_n']} features={m['features_n']} "
            f"brier={m['brier']:.8f} logloss={m['logloss']:.8f} auc={m['auc']:.8f} "
            f"top5_actual={m['top5_actual']:.5f} top5_pred={m['top5_mean_pred']:.5f} "
            f"top5_hits={m['top5_hits']}/{m['top5_rows']} actual_rate={m['actual_rate']:.5f}"
        )


def print_deltas(title, base, others):
    print("\n" + title)
    print("-" * len(title))
    for m in others:
        print(
            f"{m['label']} vs {base['label']}: "
            f"brier_delta={m['brier']-base['brier']:+.8f} "
            f"logloss_delta={m['logloss']-base['logloss']:+.8f} "
            f"auc_delta={m['auc']-base['auc']:+.8f} "
            f"top5_actual_delta={m['top5_actual']-base['top5_actual']:+.5f}"
        )


def decile_report(pd, df, feature, sample_col, threshold):
    d = df[
        (df["game_date"].dt.year == 2026)
        & df[feature].notna()
        & (pd.to_numeric(df[sample_col], errors="coerce").fillna(0) >= threshold)
    ].copy()

    if len(d) < 200:
        print(f"{feature} min_{sample_col}={threshold}: not enough rows n={len(d)}")
        return

    d["decile"] = pd.qcut(d[feature], 10, labels=False, duplicates="drop") + 1
    print(f"\nHAND HR/BBE DECILES: {feature} min_{sample_col}={threshold}")
    print("-" * 84)
    for dec, g in d.groupby("decile"):
        print(
            f"decile={int(dec):02d} n={len(g):5d} "
            f"mean={g[feature].mean():.5f} min={g[feature].min():.5f} max={g[feature].max():.5f} "
            f"actual_hr={g['actual_hr'].mean():.5f} hits={int(g['actual_hr'].sum())}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=db_path())
    ap.add_argument("--rebuild-features", action="store_true")
    args = ap.parse_args()

    (
        np, pd, ColumnTransformer, SimpleImputer, LogisticRegression,
        brier_score_loss, log_loss, roc_auc_score, Pipeline,
        OneHotEncoder, StandardScaler
    ) = imports()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")
    print("HR_HAND_PITCHER_VECTOR_A")
    print("========================")
    print("db:", args.db)

    if args.rebuild_features or not exists(conn, "pitcher_hand_features_a"):
        build_features(conn)
    else:
        print("using existing pitcher_hand_features_a")

    print("loading dataset...")
    df = load_dataset(pd, conn)
    conn.close()

    df = maybe_add_weather_hinges(df)
    for c in ["actual_hr"] + HAND_FEATURES_H3 + ["opp_hand_bbe_365d", "opp_hand_pa_365d"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print("\nHAND FEATURE COVERAGE")
    print("---------------------")
    print("rows:", len(df))
    print("date_range:", df["game_date"].min(), df["game_date"].max())
    for year, g in df.groupby(df["game_date"].dt.year):
        print(
            f"year={int(year)} rows={len(g)} "
            f"mapped={int(g['first_pitcher_id'].notna().sum())} "
            f"zero_hand_bbe={int((g['opp_hand_bbe_365d'].fillna(0)<=0).sum())} "
            f"median_hand_bbe={g['opp_hand_bbe_365d'].median():.1f} "
            f"actual_hr={g['actual_hr'].mean():.5f}"
        )

    for th in [100, 250, 500]:
        decile_report(pd, df, "opp_hand_hr_per_bbe_365d_raw", "opp_hand_bbe_365d", th)
    for th in [100, 250, 500]:
        decile_report(pd, df, "opp_hand_hr_per_bbe_365d_shrunk", "opp_hand_bbe_365d", th)

    base_features = pick_base_features(df)
    missing_pref = [c for c in PREFERRED_BASE_FEATURES if c not in df.columns]

    cat_features = []
    for c in ["batter_stand", "p_throws"]:
        if c in df.columns:
            cat_features.append(c)

    print("\nBASE FEATURE SET")
    print("----------------")
    print("base_features_n:", len(base_features))
    print("cat_features:", cat_features)
    print("base_features:", ", ".join(base_features))
    if missing_pref:
        print("missing_preferred_skipped:", ", ".join(missing_pref))

    models_all = []
    fitted = {}

    model, m = run_model(np, pd, df, base_features, cat_features, "A0_ENVIRONMENT_C_LOCKED_PROXY", ColumnTransformer, SimpleImputer, LogisticRegression, brier_score_loss, log_loss, roc_auc_score, Pipeline, OneHotEncoder, StandardScaler)
    models_all.append(m); fitted[m["label"]] = model

    model, m = run_model(np, pd, df, base_features + HAND_FEATURES_H1, cat_features, "H1_HAND_HR_BBE_ONLY", ColumnTransformer, SimpleImputer, LogisticRegression, brier_score_loss, log_loss, roc_auc_score, Pipeline, OneHotEncoder, StandardScaler)
    models_all.append(m); fitted[m["label"]] = model

    model, m = run_model(np, pd, df, base_features + HAND_FEATURES_H2, cat_features, "H2_HAND_PACKAGE", ColumnTransformer, SimpleImputer, LogisticRegression, brier_score_loss, log_loss, roc_auc_score, Pipeline, OneHotEncoder, StandardScaler)
    models_all.append(m); fitted[m["label"]] = model

    model, m = run_model(np, pd, df, base_features + HAND_FEATURES_H3, cat_features, "H3_HAND_PACKAGE_PLUS_SAMPLE", ColumnTransformer, SimpleImputer, LogisticRegression, brier_score_loss, log_loss, roc_auc_score, Pipeline, OneHotEncoder, StandardScaler)
    models_all.append(m); fitted[m["label"]] = model

    print_metrics("SUMMARY COMPARISON - ALL 2026", models_all)
    print_deltas("DELTAS VS A0 - ALL 2026", models_all[0], models_all[1:])

    trad_mask = df["first_pitcher_role"].fillna("") == "traditional"
    models_trad = []
    for label, feats in [
        ("A0_TRADITIONAL_EVAL", base_features),
        ("H1_TRADITIONAL_EVAL", base_features + HAND_FEATURES_H1),
        ("H2_TRADITIONAL_EVAL", base_features + HAND_FEATURES_H2),
        ("H3_TRADITIONAL_EVAL", base_features + HAND_FEATURES_H3),
    ]:
        model, m = run_model(np, pd, df, feats, cat_features, label, ColumnTransformer, SimpleImputer, LogisticRegression, brier_score_loss, log_loss, roc_auc_score, Pipeline, OneHotEncoder, StandardScaler, eval_mask=trad_mask)
        models_trad.append(m)
    print_metrics("SUMMARY COMPARISON - TRADITIONAL STARTER EVAL 2026", models_trad)
    print_deltas("DELTAS VS A0 - TRADITIONAL STARTER EVAL 2026", models_trad[0], models_trad[1:])

    print("\nTOP COEFFICIENTS - H2_HAND_PACKAGE")
    print("----------------------------------")
    for name, coef in coefficient_report(fitted["H2_HAND_PACKAGE"], base_features + HAND_FEATURES_H2, cat_features, 35):
        print(f"{name}: {coef:+.6f}")

    print("\nHAND FEATURE COEFFICIENTS - H2_HAND_PACKAGE")
    print("-------------------------------------------")
    coef_rows = coefficient_report(fitted["H2_HAND_PACKAGE"], base_features + HAND_FEATURES_H2, cat_features, 1000)
    coef_map = {n: c for n, c in coef_rows}
    for f in HAND_FEATURES_H2:
        print(f"{f}: {coef_map.get(f, float('nan')):+.6f}")

    print("\nREAD")
    print("----")
    print("Gate is Brier-led, not Brier-only.")
    print("Target unlock: roughly -0.0008 to -0.0010 Brier improvement with no unacceptable logloss/AUC/top-bucket damage.")
    print("If H1/H2 improves deciles but not model holdout, pitcher signal is real but not yet production-grade.")
    print("If traditional-starter eval clears but all-row does not, opener/bulk handling remains a production issue.")
    print("Bayesian pitch-zone overlap remains blocked until the cleaned handedness vector earns promotion.")


if __name__ == "__main__":
    main()
