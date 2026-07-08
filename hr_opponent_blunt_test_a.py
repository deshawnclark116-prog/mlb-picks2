#!/usr/bin/env python3
"""
HR_OPPONENT_BLUNT_TEST_A

Blunt opponent baseline test before pitch-type/zone overlap.

Compares:
A = ENVIRONMENT_C_LOCKED_A
B = ENVIRONMENT_C_LOCKED_A + simple opposing starting pitcher features

Train: 2025 only
Test:  2026 only

Run after 2025 pitch-level load finishes:
    python hr_opponent_blunt_test_a.py --rebuild-features
"""

import argparse
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

LOCKED_C_FEATURES = BASE_FEATURES + LEAGUE_FEATURES + WEATHER_FEATURES

OPP_RATE_TO_SAMPLE = {
    "opp_k_per_pa_365d": "opp_pa_365d",
    "opp_hr_per_pa_365d": "opp_pa_365d",
    "opp_hr_per_bbe_365d": "opp_bbe_365d",
    "opp_barrel_per_bbe_365d": "opp_bbe_365d",
    "opp_hardhit_per_bbe_365d": "opp_bbe_365d",
    "opp_gb_per_bbe_365d": "opp_bbe_365d",
    "opp_air_per_bbe_365d": "opp_bbe_365d",
    "opp_fb_usage_365d": "opp_pitches_365d",
    "opp_br_usage_365d": "opp_pitches_365d",
    "opp_os_usage_365d": "opp_pitches_365d",
    "opp_whiff_per_pitch_365d": "opp_pitches_365d",
    "opp_csw_per_pitch_365d": "opp_pitches_365d",
}

OPP_FEATURES = [
    "log_opp_pitches_365d",
    "log_opp_pa_365d",
    "log_opp_bbe_365d",
] + [f"{c}_shrunk" for c in OPP_RATE_TO_SAMPLE]

CONFIGS = {
    "A_ENVIRONMENT_C_LOCKED": LOCKED_C_FEATURES,
    "B_PLUS_OPPONENT_BLUNT": LOCKED_C_FEATURES + OPP_FEATURES,
}

RAW_COLS = [
    "expected_pa_v1", "temp_f", "wind_toward_pull_field", "weather_wind_mph",
    "batter_hr_per_air_60d", "batter_fly_ball_rate_30d", "batter_barrel_rate_30d",
    "batter_max_ev_60d", "pitcher_pull_air_allowed_rate_60d",
    "pitcher_hard_hit_allowed_rate_7d", "pitcher_hrfb_rate_30d", "batter_bbe_60d",
    "pitcher_bbe_allowed_60d", "league_hr_per_bbe_10d_lag", "league_hr_per_air_10d_lag",
    "league_barrel_rate_10d_lag", "league_avg_ev_10d_lag", "league_bbe_10d", "league_air_10d",
] + list(OPP_RATE_TO_SAMPLE.keys()) + ["opp_pitches_365d", "opp_pa_365d", "opp_bbe_365d"]


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
        print("Missing dependency. Install with: pip install pandas numpy scikit-learn")
        raise SystemExit(f"{type(e).__name__}: {e}")


def table_cols(conn, table):
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    except Exception:
        return []


def build_opponent_features(conn, rebuild=False):
    if "pitcher_id" not in set(table_cols(conn, "batter_games")):
        raise SystemExit("batter_games.pitcher_id not found. Need opposing starting pitcher id first.")

    if rebuild:
        with conn:
            conn.execute("DROP TABLE IF EXISTS pitcher_daily_blunt_stats")
            conn.execute("DROP TABLE IF EXISTS pitcher_blunt_game_features")

    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS pitcher_daily_blunt_stats AS
        SELECT
            game_date,
            pitcher_id,
            COUNT(*) AS pitches,
            SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END) AS pa_events,
            SUM(CASE WHEN events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS k_events,
            SUM(CASE WHEN pitch_group='FB' THEN 1 ELSE 0 END) AS fb_pitches,
            SUM(CASE WHEN pitch_group='BR' THEN 1 ELSE 0 END) AS br_pitches,
            SUM(CASE WHEN pitch_group='OS' THEN 1 ELSE 0 END) AS os_pitches,
            SUM(CASE WHEN is_whiff=1 THEN 1 ELSE 0 END) AS whiffs,
            SUM(CASE WHEN is_called_strike=1 THEN 1 ELSE 0 END) AS called_strikes,
            SUM(CASE WHEN is_bbe=1 THEN 1 ELSE 0 END) AS bbe,
            SUM(CASE WHEN is_hr=1 THEN 1 ELSE 0 END) AS hr,
            SUM(CASE WHEN is_barrel=1 THEN 1 ELSE 0 END) AS barrels,
            SUM(CASE WHEN launch_speed >= 95 THEN 1 ELSE 0 END) AS hard_hit_bbe,
            SUM(CASE WHEN bb_type='ground_ball' THEN 1 ELSE 0 END) AS gb,
            SUM(CASE WHEN bb_type IN ('fly_ball','line_drive','popup') THEN 1 ELSE 0 END) AS air
        FROM statcast_pitches
        WHERE pitcher_id IS NOT NULL AND game_date IS NOT NULL
        GROUP BY game_date, pitcher_id
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pitcher_daily_blunt_pd ON pitcher_daily_blunt_stats(pitcher_id, game_date)")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS pitcher_blunt_game_features (
            game_id TEXT,
            batter_id INTEGER,
            game_date TEXT,
            pitcher_id INTEGER,
            opp_pitches_365d REAL,
            opp_pa_365d REAL,
            opp_bbe_365d REAL,
            opp_k_per_pa_365d REAL,
            opp_hr_per_pa_365d REAL,
            opp_hr_per_bbe_365d REAL,
            opp_barrel_per_bbe_365d REAL,
            opp_hardhit_per_bbe_365d REAL,
            opp_gb_per_bbe_365d REAL,
            opp_air_per_bbe_365d REAL,
            opp_fb_usage_365d REAL,
            opp_br_usage_365d REAL,
            opp_os_usage_365d REAL,
            opp_whiff_per_pitch_365d REAL,
            opp_csw_per_pitch_365d REAL,
            PRIMARY KEY (game_id, batter_id)
        )
        """)
        conn.execute("DELETE FROM pitcher_blunt_game_features")
        conn.execute("""
        INSERT OR REPLACE INTO pitcher_blunt_game_features
        SELECT
            bg.game_id,
            bg.batter_id,
            bg.game_date,
            bg.pitcher_id,
            COALESCE(SUM(d.pitches), 0),
            COALESCE(SUM(d.pa_events), 0),
            COALESCE(SUM(d.bbe), 0),
            CASE WHEN SUM(d.pa_events) > 0 THEN 1.0 * SUM(d.k_events) / SUM(d.pa_events) END,
            CASE WHEN SUM(d.pa_events) > 0 THEN 1.0 * SUM(d.hr) / SUM(d.pa_events) END,
            CASE WHEN SUM(d.bbe) > 0 THEN 1.0 * SUM(d.hr) / SUM(d.bbe) END,
            CASE WHEN SUM(d.bbe) > 0 THEN 1.0 * SUM(d.barrels) / SUM(d.bbe) END,
            CASE WHEN SUM(d.bbe) > 0 THEN 1.0 * SUM(d.hard_hit_bbe) / SUM(d.bbe) END,
            CASE WHEN SUM(d.bbe) > 0 THEN 1.0 * SUM(d.gb) / SUM(d.bbe) END,
            CASE WHEN SUM(d.bbe) > 0 THEN 1.0 * SUM(d.air) / SUM(d.bbe) END,
            CASE WHEN SUM(d.pitches) > 0 THEN 1.0 * SUM(d.fb_pitches) / SUM(d.pitches) END,
            CASE WHEN SUM(d.pitches) > 0 THEN 1.0 * SUM(d.br_pitches) / SUM(d.pitches) END,
            CASE WHEN SUM(d.pitches) > 0 THEN 1.0 * SUM(d.os_pitches) / SUM(d.pitches) END,
            CASE WHEN SUM(d.pitches) > 0 THEN 1.0 * SUM(d.whiffs) / SUM(d.pitches) END,
            CASE WHEN SUM(d.pitches) > 0 THEN 1.0 * (SUM(d.whiffs) + SUM(d.called_strikes)) / SUM(d.pitches) END
        FROM batter_games bg
        LEFT JOIN pitcher_daily_blunt_stats d
          ON d.pitcher_id = bg.pitcher_id
         AND d.game_date < bg.game_date
         AND d.game_date >= DATE(bg.game_date, '-365 day')
        WHERE bg.actual_hr IS NOT NULL
        GROUP BY bg.game_id, bg.batter_id, bg.game_date, bg.pitcher_id
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pitcher_blunt_game ON pitcher_blunt_game_features(game_id, batter_id)")


def expr_from_tables(conn, alias, candidates):
    bg = set(table_cols(conn, "batter_games"))
    bf = set(table_cols(conn, "batter_game_features"))
    pf = set(table_cols(conn, "pitcher_game_features"))
    le = set(table_cols(conn, "league_env_lag_features"))
    ob = set(table_cols(conn, "pitcher_blunt_game_features"))
    for c in candidates:
        if c in bg: return f"bg.{c} AS {alias}"
        if c in bf: return f"bf.{c} AS {alias}"
        if c in pf: return f"pf.{c} AS {alias}"
        if c in le: return f"le.{c} AS {alias}"
        if c in ob: return f"ob.{c} AS {alias}"
    return f"NULL AS {alias}"


def load_dataset(conn, pd):
    cmap = {
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
    for c in OPP_RATE_TO_SAMPLE: cmap[c] = [c]
    for c in ["opp_pitches_365d", "opp_pa_365d", "opp_bbe_365d"]: cmap[c] = [c]

    bg = set(table_cols(conn, "batter_games"))
    def opt_bg(col): return f"bg.{col} AS {col}" if col in bg else f"NULL AS {col}"

    select = [
        "bg.game_date AS game_date", "bg.game_id AS game_id", "bg.batter_id AS batter_id",
        "bg.pitcher_id AS pitcher_id", "bg.batter_name AS batter_name",
        opt_bg("team"), opt_bg("opponent"), opt_bg("venue"),
        "bg.lineup_spot AS lineup_spot", "bg.actual_hr AS actual_hr",
    ]
    for alias, cand in cmap.items(): select.append(expr_from_tables(conn, alias, cand))

    sql = f"""
    SELECT {', '.join(select)}
    FROM batter_games bg
    LEFT JOIN batter_game_features bf ON bg.game_id = bf.game_id AND bg.batter_id = bf.batter_id
    LEFT JOIN pitcher_game_features pf ON bg.game_id = pf.game_id AND bg.batter_id = pf.batter_id
    LEFT JOIN league_env_lag_features le ON bg.game_date = le.game_date
    LEFT JOIN pitcher_blunt_game_features ob ON bg.game_id = ob.game_id AND bg.batter_id = ob.batter_id
    WHERE bg.actual_hr IS NOT NULL
    ORDER BY bg.game_date, bg.game_id, bg.lineup_spot
    """
    return pd.read_sql_query(sql, conn)


def boolish_to_float(v):
    if v is None: return 0.0
    if isinstance(v, (int, float)):
        try: return 1.0 if float(v) > 0 else 0.0
        except Exception: return 0.0
    return 1.0 if str(v).strip().lower() in ("1","true","t","yes","y","out","toward","pull","wind_out","wind toward pull") else 0.0


def shrink_rate_pair(train, test, rate_col, sample_col, out_col, shrink_k):
    prior = float(train[rate_col].dropna().median()) if train[rate_col].notna().sum() else 0.0
    for df in (train, test):
        raw = df[rate_col].fillna(prior).clip(lower=0, upper=1)
        n = df[sample_col].fillna(0).clip(lower=0)
        w = n / (n + shrink_k)
        df[out_col] = prior + (raw - prior) * w


def add_engineered_features(train, test, batter_shrink_k, opp_shrink_k):
    for df in (train, test):
        df["log_batter_bbe_60d"] = df["batter_bbe_60d"].fillna(0).clip(lower=0).map(lambda x: math.log1p(x))
        df["log_pitcher_bbe_allowed_60d"] = df["pitcher_bbe_allowed_60d"].fillna(0).clip(lower=0).map(lambda x: math.log1p(x))
        df["log_league_bbe_10d"] = df["league_bbe_10d"].fillna(0).clip(lower=0).map(lambda x: math.log1p(x))
        df["temp_over_75"] = (df["temp_f"].fillna(70) - 75).clip(lower=0)
        df["temp_over_85"] = (df["temp_f"].fillna(70) - 85).clip(lower=0)
        wind_flag = df["wind_toward_pull_field"].map(boolish_to_float)
        wind_mph = df["weather_wind_mph"].fillna(0).clip(lower=0, upper=60)
        df["wind_out_component"] = wind_flag * wind_mph
        df["wind_out_and_hot"] = df["wind_out_component"] * df["temp_over_75"]
        df["log_opp_pitches_365d"] = df["opp_pitches_365d"].fillna(0).clip(lower=0).map(lambda x: math.log1p(x))
        df["log_opp_pa_365d"] = df["opp_pa_365d"].fillna(0).clip(lower=0).map(lambda x: math.log1p(x))
        df["log_opp_bbe_365d"] = df["opp_bbe_365d"].fillna(0).clip(lower=0).map(lambda x: math.log1p(x))
    for rate_col, sample_col in RATE_TO_SAMPLE.items():
        shrink_rate_pair(train, test, rate_col, sample_col, f"{rate_col}_shrunk", batter_shrink_k)
    for rate_col, sample_col in OPP_RATE_TO_SAMPLE.items():
        shrink_rate_pair(train, test, rate_col, sample_col, f"{rate_col}_shrunk", opp_shrink_k)


def make_model(C, features, ColumnTransformer, SimpleImputer, LogisticRegression, Pipeline, StandardScaler):
    pre = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)], remainder="drop")
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", C=C))])


def safe_metric(fn, y, p):
    try: return float(fn(y, p))
    except Exception: return None

def fmt(x): return "None" if x is None else f"{float(x):.8f}"

def top5(df, col):
    cutoff = df[col].quantile(0.95)
    t = df[df[col] >= cutoff]
    return cutoff, len(t), float(t[col].mean()), float(t["actual_hr"].mean()), int(t["actual_hr"].sum())


def evaluate(name, features, train, test, args, ColumnTransformer, SimpleImputer, LogisticRegression, brier, logloss, auc, Pipeline, StandardScaler):
    model = make_model(args.c, features, ColumnTransformer, SimpleImputer, LogisticRegression, Pipeline, StandardScaler)
    model.fit(train[features], train["actual_hr"].astype(int))
    pred = model.predict_proba(test[features])[:, 1]
    out = test.copy(); out["model_prob"] = pred
    y = out["actual_hr"].astype(int).tolist()
    base = float(sum(y) / len(y)); bp = [base] * len(y)
    cutoff, n, mp, act, hits = top5(out, "model_prob")
    return {"name": name, "features_n": len(features), "baseline_brier": safe_metric(brier,y,bp), "model_brier": safe_metric(brier,y,pred.tolist()),
            "baseline_logloss": safe_metric(logloss,y,bp), "model_logloss": safe_metric(logloss,y,pred.tolist()), "model_auc": safe_metric(auc,y,pred.tolist()),
            "top5_cutoff": cutoff, "top5_rows": n, "top5_mean_pred": mp, "top5_actual": act, "top5_hits": hits, "model": model, "out": out, "features": features}


def monthly(out):
    d = out.copy(); d["month"] = d["game_date"].dt.strftime("%Y-%m")
    lines=[]
    for m,g in d.groupby("month"):
        lines.append(f"{m} rows={len(g):5d} actual={g['actual_hr'].mean():.4f} pred={g['model_prob'].mean():.4f} miss={g['model_prob'].mean()-g['actual_hr'].mean():+.4f}")
    return lines


def coef_report(model, features):
    try:
        coefs = model.named_steps["clf"].coef_[0]
        return sorted(zip(features, coefs), key=lambda x: abs(x[1]), reverse=True)
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--c", type=float, default=0.03)
    ap.add_argument("--batter-shrink-k", type=float, default=75.0)
    ap.add_argument("--opp-shrink-k", type=float, default=200.0)
    ap.add_argument("--rebuild-features", action="store_true")
    args = ap.parse_args()

    pd, ColumnTransformer, SimpleImputer, LogisticRegression, brier, logloss, auc, Pipeline, StandardScaler = require_imports()
    conn = sqlite3.connect(args.db); conn.execute("PRAGMA journal_mode=WAL")
    print("HR_OPPONENT_BLUNT_TEST_A"); print("========================"); print("db:", args.db)
    print("Building/checking opponent blunt features...")
    build_opponent_features(conn, rebuild=args.rebuild_features)
    print("pitcher_blunt_game_features:", conn.execute("SELECT COUNT(*), MIN(game_date), MAX(game_date) FROM pitcher_blunt_game_features").fetchone())
    df = load_dataset(conn, pd); conn.close()

    df["game_date"] = pd.to_datetime(df["game_date"]); df["actual_hr"] = df["actual_hr"].astype(int)
    for c in RAW_COLS:
        if c != "wind_toward_pull_field": df[c] = pd.to_numeric(df[c], errors="coerce")
    train = df[df["game_date"].dt.year == 2025].copy().sort_values("game_date")
    test = df[df["game_date"].dt.year == 2026].copy().sort_values("game_date")
    add_engineered_features(train, test, args.batter_shrink_k, args.opp_shrink_k)

    print("\nDATA"); print("----")
    print("train_rows_2025:", len(train)); print("test_rows_2026:", len(test)); print("test_actual_hr_rate:", f"{test['actual_hr'].mean():.5f}")
    print("2026 opponent missing pitch rows:", int((test["opp_pitches_365d"].fillna(0) <= 0).sum()))
    print("2026 opp_pitches_365d median/p75/p95:", f"{test['opp_pitches_365d'].median():.1f}", f"{test['opp_pitches_365d'].quantile(.75):.1f}", f"{test['opp_pitches_365d'].quantile(.95):.1f}")

    results=[]
    for name, features in CONFIGS.items():
        print(f"\nRUNNING {name}"); print("-"*(8+len(name)))
        r = evaluate(name, features, train, test, args, ColumnTransformer, SimpleImputer, LogisticRegression, brier, logloss, auc, Pipeline, StandardScaler)
        results.append(r)
        print("features_n:", r["features_n"])
        print("model_brier:", fmt(r["model_brier"]), "baseline:", fmt(r["baseline_brier"]))
        print("model_logloss:", fmt(r["model_logloss"]), "baseline:", fmt(r["baseline_logloss"]))
        print("model_auc:", fmt(r["model_auc"]))
        print("top5:", f"cutoff={r['top5_cutoff']:.5f}", f"rows={r['top5_rows']}", f"mean_pred={r['top5_mean_pred']:.5f}", f"actual={r['top5_actual']:.5f}", f"hits={r['top5_hits']}")

    print("\nSUMMARY COMPARISON"); print("==================")
    print("name | brier | logloss | auc | top5_actual | top5_hits")
    for r in results:
        print(f"{r['name']} | {fmt(r['model_brier'])} | {fmt(r['model_logloss'])} | {fmt(r['model_auc'])} | {r['top5_actual']:.5f} | {r['top5_hits']}")

    a,b = results[0], results[1]
    print("\nDELTA B vs A"); print("------------")
    print("brier_delta_B_minus_A:", f"{b['model_brier']-a['model_brier']:+.8f}")
    print("logloss_delta_B_minus_A:", f"{b['model_logloss']-a['model_logloss']:+.8f}")
    print("auc_delta_B_minus_A:", f"{b['model_auc']-a['model_auc']:+.8f}")
    print("top5_actual_delta_B_minus_A:", f"{b['top5_actual']-a['top5_actual']:+.5f}")

    for r in results:
        print(f"\nMONTHLY {r['name']}"); print("----------------" + "-"*len(r["name"]))
        for line in monthly(r["out"]): print(line)

    print("\nTOP STANDARDIZED COEFFICIENTS: B_PLUS_OPPONENT_BLUNT"); print("----------------------------------------------------")
    for name, coef in coef_report(results[1]["model"], results[1]["features"])[:30]: print(f"{name}: {coef:+.5f}")

    print("\nREAD"); print("----")
    print("This is the blunt opponent test. If B beats A, pitcher/opponent identity earns the next build slot: pitch-type/zone overlap.")

if __name__ == "__main__": main()
