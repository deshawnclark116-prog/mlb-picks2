#!/usr/bin/env python3
"""
HR_OPPONENT_RAW_AUDIT_B

Strict diagnostic audit for the blunt pitcher gate.

This does NOT build production features.

It checks:
1. Raw trailing 365d pitcher HR/BBE vs hitter actual HR.
2. Thresholds: min 100 / 250 / 500 prior BBE.
3. First-pitcher map vs diagnostic most-pitches pitcher map.
4. Traditional starter / opener / ambiguous split.
5. Batter-pitcher handedness splits.

Run:
    python hr_opponent_raw_audit_b.py --rebuild
"""

import argparse, os, sqlite3
from pathlib import Path

def db_path():
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    return p / "hr_model.sqlite" if p.parent.exists() else Path("./hr_model/hr_model.sqlite")

def imports():
    try:
        import pandas as pd
        import numpy as np
        return pd, np
    except Exception as e:
        raise SystemExit("Install deps: pip install pandas numpy\n" + repr(e))

def have_table(conn, t):
    return conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone() is not None

def build_daily_if_missing(conn):
    if have_table(conn, "pitcher_daily_gate_stats"):
        return
    with conn:
        conn.execute("""
        CREATE TABLE pitcher_daily_gate_stats AS
        SELECT game_date,pitcher_id,
               COUNT(*) pitches,
               SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END) pa,
               SUM(CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) k,
               SUM(CASE WHEN pitch_group='FB' THEN 1 ELSE 0 END) fb,
               SUM(CASE WHEN pitch_group='BR' THEN 1 ELSE 0 END) br,
               SUM(CASE WHEN pitch_group='OS' THEN 1 ELSE 0 END) os,
               SUM(CASE WHEN is_whiff=1 THEN 1 ELSE 0 END) whiff,
               SUM(CASE WHEN is_called_strike=1 THEN 1 ELSE 0 END) called,
               SUM(CASE WHEN is_bbe=1 THEN 1 ELSE 0 END) bbe,
               SUM(CASE WHEN is_hr=1 THEN 1 ELSE 0 END) hr,
               SUM(CASE WHEN is_barrel=1 THEN 1 ELSE 0 END) barrel,
               SUM(CASE WHEN launch_speed>=95 THEN 1 ELSE 0 END) hardhit,
               SUM(CASE WHEN bb_type='ground_ball' THEN 1 ELSE 0 END) gb,
               SUM(CASE WHEN bb_type IN ('fly_ball','line_drive','popup') THEN 1 ELSE 0 END) air
        FROM statcast_pitches
        WHERE pitcher_id IS NOT NULL AND game_date IS NOT NULL
        GROUP BY game_date,pitcher_id
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pdg_pd ON pitcher_daily_gate_stats(pitcher_id,game_date)")

def build_audit_tables(conn, rebuild=False):
    build_daily_if_missing(conn)
    if rebuild:
        with conn:
            for t in [
                "game_pitcher_workload_diag",
                "first_pitcher_map_diag",
                "most_pitcher_map_diag",
                "opponent_raw_audit_base",
                "opponent_raw_first_roll",
                "opponent_raw_most_roll",
                "opponent_raw_audit_features",
            ]:
                conn.execute(f"DROP TABLE IF EXISTS {t}")

    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS game_pitcher_workload_diag AS
        SELECT
            CAST(game_pk AS TEXT) AS game_id,
            game_date,
            pitcher_id,
            COUNT(*) AS game_pitches,
            COUNT(DISTINCT at_bat_number) AS game_batters_approx
        FROM statcast_pitches
        WHERE game_pk IS NOT NULL
          AND pitcher_id IS NOT NULL
          AND game_date IS NOT NULL
        GROUP BY CAST(game_pk AS TEXT), game_date, pitcher_id
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gpwl_game_pitcher ON game_pitcher_workload_diag(game_id,pitcher_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gpwl_game ON game_pitcher_workload_diag(game_id)")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS first_pitcher_map_diag AS
        WITH ranked AS (
            SELECT
                CAST(game_pk AS TEXT) AS game_id,
                game_date,
                batter_id,
                pitcher_id,
                stand,
                p_throws,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(game_pk AS TEXT), batter_id
                    ORDER BY COALESCE(at_bat_number,9999), COALESCE(pitch_number,9999)
                ) rn
            FROM statcast_pitches
            WHERE game_pk IS NOT NULL
              AND batter_id IS NOT NULL
              AND pitcher_id IS NOT NULL
              AND game_date IS NOT NULL
        )
        SELECT game_id, game_date, batter_id, pitcher_id AS first_pitcher_id, stand, p_throws
        FROM ranked
        WHERE rn=1
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fpm_game_batter ON first_pitcher_map_diag(game_id,batter_id)")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS most_pitcher_map_diag AS
        WITH ranked AS (
            SELECT
                game_id,
                game_date,
                pitcher_id,
                game_pitches,
                game_batters_approx,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id
                    ORDER BY game_pitches DESC, game_batters_approx DESC, pitcher_id
                ) rn
            FROM game_pitcher_workload_diag
        )
        SELECT
            game_id,
            game_date,
            pitcher_id AS most_pitcher_id,
            game_pitches AS most_pitcher_game_pitches,
            game_batters_approx AS most_pitcher_game_batters
        FROM ranked
        WHERE rn=1
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mpm_game ON most_pitcher_map_diag(game_id)")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS opponent_raw_audit_base AS
        SELECT
            CAST(bg.game_id AS TEXT) AS game_id,
            bg.batter_id,
            bg.game_date,
            bg.batter_name,
            bg.actual_hr,
            bg.lineup_spot,

            f.first_pitcher_id,
            f.stand,
            f.p_throws,
            fw.game_pitches AS first_pitcher_game_pitches,
            fw.game_batters_approx AS first_pitcher_game_batters,

            m.most_pitcher_id,
            m.most_pitcher_game_pitches,
            m.most_pitcher_game_batters,

            CASE
                WHEN fw.game_pitches >= 60 OR fw.game_batters_approx >= 18 THEN 'traditional'
                WHEN fw.game_pitches <= 40 OR fw.game_batters_approx <= 9 THEN 'opener'
                ELSE 'ambiguous'
            END AS first_pitcher_role,

            CASE WHEN f.first_pitcher_id = m.most_pitcher_id THEN 1 ELSE 0 END AS first_is_most
        FROM batter_games bg
        LEFT JOIN first_pitcher_map_diag f
          ON CAST(bg.game_id AS TEXT)=f.game_id
         AND bg.batter_id=f.batter_id
        LEFT JOIN game_pitcher_workload_diag fw
          ON f.game_id=fw.game_id
         AND f.first_pitcher_id=fw.pitcher_id
        LEFT JOIN most_pitcher_map_diag m
          ON CAST(bg.game_id AS TEXT)=m.game_id
        WHERE bg.actual_hr IS NOT NULL
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orab_game_batter ON opponent_raw_audit_base(game_id,batter_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orab_first_date ON opponent_raw_audit_base(first_pitcher_id,game_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orab_most_date ON opponent_raw_audit_base(most_pitcher_id,game_date)")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS opponent_raw_first_roll AS
        SELECT
            b.game_id,
            b.batter_id,
            COALESCE(SUM(d.pitches),0) AS first_pitches_365d,
            COALESCE(SUM(d.pa),0) AS first_pa_365d,
            COALESCE(SUM(d.bbe),0) AS first_bbe_365d,
            COALESCE(SUM(d.hr),0) AS first_hr_365d,
            COALESCE(SUM(d.k),0) AS first_k_365d,
            COALESCE(SUM(d.barrel),0) AS first_barrel_365d,
            COALESCE(SUM(d.hardhit),0) AS first_hardhit_365d,
            COALESCE(SUM(d.gb),0) AS first_gb_365d,
            COALESCE(SUM(d.air),0) AS first_air_365d
        FROM opponent_raw_audit_base b
        LEFT JOIN pitcher_daily_gate_stats d
          ON d.pitcher_id=b.first_pitcher_id
         AND d.game_date < b.game_date
         AND d.game_date >= DATE(b.game_date,'-365 day')
        GROUP BY b.game_id,b.batter_id
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orfr_game_batter ON opponent_raw_first_roll(game_id,batter_id)")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS opponent_raw_most_roll AS
        SELECT
            b.game_id,
            b.batter_id,
            COALESCE(SUM(d.pitches),0) AS most_pitches_365d,
            COALESCE(SUM(d.pa),0) AS most_pa_365d,
            COALESCE(SUM(d.bbe),0) AS most_bbe_365d,
            COALESCE(SUM(d.hr),0) AS most_hr_365d,
            COALESCE(SUM(d.k),0) AS most_k_365d,
            COALESCE(SUM(d.barrel),0) AS most_barrel_365d,
            COALESCE(SUM(d.hardhit),0) AS most_hardhit_365d,
            COALESCE(SUM(d.gb),0) AS most_gb_365d,
            COALESCE(SUM(d.air),0) AS most_air_365d
        FROM opponent_raw_audit_base b
        LEFT JOIN pitcher_daily_gate_stats d
          ON d.pitcher_id=b.most_pitcher_id
         AND d.game_date < b.game_date
         AND d.game_date >= DATE(b.game_date,'-365 day')
        GROUP BY b.game_id,b.batter_id
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ormr_game_batter ON opponent_raw_most_roll(game_id,batter_id)")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS opponent_raw_audit_features AS
        SELECT
            b.*,
            fr.first_pitches_365d,
            fr.first_pa_365d,
            fr.first_bbe_365d,
            fr.first_hr_365d,
            fr.first_k_365d,
            fr.first_barrel_365d,
            fr.first_hardhit_365d,
            fr.first_gb_365d,
            fr.first_air_365d,

            CASE WHEN fr.first_bbe_365d>0 THEN 1.0*fr.first_hr_365d/fr.first_bbe_365d END AS first_hr_per_bbe_365d,
            CASE WHEN fr.first_pa_365d>0 THEN 1.0*fr.first_hr_365d/fr.first_pa_365d END AS first_hr_per_pa_365d,
            CASE WHEN fr.first_pa_365d>0 THEN 1.0*fr.first_k_365d/fr.first_pa_365d END AS first_k_per_pa_365d,
            CASE WHEN fr.first_bbe_365d>0 THEN 1.0*fr.first_barrel_365d/fr.first_bbe_365d END AS first_barrel_per_bbe_365d,
            CASE WHEN fr.first_bbe_365d>0 THEN 1.0*fr.first_hardhit_365d/fr.first_bbe_365d END AS first_hardhit_per_bbe_365d,
            CASE WHEN fr.first_bbe_365d>0 THEN 1.0*fr.first_gb_365d/fr.first_bbe_365d END AS first_gb_per_bbe_365d,
            CASE WHEN fr.first_bbe_365d>0 THEN 1.0*fr.first_air_365d/fr.first_bbe_365d END AS first_air_per_bbe_365d,

            mr.most_pitches_365d,
            mr.most_pa_365d,
            mr.most_bbe_365d,
            mr.most_hr_365d,
            mr.most_k_365d,
            mr.most_barrel_365d,
            mr.most_hardhit_365d,
            mr.most_gb_365d,
            mr.most_air_365d,

            CASE WHEN mr.most_bbe_365d>0 THEN 1.0*mr.most_hr_365d/mr.most_bbe_365d END AS most_hr_per_bbe_365d,
            CASE WHEN mr.most_pa_365d>0 THEN 1.0*mr.most_hr_365d/mr.most_pa_365d END AS most_hr_per_pa_365d,
            CASE WHEN mr.most_pa_365d>0 THEN 1.0*mr.most_k_365d/mr.most_pa_365d END AS most_k_per_pa_365d,
            CASE WHEN mr.most_bbe_365d>0 THEN 1.0*mr.most_barrel_365d/mr.most_bbe_365d END AS most_barrel_per_bbe_365d,
            CASE WHEN mr.most_bbe_365d>0 THEN 1.0*mr.most_hardhit_365d/mr.most_bbe_365d END AS most_hardhit_per_bbe_365d,
            CASE WHEN mr.most_bbe_365d>0 THEN 1.0*mr.most_gb_365d/mr.most_bbe_365d END AS most_gb_per_bbe_365d,
            CASE WHEN mr.most_bbe_365d>0 THEN 1.0*mr.most_air_365d/mr.most_bbe_365d END AS most_air_per_bbe_365d
        FROM opponent_raw_audit_base b
        LEFT JOIN opponent_raw_first_roll fr
          ON b.game_id=fr.game_id AND b.batter_id=fr.batter_id
        LEFT JOIN opponent_raw_most_roll mr
          ON b.game_id=mr.game_id AND b.batter_id=mr.batter_id
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_oraf_date ON opponent_raw_audit_features(game_date)")

def load_df(pd, conn):
    df = pd.read_sql_query("SELECT * FROM opponent_raw_audit_features ORDER BY game_date, game_id, lineup_spot", conn)
    df["game_date"] = pd.to_datetime(df["game_date"])
    num_cols = [c for c in df.columns if c not in ("game_id","batter_name","stand","p_throws","first_pitcher_role")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def spearman(np, x, y):
    ok = x.notna() & y.notna()
    if ok.sum() < 100:
        return None, int(ok.sum())
    xr = x[ok].rank(method="average")
    yr = y[ok].rank(method="average")
    return float(np.corrcoef(xr, yr)[0,1]), int(ok.sum())

def print_deciles(pd, df, feature, sample_col, threshold, label):
    d = df[(df.game_date.dt.year == 2026) & df[feature].notna() & (df[sample_col].fillna(0) >= threshold)].copy()
    if len(d) < 200:
        print(f"{label} {feature} threshold={threshold}: not enough rows n={len(d)}")
        return
    try:
        d["decile"] = pd.qcut(d[feature], 10, labels=False, duplicates="drop") + 1
    except Exception as e:
        print(f"{label} {feature} threshold={threshold}: qcut error {e}")
        return
    print(f"\nDECILES {label}: {feature} min_{sample_col}={threshold}")
    print("-" * 70)
    for dec, g in d.groupby("decile"):
        print(f"decile={int(dec):02d} n={len(g):5d} mean={g[feature].mean():.5f} min={g[feature].min():.5f} max={g[feature].max():.5f} actual_hr={g.actual_hr.mean():.5f} hits={int(g.actual_hr.sum())}")

def subset_report(df, feature, sample_col, threshold, title):
    d = df[(df.game_date.dt.year==2026) & df[feature].notna() & (df[sample_col].fillna(0)>=threshold)]
    print(f"\nSUBSETS: {title} feature={feature} min_{sample_col}={threshold}")
    print("-" * 80)
    for col in ["first_pitcher_role", "first_is_most"]:
        for val, g in d.groupby(col):
            print(f"{col}={val} n={len(g):5d} actual_hr={g.actual_hr.mean():.5f} feature_mean={g[feature].mean():.5f}")
    print("\nHAND SPLITS")
    for (pt, st), g in d.groupby(["p_throws","stand"]):
        print(f"p_throws={pt} stand={st} n={len(g):5d} actual_hr={g.actual_hr.mean():.5f} feature_mean={g[feature].mean():.5f}")

def coverage(df):
    print("\nCOVERAGE")
    print("--------")
    print("rows:", len(df))
    print("range:", df.game_date.min().date(), df.game_date.max().date())
    print("mapped first pitcher rows:", int(df.first_pitcher_id.notna().sum()))
    print("first zero prior rows:", int((df.first_pitches_365d.fillna(0)<=0).sum()))
    print("most zero prior rows:", int((df.most_pitches_365d.fillna(0)<=0).sum()))
    for y,g in df.groupby(df.game_date.dt.year):
        print(f"year={y} rows={len(g)} first_zero_prior={int((g.first_pitches_365d.fillna(0)<=0).sum())} most_zero_prior={int((g.most_pitches_365d.fillna(0)<=0).sum())} actual_hr={g.actual_hr.mean():.5f}")
    d=df[df.game_date.dt.year==2026].copy()
    d["month"]=d.game_date.dt.strftime("%Y-%m")
    print("\nMONTH COVERAGE 2026")
    for m,g in d.groupby("month"):
        print(f"{m} rows={len(g)} first_zero={int((g.first_pitches_365d.fillna(0)<=0).sum())} first_bbe_med={g.first_bbe_365d.median():.1f} traditional_rows={int((g.first_pitcher_role=='traditional').sum())} opener_rows={int((g.first_pitcher_role=='opener').sum())} actual_hr={g.actual_hr.mean():.5f}")

def corr_report(np, df):
    print("\nRAW SPEARMAN DIRECTION 2026")
    print("---------------------------")
    tests = [
        ("first_hr_per_bbe_365d","first_bbe_365d"),
        ("first_hr_per_pa_365d","first_pa_365d"),
        ("first_k_per_pa_365d","first_pa_365d"),
        ("first_gb_per_bbe_365d","first_bbe_365d"),
        ("first_barrel_per_bbe_365d","first_bbe_365d"),
        ("most_hr_per_bbe_365d","most_bbe_365d"),
        ("most_hr_per_pa_365d","most_pa_365d"),
    ]
    d = df[df.game_date.dt.year==2026]
    for f,s in tests:
        for th in [100,250,500]:
            x = d.loc[d[s].fillna(0)>=th, f]
            y = d.loc[d[s].fillna(0)>=th, "actual_hr"]
            c,n = spearman(np, x, y)
            print(f"{f} min_{s}={th} n={n} spearman={None if c is None else round(c,5)} mean={x.mean():.5f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(db_path()))
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    pd, np = imports()
    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")
    print("HR_OPPONENT_RAW_AUDIT_B")
    print("=======================")
    print("db:", args.db)
    print("building/checking diagnostic tables...")
    build_audit_tables(conn, rebuild=args.rebuild)
    df = load_df(pd, conn)
    conn.close()

    coverage(df)
    corr_report(np, df)

    for th in [100,250,500]:
        print_deciles(pd, df, "first_hr_per_bbe_365d", "first_bbe_365d", th, "FIRST_PITCHER")
    for th in [100,250,500]:
        print_deciles(pd, df, "most_hr_per_bbe_365d", "most_bbe_365d", th, "MOST_PITCHES_DIAGNOSTIC")

    subset_report(df, "first_hr_per_bbe_365d", "first_bbe_365d", 250, "FIRST_PITCHER")
    subset_report(df, "most_hr_per_bbe_365d", "most_bbe_365d", 250, "MOST_PITCHES_DIAGNOSTIC")

    print("\nREAD")
    print("----")
    print("Green flag: higher raw HR/BBE deciles should generally show higher actual HR rates.")
    print("If FIRST is noisy but MOST is cleaner, opener/bulk mapping is poisoning the first-pitcher map.")
    print("If handedness splits are clearer than global, the next pitcher scalar must be handedness-conditioned.")
    print("If nothing is directional at 100/250/500 BBE, do not build pitch-zone overlap yet.")
    print("After this audit, the likely next infrastructure fix is 2024 pitch-level backfill so 2025 training rows have full prior history.")

if __name__ == "__main__":
    main()
