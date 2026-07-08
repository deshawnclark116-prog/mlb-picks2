#!/usr/bin/env python3
"""
HR_OPPONENT_RAW_AUDIT_F

Clean rebuild of the raw pitcher-vector audit.

Fixes:
- No accidental pandas syntax inside SQL.
- Robust date parsing for SQLite returning text OR microsecond integers.
- Rebuilds all diagnostic tables cleanly when --rebuild is passed.

Run:
    python hr_opponent_raw_audit_f.py --rebuild
"""

import argparse
import os
import sqlite3
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


def parse_game_date(pd, series):
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.80:
        med = float(num.dropna().abs().median()) if num.notna().any() else 0.0
        if med > 1e14:
            return pd.to_datetime(num, unit="us", errors="coerce")
        if med > 1e11:
            return pd.to_datetime(num, unit="ms", errors="coerce")
        if med > 1e8:
            return pd.to_datetime(num, unit="s", errors="coerce")
    return pd.to_datetime(series.astype(str), errors="coerce")


def have_table(conn, table):
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def build_pitcher_daily(conn):
    with conn:
        conn.execute("DROP TABLE IF EXISTS pitcher_daily_raw_audit")
        conn.execute("""
        CREATE TABLE pitcher_daily_raw_audit AS
        SELECT
            game_date,
            pitcher_id,
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
        WHERE pitcher_id IS NOT NULL
          AND game_date IS NOT NULL
        GROUP BY game_date, pitcher_id
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pdra_pitcher_date ON pitcher_daily_raw_audit(pitcher_id, game_date)")


def build_maps(conn):
    with conn:
        conn.execute("DROP TABLE IF EXISTS first_pitcher_raw_audit")
        conn.execute("""
        CREATE TABLE first_pitcher_raw_audit AS
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
                    ORDER BY COALESCE(at_bat_number, 9999), COALESCE(pitch_number, 9999)
                ) AS rn
            FROM statcast_pitches
            WHERE game_pk IS NOT NULL
              AND batter_id IS NOT NULL
              AND pitcher_id IS NOT NULL
              AND game_date IS NOT NULL
        )
        SELECT
            game_id,
            game_date,
            batter_id,
            pitcher_id AS first_pitcher_id,
            stand,
            p_throws
        FROM ranked
        WHERE rn=1
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fpra_game_batter ON first_pitcher_raw_audit(game_id, batter_id)")

        conn.execute("DROP TABLE IF EXISTS game_pitcher_workload_raw_audit")
        conn.execute("""
        CREATE TABLE game_pitcher_workload_raw_audit AS
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gpwra_game_pitcher ON game_pitcher_workload_raw_audit(game_id, pitcher_id)")

        conn.execute("DROP TABLE IF EXISTS most_pitcher_raw_audit")
        conn.execute("""
        CREATE TABLE most_pitcher_raw_audit AS
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
                ) AS rn
            FROM game_pitcher_workload_raw_audit
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mpra_game ON most_pitcher_raw_audit(game_id)")


def build_base(conn):
    with conn:
        conn.execute("DROP TABLE IF EXISTS opponent_raw_audit_base_e")
        conn.execute("""
        CREATE TABLE opponent_raw_audit_base_e AS
        SELECT
            CAST(bg.game_id AS TEXT) AS game_id,
            bg.batter_id,
            bg.game_date,
            bg.batter_name,
            bg.actual_hr,
            bg.lineup_spot,

            fp.first_pitcher_id,
            fp.stand,
            fp.p_throws,
            fw.game_pitches AS first_pitcher_game_pitches,
            fw.game_batters_approx AS first_pitcher_game_batters,

            mp.most_pitcher_id,
            mp.most_pitcher_game_pitches,
            mp.most_pitcher_game_batters,

            CASE
                WHEN fw.game_pitches >= 60 OR fw.game_batters_approx >= 18 THEN 'traditional'
                WHEN fw.game_pitches <= 40 OR fw.game_batters_approx <= 9 THEN 'opener'
                ELSE 'ambiguous'
            END AS first_pitcher_role,

            CASE WHEN fp.first_pitcher_id = mp.most_pitcher_id THEN 1 ELSE 0 END AS first_is_most
        FROM batter_games bg
        LEFT JOIN first_pitcher_raw_audit fp
          ON CAST(bg.game_id AS TEXT)=fp.game_id
         AND bg.batter_id=fp.batter_id
        LEFT JOIN game_pitcher_workload_raw_audit fw
          ON fp.game_id=fw.game_id
         AND fp.first_pitcher_id=fw.pitcher_id
        LEFT JOIN most_pitcher_raw_audit mp
          ON CAST(bg.game_id AS TEXT)=mp.game_id
        WHERE bg.actual_hr IS NOT NULL
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orabe_game_batter ON opponent_raw_audit_base_e(game_id, batter_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orabe_first_date ON opponent_raw_audit_base_e(first_pitcher_id, game_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orabe_most_date ON opponent_raw_audit_base_e(most_pitcher_id, game_date)")


def build_rolls(conn):
    with conn:
        conn.execute("DROP TABLE IF EXISTS opponent_raw_first_roll_e")
        conn.execute("""
        CREATE TABLE opponent_raw_first_roll_e AS
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
        FROM opponent_raw_audit_base_e b
        LEFT JOIN pitcher_daily_raw_audit d
          ON d.pitcher_id=b.first_pitcher_id
         AND d.game_date < b.game_date
         AND d.game_date >= DATE(b.game_date, '-365 day')
        GROUP BY b.game_id, b.batter_id
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orfre_game_batter ON opponent_raw_first_roll_e(game_id, batter_id)")

        conn.execute("DROP TABLE IF EXISTS opponent_raw_most_roll_e")
        conn.execute("""
        CREATE TABLE opponent_raw_most_roll_e AS
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
        FROM opponent_raw_audit_base_e b
        LEFT JOIN pitcher_daily_raw_audit d
          ON d.pitcher_id=b.most_pitcher_id
         AND d.game_date < b.game_date
         AND d.game_date >= DATE(b.game_date, '-365 day')
        GROUP BY b.game_id, b.batter_id
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ormre_game_batter ON opponent_raw_most_roll_e(game_id, batter_id)")


def build_features(conn):
    with conn:
        conn.execute("DROP TABLE IF EXISTS opponent_raw_audit_features_e")
        conn.execute("""
        CREATE TABLE opponent_raw_audit_features_e AS
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
        FROM opponent_raw_audit_base_e b
        LEFT JOIN opponent_raw_first_roll_e fr
          ON b.game_id=fr.game_id
         AND b.batter_id=fr.batter_id
        LEFT JOIN opponent_raw_most_roll_e mr
          ON b.game_id=mr.game_id
         AND b.batter_id=mr.batter_id
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orafe_date ON opponent_raw_audit_features_e(game_date)")


def rebuild_all(conn):
    print("building pitcher daily...")
    build_pitcher_daily(conn)
    print("building maps...")
    build_maps(conn)
    print("building base...")
    build_base(conn)
    print("building rolls...")
    build_rolls(conn)
    print("building features...")
    build_features(conn)


def load_df(pd, conn):
    df = pd.read_sql_query(
        "SELECT * FROM opponent_raw_audit_features_e ORDER BY game_date, game_id, lineup_spot",
        conn,
    )
    df["game_date"] = parse_game_date(pd, df["game_date"])
    for c in df.columns:
        if c not in ("game_id", "game_date", "batter_name", "stand", "p_throws", "first_pitcher_role"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def spearman(np, x, y):
    ok = x.notna() & y.notna()
    if int(ok.sum()) < 100:
        return None, int(ok.sum())
    xr = x[ok].rank(method="average")
    yr = y[ok].rank(method="average")
    return float(np.corrcoef(xr, yr)[0, 1]), int(ok.sum())


def coverage(df):
    print("\nCOVERAGE")
    print("--------")
    print("rows:", len(df))
    print("range:", df["game_date"].min(), df["game_date"].max())
    print("mapped first pitcher rows:", int(df["first_pitcher_id"].notna().sum()))
    print("first zero prior rows:", int((df["first_pitches_365d"].fillna(0) <= 0).sum()))
    print("most zero prior rows:", int((df["most_pitches_365d"].fillna(0) <= 0).sum()))

    for year, g in df.groupby(df["game_date"].dt.year):
        print(
            f"year={int(year)} rows={len(g)} "
            f"first_zero_prior={int((g['first_pitches_365d'].fillna(0)<=0).sum())} "
            f"most_zero_prior={int((g['most_pitches_365d'].fillna(0)<=0).sum())} "
            f"actual_hr={g['actual_hr'].mean():.5f}"
        )

    d = df[df["game_date"].dt.year == 2026].copy()
    d["month"] = d["game_date"].dt.strftime("%Y-%m")
    print("\nMONTH COVERAGE 2026")
    for month, g in d.groupby("month"):
        print(
            f"{month} rows={len(g)} "
            f"first_zero={int((g['first_pitches_365d'].fillna(0)<=0).sum())} "
            f"first_bbe_med={g['first_bbe_365d'].median():.1f} "
            f"traditional_rows={int((g['first_pitcher_role']=='traditional').sum())} "
            f"opener_rows={int((g['first_pitcher_role']=='opener').sum())} "
            f"actual_hr={g['actual_hr'].mean():.5f}"
        )


def corr_report(np, df):
    print("\nRAW SPEARMAN DIRECTION 2026")
    print("---------------------------")
    tests = [
        ("first_hr_per_bbe_365d", "first_bbe_365d"),
        ("first_hr_per_pa_365d", "first_pa_365d"),
        ("first_k_per_pa_365d", "first_pa_365d"),
        ("first_gb_per_bbe_365d", "first_bbe_365d"),
        ("first_barrel_per_bbe_365d", "first_bbe_365d"),
        ("most_hr_per_bbe_365d", "most_bbe_365d"),
        ("most_hr_per_pa_365d", "most_pa_365d"),
    ]
    d = df[df["game_date"].dt.year == 2026]
    for feature, sample in tests:
        for threshold in [100, 250, 500]:
            mask = d[sample].fillna(0) >= threshold
            c, n = spearman(np, d.loc[mask, feature], d.loc[mask, "actual_hr"])
            mean = d.loc[mask, feature].mean()
            print(f"{feature} min_{sample}={threshold} n={n} spearman={None if c is None else round(c, 5)} mean={mean:.5f}")


def deciles(pd, df, feature, sample, threshold, label):
    d = df[
        (df["game_date"].dt.year == 2026)
        & df[feature].notna()
        & (df[sample].fillna(0) >= threshold)
    ].copy()

    if len(d) < 200:
        print(f"{label} {feature} min_{sample}={threshold}: not enough n={len(d)}")
        return

    try:
        d["decile"] = pd.qcut(d[feature], 10, labels=False, duplicates="drop") + 1
    except Exception as e:
        print(f"{label} {feature} min_{sample}={threshold}: qcut error {e}")
        return

    print(f"\nDECILES {label}: {feature} min_{sample}={threshold}")
    print("-" * 80)
    for decile, g in d.groupby("decile"):
        print(
            f"decile={int(decile):02d} n={len(g):5d} "
            f"mean={g[feature].mean():.5f} "
            f"min={g[feature].min():.5f} max={g[feature].max():.5f} "
            f"actual_hr={g['actual_hr'].mean():.5f} hits={int(g['actual_hr'].sum())}"
        )


def subsets(df, feature, sample, threshold, label):
    d = df[
        (df["game_date"].dt.year == 2026)
        & df[feature].notna()
        & (df[sample].fillna(0) >= threshold)
    ].copy()

    print(f"\nSUBSETS {label}: {feature} min_{sample}={threshold}")
    print("-" * 80)

    print("ROLE SPLIT")
    for value, g in d.groupby("first_pitcher_role"):
        print(f"role={value} n={len(g):5d} actual_hr={g['actual_hr'].mean():.5f} feature_mean={g[feature].mean():.5f}")

    print("FIRST_IS_MOST SPLIT")
    for value, g in d.groupby("first_is_most"):
        print(f"first_is_most={int(value)} n={len(g):5d} actual_hr={g['actual_hr'].mean():.5f} feature_mean={g[feature].mean():.5f}")

    print("HAND SPLITS")
    for (throws, stand), g in d.groupby(["p_throws", "stand"]):
        print(f"p_throws={throws} stand={stand} n={len(g):5d} actual_hr={g['actual_hr'].mean():.5f} feature_mean={g[feature].mean():.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(db_path()))
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    pd, np = imports()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")

    print("HR_OPPONENT_RAW_AUDIT_F")
    print("=======================")
    print("db:", args.db)

    if args.rebuild or not have_table(conn, "opponent_raw_audit_features_e"):
        rebuild_all(conn)
    else:
        print("using existing opponent_raw_audit_features_e")

    df = load_df(pd, conn)
    conn.close()

    coverage(df)
    corr_report(np, df)

    for threshold in [100, 250, 500]:
        deciles(pd, df, "first_hr_per_bbe_365d", "first_bbe_365d", threshold, "FIRST_PITCHER")

    for threshold in [100, 250, 500]:
        deciles(pd, df, "most_hr_per_bbe_365d", "most_bbe_365d", threshold, "MOST_PITCHES_DIAGNOSTIC")

    subsets(df, "first_hr_per_bbe_365d", "first_bbe_365d", 250, "FIRST_PITCHER")
    subsets(df, "most_hr_per_bbe_365d", "most_bbe_365d", 250, "MOST_PITCHES_DIAGNOSTIC")

    print("\nREAD")
    print("----")
    print("Green flag: higher raw HR/BBE deciles should generally show higher actual HR rates.")
    print("If FIRST is noisy but MOST is cleaner, opener/bulk mapping is poisoning the first-pitcher map.")
    print("If handedness splits are clearer than global, the next pitcher scalar must be handedness-conditioned.")
    print("If nothing is directional at 100/250/500 BBE, do not build pitch-zone overlap yet.")
    print("After this audit, the likely next infrastructure fix is 2024 pitch-level backfill so 2025 training rows have full prior history.")


if __name__ == "__main__":
    main()
