#!/usr/bin/env python3
"""
HR_LEAGUE_ENVIRONMENT_FEATURES_A

Builds leak-safe league-wide environment features for the HR model.

Creates:
1. league_daily_rollups
   - one row per MLB regular-season game_date in statcast_bbe
   - total_bbe, total_hr, total_air_events, total_barrels, avg_ev

2. league_env_lag_features
   - one row per batter_games game_date
   - lagged 10-day league HR environment features
   - uses only dates strictly before the target game_date

Run:
    python hr_league_environment_features_a.py
"""

import argparse
import os
import sqlite3
from pathlib import Path


def default_db_path() -> Path:
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    return p / "hr_model.sqlite" if p.parent.exists() else Path("./hr_model/hr_model.sqlite")


def scalar(conn, sql, params=()):
    return conn.execute(sql, params).fetchone()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--window-days", type=int, default=10)
    args = ap.parse_args()

    db = Path(args.db)
    conn = sqlite3.connect(str(db))

    print("HR_LEAGUE_ENVIRONMENT_FEATURES_A")
    print("================================")
    print(f"db: {db}")
    print(f"window_days: {args.window_days}")

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS league_daily_rollups (
            game_date TEXT PRIMARY KEY,
            total_bbe INTEGER,
            total_hr INTEGER,
            total_air_events INTEGER,
            total_barrels INTEGER,
            avg_ev REAL
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS league_env_lag_features (
            game_date TEXT PRIMARY KEY,
            league_hr_per_bbe_10d_lag REAL,
            league_hr_per_air_10d_lag REAL,
            league_barrel_rate_10d_lag REAL,
            league_avg_ev_10d_lag REAL,
            league_bbe_10d INTEGER,
            league_air_10d INTEGER
        );
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_league_daily_date ON league_daily_rollups(game_date);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_league_lag_date ON league_env_lag_features(game_date);")

        conn.execute("DELETE FROM league_daily_rollups;")
        conn.execute("DELETE FROM league_env_lag_features;")

        conn.execute("""
        INSERT OR REPLACE INTO league_daily_rollups (
            game_date,
            total_bbe,
            total_hr,
            total_air_events,
            total_barrels,
            avg_ev
        )
        SELECT
            game_date,
            COUNT(*) AS total_bbe,
            SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS total_hr,
            SUM(CASE WHEN bb_type IN ('fly_ball', 'line_drive') THEN 1 ELSE 0 END) AS total_air_events,
            SUM(CASE WHEN launch_speed_angle = 6 THEN 1 ELSE 0 END) AS total_barrels,
            AVG(launch_speed) AS avg_ev
        FROM statcast_bbe
        WHERE game_date IS NOT NULL
          AND launch_speed IS NOT NULL
        GROUP BY game_date;
        """)

        conn.execute(f"""
        INSERT OR REPLACE INTO league_env_lag_features (
            game_date,
            league_hr_per_bbe_10d_lag,
            league_hr_per_air_10d_lag,
            league_barrel_rate_10d_lag,
            league_avg_ev_10d_lag,
            league_bbe_10d,
            league_air_10d
        )
        SELECT
            d.game_date,

            1.0 * SUM(r.total_hr) / NULLIF(SUM(r.total_bbe), 0) AS league_hr_per_bbe_10d_lag,
            1.0 * SUM(r.total_hr) / NULLIF(SUM(r.total_air_events), 0) AS league_hr_per_air_10d_lag,
            1.0 * SUM(r.total_barrels) / NULLIF(SUM(r.total_bbe), 0) AS league_barrel_rate_10d_lag,
            1.0 * SUM(r.avg_ev * r.total_bbe) / NULLIF(SUM(r.total_bbe), 0) AS league_avg_ev_10d_lag,

            SUM(r.total_bbe) AS league_bbe_10d,
            SUM(r.total_air_events) AS league_air_10d

        FROM (
            SELECT DISTINCT game_date
            FROM batter_games
            WHERE game_date IS NOT NULL
        ) d
        LEFT JOIN league_daily_rollups r
          ON r.game_date < d.game_date
         AND r.game_date >= DATE(d.game_date, '-{args.window_days} day')
        GROUP BY d.game_date;
        """)

    daily_count = scalar(conn, "SELECT COUNT(*) FROM league_daily_rollups;")
    lag_count = scalar(conn, "SELECT COUNT(*) FROM league_env_lag_features;")
    bg_dates = scalar(conn, "SELECT COUNT(DISTINCT game_date) FROM batter_games;")

    print("\nCOUNTS")
    print("------")
    print(f"batter_games_distinct_dates: {bg_dates}")
    print(f"league_daily_rollups_rows:   {daily_count}")
    print(f"league_env_lag_features_rows:{lag_count}")

    print("\nDATE RANGES")
    print("-----------")
    print("league_daily_rollups:",
          conn.execute("SELECT MIN(game_date), MAX(game_date) FROM league_daily_rollups;").fetchone())
    print("league_env_lag_features:",
          conn.execute("SELECT MIN(game_date), MAX(game_date) FROM league_env_lag_features;").fetchone())

    print("\nSAMPLE DAILY ROLLUPS")
    print("--------------------")
    for row in conn.execute("""
        SELECT game_date, total_bbe, total_hr, total_air_events, total_barrels, ROUND(avg_ev, 3)
        FROM league_daily_rollups
        ORDER BY game_date DESC
        LIMIT 10;
    """):
        print(row)

    print("\nSAMPLE LAG FEATURES")
    print("-------------------")
    for row in conn.execute("""
        SELECT
            game_date,
            ROUND(league_hr_per_bbe_10d_lag, 5),
            ROUND(league_hr_per_air_10d_lag, 5),
            ROUND(league_barrel_rate_10d_lag, 5),
            ROUND(league_avg_ev_10d_lag, 3),
            league_bbe_10d,
            league_air_10d
        FROM league_env_lag_features
        ORDER BY game_date DESC
        LIMIT 15;
    """):
        print(row)

    print("\nMONTHLY 2026 LAG FEATURE AVERAGES")
    print("---------------------------------")
    for row in conn.execute("""
        SELECT
            SUBSTR(game_date, 1, 7) AS month,
            COUNT(*) AS dates,
            ROUND(AVG(league_hr_per_bbe_10d_lag), 5) AS avg_hr_per_bbe_10d,
            ROUND(AVG(league_hr_per_air_10d_lag), 5) AS avg_hr_per_air_10d,
            ROUND(AVG(league_barrel_rate_10d_lag), 5) AS avg_barrel_10d,
            ROUND(AVG(league_avg_ev_10d_lag), 3) AS avg_ev_10d,
            ROUND(AVG(league_bbe_10d), 1) AS avg_bbe_10d
        FROM league_env_lag_features
        WHERE game_date >= '2026-04-01'
          AND game_date <= '2026-07-05'
        GROUP BY SUBSTR(game_date, 1, 7)
        ORDER BY month;
    """):
        print(row)

    null_lag = scalar(conn, """
        SELECT COUNT(*)
        FROM league_env_lag_features
        WHERE league_hr_per_bbe_10d_lag IS NULL
           OR league_hr_per_air_10d_lag IS NULL
           OR league_barrel_rate_10d_lag IS NULL;
    """)
    print("\nNULL_LAG_ROWS:", null_lag)

    conn.close()
    print("\nDONE")


if __name__ == "__main__":
    main()
