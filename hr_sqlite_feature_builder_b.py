#!/usr/bin/env python3
"""
HR SQLite Feature Builder B

Adds the missing pull-air layer to the SQLite rolling features.

Builds leak-safe rolling features from SQLite.

Important:
Final features are built per batter_game row using:
    daily.game_date < batter_games.game_date

That avoids same-day leakage.

Run:
    python hr_sqlite_feature_builder_b.py --windows 7,15,30 --export-csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
from pathlib import Path
from typing import List


def default_model_dir() -> Path:
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    if p.parent.exists():
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = Path("./hr_model")
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_db_path() -> Path:
    return default_model_dir() / "hr_model.sqlite"


def parse_windows(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def ensure_daily_rollups(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    DROP TABLE IF EXISTS batter_daily_bbe;
    CREATE TABLE batter_daily_bbe AS
    SELECT
        batter_id,
        game_date,
        COUNT(*) AS daily_bbe,
        SUM(CASE WHEN launch_speed_angle = 6 THEN 1 ELSE 0 END) AS daily_barrels,
        SUM(CASE WHEN launch_speed >= 95.0 THEN 1 ELSE 0 END) AS daily_hard_hits,
        SUM(CASE WHEN launch_angle >= 20.0 AND launch_angle <= 35.0 THEN 1 ELSE 0 END) AS daily_la_20_35,
        SUM(CASE WHEN bb_type = 'fly_ball' THEN 1 ELSE 0 END) AS daily_fly_balls,
        SUM(CASE WHEN bb_type IN ('fly_ball', 'line_drive') THEN 1 ELSE 0 END) AS daily_air_balls,
        SUM(
            CASE
              WHEN bb_type IN ('fly_ball', 'line_drive')
               AND (
                    (stand = 'R' AND hc_x IS NOT NULL AND hc_x < 125)
                 OR (stand = 'L' AND hc_x IS NOT NULL AND hc_x > 125)
               )
              THEN 1 ELSE 0
            END
        ) AS daily_pull_air,
        SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS daily_hr,
        SUM(launch_speed) AS daily_ev_sum,
        COUNT(launch_speed) AS daily_ev_count,
        AVG(launch_speed) AS daily_avg_ev,
        MAX(launch_speed) AS daily_max_ev
    FROM statcast_bbe
    GROUP BY batter_id, game_date;

    DROP TABLE IF EXISTS pitcher_daily_bbe;
    CREATE TABLE pitcher_daily_bbe AS
    SELECT
        pitcher_id,
        game_date,
        COUNT(*) AS daily_bbe,
        SUM(CASE WHEN launch_speed_angle = 6 THEN 1 ELSE 0 END) AS daily_barrels_allowed,
        SUM(CASE WHEN launch_speed >= 95.0 THEN 1 ELSE 0 END) AS daily_hard_hits_allowed,
        SUM(CASE WHEN launch_angle >= 20.0 AND launch_angle <= 35.0 THEN 1 ELSE 0 END) AS daily_la_20_35_allowed,
        SUM(CASE WHEN bb_type = 'fly_ball' THEN 1 ELSE 0 END) AS daily_fly_balls_allowed,
        SUM(CASE WHEN bb_type IN ('fly_ball', 'line_drive') THEN 1 ELSE 0 END) AS daily_air_balls_allowed,
        SUM(
            CASE
              WHEN bb_type IN ('fly_ball', 'line_drive')
               AND (
                    (stand = 'R' AND hc_x IS NOT NULL AND hc_x < 125)
                 OR (stand = 'L' AND hc_x IS NOT NULL AND hc_x > 125)
               )
              THEN 1 ELSE 0
            END
        ) AS daily_pull_air_allowed,
        SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS daily_hr_allowed,
        SUM(launch_speed) AS daily_ev_sum_allowed,
        COUNT(launch_speed) AS daily_ev_count_allowed,
        AVG(launch_speed) AS daily_avg_ev_allowed,
        MAX(launch_speed) AS daily_max_ev_allowed
    FROM statcast_bbe
    GROUP BY pitcher_id, game_date;

    CREATE INDEX IF NOT EXISTS idx_batter_daily_lookup
    ON batter_daily_bbe(batter_id, game_date);

    CREATE INDEX IF NOT EXISTS idx_pitcher_daily_lookup
    ON pitcher_daily_bbe(pitcher_id, game_date);
    """)
    conn.commit()


def build_batter_features(conn: sqlite3.Connection, windows: List[int]) -> None:
    maxw = max(windows)
    sum_exprs = []
    final_exprs = []

    for w in windows:
        suffix = f"{w}d"
        for col, out in [
            ("daily_bbe", "bbe"),
            ("daily_barrels", "barrels"),
            ("daily_hard_hits", "hard_hits"),
            ("daily_la_20_35", "la_20_35"),
            ("daily_fly_balls", "fly_balls"),
            ("daily_air_balls", "air_balls"),
            ("daily_pull_air", "pull_air"),
            ("daily_hr", "hr"),
            ("daily_ev_sum", "ev_sum"),
            ("daily_ev_count", "ev_count"),
            ("daily_max_ev", "max_ev"),
        ]:
            agg = "MAX" if col == "daily_max_ev" else "SUM"
            sum_exprs.append(
                f"{agg}(CASE WHEN bd.game_date >= date(bg.game_date, '-{w} day') THEN COALESCE(bd.{col}, 0) ELSE 0 END) AS batter_{out}_{suffix}"
            )

        final_exprs.extend([
            f"batter_bbe_{suffix}",
            f"batter_barrels_{suffix}",
            f"CASE WHEN batter_bbe_{suffix} > 0 THEN batter_barrels_{suffix} * 1.0 / batter_bbe_{suffix} END AS batter_barrel_rate_{suffix}",
            f"CASE WHEN batter_bbe_{suffix} > 0 THEN batter_hard_hits_{suffix} * 1.0 / batter_bbe_{suffix} END AS batter_hard_hit_rate_{suffix}",
            f"CASE WHEN batter_bbe_{suffix} > 0 THEN batter_la_20_35_{suffix} * 1.0 / batter_bbe_{suffix} END AS batter_la_20_35_rate_{suffix}",
            f"CASE WHEN batter_bbe_{suffix} > 0 THEN batter_fly_balls_{suffix} * 1.0 / batter_bbe_{suffix} END AS batter_fly_ball_rate_{suffix}",
            f"CASE WHEN batter_air_balls_{suffix} > 0 THEN batter_pull_air_{suffix} * 1.0 / batter_air_balls_{suffix} END AS batter_pull_air_rate_{suffix}",
            f"CASE WHEN batter_air_balls_{suffix} > 0 THEN batter_hr_{suffix} * 1.0 / batter_air_balls_{suffix} END AS batter_hr_per_air_{suffix}",
            f"CASE WHEN batter_ev_count_{suffix} > 0 THEN batter_ev_sum_{suffix} * 1.0 / batter_ev_count_{suffix} END AS batter_avg_ev_{suffix}",
            f"batter_max_ev_{suffix}",
        ])

    conn.executescript("DROP TABLE IF EXISTS batter_game_features;")
    sql = f"""
    CREATE TABLE batter_game_features AS
    WITH sums AS (
        SELECT
            bg.game_id,
            bg.batter_id,
            {", ".join(sum_exprs)}
        FROM batter_games bg
        LEFT JOIN batter_daily_bbe bd
          ON bd.batter_id = bg.batter_id
         AND bd.game_date < bg.game_date
         AND bd.game_date >= date(bg.game_date, '-{maxw} day')
        GROUP BY bg.game_id, bg.batter_id
    )
    SELECT
        game_id,
        batter_id,
        {", ".join(final_exprs)}
    FROM sums;
    """
    conn.execute(sql)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_batter_game_features_key ON batter_game_features(game_id, batter_id)")
    conn.commit()


def build_pitcher_features(conn: sqlite3.Connection, windows: List[int]) -> None:
    maxw = max(windows)
    sum_exprs = []
    final_exprs = []

    for w in windows:
        suffix = f"{w}d"
        for col, out in [
            ("daily_bbe", "bbe_allowed"),
            ("daily_barrels_allowed", "barrels_allowed"),
            ("daily_hard_hits_allowed", "hard_hits_allowed"),
            ("daily_la_20_35_allowed", "la_20_35_allowed"),
            ("daily_fly_balls_allowed", "fly_balls_allowed"),
            ("daily_air_balls_allowed", "air_balls_allowed"),
            ("daily_pull_air_allowed", "pull_air_allowed"),
            ("daily_hr_allowed", "hr_allowed"),
            ("daily_ev_sum_allowed", "ev_sum_allowed"),
            ("daily_ev_count_allowed", "ev_count_allowed"),
            ("daily_max_ev_allowed", "max_ev_allowed"),
        ]:
            agg = "MAX" if col == "daily_max_ev_allowed" else "SUM"
            sum_exprs.append(
                f"{agg}(CASE WHEN pd.game_date >= date(bg.game_date, '-{w} day') THEN COALESCE(pd.{col}, 0) ELSE 0 END) AS pitcher_{out}_{suffix}"
            )

        final_exprs.extend([
            f"pitcher_bbe_allowed_{suffix}",
            f"pitcher_barrels_allowed_{suffix}",
            f"CASE WHEN pitcher_bbe_allowed_{suffix} > 0 THEN pitcher_barrels_allowed_{suffix} * 1.0 / pitcher_bbe_allowed_{suffix} END AS pitcher_barrel_allowed_rate_{suffix}",
            f"CASE WHEN pitcher_bbe_allowed_{suffix} > 0 THEN pitcher_hard_hits_allowed_{suffix} * 1.0 / pitcher_bbe_allowed_{suffix} END AS pitcher_hard_hit_allowed_rate_{suffix}",
            f"CASE WHEN pitcher_bbe_allowed_{suffix} > 0 THEN pitcher_la_20_35_allowed_{suffix} * 1.0 / pitcher_bbe_allowed_{suffix} END AS pitcher_la_20_35_allowed_rate_{suffix}",
            f"CASE WHEN pitcher_bbe_allowed_{suffix} > 0 THEN pitcher_fly_balls_allowed_{suffix} * 1.0 / pitcher_bbe_allowed_{suffix} END AS pitcher_fly_ball_rate_{suffix}",
            f"CASE WHEN pitcher_air_balls_allowed_{suffix} > 0 THEN pitcher_pull_air_allowed_{suffix} * 1.0 / pitcher_air_balls_allowed_{suffix} END AS pitcher_pull_air_allowed_rate_{suffix}",
            f"CASE WHEN pitcher_fly_balls_allowed_{suffix} > 0 THEN pitcher_hr_allowed_{suffix} * 1.0 / pitcher_fly_balls_allowed_{suffix} END AS pitcher_hrfb_rate_{suffix}",
            f"CASE WHEN pitcher_ev_count_allowed_{suffix} > 0 THEN pitcher_ev_sum_allowed_{suffix} * 1.0 / pitcher_ev_count_allowed_{suffix} END AS pitcher_avg_ev_allowed_{suffix}",
            f"pitcher_max_ev_allowed_{suffix}",
        ])

    conn.executescript("DROP TABLE IF EXISTS pitcher_game_features;")
    sql = f"""
    CREATE TABLE pitcher_game_features AS
    WITH sums AS (
        SELECT
            bg.game_id,
            bg.batter_id,
            bg.opposing_pitcher_id,
            {", ".join(sum_exprs)}
        FROM batter_games bg
        LEFT JOIN pitcher_daily_bbe pd
          ON pd.pitcher_id = bg.opposing_pitcher_id
         AND pd.game_date < bg.game_date
         AND pd.game_date >= date(bg.game_date, '-{maxw} day')
        GROUP BY bg.game_id, bg.batter_id, bg.opposing_pitcher_id
    )
    SELECT
        game_id,
        batter_id,
        opposing_pitcher_id,
        {", ".join(final_exprs)}
    FROM sums;
    """
    conn.execute(sql)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pitcher_game_features_key ON pitcher_game_features(game_id, batter_id)")
    conn.commit()


def build_model_dataset(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    DROP VIEW IF EXISTS hr_model_dataset_view;
    CREATE VIEW hr_model_dataset_view AS
    SELECT
        bg.*,
        bf.*,
        pf.*
    FROM batter_games bg
    LEFT JOIN batter_game_features bf
      ON bg.game_id = bf.game_id
     AND bg.batter_id = bf.batter_id
    LEFT JOIN pitcher_game_features pf
      ON bg.game_id = pf.game_id
     AND bg.batter_id = pf.batter_id;
    """)
    conn.commit()


def export_csv(conn: sqlite3.Connection, out_path: Path) -> int:
    cur = conn.execute("SELECT * FROM hr_model_dataset_view")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--windows", default="7,15,30")
    ap.add_argument("--export-csv", action="store_true")
    ap.add_argument("--csv-output", default=None)
    args = ap.parse_args()

    db_path = Path(args.db)
    windows = parse_windows(args.windows)
    conn = sqlite3.connect(db_path)

    print("HR SQLITE FEATURE BUILDER B")
    print("===========================")
    print(f"db: {db_path}")
    print(f"windows: {windows}")

    ensure_daily_rollups(conn)
    batter_daily = conn.execute("SELECT COUNT(*) FROM batter_daily_bbe").fetchone()[0]
    pitcher_daily = conn.execute("SELECT COUNT(*) FROM pitcher_daily_bbe").fetchone()[0]
    print(f"batter_daily_bbe rows: {batter_daily}")
    print(f"pitcher_daily_bbe rows: {pitcher_daily}")

    build_batter_features(conn, windows)
    build_pitcher_features(conn, windows)
    build_model_dataset(conn)

    bg = conn.execute("SELECT COUNT(*) FROM batter_games").fetchone()[0]
    bf = conn.execute("SELECT COUNT(*) FROM batter_game_features").fetchone()[0]
    pf = conn.execute("SELECT COUNT(*) FROM pitcher_game_features").fetchone()[0]
    print(f"batter_games rows: {bg}")
    print(f"batter_game_features rows: {bf}")
    print(f"pitcher_game_features rows: {pf}")

    if args.export_csv:
        out = Path(args.csv_output) if args.csv_output else db_path.parent / "hr_model_dataset_sqlite.csv"
        n = export_csv(conn, out)
        print(f"exported_csv_rows: {n}")
        print(f"csv: {out}")

    conn.close()
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
