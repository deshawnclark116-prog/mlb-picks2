#!/usr/bin/env python3
"""Leak-safe SQLite rolling feature builder for HR model dataset."""
from __future__ import annotations
import argparse, csv, os, sqlite3
from pathlib import Path
from typing import List


def model_dir() -> Path:
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    if not p.parent.exists(): p = Path("./hr_model")
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_default() -> Path:
    return model_dir() / "hr_model.sqlite"


def parse_windows(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def ensure_rollups(conn: sqlite3.Connection):
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
        SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS daily_hr_allowed,
        SUM(launch_speed) AS daily_ev_sum_allowed,
        COUNT(launch_speed) AS daily_ev_count_allowed,
        AVG(launch_speed) AS daily_avg_ev_allowed,
        MAX(launch_speed) AS daily_max_ev_allowed
    FROM statcast_bbe
    GROUP BY pitcher_id, game_date;

    CREATE INDEX IF NOT EXISTS idx_batter_daily_lookup ON batter_daily_bbe(batter_id, game_date);
    CREATE INDEX IF NOT EXISTS idx_pitcher_daily_lookup ON pitcher_daily_bbe(pitcher_id, game_date);
    """)
    conn.commit()


def build_batter(conn: sqlite3.Connection, windows: List[int]):
    maxw = max(windows)
    sum_exprs, final_exprs = [], []
    for w in windows:
        s = f"{w}d"
        mapping = [
            ("daily_bbe", "bbe", "SUM"), ("daily_barrels", "barrels", "SUM"),
            ("daily_hard_hits", "hard_hits", "SUM"), ("daily_la_20_35", "la_20_35", "SUM"),
            ("daily_fly_balls", "fly_balls", "SUM"), ("daily_air_balls", "air_balls", "SUM"),
            ("daily_hr", "hr", "SUM"), ("daily_ev_sum", "ev_sum", "SUM"),
            ("daily_ev_count", "ev_count", "SUM"), ("daily_max_ev", "max_ev", "MAX"),
        ]
        for col, out, agg in mapping:
            sum_exprs.append(f"{agg}(CASE WHEN bd.game_date >= date(bg.game_date, '-{w} day') THEN COALESCE(bd.{col}, 0) ELSE 0 END) AS batter_{out}_{s}")
        final_exprs += [
            f"batter_bbe_{s}",
            f"batter_barrels_{s}",
            f"CASE WHEN batter_bbe_{s} > 0 THEN batter_barrels_{s} * 1.0 / batter_bbe_{s} END AS batter_barrel_rate_{s}",
            f"CASE WHEN batter_bbe_{s} > 0 THEN batter_hard_hits_{s} * 1.0 / batter_bbe_{s} END AS batter_hard_hit_rate_{s}",
            f"CASE WHEN batter_bbe_{s} > 0 THEN batter_la_20_35_{s} * 1.0 / batter_bbe_{s} END AS batter_la_20_35_rate_{s}",
            f"CASE WHEN batter_bbe_{s} > 0 THEN batter_fly_balls_{s} * 1.0 / batter_bbe_{s} END AS batter_fly_ball_rate_{s}",
            f"CASE WHEN batter_air_balls_{s} > 0 THEN batter_hr_{s} * 1.0 / batter_air_balls_{s} END AS batter_hr_per_air_{s}",
            f"CASE WHEN batter_ev_count_{s} > 0 THEN batter_ev_sum_{s} * 1.0 / batter_ev_count_{s} END AS batter_avg_ev_{s}",
            f"batter_max_ev_{s}",
        ]
    conn.execute("DROP TABLE IF EXISTS batter_game_features")
    sql = f"""
    CREATE TABLE batter_game_features AS
    WITH sums AS (
      SELECT bg.game_id, bg.batter_id, {', '.join(sum_exprs)}
      FROM batter_games bg
      LEFT JOIN batter_daily_bbe bd
        ON bd.batter_id = bg.batter_id
       AND bd.game_date < bg.game_date
       AND bd.game_date >= date(bg.game_date, '-{maxw} day')
      GROUP BY bg.game_id, bg.batter_id
    )
    SELECT game_id, batter_id, {', '.join(final_exprs)} FROM sums;
    """
    conn.execute(sql)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_batter_game_features_key ON batter_game_features(game_id, batter_id)")
    conn.commit()


def build_pitcher(conn: sqlite3.Connection, windows: List[int]):
    maxw = max(windows)
    sum_exprs, final_exprs = [], []
    for w in windows:
        s = f"{w}d"
        mapping = [
            ("daily_bbe", "bbe_allowed", "SUM"), ("daily_barrels_allowed", "barrels_allowed", "SUM"),
            ("daily_hard_hits_allowed", "hard_hits_allowed", "SUM"), ("daily_la_20_35_allowed", "la_20_35_allowed", "SUM"),
            ("daily_fly_balls_allowed", "fly_balls_allowed", "SUM"), ("daily_air_balls_allowed", "air_balls_allowed", "SUM"),
            ("daily_hr_allowed", "hr_allowed", "SUM"), ("daily_ev_sum_allowed", "ev_sum_allowed", "SUM"),
            ("daily_ev_count_allowed", "ev_count_allowed", "SUM"), ("daily_max_ev_allowed", "max_ev_allowed", "MAX"),
        ]
        for col, out, agg in mapping:
            sum_exprs.append(f"{agg}(CASE WHEN pd.game_date >= date(bg.game_date, '-{w} day') THEN COALESCE(pd.{col}, 0) ELSE 0 END) AS pitcher_{out}_{s}")
        final_exprs += [
            f"pitcher_bbe_allowed_{s}",
            f"pitcher_barrels_allowed_{s}",
            f"CASE WHEN pitcher_bbe_allowed_{s} > 0 THEN pitcher_barrels_allowed_{s} * 1.0 / pitcher_bbe_allowed_{s} END AS pitcher_barrel_allowed_rate_{s}",
            f"CASE WHEN pitcher_bbe_allowed_{s} > 0 THEN pitcher_hard_hits_allowed_{s} * 1.0 / pitcher_bbe_allowed_{s} END AS pitcher_hard_hit_allowed_rate_{s}",
            f"CASE WHEN pitcher_bbe_allowed_{s} > 0 THEN pitcher_la_20_35_allowed_{s} * 1.0 / pitcher_bbe_allowed_{s} END AS pitcher_la_20_35_allowed_rate_{s}",
            f"CASE WHEN pitcher_bbe_allowed_{s} > 0 THEN pitcher_fly_balls_allowed_{s} * 1.0 / pitcher_bbe_allowed_{s} END AS pitcher_fly_ball_rate_{s}",
            f"CASE WHEN pitcher_fly_balls_allowed_{s} > 0 THEN pitcher_hr_allowed_{s} * 1.0 / pitcher_fly_balls_allowed_{s} END AS pitcher_hrfb_rate_{s}",
            f"CASE WHEN pitcher_ev_count_allowed_{s} > 0 THEN pitcher_ev_sum_allowed_{s} * 1.0 / pitcher_ev_count_allowed_{s} END AS pitcher_avg_ev_allowed_{s}",
            f"pitcher_max_ev_allowed_{s}",
        ]
    conn.execute("DROP TABLE IF EXISTS pitcher_game_features")
    sql = f"""
    CREATE TABLE pitcher_game_features AS
    WITH sums AS (
      SELECT bg.game_id, bg.batter_id, bg.opposing_pitcher_id, {', '.join(sum_exprs)}
      FROM batter_games bg
      LEFT JOIN pitcher_daily_bbe pd
        ON pd.pitcher_id = bg.opposing_pitcher_id
       AND pd.game_date < bg.game_date
       AND pd.game_date >= date(bg.game_date, '-{maxw} day')
      GROUP BY bg.game_id, bg.batter_id, bg.opposing_pitcher_id
    )
    SELECT game_id, batter_id, opposing_pitcher_id, {', '.join(final_exprs)} FROM sums;
    """
    conn.execute(sql)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pitcher_game_features_key ON pitcher_game_features(game_id, batter_id)")
    conn.commit()


def build_view(conn: sqlite3.Connection):
    conn.executescript("""
    DROP VIEW IF EXISTS hr_model_dataset_view;
    CREATE VIEW hr_model_dataset_view AS
    SELECT bg.*, bf.*, pf.*
    FROM batter_games bg
    LEFT JOIN batter_game_features bf ON bg.game_id = bf.game_id AND bg.batter_id = bf.batter_id
    LEFT JOIN pitcher_game_features pf ON bg.game_id = pf.game_id AND bg.batter_id = pf.batter_id;
    """)
    conn.commit()


def export_csv(conn: sqlite3.Connection, path: Path) -> int:
    cur = conn.execute("SELECT * FROM hr_model_dataset_view")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(db_default()))
    ap.add_argument("--windows", default="15,30,60")
    ap.add_argument("--export-csv", action="store_true")
    ap.add_argument("--csv-output", default=None)
    args = ap.parse_args()
    db = Path(args.db); conn = sqlite3.connect(db)
    windows = parse_windows(args.windows)
    print("HR SQLITE FEATURE BUILDER A")
    print("===========================")
    print(f"db: {db}")
    print(f"windows: {windows}")
    ensure_rollups(conn)
    print(f"batter_daily_bbe rows: {conn.execute('SELECT COUNT(*) FROM batter_daily_bbe').fetchone()[0]}")
    print(f"pitcher_daily_bbe rows: {conn.execute('SELECT COUNT(*) FROM pitcher_daily_bbe').fetchone()[0]}")
    build_batter(conn, windows); build_pitcher(conn, windows); build_view(conn)
    print(f"batter_games rows: {conn.execute('SELECT COUNT(*) FROM batter_games').fetchone()[0]}")
    print(f"batter_game_features rows: {conn.execute('SELECT COUNT(*) FROM batter_game_features').fetchone()[0]}")
    print(f"pitcher_game_features rows: {conn.execute('SELECT COUNT(*) FROM pitcher_game_features').fetchone()[0]}")
    if args.export_csv:
        out = Path(args.csv_output) if args.csv_output else db.parent / "hr_model_dataset_sqlite.csv"
        print(f"exported_csv_rows: {export_csv(conn, out)}")
        print(f"csv: {out}")
    conn.close(); print("DONE")

if __name__ == "__main__":
    main()
