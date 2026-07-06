#!/usr/bin/env python3
"""
HR_DB_HEALTH_CHECK_A

Checks disk space, SQLite counts, date coverage, and common large files.

Run:
    python hr_db_health_check_a.py
"""
import sqlite3
import shutil
from pathlib import Path

DB = Path("/data/hr_model/hr_model.sqlite")
ROOT = Path("/data/hr_model")

def size_mb(p):
    try:
        return p.stat().st_size / 1024 / 1024
    except Exception:
        return 0

def main():
    print("HR_DB_HEALTH_CHECK_A")
    print("====================")

    try:
        total, used, free = shutil.disk_usage("/data")
        print(f"/data disk total_gb={total/1024**3:.3f} used_gb={used/1024**3:.3f} free_gb={free/1024**3:.3f}")
    except Exception as e:
        print(f"disk_usage_error: {type(e).__name__}: {e}")

    print("\nLARGE FILES")
    print("-----------")
    if ROOT.exists():
        files = []
        for p in ROOT.glob("*"):
            if p.is_file():
                files.append((size_mb(p), str(p)))
        for mb, p in sorted(files, reverse=True)[:25]:
            print(f"{mb:10.2f} MB  {p}")

    if not DB.exists():
        print(f"\nDB not found: {DB}")
        return

    conn = sqlite3.connect(DB)
    print("\nDB")
    print("--")
    print(f"path: {DB}")
    print(f"sqlite_file_mb: {size_mb(DB):.2f}")
    for suffix in ["-wal", "-shm"]:
        p = Path(str(DB) + suffix)
        if p.exists():
            print(f"{suffix}_mb: {size_mb(p):.2f}")

    print("\nTABLE COUNTS")
    print("------------")
    tables = [
        "batter_games", "statcast_bbe", "batter_daily_bbe", "pitcher_daily_bbe",
        "batter_game_features", "pitcher_game_features", "hr_odds_snapshots"
    ]
    for t in tables:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"{t:28s} {n}")
        except Exception as e:
            print(f"{t:28s} ERROR {type(e).__name__}: {e}")

    print("\nDATE RANGE")
    print("----------")
    for t, col in [("batter_games", "game_date"), ("statcast_bbe", "game_date")]:
        try:
            row = conn.execute(f"SELECT MIN({col}), MAX({col}), COUNT(DISTINCT {col}) FROM {t}").fetchone()
            print(f"{t:18s} min={row[0]} max={row[1]} distinct_dates={row[2]}")
        except Exception as e:
            print(f"{t:18s} ERROR {type(e).__name__}: {e}")

    print("\nLAST 20 BATTER_GAME DATES")
    print("-------------------------")
    try:
        rows = conn.execute("""
            SELECT game_date, COUNT(*) AS n, SUM(actual_hr) AS hr
            FROM batter_games
            GROUP BY game_date
            ORDER BY game_date DESC
            LIMIT 20
        """).fetchall()
        for d, n, hr in rows:
            print(f"{d} rows={n} hr={hr}")
    except Exception as e:
        print(f"date_count_error: {type(e).__name__}: {e}")

    print("\nWAL CHECKPOINT")
    print("--------------")
    try:
        print(conn.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchall())
    except Exception as e:
        print(f"checkpoint_error: {type(e).__name__}: {e}")
    conn.close()
    print("DONE")

if __name__ == "__main__":
    main()
