#!/usr/bin/env python3
"""
HR_EXPECTED_PA_A

Adds leakage-safe expected plate appearances to batter_games.

Best production usage after 2024/2025 backfill:
  python hr_expected_pa_a.py --lookup-start 2024-01-01 --lookup-end 2024-12-31 --apply-start 2025-01-01 --apply-end 2026-12-31

For current small sample fallback:
  python hr_expected_pa_a.py --fallback-if-empty --apply-all
"""
import argparse, os, sqlite3
from pathlib import Path

FALLBACK_SLOT_PA = {1:4.65,2:4.55,3:4.45,4:4.35,5:4.20,6:4.05,7:3.95,8:3.85,9:3.75}

def default_db_path():
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    return p / "hr_model.sqlite" if p.parent.exists() else Path("./hr_model/hr_model.sqlite")

def col_exists(conn, table, col):
    return any(r[1] == col for r in conn.execute(f"PRAGMA table_info({table})").fetchall())

def ensure_columns(conn):
    if not col_exists(conn, "batter_games", "expected_pa_v1"):
        conn.execute("ALTER TABLE batter_games ADD COLUMN expected_pa_v1 REAL")
    if not col_exists(conn, "batter_games", "expected_pa_source"):
        conn.execute("ALTER TABLE batter_games ADD COLUMN expected_pa_source TEXT")
    conn.commit()

def build_lookup(conn, lookup_start, lookup_end, min_n):
    conn.execute("DROP TABLE IF EXISTS expected_pa_lookup")
    conn.execute("""
        CREATE TABLE expected_pa_lookup (
            lineup_spot INTEGER NOT NULL,
            side TEXT NOT NULL,
            avg_pa REAL,
            n INTEGER,
            lookup_start TEXT,
            lookup_end TEXT,
            source TEXT,
            PRIMARY KEY(lineup_spot, side)
        )
    """)
    where = ["lineup_spot BETWEEN 1 AND 9", "side IN ('home','away')", "plate_appearances IS NOT NULL", "plate_appearances > 0"]
    params = []
    if lookup_start:
        where.append("game_date >= ?"); params.append(lookup_start)
    if lookup_end:
        where.append("game_date <= ?"); params.append(lookup_end)
    sql = f"""
        INSERT OR REPLACE INTO expected_pa_lookup
        (lineup_spot, side, avg_pa, n, lookup_start, lookup_end, source)
        SELECT lineup_spot, side, AVG(plate_appearances), COUNT(*), ?, ?, 'empirical'
        FROM batter_games
        WHERE {' AND '.join(where)}
        GROUP BY lineup_spot, side
        HAVING COUNT(*) >= ?
    """
    conn.execute(sql, [lookup_start, lookup_end] + params + [min_n])
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM expected_pa_lookup").fetchone()[0]

def fill_fallback_lookup(conn):
    count = conn.execute("SELECT COUNT(*) FROM expected_pa_lookup").fetchone()[0]
    if count > 0:
        return count
    rows = []
    for spot, base in FALLBACK_SLOT_PA.items():
        rows.append((spot, "away", round(base * 1.02, 3), 0, None, None, "fallback_slot_side"))
        rows.append((spot, "home", round(base * 0.98, 3), 0, None, None, "fallback_slot_side"))
    conn.executemany("""
        INSERT OR REPLACE INTO expected_pa_lookup
        (lineup_spot, side, avg_pa, n, lookup_start, lookup_end, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    return len(rows)

def apply_lookup(conn, apply_start, apply_end, apply_all):
    where = ["lineup_spot BETWEEN 1 AND 9", "side IN ('home','away')"]
    params = []
    if not apply_all:
        if apply_start:
            where.append("game_date >= ?"); params.append(apply_start)
        if apply_end:
            where.append("game_date <= ?"); params.append(apply_end)
    sql = f"""
        UPDATE batter_games
        SET expected_pa_v1 = (
              SELECT avg_pa FROM expected_pa_lookup l
              WHERE l.lineup_spot = batter_games.lineup_spot AND l.side = batter_games.side
            ),
            expected_pa_source = (
              SELECT source FROM expected_pa_lookup l
              WHERE l.lineup_spot = batter_games.lineup_spot AND l.side = batter_games.side
            )
        WHERE {' AND '.join(where)}
    """
    before = conn.total_changes
    conn.execute(sql, params)
    conn.commit()
    return conn.total_changes - before

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--lookup-start")
    ap.add_argument("--lookup-end")
    ap.add_argument("--apply-start")
    ap.add_argument("--apply-end")
    ap.add_argument("--apply-all", action="store_true")
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--fallback-if-empty", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_columns(conn)
    lookup_rows = build_lookup(conn, args.lookup_start, args.lookup_end, args.min_n)
    if args.fallback_if_empty:
        lookup_rows = fill_fallback_lookup(conn)
    updated = apply_lookup(conn, args.apply_start, args.apply_end, args.apply_all)

    print("HR_EXPECTED_PA_A")
    print("================")
    print(f"db: {args.db}")
    print(f"lookup_rows: {lookup_rows}")
    print(f"updated_batter_games: {updated}")
    print("\nLOOKUP")
    print("------")
    for r in conn.execute("SELECT lineup_spot, side, ROUND(avg_pa,3), n, source FROM expected_pa_lookup ORDER BY lineup_spot, side"):
        print(f"spot={r[0]} side={r[1]} avg_pa={r[2]} n={r[3]} source={r[4]}")
    filled = conn.execute("SELECT COUNT(*) FROM batter_games WHERE expected_pa_v1 IS NOT NULL").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM batter_games").fetchone()[0]
    print(f"\nexpected_pa_filled: {filled}/{total}")
    conn.close()

if __name__ == "__main__":
    main()
