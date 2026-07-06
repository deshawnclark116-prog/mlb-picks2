#!/usr/bin/env python3
"""Memory-safe Baseball Savant BBE date-range loader into SQLite. No Pandas."""
from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, os, sqlite3, time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import requests

SAVANT_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
USER_AGENT = "PropEdgeHRStatcastBBELoaderA/1.0"


def default_db() -> Path:
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    if not p.parent.exists(): p = Path("./hr_model")
    p.mkdir(parents=True, exist_ok=True)
    return p / "hr_model.sqlite"


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def chunks(start: dt.date, end: dt.date, days: int) -> Iterable[tuple[dt.date, dt.date]]:
    d = start
    while d <= end:
        e = min(end, d + dt.timedelta(days=max(1, days) - 1))
        yield d, e
        d = e + dt.timedelta(days=1)


def ci(x: Any) -> Optional[int]:
    try:
        if x in (None, "", "None"): return None
        return int(float(x))
    except Exception:
        return None


def cf(x: Any) -> Optional[float]:
    try:
        if x in (None, "", "None"): return None
        return float(x)
    except Exception:
        return None


def bbe_key(row: Dict[str, Any]) -> str:
    parts = [row.get(k) for k in ["game_pk", "game_date", "at_bat_number", "pitch_number", "batter", "pitcher", "launch_speed", "launch_angle", "hc_x", "hc_y", "events"]]
    return hashlib.sha256("|".join(str(p or "") for p in parts).encode()).hexdigest()


def is_bbe(row: Dict[str, Any]) -> bool:
    desc = str(row.get("description") or "").lower()
    bb = str(row.get("bb_type") or "").lower()
    ev, la = cf(row.get("launch_speed")), cf(row.get("launch_angle"))
    return ev is not None and la is not None and (desc == "hit_into_play" or bb in {"fly_ball", "line_drive", "ground_ball", "popup"})


def params(start: dt.date, end: dt.date) -> Dict[str, Any]:
    return {
        "all": "true", "hfPT": "", "hfAB": "", "hfGT": "R|", "hfPR": "", "hfZ": "", "stadium": "",
        "hfBBL": "", "hfNewZones": "", "hfPull": "", "hfC": "", "hfSea": "", "hfSit": "",
        "player_type": "pitcher", "hfOuts": "", "opponent": "", "pitcher_throws": "", "batter_stands": "",
        "hfSA": "", "game_date_gt": start.isoformat(), "game_date_lt": end.isoformat(), "team": "", "position": "",
        "hfRO": "", "home_road": "", "hfFlag": "", "hfBBT": "", "metric_1": "", "group_by": "name",
        "sort_col": "pitches", "player_event_sort": "h_launch_speed", "sort_order": "desc", "min_pitches": "0",
        "min_results": "0", "type": "details",
    }


def ensure_schema(conn: sqlite3.Connection):
    conn.executescript("""
    PRAGMA journal_mode=WAL;
    PRAGMA synchronous=NORMAL;
    CREATE TABLE IF NOT EXISTS statcast_bbe (
        bbe_key TEXT PRIMARY KEY,
        game_id TEXT NOT NULL,
        game_date TEXT NOT NULL,
        at_bat_number INTEGER,
        pitch_number INTEGER,
        batter_id INTEGER NOT NULL,
        pitcher_id INTEGER NOT NULL,
        stand TEXT,
        p_throws TEXT,
        launch_speed REAL,
        launch_angle REAL,
        launch_speed_angle INTEGER,
        bb_type TEXT,
        events TEXT,
        description TEXT,
        hc_x REAL,
        hc_y REAL
    );
    CREATE INDEX IF NOT EXISTS idx_statcast_batter_date ON statcast_bbe(batter_id, game_date);
    CREATE INDEX IF NOT EXISTS idx_statcast_pitcher_date ON statcast_bbe(pitcher_id, game_date);
    CREATE INDEX IF NOT EXISTS idx_statcast_game ON statcast_bbe(game_id);
    CREATE INDEX IF NOT EXISTS idx_statcast_date ON statcast_bbe(game_date);
    """)
    conn.commit()


def payload(row: Dict[str, Any]):
    game_id, game_date = row.get("game_pk"), row.get("game_date")
    batter, pitcher = ci(row.get("batter")), ci(row.get("pitcher"))
    if not game_id or not game_date or batter is None or pitcher is None: return None
    return (
        bbe_key(row), str(game_id), str(game_date), ci(row.get("at_bat_number")), ci(row.get("pitch_number")),
        batter, pitcher, row.get("stand"), row.get("p_throws"), cf(row.get("launch_speed")), cf(row.get("launch_angle")),
        ci(row.get("launch_speed_angle")), row.get("bb_type"), row.get("events"), row.get("description"), cf(row.get("hc_x")), cf(row.get("hc_y"))
    )


def insert_batch(conn: sqlite3.Connection, batch: list[tuple]) -> int:
    if not batch: return 0
    before = conn.total_changes
    conn.executemany("""
    INSERT OR IGNORE INTO statcast_bbe (
      bbe_key, game_id, game_date, at_bat_number, pitch_number, batter_id, pitcher_id, stand, p_throws,
      launch_speed, launch_angle, launch_speed_angle, bb_type, events, description, hc_x, hc_y
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, batch)
    conn.commit()
    return conn.total_changes - before


def ingest(conn: sqlite3.Connection, session: requests.Session, start: dt.date, end: dt.date, args) -> dict:
    print(f"\nFETCH {start} to {end}", flush=True)
    seen = bbe = inserted = 0
    batch = []
    with session.get(SAVANT_CSV_URL, params=params(start, end), timeout=args.timeout, stream=True) as r:
        print(f"status={r.status_code}", flush=True)
        if r.status_code != 200:
            return {"seen": 0, "bbe": 0, "inserted": 0}
        reader = csv.DictReader((line.decode("utf-8", errors="ignore") for line in r.iter_lines() if line))
        for row in reader:
            seen += 1
            if args.max_lines and seen > args.max_lines:
                print(f"max_lines reached: {args.max_lines}; stopping this chunk", flush=True)
                break
            if not is_bbe(row): continue
            p = payload(row)
            if p is None: continue
            bbe += 1
            batch.append(p)
            if len(batch) >= args.batch_size:
                inserted += insert_batch(conn, batch); batch = []
                print(f"  progress rows_seen={seen} bbe_seen={bbe} inserted={inserted}", flush=True)
    if batch: inserted += insert_batch(conn, batch)
    return {"seen": seen, "bbe": bbe, "inserted": inserted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db()))
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--chunk-days", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--max-lines", type=int, default=50000)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()
    db = Path(args.db); db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db); ensure_schema(conn)
    session = requests.Session(); session.headers.update({"User-Agent": USER_AGENT})
    totals = {"seen": 0, "bbe": 0, "inserted": 0}
    for s, e in chunks(parse_date(args.start_date), parse_date(args.end_date), args.chunk_days):
        st = ingest(conn, session, s, e, args)
        for k in totals: totals[k] += st[k]
        time.sleep(args.sleep)
    total = conn.execute("SELECT COUNT(*) FROM statcast_bbe").fetchone()[0]
    conn.close()
    print("\nHR STATCAST BBE LOADER A")
    print("========================")
    print(f"db: {db}")
    print(f"rows_seen: {totals['seen']}")
    print(f"bbe_seen: {totals['bbe']}")
    print(f"inserted_new: {totals['inserted']}")
    print(f"statcast_bbe_total: {total}")
    print("DONE")

if __name__ == "__main__":
    main()
