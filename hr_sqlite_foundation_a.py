#!/usr/bin/env python3
"""Create /data/hr_model/hr_model.sqlite and import existing batter-game CSV."""
from __future__ import annotations
import argparse, csv, os, sqlite3
from pathlib import Path
from typing import Any, Optional


def model_dir() -> Path:
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    if not p.parent.exists():
        p = Path("./hr_model")
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_default() -> Path:
    return model_dir() / "hr_model.sqlite"


def ci(x: Any, default: Optional[int]=None) -> Optional[int]:
    try:
        if x in (None, "", "None"): return default
        return int(float(x))
    except Exception:
        return default


def cf(x: Any, default: Optional[float]=None) -> Optional[float]:
    try:
        if x in (None, "", "None"): return default
        return float(x)
    except Exception:
        return default


def cb(x: Any) -> Optional[int]:
    if x in (None, "", "None"): return None
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}: return 1
    if s in {"false", "0", "no"}: return 0
    return None


def val(r: dict, *names: str):
    for n in names:
        if n in r and r[n] not in (None, ""):
            return r[n]
    return None


SCHEMA = r"""
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS batter_games (
    game_id TEXT NOT NULL,
    game_date TEXT NOT NULL,
    side TEXT,
    team TEXT,
    opponent TEXT,
    venue TEXT,
    park_bucket TEXT,
    hitterish_park INTEGER,
    pitcherish_park INTEGER,
    weather_condition TEXT,
    temp_f REAL,
    wind_speed_mph REAL,
    weather_wind TEXT,
    weather_wind_bucket TEXT,
    batter_id INTEGER NOT NULL,
    batter_name TEXT,
    lineup_spot INTEGER,
    batter_hand TEXT,
    opposing_pitcher_id INTEGER,
    opposing_pitcher_name TEXT,
    pitcher_hand TEXT,
    platoon_bucket TEXT,
    wind_toward_pull_field INTEGER,
    park_factor_by_batter_hand REAL,
    plate_appearances INTEGER,
    at_bats INTEGER,
    hits INTEGER,
    doubles INTEGER,
    triples INTEGER,
    home_runs INTEGER,
    rbi INTEGER,
    walks INTEGER,
    strikeouts INTEGER,
    actual_hr INTEGER NOT NULL DEFAULT 0,
    actual_hr_count INTEGER DEFAULT 0,
    PRIMARY KEY (game_id, batter_id)
);

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

CREATE TABLE IF NOT EXISTS hr_odds_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_date TEXT NOT NULL,
    game_id TEXT,
    player_id INTEGER,
    player_name TEXT,
    sportsbook TEXT,
    market TEXT,
    line REAL,
    american_odds REAL,
    decimal_odds REAL,
    implied_prob REAL,
    captured_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_batter_games_date_batter ON batter_games(game_date, batter_id);
CREATE INDEX IF NOT EXISTS idx_batter_games_date_pitcher ON batter_games(game_date, opposing_pitcher_id);
CREATE INDEX IF NOT EXISTS idx_statcast_batter_date ON statcast_bbe(batter_id, game_date);
CREATE INDEX IF NOT EXISTS idx_statcast_pitcher_date ON statcast_bbe(pitcher_id, game_date);
CREATE INDEX IF NOT EXISTS idx_statcast_game ON statcast_bbe(game_id);
CREATE INDEX IF NOT EXISTS idx_statcast_date ON statcast_bbe(game_date);
"""


def ensure_schema(conn: sqlite3.Connection):
    conn.executescript(SCHEMA)
    conn.commit()


def import_csv(conn: sqlite3.Connection, path: Path) -> int:
    if not path.exists(): raise FileNotFoundError(path)
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    sql = """
    INSERT OR REPLACE INTO batter_games (
        game_id, game_date, side, team, opponent, venue, park_bucket,
        hitterish_park, pitcherish_park, weather_condition, temp_f, wind_speed_mph,
        weather_wind, weather_wind_bucket, batter_id, batter_name, lineup_spot, batter_hand,
        opposing_pitcher_id, opposing_pitcher_name, pitcher_hand, platoon_bucket,
        wind_toward_pull_field, park_factor_by_batter_hand, plate_appearances, at_bats,
        hits, doubles, triples, home_runs, rbi, walks, strikeouts, actual_hr, actual_hr_count
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    payload = []
    for r in rows:
        game_id = str(val(r, "game_id", "game_pk") or "")
        batter_id = ci(val(r, "batter_id", "player_id"))
        if not game_id or batter_id is None: continue
        home_runs = ci(val(r, "home_runs"), 0) or 0
        payload.append((
            game_id, str(val(r, "game_date") or ""), val(r, "side"), val(r, "team"), val(r, "opponent"),
            val(r, "venue"), val(r, "park_bucket"), cb(val(r, "hitterish_park")), cb(val(r, "pitcherish_park")),
            val(r, "weather_condition"), cf(val(r, "temp_f", "weather_temp_f")), cf(val(r, "wind_speed_mph", "weather_wind_mph")),
            val(r, "weather_wind"), val(r, "weather_wind_bucket"), batter_id,
            val(r, "batter_name", "player", "player_name"), ci(val(r, "lineup_spot")), val(r, "batter_hand", "batter_side"),
            ci(val(r, "opposing_pitcher_id", "opp_pitcher_id")), val(r, "opposing_pitcher_name", "opp_pitcher"),
            val(r, "pitcher_hand", "opposing_pitcher_hand", "opp_pitcher_hand"), val(r, "platoon_bucket"),
            cb(val(r, "wind_toward_pull_field")), cf(val(r, "park_factor_by_batter_hand")), ci(val(r, "plate_appearances")),
            ci(val(r, "at_bats")), ci(val(r, "hits")), ci(val(r, "doubles")), ci(val(r, "triples")), home_runs,
            ci(val(r, "rbi")), ci(val(r, "walks")), ci(val(r, "strikeouts")), ci(val(r, "actual_hr"), 1 if home_runs > 0 else 0) or 0,
            ci(val(r, "actual_hr_count"), home_runs) or home_runs,
        ))
    conn.executemany(sql, payload)
    conn.commit()
    return len(payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(db_default()))
    ap.add_argument("--import-csv", default=None)
    args = ap.parse_args()
    db = Path(args.db); db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    ensure_schema(conn)
    imported = import_csv(conn, Path(args.import_csv)) if args.import_csv else 0
    print("HR SQLITE FOUNDATION A")
    print("======================")
    print(f"db: {db}")
    print(f"imported_batter_games: {imported}")
    for t in ["batter_games", "statcast_bbe", "hr_odds_snapshots"]:
        print(f"{t}: {conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]}")
    conn.close()
    print("DONE")

if __name__ == "__main__":
    main()
