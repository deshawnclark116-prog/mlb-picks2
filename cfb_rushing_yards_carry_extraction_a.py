#!/usr/bin/env python3
"""
CFB_RUSHING_YARDS_CARRY_EXTRACTION_A

Prerequisite for a real Monte Carlo rushing-yards simulator (the user
asked for CFB models "almost just like the mlb model" -- ksim.py, which
sims a full outcome distribution instead of just a binary over/under
classifier). ksim.py doesn't actually have real per-plate-appearance
outcome data (strikeout-or-not is simulated from an assumed Bernoulli
rate); CFB's raw source turns out to have something BETTER for this
purpose: cfbfastR-data's player_stats CSV is genuinely PLAY-LEVEL, one
row per play, with rush_yds giving the EXACT yards gained on that
specific carry (confirmed directly: Mark Fletcher Jr.'s 2025 week-1 log
extracted here is [3,5,-3,15,3,2,2,11,9,3,6,6,1,1,2], 15 real carries
summing to exactly 66 yards -- matches his known game-level total
exactly). That means the simulator can bootstrap-resample from a
player's own REAL recorded per-carry outcomes instead of needing an
assumed parametric distribution -- a more direct fit for a continuous-
ish per-play outcome than ksim's binary-event model needed for
strikeouts (which is the RIGHT model for that domain, not a limitation
being worked around here -- just a different domain needing a different
technique).

Writes one row per real carry: (player_id, season, week, game_id,
carry_index, yards) -- a genuine play-level log, not a re-aggregation.
Position-filtered to RB only for now (this market's pilot); same
FBS-vs-FBS game scoping as cfb_player_games_foundation_a.py.

Run
---
python -u cfb_rushing_yards_carry_extraction_a.py --season 2022 2023 2024 2025 \
    --raw-dir /data/cfb_raw --db cfb_models/cfb_carry_log.sqlite
"""
import argparse
import csv
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS rush_carries (
    player_id TEXT NOT NULL,
    player_name TEXT,
    team TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT NOT NULL,
    carry_index INTEGER NOT NULL,
    yards INTEGER NOT NULL,
    PRIMARY KEY (player_id, season, game_id, carry_index)
);
CREATE INDEX IF NOT EXISTS idx_carries_player_season ON rush_carries(player_id, season, week);
"""


def to_int(v):
    try:
        return int(round(float(v)))
    except Exception:
        return None


def load_games(schedule_path):
    """Same FBS-vs-FBS game_id filter as cfb_player_games_foundation_a.py --
    mirrors its own schedules-file parsing so the population is identical.

    Real bug found and fixed here (caught before this script ever shipped
    anywhere, not present in the original foundation script): postseason
    games get their OWN week numbering starting back at 1 in cfbfastR's
    schedule data (confirmed directly: Mark Fletcher Jr.'s "week 1 2025"
    query returned 5 different games before this fix -- one real regular-
    season week-1 game against Notre Dame, plus 4 real POSTSEASON games
    against Texas A&M/Ohio State/Ole Miss/Indiana, all also labeled
    week=1). cfb_player_games_foundation_a.py already filters
    season_type == "regular" for exactly this reason; this script didn't
    have that filter yet."""
    games = {}
    with open(schedule_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("home_division") or "").lower() != "fbs":
                continue
            if (r.get("away_division") or "").lower() != "fbs":
                continue
            if r.get("season_type") != "regular":
                continue
            gid = r.get("game_id")
            week = to_int(r.get("week"))
            if not gid or week is None:
                continue
            games[gid] = week
    return games


def extract_season(ps_path, games, season, conn):
    n_rows = n_matched = n_carries = 0
    counters = {}  # (player_id, game_id) -> next carry_index
    batch = []
    with open(ps_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n_rows += 1
            gid = r.get("game_id")
            week = games.get(gid)
            if week is None:
                continue
            rid = r.get("rush_player_id")
            if not rid or rid == "NA":
                continue
            yards = to_int(r.get("rush_yds"))
            if yards is None:
                continue
            n_matched += 1
            key = (rid, gid)
            idx = counters.get(key, 0)
            counters[key] = idx + 1
            batch.append({
                "player_id": rid, "player_name": r.get("rush_player"),
                "team": r.get("team"), "season": season, "week": week,
                "game_id": gid, "carry_index": idx, "yards": yards,
            })
            n_carries += 1
            if len(batch) >= 20000:
                conn.executemany(
                    "INSERT OR REPLACE INTO rush_carries (player_id, player_name, team, "
                    "season, week, game_id, carry_index, yards) VALUES (:player_id, "
                    ":player_name, :team, :season, :week, :game_id, :carry_index, :yards)",
                    batch)
                batch = []
    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO rush_carries (player_id, player_name, team, "
            "season, week, game_id, carry_index, yards) VALUES (:player_id, "
            ":player_name, :team, :season, :week, :game_id, :carry_index, :yards)",
            batch)
    conn.commit()
    print(f"  season {season}: {n_rows} plays read, {n_matched} rush plays in FBS-vs-FBS games "
          f"-> {n_carries} carry rows written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, nargs="+", required=True)
    ap.add_argument("--raw-dir", default="/data/cfb_raw")
    ap.add_argument("--db", required=True)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)

    print("CFB_RUSHING_YARDS_CARRY_EXTRACTION_A\n=====================================")
    for season in args.season:
        sched_path = raw_dir / f"schedules_{season}.csv"
        ps_path = raw_dir / f"player_stats_{season}.csv"
        if not sched_path.exists() or not ps_path.exists():
            print(f"  season {season}: SKIP -- missing {sched_path} or {ps_path}")
            continue
        games = load_games(sched_path)
        print(f"  season {season}: {len(games)} FBS-vs-FBS games in schedule")
        extract_season(ps_path, games, season, conn)

    n_total = conn.execute("SELECT COUNT(*) FROM rush_carries").fetchone()[0]
    n_players = conn.execute("SELECT COUNT(DISTINCT player_id) FROM rush_carries").fetchone()[0]
    conn.close()
    print(f"\ntotal: {n_total} carry rows, {n_players} distinct players -- db: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
