#!/usr/bin/env python3
"""
CFB_PASS_RECV_EVENT_EXTRACTION_A

Same idea as cfb_rushing_yards_carry_extraction_a.py (real per-event
outcomes for a Monte Carlo simulator, not a pre-aggregated rate) applied
to the other 3 FBS markets: receiving_yards, passing_yards, passing_
touchdowns. One pass over each season's play-level player_stats CSV
writes two new event tables into the same cfb_carry_log.sqlite used for
rushing:

  recv_catches       one row per real reception: (player, game, catch
                     index, yards) -- identical shape to rush_carries,
                     drives the receiving_yards simulator via the exact
                     same cfb_rush_sim.simulate() function (the bootstrap
                     math -- sample event count from a recent game, sample
                     each event's yards from a pooled recent-outcome set
                     -- doesn't care whether the event is a carry or a
                     catch).

  pass_attempts_log  one row per real pass attempt (completions,
                     incompletions, AND interceptions -- all 3 are real
                     attempts for passing_yards purposes): (player, game,
                     attempt index, yards, is_touchdown). Incompletions
                     and interceptions get yards=0 (verified against the
                     existing shipped passing_yards aggregate logic in
                     cfb_player_games_foundation_a.py, which only adds
                     completion_yds -- zero contribution from any other
                     attempt outcome). is_touchdown is 1 only when a
                     completion_player_id row ALSO carries a
                     touchdown_player_id (same inference already
                     validated there against Dillon Gabriel's real 2024
                     stat line).

Same FBS-vs-FBS regular-season game filter as cfb_player_games_
foundation_a.py / cfb_rushing_yards_carry_extraction_a.py.

Run
---
python -u cfb_pass_recv_event_extraction_a.py --season 2022 2023 2024 2025 \
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
CREATE TABLE IF NOT EXISTS recv_catches (
    player_id TEXT NOT NULL,
    player_name TEXT,
    team TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT NOT NULL,
    catch_index INTEGER NOT NULL,
    yards INTEGER NOT NULL,
    PRIMARY KEY (player_id, season, game_id, catch_index)
);
CREATE INDEX IF NOT EXISTS idx_catches_player_season ON recv_catches(player_id, season, week);

CREATE TABLE IF NOT EXISTS pass_attempts_log (
    player_id TEXT NOT NULL,
    player_name TEXT,
    team TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT NOT NULL,
    attempt_index INTEGER NOT NULL,
    yards INTEGER NOT NULL,
    is_touchdown INTEGER NOT NULL,
    PRIMARY KEY (player_id, season, game_id, attempt_index)
);
CREATE INDEX IF NOT EXISTS idx_attempts_player_season ON pass_attempts_log(player_id, season, week);
"""


def to_int(v):
    try:
        return int(round(float(v)))
    except Exception:
        return None


def load_games(schedule_path):
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


def flush(conn, table, cols, batch):
    if not batch:
        return
    placeholders = ", ".join(f":{c}" for c in cols)
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
        batch)


def extract_season(ps_path, games, season, conn):
    n_rows = n_matched = n_catches = n_attempts = 0
    catch_counters = {}   # (player_id, game_id) -> next catch_index
    attempt_counters = {}  # (player_id, game_id) -> next attempt_index
    catch_batch = []
    attempt_batch = []
    recv_cols = ["player_id", "player_name", "team", "season", "week",
                 "game_id", "catch_index", "yards"]
    pass_cols = ["player_id", "player_name", "team", "season", "week",
                 "game_id", "attempt_index", "yards", "is_touchdown"]

    with open(ps_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n_rows += 1
            gid = r.get("game_id")
            week = games.get(gid)
            if week is None:
                continue
            n_matched += 1
            team = r.get("team")

            recid = r.get("reception_player_id")
            if recid and recid != "NA":
                yards = to_int(r.get("reception_yds"))
                if yards is not None:
                    key = (recid, gid)
                    idx = catch_counters.get(key, 0)
                    catch_counters[key] = idx + 1
                    catch_batch.append({
                        "player_id": recid, "player_name": r.get("reception_player"),
                        "team": team, "season": season, "week": week,
                        "game_id": gid, "catch_index": idx, "yards": yards,
                    })
                    n_catches += 1

            has_td = bool(r.get("touchdown_player_id") and r.get("touchdown_player_id") != "NA")

            cpid = r.get("completion_player_id")
            icpid = r.get("incompletion_player_id")
            itpid = r.get("interception_thrown_player_id")
            attempt_pid, attempt_name, attempt_yards, attempt_td = None, None, 0, 0
            if cpid and cpid != "NA":
                attempt_pid, attempt_name = cpid, r.get("completion_player")
                attempt_yards = to_int(r.get("completion_yds")) or 0
                attempt_td = 1 if has_td else 0
            elif icpid and icpid != "NA":
                attempt_pid, attempt_name = icpid, r.get("incompletion_player")
            elif itpid and itpid != "NA":
                attempt_pid, attempt_name = itpid, r.get("interception_thrown_player")

            if attempt_pid:
                key = (attempt_pid, gid)
                idx = attempt_counters.get(key, 0)
                attempt_counters[key] = idx + 1
                attempt_batch.append({
                    "player_id": attempt_pid, "player_name": attempt_name,
                    "team": team, "season": season, "week": week,
                    "game_id": gid, "attempt_index": idx, "yards": attempt_yards,
                    "is_touchdown": attempt_td,
                })
                n_attempts += 1

            if len(catch_batch) >= 20000:
                flush(conn, "recv_catches", recv_cols, catch_batch)
                catch_batch = []
            if len(attempt_batch) >= 20000:
                flush(conn, "pass_attempts_log", pass_cols, attempt_batch)
                attempt_batch = []

    flush(conn, "recv_catches", recv_cols, catch_batch)
    flush(conn, "pass_attempts_log", pass_cols, attempt_batch)
    conn.commit()
    print(f"  season {season}: {n_rows} plays read, {n_matched} in FBS-vs-FBS games "
          f"-> {n_catches} catch rows, {n_attempts} pass-attempt rows")


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

    print("CFB_PASS_RECV_EVENT_EXTRACTION_A\n=================================")
    for season in args.season:
        sched_path = raw_dir / f"schedules_{season}.csv"
        ps_path = raw_dir / f"player_stats_{season}.csv"
        if not sched_path.exists() or not ps_path.exists():
            print(f"  season {season}: SKIP -- missing {sched_path} or {ps_path}")
            continue
        games = load_games(sched_path)
        print(f"  season {season}: {len(games)} FBS-vs-FBS games in schedule")
        extract_season(ps_path, games, season, conn)

    n_catches = conn.execute("SELECT COUNT(*) FROM recv_catches").fetchone()[0]
    n_catch_players = conn.execute("SELECT COUNT(DISTINCT player_id) FROM recv_catches").fetchone()[0]
    n_attempts = conn.execute("SELECT COUNT(*) FROM pass_attempts_log").fetchone()[0]
    n_attempt_players = conn.execute("SELECT COUNT(DISTINCT player_id) FROM pass_attempts_log").fetchone()[0]
    conn.close()
    print(f"\ntotal: {n_catches} catch rows ({n_catch_players} players), "
          f"{n_attempts} pass-attempt rows ({n_attempt_players} players) -- db: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
