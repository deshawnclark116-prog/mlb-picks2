#!/usr/bin/env python3
"""
NFL_PBP_FOUNDATION_A

Real, per-PLAY NFL data (nflverse's play-by-play release -- verified
live: https://github.com/nflverse/nflverse-data/releases/download/pbp/
play_by_play_{season}.csv.gz, real columns rusher_player_id/rushing_yards
per rush attempt and receiver_player_id/receiving_yards per target,
same gsis_id player_id namespace nfl_player_games_foundation_a.py
already uses -- direct join, no crosswalk needed). This is the NFL
analog of cfbfastR-data's already-play-level player_stats that CFB's
carry-log/simulator system is built on (see cfb_rush_sim.py /
cfb_rushing_yards_carry_extraction_a.py) -- NFL's own weekly stats feed
(stats_player_week_*.csv) is only game-level aggregates, so a per-touch
empirical outcome pool needs this separate, real PBP source.

Extracts two per-event pools into nfl_models/nfl_carry_log.sqlite:

  rush_carries   one row per real rush attempt (rush_attempt==1 AND a
                 rusher_player_id present), yards = that carry's real
                 rushing_yards.
  recv_targets   one row per real target (play_type=='pass' AND a
                 receiver_player_id present), yards = real receiving_
                 yards on a completion, 0 on an incompletion -- workload
                 is TARGETS, not receptions, matching NFL's already-
                 shipped receiving_yards classifier's own volume feature
                 (recent3_avg_targets), not CFB's reception-based
                 convention. An incompletion is a real 0-yard outcome
                 for that target, not a missing observation -- excluding
                 it would systematically overstate every target's
                 expected yards.

This script ONLY loads raw per-play pools -- no feature engineering, no
eligibility logic, no model, mirroring the separation used throughout
this repo (nfl_player_games_foundation_a.py is the same shape for the
game-level source).

Run:
    python -u nfl_pbp_foundation_a.py --seasons 2022 2023 2024 \
        --db nfl_models/nfl_carry_log.sqlite
"""
import argparse
import csv
import gzip
import io
import sqlite3
import sys
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
DEFAULT_DB = Path("nfl_models/nfl_carry_log.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS rush_carries (
    player_id TEXT NOT NULL,
    player_name TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT NOT NULL,
    carry_index INTEGER NOT NULL,
    yards REAL
);
CREATE INDEX IF NOT EXISTS idx_rc_player_season ON rush_carries(player_id, season, week);
CREATE INDEX IF NOT EXISTS idx_rc_game ON rush_carries(game_id);

CREATE TABLE IF NOT EXISTS recv_targets (
    player_id TEXT NOT NULL,
    player_name TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT NOT NULL,
    target_index INTEGER NOT NULL,
    yards REAL
);
CREATE INDEX IF NOT EXISTS idx_rt_player_season ON recv_targets(player_id, season, week);
CREATE INDEX IF NOT EXISTS idx_rt_game ON recv_targets(game_id);
"""


def fetch_season_csv_rows(season, local_gz=None):
    if local_gz:
        with gzip.open(local_gz, "rt", newline="") as f:
            yield from csv.DictReader(f)
        return
    url = PBP_URL.format(season=season)
    print(f"  fetching {url} ...", flush=True)
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    with gzip.open(io.BytesIO(r.content), "rt", newline="") as f:
        yield from csv.DictReader(f)


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def build(seasons, db_path, local_gz_template=None):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.executescript(SCHEMA)

    for season in seasons:
        local_gz = local_gz_template.format(season=season) if local_gz_template else None
        con.execute("DELETE FROM rush_carries WHERE season=?", (season,))
        con.execute("DELETE FROM recv_targets WHERE season=?", (season,))
        n_rush = n_recv = 0
        carry_idx_by_game = {}
        target_idx_by_game = {}
        for row in fetch_season_csv_rows(season, local_gz):
            try:
                week = int(row.get("week") or 0)
            except Exception:
                continue
            game_id = row.get("game_id")
            if not game_id or not week:
                continue

            if row.get("rush_attempt") == "1" and row.get("rusher_player_id"):
                pid = row["rusher_player_id"]
                yards = _f(row.get("rushing_yards"))
                if yards is None:
                    continue
                key = (pid, game_id)
                carry_idx_by_game[key] = carry_idx_by_game.get(key, 0) + 1
                con.execute(
                    "INSERT INTO rush_carries (player_id, player_name, season, week, game_id, "
                    "carry_index, yards) VALUES (?,?,?,?,?,?,?)",
                    (pid, row.get("rusher_player_name"), season, week, game_id,
                     carry_idx_by_game[key], yards))
                n_rush += 1

            if row.get("play_type") == "pass" and row.get("receiver_player_id"):
                pid = row["receiver_player_id"]
                yards = _f(row.get("receiving_yards")) if row.get("complete_pass") == "1" else 0.0
                key = (pid, game_id)
                target_idx_by_game[key] = target_idx_by_game.get(key, 0) + 1
                con.execute(
                    "INSERT INTO recv_targets (player_id, player_name, season, week, game_id, "
                    "target_index, yards) VALUES (?,?,?,?,?,?,?)",
                    (pid, row.get("receiver_player_name"), season, week, game_id,
                     target_idx_by_game[key], yards))
                n_recv += 1

        con.commit()
        print(f"  season {season}: {n_rush} rush-carry rows, {n_recv} target rows")

    total_rush = con.execute("SELECT COUNT(*) FROM rush_carries").fetchone()[0]
    total_recv = con.execute("SELECT COUNT(*) FROM recv_targets").fetchone()[0]
    print(f"\ndone: {total_rush} total rush carries, {total_recv} total targets -> {db_path}")
    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--local-gz-template", default=None,
                     help="Optional local path template with {season}, e.g. /tmp/pbp_{season}.csv.gz "
                          "-- bypasses the live fetch (used for local dev/testing).")
    args = ap.parse_args()

    print("NFL_PBP_FOUNDATION_A\n=====================")
    print(f"seasons: {args.seasons}\n")
    build(args.seasons, args.db, args.local_gz_template)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
