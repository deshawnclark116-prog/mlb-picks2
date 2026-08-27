#!/usr/bin/env python3
"""
CFB_PLAYER_GAMES_FOUNDATION_A

Data foundation for a new college football pipeline -- the CFB analog of
nfl_player_games_foundation_a.py. Builds cfb_model.sqlite with two tables:

    player_games   one row per player per game (aggregated rushing/
                    receiving stat lines, joined to schedule)
    games           one row per FBS-vs-FBS game (schedule/matchup context)

Source: cfbfastR-data (github.com/sportsdataverse/cfbfastR-data), the free,
actively maintained college football data project analogous to nflverse --
no API key needed, plain CSV files served from raw.githubusercontent.com.
Three files per season:
    player_stats   PLAY-LEVEL rows with per-play player attribution
                   (rush_player_id/rush_yds, reception_player_id/
                   reception_yds, etc.) -- NOT pre-aggregated like nflverse's
                   stats_player_week, so this script aggregates it into
                   per-player-per-game totals itself (confirmed via a real
                   header/row inspection before writing this, not assumed).
    rosters        athlete_id -> position, one row per player per season.
    schedules      one row per game: teams, division, home/away, points,
                   date. Division matters a LOT here: college football
                   spans FBS/FCS/II/III, and a real header check on 2024
                   showed only 920 of 3801 scheduled games are FBS-vs-FBS
                   (the rest include massive-mismatch games against
                   lower-division opponents) -- scoped to FBS-vs-FBS only,
                   the same population real CFB player props are offered on.

Only rushing_yards (RB) and receiving_yards (WR) are built here, mirroring
exactly which two markets came first for the NFL build in this repo.
Passing/QB props are a deliberate non-goal for this first pass.

NOTE (found by inspecting the raw data, not assumed): target_player_id is
NOT a usable "pass attempted at this receiver" signal in this dataset --
checked directly against a real season: populated on 0 of 54,800 real
completions and only ~21% of incompletions. So there is no "targets"
column here at all (unlike the NFL build, which used targets as a feature)
-- receptions is used as the opportunity/volume proxy instead.

This script is the data-loading step ONLY -- no feature engineering, no
models, no strict-D1 logic (that belongs in a clean-baseline script
downstream, same division of labor nfl_player_games_foundation_a.py uses).

Run
---
python -u cfb_player_games_foundation_a.py --season 2022 2023 2024 2025
python -u cfb_player_games_foundation_a.py --season 2022 2023 2024 2025 \
    --raw-dir /data/cfb_raw --db cfb_models/cfb_model.sqlite
"""
import argparse
import csv
import sqlite3
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

BASE_URL = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main"
DEFAULT_RAW_DIR = Path("/data/cfb_raw")
DEFAULT_DB = Path("/data/cfb_model/cfb_model.sqlite")
UA = {"User-Agent": "cfb-foundation/1.0"}

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_date TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_points INTEGER,
    away_points INTEGER,
    neutral_site INTEGER
);
CREATE INDEX IF NOT EXISTS idx_games_season_week ON games(season, week);

CREATE TABLE IF NOT EXISTS player_games (
    player_id TEXT NOT NULL,
    player_name TEXT,
    position TEXT,
    team TEXT NOT NULL,
    opponent TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT,
    game_date TEXT,
    is_home INTEGER,
    carries INTEGER,
    rushing_yards INTEGER,
    receptions INTEGER,
    receiving_yards INTEGER,
    PRIMARY KEY (player_id, season, week, game_id)
);
CREATE INDEX IF NOT EXISTS idx_pg_player_season ON player_games(player_id, season, week);
CREATE INDEX IF NOT EXISTS idx_pg_team_week ON player_games(team, season, week);
CREATE INDEX IF NOT EXISTS idx_pg_game ON player_games(game_id);
"""


def _fetch(url, dest, timeout=180):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        f.write(r.read())


def _local_or_fetch(raw_dir, kind, filename_fmt, season, timeout=180):
    path = raw_dir / filename_fmt.format(season=season)
    if path.exists():
        return path
    subdir = {"player_stats": "player_stats/csv", "rosters": "rosters/csv",
              "schedules": "schedules/csv"}[kind]
    url = f"{BASE_URL}/{subdir}/{path.name}"
    print(f"  fetching {url} ...", flush=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    _fetch(url, path, timeout=timeout)
    return path


def to_int(v):
    try:
        if v is None or v == "" or v == "NA":
            return None
        return int(round(float(v)))
    except Exception:
        return None


def load_schedules(path, season):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    games = {}
    for r in rows:
        if r.get("home_division") != "fbs" or r.get("away_division") != "fbs":
            continue
        if r.get("season_type") != "regular":
            continue  # postseason/bowls handled separately if ever needed; regular season only for now
        gid = r["game_id"]
        games[gid] = {
            "game_id": gid, "season": int(r["season"]), "week": to_int(r["week"]),
            "game_date": (r.get("start_date") or "")[:10],
            "home_team": r["home_team"], "away_team": r["away_team"],
            "home_points": to_int(r.get("home_points")), "away_points": to_int(r.get("away_points")),
            "neutral_site": 1 if r.get("neutral_site") == "TRUE" else 0,
        }
    return games


def load_positions(path, season):
    rows = csv.DictReader(open(path, newline="", encoding="utf-8"))
    pos = {}
    for r in rows:
        aid = r.get("athlete_id")
        p = r.get("position")
        if aid and p:
            pos[aid] = p
    return pos


def aggregate_player_stats(path, games):
    """Play-level rows -> per-(game_id, player_id) rushing/receiving totals.
    Only plays belonging to an FBS-vs-FBS game (already filtered into
    `games`) are counted."""
    rush = defaultdict(lambda: {"carries": 0, "rushing_yards": 0, "name": None, "team": None})
    recv = defaultdict(lambda: {"receptions": 0, "receiving_yards": 0, "name": None, "team": None})
    n_rows = 0
    n_matched = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n_rows += 1
            gid = r.get("game_id")
            if gid not in games:
                continue
            n_matched += 1
            team = r.get("team")
            rid = r.get("rush_player_id")
            if rid and rid != "NA":
                key = (gid, rid)
                d = rush[key]
                d["carries"] += 1
                d["rushing_yards"] += to_int(r.get("rush_yds")) or 0
                d["name"] = r.get("rush_player") or d["name"]
                d["team"] = team
            recid = r.get("reception_player_id")
            if recid and recid != "NA":
                key = (gid, recid)
                d = recv[key]
                d["receptions"] += 1
                d["receiving_yards"] += to_int(r.get("reception_yds")) or 0
                d["name"] = r.get("reception_player") or d["name"]
                d["team"] = team
    print(f"    {n_rows} plays read, {n_matched} in an FBS-vs-FBS game")
    return rush, recv


def build_player_games(games, rush, recv, positions, season):
    out = {}
    all_keys = set(rush.keys()) | set(recv.keys())
    for (gid, pid) in all_keys:
        g = games.get(gid)
        if g is None:
            continue
        r = rush.get((gid, pid), {})
        v = recv.get((gid, pid), {})
        team = r.get("team") or v.get("team")
        name = r.get("name") or v.get("name")
        if team == g["home_team"]:
            opponent, is_home = g["away_team"], 1
        elif team == g["away_team"]:
            opponent, is_home = g["home_team"], 0
        else:
            continue  # team name mismatch between player_stats and schedules -- skip rather than guess
        out[(pid, season, g["week"], gid)] = {
            "player_id": pid, "player_name": name, "position": positions.get(pid),
            "team": team, "opponent": opponent, "season": season, "week": g["week"],
            "game_id": gid, "game_date": g["game_date"], "is_home": is_home,
            "carries": r.get("carries", 0), "rushing_yards": r.get("rushing_yards", 0),
            "receptions": v.get("receptions", 0),
            "receiving_yards": v.get("receiving_yards", 0),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, nargs="+", required=True)
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    ap.add_argument("--db", default=str(DEFAULT_DB))
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("CFB_PLAYER_GAMES_FOUNDATION_A\n=============================")
    print(f"seasons: {args.season}")

    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)

    all_games = {}
    all_pg = {}
    for season in args.season:
        print(f"\nseason {season}")
        sched_path = _local_or_fetch(raw_dir, "schedules", "schedules_{season}.csv", season)
        games = load_schedules(sched_path, season)
        print(f"  {len(games)} FBS-vs-FBS regular season games")

        ros_path = _local_or_fetch(raw_dir, "rosters", "rosters_{season}.csv", season)
        positions = load_positions(ros_path, season)
        print(f"  {len(positions)} players with a known position")

        ps_path = _local_or_fetch(raw_dir, "player_stats", "player_stats_{season}.csv", season)
        rush, recv = aggregate_player_stats(ps_path, games)

        pg = build_player_games(games, rush, recv, positions, season)
        print(f"  {len(pg)} player-game rows built")

        all_games.update(games)
        all_pg.update(pg)

    conn.executemany(
        "INSERT OR REPLACE INTO games (game_id, season, week, game_date, home_team, "
        "away_team, home_points, away_points, neutral_site) VALUES "
        "(:game_id, :season, :week, :game_date, :home_team, :away_team, :home_points, "
        ":away_points, :neutral_site)",
        list(all_games.values()))

    conn.executemany(
        "INSERT OR REPLACE INTO player_games (player_id, player_name, position, team, "
        "opponent, season, week, game_id, game_date, is_home, carries, rushing_yards, "
        "receptions, receiving_yards) VALUES (:player_id, :player_name, "
        ":position, :team, :opponent, :season, :week, :game_id, :game_date, :is_home, "
        ":carries, :rushing_yards, :receptions, :receiving_yards)",
        list(all_pg.values()))
    conn.commit()

    n_games = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    n_pg = conn.execute("SELECT COUNT(*) FROM player_games").fetchone()[0]
    by_season = conn.execute(
        "SELECT season, COUNT(*), SUM(CASE WHEN position='RB' THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN position='WR' THEN 1 ELSE 0 END) "
        "FROM player_games GROUP BY season ORDER BY season").fetchall()
    conn.close()

    print(f"\nDB WRITTEN: {db_path}")
    print(f"  games: {n_games}   player_games: {n_pg}")
    print(f"  {'season':8s}{'rows':>8s}{'RB rows':>10s}{'WR rows':>10s}")
    for s, n, rb, wr in by_season:
        print(f"  {s:<8}{n:>8}{rb:>10}{wr:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
