#!/usr/bin/env python3
"""
NHL_PLAYER_GAMES_FOUNDATION_A

Real per-game skater and goalie stat lines, pulled from the NHL's own
public bulk stats REST API (api.nhle.com/stats/rest/en/{skater,goalie}/
summary?isGame=true) -- the same no-auth backend nhl_live_foundation_a.py
already uses for team schedules/scores, a different endpoint family.
isGame=true returns one row PER PLAYER PER GAME (not a season aggregate)
-- confirmed by direct inspection (gameId/gameDate present on every row).

That endpoint hard-caps at 10,000 rows per query (confirmed empirically:
start=15000 on an unfiltered full-season query returns 0 rows even
though a full season has far more skater-game rows than that) -- so this
fetches one TEAM-SEASON at a time (cayenneExp teamAbbrevs filter), same
team-loop shape as nhl_live_foundation_a.py's club-schedule-season calls.
A team-season is ~1400-1500 skater-game rows, safely under the cap.

Two tables, written into the SAME nhl_model.sqlite the games table
already lives in (so games.home_abbrev/away_abbrev join cleanly against
these tables' team/opponent columns, which are abbreviations, not the
games table's home_team/away_team full display names):

  skater_games   one row per skater per game: goals, assists, points,
                 shots, time-on-ice. All skater positions (C/L/R/D) --
                 real book point/shot props aren't position-restricted.
  goalie_games   one row per goalie per game: saves, shots_against,
                 goals_against, decision.

Regular season only (gameTypeId=2), same convention as every other
market in this repo.

Run
---
python -u nhl_player_games_foundation_a.py --seasons 2018 2019 2020 2021 2022 2023 2024 2025 2026 \
    --db nhl_models/nhl_model.sqlite
"""
import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

BASE = "https://api.nhle.com/stats/rest/en"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
}
REQUEST_SLEEP = 0.15

TEAM_ABBREVS = [
    "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL",
    "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR",
    "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN",
    "VGK", "WPG", "WSH",
]
PAGE_LIMIT = 2000  # requested page size -- the API silently caps each
# actual response at 100 rows regardless of this value (confirmed
# empirically: limit=2000 and limit=100 both return exactly 100 rows,
# and start=100 successfully returns the NEXT 100) -- so pagination
# below walks by the server's true page size and stops using the
# response's own "total" field, not by comparing against this constant.


def get(url, tries=4, timeout=30):
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def fetch_team_game_rows(kind, abbrev, season_start_year):
    nhl_season_id = f"{season_start_year}{season_start_year + 1}"
    cayenne = f'seasonId={nhl_season_id} and gameTypeId=2 and teamAbbrevs="{abbrev}"'
    rows = []
    start = 0
    total = None
    while total is None or start < total:
        q = urllib.parse.urlencode({
            "isAggregate": "false", "isGame": "true",
            "start": start, "limit": PAGE_LIMIT, "cayenneExp": cayenne,
        })
        time.sleep(REQUEST_SLEEP)
        data = get(f"{BASE}/{kind}/summary?{q}")
        if not data or not data.get("data"):
            break
        rows.extend(data["data"])
        total = data.get("total", len(rows))
        page_size = len(data["data"])
        if page_size == 0:
            break
        start += page_size
    return rows


def build_skater_rows(raw, season):
    out = {}
    for r in raw:
        pid, gid = r.get("playerId"), r.get("gameId")
        if not pid or not gid:
            continue
        out[(pid, gid)] = {
            "player_id": pid, "player_name": r.get("skaterFullName"),
            "position": r.get("positionCode"), "team": r.get("teamAbbrev"),
            "opponent": r.get("opponentTeamAbbrev"), "season": season,
            "game_id": gid, "game_date": r.get("gameDate"),
            "is_home": 1 if r.get("homeRoad") == "H" else 0,
            "goals": r.get("goals") or 0, "assists": r.get("assists") or 0,
            "points": r.get("points") or 0, "shots": r.get("shots") or 0,
            "toi_seconds": r.get("timeOnIcePerGame") or 0,
        }
    return out


def build_goalie_rows(raw, season):
    out = {}
    for r in raw:
        pid, gid = r.get("playerId"), r.get("gameId")
        if not pid or not gid:
            continue
        decision = "W" if (r.get("wins") or 0) > 0 else (
            "L" if (r.get("losses") or 0) > 0 else (
                "OTL" if (r.get("otLosses") or 0) > 0 else None))
        out[(pid, gid)] = {
            "player_id": pid, "player_name": r.get("goalieFullName"),
            "team": r.get("teamAbbrev"), "opponent": r.get("opponentTeamAbbrev"),
            "season": season, "game_id": gid, "game_date": r.get("gameDate"),
            "is_home": 1 if r.get("homeRoad") == "H" else 0,
            "saves": r.get("saves") or 0, "shots_against": r.get("shotsAgainst") or 0,
            "goals_against": r.get("goalsAgainst") or 0, "decision": decision,
            "toi_seconds": r.get("timeOnIce") or 0,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--db", required=True)
    args = ap.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS skater_games (
            player_id INTEGER, game_id INTEGER, player_name TEXT, position TEXT,
            team TEXT, opponent TEXT, season INTEGER, game_date TEXT, is_home INTEGER,
            goals INTEGER, assists INTEGER, points INTEGER, shots INTEGER,
            toi_seconds REAL,
            PRIMARY KEY (player_id, game_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS goalie_games (
            player_id INTEGER, game_id INTEGER, player_name TEXT,
            team TEXT, opponent TEXT, season INTEGER, game_date TEXT, is_home INTEGER,
            saves INTEGER, shots_against INTEGER, goals_against INTEGER,
            decision TEXT, toi_seconds REAL,
            PRIMARY KEY (player_id, game_id)
        )
    """)

    print("NHL_PLAYER_GAMES_FOUNDATION_A\n==============================")
    print(f"seasons={args.seasons}  db={db_path}")

    for season in args.seasons:
        skater_rows = {}
        goalie_rows = {}
        for abbrev in TEAM_ABBREVS:
            raw_sk = fetch_team_game_rows("skater", abbrev, season)
            skater_rows.update(build_skater_rows(raw_sk, season))
            raw_g = fetch_team_game_rows("goalie", abbrev, season)
            goalie_rows.update(build_goalie_rows(raw_g, season))
        print(f"  season {season}-{season+1}: {len(skater_rows)} skater-game rows, "
              f"{len(goalie_rows)} goalie-game rows", flush=True)

        if skater_rows:
            conn.executemany("""
                INSERT OR REPLACE INTO skater_games
                (player_id, game_id, player_name, position, team, opponent, season,
                 game_date, is_home, goals, assists, points, shots, toi_seconds)
                VALUES (:player_id, :game_id, :player_name, :position, :team, :opponent,
                 :season, :game_date, :is_home, :goals, :assists, :points, :shots, :toi_seconds)
            """, list(skater_rows.values()))
        if goalie_rows:
            conn.executemany("""
                INSERT OR REPLACE INTO goalie_games
                (player_id, game_id, player_name, team, opponent, season, game_date,
                 is_home, saves, shots_against, goals_against, decision, toi_seconds)
                VALUES (:player_id, :game_id, :player_name, :team, :opponent, :season,
                 :game_date, :is_home, :saves, :shots_against, :goals_against, :decision, :toi_seconds)
            """, list(goalie_rows.values()))
        conn.commit()

    n_sk = conn.execute("SELECT COUNT(*) FROM skater_games").fetchone()[0]
    n_g = conn.execute("SELECT COUNT(*) FROM goalie_games").fetchone()[0]
    conn.close()
    print(f"\nDB now has {n_sk} skater_games rows, {n_g} goalie_games rows total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
