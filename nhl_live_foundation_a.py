#!/usr/bin/env python3
"""
NHL_LIVE_FOUNDATION_A

First-ever NHL data source in this repo. Pulls real, final game-by-game
results (and the real upcoming schedule) straight from the NHL's own
public, no-auth API (api-web.nhle.com -- the same backend NHL.com itself
uses; confirmed reachable and returning real historical + live data by
direct inspection), one team-season at a time via the
club-schedule-season endpoint, which returns a team's ENTIRE season
(preseason + regular season + playoffs) including real final scores for
already-played games and real dates for not-yet-played ones. Every real
game is returned twice (once per team's own schedule) and deduped here
by game_id.

Population: regular season only (gameType == 2) -- same convention as
every other market in this repo (CFB/NFL exclude preseason/playoffs from
their own training population too). Team roster of abbreviations spans
every team that has existed in the league since the 2018-19 season,
including the two realignments this window covers: Seattle Kraken
joined as an expansion team in 2021-22 (SEA), and the Arizona Coyotes
relocated to Utah for 2024-25 (ARI -> UTA). Querying an abbreviation for
a season before/after a team existed simply returns 0 real games for
that team-season combo -- handled as a no-op, not an error.

`season` is stored as the season's START year (e.g. 2018 for the
2018-19 season) and `week` is a simple date-bucket label (days since
that season's real first regular-season game, integer-divided by 7,
1-indexed) -- NOT a real NHL concept (the league has no "week 1"), it
exists purely so the champion-gate/walkforward-stability scripts can
reuse the exact same season/week-bucketed DEV/VAL/HOLDOUT-split and
week-block-bootstrap machinery already proven for CFB and NFL, without
inventing a new validation methodology for a sport that plays
near-daily. The live serving builder does NOT use this week bucket for
its own asof-lookup (see nhl_serving_builder_a.py's MoneylineEngine
docstring for why a week-exact lookup is the wrong tool for a sport
that plays most days of the week) -- it only matters for training/
validation here.

Run
---
python -u nhl_live_foundation_a.py --seasons 2018 2019 2020 2021 2022 2023 2024 2025 2026 \
    --db nhl_models/nhl_model.sqlite
"""
import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

BASE = "https://api-web.nhle.com/v1"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
}
REQUEST_SLEEP = 0.15

# Every team abbreviation that has played a real NHL regular-season game
# since the 2018-19 season, including both sides of the two realignments
# this window covers (Seattle expansion 2021-22, Arizona -> Utah 2024-25).
TEAM_ABBREVS = [
    "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL",
    "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR",
    "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN",
    "VGK", "WPG", "WSH",
]

REGULAR_SEASON_GAME_TYPE = 2
SCHEMA_TABLES = ("games",)


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


def fetch_team_season(abbrev, season_start_year):
    nhl_season_id = f"{season_start_year}{season_start_year + 1}"
    time.sleep(REQUEST_SLEEP)
    data = get(f"{BASE}/club-schedule-season/{abbrev}/{nhl_season_id}")
    if not data:
        return []
    return data.get("games") or []


def team_display_name(team_obj):
    place = (team_obj.get("placeName") or {}).get("default") or ""
    common = (team_obj.get("commonName") or {}).get("default") or ""
    return f"{place} {common}".strip() or team_obj.get("abbrev")


def build_games(raw_games, season_start_year, seen_game_ids):
    out = {}
    for g in raw_games:
        gid = g.get("id")
        if not gid or gid in seen_game_ids:
            continue
        if g.get("gameType") != REGULAR_SEASON_GAME_TYPE:
            continue
        state = g.get("gameState")
        home = g.get("homeTeam") or {}
        away = g.get("awayTeam") or {}
        game_date = g.get("gameDate") or (g.get("startTimeUTC") or "")[:10]
        if not game_date or not home.get("abbrev") or not away.get("abbrev"):
            continue
        completed = state in ("OFF", "FINAL")
        seen_game_ids.add(gid)
        out[gid] = {
            "game_id": gid,
            "season": season_start_year,
            "game_date": game_date,
            "home_team": team_display_name(home),
            "away_team": team_display_name(away),
            "home_abbrev": home.get("abbrev"),
            "away_abbrev": away.get("abbrev"),
            "home_score": home.get("score") if completed else None,
            "away_score": away.get("score") if completed else None,
            "neutral_site": 1 if g.get("neutralSite") else 0,
            "game_state": state,
        }
    return out


def assign_weeks(games_by_season):
    """week = 1-indexed 7-day bucket since that season's real earliest
    regular-season game date -- see module docstring for why this is a
    label for training/validation only, not a real NHL concept."""
    for season, games in games_by_season.items():
        dates = [datetime.fromisoformat(g["game_date"]).date() for g in games]
        if not dates:
            continue
        season_start = min(dates)
        for g in games:
            d = datetime.fromisoformat(g["game_date"]).date()
            g["week"] = ((d - season_start).days // 7) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", required=True,
                     help="season START years, e.g. 2018 for the 2018-19 season")
    ap.add_argument("--db", required=True)
    args = ap.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id INTEGER PRIMARY KEY,
            season INTEGER NOT NULL,
            week INTEGER,
            game_date TEXT NOT NULL,
            home_team TEXT NOT NULL, away_team TEXT NOT NULL,
            home_abbrev TEXT, away_abbrev TEXT,
            home_score INTEGER, away_score INTEGER,
            neutral_site INTEGER DEFAULT 0,
            game_state TEXT
        )
    """)

    print("NHL_LIVE_FOUNDATION_A\n======================")
    print(f"seasons={args.seasons}  db={db_path}")

    games_by_season = {}
    for season in args.seasons:
        seen_game_ids = set()
        season_games = {}
        n_teams_with_data = 0
        for abbrev in TEAM_ABBREVS:
            raw = fetch_team_season(abbrev, season)
            if not raw:
                continue
            g = build_games(raw, season, seen_game_ids)
            if g:
                n_teams_with_data += 1
            season_games.update(g)
        games_by_season[season] = list(season_games.values())
        n_completed = sum(1 for g in season_games.values() if g["home_score"] is not None)
        print(f"  season {season}-{season+1}: {len(season_games)} regular-season games "
              f"({n_completed} completed, {len(season_games) - n_completed} scheduled), "
              f"{n_teams_with_data} teams found", flush=True)

    assign_weeks(games_by_season)

    all_rows = [g for games in games_by_season.values() for g in games]
    if all_rows:
        conn.executemany("""
            INSERT OR REPLACE INTO games
            (game_id, season, week, game_date, home_team, away_team, home_abbrev,
             away_abbrev, home_score, away_score, neutral_site, game_state)
            VALUES (:game_id, :season, :week, :game_date, :home_team, :away_team,
             :home_abbrev, :away_abbrev, :home_score, :away_score, :neutral_site, :game_state)
        """, all_rows)
    conn.commit()

    n_games = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    n_completed = conn.execute(
        "SELECT COUNT(*) FROM games WHERE home_score IS NOT NULL").fetchone()[0]
    conn.close()
    print(f"\nDB now has {n_games} total regular-season games ({n_completed} completed) "
          f"across seasons {sorted(games_by_season)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
