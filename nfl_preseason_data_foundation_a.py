#!/usr/bin/env python3
"""
NFL_PRESEASON_DATA_FOUNDATION_A

Phase 1 of the preseason props build. nflverse's player-stats feeds
(verified directly against the live 'stats_player' release, 2025 season:
zero PRE rows) do not carry any preseason player box scores at all --
neither the legacy 'player_stats' tag nor the actively-maintained
'stats_player' tag has them, and the 'schedules' release has no PRE
game_type rows either. So this is a genuinely separate data source from
everything else in this repo.

Source: ESPN's public site API (site.api.espn.com), same one many open
sports-data tools use. No key required. Verified live:
  - scoreboard endpoint with seasontype=1 (preseason) returns real games,
    week=1 is Hall-of-Fame weekend, weeks 2-4 are Preseason Weeks 1-3
  - summary endpoint (?event=<id>) returns full player-level box scores
    (boxscore.players[team].statistics[group].athletes[]) for rushing and
    receiving, with stable numeric athlete ids
  - real data confirmed back through at least the 2021 preseason

This script ONLY loads raw box scores into a queryable db -- no feature
engineering, no eligibility logic, no model. Mirrors the separation used
throughout this repo (nfl_player_games_foundation_a.py is the same shape
for the regular-season source).

Run:
    python -u nfl_preseason_data_foundation_a.py --seasons 2021 2022 2023 2024 2025 2026 \
        --db nfl_models/nfl_preseason.sqlite
"""
import argparse
import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

UA = {"User-Agent": "nfl-preseason-foundation/1.0"}
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
DEFAULT_DB = Path("/data/nfl_model/nfl_preseason.sqlite")

# ESPN week numbering within seasontype=1: 1=Hall of Fame weekend,
# 2/3/4=Preseason Weeks 1/2/3. Labeled explicitly rather than assumed --
# verified live against the 2026 calendar payload (calendar[].entries[]
# labels: "Hall of Fame Weekend", "Preseason Week 1/2/3").
PRESEASON_WEEKS = {1: "HOF", 2: "PRE1", 3: "PRE2", 4: "PRE3"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    season INTEGER,
    week INTEGER,
    week_label TEXT,
    game_date TEXT,
    home_team TEXT,
    away_team TEXT,
    home_score INTEGER,
    away_score INTEGER
);
CREATE TABLE IF NOT EXISTS player_games (
    athlete_id TEXT,
    player_name TEXT,
    team TEXT,
    opponent TEXT,
    is_home INTEGER,
    season INTEGER,
    week INTEGER,
    week_label TEXT,
    game_id TEXT,
    carries INTEGER,
    rushing_yards INTEGER,
    targets INTEGER,
    receptions INTEGER,
    receiving_yards INTEGER,
    PRIMARY KEY (athlete_id, game_id)
);
"""


def http_json(url, params, timeout=30, retries=4):
    from urllib.parse import urlencode
    full = f"{url}?{urlencode(params)}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(full, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"  retry ({attempt+1}/{retries}) {full}: {e}")
            time.sleep(2 * (attempt + 1))
    return {}


def list_games(season, week):
    data = http_json(SCOREBOARD, {"seasontype": 1, "week": week, "dates": season})
    out = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        out.append({
            "game_id": ev.get("id"),
            "date": comp.get("date"),
            "home_team": home.get("team", {}).get("abbreviation"),
            "away_team": away.get("team", {}).get("abbreviation"),
            "home_score": home.get("score"),
            "away_score": away.get("score"),
        })
    return out


def extract_player_rows(game_meta, season, week):
    data = http_json(SUMMARY, {"event": game_meta["game_id"]})
    box = data.get("boxscore", {})
    teams = box.get("players", [])
    rows_by_athlete = {}

    for team_block in teams:
        team_abbr = team_block.get("team", {}).get("abbreviation")
        if team_abbr == game_meta["home_team"]:
            opp, is_home = game_meta["away_team"], 1
        elif team_abbr == game_meta["away_team"]:
            opp, is_home = game_meta["home_team"], 0
        else:
            continue

        for stat_group in team_block.get("statistics", []):
            gname = stat_group.get("name")
            if gname not in ("rushing", "receiving"):
                continue
            labels = stat_group.get("labels", [])
            for ath in stat_group.get("athletes", []):
                aid = ath.get("athlete", {}).get("id")
                name = ath.get("athlete", {}).get("displayName")
                stats = ath.get("stats", [])
                if not aid or len(stats) != len(labels):
                    continue
                vals = dict(zip(labels, stats))
                row = rows_by_athlete.setdefault(aid, {
                    "athlete_id": aid, "player_name": name, "team": team_abbr,
                    "opponent": opp, "is_home": is_home,
                    "carries": None, "rushing_yards": None,
                    "targets": None, "receptions": None, "receiving_yards": None,
                })
                if gname == "rushing":
                    row["carries"] = _to_int(vals.get("CAR"))
                    row["rushing_yards"] = _to_int(vals.get("YDS"))
                elif gname == "receiving":
                    row["targets"] = _to_int(vals.get("TGTS"))
                    row["receptions"] = _to_int(vals.get("REC"))
                    row["receiving_yards"] = _to_int(vals.get("YDS"))
    return list(rows_by_athlete.values())


def _to_int(v):
    try:
        if v is None or v == "--":
            return None
        return int(round(float(v)))
    except Exception:
        return None


def build(seasons, db_path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.executescript(SCHEMA)

    total_games = total_rows = 0
    for season in seasons:
        for week, label in PRESEASON_WEEKS.items():
            games = list_games(season, week)
            print(f"season={season} week={week} ({label}): {len(games)} final games")
            for g in games:
                con.execute(
                    "INSERT OR REPLACE INTO games VALUES (?,?,?,?,?,?,?,?,?)",
                    (g["game_id"], season, week, label, g["date"],
                     g["home_team"], g["away_team"], g["home_score"], g["away_score"]),
                )
                rows = extract_player_rows(g, season, week)
                for r in rows:
                    con.execute(
                        "INSERT OR REPLACE INTO player_games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (r["athlete_id"], r["player_name"], r["team"], r["opponent"],
                         r["is_home"], season, week, label, g["game_id"],
                         r["carries"], r["rushing_yards"],
                         r["targets"], r["receptions"], r["receiving_yards"]),
                    )
                total_rows += len(rows)
                total_games += 1
                time.sleep(0.1)
            con.commit()

    print(f"\ndone: {total_games} games, {total_rows} player-game rows -> {db_path}")
    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args()
    build(args.seasons, args.db)


if __name__ == "__main__":
    main()
