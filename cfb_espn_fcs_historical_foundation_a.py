#!/usr/bin/env python3
"""
CFB_ESPN_FCS_HISTORICAL_FOUNDATION_A

First step of extending CFB coverage to FCS players (user asked for "all
players", not just FBS). cfbfastR-data -- the source used for every FBS
market in this repo -- is FBS-only; there is no equivalent free historical
FCS play-by-play source found. ESPN's public site API, already proven
reliable for the live FBS season (cfb_espn_live_foundation_a.py), turns
out to cover FCS identically: confirmed by direct testing, not assumed --
`/scoreboard?groups=81` returns real FCS games in the same shape as
groups=80 FBS, `/summary?event=<id>` returns the same passing/rushing/
receiving boxscore categories for a real completed FCS game (tested:
Robert Morris @ Wagner, 2026-08-29), and `/teams/{id}/roster` returns the
same position-tagged roster shape for a real FCS team (tested: Wagner,
100 players, real position tags).

Since there's no independent historical FCS team list to cross-check
against (the whole point of this script is to BUILD one), FBS/FCS
classification here trusts ESPN's groups.parent.id directly (=="81"),
same as the ORIGINAL (pre-cross-check) FBS approach. The cross-check
fix added to cfb_espn_live_foundation_a.py addressed a real but narrow
failure mode (an FCS team occasionally mislabeled AS FBS when the OTHER
side of the same query is a real FBS team) -- scoped to FCS-vs-FCS games
only here (population design mirrors this repo's own FBS-vs-FBS scoping
everywhere else, for the same reason: mixing in FBS blowout "buy games"
would add exactly the noisy-mismatch population this repo already found
and fixed for FBS Group-of-5 games), so that specific failure mode isn't
in play -- both sides already have to agree they're FCS.

Writes a SEPARATE db (cfb_fcs_model.sqlite) with the IDENTICAL games/
player_games schema used everywhere else in this repo, so every existing
feature-engineering/baseline/champion-gate/serving script can be reused
against it unchanged by just pointing --source/--db at this file instead
of cfb_model.sqlite. Kept as a separate file (not merged into the FBS db)
so an FCS and FBS team that happen to share a school-name substring can
never collide, and so the FBS pipeline's validated behavior is provably
untouched by this addition.

Historical (not live): scans one full season window per --season, ALL
days, keeping only completed FCS-vs-FCS games -- unlike the live script,
there's no future-schedule-visibility need here since this is building a
frozen training set, not serving picks.

Run
---
python -u cfb_espn_fcs_historical_foundation_a.py --season 2022 2023 2024 2025 --db cfb_models/cfb_fcs_model.sqlite
"""
import argparse
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

BASE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
REQUEST_SLEEP = 0.15
FCS_PARENT_GROUP_ID = "81"

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
    neutral_site INTEGER,
    home_conference TEXT,
    away_conference TEXT
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
    pass_attempts INTEGER,
    completions INTEGER,
    passing_yards INTEGER,
    passing_touchdowns INTEGER,
    passing_interceptions INTEGER,
    PRIMARY KEY (player_id, season, week, game_id)
);
CREATE INDEX IF NOT EXISTS idx_pg_player_season ON player_games(player_id, season, week);
CREATE INDEX IF NOT EXISTS idx_pg_team_week ON player_games(team, season, week);
CREATE INDEX IF NOT EXISTS idx_pg_game ON player_games(game_id);
"""


def get(url, tries=4, timeout=30):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                import json as _json
                return _json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (404, 400):
                return None
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


CONF_RE = re.compile(r"\bin\s+(.+)$")


class TeamInfoCache:
    def __init__(self):
        self.cache = {}

    def get(self, team_id, team_name_hint=None):
        if team_id in self.cache:
            return self.cache[team_id]
        time.sleep(REQUEST_SLEEP)
        data = get(f"{BASE}/teams/{team_id}?enable=groups")
        info = {"fcs": False, "conference": None, "name": team_name_hint}
        if data and data.get("team"):
            t = data["team"]
            info["name"] = t.get("displayName") or team_name_hint
            groups = t.get("groups") or {}
            parent = (groups.get("parent") or {}).get("id")
            info["fcs"] = parent == FCS_PARENT_GROUP_ID
            m = CONF_RE.search(t.get("standingSummary") or "")
            if m:
                info["conference"] = m.group(1).strip()
        self.cache[team_id] = info
        return info


class RosterCache:
    def __init__(self):
        self.cache = {}

    def get(self, team_id, season):
        key = (team_id, season)
        if key in self.cache:
            return self.cache[key]
        time.sleep(REQUEST_SLEEP)
        data = get(f"{BASE}/teams/{team_id}/roster")
        pos_map = {}
        if data:
            for group in data.get("athletes") or []:
                for item in group.get("items") or []:
                    pid = item.get("id")
                    pos = (item.get("position") or {}).get("abbreviation")
                    if pid and pos:
                        pos_map[pid] = pos
        self.cache[key] = pos_map
        return pos_map


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def default_season_window(season):
    # FCS regular season: mid-Aug through mid-Nov (playoffs run Nov-Jan
    # but this repo's markets are regular-season props, matching the FBS
    # scripts' own Aug20-Dec15 convention closely enough for training data).
    return date(season, 8, 20), date(season, 11, 25)


def fetch_scoreboard(d):
    ds = d.strftime("%Y%m%d")
    time.sleep(REQUEST_SLEEP)
    data = get(f"{BASE}/scoreboard?dates={ds}&groups=81&limit=300")
    if not data:
        return []
    return data.get("events") or []


def _to_int(v):
    try:
        if v is None or v == "" or v == "--":
            return None
        return int(round(float(v)))
    except Exception:
        return None


def extract_player_stats(box_team, roster, team_id, season):
    out = {}
    for cat in box_team.get("statistics") or []:
        name = cat.get("name")
        if name not in ("passing", "rushing", "receiving"):
            continue
        labels = cat.get("labels") or []
        for a in cat.get("athletes") or []:
            pid = (a.get("athlete") or {}).get("id")
            if not pid:
                continue
            stats = a.get("stats") or []
            row = dict(zip(labels, stats))
            d = out.setdefault(pid, {"player_name": (a.get("athlete") or {}).get("displayName")})
            if name == "passing":
                ca = (row.get("C/ATT") or "0/0").split("/")
                d["completions"] = _to_int(ca[0]) if len(ca) == 2 else None
                d["pass_attempts"] = _to_int(ca[1]) if len(ca) == 2 else None
                d["passing_yards"] = _to_int(row.get("YDS"))
                d["passing_touchdowns"] = _to_int(row.get("TD"))
                d["passing_interceptions"] = _to_int(row.get("INT"))
            elif name == "rushing":
                d["carries"] = _to_int(row.get("CAR"))
                d["rushing_yards"] = _to_int(row.get("YDS"))
            elif name == "receiving":
                d["receptions"] = _to_int(row.get("REC"))
                d["receiving_yards"] = _to_int(row.get("YDS"))
    return out


def build_from_events(events, season, team_cache, roster_cache, seen_game_ids):
    games_out = {}
    pg_out = {}
    for e in events:
        gid = e.get("id")
        if not gid or gid in seen_game_ids:
            continue
        status = (e.get("status") or {}).get("type") or {}
        if not status.get("completed"):
            continue  # historical build: completed games only
        comp = (e.get("competitions") or [{}])[0]
        competitors = comp.get("competitors") or []
        if len(competitors) != 2:
            continue
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        home_id, away_id = home["team"]["id"], away["team"]["id"]
        home_info = team_cache.get(home_id, home["team"].get("displayName"))
        away_info = team_cache.get(away_id, away["team"].get("displayName"))
        if not (home_info["fcs"] and away_info["fcs"]):
            continue  # FCS-vs-FCS only -- same reasoning as FBS-vs-FBS elsewhere

        week = (e.get("week") or {}).get("number")
        game_date = (e.get("date") or "")[:10]
        if not week or not game_date:
            continue

        seen_game_ids.add(gid)
        games_out[gid] = {
            "game_id": gid, "season": season, "week": week, "game_date": game_date,
            "home_team": home_info["name"], "away_team": away_info["name"],
            "home_points": _to_int(home.get("score")), "away_points": _to_int(away.get("score")),
            "neutral_site": 1 if comp.get("neutralSite") else 0,
            "home_conference": home_info["conference"], "away_conference": away_info["conference"],
        }

        time.sleep(REQUEST_SLEEP)
        box = get(f"{BASE}/summary?event={gid}")
        if not box:
            continue
        players_blocks = (box.get("boxscore") or {}).get("players") or []
        for side_team_id, side_name, opp_name, is_home in (
            (home_id, home_info["name"], away_info["name"], 1),
            (away_id, away_info["name"], home_info["name"], 0),
        ):
            block = next((p for p in players_blocks if p.get("team", {}).get("id") == side_team_id), None)
            if not block:
                continue
            roster = roster_cache.get(side_team_id, season)
            stats = extract_player_stats(block, roster, side_team_id, season)
            for pid, d in stats.items():
                pos = roster.get(pid)
                if pos not in ("QB", "RB", "WR"):
                    continue
                pg_out[(pid, season, week, gid)] = {
                    "player_id": pid, "player_name": d.get("player_name"), "position": pos,
                    "team": side_name, "opponent": opp_name, "season": season, "week": week,
                    "game_id": gid, "game_date": game_date, "is_home": is_home,
                    "carries": d.get("carries", 0) or 0, "rushing_yards": d.get("rushing_yards", 0) or 0,
                    "receptions": d.get("receptions", 0) or 0, "receiving_yards": d.get("receiving_yards", 0) or 0,
                    "pass_attempts": d.get("pass_attempts", 0) or 0, "completions": d.get("completions", 0) or 0,
                    "passing_yards": d.get("passing_yards", 0) or 0,
                    "passing_touchdowns": d.get("passing_touchdowns", 0) or 0,
                    "passing_interceptions": d.get("passing_interceptions", 0) or 0,
                }
    return games_out, pg_out


def run_season(season, conn, team_cache, roster_cache):
    start, end = default_season_window(season)
    seen_game_ids = set()
    all_games = {}
    all_pg = {}
    n_days = 0
    for d in daterange(start, end):
        events = fetch_scoreboard(d)
        n_days += 1
        if not events:
            continue
        g, pg = build_from_events(events, season, team_cache, roster_cache, seen_game_ids)
        all_games.update(g)
        all_pg.update(pg)
        if g:
            print(f"  {d}: +{len(g)} FCS-vs-FCS games, +{len(pg)} player-game rows "
                  f"(season {season} running total games={len(all_games)} rows={len(all_pg)})", flush=True)

    print(f"season {season}: scanned {n_days} days, {len(all_games)} FCS-vs-FCS games, "
          f"{len(all_pg)} QB/RB/WR player-game rows")

    if all_games:
        conn.executemany(
            "INSERT OR REPLACE INTO games (game_id, season, week, game_date, home_team, "
            "away_team, home_points, away_points, neutral_site, home_conference, away_conference) VALUES "
            "(:game_id, :season, :week, :game_date, :home_team, :away_team, :home_points, "
            ":away_points, :neutral_site, :home_conference, :away_conference)",
            list(all_games.values()))
    if all_pg:
        conn.executemany(
            "INSERT OR REPLACE INTO player_games (player_id, player_name, position, team, "
            "opponent, season, week, game_id, game_date, is_home, carries, rushing_yards, "
            "receptions, receiving_yards, pass_attempts, completions, passing_yards, "
            "passing_touchdowns, passing_interceptions) VALUES (:player_id, :player_name, "
            ":position, :team, :opponent, :season, :week, :game_id, :game_date, :is_home, "
            ":carries, :rushing_yards, :receptions, :receiving_yards, :pass_attempts, "
            ":completions, :passing_yards, :passing_touchdowns, :passing_interceptions)",
            list(all_pg.values()))
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, nargs="+", required=True)
    ap.add_argument("--db", required=True)
    args = ap.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)

    print("CFB_ESPN_FCS_HISTORICAL_FOUNDATION_A\n=====================================")
    print(f"seasons={args.season}  db={db_path}")

    team_cache = TeamInfoCache()
    roster_cache = RosterCache()
    for season in args.season:
        run_season(season, conn, team_cache, roster_cache)

    n_games = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    n_pg = conn.execute("SELECT COUNT(*) FROM player_games").fetchone()[0]
    conn.close()
    print(f"\nDB now has {n_games} games and {n_pg} player_games rows total: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
