#!/usr/bin/env python3
"""
CFB_ESPN_LIVE_FOUNDATION_A

Live current-season data source, replacing cfbfastR-data for the season
that's actually being served. cfbfastR-data is a community mirror that
lags real games by days-to-weeks at the start of a season (confirmed by
direct inspection: no 2026 file existed at all despite real 2026 games
already being played) -- ESPN's public scoreboard/box-score/roster
endpoints (undocumented but no-auth, used by ESPN's own site/app, widely
relied on by hobby sports-data projects) already have the live 2026
schedule and box scores same-day.

Historical seasons (2018-2025) stay sourced from cfbfastR-data via
cfb_player_games_foundation_a.py -- that data is already fetched, cached,
and fully validated; no reason to touch it. This script is ONLY for the
current/live season, and writes into the SAME games/player_games schema
so every already-validated model, baseline, and the serving engine work
completely unchanged -- only the data-ingestion layer is new.

Population scoping matches every other market here: FBS-vs-FBS only.
ESPN's scoreboard groups=80 filter is NOT sufficient by itself -- it
still includes FBS-vs-FCS "buy games" (confirmed directly: Georgia
Southern... no, confirmed: Penn State vs Nevada groups=80 included, but
so did Oregon vs Montana State (FCS) and Arizona State vs Northern
Arizona (FCS)). Real discriminator (confirmed by direct inspection):
team.groups.parent.id == "80" for FBS, "81" for FCS -- both teams must
be FBS or the game is dropped, exactly matching cfbfastR-data's own
home_division=fbs/away_division=fbs filter.

Conference name isn't a direct field on the team-detail response; it's
parsed from standingSummary (e.g. "1st in Big Ten" -> "Big Ten"), the
only place ESPN's public API actually names it in this response shape.

Position comes from each team's roster endpoint (real position data, not
inferred from which stat category a player appears in), fetched once per
team per season and cached.

Run
---
python -u cfb_espn_live_foundation_a.py --season 2026 --db cfb_models/cfb_model.sqlite
python -u cfb_espn_live_foundation_a.py --season 2026 --start 2026-08-23 --end 2026-09-05 --db /tmp/test.sqlite
"""
import argparse
import json
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
REQUEST_SLEEP = 0.15  # politeness pacing against an undocumented public endpoint
FBS_PARENT_GROUP_ID = "80"

SCHEMA_CHECK_TABLES = ("games", "player_games")


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


CONF_RE = re.compile(r"\bin\s+(.+)$")


class TeamInfoCache:
    """Lazily fetches + caches each team's FBS status and conference name."""

    def __init__(self):
        self.cache = {}  # team_id -> {"fbs": bool, "conference": str|None, "name": str}

    def get(self, team_id, team_name_hint=None):
        if team_id in self.cache:
            return self.cache[team_id]
        time.sleep(REQUEST_SLEEP)
        data = get(f"{BASE}/teams/{team_id}?enable=groups")
        info = {"fbs": False, "conference": None, "name": team_name_hint}
        if data and data.get("team"):
            t = data["team"]
            info["name"] = t.get("displayName") or team_name_hint
            groups = t.get("groups") or {}
            parent = (groups.get("parent") or {}).get("id")
            info["fbs"] = parent == FBS_PARENT_GROUP_ID
            m = CONF_RE.search(t.get("standingSummary") or "")
            if m:
                info["conference"] = m.group(1).strip()
        self.cache[team_id] = info
        return info


class RosterCache:
    """Lazily fetches + caches each team's season roster: player_id -> position abbrev."""

    def __init__(self):
        self.cache = {}  # (team_id, season) -> {player_id: position_abbrev}

    def get(self, team_id, season):
        key = (team_id, season)
        if key in self.cache:
            return self.cache[key]
        time.sleep(REQUEST_SLEEP)
        # No ?season= param -- confirmed by direct testing that it does NOT
        # return a historical roster (season=2025 returned the postseason
        # bracket context, 0 players); omitting it returns the team's
        # CURRENT roster, which is exactly what's needed since this script
        # only ever runs against the live/current season anyway.
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
    # CFB regular season runs late Aug through early/mid December.
    return date(season, 8, 20), date(season, 12, 15)


def fetch_scoreboard(d):
    ds = d.strftime("%Y%m%d")
    time.sleep(REQUEST_SLEEP)
    data = get(f"{BASE}/scoreboard?dates={ds}&groups=80&limit=300")
    if not data:
        return []
    return data.get("events") or []


def extract_player_stats(box_team, side_athletes_positions, team_id, season):
    """box_team: one entry of boxscore['players'] (one team's stat block).
    Returns list of dicts: player stat lines keyed by category presence."""
    out = {}  # player_id -> accumulated stat dict
    for cat in box_team.get("statistics") or []:
        name = cat.get("name")
        if name not in ("passing", "rushing", "receiving"):
            continue
        labels = cat.get("labels") or []
        for a in cat.get("athletes") or []:
            athlete = a.get("athlete") or {}
            pid = athlete.get("id")
            pname = athlete.get("displayName")
            vals = a.get("stats") or []
            if not pid or len(vals) != len(labels):
                continue
            row = dict(zip(labels, vals))
            d = out.setdefault(pid, {"player_name": pname})
            if name == "rushing":
                d["carries"] = _to_int(row.get("CAR"))
                d["rushing_yards"] = _to_int(row.get("YDS"))
            elif name == "receiving":
                d["receptions"] = _to_int(row.get("REC"))
                d["receiving_yards"] = _to_int(row.get("YDS"))
            elif name == "passing":
                catt = row.get("C/ATT") or ""
                comp, att = (catt.split("/") + [None, None])[:2] if "/" in catt else (None, None)
                d["completions"] = _to_int(comp)
                d["pass_attempts"] = _to_int(att)
                d["passing_yards"] = _to_int(row.get("YDS"))
                d["passing_touchdowns"] = _to_int(row.get("TD"))
                d["passing_interceptions"] = _to_int(row.get("INT"))
    return out


def _to_int(v):
    try:
        if v is None or v == "" or v == "--":
            return None
        return int(round(float(v)))
    except Exception:
        return None


def build_from_events(events, season, team_cache, roster_cache, seen_game_ids):
    games_out = {}
    pg_out = {}
    for e in events:
        gid = e.get("id")
        if not gid or gid in seen_game_ids:
            continue
        status = (e.get("status") or {}).get("type") or {}
        if not status.get("completed"):
            continue
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
        if not (home_info["fbs"] and away_info["fbs"]):
            continue  # FBS-vs-FBS only, matches every other CFB market's scoping

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--start", type=str, default=None, help="YYYY-MM-DD, default season-typical start")
    ap.add_argument("--end", type=str, default=None, help="YYYY-MM-DD, default today or season-typical end")
    ap.add_argument("--db", required=True)
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"FAIL: --db {db_path} does not exist -- run cfb_player_games_foundation_a.py "
              f"first to create the schema (this script only adds live-season rows).")
        return 1
    conn = sqlite3.connect(str(db_path))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = set(SCHEMA_CHECK_TABLES) - tables
    if missing:
        print(f"FAIL: db missing table(s) {missing} -- wrong db or schema out of date")
        return 1

    default_start, default_end = default_season_window(args.season)
    start = date.fromisoformat(args.start) if args.start else default_start
    end = date.fromisoformat(args.end) if args.end else min(default_end, date.today())

    print("CFB_ESPN_LIVE_FOUNDATION_A\n===========================")
    print(f"season={args.season}  window={start}..{end}  db={db_path}")

    team_cache = TeamInfoCache()
    roster_cache = RosterCache()
    seen_game_ids = set()
    all_games = {}
    all_pg = {}

    n_days = 0
    for d in daterange(start, end):
        events = fetch_scoreboard(d)
        n_days += 1
        if not events:
            continue
        g, pg = build_from_events(events, args.season, team_cache, roster_cache, seen_game_ids)
        all_games.update(g)
        all_pg.update(pg)
        if g:
            print(f"  {d}: +{len(g)} FBS-vs-FBS games, +{len(pg)} player-game rows "
                  f"(running total games={len(all_games)} rows={len(all_pg)})", flush=True)

    print(f"\nscanned {n_days} days")
    print(f"FBS-vs-FBS completed games found: {len(all_games)}")
    print(f"QB/RB/WR player-game rows found: {len(all_pg)}")

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

    n_games = conn.execute("SELECT COUNT(*) FROM games WHERE season=?", (args.season,)).fetchone()[0]
    n_pg = conn.execute("SELECT COUNT(*) FROM player_games WHERE season=?", (args.season,)).fetchone()[0]
    conn.close()
    print(f"\nDB now has {n_games} games and {n_pg} player_games rows for season {args.season}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
