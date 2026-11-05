"""Publish late lineup statuses (injuries, scratches, starters) for the app to read.

The app host can read this public repo but may not be able to reach ESPN or the NHL API. This job fetches them and writes
docs/lineup_overlay/{nba,nhl}.json. It publishes STATUSES ONLY: never a forecast, never a model input. The app
(server/lineupOverlay.ts in the frontend repo) applies them: OUT removes a player, a posted starter list confirms players.

Schema LATE_LINEUP_PUBLISHED_V1:
  {schema, sport, generated_at, games: {<game_id>: {lineupPosted, sources, players: {"<teamKey>|<normalizedName>": STATUS}}}}
  NBA   game_id = ESPN event id (same id the NBA V3 forecasts use), teamKey = ESPN team id
  NHL   game_id = NHL game id (same id the NHL ledger uses),         teamKey = official NHL team abbreviation

Writes a file only when its content changed or it is older than 25 minutes (the app ignores files older than 3 hours).
Stdlib only.
"""
import json
import re
import sys
import time
import unicodedata
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SCHEMA = "LATE_LINEUP_PUBLISHED_V1"
OUT = Path("docs/lineup_overlay")
FRESH_S = 25 * 60
UA = {"User-Agent": "Mozilla/5.0"}
ESPN = "https://site.api.espn.com/apis/site/v2/sports/"
NHL_API = "https://api-web.nhle.com/v1"
ESPN_NHL_ALIAS = {"TB": "TBL", "NJ": "NJD", "LA": "LAK", "SJ": "SJS", "UTAH": "UTA", "WAS": "WSH", "MON": "MTL"}
STATUS_MAP = {
    "out": "OUT", "injured reserve": "OUT", "ir": "OUT", "suspension": "OUT", "suspended": "OUT", "inactive": "OUT",
    "not with team": "OUT", "ltir": "OUT", "doubtful": "DOUBTFUL",
    "questionable": "QUESTIONABLE", "day-to-day": "QUESTIONABLE", "day to day": "QUESTIONABLE", "game time decision": "QUESTIONABLE",
    "probable": "IN", "available": "IN", "active": "IN",
}
RANK = {"OUT": 3, "DOUBTFUL": 2, "QUESTIONABLE": 1, "IN": 0}


def norm_name(n):
    s = unicodedata.normalize("NFKD", n or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def map_status(raw):
    return STATUS_MAP.get((raw or "").strip().lower(), "QUESTIONABLE")   # unknown wording is never treated as healthy


def set_status(d, key, s):
    cur = d.get(key)
    if cur is None or s == "OUT" or (cur != "OUT" and RANK[s] >= RANK[cur]):
        d[key] = s


def get(url, tries=3):
    err = None
    for a in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
                return json.load(r)
        except Exception as e:                       # a missing page before a game is normal
            err = e
            time.sleep(1.5 * (a + 1))
    print("fetch failed:", url, str(err)[:80], file=sys.stderr)
    return None


def espn_injuries(summary, team_key, players):
    for t in summary.get("injuries", []) or []:
        tk = team_key(t.get("team") or {})
        for x in t.get("injuries", []) or []:
            a = x.get("athlete") or {}
            nm = a.get("displayName") or a.get("fullName")
            if nm:
                set_status(players, tk + "|" + norm_name(nm), map_status(x.get("status")))


def espn_starters(summary, players):
    any_ = False
    for t in (summary.get("boxscore") or {}).get("players", []) or []:
        tk = str((t.get("team") or {}).get("id", ""))
        for g in t.get("statistics", []) or []:
            for a in g.get("athletes", []) or []:
                if a.get("starter") is True and (a.get("athlete") or {}).get("displayName"):
                    set_status(players, tk + "|" + norm_name(a["athlete"]["displayName"]), "IN")
                    any_ = True
    return any_


def et_dates(now):
    et = ZoneInfo("America/New_York")
    return [(now.astimezone(et) + timedelta(days=d)).strftime("%Y%m%d") for d in (-1, 0, 1)]


def build_nba(now):
    games = {}
    seen = set()
    for d in et_dates(now):
        sb = get(ESPN + "basketball/nba/scoreboard?dates=" + d) or {}
        for e in sb.get("events", []) or []:
            if e["id"] in seen:
                continue
            seen.add(e["id"])
            tip = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
            if not (-3 * 3600 <= (tip - now).total_seconds() <= 36 * 3600):
                continue
            s = get(ESPN + "basketball/nba/summary?event=" + e["id"])
            if not s:
                continue
            players = {}
            espn_injuries(s, lambda t: str(t.get("id", "")), players)
            started = espn_starters(s, players)
            games[e["id"]] = {"lineupPosted": started, "players": players,
                              "sources": ["espn_injuries"] + (["espn_starters"] if started else [])}
    return games


def nhl_official(landing, rail, players):
    posted = bool((landing or {}).get("rosterSpots"))
    ab = {}
    for side in ("awayTeam", "homeTeam"):
        t = (landing or {}).get(side) or {}
        ab[side] = t.get("abbrev")
        ab[str(t.get("id"))] = t.get("abbrev")
    for s in (landing or {}).get("rosterSpots", []) or []:
        tk = ab.get(str(s.get("teamId")))
        nm = ((s.get("firstName") or {}).get("default", "") + " " + (s.get("lastName") or {}).get("default", "")).strip()
        if tk and nm:
            set_status(players, tk + "|" + norm_name(nm), "IN")
    gi = (rail or {}).get("gameInfo", {}) or {}
    for side in ("awayTeam", "homeTeam"):
        for s in (gi.get(side, {}) or {}).get("scratches", []) or []:
            nm = ((s.get("firstName") or {}).get("default", "") + " " + (s.get("lastName") or {}).get("default", "")).strip()
            if ab.get(side) and nm:          # scratches alone do NOT mean the full lineup is posted
                set_status(players, ab[side] + "|" + norm_name(nm), "OUT")
    return posted


def build_nhl(now):
    games = {}
    et = ZoneInfo("America/New_York")
    espn_by_pair = {}
    for d in et_dates(now):
        sb = get(ESPN + "hockey/nhl/scoreboard?dates=" + d) or {}
        for e in sb.get("events", []) or []:
            ab = [ESPN_NHL_ALIAS.get(c["team"]["abbreviation"], c["team"]["abbreviation"])
                  for c in e["competitions"][0]["competitors"]]
            if len(ab) == 2:
                espn_by_pair["|".join(sorted(ab))] = e["id"]
    seen = set()
    for off in (-1, 0, 1):
        day = (now.astimezone(et) + timedelta(days=off)).strftime("%Y-%m-%d")
        sch = get("%s/schedule/%s" % (NHL_API, day)) or {}
        for w in sch.get("gameWeek", []) or []:
            if w.get("date") != day:
                continue
            for g in w.get("games", []) or []:
                gid = g["id"]
                if gid in seen or g.get("gameType") != 2:
                    continue
                seen.add(gid)
                tip = datetime.fromisoformat(g["startTimeUTC"].replace("Z", "+00:00"))
                if not (-3 * 3600 <= (tip - now).total_seconds() <= 36 * 3600):
                    continue
                players, sources = {}, []
                land = get("%s/gamecenter/%s/landing" % (NHL_API, gid), tries=2)
                rail = get("%s/gamecenter/%s/right-rail" % (NHL_API, gid), tries=2)
                posted = False
                if land or rail:
                    posted = nhl_official(land, rail, players)
                    sources.append("nhl_official")
                pair = "|".join(sorted([g["awayTeam"]["abbrev"], g["homeTeam"]["abbrev"]]))
                eid = espn_by_pair.get(pair)
                if eid:
                    s = get(ESPN + "hockey/nhl/summary?event=" + eid, tries=2)
                    if s:
                        espn = {}
                        espn_injuries(s, lambda t: ESPN_NHL_ALIAS.get(str(t.get("abbreviation")), str(t.get("abbreviation"))), espn)
                        for k, v in espn.items():
                            set_status(players, k, v)
                        sources.append("espn_injuries")
                if sources:
                    games[str(gid)] = {"lineupPosted": posted, "players": players, "sources": sources}
    return games


def write(sport, games, now):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / (sport + ".json")
    try:
        old = json.loads(f.read_text())
    except Exception:
        old = None
    if old and old.get("schema") == SCHEMA and old.get("games") == games:
        age = (now - datetime.fromisoformat(old["generated_at"].replace("Z", "+00:00"))).total_seconds()
        if age < FRESH_S:
            print(sport, "unchanged, not rewritten")
            return
    doc = {"schema": SCHEMA, "sport": sport, "generated_at": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
           "note": "Injury/scratch/starter statuses only. Never a forecast; applied by the frontend's lineupOverlay.",
           "games": games}
    f.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(sport, "written,", len(games), "games,", sum(1 for g in games.values() for v in g["players"].values() if v == "OUT"), "players OUT")


def main():
    now = datetime.now(timezone.utc)
    ok = 0
    for sport, fn in (("nba", build_nba), ("nhl", build_nhl)):
        try:
            games = fn(now)
            if games:                       # never overwrite a good file with an empty one after a failed fetch
                write(sport, games, now)
                ok += 1
            else:
                print(sport, "no games found; file left as is")
        except Exception as e:
            print(sport, "failed:", e, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
