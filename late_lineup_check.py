"""Late lineup check: one shared step that runs close to game time and turns the *early* forecasts into
*lineup-aware* ones. The early forecast is never overwritten; the adjusted numbers are written beside it so
both can be graded against the real result.

Sources (all free, no keys):
  NBA / NHL / NFL : ESPN game summary `injuries` (Out / Doubtful / Questionable / Day-To-Day ...)
  NHL             : official api-web.nhle.com `right-rail` scratches + `landing` rosterSpots (appear close to puck drop)
  MLB             : statsapi.mlb.com schedule?hydrate=lineups (posted batting order)

What it produces (per game): {player_key: status} where status is OUT / DOUBTFUL / QUESTIONABLE / IN / UNKNOWN,
plus `lineup_posted` (True when the league's own lineup is out). Player keys are normalised names, because
ESPN, NHL and MLB use different ids.

Adjustment rules are per sport and each one is only enabled after it beat the early forecast on finished games
(see ADJUSTMENTS). Nothing here touches a locked engine or a frozen model file.

CLI:  python late_lineup_check.py overlay --sport nhl --date 2026-11-03 --out docs/lineup_overlay_nhl.json
"""
import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.request
from datetime import datetime, timezone

UA = {"User-Agent": "Mozilla/5.0"}
ESPN = "https://site.api.espn.com/apis/site/v2/sports/"
ESPN_PATH = {"nba": "basketball/nba", "nhl": "hockey/nhl", "nfl": "football/nfl"}
NHL_API = "https://api-web.nhle.com/v1"
MLB_API = "https://statsapi.mlb.com/api/v1"

# Higher = more likely to miss. Anything not listed counts as QUESTIONABLE (we never silently treat it as IN).
STATUS_MAP = {
    "out": "OUT", "injured reserve": "OUT", "ir": "OUT", "suspension": "OUT", "suspended": "OUT",
    "inactive": "OUT", "not with team": "OUT", "ltir": "OUT",
    "doubtful": "DOUBTFUL",
    "questionable": "QUESTIONABLE", "day-to-day": "QUESTIONABLE", "day to day": "QUESTIONABLE", "game time decision": "QUESTIONABLE",
    "probable": "IN", "available": "IN", "active": "IN",
}

# Per-sport adjustment of the early forecast. `enabled` is flipped only by a measured improvement on finished games.
#   out_mean: multiplier applied to the mean of a player who is OUT (0 = not playing)
#   in_scale: multiplier for players confirmed in the lineup (fitted on older games, scored on newer games)
# NHL numbers: backtest of 9,082 graded T30 shot forecasts, fitted on 2026-10-07..19, scored on 2026-10-20..11-02.
#   Perfect-lineup ceiling: P(1+ shot) Brier 0.1692 -> 0.1425, P(3+ shots) Brier 0.1111 -> 0.1097.
# NBA numbers: backtest of 5,344 board rows (8 days, 2026-10-27..11-03), scale fitted on 10-27..30, scored on 10-31..11-03.
#   The NBA board mean is already discounted by p_active (avg 0.65 for players who played), so a confirmed-in player gets
#   mean / p_active * scale. Perfect-lineup ceiling on held-out games, mean abs error: points 5.02->3.90, rebounds 1.85->1.44,
#   assists 1.39->1.11, threes 0.88->0.75. `unlisted_out` False: the NBA has no pregame bench list, so absence proves nothing.
ADJUSTMENTS = {
    "nhl": {"enabled": True, "out_mean": 0.0, "doubtful_mean": 0.25, "in_scale": 1.058, "unlisted_out": True},
    "nba": {"enabled": True, "out_mean": 0.0, "doubtful_mean": 0.25, "in_scale": 1.0, "unlisted_out": False,
            "remove_p_active": True,
            "market_scale": {"points": 0.828, "rebounds": 0.942, "assists": 0.912, "three_pointers_made": 0.824}},
    "nfl": {"enabled": False, "out_mean": 0.0, "doubtful_mean": 0.25, "in_scale": 1.0},
    "mlb": {"enabled": False, "out_mean": 0.0, "doubtful_mean": 0.25, "in_scale": 1.0},
}


# ESPN abbreviation -> official league abbreviation (only where they differ)
ABBREV_ALIAS = {
    "nhl": {"TB": "TBL", "NJ": "NJD", "LA": "LAK", "SJ": "SJS", "UTAH": "UTA", "WAS": "WSH", "MON": "MTL"},
}


def canon_team(sport, ab):
    return ABBREV_ALIAS.get(sport, {}).get(ab, ab)


def norm_name(name):
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s.lower())
    return re.sub(r"[^a-z]", "", s)


def map_status(raw):
    return STATUS_MAP.get((raw or "").strip().lower(), "QUESTIONABLE")


def _get(url, timeout=25, retries=3):
    err = None
    for a in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # network / 404 before a game is posted is normal
            err = e
            time.sleep(1.5 * (a + 1))
    return {"_error": str(err)[:120]}


# ---------------------------------------------------------------- parsers (pure; unit-tested on saved payloads)
def parse_espn_injuries(summary, sport=None):
    """ESPN summary -> {team_abbrev: {player_key: status}}."""
    out = {}
    for t in summary.get("injuries", []) or []:
        ab = (t.get("team") or {}).get("abbreviation") or (t.get("team") or {}).get("displayName", "")
        d = out.setdefault(canon_team(sport, ab), {})
        for x in t.get("injuries", []) or []:
            nm = (x.get("athlete") or {}).get("displayName") or (x.get("athlete") or {}).get("fullName")
            if nm:
                d[norm_name(nm)] = map_status(x.get("status"))
    return out


def parse_nhl_official(landing, rail):
    """NHL api-web -> (lineup_posted, {team_abbrev: {player_key: status}}). Scratches are OUT, rostered players IN."""
    posted = bool((landing or {}).get("rosterSpots"))
    out = {}
    abbrev = {}
    for side in ("awayTeam", "homeTeam"):
        tm = (landing or {}).get(side) or {}
        abbrev[side] = tm.get("abbrev")
        abbrev[tm.get("id")] = tm.get("abbrev")
    for s in (landing or {}).get("rosterSpots", []) or []:
        ab = abbrev.get(s.get("teamId"))
        nm = ((s.get("firstName") or {}).get("default", "") + " " + (s.get("lastName") or {}).get("default", "")).strip()
        if ab and nm:
            out.setdefault(ab, {})[norm_name(nm)] = "IN"
    gi = (rail or {}).get("gameInfo", {}) or {}
    for side in ("awayTeam", "homeTeam"):
        ab = abbrev.get(side)
        for s in (gi.get(side, {}) or {}).get("scratches", []) or []:
            nm = ((s.get("firstName") or {}).get("default", "") + " " + (s.get("lastName") or {}).get("default", "")).strip()
            if ab and nm:
                out.setdefault(ab, {})[norm_name(nm)] = "OUT"
                posted = posted or True
    return posted, out


def parse_mlb_lineups(schedule):
    """statsapi schedule?hydrate=lineups -> {game_pk: (posted, {team_abbrev_or_name: {player_key: 'IN'}})}."""
    res = {}
    for d in schedule.get("dates", []) or []:
        for g in d.get("games", []) or []:
            lu = g.get("lineups") or {}
            posted = bool(lu.get("homePlayers") or lu.get("awayPlayers"))
            teams = {}
            for side, key in (("away", "awayPlayers"), ("home", "homePlayers")):
                nm = ((g.get("teams") or {}).get(side) or {}).get("team", {}).get("name", side)
                teams[nm] = {norm_name(p.get("fullName")): "IN" for p in lu.get(key, []) or []}
            res[g.get("gamePk")] = (posted, teams)
    return res


def merge(*maps):
    """Later maps win; OUT is never downgraded by a weaker source."""
    rank = {"OUT": 3, "DOUBTFUL": 2, "QUESTIONABLE": 1, "IN": 0, "UNKNOWN": -1}
    out = {}
    for m in maps:
        for team, players in m.items():
            d = out.setdefault(team, {})
            for k, v in players.items():
                if k not in d or v == "OUT" or (rank[v] >= rank[d[k]] and d[k] != "OUT"):
                    d[k] = v
    return out


# ---------------------------------------------------------------- adjustment (pure)
def adjust_mean(sport, mean, status, lineup_posted, p_active=None, market=None):
    """Return (adjusted_mean, label). Players not mentioned stay UNKNOWN unless the league lineup is posted
    (then a player missing from it is treated as not playing)."""
    a = ADJUSTMENTS[sport]
    if not a["enabled"]:
        return mean, "DISABLED_SPORT"
    if status == "OUT":
        return mean * a["out_mean"], "CONFIRMED_OUT"
    if status == "DOUBTFUL":
        return mean * a["doubtful_mean"], "DOUBTFUL"
    if status == "IN" and (lineup_posted or a.get("remove_p_active")):
        m = mean
        if a.get("remove_p_active") and p_active:
            m = mean / max(p_active, 0.05)
        scale = (a.get("market_scale") or {}).get(market, a["in_scale"])
        return m * scale, "CONFIRMED_IN"
    if status is None and lineup_posted and a.get("unlisted_out", True):
        return mean * a["out_mean"], "NOT_IN_POSTED_LINEUP"
    return mean, ("QUESTIONABLE" if status == "QUESTIONABLE" else "NO_NEW_INFO")


def apply_overlay(sport, rows, game_status):
    """rows: dicts with player (name), team (abbrev), mean. game_status: {'lineup_posted': bool, 'teams': {...}}.
    Returns new rows; the original `mean` is kept as `mean_early`."""
    teams = game_status.get("teams", {})
    posted = bool(game_status.get("lineup_posted"))
    out = []
    for r in rows:
        st = (teams.get(r.get("team")) or {}).get(norm_name(r.get("player")))
        adj, label = adjust_mean(sport, r["mean"], st, posted, r.get("p_active"), r.get("market"))
        out.append(dict(r, mean_early=r["mean"], mean=adj, lineup_label=label, lineup_status=st or "UNKNOWN"))
    return out


# ---------------------------------------------------------------- fetch (network)
def espn_events(sport, date):
    sb = _get(ESPN + ESPN_PATH[sport] + "/scoreboard?dates=" + date.replace("-", ""))
    return sb.get("events", []) or []


def fetch_game_status(sport, event_id, nhl_game_id=None):
    """Combine every source we have for one game. Safe to call repeatedly as game time approaches."""
    teams, posted, sources = {}, False, []
    if sport in ESPN_PATH:
        s = _get(ESPN + ESPN_PATH[sport] + "/summary?event=" + str(event_id))
        if "_error" not in s:
            teams = merge(teams, parse_espn_injuries(s, sport)); sources.append("espn_injuries")
    if sport == "nhl" and nhl_game_id:
        land = _get("%s/gamecenter/%s/landing" % (NHL_API, nhl_game_id))
        rail = _get("%s/gamecenter/%s/right-rail" % (NHL_API, nhl_game_id))
        p, t = parse_nhl_official(land, rail)
        teams = merge(teams, t); posted = posted or p; sources.append("nhl_official")
    return {"lineup_posted": posted, "teams": teams, "sources": sources,
            "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def nhl_game_ids(date):
    """{frozenset({away, home}): game_id} from the official schedule for one date (regular season + playoffs)."""
    s = _get("%s/schedule/%s" % (NHL_API, date))
    out = {}
    for w in s.get("gameWeek", []) or []:
        if w.get("date") == date:
            for g in w.get("games", []) or []:
                out[frozenset((g["awayTeam"]["abbrev"], g["homeTeam"]["abbrev"]))] = g["id"]
    return out


def build_overlay(sport, date):
    games = {}
    nhl_ids = nhl_game_ids(date) if sport == "nhl" else {}
    for e in espn_events(sport, date):
        comp = e["competitions"][0]
        ab = [canon_team(sport, c["team"]["abbreviation"]) for c in comp["competitors"]]
        gid = nhl_ids.get(frozenset(ab))
        games[e["id"]] = dict(fetch_game_status(sport, e["id"], nhl_game_id=gid), matchup=e["shortName"], start=e["date"],
                              teams_list=ab, state=e["status"]["type"]["name"], nhl_game_id=gid)
    return {"schema": "LATE_LINEUP_OVERLAY_V1", "sport": sport, "date": date, "adjustments": ADJUSTMENTS[sport], "games": games}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["overlay"])
    ap.add_argument("--sport", required=True, choices=sorted(ADJUSTMENTS))
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ov = build_overlay(a.sport, a.date)
    with open(a.out, "w") as f:
        json.dump(ov, f, indent=1, sort_keys=True)
    print("wrote", a.out, "games", len(ov["games"]))


if __name__ == "__main__":
    sys.exit(main())
