"""NHL V2 Phase1A-SOG acquisition helpers (research only).

Ported as DATA ENGINEERING ONLY from codex/nhl-outcome-engine-v1 nhl_sog_phase1a_data.py (acquire_window / assemble): same source
(api-web schedule + stats-REST skater/summary + skater/timeonice over weekly gameDate windows, regular season), same duplicate and
team-participant guards. Adds explicit per-request retrieval metadata, a polite request gap and no on-disk cache. No betting-market data."""
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

API = "https://api-web.nhle.com/v1"
ST = "https://api.nhle.com/stats/rest/en"
UTC = timezone.utc
REQUEST_GAP_S = 0.4
USER_AGENT = "nhl-v2-research/1.0 (research; low-rate)"


def window_list(season):
    """Weekly 7-day windows Sep 28 .. ~Jun 1 of the season (identical to the V1 acquisition)."""
    lo = datetime(season, 9, 28)
    end = datetime(season + 1, 6, 1)
    out = []
    while lo <= end:
        out.append((lo.strftime("%Y-%m-%d"), (lo + timedelta(days=6)).strftime("%Y-%m-%d")))
        lo += timedelta(days=7)
    return out


def fetch(url, now=lambda: datetime.now(UTC), sleep=time.sleep):
    """GET with retrieval metadata. Returns (raw_bytes, meta). meta carries start/completion times and the payload sha256."""
    last = None
    for attempt in range(4):
        try:
            sleep(REQUEST_GAP_S)
            t0 = now()
            r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=120)
            raw = r.read()
            t1 = now()
            return raw, {"url": url, "retrieval_started_utc": t0.isoformat(), "retrieval_completed_utc": t1.isoformat(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "http_status": r.status}
        except Exception as e:                                                  # noqa
            last = e
            if attempt == 3:
                raise
            sleep(2 * (attempt + 1))
    raise last


def stats_url(path, lo, hi):
    cay = f'gameDate>="{lo}" and gameDate<="{hi}" and gameTypeId=2'
    return ST + path + "?" + urllib.parse.urlencode({"isGame": "true", "limit": "-1", "cayenneExp": cay}, quote_via=urllib.parse.quote)


def acquire_window(lo, hi, states=("OFF", "FINAL"), fetcher=fetch):
    sraw, smeta = fetcher(f"{API}/schedule/{lo}")
    sched = json.loads(sraw)
    games = []
    for day in sched["gameWeek"]:
        if lo <= day["date"] <= hi:
            for g in day["games"]:
                if g["gameType"] == 2 and g["gameState"] in states:
                    games.append({"game_id": g["id"], "game_start_utc": g["startTimeUTC"], "home_abbrev": g["homeTeam"]["abbrev"], "away_abbrev": g["awayTeam"]["abbrev"],
                                  "home_team_id": g["homeTeam"]["id"], "away_team_id": g["awayTeam"]["id"], "date": day["date"]})
    out = {"window": [lo, hi], "games": games, "provenance": {"schedule": smeta}}
    for name, path in (("summary", "/skater/summary"), ("timeonice", "/skater/timeonice")):
        raw, meta = fetcher(stats_url(path, lo, hi))
        d = json.loads(raw)
        out[name] = d["data"]
        out["provenance"][name] = dict(meta, reported_total=d["total"], returned_rows=len(d["data"]), guard_ok=bool(d["total"] < 10000 and len(d["data"]) == d["total"]))
    return out


def assemble(windows):
    """windows -> (games {id: dict}, rows [dict]); missing components stay None (no silent zero filling); duplicate / non-participant rows raise."""
    games, rows, seen = {}, [], set()
    for w in windows:
        for g in w["games"]:
            games[g["game_id"]] = g
    for w in windows:
        tio = {(r["gameId"], r["playerId"]): r for r in w["timeonice"]}
        for r in w["summary"]:
            g = games.get(r["gameId"])
            if g is None:
                raise RuntimeError("summary row of game %s has no completed regular-season game in the schedule" % r["gameId"])
            team = r["teamAbbrev"]
            tid = g["home_team_id"] if team == g["home_abbrev"] else g["away_team_id"] if team == g["away_abbrev"] else None
            if tid is None:
                raise RuntimeError("team %s is not a participant of game %s" % (team, r["gameId"]))
            key = (r["gameId"], tid, r["playerId"])
            if key in seen:
                raise RuntimeError("duplicate skater-game row %s" % (key,))
            seen.add(key)
            t = tio.get((r["gameId"], r["playerId"]), {})
            rows.append({"game_id": r["gameId"], "season_start_year": int(str(r["gameId"])[:4]), "game_start_utc": g["game_start_utc"], "team_id": tid, "team_abbrev": team, "opponent": r["opponentTeamAbbrev"],
                         "player_id": r["playerId"], "position": r.get("positionCode"), "sog": r["shots"], "toi_sec": t.get("timeOnIce"), "ev_toi_sec": t.get("evTimeOnIce"), "pp_toi_sec": t.get("ppTimeOnIce"),
                         "sh_toi_sec": t.get("shTimeOnIce"), "ot_toi_sec": t.get("otTimeOnIce"), "shifts": t.get("shifts")})
    rows.sort(key=lambda r: (r["game_start_utc"], r["game_id"], r["team_id"], r["player_id"]))
    return games, rows
