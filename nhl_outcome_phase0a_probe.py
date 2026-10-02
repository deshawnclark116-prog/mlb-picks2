"""
NHL_OUTCOME_PHASE0A_PROBE -- bounded data/chronology feasibility probe for the NHL shots-on-goal outcome engine. No model, no tuning, no season replay.

Fetches a small deterministic sample (hash-selected game ids), caches every response by sha256, and compares independent sources on the same games:
  api-web boxscore / play-by-play / right-rail / landing, stats-rest shiftcharts / timeonice / skater summary, and the repo's own nhl_model.sqlite rows.
  python -u nhl_outcome_phase0a_probe.py --cache DIR --out FILE
"""
import argparse
import csv
import hashlib
import io
import json
import re
import sqlite3
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
API = "https://api-web.nhle.com/v1"
STATS = "https://api.nhle.com/stats/rest/en"
SAMPLE_SEASONS = (2018, 2021, 2023, 2025)
COVERAGE_SEASONS = tuple(range(2009, 2026))
OPS = []


def now():
    return datetime.now(timezone.utc).isoformat()


def get(url, cache, kind="json"):
    t0 = time.time()
    key = hashlib.sha256(url.encode()).hexdigest()
    f = Path(cache) / f"{key}.bin"
    meta = Path(cache) / f"{key}.meta.json"
    if f.exists() and meta.exists():
        raw, m = f.read_bytes(), json.loads(meta.read_text())
        m["from_cache"] = True
    else:
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "nhl-phase0a"}), timeout=90)
            raw = r.read()
            m = {"url": url, "http_status": r.status, "retrieved_at_utc": now(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "from_cache": False,
                 "headers": {k: v for k, v in r.headers.items() if k.lower() in ("last-modified", "etag", "cache-control", "age", "date", "content-type")}}
        except urllib.error.HTTPError as e:
            return None, {"url": url, "http_status": e.code, "retrieved_at_utc": now(), "error": str(e)}
        f.write_bytes(raw); meta.write_text(json.dumps(m))
    OPS.append({"url": url, "seconds": round(time.time() - t0, 2), "from_cache": m["from_cache"], "bytes": m.get("bytes")})
    return (json.loads(raw) if kind == "json" else raw), m


def toi_s(s):
    if s is None or s == "":
        return None
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def pick_games(con):
    out = []
    for s in SAMPLE_SEASONS:
        ids = [r[0] for r in con.execute("select game_id from games where season=? and game_state in ('OFF','FINAL')", (s,))]
        ids.sort(key=lambda g: hashlib.sha256(f"nhl-phase0a-{g}".encode()).hexdigest())
        out += ids[:2]
    return out


def analyse_game(gid, cache, con, html=False):
    ev = {"game_id": gid, "sources": {}}
    box, m1 = get(f"{API}/gamecenter/{gid}/boxscore", cache); ev["sources"]["boxscore"] = m1
    pbp, m2 = get(f"{API}/gamecenter/{gid}/play-by-play", cache); ev["sources"]["play_by_play"] = m2
    rr, m3 = get(f"{API}/gamecenter/{gid}/right-rail", cache); ev["sources"]["right_rail"] = m3
    sh, m4 = get(f"{STATS}/shiftcharts?cayenneExp=gameId={gid}", cache); ev["sources"]["shiftcharts"] = m4
    ti, m5 = get(f"{STATS}/skater/timeonice?isGame=true&cayenneExp=gameId={gid}", cache); ev["sources"]["timeonice_stats_rest"] = m5
    sm, m6 = get(f"{STATS}/skater/summary?isGame=true&limit=-1&cayenneExp=gameId={gid}", cache); ev["sources"]["skater_summary_stats_rest"] = m6
    if not (box and pbp):
        ev["error"] = "boxscore / play-by-play unavailable"; return ev
    ev["identity"] = {"season": box["season"], "gameType": box["gameType"], "gameDate": box["gameDate"], "startTimeUTC": box["startTimeUTC"], "venueUTCOffset": box.get("venueUTCOffset"),
                      "away": box["awayTeam"]["abbrev"], "home": box["homeTeam"]["abbrev"], "awayTeamId": box["awayTeam"]["id"], "homeTeamId": box["homeTeam"]["id"], "gameState": box["gameState"],
                      "final_sog": {"away": box["awayTeam"].get("sog"), "home": box["homeTeam"].get("sog")}, "scheduleState": box.get("gameScheduleState")}
    row = con.execute("select game_date, home_abbrev, away_abbrev, home_score, away_score from games where game_id=?", (gid,)).fetchone()
    ev["local_games_row"] = row
    # ---- skaters from the boxscore
    bx = {}
    for side in ("awayTeam", "homeTeam"):
        for pos in ("forwards", "defense"):
            for p in box["playerByGameStats"][side][pos]:
                bx[p["playerId"]] = {"team": box[side]["abbrev"], "sog": p.get("sog"), "toi": toi_s(p.get("toi")), "shifts": p.get("shifts"), "pos": p["position"], "name": p["name"]["default"]}
    gl = {side: [g["playerId"] for g in box["playerByGameStats"][side]["goalies"]] for side in ("awayTeam", "homeTeam")}
    ev["boxscore_skaters"] = len(bx)
    ev["goalies_dressed"] = gl
    ev["goalie_starter_flag_in_boxscore"] = [k for k in box["playerByGameStats"]["homeTeam"]["goalies"][0].keys() if "start" in k.lower()]
    # ---- play-by-play event attribution
    plays = pbp["plays"]
    cnt = Counter(p["typeDescKey"] for p in plays)
    att = defaultdict(Counter)
    sit = Counter()
    for p in plays:
        d = p.get("details", {})
        t = p["typeDescKey"]
        if t in ("shot-on-goal", "goal", "missed-shot", "blocked-shot") and p["periodDescriptor"].get("periodType") != "SO":      # shootout shots are not credited as SOG in the boxscore
            sh_id = d.get("shootingPlayerId") or d.get("scoringPlayerId")
            if sh_id:
                att[sh_id][t] += 1
            sit[p.get("situationCode")] += 1
    pb_sog = {pid: c["shot-on-goal"] + c["goal"] for pid, c in att.items()}
    ev["pbp"] = {"has_shootout_period": any(p["periodDescriptor"].get("periodType") == "SO" for p in plays), "n_events": len(plays), "event_types": dict(cnt), "n_rosterSpots": len(pbp["rosterSpots"]), "has_situationCode": "situationCode" in plays[5],
                 "example_shot_event": next(p for p in plays if p["typeDescKey"] == "shot-on-goal"), "n_distinct_situation_codes": len(sit)}
    ev["sog_boxscore_vs_pbp"] = {"n_players_compared": 0, "mismatches": []}
    for pid, b in bx.items():
        a = pb_sog.get(pid, 0)
        ev["sog_boxscore_vs_pbp"]["n_players_compared"] += 1
        if (b["sog"] or 0) != a:
            ev["sog_boxscore_vs_pbp"]["mismatches"].append({"player_id": pid, "name": b["name"], "boxscore_sog": b["sog"], "pbp_sog_plus_goals": a})
    ev["team_sog_check"] = {"boxscore_team_sog_sum_players": {t: sum((b["sog"] or 0) for b in bx.values() if b["team"] == t) for t in {b["team"] for b in bx.values()}}, "boxscore_team_field": ev["identity"]["final_sog"]}
    # ---- local DB comparison
    loc = {r[0]: r for r in con.execute("select player_id, shots, toi_seconds, team from skater_games where game_id=?", (gid,))}
    ev["local_db_skater_rows"] = len(loc)
    ev["local_db_vs_boxscore"] = {"sog_mismatches": [(pid, loc[pid][1], bx[pid]["sog"]) for pid in loc if pid in bx and (loc[pid][1] or 0) != (bx[pid]["sog"] or 0)],
                                  "toi_mismatch_gt_1s": [(pid, loc[pid][2], bx[pid]["toi"]) for pid in loc if pid in bx and bx[pid]["toi"] is not None and abs((loc[pid][2] or 0) - bx[pid]["toi"]) > 1],
                                  "ids_in_local_not_boxscore": [pid for pid in loc if pid not in bx], "ids_in_boxscore_not_local": [pid for pid in bx if pid not in loc]}
    # ---- stats-rest summary / TOI by strength
    if sm and sm.get("data"):
        smd = {r["playerId"]: r for r in sm["data"]}
        ev["skater_summary_vs_boxscore"] = {"n_rows": len(smd), "sog_mismatches": [(pid, smd[pid].get("shots"), bx[pid]["sog"]) for pid in smd if pid in bx and (smd[pid].get("shots") or 0) != (bx[pid]["sog"] or 0)],
                                            "fields": sorted(next(iter(smd.values())).keys())}
    else:
        ev["skater_summary_vs_boxscore"] = {"available": False}
    if ti and ti.get("data"):
        td = {r["playerId"]: r for r in ti["data"]}
        ex = next(iter(td.values()))
        bad = [(pid, r["timeOnIce"], bx[pid]["toi"]) for pid, r in td.items() if pid in bx and bx[pid]["toi"] is not None and abs(r["timeOnIce"] - bx[pid]["toi"]) > 1]
        sumcheck = [pid for pid, r in td.items() if abs((r.get("evTimeOnIce") or 0) + (r.get("ppTimeOnIce") or 0) + (r.get("shTimeOnIce") or 0) - r["timeOnIce"]) > 1]
        ev["strength_toi"] = {"available": True, "n_rows": len(td), "example_row": {k: ex[k] for k in ("playerId", "skaterFullName", "teamAbbrev", "timeOnIce", "evTimeOnIce", "ppTimeOnIce", "shTimeOnIce", "shifts")},
                              "total_toi_mismatch_vs_boxscore_gt_1s": bad, "ev_pp_sh_not_summing_to_total": sumcheck}
    else:
        ev["strength_toi"] = {"available": False}
    # ---- shifts
    if sh and sh.get("data"):
        rows_all = [r for r in sh["data"] if r.get("typeCode") == 517]
        seen, rows = set(), []
        for r in rows_all:
            k = (r["playerId"], r["period"], r["startTime"], r["endTime"])
            if k in seen:
                continue
            seen.add(k); rows.append(r)
        per = defaultdict(lambda: [0, 0])
        for r in rows:
            dur = toi_s(r["duration"]) if r.get("duration") else 0
            per[r["playerId"]][0] += dur or 0
            per[r["playerId"]][1] += 1
        bad = [(pid, v[0], bx[pid]["toi"]) for pid, v in per.items() if pid in bx and bx[pid]["toi"] is not None and abs(v[0] - bx[pid]["toi"]) > 2]
        ev["shifts"] = {"available": True, "n_raw_shift_rows": len(rows_all), "n_duplicate_rows_removed": len(rows_all) - len(rows), "n_shift_rows": len(rows), "n_players": len(per), "example": {k: rows[0][k] for k in ("playerId", "period", "shiftNumber", "startTime", "endTime", "duration", "teamAbbrev")},
                        "shift_toi_sum_vs_boxscore_toi_gt_2s": bad, "shift_count_mismatch_vs_boxscore": [(pid, v[1], bx[pid]["shifts"]) for pid, v in per.items() if pid in bx and bx[pid]["shifts"] is not None and v[1] != bx[pid]["shifts"]]}
    else:
        ev["shifts"] = {"available": False}
    # ---- scratches / officials in right-rail (post-hoc)
    if rr:
        gi = rr.get("gameInfo") or {}
        ev["right_rail_gameInfo"] = {"keys": sorted(gi.keys()), "away_scratches": len((gi.get("awayTeam") or {}).get("scratches", [])), "home_scratches": len((gi.get("homeTeam") or {}).get("scratches", []))}
    if html:
        season = str(box["season"])
        n = str(gid)[-6:]
        for tag in ("TH", "TV"):
            raw, mh = get(f"https://www.nhl.com/scores/htmlreports/{season}/{tag}{n}.HTM", cache, kind="raw")
            ev["sources"][f"html_{tag}"] = mh
            if raw:
                ev[f"html_{tag}_has_shift_rows"] = bool(re.search(rb"Shift #", raw)) or b"TOI" in raw
    return ev


def coverage(cache):
    out = {}
    for yr in COVERAGE_SEASONS:
        n = int(hashlib.sha256(f"nhl-cov-{yr}".encode()).hexdigest(), 16) % 1100 + 1
        gid = int(f"{yr}02{n:04d}")
        b, mb = get(f"{API}/gamecenter/{gid}/boxscore", cache)
        p, mp = get(f"{API}/gamecenter/{gid}/play-by-play", cache)
        s, ms = get(f"{STATS}/shiftcharts?cayenneExp=gameId={gid}", cache)
        t, mt = get(f"{STATS}/skater/timeonice?isGame=true&cayenneExp=gameId={gid}", cache)
        nsh = sum(1 for e in (p or {}).get("plays", []) if e["typeDescKey"] in ("shot-on-goal", "goal"))
        with_sit = sum(1 for e in (p or {}).get("plays", []) if "situationCode" in e)
        xy = sum(1 for e in (p or {}).get("plays", []) if e["typeDescKey"] == "shot-on-goal" and "xCoord" in e.get("details", {}))
        out[yr] = {"game_id": gid, "boxscore": bool(b), "box_players": (len(b["playerByGameStats"]["homeTeam"]["forwards"]) if b else 0), "pbp_events": len((p or {}).get("plays", [])), "pbp_sog_events": nsh, "pbp_with_situationCode": with_sit,
                   "pbp_sog_with_xy": xy, "pbp_rosterSpots": len((p or {}).get("rosterSpots", [])), "shiftcharts_rows": (s or {}).get("total"), "timeonice_rows": (t or {}).get("total"), "boxscore_status": mb.get("http_status")}
    return out


def pregame_probe():
    """Live (uncached) look at what the providers expose BEFORE puck drop, and how historical rosters are exposed. Never cached: the point is the current state."""
    def live(url):
        t0 = time.time()
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "nhl-phase0a"}), timeout=60)
            raw = r.read()
            return json.loads(raw), {"url": url, "retrieved_at_utc": now(), "sha256": hashlib.sha256(raw).hexdigest(), "headers": {k: v for k, v in r.headers.items() if k.lower() in ("last-modified", "etag", "cache-control")}, "seconds": round(time.time() - t0, 2)}
        except urllib.error.HTTPError as e:
            return None, {"url": url, "http_status": e.code}
    out = {"games": [], "rosters": {}}
    sc, m = live(f"{API}/schedule/now")
    out["schedule_now"] = {**m, "regularSeasonStartDate": sc.get("regularSeasonStartDate"), "n_days": len(sc["gameWeek"])}
    games = [(d["date"], g) for d in sc["gameWeek"] for g in d["games"]]
    fut = [g for g in games if g[1]["gameState"] in ("FUT", "PRE")][:2]
    liv = [g for g in games if g[1]["gameState"] in ("LIVE", "CRIT")][:1]
    for date, g in fut + liv:
        gid = g["id"]
        e = {"game_id": gid, "date": date, "gameState_at_probe": g["gameState"], "startTimeUTC": g["startTimeUTC"], "endpoints": {}}
        for ep in ("landing", "right-rail", "boxscore", "play-by-play"):
            d, mm = live(f"{API}/gamecenter/{gid}/{ep}")
            info = {"meta": mm}
            if d:
                if ep == "right-rail":
                    gi = d.get("gameInfo") or {}
                    info["gameInfo_keys"] = sorted(gi.keys())
                    info["scratches"] = {k: len((gi.get(k) or {}).get("scratches", [])) for k in ("awayTeam", "homeTeam")}
                if ep == "boxscore":
                    pb = d.get("playerByGameStats") or {}
                    info["n_skaters_listed"] = sum(len(pb.get(s, {}).get(q, [])) for s in ("awayTeam", "homeTeam") for q in ("forwards", "defense"))
                    info["n_goalies_listed"] = sum(len(pb.get(s, {}).get("goalies", [])) for s in ("awayTeam", "homeTeam"))
                if ep == "play-by-play":
                    info["n_plays"] = len(d.get("plays", [])); info["n_rosterSpots"] = len(d.get("rosterSpots", []))
                if ep == "landing":
                    info["has_matchup"] = "matchup" in d
                    info["keys"] = sorted(d.keys())
            e["endpoints"][ep] = info
        out["games"].append(e)
    for label, path in (("TOR_2022_2023", f"{API}/roster/TOR/20222023"), ("TOR_current", f"{API}/roster/TOR/current")):
        d, mm = live(path)
        out["rosters"][label] = {"meta": mm, "n": (sum(len(d.get(k, [])) for k in ("forwards", "defensemen", "goalies")) if d else None), "has_effective_date_or_history": bool(d) and any(k in d for k in ("asOf", "effectiveDate", "transactions"))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(REPO / "nhl_models" / "nhl_model.sqlite")
    t0 = time.time()
    games = pick_games(con)
    res = {"sample_game_ids": games, "sample_rule": "per season in %s the 2 completed regular-season game ids with the smallest sha256('nhl-phase0a-'+game_id) from nhl_model.sqlite.games" % (SAMPLE_SEASONS,), "games": []}
    for i, g in enumerate(games):
        res["games"].append(analyse_game(g, a.cache, con, html=(i in (0, len(games) - 1))))
    res["sample_seconds"] = round(time.time() - t0, 1)
    t1 = time.time()
    res["pregame_probe"] = pregame_probe()
    res["coverage_probe"] = coverage(a.cache)
    res["coverage_seconds"] = round(time.time() - t1, 1)
    res["operations"] = {"n_requests": len(OPS), "n_network": sum(not o["from_cache"] for o in OPS), "bytes_network": sum(o["bytes"] or 0 for o in OPS if not o["from_cache"]), "total_seconds": round(time.time() - t0, 1)}
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["operations"]))


if __name__ == "__main__":
    main()
