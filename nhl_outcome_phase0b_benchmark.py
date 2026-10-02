"""
NHL_OUTCOME_PHASE0B_BENCHMARK -- acquisition strategies for the minimal skater-game table, on one deterministic sample (3 one-week windows in 3 seasons). No model, no full-season crawl.

  M1 per-game      boxscore + stats-REST timeonice(gameId)                    2 requests / game
  M2 batched       api-web schedule(week) + stats-REST summary(date window) + timeonice(date window)    3 requests / week
  M3 existing      the repo's nhl_player_games_foundation_a method (team-season, teamAbbrevs filter, 100-row pages)  (run for 3 team-seasons)
Truth = boxscore player sets per game. Network timings are fresh (no cache) and recorded per strategy.
  python -u nhl_outcome_phase0b_benchmark.py --out FILE --table-out FILE
"""
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

API = "https://api-web.nhle.com/v1"
ST = "https://api.nhle.com/stats/rest/en"
WEEKS = [("2023-11-13", "2023-11-19", "baseline week 2023-24"), ("2024-03-04", "2024-03-10", "trade-deadline week 2023-24 (deadline 2024-03-08)"), ("2025-10-13", "2025-10-19", "early-season week 2025-26 (call-ups / cold starts)")]
TEAM_SEASONS = [("TOR", 2023), ("MTL", 2023), ("MTL", 2025)]
NET = {"requests": 0, "bytes": 0, "seconds": 0.0}


def get(url, counter):
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "nhl-phase0b"}), timeout=120)
    raw = r.read()
    dt = time.time() - t0
    counter["requests"] += 1; counter["bytes"] += len(raw); counter["seconds"] += dt
    return json.loads(raw)


def qs(path, **kw):
    return ST + path + "?" + urllib.parse.urlencode(kw, quote_via=urllib.parse.quote)


def mmss(s):
    m, x = s.split(":")
    return int(m) * 60 + int(x)


def week_games(lo, counter):
    """Completed regular-season games of the 7-day window, from api-web schedule (one request)."""
    d = get(f"{API}/schedule/{lo}", counter)
    out = {}
    for day in d["gameWeek"]:
        for g in day["games"]:
            if g["gameType"] == 2 and g["gameState"] in ("OFF", "FINAL"):
                out[g["id"]] = {"startTimeUTC": g["startTimeUTC"], "away": g["awayTeam"]["abbrev"], "home": g["homeTeam"]["abbrev"], "awayTeamId": g["awayTeam"]["id"], "homeTeamId": g["homeTeam"]["id"], "date": day["date"]}
    return out


def m1_per_game(games, counter):
    rows = {}
    for gid, g in games.items():
        box = get(f"{API}/gamecenter/{gid}/boxscore", counter)
        ti = get(qs("/skater/timeonice", isGame="true", limit="-1", cayenneExp=f"gameId={gid}"), counter)
        tid = {r["playerId"]: r for r in ti["data"]}
        for side, other in (("awayTeam", "homeTeam"), ("homeTeam", "awayTeam")):
            for pos in ("forwards", "defense"):
                for p in box["playerByGameStats"][side][pos]:
                    t = tid.get(p["playerId"], {})
                    rows[(gid, p["playerId"])] = {"gameId": gid, "playerId": p["playerId"], "teamId": box[side]["id"], "team": box[side]["abbrev"], "opponent": box[other]["abbrev"], "startTimeUTC": box["startTimeUTC"],
                                                  "sog": p["sog"], "toi": mmss(p["toi"]), "ev": t.get("evTimeOnIce"), "pp": t.get("ppTimeOnIce"), "sh": t.get("shTimeOnIce"), "shifts": p.get("shifts"), "pos": p["position"], "name": p["name"]["default"]}
    return rows


def window_rows(path, lo, hi, counter):
    d = get(qs(path, isGame="true", limit="-1", cayenneExp=f'gameDate>="{lo}" and gameDate<="{hi}" and gameTypeId=2'), counter)
    if d["total"] >= 10000 or len(d["data"]) != d["total"]:
        raise RuntimeError(f"window {lo}..{hi} hit the 10,000-row cap or truncated: total={d['total']} rows={len(d['data'])}")
    return d["data"]


def m2_batched(lo, hi, games, counter):
    sm = window_rows("/skater/summary", lo, hi, counter)
    ti = window_rows("/skater/timeonice", lo, hi, counter)
    tid = {(r["gameId"], r["playerId"]): r for r in ti}
    teamid = {}
    for gid, g in games.items():
        teamid[(gid, g["away"])] = g["awayTeamId"]; teamid[(gid, g["home"])] = g["homeTeamId"]
    rows = {}
    for r in sm:
        k = (r["gameId"], r["playerId"])
        t = tid.get(k, {})
        g = games.get(r["gameId"])
        rows[k] = {"gameId": r["gameId"], "playerId": r["playerId"], "teamId": teamid.get((r["gameId"], r["teamAbbrev"])), "team": r["teamAbbrev"], "opponent": r["opponentTeamAbbrev"], "startTimeUTC": g["startTimeUTC"] if g else None,
                   "sog": r["shots"], "toi": t.get("timeOnIce"), "ev": t.get("evTimeOnIce"), "pp": t.get("ppTimeOnIce"), "sh": t.get("shTimeOnIce"), "shifts": t.get("shifts"), "pos": r["positionCode"], "name": r["skaterFullName"]}
    return rows, {"summary_rows": len(sm), "timeonice_rows": len(ti), "summary_keys_not_in_timeonice": len({(r["gameId"], r["playerId"]) for r in sm} - set(tid)), "timeonice_keys_not_in_summary": len(set(tid) - {(r["gameId"], r["playerId"]) for r in sm})}


def m3_existing(abbrev, season, counter):
    rows, start, total, pages = {}, 0, None, 0
    while total is None or start < total:
        q = urllib.parse.urlencode({"isAggregate": "false", "isGame": "true", "start": start, "limit": 2000, "cayenneExp": f'seasonId={season}{season + 1} and gameTypeId=2 and teamAbbrevs="{abbrev}"'})
        d = get(f"{ST}/skater/summary?{q}", counter)
        pages += 1
        if not d.get("data"):
            break
        for r in d["data"]:
            rows[(r["gameId"], r["playerId"])] = r
        total = d.get("total", len(rows)); start += len(d["data"])
    return rows, pages


def compare(truth, cand, games, label):
    miss, extra, sogm, toim, shm = [], [], [], [], []
    for k, t in truth.items():
        c = cand.get(k)
        if c is None:
            miss.append(k); continue
        if (c.get("sog") if isinstance(c, dict) and "sog" in c else c.get("shots")) != t["sog"]:
            sogm.append(k)
        if "toi" in c and c["toi"] is not None and abs(c["toi"] - t["toi"]) > 1:
            toim.append(k)
        if "shifts" in c and c["shifts"] is not None and t["shifts"] is not None and c["shifts"] != t["shifts"]:
            shm.append(k)
    extra = [k for k in cand if k not in truth and k[0] in games]
    return {"label": label, "truth_rows": len(truth), "candidate_rows_in_sample_games": sum(1 for k in cand if k[0] in games), "missing_vs_boxscore": len(miss), "extra_vs_boxscore": len(extra), "sog_mismatch": len(sogm),
            "toi_mismatch_gt_1s": len(toim), "shift_count_mismatch": len(shm), "missing_examples": [{"gameId": k[0], "playerId": k[1], "name": truth[k]["name"], "team": truth[k]["team"], "sog": truth[k]["sog"]} for k in miss[:5]],
            "missing_players_with_sog_gt0": sum(1 for k in miss if truth[k]["sog"] > 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--table-out", required=True)
    a = ap.parse_args()
    res = {"weeks": [], "strategies": {}, "sample_rule": "three fixed one-week windows (see weeks); every completed regular-season game in each window"}
    c1, c2, c3 = ({"requests": 0, "bytes": 0, "seconds": 0.0} for _ in range(3))
    all_games, truth, m2tab = {}, {}, {}
    t1 = t2 = 0.0
    for lo, hi, label in WEEKS:
        games = week_games(lo, c2)
        all_games.update(games)
        s = time.time(); tr = m1_per_game(games, c1); t1 += time.time() - s
        truth.update(tr)
        s = time.time(); rows, detail = m2_batched(lo, hi, games, c2); t2 += time.time() - s
        in_sample = {k: v for k, v in rows.items() if k[0] in games}
        m2tab.update(in_sample)
        res["weeks"].append({"window": [lo, hi], "label": label, "games": len(games), "m2_detail": detail, "m2_vs_boxscore": compare(tr, rows, games, "M2 batched")})
    res["strategies"]["M1_per_game_boxscore_plus_timeonice"] = {**c1, "bytes_MB": round(c1["bytes"] / 1e6, 1), "games": len(all_games), "requests_per_game": 2, "wall_seconds": round(t1, 1), "rows": len(truth), "mismatch_vs_truth": "n/a (is the truth source for set membership)"}
    res["strategies"]["M2_batched_date_window"] = {**c2, "bytes_MB": round(c2["bytes"] / 1e6, 1), "windows": len(WEEKS), "requests_per_week": 3, "wall_seconds": round(t2, 1), "rows": len(m2tab)}
    # M3: the existing method
    m3res = []
    t3 = time.time()
    for ab, season in TEAM_SEASONS:
        rows, pages = m3_existing(ab, season, c3)
        sample_team_games = {g for g, v in all_games.items() if ab in (v["away"], v["home"]) and str(g).startswith(str(season))}
        tr = {k: v for k, v in truth.items() if k[0] in sample_team_games and v["team"] == ab}
        m3res.append({"team_season": f"{ab} {season}", "pages": pages, "rows_total": len(rows), "vs_boxscore_for_this_team_in_sample_weeks": compare(tr, {k: {**v, "sog": v["shots"], "toi": v.get("timeOnIcePerGame")} for k, v in rows.items()}, sample_team_games, "M3 existing")})
    res["strategies"]["M3_existing_team_season_loop"] = {**c3, "bytes_MB": round(c3["bytes"] / 1e6, 1), "wall_seconds": round(time.time() - t3, 1), "team_seasons_run": len(TEAM_SEASONS), "per_team_season": m3res}
    # traded players by season aggregate (isGame=false): teamAbbrevs containing a comma
    d = get(qs("/skater/summary", isAggregate="false", isGame="false", limit="-1", cayenneExp="seasonId=20232024 and gameTypeId=2"), c2)
    traded23 = {r["playerId"]: r["teamAbbrevs"] for r in d["data"] if "," in (r.get("teamAbbrevs") or "")}
    d2 = get(qs("/skater/summary", isAggregate="false", isGame="false", limit="-1", cayenneExp="seasonId=20252026 and gameTypeId=2"), c2)
    traded25 = {r["playerId"]: r["teamAbbrevs"] for r in d2["data"] if "," in (r.get("teamAbbrevs") or "")}
    tr_in_truth = {k for k, v in truth.items() if v["playerId"] in traded23 or v["playerId"] in traded25}
    res["traded_players"] = {"definition": "season-aggregate row with a comma in teamAbbrevs (played for >=2 teams)", "n_traded_2023_24": len(traded23), "n_traded_2025_26_so_far": len(traded25), "traded_skater_games_in_sample": len(tr_in_truth),
                             "traded_skater_games_found_by_M2": sum(1 for k in tr_in_truth if k in m2tab), "examples": [{"playerId": truth[k]["playerId"], "name": truth[k]["name"], "gameId": k[0], "team": truth[k]["team"], "m2_has_row": k in m2tab} for k in sorted(tr_in_truth)[:6]]}
    # does the existing method ever return a multi-team player? (rows keyed by player -> distinct teams for the season)
    res["cap_test"] = {}
    for lo, hi in (("2023-10-10", "2023-11-30"),):
        d = get(qs("/skater/summary", isGame="true", limit="-1", cayenneExp=f'gameDate>="{lo}" and gameDate<="{hi}" and gameTypeId=2'), c2)
        res["cap_test"] = {"window": [lo, hi], "reported_total": d["total"], "rows_returned": len(d["data"]), "distinct_games": len({r["gameId"] for r in d["data"]}), "silently_truncated_at_10000": d["total"] == 10000,
                           "rule": "a window is accepted only if total < 10000 and rows == total; weekly windows are ~1,600-2,200 rows"}
    # 100-row pagination on one week: unsorted vs explicitly sorted vs limit=-1
    lo, hi = WEEKS[0][0], WEEKS[0][1]
    wk0 = {k for k in m2tab if all_games[k[0]]["date"] >= lo and all_games[k[0]]["date"] <= hi}
    SORT = '[{"property":"gameId","direction":"ASC"},{"property":"playerId","direction":"ASC"}]'

    def paged(cay, sort, extra=None):
        cp = {"requests": 0, "bytes": 0, "seconds": 0.0}
        keys, start = [], 0
        while True:
            kw = dict(isGame="true", start=str(start), limit="100", cayenneExp=cay, **(extra or {}))
            if sort:
                kw["sort"] = SORT
            d = get(qs("/skater/summary", **kw), cp)
            if not d["data"]:
                break
            keys += [(r["gameId"], r["playerId"]) for r in d["data"]]
            start += len(d["data"])
            if start >= d["total"]:
                break
        return keys, d["total"], cp
    cay = f'gameDate>="{lo}" and gameDate<="{hi}" and gameTypeId=2'
    res["pagination_check"] = {"window": [lo, hi], "page_size": 100, "reference_rows_limit_minus_1": len(wk0)}
    for tag, srt in (("unsorted", False), ("explicit_sort_gameId_playerId", True)):
        keys, total, cp = paged(cay, srt)
        res["pagination_check"][tag] = {**cp, "rows_returned": len(keys), "unique_keys": len(set(keys)), "duplicates": len(keys) - len(set(keys)), "reported_total": total, "missing_vs_limit_minus_1": len(wk0 - set(keys)), "complete": set(keys) == wk0}
    # defect diagnosis on the existing method (TOR 2023-24, sample-week games)
    tor_truth = {k: v for k, v in truth.items() if v["team"] == "TOR" and str(k[0]).startswith("2023")}
    ce = 'seasonId=20232024 and gameTypeId=2 and teamAbbrevs="TOR"'
    agg = get(qs("/skater/summary", isAggregate="false", isGame="false", limit="-1", cayenneExp="seasonId=20232024 and gameTypeId=2"), c2)["data"]
    tr_ids = {r["playerId"] for r in agg if "," in (r.get("teamAbbrevs") or "")}
    diag = {"team_season": "TOR 2023-24", "sample_truth_rows": len(tor_truth)}
    for tag, srt in (("existing_unsorted_100_per_page", False), ("explicit_sort_100_per_page", True)):
        keys, total, cp = paged(ce, srt, {"isAggregate": "false"})
        ks = set(keys)
        diag[tag] = {"rows_returned": len(keys), "unique": len(ks), "reported_total": total, "duplicate_rows_that_displace_real_rows": len(keys) - len(ks), "missing_sample_rows": sum(1 for k in tor_truth if k not in ks)}
    d1 = get(qs("/skater/summary", isAggregate="false", isGame="true", limit="-1", cayenneExp=ce), c2)
    ks1 = {(r["gameId"], r["playerId"]) for r in d1["data"]}
    miss1 = [tor_truth[k] for k in tor_truth if k not in ks1]
    diag["team_filter_with_limit_minus_1"] = {"rows": len(d1["data"]), "reported_total": d1["total"], "missing_sample_rows": len(miss1), "missing_players": sorted({(m["name"], m["playerId"] in tr_ids) for m in miss1}),
                                              "note": "name, is_multi_team_season_player; the teamAbbrevs filter itself drops players even with no pagination"}
    mw = {k for k in m2tab if m2tab[k]["team"] == "TOR" and str(k[0]).startswith("2023")}
    diag["date_window_query_for_same_games"] = {"missing_sample_rows": sum(1 for k in tor_truth if k not in m2tab)}
    diag["conclusion"] = "two independent causes: (1) unsorted offset pagination returns duplicate rows and silently drops real ones; explicit sort fixes it; (2) the teamAbbrevs team filter excludes some players (multi-team-season players and others) regardless of pagination; a gameDate-window query without a team filter has neither problem."
    res["missing_player_defect_diagnosis"] = diag
    # completeness invariant usable without boxscore
    inv = defaultdict(lambda: [0, 0, 0])
    for k, r in m2tab.items():
        acc = inv[(r["gameId"], r["team"])]
        acc[0] += 1; acc[1] += (r["toi"] or 0)
    res["completeness_invariant"] = {"rule": "per team-game: >=17 skater rows and sum(skater TOI) in [16500, 19500] s for a regulation game (5 skaters x 3600 s minus penalties; OT adds up to 1500 s)",
                                     "team_games_checked": len(inv), "violations": [{"gameId": g, "team": t, "rows": v[0], "toi_sum": v[1]} for (g, t), v in inv.items() if v[0] < 17 or not (16500 <= v[1] <= 21000)][:10]}
    res["estimate_full_season"] = {}
    per_week_req = 3
    weeks_per_season = 28
    res["estimate_full_season"] = {"games_per_season": 1312, "M1_per_game_requests": 1312 * 2, "M1_seconds_est": round(1312 * 2 * (res["strategies"]["M1_per_game_boxscore_plus_timeonice"]["wall_seconds"] / max(res["strategies"]["M1_per_game_boxscore_plus_timeonice"]["requests"], 1)), 0),
                                   "M2_requests": weeks_per_season * per_week_req, "M2_seconds_est": round(weeks_per_season * res["strategies"]["M2_batched_date_window"]["wall_seconds"] / len(WEEKS), 0),
                                   "M3_requests_est": round(33 * res["strategies"]["M3_existing_team_season_loop"]["requests"] / len(TEAM_SEASONS)), "M3_seconds_est": round(33 * res["strategies"]["M3_existing_team_season_loop"]["wall_seconds"] / len(TEAM_SEASONS), 0)}
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    json.dump([{**v} for v in m2tab.values()], open(a.table_out, "w"))
    print(json.dumps({"strategies": res["strategies"], "estimate": res["estimate_full_season"]}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
