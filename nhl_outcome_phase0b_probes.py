"""
NHL_OUTCOME_PHASE0B_PROBES -- (C) shot-attempt / PBP join test and (D) PP / deployment proxies from PRIOR games only. Small deterministic samples; no model, no fitting.

  python -u nhl_outcome_phase0b_probes.py --cache DIR --out FILE --bench-table m2_table.json
"""
import argparse
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nhl_outcome_phase0b_benchmark as BM

API, ST = BM.API, BM.ST
UTC = timezone.utc
import nhl_outcome_contract as CT
GAME_LEN = timedelta(minutes=CT.GAME_MAX_MINUTES)          # deterministic cutoff rule from the contract
PRIOR_WINDOWS = [("2023-10-10", "2023-10-15"), ("2023-10-16", "2023-10-22"), ("2023-10-23", "2023-10-29"), ("2023-10-30", "2023-11-05"), ("2023-11-06", "2023-11-12")]
PBP_SAMPLE_GAMES = [2023020231, 2023020251, 2025020444, 2025020719, 2018020442, 2021021037]       # includes a shootout game (2025020444)


def cached(url, cache):
    k = hashlib.sha256(url.encode()).hexdigest()
    f = Path(cache) / f"{k}.bin"
    if f.exists():
        return json.loads(f.read_bytes()), hashlib.sha256(f.read_bytes()).hexdigest(), True
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "nhl-phase0b"}), timeout=120).read()
    f.write_bytes(raw)
    return json.loads(raw), hashlib.sha256(raw).hexdigest(), False


def parse_utc(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


# ------------------------------------------------------------------ C
def derive_attempts(gid, cache, table):
    pbp, sha, _ = cached(f"{API}/gamecenter/{gid}/play-by-play", cache)
    box, bsha, _ = cached(f"{API}/gamecenter/{gid}/boxscore", cache)
    home_id, away_id = box["homeTeam"]["id"], box["awayTeam"]["id"]
    dressed = {p["playerId"]: (side, p) for side in ("awayTeam", "homeTeam") for pos in ("forwards", "defense") for p in box["playerByGameStats"][side][pos]}
    team_of_player = {pid: box[side]["id"] for pid, (side, _) in dressed.items()}
    att = defaultdict(Counter)
    so_events = 0
    unattributed = 0
    owner_mismatch = 0
    for e in pbp["plays"]:
        t = e["typeDescKey"]
        if t not in ("shot-on-goal", "goal", "missed-shot", "blocked-shot"):
            continue
        if e["periodDescriptor"].get("periodType") == "SO":
            so_events += 1
            continue
        d = e.get("details", {})
        sid = d.get("shootingPlayerId") or d.get("scoringPlayerId")
        if not sid:
            unattributed += 1
            continue
        kind = "goal" if t == "goal" else t
        att[sid][kind] += 1
        # team consistency: for shots/goals/misses the event owner is the shooter's team; for blocked shots the event owner is the shooter's team too?  recorded, not assumed
        if d.get("eventOwnerTeamId") is not None and team_of_player.get(sid) is not None and d["eventOwnerTeamId"] != team_of_player[sid]:
            owner_mismatch += 1
    rows = []
    for pid, (side, p) in dressed.items():
        c = att.get(pid, Counter())
        sog = c["shot-on-goal"] + c["goal"]
        attempts = sog + c["missed-shot"] + c["blocked-shot"]
        t = table.get((gid, pid))
        toi = t["toi"] if t else None
        rows.append({"playerId": pid, "name": p["name"]["default"], "box_sog": p["sog"], "pbp_sog": sog, "missed": c["missed-shot"], "blocked_attempts": c["blocked-shot"], "goals": c["goal"], "attempts": attempts,
                     "toi_s": toi, "attempts_per_60": round(attempts * 3600 / toi, 2) if toi else None, "joined_to_minimal_table": t is not None})
    shooters_not_dressed = [pid for pid in att if pid not in dressed]
    goalie_ids = {g["playerId"] for side in ("awayTeam", "homeTeam") for g in box["playerByGameStats"][side]["goalies"]}
    return {"game_id": gid, "pbp_sha256": sha, "boxscore_sha256": bsha, "n_shootout_shot_events_excluded": so_events, "has_shootout": any(e["periodDescriptor"].get("periodType") == "SO" for e in pbp["plays"]),
            "unattributed_shot_events": unattributed, "shooter_team_mismatch_with_eventOwnerTeamId": owner_mismatch, "shooters_not_in_dressed_skaters": shooters_not_dressed, "shooters_who_are_goalies": [p for p in shooters_not_dressed if p in goalie_ids],
            "sog_mismatch_players": [r for r in rows if r["box_sog"] != r["pbp_sog"]], "attempts_lt_sog": [r for r in rows if r["attempts"] < r["pbp_sog"]],
            "team_sog_pbp_vs_box": {"away": [sum(r["pbp_sog"] for r in rows if dressed[r["playerId"]][0] == "awayTeam"), box["awayTeam"]["sog"]], "home": [sum(r["pbp_sog"] for r in rows if dressed[r["playerId"]][0] == "homeTeam"), box["homeTeam"]["sog"]]},
            "join_rate_to_minimal_table": round(sum(r["joined_to_minimal_table"] for r in rows) / len(rows), 3), "n_skaters": len(rows),
            "totals": {"attempts": sum(r["attempts"] for r in rows), "sog": sum(r["pbp_sog"] for r in rows), "missed": sum(r["missed"] for r in rows), "blocked": sum(r["blocked_attempts"] for r in rows)},
            "top_attempt_rows": sorted(rows, key=lambda r: -r["attempts"])[:4]}


# ------------------------------------------------------------------ D
def team_totals(table):
    tot = defaultdict(lambda: {"pp": 0, "ev": 0, "sh": 0})
    for (g, p), r in table.items():
        for k in ("pp", "ev", "sh"):
            tot[(g, r["team"])][k] += r[k] or 0
    return tot


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def deployment_probe(table, target_ids, cache, counter):
    """table: {(gameId, playerId): row} covering prior windows + the target week. For every target team-game, features use only rows of games completed (start + 3h) before the target game's cutoff."""
    start_of = {g: parse_utc(r["startTimeUTC"]) for (g, _), r in table.items() if r["startTimeUTC"]}
    tt = team_totals(table)
    by_player = defaultdict(list)
    for (g, p), r in table.items():
        by_player[p].append((start_of[g], g, r))
    for p in by_player:
        by_player[p].sort()
    out_rows, leak_checks = [], 0
    cutoff_min = 90
    for g in sorted(target_ids):
        T = start_of[g] - timedelta(minutes=cutoff_min)
        roster = [(p, r) for (gg, p), r in table.items() if gg == g]
        per_team = defaultdict(list)
        for p, r in roster:
            hist = [(s, gg, rr) for (s, gg, rr) in by_player[p] if s + GAME_LEN <= T and gg != g]
            leak_checks += sum(1 for (s, gg, rr) in hist if s >= start_of[g])
            last5 = hist[-5:]
            cat = CT.classify_player(p, r["team"], [{"team": rr["team"]} for (_, _, rr) in hist])
            if not last5:
                per_team[r["team"]].append({"playerId": p, "cold": True, "cat": cat, "target": r}); continue
            def share(rr, key):
                den = tt[(rr["gameId"], rr["team"])][key]
                return (rr[key] or 0) / den if den else 0.0
            pp5 = [share(rr, "pp") for (_, _, rr) in last5]; ev5 = [share(rr, "ev") for (_, _, rr) in last5]
            per_team[r["team"]].append({"playerId": p, "cold": False, "cat": cat, "n_prior": len(last5), "pp_share5": sum(pp5) / len(pp5), "ev_share5": sum(ev5) / len(ev5), "shifts5": sum(rr["shifts"] or 0 for (_, _, rr) in last5) / len(last5),
                                        "pp_share_last": pp5[-1], "pp_change": (sum(pp5[-2:]) / len(pp5[-2:])) - sum(pp5) / len(pp5), "target": r})
        for team, lst in per_team.items():
            warm = [x for x in lst if not x["cold"]]
            ranked = sorted(warm, key=lambda x: -x["pp_share5"])
            for i, x in enumerate(ranked):
                x["pp_rank"] = i + 1
                x["proxy"] = "PP1" if i < 5 and x["pp_share5"] > 0.05 else "PP2" if i < 10 and x["pp_share5"] > 0.02 else "NONE"
            ranked_last = sorted(warm, key=lambda x: -x["pp_share_last"])
            tgt_rank = sorted(lst, key=lambda x: -(x["target"]["pp"] or 0))
            top5_t = {x["playerId"] for x in tgt_rank[:5] if (x["target"]["pp"] or 0) > 0}
            out_rows.append({"game": g, "team": team, "warm": warm, "cold": [x for x in lst if x["cold"]], "top5_target": top5_t, "proxy_top5": {x["playerId"] for x in ranked[:5]}, "last_game_top5": {x["playerId"] for x in ranked_last[:5]}})
    return out_rows, leak_checks


def summarize_deployment(rows):
    recall_proxy, recall_last, n_tg = [], [], 0
    xs, ys, ev_x, ev_y, sh_x, sh_y = [], [], [], [], [], []
    pp1_hits = pp1_n = pp2_hits = pp2_n = none_hits = none_n = 0
    cold = Counter()
    chg = []
    for r in rows:
        if not r["top5_target"]:
            continue
        n_tg += 1
        recall_proxy.append(len(r["top5_target"] & r["proxy_top5"]) / len(r["top5_target"]))
        recall_last.append(len(r["top5_target"] & r["last_game_top5"]) / len(r["top5_target"]))
        for x in r["warm"]:
            t = x["target"]
            tg = team_tot_of(r, t)
            xs.append(x["pp_share5"]); ys.append((t["pp"] or 0))
            ev_x.append(x["ev_share5"]); ev_y.append(t["ev"] or 0)
            sh_x.append(x["shifts5"]); sh_y.append(t["shifts"] or 0)
            chg.append(abs(x["pp_change"]))
            if x["proxy"] == "PP1":
                pp1_n += 1; pp1_hits += (t["pp"] or 0) >= 60
            elif x["proxy"] == "PP2":
                pp2_n += 1; pp2_hits += (t["pp"] or 0) >= 60
            else:
                none_n += 1; none_hits += (t["pp"] or 0) >= 60
        cold["cold_start_players"] += len(r["cold"])
        cold["warm_players"] += len(r["warm"])
        for x in r["warm"] + r["cold"]:
            cold["state_" + x["cat"]] += 1
    return {"team_games_with_pp_time": n_tg, "mean_recall_of_target_top5_PP_by_prior5_proxy": round(sum(recall_proxy) / len(recall_proxy), 3), "mean_recall_by_last_game_top5": round(sum(recall_last) / len(recall_last), 3),
            "chance_recall_top5_of_~18_skaters": round(5 / 18, 3),
            "spearman_prior5_pp_share_vs_target_pp_toi": round(spearman(xs, ys), 3), "spearman_prior5_ev_share_vs_target_ev_toi": round(spearman(ev_x, ev_y), 3), "spearman_prior5_shifts_vs_target_shifts": round(spearman(sh_x, sh_y), 3),
            "P(target_PP_TOI>=60s | proxy PP1)": [pp1_hits, pp1_n, round(pp1_hits / pp1_n, 3) if pp1_n else None], "P(.. | proxy PP2)": [pp2_hits, pp2_n, round(pp2_hits / pp2_n, 3) if pp2_n else None],
            "P(.. | proxy NONE)": [none_hits, none_n, round(none_hits / none_n, 3) if none_n else None], "median_abs_rolling_pp_share_change": round(sorted(chg)[len(chg) // 2], 4), "players": dict(cold)}


def team_tot_of(r, t):
    return None


def cold_start_classification(rows, cache, upto_date):
    """For players with NO prior game in the loaded window, ask the stats API for ANY earlier game (all seasons) to separate 'NHL debut' from 'returning / not yet played this window'."""
    cold = {}
    for r in rows:
        for x in r["cold"]:
            cold[x["playerId"]] = x["target"]
    res = {"n_cold_in_window": len(cold), "debut": [], "has_earlier_games": []}
    for pid, t in sorted(cold.items())[:40]:
        url = f"{ST}/skater/summary?" + urllib.parse.urlencode({"isGame": "true", "limit": "1", "cayenneExp": f'playerId={pid} and gameDate<"{upto_date}" and gameTypeId=2'}, quote_via=urllib.parse.quote)
        d, _, _ = cached(url, cache)
        (res["has_earlier_games"] if d["total"] > 0 else res["debut"]).append({"playerId": pid, "name": t["name"], "team": t["team"], "earlier_regular_season_games_all_seasons": d["total"]})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True); ap.add_argument("--bench-table", required=True)
    a = ap.parse_args()
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tab = {(r["gameId"], r["playerId"]): r for r in json.load(open(a.bench_table))}
    res = {"C_shot_attempts": {"sample_games": PBP_SAMPLE_GAMES, "games": []}, "D_deployment": {}}
    # games not in the benchmark table (other seasons) get their rows from boxscore/timeonice via cache for the join test
    for gid in PBP_SAMPLE_GAMES:
        t = dict(tab)
        if not any(k[0] == gid for k in t):
            box, _, _ = cached(f"{API}/gamecenter/{gid}/boxscore", a.cache)
            ti, _, _ = cached(f"{ST}/skater/timeonice?" + urllib.parse.urlencode({"isGame": "true", "limit": "-1", "cayenneExp": f"gameId={gid}"}, quote_via=urllib.parse.quote), a.cache)
            tid = {r["playerId"]: r for r in ti["data"]}
            for side in ("awayTeam", "homeTeam"):
                for pos in ("forwards", "defense"):
                    for p in box["playerByGameStats"][side][pos]:
                        t[(gid, p["playerId"])] = {"toi": BM.mmss(p["toi"]), "team": box[side]["abbrev"]}
        res["C_shot_attempts"]["games"].append(derive_attempts(gid, a.cache, t))
    cg = res["C_shot_attempts"]["games"]
    res["C_shot_attempts"]["summary"] = {"games": len(cg), "sog_mismatch_players": sum(len(g["sog_mismatch_players"]) for g in cg), "attempts_lt_sog": sum(len(g["attempts_lt_sog"]) for g in cg), "shooters_not_in_dressed_skaters": sum(len(g["shooters_not_in_dressed_skaters"]) for g in cg),
                                         "unattributed_shot_events": sum(g["unattributed_shot_events"] for g in cg), "shootout_games": [g["game_id"] for g in cg if g["has_shootout"]], "shootout_events_excluded": sum(g["n_shootout_shot_events_excluded"] for g in cg),
                                         "min_join_rate_to_minimal_table": min(g["join_rate_to_minimal_table"] for g in cg), "team_sog_equal": all(g["team_sog_pbp_vs_box"]["away"][0] == g["team_sog_pbp_vs_box"]["away"][1] and g["team_sog_pbp_vs_box"]["home"][0] == g["team_sog_pbp_vs_box"]["home"][1] for g in cg),
                                         "shooter_team_mismatch_events": sum(g["shooter_team_mismatch_with_eventOwnerTeamId"] for g in cg)}
    # D: for two target weeks (a November week and the trade-deadline week) load the 5 prior weekly windows + the target week
    res["D_deployment"] = {"targets": []}
    for tlo in ("2023-11-13", "2024-03-04"):
        t_start = datetime.strptime(tlo, "%Y-%m-%d")
        wins = [((t_start - timedelta(days=7 * k)).strftime("%Y-%m-%d"), (t_start - timedelta(days=7 * k - 6)).strftime("%Y-%m-%d")) for k in range(5, 0, -1)]
        thi = (t_start + timedelta(days=6)).strftime("%Y-%m-%d")
        counter = {"requests": 0, "bytes": 0, "seconds": 0.0}
        dtab, games_all = {}, {}
        for lo, hi in wins + [(tlo, thi)]:
            games = BM.week_games(lo, counter)
            games_all.update(games)
            rows, _ = BM.m2_batched(lo, hi, games, counter)
            dtab.update({k: v for k, v in rows.items() if k[0] in games})
        target = {g for g, v in games_all.items() if tlo <= v["date"] <= thi}
        drows, leaks = deployment_probe(dtab, target, a.cache, counter)
        res["D_deployment"]["targets"].append({"target_week": [tlo, thi], "prior_windows": wins, "acquisition": {**counter, "windows": len(wins) + 1}, "target_games": len(target), "rows_loaded": len(dtab),
                                               "leak_checks_prior_rows_with_start_ge_target_start": leaks, "results": summarize_deployment(drows), "cold_start": cold_start_classification(drows, a.cache, tlo)})
    res["D_deployment"]["cutoff_rule"] = "features for a target game use only rows of games with start + %d min <= (target start - 90 min); never the target game" % CT.GAME_MAX_MINUTES
    res["D_deployment"]["definitions"] = {"pp_share": "player PP TOI / sum of all skaters' PP TOI of that team-game", "ev_share": "player EV TOI / sum of team EV TOI", "prior5": "mean over the last <=5 prior games", "PP1 proxy": "prior-5 PP-share rank <= 5 and share > 0.05",
                                          "PP2 proxy": "rank 6-10 and share > 0.02", "pp_change": "mean(last 2 prior games) - mean(last 5)", "state": "KNOWN / ACQUIRED / COLD from nhl_outcome_contract.classify_player"}
    res["D_deployment"]["linemates_from_shift_overlap"] = "not tested (optional; deferred)"
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(json.dumps({"C": res["C_shot_attempts"]["summary"], "D": [(t["target_week"], t["results"]) for t in res["D_deployment"]["targets"]], "seconds": res["seconds"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
