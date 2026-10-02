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


def _team_totals(rows):
    tot = defaultdict(lambda: {"pp": 0, "ev": 0})
    for r in rows:
        tot[(r["gameId"], r["team"])]["pp"] += r["pp"] or 0
        tot[(r["gameId"], r["team"])]["ev"] += r["ev"] or 0
    return tot


def deployment_probe(rows, games_meta, target_ids, cutoff_minutes=90, grading_rows=None):
    """Candidate universe + prior features FIRST, from allowed prior rows only; target-game rows are read afterwards, for grading and coverage only.

    rows          every row of the loaded history (list of dicts: gameId, playerId, team, startTimeUTC, pp, ev, sh, shifts, sog, toi, ...). Target-game rows may be present in it; they are removed by the
                  contract's allowed-rows rule before ANYTHING is computed.
    games_meta    {gameId: {"startTimeUTC", "home", "away"}} from the SCHEDULE: the target team identities never come from target-game skater rows.
    grading_rows  (optional) the rows used for grading; default = `rows`.
    Returns one dict per target team-game."""
    grading_rows = rows if grading_rows is None else grading_rows
    out = []
    for gid in sorted(target_ids):
        meta = games_meta[gid]
        T = CT.cutoff_time(CT.parse_utc(meta["startTimeUTC"]), "T90") if cutoff_minutes == 90 else CT.parse_utc(meta["startTimeUTC"]) - timedelta(minutes=cutoff_minutes)
        allowed = CT.prior_rows(rows, gid, T)                                   # nothing outside this list is visible to candidate construction or features
        totals = _team_totals(allowed)
        by_team_player = defaultdict(list)
        for r in allowed:
            by_team_player[(r["team"], r["playerId"])].append(r)
        for team in (meta["home"], meta["away"]):
            uni = CT.candidate_universe(allowed, gid, team, T)
            feats = {}
            for pid in uni["candidates"]:
                hist = sorted(by_team_player[(team, pid)], key=lambda r: (r["startTimeUTC"], r["gameId"]))[-5:]          # CURRENT-TEAM appearances only
                pp_sh = [(r["pp"] or 0) / totals[(r["gameId"], team)]["pp"] if totals[(r["gameId"], team)]["pp"] else 0.0 for r in hist]
                ev_sh = [(r["ev"] or 0) / totals[(r["gameId"], team)]["ev"] if totals[(r["gameId"], team)]["ev"] else 0.0 for r in hist]
                feats[pid] = {"n_current_team_appearances": len(hist), "prior_pp_toi_mean": sum(r["pp"] or 0 for r in hist) / len(hist), "prior_ev_toi_mean": sum(r["ev"] or 0 for r in hist) / len(hist),
                              "prior_shifts_mean": sum(r["shifts"] or 0 for r in hist) / len(hist), "pp_allocation_share5": sum(pp_sh) / len(pp_sh), "ev_allocation_share5": sum(ev_sh) / len(ev_sh),
                              "pp_share_change_recent_vs_long": (sum(pp_sh[-2:]) / len(pp_sh[-2:]) - sum(pp_sh) / len(pp_sh)) if len(pp_sh) >= 3 else None}
            ranked = sorted(feats, key=lambda p: (-feats[p]["pp_allocation_share5"], str(p)))
            for i, pid in enumerate(ranked):
                feats[pid]["pp_rank"] = i + 1
                feats[pid]["pp_proxy"] = "PP1" if i < 5 and feats[pid]["pp_allocation_share5"] > 0.05 else "PP2" if i < 10 and feats[pid]["pp_allocation_share5"] > 0.02 else "NONE"
            res = {"game_id": gid, "team": team, "cutoff": CT.cutoff_time(CT.parse_utc(meta["startTimeUTC"]), "T90").strftime("%Y-%m-%dT%H:%M:%SZ"), "candidates": uni["candidates"], "team_games_used": uni["team_games"],
                   "n_prior_team_games_available": uni["n_prior_team_games_available"], "features": feats}
            # ---- GRADING ONLY (candidate construction is complete) ----
            actual = [r for r in grading_rows if r["gameId"] == gid and r["team"] == team]
            cset = set(uni["candidates"])
            obs = [r for r in actual if r["playerId"] in cset]
            unobs = [r for r in actual if r["playerId"] not in cset]
            tot_sog = sum(r["sog"] or 0 for r in actual); obs_sog = sum(r["sog"] or 0 for r in obs)
            res["grading"] = {"n_candidates_at_T": len(cset), "n_actual_target_skaters": len(actual), "n_actual_skaters_observable_at_T": len(obs), "n_unobservable_at_T": len(unobs),
                              "actual_player_coverage": round(len(obs) / len(actual), 4) if actual else None, "total_actual_SOG": tot_sog, "observable_actual_SOG": obs_sog,
                              "SOG_coverage": round(obs_sog / tot_sog, 4) if tot_sog else None,
                              "n_candidates_who_did_not_play": len(cset - {r["playerId"] for r in actual}),
                              "unobservable": [{"playerId": r["playerId"], "sog": r["sog"], "diagnostic_grading_only": CT.diagnose_unobservable(r["playerId"], team, gid, rows, T)} for r in unobs],
                              "actual": {r["playerId"]: {"pp": r["pp"], "ev": r["ev"], "shifts": r["shifts"], "sog": r["sog"]} for r in actual}}
            out.append(res)
    return out


def summarize_deployment(tgs):
    xs, ys, ev_x, ev_y, sh_x, sh_y = [], [], [], [], [], []
    rec_p, rec_l, n_tg = [], [], 0
    hits = {"PP1": [0, 0], "PP2": [0, 0], "NONE": [0, 0]}
    chg = []
    cov, sogcov = [], []
    diag = Counter()
    n_cand = n_act = n_obs = n_unobs = 0
    tot_sog = obs_sog = 0
    for t in tgs:
        g = t["grading"]
        n_cand += g["n_candidates_at_T"]; n_act += g["n_actual_target_skaters"]; n_obs += g["n_actual_skaters_observable_at_T"]; n_unobs += g["n_unobservable_at_T"]
        tot_sog += g["total_actual_SOG"]; obs_sog += g["observable_actual_SOG"]
        if g["actual_player_coverage"] is not None:
            cov.append(g["actual_player_coverage"])
        if g["SOG_coverage"] is not None:
            sogcov.append(g["SOG_coverage"])
        for u in g["unobservable"]:
            diag[u["diagnostic_grading_only"]] += 1
        act = g["actual"]
        top5 = {p for p, v in sorted(act.items(), key=lambda kv: -(kv[1]["pp"] or 0))[:5] if (v["pp"] or 0) > 0 and p in t["features"]}
        if top5:
            n_tg += 1
            rank5 = {p for p, f in t["features"].items() if f["pp_rank"] <= 5}
            rec_p.append(len(top5 & rank5) / len(top5))
        for p, f in t["features"].items():
            a = act.get(p)
            if a is None:
                continue                                                          # candidate who did not play: no realized value to compare
            xs.append(f["pp_allocation_share5"]); ys.append(a["pp"] or 0); ev_x.append(f["ev_allocation_share5"]); ev_y.append(a["ev"] or 0); sh_x.append(f["prior_shifts_mean"]); sh_y.append(a["shifts"] or 0)
            hits[f["pp_proxy"]][1] += 1; hits[f["pp_proxy"]][0] += (a["pp"] or 0) >= 60
            if f["pp_share_change_recent_vs_long"] is not None:
                chg.append(abs(f["pp_share_change_recent_vs_long"]))
    sp = lambda a, b: round(spearman(a, b), 3) if a and spearman(a, b) is not None else None
    ratio = lambda h: [h[0], h[1], round(h[0] / h[1], 3) if h[1] else None]
    return {"team_games": len(tgs), "team_games_with_pp_time_among_candidates": n_tg, "coverage": {"candidates_at_T": n_cand, "actual_target_skaters": n_act, "observable_at_T": n_obs, "unobservable_at_T": n_unobs,
            "mean_player_coverage": round(sum(cov) / len(cov), 4) if cov else None, "total_actual_SOG": tot_sog, "observable_actual_SOG": obs_sog, "pooled_SOG_coverage": round(obs_sog / tot_sog, 4) if tot_sog else None,
            "mean_team_game_SOG_coverage": round(sum(sogcov) / len(sogcov), 4) if sogcov else None, "unobservable_diagnostics_grading_only": dict(diag)},
            "mean_recall_of_actual_top5_PP_by_candidate_rank_top5": round(sum(rec_p) / len(rec_p), 3) if rec_p else None, "spearman_prior5_pp_allocation_vs_target_pp_toi": sp(xs, ys),
            "spearman_prior5_ev_allocation_vs_target_ev_toi": sp(ev_x, ev_y), "spearman_prior5_shifts_vs_target_shifts": sp(sh_x, sh_y),
            "P(target_PP_TOI>=60s | proxy PP1)": ratio(hits["PP1"]), "P(.. | proxy PP2)": ratio(hits["PP2"]), "P(.. | proxy NONE)": ratio(hits["NONE"]),
            "median_abs_recent_vs_long_pp_share_change": round(sorted(chg)[len(chg) // 2], 4) if chg else None,
            "note": "allocation shares (not true PP opportunity shares); realized target values are used only to grade; candidates who did not play are excluded from the value comparisons"}


def debut_check(tgs, cache):
    """GRADING-ONLY: separate 'no loaded history' unobservables into true NHL debut vs earlier history outside the loaded window (one cached stats query per player, max 40)."""
    pend = [(u["playerId"], t["team"], t["game_id"]) for t in tgs for u in t["grading"]["unobservable"] if u["diagnostic_grading_only"] == "NO_LOADED_HISTORY"]
    res = {"n_no_loaded_history": len(pend), "checked": 0, "true_debut_no_earlier_regular_season_game_in_any_season": 0, "earlier_history_outside_loaded_window": 0}
    for pid, team, gid in sorted(set(pend))[:40]:
        date = None
        url = f"{ST}/skater/summary?" + urllib.parse.urlencode({"isGame": "true", "limit": "1", "cayenneExp": f'playerId={pid} and gameId<{gid} and gameTypeId=2'}, quote_via=urllib.parse.quote)
        d, _, _ = cached(url, cache)
        res["checked"] += 1
        res["earlier_history_outside_loaded_window" if d["total"] > 0 else "true_debut_no_earlier_regular_season_game_in_any_season"] += 1
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
    # D: for two target weeks load the 5 prior weekly windows + the target week; the TARGET TEAMS come from the schedule, never from target skater rows
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
        meta = {g: {"startTimeUTC": v["startTimeUTC"], "home": v["home"], "away": v["away"]} for g, v in games_all.items()}
        target = {g for g, v in games_all.items() if tlo <= v["date"] <= thi}
        rows_list = list(dtab.values())
        tgs = deployment_probe(rows_list, meta, target)
        res["D_deployment"]["targets"].append({"target_week": [tlo, thi], "prior_windows": wins, "acquisition": {**counter, "windows": len(wins) + 1}, "target_games": len(target), "rows_loaded": len(rows_list),
                                               "results": summarize_deployment(tgs), "debut_check_grading_only": debut_check(tgs, a.cache),
                                               "team_games": [{k: v for k, v in t.items() if k not in ("features",)} | {"grading": {k: v for k, v in t["grading"].items() if k != "actual"}} for t in tgs][:6], "n_team_games_listed": min(6, len(tgs))})
    res["D_deployment"]["cutoff_rule"] = "T = start - 90 min; source games must satisfy start + %d min <= T; target game id excluded" % CT.GAME_MAX_MINUTES
    res["D_deployment"]["definitions"] = {"candidate_universe": "nhl_outcome_contract.candidate_universe (same-team appearances in the last %d completed team games)" % CT.LOOKBACK_TEAM_GAMES, "pp_allocation_share": "player PP TOI / sum of team skater PP TOI in that prior game",
                                          "ev_allocation_share": "same for EV TOI", "prior5": "mean over the last <=5 CURRENT-TEAM appearances", "PP1 proxy": "candidate PP-allocation rank <= 5 and share > 0.05", "PP2 proxy": "rank 6-10 and share > 0.02",
                                          "recent_vs_long": "mean(last 2) - mean(last 5), needs >= 3 appearances"}
    res["D_deployment"]["linemates_from_shift_overlap"] = "not tested (optional; deferred)"
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(json.dumps({"C": res["C_shot_attempts"]["summary"], "D": [(t["target_week"], t["results"]["coverage"], t["results"]["mean_recall_of_actual_top5_PP_by_candidate_rank_top5"]) for t in res["D_deployment"]["targets"]], "seconds": res["seconds"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
