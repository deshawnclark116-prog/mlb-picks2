"""
NHL_FWD_EVAL (Phase 0C) -- horizon evidence metrics from forward captures and postgame truth, plus the PRE-REGISTERED recommendation rule. No model fitting.

A horizon is evaluated/recommended only with >= MIN_GAMES completed games with postgame truth on >= MIN_DATES distinct game dates and >= MIN_VALID_PER_HORIZON COMPLETE_VALID observations at that horizon.
Qualification (play-by-play rosterSpots as the candidate-universe roster source): capture_success_rate >= 0.99 over ELIGIBLE OPPORTUNITIES (protocol-2 pre-live amendment: planned non-revised key, postgame truth eventually exists, planned_at <= cutoff - WAKE_LEAD_SECONDS; every non-COMPLETE_VALID status is a failure; late enrollment is reported and excluded), final dressed SKATER recall >= 0.995, final SOG coverage >= 0.995.
The earliest (farthest from puck drop) qualifying horizon is recommended; otherwise BLOCKER_INSUFFICIENT_LIVE_TIMING_EVIDENCE stays. Goalie starter confirmation is not part of this pass.
"""
import json
from collections import defaultdict
from datetime import timedelta

import nhl_fwd_capture as C

MIN_GAMES = 50
MIN_DATES = 3
MIN_VALID_PER_HORIZON = 40
MIN_SUCCESS_RATE = 0.99
MIN_SKATER_RECALL = 0.995
MIN_SOG_COVERAGE = 0.995
BLOCKER = "BLOCKER_INSUFFICIENT_LIVE_TIMING_EVIDENCE"


def latest_by(rows, key):
    out = {}
    for r in rows:
        out[r[key]] = r
    return out


def roster_metrics(pbp, truth_sk_by_team, truth_gl_by_team, sog_by_skater):
    spots = pbp.get("rosterSpots", [])
    r_sk = {s["playerId"] for s in spots if s.get("positionCode") != "G"}
    r_gl = {s["playerId"] for s in spots if s.get("positionCode") == "G"}
    final_sk = {p for ids in truth_sk_by_team.values() for p in ids}
    final_gl = {p for ids in truth_gl_by_team.values() for p in ids}
    inter = r_sk & final_sk
    tot_sog = sum(sog_by_skater.values())
    return {"final_skaters": len(final_sk), "roster_skaters": len(r_sk), "hit": len(inter), "extra": len(r_sk - final_sk), "union": len(r_sk | final_sk),
            "sog_in_roster": sum(v for k, v in sog_by_skater.items() if int(k) in r_sk), "sog_total": tot_sog, "goalies_final": len(final_gl), "goalies_hit": len(r_gl & final_gl), "roster_empty": len(spots) == 0}


def box_metrics(box, final_sk):
    pb = box.get("playerByGameStats") or {}
    ids = {p["playerId"] for side in ("awayTeam", "homeTeam") for pos in ("forwards", "defense") for p in pb.get(side, {}).get(pos, [])}
    gl = [g for side in ("awayTeam", "homeTeam") for g in pb.get(side, {}).get("goalies", [])]
    return {"skater_set_available": len(ids) > 0, "n_skaters": len(ids), "hit": len(ids & final_sk), "final_skaters": len(final_sk), "goalie_field_available": len(gl) > 0, "starter_field_values_observed_unvalidated": [g.get("starter") for g in gl]}


def rail_metrics(rail, final_sk):
    """Availability = the scratches field is PRESENT and a list for both teams (an empty list is available with zero scratches; an absent / non-list field is unavailable)."""
    gi = (rail.get("gameInfo") or {})
    lists = [(gi.get(side) or {}).get("scratches") for side in ("awayTeam", "homeTeam")]
    avail = all(isinstance(x, list) for x in lists)
    ids = [s["id"] for x in lists if isinstance(x, list) for s in x if isinstance(s, dict) and "id" in s] if avail else []
    return {"scratch_list_available": avail, "n_scratches": len(ids), "scratches_who_actually_played": sum(1 for i in ids if i in final_sk)}


def evaluate(st, gate_override=None):
    planned = st.planned.read()
    results = latest_by(st.results.read(), "key")
    obs = defaultdict(list)
    for o in st.obs.read():
        obs[o["key"]].append(o)
    truth = latest_by(st.postgame.read(), "game_id")
    games_truth = {g for g in truth}
    dates = {truth[g]["game_date"] for g in truth}
    min_games = (gate_override or {}).get("min_games", MIN_GAMES)
    sample_ok = len(games_truth) >= min_games and len(dates) >= (gate_override or {}).get("min_dates", MIN_DATES)
    out = {"protocol_version": C.PROTOCOL_VERSION, "games_with_postgame_truth": len(games_truth), "distinct_game_dates_with_truth": len(dates), "min_sample": {"games": min_games, "dates": MIN_DATES, "valid_per_horizon": MIN_VALID_PER_HORIZON}, "denominator": "eligible opportunities (see module docstring)", "horizons": {}}
    for hz in C.HORIZON_ORDER:
        keys = [p for p in planned if p["horizon"] == hz]
        res = [results[p["key"]] for p in keys if p["key"] in results]
        attempted = [r for r in res if r["status"] != C.REVISED and any(o for o in obs.get(r["key"], []))]
        non_revised = [p for p in keys if results.get(p["key"], {}).get("status") != C.REVISED]
        elig, late = [], []
        for p in non_revised:                                                  # eligible opportunity: non-revised key + postgame truth eventually exists + planned_at <= cutoff - WAKE_LEAD_SECONDS
            if p["game_id"] not in games_truth:
                continue
            (elig if C.parse_iso(p["planned_at"]) <= C.parse_iso(p["cutoff"]) - timedelta(seconds=C.WAKE_LEAD_SECONDS) else late).append(p)
        elig_valid = [results[p["key"]] for p in elig if results.get(p["key"], {}).get("status") == C.COMPLETE_VALID]
        valid = elig_valid                                                     # truth metrics and the qualification sample use eligible COMPLETE_VALID keys only
        all_valid = [r for r in res if r["status"] == C.COMPLETE_VALID]
        h = {"minutes_before_puck_drop": C.HORIZONS[hz], "games_planned": len(keys), "games_attempted": len(attempted), "complete_valid": len(valid), "missed_cutoff": sum(1 for r in res if r["status"] == C.MISSED),
             "early_not_horizon_evidence": sum(1 for r in res if r["status"] == C.EARLY), "provider_failures": sum(1 for r in res if r["status"] == C.PROVIDER_FAILURE), "refused": sum(1 for r in res if r["status"] == C.REFUSED),
             "schedule_revised": sum(1 for r in res if r["status"] == C.REVISED),
             "eligible_opportunities": len(elig), "eligible_complete_valid": len(elig_valid), "capture_success_rate": (len(elig_valid) / len(elig)) if elig else None,
             "eligible_failure_breakdown": {st_: sum(1 for p in elig if results.get(p["key"], {}).get("status", "NO_RESULT") == st_) for st_ in sorted({results.get(p["key"], {}).get("status", "NO_RESULT") for p in elig} - {C.COMPLETE_VALID})},
             "late_enrollment_not_eligible": len(late), "late_enrollment_status": C.LATE_ENROLLMENT, "complete_valid_all_keys_diagnostic": len(all_valid),
             "attempted_only_success_rate_diagnostic": (len(all_valid) / len(attempted)) if attempted else None}
        eps = defaultdict(lambda: [0, 0])
        for r in attempted:
            for o in obs[r["key"]]:
                eps[o["endpoint"]][0] += 1; eps[o["endpoint"]][1] += 1 if o.get("raw_sha256") else 0
        h["endpoint_availability_rate"] = {e: (v[1] / v[0] if v[0] else None) for e, v in eps.items()}
        # ---- truth-based metrics on COMPLETE_VALID observations of games with postgame truth
        rm = {"final_skaters": 0, "hit": 0, "extra": 0, "union": 0, "roster_skaters": 0, "sog_in_roster": 0, "sog_total": 0, "goalies_final": 0, "goalies_hit": 0, "empty": 0, "n": 0}
        bx = {"n": 0, "available": 0, "hit": 0, "final": 0, "goalie_field": 0}
        rl = {"n": 0, "available": 0, "n_scratches": 0, "played": 0}
        for r in valid:
            t = truth.get(r["game_id"])
            if not t:
                continue
            o = {x["endpoint"]: x for x in obs[r["key"]]}
            final_sk = {p for ids in t["final_dressed_skater_ids_by_team"].values() for p in ids}
            m = roster_metrics(json.loads(st.blobs.get(o["play-by-play"]["raw_sha256"])), t["final_dressed_skater_ids_by_team"], t["final_goalie_ids_by_team"], t["final_sog_by_skater"])
            rm["n"] += 1; rm["empty"] += int(m["roster_empty"])
            for k_, f in (("final_skaters", "final_skaters"), ("hit", "hit"), ("extra", "extra"), ("union", "union"), ("roster_skaters", "roster_skaters"), ("sog_in_roster", "sog_in_roster"), ("sog_total", "sog_total"), ("goalies_final", "goalies_final"), ("goalies_hit", "goalies_hit")):
                rm[k_] += m[f]
            b = box_metrics(json.loads(st.blobs.get(o["boxscore"]["raw_sha256"])), final_sk)
            bx["n"] += 1; bx["available"] += int(b["skater_set_available"]); bx["goalie_field"] += int(b["goalie_field_available"])
            if b["skater_set_available"]:
                bx["hit"] += b["hit"]; bx["final"] += b["final_skaters"]
            q = rail_metrics(json.loads(st.blobs.get(o["right-rail"]["raw_sha256"])), final_sk)
            rl["n"] += 1; rl["available"] += int(q["scratch_list_available"]); rl["n_scratches"] += q["n_scratches"]; rl["played"] += q["scratches_who_actually_played"]
        d = lambda a, b: (a / b) if b else None
        h["with_truth"] = {"complete_valid_games_with_truth": rm["n"],
                           "pbp_rosterSpots": {"final_dressed_skater_recall": d(rm["hit"], rm["final_skaters"]), "extra_players_total": rm["extra"], "precision": d(rm["hit"], rm["roster_skaters"]), "jaccard": d(rm["hit"], rm["union"]),
                                               "final_sog_coverage": d(rm["sog_in_roster"], rm["sog_total"]), "goalie_id_recall": d(rm["goalies_hit"], rm["goalies_final"]), "games_with_empty_rosterSpots": rm["empty"]},
                           "right_rail": {"scratch_list_availability": d(rl["available"], rl["n"]), "mean_scratches_when_available": d(rl["n_scratches"], rl["available"]), "listed_scratches_who_played": rl["played"],
                                          "scratches_not_in_final_dressed_fraction": (1 - d(rl["played"], rl["n_scratches"])) if rl["n_scratches"] else None},
                           "boxscore": {"pregame_skater_set_availability": d(bx["available"], bx["n"]), "final_dressed_skater_recall_when_present": d(bx["hit"], bx["final"]), "goalie_field_availability": d(bx["goalie_field"], bx["n"]),
                                        "starter_fields": "UNVALIDATED; no confirmed starter is inferred"}}
        h["_valid_with_truth"] = rm["n"]
        out["horizons"][hz] = h
    # ---- registered recommendation rule
    if not sample_ok:
        out["evaluation"] = "NOT_EVALUATED_INSUFFICIENT_SAMPLE"; out["recommended_horizon"] = None; out["status"] = BLOCKER
        out["sample_gate"] = {"games_with_truth": len(games_truth), "required_games": min_games, "dates": len(dates), "required_dates": MIN_DATES}
    else:
        q = []
        for hz in C.HORIZON_ORDER:
            h = out["horizons"][hz]; w = h["with_truth"]["pbp_rosterSpots"]
            if h["_valid_with_truth"] >= MIN_VALID_PER_HORIZON and (h["capture_success_rate"] or 0) >= MIN_SUCCESS_RATE and (w["final_dressed_skater_recall"] or 0) >= MIN_SKATER_RECALL and (w["final_sog_coverage"] or 0) >= MIN_SOG_COVERAGE:
                q.append(hz)
        out["evaluation"] = "EVALUATED"; out["qualifying_horizons"] = q
        out["recommended_horizon"] = q[0] if q else None                       # HORIZON_ORDER runs from farthest to nearest puck drop: earliest qualifying
        out["status"] = "HORIZON_RECOMMENDED" if q else BLOCKER
    for h in out["horizons"].values():
        h.pop("_valid_with_truth", None)
    return out
