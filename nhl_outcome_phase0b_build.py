"""Assemble the Phase 0B artifacts from the benchmark / probe outputs and the forward-snapshot observations. No model."""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import nhl_outcome_contract as CT

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_outcome_engine"


def observation_timeline(obs):
    """Per game: for every field the observation list shows, the largest minutes_to_start at which it was PRESENT and the smallest at which it was still ABSENT (bracket of first appearance)."""
    by = defaultdict(lambda: defaultdict(list))
    for r in obs:
        if "availability" not in r:
            continue
        a, ep, m = r["availability"], r["endpoint"], r["minutes_to_start"]
        by[r["game_id"]]["_n"].append(m)
        if ep == "right-rail":
            by[r["game_id"]]["scratches_present"].append((m, a["scratches_away"] + a["scratches_home"] > 0))
            by[r["game_id"]]["scratch_counts"].append((m, a["scratches_away"], a["scratches_home"]))
            by[r["game_id"]]["officials_present"].append((m, a["n_referees"] > 0))
        elif ep == "play-by-play":
            by[r["game_id"]]["rosterSpots_present"].append((m, a["n_rosterSpots"] > 0))
            by[r["game_id"]]["rosterSpots_counts"].append((m, a["n_rosterSpots"], a["n_rosterSpot_goalies"]))
            by[r["game_id"]]["plays_present"].append((m, a["n_plays"] > 0))
        elif ep == "boxscore":
            by[r["game_id"]]["boxscore_skaters_present"].append((m, a["n_skaters"] > 0))
            by[r["game_id"]]["boxscore_goalies_present"].append((m, a["n_goalies"] > 0))
            by[r["game_id"]]["starter_flag_present"].append((m, any(x is not None for x in a["starter_flags"])))
        elif ep == "landing":
            by[r["game_id"]]["matchup_present"].append((m, a["has_matchup"]))
    out = {}
    for g, fields in by.items():
        e = {"observations": len(fields["_n"]), "earliest_observed_min_to_start": max(fields["_n"]), "latest_observed_min_to_start": min(fields["_n"])}
        for f, lst in fields.items():
            if f in ("_n", "scratch_counts", "rosterSpots_counts"):
                continue
            lst = sorted(lst, key=lambda x: -x[0])
            present = [m for m, v in lst if v]; absent = [m for m, v in lst if not v]
            first_present = max(present) if present else None
            e[f] = {"first_seen_at_min_to_start": first_present, "last_absent_at_min_to_start": min([m for m in absent if first_present is None or m > first_present], default=None) if absent else None, "present_in_earliest_observation": bool(lst and lst[0][1])}
        e["rosterSpots_count_timeline"] = [(m, n, gk) for m, n, gk in sorted(fields["rosterSpots_counts"], key=lambda x: -x[0])]
        e["scratch_count_timeline"] = [(m, a, h) for m, a, h in sorted(fields["scratch_counts"], key=lambda x: -x[0])]
        out[g] = e
    return out


def compress_timeline(tl):
    """Keep only the points where the value changes."""
    out, last = [], None
    for t in tl:
        key = tuple(t[1:])
        if key != last:
            out.append(list(t)); last = key
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True); ap.add_argument("--probes", required=True); ap.add_argument("--obs", required=True); ap.add_argument("--m2-table", required=True)
    ap.add_argument("--recommend", required=True, help="json file with recommended_horizons + notes written after reading the observations")
    a = ap.parse_args()
    bench, probes = json.load(open(a.bench)), json.load(open(a.probes))
    obs = [json.loads(l) for l in open(a.obs) if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence").mkdir(exist_ok=True)
    tab = json.load(open(a.m2_table))
    tab = [{"gameId": r["gameId"], "playerId": r["playerId"], "teamId": r["teamId"], "team": r["team"], "opponent": r["opponent"], "startTimeUTC": r["startTimeUTC"], "sog": r["sog"], "toi": r["toi"], "ev": r["ev"], "pp": r["pp"], "sh": r["sh"], "shifts": r["shifts"], "pos": r["pos"]} for r in tab]
    (OUT / "evidence" / "m2_sample_table.json").write_text(json.dumps(tab, separators=(",", ":")))
    # ---- historical contract
    contract = dict(CT.CONTRACT)
    dep = probes["D_deployment"]["targets"]
    contract["measured_on_sample"] = {"unobservable_at_T_rate": [{"target_week": t["target_week"], "dressed_skaters": sum(t["results"]["players"].get(k, 0) for k in ("state_KNOWN", "state_ACQUIRED", "state_COLD")), "state_counts": {k: v for k, v in t["results"]["players"].items() if k.startswith("state_")},
                                                                  "cold_start_split": {"nhl_debut": len(t["cold_start"]["debut"]), "had_earlier_games_outside_loaded_window": len(t["cold_start"]["has_earlier_games"])}} for t in dep],
                                     "note": "COLD here is relative to a 5-week loaded window; loading the prior season turns most of them KNOWN/ACQUIRED. Loaded-history depth is a Phase 1 parameter (>= 1 prior season required).",
                                     "leak_checks": [t["leak_checks_prior_rows_with_start_ge_target_start"] for t in dep]}
    contract["executable"] = {"module": "nhl_outcome_contract.py", "functions": ["cutoff_time", "completed_before", "prior_rows", "classify_player", "shot_counts_from_events", "strength_consistent"], "tests": "tests/test_nhl_phase0b.py"}
    (OUT / "phase0b_historical_contract.json").write_text(json.dumps(contract, indent=1))
    # ---- acquisition benchmark (+ C and D evidence)
    est = bench["estimate_full_season"]
    bench_out = {**bench, "C_shot_attempts": probes["C_shot_attempts"], "D_deployment": probes["D_deployment"], "sample": {"weeks": [w["window"] for w in bench["weeks"]], "labels": [w["label"] for w in bench["weeks"]]},
                 "fastest_leak_safe_method": {"method": "M2 batched: api-web schedule(week) + stats-REST skater/summary + skater/timeonice over a gameDate window (limit=-1, no team filter), guarded: accept a window only if reported total < 10000 and rows == total; weekly windows",
                                              "why": "complete vs boxscore on 151 games / 5,435 skater-games incl. 867 traded-player games; 3 requests per week; ~21 s per season vs ~834 s per-game and ~101 s (incomplete) for the existing loop",
                                              "leak_safety": "rows are post-game final values: they are leak-safe ONLY when read through the contract's completed-before-T rule; a retrieval date is stored with each window so as-of reconstruction is possible from our own snapshots going forward",
                                              "projected_full_season_runtime_seconds": est["M2_seconds_est"], "projected_requests": est["M2_requests"],
                                              "projected_full_season_with_pbp_for_attempts": "play-by-play is per game: 1,312 requests/season (~110 KB each); at the measured ~0.3 s/request ~7 min sequential per season, pbp only if attempts are required (see C)"},
                 "defect_status": "SOLVED for acquisition: gameDate-window queries without the team filter are complete; the repo's skater_games table must be rebuilt (not patched)"}
    (OUT / "phase0b_acquisition_benchmark.json").write_text(json.dumps(bench_out, indent=1, default=str))
    # ---- forward snapshot contract
    tl = observation_timeline(obs)
    for g in tl.values():
        g["rosterSpots_count_timeline"] = compress_timeline(g["rosterSpots_count_timeline"]); g["scratch_count_timeline"] = compress_timeline(g["scratch_count_timeline"])
    rec = json.load(open(a.recommend))
    fwd = {"version": "nhl-forward-snapshot-contract-1", "purpose": "historical pregame availability cannot be reconstructed; from 2026-27 on every pregame state is captured forward, append-only, hash-addressed",
           "snapshot_record_fields": ["game_id", "endpoint", "source_url", "retrieval_ts", "intended_horizon", "minutes_to_start", "game_start", "sha256", "bytes", "http_status", "etag", "game_state", "availability"],
           "availability_state": {"landing": ["has_matchup", "has_goalie_comparison", "goalie_starter_field"], "right-rail": ["scratches_away", "scratches_home", "n_referees"], "boxscore": ["n_skaters", "n_goalies", "starter_flags"], "play-by-play": ["n_plays", "n_rosterSpots", "n_rosterSpot_goalies"]},
           "rules": ["retrieval_ts is taken after the last byte arrived; a snapshot is valid for horizon H only if retrieval_ts <= game_start - H", "an early retrieval is never relabelled as a later horizon; intended_horizon is the nearest candidate and minutes_to_start is the exact value",
                     "raw bytes are stored by sha256 and never overwritten; observations are append-only", "no pregame observation is recorded after the scheduled start", "a capture that cannot reach the provider records the failure, never a substitute",
                     "a post-game capture (final boxscore, pbp, right-rail) is recorded separately and labelled POSTGAME; it is the target source, never a feature"],
           "observation_run": {"watcher": "nhl_outcome_snapshot.py watch", "n_observations": len(obs), "games": sorted({o["game_id"] for o in obs}), "first_retrieval_ts": min(o["retrieval_ts"] for o in obs), "last_retrieval_ts": max(o["retrieval_ts"] for o in obs),
                               "cadence": "every 120 s for games within 150 min of the start; every 30 min for one game ~24 h out"},
           "observed_field_timing": tl, "recommended_horizons": rec["recommended_horizons"], "interpretation": rec["interpretation"], "durable_scheduler": rec["durable_scheduler"], "open_items": rec["open_items"]}
    (OUT / "phase0b_forward_snapshot_contract.json").write_text(json.dumps(fwd, indent=1, default=str))
    print("ok")


if __name__ == "__main__":
    main()
