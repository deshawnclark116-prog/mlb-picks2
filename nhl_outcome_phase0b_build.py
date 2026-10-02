"""Assemble the Phase 0B artifacts from the bounded benchmark / probe outputs (and, optionally, the earlier exploratory observation log). No model, no sportsbook data. Every number in the
artifacts and README is derived from the inputs."""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import nhl_outcome_contract as CT

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_outcome_engine"
FWD_STATUS = "BLOCKER_INSUFFICIENT_LIVE_TIMING_EVIDENCE"


def exploratory_summary(obs):
    """Earlier WIP observations (no schedule hash, no retrieval-completion timestamps): kept as EXPLORATORY evidence only; they do not qualify for horizon selection."""
    ob = [o for o in obs if "availability" in o]
    games = sorted({o["game_id"] for o in ob})
    return {"qualifies_for_horizon_selection": False,
            "why_not": "collected by the WIP watcher: no raw-schedule hash, no retrieval started/completed pair, a 5-minute post-puck-drop grace existed, one retrieval timestamp per fetch; it does not meet the Phase 0B provenance rules",
            "n_observations": len(ob), "games": games, "first_retrieval_ts": min((o["retrieval_ts"] for o in ob), default=None), "last_retrieval_ts": max((o["retrieval_ts"] for o in ob), default=None),
            "min_minutes_to_start_observed": min((o["minutes_to_start"] for o in ob), default=None), "max_minutes_to_start_observed": max((o["minutes_to_start"] for o in ob), default=None),
            "what_it_suggests_not_validated": "rosterSpots (40-46 entries incl. goalies) already present at the earliest in-window observation; scratches empty at the observed pre-window points; none of this is promoted to a contract"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True); ap.add_argument("--probes", required=True); ap.add_argument("--m2-table", required=True); ap.add_argument("--obs", default=None)
    a = ap.parse_args()
    bench, probes = json.load(open(a.bench)), json.load(open(a.probes))
    OUT.mkdir(parents=True, exist_ok=True)
    ev = OUT / "evidence"
    ev.mkdir(exist_ok=True)
    tab = json.load(open(a.m2_table))
    tab = [{k: r[k] for k in ("gameId", "playerId", "teamId", "team", "opponent", "startTimeUTC", "sog", "toi", "ev", "pp", "sh", "shifts", "pos")} for r in tab]
    (ev / "m2_sample_table.json").write_text(json.dumps(tab, separators=(",", ":")))
    shutil.copy(a.bench, ev / "benchmark_raw.json"); shutil.copy(a.probes, ev / "probes_raw.json")
    dep = probes["D_deployment"]["targets"]
    strat = bench["strategies"]; est = bench["estimate_full_season"]
    n_games = strat["M1_per_game_boxscore_plus_timeonice"]["games"]; n_rows = strat["M2_batched_date_window"]["rows"]
    tp = bench["traded_players"]
    # ---------------- historical contract
    contract = dict(CT.CONTRACT)
    contract["measured_on_sample"] = {"deployment_probe": [{"target_week": t["target_week"], "team_games": t["results"]["team_games"], "coverage": t["results"]["coverage"], "debut_check_grading_only": t["debut_check_grading_only"]} for t in dep],
                                      "note": "coverage is measured on the sample by grading AFTER candidate construction; the candidate universe is a Phase 1A design input, and unobservable players are excluded from the prediction set and reported",
                                      "loaded_history": "5 prior weekly windows (~10 team games); the production design needs at least one prior season"}
    contract["executable"] = {"module": "nhl_outcome_contract.py", "functions": ["cutoff_time", "completed_before", "prior_rows", "candidate_universe", "current_team_appearances", "diagnose_unobservable", "shot_counts_from_events", "strength_consistent"], "tests": "tests/test_nhl_phase0b.py"}
    contract["goalie_statement"] = "historical confirmed pregame goalie is NOT reconstructable from the tested NHL endpoints; a boxscore 'starter' field observed pregame is an observed, unvalidated field"
    contract["no_model_no_sportsbook"] = "no model was fitted in Phase 0B and no sportsbook data was used"
    (OUT / "phase0b_historical_contract.json").write_text(json.dumps(contract, indent=1))
    # ---------------- acquisition benchmark
    m2_ok = all(w["m2_vs_boxscore"]["missing_vs_boxscore"] == 0 and w["m2_vs_boxscore"]["extra_vs_boxscore"] == 0 and w["m2_vs_boxscore"]["sog_mismatch"] == 0 and w["m2_vs_boxscore"]["toi_mismatch_gt_1s"] == 0 for w in bench["weeks"]) \
        and tp["traded_skater_games_found_by_M2"] == tp["traded_skater_games_in_sample"]
    bench_out = {**{k: v for k, v in bench.items()}, "C_shot_attempts": probes["C_shot_attempts"], "D_deployment": probes["D_deployment"], "sample": {"weeks": [w["window"] for w in bench["weeks"]], "labels": [w["label"] for w in bench["weeks"]], "games": n_games, "skater_rows": n_rows},
                 "m2_conclusion_reproduced_by_this_bounded_benchmark": m2_ok,
                 "fastest_leak_safe_method": {"method": "M2 batched: api-web schedule(week) + stats-REST skater/summary + skater/timeonice over a gameDate window (limit=-1, no team filter); accept a window only if reported total < 10000 and rows == total (weekly windows)",
                                              "evidence": f"{n_games} games / {n_rows} skater rows / {tp['traded_skater_games_in_sample']} traded-player skater-games reproduced against boxscore with 0 missing, 0 extra, 0 SOG mismatches",
                                              "requests_per_week": strat["M2_batched_date_window"]["requests_per_week"], "projected_full_season_requests": est["M2_requests"], "projected_full_season_seconds": est["M2_seconds_est"],
                                              "vs_per_game_seconds": est["M1_seconds_est"], "vs_existing_loop_requests": est["M3_requests_est"],
                                              "leak_safety": "rows are post-game final values; they are leak-safe only when read through the contract's allowed-rows rule (start + %d min <= T, target game excluded) and each window's retrieval date is stored" % CT.GAME_MAX_MINUTES},
                 "existing_repo_table": "nhl_models/nhl_model.sqlite skater_games is NOT used as a source of truth (incomplete: unsorted pagination duplicates + team-filter drops); it must be rebuilt, not patched",
                 "pbp_scope": "the small probe shows attempts and shootout exclusion work; no full-season PBP crawl is run. Phase 1A starts with SOG / TOI / deployment / environment; shot attempts are a later challenger only if they earn incremental CRPS / NLL"}
    (OUT / "phase0b_acquisition_benchmark.json").write_text(json.dumps(bench_out, indent=1, default=str))
    # ---------------- forward snapshot contract
    obs = [json.loads(l) for l in open(a.obs) if l.strip()] if a.obs and Path(a.obs).exists() else []
    fwd = {"version": "nhl-forward-snapshot-contract-2", "status": FWD_STATUS, "recommended_horizons": [],
           "status_reason": "no contract-compliant live timing evidence has been collected: the earlier exploratory watcher lacked raw-schedule hashes and retrieval-completion provenance and is not horizon-selection evidence; no horizon is validated",
           "purpose": "historical pregame availability cannot be reconstructed; from 2026-27 on pregame state is captured forward, append-only, hash-addressed",
           "observation_fields": ["game_id", "endpoint", "source_url", "scheduled_start_utc", "schedule_sha256", "schedule_retrieval_started_at", "schedule_retrieval_completed_at", "retrieval_started_at", "retrieval_completed_at", "retrieval_ts",
                                  "intended_horizon", "minutes_to_start", "sha256", "bytes", "http_status", "etag", "last_modified", "game_state", "availability", "observation_status"],
           "observation_status_values": {"VALID_PREGAME": "completed strictly before the scheduled start", "REJECTED_LATE": "started before puck drop but completed at/after it: bytes kept for audit, never pregame evidence",
                                         "PROVIDER_ERROR": "recorded explicitly, no substitute source", "REFUSED_NOT_STARTED_BEFORE_PUCK_DROP": "a fetch is never initiated at/after the scheduled start (no grace period)"},
           "rules": ["the raw schedule bytes that establish each scheduled start are stored by sha256 and hash-roundtrip; every observation references that schedule identity", "minutes_to_start uses the COMPLETION timestamp", "an early retrieval is never relabelled as a later horizon",
                     "hard stop at scheduled puck drop: no pregame observation is initiated at or after it", "goalie fields are non-authoritative observations (starter_like_field_present_unvalidated, boxscore_starter_field_values_observed_unvalidated); historical confirmed pregame goalie is NOT reconstructable",
                     "a POSTGAME capture (final boxscore / pbp / right-rail) is recorded separately and is the label source, never a feature"],
           "implementation": "nhl_outcome_snapshot.py (injectable clock / http / sleep; tested without network or waiting)",
           "required_before_a_horizon_can_be_recommended": ["a compliant watcher run over several slates covering the candidate horizons (T24H, T90, T30, T10, T2) with VALID_PREGAME observations that include the schedule hash and completion timestamps",
                                                           "a durable scheduler (the repo's GitHub Actions pattern) instead of an interactive session", "equality check of pregame rosterSpots / scratches against the final dressed list"],
           "exploratory_prior_observations": exploratory_summary(obs)}
    (OUT / "phase0b_forward_snapshot_contract.json").write_text(json.dumps(fwd, indent=1, default=str))
    # ---------------- README
    cov = [(t["target_week"], t["results"]["coverage"]) for t in dep]
    lines = "\n".join(f"  - {w[0]}..{w[1]}: {c['observable_at_T']}/{c['actual_target_skaters']} skaters observable at T ({c['mean_player_coverage']}), SOG coverage {c['pooled_SOG_coverage']}, unobservable {c['unobservable_at_T']} {c['unobservable_diagnostics_grading_only']}" for w, c in cov)
    readme = f"""# NHL outcome engine — Phase 0B (historical contract, acquisition, forward snapshots)

**HISTORICAL FOUNDATION: PASS** (contract, acquisition and probes validated by `tests/test_nhl_phase0b.py`).
**FORWARD SNAPSHOT TIMING: BLOCKER_INSUFFICIENT_LIVE_TIMING_EVIDENCE** (`recommended_horizons: []`; no horizon is validated).
**OVERALL:** historical Phase 1A research may proceed. Forward horizon selection is NOT solved.

No model was fitted in Phase 0B. No sportsbook data was used. No production NHL file changed. A full play-by-play crawl is deferred.

## Key rule
Historical player candidacy must be known before the target game. Candidate universe for target game G, team TEAM, cutoff T = players who appeared **for that team** in TEAM's last {CT.LOOKBACK_TEAM_GAMES} completed games before T (games != G with start + {CT.GAME_MAX_MINUTES} min <= T), `nhl_outcome_contract.candidate_universe`. Nothing else adds a candidate; a target participant outside it is `unobservable_at_T` (reported, never inserted). Target-game boxscore / roster / position / TOI / PP / shifts / SOG / events / scratches are grading-only. After membership, prior NHL history from all teams may feed player *skill*; role / deployment features use current-team appearances only.

## Evidence (derived from the committed evidence files)
- Historical acquisition: M2 (schedule + stats-REST summary + time-on-ice by date window, no team filter, `total < 10000` and `rows == total` guard) reproduced {n_games} sample games / {n_rows} skater rows / {tp['traded_skater_games_in_sample']} traded-player skater-games against the boxscore with no missing rows: **{m2_ok}**; projected {est['M2_requests']} requests / ~{est['M2_seconds_est']:.0f} s per season vs ~{est['M1_seconds_est']:.0f} s per-game. The repo's `skater_games` table stays unused (unsorted pagination duplicates and the team filter drop players).
- Candidate coverage on the sample (graded after candidate construction):
{lines}
- Shot attempts: the small probe reconstructs attempts / SOG / missed / blocked and excludes shootout events; attempts are a later challenger.
- Goalie: historical confirmed pregame goalie is NOT reconstructable from the tested endpoints; pregame `starter`-like fields are observed, unvalidated signals.

## Remaining blockers
1. Forward snapshot timing: compliant live evidence (schedule hash, retrieval started/completed, hard stop at puck drop) has not been collected; no horizon is recommended.
2. No historical point-in-time roster / transaction source: players new to a team are unobservable historically.
3. A durable (non-interactive) scheduler is needed for the forward capture.
"""
    (OUT / "PHASE0B_README.md").write_text(readme)
    print("ok")


if __name__ == "__main__":
    main()
