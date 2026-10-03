"""
NFL_PHASE1C_FREEZE_CANDIDATE  (Phase 1C, shadow research)

Builds nfl_models/nfl_player_outcome_phase1_freeze_candidate.json from the Phase 1C artifacts. This is NOT the freeze record:
it is the package a reviewer audits. The final "nfl_player_outcome_phase1_freeze.json" is never created by this code.

  python -u nfl_phase1c_freeze_candidate.py --scratch DIR --adj FILES...
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import nfl_phase1_forecast as FC
import nfl_phase1c_evaluate as EV

REPO = Path(__file__).resolve().parent
P1C = REPO / "nfl_models" / "nfl_player_outcome_phase1c"
NEVER_CREATED = REPO / "nfl_models" / "nfl_player_outcome_phase1_freeze.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def arr_hash(o):
    h = hashlib.sha256()
    def walk(x):
        if isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode()); walk(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        else:
            h.update(np.ascontiguousarray(np.asarray(x, float)).tobytes())
    walk(o)
    return h.hexdigest()


def build(scratch, adj_files, fit_end=202603):
    import nfl_phase1c_dryrun as DR
    load = lambda n: json.load(open(P1C / n)) if (P1C / n).exists() else None
    conv = load("simulation_convergence.json"); cal = load("calibration_depth.json"); uni = load("universe_comparison.json"); st = load("state_static_vs_dynamic.json")
    ctxless = DR.make_bundle(scratch, adj_files, (conv or {}).get("chosen_n") or 0, calibration=(cal or {}).get("final_maps") or {}, fit_end=fit_end, records_file="records_depth.pkl")
    eff = ctxless.eff
    proto = REPO / "nfl_models" / "nfl_player_outcome_phase1_protocol.json"
    files = ["nfl_phase1_forecast.py", "nfl_phase1_score.py", "nfl_phase1_store.py", "nfl_phase1_snapshots.py", "nfl_phase1c_sim.py", "nfl_phase1c_fit.py", "nfl_phase1c_script.py",
             "nfl_phase1c_metrics.py", "nfl_phase1c_constants.py", "nfl_phase1c_inputs.py", "nfl_phase1c_adjudicate.py", "nfl_phase1c_evaluate.py", "nfl_phase1c_dryrun.py",
             "nfl_phase1_efficiency.py", "nfl_phase1_event_models.py", "nfl_phase1_rushing_efficiency.py", "nfl_phase1_receiving_efficiency.py", "nfl_phase1_passing_efficiency.py",
             "nfl_phase1_defense_events.py", "nfl_phase1b_data.py", "nfl_phase1_data.py", "nfl_phase1_common.py", "nfl_phase1_availability.py", "nfl_phase1_team_environment.py",
             "nfl_phase1_role_state.py", "nfl_phase1_opportunity.py", "nfl_context_v4.py"]
    frozen_files = {f: sha(REPO / f) for f in files}
    for extra in ("nfl_models/nfl_player_outcome_phase1_protocol.json", "nfl_models/nfl_player_outcome_phase1c/constants.json", "nfl_models/nfl_player_outcome_phase1c/selected_components.json",
                  "nfl_models/nfl_player_outcome_phase1c/adjudication_rule.json", "nfl_models/nfl_player_outcome_phase1c/adjudication_results.json"):
        frozen_files[extra] = sha(REPO / extra)
    proto_j = json.load(open(proto))
    chosen_n = (conv or {}).get("chosen_n")
    man = {
        "STATUS": "FREEZE CANDIDATE FOR AUDIT. NOT A FREEZE. nfl_player_outcome_phase1_freeze.json does not exist and must not be created before the audit.",
        "code_commit_sha": FC.git_sha(),
        "commit_note": "the SHA is the checkout at build time; the final freeze must be built from a clean commit that contains this manifest's inputs",
        "protocol_version": FC.PROTOCOL_VERSION, "protocol_file": "nfl_models/nfl_player_outcome_phase1_protocol.json", "protocol_amendments": [a["id"] for a in proto_j.get("amendments_v1_1", [])],
        "model_version": ctxless.model_version, "sim_version": FC.SIM_VERSION,
        "selected_component_architecture": ctxless.config,
        "temporal_adjudication": {"rule": "nfl_models/nfl_player_outcome_phase1c/adjudication_rule.json", "results": "adjudication_results.json", "table": "adjudication_table.md"},
        "model_hyper_parameters_priors_at_fit_end": {"fit_end": fit_end, "fit_window": "2022 wk2 through 2026 wk3 (all burned data); inner validation = last 25% of weeks", "hyper": eff["hyper"]},
        "fitted_arrays_sha256": {k: arr_hash(v) for k, v in eff.items() if k not in ("hyper",)},
        "feature_lists": {"team": "home flag + as-of team-offense rates (rush: >=15yd rate, negative rate, yards/carry; pass: completion, air/completion, YAC/completion, sack, INT, TD rates)",
                          "opp": "same rates for the opponent's defence allowed", "scheme": {"rush": ["def_stacked_rate", "def_avg_box", "def_blitz_rate", "def_rush_epa"],
                                                                                            "pass": ["def_man_rate", "def_c0", "def_c1", "def_c2", "def_c4", "def_c6", "def_blitz_rate", "def_pressure_rate"]},
                          "personnel": "injury-report OL/DL/LB/DB Out/Doubtful (+0.5 Questionable) counts (historical final-weekly-report proxy; forward: immutable snapshot)",
                          "used_by_adjudicated_components": {k: v["level"] for k, v in ctxless.config.items()}},
        "data_source_rules": {"as_of": "week-level for all stats histories; injuries/rosters/depth from immutable snapshots retrieved <= cutoff (nfl_phase1_snapshots); assumptions A1-A7 in nfl_phase1_data.AS_OF_ASSUMPTIONS",
                              "forbidden": ["sportsbook lines/odds as features", "same-game participation / snaps / box / personnel", "target-game outcomes", "retrievals after the forecast cutoff"],
                              "sportsbook_use": "only inside the v2 comparator (nfl_phase1c_v2comp), never in the engine"},
        "calibration_rules": {"evidence_file": "calibration_depth.json", "method": "target-specific; candidate PIT map / dispersion scaling / conformal margin; adopted only if |cov80-0.8| and |cov50-0.5| both improve out-of-time and CRPS worsens <= 0.3%",
                              "final_maps": (cal or {}).get("final_maps"), "evidence": "calibration.json"},
        "simulation": {"n_draws": chosen_n, "n_draws_rule": "smallest tested N meeting the pre-committed Monte Carlo criteria (simulation_convergence.json)", "rng": "numpy PCG64 default_rng seeded by crc32(repr((model_version, game_id, horizon)))",
                       "rng_version": f"numpy {np.__version__}", "constants": "nfl_models/nfl_player_outcome_phase1c/constants.json", "game_script": {"adopted": "S0 static pregame volumes (Phase 1A team-environment negative binomials)",
                                                                                      "S1_dynamic_result": "not adopted: S1 CRPS worse than S0 on every scored outcome (state_static_vs_dynamic.json)"}},
        "forecast_universe": "Phase 1A pregame universe: RB/QB carry candidates, RB/WR/TE target candidates, QB attempt candidates, DL/LB/DB defenders (non-participants score 0); "
                             "ADOPTED: depth-chart-extended candidate universe (decision block in universe_comparison.json); calibration maps and evidence from *_depth.json files",
        "horizons": {"T24": "cutoff = kickoff - 24h; required snapshots: injuries (+ depth charts for 2025+); P(active) from Phase 1A T24 model",
                     "T90": "cutoff = kickoff - 90m; required snapshots: injuries + weekly_rosters (+ depth); game-day INA collapses P(active) to 0; independent download from T24"},
        "scoring_rules": {"logger": "nfl_phase1_score.py", "official_source": "nflverse stats_player_week (sha256 recorded); revisions append new revision records", "primary_outcomes": sorted(FC.OUTCOMES),
                          "primary_metrics": ["CRPS (quantile-grid proxy)", "MAE of median", "bias of mean", "interval coverage 50/80/90", "event logloss/Brier/ECE"]},
        "final_gates": {"component_gates": proto_j.get("component_gates"), "end_to_end_gates": proto_j.get("end_to_end_gates"), "clean_forward_window": proto_j.get("clean_forward_window"),
                        "phase1c_additional": "calibration coverage (PIT) within 0.03 of nominal on the frozen window; convergence N fixed; idempotency/chaos suites green; no HARD ERRORs in the logs"},
        "files_and_sha256_that_would_be_frozen": frozen_files,
        "not_created": str(NEVER_CREATED.relative_to(REPO)), "final_freeze_file_exists": NEVER_CREATED.exists(),
    }
    (REPO / "nfl_models" / "nfl_player_outcome_phase1_freeze_candidate.json").write_text(json.dumps(man, indent=1, default=float))
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--adj", nargs="+", required=True)
    a = ap.parse_args()
    m = build(a.scratch, a.adj)
    print(m["model_version"], m["simulation"]["n_draws"], m["final_freeze_file_exists"])
