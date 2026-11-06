#!/usr/bin/env python3
"""Writes phase1a_v1_migration_audit.json: component-by-component classification of the burned V1 SOG corpus (read from the pinned git commit)."""
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_player_outcome_v2"
V1 = "cea38be352e5539239b29520977e0b19962d2372"
EO = "nhl_models/nhl_outcome_engine/"


def blob(path):
    return subprocess.run(["git", "show", "%s:%s" % (V1, path)], cwd=str(REPO), check=True, capture_output=True).stdout


def fp(path):
    b = blob(path)
    return {"path": path, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "lines": b.count(b"\n")}


def comp(name, files, cls, why, v2_target=None):
    d = {"component": name, "files": [fp(f) for f in files], "classification": cls, "reason": why}
    if v2_target:
        d["v2_location"] = v2_target
    return d


def main():
    h = json.loads(blob(EO + "phase1a_2025_holdout.json"))
    b2 = h["summaries"]["B2"]
    attempt = json.loads(blob(EO + "phase1b_attempt_signal_results.json"))
    components = [
        comp("SOG count mean: B1 PoissonRegressor on ALL candidate rows (alpha=0.001) with training-only median imputation + missing indicators + standardization", ["nhl_sog_phase1a_models.py", EO + "phase1a_selected_architecture.json"],
             "PORT_AS_FIXED_ARCHITECTURE", "Selected B2 mean model; hyper-parameter alpha=0.001 frozen by V1 development folds on 2018-2023 targets; NOT re-tuned.", "nhl_v2_phase1a_sog_model.py"),
        comp("NB2 dispersion: bounded ML of log alpha in [1e-6,20] with mean fixed, training rows only", ["nhl_sog_phase1a_models.py"], "PORT_AS_FIXED_ARCHITECTURE", "Count distribution layer of B2.", "nhl_v2_phase1a_sog_model.py"),
        comp("33 prior-game features (position, home, rest, back-to-back, participation, SOG history, SOG/60, TOI, PP TOI, PP allocation share, shifts, team SOG, opponent SOG allowed)", ["nhl_sog_phase1a_data.py"],
             "PORT_AS_FIXED_ARCHITECTURE", "Feature definitions are the B2 inputs. Ported with two declared changes: horizon-parameterised cutoff, and source games without loaded rows are excluded (no-op historically).", "nhl_v2_phase1a_sog_data.py"),
        comp("Candidate-universe construction (players who appeared for the team in its last 10 completed team games before T)", ["nhl_sog_phase1a_data.py", "nhl_outcome_contract.py"], "PORT_AS_FIXED_ARCHITECTURE",
             "Same universe definition; audited for realism in the V2 candidate-population audit; non-participant zero rows are reported separately.", "nhl_v2_phase1a_sog_data.py"),
        comp("Official-source acquisition (api-web schedule + stats-REST skater/summary + skater/timeonice, weekly windows)", ["nhl_sog_phase1a_data.py"], "PORT_DATA_ENGINEERING_ONLY",
             "Source re-verified reachable; 6 sampled windows (8.8k rows) re-fetched and byte-identical to the frozen rows; ported with retrieval metadata.", "nhl_v2_phase1a_sog_acquire.py"),
        comp("Frozen per-season gzip JSONL data + manifest (retrieved 2026-10-02)", [EO + "phase1a_data_manifest.json", EO + "phase1a_data/skater_games_2025.jsonl.gz", EO + "phase1a_data/games_2025.jsonl.gz"], "PORT_DATA_ENGINEERING_ONLY",
             "Vendored into V2 with per-file sha256 equal to the V1 manifest; the V2 research input is the pinned files, never live historical URLs.", "nhl_models/nhl_player_outcome_v2/phase1a_data/"),
        comp("Data-quality gates (unique rows, schedule membership, SOG range, TOI consistency, ordering, mutation check, candidate equivalence)", ["nhl_sog_phase1a_data.py", EO + "phase1a_data_quality.json"], "PORT_TEST_GUARD",
             "Re-implemented as V2 gates and tests; V1 pass status is reference only.", "nhl_v2_phase1a_sog_data.py + tests"),
        comp("Shootout exclusion + shot-event counting helper", ["nhl_outcome_contract.py"], "PORT_TEST_GUARD", "Player SOG comes from the official summary (shootout shots never credited); helper kept as a guard concept.", "tests"),
        comp("Historical timing contract (source game usable iff start+210 min <= T)", ["nhl_outcome_contract.py"], "PORT_AS_FIXED_ARCHITECTURE", "Applied per forecast horizon (T24H, T90, T30).", "nhl_v2_phase1a_sog_data.py"),
        comp("Evaluation metrics: discrete CRPS, NLL, randomized PIT, central-interval coverage, threshold Brier/ECE, game-macro aggregation, blocked bootstrap", ["nhl_sog_phase1a_metrics.py"], "PORT_AS_FIXED_ARCHITECTURE",
             "Metric definitions are protocol; ported (V2 adds Brier skill and calibration tables).", "nhl_v2_phase1a_sog_model.py"),
        comp("B0 empirical-shrinkage mixture", ["nhl_sog_phase1a_models.py"], "BURNED_RESULT_REFERENCE_ONLY", "Primitive V1 reference; V2 comparators are the Phase0 simple and human baselines. Not ported."),
        comp("B1 Poisson (no dispersion)", ["nhl_sog_phase1a_models.py"], "BURNED_RESULT_REFERENCE_ONLY", "Used only as the mean inside B2."),
        comp("B3 structural participation mixture", ["nhl_sog_phase1a_models.py"], "DO_NOT_PORT", "Not the selected architecture; no automatic adoption. Participation/availability stays a separate, uncertified layer."),
        comp("Tournament orchestration (dev folds, 2024 confirmation, 2025 holdout, G1-G7/H1-H6 gates)", ["nhl_sog_phase1a_run.py"], "DO_NOT_PORT", "Architecture search is finished; no B2-vs-B3 comparison, no re-selection."),
        comp("V1 burned results (2025 B2: MAE %.4f, CRPS %.5f, NLL %.5f, P(SOG>=3) Brier %.5f)" % (b2["mae_mean_prediction"], b2["crps_macro_game"], b2["nll_macro_game"], 0.11700547845476063),
             [EO + "phase1a_2025_holdout.json", EO + "phase1a_2024_confirmation.json", EO + "phase1a_dev_results.json"], "BURNED_RESULT_REFERENCE_ONLY", "2024 and 2025 are burned; used only as reproduction targets, never as validation."),
        comp("Phase1B shot-attempt extension (HYBRID attempt source; mean relative CRPS gain 0.339%% vs 0.5%% materiality)", ["nhl_models/nhl_outcome_engine/phase1b_attempt_signal_results.json", EO + "PHASE1B_ATTEMPT_README.md"],
             "REJECTED_DO_NOT_REOPEN", "Status %s; improved all folds but failed the preregistered materiality gate; not reopened, no variants." % attempt["status"]),
        comp("Forward-capture contract concepts (T24H/T90/T30/T10/T2, schedule hash, retrieval start/completion, immutable blobs, no backfill, hard stop at puck drop, postgame separated)",
             [EO + "phase0b_forward_snapshot_contract.json", EO + "phase0c_capture_protocol.json"], "PORT_DATA_ENGINEERING_ONLY", "Principles preserved in the V2 forward protocol; the deployed collector on main already implements the capture.", "phase1a_sog_forward_protocol.json"),
        comp("Forward-capture code (nhl_fwd_capture/state/eval, nhl_outcome_snapshot)", ["nhl_fwd_capture.py", "nhl_fwd_state.py", "nhl_fwd_eval.py", "nhl_outcome_snapshot.py"], "PORT_DATA_ENGINEERING_ONLY",
             "Byte-identical to the deployed code on main; V2 reads its pregame observations, never modifies it."),
        comp("V1 forward scheduler variant", ["nhl_fwd_scheduler.py"], "DO_NOT_PORT", "Differs from the deployed main version; production scheduler untouched."),
        comp("Exploratory forward observations (WIP watcher)", [EO + "phase0b_forward_snapshot_contract.json"], "BLOCKED_TIMING", "Do not qualify as clean evidence; availability/rosterSpots remain OBSERVED_NOT_CERTIFIED until the capture-integrity gate passes."),
        comp("Historical target-game lineup / scratches / confirmed goalie", [EO + "phase0b_forward_snapshot_contract.json"], "BLOCKED_TIMING", "No archived pregame source exists for history."),
        comp("Goals head G1, assists A1, goalie G0, scoring S0, team T0, registry", [EO + "NHL_OUTCOME_ENGINE_REGISTRY.json", EO + "PHASE_GOALS_G1_README.md", EO + "PHASE_ASSISTS_A1_README.md"], "DO_NOT_PORT", "Out of scope for SOG; no market borrows SOG credibility."),
    ]
    out = {"artifact": "phase1a_v1_migration_audit", "v1_branch": "origin/codex/nhl-outcome-engine-v1", "v1_pinned_commit": V1, "v1_pr": "none", "status_of_v1": "BURNED_PRIOR_RESEARCH_CORPUS",
           "classification_vocabulary": ["PORT_AS_FIXED_ARCHITECTURE", "PORT_DATA_ENGINEERING_ONLY", "PORT_TEST_GUARD", "BURNED_RESULT_REFERENCE_ONLY", "REJECTED_DO_NOT_REOPEN", "BLOCKED_TIMING", "DO_NOT_PORT"],
           "components": components,
           "burned_reference_numbers": {"b2_2025_holdout": {"n_rows": b2["n_rows"], "crps_macro_game": b2["crps_macro_game"], "nll_macro_game": b2["nll_macro_game"], "mae_mean_prediction": b2["mae_mean_prediction"],
                                                            "bias_mean_prediction": b2["bias_mean_prediction"], "p_ge_3_brier": 0.11700547845476063, "central_coverage_50_80_90": [b2["coverage"]["50"], b2["coverage"]["80"], b2["coverage"]["90"]]},
                                        "phase1b_attempt": {"relative_crps_improvement_mean_of_folds": attempt["relative_crps_improvement_mean_of_folds"], "status": attempt["status"]}},
           "evidence_grade": "BURNED_PRIOR_RESEARCH_NOT_VALIDATION", "nothing_copied_blindly": "V1 files are not merged; only the listed components are re-implemented or vendored with hashes in V2 paths",
           "declared_changes_to_ported_builder": ["cutoff parameterised by forecast horizon (V1 used T90 only)", "source games without loaded player rows are excluded from team histories (identical result on complete historical data; required for live forward data)",
                                                  "features can be built for a game without reading its target rows (forward mode)"]}
    (OUT / "phase1a_v1_migration_audit.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(len(components), "components")


if __name__ == "__main__":
    main()
