"""
CFB_PHASE1_FREEZE -- builds cfb_phase1_freeze_candidate.json: the frozen core-engine bundle (components, features, hyper-parameters, data hashes, code hashes, universe / newcomer policy, simulator constants, N, calibration method, blockers) BEFORE any 2025 scoring.
The model artifact is the engine fitted on the 2019-2024 window (seasons <= 2024 only); the fit is deterministic, so the confirmation re-fits and must reproduce the artifact hash exactly.
  python cfb_phase1_freeze.py build --tests "<recorded test summary json>"
"""
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

import cfb_phase1_common as C
import cfb_phase1_data as D
import cfb_phase1_forecast as FC
import cfb_phase1_role_state as RS
import cfb_phase1_team_environment as TE
import cfb_phase1b_components as K
import cfb_phase1b_experiments as E

REPO = Path(__file__).resolve().parent
OUT = REPO / "cfb_models" / "cfb_outcome_engine"
CODE_FILES = ["cfb_phase1_common.py", "cfb_phase1_data.py", "cfb_phase1_team_environment.py", "cfb_phase1_role_state.py", "cfb_phase1_opportunity.py", "cfb_phase1_efficiency.py", "cfb_phase1b_components.py", "cfb_phase1b_harness.py", "cfb_phase1_forecast.py",
              "cfb_phase1_sim.py", "cfb_phase1_integrated.py", "cfb_phase1_convergence.py", "cfb_phase1_freeze.py", "cfb_phase1_confirm2025.py"]
GOV_FILES = ["phase1_protocol.json", "phase1_protocol_amendment_1.json", "phase1b_core_engine_protocol.json", "phase1b_development_amendment_1.json", "phase1b_development_amendment_2.json", "phase1b_simulator_registration.json", "phase1b_target_recovery_gate.json",
             "phase1b_target_recovery_audit.json", "cfb_candidate_universe_contract.json", "cfb_shared_state_contract.json", "cfb_newcomer_state_contract.json", "cfb_identity_contract.json"]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_json(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=float).encode()).hexdigest()


def fit_bundle_engine(choices=None):
    """Fit the engine on the 2019-2024 window (the training window of the 2025 confirmation). Seasons <= 2024 only."""
    tg, pg, rows, team = E.load()
    pm = E.pos_map(pg)
    ev = D.load_events(list(range(2018, 2025)))
    team_by_key = {(r["game_id"], r["team"]): r for r in team}
    eng = FC.CoreEngine(choices or FC.DEFAULT_CHOICES).fit(rows, team, ev, pm, team_by_key)
    return eng


def build(tests_summary=None, int_dev_path=None):
    eng = fit_bundle_engine()
    art = eng.artifact()
    freeze = {"freeze_version": "cfb-phase1-freeze-candidate-1", "status": "FROZEN_CANDIDATE_BEFORE_2025", "branch_head_at_freeze": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
              "component_choices": eng.ch, "training_window": "2019-2024 (2018 warm-up history)", "scoring_target": "2025 regular season (LATE_PERIOD_INTEGRATED_CONFIRMATION_PREVIOUSLY_EXPOSED), scored ONCE per bundle hash",
              "feature_lists": {"team_and_context_v2": TE.TEAM_FEATURES_V2, "player_role_state": RS.ROLE_FEATURES, "player_count_windows_l20": RS.L20_FEATURES, "role_model_inputs": K.ROLE_NAMES, "hazard_context": FC.CTX},
              "hyperparameters": {"blend_k": TE.BLEND_K, "elo": {"init_fbs": TE.INIT_FBS, "init_fcs": TE.INIT_FCS, "K": TE.ELO_K, "hfa": TE.ELO_HFA, "season_regression": TE.ELO_REGRESS}, "poisson_alpha": TE.POISSON_ALPHA, "logistic_C": 1.0, "kappa_bounds": [0.3, 800.0],
                                  "yard_clip": [-10, 99], "event_window": 150, "rush_kappa": eng.ch["rush_kappa"], "rec_kappa": eng.ch["rec_kappa"], "completion_prior_k": 80, "hazard_prior_k": {"rush_td": 100, "rec_td": 60, "int": "league constant (valid seasons only)"},
                                  "int_valid_seasons": list(FC.INT_VALID_SEASONS), "resid_shift": eng.ch["resid_shift"], "recency_team_games": D.RECENCY_TEAM_GAMES, "coverage_valid_plays": [C.MIN_VALID_PLAYS, C.MAX_VALID_PLAYS]},
              "data_hashes": {"research_tables_manifest_sha256": json.loads((D.DATA / "manifest.json").read_text())["manifest_content_sha256"], "files": json.loads((D.DATA / "manifest.json").read_text())["files"]},
              "candidate_universe_version": {"contract": "cfb_candidate_universe_contract.json", "contract_sha256": sha_file(OUT / "cfb_candidate_universe_contract.json"), "rule": "last appearance for the team within the team's last 12 completed games; skill positions; earlier weeks only"},
              "newcomer_policy": {"contract": "cfb_newcomer_state_contract.json", "contract_sha256": sha_file(OUT / "cfb_newcomer_state_contract.json"), "historical": "explicit OTHER bucket per allocation; unobservable newcomers never scored as rows", "forward": "timestamped live roster adds zero-history candidates with wide high-uncertainty priors"},
              "simulator": {"module": "cfb_phase1_sim.py", "registration": "phase1b_simulator_registration.json", "registration_sha256": sha_file(OUT / "phase1b_simulator_registration.json"), "N_confirmation": 4000, "seed_rule": "1000 * season + game ordinal", "identities": "checked every game"},
              "calibration_method": "importance resampling of joint draws to the validated direct team-points component (NB2 C1 state GLM); no other post-hoc calibration; no Platt layer",
              "code_hashes": {f: sha_file(REPO / f) for f in CODE_FILES if (REPO / f).exists()}, "governance_hashes": {f: sha_file(OUT / f) for f in GOV_FILES},
              "model_artifact_sha256": sha_json(art), "model_artifact": art, "tests": tests_summary or "recorded by the caller before commit",
              "known_blockers": ["no depth chart / injury / snap / route / true-target / recruiting / portal / coaching sources (BLOCKED_HISTORICAL_PIT or no source)", "interception attribution defective in 2021-2023 (INT fitted on valid seasons only, league-constant)", "gadget (WR / TE) carry allocation worse than per-row baseline marginals on tiny CRPS (G4 recorded, not rescued)",
                                "independent T2 B0 remains the simple reference for pass attempts (V2 PIT-KS gate)", "player universe FBS-vs-FBS only; unobservable newcomers are an explicit residual", "team-points calibration relies on importance resampling to the direct component",
                                "forward newcomer priors cannot be validated historically", "all 2019-2024 results are burned development, not forward evidence"]}
    freeze["bundle_hash"] = sha_json({k: v for k, v in freeze.items() if k not in ("branch_head_at_freeze", "tests")})
    (OUT / "cfb_phase1_freeze_candidate.json").write_text(json.dumps(freeze, indent=1, sort_keys=True, default=float) + "\n")
    return freeze


if __name__ == "__main__":
    if sys.argv[1] == "build":
        t = json.loads(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--tests" else None
        f = build(t)
        print(f["bundle_hash"], f["model_artifact_sha256"])
