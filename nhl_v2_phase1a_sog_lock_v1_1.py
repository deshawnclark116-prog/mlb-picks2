#!/usr/bin/env python3
"""Builds the NHL V2 Phase1A-SOG engine v1.1 lock (PRE_FIRST_FORECAST_INFRASTRUCTURE_HARDENING) and the model-identity proof against v1.0.

v1.0 produced ZERO real forecasts. Models are NOT refit: the v1.1 lock points at the byte-identical v1.0 model files; only forward.py / grade.py differ."""
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nhl_v2_phase1a_sog_forward as F

OUT = F.OUT
V10_COMMIT = "ec031c29fd6357691923e95f45f7c37a1b98ff7f"
SCIENCE_FILES = ("nhl_v2_phase1a_sog_acquire.py", "nhl_v2_phase1a_sog_data.py", "nhl_v2_phase1a_sog_model.py", "nhl_v2_phase1a_sog_compare.py", "nhl_v2_phase1a_sog_quality.py")
INFRA_FILES = ("nhl_v2_phase1a_sog_forward.py", "nhl_v2_phase1a_sog_grade.py")
PROTOCOL_FILES = ("phase1a_sog_protocol.json", "phase1a_sog_protocol_amendment_1.json", "phase1a_sog_forward_protocol.json", "phase1a_sog_protocol_amendment_2.json", "phase1a_sog_data_manifest.json",
                  "phase1a_sog_burned_reproduction.json", "phase1a_v1_migration_audit.json", "phase1a_sog_engine_lock.json")


def git_show(path):
    return subprocess.run(["git", "show", "%s:%s" % (V10_COMMIT, path)], cwd=str(F.REPO), check=True, capture_output=True).stdout


def main():
    old_lock = json.loads(git_show("nhl_models/nhl_player_outcome_v2/phase1a_sog_engine_lock.json"))
    proof = {"artifact": "phase1a_sog_model_identity_proof", "v1_0_commit": V10_COMMIT, "v1_0_real_forecasts": 0, "checks": {}}
    ok = True
    for h in F.HORIZONS:
        name = "engine_%s.json" % h
        old = json.loads(git_show("nhl_models/nhl_player_outcome_v2/phase1a_sog_models/" + name))
        cur_bytes = (OUT / "phase1a_sog_models" / name).read_bytes()
        new = json.loads(cur_bytes)
        c = {"file_sha256_equal": F.sha_file(OUT / "phase1a_sog_models" / name) == old_lock["model_sha256"][name], "coef_identical": old["coef"] == new["coef"], "intercept_identical": old["intercept"] == new["intercept"],
             "nb2_dispersion_identical": old["nb2"] == new["nb2"], "feature_order_identical": old["preprocessing"]["feature_order"] == new["preprocessing"]["feature_order"],
             "preprocessing_identical": old["preprocessing"] == new["preprocessing"], "poisson_alpha_identical": old["poisson_alpha"] == new["poisson_alpha"], "artifact_bytes_identical": git_show("nhl_models/nhl_player_outcome_v2/phase1a_sog_models/" + name) == cur_bytes}
        proof["checks"][h] = c; ok = ok and all(c.values())
    proof["checks"]["comparator_constants_and_base_rates_identical_in_lock"] = True        # copied verbatim below
    proof["science_modules_byte_identical"] = {f: F.sha_file(F.REPO / f) == old_lock["code_sha256"][f] for f in SCIENCE_FILES}
    ok = ok and all(proof["science_modules_byte_identical"].values())
    proof["infrastructure_modules_changed"] = {f: {"v1_0_sha256": old_lock["code_sha256"][f], "v1_1_sha256": F.sha_file(F.REPO / f)} for f in INFRA_FILES}
    proof["all_model_and_science_identity_checks_pass"] = bool(ok)
    if not ok:
        raise SystemExit("identity proof FAILED: " + json.dumps(proof["checks"]))
    now = datetime.now(timezone.utc)
    prev = subprocess.run(["git", "show", "HEAD:nhl_models/nhl_player_outcome_v2/phase1a_sog_engine_lock_v1_1.json"], cwd=str(F.REPO), capture_output=True)
    prior_v1_1 = F.sha_text(prev.stdout.decode()) if prev.returncode == 0 else None
    lock = {k: old_lock[k] for k in ("architecture", "horizons", "fit", "model_sha256", "comparator_constants", "base_rates", "meaningful_rule", "no_parameter_refit_during_forward_window", "availability_used_in_forecast",
                                      "no_betting_market_inputs", "no_simulation", "table_sha256", "burned_reproduction_status")}
    lock.update({"artifact": "phase1a_sog_engine_lock_v1_1", "lock_id": "nhl-v2-sog-engine-lock-1.1", "engine_version": F.ENGINE_VERSION, "supersedes": "nhl-v2-sog-b2-1.0", "supersedes_lock_file": "phase1a_sog_engine_lock.json",
                 "reason": "PRE_FIRST_FORECAST_INFRASTRUCTURE_HARDENING", "v1_0_real_forecasts": 0,
                 "changes": ["decision key includes scheduled start (revision semantics)", "per-run content-addressed source manifests with full retrieval provenance", "hard source-completeness gate and explicit non-forecast statuses"],
                 "locked_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "eligible_from_cutoff_utc": (now + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "eligibility_note": "only horizon cutoffs at/after eligible_from_cutoff_utc can produce valid forecasts, and only after this lock commit is pushed and CI is green",
                 "code_sha256": {f: F.sha_file(F.REPO / f) for f in F.LOCK_CODE_FILES}, "protocol_sha256": {f: F.sha_file(OUT / f) for f in PROTOCOL_FILES}, "identity_proof": "phase1a_sog_model_identity_proof.json",
                 "reissued_before_any_forecast": prior_v1_1 is not None, "prior_v1_1_lock_text_sha256": prior_v1_1, "reissue_reason": "stray test/probe blobs were committed and store_blob bound its directory at import; fixed before any forecast" if prior_v1_1 else None})
    (OUT / "phase1a_sog_model_identity_proof.json").write_text(json.dumps(proof, indent=1, sort_keys=True) + "\n")
    F.LOCK.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
    print("v1.1 lock written", lock["locked_at_utc"], lock["eligible_from_cutoff_utc"])


if __name__ == "__main__":
    main()
