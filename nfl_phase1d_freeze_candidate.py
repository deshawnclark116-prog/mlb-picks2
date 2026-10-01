"""
NFL_PHASE1D_FREEZE_CANDIDATE  (Phase 1D)  -- build and verify nfl_models/nfl_player_outcome_phase1_freeze_candidate_v2.json

This is a FREEZE CANDIDATE for audit. It is NOT the freeze: nfl_player_outcome_phase1_freeze.json is never created by this code.

  build    (run at a clean, committed HEAD)   python nfl_phase1d_freeze_candidate.py build [--skip-tests]
  verify   (after build, nothing modified)    python nfl_phase1d_freeze_candidate.py verify

build refuses a dirty working tree (tracked files), records HEAD, hashes every file that would be frozen from disk, records the artifact / calibration / hyper-parameter / constants / protocol
hashes and the results of the complete test suite run at that HEAD. verify re-checks HEAD, every hash and that no tracked file changed.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
P1D = REPO / "nfl_models" / "nfl_player_outcome_phase1d"
CAND = REPO / "nfl_models" / "nfl_player_outcome_phase1_freeze_candidate_v2.json"
FINAL = REPO / "nfl_models" / "nfl_player_outcome_phase1_freeze.json"
VALIDATION = [REPO / "freeze_candidate_validation_v2.md", P1D / "freeze_candidate_validation_v2.md"]
TESTS = ["tests/test_nfl_player_outcome_invariants.py", "tests/test_nfl_phase1_invariants.py", "tests/test_nfl_phase1b_invariants.py", "tests/test_nfl_phase1c_invariants.py", "tests/test_nfl_phase1d.py"]
ARTIFACT_DIR = P1D / "artifacts_freeze_fit_2026wk3"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()


def tracked_dirty():
    return git("status", "--porcelain", "--untracked-files=no")


def files_to_freeze():
    out = set()
    for pat in ("nfl_phase1*.py", "nfl_context_v4.py", "tests/test_nfl_phase1*.py"):
        out |= {str(p.relative_to(REPO)) for p in REPO.glob(pat)}
    out |= {"tests/test_nfl_player_outcome_invariants.py"}
    P1C = REPO / "nfl_models" / "nfl_player_outcome_phase1c"
    for n in ("constants.json", "selected_components.json", "adjudication_rule.json", "adjudication_results.json", "calibration_depth.json", "simulation_convergence.json"):
        out.add(str((P1C / n).relative_to(REPO)))
    out.add("nfl_models/nfl_player_outcome_phase1_protocol.json")
    for p in P1D.rglob("*"):
        if p.is_file() and "partial" not in p.name and p.name != "freeze_candidate_validation_v2.md":
            out.add(str(p.relative_to(REPO)))
    out = {f for f in out if f != str(CAND.relative_to(REPO)) and (REPO / f).exists()}
    return sorted(out)


DATA_DIR = os.environ.get("NFL_DATA_DIR", "/tmp/nflcsv")
DATA_TESTS = ("tests/test_nfl_player_outcome_invariants.py", "tests/test_nfl_phase1_invariants.py", "tests/test_nfl_phase1b_invariants.py")     # also run their data invariants


def run_tests():
    res = {}
    for t in TESTS:
        args = ["--data-dir", DATA_DIR] if t in DATA_TESTS else []
        env = {**os.environ, "NFL_SKIP_CANDIDATE_TEST": "1"}        # the candidate-v2 test cannot pass before the candidate exists; `verify` + the full suite are re-run after the build
        p = subprocess.run([sys.executable, t, *args], cwd=REPO, capture_output=True, text=True, timeout=7200, env=env)
        out = p.stdout + p.stderr
        res[t] = {"exit_code": p.returncode, "pass_lines": len(re.findall(r"^PASS", out, re.M)), "fail_lines": len(re.findall(r"^FAIL", out, re.M)),
                  "summary_tail": out.strip().splitlines()[-1][:200] if out.strip() else ""}
    return res


def build(skip_tests=False, files_added_after_build=None):
    dirty = tracked_dirty()
    if dirty:
        raise SystemExit(f"working tree has modified tracked files; commit first:\n{dirty}")
    head = git("rev-parse", "HEAD")
    import nfl_phase1_forecast as FC
    ART = json.loads((ARTIFACT_DIR / "manifest.json").read_text())
    cal = json.loads((P1D / "calibration_freeze.json").read_text())
    proto = json.loads((REPO / "nfl_models" / "nfl_player_outcome_phase1_protocol.json").read_text())
    manifest_files = files_to_freeze()
    tests = None if skip_tests else run_tests()
    m = {
        "STATUS": "FREEZE CANDIDATE v2 FOR AUDIT. NOT A FREEZE. nfl_player_outcome_phase1_freeze.json does not exist and must not be created before the audit.",
        "research_interpretation": __import__("nfl_phase1d_report").RESEARCH_INTERPRETATION,
        "supersedes": "nfl_models/nfl_player_outcome_phase1_freeze_candidate.json (stale: built at 658c498 while later code existed)",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit_sha": head,
        "working_tree_clean_at_build": True,
        "files_added_after_build": files_added_after_build if files_added_after_build is not None else [
            "nfl_models/nfl_player_outcome_phase1_freeze_candidate_v2.json", "freeze_candidate_validation_v2.md", "nfl_models/nfl_player_outcome_phase1d/freeze_candidate_validation_v2.md"],
        "commit_note": "the candidate is built from HEAD; the commit that stores this file and the validation documents adds only the paths in files_added_after_build (checked by tests/test_nfl_phase1d.py)",
        "protocol": {"file": "nfl_models/nfl_player_outcome_phase1_protocol.json", "version": proto["version"], "sha256": sha(REPO / "nfl_models" / "nfl_player_outcome_phase1_protocol.json"),
                     "amendment_G_registered": any(a["id"] == "G" for a in proto.get("amendments_v1_2", []))},
        "model_composition": {
            "model_version_rule": "p1c-<sha256 of {component architecture, constants, calibration maps, n_draws, sim version, qb_adjust, code hashes of FC.CODE_FILES, fit_end, target week, weekly Phase 1A artifact "
                                  "bundle hash, Phase 1B hyper-parameter hash}>; a new model_version therefore exists for every weekly refit (new artifact bundle hash) and for any code change",
            "code_hashes_forecast_path": FC.file_hashes(),
            "components": json.loads((REPO / "nfl_models" / "nfl_player_outcome_phase1c" / "selected_components.json").read_text())["components"],
            "phase1a_frozen_selection_sha256": sha(P1D / "phase1a_frozen_selection.json"),
            "phase1b_frozen_hyper_sha256": sha(P1D / "phase1b_frozen_hyper.json"),
            "constants_sha256": sha(REPO / "nfl_models" / "nfl_player_outcome_phase1c" / "constants.json"),
            "calibration": {"label": cal["label"], "all_maps_sha256": cal["all_maps_sha256"], "maps": {k: v["sha256"] for k, v in cal["maps"].items()}},
            "n_draws": 25000, "n_draws_rule": "engineering selection (n_engineering_criteria.json, verified in n_engineering_audit.json)",
            "universe": "depth-chart-extended candidate universe (development-selected, frozen)", "game_script": "S0 static pregame volumes",
            "freeze_artifact_bundle": {"dir": str(ARTIFACT_DIR.relative_to(REPO)), "bundle_sha256": ART["bundle_sha256"], "files": ART["files"], "meta": ART["meta"]},
            "walk_forward_refit": "nfl_phase1d_p1a.fit_artifacts + nfl_phase1c_fit.fit_all(frozen hyper-parameters); training = completed weeks before the target week from 2023 on; every refit logged in weekly_fits.jsonl"},
        "baselines": {"historical": "nfl_phase1d_baselines.py HB1", "availability": "status-only lookup of the weekly artifact", "anytime_td": "HB1 anytime-TD"},
        "gates": "protocol Amendment G (nfl_phase1d_gates.AMENDMENT_G)",
        "forecast_horizons": {"primary": "T24", "secondary": "T90 (separately reported)"},
        "clean_forward_window": proto["clean_forward_window"],
        "tests_at_head": tests,
        "files_sha256": {f: sha(REPO / f) for f in manifest_files},
        "n_files": len(manifest_files),
        "not_created": str(FINAL.relative_to(REPO)), "final_freeze_file_exists": FINAL.exists(),
    }
    CAND.write_text(json.dumps(m, indent=1, sort_keys=True))
    return m


def verify():
    m = json.loads(CAND.read_text())
    head = git("rev-parse", "HEAD")
    problems = []
    if m["code_commit_sha"] != head:
        problems.append(f"candidate sha {m['code_commit_sha']} != HEAD {head}")
    for rel, h in m["files_sha256"].items():
        p = REPO / rel
        if not p.exists():
            problems.append(f"missing {rel}")
        elif sha(p) != h:
            problems.append(f"hash mismatch {rel}")
    d = tracked_dirty()
    if d:
        problems.append(f"modified tracked files: {d}")
    if FINAL.exists():
        problems.append("final freeze file exists")
    return {"head": head, "candidate_sha": m["code_commit_sha"], "sha_equals_head": m["code_commit_sha"] == head, "n_files_checked": len(m["files_sha256"]), "problems": problems, "ok": not problems}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify"])
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        m = build(a.skip_tests)
        print(m["code_commit_sha"], m["n_files"], {k: (v["pass_lines"], v["fail_lines"]) for k, v in (m["tests_at_head"] or {}).items()})
    else:
        r = verify()
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["ok"] else 1)
