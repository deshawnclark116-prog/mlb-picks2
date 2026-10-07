"""Engine-lock, ledger-integrity and workflow-guard tests for NHL V2 Phase1A-SOG."""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "nhl_models" / "nhl_player_outcome_v2"
import nhl_v2_phase1a_sog_forward as F


def git(*a):
    return subprocess.run(["git"] + list(a), cwd=str(REPO), capture_output=True, text=True)


def test_lock_matches_code_models_and_protocols():
    if not F.LOCK.exists():
        return
    lock = F.verify_lock()
    assert lock["architecture"] == "B2_FIXED" and lock["no_parameter_refit_during_forward_window"] is True and lock["availability_used_in_forecast"] is False
    assert set(lock["horizons"]) == {"T24H", "T90", "T30"}
    for h, f in lock["fit"]["per_horizon"].items():
        assert f["converged"] and f["poisson_alpha"] == 0.001
    assert lock["burned_reproduction_status"] == "B2_REPRODUCED"
    assert lock["engine_version"] == "nhl-v2-sog-b2-1.1" and lock["supersedes"] == "nhl-v2-sog-b2-1.0" and lock["v1_0_real_forecasts"] == 0 and lock["reason"] == "PRE_FIRST_FORECAST_INFRASTRUCTURE_HARDENING"


V10 = "ec031c29fd6357691923e95f45f7c37a1b98ff7f"


def test_v1_0_lock_is_preserved_unmodified_and_models_are_identical():
    base = "nhl_models/nhl_player_outcome_v2/"
    old = git("show", V10 + ":" + base + "phase1a_sog_engine_lock.json")
    if old.returncode != 0:
        return
    assert old.stdout == (OUT / "phase1a_sog_engine_lock.json").read_text()
    for h in ("T24H", "T90", "T30"):
        f = "phase1a_sog_models/engine_%s.json" % h
        assert git("show", V10 + ":" + base + f).stdout == (OUT / f).read_text()
    for f in ("nhl_v2_phase1a_sog_acquire.py", "nhl_v2_phase1a_sog_data.py", "nhl_v2_phase1a_sog_model.py", "nhl_v2_phase1a_sog_compare.py", "nhl_v2_phase1a_sog_quality.py"):
        assert git("show", V10 + ":" + f).stdout == (REPO / f).read_text(), f


def test_identity_proof_all_checks_true():
    p = OUT / "phase1a_sog_model_identity_proof.json"
    if not p.exists():
        return
    pr = json.loads(p.read_text())
    assert pr["all_model_and_science_identity_checks_pass"] is True and pr["v1_0_real_forecasts"] == 0
    for h, c in pr["checks"].items():
        if isinstance(c, dict):
            assert all(c.values()), h
    assert all(pr["science_modules_byte_identical"].values())


def test_ledger_chain_valid_and_history_only_appends():
    if not F.LEDGER.exists():
        return
    F.Ledger().verify()
    path = "nhl_models/nhl_player_outcome_v2/phase1a_sog_forward/ledger.jsonl"
    commits = git("log", "--reverse", "--format=%H", "--", path).stdout.split()
    prev = ""
    for c in commits:
        cur = git("show", c + ":" + path).stdout
        assert cur.startswith(prev), "ledger rewritten in " + c
        prev = cur


def test_every_valid_forecast_respects_lock_window_and_firewall():
    if not F.LEDGER.exists() or not F.LOCK.exists():
        return
    lock = json.loads(F.LOCK.read_text())
    eligible = F.parse_iso(lock["eligible_from_cutoff_utc"]); lock_at = F.parse_iso(lock["locked_at_utc"])
    for r in F.Ledger().rows():
        if r["record_type"] == "FORECAST":
            c, g = F.parse_iso(r["cutoff_at"]), F.parse_iso(r["generated_at"])
            assert c >= eligible and g >= lock_at and c.timestamp() - F.WINDOW_S <= g.timestamp() <= c.timestamp()
            assert r["availability_confidence"] == "NOT_CERTIFIED" and r["availability_used_in_forecast"] is False
            assert re.search(r"odds|sportsbook|bookmaker", json.dumps(r).lower()) is None
            assert r["lock_sha256"] and r["engine_version"] == lock["engine_version"] and r["source_manifest_sha256"]
    assert F.audit_ledger() == []


def test_forward_workflow_is_dispatch_only_and_research_scoped():
    wf = (REPO / ".github" / "workflows" / "nhl_v2_phase1a_sog_forward.yml").read_text()
    assert "workflow_dispatch" in wf and "schedule:" not in wf and "cron" not in wf
    ci = (REPO / ".github" / "workflows" / "nhl_v2_phase1a_sog_ci.yml").read_text()
    assert "cron" not in ci


def test_snapshot_matches_files_when_present():
    import hashlib
    p = OUT / "phase1a_sog_snapshot.json"
    if not p.exists():
        return
    s = json.loads(p.read_text())
    sha = lambda x: hashlib.sha256(Path(x).read_bytes()).hexdigest()
    for n, h in s["artifact_sha256"].items():
        assert sha(OUT / n) == h, n
    for n, h in s["code_sha256"].items():
        assert sha(REPO / n) == h, n
    for n, h in s["models_sha256"].items():
        assert sha(OUT / "phase1a_sog_models" / n) == h, n
    assert sha(OUT / "phase1a_sog_engine_lock.json") == s["lock_sha256"]
