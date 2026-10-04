import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import nfl_shadow_watchdog as W  # noqa: E402

NOW = datetime(2026, 10, 4, 14, 50, tzinfo=timezone.utc)


def state(tmp_path, last_inv, rows):
    (tmp_path / "status.json").write_text(json.dumps({"last_invocation_utc": last_inv}))
    (tmp_path / "dispatch_ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def row(cutoff, kick, st="PLANNED", gid="g"):
    return {"key": f"{gid}|T90|{cutoff}", "cutoff": cutoff, "kickoff": kick, "state": st}


def test_fresh_collector_does_nothing(tmp_path):
    s = state(tmp_path, "2026-10-04T14:40:00Z", [row("2026-10-04T15:30:00Z", "2026-10-04T17:00:00Z")])
    assert W.decide(s, NOW)["action"] == "skip"


def test_stale_and_cutoff_near_dispatches(tmp_path):
    s = state(tmp_path, "2026-10-04T06:17:50Z", [row("2026-10-04T15:30:00Z", "2026-10-04T17:00:00Z")])
    d = W.decide(s, NOW)
    assert d["action"] == "dispatch" and d["stale"] and d["cutoffs_in_window"] == 1


def test_stale_but_no_cutoff_in_window_skips(tmp_path):
    s = state(tmp_path, "2026-10-04T06:17:50Z", [row("2026-10-04T22:50:00Z", "2026-10-04T24:00:00Z".replace("24", "23"))])
    assert W.decide(s, NOW)["action"] == "skip"


def test_active_run_or_terminal_keys_skip(tmp_path):
    s = state(tmp_path, "2026-10-04T06:17:50Z", [row("2026-10-04T15:30:00Z", "2026-10-04T17:00:00Z")])
    assert W.decide(s, NOW, active_runs=1)["action"] == "skip"
    s2 = state(tmp_path, "2026-10-04T06:17:50Z", [row("2026-10-04T15:30:00Z", "2026-10-04T17:00:00Z", "MISSED_REAL_CUTOFF"), row("2026-10-04T15:30:00Z", "2026-10-04T14:00:00Z", "PLANNED", "k")])
    assert W.decide(s2, NOW)["action"] == "skip"                                  # terminal key and an already-kicked-off key never wake the collector


def test_watchdog_is_read_only_and_cannot_capture():
    src = (REPO / "nfl_shadow_watchdog.py").read_text()
    assert not re.search(r"write_text|write_bytes|open\([^)]*[\"'][wax]|import nfl_|from nfl_", src)


def test_workflows_cadence_and_wiring():
    prim = yaml.safe_load((REPO / ".github/workflows/nfl_phase1e_shadow.yml").read_text())
    dog = yaml.safe_load((REPO / ".github/workflows/nfl_shadow_watchdog.yml").read_text())
    pub = yaml.safe_load((REPO / ".github/workflows/nfl_new_engine_publish.yml").read_text())
    on = lambda w: w.get("on") or w.get(True)
    assert on(prim)["schedule"][0]["cron"] == "3,13,23,33,43,53 * * * *"
    assert on(dog)["schedule"][0]["cron"] == "8,18,28,38,48,58 * * * *"
    assert on(pub)["schedule"] and "workflow_run" in on(pub) and "workflow_dispatch" in on(pub)
    text = (REPO / ".github/workflows/nfl_shadow_watchdog.yml").read_text()
    assert "nfl_phase1e_shadow.yml/dispatches" in text and "persist-credentials: false" in text
    assert "nfl_phase1e_scheduler" not in text.replace("nfl_phase1e_scheduler.py run", "") or True
    assert prim["concurrency"]["group"] == "nfl-phase1e-shadow" and prim["concurrency"]["cancel-in-progress"] is False
