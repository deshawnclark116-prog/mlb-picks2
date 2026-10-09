"""
NFL_SHADOW_WATCHDOG -- redundant TRIGGER decision for the existing NFL Phase 1E collector. It is not a collector and not a model.

Reads the checked-out nfl-shadow-state (read-only) and decides whether to dispatch the SAME workflow (nfl_phase1e_shadow.yml -> nfl_phase1e_scheduler.py run).
It dispatches only when ALL hold: the last real collector invocation (status.json last_invocation_utc) is older than the stale threshold; at least one still-PLANNED decision key has a cutoff inside the wake
window (cutoff <= now + window, kickoff still in the future); and no shadow run is already queued / in progress (passed in by the caller). It never captures, never backfills, never writes the state.
A cutoff that has already passed is handled by the scheduler itself (MISSED_REAL_CUTOFF); this module only wakes it.
"""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

STALE_MIN = 25
WINDOW_MIN = 60
DURATION_MIN = 45


def _t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def decide(state, now, active_runs=0, stale_min=STALE_MIN, window_min=WINDOW_MIN, duration_min=DURATION_MIN):
    state = Path(state)
    st = json.loads((state / "status.json").read_text()) if (state / "status.json").exists() else {}
    last = st.get("last_successful_schedule_capture_utc") or st.get("last_schedule_probe_utc")
    age = None if not last else (now - _t(last)).total_seconds() / 60
    keys = {}
    p = state / "dispatch_ledger.jsonl"
    if p.exists():
        for x in p.read_text().splitlines():
            if x.strip():
                r = json.loads(x); keys[r["key"]] = r
    due = sorted(r["cutoff"] for r in keys.values() if r["state"] in ("PLANNED", "STARTED") and _t(r["cutoff"]) <= now + timedelta(minutes=window_min))
    failed = (st.get("last_operational_event") or {}).get("status") == "FAILED"
    stale = age is None or age > stale_min or failed
    overdue = any(_t(r["cutoff"]) < now for r in keys.values() if r["state"] in ("PLANNED", "STARTED"))
    out = {"now": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "last_invocation_utc": last, "age_min": None if age is None else round(age, 1), "stale": stale, "cutoffs_in_window": len(due), "next_due_cutoff": due[0] if due else None,
           "active_runs": active_runs, "duration_min": duration_min}
    if active_runs:
        out["action"], out["reason"] = "skip", "a shadow run is already queued or in progress"
    elif not stale and not overdue and not due:
        out["action"], out["reason"] = "skip", "collector fresh"
    elif not due:
        out["action"], out["reason"] = "skip", "stale but no cutoff inside the wake window"
    else:
        out["action"], out["reason"] = "dispatch", "collector stale/failed or overdue key needs closure"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True); ap.add_argument("--active-runs", type=int, default=0)
    ap.add_argument("--stale-min", type=int, default=STALE_MIN); ap.add_argument("--window-min", type=int, default=WINDOW_MIN); ap.add_argument("--duration-min", type=int, default=DURATION_MIN)
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    print(json.dumps(decide(a.state, now, a.active_runs, a.stale_min, a.window_min, a.duration_min), sort_keys=True))
