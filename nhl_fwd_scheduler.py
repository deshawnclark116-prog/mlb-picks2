"""
NHL_FWD_SCHEDULER (Phase 0C) -- durable forward-snapshot dispatcher. One invocation = one GitHub Actions job: observe the schedule, plan decision keys, wait INSIDE the job until the registered capture start of each due
cutoff group, capture the group's games concurrently, capture postgame truth, evaluate, write status/readiness, persist state to the `nhl-forward-state` branch. No daemon, no interactive watcher, no prediction.

  python nhl_fwd_scheduler.py run       --state DIR [--duration-min 25] [--persist]
  python nhl_fwd_scheduler.py status    --state DIR
  python nhl_fwd_scheduler.py readiness --state DIR            (exit 2 when not ready)
  python nhl_fwd_scheduler.py evaluate  --state DIR
"""
import argparse
import ast
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nhl_fwd_capture as C
import nhl_fwd_eval as EV
import nhl_fwd_ops as OPS
import nhl_fwd_state as S
import nhl_outcome_snapshot as SN

REPO = Path(__file__).resolve().parent
UTC = timezone.utc
OUT = REPO / "nhl_models" / "nhl_outcome_engine"
PROTOCOL_FILE = OUT / "phase0c_capture_protocol.json"
CONTRACT_FILE = OUT / "phase0b_forward_snapshot_contract.json"
WORKFLOW = REPO / ".github" / "workflows" / "nhl_forward_snapshot.yml"
WATCHDOG_WORKFLOW = REPO / ".github" / "workflows" / "nhl_forward_watchdog.yml"
CODE_FILES = ["nhl_fwd_state.py", "nhl_fwd_capture.py", "nhl_fwd_eval.py", "nhl_fwd_scheduler.py", "nhl_fwd_ops.py", "nhl_outcome_snapshot.py"]
FORBIDDEN_IMPORT_PREFIXES = ("nfl_", "nhl_sog", "nhl_serving", "nhl_live", "nhl_player", "xgboost", "sklearn", "numpy", "scipy", "pandas", "lightgbm")
WAIT_STEP_S = 30


def utcnow():
    return datetime.now(UTC)


def git_persist(state_root, message, attempts=4, sleep=time.sleep):
    """Commit and push the state branch; FAILS LOUDLY (raises) if every push attempt fails."""
    def g(*a, check=True):
        return subprocess.run(["git", *a], cwd=state_root, capture_output=True, text=True, check=check)
    g("config", "user.name", "nhl-forward-bot"); g("config", "user.email", "nhl-forward-bot@users.noreply.github.com")
    g("add", "-A")
    if g("diff", "--cached", "--quiet", check=False).returncode == 0:
        return "no_change"
    g("commit", "-q", "-m", message)
    for i in range(attempts):
        if g("push", "origin", "HEAD:nhl-forward-state", check=False).returncode == 0:
            return "pushed"
        g("pull", "--rebase", "origin", "nhl-forward-state", check=False)
        sleep(2 * (i + 1))
    raise S.HardError("state branch push failed after retries: durable state was NOT saved")


class Runner:
    def __init__(self, state_root, http=SN.real_http, clock=utcnow, sleep=time.sleep, map_fn=C.default_map, persist=None, log=print, provenance=None):
        self.state_root = Path(state_root)
        self.st = C.State(state_root)
        self.prior_last_invocation = prior_last_invocation(state_root)                      # operational: when the collector last ran BEFORE this invocation
        self.provenance = provenance or OPS.provenance({}, clock())
        OPS.record_invocation(state_root, self.provenance, clock())
        self.http, self.clock, self.sleep, self.map_fn, self.persist, self.log = http, clock, sleep, map_fn, persist, log

    def wait_until(self, t):
        while True:
            n = self.clock()
            if n >= t:
                return
            self.sleep(min((t - n).total_seconds(), WAIT_STEP_S))

    def observe(self):
        now = self.clock()
        sched = C.observe_schedule(self.st, self.http, self.clock)
        new, revised = C.plan_keys(self.st, sched, self.clock())
        return sched, new, revised

    def process_group(self, group):
        st = self.st
        cutoff = C.parse_iso(group[0]["cutoff"])
        if self.clock() < cutoff - timedelta(seconds=C.CAPTURE_START_LEAD_SECONDS):
            self.wait_until(cutoff - timedelta(seconds=C.CAPTURE_START_LEAD_SECONDS))     # woke early; capture only inside the registered capture window
        if self.clock() < cutoff:
            self.observe()                                                                 # schedule re-check at capture start: a revised start closes the old key and creates a new one
        done = {r["key"] for r in st.results.read()}
        live = [k for k in group if k["key"] not in done]
        now = self.clock()
        out = []
        past = [k for k in live if C.parse_iso(k["cutoff"]) <= now and not any(o["key"] == k["key"] for o in st.obs.read())]
        for k in past:                                                                     # cutoff already passed without any capture: MISSED, never backfilled
            out.append(st.results.append_unique(["key"], {"key": k["key"], "game_id": k["game_id"], "horizon": k["horizon"], "scheduled_start_utc": k["scheduled_start_utc"], "cutoff": k["cutoff"], "status": C.MISSED,
                                                          "reason": "cutoff_passed_before_capture_started", "finalized_at": C.iso(now), "schedule_sha256": k["schedule_sha256"]}, immutable=("status",))[0])
        todo = [k for k in live if k not in past]
        res = self.map_fn(lambda k: C.capture_key(st, k, self.http, self.clock), todo)
        out += [r for r in res if r]
        if self.persist:
            self.persist("capture group " + group[0]["cutoff"])
        return out

    def postgame_pass(self):
        todo = C.postgame_candidates(self.st, self.clock())
        res = self.map_fn(lambda r: C.capture_postgame(self.st, r["game_id"], r, self.http, self.clock), todo)
        return [r for r in res if r]

    def tick(self):
        sched, new, revised = self.observe()
        groups = C.due_groups(self.st, self.clock())
        processed = 0
        for g in groups:
            self.process_group(g); processed += 1
        pg = self.postgame_pass()
        ev = EV.evaluate(self.st)
        rd = readiness(self.state_root, self.http, self.clock, prior_last_invocation=self.prior_last_invocation)
        status = write_status(self.st, self.clock(), ev, rd, {"groups_processed": processed, "new_keys": len(new), "revised_keys": len(revised), "postgame_captured_this_tick": sum(1 for r in pg if r.get("label") == "POSTGAME_TRUTH"),
                                                      "collector_health": OPS.collector_health(C.iso(self.clock()), self.provenance["workflow_event_name"], self.clock(), self.prior_last_invocation), "invocation_provenance": self.provenance})
        if self.persist:
            self.persist("tick status")
        return {"groups": processed, "new_keys": len(new), "revised": len(revised), "postgame": len(pg), "status": ev["status"], "ready": rd["READY"]}

    def run(self, duration_min=25):
        t_end = self.clock() + timedelta(minutes=duration_min)
        outs = []
        while True:
            outs.append(self.tick())
            nxt = next_cutoff(self.st, self.clock())
            if nxt is None or (nxt - self.clock()).total_seconds() > C.WAKE_LEAD_SECONDS or self.clock() >= t_end:
                break
        return outs


def prior_last_invocation(state_root):
    p = Path(state_root) / "status.json"
    try:
        return json.loads(p.read_text()).get("last_invocation_utc") if p.exists() else None
    except Exception:                                                           # noqa
        return None


def watchdog_decision(st, last_invocation_iso, now):
    """Redundant-trigger decision. run only when (a) the primary looks stale (no invocation within WATCHDOG_FRESH_MINUTES) AND (b) an open cutoff lies inside the registered wake window. Pure; creates nothing."""
    gap = (now - C.parse_iso(last_invocation_iso)).total_seconds() / 60 if last_invocation_iso else None
    if gap is not None and gap <= OPS.WATCHDOG_FRESH_MINUTES:
        return {"action": "skip", "reason": "collector_fresh", "minutes_since_last_invocation": round(gap, 2)}
    nxt = next_cutoff(st, now)
    if nxt is None or (nxt - now).total_seconds() > C.WAKE_LEAD_SECONDS:
        return {"action": "skip", "reason": "no_open_cutoff_in_wake_window", "minutes_since_last_invocation": round(gap, 2) if gap is not None else None, "next_cutoff": C.iso(nxt) if nxt else None}
    return {"action": "run", "reason": "collector_stale_and_cutoff_in_wake_window", "minutes_since_last_invocation": round(gap, 2) if gap is not None else None, "next_cutoff": C.iso(nxt)}


def next_cutoff(st, now):
    done = {r["key"] for r in st.results.read()}
    cs = sorted(C.parse_iso(r["cutoff"]) for r in st.planned.read() if r["key"] not in done and C.parse_iso(r["cutoff"]) > now)
    return cs[0] if cs else None


# ------------------------------------------------------------------ status / readiness
def write_status(st, now, ev, rd, extra=None):
    done = {r["key"]: r for r in st.results.read()}
    planned = st.planned.read()
    counts = {}
    for p in planned:
        r = done.get(p["key"])
        s = r["status"] if r else "OPEN"
        counts.setdefault(p["horizon"], {}).setdefault(s, 0)
        counts[p["horizon"]][s] += 1
    sched = st.schedule.read()
    openk = sorted((p for p in planned if p["key"] not in done), key=lambda p: p["cutoff"])[:10]
    stat = {"last_invocation_utc": C.iso(now), "protocol_version": C.PROTOCOL_VERSION, "planned_keys": len(planned), "counts_by_horizon_and_status": counts,
            "next_planned_cutoffs": [{"key": p["key"], "cutoff": p["cutoff"], "horizon": p["horizon"]} for p in openk], "most_recent_schedule_retrieval": sched[-1] if sched else None,
            "postgame_truth_count": ev["games_with_postgame_truth"], "distinct_game_dates_with_truth": ev["distinct_game_dates_with_truth"],
            "minimum_horizon_evaluation_sample_exists": ev["evaluation"] == "EVALUATED", "horizon_status": ev["status"], "recommended_horizon": ev["recommended_horizon"],
            "readiness": {"READY": rd["READY"], "failing": [c["name"] for c in rd["checks"] if not c["ok"]]}, "state_blob_bytes": st.blobs.size_bytes(), **(extra or {})}
    S.write_json_atomic(st.root / "status.json", stat)
    S.write_json_atomic(st.root / "readiness.json", rd)
    S.write_json_atomic(st.root / "evaluation.json", ev)
    return stat


def _chk(name, ok, detail=None):
    return {"name": name, "ok": bool(ok), "detail": detail}


def imports_of(path):
    tree = ast.parse(Path(path).read_text())
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            names.add(n.module.split(".")[0])
    return names


def registered_constants():
    return {"HORIZONS_MINUTES": C.HORIZONS, "CAPTURE_START_LEAD_SECONDS": C.CAPTURE_START_LEAD_SECONDS, "WAKE_LEAD_SECONDS": C.WAKE_LEAD_SECONDS, "VALID_WINDOW_SECONDS": C.VALID_WINDOW_SECONDS, "MAX_WORKERS": C.MAX_WORKERS,
            "REQUIRED_ENDPOINTS": list(C.REQUIRED_ENDPOINTS), "MIN_GAMES": EV.MIN_GAMES, "MIN_DATES": EV.MIN_DATES, "MIN_VALID_PER_HORIZON": EV.MIN_VALID_PER_HORIZON, "MIN_SUCCESS_RATE": EV.MIN_SUCCESS_RATE,
            "MIN_SKATER_RECALL": EV.MIN_SKATER_RECALL, "MIN_SOG_COVERAGE": EV.MIN_SOG_COVERAGE}


def liveness_checks(state_root, clock, prior_last_invocation=None):
    """INFRASTRUCTURE health only; deliberately separate from scientific readiness and never an input to evidence qualification."""
    last = prior_last_invocation if prior_last_invocation is not None else globals()["prior_last_invocation"](state_root)
    h = OPS.collector_health(last, None, clock())
    wd = WATCHDOG_WORKFLOW.read_text() if WATCHDOG_WORKFLOW.exists() else ""
    wf = WORKFLOW.read_text() if WORKFLOW.exists() else ""
    wd_ok = bool(wd) and "group: nhl-forward-snapshot" in wd and "cancel-in-progress: false" in wd and "nhl_fwd_scheduler.py watchdog" in wd
    return [_chk("recent_invocation_within_stale_threshold", h["healthy"], {k: h[k] for k in ("last_invocation_utc", "minutes_since_last_invocation", "stale_after_minutes")}),
            _chk("primary_cron_off_quarter_hour", all(m % 15 != 0 for m in (OPS.cron_minutes(next((l.split('"')[1] for l in wf.splitlines() if l.strip().startswith('- cron')), '')) or [0]))),
            _chk("watchdog_workflow_shares_concurrency_group", wd_ok)]


def readiness(state_root, http, clock, check_schedule=True, prior_last_invocation=None):
    checks = []
    wf = WORKFLOW.read_text() if WORKFLOW.exists() else ""
    body = "\n".join(l for l in wf.splitlines() if not l.strip().startswith("#"))
    checks.append(_chk("workflow_configured", wf and "schedule:" in wf and "workflow_dispatch" in wf and "cancel-in-progress: false" in wf and "concurrency:" in wf and "nhl_fwd_scheduler.py" in wf and "nhl-forward-state" in wf
                       and not any(x in body for x in ("build.py", "nfl_", "cfb_", "mlb_", "nhl_sog", "nhl_serving", "docs/")), {"path": str(WORKFLOW.relative_to(REPO))}))
    try:
        Path(state_root).mkdir(parents=True, exist_ok=True)
        probe = Path(state_root) / ".write_probe"
        probe.write_text("x"); probe.unlink()
        checks.append(_chk("state_writable", True))
    except Exception as e:                                                      # noqa
        checks.append(_chk("state_writable", False, str(e)))
    try:
        bs = S.BlobStore(state_root)
        raw = b"phase0c-roundtrip-" + C.iso(clock()).encode()
        info = bs.put(raw)
        ok = bs.get(info["sha256"]) == raw and S.sha256_hex(raw) == info["sha256"]
        checks.append(_chk("cas_gzip_hash_roundtrip", ok))
    except Exception as e:                                                      # noqa
        checks.append(_chk("cas_gzip_hash_roundtrip", False, str(e)))
    if check_schedule:
        try:
            raw, _h, status = http(f"{C.API}/schedule/now")
            checks.append(_chk("schedule_reachable", status == 200 and bool(C.parse_schedule(raw)) or status == 200, {"http_status": status}))
        except Exception as e:                                                  # noqa
            checks.append(_chk("schedule_reachable", False, f"{type(e).__name__}: {e}"[:150]))
    try:
        contract = json.loads(CONTRACT_FILE.read_text())
        checks.append(_chk("phase0b_contract_version_recognized", contract.get("version") == "nhl-forward-snapshot-contract-2" and contract.get("status") == "BLOCKER_INSUFFICIENT_LIVE_TIMING_EVIDENCE", {"version": contract.get("version")}))
    except Exception as e:                                                      # noqa
        checks.append(_chk("phase0b_contract_version_recognized", False, str(e)[:150]))
    try:
        proto = json.loads(PROTOCOL_FILE.read_text())
        checks.append(_chk("horizon_timing_constants_match_preregistered", proto["constants"] == json.loads(json.dumps(registered_constants())), None))
        checks.append(_chk("capture_protocol_version_matches_registered", proto.get("protocol_version") == C.PROTOCOL_VERSION and proto.get("amendments", [{}])[-1].get("id") == "PRE_LIVE_AMENDMENT_1", {"code": C.PROTOCOL_VERSION, "registered": proto.get("protocol_version")}))
    except Exception as e:                                                      # noqa
        checks.append(_chk("horizon_timing_constants_match_preregistered", False, str(e)[:150]))
    bad = {}
    for f in CODE_FILES:
        p = REPO / f
        if p.exists():
            b = [n for n in imports_of(p) if n.startswith(FORBIDDEN_IMPORT_PREFIXES)]
            if b:
                bad[f] = b
    checks.append(_chk("no_production_or_model_code_invoked", not bad, bad or None))
    sci = all(c["ok"] for c in checks)
    live = liveness_checks(state_root, clock, prior_last_invocation)
    return {"checked_at": C.iso(clock()), "protocol_version": C.PROTOCOL_VERSION, "READY": sci, "SCIENTIFIC_READY": sci, "COLLECTOR_LIVENESS_HEALTHY": all(c["ok"] for c in live), "checks": checks, "liveness_checks": live}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "watchdog", "status", "readiness", "evaluate"])
    ap.add_argument("--state", required=True)
    ap.add_argument("--duration-min", type=float, default=25)
    ap.add_argument("--persist", action="store_true")
    a = ap.parse_args()
    if a.cmd == "status":
        p = Path(a.state) / "status.json"
        print(p.read_text() if p.exists() else json.dumps({"error": "no status.json: the scheduler has never run on this state"})); return
    if a.cmd == "evaluate":
        print(json.dumps(EV.evaluate(C.State(a.state)), indent=1, default=str)); return
    if a.cmd == "readiness":
        r = readiness(a.state, SN.real_http, utcnow)
        S.write_json_atomic(Path(a.state) / "readiness.json", r)
        print(json.dumps(r, indent=1)); sys.exit(0 if r["READY"] else 2)
    persist = (lambda msg: git_persist(a.state, "nhl forward capture: " + msg)) if a.persist else None
    if a.cmd == "watchdog":                                                      # redundant TRIGGER for the same collector: no second collector, no new rules
        d = watchdog_decision(C.State(a.state), prior_last_invocation(a.state), utcnow())
        print(json.dumps({"watchdog": d})); 
        if d["action"] != "run":
            return
    r = Runner(a.state, persist=persist, log=lambda m: print(m, flush=True), provenance=OPS.provenance({**os.environ, "NHL_INVOKER": "watchdog" if a.cmd == "watchdog" else os.environ.get("NHL_INVOKER", "primary")}, utcnow()))
    print(json.dumps(r.run(a.duration_min), default=str))


if __name__ == "__main__":
    main()
