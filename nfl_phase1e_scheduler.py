"""
NFL_PHASE1E_SCHEDULER (Phase 1E, shadow) -- dispatcher that runs the frozen forecast runner at every real cutoff, exactly once per (game, horizon).

    python nfl_phase1e_scheduler.py tick   --root DIR [--n 100000] [--lead-min 20]      one pass (idempotent; safe to call from cron / a loop / after a crash)
    python nfl_phase1e_scheduler.py loop   --root DIR [--every-sec 120]                  tick forever
    python nfl_phase1e_scheduler.py status --root DIR

State (all append-only, under --root):
    dispatch_ledger.jsonl   one row per state transition of a decision key (game_id, horizon, cutoff):
                            PLANNED(first seen) -> STARTED -> DONE | FAILED | MISSED_REAL_CUTOFF
    schedule_probes.jsonl   one row per real schedule retrieval (sha256, retrieval ts, Last-Modified)
The decision key contains the cutoff, so a kickoff revision creates a NEW key; an earlier decision is never edited.

Rules (protocol Amendment G, schedule module):
  * the real schedule is retrieved every tick; kickoff / cutoff come from THAT snapshot; cutoff = kickoff - 24h / - 90m
  * a (game, horizon) whose cutoff has passed without a retrieved snapshot is logged MISSED_REAL_CUTOFF (terminal) -- never backfilled, never backdated
  * execution starts when 0 < cutoff - now <= lead; groups with the same kickoff run together (one snapshot set)
  * restart safety: a STARTED key whose snapshot set already exists reuses it (identical bytes -> identical forecast ids -> 0 new forecasts; a different byte -> HardError)
  * a lagging provider is retried at the next tick until the cutoff, then FAILED(provider_lag) -- no data is invented
Shadow only: nothing here touches production serving.

Deployment (cron, every 5 minutes; the lock makes overlapping invocations safe):
    */5 * * * *  cd /path/to/repo && python3 nfl_phase1e_scheduler.py tick --root /var/nfl_shadow --n 100000 >> /var/nfl_shadow/dispatch.log 2>&1
or a supervised daemon:
    python3 nfl_phase1e_scheduler.py loop --root /var/nfl_shadow --n 100000
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nfl_phase1_store as ST
import nfl_phase1_store_lock as LK
import nfl_phase1d_cas as CAS
import nfl_phase1d_schedule as SCH

UTC = timezone.utc
SCHEDULER_VERSION = "phase1e-scheduler-3"
TERMINAL = ("DONE", "PARTIAL_V2_MISSING", "FAILED", "MISSED_REAL_CUTOFF")
DEFAULT_LEAD_MIN = 20
PREFIT_GUARD = timedelta(minutes=50)           # a prefit (~20 min) starts only when no cutoff is due within this window
RETRYABLE_UNTIL_KICKOFF = ("prefit_missing", "prefit_hash_mismatch", "prefit_code_identity_mismatch", "stored_snapshot_invalid")
WAIT_STEP_S = 30


def utcnow():
    return datetime.now(UTC)


def key_of(gid, hz, cutoff):
    return f"{gid}|{hz}|{SCH.iso(cutoff)}"


class Dispatcher:
    def __init__(self, root, n_draws, lead_min=DEFAULT_LEAD_MIN, runner=None, fetch_schedule=None, clock=utcnow, log=print, horizons=("T24", "T90"), sleep=time.sleep, capture_start_lead_s=240):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.n_draws, self.lead, self.log, self.clock, self.horizons = n_draws, timedelta(minutes=lead_min), log, clock, horizons
        self.sleep, self.capture_lead = sleep, timedelta(seconds=capture_start_lead_s)     # waking early (lead) is separate from capturing (capture_lead)
        self.ledger = CAS.Ledger(self.root, name="dispatch_ledger.jsonl")
        self.probes = CAS.Ledger(self.root, name="schedule_probes.jsonl")
        self.lock = LK.StoreLock(self.root / "dispatcher.lock", timeout=1, bind_dir=self.root)    # one dispatcher at a time; a second tick exits immediately
        self._runner = runner
        self.fetch_schedule = fetch_schedule or self._fetch_schedule

    @property
    def runner(self):
        if self._runner is None:
            import nfl_phase1e_live as LVE
            self._runner = LVE.LiveRunner(self.root, self.n_draws, cas_root=self.root / "cas", log=self.log)
        return self._runner

    # ---------------------------------------------------------------- schedule
    def _fetch_schedule(self):
        import nfl_phase1e_live as LVE
        raw, lm, _ = LVE.http_get(CAS.provider_url("games.csv"))
        return raw, lm

    def probe_schedule(self):
        raw, lm = self.fetch_schedule()
        now = self.clock()
        store = CAS.BlobStore(self.root / "cas")
        stored, raw_sha, _ = CAS.sanitize_schedule(raw)
        info = store.put(stored)
        self.probes.append_many([{"retrieval_ts": SCH.iso(now), "sha256": info["sha256"], "raw_sha256": raw_sha, "last_modified": lm, "scheduler_version": SCHEDULER_VERSION}])
        return stored, now

    # ---------------------------------------------------------------- state
    def states(self):
        st = {}
        for r in self.ledger.read():
            st[r["key"]] = r                    # last row wins; rows are never edited
        return st

    def record(self, key, gid, hz, cutoff, kick, state, **kw):
        row = {"key": key, "game_id": gid, "horizon": hz, "cutoff": SCH.iso(cutoff), "kickoff": SCH.iso(kick), "state": state, "at": SCH.iso(self.clock()), "scheduler_version": SCHEDULER_VERSION, **kw}
        self.ledger.append_many([row])
        return row

    # ---------------------------------------------------------------- one pass
    def tick(self):
        try:
            self.lock.acquire()
        except LK.LockError:
            return {"skipped": "another dispatcher holds the lock"}
        try:
            out, sched, now = self._tick()
            self.last_sched = sched
            return out
        finally:
            self.lock.release()

    def _tick(self):
        sched_bytes, now = self.probe_schedule()
        sched = SCH.parse_schedule(sched_bytes)
        st = self.states()
        out = {"now": SCH.iso(now), "planned": 0, "missed": 0, "started": 0, "done": 0, "failed": 0, "waiting": 0, "expired": 0}
        out["expired"] += self.finalize_expired(st, now)
        st = self.states()
        due = defaultdict(list)                  # (season, week, hz, kick) -> [game ids]
        for gid, g in sorted(sched.items()):
            if g["kick"] <= now:
                continue                         # the game has started: no pregame decision can exist any more (finalize_expired closes its open keys)
            for hz in self.horizons:
                cutoff = SCH.forecast_cutoff(g["kick"], hz)
                key = key_of(gid, hz, cutoff)       # the key contains the cutoff: a kickoff revision is a NEW key; old keys are never edited
                cur = st.get(key)
                if cur and cur["state"] in TERMINAL:
                    continue
                if cur is None:
                    self.record(key, gid, hz, cutoff, g["kick"], "PLANNED"); out["planned"] += 1
                s_exist = self.runner.existing_set(self.runner.group_name(g["season"], g["week"], hz, g["kick"])) if (now >= cutoff or cutoff - now <= self.lead) else None
                if now >= cutoff:
                    if s_exist is None:
                        if cur and cur["state"] == "STARTED" and str(cur.get("reason", "")).startswith("prefit_"):
                            self.record(key, gid, hz, cutoff, g["kick"], "FAILED", reason="PREFIT_NOT_READY_AT_CUTOFF: no valid pre-cutoff prefit when the cutoff was reached (" + str(cur.get("reason"))[:200] + "); never forecast later from a post-cutoff artifact")
                            out["failed"] += 1; self.log(f"PREFIT_NOT_READY_AT_CUTOFF {key}")
                            continue
                        self.record(key, gid, hz, cutoff, g["kick"], "MISSED_REAL_CUTOFF", reason="cutoff passed before any pre-cutoff snapshot of this key existed (never backfilled)")
                        out["missed"] += 1; self.log(f"MISSED_REAL_CUTOFF {key}")
                        continue
                    due[(g["season"], g["week"], hz, g["kick"])].append(gid)       # restart: the stored pre-cutoff snapshot is reused (same bytes, same identity)
                elif cutoff - now <= self.lead or s_exist is not None:
                    due[(g["season"], g["week"], hz, g["kick"])].append(gid)
                else:
                    out["waiting"] += 1
        for (season, week, hz, kick), gids in sorted(due.items(), key=lambda x: (SCH.forecast_cutoff(x[0][3], x[0][2]), x[0][3])):
            cutoff = SCH.forecast_cutoff(kick, hz)
            if self.runner.existing_set(self.runner.group_name(season, week, hz, kick)) is None and self.clock() < cutoff - self.capture_lead:
                self.wait_until(cutoff - self.capture_lead)         # woke early; the horizon snapshot is captured only inside the pre-registered capture window
            for gid in gids:
                self.record(key_of(gid, hz, cutoff), gid, hz, cutoff, kick, "STARTED")
            out["started"] += len(gids)
            run_id = f"live-{SCH.iso(self.clock())}"
            try:
                rec, how, logs = self.runner.run_group(season, week, hz, kick, gids, run_id)
            except CAS.CASError as e:
                msg = str(e)
                if msg.startswith("MISSED_REAL_CUTOFF"):
                    state = "MISSED_REAL_CUTOFF"
                elif msg.startswith("PREFIT_NOT_READY_AT_CUTOFF"):
                    state = "FAILED"                  # terminal: a post-cutoff prefit can never promote this key
                elif msg.startswith(RETRYABLE_UNTIL_KICKOFF):
                    state = None                  # operational, retryable until kickoff; resumes from the stored snapshot (or is closed by finalize_expired)
                elif self.clock() >= cutoff or not msg.startswith(("provider_lag", "early_snapshot_rejected")):
                    state = "FAILED"
                else:
                    state = None                  # provider lag before the cutoff: retry at the next tick
                for gid in gids:
                    self.record(key_of(gid, hz, cutoff), gid, hz, cutoff, kick, state or "STARTED", reason=msg[:500])
                out["failed" if state == "FAILED" else "missed" if state else "waiting"] += len(gids)
                self.log(f"{hz} {gids}: {msg[:300]}")
                continue
            by = {l["game_id"]: l for l in logs}
            for gid in gids:
                l = by.get(gid, {"status": "SAFE_EXPLICIT_FAILURE", "reason": "no status row"})
                ok = l.get("status") == "FORECAST_SUCCESS"
                joint = ok and l.get("joint_ready", True)
                self.record(key_of(gid, hz, cutoff), gid, hz, cutoff, kick, "DONE" if joint else "PARTIAL_V2_MISSING" if ok else "FAILED", snapshot_set_id=rec["set_id"], snapshot_how=how,
                            retrieval_ts=rec["retrieval_ts"], status=l.get("status"), reason=l.get("reason"), n_records=l.get("n_records"), written=l.get("written"), verified_duplicates=l.get("verified_duplicates"), v2=l.get("v2"))
                out["done" if joint else "failed"] += 1
        return out, sched, now

    def wait_until(self, t):
        """Wait inside the durable job until the pre-registered capture start. Injected clock / sleep make this testable without real waiting."""
        while True:
            n = self.clock()
            if n >= t:
                return
            self.sleep(min((t - n).total_seconds(), WAIT_STEP_S))

    def finalize_expired(self, st, now):
        """Close every non-terminal key whose game has already kicked off: with a stored snapshot -> FAILED (could not finish before kickoff), without -> MISSED_REAL_CUTOFF. Never produces a forecast."""
        n = 0
        for key, r in sorted(st.items()):
            if r["state"] in TERMINAL or SCH.parse_iso(r["kickoff"]) > now:
                continue
            gname = None
            has_set = False
            try:
                for rr in CAS.Ledger(self.root / "cas", name="manifest.jsonl").read():
                    if rr.get("type") == "set" and rr["group"].startswith("LIVE_") and rr["group"].endswith(f"_{r['horizon']}_{r['kickoff']}"):
                        has_set = True
            except CAS.CASError:
                pass
            self.record(key, r["game_id"], r["horizon"], SCH.parse_iso(r["cutoff"]), SCH.parse_iso(r["kickoff"]), "FAILED" if has_set else "MISSED_REAL_CUTOFF",
                        reason="kickoff reached before the decision completed" + (" (a valid pre-cutoff snapshot existed)" if has_set else " (no pre-cutoff snapshot; never backfilled)"))
            n += 1
        return n

    def maybe_prefit(self, sched, now):
        """Run the (expensive) prefit for the nearest week lacking one, ONLY if prerequisites are met and no cutoff is due within PREFIT_GUARD. Never at a cutoff."""
        import nfl_phase1e_ops as OPS
        for gid, g in sorted(sched.items()):                   # guard evaluated directly from the REAL schedule, not from dispatch-ledger rows
            for hz in self.horizons:
                c = SCH.forecast_cutoff(g["kick"], hz)
                if now <= c <= now + PREFIT_GUARD:
                    return {"prefit": "deferred: a real cutoff is within the guard window", "cutoff": SCH.iso(c), "game_id": gid, "horizon": hz}
        weeks = sorted({(g["season"], g["week"]) for g in sched.values() if now < g["kick"] <= now + timedelta(days=5)})
        for (s, w) in weeks:
            if OPS.prefit_row(self.root, s, w) is None:
                try:
                    row, how = OPS.run_prefit(self.runner, s, w, now_fn=self.clock, log=self.log)
                    return {"prefit": how, "season": s, "week": w, "artifact_bundle_sha256": row["artifact_bundle_sha256"]}
                except CAS.CASError as e:
                    return {"prefit": "not_ready", "season": s, "week": w, "reason": str(e)[:300]}
        return {"prefit": "none_needed"}

    def prefit_only(self):
        """Prefit-only path: store the schedule provenance, run the (guarded) prefit, write status. Dispatches NOTHING: no key transition, no live snapshot, no forecast / v2 / event write."""
        try:
            self.lock.acquire()
        except LK.LockError:
            return {"skipped": "another dispatcher holds the lock"}
        try:
            sched_bytes, now = self.probe_schedule()
            sched = SCH.parse_schedule(sched_bytes)
            extra = self.maybe_prefit(sched, now)
            self.heartbeat(sched, extra)
            return {"mode": "prefit-only", **extra}
        finally:
            self.lock.release()

    def run(self, duration_min=0, every_sec=60, do_prefit=True):
        """One workflow invocation: tick (repeatedly for `duration_min`), then prefit if allowed, then readiness + heartbeat. No daemon: all state is in the append-only files."""
        import nfl_phase1e_ops as OPS
        t_end = self.clock() + timedelta(minutes=duration_min)
        outs = []
        while True:
            r = self.tick()
            outs.append(r)
            if self.clock() + timedelta(seconds=every_sec) >= t_end:
                break
            self.sleep(every_sec)
        extra = {}
        sched, now = None, self.clock()
        probes = self.probes.read()
        if do_prefit and probes:
            stored = CAS.BlobStore(self.root / "cas").get(probes[-1]["sha256"])
            sched = SCH.parse_schedule(stored)
            extra = self.maybe_prefit(sched, now)
        self.heartbeat(sched, extra)
        return {"ticks": outs[-3:], "n_ticks": len(outs), **extra}

    def heartbeat(self, sched=None, extra=None):
        import nfl_phase1e_ops as OPS
        now = self.clock()
        if sched is None:
            probes = self.probes.read()
            sched = SCH.parse_schedule(CAS.BlobStore(self.root / "cas").get(probes[-1]["sha256"])) if probes else {}
        rd = OPS.readiness(self.root, sched, now)
        return OPS.write_status(self.root, now, self.states(), sched, SCHEDULER_VERSION, rd, extra)


def preflight(root, game_id, horizon, sample_dir, reference_root=None, fetch_schedule=None, head=True, cas_root=None):
    """No-write preflight of one game-horizon: both paths must resolve the game from the SAME schedule bytes, and everything the v2 comparator needs must exist BEFORE Phase 1 runs."""
    import hashlib
    import tempfile
    import urllib.request
    import nfl_phase1e_live as LVE
    import nfl_phase1e_sources as SRC
    import nfl_phase1e_v2 as V2
    root = Path(root)
    out = {"game_id": game_id, "horizon": horizon, "checked_at": SCH.iso(utcnow()), "problems": []}
    P = out["problems"]
    raw, lm = (fetch_schedule or Dispatcher(root, 0, runner=object())._fetch_schedule)()
    san, raw_sha, _ = CAS.sanitize_schedule(raw)
    out["schedule"] = {"raw_sha256": hashlib.sha256(raw).hexdigest(), "phase1_stored_sha256": hashlib.sha256(san).hexdigest(), "provider_last_modified": lm,
                       "sanitize_is_deterministic": CAS.sanitize_schedule(raw)[0] == san, "phase1_schedule_has_market_columns": b"spread_line" in san}
    p1 = SCH.parse_schedule(san).get(game_id)
    v2v = V2.schedule_view(raw).get(game_id)
    if p1 is None or v2v is None:
        P.append("game absent from the schedule on the " + ("phase1" if p1 is None else "v2") + " side")
    else:
        cut = SCH.forecast_cutoff(p1["kick"], horizon)
        v2cut = SCH.forecast_cutoff(SCH.parse_iso(v2v[2]), horizon)
        same = {"game_id": True, "home": p1["home"] == v2v[0], "away": p1["away"] == v2v[1], "kickoff": SCH.iso(p1["kick"]) == v2v[2], "cutoff": cut == v2cut}
        out["shared"] = {"game_id": game_id, "home": p1["home"], "away": p1["away"], "kickoff": SCH.iso(p1["kick"]), "cutoff": SCH.iso(cut), "same_on_both_paths": same,
                         "both_derive_from_one_retrieval": "LiveRunner.snapshot_live stores sanitized bytes (Phase 1 CAS) and the raw bytes (v2 raw store) from ONE download; provider_lag row links group -> raw sha"}
        if not all(same.values()):
            P.append(f"schedule resolution differs between Phase 1 and v2: {same}")
        out["cutoff_in_future"] = utcnow() < cut
        if not out["cutoff_in_future"]:
            P.append("cutoff already passed (MISSED_REAL_CUTOFF)")
    # v2 requirements
    v2r = {"frozen_artifacts_match_manifest": False, "raw_store_writable": os.access(root, os.W_OK) or not root.exists(), "provenance_ledger_writable": os.access(root, os.W_OK) or not root.exists()}
    try:
        man = json.loads((V2.P1E / "v2_comparator_manifest.json").read_text())
        fr = V2.Frozen()
        v2r["frozen_artifacts_match_manifest"] = all(fr.hashes[oc]["artifact_sha256"] == man["artifacts"][oc]["sha256"] and fr.hashes[oc]["residual_sha256"] == man["artifacts"][oc]["residual_sha256"] for oc in V2.MARKETS) and man["code_hash"] == V2.code_hash()
        v2r["comparator_artifact_sha256"] = {oc: fr.hashes[oc]["artifact_sha256"] for oc in V2.MARKETS}
        v2r["production_artifact_context_present"] = all(oc in fr.prod for oc in V2.MARKETS)
        if sample_dir and p1:
            td = Path(tempfile.mkdtemp(prefix="v2pre_"))
            for f in Path(sample_dir).iterdir():
                if f.name != "games.csv":
                    (td / f.name).symlink_to(f.resolve())
            (td / "games.csv").write_bytes(raw)
            recs = V2.build_records(fr, str(td), p1["season"], p1["week"], [game_id], horizon, {game_id: SCH.iso(p1["kick"])}, {game_id: SCH.iso(cut)}, {"retrieval_ts": SCH.iso(utcnow()), "input_hashes": {}})
            v2r["dry_build_records"] = {"n_records": len(recs), "n_eligible": sum(r["eligibility"] == "eligible" for r in recs), "dry_run_not_written": True}
            v2r["dry_build_ok"] = v2r["dry_build_records"]["n_eligible"] > 0
        else:
            v2r["dry_build_ok"] = False
    except Exception as e:                                       # noqa
        v2r["error"] = f"{type(e).__name__}: {e}"
    out["v2"] = v2r
    v2_ok = all(v2r.get(k) for k in ("frozen_artifacts_match_manifest", "raw_store_writable", "provenance_ledger_writable", "dry_build_ok")) and out.get("shared", {}).get("same_on_both_paths") and all(out["shared"]["same_on_both_paths"].values())
    # phase 1 requirements
    p1r = {"registry_sources": SRC.build_registry()["n_sources"], "required_sources_reachable": None}
    if head:
        bad = []
        for s_ in SRC.build_registry()["sources"]:
            if s_["required_at_T90"]:
                try:
                    req = urllib.request.Request(s_["endpoint"], method="HEAD", headers={"User-Agent": "nfl-phase1-snapshots"})
                    urllib.request.urlopen(req, timeout=60).close()
                except Exception as e:                           # noqa
                    bad.append(f"{s_['logical_name']}: {e}")
        p1r["required_sources_reachable"] = not bad
        p1r["unreachable"] = bad
    cas_root = Path(cas_root) if cas_root else root / "cas"
    p1r["cas_writable"] = os.access(cas_root if cas_root.exists() else root, os.W_OK)
    mv = None
    if reference_root:
        recs = [r for r in ST.Store(reference_root, "forecasts").all_records() if r["game_id"] == game_id and r["horizon"] == horizon]
        mv = recs[0]["model_version"] if recs else None
    p1r["model_version"] = mv
    p1r["model_version_source"] = "dry-run record on real data at this code SHA (n_draws inside the version = 50000 for that run); the live value is fixed when the live root fits the weekly artifact" if mv else None
    p1r["weekly_artifact_cached_in_live_root"] = bool(CAS.Ledger(root, name="weekly_fits.jsonl").read())
    try:
        aud = json.loads((V2.P1E / "simulation_n_final_audit.json").read_text())
        nf = aud.get("selected_N")
        if nf is None and aud.get("R11") == "BLOCKER":
            p1r_unbound = "UNBOUND_R11_BLOCKER (no N passed C1-C4; the live rehearsal uses --n as an operational N only)"
    except Exception:                                            # noqa
        nf = None
    p1r["final_N_status"] = str(nf) if nf else (locals().get("p1r_unbound") or "PENDING")
    p1_ok = bool(out.get("shared")) and all(out["shared"]["same_on_both_paths"].values()) and p1r["required_sources_reachable"] is not False and p1r["cas_writable"] and out.get("cutoff_in_future", False)
    out["phase1"] = p1r
    out["PHASE1_READY"], out["V2_READY"] = bool(p1_ok), bool(v2_ok)
    out["JOINT_READY"] = LVE.joint_ready(p1_ok, "V2_LOGGED" if v2_ok else "V2_FAILED")
    out["FINAL_N"] = p1r["final_N_status"]
    out["note"] = "no store was written; the daemon must not be started while FINAL_N is PENDING"
    return out


def status(root):
    led = CAS.Ledger(root, name="dispatch_ledger.jsonl").read()
    last = {}
    for r in led:
        last[r["key"]] = r
    cnt = defaultdict(int)
    for r in last.values():
        cnt[r["state"]] += 1
    return {"keys": len(last), "by_state": dict(cnt), "rows": len(led)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tick", "loop", "status", "preflight", "run", "readiness", "prefit"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--lead-min", type=int, default=DEFAULT_LEAD_MIN)
    ap.add_argument("--every-sec", type=int, default=120)
    ap.add_argument("--duration-min", type=float, default=0)
    ap.add_argument("--game"); ap.add_argument("--horizon", default="T90"); ap.add_argument("--sample-dir"); ap.add_argument("--reference-root")
    a = ap.parse_args()
    if a.cmd == "preflight":
        r = preflight(a.root, a.game, a.horizon, a.sample_dir, a.reference_root)
        print(json.dumps(r, indent=1, default=str)); return
    if a.cmd == "status":
        sp = Path(a.root) / "status.json"
        print(sp.read_text() if sp.exists() else json.dumps({"error": "no status.json: the scheduler has never run on this root", **status(a.root)}, indent=1)); return
    if a.cmd == "readiness":
        import nfl_phase1e_ops as OPS
        d = Dispatcher(a.root, a.n, a.lead_min, log=lambda m: print(m, flush=True))
        raw, _lm = d._fetch_schedule()
        san, _, _ = CAS.sanitize_schedule(raw)
        r = OPS.readiness(a.root, SCH.parse_schedule(san), utcnow(), deep=True)
        print(json.dumps(r, indent=1)); sys.exit(0 if r["READY"] else 2)
    if a.cmd == "prefit":
        d = Dispatcher(a.root, a.n, a.lead_min, log=lambda m: print(m, flush=True))
        print(json.dumps(d.prefit_only(), default=str)); return
    if a.cmd == "run":
        d = Dispatcher(a.root, a.n, a.lead_min, log=lambda m: print(m, flush=True))
        print(json.dumps(d.run(a.duration_min, a.every_sec), default=str)); return
    d = Dispatcher(a.root, a.n, a.lead_min, log=lambda m: print(m, flush=True))
    if a.cmd == "tick":
        print(json.dumps(d.tick()))
    else:
        while True:
            try:
                print(json.dumps(d.tick()), flush=True)
            except Exception as e:                       # never die: the next tick retries from the persisted state
                print(f"tick error {type(e).__name__}: {e}", flush=True)
            time.sleep(a.every_sec)


if __name__ == "__main__":
    main()
