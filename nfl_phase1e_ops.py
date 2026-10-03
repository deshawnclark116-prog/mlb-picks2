"""
NFL_PHASE1E_OPS -- operational layer for durable shadow dispatch: immutable prefit artifacts, per-event provenance records, readiness preflight, heartbeat / status.
No predictive content: nothing here changes a model, feature, calibration, eligibility rule or simulation. Shadow only.
"""
import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nfl_phase1_data as P1
import nfl_phase1_forecast as FC
import nfl_phase1_store as ST
import nfl_phase1d_cas as CAS
import nfl_phase1d_p1a as P
import nfl_phase1d_runner as RN
import nfl_phase1d_schedule as SCH

REPO = Path(__file__).resolve().parent
UTC = timezone.utc
WORKFLOW = REPO / ".github" / "workflows" / "nfl_phase1e_shadow.yml"
OPS_VERSION = "phase1e-ops-1"


def utcnow():
    return datetime.now(UTC)


def canon_sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


# ------------------------------------------------------------------ prefit artifacts
def code_identity():
    """Deterministic identifier of the forecast code + frozen hyper-parameter files that determine what a weekly artifact means (no predictive change: it only NAMES the code state)."""
    import nfl_phase1d_live as LV
    return canon_sha({"code_hashes": FC.file_hashes(), "p1b_hyper": hashlib.sha256(LV.P1B_HYPER.read_bytes()).hexdigest(), "p1a_frozen": P.load_frozen() and hashlib.sha256(json.dumps(P.load_frozen(), sort_keys=True, default=str).encode()).hexdigest()})


def prefit_ledger(root):
    return CAS.Ledger(root, name="prefit_ledger.jsonl")


def prefit_row(root, season, week):
    rows = [r for r in prefit_ledger(root).read() if r["season"] == season and r["week"] == week]
    return rows[-1] if rows else None


def load_verified_prefit(root, row):
    """Load the artifact bundle and verify every blob and the bundle hash against the prefit record, and that the record was made by the CURRENT frozen forecast code / hyper-parameters.
    Mismatch -> CASError('prefit_hash_mismatch' / 'prefit_code_identity_mismatch')."""
    if row.get("code_identity") != code_identity():
        raise CAS.CASError("prefit_code_identity_mismatch: the prefit was produced by different frozen forecast code / hyper-parameters than the current code")
    path = Path(root) / "artifacts" / row["artifact_bundle_sha256"]
    try:
        art = P.Artifacts.load(path)
    except Exception as e:                                                 # noqa
        raise CAS.CASError(f"prefit_hash_mismatch: {type(e).__name__}: {e}")
    if art.manifest()["bundle_sha256"] != row["artifact_bundle_sha256"]:
        raise CAS.CASError("prefit_hash_mismatch: bundle hash differs from the prefit record")
    return art


def prefit_timing_ok(row, cutoff):
    """The prefit used for a forecast must have existed BEFORE that exact cutoff: created_at <= cutoff AND its source snapshot retrieved <= cutoff (and the record must be self-consistent: created_at >= fit retrieval)."""
    try:
        created = SCH.parse_iso(row["created_at"])
        fit_ts = SCH.parse_iso(row["source_identity"]["fit_retrieval_ts"])
    except Exception:                                                          # noqa
        return False, "prefit record lacks created_at / source_identity.fit_retrieval_ts"
    if created < fit_ts:
        return False, f"prefit_record_inconsistent: created_at {row['created_at']} precedes its source retrieval {row['source_identity']['fit_retrieval_ts']}"
    if created > cutoff:
        return False, f"prefit created_at {row['created_at']} is after the cutoff {SCH.iso(cutoff)}"
    if fit_ts > cutoff:
        return False, f"prefit source snapshot retrieved {SCH.iso(fit_ts)} is after the cutoff {SCH.iso(cutoff)}"
    return True, None


def prefit_gate(root, season, week, cutoff, now):
    """Return the verified prefit row usable for a forecast with this cutoff, or raise CASError.
    prefit_* (retryable): none usable yet and the cutoff has not been reached.  PREFIT_NOT_READY_AT_CUTOFF (terminal): the cutoff was reached (or the record is timing-invalid) without a valid pre-cutoff prefit.
    A post-cutoff prefit is never used for this key and is never backdated."""
    rows = [r for r in prefit_ledger(root).read() if r["season"] == season and r["week"] == week]
    if not rows:
        if now >= cutoff:
            raise CAS.CASError(f"PREFIT_NOT_READY_AT_CUTOFF: no prefit existed when the cutoff {SCH.iso(cutoff)} was reached")
        raise CAS.CASError(f"prefit_missing: no prefit artifact for season {season} week {week} (run the prefit step; a cutoff forecast never fits)")
    good, why = [], []
    for r in rows:
        ok, reason = prefit_timing_ok(r, cutoff)
        (good if ok else why).append(r if ok else reason)
    if not good:
        raise CAS.CASError("PREFIT_NOT_READY_AT_CUTOFF: no prefit record satisfies created_at <= cutoff and source retrieval <= cutoff: " + "; ".join(why[:3]))
    row = good[-1]
    try:
        load_verified_prefit(root, row)
    except CAS.CASError as e:
        if now >= cutoff:
            raise CAS.CASError(f"PREFIT_NOT_READY_AT_CUTOFF: the pre-cutoff prefit is not usable at the cutoff ({e})")
        raise
    return row


def prefit_prereq_problems(stats_bytes, sched_games, season, week, now):
    """The weekly artifact trains on COMPLETED weeks before the target week only. Every game of earlier weeks whose kickoff+24h has passed must be present in the provider's stats file."""
    import csv, io
    have = {(r["week"], r["team"]) for r in csv.DictReader(io.StringIO(stats_bytes.decode("utf-8")))}
    miss = []
    for g in sched_games.values():
        if g["season"] == season and g["week"] < week and g["kick"] + timedelta(hours=24) <= now:
            for tm in (g["home"], g["away"]):
                if (str(g["week"]), tm) not in have:
                    miss.append(f"{season} wk{g['week']} {tm}")
    return miss


def run_prefit(runner, season, week, now_fn=utcnow, log=print):
    """Create (idempotently) the immutable prefit artifact for (season, week) from a REAL snapshot taken now. Not a forecast; the snapshot set is labelled PREFIT."""
    import nfl_phase1d_live as LV
    import nfl_phase1e_live as LVE
    row = prefit_row(runner.root, season, week)
    if row:
        load_verified_prefit(runner.root, row)
        return row, "existing"
    got, audit, unavailable, _raw = runner.fetcher()
    retrieval = now_fn()
    sched = SCH.parse_schedule(got["games.csv"][0] if isinstance(got["games.csv"], tuple) else got["games.csv"], seasons={season})
    week_games = {g: v for g, v in sched.items() if v["week"] == week}
    if not week_games:
        raise CAS.CASError(f"prefit: no games for season {season} week {week} in the schedule")
    miss = prefit_prereq_problems(got[f"stats_player_week_{season}.csv"], sched, season, week, retrieval)
    if miss:
        raise CAS.CASError("prefit_prereq_incomplete: completed earlier-week games missing from provider data: " + "; ".join(miss[:6]))
    first_kick = min(v["kick"] for v in week_games.values())
    srcs = {n: (CAS.provider_id(n), (lambda b=b: b)) for n, b in got.items()}
    for n, msg in unavailable.items():
        srcs[n] = (CAS.provider_id(n), (lambda m=msg: (_ for _ in ()).throw(CAS.Unavailable(m))))
    group = f"PREFIT_{season}_{week:02d}_{SCH.iso(retrieval)}"
    rec = CAS.take_snapshot_set(runner.store, runner.ledger, srcs, "PREFIT", first_kick, retrieval, retrieval, group,
                                note="PREFIT snapshot of the real provider (training data identity); not a forecast", time_travel=False)
    asof = CAS.materialize(runner.store, rec, runner.root / "tmp" / f"prefit_{rec['content_id'][:16]}")
    targets = sorted({(season, week, t) for v in week_games.values() for t in (v["home"], v["away"])})
    D = P1.Data(str(asof))
    U = P1.build(D, depth_universe=True, targets=targets)
    art = RN.Runner.weekly_artifacts(runner, D, U, (season, week), rec["content_id"])          # unchanged Phase 1D walk-forward refit; cached per (season, week)
    man = art.manifest()
    wf = [r for r in runner.wf_ledger.read() if r["season"] == season and r["week"] == week][-1]
    row = {"season": season, "week": week, "prefit_version": OPS_VERSION, "code_identity": code_identity(), "code_sha": FC.git_sha(), "training_cutoff_last_completed_week": wf.get("training_cutoff_last_week"),
           "source_identity": {"fit_snapshot_set_id": rec["set_id"], "fit_snapshot_content_id": rec["content_id"], "fit_retrieval_ts": rec["retrieval_ts"], "fit_set_content_id_in_weekly_ledger": wf.get("fit_set_content_id")},
           "artifact_bundle_sha256": man["bundle_sha256"], "artifact_dir": f"artifacts/{man['bundle_sha256']}", "created_at": SCH.iso(now_fn()), "fit_seconds": wf.get("fit_seconds"),
           "note": "weekly walk-forward artifact trained on completed weeks before the target week; independent of N and of the cutoff"}
    load_verified_prefit(runner.root, row)
    prefit_ledger(runner.root).append_many([row])
    return row, "created"


# ------------------------------------------------------------------ per-event provenance
def live_events(root):
    return CAS.Ledger(root, name="live_events.jsonl")


def record_live_event(runner, rec, how, hz, kick, log):
    """One append-only provenance row per (game, horizon): intended horizon, kickoff, cutoff, actual retrieval timestamp, delta, schedule identity, prefit / v2 identity. Same key + same snapshot -> no new row;
    same key + a DIFFERENT snapshot -> HardError (a live decision is never replaced)."""
    gid = log["game_id"]
    cutoff = SCH.forecast_cutoff(kick, hz)
    retrieval = SCH.parse_iso(rec["retrieval_ts"])
    lag = [r for r in runner.lag.read() if r["group"] == runner.group_name(log["season"], log["week"], hz, kick) and r.get("raw_schedule_sha256")]
    pre = getattr(runner, "prefit_used", None) or {}
    row = {"key": f"{gid}|{hz}|{SCH.iso(cutoff)}", "game_id": gid, "intended_horizon": hz, "kickoff": SCH.iso(kick), "cutoff": SCH.iso(cutoff), "retrieval_ts": rec["retrieval_ts"],
           "retrieval_minus_cutoff_seconds": (retrieval - cutoff).total_seconds(), "seconds_before_cutoff": (cutoff - retrieval).total_seconds(), "retrieved_before_cutoff": retrieval <= cutoff,
           "effective_minutes_before_kickoff": round((kick - retrieval).total_seconds() / 60, 3),
           "clean_forward_timing_ok": bool(retrieval <= cutoff and (cutoff - retrieval) <= timedelta(minutes=5) and pre.get("created_at") and SCH.parse_iso(pre["created_at"]) <= cutoff),
           "snapshot_how": how, "schedule_snapshot_sha256": rec["files"].get("games.csv"), "schedule_raw_sha256": lag[-1]["raw_schedule_sha256"] if lag else None, "snapshot_set_id": rec["set_id"], "snapshot_content_id": rec["content_id"],
           "prefit_artifact_sha256": pre.get("artifact_bundle_sha256"), "prefit_created_at": pre.get("created_at"), "prefit_code_identity": pre.get("code_identity"), "model_version": log.get("model_version"),
           "phase1_status": log.get("status"), "phase1_n_records": log.get("n_records"), "v2": log.get("v2"), "joint_ready": bool(log.get("joint_ready")), "n_draws_operational": runner.n_draws,
           "label": "LIVE shadow event. Temporal evidence: clean-forward iff clean_forward_timing_ok (retrieved within 5 min before the real cutoff, prefit existed before the cutoff). Operational N only (R11 blocker): not freeze evidence, not production-promotion evidence", "ops_version": OPS_VERSION}
    led = live_events(runner.root)
    prev = [r for r in led.read() if r["key"] == row["key"]]
    if prev:
        if prev[-1]["snapshot_set_id"] != row["snapshot_set_id"]:
            raise ST.HardError(f"live event {row['key']} already recorded with a different snapshot ({prev[-1]['snapshot_set_id']} vs {row['snapshot_set_id']})")
        return prev[-1], False
    led.append_many([row])
    return row, True


# ------------------------------------------------------------------ readiness
def _check(name, ok, detail=None):
    return {"name": name, "ok": bool(ok), "detail": detail}


def workflow_config():
    if not WORKFLOW.exists():
        return {"exists": False}
    t = WORKFLOW.read_text()
    return {"exists": True, "schedule": "schedule:" in t and "cron:" in t, "workflow_dispatch": "workflow_dispatch" in t, "concurrency": "concurrency:" in t and "cancel-in-progress: false" in t,
            "invokes_scheduler": "nfl_phase1e_scheduler.py" in t, "touches_other_sports": any(x in t for x in ("build.py", "cfb_", "nhl_", "mlb_", "nfl_serving_builder")), "path": str(WORKFLOW.relative_to(REPO))}


PREFIT_LOOKAHEAD = timedelta(days=5)   # THE prefit eligibility window: weeks with kickoff in (now, now + PREFIT_LOOKAHEAD]. Shared by maybe_prefit() and readiness() so they cannot drift.


def prefit_window_weeks(upcoming, now):
    """Split upcoming schedule weeks into (required_now, future_not_due): {(season, week): [game ids]} each. A week is prefit-actionable iff it has a kickoff within PREFIT_LOOKAHEAD."""
    required, future = {}, {}
    for gid, g in (upcoming or {}).items():
        if g["kick"] <= now:
            continue
        (required if g["kick"] - now <= PREFIT_LOOKAHEAD else future).setdefault((g["season"], g["week"]), []).append(gid)
    for wk in list(future):                                                   # a week with any game inside the window is required now
        if wk in required:
            del future[wk]
    return required, future


def readiness(root, upcoming=None, now=None, deep=False):
    """Concise machine-readable readiness. `upcoming` = {game_id: {season, week, kick}} (from the already snapshotted schedule); no network unless deep=True."""
    import nfl_phase1e_scheduler as DS
    import nfl_phase1e_sources as SRC
    import nfl_phase1e_v2 as V2
    root, now = Path(root), now or utcnow()
    checks = []
    wf = workflow_config()
    checks.append(_check("workflow_configured", wf.get("exists") and wf.get("schedule") and wf.get("workflow_dispatch") and wf.get("concurrency") and wf.get("invokes_scheduler") and not wf.get("touches_other_sports"), wf))
    reg = SRC.build_registry()
    checks.append(_check("source_registry_complete", reg["n_sources"] == len(CAS.logical_files()) and reg["no_silent_substitution"] is True and all(s["endpoint"].startswith("https://github.com/nflverse/") for s in reg["sources"]), {"n_sources": reg["n_sources"]}))
    try:
        root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=root, prefix=".w"); os.close(fd); os.unlink(tmp)
        checks.append(_check("store_writable", True, str(root)))
    except Exception as e:                                                    # noqa
        checks.append(_check("store_writable", False, str(e)))
    # schedule resolvable + prefit per week that is ACTIONABLE NOW (same window as maybe_prefit); later weeks are visible but non-blocking
    weeks, future = prefit_window_weeks(upcoming, now)
    checks.append(_check("schedule_resolvable", bool(upcoming), {"n_upcoming_games": len(upcoming or {}), "prefit_required_now": sorted(f"{s}-wk{w}" for s, w in weeks),
                                                                "future_prefit_not_due": sorted(f"{s}-wk{w}" for s, w in future), "prefit_lookahead_days": PREFIT_LOOKAHEAD.days}))
    pf = {}
    for (s, w), gids in sorted(weeks.items()):
        row = prefit_row(root, s, w)
        if row is None:
            pf[f"{s}-wk{w}"] = {"ok": False, "status": "prefit_required_now", "reason": "no prefit artifact"}
            continue
        try:
            load_verified_prefit(root, row); pf[f"{s}-wk{w}"] = {"ok": True, "status": "prefit_required_now", "artifact_bundle_sha256": row["artifact_bundle_sha256"], "created_at": row["created_at"], "training_cutoff_last_completed_week": row["training_cutoff_last_completed_week"]}
        except Exception as e:                                                # noqa
            pf[f"{s}-wk{w}"] = {"ok": False, "status": "prefit_required_now", "reason": str(e)[:200]}
    informational = {f"{s}-wk{w}": {"ok": None, "status": "future_prefit_not_due", "informational": True} for (s, w) in future}
    checks.append(_check("prefit_artifacts_verified", all(v["ok"] for v in pf.values()), {**pf, **informational} if (pf or informational) else {"note": "no upcoming week"}))
    try:
        RN.load_config(); import nfl_phase1d_live as LV; LV.load_p1b_hyper(); P.load_frozen()
        checks.append(_check("phase1_runner_ready", True, None))
    except Exception as e:                                                    # noqa
        checks.append(_check("phase1_runner_ready", False, f"{type(e).__name__}: {e}"))
    try:
        man = json.loads((V2.P1E / "v2_comparator_manifest.json").read_text())
        fr = V2.Frozen()
        ok = all(fr.hashes[oc]["artifact_sha256"] == man["artifacts"][oc]["sha256"] and fr.hashes[oc]["residual_sha256"] == man["artifacts"][oc]["residual_sha256"] for oc in V2.MARKETS) and man["code_hash"] == V2.code_hash()
        checks.append(_check("frozen_v2_ready", ok, {"comparator_version": V2.COMPARATOR_VERSION, "artifact_sha256": {oc: fr.hashes[oc]["artifact_sha256"] for oc in V2.MARKETS}}))
    except Exception as e:                                                    # noqa
        checks.append(_check("frozen_v2_ready", False, f"{type(e).__name__}: {e}"))
    if deep:
        import urllib.request
        bad = []
        for s_ in reg["sources"]:
            if s_["required_at_T90"]:
                try:
                    urllib.request.urlopen(urllib.request.Request(s_["endpoint"], method="HEAD", headers={"User-Agent": "nfl-phase1-snapshots"}), timeout=60).close()
                except Exception as e:                                        # noqa
                    bad.append(f"{s_['logical_name']}: {e}")
        checks.append(_check("required_sources_reachable", not bad, bad[:5]))
    return {"checked_at": SCH.iso(now), "ops_version": OPS_VERSION, "READY": all(c["ok"] for c in checks), "checks": checks, "note": "readiness is operational only; R11 (simulation N) remains a BLOCKER and nothing here is freeze evidence"}


# ------------------------------------------------------------------ heartbeat / status
def write_status(root, now, states, upcoming, scheduler_version, readiness_result=None, extra=None):
    root = Path(root)
    last = {}
    for r in states.values():
        last[r["state"]] = last.get(r["state"], 0) + 1
    nxt = sorted((r for r in states.values() if r["state"] not in DS_TERMINAL), key=lambda r: r["cutoff"])[:10]
    lag = CAS.Ledger(root, name="provider_lag.jsonl").read() if (root / "provider_lag.jsonl").exists() else []
    ok_lag = [r for r in lag if not r.get("provider_lag_missing")]
    probes = CAS.Ledger(root, name="schedule_probes.jsonl").read() if (root / "schedule_probes.jsonl").exists() else []
    st = {"last_invocation_utc": SCH.iso(now), "scheduler_version": scheduler_version, "ops_version": OPS_VERSION, "planned_keys": len(states), "counts_by_state": last,
          "next_cutoffs": [{"key": r["key"], "cutoff": r["cutoff"], "kickoff": r["kickoff"], "state": r["state"]} for r in nxt],
          "last_successful_provider_retrieval_utc": ok_lag[-1]["retrieval_ts"] if ok_lag else None, "last_schedule_probe_utc": probes[-1]["retrieval_ts"] if probes else None,
          "readiness": {"READY": readiness_result["READY"], "checked_at": readiness_result["checked_at"], "failing": [c["name"] for c in readiness_result["checks"] if not c["ok"]]} if readiness_result else None, **(extra or {})}
    tmp = root / ".status.tmp"
    tmp.write_text(json.dumps(st, indent=1, sort_keys=True))
    os.replace(tmp, root / "status.json")
    if readiness_result:
        tmp = root / ".readiness.tmp"; tmp.write_text(json.dumps(readiness_result, indent=1, sort_keys=True)); os.replace(tmp, root / "readiness.json")
    return st


DS_TERMINAL = ("DONE", "PARTIAL_V2_MISSING", "FAILED", "MISSED_REAL_CUTOFF")
