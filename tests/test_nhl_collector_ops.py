"""NHL collector operational-liveness tests: off-quarter-hour cron, provenance, collector_health, scientific-vs-liveness readiness, watchdog decisions, duplicate-invocation safety, no early / no backfill evidence.
Infrastructure only: asserts that none of it changes evidence. python tests/test_nhl_collector_ops.py"""
import hashlib
import json
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tests"))
import nhl_fwd_capture as C  # noqa: E402
import nhl_fwd_ops as OPS  # noqa: E402
import nhl_fwd_scheduler as SC  # noqa: E402
import test_nhl_phase0c as T  # noqa: E402
import yaml  # noqa: E402

START = T.START
UTC = T.UTC


def evidence_hash(root):
    h = hashlib.sha256()
    for n in ("planned.jsonl", "observations.jsonl", "results.jsonl", "postgame.jsonl"):
        p = Path(root) / n
        h.update(p.read_bytes() if p.exists() else b"-")
    return h.hexdigest()


def runner_at(t, clk, w, prov=None):
    sleeps = []

    def sleep(sec):
        sleeps.append(sec); clk.advance(sec)
    return SC.Runner(t, http=w.http, clock=clk, sleep=sleep, map_fn=T.seq_map, log=lambda m: None, provenance=prov), sleeps


def test_primary_cron_is_off_the_contended_marks_and_keeps_a_15_minute_cadence():
    d = yaml.safe_load((REPO / ".github/workflows/nhl_forward_snapshot.yml").read_text())
    cron = (d.get("on") or d.get(True))["schedule"][0]["cron"]
    mins = OPS.cron_minutes(cron)
    assert mins == [7, 22, 37, 52] and not set(mins) & {0, 15, 30, 45} and [b - a for a, b in zip(mins, mins[1:])] == [15, 15, 15] and 60 - mins[-1] + mins[0] == 15
    wd = yaml.safe_load((REPO / ".github/workflows/nhl_forward_watchdog.yml").read_text())
    wc = OPS.cron_minutes((wd.get("on") or wd.get(True))["schedule"][0]["cron"])
    assert wc == [11, 26, 41, 56] and not set(wc) & {0, 15, 30, 45} and not set(wc) & set(mins)


def test_watchdog_workflow_is_the_same_collector_not_a_second_one():
    raw = (REPO / ".github/workflows/nhl_forward_watchdog.yml").read_text()
    d = yaml.safe_load(raw)
    p = yaml.safe_load((REPO / ".github/workflows/nhl_forward_snapshot.yml").read_text())
    assert d["concurrency"] == p["concurrency"] == {"group": "nhl-forward-snapshot", "cancel-in-progress": False}
    body = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("#"))
    assert "nhl_fwd_scheduler.py watchdog" in body and "STATE_BRANCH: nhl-forward-state" in body and "pushed=0" in body and "--persist" in body
    for other in ("build.py", "nfl_", "cfb_", "mlb_", "nhl_sog", "nhl_serving", "nhl_goals", "docs/", "pip install", "xgboost", "predict"):
        assert other not in body, other
    assert not [n for n in SC.imports_of(REPO / "nhl_fwd_ops.py") if n.startswith(SC.FORBIDDEN_IMPORT_PREFIXES)] and "nhl_fwd_ops.py" in SC.CODE_FILES


def test_intended_slot_and_provenance_are_operational_only():
    now = T.datetime(2026, 10, 3, 10, 32, 1, tzinfo=UTC)
    assert OPS.intended_slot("7,22,37,52 * * * *", now) == T.datetime(2026, 10, 3, 10, 22, tzinfo=UTC)
    p = OPS.provenance({"NHL_EVENT_NAME": "schedule", "NHL_RUN_ID": "123", "NHL_CRON": "7,22,37,52 * * * *"}, now)
    assert p["workflow_event_name"] == "schedule" and p["run_id"] == "123" and p["intended_slot_utc"].startswith("2026-10-03T10:22") and p["delay_from_intended_slot_min"] == 10.02 and p["scheduled_event_timestamp_available"] is False
    assert OPS.provenance({"NHL_EVENT_NAME": "workflow_dispatch"}, now)["intended_slot_utc"] is None


def test_collector_health_pure_function_stale_threshold_and_historical_gap():
    now = T.datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
    fresh = OPS.collector_health(OPS.iso(now - timedelta(minutes=10)), "schedule", now)
    stale = OPS.collector_health(OPS.iso(now - timedelta(minutes=36)), "schedule", now)
    assert fresh["healthy"] is True and stale["healthy"] is False and fresh["stale_after_minutes"] == 35
    assert OPS.collector_health(None, None, now)["healthy"] is False
    h = OPS.collector_health(OPS.iso(now), "schedule", now, previous_invocation_iso=OPS.iso(now - timedelta(minutes=300)))
    assert h["healthy"] is True and h["gap_before_this_invocation_min"] == 300.0 and h["gap_before_this_invocation_exceeded_stale_threshold"] is True


def test_status_carries_health_and_provenance_and_readiness_separates_scientific_from_liveness():
    with tempfile.TemporaryDirectory() as t:
        clk, w, r, _ = T.setup(t, {2026020300: {"start": START}}, START - timedelta(days=3))
        r.tick()
        st = json.loads((Path(t) / "status.json").read_text())
        assert st["collector_health"]["healthy"] is True and st["collector_health"]["last_event_type"] == "local" and st["invocation_provenance"]["invoker"] == "primary"
        rd = json.loads((Path(t) / "readiness.json").read_text())
        assert rd["READY"] is True and rd["SCIENTIFIC_READY"] is True and "COLLECTOR_LIVENESS_HEALTHY" in rd and "liveness_checks" in rd
        assert len(OPS.read_ops(t)) == 1
        clk.advance(40 * 60)                                                                    # nobody invoked for 40 min: infrastructure unhealthy, science untouched
        rd2 = SC.readiness(t, w.http, clk)
        assert rd2["SCIENTIFIC_READY"] is True and rd2["READY"] is True and rd2["COLLECTOR_LIVENESS_HEALTHY"] is False
        assert not any(c["name"] in ("recent_invocation_within_stale_threshold", "watchdog_workflow_shares_concurrency_group") for c in rd2["checks"])   # liveness is never a scientific check
        w.fail.add(f"{T.API}/schedule/now")
        rd3 = SC.readiness(t, w.http, clk)
        assert rd3["SCIENTIFIC_READY"] is False and rd3["READY"] is False                         # scientific failures still fail READY


def test_stale_period_never_rewrites_old_evidence_and_ops_fields_do_not_change_evidence():
    games = {2026020301: {"start": START}}
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        for root, prov in ((a, None), (b, OPS.provenance({"NHL_EVENT_NAME": "schedule", "NHL_RUN_ID": "9", "NHL_CRON": "7,22,37,52 * * * *", "NHL_INVOKER": "watchdog"}, START - timedelta(days=3)))):
            clk, w, _, _ = T.setup(root, games, START - timedelta(days=3))
            r, _s = runner_at(root, clk, w, prov)
            r.tick()
        assert evidence_hash(a) == evidence_hash(b)                                             # provenance is not a scientific feature
        before = evidence_hash(a)
        clk, w, _, _ = T.setup(a, games, START - timedelta(days=3, minutes=-300))                  # 5 h later, collector was dark
        r, _s = runner_at(a, clk, w); r.tick()
        st = json.loads((Path(a) / "status.json").read_text())
        assert st["collector_health"]["gap_before_this_invocation_exceeded_stale_threshold"] is True and st["collector_health"]["healthy"] is True
        assert evidence_hash(a) == before                                                       # historical outage is exposed, evidence is not rewritten


def test_watchdog_skips_when_fresh_or_no_cutoff_and_runs_only_when_stale_with_cutoff_in_wake_window():
    with tempfile.TemporaryDirectory() as t:
        clk, w, r, _ = T.setup(t, {2026020302: {"start": START}}, START - timedelta(days=3))
        r.tick()
        st, now = r.st, clk()
        last = json.loads((Path(t) / "status.json").read_text())["last_invocation_utc"]
        assert SC.watchdog_decision(st, last, now) == {"action": "skip", "reason": "collector_fresh", "minutes_since_last_invocation": 0.0}
        stale_now = now + timedelta(minutes=20)
        d = SC.watchdog_decision(st, last, stale_now)
        assert d["action"] == "skip" and d["reason"] == "no_open_cutoff_in_wake_window"                    # stale, but the next cutoff (T24H) is days away
        cutoff = SC.next_cutoff(st, stale_now)
        at = cutoff - timedelta(seconds=C.WAKE_LEAD_SECONDS - 60)
        d = SC.watchdog_decision(st, last, at)
        assert d["action"] == "run" and d["reason"] == "collector_stale_and_cutoff_in_wake_window"
        assert SC.watchdog_decision(st, last, cutoff - timedelta(seconds=C.WAKE_LEAD_SECONDS + 60))["action"] == "skip"
        assert SC.watchdog_decision(st, None, at)["action"] == "run"                                       # never invoked + cutoff approaching


def test_primary_then_watchdog_cannot_duplicate_evidence_and_neither_captures_early_nor_backfills():
    games = {2026020303: {"start": START}}
    with tempfile.TemporaryDirectory() as t:
        clk, w, _, _ = T.setup(t, games, START - timedelta(minutes=1440 + 19))                    # first cutoff (T24H) 19 min away: inside the wake window
        r1, s1 = runner_at(t, clk, w, OPS.provenance({"NHL_EVENT_NAME": "schedule", "NHL_INVOKER": "primary", "NHL_CRON": "7,22,37,52 * * * *"}, clk()))
        cutoff = START - timedelta(minutes=1440)
        r1.tick()                                                                               # tick waits inside the job until cutoff-120s (never earlier), then captures
        assert any(sec > 0 for sec in s1) and clk() >= cutoff - timedelta(seconds=C.CAPTURE_START_LEAD_SECONDS)
        r1.run(duration_min=25)
        obs1 = [o for o in r1.st.obs.read() if o["horizon"] == "T24H"]
        assert obs1 and all(C.parse_iso(o["started_at"]) >= cutoff - timedelta(seconds=C.CAPTURE_START_LEAD_SECONDS) for o in obs1 if "started_at" in o)
        counts = (len(r1.st.planned.read()), len(r1.st.obs.read()), len(r1.st.results.read()))
        h = evidence_hash(t)
        r2, _s = runner_at(t, clk, w, OPS.provenance({"NHL_EVENT_NAME": "schedule", "NHL_INVOKER": "watchdog"}, clk()))
        r2.run(duration_min=25)                                                                 # watchdog duplicate invocation afterwards
        r2.tick()
        assert (len(r2.st.planned.read()), len(r2.st.obs.read()), len(r2.st.results.read())) == counts and evidence_hash(t) == h
        assert {o["invoker"] for o in OPS.read_ops(t)} == {"primary", "watchdog"}


def test_cutoff_missed_by_a_dark_collector_is_recorded_missed_never_backfilled():
    games = {2026020304: {"start": START}}
    with tempfile.TemporaryDirectory() as t:
        clk, w, r, _ = T.setup(t, games, START - timedelta(minutes=1440 + 30))
        r.tick()                                                                                # plans keys; collector then goes dark through the T24H cutoff
        clk.advance(40 * 60)
        r2, _s = runner_at(t, clk, w); r2.tick()
        res = {x["key"]: x for x in r2.st.results.read() if x["horizon"] == "T24H"}
        assert res and all(x["status"] == C.MISSED for x in res.values())
        assert not [o for o in r2.st.obs.read() if o["horizon"] == "T24H"]                      # no late capture relabelled as horizon evidence


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
