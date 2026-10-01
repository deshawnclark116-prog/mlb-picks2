"""
Phase 1E tests (blocker clearance: N audit, real-provider live operation, v2 forward logging). Standalone and pytest-compatible.

  python -u tests/test_nfl_phase1e.py

Unit tests use synthetic schedules / fake runners (no network). Result tests assert on the recorded Phase 1E evidence files.
"""
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nfl_phase1_store as ST  # noqa: E402
import nfl_phase1d_cas as CAS  # noqa: E402
import nfl_phase1d_schedule as SCH  # noqa: E402
import nfl_phase1e_live as LVE  # noqa: E402
import nfl_phase1e_scheduler as DS  # noqa: E402
import nfl_phase1e_sources as SRC  # noqa: E402
import nfl_phase1e_v2 as V2  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
P1E = REPO / "nfl_models" / "nfl_player_outcome_phase1e"
UTC = timezone.utc
KICK = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)            # 13:00 ET


def sched_bytes(kick=KICK, gid="2026_04_AAA_BBB"):
    et = kick.astimezone(SCH.P1.ET)
    rows = [{"game_id": gid, "season": "2026", "game_type": "REG", "week": "4", "gameday": et.strftime("%Y-%m-%d"), "gametime": et.strftime("%H:%M"), "away_team": "AAA", "home_team": "BBB",
             "result": "", "spread_line": "-3.5", "total_line": "44.5"}]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    return buf.getvalue().encode()


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class FakeRunner:
    def __init__(self, root):
        self.calls, self.sets = [], {}
        self.crash_once = False

    def group_name(self, season, week, hz, kick):
        return f"G_{season}_{week}_{hz}_{SCH.iso(kick)}"

    def existing_set(self, group):
        return self.sets.get(group)

    def run_group(self, season, week, hz, kick, gids, run_id):
        group = self.group_name(season, week, hz, kick)
        self.calls.append((hz, tuple(gids)))
        if group not in self.sets:
            self.sets[group] = {"set_id": "S" + group[-8:], "retrieval_ts": SCH.iso(SCH.forecast_cutoff(kick, hz) - timedelta(minutes=5))}
        if self.crash_once:
            self.crash_once = False
            raise RuntimeError("simulated crash after the snapshot was stored")
        return self.sets[group], "retrieved_now", [{"game_id": g, "status": "FORECAST_SUCCESS", "n_records": 3, "written": 3} for g in gids]


def make(tmp, clock, sched=None, runner=None):
    sb = [sched or sched_bytes()]
    d = DS.Dispatcher(tmp, 1000, runner=runner or FakeRunner(tmp), fetch_schedule=lambda: (sb[0], "Thu, 01 Oct 2026 00:00:00 GMT"), clock=clock, log=lambda m: None)
    return d, sb


def states(d):
    return {k: v["state"] for k, v in d.states().items()}


# ------------------------------------------------------------------ scheduler
def test_dispatch_exactly_once_per_game_and_horizon():
    with tempfile.TemporaryDirectory() as t:
        clk = Clock(KICK - timedelta(hours=30))
        d, _ = make(t, clk)
        d.tick(); assert d.runner.calls == []                                                   # nothing due: T24 cutoff is 6h away
        clk.t = KICK - timedelta(hours=24) - timedelta(minutes=10)
        d.tick(); assert d.runner.calls == [("T24", ("2026_04_AAA_BBB",))]
        d.tick(); d.tick(); assert len(d.runner.calls) == 1                                    # same key is terminal: not run again
        clk.t = KICK - timedelta(minutes=100)
        d.tick(); assert d.runner.calls[-1][0] == "T90" and len(d.runner.calls) == 2
        clk.t = KICK - timedelta(minutes=95); d.tick(); assert len(d.runner.calls) == 2
        assert sorted(states(d).values()) == ["DONE", "DONE"]
        # a NEW dispatcher on the same root (process restart) does not repeat anything
        d2, _ = make(t, clk, runner=FakeRunner(t)); d2.tick(); assert d2.runner.calls == []


def test_missed_real_cutoff_is_logged_never_backfilled():
    with tempfile.TemporaryDirectory() as t:
        clk = Clock(KICK - timedelta(hours=20))                                                  # T24 cutoff already passed, game not started
        d, _ = make(t, clk)
        d.tick()
        st = states(d)
        t24 = [k for k in st if "|T24|" in k][0]
        assert st[t24] == "MISSED_REAL_CUTOFF" and d.runner.calls == []
        clk.t = KICK - timedelta(hours=10); d.tick(); assert d.runner.calls == [] and states(d)[t24] == "MISSED_REAL_CUTOFF"    # never revived, never backfilled
        clk.t = KICK - timedelta(minutes=100); d.tick()
        assert d.runner.calls == [("T90", ("2026_04_AAA_BBB",))]
        clk.t = KICK + timedelta(minutes=1); d.tick()
        assert len(d.runner.calls) == 1                                                          # a started game gets no pregame decision


def test_restart_after_crash_reuses_stored_snapshot():
    with tempfile.TemporaryDirectory() as t:
        clk = Clock(KICK - timedelta(hours=24) - timedelta(minutes=10))
        r = FakeRunner(t); r.crash_once = True
        d, _ = make(t, clk, runner=r)
        try:
            d.tick(); assert False
        except RuntimeError:
            pass
        assert [v for k, v in states(d).items() if "|T24|" in k] == ["STARTED"]
        clk.t = KICK - timedelta(hours=24) + timedelta(minutes=1)                                # the cutoff has passed but the pre-cutoff snapshot exists
        d2, _ = make(t, clk, runner=r); d2.tick()
        assert [v for k, v in states(d2).items() if "|T24|" in k] == ["DONE"]
        assert len(r.sets) == 1 and len(r.calls) == 2                                            # one snapshot set for the key, resumed not re-retrieved


def test_kickoff_revision_creates_a_new_decision_key():
    with tempfile.TemporaryDirectory() as t:
        clk = Clock(KICK - timedelta(hours=40))
        d, sb = make(t, clk)
        d.tick(); k1 = set(states(d))
        sb[0] = sched_bytes(KICK + timedelta(hours=3))                                           # flexed later
        d.tick(); k2 = set(states(d))
        assert k2 > k1 and len(k2) == 4                                                          # 2 horizons x 2 kickoffs; old rows never edited
        rows = d.ledger.read()
        assert all(r["state"] == "PLANNED" for r in rows)


def test_second_dispatcher_cannot_run_concurrently():
    with tempfile.TemporaryDirectory() as t:
        clk = Clock(KICK - timedelta(hours=40))
        a, _ = make(t, clk); b, _ = make(t, clk)
        a.lock.acquire()
        try:
            assert "skipped" in b.tick()
        finally:
            a.lock.release()


# ------------------------------------------------------------------ live runner pieces
def test_provider_lag_policy():
    stats = b"player_id,season,week,team\nx,2026,3,AAA\ny,2026,3,BBB\n"
    sched = SCH.parse_schedule(b"game_id,season,game_type,week,gameday,gametime,away_team,home_team\n2026_03_AAA_BBB,2026,REG,3,2026-09-20,13:00,AAA,BBB\n2026_03_CCC_DDD,2026,REG,3,2026-09-20,13:00,CCC,DDD\n")
    cutoff = datetime(2026, 9, 30, tzinfo=UTC)
    miss = LVE.provider_lag_problems({"stats_player_week_2026.csv": stats}, sched, cutoff, 2026)
    assert sorted(miss) == ["2026 wk3 CCC", "2026 wk3 DDD"]
    assert LVE.provider_lag_problems({"stats_player_week_2026.csv": stats}, {k: v for k, v in sched.items() if "AAA" in k}, cutoff, 2026) == []
    # a game that is not yet complete by the cutoff is not required
    assert LVE.provider_lag_problems({"stats_player_week_2026.csv": stats}, sched, datetime(2026, 9, 20, 18, tzinfo=UTC), 2026) == []


def test_guarded_store_refuses_after_kickoff():
    with tempfile.TemporaryDirectory() as t:
        inner = ST.Store(t, "forecasts")
        g = LVE.GuardedStore(inner, {"G": datetime(2000, 1, 1, tzinfo=UTC)})
        try:
            g.append_batch("b", {"game_id": "G"}, [{"id": "x"}]); assert False
        except LVE.LateGeneration:
            pass
        g2 = LVE.GuardedStore(inner, {"G": datetime(2999, 1, 1, tzinfo=UTC)})
        assert g2.append_batch("b", {"game_id": "G"}, [{"id": "x", "v": 1}])["written"] == 1


def test_live_snapshot_never_after_cutoff_and_reuse():
    class R(LVE.LiveRunner):
        def __init__(self, root, fetcher):
            super().__init__(root, 1000, cas_root=Path(root) / "cas", fetcher=fetcher)
    with tempfile.TemporaryDirectory() as t:
        past_kick = datetime.now(UTC) + timedelta(minutes=30)                                     # T90 cutoff is already past
        fetched = []
        r = R(t, lambda: fetched.append(1))
        try:
            r.snapshot_live(2026, 4, "T90", past_kick, ["g"]); assert False
        except CAS.CASError as e:
            assert str(e).startswith("MISSED_REAL_CUTOFF")
        assert fetched == []                                                                       # no retrieval is even attempted after the cutoff


# ------------------------------------------------------------------ source registry
def test_registry_covers_every_live_source_with_policy():
    reg = SRC.build_registry()
    names = {s["logical_name"] for s in reg["sources"]}
    assert names == set(CAS.logical_files())
    for s in reg["sources"]:
        for k in ("provider", "endpoint", "expected_schema_required_columns", "parser_version", "cadence", "availability_lag", "required_at_T24", "required_at_T90", "failure_policy"):
            assert k in s and s[k] not in (None, ""), (s["logical_name"], k)
        assert s["endpoint"] == CAS.provider_url(s["logical_name"]) and s["endpoint"].startswith("https://github.com/nflverse/nflverse-data/releases/download/")
        assert (not s["required_at_T24"]) == s["logical_name"].startswith(CAS.OPTIONAL_LIVE)
    assert reg["no_silent_substitution"] is True


# ------------------------------------------------------------------ v2 logger
def v2rec(i, pt=10.0):
    return {"id": f"v2-{i}", "game_id": "G", "player_id": f"P{i}", "outcome": "rush_yds", "horizon": "T24", "cutoff": "2026-10-03T17:00:00.000000Z", "kind": "v2_forward_forecast",
            "v2_frozen_comparator": {"point": pt}, "v2_live_production": {"point": pt + 1}}


def test_v2_store_idempotent_and_conflict_is_hard_error():
    with tempfile.TemporaryDirectory() as t:
        st = ST.Store(t, "v2forecasts")
        recs = [v2rec(i) for i in range(3)]
        assert st.append_batch("b", {"generated_at": "a"}, recs)["written"] == 3
        r2 = st.append_batch("b", {"generated_at": "LATER HEADER"}, recs)
        assert r2["written"] == 0 and r2["verified_duplicates"] == 3                                   # same id, same bytes
        try:
            st.append_batch("b", {}, [v2rec(1, pt=99.0)]); assert False
        except ST.HardError:
            pass
        assert len(st.all_records()) == 3


def test_v2_matching_is_deterministic_and_order_independent():
    p = [{"game_id": "G", "player_id": f"P{i}", "outcome": "rush_yds", "horizon": "T24", "cutoff": "c", "id": f"p{i}"} for i in range(5)] + [{"game_id": "G", "player_id": "PX", "outcome": "rec_yds", "horizon": "T24", "cutoff": "c", "id": "pX"}]
    v = [{**v2rec(i), "cutoff": "c"} for i in range(0, 4)] + [{**v2rec(9), "cutoff": "c"}]
    v.append({"game_id": "G", "player_id": "PX", "outcome": "rec_yds", "horizon": "T24", "cutoff": "c", "kind": "v2_forward_abstention", "abstain_reason": "min_history", "id": "va"})
    a = V2.match(p, v)
    b = V2.match(list(reversed(p)), list(reversed(v)))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    pairs, p_only, v_only = a
    assert [x["key"][1] for x in pairs] == ["P0", "P1", "P2", "P3"]
    assert {x["key"][1]: x["reason"] for x in p_only} == {"P4": "player_absent_from_v2_candidate_set", "PX": "v2_abstained: min_history"}
    assert [x["key"][1] for x in v_only] == ["P9"]
    assert set(pairs[0]) >= {"phase1_forecast", "v2_frozen_comparator", "v2_live_production"}


def test_v2_phase1_snapshot_never_contains_sportsbook_columns():
    out, raw_sha, tr = CAS.sanitize_schedule(sched_bytes())
    assert b"spread_line" not in out and b"total_line" not in out and raw_sha == hashlib.sha256(sched_bytes()).hexdigest()


def test_v2_frozen_artifacts_match_manifest():
    m = json.loads((P1E / "v2_comparator_manifest.json").read_text())
    for oc, a in m["artifacts"].items():
        assert hashlib.sha256((REPO / a["file"]).read_bytes()).hexdigest() == a["sha256"]
        assert hashlib.sha256((REPO / a["residual_file"]).read_bytes()).hexdigest() == a["residual_sha256"]
    assert m["code_hash"] == V2.code_hash() and m["no_future_refitting"] is True
    assert set(m["comparator_policy"]) >= {"phase1_forecast", "v2_frozen_comparator", "v2_live_production"}


# ------------------------------------------------------------------ evidence (result) tests
def jl(p):
    return json.loads((P1E / p).read_text())


def test_plan_committed_before_any_result():
    log = subprocess.check_output(["git", "log", "--reverse", "--format=%H", "--", "nfl_models/nfl_player_outcome_phase1e/blocker_clearance_plan.json"], cwd=REPO, text=True).split()
    first_plan = log[0]
    others = subprocess.check_output(["git", "log", "--reverse", "--format=%H", "--", "nfl_models/nfl_player_outcome_phase1e"], cwd=REPO, text=True).split()
    assert others[0] == first_plan, "the first Phase 1E commit must be the plan"


def test_n_audit_is_consistent_with_the_registered_rule():
    a = jl("simulation_n_final_audit.json")
    crit = json.loads((REPO / "nfl_models/nfl_player_outcome_phase1d/n_engineering_criteria.json").read_text())
    assert a["criteria_file"].endswith("n_engineering_criteria.json") and a["reference_N"] == crit["reference_N"]
    runs = a["candidates"]
    assert [r["N"] for r in runs][0] == 50000
    sel = a["selected_N"]
    passing = [r["N"] for r in runs if all(r["verdict"][c] for c in ("C1", "C2", "C3", "C4"))]
    assert (sel == min(passing)) if passing else (sel is None)
    if 50000 in passing:
        assert len(runs) == 1                                                                        # 50k passes -> stop, 100k not run
    else:
        assert [r["N"] for r in runs] == [50000, 100000]
    for r in runs:
        assert r["peak_rss_mb"] and r["seconds"] and len(r["repeat_design"]["plan"]) == 5


def test_real_provider_fetch_audit():
    a = jl("real_provider_fetch_audit.json")
    assert a["uses_local_mirror"] is False and a["uses_cached_dev_file"] is False
    assert a["required_sources_all_ok"] and a["acceptance"]["all_schema_ok"] and a["acceptance"]["every_blob_roundtrip_verified"]
    for r in a["results"]:
        if not r["optional"]:
            assert r["status"] == "OK" and r["raw_sha256"] and r["stored_sha256"] and r["retrieved_at_utc"] and r["url"].startswith("https://github.com/nflverse/")


def test_v2_time_travel_equivalence_evidence():
    e = jl("v2_time_travel_equivalence.json")
    assert e["pass"] is True and len(e["weeks"]) >= 4
    for w in e["weeks"].values():
        for oc in ("rush_yds", "rec_yds"):
            assert w[oc]["all_within_tol"] and w[oc]["n_unexplained_missing"] == 0 and w[oc]["max_abs_diff"] <= 1e-9
    assert e["perturbation"]["prior_history_change"]["changed"]


def test_no_final_freeze_file():
    assert not (REPO / "nfl_models" / "nfl_player_outcome_phase1_freeze.json").exists()


if __name__ == "__main__":
    only = sys.argv[1:]
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and (not only or n in only):
            try:
                f(); print("PASS", n, flush=True)
            except Exception as e:  # noqa
                import traceback; traceback.print_exc()
                fails += 1; print("FAIL", n, repr(e), flush=True)
    sys.exit(1 if fails else 0)
