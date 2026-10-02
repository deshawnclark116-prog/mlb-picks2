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
import nfl_phase1e_ops as OPS  # noqa: E402
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


def test_week4_pit_at_cle_real_cutoffs():
    """Real 2026 wk4 Thursday game (provider row: gameday 2026-10-01, gametime 20:15 America/New_York): T24 = 2026-10-01 00:15Z, T90 = 2026-10-01 22:45Z."""
    raw = b"game_id,season,game_type,week,gameday,gametime,away_team,home_team,result\n2026_04_PIT_CLE,2026,REG,4,2026-10-01,20:15,PIT,CLE,\n"
    g = SCH.parse_schedule(raw)["2026_04_PIT_CLE"]
    assert SCH.iso(g["kick"]) == "2026-10-02T00:15:00.000000Z"
    assert SCH.iso(SCH.forecast_cutoff(g["kick"], "T24")) == "2026-10-01T00:15:00.000000Z"
    assert SCH.iso(SCH.forecast_cutoff(g["kick"], "T90")) == "2026-10-01T22:45:00.000000Z"
    with tempfile.TemporaryDirectory() as t:
        clk = Clock(datetime(2026, 10, 1, 2, 36, tzinfo=UTC))            # the moment the correction was requested (2026-09-30 22:36 ET)
        d = DS.Dispatcher(t, 1000, runner=FakeRunner(t), fetch_schedule=lambda: (raw, "x"), clock=clk, log=lambda m: None)
        d.tick()
        st = states(d)
        assert st["2026_04_PIT_CLE|T24|2026-10-01T00:15:00.000000Z"] == "MISSED_REAL_CUTOFF"
        assert st["2026_04_PIT_CLE|T90|2026-10-01T22:45:00.000000Z"] == "PLANNED" and d.runner.calls == []
        clk.t = datetime(2026, 10, 1, 22, 30, tzinfo=UTC)               # 15 min before the T90 cutoff: due
        d.tick()
        assert d.runner.calls == [("T90", ("2026_04_PIT_CLE",))]


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


def test_phase1_cannot_be_fully_ready_without_v2_provenance():
    assert LVE.joint_ready(True, "V2_LOGGED") is True
    assert LVE.joint_ready(True, "V2_FAILED") is False and LVE.joint_ready(True, None) is False and LVE.joint_ready(False, "V2_LOGGED") is False

    class Partial(FakeRunner):                                      # Phase 1 succeeds, v2 lacks its shared schedule provenance
        def run_group(self, season, week, hz, kick, gids, run_id):
            rec, how, logs = super().run_group(season, week, hz, kick, gids, run_id)
            for l in logs:
                l["v2"] = {"status": "V2_FAILED", "reason": "no provider-lag row (raw schedule hash) for the group"}
                l["joint_ready"] = LVE.joint_ready(True, "V2_FAILED")
            return rec, how, logs
    with tempfile.TemporaryDirectory() as t:
        clk = Clock(KICK - timedelta(hours=24) - timedelta(minutes=10))
        d, _ = make(t, clk, runner=Partial(t))
        d.tick()
        assert [v for k, v in states(d).items() if "|T24|" in k] == ["PARTIAL_V2_MISSING"]          # never reported DONE


def test_one_job_invokes_both_paths_with_the_same_snapshot_and_schedule():
    calls = []

    class R(LVE.LiveRunner):
        def __init__(self, root):
            super().__init__(root, 1000, cas_root=Path(root) / "cas")

        def snapshot_live(self, season, week, hz, kick, ids):
            return {"set_id": "SET", "content_id": "CID", "retrieval_ts": SCH.iso(SCH.forecast_cutoff(kick, hz) - timedelta(minutes=5)), "files": {}}, "retrieved_now"

        def log_v2(self, rec, season, week, hz, kick, game_ids, run_id):
            calls.append(("v2", rec["set_id"], SCH.iso(kick), tuple(game_ids), hz))
            return {g: {"status": "V2_LOGGED"} for g in game_ids}

        def run_context(self, hz, entries, season, week, run_id, fstore, bstore=None, only_games=None):
            calls.append(("phase1", entries[0][1]["set_id"], SCH.iso(entries[0][0]), tuple(entries[0][2]), hz))
            return [{"game_id": g, "status": "FORECAST_SUCCESS"} for g in entries[0][2]]
    with tempfile.TemporaryDirectory() as t:
        r = R(t)
        rec, how, logs = r.run_group(2026, 4, "T90", KICK, ["2026_04_AAA_BBB"], "x")
        assert [c[1:] for c in calls] == [calls[0][1:], calls[0][1:]] and {c[0] for c in calls} == {"v2", "phase1"}
        assert logs[0]["joint_ready"] is True


def test_preflight_detects_schedule_or_provenance_problems_before_phase1():
    import nfl_phase1e_scheduler as D
    raw = sched_bytes(datetime.now(UTC) + timedelta(days=2))
    with tempfile.TemporaryDirectory() as t:
        r = D.preflight(t, "2026_04_AAA_BBB", "T90", None, fetch_schedule=lambda: (raw, "x"), head=False)
        assert r["shared"]["same_on_both_paths"] == {"game_id": True, "home": True, "away": True, "kickoff": True, "cutoff": True}
        assert r["V2_READY"] is False and r["JOINT_READY"] is False                                # no sample data -> v2 dry build not proven -> not ready
        r2 = D.preflight(t, "2026_04_ZZZ_YYY", "T90", None, fetch_schedule=lambda: (raw, "x"), head=False)
        assert r2["problems"] and r2["PHASE1_READY"] is False and r2["V2_READY"] is False


# ------------------------------------------------------------------ durable-dispatch patch (Phase 1E operational hardening)
def raw_schedule(rows):
    """rows: [(game_id, kickoff_utc)] -> a provider-style games.csv (with market columns) for the live fake fetcher."""
    out = []
    for gid, k in rows:
        et = k.astimezone(SCH.P1.ET)
        out.append({"game_id": gid, "season": "2026", "game_type": "REG", "week": "4", "gameday": et.strftime("%Y-%m-%d"), "gametime": et.strftime("%H:%M"), "away_team": gid.split("_")[2], "home_team": gid.split("_")[3], "result": "", "spread_line": "-3", "total_line": "44"})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(out[0]), lineterminator="\n"); w.writeheader(); w.writerows(out)
    return buf.getvalue().encode()


def fake_fetcher(raw, calls=None):
    def f():
        if calls is not None:
            calls.append(1)
        san = CAS.sanitize_schedule(raw)
        got = {"games.csv": san, "stats_player_week_2026.csv": b"player_id,season,week,team\n", "injuries_2026.csv": b"season,week\n"}
        return got, {"games.csv": {"raw_sha256": hashlib.sha256(raw).hexdigest()}}, {}, raw
    return f


def live_runner(root, fetcher):
    return LVE.LiveRunner(root, 1000, cas_root=Path(root) / "cas", fetcher=fetcher, log=lambda m: None)


class PatchClock:
    def __init__(self, t):
        self.t = t
        self.old = LVE.utcnow
        LVE.utcnow = lambda: self.t

    def close(self):
        LVE.utcnow = self.old


def test_cutoffs_are_derived_from_real_kickoffs_not_hardcoded():
    et = SCH.P1.ET
    games = {"2026_04_AAA_BBB": datetime(2026, 10, 4, 9, 30, tzinfo=et), "2026_04_CCC_DDD": datetime(2026, 10, 4, 13, 0, tzinfo=et), "2026_04_EEE_FFF": datetime(2026, 10, 4, 16, 25, tzinfo=et), "2026_04_GGG_HHH": datetime(2026, 10, 4, 20, 20, tzinfo=et)}
    sched = SCH.parse_schedule(raw_schedule([(g, k.astimezone(UTC)) for g, k in games.items()]))
    exp = {"2026_04_AAA_BBB": ("2026-10-03T13:30:00.000000Z", "2026-10-04T12:00:00.000000Z"), "2026_04_CCC_DDD": ("2026-10-03T17:00:00.000000Z", "2026-10-04T15:30:00.000000Z"),
           "2026_04_EEE_FFF": ("2026-10-03T20:25:00.000000Z", "2026-10-04T18:55:00.000000Z"), "2026_04_GGG_HHH": ("2026-10-04T00:20:00.000000Z", "2026-10-04T22:50:00.000000Z")}
    for gid, (t24, t90) in exp.items():
        assert SCH.iso(SCH.forecast_cutoff(sched[gid]["kick"], "T24")) == t24 and SCH.iso(SCH.forecast_cutoff(sched[gid]["kick"], "T90")) == t90, gid
    with tempfile.TemporaryDirectory() as t:                                   # the dispatcher plans a distinct key per derived cutoff, nothing special-cased
        clk = Clock(datetime(2026, 10, 3, 0, 0, tzinfo=UTC))
        d = DS.Dispatcher(t, 1000, runner=FakeRunner(t), fetch_schedule=lambda: (raw_schedule([(g, k.astimezone(UTC)) for g, k in games.items()]), "x"), clock=clk, log=lambda m: None)
        d.tick()
        keys = set(states(d))
        assert len(keys) == 8 and {k.split("|")[2] for k in keys} == {v for pair in exp.values() for v in pair}


def test_live_snapshot_provenance_delta_and_exact_cutoff_label():
    with tempfile.TemporaryDirectory() as t:
        kick = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
        cut = SCH.forecast_cutoff(kick, "T90")
        clk = PatchClock(cut - timedelta(minutes=12))
        try:
            r = live_runner(t, fake_fetcher(raw_schedule([("2026_04_AAA_BBB", kick)])))
            rec, how = r.snapshot_live(2026, 4, "T90", kick, ["2026_04_AAA_BBB"])
            assert how == "retrieved_now" and rec["horizon"] == "T90" and rec["cutoff"] == SCH.iso(cut) and rec["retrieval_ts"] == SCH.iso(cut - timedelta(minutes=12)) and not rec["time_travel"]
            assert SCH.parse_iso(rec["retrieval_ts"]) <= SCH.parse_iso(rec["cutoff"])
            row, new = OPS.record_live_event(r, rec, how, "T90", kick, {"game_id": "2026_04_AAA_BBB", "season": 2026, "week": 4, "status": "FORECAST_SUCCESS", "n_records": 3, "v2": {"status": "V2_LOGGED"}, "joint_ready": True, "model_version": "mv"})
            assert new and row["seconds_before_cutoff"] == 720.0 and row["retrieval_minus_cutoff_seconds"] == -720.0 and row["retrieved_before_cutoff"] is True
            assert row["intended_horizon"] == "T90" and row["kickoff"] == SCH.iso(kick) and row["cutoff"] == SCH.iso(cut) and row["schedule_snapshot_sha256"] and row["schedule_raw_sha256"]
            row2, new2 = OPS.record_live_event(r, rec, how, "T90", kick, {"game_id": "2026_04_AAA_BBB", "season": 2026, "week": 4, "status": "FORECAST_SUCCESS", "v2": {"status": "V2_LOGGED"}, "joint_ready": True})
            assert not new2 and len(OPS.live_events(t).read()) == 1                                  # idempotent
            other = dict(rec, set_id="DIFFERENT")
            try:
                OPS.record_live_event(r, other, how, "T90", kick, {"game_id": "2026_04_AAA_BBB", "season": 2026, "week": 4}); assert False
            except ST.HardError:
                pass                                                                                  # same key, different snapshot: HARD ERROR
        finally:
            clk.close()


def test_early_snapshot_is_rejected_not_relabelled_and_late_is_missed():
    with tempfile.TemporaryDirectory() as t:
        kick = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
        cut = SCH.forecast_cutoff(kick, "T24")
        calls = []
        clk = PatchClock(cut - timedelta(hours=3))                                                   # an early wake-up: 3 h before the cutoff
        try:
            r = live_runner(t, fake_fetcher(raw_schedule([("2026_04_AAA_BBB", kick)]), calls))
            try:
                r.snapshot_live(2026, 4, "T24", kick, ["2026_04_AAA_BBB"]); assert False
            except CAS.CASError as e:
                assert str(e).startswith("early_snapshot_rejected") and "not labelled T24" in str(e)
            assert r.existing_set(r.group_name(2026, 4, "T24", kick)) is None and CAS.Ledger(Path(t) / "cas").read() == []      # nothing stored under the T24 label
            clk.t = cut + timedelta(seconds=1)
            n = len(calls)
            try:
                r.snapshot_live(2026, 4, "T24", kick, ["2026_04_AAA_BBB"]); assert False
            except CAS.CASError as e:
                assert str(e).startswith("MISSED_REAL_CUTOFF")
            assert len(calls) == n                                                                    # no retrieval is even attempted after the cutoff
            clk.t = cut - timedelta(minutes=5)                                                        # inside the window it is accepted
            rec, how = r.snapshot_live(2026, 4, "T24", kick, ["2026_04_AAA_BBB"])
            assert how == "retrieved_now" and rec["retrieval_ts"] == SCH.iso(cut - timedelta(minutes=5))
        finally:
            clk.close()


def test_stored_snapshot_is_reused_after_cutoff_with_identical_identity_and_tamper_is_detected():
    with tempfile.TemporaryDirectory() as t:
        kick = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
        cut = SCH.forecast_cutoff(kick, "T90")
        calls = []
        clk = PatchClock(cut - timedelta(minutes=10))
        try:
            r = live_runner(t, fake_fetcher(raw_schedule([("2026_04_AAA_BBB", kick)]), calls))
            rec, _ = r.snapshot_live(2026, 4, "T90", kick, ["2026_04_AAA_BBB"])
            clk.t = cut + timedelta(minutes=20)                                                       # the process died; restart after the cutoff
            r2 = live_runner(t, fake_fetcher(raw_schedule([("2026_04_AAA_BBB", kick)]), calls))
            rec2, how2 = r2.snapshot_live(2026, 4, "T90", kick, ["2026_04_AAA_BBB"])
            assert how2 == "reused_after_restart" and rec2["set_id"] == rec["set_id"] and rec2["content_id"] == rec["content_id"] and rec2["retrieval_ts"] == rec["retrieval_ts"] and len(calls) == 1
            sha = rec["files"]["games.csv"]
            blob = r2.store.path(sha); os.chmod(blob, 0o644); blob.write_bytes(b"tampered"); 
            try:
                r2.snapshot_live(2026, 4, "T90", kick, ["2026_04_AAA_BBB"]); assert False
            except CAS.CASError as e:
                assert str(e).startswith("stored_snapshot_invalid")
        finally:
            clk.close()


def _install_prefit(root, tamper=False):
    src = REPO / "nfl_models" / "nfl_player_outcome_phase1d" / "artifacts_freeze_fit_2026wk3"
    man = json.loads((src / "manifest.json").read_text())
    dst = Path(root) / "artifacts" / man["bundle_sha256"]
    import shutil
    shutil.copytree(src, dst)
    if tamper:
        f = sorted(p for p in dst.iterdir() if p.name != "manifest.json")[0]
        f.write_bytes(f.read_bytes() + b"x")
    row = {"season": 2026, "week": 4, "prefit_version": OPS.OPS_VERSION, "code_identity": "x", "training_cutoff_last_completed_week": 3, "source_identity": {"fit_snapshot_set_id": "s"},
           "artifact_bundle_sha256": man["bundle_sha256"], "created_at": "2026-10-01T00:00:00.000000Z"}
    OPS.prefit_ledger(root).append_many([row])
    return row


def test_prefit_is_loaded_not_refit_and_hash_mismatch_is_detected():
    orig = RN_fit_guard()
    try:
        with tempfile.TemporaryDirectory() as t:
            r = live_runner(t, fake_fetcher(raw_schedule([("2026_04_AAA_BBB", KICK)])))
            try:
                r.weekly_artifacts(None, None, (2026, 4), "c"); assert False
            except CAS.CASError as e:
                assert str(e).startswith("prefit_missing")                                                  # a cutoff never fits
            row = _install_prefit(t)
            a1 = r.weekly_artifacts(None, None, (2026, 4), "c"); a2 = r.weekly_artifacts(None, None, (2026, 4), "c")
            assert a1.manifest()["bundle_sha256"] == row["artifact_bundle_sha256"] == a2.manifest()["bundle_sha256"] and r.prefit_used["artifact_bundle_sha256"] == row["artifact_bundle_sha256"]
        with tempfile.TemporaryDirectory() as t:
            r = live_runner(t, fake_fetcher(raw_schedule([("2026_04_AAA_BBB", KICK)])))
            _install_prefit(t, tamper=True)
            try:
                r.weekly_artifacts(None, None, (2026, 4), "c"); assert False
            except CAS.CASError as e:
                assert str(e).startswith("prefit_hash_mismatch")
    finally:
        orig()


def RN_fit_guard():
    """Make any attempt to FIT during these tests fail loudly."""
    import nfl_phase1d_runner as RN
    import nfl_phase1d_p1a as P
    old = P.fit_artifacts
    def boom(*a, **k):
        raise AssertionError("a live cutoff attempted to fit the weekly artifact")
    P.fit_artifacts = boom
    return lambda: setattr(P, "fit_artifacts", old)


class PersistRunner(FakeRunner):
    """File-backed fake runner so a REAL process death (os._exit) can be survived by a new process."""

    def __init__(self, root, die_after_snapshot=False):
        super().__init__(root)
        self.f = Path(root) / "fake_sets.json"
        self.sets = json.loads(self.f.read_text()) if self.f.exists() else {}
        self.die = die_after_snapshot

    def run_group(self, season, week, hz, kick, gids, run_id):
        group = self.group_name(season, week, hz, kick)
        if group not in self.sets:
            self.sets[group] = {"set_id": "S" + group[-8:], "retrieval_ts": SCH.iso(SCH.forecast_cutoff(kick, hz) - timedelta(minutes=5))}
            self.f.write_text(json.dumps(self.sets))
        if self.die:
            os._exit(9)                                                                                  # process death after the snapshot is stored, before any forecast completes
        return self.sets[group], "reused_after_restart", [{"game_id": g, "status": "FORECAST_SUCCESS", "n_records": 3, "written": 3} for g in gids]


def test_real_process_death_after_snapshot_then_restart_resumes_same_snapshot():
    with tempfile.TemporaryDirectory() as t:
        code = ("import sys, os, json; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
                "import test_nfl_phase1e as T\n"
                "from datetime import datetime, timezone\n"
                "clk = T.Clock(datetime.fromtimestamp(float(os.environ['NOW']), timezone.utc))\n"
                "d = T.DS.Dispatcher(%r, 1000, runner=T.PersistRunner(%r, die_after_snapshot=os.environ['DIE'] == '1'), fetch_schedule=lambda: (T.sched_bytes(), 'x'), clock=clk, log=lambda m: None)\n"
                "print(json.dumps(d.tick()))\n") % (str(REPO), str(REPO / "tests"), t, t)
        env = {**os.environ, "NOW": str((KICK - timedelta(hours=24) - timedelta(minutes=10)).timestamp()), "DIE": "1"}
        p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert p.returncode == 9, p.stderr[-500:]                                                         # killed mid-run
        led = [json.loads(x) for x in (Path(t) / "dispatch_ledger.jsonl").read_text().splitlines()]
        assert [r["state"] for r in led if "|T24|" in r["key"]][-1] == "STARTED"
        env2 = {**os.environ, "NOW": str((KICK - timedelta(hours=24) + timedelta(minutes=3)).timestamp()), "DIE": "0"}      # restart AFTER the cutoff
        p2 = subprocess.run([sys.executable, "-c", code], env=env2, capture_output=True, text=True)
        assert p2.returncode == 0, p2.stderr[-500:]
        led = [json.loads(x) for x in (Path(t) / "dispatch_ledger.jsonl").read_text().splitlines()]
        done = [r for r in led if "|T24|" in r["key"] and r["state"] == "DONE"]
        assert len(done) == 1 and done[0]["snapshot_set_id"] == json.loads((Path(t) / "fake_sets.json").read_text()).popitem()[1]["set_id"]
        p3 = subprocess.run([sys.executable, "-c", code], env=env2, capture_output=True, text=True)         # a duplicate invocation changes nothing
        led3 = [json.loads(x) for x in (Path(t) / "dispatch_ledger.jsonl").read_text().splitlines()]
        assert len([r for r in led3 if r["state"] == "DONE" and "|T24|" in r["key"]]) == 1


def test_prefit_missing_is_retryable_until_kickoff_then_closed():
    class R(FakeRunner):
        def __init__(self, root):
            super().__init__(root); self.fail = True

        def run_group(self, season, week, hz, kick, gids, run_id):
            if self.fail:
                group = self.group_name(season, week, hz, kick)
                self.sets[group] = {"set_id": "S1", "retrieval_ts": SCH.iso(SCH.forecast_cutoff(kick, hz) - timedelta(minutes=5))}
                self.fail = False
                raise CAS.CASError("prefit_missing: no prefit artifact for season 2026 week 4")
            return super().run_group(season, week, hz, kick, gids, run_id)
    with tempfile.TemporaryDirectory() as t:
        clk = Clock(KICK - timedelta(hours=24) - timedelta(minutes=10))
        d, _ = make(t, clk, runner=R(t)); d.tick()
        assert [v for k, v in states(d).items() if "|T24|" in k] == ["STARTED"]                              # not terminal
        clk.t = KICK - timedelta(hours=20); d.tick()                                                         # after the cutoff, before kickoff: resumes from the stored snapshot
        assert [v for k, v in states(d).items() if "|T24|" in k] == ["DONE"]
    with tempfile.TemporaryDirectory() as t:                                                                 # never resolved before kickoff -> closed, never forecast
        class Never(FakeRunner):
            def run_group(self, *a, **k):
                raise CAS.CASError("prefit_missing: x")
        clk = Clock(KICK - timedelta(hours=24) - timedelta(minutes=10))
        d, _ = make(t, clk, runner=Never(t)); d.tick()
        clk.t = KICK + timedelta(minutes=1); d.tick()
        assert [v for k, v in states(d).items() if "|T24|" in k] == ["MISSED_REAL_CUTOFF"] and d.runner.calls == []


def test_status_heartbeat_and_readiness_report():
    with tempfile.TemporaryDirectory() as t:
        clk = Clock(KICK - timedelta(hours=30))
        d, _ = make(t, clk)
        d.tick()
        sched = SCH.parse_schedule(d.probes and CAS.BlobStore(Path(t) / "cas").get(d.probes.read()[-1]["sha256"]))
        rd = OPS.readiness(t, sched, clk())
        names = {c["name"]: c["ok"] for c in rd["checks"]}
        assert names["workflow_configured"] and names["source_registry_complete"] and names["store_writable"] and names["schedule_resolvable"] and names["phase1_runner_ready"] and names["frozen_v2_ready"]
        assert names["prefit_artifacts_verified"] is False and rd["READY"] is False                         # no prefit yet: NOT ready, loudly
        st = OPS.write_status(t, clk(), d.states(), sched, DS.SCHEDULER_VERSION, rd)
        on_disk = json.loads((Path(t) / "status.json").read_text())
        assert on_disk["last_invocation_utc"] == SCH.iso(clk()) and on_disk["counts_by_state"]["PLANNED"] == 2 and on_disk["next_cutoffs"][0]["cutoff"] == SCH.iso(SCH.forecast_cutoff(KICK, "T24"))
        assert on_disk["readiness"]["READY"] is False and "prefit_artifacts_verified" in on_disk["readiness"]["failing"] and (Path(t) / "readiness.json").exists()
        _install_prefit(t)                                                                                    # with a verified prefit the check turns green
        assert {c["name"]: c["ok"] for c in OPS.readiness(t, sched, clk())["checks"]}["prefit_artifacts_verified"] is True


def test_phase1_and_v2_use_one_event_and_missing_v2_is_not_joint_ready():
    seen = []

    class R(LVE.LiveRunner):
        def __init__(self, root, v2_status):
            super().__init__(root, 1000, cas_root=Path(root) / "cas"); self.v2s = v2_status

        def snapshot_live(self, season, week, hz, kick, ids):
            return {"set_id": "SET", "content_id": "CID", "retrieval_ts": "2026-10-04T15:20:00.000000Z", "files": {"games.csv": "h"}}, "retrieved_now"

        def log_v2(self, rec, season, week, hz, kick, game_ids, run_id):
            seen.append(("v2", rec["set_id"], rec["retrieval_ts"], SCH.iso(kick), hz, tuple(game_ids)))
            return {g: {"status": self.v2s} for g in game_ids}

        def run_context(self, hz, entries, season, week, run_id, fstore, bstore=None, only_games=None):
            seen.append(("phase1", entries[0][1]["set_id"], entries[0][1]["retrieval_ts"], SCH.iso(entries[0][0]), hz, tuple(entries[0][2])))
            return [{"game_id": g, "season": season, "week": week, "status": "FORECAST_SUCCESS", "model_version": "mv"} for g in entries[0][2]]
    for v2s, joint in (("V2_LOGGED", True), ("V2_FAILED", False)):
        seen.clear()
        with tempfile.TemporaryDirectory() as t:
            r = R(t, v2s)
            rec, how, logs = r.run_group(2026, 4, "T90", KICK, ["2026_04_AAA_BBB"], "x")
            assert [x[1:] for x in seen if x[0] == "v2"] == [x[1:] for x in seen if x[0] == "phase1"] and len(seen) == 2          # same snapshot, retrieval, kickoff, horizon, games
            assert logs[0]["joint_ready"] is joint
            ev = OPS.live_events(t).read()[0]
            assert ev["joint_ready"] is joint and ev["v2"]["status"] == v2s and ev["phase1_status"] == "FORECAST_SUCCESS"


def test_finalize_expired_closes_open_keys_without_forecasting():
    with tempfile.TemporaryDirectory() as t:
        clk = Clock(KICK - timedelta(hours=40))
        d, _ = make(t, clk); d.tick()                                                                         # PLANNED rows only
        clk.t = KICK + timedelta(hours=1); d.tick()
        assert sorted(states(d).values()) == ["MISSED_REAL_CUTOFF", "MISSED_REAL_CUTOFF"] and d.runner.calls == []


def test_duplicate_invocation_zero_duplicate_forecasts_in_the_store():
    with tempfile.TemporaryDirectory() as t:
        st = ST.Store(t, "forecasts")
        recs = [{"id": f"f{i}", "v": i} for i in range(4)]
        assert st.append_batch("b", {"run": 1}, recs)["written"] == 4
        r = st.append_batch("b", {"run": 2}, recs)
        assert r["written"] == 0 and r["verified_duplicates"] == 4 and len(st.all_records()) == 4
        try:
            st.append_batch("b", {"run": 3}, [{"id": "f1", "v": 999}]); assert False
        except ST.HardError:
            pass


def test_github_workflow_is_the_durable_shadow_entrypoint():
    wf = (REPO / ".github" / "workflows" / "nfl_phase1e_shadow.yml").read_text()
    try:
        import yaml
        d = yaml.safe_load(wf)
        on = d.get("on") or d.get(True)
        assert "schedule" in on and "workflow_dispatch" in on and on["schedule"][0]["cron"]
        assert d["concurrency"]["group"] == "nfl-phase1e-shadow" and d["concurrency"]["cancel-in-progress"] is False
    except ImportError:
        assert "schedule:" in wf and "workflow_dispatch:" in wf and "cancel-in-progress: false" in wf
    assert "nfl_phase1e_scheduler.py run" in wf and "nfl_phase1e_scheduler.py readiness" in wf and "nfl-shadow-state" in wf
    body = "\n".join(l for l in wf.splitlines() if not l.strip().startswith("#"))
    for other in ("build.py", "cfb_", "nhl_", "mlb_", "nfl_serving_builder", "docs/"):
        assert other not in body.replace("nfl_phase1e_scheduler.py", ""), other
    assert OPS.workflow_config()["invokes_scheduler"] and not OPS.workflow_config()["touches_other_sports"]
    allsports = (REPO / ".github" / "workflows" / "all_sports_predictions.yml").read_text()
    assert "nfl_phase1e" not in allsports                                                                   # the existing workflow is untouched


def test_runbook_states_the_required_caveats():
    t = (REPO / "nfl_models" / "nfl_player_outcome_phase1e" / "RUNBOOK.md").read_text()
    for needle in ("AD_HOC_PREGAME", "not clean T24/T90 evidence", "R11", "BLOCKER", "workflow_dispatch", "MISSED_REAL_CUTOFF", "status.json"):
        assert needle in t, needle


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
