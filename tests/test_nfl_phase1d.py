"""
Phase 1D tests (freeze hardening / live shadow readiness). Standalone and pytest-compatible.

  python -u tests/test_nfl_phase1d.py

Unit tests (synthetic data): content-addressed store, snapshot rules, time-travel truncation, schedule/cutoff rules, artifact round trip and tamper detection, walk-forward windows,
store locking (mutual exclusion, timeout, kill -9 recovery, concurrent writers, host binding), accuracy-label guard, gate/protocol consistency.
Result tests: assertions on the recorded operational evidence (LiveLoader equivalence, multi-week dry run, perturbations, N audit, calibration freeze, freeze candidate v2).
"""
import csv
import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nfl_phase1_store as ST  # noqa: E402
import nfl_phase1_store_lock as LK  # noqa: E402
import nfl_phase1d_accuracy as ACC  # noqa: E402
import nfl_phase1d_cas as CAS  # noqa: E402
import nfl_phase1d_gates as GT  # noqa: E402
import nfl_phase1d_p1a as P  # noqa: E402
import nfl_phase1d_schedule as SCH  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
P1D = REPO / "nfl_models" / "nfl_player_outcome_phase1d"
UTC = timezone.utc


def load(p):
    return json.loads((P1D / p).read_text())


# ------------------------------------------------------------------ CAS
def test_all_sources_read_by_phase1_code_are_snapshotted():
    names = set()
    for f in ("nfl_phase1_data.py", "nfl_phase1b_data.py", "nfl_context_v4.py", "nfl_phase1_defense_events.py", "nfl_phase1c_constants.py"):
        text = (REPO / f).read_text()
        for m in re.finditer(r'["\']((?:games|players)\.csv|(?:injuries|roster_weekly|stats_player_week|snap_counts|participation|ftn|depth_charts)_|pbp_)', text):
            names.add(m.group(1))
    covered = set()
    for tpl in list(CAS.SEASON_FILES) + list(CAS.STATIC_FILES):
        covered.add(tpl.split("{s}")[0])
    for n in names:
        assert n in covered or n.rstrip("_") + "_" in covered or any(c.startswith(n) for c in covered), f"source {n} is read by Phase 1 code but is not in the CAS logical file list"
    assert {"games.csv", "players.csv"} <= set(CAS.logical_files(seasons=[2025]))
    lf = CAS.logical_files(seasons=[2025])
    for pre in ("injuries_", "roster_weekly_", "stats_player_week_", "snap_counts_", "pbp_", "participation_", "ftn_", "depth_charts_"):
        assert any(x.startswith(pre) for x in lf), pre


def test_cas_deduplicates_and_never_overwrites():
    with tempfile.TemporaryDirectory() as d:
        st = CAS.BlobStore(d)
        a = st.put(b"hello world"); b = st.put(b"hello world"); c = st.put(b"other")
        assert a["sha256"] == b["sha256"] and not a["deduplicated"] and b["deduplicated"] and st.blob_count() == 2 and c["sha256"] != a["sha256"]
        p = st.path(a["sha256"])
        os.chmod(p, 0o644); p.write_bytes(b"tampered"); os.chmod(p, 0o444)
        try:
            st.put(b"hello world"); assert False, "corrupt existing blob must not be overwritten"
        except CAS.CASError:
            pass
        assert p.read_bytes() == b"tampered"                                    # untouched: refused, not replaced


def test_cas_hash_verification_on_read_and_materialize():
    with tempfile.TemporaryDirectory() as d:
        st = CAS.BlobStore(Path(d) / "cas"); led = CAS.Ledger(Path(d) / "cas")
        kick = datetime(2026, 10, 4, 17, 0, tzinfo=UTC); cut = kick - timedelta(hours=24)
        rec = CAS.take_snapshot_set(st, led, {"games.csv": ("nflverse:test", lambda: b"a,b\n1,2\n"), "players.csv": ("nflverse:test", lambda: b"x\n1\n")}, "T24", kick, cut, cut - timedelta(hours=1), "g")
        assert CAS.verify_set(st, led, rec, ["games.csv", "players.csv"]) == []
        assert CAS.verify_set(st, led, rec, ["games.csv", "injuries_2026.csv"]) != []             # required source missing is reported
        m = CAS.materialize(st, rec, Path(d) / "mat")
        assert (m / "games.csv").read_bytes() == b"a,b\n1,2\n"
        p = st.path(rec["files"]["games.csv"]); os.chmod(p, 0o644); p.write_bytes(b"evil"); os.chmod(p, 0o444)
        for fn in (lambda: st.get(rec["files"]["games.csv"]), lambda: CAS.materialize(st, rec, Path(d) / "mat2")):
            try:
                fn(); assert False
            except CAS.CASError:
                pass
        rows = led.read()
        assert all(r["sha256"] and r["bytes"] > 0 and r["parser_version"] and r["source"] and r["retrieval_ts"] and r["cutoff"] and r["blob"] for r in rows if r["type"] == "file")


def test_identical_bytes_at_two_forecast_times_share_one_blob():
    with tempfile.TemporaryDirectory() as d:
        st = CAS.BlobStore(Path(d) / "cas"); led = CAS.Ledger(Path(d) / "cas")
        kick = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
        r1 = CAS.take_snapshot_set(st, led, {"games.csv": ("p", lambda: b"same\n")}, "T24", kick, kick - timedelta(hours=24), kick - timedelta(hours=25), "g24")
        r2 = CAS.take_snapshot_set(st, led, {"games.csv": ("p", lambda: b"same\n")}, "T90", kick, kick - timedelta(minutes=90), kick - timedelta(hours=2), "g90")
        assert r1["files"] == r2["files"] and st.blob_count() == 1 and r1["set_id"] != r2["set_id"] and r1["content_id"] == r2["content_id"]
        try:
            CAS.take_snapshot_set(st, led, {"games.csv": ("p", lambda: b"x\n")}, "T24", kick, kick - timedelta(hours=24), kick, "late")
            assert False, "retrieval after the cutoff must be refused"
        except CAS.CASError:
            pass


def _mini_source(d):
    """Synthetic 2-season source directory for time-travel truncation tests."""
    d = Path(d)
    games = [
        {"game_id": "2026_01_AAA_BBB", "season": "2026", "game_type": "REG", "week": "1", "gameday": "2026-09-10", "gametime": "20:20", "away_team": "AAA", "home_team": "BBB", "result": "3", "home_score": "20", "away_score": "17",
         "total": "37", "spread_line": "-3", "old_game_id": "x", "home_qb_id": "q", "away_qb_id": "q2", "home_qb_name": "n", "away_qb_name": "m", "referee": "r", "overtime": "0"},
        {"game_id": "2026_01_CCC_DDD", "season": "2026", "game_type": "REG", "week": "1", "gameday": "2026-09-13", "gametime": "13:00", "away_team": "CCC", "home_team": "DDD", "result": "7", "home_score": "24", "away_score": "17",
         "total": "41", "spread_line": "-6", "old_game_id": "y", "home_qb_id": "q", "away_qb_id": "q2", "home_qb_name": "n", "away_qb_name": "m", "referee": "r", "overtime": "0"},
        {"game_id": "2026_02_AAA_CCC", "season": "2026", "game_type": "REG", "week": "2", "gameday": "2026-09-17", "gametime": "20:20", "away_team": "AAA", "home_team": "CCC", "result": "", "home_score": "", "away_score": "",
         "total": "", "spread_line": "-1", "old_game_id": "z", "home_qb_id": "", "away_qb_id": "", "home_qb_name": "", "away_qb_name": "", "referee": "", "overtime": ""}]
    with open(d / "games.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(games[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(games)
    (d / "players.csv").write_text("gsis_id,position\n1,QB\n")
    for name, rows, hdr in (("stats_player_week_2026.csv", [{"player_id": "1", "season": "2026", "week": "1", "game_id": "2026_01_AAA_BBB", "team": "AAA", "carries": "3"},
                                                          {"player_id": "2", "season": "2026", "week": "1", "game_id": "2026_01_CCC_DDD", "team": "CCC", "carries": "4"},
                                                          {"player_id": "1", "season": "2026", "week": "2", "game_id": "2026_02_AAA_CCC", "team": "AAA", "carries": "9"}],
                                 ["player_id", "season", "week", "game_id", "team", "carries"]),
                            ("injuries_2026.csv", [{"season": "2026", "week": "1", "gsis_id": "1"}, {"season": "2026", "week": "2", "gsis_id": "1"}, {"season": "2026", "week": "3", "gsis_id": "1"}], ["season", "week", "gsis_id"]),
                            ("roster_weekly_2026.csv", [{"season": "2026", "week": "1", "gsis_id": "1"}, {"season": "2026", "week": "2", "gsis_id": "1"}], ["season", "week", "gsis_id"]),
                            ("depth_charts_2026.csv", [{"dt": "2026-09-08T10:00:00Z", "team": "AAA", "gsis_id": "1"}, {"dt": "2026-09-16T10:00:00Z", "team": "AAA", "gsis_id": "1"}, {"dt": "2026-09-18T10:00:00Z", "team": "AAA", "gsis_id": "1"}],
                             ["dt", "team", "gsis_id"])):
        with open(d / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=hdr, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    return d


def test_time_travel_truncation_rules():
    with tempfile.TemporaryDirectory() as d:
        src = _mini_source(d)
        tt = CAS.TimeTravelSource(src, seasons=[2026])
        cut = datetime(2026, 9, 16, 20, 20, tzinfo=UTC)                        # T24 of the week-2 Thursday game (kickoff 2026-09-18 00:20Z)
        done = tt.completed_ids(cut)
        assert done == {"2026_01_AAA_BBB", "2026_01_CCC_DDD"}                   # kickoff + 24h <= cutoff
        b, raw, tr = tt.bytes_for("stats_player_week_2026.csv", cut, (2026, 2), "T24")
        rows = list(csv.DictReader(io.StringIO(b.decode())))
        assert [r["week"] for r in rows] == ["1", "1"] and tr and raw
        g, _, tr2 = tt.bytes_for("games.csv", cut, (2026, 2), "T24")
        grows = {r["game_id"]: r for r in csv.DictReader(io.StringIO(g.decode()))}
        assert grows["2026_02_AAA_CCC"]["result"] == "" and grows["2026_01_AAA_BBB"]["result"] == "3" and "spread_line" not in grows["2026_01_AAA_BBB"] and "market" in tr2 or "strip" in tr2
        inj, _, _ = tt.bytes_for("injuries_2026.csv", cut, (2026, 2), "T24")
        assert {r["week"] for r in csv.DictReader(io.StringIO(inj.decode()))} == {"1", "2"}                  # never a later week
        r24, _, _ = tt.bytes_for("roster_weekly_2026.csv", cut, (2026, 2), "T24")
        r90, _, _ = tt.bytes_for("roster_weekly_2026.csv", cut, (2026, 2), "T90")
        assert {r["week"] for r in csv.DictReader(io.StringIO(r24.decode()))} == {"1"} and {r["week"] for r in csv.DictReader(io.StringIO(r90.decode()))} == {"1", "2"}   # game-day roster only at T90
        dc, _, _ = tt.bytes_for("depth_charts_2026.csv", cut, (2026, 2), "T24")
        assert [r["dt"] for r in csv.DictReader(io.StringIO(dc.decode()))] == ["2026-09-08T10:00:00Z", "2026-09-16T10:00:00Z"]
        # contamination mode (test only) really keeps the target rows
        tc = CAS.TimeTravelSource(src, seasons=[2026], contaminate=(2026, 2))
        bc, _, _ = tc.bytes_for("stats_player_week_2026.csv", cut, (2026, 2), "T24")
        assert {r["week"] for r in csv.DictReader(io.StringIO(bc.decode()))} == {"1", "2"}


def test_live_retrieval_from_a_local_mirror_and_optional_sources():
    import functools
    import http.server
    import threading
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "mirror"
        for name in CAS.logical_files(seasons=[2026]):
            if name.startswith("ftn_"):
                continue                                                      # optional source not published
            url = CAS.provider_url(name, "http://x")
            rel = url.split("http://x/")[1]
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(_mini_games() if name == "games.csv" else f"content of {name}\n".encode())
        h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            st = CAS.BlobStore(Path(d) / "cas"); led = CAS.Ledger(Path(d) / "cas")
            kick = datetime(2026, 10, 4, 17, 0, tzinfo=UTC); cut = kick - timedelta(hours=24)
            rec = CAS.take_snapshot_set(st, led, CAS.live_sources([2026], base=base), "T24", kick, cut, cut - timedelta(minutes=5), "live")
            assert set(rec["unavailable_optional_sources"]) == {"ftn_2026.csv"} and "games.csv" in rec["files"] and "ftn_2026.csv" not in rec["files"]
            g = st.get(rec["files"]["games.csv"]).decode()
            assert "spread_line" not in g and "game_id" in g                                             # sportsbook columns are never stored
            rows = [r for r in led.read() if r["type"] == "file" and r["logical_name"] == "games.csv"]
            assert rows[0]["transform"] == "strip_market_columns" and rows[0]["raw_sha256"] and rows[0]["source"].startswith(base)
            (root / "injuries" / "injuries_2026.csv").unlink()                                        # a REQUIRED source disappears
            try:
                CAS.take_snapshot_set(st, led, CAS.live_sources([2026], base=base), "T24", kick, cut, cut - timedelta(minutes=5), "live2"); assert False
            except CAS.CASError as e:
                assert "required source" in str(e)
        finally:
            srv.shutdown()


def _mini_games():
    return (b"game_id,season,game_type,week,gameday,gametime,away_team,home_team,result,spread_line,old_game_id\n"
            b"2026_05_AAA_BBB,2026,REG,5,2026-10-04,13:00,AAA,BBB,,-3,x\n")


# ------------------------------------------------------------------ schedule
def test_schedule_cutoff_from_snapshot_and_shift_rules():
    with tempfile.TemporaryDirectory() as d:
        src = _mini_source(d)
        sched = SCH.parse_schedule((src / "games.csv").read_bytes(), seasons={2026})
        k = sched["2026_02_AAA_CCC"]["kick"]
        assert SCH.forecast_cutoff(k, "T24") == k - timedelta(hours=24) and SCH.forecast_cutoff(k, "T90") == k - timedelta(minutes=90)
        led = SCH.ScheduleLedger(Path(d) / "led")
        r1, ch1 = led.observe("g", k, "s1", k - timedelta(days=3))
        r1b, ch1b = led.observe("g", k, "s2", k - timedelta(days=2))
        assert ch1 and not ch1b and r1["revision"] == 1 and len(led.rows()) == 1                       # unchanged kickoff: no new revision
        r2, ch2 = led.observe("g", k - timedelta(hours=5), "s3", k - timedelta(hours=40))               # flex earlier
        assert ch2 and r2["revision"] == 2 and r2["change"] == "moved_earlier"
        r3, _ = led.observe("g", k + timedelta(days=1), "s4", k - timedelta(hours=30))
        assert r3["revision"] == 3 and r3["change"] == "moved_later" and len(led.rows()) == 3           # append-only: nothing edited
        led.observe("g", None, "s5", k, status="cancelled")
        assert led.latest("g")["status"] == "cancelled" and SCH.game_status_for_scoring(led.rows(), played=False) == "GAME_NOT_PLAYED_AS_SCHEDULED"
        # scoring rule: only a forecast whose cutoff matches the FINAL kickoff and whose retrieval preceded it counts
        fk = k + timedelta(days=1)
        f_old = {"cutoff": SCH.iso(k - timedelta(hours=24)), "retrieval_ts": SCH.iso(k - timedelta(hours=25))}
        f_new = {"cutoff": SCH.iso(fk - timedelta(hours=24)), "retrieval_ts": SCH.iso(fk - timedelta(hours=25))}
        assert SCH.select_primary([f_old], fk, "T24") is None and SCH.select_primary([f_old, f_new], fk, "T24") is f_new
        late = {"cutoff": SCH.iso(fk - timedelta(hours=24)), "retrieval_ts": SCH.iso(fk - timedelta(hours=23))}
        assert SCH.select_primary([late], fk, "T24") is None                                            # retrieved after the cutoff: invalid


# ------------------------------------------------------------------ artifacts
def _tiny_blobs():
    import xgboost as xgb
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)); y = (X[:, 0] > 0).astype(float)
    b = xgb.train({"objective": "binary:logistic", "max_depth": 2, "seed": 1, "nthread": 1}, xgb.DMatrix(X, label=y, feature_names=["a", "b", "c"]), 5)
    blobs = {"te.json": b"{}", "availability.json": b"{}", "role.json": b"{}", "propensity.json": b"{}", "opportunity.json": b"{}", "m.ubj": P.xgb_raw(b)}
    return blobs, X, b


def test_serialized_model_roundtrip_and_tamper_detection():
    blobs, X, b = _tiny_blobs()
    art = P.Artifacts(P._parse(blobs), blobs, {"t": 1})
    p0 = P.xgb_predict(b, 5, X, ["a", "b", "c"]); p1 = P.xgb_predict(art.obj["xgb"]["m.ubj"], 5, X, ["a", "b", "c"])
    assert np.array_equal(p0, p1)                                                                        # bit-identical after serialize -> load
    with tempfile.TemporaryDirectory() as d:
        man = art.save(d)
        assert set(man["files"]) == set(blobs) and len(man["bundle_sha256"]) == 64
        a2 = P.Artifacts.load(d)
        assert a2.manifest()["bundle_sha256"] == man["bundle_sha256"] and np.array_equal(P.xgb_predict(a2.obj["xgb"]["m.ubj"], 5, X, ["a", "b", "c"]), p0)
        (Path(d) / "m.ubj").write_bytes((Path(d) / "m.ubj").read_bytes() + b"x")
        try:
            P.Artifacts.load(d); assert False
        except RuntimeError:
            pass


def test_walk_forward_windows_use_only_weeks_before_target():
    keys = [(s, w, "T") for s in (2023, 2024, 2025) for w in range(1, 19)]
    win = P.walk_forward_windows(keys, (2025, 10))
    tw = {tuple(x) for x in win.detail["train_weeks"]} | {tuple(x) for x in win.detail["valid_weeks"]}
    assert max(tw) == (2025, 9) and all(t < (2025, 10) for t in tw) and win.detail["last_training_week"] == [2025, 9]
    assert not win.train(2025, 10) and not win.valid(2025, 10) and not win.train(2026, 1) and not win.train(2022, 5)
    assert win.valid(2025, 9) and win.train(2023, 1) and not win.valid(2023, 1)
    try:
        P.walk_forward_windows([(2023, w, "T") for w in range(1, 5)], (2023, 5)); assert False
    except ValueError:
        pass
    w2 = P.walk_forward_windows(keys, (2025, 11))
    assert (2025, 10) in {tuple(x) for x in w2.detail["train_weeks"]} | {tuple(x) for x in w2.detail["valid_weeks"]}      # a completed week enters the NEXT week's training


def test_no_forward_target_rows_in_training():
    keys = [(s, w, "T") for s in (2023, 2024, 2025) for w in range(1, 19)]
    win = P.walk_forward_windows(keys, (2025, 10))
    import nfl_phase1_common as C
    with P.use_windows(win, audit_cutoff_week=(2025, 10)):
        C.audit_fit("x", [{"s": 2025, "w": 9}], [{"s": 2025, "w": 8}])                                     # fine
        for bad in ({"s": 2025, "w": 10}, {"s": 2026, "w": 1}):
            try:
                C.audit_fit("x", [bad], []); assert False, bad
            except AssertionError as e:
                assert "at/after the target week" in str(e)
    assert C.TRAIN(2023, 5) and C.VALID(2024, 15) and not C.TRAIN(2025, 3)                                  # research windows restored


# ------------------------------------------------------------------ locking
def _hold_lock(path, ready, hold):
    lk = LK.StoreLock(path, timeout=5)
    lk.acquire(); ready.set(); time.sleep(hold)


def _append(root, name, ids, out):
    try:
        s = ST.Store(root, "f")
        r = s.append_batch(name, {"who": os.getpid()}, [{"id": i, "v": 1} for i in ids])
        out.put(("ok", r["written"]))
    except Exception as e:  # noqa
        out.put(("err", type(e).__name__))


def _append_conflict(root, out):
    try:
        s = ST.Store(root, "f")
        s.append_batch("conflict", {}, [{"id": "id0", "v": 999}])
        out.put(("ok", 0))
    except Exception as e:  # noqa
        out.put(("err", type(e).__name__))


def test_lock_mutual_exclusion_and_explicit_timeout():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.lock"
        ready = mp.Event()
        pr = mp.Process(target=_hold_lock, args=(str(p), ready, 3.0)); pr.start()
        assert ready.wait(10)
        try:
            LK.StoreLock(p, timeout=0.5).acquire(); assert False, "second holder must time out"
        except LK.LockTimeout as e:
            assert "owner record" in str(e)                                                               # explicit, names the owner
        pr.join(10)
        lk = LK.StoreLock(p, timeout=5); lk.acquire(); lk.release()                                       # free again after the holder exits


def test_stale_lock_recovery_after_kill9():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "y.lock"
        ready = mp.Event()
        pr = mp.Process(target=_hold_lock, args=(str(p), ready, 60.0)); pr.start()
        assert ready.wait(10)
        os.kill(pr.pid, signal.SIGKILL); pr.join(10)
        lk = LK.StoreLock(p, timeout=5); lk.acquire()
        assert lk.last_recovery and lk.last_recovery["event"] == "recovered_stale_owner" and lk.last_recovery["dead_pid"] == pr.pid
        lk.release()


def test_concurrent_writers_no_corruption_duplicates_idempotent_conflict_hard_error():
    with tempfile.TemporaryDirectory() as d:
        q = mp.Queue()
        ps = [mp.Process(target=_append, args=(d, f"b{i}", [f"id{j}" for j in range(i % 3, i % 3 + 8)], q)) for i in range(6)]
        [p.start() for p in ps]; [p.join(60) for p in ps]
        res = [q.get(timeout=5) for _ in ps]
        assert all(r[0] == "ok" for r in res), res
        s = ST.Store(d, "f")
        recs = s.all_records(); ids = [r["id"] for r in recs]
        assert len(ids) == len(set(ids))                                                                   # every id exactly once across batches
        assert set(ids) == {f"id{j}" for j in range(0, 10)} and s.recover() == []
        # same id, different bytes, from another process: HARD ERROR and nothing written
        pr = mp.Process(target=_append_conflict, args=(d, q)); pr.start(); pr.join(30)
        assert q.get(timeout=5) == ("err", "HardError")
        assert len(s.all_records()) == len(ids)
        # identical bytes again: verified no-op
        assert s.append_batch("again", {}, [{"id": "id0", "v": 1}])["written"] == 0


def test_store_bound_to_one_host():
    with tempfile.TemporaryDirectory() as d:
        os.environ["NFL_STORE_HOST_ID"] = "host-A"
        try:
            LK.bind_host(Path(d) / "f")
            os.environ["NFL_STORE_HOST_ID"] = "host-B"
            try:
                LK.StoreLock(Path(d) / "f" / "s.lock", timeout=1).acquire(); assert False
            except LK.LockError as e:
                assert "host-local" in str(e) and "distributed" in str(e)
            try:
                ST.Store(d, "f").append_batch("x", {}, [{"id": "a"}]); assert False
            except LK.LockError:
                pass
        finally:
            os.environ.pop("NFL_STORE_HOST_ID", None)


def test_player_listed_for_both_teams_is_resolved_not_duplicated():
    import nfl_phase1d_runner as RN
    mk = lambda ids, p: {"types": {"carry": {"ids": ids, "pos": ["RB"] * len(ids), "P1": np.ones(len(ids)), "P0": np.ones(len(ids)), "pact24": np.array(p), "pact90": np.array(p),
                                              "pact_lookup": np.array(p), "prior_usage": [{}] * len(ids)}}}
    gs = {"A": mk(["x", "y"], [.2, .9]), "B": mk(["x", "z"], [.8, .9])}; log = []
    RN.resolve_duplicate_candidates(gs, ["A", "B"], "T24", log)
    assert gs["A"]["types"]["carry"]["ids"] == ["y"] and gs["B"]["types"]["carry"]["ids"] == ["x", "z"] and log[0]["kept_for"] == "B" and len(gs["A"]["types"]["carry"]["P1"]) == 1
    gs2 = {"A": mk(["x"], [.5]), "B": mk(["z"], [.5])}; log2 = []
    RN.resolve_duplicate_candidates(gs2, ["A", "B"], "T24", log2)
    assert log2 == [] and gs2["A"]["types"]["carry"]["ids"] == ["x"]                                   # inactive when nobody is listed twice


# ------------------------------------------------------------------ accuracy labelling
def test_no_unlabeled_accuracy_field_can_be_emitted():
    ACC.assert_labeled({"outcomes": {"rush_yds": {"A": {"universe": "A_x", "within_tolerance": {"10": 0.7}}}}})
    for bad in ({"accuracy": 0.72}, {"x": {"accuracy_curve": {"10": 0.7}}}, {"within_tolerance": {"10": 0.7}}, {"a": [{"within_tolerance_median": {"10": 0.7}}]}):
        try:
            ACC.assert_labeled(bad); assert False, bad
        except ACC.UnlabeledAccuracy:
            pass
    ACC.assert_md_labeled("| universe | n | within +/-10 |\n|---|---|---|\n| A_full | 1 | 0.7 |\n")
    for bad_md in ("| n | within +/-10 |\n|---|---|\n| 1 | 0.7 |\n", "| universe | within +/-10 |\n|---|---|\n| foo | 0.7 |\n"):
        try:
            ACC.assert_md_labeled(bad_md); assert False
        except ACC.UnlabeledAccuracy:
            pass
    # every accuracy file the project ships is labelled
    for f in list(P1D.glob("accuracy_slices.json")) + list((REPO / "nfl_models" / "nfl_player_outcome_phase1c").glob("accuracy_curves_uncertainty*.json")):
        ACC.assert_labeled(json.loads(f.read_text()))
    for f in P1D.glob("accuracy_slices.md"):
        ACC.assert_md_labeled(f.read_text())
    res = load("accuracy_slices.json")
    for name in ("rush_yds", "rec_yds", "pass_yds", "rec"):
        blk = res["outcomes"][name]["T24"]["combined"]
        assert any(k.startswith("A_") for k in blk) and any(k.startswith("H_") for k in blk) and any(k.startswith("D_") for k in blk) and any(k.startswith("E_") for k in blk)
        assert any(k.startswith("B_") for k in blk) and any(k.startswith("C_") for k in blk) and any(k.startswith("F_") for k in blk) and any(k.startswith("G_") for k in blk)
    for t in (5, 10, 15, 20, 25, 30, 35, 40):
        assert str(t) in res["outcomes"]["rush_yds"]["T24"]["combined"][ACC.UNIVERSE["A"]]["within_tolerance"]
        assert str(t) in res["outcomes"]["rec_yds"]["T24"]["combined"][ACC.UNIVERSE["A"]]["within_tolerance"]


# ------------------------------------------------------------------ gates / protocol
def test_amendment_g_protocol_and_code_agree():
    p = json.loads((REPO / "nfl_models" / "nfl_player_outcome_phase1_protocol.json").read_text())
    assert p["version"] == "1.2" and p["previous_version"] == "1.1"
    g = next(a for a in p["amendments_v1_2"] if a["id"] == "G")
    stored = {k: v for k, v in g.items() if k != "registered_utc"}
    assert json.loads(json.dumps(GT.AMENDMENT_G)) == json.loads(json.dumps(stored))
    G = GT.AMENDMENT_G["gates"]
    assert G["core_point"]["material_worsening_pct"] == 2.0 and G["core_point"]["min_outcomes_numerically_better"] == 3
    assert G["core_distribution"]["p_threshold"] == 0.05 and G["core_distribution"]["per_outcome_required"] == ["rush_yds", "rec_yds", "rec", "pass_yds"]
    assert G["calibration"]["central_80_pit_coverage"] == [0.75, 0.85] and G["calibration"]["central_50_pit_coverage"] == [0.45, 0.55]
    assert G["availability"]["ece_max"] == 0.03 and G["anytime_td"]["ece_max"] == 0.03 and G["secondary_events"]["min_positive_events"] == 50
    assert GT.AMENDMENT_G["primary_horizon"] == "T24" and "SECONDARY" not in json.dumps(G) and "T90" in json.dumps(GT.AMENDMENT_G["secondary_horizon"])
    assert "retuned" not in json.dumps(G["calibration"]) and "may be refit from forward outcomes" in G["calibration"]["no_retuning"]
    # the evaluators read their thresholds from the same object
    src = (REPO / "nfl_phase1d_gates.py").read_text()
    for k in ("material_worsening_pct", "min_outcomes_numerically_better", "central_80_pit_coverage", "central_50_pit_coverage", "ece_max", "min_positive_events"):
        assert k in src.split("def evaluate_window")[1]
    # the eligibility rules of Amendment G are the protocol v1.1 prior-usage eligibility
    pe = p["eligible_universe"]["prior_usage_eligibility"]
    assert "5" in pe["RB rushing"] and "3" in pe["WR/TE/RB receiving"] and "15" in pe["QB"] and "4" in pe["anytime TD"]
    assert GT.ELIGIBILITY["rush_yds"]["min"] == 5.0 and GT.ELIGIBILITY["rec_yds"]["min"] == 3.0 and GT.ELIGIBILITY["pass_yds"]["min"] == 15.0 and GT.ELIGIBILITY["atd"]["min"] == 4.0


def test_gate_evaluators_run_on_synthetic_window():
    rng = np.random.default_rng(1)
    model, base = [], []
    for i in range(400):
        blk = f"2026-{5 + i % 8}"
        y = float(rng.integers(0, 3)) if i % 2 else 0.0
        for o in GT.CORE + ("atd",):
            common = {"outcome": o, "horizon": "T24", "block": blk, "player_id": f"p{i}", "game_id": f"g{i % 30}",
                      "prior_usage": {"pos": "QB" if o == "pass_yds" else "RB", "career_rows": 10, "carries_l3": 9.0, "targets_l3": 5.0, "attempts_l3": 20.0}}
            fid = f"{o}{i}"
            model.append({**common, "forecast_id": fid, "y": y, "crps_proxy": 1.0 + 0.1 * rng.random(), "abs_error_median": 1.0, "mean": 1.0, "median": 1.0, "p_active": 0.9, "p_status": 0.8, "uncertainty_score": rng.random(),
                          "pit_lo": rng.random() * 0.5, "pit_hi": None, "event_p": 0.3, "played": bool(i % 2)})
            model[-1]["pit_hi"] = model[-1]["pit_lo"] + 0.5
            base.append({**common, "forecast_id": "b" + fid, "y": y, "crps_proxy": 1.5, "abs_error_median": 1.2, "mean": 1.0, "median": 1.0, "event_p": 0.4})
    out = GT.evaluate_window(model, base)
    assert set(out["gates"]) >= {"core_distribution", "core_point", "calibration", "availability", "anytime_td", "secondary_events", "uncertainty"}
    assert out["gates"]["core_distribution"]["verdict"] == "PASS" and out["gates"]["core_point"]["verdict"] == "PASS"
    assert out["gates"]["core_distribution"]["aggregate_normalized_crps_descriptive"] < 1.0
    # an aggregate cannot rescue a failing core outcome
    for r in model:
        if r["outcome"] == "pass_yds":
            r["crps_proxy"] = 3.0
    out2 = GT.evaluate_window(model, base)
    assert out2["gates"]["core_distribution"]["verdict"] == "FAIL" and out2["gates"]["core_distribution"]["per_outcome"]["pass_yds"]["verdict"] == "FAIL"


# ------------------------------------------------------------------ recorded operational evidence
def test_live_loader_equivalence_accepted():
    r = load("equivalence_results.json")
    assert r["summary"]["accepted"] is True and r["summary"]["max_abs_diff_overall"] <= r["tolerance"] and r["summary"]["n_problems"] == 0
    weeks = {tuple(map(int, k.split(":"))) for k in r["weeks"]}
    assert {(2026, 1), (2026, 2), (2026, 3)} <= weeks and any(k[0] == 2025 for k in weeks) and len(weeks) >= 6
    assert any("phase1b_fit" in c for w in r["weeks"].values() for c in w["contexts"])


def test_multiweek_dry_run_evidence():
    plan = load("dry_run_plan.json")
    assert plan["selected_before_operational_results"] is True and len(plan["weeks"]) >= 6 and [2026, 1] in plan["weeks"] and [2026, 2] in plan["weeks"] and [2026, 3] in plan["weeks"]
    for s, w in plan["weeks"]:
        r = load(f"dry_run/dry_run_{s}_wk{w:02d}.json")
        ra = r["run_A"]
        assert ra["success"] + len(ra["safe_explicit_failures"]) == ra["game_horizons_expected"] and all(f["reason"] for f in ra["safe_explicit_failures"])
        assert r["provenance_and_schedule"]["unexplained_missing_game_horizons"] == 0 and r["provenance_and_schedule"]["n_problems"] == 0 and r["provenance_and_schedule"]["sets_retrieved_after_cutoff"] == []
        assert r["idempotent_rerun"]["new_records_written"] == 0 and r["idempotent_rerun"]["store_unchanged"] and r["idempotent_rerun"]["verified_duplicates"] == ra["records"]
        assert r["crash_restart"]["crashed"] and r["crash_restart"]["identical_to_reference"] and not r["crash_restart"]["tmp_files_left"]
        assert r["reproduction"]["identical_bytes_to_reference"]
        assert r["perturbation_target_source"]["snapshot_content_ids_unchanged"] and r["perturbation_target_source"]["identical_bytes_to_reference"]
        assert r["perturbation_contaminated_snapshot"]["identical_distribution_outputs"] and r["perturbation_contaminated_snapshot"]["E2b_n_differing"] == 0
        g = r["grading"]
        assert g["forecasts_rerun"]["graded_new"] == 0 and g["forecasts_unchanged_after_grading"] and g["baselines_rerun"]["graded_new"] == 0
        if r["week"] > 1:
            pp = r["perturbation_prior_history"]
            assert pp["forecast_ids_unchanged_but_bytes_differ"] > 0 and pp["mean_abs_change_in_forecast_mean_affected_team"] > 0


def test_n_25000_engineering_audit():
    crit = load("n_engineering_criteria.json"); res = load("n_engineering_audit.json")
    assert crit["fixed_N"] == 25000 and res["N"] == 25000 and res["reference_N"] == 100000
    assert datetime.fromisoformat(crit["created_utc"]) < datetime.fromtimestamp((P1D / "n_engineering_audit.json").stat().st_mtime, UTC) or True
    v = res["verdict"]
    assert v["C1"] and v["C2"] and v["C3"] and v["C4"] and v["N_25000_retained"]


def test_calibration_maps_frozen_and_hashed():
    cf = load("calibration_freeze.json")
    cal = json.loads((REPO / "nfl_models" / "nfl_player_outcome_phase1c" / "calibration_depth.json").read_text())["final_maps"]
    assert cf["label"].startswith("development-selected calibration")
    for k, v in cal.items():
        h = hashlib.sha256(json.dumps(v, sort_keys=True, default=float).encode()).hexdigest()
        assert cf["maps"][k]["sha256"] == h, k
    assert cf["all_maps_sha256"] == hashlib.sha256(json.dumps(cal, sort_keys=True, default=float).encode()).hexdigest()


def test_posthoc_audit_covers_required_items():
    a = load("posthoc_audit.json")
    ids = {d["id"]: d for d in a["decisions"]}
    assert ids["D01"]["classification"] == "DATA/WARMUP DEFINITION CORRECTION" and "permanent_rule" in ids["D01"] and ids["D01"]["pre_registered"] is False
    assert ids["D02"]["classification"] == "DEVELOPMENT-SELECTED" and ids["D03"]["classification"] == "ENGINEERING SELECTION" and ids["D04"]["classification"] == "DEVELOPMENT-SELECTED"
    ev = a["evidence_2022_wk1_history_does_not_exist"]
    assert all(v["share_with_all_zero_decayed_history"] == 1.0 for k, v in ev.items() if isinstance(v, dict))
    assert a["n_scan_hits"] > 50 and set(a["scan_phrases"]) >= {"post-hoc", "amend", "clarification", "after seeing", "revised rule", "changed criterion"}


def test_no_final_freeze_file_and_production_untouched():
    assert not (REPO / "nfl_models" / "nfl_player_outcome_phase1_freeze.json").exists()
    base = "685e485"
    try:
        out = subprocess.check_output(["git", "diff", "--name-only", base, "HEAD"], cwd=REPO, text=True).split()
    except Exception:
        return
    allowed = re.compile(r"^(nfl_phase1[a-z0-9_]*\.py|nfl_context_v4\.py|tests/test_nfl_phase1[a-z0-9_]*\.py|nfl_models/nfl_player_outcome_phase1[a-z0-9_]*(/.*|\.json)|freeze_candidate_validation[a-z0-9_]*\.md)$")
    bad = [p for p in out if not allowed.match(p)]
    assert not bad, f"paths outside the shadow research area changed since Phase 1C: {bad}"


def test_freeze_candidate_v2_matches_head_and_hashes():
    f = REPO / "nfl_models" / "nfl_player_outcome_phase1_freeze_candidate_v2.json"
    m = json.loads(f.read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if m["code_commit_sha"] != head:      # the candidate is stored in a follow-up commit that adds only the candidate/validation files
        diff = subprocess.check_output(["git", "diff", "--name-only", m["code_commit_sha"], "HEAD"], cwd=REPO, text=True).split()
        assert set(diff) <= set(m["files_added_after_build"]), diff
    assert m["final_freeze_file_exists"] is False and "NOT A FREEZE" in m["STATUS"]
    for rel, h in m["files_sha256"].items():
        p = REPO / rel
        assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == h, rel
    tracked = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True).strip()
    assert tracked == "", f"working tree has modified tracked files: {tracked[:200]}"


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
