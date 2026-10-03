"""NHL Phase 0C tests: durable forward-capture state, registered timing, grouping, postgame truth, evaluation rule, readiness. Injected clock / sleep / http; nothing waits in real time. python tests/test_nhl_phase0c.py"""
import gzip
import json
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import nhl_fwd_capture as C  # noqa: E402
import nhl_fwd_eval as EV  # noqa: E402
import nhl_fwd_scheduler as SC  # noqa: E402
import nhl_fwd_state as S  # noqa: E402

UTC = timezone.utc
START = datetime(2026, 10, 10, 23, 0, tzinfo=UTC)
API = C.API


def iso_z(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


class Clock:
    def __init__(self, t):
        self.t = t
        self.lock = threading.Lock()

    def __call__(self):
        return self.t

    def advance(self, sec):
        with self.lock:
            self.t = self.t + timedelta(seconds=sec)


class World:
    """Fake NHL provider. games: {id: {"start": dt, "state": "FUT", "date": "2026-10-10"}}; every http call advances the clock by `latency` seconds and is logged with its START time."""

    def __init__(self, clock, games, latency=1.0):
        self.clock, self.games, self.latency, self.calls, self.fail = clock, games, latency, [], set()
        self.lock = threading.Lock()
        self.skaters = {}

    def schedule_bytes(self):
        days = {}
        for gid, g in sorted(self.games.items()):
            days.setdefault(g.get("date", "2026-10-10"), []).append({"id": gid, "gameType": 2, "gameState": g.get("state", "FUT"), "startTimeUTC": iso_z(g["start"]), "awayTeam": {"abbrev": f"A{gid % 100}"}, "homeTeam": {"abbrev": f"H{gid % 100}"}})
        return json.dumps({"gameWeek": [{"date": d, "games": gs} for d, gs in sorted(days.items())]}).encode()

    def body(self, gid, ep):
        g = self.games[gid]
        state = g.get("state", "FUT")
        ids = self.skaters.get(gid, [gid * 100 + i for i in range(1, 37)])
        if ep == "landing":
            return {"gameState": state, "matchup": {"goalieComparison": {"x": 1}}}
        if ep == "right-rail":
            return {"gameInfo": {"awayTeam": {"scratches": [{"id": gid * 100 + 90}]}, "homeTeam": {"scratches": []}, "referees": []}}
        if ep == "boxscore":
            half = len(ids) // 2
            mk = lambda lst: {"forwards": [{"playerId": i, "sog": 1} for i in lst[:max(1, len(lst) * 2 // 3)]], "defense": [{"playerId": i, "sog": 0} for i in lst[max(1, len(lst) * 2 // 3):]], "goalies": [{"playerId": gid * 100 + 98, "starter": True}]}
            return {"gameState": state, "awayTeam": {"abbrev": "AAA", "sog": half}, "homeTeam": {"abbrev": "HHH", "sog": len(ids) - half}, "playerByGameStats": {"awayTeam": mk(ids[:half]), "homeTeam": mk(ids[half:])}}
        if ep == "play-by-play":
            return {"gameState": state, "plays": [], "rosterSpots": [{"playerId": i, "positionCode": "C"} for i in ids] + [{"playerId": gid * 100 + 98, "positionCode": "G"}]}
        raise KeyError(ep)

    def http(self, url):
        started = self.clock()
        with self.lock:
            self.calls.append((url, started))
        self.clock.advance(self.latency)
        if url in self.fail:
            raise OSError("provider down")
        if url.endswith("schedule/now"):
            return self.schedule_bytes(), {"ETag": "s"}, 200
        gid = int(url.split("/gamecenter/")[1].split("/")[0]); ep = url.rsplit("/", 1)[1]
        return json.dumps(self.body(gid, ep), sort_keys=True).encode(), {"ETag": "e"}, 200


def seq_map(fn, items, workers=1):
    return [fn(x) for x in items]


def setup(t, games, now, latency=1.0, map_fn=seq_map):
    clk = Clock(now)
    w = World(clk, games, latency)
    sleeps = []

    def sleep(sec):
        sleeps.append(sec); clk.advance(sec)
    r = SC.Runner(t, http=w.http, clock=clk, sleep=sleep, map_fn=map_fn, log=lambda m: None)
    return clk, w, r, sleeps


def keyrows(st, hz=None):
    return [p for p in st.planned.read() if hz is None or p["horizon"] == hz]


def result_of(st, gid, hz):
    r = [x for x in st.results.read() if x["game_id"] == gid and x["horizon"] == hz]
    return r[-1] if r else None


# ------------------------------------------------------------------ state: CAS + ledger
def test_gzip_blob_roundtrip_identity_and_tamper_detection():
    with tempfile.TemporaryDirectory() as t:
        b = S.BlobStore(t)
        raw = b'{"a": 1, "b": [1,2,3]}' * 50
        info = b.put(raw)
        assert info["sha256"] == S.sha256_hex(raw) and info["stored"] == "new" and b.put(raw)["stored"] == "existing"
        p = b.path(info["sha256"])
        assert p.name == info["sha256"] + ".gz" and p.read_bytes()[:2] == b"\x1f\x8b" and gzip.decompress(p.read_bytes()) == raw and b.get(info["sha256"]) == raw
        assert len(list(b.dir.glob("*.gz"))) == 1 and not b.verify_all()
        p.write_bytes(S.det_gzip(b"different bytes"))                                         # same sha name, different bytes
        for fn in (lambda: b.get(info["sha256"]), lambda: b.put(raw)):
            try:
                fn(); assert False
            except S.HardError:
                pass
        assert b.verify_all()
        try:
            b.put_as("0" * 64, raw); assert False
        except S.HardError:
            pass


def test_ledger_same_key_different_bytes_is_hard_error_and_duplicates_are_skipped():
    with tempfile.TemporaryDirectory() as t:
        L = S.Ledger(t, "x.jsonl")
        assert L.append_unique(["key"], {"key": "a", "sha": "1"}, immutable=("sha",))[1] is True
        assert L.append_unique(["key"], {"key": "a", "sha": "1"}, immutable=("sha",))[1] is False and len(L.read()) == 1
        try:
            L.append_unique(["key"], {"key": "a", "sha": "2"}, immutable=("sha",)); assert False
        except S.HardError:
            pass


# ------------------------------------------------------------------ schedule / planning
def test_same_schedule_same_identity_and_duplicate_invocation_adds_nothing():
    with tempfile.TemporaryDirectory() as t:
        clk, w, r, _ = setup(t, {2026020100: {"start": START}}, START - timedelta(days=3))
        a = C.observe_schedule(r.st, w.http, clk); b = C.observe_schedule(r.st, w.http, clk)
        assert a["schedule_sha256"] == b["schedule_sha256"]
        n1, _ = C.plan_keys(r.st, a, clk()); n2, _ = C.plan_keys(r.st, b, clk())
        assert len(n1) == 5 and n2 == [] and len(keyrows(r.st)) == 5                                      # one key per horizon; the duplicate adds none
        assert r.st.blobs.get(a["schedule_sha256"]) == w.schedule_bytes()                                  # the exact raw schedule bytes are stored
        for k in keyrows(r.st):
            assert k["schedule_sha256"] == a["schedule_sha256"] and k["schedule_retrieval_started_at"] and k["schedule_retrieval_completed_at"]
            assert k["key"] == f"{k['game_id']}|{k['horizon']}|{k['scheduled_start_utc']}" and k["cutoff"] == C.iso(START - timedelta(minutes=C.HORIZONS[k["horizon"]]))
        r.tick(); n_rows = (len(r.st.planned.read()), len(r.st.obs.read()), len(r.st.results.read()))
        r.tick()
        assert n_rows == (len(r.st.planned.read()), len(r.st.obs.read()), len(r.st.results.read()))      # duplicate invocation: no duplicate records


def test_schedule_revision_creates_new_decision_key_old_evidence_immutable():
    with tempfile.TemporaryDirectory() as t:
        gid = 2026020101
        clk, w, r, _ = setup(t, {gid: {"start": START}}, START - timedelta(days=3))
        r.tick()
        old = {k["key"]: k for k in keyrows(r.st)}
        w.games[gid]["start"] = START + timedelta(hours=3)                                               # the league flexes the game later
        r.tick()
        new = [k for k in keyrows(r.st) if k["key"] not in old]
        assert len(new) == 5 and all(k["scheduled_start_utc"] == C.iso(START + timedelta(hours=3)) for k in new)
        for k in old.values():
            assert result_of(r.st, gid, k["horizon"]) and any(x["status"] == C.REVISED for x in r.st.results.read() if x["key"] == k["key"])
            assert [x for x in r.st.planned.read() if x["key"] == k["key"]][0] == k                       # old key row unchanged
        assert all(k["schedule_sha256"] != next(iter(old.values()))["schedule_sha256"] for k in new)      # the new key points at the new schedule bytes


# ------------------------------------------------------------------ timing semantics
def test_wait_inside_job_then_capture_in_the_registered_window_and_complete_valid():
    with tempfile.TemporaryDirectory() as t:
        gid = 2026020102
        cutoff = START - timedelta(minutes=90)
        clk, w, r, sleeps = setup(t, {gid: {"start": START}}, cutoff - timedelta(minutes=15))
        # plan first (keys are planned on the first observation), then tick again at the same moment
        r.tick()
        res = result_of(r.st, gid, "T90")
        assert res["status"] == C.COMPLETE_VALID and sum(sleeps) >= 12 * 60 and max(sleeps) <= SC.WAIT_STEP_S                      # waited (simulated) inside the job
        obs = [o for o in r.st.obs.read() if o["key"] == res["key"]]
        assert [o["endpoint"] for o in obs] == list(C.REQUIRED_ENDPOINTS)
        for o in obs:
            started, done = C.parse_iso(o["retrieval_started_at"]), C.parse_iso(o["retrieval_completed_at"])
            assert o["observation_status"] == C.VALID and started < cutoff and cutoff - timedelta(seconds=180) <= done <= cutoff
            assert cutoff - timedelta(seconds=C.CAPTURE_START_LEAD_SECONDS) - timedelta(seconds=1) <= started                        # capture began ~120 s before the cutoff
            assert o["schedule_sha256"] and o["source_url"].endswith(o["endpoint"]) and o["raw_sha256"] and r.st.blobs.get(o["raw_sha256"])
            assert o["seconds_before_cutoff"] >= 0


def test_early_snapshot_is_never_horizon_evidence_and_late_completion_is_never_valid():
    start, cutoff = START, START - timedelta(minutes=90)
    krow = {"key": "k", "game_id": 1, "horizon": "T90", "scheduled_start_utc": C.iso(start), "cutoff": C.iso(cutoff), "schedule_sha256": "s", "schedule_retrieval_started_at": "a", "schedule_retrieval_completed_at": "b"}
    for t0, lat, want in ((cutoff - timedelta(minutes=10), 1, C.EARLY), (cutoff - timedelta(seconds=181) - timedelta(seconds=1), 1, C.EARLY), (cutoff - timedelta(seconds=100), 1, C.VALID),
                          (cutoff - timedelta(seconds=180), 0.0, C.VALID), (cutoff - timedelta(seconds=3), 10, C.MISSED), (cutoff + timedelta(seconds=1), 1, C.MISSED)):
        with tempfile.TemporaryDirectory() as t:
            clk = Clock(t0)
            w = World(clk, {1: {"start": start}}, lat)
            st = C.State(t)
            row, _ = C.capture_endpoint(st, krow, "landing", w.http, clk)
            assert row["observation_status"] == want, (t0, lat, want, row["observation_status"])
            if want == C.MISSED and t0 >= cutoff:
                assert w.calls == []                                                              # not initiated after the cutoff
    with tempfile.TemporaryDirectory() as t:                                                       # an early / late key can never be COMPLETE_VALID
        clk = Clock(cutoff - timedelta(minutes=10)); w = World(clk, {1: {"start": start}}); st = C.State(t)
        res = C.capture_key(st, krow, w.http, clk)
        assert res["status"] == C.EARLY and res["status"] != C.COMPLETE_VALID


def test_no_endpoint_request_starts_at_or_after_puck_drop():
    for off in (0, 1, 600):
        with tempfile.TemporaryDirectory() as t:
            clk = Clock(START + timedelta(seconds=off)); w = World(clk, {1: {"start": START}}); st = C.State(t)
            krow = {"key": "k", "game_id": 1, "horizon": "T2", "scheduled_start_utc": C.iso(START), "cutoff": C.iso(START - timedelta(minutes=2)), "schedule_sha256": "s", "schedule_retrieval_started_at": "a", "schedule_retrieval_completed_at": "b"}
            res = C.capture_key(st, krow, w.http, clk)
            assert w.calls == [] and res["status"] == C.REFUSED and all(o["observation_status"] == C.REFUSED for o in st.obs.read())


def test_missed_cutoff_is_never_backfilled():
    with tempfile.TemporaryDirectory() as t:
        gid = 2026020103
        cutoff = START - timedelta(minutes=90)
        clk, w, r, _ = setup(t, {gid: {"start": START}}, START - timedelta(days=3))
        r.tick()                                                                                    # keys planned, nothing due
        clk.t = cutoff + timedelta(minutes=5)                                                       # the runner was down across the T90 cutoff
        r.tick()
        res = result_of(r.st, gid, "T90")
        assert res["status"] == C.MISSED and res["reason"] == "cutoff_passed_before_capture_started"
        assert not [o for o in r.st.obs.read() if o["horizon"] == "T90"]                           # nothing was fetched for it, now or later
        n = len(w.calls)
        clk.t = START - timedelta(minutes=20); r.tick()
        assert result_of(r.st, gid, "T90")["status"] == C.MISSED and not [o for o in r.st.obs.read() if o["horizon"] == "T90"]
        # a game first seen after a cutoff: planned, immediately MISSED (never captured later)
    with tempfile.TemporaryDirectory() as t:
        clk, w, r, _ = setup(t, {gid: {"start": START}}, START - timedelta(hours=20))
        r.tick()
        assert result_of(r.st, gid, "T24H")["status"] == C.MISSED and result_of(r.st, gid, "T24H")["reason"] == "first_seen_after_cutoff"


# ------------------------------------------------------------------ restart safety
def test_death_after_blob_write_and_after_metadata_write_then_restart_reuses_verified_bytes():
    with tempfile.TemporaryDirectory() as t:
        gid = 2026020104
        cutoff = START - timedelta(minutes=90)
        clk = Clock(cutoff - timedelta(seconds=100)); w = World(clk, {gid: {"start": START}}); st = C.State(t)
        krow = {"key": f"{gid}|T90|{C.iso(START)}", "game_id": gid, "horizon": "T90", "scheduled_start_utc": C.iso(START), "cutoff": C.iso(cutoff), "schedule_sha256": "s", "schedule_retrieval_started_at": "a", "schedule_retrieval_completed_at": "b"}
        # (1) process death AFTER the blob write, BEFORE the metadata write of the 3rd endpoint
        n = {"i": 0}
        orig = st.obs.append_unique

        def dying(*a, **k):
            n["i"] += 1
            if n["i"] == 3:
                raise SystemExit("process death after blob write")
            return orig(*a, **k)
        st.obs.append_unique = dying
        try:
            C.capture_key(st, krow, w.http, clk); assert False
        except SystemExit:
            pass
        assert len(st.obs.read()) == 2 and len(list(st.blobs.dir.glob("*.gz"))) == 3                      # 2 metadata rows, 3 blobs (the 3rd has no row)
        calls_before = len(w.calls)
        # restart: a new process / state object
        st2 = C.State(t)
        res = C.capture_key(st2, krow, w.http, clk)
        assert res["status"] == C.COMPLETE_VALID and len(w.calls) == calls_before + 2                    # the two recorded endpoints were REUSED (verified), only the 2 missing ones fetched
        assert len(st2.obs.read()) == 4 and len(list(st2.blobs.dir.glob("*.gz"))) == 4                    # the re-fetched endpoint deduplicated onto the same blob
        # (2) death after the metadata write: a corrupted stored blob is detected on reuse
    with tempfile.TemporaryDirectory() as t:
        clk = Clock(cutoff - timedelta(seconds=100)); w = World(clk, {gid: {"start": START}}); st = C.State(t)
        C.capture_endpoint(st, krow, "landing", w.http, clk)
        sha = st.obs.read()[0]["raw_sha256"]
        st.blobs.path(sha).write_bytes(S.det_gzip(b"tampered"))
        try:
            C.capture_endpoint(C.State(t), krow, "landing", w.http, clk); assert False
        except S.HardError:
            pass


def test_restart_after_cutoff_with_partial_capture_finalizes_without_backfill():
    with tempfile.TemporaryDirectory() as t:
        gid = 2026020105
        cutoff = START - timedelta(minutes=90)
        clk = Clock(cutoff - timedelta(seconds=100)); w = World(clk, {gid: {"start": START}}); st = C.State(t)
        krow = {"key": f"{gid}|T90|{C.iso(START)}", "game_id": gid, "horizon": "T90", "scheduled_start_utc": C.iso(START), "cutoff": C.iso(cutoff), "schedule_sha256": "s", "schedule_retrieval_started_at": "a", "schedule_retrieval_completed_at": "b"}
        C.capture_endpoint(st, krow, "landing", w.http, clk); C.capture_endpoint(st, krow, "right-rail", w.http, clk)
        clk.t = cutoff + timedelta(minutes=3)                                                      # the process died; restart after the cutoff
        res = C.capture_key(C.State(t), krow, w.http, clk)
        eps = {o["endpoint"]: o["observation_status"] for o in C.State(t).obs.read()}
        assert res["status"] == C.MISSED and eps["landing"] == C.VALID and eps["right-rail"] == C.VALID and eps["boxscore"] == C.MISSED and eps["play-by-play"] == C.MISSED


# ------------------------------------------------------------------ concurrency / grouping
def test_simultaneous_games_are_captured_as_one_group_without_merging_identities():
    with tempfile.TemporaryDirectory() as t:
        games = {2026020200 + i: {"start": START} for i in range(10)}
        cutoff = START - timedelta(minutes=90)
        clk, w, r, sleeps = setup(t, games, cutoff - timedelta(minutes=10), latency=0.01, map_fn=C.default_map)       # real threads
        r.tick()
        groups = C.due_groups(r.st, clk(), wake_seconds=-10 ** 9)                                  # nothing is open at T90 any more
        res = [x for x in r.st.results.read() if x["horizon"] == "T90"]
        assert len(res) == 10 and all(x["status"] == C.COMPLETE_VALID for x in res)
        obs = [o for o in r.st.obs.read() if o["horizon"] == "T90"]
        assert len(obs) == 40
        by = {}
        for o in obs:
            assert str(o["game_id"]) in o["source_url"] and o["key"].startswith(f"{o['game_id']}|T90|")
            by.setdefault((o["game_id"], o["endpoint"]), set()).add(o["raw_sha256"])
        assert all(len(v) == 1 for v in by.values()) and len({next(iter(v)) for (g, e), v in by.items() if e == "boxscore"}) == 10          # distinct games -> distinct bytes -> distinct identities
        assert sum(sleeps) >= 7 * 60                                                                # a single wait for the whole group, not one per game
        started = sorted(C.parse_iso(o["retrieval_started_at"]) for o in obs)
        assert started[-1] - started[0] < timedelta(seconds=C.CAPTURE_START_LEAD_SECONDS)           # all games started inside the capture window together


def test_default_map_uses_at_most_eight_workers():
    seen = {}
    import concurrent.futures as cf
    orig = C.ThreadPoolExecutor

    class Spy(orig):
        def __init__(self, max_workers=None, **k):
            seen["w"] = max_workers
            super().__init__(max_workers=max_workers, **k)
    C.ThreadPoolExecutor = Spy
    try:
        assert C.default_map(lambda x: x * 2, range(20)) == [x * 2 for x in range(20)]
    finally:
        C.ThreadPoolExecutor = orig
    assert seen["w"] == 8 and C.MAX_WORKERS == 8


# ------------------------------------------------------------------ postgame truth
def test_postgame_truth_is_labelled_idempotent_and_never_a_pregame_snapshot():
    with tempfile.TemporaryDirectory() as t:
        gid = 2026020106
        clk, w, r, _ = setup(t, {gid: {"start": START}}, START - timedelta(days=3))
        r.tick()
        clk.t = START + timedelta(hours=4); w.games[gid]["state"] = "OFF"; w.skaters[gid] = [gid * 100 + i for i in range(1, 37)]
        r.tick()
        pg = r.st.postgame.read()
        assert len(pg) == 1 and pg[0]["label"] == "POSTGAME_TRUTH" and pg[0]["never_a_pregame_feature"] is True and set(pg[0]["endpoints"]) == set(C.POSTGAME_ENDPOINTS)
        assert sum(len(v) for v in pg[0]["final_dressed_skater_ids_by_team"].values()) == 36 and sum(pg[0]["final_sog_by_skater"].values()) == 24 and set(pg[0]["final_team_sog"]) == {"AAA", "HHH"}
        assert pg[0]["final_goalie_ids_by_team"]["AAA"]
        calls = len(w.calls)
        r.tick(); assert len(r.st.postgame.read()) == 1 and not [c for c in w.calls[calls:] if "/boxscore" in c[0]]                  # idempotent: not captured again
        assert all(o.get("label") == "PREGAME_HORIZON_CAPTURE" for o in r.st.obs.read()) and not any(x["key"].startswith(str(gid)) and x["status"] == "POSTGAME_TRUTH" for x in r.st.results.read())
        # a pregame horizon capture of a FINAL game is rejected, not stored as horizon evidence
        krow = {"key": "k2", "game_id": gid, "horizon": "T2", "scheduled_start_utc": C.iso(START + timedelta(days=1)), "cutoff": C.iso(START + timedelta(days=1) - timedelta(minutes=2)), "schedule_sha256": "s", "schedule_retrieval_started_at": "a",
                "schedule_retrieval_completed_at": "b"}
        clk.t = START + timedelta(days=1) - timedelta(seconds=200)
        row, _ = C.capture_endpoint(r.st, krow, "boxscore", w.http, clk)
        assert row["observation_status"] == C.NOT_PREGAME
    with tempfile.TemporaryDirectory() as t:                                                       # not final yet -> pending, nothing recorded
        gid = 2026020107
        clk, w, r, _ = setup(t, {gid: {"start": START}}, START + timedelta(hours=4)); w.games[gid]["state"] = "LIVE"
        r.st.planned.append({"key": "x", "game_id": gid, "horizon": "T2", "scheduled_start_utc": C.iso(START), "cutoff": C.iso(START - timedelta(minutes=2)), "game_date": "2026-10-10", "schedule_sha256": "s",
                             "schedule_retrieval_started_at": "a", "schedule_retrieval_completed_at": "b"})
        out = r.postgame_pass()
        assert r.st.postgame.read() == [] and out[0]["status"] == "POSTGAME_PENDING"


# ------------------------------------------------------------------ evaluation + registered recommendation rule
def synth_state(t, n_games=50, n_dates=3, recall_by_horizon=None, valid_per_horizon=None, late_games=0):
    """Directly populate a state with truth + COMPLETE_VALID observations for every horizon (no model, no network)."""
    st = C.State(t)
    recall_by_horizon = recall_by_horizon or {}
    for gi in range(n_games):
        gid = 2026030000 + gi
        date = f"2026-10-{10 + gi % n_dates:02d}"
        sk = [gid * 100 + i for i in range(1, 37)]
        truth = {"game_id": gid, "label": "POSTGAME_TRUTH", "never_a_pregame_feature": True, "game_date": date, "scheduled_start_utc": "2026-10-10T23:00:00.000000Z", "endpoints": {},
                 "final_dressed_skater_ids_by_team": {"AAA": sk[:18], "HHH": sk[18:]}, "final_goalie_ids_by_team": {"AAA": [gid * 100 + 98], "HHH": [gid * 100 + 99]}, "final_sog_by_skater": {str(i): 1 for i in sk}, "final_team_sog": {"AAA": 18, "HHH": 18}}
        st.postgame.append(truth)
        for hz in C.HORIZON_ORDER:
            key = f"{gid}|{hz}|2026-10-10T23:00:00.000000Z"
            cutoff = C.parse_iso("2026-10-10T23:00:00.000000Z") - timedelta(minutes=C.HORIZONS[hz])
            late = gi >= n_games - late_games                                                          # enrolled after cutoff - WAKE_LEAD_SECONDS
            planned_at = cutoff - timedelta(seconds=C.WAKE_LEAD_SECONDS + (-5 if late else 5))
            st.planned.append({"key": key, "game_id": gid, "horizon": hz, "scheduled_start_utc": "2026-10-10T23:00:00.000000Z", "cutoff": C.iso(cutoff), "game_date": date, "schedule_sha256": "s", "planned_at": C.iso(planned_at)})
            if valid_per_horizon is not None and gi >= valid_per_horizon.get(hz, 10 ** 9):
                st.results.append({"key": key, "game_id": gid, "horizon": hz, "status": C.MISSED}); continue                       # zero observations
            rec = recall_by_horizon.get(hz, 1.0)
            keep = sk[: int(round(len(sk) * rec))]
            blobs = {"play-by-play": {"gameState": "FUT", "rosterSpots": [{"playerId": i, "positionCode": "C"} for i in keep] + [{"playerId": 999, "positionCode": "C"}, {"playerId": gid * 100 + 98, "positionCode": "G"}, {"playerId": gid * 100 + 99, "positionCode": "G"}]},
                     "boxscore": {"gameState": "FUT", "playerByGameStats": {}}, "right-rail": {"gameInfo": {"awayTeam": {"scratches": [{"id": 5}]}, "homeTeam": {"scratches": []}}}, "landing": {"gameState": "FUT"}}
            for ep, body in blobs.items():
                info = st.blobs.put(json.dumps(body, sort_keys=True).encode())
                st.obs.append({"key": key, "endpoint": ep, "raw_sha256": info["sha256"], "horizon": hz, "observation_status": C.VALID})
            st.results.append({"key": key, "game_id": gid, "horizon": hz, "status": C.COMPLETE_VALID})
    return st


def test_evaluation_not_run_before_minimum_sample_and_no_recommendation():
    with tempfile.TemporaryDirectory() as t:
        st = synth_state(t, n_games=49, n_dates=3)
        ev = EV.evaluate(st)
        assert ev["evaluation"] == "NOT_EVALUATED_INSUFFICIENT_SAMPLE" and ev["recommended_horizon"] is None and ev["status"] == EV.BLOCKER
    with tempfile.TemporaryDirectory() as t:
        st = synth_state(t, n_games=60, n_dates=2)
        assert EV.evaluate(st)["status"] == EV.BLOCKER and EV.evaluate(st)["evaluation"] == "NOT_EVALUATED_INSUFFICIENT_SAMPLE"          # 3 distinct dates required
    with tempfile.TemporaryDirectory() as t:
        assert EV.evaluate(C.State(t))["status"] == EV.BLOCKER


def test_recommendation_rule_earliest_qualifying_horizon_and_thresholds():
    with tempfile.TemporaryDirectory() as t:
        st = synth_state(t, n_games=60, n_dates=3)                                                  # every horizon is perfect
        ev = EV.evaluate(st)
        assert ev["evaluation"] == "EVALUATED" and ev["qualifying_horizons"] == C.HORIZON_ORDER and ev["recommended_horizon"] == "T24H" and ev["status"] == "HORIZON_RECOMMENDED"
        w = ev["horizons"]["T90"]["with_truth"]["pbp_rosterSpots"]
        assert w["final_dressed_skater_recall"] == 1.0 and w["extra_players_total"] == 60 and w["goalie_id_recall"] == 1.0 and w["precision"] < 1.0 and w["jaccard"] < 1.0 and w["final_sog_coverage"] == 1.0
        rl = ev["horizons"]["T90"]["with_truth"]["right_rail"]
        assert rl["scratch_list_availability"] == 1.0 and rl["mean_scratches_when_available"] == 1.0
    with tempfile.TemporaryDirectory() as t:
        st = synth_state(t, n_games=60, n_dates=3, recall_by_horizon={"T24H": 0.95, "T90": 0.97})  # early horizons miss players -> the first qualifying is T30
        ev = EV.evaluate(st)
        assert ev["qualifying_horizons"] == ["T30", "T10", "T2"] and ev["recommended_horizon"] == "T30"
    with tempfile.TemporaryDirectory() as t:
        st = synth_state(t, n_games=60, n_dates=3, valid_per_horizon={"T24H": 39})                  # only 39 COMPLETE_VALID at T24H -> not eligible
        ev = EV.evaluate(st)
        assert "T24H" not in ev["qualifying_horizons"] and ev["recommended_horizon"] == "T90"
    with tempfile.TemporaryDirectory() as t:
        st = synth_state(t, n_games=60, n_dates=3, recall_by_horizon={h: 0.9 for h in C.HORIZON_ORDER})
        ev = EV.evaluate(st)
        assert ev["recommended_horizon"] is None and ev["status"] == EV.BLOCKER and ev["qualifying_horizons"] == []


def test_capture_success_rate_uses_eligible_opportunities_not_attempted_only():
    with tempfile.TemporaryDirectory() as t:                                                       # 60 eligible, 50 COMPLETE_VALID, 10 zero-observation MISSED => 50/60, no qualification
        st = synth_state(t, n_games=60, n_dates=3, valid_per_horizon={h: 50 for h in C.HORIZON_ORDER})
        ev = EV.evaluate(st)
        for hz in C.HORIZON_ORDER:
            h = ev["horizons"][hz]
            assert h["eligible_opportunities"] == 60 and h["eligible_complete_valid"] == 50 and abs(h["capture_success_rate"] - 50 / 60) < 1e-12 and h["capture_success_rate"] < 0.99
            assert h["games_attempted"] == 50 and h["attempted_only_success_rate_diagnostic"] == 1.0 and h["eligible_failure_breakdown"] == {C.MISSED: 10}     # the old attempted-only denominator would have read 50/50 = 1.0
        assert ev["evaluation"] == "EVALUATED" and ev["qualifying_horizons"] == [] and ev["recommended_horizon"] is None and ev["status"] == EV.BLOCKER
    with tempfile.TemporaryDirectory() as t:                                                       # all non-VALID statuses are failures; revised keys are excluded; no result at all is a failure
        st = synth_state(t, n_games=60, n_dates=3)
        keys = [p for p in st.planned.read() if p["horizon"] == "T90"]
        st.results.path.write_text("".join(json.dumps(r) + "\n" for r in st.results.read() if not (r["horizon"] == "T90" and r["game_id"] in {k["game_id"] for k in keys[:5]})))
        for st_, k in zip((C.MISSED, C.PROVIDER_FAILURE, C.REFUSED, C.NOT_PREGAME, C.EARLY), keys[:5]):
            st.results.append({"key": k["key"], "game_id": k["game_id"], "horizon": "T90", "status": st_})
        st.results.append({"key": keys[5]["key"], "game_id": keys[5]["game_id"], "horizon": "T90", "status": C.REVISED})                      # later row for the same key: latest wins (revised, excluded)
        h = EV.evaluate(st)["horizons"]["T90"]
        assert h["eligible_opportunities"] == 59 and h["eligible_complete_valid"] == 54 and abs(h["capture_success_rate"] - 54 / 59) < 1e-12
        assert h["eligible_failure_breakdown"] == {C.MISSED: 1, C.PROVIDER_FAILURE: 1, C.REFUSED: 1, C.NOT_PREGAME: 1, C.EARLY: 1}
    with tempfile.TemporaryDirectory() as t:                                                       # late enrollment: reported, excluded from the denominator
        st = synth_state(t, n_games=60, n_dates=3, late_games=6)
        h = EV.evaluate(st)["horizons"]["T90"]
        assert h["eligible_opportunities"] == 54 and h["late_enrollment_not_eligible"] == 6 and h["late_enrollment_status"] == "LATE_ENROLLMENT_NOT_ELIGIBLE" and h["capture_success_rate"] == 1.0
    with tempfile.TemporaryDirectory() as t:                                                       # a game without postgame truth is not an eligible opportunity (yet)
        st = synth_state(t, n_games=60, n_dates=3)
        st.planned.append({"key": "9|T90|x", "game_id": 9, "horizon": "T90", "scheduled_start_utc": "x", "cutoff": "2026-10-10T21:30:00.000000Z", "game_date": "2026-10-10", "schedule_sha256": "s", "planned_at": "2026-10-09T00:00:00.000000Z"})
        assert EV.evaluate(st)["horizons"]["T90"]["eligible_opportunities"] == 60


def test_right_rail_scratch_availability_present_list_even_if_empty():
    both_empty = {"gameInfo": {"awayTeam": {"scratches": []}, "homeTeam": {"scratches": []}}}
    absent = {"gameInfo": {"awayTeam": {"scratches": []}, "homeTeam": {}}}
    no_info = {}
    non_list = {"gameInfo": {"awayTeam": {"scratches": None}, "homeTeam": {"scratches": "x"}}}
    populated = {"gameInfo": {"awayTeam": {"scratches": [{"id": 5}, {"id": 6}]}, "homeTeam": {"scratches": [{"id": 7}]}}}
    m = EV.rail_metrics(both_empty, {1}); assert m["scratch_list_available"] is True and m["n_scratches"] == 0
    for bad in (absent, no_info, non_list):
        m = EV.rail_metrics(bad, {1}); assert m["scratch_list_available"] is False and m["n_scratches"] == 0
    m = EV.rail_metrics(populated, {6}); assert m["scratch_list_available"] is True and m["n_scratches"] == 3 and m["scratches_who_actually_played"] == 1


def test_ledger_is_thread_safe_valid_jsonl_and_one_row_per_key():
    import random
    with tempfile.TemporaryDirectory() as t:
        L = S.Ledger(t, "stress.jsonl")
        assert hasattr(L._lock, "acquire") and type(L._lock).__name__ == type(threading.RLock()).__name__
        errors = []

        def worker(wid):
            rnd = random.Random(wid)
            try:
                for i in range(120):
                    k = rnd.randrange(40)                                                              # heavy duplicate-key contention across 16 threads
                    L.append_unique(["k"], {"k": f"dup{k}", "sha": f"v{k}", "pad": "x" * rnd.randrange(50, 4000)}, immutable=("sha",))
                    L.append_unique(["k"], {"k": f"own{wid}-{i}", "sha": "s"}, immutable=("sha",))      # distinct keys
                    L.read()
            except Exception as e:                                                                     # noqa
                errors.append(repr(e))
        ts = [threading.Thread(target=worker, args=(w,)) for w in range(16)]
        [x.start() for x in ts]; [x.join() for x in ts]
        assert not errors, errors[:3]
        raw = L.path.read_bytes()
        assert raw.endswith(b"\n")
        rows = [json.loads(line) for line in raw.split(b"\n") if line]                                  # every line is complete valid JSON (no partial / interleaved rows)
        keys = [r["k"] for r in rows]
        assert len(keys) == len(set(keys)) and len(rows) == len({f"dup{k}" for k in range(40)} & set(keys)) + 16 * 120
        assert {f"own{w}-{i}" for w in range(16) for i in range(120)} <= set(keys)
        for r in rows:
            if r["k"].startswith("dup"):
                assert r["sha"] == "v" + r["k"][3:]
        try:
            L.append_unique(["k"], {"k": "dup0", "sha": "CHANGED"}, immutable=("sha",)) if "dup0" in keys else (_ for _ in ()).throw(S.HardError("n/a"))
            assert False
        except S.HardError:
            pass


def test_protocol_amendment_is_registered_and_versioned():
    proto = json.loads((REPO / "nhl_models" / "nhl_outcome_engine" / "phase0c_capture_protocol.json").read_text())
    assert proto["protocol_version"] == C.PROTOCOL_VERSION == "nhl-forward-capture-protocol-2"
    a = proto["amendments"][-1]
    assert a["id"] == "PRE_LIVE_AMENDMENT_1" and a["registered_before_any_compliant_live_evidence"] is True and a["base_research_commit"] == "717bab9b908313e8839802c4ef59e3269b4178d5"
    assert proto["capture_success_rate"]["late_enrollment"].startswith("LATE_ENROLLMENT_NOT_ELIGIBLE") and proto["status"] == "PREREGISTERED_BEFORE_ANY_COMPLIANT_LIVE_EVIDENCE"
    with tempfile.TemporaryDirectory() as t:
        clk, w, r, _ = setup(t, {2026020400: {"start": START}}, START - timedelta(days=3))
        r.tick()
        rd = json.loads((Path(t) / "readiness.json").read_text())
        assert {c["name"]: c["ok"] for c in rd["checks"]}["capture_protocol_version_matches_registered"] and rd["protocol_version"] == C.PROTOCOL_VERSION and rd["READY"] is True
        assert json.loads((Path(t) / "status.json").read_text())["horizon_status"] == EV.BLOCKER


# ------------------------------------------------------------------ readiness / status / workflow / isolation
def test_readiness_status_and_registered_constants():
    with tempfile.TemporaryDirectory() as t:
        clk, w, r, _ = setup(t, {2026020300: {"start": START}}, START - timedelta(days=3))
        out = r.tick()
        rd = json.loads((Path(t) / "readiness.json").read_text())
        names = {c["name"]: c["ok"] for c in rd["checks"]}
        for n in ("workflow_configured", "state_writable", "cas_gzip_hash_roundtrip", "schedule_reachable", "phase0b_contract_version_recognized", "horizon_timing_constants_match_preregistered", "no_production_or_model_code_invoked"):
            assert names[n], n
        assert rd["READY"] is True
        st = json.loads((Path(t) / "status.json").read_text())
        for k in ("last_invocation_utc", "next_planned_cutoffs", "counts_by_horizon_and_status", "most_recent_schedule_retrieval", "postgame_truth_count", "distinct_game_dates_with_truth", "minimum_horizon_evaluation_sample_exists"):
            assert k in st
        assert st["minimum_horizon_evaluation_sample_exists"] is False and st["horizon_status"] == EV.BLOCKER and st["next_planned_cutoffs"][0]["cutoff"] == C.iso(START - timedelta(minutes=1440))
        w.fail.add(f"{API}/schedule/now")
        assert SC.readiness(t, w.http, clk)["READY"] is False
    proto = json.loads((REPO / "nhl_models" / "nhl_outcome_engine" / "phase0c_capture_protocol.json").read_text())
    assert proto["constants"] == json.loads(json.dumps(SC.registered_constants())) and proto["constants"]["CAPTURE_START_LEAD_SECONDS"] == 120 and proto["constants"]["VALID_WINDOW_SECONDS"] == 180


def test_forward_status_remains_blocker_and_contract_unchanged():
    c = json.loads((REPO / "nhl_models" / "nhl_outcome_engine" / "phase0b_forward_snapshot_contract.json").read_text())
    assert c["status"] == "BLOCKER_INSUFFICIENT_LIVE_TIMING_EVIDENCE" and c["recommended_horizons"] == []


def test_workflow_configuration_and_isolation():
    wf = (REPO / ".github" / "workflows" / "nhl_forward_snapshot.yml").read_text()
    import yaml
    d = yaml.safe_load(wf)
    on = d.get("on") or d.get(True)
    assert "schedule" in on and "workflow_dispatch" in on and on["schedule"][0]["cron"] == "*/15 * * * *"
    assert d["concurrency"] == {"group": "nhl-forward-snapshot", "cancel-in-progress": False}
    body = "\n".join(l for l in wf.splitlines() if not l.strip().startswith("#"))
    assert "nhl_fwd_scheduler.py run" in body and "nhl-forward-state" in body and "pushed=0" in body and "::error::state branch push failed" in body and "exit 1" in body
    for other in ("build.py", "nfl_", "cfb_", "mlb_", "nhl_sog", "nhl_serving", "docs/", "pip install"):
        assert other not in body, other
    for f in SC.CODE_FILES:
        bad = [n for n in SC.imports_of(REPO / f) if n.startswith(SC.FORBIDDEN_IMPORT_PREFIXES)]
        assert not bad, (f, bad)
    assert (REPO / "nhl_models" / "nhl_outcome_engine" / "PHASE0C_RUNBOOK.md").exists()


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
