"""Focused NHL Phase 0B tests (historical candidate universe, acquisition evidence, forward snapshot provenance). python tests/test_nhl_phase0b.py"""
import copy
import hashlib
import json
import random
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import nhl_outcome_contract as CT  # noqa: E402
import nhl_outcome_phase0b_probes as PR  # noqa: E402
import nhl_outcome_snapshot as SN  # noqa: E402

E = REPO / "nhl_models" / "nhl_outcome_engine"
UTC = timezone.utc
T0 = datetime(2023, 11, 1, 19, 0, tzinfo=UTC)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def row(g, p, team, start, pp=60, ev=900, shifts=20, sog=2):
    return {"gameId": g, "playerId": p, "team": team, "startTimeUTC": iso(start), "pp": pp, "ev": ev, "sh": 0, "toi": pp + ev, "shifts": shifts, "sog": sog}


def world(n_games=12):
    """AAA plays games 1..n (every 2 days) vs BBB. AAA regulars a1..a5 (a1 has the most PP); BBB regulars b1..b5.
    'old' plays for AAA only in game 1; 'x1' plays for CCC in games 1-5 (other team) and for AAA from game 7; 'y' plays for CCC with a lot of PP (all games) and for AAA only in game 9+."""
    rows, meta = [], {}
    for g in range(1, n_games + 1):
        st = T0 + timedelta(days=2 * (g - 1))
        meta[g] = {"startTimeUTC": iso(st), "home": "AAA", "away": "BBB"}
        for i in range(1, 6):
            rows.append(row(g, f"a{i}", "AAA", st, pp=200 - 30 * i))
            rows.append(row(g, f"b{i}", "BBB", st, pp=190 - 30 * i))
        if g == 1:
            rows.append(row(g, "old", "AAA", st))
        if g <= 5:
            rows.append(row(100 + g, "x1", "CCC", st - timedelta(hours=3), pp=50)); meta[100 + g] = {"startTimeUTC": iso(st - timedelta(hours=3)), "home": "CCC", "away": "DDD"}
        if g >= 7:
            rows.append(row(g, "x1", "AAA", st, pp=40))
        if g <= 8:
            rows.append(row(200 + g, "y", "CCC", st - timedelta(hours=3), pp=300)); meta[200 + g] = {"startTimeUTC": iso(st - timedelta(hours=3)), "home": "CCC", "away": "DDD"}
        if g >= 9:
            rows.append(row(g, "y", "AAA", st, pp=10))
    return rows, meta


def cut(meta, g):
    return CT.cutoff_time(CT.parse_utc(meta[g]["startTimeUTC"]), "T90")


def probe(rows, meta, target):
    return {(t["game_id"], t["team"]): t for t in PR.deployment_probe(rows, meta, {target})}


def hist(rows, meta, target, team="AAA"):
    return CT.candidate_universe(rows, target, team, cut(meta, target))


# ------------------------------------------------------------------ A / D : historical candidate universe
def test_mutating_target_game_rows_cannot_change_candidates_or_features():
    rows, meta = world()
    base = probe(rows, meta, 12)
    mut = []
    for r in rows:
        if r["gameId"] == 12:
            r = {**r, "playerId": "Z" + str(r["playerId"]), "pp": 999, "ev": 1, "sh": 77, "toi": 5, "shifts": 99, "sog": 42}
        mut.append(r)
    after = probe(mut, meta, 12)
    for k in base:
        assert base[k]["candidates"] == after[k]["candidates"] and base[k]["team_games_used"] == after[k]["team_games_used"]
        assert json.dumps(base[k]["features"], sort_keys=True) == json.dumps(after[k]["features"], sort_keys=True)
    assert base[(12, "AAA")]["grading"]["n_actual_target_skaters"] == after[(12, "AAA")]["grading"]["n_actual_target_skaters"] and base[(12, "AAA")]["grading"]["actual"] != after[(12, "AAA")]["grading"]["actual"]


def test_target_only_player_is_unobservable_and_never_a_candidate():
    rows, meta = world()
    base = probe(rows, meta, 12)[(12, "AAA")]
    rows2 = rows + [row(12, "newbie", "AAA", datetime.strptime(meta[12]["startTimeUTC"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC), sog=3)]
    t = probe(rows2, meta, 12)[(12, "AAA")]
    assert t["candidates"] == base["candidates"] and "newbie" not in t["candidates"] and "newbie" not in t["features"]
    g = t["grading"]
    assert g["n_unobservable_at_T"] == base["grading"]["n_unobservable_at_T"] + 1 and any(u["playerId"] == "newbie" and u["diagnostic_grading_only"] == "NO_LOADED_HISTORY" for u in g["unobservable"])
    assert g["total_actual_SOG"] == base["grading"]["total_actual_SOG"] + 3 and g["observable_actual_SOG"] == base["grading"]["observable_actual_SOG"] and g["SOG_coverage"] < base["grading"]["SOG_coverage"]


def test_other_team_history_does_not_make_a_historical_candidate_then_first_same_team_game_does():
    rows, meta = world()
    assert "x1" not in hist(rows, meta, 6)["candidates"]                       # x1 has only CCC history before game 7
    t = probe(rows, meta, 7)[(7, "AAA")]
    assert "x1" not in t["candidates"]
    unobs = {u["playerId"]: u["diagnostic_grading_only"] for u in t["grading"]["unobservable"]}
    assert unobs["x1"] == "PRIOR_NHL_HISTORY_ELSEWHERE"                        # graded as unobservable; never called an ACQUIRED candidate
    assert "x1" in hist(rows, meta, 8)["candidates"]                           # after his first completed AAA game he may enter a LATER target game
    assert "ACQUIRED" not in json.dumps(CT.CONTRACT)


def test_same_team_appearance_older_than_ten_team_games_does_not_establish_membership():
    rows, meta = world(12)
    u = hist(rows, meta, 12)
    assert u["team_games"] == list(range(2, 12)) and len(u["team_games"]) == CT.LOOKBACK_TEAM_GAMES and "old" not in u["candidates"]
    assert "old" in hist(rows, meta, 10)["candidates"]                          # game 1 is still within the last 10 team games before game 10 (games 1..9)
    t = probe(rows, meta, 12)[(12, "AAA")]
    rows2 = rows + [row(12, "old", "AAA", CT.parse_utc(meta[12]["startTimeUTC"]))]
    assert {u["playerId"]: u["diagnostic_grading_only"] for u in probe(rows2, meta, 12)[(12, "AAA")]["grading"]["unobservable"]}["old"] == "STALE_SAME_TEAM_HISTORY_OUTSIDE_MEMBERSHIP_WINDOW"


def test_incomplete_source_game_is_excluded_and_boundary_is_inclusive():
    rows, meta = world(6)
    T = cut(meta, 6)
    late_start = T - timedelta(minutes=CT.GAME_MAX_MINUTES) + timedelta(minutes=1)             # start + 210 min = T + 1 min > T  -> excluded
    edge_start = T - timedelta(minutes=CT.GAME_MAX_MINUTES)                                     # start + 210 min == T -> included
    r_late = [row(500, "late_p", "AAA", late_start)]
    r_edge = [row(501, "edge_p", "AAA", edge_start)]
    u = CT.candidate_universe(rows + r_late + r_edge, 6, "AAA", T)
    assert "late_p" not in u["candidates"] and 500 not in u["team_games"] and "edge_p" in u["candidates"] and 501 in u["team_games"]


def test_target_game_id_is_excluded_regardless_of_timestamps():
    rows, meta = world(6)
    T = cut(meta, 6)
    mis = [row(6, "ghost", "AAA", T - timedelta(days=30))]                                       # the target game with a misleading early timestamp
    u = CT.candidate_universe(rows + mis, 6, "AAA", T)
    assert "ghost" not in u["candidates"] and 6 not in u["team_games"]
    assert all(r["gameId"] != 6 for r in CT.prior_rows(rows + mis, 6, T))


def test_candidates_and_features_are_deterministic_under_row_order():
    rows, meta = world()
    a = probe(rows, meta, 12)
    for seed in (1, 2, 3):
        sh = rows[:]; random.Random(seed).shuffle(sh)
        b = probe(sh, meta, 12)
        assert json.dumps({str(k): (v["candidates"], v["team_games_used"], v["features"]) for k, v in a.items()}, sort_keys=True) == json.dumps({str(k): (v["candidates"], v["team_games_used"], v["features"]) for k, v in b.items()}, sort_keys=True)
    assert json.dumps(probe(list(reversed(rows)), meta, 12)[(12, "AAA")]["candidates"]) == json.dumps(a[(12, "AAA")]["candidates"])


def test_target_game_rows_are_used_only_for_labels_and_coverage():
    rows, meta = world()
    t = probe(rows, meta, 12)[(12, "AAA")]
    g = t["grading"]
    actual_rows = [r for r in rows if r["gameId"] == 12 and r["team"] == "AAA"]
    assert g["n_actual_target_skaters"] == len(actual_rows) and g["total_actual_SOG"] == sum(r["sog"] for r in actual_rows)
    assert g["n_candidates_at_T"] == len(t["candidates"]) and g["n_actual_skaters_observable_at_T"] + g["n_unobservable_at_T"] == g["n_actual_target_skaters"]
    for k in ("n_candidates_at_T", "n_actual_target_skaters", "n_actual_skaters_observable_at_T", "n_unobservable_at_T", "actual_player_coverage", "total_actual_SOG", "observable_actual_SOG", "SOG_coverage"):
        assert k in g
    # the team identities come from the schedule metadata: with the target rows removed entirely the candidates are unchanged
    no_target = [r for r in rows if r["gameId"] != 12]
    t2 = probe(no_target, meta, 12)[(12, "AAA")]
    assert t2["candidates"] == t["candidates"] and t2["grading"]["n_actual_target_skaters"] == 0
    assert not hasattr(PR, "team_tot_of")                                                  # dead helper removed


def test_current_team_role_features_ignore_old_team_rows():
    rows, meta = world()
    t = probe(rows, meta, 12)[(12, "AAA")]
    f = t["features"]["y"]                                                                  # y: 8 CCC games with PP 300, AAA games 9..11 with PP 10
    assert f["n_current_team_appearances"] == 3 and abs(f["prior_pp_toi_mean"] - 10) < 1e-9
    assert [r["gameId"] for r in CT.current_team_appearances(rows, 12, "AAA", "y", cut(meta, 12))] == [9, 10, 11]
    mut = [({**r, "pp": 9999, "ev": 1, "shifts": 99} if (r["team"] == "CCC") else r) for r in rows]                # old-team role rows changed
    assert json.dumps(probe(mut, meta, 12)[(12, "AAA")]["features"], sort_keys=True) == json.dumps(t["features"], sort_keys=True)
    assert "pp_allocation_share5" in f and "ev_allocation_share5" in f and f["pp_share_change_recent_vs_long"] is not None


# ------------------------------------------------------------------ B/H : acquisition + probe evidence
def test_shootout_exclusion_and_attempt_attribution():
    ev = [{"typeDescKey": "shot-on-goal", "periodDescriptor": {"periodType": "REG"}, "details": {"shootingPlayerId": 1}},
          {"typeDescKey": "goal", "periodDescriptor": {"periodType": "OT"}, "details": {"scoringPlayerId": 1}},
          {"typeDescKey": "missed-shot", "periodDescriptor": {"periodType": "REG"}, "details": {"shootingPlayerId": 1}},
          {"typeDescKey": "blocked-shot", "periodDescriptor": {"periodType": "REG"}, "details": {"shootingPlayerId": 1, "blockingPlayerId": 2}},
          {"typeDescKey": "shot-on-goal", "periodDescriptor": {"periodType": "SO"}, "details": {"shootingPlayerId": 1}},
          {"typeDescKey": "goal", "periodDescriptor": {"periodType": "SO"}, "details": {"scoringPlayerId": 1}}]
    c = CT.shot_counts_from_events(ev)[1]
    assert CT.sog(c) == 2 and CT.attempts(c) == 4 and 2 not in CT.shot_counts_from_events(ev)
    p = json.loads((E / "phase0b_acquisition_benchmark.json").read_text())["C_shot_attempts"]["summary"]
    assert p["sog_mismatch_players"] == 0 and p["shootout_events_excluded"] > 0 and p["shootout_games"] and p["min_join_rate_to_minimal_table"] == 1.0


def test_toi_strength_consistency_in_sample_table():
    tab = json.loads((E / "evidence" / "m2_sample_table.json").read_text())
    assert len(tab) > 1000
    bad = [r for r in tab if not CT.strength_consistent(r, tol=1)]
    assert len(bad) <= 0.002 * len(tab), len(bad)
    assert all(r["teamId"] and r["startTimeUTC"] and r["opponent"] for r in tab)


def test_traded_player_acquisition_completeness_and_defect_diagnosis():
    b = json.loads((E / "phase0b_acquisition_benchmark.json").read_text())
    tp = b["traded_players"]
    assert tp["traded_skater_games_in_sample"] > 0 and tp["traded_skater_games_found_by_M2"] == tp["traded_skater_games_in_sample"]
    for w in b["weeks"]:
        m = w["m2_vs_boxscore"]
        assert m["missing_vs_boxscore"] == 0 and m["extra_vs_boxscore"] == 0 and m["sog_mismatch"] == 0 and m["toi_mismatch_gt_1s"] == 0
    d = b["missing_player_defect_diagnosis"]
    assert d["date_window_query_for_same_games"]["missing_sample_rows"] == 0 and d["team_filter_with_limit_minus_1"]["missing_sample_rows"] > 0
    assert d["existing_unsorted_100_per_page"]["duplicate_rows_that_displace_real_rows"] > 0 and d["explicit_sort_100_per_page"]["duplicate_rows_that_displace_real_rows"] == 0
    assert b["cap_test"]["silently_truncated_at_10000"] is True and b["m2_conclusion_reproduced_by_this_bounded_benchmark"] is True
    n_games = b["sample"]["games"]
    assert str(n_games) in (E / "PHASE0B_README.md").read_text()                             # README numbers come from the evidence


def test_deterministic_cutoff_handling():
    st = datetime(2026, 10, 3, 23, 0, tzinfo=UTC)
    assert CT.cutoff_time(st, "T90") == datetime(2026, 10, 3, 21, 30, tzinfo=UTC) and CT.cutoff_time(st, "T24H") == datetime(2026, 10, 2, 23, 0, tzinfo=UTC)
    assert CT.cutoff_time(st, "T90") == CT.cutoff_time(st.astimezone(timezone(timedelta(hours=-4))), "T90")
    try:
        CT.cutoff_time(datetime(2026, 10, 3, 23, 0), "T90"); assert False
    except ValueError:
        pass
    T = CT.cutoff_time(st, "T90")
    edge = T - timedelta(minutes=CT.GAME_MAX_MINUTES)
    assert CT.completed_before(edge, T) is True and CT.completed_before(edge + timedelta(minutes=1), T) is False and CT.completed_before(st, T) is False


# ------------------------------------------------------------------ E : forward snapshot provenance
class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def sched_json(start):
    return json.dumps({"gameWeek": [{"date": "2026-10-04", "games": [{"id": 2026020035, "startTimeUTC": start.strftime("%Y-%m-%dT%H:%M:%SZ")}]}]}).encode()


def make_http(clock, start, durations=None, log=None):
    durations = durations or {}

    def http(url):
        if log is not None:
            log.append((url, clock.t))
        clock.t = clock.t + timedelta(seconds=durations.get(url.rsplit("/", 1)[-1], 1))
        if url.endswith("schedule/now"):
            return sched_json(start), {"ETag": "s"}, 200
        ep = url.rsplit("/", 1)[-1]
        body = {"landing": {"gameState": "FUT", "matchup": {"goalieComparison": {"x": 1}}}, "right-rail": {"gameInfo": {"awayTeam": {"scratches": [1]}, "homeTeam": {}, "referees": []}},
                "boxscore": {"gameState": "FUT", "playerByGameStats": {}}, "play-by-play": {"gameState": "FUT", "plays": [], "rosterSpots": [{"positionCode": "G"}]}}[ep]
        return json.dumps(body).encode(), {"ETag": "e"}, 200
    return http


def test_raw_schedule_hash_provenance_and_endpoint_timestamps():
    start = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
    clk = Clock(start - timedelta(minutes=95))
    with tempfile.TemporaryDirectory() as t:
        http = make_http(clk, start)
        sched, starts = SN.fetch_schedule(t, http, clk)
        raw = sched_json(start)
        assert sched["schedule_sha256"] == hashlib.sha256(raw).hexdigest() and SN.verify_blob(t, sched["schedule_sha256"])           # raw schedule bytes stored and hash-roundtrip
        assert (Path(t) / "blobs" / sched["schedule_sha256"]).read_bytes() == raw and starts[2026020035] == start
        assert SN.parse_iso(sched["schedule_retrieval_completed_at"]) > SN.parse_iso(sched["schedule_retrieval_started_at"])
        rows = SN.observe(t, 2026020035, start, sched, http, clk)
        assert [r["endpoint"] for r in rows] == list(SN.ENDPOINTS)
        for r in rows:
            assert r["observation_status"] == SN.VALID and r["schedule_sha256"] == sched["schedule_sha256"] and r["scheduled_start_utc"] == SN.iso(start)
            assert SN.parse_iso(r["retrieval_completed_at"]) > SN.parse_iso(r["retrieval_started_at"]) and r["retrieval_ts"] == r["retrieval_completed_at"] and r["source_url"].endswith(r["endpoint"])
            assert abs(r["minutes_to_start"] - (start - SN.parse_iso(r["retrieval_completed_at"])).total_seconds() / 60) < 1e-3 and SN.verify_blob(t, r["sha256"])
            assert {"http_status", "bytes", "etag", "game_state", "availability", "intended_horizon"} <= set(r)
        logged = [json.loads(l) for l in (Path(t) / "observations.jsonl").read_text().splitlines()]
        assert logged[0]["record_type"] == "schedule" and sum(1 for l in logged if l["record_type"] == "observation") == 4


def test_no_pregame_fetch_is_initiated_at_or_after_puck_drop_and_late_completion_is_rejected():
    start = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
    for offset in (0, 1, 300):                                                                  # at puck drop, 1 s after, 5 min after (no grace period)
        clk = Clock(start + timedelta(seconds=offset))
        calls = []
        with tempfile.TemporaryDirectory() as t:
            sched = {"schedule_sha256": "x", "schedule_retrieval_started_at": "a", "schedule_retrieval_completed_at": "b"}
            rows = SN.observe(t, 1, start, sched, make_http(clk, start, log=calls), clk)
            assert [r["observation_status"] for r in rows] == [SN.REFUSED] * 4 and calls == [] and all("sha256" not in r for r in rows)
    clk = Clock(start - timedelta(seconds=3))                                                  # started before puck drop, completes after it
    with tempfile.TemporaryDirectory() as t:
        sched = {"schedule_sha256": "x", "schedule_retrieval_started_at": "a", "schedule_retrieval_completed_at": "b"}
        rows = SN.observe(t, 1, start, sched, make_http(clk, start, durations={"landing": 10, "right-rail": 10, "boxscore": 10, "play-by-play": 10}), clk)
        first = rows[0]
        assert first["observation_status"] == SN.REJECTED_LATE and first["intended_horizon"] == "NONE_REJECTED_LATE" and first["minutes_to_start"] < 0 and SN.verify_blob(t, first["sha256"])    # bytes retained for audit
        assert all(r["observation_status"] in (SN.REJECTED_LATE, SN.REFUSED) for r in rows)
    clk = Clock(start - timedelta(seconds=1))                                                  # completes exactly AT the start: still not pregame
    with tempfile.TemporaryDirectory() as t:
        sched = {"schedule_sha256": "x", "schedule_retrieval_started_at": "a", "schedule_retrieval_completed_at": "b"}
        assert SN.observe(t, 1, start, sched, make_http(clk, start, durations={"landing": 1}), clk)[0]["observation_status"] == SN.REJECTED_LATE


def test_watcher_hard_stop_at_puck_drop_with_injected_time():
    start = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
    clk = Clock(start - timedelta(minutes=10))
    calls = []
    with tempfile.TemporaryDirectory() as t:
        SN.watch(t, 150, 120, 3.0, http=make_http(clk, start, log=calls), now_fn=clk, sleep=lambda s: setattr(clk, "t", clk.t + timedelta(seconds=s)))        # never really sleeps
        obs = [json.loads(l) for l in (Path(t) / "observations.jsonl").read_text().splitlines()]
        o = [x for x in obs if x["record_type"] == "observation"]
        assert o and all(SN.parse_iso(x["retrieval_started_at"]) < start for x in o if x["observation_status"] != SN.REFUSED)
        assert all(SN.parse_iso(u[1].strftime("%Y-%m-%dT%H:%M:%SZ")) < start for u in calls if "gamecenter" in u[0])                     # no game endpoint request was initiated at/after the start


def test_goalie_signal_wording_is_non_authoritative():
    d = SN.availability("landing", {"matchup": {"goalieComparison": {"leaders": [{"starter": True}]}}})
    assert "starter_like_field_present_unvalidated" in d and not any("confirmed" in k for k in d)
    b = SN.availability("boxscore", {"playerByGameStats": {"homeTeam": {"goalies": [{"starter": True}]}, "awayTeam": {"goalies": []}}})
    assert "boxscore_starter_field_values_observed_unvalidated" in b and "starter_flags" not in b
    c = json.loads((E / "phase0b_historical_contract.json").read_text())
    assert "NOT reconstructable" in c["goalie_statement"]


def test_forward_horizon_blocker_semantics():
    f = json.loads((E / "phase0b_forward_snapshot_contract.json").read_text())
    assert isinstance(f["recommended_horizons"], list)
    if not f["recommended_horizons"]:
        assert f["status"] == "BLOCKER_INSUFFICIENT_LIVE_TIMING_EVIDENCE"
    assert f["status"] == "BLOCKER_INSUFFICIENT_LIVE_TIMING_EVIDENCE" and f["recommended_horizons"] == []                          # this commit: nothing validated
    assert f["exploratory_prior_observations"]["qualifies_for_horizon_selection"] is False
    assert not any("validated" in str(v).lower() and "not" not in str(v).lower() for v in f["recommended_horizons"])
    assert {"schedule_sha256", "retrieval_started_at", "retrieval_completed_at", "observation_status"} <= set(f["observation_fields"])
    assert "REJECTED_LATE" in f["observation_status_values"]


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
