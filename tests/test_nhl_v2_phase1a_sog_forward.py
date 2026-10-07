"""Forward ledger / validity-window / lock / grader tests for NHL V2 Phase1A-SOG (synthetic fixtures; research only)."""
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
import nhl_v2_phase1a_sog_data as D
import nhl_v2_phase1a_sog_forward as F
import nhl_v2_phase1a_sog_grade as G
import nhl_v2_phase1a_sog_model as M
from test_nhl_v2_phase1a_sog import synth

UTC = timezone.utc
LOCK = {"eligible_from_cutoff_utc": "2026-10-08T00:00:00Z", "comparator_constants": {"F": {"rate_hr": 7.0, "toi_hr": 0.25, "n": 1}, "D": {"rate_hr": 4.5, "toi_hr": 0.33, "n": 1}},
        "base_rates": {"FULL": {"F": [.7, .5, .3, .2, .1], "D": [.5, .3, .1, .05, .02], "U": [.6, .4, .2, .1, .05]}, "MEANINGFUL": {"F": [.7, .5, .3, .2, .1], "D": [.5, .3, .1, .05, .02], "U": [.6, .4, .2, .1, .05]}}}


def t(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def test_ledger_chain_append_only_and_tamper_detection(tmp_path):
    L = F.Ledger(tmp_path / "l.jsonl")
    for i in range(4):
        L.append({"record_type": "FORECAST", "game_id": i, "forecast_horizon": "T90", "x": i})
    assert L.verify() == 4
    rows = L.rows()
    assert rows[0]["prev_hash"] == F.GENESIS and rows[1]["prev_hash"] == rows[0]["row_hash"] and [r["seq"] for r in rows] == [0, 1, 2, 3]
    txt = (tmp_path / "l.jsonl").read_text().splitlines()
    (tmp_path / "t1.jsonl").write_text("\n".join([txt[0].replace('"x":0', '"x":9')] + txt[1:]) + "\n")
    with pytest.raises(F.LockError):
        F.Ledger(tmp_path / "t1.jsonl").verify()
    (tmp_path / "t2.jsonl").write_text("\n".join([txt[0]] + txt[2:]) + "\n")
    with pytest.raises(F.LockError):
        F.Ledger(tmp_path / "t2.jsonl").verify()


def games_at(start):
    return {1: {"game_id": 1, "game_start_utc": start, "state": "FUT", "home_abbrev": "A", "away_abbrev": "B", "home_team_id": 1, "away_team_id": 2, "date": start[:10]}}


def test_plan_window_boundaries_and_eligible_from():
    g = games_at("2026-10-08T23:00:00Z")
    c90 = t("2026-10-08T21:30:00Z")
    assert [p[1] for p in F.plan(c90 - timedelta(seconds=901), g, LOCK, set())] == []
    assert [p[1] for p in F.plan(c90 - timedelta(seconds=900), g, LOCK, set())] == ["T90"]
    assert [p[1] for p in F.plan(c90, g, LOCK, set())] == ["T90"]
    assert [p[1] for p in F.plan(c90 + timedelta(seconds=1), g, LOCK, set())] == []
    assert F.plan(c90, g, LOCK, {(1, "T90")}) == []
    early = games_at("2026-10-08T01:00:00Z")              # T24H cutoff 2026-10-07T01:00 < eligible_from
    now = t("2026-10-07T01:00:00Z")
    assert F.plan(now, early, LOCK, set()) == []
    assert (1, "T24H") not in {(a, b) for a, b, c in F.missed(t("2026-10-08T00:30:00Z"), early, LOCK, set())}


def test_missed_only_after_cutoff_and_after_eligible_from_and_never_duplicated():
    g = games_at("2026-10-09T23:00:00Z")
    m = F.missed(t("2026-10-09T21:31:00Z"), g, LOCK, set())
    assert [x[1] for x in m] == ["T24H", "T90"]                    # T30 cutoff 22:30 is still in the future
    assert F.missed(t("2026-10-09T21:31:00Z"), g, LOCK, {(1, "T24H"), (1, "T90")}) == []
    assert F.missed(t("2026-10-07T00:00:00Z"), g, LOCK, set()) == []


def test_verify_lock_detects_code_model_and_protocol_changes(tmp_path):
    (tmp_path / "f.py").write_text("x=1")
    good = {"code_sha256": {"f.py": hashlib.sha256(b"x=1").hexdigest()}, "model_sha256": {}, "protocol_sha256": {}}
    p = tmp_path / "lock.json"; p.write_text(json.dumps(good))
    assert F.verify_lock(p, tmp_path)
    (tmp_path / "f.py").write_text("x=2")
    with pytest.raises(F.LockError):
        F.verify_lock(p, tmp_path)


def test_availability_extraction_never_certifies():
    landing = {"rosterSpots": [{"playerId": 7, "teamId": 1, "firstName": {"default": "A"}, "lastName": {"default": "B"}}]}
    rail = {"gameInfo": {"awayTeam": {"scratches": [{"id": 9, "firstName": {"default": "S"}, "lastName": {"default": "C"}}]}, "homeTeam": {"scratches": []}}}
    spots, scr, names = F.extract_availability(landing, rail, None)
    assert 7 in spots and scr == {9} and names[7] == "A B" and names[9] == "S C"
    assert F.extract_availability(None, None, None) == ({}, set(), {})


def make_world(tmp_path, monkeypatch, now_offset=-300, late=False):
    games0, rows0 = synth(n_teams=6, n_games=120)
    last = max(g["game_start_utc"] for g in games0.values())
    start = (datetime.strptime(last, "%Y-%m-%dT%H:%M:%SZ") + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    gid = max(games0) + 1
    gl = {gid: {"game_id": gid, "game_start_utc": start, "home_abbrev": "T1", "away_abbrev": "T2", "home_team_id": 1, "away_team_id": 2, "date": start[:10], "state": "FUT"}}
    train, _ = D.build_rows(games0, rows0, target_seasons=[2019], horizon_min=90)
    art = M.fit_b2(train, "T90")
    models = tmp_path / "phase1a_sog_models"; models.mkdir()
    (models / "engine_T90.json").write_text(json.dumps(art))
    monkeypatch.setattr(F, "OUT", tmp_path)
    monkeypatch.setattr(F, "LOCK", tmp_path / "lock.json"); (tmp_path / "lock.json").write_text("{}")
    lock = dict(LOCK, eligible_from_cutoff_utc="2019-01-01T00:00:00Z")
    monkeypatch.setattr(F, "verify_lock", lambda *a, **k: lock)
    monkeypatch.setattr(F, "BLOBS", tmp_path / "blobs")
    prov = [{"kind": "schedule", "sha256": "ab" * 32, "retrieval_completed_utc": "2019-01-01T00:00:00+00:00"}]
    monkeypatch.setattr(F, "live_season", lambda fetcher, now: (gl, [], prov))
    cutoff = F.cutoff_of(start, "T90")
    now = cutoff + timedelta(seconds=now_offset)
    calls = [0]

    def now_fn():
        calls[0] += 1
        return now + (timedelta(seconds=400) if (late and calls[0] > 1) else timedelta(0))

    def fetcher(url):
        raw = json.dumps({"rosterSpots": [], "gameInfo": {"awayTeam": {"scratches": []}, "homeTeam": {"scratches": []}}, "forwards": [], "defensemen": []}).encode()
        return raw, {"url": url, "retrieval_started_utc": now.isoformat(), "retrieval_completed_utc": now.isoformat(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "http_status": 200}
    return gid, (games0, rows0), fetcher, now_fn, art


def test_run_produces_valid_immutable_forecasts_inside_window(tmp_path, monkeypatch):
    gid, frozen, fetcher, now_fn, art = make_world(tmp_path, monkeypatch)
    L = F.Ledger(tmp_path / "ledger.jsonl")
    res = F.run(now_fn=now_fn, fetcher=fetcher, ledger=L, frozen=frozen, log=lambda *a: None)
    rows = L.rows()
    fcs = [r for r in rows if r["record_type"] == "FORECAST" and r["forecast_horizon"] == "T90"]
    assert res["forecasts"] == len(fcs) > 10 and L.verify() == len(rows)
    r = fcs[0]
    for k in ("forecast_id", "run_id", "engine_version", "git_sha", "game_id", "player_id", "team", "opponent", "scheduled_start", "forecast_horizon", "cutoff_at", "generated_at", "candidate_status", "availability_state",
              "availability_confidence", "feature_snapshot_sha256", "raw_source_hashes", "expected_sog", "median_sog", "variance", "dispersion", "P1", "P2", "P3", "P4", "P5", "receipt", "comparators"):
        assert k in r, k
    assert r["availability_confidence"] == "NOT_CERTIFIED" and r["availability_used_in_forecast"] is False and r["receipt"]["target_game_lineup_or_pp_state_used"] is False
    assert r["availability_state"] == "NO_ROSTER_DATA_PUBLISHED_YET"
    assert r["P1"] >= r["P2"] >= r["P3"] >= r["P4"] >= r["P5"] and r["variance"] >= r["expected_sog"]
    assert abs(r["generated_at"] and 0) == 0 and F.parse_iso(r["generated_at"]) <= F.parse_iso(r["cutoff_at"])
    assert not any(k in json.dumps(r).lower() for k in ("odds", "sportsbook", "bookmaker"))
    # re-running in the same window creates no duplicate and no rewrite
    before = (tmp_path / "ledger.jsonl").read_text()
    F.run(now_fn=now_fn, fetcher=fetcher, ledger=L, frozen=frozen, log=lambda *a: None)
    after = (tmp_path / "ledger.jsonl").read_text()
    assert after.startswith(before) and len([x for x in L.rows() if x["record_type"] == "FORECAST" and x["forecast_horizon"] == "T90"]) == len(fcs)


def test_run_outside_window_creates_no_forecast_and_records_missed(tmp_path, monkeypatch):
    gid, frozen, fetcher, now_fn, art = make_world(tmp_path, monkeypatch, now_offset=+60)
    L = F.Ledger(tmp_path / "ledger.jsonl")
    res = F.run(now_fn=now_fn, fetcher=fetcher, ledger=L, frozen=frozen, log=lambda *a: None)
    types = [(r["record_type"], r["forecast_horizon"]) for r in L.rows()]
    assert res["forecasts"] == 0 and ("MISSED_CUTOFF", "T90") in types and not any(t_[0] == "FORECAST" for t_ in types)


def test_late_completion_is_invalid_audit_only(tmp_path, monkeypatch):
    gid, frozen, fetcher, now_fn, art = make_world(tmp_path, monkeypatch, now_offset=-60, late=True)
    L = F.Ledger(tmp_path / "ledger.jsonl")
    F.run(now_fn=now_fn, fetcher=fetcher, ledger=L, frozen=frozen, log=lambda *a: None)
    kinds = {r["record_type"] for r in L.rows()}
    assert "INVALID_LATE" in kinds and "FORECAST" not in kinds


# ------------------------------------------------------------------ grader
def fake_official(game_id, rows, state="FINAL"):
    def fetcher(url):
        if "/schedule/" in url:
            body = {"gameWeek": [{"date": "2026-10-09", "games": [{"id": game_id, "gameType": 2, "gameState": state, "startTimeUTC": "2026-10-09T23:00:00Z", "homeTeam": {"abbrev": "A", "id": 1}, "awayTeam": {"abbrev": "B", "id": 2}}]}]}
        elif "/skater/summary" in url:
            body = {"total": len(rows), "data": [{"gameId": game_id, "playerId": r[0], "teamAbbrev": "A" if r[1] == 1 else "B", "opponentTeamAbbrev": "B" if r[1] == 1 else "A", "positionCode": "C", "shots": r[2]} for r in rows]}
        else:
            body = {"total": len(rows), "data": [{"gameId": game_id, "playerId": r[0], "timeOnIce": 1000, "ppTimeOnIce": 100, "shifts": 20, "evTimeOnIce": 850, "shTimeOnIce": 50, "otTimeOnIce": 0} for r in rows]}
        raw = json.dumps(body).encode()
        return raw, {"url": url, "retrieval_started_utc": "x", "retrieval_completed_utc": "y", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "http_status": 200}
    return fetcher


def fc_row(fid, pid, mu=1.5, game=77, meaningful=True, pos="F"):
    return {"record_type": "FORECAST", "forecast_id": fid, "game_id": game, "player_id": pid, "forecast_horizon": "T90", "scheduled_start": "2026-10-09T23:00:00Z", "cutoff_at": "2026-10-09T21:30:00Z", "schedule_date": "2026-10-09",
            "expected_sog": mu, "median_sog": 1.0, "variance": 2.0, "dispersion": 0.15, "P1": .7, "P2": .45, "P3": .25, "P4": .12, "P5": .05, "meaningful_expected_participant": meaningful,
            "comparators": {"comparator_eligible": True, "human_frozen_mean": 1.4, "simple_prior10_mean": 1.3},
            "receipt": {"position": pos, "toi_seconds": {"recent3": 1000.0}, "sog_per60_app10": 5.0, "team_sog_for_mean5": 30.0, "pp_toi_recent3_seconds": 60.0}}


def test_grader_scores_only_final_handles_absent_players_and_is_idempotent(tmp_path):
    L = F.Ledger(tmp_path / "l.jsonl")
    for pid in (11, 12, 13):
        L.append(fc_row("77-T90-%d" % pid, pid))
    gp = tmp_path / "g.jsonl"
    assert G.grade(fetcher=fake_official(77, [(11, 1, 4), (12, 2, 0)], state="LIVE"), ledger=L, grades_path=gp, log=lambda *a: None) == 0
    n = G.grade(fetcher=fake_official(77, [(11, 1, 4), (12, 2, 0)]), ledger=L, grades_path=gp, log=lambda *a: None)
    assert n == 3
    g = {r["player_id"]: r for r in F.Ledger(gp).rows()}
    assert g[11]["actual_sog"] == 4 and g[11]["played"] and g[13]["actual_sog"] == 0 and not g[13]["played"]
    assert g[11]["diagnostics_postgame_only"]["actual_team_sog"] == 4
    assert G.grade(fetcher=fake_official(77, [(11, 1, 4)]), ledger=L, grades_path=gp, log=lambda *a: None) == 0
    assert F.Ledger(gp).verify() == 3


def test_join_rejects_grade_for_modified_forecast_and_censors_need_evidence(tmp_path):
    L = F.Ledger(tmp_path / "l.jsonl"); L.append(fc_row("77-T90-11", 11))
    gp = tmp_path / "g.jsonl"
    G.grade(fetcher=fake_official(77, [(11, 1, 2)]), ledger=L, grades_path=gp, log=lambda *a: None)
    assert len(G.join(L, gp, tmp_path / "none.jsonl")) == 1
    bad = F.Ledger(tmp_path / "bad.jsonl"); bad.append(dict(fc_row("77-T90-11", 11), expected_sog=9.9))
    with pytest.raises(ValueError):
        G.join(bad, gp, tmp_path / "none.jsonl")
    cp = tmp_path / "c.jsonl"
    cp.write_text(json.dumps({"forecast_id": "77-T90-11", "reason": "BAD_SHOOTING_NIGHT", "evidence": "x"}) + "\n")
    with pytest.raises(ValueError):
        G.load_censors(cp)
    cp.write_text(json.dumps({"forecast_id": "77-T90-11", "reason": "IN_GAME_INJURY_ABNORMAL_LOST_ICE_TIME", "evidence": "official game note"}) + "\n")
    pairs = G.join(L, gp, cp)
    assert pairs[0][2] is True


def test_large_miss_classification_uses_fixed_ratios_only():
    f = fc_row("a", 1, mu=1.0)
    g = {"actual_sog": 6, "played": True, "diagnostics_postgame_only": {"actual_toi_seconds": 1500.0, "actual_pp_toi_seconds": 300.0, "actual_shifts": 30, "actual_team_sog": 45}}
    assert G.classify_miss(f, g) == "MULTIPLE"
    f["receipt"]["sog_per60_app10"] = 17.0
    g2 = {"actual_sog": 5, "played": True, "diagnostics_postgame_only": {"actual_toi_seconds": 1000.0, "actual_pp_toi_seconds": 60.0, "actual_shifts": 25, "actual_team_sog": 30}}
    assert G.classify_miss(f, g2) == "UNRESOLVED"
    assert G.classify_miss(f, {"actual_sog": 2, "played": True, "diagnostics_postgame_only": g2["diagnostics_postgame_only"]}) is None
    g3 = {"actual_sog": 0, "played": False, "diagnostics_postgame_only": {"actual_toi_seconds": None, "actual_pp_toi_seconds": None, "actual_shifts": None, "actual_team_sog": None}}
    assert G.classify_miss(fc_row("b", 2, mu=3.9), g3) == "AVAILABILITY"


def test_evaluate_forward_uses_stored_comparators_and_both_ledgers(tmp_path):
    L = F.Ledger(tmp_path / "l.jsonl")
    rng = np.random.default_rng(1)
    rows_ = []
    for gi in range(40):
        for pid in range(10):
            fid = "%d-T90-%d" % (1000 + gi, pid)
            r = fc_row(fid, pid, mu=1.5, game=1000 + gi, meaningful=(pid < 7), pos="F" if pid < 6 else "D")
            r["scheduled_start"] = (datetime(2026, 10, 9, 23, tzinfo=UTC) + timedelta(days=gi)).strftime("%Y-%m-%dT%H:%M:%SZ"); r["schedule_date"] = r["scheduled_start"][:10]
            L.append(r)
    gp = tmp_path / "g.jsonl"; G_ = F.Ledger(gp)
    for r in L.rows():
        G_.append({"record_type": "GRADE", "forecast_id": r["forecast_id"], "forecast_row_hash": r["row_hash"], "game_id": r["game_id"], "player_id": r["player_id"], "forecast_horizon": "T90", "graded_at": "x",
                   "played": True, "actual_sog": int(rng.poisson(1.5)), "diagnostics_postgame_only": {"actual_toi_seconds": 1000, "actual_pp_toi_seconds": 60, "actual_shifts": 20, "actual_team_sog": 30}, "official_source_hashes": {}, "game_state": "FINAL"})
    cp = tmp_path / "c.jsonl"; cp.write_text(json.dumps({"forecast_id": "1000-T90-0", "reason": "EJECTION_OR_MISCONDUCT_ABNORMAL_EXIT", "evidence": "official"}) + "\n")
    pairs = G.join(L, gp, cp)
    raw = G.evaluate(pairs, LOCK, "T90", clean=False, population="FULL")
    clean = G.evaluate(pairs, LOCK, "T90", clean=True, population="FULL")
    mean = G.evaluate(pairs, LOCK, "T90", population="MEANINGFUL")
    assert raw["V2_B2"]["n_rows"] == 400 and clean["V2_B2"]["n_rows"] == 399 and mean["V2_B2"]["n_rows"] == 280
    assert raw["comparators_identical_rows"]["identical_rows"] == 400 and "paired_crps_V2_B2_minus_human_frozen" in raw["comparators_identical_rows"]
    assert set(raw["V2_B2"]["thresholds"]) == {"P(SOG>=%d)" % k for k in range(1, 6)}


def test_production_join_accepts_only_pregame_logged_probabilities():
    pairs = [(dict(fc_row("a", 5), cutoff_at="2026-10-09T21:30:00Z", schedule_date="2026-10-09"), None, False), (dict(fc_row("b", 6), cutoff_at="2026-10-09T21:30:00Z", schedule_date="2026-10-09"), None, False)]
    log = "\n".join([json.dumps({"market": "shots_on_goal_early_season", "player_id": 5, "game_date": "2026-10-09", "prob_over": 0.3, "logged_at": "2026-10-09T10:00:00+00:00"}),
                     json.dumps({"market": "shots_on_goal", "player_id": 6, "game_date": "2026-10-09", "prob_over": 0.4, "logged_at": "2026-10-09T22:00:00+00:00"})])
    assert G.production_join(pairs, log) == {"a": 0.3}
