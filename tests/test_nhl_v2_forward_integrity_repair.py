"""Adversarial integrity tests for immutable NHL forward forecasts and grading."""
import json
from datetime import datetime, timezone

import pytest

import nhl_v2_phase1a_sog_forward as F
import nhl_v2_phase1a_sog_grade as G

START = "2026-10-07T23:30:00Z"
NOW = lambda: datetime(2026, 10, 8, 12, tzinfo=timezone.utc)


def add_forecast(ledger, *, player_id, team, start=START, horizon="T90", game_id=2026020053):
    # The v1.1 ID did NOT include team. Intentionally reproduce the bug.
    fid = f"{game_id}-{horizon}-{start}-{player_id}"
    return ledger.append({
        "record_type": "FORECAST", "game_id": game_id,
        "player_id": player_id, "team": team,
        "forecast_horizon": horizon, "scheduled_start": start,
        "forecast_id": fid, "expected_sog": 1.4,
        "meaningful_expected_participant": True,
    })


def test_colliding_teams_quarantined_and_hash_join_is_idempotent(tmp_path, monkeypatch):
    forecasts = F.Ledger(tmp_path / "forecasts.jsonl")
    a = add_forecast(forecasts, player_id=999, team="WSH")
    b = add_forecast(forecasts, player_id=999, team="PIT")
    c = add_forecast(forecasts, player_id=100, team="WSH")
    path = tmp_path / "grades.jsonl"
    called = []
    def official(fetcher, gid, date):
        called.append((gid, date))
        return "OFF", {
            "players": {100: {"sog": 2, "toi": 1100, "pp_toi": 100, "shifts": 20, "team_id": 77}},
            "team_sog": {77: 26},
        }, {"schedule_state": "OK", "official_start_utc": START}
    monkeypatch.setattr(G, "official_game", official)
    assert G.identity_collisions(forecasts.rows()) == {a["row_hash"], b["row_hash"]}
    before = forecasts.path.read_bytes()
    assert G.grade(ledger=forecasts, grades_path=path, now_fn=NOW, log=lambda *args: None) == 3
    assert forecasts.path.read_bytes() == before
    graded = F.Ledger(path).rows()
    assert [x["record_type"] for x in graded].count("UNGRADED") == 2
    assert [x["record_type"] for x in graded].count("GRADE") == 1
    assert {x["forecast_row_hash"] for x in graded} == {a["row_hash"], b["row_hash"], c["row_hash"]}
    assert {x["reason"] for x in graded if x["record_type"] == "UNGRADED"} == {"AMBIGUOUS_PLAYER_TEAM_IDENTITY"}
    pairs = G.join(ledger=forecasts, grades_path=path, censor_path=tmp_path / "absent.jsonl")
    assert len(pairs) == 1
    assert pairs[0][0]["row_hash"] == c["row_hash"] and pairs[0][1]["actual_sog"] == 2
    assert G.grade(ledger=forecasts, grades_path=path, now_fn=NOW, log=lambda *args: None) == 0
    assert len(F.Ledger(path).rows()) == 3
    assert len(called) == 1


def test_same_player_in_revised_start_is_a_different_decision(tmp_path):
    l = F.Ledger(tmp_path / "decisions.jsonl")
    a = add_forecast(l, player_id=201, team="WSH")
    b = add_forecast(l, player_id=201, team="WSH", start="2026-10-08T01:00:00Z")
    assert a["row_hash"] != b["row_hash"]
    assert G.identity_collisions(l.rows()) == set()


def test_final_without_complete_stats_does_not_invent_zero(tmp_path, monkeypatch):
    forecasts = F.Ledger(tmp_path / "fc.jsonl")
    add_forecast(forecasts, player_id=222, team="PIT")
    grades = tmp_path / "gr.jsonl"
    monkeypatch.setattr(G, "official_game", lambda *args: (
        "OFF", None, {"schedule_state": "OK", "official_start_utc": START}))
    assert G.grade(ledger=forecasts, grades_path=grades, now_fn=NOW, log=lambda *args: None) == 0
    assert F.Ledger(grades).rows() == []


def test_existing_grade_for_ambiguous_identity_fails_closed(tmp_path):
    forecasts = F.Ledger(tmp_path / "fc.jsonl")
    a = add_forecast(forecasts, player_id=333, team="WSH")
    add_forecast(forecasts, player_id=333, team="PIT")
    grades = tmp_path / "gr.jsonl"
    F.Ledger(grades).append({
        "record_type": "GRADE", "forecast_id": a["forecast_id"],
        "forecast_row_hash": a["row_hash"], "game_id": a["game_id"],
        "player_id": a["player_id"], "forecast_horizon": a["forecast_horizon"],
    })
    with pytest.raises(ValueError, match="existing GRADE for ambiguous"):
        G.grade(ledger=forecasts, grades_path=grades, now_fn=NOW)
    with pytest.raises(ValueError, match="ambiguous player/team"):
        G.join(ledger=forecasts, grades_path=grades, censor_path=tmp_path / "no.jsonl")


def test_grade_rejects_claimed_identity_mismatch(tmp_path):
    forecasts = F.Ledger(tmp_path / "fc.jsonl")
    a = add_forecast(forecasts, player_id=10, team="WSH")
    grades = tmp_path / "gr.jsonl"
    F.Ledger(grades).append({
        "record_type": "GRADE", "forecast_id": a["forecast_id"],
        "forecast_row_hash": a["row_hash"], "game_id": a["game_id"],
        "player_id": 11, "forecast_horizon": a["forecast_horizon"],
    })
    with pytest.raises(ValueError, match="does not match immutable forecast identity"):
        G.join(ledger=forecasts, grades_path=grades, censor_path=tmp_path / "no.jsonl")


def test_duplicate_grades_for_same_hash_rejected(tmp_path):
    forecasts = F.Ledger(tmp_path / "fc.jsonl")
    a = add_forecast(forecasts, player_id=10, team="WSH")
    grades = tmp_path / "gr.jsonl"
    for _ in range(2):
        F.Ledger(grades).append({
            "record_type": "GRADE", "forecast_id": a["forecast_id"],
            "forecast_row_hash": a["row_hash"], "game_id": a["game_id"],
            "player_id": a["player_id"], "forecast_horizon": a["forecast_horizon"],
        })
    with pytest.raises(ValueError, match="duplicate grade"):
        G.join(ledger=forecasts, grades_path=grades, censor_path=tmp_path / "no.jsonl")
    with pytest.raises(ValueError, match="duplicate grading"):
        G.grade(ledger=forecasts, grades_path=grades, now_fn=NOW)


def test_real_october_7_ledger_identity_collisions_are_immutable():
    """Pre-outcome forensic regression test against actual forward hash-chained ledger."""
    ledger = F.Ledger()
    assert ledger.verify() == 926
    forecasts = [x for x in ledger.rows() if x.get("record_type") == "FORECAST"]
    assert len(forecasts) == 926
    october = [x for x in forecasts if x["schedule_date"] == "2026-10-07"]
    assert len(october) == 352
    assert {x["forecast_horizon"] for x in october} == {"T90", "T30"}
    bad = G.identity_collisions(october)
    assert len(bad) == 8
    for horizon in ("T90", "T30"):
        rows = [x for x in october if x["forecast_horizon"] == horizon]
        assert len(rows) == 176
        quarantined = [x for x in rows if x["row_hash"] in bad]
        assert len(quarantined) == 4
        assert {x["player_id"] for x in quarantined} == {8477845, 8482148}
        assert {x["team"] for x in quarantined} == {"WSH", "PIT"}
        assert len([x for x in rows if x["row_hash"] not in bad]) == 172
        assert all(x["comparators"]["human_frozen_mean"] is None for x in rows)
        assert all(x["availability_state"] == "NO_ROSTER_DATA_PUBLISHED_YET" for x in rows)
