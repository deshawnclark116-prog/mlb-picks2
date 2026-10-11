"""Adversarial integrity tests for immutable NHL forward forecasts and grading."""
import json
from datetime import datetime, timezone

import pytest

import nhl_v2_phase1a_sog_forward as F
import nhl_v2_forward_integrity_grade as G

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
        players = {100: {"sog": 2, "toi": 1100, "pp_toi": 100, "shifts": 20, "team_id": 77}}
        players.update({
            pid: {"sog": 0, "toi": 900, "pp_toi": 0, "shifts": 15,
                  "team_id": 77 if pid % 2 == 0 else 88}
            for pid in range(200, 230)
        })
        return "OFF", {
            "players": players, "team_sog": {77: 26, 88: 21},
        }, {"schedule_state": "OK", "official_start_utc": START, "team_ids": {"WSH": 77, "PIT": 88}}
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
        "OFF", None, {"schedule_state": "OK", "official_start_utc": START, "team_ids": {"WSH": 77, "PIT": 88}}))
    assert G.grade(ledger=forecasts, grades_path=grades, now_fn=NOW, log=lambda *args: None) == 0
    assert F.Ledger(grades).rows() == []


def test_sparse_official_final_table_is_not_interpreted_as_absent_players(tmp_path, monkeypatch):
    forecasts = F.Ledger(tmp_path / "fc.jsonl")
    add_forecast(forecasts, player_id=333, team="WSH")
    grades = tmp_path / "gr.jsonl"
    monkeypatch.setattr(G, "official_game", lambda *args: (
        "OFF", {
            "players": {901: {"sog": 1, "toi": 900, "pp_toi": 0, "shifts": 14, "team_id": 77}},
            "team_sog": {77: 1},
        }, {"schedule_state": "OK", "official_start_utc": START, "team_ids": {"WSH": 77, "PIT": 88}}))
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
    assert ledger.verify() >= 926
    forecasts = [x for x in ledger.rows() if x.get("record_type") == "FORECAST"]
    assert len(forecasts) >= 926
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

@pytest.mark.parametrize('wrong_team,changed_start,reason', [
    (True, False, 'OFFICIAL_PLAYER_TEAM_IDENTITY_UNVERIFIED_OR_MISMATCH'),
    (False, True, 'SCHEDULE_START_CHANGED_NO_LINKAGE'),
])
def test_single_team_mismatch_and_schedule_revision_quarantined(tmp_path, monkeypatch, wrong_team, changed_start, reason):
    forecasts = F.Ledger(tmp_path / 'fc.jsonl')
    add_forecast(forecasts, player_id=100, team='WSH')
    players = {pid: {'sog': 1, 'toi': 900, 'pp_toi': 0, 'shifts': 14,
                     'team_id': 77 if pid % 2 == 0 else 88} for pid in range(100, 136)}
    players[100]['team_id'] = 88 if wrong_team else 77
    monkeypatch.setattr(G, 'official_game', lambda *args: ('OFF', {
        'players': players, 'team_sog': {77: 18, 88: 18}}, {
        'schedule_state': 'OK', 'team_ids': {'WSH': 77, 'PIT': 88},
        'official_start_utc': '2026-10-08T01:00:00Z' if changed_start else START}))
    grades = tmp_path / 'grades.jsonl'
    assert G.grade(ledger=forecasts, grades_path=grades, now_fn=NOW) == 1
    assert F.Ledger(grades).rows()[0]['reason'] == reason
    assert G.grade(ledger=forecasts, grades_path=grades, now_fn=NOW) == 0
    assert not G.join(ledger=forecasts, grades_path=grades, censor_path=tmp_path / 'none')


def test_manual_workflow_uses_dispatched_ref_for_checkout_and_push():
    from pathlib import Path
    text = Path('.github/workflows/nhl_v2_phase1a_sog_forward.yml').read_text()
    assert 'ref: ${{ github.ref }}' in text
    assert 'ref: codex/nhl-outcome-engine-v2' not in text
    assert 'HEAD:$GITHUB_REF_NAME' in text
    assert 'python nhl_v2_forward_integrity_grade.py grade' in text


@pytest.mark.parametrize('missing,wrong_sog,expected_ok',[(0,False,True),(2,False,False),(0,True,False)])
def test_official_complete_stats_need_exact_final_boxscore_not_just_row_floor(monkeypatch,missing,wrong_sog,expected_ok):
    import nhl_v2_phase1a_sog_acquire as A
    gid=2026020053
    game={'id':gid,'gameState':'OFF','startTimeUTC':START,'gameScheduleState':'OK',
          'awayTeam':{'abbrev':'WSH','id':77},'homeTeam':{'abbrev':'PIT','id':88}}
    skaters=[{'player_id':pid,'game_id':gid,'team_id':77 if pid<118 else 88,'sog':1,'toi_sec':900,'pp_toi_sec':0,'shifts':14} for pid in range(100,136)]
    box={'id':gid,'gameState':'OFF','awayTeam':{'id':77},'homeTeam':{'id':88},'playerByGameStats':{
        'awayTeam':{'forwards':[{'playerId':p,'sog':1} for p in range(100,112)],'defense':[{'playerId':p,'sog':1} for p in range(112,118)]},
        'homeTeam':{'forwards':[{'playerId':p,'sog':1} for p in range(118,130)],'defense':[{'playerId':p,'sog':1} for p in range(130,136)]}}}
    if missing:skaters=skaters[:-missing]
    if wrong_sog:skaters[0]['sog']=2
    meta={'sha256':'a'*64}
    def fetch(url):return json.dumps(box if '/boxscore' in url else {'gameWeek':[{'games':[game]}]}).encode(),meta
    monkeypatch.setattr(A,'acquire_window',lambda *args,**kwargs:{'provenance':{'summary':meta,'timeonice':meta}})
    monkeypatch.setattr(A,'assemble',lambda windows:({gid:game},skaters))
    st,data,provenance=G.official_game(fetch,gid,'2026-10-07')
    assert (data is not None)==expected_ok
    assert 'boxscore' in provenance


def test_manual_workflow_prevents_forecast_fork_and_checks_upstream_sync():
    from pathlib import Path
    text=Path('.github/workflows/nhl_v2_phase1a_sog_forward.yml').read_text()
    assert 'Forecast ledger has a single owner' in text
    assert 'git merge-base --is-ancestor FETCH_HEAD HEAD' in text
