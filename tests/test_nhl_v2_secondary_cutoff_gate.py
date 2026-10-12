"""Real NHL cutoff trigger, no arbitrary backfills."""
from datetime import datetime,timezone
import nhl_v2_secondary_cutoff_gate as B

NOW=datetime(2026,10,11,20,23,tzinfo=timezone.utc)
def schedule(start="2026-10-11T22:00:00Z",game_id=1):
    return {"gameWeek":[{"date":"2026-10-11","games":[{
        "id":game_id,"gameType":2,"gameState":"FUT",
        "gameScheduleState":"OK","startTimeUTC":start
    }]}]}


def test_open_window_requires_real_regulation_start():
    got=B.due(NOW,schedule())
    assert len(got)==1
    assert got[0]["horizon_minutes"]==90
    assert got[0]["seconds_until_cutoff"]==420


def test_after_deadline_skipped():
    later=datetime(2026,10,11,20,31,tzinfo=timezone.utc)
    assert B.due(later,schedule())==[]


def test_invalid_game_states_refuse_trigger():
    for attribute,value in [("gameState","OFF"),("gameType",3),
                            ("gameScheduleState","PPD")]:
        src=schedule()
        src["gameWeek"][0]["games"][0][attribute]=value
        assert B.due(NOW,src)==[]


def test_green_zero_forecast_run_cannot_suppress_an_eligible_backup():
    recent={"workflow_runs":[{"event":"schedule","conclusion":"success",
                              "created_at":"2026-10-11T20:21:00Z"}]}
    failed={"workflow_runs":[{"event":"schedule","conclusion":"failure",
                              "created_at":"2026-10-11T20:21:00Z"}]}
    assert B.decision(NOW,lambda:schedule(),lambda:recent)["should_run"]
    assert B.decision(NOW,lambda:schedule(),lambda:failed)["should_run"]


def test_missing_primary_and_missing_schedule_have_opposite_fates():
    assert B.decision(NOW,lambda:schedule(),lambda:{"workflow_runs":[]})["should_run"]
    assert not B.decision(NOW,lambda:{})["should_run"]


def test_24_hour_horizon_still_available():
    s=schedule("2026-10-12T20:30:00Z")
    got=B.due(NOW,s)
    assert len(got)==1 and got[0]["horizon_minutes"]==1440


def test_advance_wake_window_prepares_model_before_15_minute_deadline():
    now=datetime(2026,10,11,19,0,tzinfo=timezone.utc)
    # 90-min deadline at 20:30, still 90 minutes away.
    candidates=B.upcoming(now,schedule("2026-10-11T22:00:00Z"))
    assert len(candidates)==1
    assert candidates[0]["horizon_minutes"]==90
    assert candidates[0]["seconds_until_cutoff"]==5400
    assert B.decision(now,lambda:schedule("2026-10-11T22:00:00Z"))["should_run"]


def test_worker_gate_cannot_invite_missed_historical_game():
    now=datetime(2026,10,11,23,0,tzinfo=timezone.utc)
    assert B.upcoming(now,schedule("2026-10-11T22:00:00Z"))==[]
    assert not B.decision(now,lambda:schedule("2026-10-11T22:00:00Z"))["should_run"]
