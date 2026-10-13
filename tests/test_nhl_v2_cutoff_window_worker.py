"""Prove warm NHL B2 worker genuinely captures every available pregame cutoff."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import pytest
import nhl_v2_cutoff_window_worker as W

UTC=timezone.utc
CUTOFF=datetime(2026,10,12,22,0,tzinfo=UTC)
START="2026-10-12T23:30:00Z"

def schedule(start=START,game_id=2026021234):
    return {"gameWeek":[{"games":[{"id":game_id,"gameType":2,
            "gameState":"FUT","gameScheduleState":"OK",
            "startTimeUTC":start}]}]}

class Clock:
    def __init__(self, now):
        self.value=now
        self.waits=[]
    def now(self):
        return self.value
    def sleep(self,secs):
        self.waits.append(secs)
        self.value+=timedelta(seconds=secs)

def path(root):
    root=Path(root)
    ledger=root/"nhl_models/nhl_player_outcome_v2/phase1a_sog_forward/ledger.jsonl"
    ledger.parent.mkdir(parents=True,exist_ok=True)
    ledger.write_text("")
    return ledger


def test_warm_worker_captures_before_cutoff_and_persists_before_ending():
    with tempfile.TemporaryDirectory() as td:
        fc=path(td);clk=Clock(CUTOFF-timedelta(minutes=16))
        events=[]
        def model(batch):
            events.append(("model",clk.now()))
            for e in batch:
                with fc.open("a") as out:
                    out.write(json.dumps({
                        "record_type":"FORECAST","game_id":int(e["game_id"]),
                        "player_id":8470001,"forecast_horizon":"T90",
                        "scheduled_start":START,
                        "generated_at":clk.now().isoformat(),
                        "cutoff_at":e["cutoff"]})+"\n")
        def persist(batch):
            events.append(("persist",clk.now()))
        def publish():
            events.append(("publish",clk.now()))
        out=W.session(Path(td),Path(td),duration_minutes=19,
                      now_fn=clk.now,sleep=clk.sleep,get_schedule=schedule,
                      forecast=model,persist=persist,publish_fn=publish)
        assert out["verified_forecast_decisions"]==1
        assert len(events)==3
        assert [e[0] for e in events]==["model","persist","publish"]
        assert events[0][1]<CUTOFF
        assert events[0][1]>=CUTOFF-timedelta(minutes=15)
        assert events[1][1]==events[0][1]


def test_no_genuine_forecast_at_cutoff_turns_the_run_red():
    with tempfile.TemporaryDirectory() as td:
        path(td);clk=Clock(CUTOFF-timedelta(minutes=13))
        with pytest.raises(RuntimeError,match="no real pregame forecast"):
            W.session(Path(td),Path(td),duration_minutes=2,
                      now_fn=clk.now,sleep=clk.sleep,get_schedule=schedule,
                      forecast=lambda batch:None,persist=lambda batch:None,
                      publish_fn=lambda:None)


def test_no_new_model_execution_when_games_already_over():
    with tempfile.TemporaryDirectory() as td:
        path(td);clk=Clock(CUTOFF+timedelta(minutes=1));calls=[]
        result=W.session(Path(td),Path(td),duration_minutes=2,
                         now_fn=clk.now,sleep=clk.sleep,get_schedule=schedule,
                         forecast=lambda batch:calls.append(batch),
                         persist=lambda batch:None,publish_fn=lambda:None)
        assert result["attempted_game_horizons"]==0
        assert calls==[]


def test_worker_does_not_fake_forecasts_when_schedule_is_unavailable():
    with tempfile.TemporaryDirectory() as td:
        path(td);clk=Clock(CUTOFF-timedelta(minutes=13))
        with pytest.raises(RuntimeError,match="schedule was unavailable"):
            W.session(Path(td),Path(td),duration_minutes=2,
                      now_fn=clk.now,sleep=clk.sleep,
                      get_schedule=lambda: (_ for _ in ()).throw(RuntimeError("upstream offline")),
                      forecast=lambda batch:None,persist=lambda batch:None,
                      publish_fn=lambda:None)


def test_identity_match_is_exact_game_start_horizon():
    candidate={"game_id":"2026021234",
               "start":"2026-10-12T23:30:00+00:00",
               "cutoff":"2026-10-12T22:00:00+00:00",
               "horizon_minutes":90}
    original=[{"record_type":"FORECAST","game_id":2026021234,
               "scheduled_start":START,"forecast_horizon":"T90",
               "player_id":8470001}]
    assert W.decision_status(original,[candidate])[0]["status"]=="FORECAST"
    original[0]["scheduled_start"]="2026-10-12T22:30:00Z"
    assert W.decision_status(original,[candidate])[0]["status"]=="NO_SAVED_DECISION"


def test_original_model_lock_is_verified_at_start_before_first_schedule_poll(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        path(td)
        clk=Clock(CUTOFF-timedelta(minutes=90))
        calls=[]
        def locked_engine_fails(original):
            calls.append(("lock_failure",clk.now()))
            raise RuntimeError("frozen source lock invalid")
        def forbidden_poll():
            calls.append(("unexpected_schedule",clk.now()))
            raise AssertionError("official schedule should not be needed after broken model prewarm")
        monkeypatch.setattr(W,"verify_original",locked_engine_fails)
        with pytest.raises(RuntimeError,match="frozen source lock invalid"):
            W.session(Path(td),Path(td),duration_minutes=1,
                      now_fn=clk.now,sleep=clk.sleep,get_schedule=forbidden_poll,
                      publish_fn=lambda:None)
        assert calls==[("lock_failure",CUTOFF-timedelta(minutes=90))]
