"""NHL coverage SLA must detect games entirely missing from model ledger."""
import hashlib
import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import pytest
import nhl_v2_independent_schedule_sla as S

NOW=datetime(2026,10,11,22,15,tzinfo=timezone.utc)
LOCK=datetime(2026,10,7,tzinfo=timezone.utc)
START="2026-10-11T22:00:00Z"
SCHEDULE={"gameWeek":[{"games":[
    {"id":2026020123,"gameType":2,"gameState":"FINAL",
     "gameScheduleState":"OK","startTimeUTC":START}]}]}

def forecast(horizon="T90",player_id=8470001,generated="2026-10-11T20:28:00Z"):
    return {"record_type":"FORECAST","game_id":2026020123,
            "player_id":player_id,
            "forecast_horizon":horizon,
            "scheduled_start":START,
            "generated_at":generated}


def test_official_schedule_missing_game_is_red_even_with_zero_model_records():
    result=S.evaluate([SCHEDULE],[],NOW,LOCK,hours=10)
    assert result["expected_game_horizon_decisions"]==2
    assert result["missing_or_failed"]==2
    assert all(x["reason"]=="NO_LEDGER_DECISION_RECORDED"
               for x in result["missed_examples"])
    assert result["status"]=="FAILED_REAL_PREGAME_CUTOFF_SLA"


def test_one_real_horizon_cannot_cover_others():
    out=S.evaluate([SCHEDULE],[forecast()],NOW,LOCK,hours=10)
    assert out["genuine_pregame_game_horizon_forecasts"]==1
    assert out["missing_or_failed"]==1
    assert out["per_horizon"]["T90"]["forecasted"]==1
    assert out["per_horizon"]["T30"]["missing"]==1


def test_postcutoff_forecast_does_not_forge_successful_delivery():
    fc=forecast(generated="2026-10-11T20:31:00Z")
    result=S.evaluate([SCHEDULE],[fc],NOW,LOCK,hours=10)
    assert result["per_horizon"]["T90"]["missing"]==1


def test_player_id_collision_no_longer_counts_as_delivered_coverage():
    out=S.evaluate([SCHEDULE],[forecast(),forecast()],NOW,LOCK,hours=10)
    assert out["genuine_pregame_game_horizon_forecasts"]==0
    mixed=S.evaluate([SCHEDULE],[forecast(),forecast(),forecast(player_id=8470002)],NOW,LOCK,hours=10)
    assert mixed["genuine_pregame_game_horizon_forecasts"]==1


def test_ledger_source_cannot_supply_or_invent_official_nhl_schedule():
    assert S.evaluate([{"gameWeek":[]}],[forecast()],NOW,LOCK,hours=10)["expected_game_horizon_decisions"]==0
    with pytest.raises(ValueError,match="schedule"):
        S.evaluate([{}],[],NOW,LOCK,hours=10)


def test_hash_chain_verification_does_not_accept_edited_original_forecasts(tmp_path):
    p=tmp_path/"ledger.jsonl"
    first={"record_type":"FORECAST","game_id":2026020123,"seq":0,
           "prev_hash":"0"*64}
    first["row_hash"]=hashlib.sha256(json.dumps(first,sort_keys=True,
                     separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    p.write_text(json.dumps(first)+"\n")
    rows,checksum=S.verify_ledger(p)
    assert len(rows)==1 and len(checksum)==64
    first["game_id"]=2026020999
    p.write_text(json.dumps(first)+"\n")
    with pytest.raises(ValueError,match="hash-chain"):
        S.verify_ledger(p)
