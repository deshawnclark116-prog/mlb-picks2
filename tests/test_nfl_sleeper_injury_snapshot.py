"""Proof of once-a-day provider requests and no fake official inactives."""
import json
from datetime import datetime,timedelta,timezone

import pytest
import nfl_sleeper_injury_snapshot as S

NOW=datetime(2026,11,8,13,16,tzinfo=timezone.utc)
TIP=datetime(2026,11,8,18,0,tzinfo=timezone.utc)

class R:
    def __init__(self,payload):self._payload=payload
    def raise_for_status(self):pass
    @property
    def content(self):return b"{}"
    def json(self):return self._payload

def player(i,st=None,team="DAL",name=None,gsis=None):
    return {"sport":"nfl","team":team,"full_name":name or "Sleeper Mock "+str(i),
            "injury_status":st,"gsis_id":gsis or f"00-{i:07d}",
            "espn_id":str(i),"status":"Active"}

def fake_payload():
    data={str(i):player(i, "Out" if i%100==0 else
                             "Questionable" if i%47==0 else
                             "IR" if i%57==0 else None)
          for i in range(1,1802)}
    data["100"]={**player(100,"Out",name="Example Receiver",gsis="00-0000100")}
    data["47"]={**player(47,"Doubtful",name="Example Runner",gsis="00-0000047")}
    return data

def capture(tmp_path):
    path=tmp_path/"snapshot.json"
    calls=[]
    def get(url,timeout,headers):
        calls.append(url)
        assert url==S.URL and timeout<=30
        assert headers["User-Agent"]
        return R(fake_payload())
    d,new=S.capture_once_per_day(path,NOW,http_get=get,clock=lambda:NOW)
    assert new is True and len(calls)==1
    assert d["injured_player_count"]>25
    assert not d["source_issue_time_verified"]
    path.write_text(json.dumps(d))
    return d,path,calls,get

def test_once_per_day_fetch_is_deterministic_and_previous_payload_is_reused(tmp_path):
    d,path,calls,fetch=capture(tmp_path)
    again,new=S.capture_once_per_day(path,NOW+timedelta(hours=4),http_get=fetch,clock=lambda:NOW+timedelta(hours=4))
    assert not new
    assert len(calls)==1
    assert again==d
    assert again["injury_start_date_is_not_status_report_time"] is True
    assert len(d["players"])>25
    assert S.read(path)["schema"]==S.SCHEMA

def test_out_ir_and_doubtful_have_correct_different_decisions(tmp_path):
    d,_,_,_=capture(tmp_path)
    out=S.verdict(d,{"player":"Example Receiver","team":"DAL",
                     "player_id":"00-0000100"},NOW,TIP)
    assert out["status"]=="CURRENT_OUT_OR_IR_PLATFORM_DESIGNATION"
    assert out["suppress"] is True
    runner=S.verdict(d,{"player":"Example Runner","team":"DAL",
                        "player_id":"00-0000047"},NOW,TIP)
    assert runner["status"]=="INJURY_PARTICIPATION_UNCERTAIN"
    assert not runner["suppress"]
    assert not S.verdict(d,{"player":"Example Receiver","team":"PHI","player_id":"00-0000100"},NOW,TIP)["suppress"]
    assert S.verdict(d,{"player":"Example Receiver","team":"DAL","player_id":"00-9999999"},NOW,TIP)["status"]=="GSIS_PLAYER_ID_CONFLICT"
    assert not S.verdict(d,{"player":"Example Receiver","team":"DAL","player_id":"00-9999999"},NOW,TIP)["suppress"]

def test_out_status_not_used_after_game_or_beyond_20hr_source_ttl(tmp_path):
    d,_,_,_=capture(tmp_path)
    p={"player":"Example Receiver","team":"DAL","player_id":"00-0000100"}
    assert not S.verdict(d,p,TIP,TIP)["suppress"]
    later=NOW+timedelta(hours=21)
    assert not S.verdict(d,p,later,later+timedelta(hours=2))["suppress"]
    future={**d,"captured_at_utc":(NOW+timedelta(minutes=1)).isoformat()}
    assert not S.verdict(future,p,NOW,TIP)["suppress"]

def test_sleeper_report_does_not_backfill_projections_or_remove_risky_non_out(tmp_path):
    d,_,_,_=capture(tmp_path)
    picks=[
      {"player":"Example Receiver","player_id":"00-0000100","team":"DAL",
       "opponent":"PHI","market":"receiving_yards","projected_median":72.4,"model_prob":.71},
      {"player":"Example Runner","player_id":"00-0000047","team":"DAL",
       "opponent":"PHI","market":"rushing_yards","projected_median":58.1,"model_prob":.64},
      {"player":"Example Receiver","player_id":"00-0000100","team":"PHI",
       "opponent":"DAL","market":"receiving_yards","projected_median":51.2,"model_prob":.62},
    ]
    sch=[{"home_team":"DAL","away_team":"PHI","kickoff_utc":TIP.isoformat()}]
    got,audit=S.apply(picks,d,sch,NOW)
    assert [x["team"] for x in got]==["DAL","PHI"]
    assert got[0]["player"]=="Example Runner"
    assert got[0]["game_day_availability_uncertain"] is True
    assert got[0]["model_prob"]==.64 and got[0]["projected_median"]==58.1
    assert got[1]==picks[2]
    assert audit["suppressed_current_out_or_ir"]==1
    assert audit["uncertain_participation_picks"]==1
    assert audit["source_does_not_verify_official_inactives"]
    assert audit["no_numerical_model_change"]
    assert len(picks)==3

def test_corrupt_or_missing_cache_may_not_claim_healthy_or_inactive(tmp_path):
    path=tmp_path/"missing"
    assert S.read(path)["status"]=="SOURCE_NOT_CACHED"
    path.write_text("not json")
    assert S.read(path)["status"]=="SOURCE_NOT_CACHED"
    path.write_text('{"schema":"BAD","players":[],"status":"CAPTURED"}')
    assert S.read(path)["status"]=="CACHE_SCHEMA_INVALID"
    assert not S.verdict(S.read(path),{"team":"DAL","player":"Example Receiver"},NOW,TIP)["suppress"]

def test_ambiguous_name_and_missing_external_ids_are_not_hard_excluded(tmp_path):
    d,_,_,_=capture(tmp_path)
    row=next(x for x in d["players"] if x["player"]=="Example Receiver")
    duplicate={**row,"sleeper_id":"another"}
    wrong={**d,"players":[*d["players"],duplicate]}
    assert S.verdict(wrong,{"player":"Example Receiver","team":"DAL"},NOW,TIP)["status"]=="INJURY_IDENTITY_AMBIGUOUS"
    no_id={**d,"players":[
        {**x,"gsis_id":None,"espn_id":None}
        if x["player"]=="Example Receiver" else x for x in d["players"]
    ]}
    assert S.verdict(no_id,{"player":"Example Receiver","team":"DAL"},NOW,TIP)["status"]=="INJURY_PERSON_ID_UNVERIFIED"

def test_unreasonable_or_partial_provider_response_does_not_overwrite_a_previous_cached_file(tmp_path):
    path=tmp_path/"snapshot.json"
    path.write_text("known original")
    with pytest.raises(ValueError,match="UNVERIFIABLE"):
        S.capture_once_per_day(path,NOW,http_get=lambda u,timeout,headers:R({}),
                               clock=lambda:NOW)
    assert path.read_text()=="known original"
    with pytest.raises(ValueError,match="UNEXPECTEDLY_LARGE"):
        class Big:
            def raise_for_status(self):pass
            @property
            def content(self):return b"X"*22_000_001
        S.capture_once_per_day(path,NOW,http_get=lambda u,timeout,headers:Big(),
                               clock=lambda:NOW)
    assert path.read_text()=="known original"


def test_live_daily_capture_is_only_triggered_once_by_workflow_deployment_and_schedule():
    from pathlib import Path
    workflow=(Path(__file__).resolve().parents[1]/".github/workflows/nfl_sleeper_daily_injury_snapshot.yml").read_text()
    assert "cron: \"15 13 * * *\"" in workflow
    assert "branches: [main]" in workflow
    assert "paths:" in workflow
    assert "- .github/workflows/nfl_sleeper_daily_injury_snapshot.yml" in workflow
    assert "git add -- docs/nfl_sleeper_injury_snapshot.json" in workflow
    assert "concurrency:" in workflow and "cancel-in-progress: false" in workflow
    assert "on:" in workflow and "workflow_dispatch: {}" in workflow
