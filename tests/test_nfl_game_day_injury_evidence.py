"""Source-truth NFL injury availability tests: no fake active/inactive status."""
from datetime import datetime,timedelta,timezone

import pytest
import nfl_game_day_injury_evidence as I

NOW=datetime(2026,11,7,23,0,tzinfo=timezone.utc)
TIP=datetime(2026,11,8,18,0,tzinfo=timezone.utc)
DATE="2026-11-07T19:00:00Z"

class FakeResponse:
    def __init__(self,payload):self.payload=payload
    def raise_for_status(self):pass
    def json(self):return self.payload

def payload(status="Out",date=DATE,team="DAL",player="Fake Receiver",pid="1001"):
    return {
        "season":{"year":2026,"type":2},
        "injuries":[
            {"team":{"abbreviation":team},
             "injuries":[{"athlete":{"id":pid,"displayName":player},
                          "status":status,"date":date}]}
        ]
    }

def snapshot(source):
    return I.capture_league_injuries(2026,NOW,http_get=lambda url,timeout:FakeResponse(source),clock=lambda:NOW)

def test_recent_verifiable_out_is_hard_excluded_only_for_exact_team_and_name():
    report=snapshot(payload())
    assert report["status"]=="CAPTURED"
    a=I.classify_for_pick(report,"DAL","Fake Receiver",NOW,TIP)
    assert a["hard_out"] is True
    assert a["status"]=="RECENT_VERIFIED_REPORTED_OUT"
    assert a["espn_athlete_id"]=="1001"
    assert I.classify_for_pick(report,"MIA","Fake Receiver",NOW,TIP)["hard_out"] is False
    assert I.classify_for_pick(report,"DAL","Different Receiver",NOW,TIP)["hard_out"] is False
    # Suffix normalization is okay but cannot cross team identity.
    assert I.classify_for_pick(report,"DAL","Fake Receiver Jr.",NOW,TIP)["hard_out"] is True
    assert I.classify_for_pick(report,"DAL","Fake Receiver",TIP,TIP)["hard_out"] is False

def test_doubtful_questionable_are_not_converted_to_certain_absence():
    for stat in ("Doubtful","Questionable","Limited","Day-to-Day"):
        result=I.classify_for_pick(snapshot(payload(stat)),"DAL","Fake Receiver",NOW,TIP)
        assert result["status"]=="INJURY_PARTICIPATION_UNCERTAIN"
        assert result["hard_out"] is False

def test_stale_future_undated_and_wrong_year_cannot_hard_remove():
    for date in ("2026-09-12T00:00:00Z","2026-11-10T00:00:00Z",
                 "2026-11-07",None):
        result=I.classify_for_pick(snapshot(payload(date=date)),"DAL","Fake Receiver",NOW,TIP)
        assert result["hard_out"] is False
    snap=snapshot({**payload(),"season":{"year":2025}})
    assert snap["status"]=="SOURCE_UNAVAILABLE"
    assert snap["reason"]=="WRONG_OR_MISSING_SEASON"
    assert I.classify_for_pick(snap,"DAL","Fake Receiver",NOW,TIP)["hard_out"] is False

def test_contradictory_latest_reports_and_unidentified_athletes_fail_closed():
    data=payload()
    data["injuries"][0]["injuries"].append({
        "athlete":{"id":"1001","displayName":"Fake Receiver"},
        "status":"Questionable","date":DATE
    })
    r=I.classify_for_pick(snapshot(data),"DAL","Fake Receiver",NOW,TIP)
    assert r["status"]=="CONFLICTING_LATEST_REPORTS"
    assert not r["hard_out"]
    data=payload()
    data["injuries"][0]["injuries"].append({
        "athlete":{"id":"2222","displayName":"Fake Receiver"},
        "status":"Out","date":DATE
    })
    r=I.classify_for_pick(snapshot(data),"DAL","Fake Receiver",NOW,TIP)
    assert r["status"]=="AMBIGUOUS_ATHLETE_ID"
    assert not r["hard_out"]
    r=I.classify_for_pick(snapshot(payload(pid="")),"DAL","Fake Receiver",NOW,TIP)
    assert r["status"]=="AMBIGUOUS_ATHLETE_ID"

def test_report_later_than_captured_time_and_capture_delay_are_unusable():
    report=snapshot(payload())
    wrong={**report,"checked_at_utc":"2026-11-07T23:30:00Z"}
    assert not I.classify_for_pick(wrong,"DAL","Fake Receiver",NOW,TIP)["hard_out"]
    old={**report,"checked_at_utc":"2026-11-07T21:00:00Z"}
    assert not I.classify_for_pick(old,"DAL","Fake Receiver",NOW,TIP)["hard_out"]

def test_filters_only_undoubted_out_but_keeps_numeric_originals_and_uncertainty():
    report=snapshot({
        "season":{"year":2026},"injuries":[{
            "team":{"abbreviation":"DAL"},
            "injuries":[
                {"athlete":{"id":"1","displayName":"Fake Receiver"},"status":"Out","date":DATE},
                {"athlete":{"id":"2","displayName":"Risky Runner"},"status":"Doubtful","date":DATE}
            ]
        }]
    })
    picks=[
        {"player":"Fake Receiver","team":"DAL","opponent":"PHI",
         "market":"receiving_yards","projected_median":92.5,"model_prob":.76},
        {"player":"Risky Runner","team":"DAL","opponent":"PHI",
         "market":"rushing_yards","projected_median":45.2,"model_prob":.63},
        {"player":"Fake Receiver","team":"PHI","opponent":"DAL",
         "market":"receiving_yards","projected_median":60.4,"model_prob":.65},
    ]
    scheduled=[{"home_team":"DAL","away_team":"PHI","kickoff_utc":TIP.isoformat()}]
    got,evidence=I.apply_to_serving_picks(picks,report,scheduled,NOW)
    assert len(got)==2
    assert got[0]["player"]=="Risky Runner"
    assert got[0]["projected_median"]==45.2
    assert got[0]["model_prob"]==.63
    assert got[0]["game_day_availability_uncertain"] is True
    assert got[1]==picks[2]
    assert evidence["confirmed_out_suppressed"]==1
    assert evidence["uncertain_participation_picks"]==1
    assert evidence["does_not_modify_model_medians_or_probabilities"] is True
    assert len(picks)==3
    assert "game_day_availability_uncertain" not in picks[1]
    assert I.apply_to_serving_picks(picks,I._empty(NOW,"OFFLINE"),scheduled,NOW)[0]==picks

def test_rejects_unscheduled_pick_and_duplicate_scheduled_game_identity():
    p={"player":"Fake Receiver","team":"DAL","opponent":"WAS","market":"receiving_yards"}
    with pytest.raises(ValueError,match="WITHOUT_SCHEDULED_GAME"):
        I.apply_to_serving_picks([p],snapshot(payload()),[
            {"home_team":"DAL","away_team":"PHI","kickoff_utc":TIP.isoformat()}],NOW)
    with pytest.raises(ValueError,match="DUPLICATE_SCHEDULE_PAIR"):
        I.apply_to_serving_picks([],snapshot(payload()),[
            {"home_team":"DAL","away_team":"PHI","kickoff_utc":TIP.isoformat()},
            {"home_team":"PHI","away_team":"DAL","kickoff_utc":TIP.isoformat()}],NOW)

def test_source_http_exception_does_not_invent_active_or_out():
    def broken(url,timeout):
        raise OSError("gateway down")
    src=I.capture_league_injuries(2026,NOW,http_get=broken,clock=lambda:NOW)
    assert src["status"]=="SOURCE_UNAVAILABLE"
    assert src["reason"]=="HTTP_OR_JSON_OSError"
    assert not I.classify_for_pick(src,"DAL","Fake Receiver",NOW,TIP)["hard_out"]
