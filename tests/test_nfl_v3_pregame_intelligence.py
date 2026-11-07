"""Live model intelligence must be source-matched, pregame and fail closed."""
import copy
import json
from datetime import datetime,timezone
import pytest
from nfl_v3_pregame_intelligence import build,SCHEMA

NOW=datetime(2026,11,7,22,0,tzinfo=timezone.utc)
AT="2026-11-07T21:30:00Z"
KICK="2026-11-08T18:00:00Z"


def fixture():
    serving={
        "generated_at_utc":AT,"season":2026,"week":9,
        "scheduled_games":[
            {"home_team":"ATL","away_team":"CIN","kickoff_utc":KICK},
            {"home_team":"IND","away_team":"DAL","kickoff_utc":KICK}],
        "picks":[
            {"player":"Player A","team":"ATL","opponent":"CIN",
             "market":"receiving_yards","kickoff_utc":KICK},
            {"player":"Player B","team":"ATL","opponent":"CIN",
             "market":"rushing_yards","kickoff_utc":KICK},
            {"player":"Player C","team":"DAL","opponent":"IND",
             "market":"anytime_touchdowns","kickoff_utc":KICK}],
    }
    shadow={"schema":"NFL_V2_SHADOW","games":[{
        "game_id":"2026_09_CIN_ATL","away":"CIN","home":"ATL",
        "kickoff_utc":KICK,
        "horizons":{"T24":{"n_records":400,"n_draws":100000}}}]}
    roles={
        "schema":"nfl-role-evidence-coverage-v1",
        "season":2026,"week":9,
        "forecast_generated_at_utc":AT,"captured_at_utc":AT,
        "player_market_evidence":[
            {"player":"Player A","team":"ATL","opponent":"CIN","market":"receiving_yards",
             "kickoff_utc":KICK,"roster_status":"ROSTER_LISTED_NOT_STARTER_VERIFIED",
             "role_evidence_complete_for_showcase":True,"warnings":[],
             "has_verified_starting_role":False},
            {"player":"Player B","team":"ATL","opponent":"CIN","market":"rushing_yards",
             "kickoff_utc":KICK,"roster_status":"INJURY_CAUTION_NOT_CONFIRMED_ACTIVE",
             "role_evidence_complete_for_showcase":False,"warnings":["CURRENT_ROSTER_INJURY"],
             "has_verified_starting_role":False}
        ]
    }
    conflicts={
        "role_conflicts":[{"game_id":"2026_09_CIN_ATL","team":"ATL",
                           "horizon":"T24","flags":["MULTIPLE_QBS_PROJECTED_FOR_SUBSTANTIVE_PASSING"]}]
    }
    return serving,shadow,roles,conflicts


def test_game_intelligence_evidence_stays_distinct_from_actual_model_and_starter_certification():
    args=fixture()
    before=json.dumps(args,sort_keys=True)
    report=build(*args,now=NOW)
    assert report["schema"]==SCHEMA
    assert report["new_model_prediction_changes"] is False
    assert report["unverified_starters_are_not_treated_as_confirmed"] is True
    assert report["summary"]["games_remaining"]==2
    assert report["summary"]["serving_player_outcomes"]==3
    assert report["summary"]["games_with_immutable_T24"]==1
    assert report["summary"]["games_with_immutable_T90"]==0
    assert report["summary"]["games_with_QB_conflicts"]==1
    assert report["summary"]["roster_injury_cautions"]==1
    game=next(x for x in report["games"] if x["home"]=="ATL")
    assert game["serving_yardage_outcomes"]==2
    assert game["served_yards_with_matched_role_evidence"]==2
    assert game["role_evidence_complete_no_warnings"]==1
    assert "UNRESOLVED_QB_ROLE_CONTRADICTION" in game["unresolved_conditions"]
    assert "INJURY_OR_ROSTER_STATUS_CAUTION" in game["unresolved_conditions"]
    assert game["shadow"]["T24"]["forecast_record_count"]==400
    assert game["shadow"]["T90"]["forecast_present"] is False
    assert json.dumps(args,sort_keys=True)==before


def test_future_only_filters_games_that_have_started_and_their_serving_rows():
    a=list(fixture())
    a[0]["scheduled_games"][1]["kickoff_utc"]="2026-11-07T21:45:00Z"
    a[0]["picks"][-1]["kickoff_utc"]="2026-11-07T21:45:00Z"
    report=build(*a,now=NOW)
    assert report["summary"]["games_remaining"]==1
    assert report["summary"]["serving_player_outcomes"]==2


def test_identity_mismatch_and_postkickoff_picks_fail_closed():
    a=list(fixture())
    a[2]["forecast_generated_at_utc"]="2026-11-07T21:20:00Z"
    with pytest.raises(ValueError,match="WRONG_FORECAST"):
        build(*a,now=NOW)
    b=list(fixture())
    b[0]["picks"][0]["kickoff_utc"]="2026-11-07T21:00:00Z"
    with pytest.raises(ValueError,match="POSTKICKOFF_SERVING_PICK"):
        build(*b,now=NOW)


def test_conflicting_shadow_and_duplicate_roster_cannot_create_false_certainty():
    a=list(fixture())
    a[1]["games"].append(copy.deepcopy(a[1]["games"][0]))
    with pytest.raises(ValueError,match="DUPLICATE_SHADOW_GAME"):
        build(*a,now=NOW)
    b=list(fixture())
    b[2]["player_market_evidence"].append(
        copy.deepcopy(b[2]["player_market_evidence"][0]))
    with pytest.raises(ValueError,match="DUPLICATE_ROLE_EVIDENCE"):
        build(*b,now=NOW)


def test_missing_role_source_never_fabricates_verified_athlete():
    a=list(fixture())
    a[2]["player_market_evidence"]=[]
    report=build(*a,now=NOW)
    g=next(x for x in report["games"] if x["home"]=="ATL")
    assert g["role_evidence_rows"]==0
    assert g["served_yards_with_matched_role_evidence"]==0
    assert "PUBLISHED_YARDS_WITHOUT_MATCHED_ROLE_EVIDENCE" in g["unresolved_conditions"]
    assert g["shadow"]["T24"]["game_start_verified"] is False
