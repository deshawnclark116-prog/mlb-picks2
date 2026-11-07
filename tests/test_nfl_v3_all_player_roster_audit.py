"""Every original NFL Phase1 player is separately source-checked; never fake DNP."""
from datetime import datetime,timezone
import copy
import pytest

from nfl_v3_all_player_roster_audit import build, SCHEMA

NOW=datetime(2026,11,7,23,0,tzinfo=timezone.utc)
FUTURE="2026-11-08T18:00:00Z"
FIELDS=[
    "game_id","player","team","opp","pos","outcome",
    "p_active","exp_opp","role_share","mean","median","sd",
    "p10","p25","p75","p90","p0","p_ge1","unc","unc_reasons","id"
]


def datum(name,team,opp,market,identifier):
    return ["2026_09_ARI_SEA",name,team,opp,"QB" if market=="pass_yds" else "DL",
            market,.9,25,None,20,18,4,2,10,23,38,.04,.2,.1,[],identifier]


def doc():
    return {
        "games":[{"game_id":"2026_09_ARI_SEA","home":"SEA","away":"ARI",
                  "kickoff_utc":FUTURE,"horizons":{"T24":{"n_records":4}}}],
        "row_fields":FIELDS,
        "forecasts":{"T24":[
            datum("Known Quarterback","ARI","SEA","pass_yds","a"),
            datum("Known Quarterback","ARI","SEA","pass_td","b"),
            datum("Known Defender","SEA","ARI","tackles","c"),
            datum("Unverified Defender","SEA","ARI","sacks","d"),
            datum("Player on IR","SEA","ARI","def_int","e"),
        ],"T90":[]},
    }


def roster(team,checked):
    assert checked.startswith("2026-11-07T23:00")
    if team=="ARI":
        return {"status":"ROSTER_CAPTURED","players":{
            "known quarterback":[{"status":"ROSTER_LISTED_NOT_STARTER_VERIFIED"}]}}
    if team=="SEA":
        return {"status":"ROSTER_CAPTURED","players":{
            "known defender":[{"status":"INJURY_CAUTION_NOT_CONFIRMED_ACTIVE"}],
            "player on ir":[{"status":"LISTED_UNAVAILABLE"}]}}
    raise AssertionError("unrecognized team")


def test_all_player_markets_are_included_but_per_player_unique_within_game():
    original=doc()
    before=copy.deepcopy(original)
    report=build(original,now=NOW,roster_fetch=roster)
    assert report["schema"]==SCHEMA
    assert report["games_covered"]==1
    assert report["distinct_players_by_game_horizon"]==4
    assert report["roster_identity_status_counts"]=={
        "INJURY_CAUTION_NOT_CONFIRMED_ACTIVE":1,
        "LISTED_UNAVAILABLE":1,
        "ROSTER_IDENTITY_NOT_VERIFIED":1,
        "ROSTER_LISTED_NOT_STARTER_VERIFIED":1
    }
    sea=next(x for x in report["per_game_team_horizon"] if x["team"]=="SEA")
    assert sea["named_players"]==3
    assert sea["listed_unavailable"]==1
    assert sea["unverified_roster_identity"]==1
    assert sea["explicit_injury_cautions"]==1
    assert report["no_numerical_forecast_or_frontend_changes"] is True
    assert report["roster_absence_is_not_verified_player_inactivity"] is True
    assert report["roster_presence_is_not_game_day_starter_certification"] is True
    assert original==before


def test_roster_source_outage_is_unknown_for_every_player():
    def down(team,checked):
        raise TimeoutError("network unavailable")
    r=build(doc(),now=NOW,roster_fetch=down)
    assert r["roster_identity_status_counts"]["ROSTER_SOURCE_UNAVAILABLE"]==4
    assert r["source_status_by_team"]=={
        "ARI":"ROSTER_SOURCE_UNAVAILABLE","SEA":"ROSTER_SOURCE_UNAVAILABLE"}


def test_completed_games_have_no_new_roster_queries_or_false_pretip_certification():
    def prohibited(team,checked):
        raise AssertionError("post-kickoff network call")
    r=build(doc(),now=datetime(2026,11,8,18,1,tzinfo=timezone.utc),
            roster_fetch=prohibited)
    assert r["games_covered"]==0
    assert r["distinct_players_by_game_horizon"]==0
    assert r["per_game_team_horizon"]==[]


def test_wrong_team_and_invalid_forecast_source_fail_closed():
    d=doc()
    d["forecasts"]["T24"][0][2]="CIN"
    with pytest.raises(ValueError,match="TEAM_DOES_NOT_MATCH_GAME"):
        build(d,now=NOW,roster_fetch=roster)
    d=doc()
    d["forecasts"]["T24"][0][5]="fake_market"
    with pytest.raises(ValueError,match="UNKNOWN_MODEL_MARKET"):
        build(d,now=NOW,roster_fetch=roster)
    d=doc()
    d["row_fields"].append("market")
    with pytest.raises(ValueError,match="BAD_FORECAST_ROW_SHAPE"):
        build(d,now=NOW,roster_fetch=roster)


def test_same_name_both_teams_flagged_not_automatically_removed():
    d=doc()
    d["forecasts"]["T24"].append(
        datum("Known Quarterback","SEA","ARI","int","z"))
    r=build(d,now=NOW,roster_fetch=roster)
    assert len(r["ambiguous_same_name_across_opponents"])==1
    assert r["ambiguous_same_name_across_opponents"][0]["identity_verified"] is False
    assert r["distinct_players_by_game_horizon"]==5


def test_two_separate_horizons_are_not_silently_mixed():
    d=doc()
    d["forecasts"]["T90"]=[datum("Known Quarterback","ARI","SEA","pass_yds","a90")]
    r=build(d,now=NOW,roster_fetch=roster)
    assert r["distinct_players_by_game_horizon"]==5
    assert len({x["horizon"] for x in r["per_game_team_horizon"]})==2
