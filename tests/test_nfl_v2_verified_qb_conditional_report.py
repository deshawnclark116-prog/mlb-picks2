"""When a confirmed starter is mis-modeled as a backup, test the predictive consequence."""
import copy
from datetime import datetime,timezone
import pytest
import nfl_v2_verified_qb_conditional_report as Q

NOW=datetime(2026,10,17,21,30,tzinfo=timezone.utc)
FIELDS=["game_id","player","team","opp","pos","outcome","p_active",
        "exp_opp","role_share","mean","median","sd","p10","p25",
        "p75","p90","p0","p_ge1","unc","unc_reasons","id"]

def row(player,market,p,mean,opp):
    values={"game_id":"2026_06_CAR_PHI","player":player,"team":"PHI",
            "opp":"CAR","pos":"QB","outcome":market,"p_active":p,
            "exp_opp":opp,"role_share":.22 if player=="tanner mckee" else .75,
            "mean":mean,"median":0,"id":player+"_"+market}
    return [values.get(k) for k in FIELDS]

def fixture():
    d={"schema":"nfl-new-engine-shadow-v1","row_fields":FIELDS,
       "status":{"publication_state":"PUBLISHED"},
       "games":[{"game_id":"2026_06_CAR_PHI","home":"PHI","away":"CAR",
          "kickoff_utc":"2026-10-18T17:00:00Z",
          "horizons":{"T24":{"cutoff":"2026-10-17T17:00:00Z"}}}],
       "forecasts":{"T24":[
          row("tanner mckee","pass_yds",.1852,38.6378,5.0061),
          row("tanner mckee","pass_td",.1852,.1882,5.0061),
          row("tanner mckee","int",.1852,.0988,5.0061),
          row("tanner mckee","rush_yds",.1852,2.9552,.4361),
          row("jalen hurts","pass_yds",.0057,1.2074,.1537)],
          "T90":[]}}
    e={"schema":"nfl-confirmed-starting-quarterback-source-evidence-v1",
       "assertions":[{"game_id":"2026_06_CAR_PHI","team":"PHI",
         "opponent":"CAR","kickoff_utc":"2026-10-18T17:00:00Z",
         "verified_starter":"Tanner McKee","out_players":["Jalen Hurts"],
         "official_sources":[{"url":"https://www.philadelphiaeagles.com/news/panthers-vs-eagles-injury-report-2026-nfl-week-6-saquon-barkley-devonta-smith",
          "published_at_utc":"2026-10-16T20:05:00Z",
          "supports":["Jalen Hurts ruled out","Tanner McKee will start"]}]}]}
    return d,e

def test_verified_starter_reconstructs_mean_opportunity_without_faking_forecast():
    b,e=fixture();before=copy.deepcopy(b)
    d=Q.build(b,e,now=NOW);r=d["scenarios"][0]
    assert b==before,"immutable original must never be rewritten"
    assert r["starter_original_p_active"]==.1852
    assert r["official_verified_starter"]=="Tanner McKee"
    assert r["qb_stat_scenarios"]["pass_yds"]["original_unconditional_mean"]==38.6378
    assert r["qb_stat_scenarios"]["pass_yds"]["conditional_mean_if_active"]==208.63
    assert r["conditional_mean_pass_opportunities_if_active"]==27.03
    assert r["status"]=="ROLE_CONFIRMED_MODEL_PROBABILITY_CONTRADICTION"
    assert r["qb_stat_scenarios"]["pass_yds"]["original_median_including_not_playing"]==0
    assert r["not_original_v2_prediction"] and r["requires_full_team_resimulation"]
    assert d["original_immutable_v2_board_unchanged"]

def test_source_published_after_original_cutoff_cannot_justify_backdated_scenario():
    b,e=fixture()
    e["assertions"][0]["official_sources"][0]["published_at_utc"]="2026-10-17T18:00:00Z"
    with pytest.raises(ValueError,match="SOURCE_NOT_PUBLIC_BEFORE_MODEL_CUTOFF"):
        Q.build(b,e,now=NOW)

def test_no_verified_starter_or_wrong_game_identity_fails_closed():
    b,e=fixture();e["assertions"][0]["verified_starter"]="Jalen Hurts"
    with pytest.raises(ValueError,match="SOURCE_DOES_NOT_VERIFY_STARTER"):
        Q.build(b,e,now=NOW)
    b,e=fixture();e["assertions"][0]["opponent"]="DAL"
    with pytest.raises(ValueError,match="SOURCE_GAME_TEAMS_MISMATCH"):
        Q.build(b,e,now=NOW)

def test_no_claim_of_pregame_scenario_after_kickoff():
    b,e=fixture()
    with pytest.raises(ValueError,match="GAME_ALREADY_STARTED"):
        Q.build(b,e,now=datetime(2026,10,18,18,tzinfo=timezone.utc))

def test_qb_active_probability_has_to_match_within_original_forecast():
    b,e=fixture()
    b["forecasts"]["T24"][1][FIELDS.index("p_active")]=.84
    with pytest.raises(ValueError,match="INCONSISTENT_STARTER_PLAY_PROB"):
        Q.build(b,e,now=NOW)

def test_prior_horizon_only_when_latest_not_genuinely_published():
    b,e=fixture();b["games"][0]["horizons"]["T90"]={"cutoff":"2026-10-18T15:30:00Z"}
    d=Q.build(b,e,now=NOW)
    assert len(d["scenarios"])==1
    assert d["scenarios"][0]["horizon"]=="T24"
