"""Compact NFL grading dashboard must not falsify projections or mix research with bets."""
import copy
import pytest
from nfl_all_grade_summary import build

G="2026_06_WAS_SF"
def inputs():
    production={
        "schema":"NFL_FIRST_SEEN_ALL_SERVED_PREDICTIONS_GRADE_V1",
        "production_vs_research_separate":True,
        "original_count":3,"accounted_for":3,
        "verified_outcome_count":2,"unverified_count":1,
        "directional_wins":1,"directional_losses":0,"directional_pushes":0,
        "status_counts":{"GRADED":2,"NO_VERIFIED_PLAYER_MARKET_ACTUAL_NOT_ZERO":1},
        "by_game":[{"game_id":G,"season":2026,"week":6,"original_predictions":3,
                    "graded":2,"unverified":1,"by_market":{
                        "rushing_yards":{"original_predictions":2,
                                         "actual_verified":1,
                                         "numeric_predictions_scored":1,
                                         "absolute_error_mean":3.5,
                                         "explicit_direction_win":0,"explicit_direction_loss":0,
                                         "explicit_direction_push":0},
                        "anytime_touchdowns":{"original_predictions":1,
                                         "actual_verified":1,
                                         "numeric_predictions_scored":0,
                                         "explicit_direction_win":1,"explicit_direction_loss":0,
                                         "explicit_direction_push":0}}}]
    }
    v2={"schema":"NFL_V2_FIRST_SEEN_REAL_FINAL_NUMERIC_SCORECARD_V1",
        "every_genuine_original_forecast_accounted":True,
        "total_original_pregame_forecasts":5,"graded_player_outcomes":3,
        "unverified_original_forecasts":2,
        "late_original_pregame_forecasts":2,
        "by_horizon":{"T24":{"n":1,"mean_mae":7.0}},
        "by_late_horizon":{"T24_LATE":{"n":2,"mean_mae":4.0}},
        "exclusions":{"NO_UNIQUE_PLAYER_ACTUAL_NOT_ZERO":2},
        "by_game_horizon":[
            {"game_id":G,"horizon":"T24",
             "original_forecasts":3,"graded":1,"unverified":2,
             "markets":{"pass_yds":{"original_forecasts":3,"graded":1,
                                     "unverified":2,"median_mae":9}}},
            {"game_id":G,"horizon":"T24_LATE",
             "original_forecasts":2,"graded":2,"unverified":0,
             "markets":{"rush_yds":{"original_forecasts":2,"graded":2,
                                     "unverified":0,"median_mae":4}}}]}
    coverage={"schema":"NFL_V2_POSTGAME_GRADING_COVERAGE_V1",
        "asof_utc":"2026-10-20T14:16:39Z",
        "status":"ALL_ORIGINAL_PREDICTIONS_ACCOUNTED_PARTIAL_ACTUAL_VERIFICATION",
        "recent_completed_game_count":1,
        "recent_frozen_game_horizons_ungraded":0,
        "recent_game_horizons_missing_original_forecast":1,
        "recent_late_cutoff_pregame_original_predictions":2,
        "recent_unverified_original_predictions":2,
        "recent_completed_games":[{"game_id":G,"home":"SF","away":"WAS",
            "home_score":23,"away_score":17}]}
    return production,v2,coverage

def test_report_keeps_three distinct_truths_and_aggregate_parity():
    d=build(*inputs())
    assert d["game_count"]==1
    assert d["production"]["original_saved"]==3
    assert d["production"]["actuals_verified"]==2
    assert d["production"]["original_recorded_direction_wins"]==1
    assert d["original_v2"]["original_saved"]==5
    assert d["original_v2"]["late_but_pregame_originals_not_on_time_T24_T90"]==2
    assert d["games"][0]["v2_cohorts"]["T24_LATE"]["strict_horizon_qualified"] is False
    assert d["games"][0]["v2_cohorts"]["T24"]["strict_horizon_qualified"] is True
    assert d["games"][0]["serving"]["by_market"]["rushing_yards"]["numeric_predictions_scored"]==1
    assert d["games"][0]["final"]["final_home_score"]==23
    assert d["completed_games_audit"]["archived_game_horizons_missing_from_grade"]==0

def test_original_record_accounting_mismatch_fails_closed():
    p,v,c=inputs()
    p["accounted_for"]=2
    with pytest.raises(ValueError,match="Production"):build(p,v,c)
    p,v,c=inputs()
    v["unverified_original_forecasts"]=1
    with pytest.raises(ValueError,match="V2"):build(p,v,c)
    p,v,c=inputs()
    v["by_game_horizon"][1]["original_forecasts"]=3
    with pytest.raises(ValueError,match="Per-game"):build(p,v,c)

def test_same_unresolved_matchup_in_different_weeks_stays_distinct():
    p,v,c=inputs()
    p["by_game"].append({"game_id":"UNRESOLVED_SF_WAS","season":2026,"week":5,
                         "original_predictions":1,"graded":0,"unverified":1,
                         "by_market":{}})
    p["by_game"].append({"game_id":"UNRESOLVED_SF_WAS","season":2026,"week":6,
                         "original_predictions":1,"graded":0,"unverified":1,
                         "by_market":{}})
    p["original_count"]=p["accounted_for"]=5
    p["unverified_count"]=3
    d=build(p,v,c)
    assert d["game_count"]==3
    assert len({g["game_id"] for g in d["games"]})==3

def test_unknown_data_sources_rejected():
    p,v,c=inputs()
    c["schema"]="fiction"
    with pytest.raises(ValueError,match="coverage"):build(p,v,c)
