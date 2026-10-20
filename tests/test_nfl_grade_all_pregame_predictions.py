"""No fabricated betting selections, no erased forecasts, no missing-player zeros."""
import json
from datetime import datetime,timezone
import pytest
import nfl_grade_all_pregame_predictions as G
import nfl_recent_espn_final_truth as E

NOW=datetime(2026,10,20,13,0,tzinfo=timezone.utc)
GID="2026_06_WAS_SF"
K="2026-10-20T00:15:00Z"

def game():
    return {"game_id":GID,"season":2026,"week":6,"season_type":"REG",
            "kickoff_utc":K,"home_team":"SF","away_team":"WAS",
            "home_score":23,"away_score":17}

def player(pid="00-000001",name="Sample Runner",rush=64,rec=15,tds=1):
    return {"game_id":GID,"season":2026,"week":6,"season_type":"REG",
            "team":"SF","player_id":pid,"player_name":name,
            "rushing_yards":rush,"receiving_yards":rec,
            "rushing_tds":tds,"receiving_tds":0,"def_sacks":0}

def prediction(market="rushing_yards",player_name="Sample Runner",
               pid="00-000001",line=None,side=None):
    d={"season":2026,"week":6,"market":market,"player":player_name,
       "player_id":pid,"team":"SF","opponent":"WAS",
       "logged_at":"2026-10-19T20:00:00Z","kickoff_utc":K,
       "model_source":"v2_context","projected_median":67.5,
       "projected_mean":71.3,"projected_low":20,"projected_high":110,
       "pick":"Projected 68 yards"}
    if side is not None:
        d["line"]=line
        d["pick"]=f"{side} {line}"
        d["model_prob"]=0.7
    return d

def test_every_original_number_even_without_sportsbook_line_is_graded():
    d=G.score_predictions([prediction()], [game()], [player()],now=NOW)
    assert d["accounted_for"]==d["original_count"]==1
    row=d["all_predictions"][0]
    assert row["status"]=="GRADED"
    assert row["actual"]==64
    assert row["predicted_median"]==67.5
    assert row["absolute_stat_error"]==3.5
    assert row["absolute_mean_error"]==7.3
    assert row["actual_inside_p10_p90"] is True
    assert row["direction_grade"] is None
    assert d["by_game"][0]["by_market"]["rushing_yards"]["numeric_predictions_scored"]==1

def test_original_explicit_side_is_graded_not_invented():
    rec=prediction(side="OVER",line=59.5)
    d=G.score_predictions([rec], [game()], [player()],now=NOW)
    row=d["all_predictions"][0]
    assert row["direction_grade"]=="win"
    assert row["brier_of_explicit_side"]==pytest.approx(0.09)
    assert d["directional_wins"]==1
    assert d["directional_losses"]==0

def test_anytime_touchdown_predictions_are_not_dropped():
    rec=prediction(market="anytime_touchdowns",side="OVER",line=0.5)
    rec["projected_median"]=None
    d=G.score_predictions([rec], [game()], [player()],now=NOW)
    row=d["all_predictions"][0]
    assert row["actual"]==1
    assert row["direction_grade"]=="win"

def test_half_sack_exact_threshold_is_push():
    rec=prediction(market="sacks",side="UNDER",line=0.5)
    rec["projected_median"]=None
    p=player()
    p["def_sacks"]=0.5
    row=G.score_predictions([rec],[game()],[p],now=NOW)["all_predictions"][0]
    assert row["direction_grade"]=="push"
    assert row["brier_of_explicit_side"] is None

def test_wrong_frozen_kickoff_not_graded():
    rec=prediction()
    rec["kickoff_utc"]="2026-10-21T00:15:00Z"
    row=G.score_predictions([rec],[game()],[player()],now=NOW)["all_predictions"][0]
    assert row["status"]=="ORIGINAL_GAME_KICKOFF_MISMATCH"
    assert row["actual"] is None

def test_late_logged_prediction_not_retroactively_graded():
    rec=prediction()
    rec["logged_at"]="2026-10-20T02:00:00Z"
    row=G.score_predictions([rec],[game()],[player()],now=NOW)["all_predictions"][0]
    assert row["status"]=="NOT_PROVABLY_SAVED_BEFORE_KICKOFF"

def test_unverified_player_is_not_silently_scored_zero():
    row=G.score_predictions([prediction()],[game()],[],now=NOW)["all_predictions"][0]
    assert row["status"]=="NO_VERIFIED_PLAYER_MARKET_ACTUAL_NOT_ZERO"
    assert row["actual"] is None

def test_suspect_name_identity_not_matched_across_teams():
    bad=player();bad["team"]="WAS"
    row=G.score_predictions([prediction()],[game()],[bad],now=NOW)["all_predictions"][0]
    assert row["status"]=="NO_VERIFIED_PLAYER_MARKET_ACTUAL_NOT_ZERO"

def test_each_input_forecast_has_a_row_even_for_duplicate_and_unknown_market():
    raw=prediction()
    bad={**prediction(),"market":"tackles"}
    d=G.score_predictions([raw,raw,bad],[game()],[player()],now=NOW)
    assert d["original_count"]==d["accounted_for"]==3
    assert d["verified_outcome_count"]==1
    assert d["status_counts"]["DUPLICATE_FIRST_SEEN_KEY_QUARANTINED"]==1
    assert d["status_counts"]["UNSUPPORTED_OR_UNMODELED_MARKET"]==1

def test_current_week_espn_final_can_score_before_nflverse_publishes():
    ep={"schema":"NFL_RECENT_FINAL_ESPN_PLAYER_TRUTH_V1","games":{
        GID:{**game(),"source":"ESPN_FINAL_GAME_BOXSCORE","away":"WAS","home":"SF",
            "players":[{"team":"SF","name_key":"sample runner","player":"Sample Runner",
                        "espn_athlete_id":"3494819",
                        "offense_appeared":True,"defense_appeared":False,
                        "stats":{"rush_yds":64,"rec_yds":0,"rush_td":0,"rec_td":1}}]}}}
    row=G.score_predictions([prediction()],[],[],espn=ep,now=NOW)["all_predictions"][0]
    assert row["status"]=="GRADED"
    assert row["actual_source"]=="ESPN_FINAL_BOX"
    assert row["actual"]==64

def test_conflicting_independent_actuals_quarantined():
    ep={"schema":"NFL_RECENT_FINAL_ESPN_PLAYER_TRUTH_V1","games":{
        GID:{**game(),"source":"ESPN_FINAL_GAME_BOXSCORE","away":"WAS","home":"SF",
            "players":[{"team":"SF","name_key":"sample runner","player":"Sample Runner",
                        "espn_athlete_id":"3494819",
                        "offense_appeared":True,"defense_appeared":False,
                        "stats":{"rush_yds":19,"rec_yds":0,"rush_td":0,"rec_td":0}}]}}}
    row=G.score_predictions([prediction()],[game()],[player()],espn=ep,now=NOW)["all_predictions"][0]
    assert row["status"]=="CONFLICTING_FINAL_STAT_SOURCES"
    assert row["actual"] is None

def test_source_recent_only_completed_and_identity_checked():
    real={GID:{"game_id":GID,"espn_id":"test123","week":6,
               "home":"SF","away":"WAS","kickoff_utc":K,
               "home_score":23,"away_score":17}}
    def fake(path):
        assert path=="summary?event=test123"
        return {"boxscore":{"players":[
            {"team":{"abbreviation":"SF"},"statistics":[{
                "name":"rushing","labels":["YDS","TD"],
                "athletes":[{"athlete":{"id":"42","displayName":"Sample Runner"},
                             "stats":["64","1"]}]}]},
            {"team":{"abbreviation":"WSH"},"statistics":[{
                "name":"passing","labels":["YDS"],
                "athletes":[{"athlete":{"id":"43","displayName":"Sample QB"},
                             "stats":["200"]}]}]}]}}
    out=E.build(real,fetch_summary=fake,now=NOW)
    assert out["games_with_verified_boxscore"]==1
    assert out["games"][GID]["players"][0]["stats"]["rush_yds"]==64

def test_2026_week7_not_hardcoded_out_of_grader():
    g=game();g["week"]=7;g["game_id"]="2026_07_WAS_SF"
    p=player();p["week"]=7;p["game_id"]=g["game_id"]
    r=prediction();r["week"]=7
    d=G.score_predictions([r],[g],[p],now=NOW)
    assert d["accounted_for"]==1
