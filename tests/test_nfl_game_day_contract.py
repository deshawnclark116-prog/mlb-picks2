"""Strict tests for the weekly NFL publishing gate; no model retraining."""
from datetime import datetime,timezone
import copy,json,sqlite3
from pathlib import Path
import pytest
from nfl_game_day_contract import validate
from nfl_pregame_delivery import schedule_manifest

NOW="2026-10-10T23:35:00Z"
GOOD={
 "generated_at_utc":"2026-10-10T23:31:05Z", "season":2026,"week":5,
 "scheduled_games":[
  {"home_team":"JAX","away_team":"PHI","kickoff_utc":"2026-10-11T13:30:00Z"},
  {"home_team":"ARI","away_team":"DET","kickoff_utc":"2026-10-11T20:25:00Z"}],
 "picks":[{"market":"rushing_yards","player":"Runner","player_id":"p1","team":"PHI","opponent":"JAX",
           "kickoff_utc":"2026-10-11T13:30:00Z","projected_median":55,"projected_low":5,"projected_high":110}]}
def check(doc):
    return validate(doc,now=NOW)
def test_valid_slate_and_no_player_game_are_distinct():
    status=check(copy.deepcopy(GOOD))
    assert status["scheduled"]==2 and status["future"]==2
    assert status["predictions"]==1 and status["numeric"]==1
    assert status["games_without_eligible_picks"]==[["ARI","DET"]]
def test_empty_weekly_schedule_cannot_be_published_as_complete():
    x=copy.deepcopy(GOOD);x["scheduled_games"]=[]
    with pytest.raises(ValueError,match="SCHEDULE_MANIFEST_MISSING"):check(x)
def test_cannot_change_or_duplicate_scheduled_kickoff():
    x=copy.deepcopy(GOOD);x["picks"][0]["kickoff_utc"]="2026-10-11T20:25:00Z"
    with pytest.raises(ValueError,match="PICK_KICKOFF_MISMATCH"):check(x)
    x=copy.deepcopy(GOOD);x["scheduled_games"].append(copy.deepcopy(x["scheduled_games"][0]))
    with pytest.raises(ValueError,match="duplicate"):check(x)
def test_no_postkickoff_pick_and_no_backfill_to_historical_games():
    x=copy.deepcopy(GOOD);x["generated_at_utc"]="2026-10-11T15:00:00Z"
    with pytest.raises(ValueError,match="LATE_PREDICTION_OR_BACKFILL"):
        validate(x,now="2026-10-11T15:01:00Z")
def test_no_fabricated_numerical_projections_or_impossible_td_probability():
    x=copy.deepcopy(GOOD);x["picks"][0]["projected_median"]=140
    with pytest.raises(ValueError,match="INVALID_NUMERICAL_PROJECTION"):check(x)
    x=copy.deepcopy(GOOD);x["picks"][0].update(market="anytime_touchdowns",model_prob=1.5)
    with pytest.raises(ValueError,match="INVALID_TD_PROBABILITY"):check(x)
def test_duplicate_players_cannot_double_count_per_market():
    x=copy.deepcopy(GOOD);x["picks"].append(copy.deepcopy(x["picks"][0]))
    with pytest.raises(ValueError,match="DUPLICATED_PLAYER_MARKET"):check(x)
def test_validation_checks_the_authoritative_foundation_not_just_the_website(tmp_path):
    db=tmp_path/"nfl.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE games(home_team TEXT,away_team TEXT,kickoff_utc TEXT,season INTEGER,week INTEGER)")
        con.executemany("INSERT INTO games VALUES(?,?,?,?,?)",
            [("JAX","PHI","2026-10-11T13:30:00Z",2026,5),("ARI","DET","2026-10-11T20:25:00Z",2026,5)])
    assert validate(GOOD,db,now=NOW)["status"]=="VALID_PREGAME_PUBLICATION"
    x=copy.deepcopy(GOOD);x["scheduled_games"].pop()
    with pytest.raises(ValueError,match="PUBLISHED_SCHEDULE_DIFFERS"):validate(x,db,now=NOW)
def test_real_2026_week5_frozen_weekly_predictions_against_live_ingested_schedule():
    # CI refreshes the same real nflverse 2025/26 schedule into its local DB.
    root=Path(__file__).resolve().parent.parent
    old=json.loads((root/"docs/nfl_predictions_2026_w05.json").read_text())
    with sqlite3.connect(root/"nfl_models/nfl_model.sqlite") as con:
        rows=con.execute("SELECT home_team,away_team,kickoff_utc FROM games WHERE season=2026 AND week=5").fetchall()
    assert len(rows)==15
    doc={**old,"scheduled_games":schedule_manifest(rows)}
    report=validate(doc,root/"nfl_models/nfl_model.sqlite",now=old["generated_at_utc"])
    print("REAL_W5_PUBLICATION",json.dumps(report,sort_keys=True))
    assert report["scheduled"]==15 and report["future"]==14
    assert report["numeric"]>=100 and report["predictions"]>=100
    assert report["games_without_eligible_picks"]==[]
