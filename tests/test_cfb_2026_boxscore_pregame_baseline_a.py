"""CFB full-slate naive box-score benchmark: future-only, no missing-stat fabrication."""
import copy
import sqlite3
from datetime import datetime,timezone,timedelta

import pytest

import cfb_2026_boxscore_pregame_baseline_a as B

AT=datetime(2026,10,10,13,0,tzinfo=timezone.utc)

def fixture():
    con=sqlite3.connect(":memory:")
    con.row_factory=sqlite3.Row
    con.executescript("""
      CREATE TABLE games (game_id TEXT,season INT,week INT,kickoff_utc TEXT,
          home_team TEXT,away_team TEXT,home_points INT,away_points INT);
      CREATE TABLE schedule_snapshot (game_id TEXT,espn_status TEXT);
      CREATE TABLE player_games (player_id TEXT,game_id TEXT,team TEXT,
          position TEXT,season INT,week INT,rushing_yards INT,passing_touchdowns INT,
          rushing_touchdowns INT,receiving_touchdowns INT);
    """)
    for week in (1,2,3):
        gid=f"G{week}"
        con.execute("INSERT INTO games VALUES(?,?,?,?,?,?,?,?)",
                    (gid,2026,week,f"2026-09-0{week}T16:00:00Z","A","B",24,10))
        con.execute("INSERT INTO schedule_snapshot VALUES(?,?)",(gid,"STATUS_FINAL"))
        con.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?)",
                    ("P1",gid,"A","RB",2026,week,50+week*10,None,0,week%2))
        con.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?)",
                    ("Q1",gid,"A","QB",2026,week,None,week,None,None))
        con.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?)",
                    ("W1",gid,"A","WR",2026,week,None,None,0,week%2))
    con.execute("INSERT INTO games VALUES(?,?,?,?,?,?,?,?)",
                ("NEXT",2026,6,"2026-10-10T16:00:00Z","A","B",None,None))
    con.execute("INSERT INTO schedule_snapshot VALUES(?,?)",("NEXT","STATUS_SCHEDULED"))
    def row(p,market):
        return {"market":market,"game_id":"NEXT","player_id":p,"player":p,
                "team":"A","opponent":"B","season":2026,"week":6,
                "kickoff_utc":"2026-10-10T16:00:00Z",
                "roster_verification":"ON_CURRENT_ROSTER_SNAPSHOT"}
    board={"season":2026,"week":6,
           "generated_at_utc":"2026-10-10T12:00:00Z",
           "picks":[row("P1","rushing_yards"),row("Q1","passing_touchdowns"),
                    row("W1","anytime_touchdowns")]}
    return con,board

def report(db,board,at=AT):
    return B.snapshot(db,board,at=at,sha256="f"*64)

def test_all_slate_markets_only_actual_prior_game_statistics():
    db,board=fixture()
    r=report(db,board)
    assert r["baseline_rows"]==3
    assert r["by_market"]=={"rushing_yards":1,"passing_touchdowns":1,"anytime_touchdowns":1}
    marks={x["market"]:x for x in r["forecasts"]}
    assert marks["rushing_yards"]["point_mean_last3"]==70
    assert marks["rushing_yards"]["point_median_last3"]==70
    assert marks["passing_touchdowns"]["point_mean_last3"]==2
    assert marks["anytime_touchdowns"]["point_mean_last3"]==round(2/3,4)
    assert all(x["not_a_calibrated_interval"] and
               x["original_classifier_probability_not_used"] for x in r["forecasts"])
    assert r["historical_forecasts_backfilled"]==0
    assert r["production_model_modified"] is False

def test_pre_kickoff_forecast_required_not_fake_retroactive():
    db,board=fixture()
    late=datetime(2026,10,10,16,0,tzinfo=timezone.utc)
    r=report(db,board,late)
    assert r["baseline_rows"]==0
    assert r["excluded"]["NOT_PROVABLY_PREGAME"]==3

def test_bad_roster_and_identity_are_rejected():
    db,board=fixture()
    board["picks"][0]["team"]="B"
    board["picks"][1]["roster_verification"]="UNVERIFIED"
    r=report(db,board)
    assert r["baseline_rows"]==1
    assert r["excluded"]["INSUFFICIENT_PRIOR_3_SAME_TEAM_FINAL_GAME_STATS"]==1
    assert r["excluded"]["ROSTER_NOT_CONFIRMED_IN_BOARD"]==1

def test_incorrect_team_outside_game_explicitly_blocked():
    db,board=fixture()
    board["picks"][0]["team"]="C"
    r=report(db,board)
    assert r["baseline_rows"]==2
    assert r["excluded"]["PLAYER_TEAM_NOT_PARTICIPATING"]==1


def test_future_game_results_must_be_final_before_snapshot():
    db,board=fixture()
    db.execute("UPDATE schedule_snapshot SET espn_status='STATUS_SCHEDULED' WHERE game_id='G3'")
    r=report(db,board)
    assert r["baseline_rows"]==0
    assert r["excluded"]["INSUFFICIENT_PRIOR_3_SAME_TEAM_FINAL_GAME_STATS"]==3

def test_no_missing_stat_field_can_turn_into_zero_touchdown():
    db,board=fixture()
    db.execute("UPDATE player_games SET passing_touchdowns=NULL WHERE player_id='Q1' AND game_id='G3'")
    r=report(db,board)
    assert r["by_market"].get("passing_touchdowns")==None
    assert r["baseline_rows"]==2

def test_a_board_kickoff_mismatch_is_not_a_valid_forecast():
    db,board=fixture()
    board["picks"][0]["kickoff_utc"]="2026-10-10T18:00:00Z"
    r=report(db,board)
    assert r["baseline_rows"]==2
    assert r["excluded"]["KICKOFF_DIFFERENCE_NEEDS_MANUAL_REVIEW"]==1

def test_duplicate_player_market_refused_and_no_stats_aggregation():
    db,board=fixture()
    board["picks"].append(copy.deepcopy(board["picks"][0]))
    with pytest.raises(B.BaselineError,match="DUPLICATE_CURRENT_BOARD_PLAYER_MARKET"):
        report(db,board)

def test_incorrect_board_week_and_future_created_timestamp_fail_closed():
    db,board=fixture()
    board["week"]=5
    with pytest.raises(B.BaselineError,match="NOT_ORIGINAL_2026_WEEK6_BOARD"):
        report(db,board)
    board["week"]=6
    board["generated_at_utc"]="2026-10-10T16:05:00Z"
    with pytest.raises(B.BaselineError,match="BOARD_CREATED_IN_FUTURE"):
        report(db,board)
