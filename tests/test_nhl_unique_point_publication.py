"""Never list one NHL player anytime-point outcome twice as two bets.
Lock real NHL Eastern game calendar, unfinished-slate priority, archive integrity.
"""
from datetime import datetime,timezone
import copy
import sqlite3
from nhl_serving_builder_a import (
    select_one_public_points_outcome, eastern_game_calendar_date,
    infer_target_date
)

def point(pid,market,prob,team="FLA",opponent="CHI",date="2026-10-25"):
    return {"market":market,"player_id":pid,"player":"Named skater",
        "team":team,"opponent":opponent,"season":2026,"game_date":date,
        "kickoff_utc":date+"T23:00:00Z","pick":"OVER 0.5",
        "model_prob":prob, "source_model":"prior" if market.endswith("_early_season") else "current"}

def test_duplicate_point_models_are_one_event_and_current_source_wins_not_most_optimistic():
    rows=[point(8479314,"points_early_season",.91),
          point(8479314,"points",.52),
          point(8480011,"points_early_season",.74),
          {"market":"moneyline","team":"Florida Panthers",
           "opponent":"Chicago Blackhawks","pick":"Florida Panthers ML"}]
    original=copy.deepcopy(rows)
    served,audit=select_one_public_points_outcome(rows)
    assert len(served)==3
    main=next(x for x in served if x.get("player_id")==8479314)
    assert main["market"]=="points"
    assert main["source_market"]=="points"
    assert main["model_prob"]==.52 # NO max-probability shopping
    fallback=next(x for x in served if x.get("player_id")==8480011)
    assert fallback["market"]=="points"
    assert fallback["source_market"]=="points_early_season"
    assert audit["raw_anytime_point_rows"]==3
    assert audit["public_anytime_point_rows"]==2
    assert audit["superseded_model_generation_rows"]==1
    assert rows==original # pregame originals remain untouched

def test_73_raw_point_estimates_reduce_to_50_distinct_outcomes_no_book_needed():
    rows=[point(90000+i,"points",.55) for i in range(50)]
    rows.extend(point(90000+i,"points_early_season",.85) for i in range(23))
    served,audit=select_one_public_points_outcome(rows)
    assert len(served)==50
    assert audit["raw_anytime_point_rows"]==73
    assert audit["public_anytime_point_rows"]==50
    assert audit["superseded_model_generation_rows"]==23
    assert all(p["model_prob"]==.55 for p in served)
    assert all(p["market"]=="points" for p in served)

def test_different_game_or_player_is_never_silently_combined():
    a=point(1,"points",.55)
    b=point(1,"points_early_season",.54,date="2026-10-26")
    c=point(2,"points",.51)
    d=point(None,"points",.99)
    rows,audit=select_one_public_points_outcome([a,b,c,d])
    assert len(rows)==3
    assert audit["quarantined_unidentified_rows"]==1
    assert sorted(x["game_date"] for x in rows)==[
        "2026-10-25","2026-10-25","2026-10-26"]

def test_live_nhl_game_day_stays_oct24_after_utc_has_become_oct25():
    clock=datetime.fromisoformat("2026-10-25T01:07:12+00:00")
    assert eastern_game_calendar_date(clock)=="2026-10-24"
    morning=datetime.fromisoformat("2026-10-25T14:00:00+00:00")
    assert eastern_game_calendar_date(morning)=="2026-10-25"

def test_only_unfinished_scheduled_games_can_become_current_board():
    db=sqlite3.connect(":memory:")
    db.execute("CREATE TABLE games (game_date TEXT, home_score INTEGER, away_score INTEGER)")
    db.executemany("INSERT INTO games VALUES (?,?,?)",[
      ("2026-10-24",4,2),("2026-10-24",None,None),
      ("2026-10-25",None,None),("2026-10-26",None,None)])
    assert infer_target_date(db,"2026-10-24")=="2026-10-24"
    db.execute("UPDATE games SET home_score=3,away_score=2 WHERE game_date='2026-10-24'")
    assert infer_target_date(db,"2026-10-24")=="2026-10-25"
    assert infer_target_date(db,"2026-10-27") is None
