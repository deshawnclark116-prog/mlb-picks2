"""Leak-safe, no-odds numerical accuracy scoring against same-game yard actuals."""
import sqlite3
import pytest
from pathlib import Path
from nfl_numeric_accuracy_audit import evaluate,normalize_name

def make_db(tmp_path):
    db=tmp_path/"sports.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE games(season INTEGER,week INTEGER,home_team TEXT,away_team TEXT,kickoff_utc TEXT)")
        con.execute("CREATE TABLE player_games(season INTEGER,week INTEGER,player_id TEXT,player_name TEXT,team TEXT,opponent TEXT,rushing_yards REAL,receiving_yards REAL)")
        for wk,kick,yds in [(1,"2026-09-13T17:00:00Z",40),(2,"2026-09-20T17:00:00Z",45),(3,"2026-09-27T17:00:00Z",50),(4,"2026-10-04T17:00:00Z",60)]:
            con.execute("INSERT INTO games VALUES(?,?,?,?,?)",(2026,wk,"JAX","PHI",kick))
            con.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?)",(2026,wk,"id-one","Runner One","PHI","JAX",yds,yds))
    return db

def log(**override):
    x={"season":2026,"week":4,"market":"rushing_yards","player":"Runner One",
       "player_id":"id-one","team":"PHI","opponent":"JAX","logged_at":"2026-10-03T20:00:00Z",
       "projected_median":70,"model_source":"v2_context"}
    x.update(override)
    return x

def test_absolute_prediction_error_and_same_game_historical_average(tmp_path):
    d=evaluate([log()],make_db(tmp_path))
    assert d["eligible_sample"]==1
    assert d["overall"]["mae"]==10
    assert d["overall"]["baseline_last3_mae"]==15
    assert d["overall"]["improvement_vs_last3_yds"]==5
    assert d["overall"]["status"]=="BEATS_LAST3_OBSERVED"
    assert d["forecasts"][0]["actual"]==60
    assert d["forecasts"][0]["baseline_last3"]==45
    assert d["automatic_promotion"] is False

def test_no_after_kickoff_prediction_can_be_graded_as_pregame(tmp_path):
    d=evaluate([log(logged_at="2026-10-04T17:00:00Z")],make_db(tmp_path))
    assert d["eligible_sample"]==0
    assert d["excluded"]["after_kickoff"]==1

def test_no_future_results_count_as_realized_accuracy(tmp_path):
    d=evaluate([log(week=5,logged_at="2026-10-06T14:00:00Z")],make_db(tmp_path))
    assert d["eligible_sample"]==0
    assert d["excluded"]["missing_schedule_kickoff"]==1

def test_no_fake_betting_hit_rates_or_invalid_projections(tmp_path):
    db=make_db(tmp_path)
    d=evaluate([log(projected_median=float("nan")),
                log(market="anytime_touchdowns",model_prob=.9)],db)
    assert d["eligible_sample"]==0
    assert "hit_rate" not in d["overall"]

def test_duplicate_first_seen_record_cannot_double_weight_score(tmp_path):
    d=evaluate([log(),log(projected_median=58)],make_db(tmp_path))
    assert d["eligible_sample"]==1
    assert d["forecasts"][0]["projected_median"]==70
    assert d["excluded"]["duplicate_first_seen_key"]==1

def test_previous_game_not_finished_by_forecast_does_not_leak_into_baseline(tmp_path):
    db=make_db(tmp_path)
    d=evaluate([log(logged_at="2026-09-27T18:00:00Z")],db)
    assert d["eligible_sample"]==1
    assert d["forecasts"][0]["baseline_last3"]==42.5

def test_ambiguous_name_fallback_is_not_mistaken_for_known_identity(tmp_path):
    db=make_db(tmp_path)
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?)",(2026,4,"id-two","Runner One","PHI","JAX",99,99))
    d=evaluate([log(player_id="external-identifier")],db)
    assert d["eligible_sample"]==0
    assert d["excluded"]["missing_or_ambiguous_actual"]==1

def test_name_normalization_preserves_simple_player_matches():
    assert normalize_name("D'Andre Smith Jr.")=="dandre smith"
