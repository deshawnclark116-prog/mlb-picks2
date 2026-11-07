"""Test raw team pace proxy with forward-only baselines, no winner feature leak."""
import sqlite3
from datetime import datetime,timezone,timedelta

from nfl_v3_joint_pace_falsification import empirical


def create():
    con=sqlite3.connect(":memory:")
    con.executescript("""
        CREATE TABLE games(
          game_id TEXT PRIMARY KEY,season INT,week INT,season_type TEXT,
          home_team TEXT,away_team TEXT,kickoff_utc TEXT,
          home_score INT,away_score INT);
        CREATE TABLE player_games(
          player_id TEXT,game_id TEXT,season INT,week INT,team TEXT,
          season_type TEXT,carries REAL,attempts REAL);
    """)
    return con


def add(con,w,home_runs=25,away_runs=25,home_pass=35,away_pass=35,
        score_home=25,score_away=17,complete=True):
    gid=f"2025_{w:02d}_AAA_BBB"
    kick=(datetime(2025,9,1,tzinfo=timezone.utc)+timedelta(weeks=w)).isoformat()
    con.execute("INSERT INTO games VALUES(?,?,?,?,?,?,?,?,?)",
                (gid,2025,w,"REG","AAA","BBB",kick,
                 score_home if complete else None,
                 score_away if complete else None))
    for team,runs,passes in (("AAA",home_runs,home_pass),
                            ("BBB",away_runs,away_pass)):
        for i in range(8):
            con.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?)",
                        (f"{team}_{i}",gid,2025,w,team,"REG",
                         runs if i==0 else 0,
                         passes if i==1 else 0))
    con.commit()
    return kick


def test_actual_game_never_enters_its_own_four_prior_game_baseline():
    c=create()
    for w in range(1,5):add(c,w)
    add(c,5,home_runs=30,away_runs=20,home_pass=35,away_pass=40)
    at="2025-11-01T00:00:00Z"
    a=empirical(c,at)
    one=a["by_season"]["2025"]
    assert one["all_completed_source_games"]==5
    assert one["paired_games_with_four_prior_team_games"]==1
    assert one["pace_residual_correlation"] is None
    assert one["mean_winner_minus_loser_rush_share"] is not None
    assert a["model_changed"] is False
    assert a["production_promotion_allowed"] is False
    assert a["fixed_prior_team_games"]==4


def test_future_final_is_excluded_from_asof_and_cannot_change_old_label():
    c=create()
    for w in range(1,5):add(c,w)
    add(c,5,home_runs=30,away_runs=20,home_pass=35,away_pass=40)
    before=empirical(c,"2025-10-10T00:00:00Z")
    future=add(c,6,home_runs=50,away_runs=5,home_pass=20,away_pass=65)
    after=empirical(c,"2025-10-10T00:00:00Z")
    assert before==after
    assert after["by_season"]["2025"]["paired_games_with_four_prior_team_games"]==1


def test_score_is_a_validation_label_and_not_pretip_volume_or_pacing_feature():
    c=create()
    for w in range(1,5):add(c,w)
    add(c,5,home_runs=30,away_runs=20,home_pass=35,away_pass=40)
    before=empirical(c,"2025-11-01T00:00:00Z")
    c.execute("UPDATE games SET home_score=10,away_score=27 WHERE week=5")
    c.commit()
    after=empirical(c,"2025-11-01T00:00:00Z")
    a=before["by_season"]["2025"]
    b=after["by_season"]["2025"]
    assert a["pace_residual_correlation"]==b["pace_residual_correlation"]
    assert a["mean_winner_minus_loser_rush_share"]== -b["mean_winner_minus_loser_rush_share"]


def test_unknown_or_unfinished_game_is_not_treated_as_completed():
    c=create()
    for w in range(1,5):add(c,w)
    add(c,5,home_runs=50,away_runs=5,complete=False)
    a=empirical(c,"2025-11-01T00:00:00Z")
    assert a["by_season"]=={}


def test_comparisons_stay_uncalibrated_without_sufficient_independent_games():
    c=create()
    for w in range(1,9):add(c,w,home_runs=22+w,away_runs=23+w)
    a=empirical(c,"2025-12-01T00:00:00Z")
    s=a["by_season"]["2025"]
    assert s["paired_games_with_four_prior_team_games"]==4
    assert s["game_cluster_bootstrap_95pct_interval"] is None
    assert s["hypothesis_positive_common_tempo_supported"] is False
    assert a["inspection_cohort_2026_not_promotion_evidence"] is True
