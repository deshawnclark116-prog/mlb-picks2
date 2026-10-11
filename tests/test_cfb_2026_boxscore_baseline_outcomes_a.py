"""Never re-create historical baselines to grade; test frozen pregame artifact integrity."""
from datetime import datetime,timezone
import sqlite3,copy
import pytest
import cfb_2026_boxscore_pregame_baseline_a as B
import cfb_2026_boxscore_baseline_outcomes_a as G


def fixture():
    con=sqlite3.connect(":memory:")
    con.row_factory=sqlite3.Row
    con.executescript("""
      CREATE TABLE games(game_id TEXT,season INT,week INT,kickoff_utc TEXT,
                         home_team TEXT,away_team TEXT,home_points INT,away_points INT);
      CREATE TABLE schedule_snapshot(game_id TEXT,season INT,espn_status TEXT);
      CREATE TABLE player_games(player_id TEXT,game_id TEXT,team TEXT,position TEXT,
          season INT,week INT,rushing_yards INT,passing_touchdowns INT,
          rushing_touchdowns INT,receiving_touchdowns INT);
    """)
    for week,yds in [(1,60),(2,70),(3,80)]:
        gid=f"G{week}"
        con.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?)",
                    (gid,2026,week,f"2026-09-0{week}T16:00:00Z","A","B",14,7))
        con.execute("INSERT INTO schedule_snapshot VALUES(?,?,?)",(gid,2026,"STATUS_FINAL"))
        con.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?)",
                    ("RB1",gid,"A","RB",2026,week,yds,None,1,0))
    con.execute("INSERT INTO games VALUES(?,?,?,?,?,?,?,?)",
                ("G6",2026,6,"2026-10-10T16:00:00Z","A","B",None,None))
    con.execute("INSERT INTO schedule_snapshot VALUES(?,?,?)",("G6",2026,"STATUS_SCHEDULED"))
    board={"season":2026,"week":6,"generated_at_utc":"2026-10-10T12:15:00Z",
           "picks":[{"market":"rushing_yards","game_id":"G6","player_id":"RB1",
                    "player":"RB1","team":"A","opponent":"B","season":2026,"week":6,
                    "kickoff_utc":"2026-10-10T16:00:00Z",
                    "roster_verification":"ON_CURRENT_ROSTER_SNAPSHOT"}]}
    artifact=B.snapshot(con,board,at=datetime(2026,10,10,13,0,tzinfo=timezone.utc),sha256="a"*64)
    source={"id":38053200000,"status":"completed","conclusion":"success",
            "name":"CFB 2026 event data qualification (research, no serving)",
            "head_sha":"b"*40,
            "created_at":"2026-10-10T12:30:00Z",
            "run_started_at":"2026-10-10T12:31:00Z",
            "updated_at":"2026-10-10T13:08:00Z"}
    assert artifact["baseline_rows"]==1
    return con,artifact,source


def final(con,yards=84):
    con.execute("UPDATE games SET home_points=24,away_points=10 WHERE game_id='G6'")
    con.execute("UPDATE schedule_snapshot SET espn_status='STATUS_FINAL' WHERE game_id='G6'")
    con.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?)",
                ("RB1","G6","A","RB",2026,6,yards,None,1,0))
    con.commit()


def test_incomplete_saturday_game_does_not_grade():
    con,paper,source=fixture()
    r=G.evaluate(paper,con,source)
    assert r["graded_n"]==0
    assert r["ungraded_reasons"]=={"GAME_NOT_FINAL":1}


def test_after_final_original_mean_accuracy_is_scored_without_refitting():
    con,paper,source=fixture()
    final(con)
    r=G.evaluate(paper,con,source,original_sha="sha_test")
    assert r["graded_n"]==1
    assert r["overall"]["mean_baseline_MAE"]==14
    assert r["overall"]["median_baseline_MAE"]==14
    assert r["original_source_artifact_sha256"]=="sha_test"
    assert r["no_model_promotions_or_past_predictions_rewritten"]


def test_missing_player_stat_not_scored_as_zero():
    con,paper,source=fixture()
    final(con)
    con.execute("DELETE FROM player_games WHERE game_id='G6'")
    r=G.evaluate(paper,con,source)
    assert r["graded_n"]==0
    assert r["ungraded_reasons"]=={"NO_OFFICIAL_PLAYER_LINE":1}


def test_modified_history_after_original_forecast_is_rejected():
    con,paper,source=fixture()
    final(con)
    con.execute("UPDATE player_games SET rushing_yards=90 WHERE game_id='G3'")
    with pytest.raises(G.BaselineGradeError,match="ORIGINAL_SOURCE_REVISION_CHANGED_BASELINE"):
        G.evaluate(paper,con,source)


@pytest.mark.parametrize("mutate,msg",[
    (lambda p,s: p["forecasts"][0].update(point_mean_last3=95),
     "MEAN_CHANGED_AFTER_PREGAME"),
    (lambda p,s: p["forecasts"][0].update(team="B"),
     "ORIGINAL_PRIOR_PLAYER_TEAM_AMBIGUOUS"),
    (lambda p,s: p["forecasts"][0].update(research_generated_at_utc="2026-10-10T17:00:00Z"),
     "NO_GENUINE_ORIGINAL_PREGAME_PROOF"),
    (lambda p,s: s.update(conclusion="failure"),
     "UNVERIFIED_GITHUB_ORIGINAL_RUN"),
    (lambda p,s: s.update(head_sha="not-a-sha"),
     "MISSING_ORIGIN_COMMIT"),
    (lambda p,s: p["forecasts"][0].update(point_median_last3=100),
     "MEDIAN_CHANGED_AFTER_PREGAME"),
])
def test_mutated_forecast_or_origin_must_not_grade(mutate,msg):
    con,paper,source=fixture()
    final(con)
    mutate(paper,source)
    with pytest.raises(G.BaselineGradeError,match=msg):
        G.evaluate(paper,con,source)


def test_duplicate_forecasts_cannot_inflate_evaluation():
    con,paper,source=fixture()
    final(con)
    paper["forecasts"].append(copy.deepcopy(paper["forecasts"][0]))
    paper["baseline_rows"]=2
    with pytest.raises(G.BaselineGradeError,match="DUPLICATE_ORIGINAL_PLAYER_STAT_FORECAST"):
        G.evaluate(paper,con,source)
