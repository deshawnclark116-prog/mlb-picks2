import sqlite3
from datetime import datetime,timezone
import cfb_numeric_shadow_forward as m

NOW=datetime(2026,10,11,16,tzinfo=timezone.utc)
KICKOFF='2026-10-13T23:00:00Z'
MODEL=m.MODEL

def db():
 c=sqlite3.connect(':memory:')
 c.executescript("""
 CREATE TABLE games(game_id TEXT,season INT,week INT,home_team TEXT,
 away_team TEXT,kickoff_utc TEXT,home_points INT,away_points INT);
 CREATE TABLE schedule_snapshot(game_id TEXT,espn_state TEXT);
 CREATE TABLE player_games(game_id TEXT,player_id TEXT,team TEXT,
 rushing_yards REAL,receiving_yards REAL,passing_touchdowns REAL);
 """)
 c.execute("INSERT INTO games VALUES(?,?,?,?,?,?,?,?)",
 ("game123",2026,7,"Hawks","Bears",KICKOFF,None,None))
 c.execute("INSERT INTO schedule_snapshot VALUES(?,?)",("game123","pre"))
 c.commit()
 return c

BOARD={"season":2026,"week":7,"generated_at_utc":"2026-10-11T13:00:00Z"}
def numeric():
 return {"research_model":MODEL,"status":"FROZEN_SHADOW_NO_PROMOTION",
  "upcoming_asof":BOARD["generated_at_utc"],
  "upcoming_numeric_projections_RESEARCH_ONLY":[{
   "game_id":"game123","player_id":"A123","player":"Real Runner",
   "team":"Hawks","opponent":"Bears","market":"rushing_yards",
   "expected_stat":78.21,"kickoff_utc":KICKOFF
  }]}

def test_first_pregame_forecast_is_real_and_immutable():
 c=db()
 rows,status=m.publish(c,numeric(),BOARD,[],NOW)
 assert status['new_first_pregame']==1
 assert len(rows)==1 and rows[0]['expected_stat']==78.21
 assert rows[0]['published_at_utc']=='2026-10-11T16:00:00Z'
 assert rows[0]['not_a_betting_pick'] is True
 # Publishing a refreshed stronger prediction cannot rewrite original.
 doc=numeric()
 doc['upcoming_numeric_projections_RESEARCH_ONLY'][0]['expected_stat']=102.0
 again,audit=m.publish(c,doc,BOARD,rows,NOW)
 assert again==[]
 assert audit['omitted_reasons']['prior_immutable_prediction_exists']==1
 assert rows[0]['expected_stat']==78.21

def test_real_numeric_grading_uses_final_actual_and_never_zero_fills_missing():
 c=db()
 entries,_=m.publish(c,numeric(),BOARD,[],NOW)
 assert m.grade(c,entries)['ungraded']['game_not_final']==1
 c.execute("UPDATE games SET home_points=42,away_points=7")
 assert m.grade(c,entries)['ungraded']['player_no_stat_line_after_final_NOT_ZERO']==1
 c.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?)",("game123","A123","Hawks",61,0,0))
 r=m.grade(c,entries)
 assert r['graded_count']==1
 assert r['by_market']['rushing_yards']['mae']==17.21
 assert r['by_market']['rushing_yards']['signed_bias']==17.21
 assert not r['betting_edge_calibrated']
 assert r['promotion']=='NOT_AUTHORIZED'

def test_duplicate_or_postkickoff_predictions_never_become_fresh_grading_truth():
 c=db()
 rows,_=m.publish(c,numeric(),BOARD,[],NOW)
 c.execute("UPDATE games SET home_points=40,away_points=35")
 c.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?)",("game123","A123","Hawks",72,0,0))
 later={**rows[0],"published_at_utc":"2026-10-13T23:05:00Z","expected_stat":72}
 r=m.grade(c,[later,rows[0]])
 assert r["graded_count"]==1
 assert r['duplicate_or_later_ledger_rows_never_used']==1
 assert r['evaluations'][0]['predicted_mean']==78.21

def test_wrong_game_or_teams_stale_model_and_post_kickoff_all_fail_closed():
 c=db()
 d=numeric();d['upcoming_asof']="2026-10-10T01:00:00Z"
 try:m.publish(c,d,BOARD,[],NOW);assert False
 except ValueError:pass
 d=numeric();d['upcoming_numeric_projections_RESEARCH_ONLY'][0]['team']="Not Playing"
 assert m.publish(c,d,BOARD,[],NOW)[0]==[]
 d=numeric()
 assert m.publish(c,d,BOARD,[],datetime(2026,10,14,tzinfo=timezone.utc))[0]==[]
 d=numeric();d["status"]="PROMOTED"
 try:m.publish(c,d,BOARD,[],NOW);assert False
 except ValueError:pass

def test_espn_still_in_play_rejected_even_before_scheduled_kickoff():
 c=db()
 c.execute("UPDATE schedule_snapshot SET espn_state='in'")
 assert m.publish(c,numeric(),BOARD,[],NOW)[0]==[]
