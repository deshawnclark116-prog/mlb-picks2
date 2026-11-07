"""As-of roster/usage receipt tests: a 0.4-yard backup never becomes top six via HIGH."""
import sqlite3
from datetime import datetime,timezone
import nfl_role_snapshot_batch as mod

def db():
 c=sqlite3.connect(":memory:")
 c.executescript("""
 CREATE TABLE games(game_id TEXT,season INT,week INT,home_team TEXT,away_team TEXT);
 CREATE TABLE player_games(player_id TEXT,player_name TEXT,team TEXT,
 season INT,week INT,game_id TEXT,season_type TEXT,
 carries REAL,targets REAL,rushing_yards REAL,receiving_yards REAL);
 """)
 for w in range(1,5):
  gid=f"2026_{w:02}_CHI_GB"
  c.execute("INSERT INTO games VALUES(?,?,?,?,?)",(gid,2026,w,"GB","CHI"))
  for name,team,att,mkt in [
   ("Established Runner","CHI",14,"rush"),
   ("Third String Back","CHI",0,"rush"),
   ("Role Receiver","GB",8,"rec")]:
   vol=(att,0) if mkt=="rush" else (0,att)
   c.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?,?)",
      (name,name,team,2026,w,gid,"REG",vol[0],vol[1],vol[0]*4,vol[1]*8))
 c.commit()
 return c

NOW=datetime(2026,10,11,12,0,tzinfo=timezone.utc)
FC={"season":2026,"week":5,"generated_at_utc":"2026-10-11T08:30:00Z",
"scheduled_games":[{"home_team":"GB","away_team":"CHI","kickoff_utc":"2026-10-11T17:00:00Z"}],
"picks":[
 {"player":"Established Runner","team":"CHI","opponent":"GB","market":"rushing_yards",
  "projected_median":59.8,"model_source":"v2_context","player_id":"one"},
 {"player":"Third String Back","team":"CHI","opponent":"GB","market":"rushing_yards",
  "projected_median":0.4,"model_source":"v2_context","player_id":"two","confidence":"HIGH"},
 {"player":"Role Receiver","team":"GB","opponent":"CHI","market":"receiving_yards",
  "projected_median":45.5,"model_source":"v2_context","player_id":"three"}
]}

def roster(t,now):
 player_names={"CHI":["Established Runner","Third String Back"],"GB":["Role Receiver"]}
 return {"status":"ROSTER_CAPTURED","players":{
  mod.role.normalize(p):[{"status":"ROSTER_LISTED_NOT_STARTER_VERIFIED"}]
  for p in player_names[t]}}

def test_0_point_4_high_stability_is_not_featured_while_valid_roles_are():
 result=mod.capture(FC,db(),now=NOW,roster_fetch=roster)
 assert len(result["player_market_evidence"])==3
 by={x["player"]:x for x in result["player_market_evidence"]}
 assert by["Established Runner"]["role_evidence_complete_for_showcase"]
 assert by["Role Receiver"]["role_evidence_complete_for_showcase"]
 assert not by["Third String Back"]["role_evidence_complete_for_showcase"]
 assert by["Third String Back"]["recent_opportunities_per_observed_game"]==0
 assert "LOW_OR_UNKNOWN_PRIOR_OPPORTUNITY" in by["Third String Back"]["warnings"]
 assert result["n_with_complete_historical_role_evidence"]==2
 assert result["is_betting_probability_calibrated"] is False
 assert result["starter_status_verified"] is False

def test_no_roster_cannot_certify_an_official_model_showcase():
 def outage(t,now):return {"status":"ROSTER_SOURCE_UNAVAILABLE","players":{}}
 result=mod.capture(FC,db(),now=NOW,roster_fetch=outage)
 assert result["n_with_complete_historical_role_evidence"]==0
 assert all("PLAYER_CURRENT_ROSTER_STATUS_UNCERTAIN" in x["warnings"]
            for x in result["player_market_evidence"])

def test_post_kickoff_never_relabels_previous_forecasts():
 later=datetime(2026,10,11,17,0,tzinfo=timezone.utc)
 result=mod.capture(FC,db(),now=later,roster_fetch=roster)
 assert result["player_market_evidence"]==[]
 assert result["n_with_complete_historical_role_evidence"]==0

def test_upcoming_game_stats_cannot_enter_asof_role_receipt():
 c=db()
 before=mod.capture(FC,c,now=NOW,roster_fetch=roster)
 gid="2026_05_CHI_GB"
 c.execute("INSERT INTO games VALUES(?,?,?,?,?)",(gid,2026,5,"GB","CHI"))
 c.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?,?)",
  ("two","Third String Back","CHI",2026,5,gid,"REG",42,0,200,0))
 after=mod.capture(FC,c,now=NOW,roster_fetch=roster)
 assert before["player_market_evidence"]==after["player_market_evidence"]

def test_distinct_roster_player_names_cannot_be_guessed():
 def missing(t,now):return {"status":"ROSTER_CAPTURED","players":{}}
 result=mod.capture(FC,db(),now=NOW,roster_fetch=missing)
 assert result["n_with_complete_historical_role_evidence"]==0


def test_variance_and_volume_weighted_role_is_derived_from_real_prior_team_games():
 c=db()
 # The same player's game-level rush share moves from a large piece of
 # small backfield to one third of a fast-paced committee. Arithmetic
 # mean of shares and pooled volume share are genuinely different.
 gid="2026_04_CHI_GB"
 c.execute("UPDATE player_games SET carries=20 WHERE game_id=? AND player_name=?",
           (gid,"Established Runner"))
 c.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?,?)",
           ("other","Other Runner","CHI",2026,4,gid,"REG",40,0,120,0))
 c.commit()
 result=mod.capture(FC,c,now=NOW,roster_fetch=roster)
 entry=next(x for x in result["player_market_evidence"] if x["player"]=="Established Runner")
 assert entry["recent_team_share"]==round((1+1+20/60)/3,4)
 assert entry["role_share_volume_weighted"]==round((14+14+20)/(14+14+60),4)
 assert entry["role_share_game_to_game_range"]==round(1-20/60,4)
 assert entry["role_share_sample_games"]==3
 assert entry["player_observed_opportunities_total"]==48
 assert entry["team_observed_opportunities_total"]==88
 assert len(entry["recent_team_game_observations"])==3
 assert all(x["stat_row_observed"] is True for x in entry["recent_team_game_observations"])
 assert result["raw_role_uncertainty_added"] is True
 assert result["is_betting_probability_calibrated"] is False


def test_missing_player_game_row_not_mislabeled_zero_or_confirmed_inactive():
 c=db()
 gid="2026_03_CHI_GB"
 c.execute("DELETE FROM player_games WHERE game_id=? AND player_name=?",
           (gid,"Established Runner"))
 c.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?,?)",
           ("other","Other Runner","CHI",2026,3,gid,"REG",10,0,20,0))
 c.commit()
 result=mod.capture(FC,c,now=NOW,roster_fetch=roster)
 entry=next(x for x in result["player_market_evidence"] if x["player"]=="Established Runner")
 assert entry["observed_stat_rows"]==2
 assert entry["role_share_sample_games"]==2
 unknown=[x for x in entry["recent_team_game_observations"] if x["week"]==3]
 assert len(unknown)==1
 assert unknown[0]["stat_row_observed"] is False
 assert unknown[0]["opportunities_when_observed"] is None
 assert unknown[0]["team_observed_opportunities"]==10
 assert entry["missing_team_game_stat_row_is_unknown_not_DNP"] is True
 assert "THREE_PRIOR_TEAM_GAME_ROWS_UNVERIFIED" in entry["warnings"]
 assert entry["role_evidence_complete_for_showcase"] is False


def test_future_game_data_does_not_influence_role_volatility_receipt():
 c=db()
 before=mod.capture(FC,c,now=NOW,roster_fetch=roster)
 c.execute("INSERT INTO games VALUES(?,?,?,?,?)",
           ("2026_05_CHI_GB",2026,5,"GB","CHI"))
 c.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?,?)",
           ("one","Established Runner","CHI",2026,5,"2026_05_CHI_GB",
            "REG",99,0,700,0))
 after=mod.capture(FC,c,now=NOW,roster_fetch=roster)
 assert before["player_market_evidence"]==after["player_market_evidence"]
