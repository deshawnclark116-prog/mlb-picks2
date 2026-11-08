import copy
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
import nfl_model_target_forward as F

class TargetForwardContract(unittest.TestCase):
 def setUp(self):
  self.clock=datetime(2026,11,8,17,45,tzinfo=timezone.utc)
  self.kickoff="2026-11-08T18:00:00Z"
  chosen={"yards":35,"probability":.81,"wilson_lower_95":.80}
  self.player={"market":"receiving_yards","player_id":"ESPN123",
   "player":"Example Wideout","team":"CLE","opponent":"NO",
   "kickoff_utc":self.kickoff,"projected_mean":50.,
   "projected_median":45.,"model_source":"blended_sim",
   "projection_probability_source":"UNCALIBRATED_HISTORICAL_USAGE_BOOTSTRAP",
   "nfl_model_target_event":"P_YARDS_AT_LEAST_FIXED_MEANINGFUL_THRESHOLD",
   "projection_simulation_n":10000,"nfl_model_target":chosen,
   "nfl_model_target_grid":{
    "schema":"nfl_meaningful_yardage_targets_v1",
    "simulation_n":10000,"minimum_absolute_yards":25,
    "minimum_relative_mean_fraction":.55,"required_wilson_lower_95":.65,
    "grid_step_yards":5,"conditional_on_modeled_opportunities":True,
    "forward_calibrated":False,"official_active_status_verified":False,
    "chosen_target":chosen,
    "grid":[{"yards":30,"probability":.92,"wilson_lower_95":.91},
            chosen,{"yards":40,"probability":.49,"wilson_lower_95":.48}]
   }}
  self.board={"season":2026,"week":9,
    "generated_at_utc":"2026-11-08T17:41:00Z","picks":[self.player]}
  self.role={"season":2026,"week":9,
    "schema":"nfl-role-evidence-coverage-v1",
    "status":"PREGAME_RESEARCH_ROLE_SNAPSHOT",
    "forecast_generated_at_utc":self.board["generated_at_utc"],
    "captured_at_utc":"2026-11-08T17:41:01Z",
    "player_market_evidence":[{
     "player_id":"ESPN123","player":"Example Wideout","team":"CLE",
     "opponent":"NO","market":"receiving_yards",
     "kickoff_utc":self.kickoff,
     "published_model_median":45.,"published_model_source":"blended_sim",
     "role_evidence_complete_for_showcase":True,"observed_stat_rows":3,
     "roster_status":"ROSTER_LISTED_NOT_STARTER_VERIFIED",
     "warnings":[],"has_verified_starting_role":False,
     "has_calibrated_book_bet_probability":False
    }]}

 def test_pregame_capture_and_original_first_seen_lock(self):
  self.assertTrue(F.valid_target(self.player))
  r=F.capture_rows(self.board,self.role,self.clock)
  self.assertEqual(len(r),1)
  self.assertEqual(r[0]["target_yards"],35)
  self.assertFalse(r[0]["forward_calibrated"])
  with tempfile.TemporaryDirectory() as td:
   fn=Path(td)/"firstseen.jsonl"
   self.assertEqual(F.capture(self.board,self.role,self.clock,fn)["new_firstseen"],1)
   self.player["nfl_model_target"]["probability"]=.91
   self.assertEqual(F.capture(self.board,self.role,self.clock,fn)["new_firstseen"],0)
   frozen=json.loads(fn.read_text().strip())
   self.assertEqual(frozen["model_hit_probability"],.81)

 def test_does_not_backfill_live_game_or_unverified_role(self):
  after=datetime(2026,11,8,18,1,tzinfo=timezone.utc)
  self.assertEqual(F.capture_rows(self.board,self.role,after),[])
  denied=copy.deepcopy(self.role)
  denied["player_market_evidence"][0]["warnings"]=["INJURY_UNKNOWN"]
  self.assertEqual(F.capture_rows(self.board,denied,self.clock),[])
  bad=copy.deepcopy(self.player)
  bad["nfl_model_target"]["yards"]=5
  self.assertFalse(F.valid_target(bad))
  bad=copy.deepcopy(self.player)
  bad["projection_probability_source"]="UNAVAILABLE_V2_CONTEXT_NO_MATCHED_DISTRIBUTION"
  self.assertFalse(F.valid_target(bad))

 def test_forward_only_grader_uses_correct_game_and_no_assumed_zero(self):
  first=F.capture_rows(self.board,self.role,self.clock)[0]
  second={**first,"player":"Missing Receiver","player_id":"ESPN999"}
  with tempfile.TemporaryDirectory() as td:
   db=Path(td)/"nfl.sqlite"
   con=sqlite3.connect(db)
   con.execute("CREATE TABLE player_games (season INTEGER,week INTEGER,player_id TEXT,player_name TEXT,team TEXT,opponent TEXT,rushing_yards REAL,receiving_yards REAL)")
   con.execute("INSERT INTO player_games VALUES(2026,9,'GSIS123','Example Wideout','CLE','NO',0,39)")
   con.execute("INSERT INTO player_games VALUES(2026,8,'GSIS123','Example Wideout','CLE','PIT',0,0)")
   con.commit();con.close()
   early=F.grade([first,second],db,datetime(2026,11,8,20,tzinfo=timezone.utc).timestamp())
   self.assertEqual(early["summary"]["graded"],0)
   self.assertEqual(early["summary"]["pending"],2)
   final=F.grade([first,second],db,datetime(2026,11,9,3,tzinfo=timezone.utc).timestamp())
   self.assertEqual(final["summary"]["graded"],1)
   self.assertEqual(final["summary"]["unresolved"],1)
   self.assertEqual(final["summary"]["observed_hit_rate"],1.0)
   self.assertAlmostEqual(final["summary"]["brier"],(.19)**2,places=5)

if __name__=="__main__":
 unittest.main()
