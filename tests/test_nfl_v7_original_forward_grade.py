"""V7 postgame grades must be original-only and independent of the result."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from nfl_v7_original_forward_grade import grade, validate


GAME = "2026_11_A_B"
KICK = "2026-11-19T01:15:00Z"


def model_row(pid,name,med,returner=False):
    return {"player_id":pid,"player":name,"team":"A","opponent":"B",
       "market":"receiving_yards","projected_median":med,"simulations":20000,
       "returner_added_to_player_universe":returner}


def original(rows):
    return {"schema":"NFL_V7_PREGAME_ROSTER_RETURNER_JOINT_SHADOW_A",
        "status":"RESEARCH_ONLY_NOT_PROMOTED","game_id":GAME,
        "kickoff_utc":KICK,"capture_observed_at_utc":"2026-11-18T18:00:00Z",
        "source_cutoff_utc":"2026-11-18T17:00:00Z",
        "official_bets":0,"conditional_player_predictions_not_calibrated":True,
        "model":{"game_id":GAME,"kickoff_utc":KICK,"home":"A","away":"B",
                 "projections":rows}}


class GradeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.v7=self.root/"v7";self.v7.mkdir()
        self.v6=self.root/"v6";self.v6.mkdir()
        self.db=self.root/"stats.sqlite"
        con=sqlite3.connect(self.db)
        con.execute("CREATE TABLE games(game_id TEXT,home_score INT,away_score INT)")
        con.execute("""CREATE TABLE player_games(game_id TEXT,season_type TEXT,
                        player_id TEXT,team TEXT,receiving_yards REAL,
                        rushing_yards REAL,passing_yards REAL)""")
        con.execute("INSERT INTO games VALUES (?,23,13)",(GAME,))
        con.executemany("INSERT INTO player_games VALUES (?,?,?,?,?,?,?)",[
            (GAME,"REG","456","A",10,0,0),
            (GAME,"REG","123","A",20,0,0)])
        con.commit();con.close()
        self.write([
            model_row("456","Regular",15),
            model_row("123","Returning",15,True),
            model_row("789","Unverified",100)],[
            model_row("456","Regular",22)])

    def tearDown(self):
        self.temp.cleanup()

    def write(self, rows, previous):
        (self.v7/(GAME+".json")).write_text(json.dumps(original(rows)))
        (self.v6/(GAME+".json")).write_text(json.dumps({
            "game_id":GAME,"source_snapshot_utc":"2026-11-17T19:00:00Z",
            "projections":previous}))

    def test_paired_grading_never_selects_winners_or_zeroes_unknown(self):
        o=grade(self.v7,self.v6,self.db,asof="2026-11-20T20:00:00Z")
        self.assertFalse(o["model_promoted"])
        self.assertFalse(o["eligible_for_production_promotion"])
        self.assertEqual(o["v7_all_original_graded"],2)
        self.assertEqual(o["v7_returner_only_graded"],1)
        self.assertEqual(o["games"][0]["original_count"],3)
        self.assertEqual(o["games"][0]["unresolved"],1)
        self.assertEqual(o["paired_summary"]["n"],1)
        self.assertAlmostEqual(o["paired_summary"]["v7_mae"],5)
        self.assertAlmostEqual(o["paired_summary"]["v6b_mae"],12)
        self.assertEqual(o["excluded"]["new_v7_population_no_v6b_same_player"],1)
        self.assertEqual(o["excluded"]["no_unique_verified_player_stat_row"],1)

    def test_not_completed_is_excluded_not_scored(self):
        o=grade(self.v7,self.v6,self.db,asof="2026-11-18T22:00:00Z")
        self.assertEqual(o["v7_all_original_graded"],0)
        self.assertEqual(o["excluded"]["not_started_yet"],3)

    def test_duplicate_forecast_identity_rejected(self):
        a=original([model_row("456","R",15),model_row("456","R",15)])
        with self.assertRaisesRegex(ValueError,"DUPLICATE"):
            validate(a)

    def test_post_tip_rewrite_and_future_history_rejected(self):
        a=original([model_row("456","R",15)])
        a["capture_observed_at_utc"]=KICK
        with self.assertRaisesRegex(ValueError,"NOT_PRETIP"):
            validate(a)
        a=original([model_row("456","R",15)])
        a["source_cutoff_utc"]=KICK
        with self.assertRaisesRegex(ValueError,"HISTORY_AFTER_SOURCE"):
            validate(a)

    def test_opponent_player_never_assigned_to_forecast_team(self):
        a=original([model_row("456","R",15)])
        a["model"]["projections"][0]["team"]="C"
        with self.assertRaisesRegex(ValueError,"WRONG_GAME_TEAM"):
            validate(a)

    def test_unresolved_duplicate_truth_is_not_selected(self):
        con=sqlite3.connect(self.db)
        con.execute("INSERT INTO player_games VALUES (?,?,?,?,?,?,?)",
                    (GAME,"REG","456","A",40,0,0))
        con.commit();con.close()
        o=grade(self.v7,self.v6,self.db,asof="2026-11-20T20:00:00Z")
        self.assertEqual(o["paired_summary"]["n"],0)
        self.assertEqual(o["games"][0]["unresolved"],2)
        self.assertEqual(o["v7_all_original_graded"],1)

    def test_published_live_champion_pairs_only_exact_original(self):
        f=self.v7/(GAME+".json")
        original_doc=json.loads(f.read_text())
        original_doc["frozen_live_incumbent"]={
            "status":"ORIGINAL_SAME_GAME_PUBLISHED_PRODUCTION",
            "game_id":GAME,"kickoff_utc":KICK,
            "generated_at_utc":"2026-11-18T16:30:00Z",
            "observed_at_utc":"2026-11-18T18:00:00Z",
            "rows":[{"team":"A","player_id":"456",
                     "market":"receiving_yards","projected_median":12,
                     "model_source":"v2_context"}]}
        f.write_text(json.dumps(original_doc))
        result=grade(self.v7,self.v6,self.db,asof="2026-11-20T20:00:00Z")
        self.assertEqual(result["published_incumbent_paired_summary"]["n"],1)
        self.assertAlmostEqual(result["published_incumbent_paired_summary"]["v7_mae"],5)
        self.assertAlmostEqual(result["published_incumbent_paired_summary"]["published_incumbent_mae"],2)
        self.assertEqual(result["games"][0]["paired_live_incumbent"],1)
        self.assertFalse(result["model_promoted"])

    def test_published_incumbent_recorded_after_pretip_observation_rejected(self):
        f=self.v7/(GAME+".json")
        original_doc=json.loads(f.read_text())
        original_doc["frozen_live_incumbent"]={
            "status":"ORIGINAL_SAME_GAME_PUBLISHED_PRODUCTION",
            "game_id":GAME,"kickoff_utc":KICK,
            "generated_at_utc":"2026-11-18T16:30:00Z",
            "observed_at_utc":"2026-11-18T18:01:00Z",
            "rows":[]}
        f.write_text(json.dumps(original_doc))
        with self.assertRaisesRegex(ValueError,"INVALID_OR_LATE_LIVE"):
            grade(self.v7,self.v6,self.db,asof="2026-11-20T20:00:00Z")


if __name__=="__main__":
    unittest.main()
