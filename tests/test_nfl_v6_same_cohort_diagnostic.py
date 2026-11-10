"""Same-game and same-player comparison cannot manufacture model improvement."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from nfl_v6_same_cohort_diagnostic import compare


def forecast():
    return {"version":"NFL_V6_JOINT_OPPORTUNITY_SHADOW_A",
      "status":"RESEARCH_SHADOW_NOT_PROMOTED","season":2026,"week":9,
      "games":[{"game_id":"2026_09_BUF_MIN","home":"MIN","away":"BUF",
        "kickoff_utc":"2026-11-10T01:15:00Z","projections":[
          {"player_id":"GSIS-1","player":"Justin Jefferson",
            "team":"MIN","opponent":"BUF","market":"receiving_yards",
            "projected_mean":58.0,"naive_recent3_mean":51.0,
            "rounded_mean":58,"prob_reach_rounded_mean":.449,
            "prob_within_15_yards_of_rounded_mean":.19}]}]}


def firstseen(name="Justin Jefferson", recorded="2026-11-10T00:54:00Z"):
    return {"schema":"NFL_NUMERIC_PRECISION_FIRSTSEEN_V1",
      "season":2026,"week":9,"player_id":"ESPN-1","player":name,
      "team":"MIN","opponent":"BUF","market":"receiving_yards",
      "kickoff_utc":"2026-11-10T01:15:00Z",
      "recorded_at_utc":recorded,
      "source_generated_at_utc":"2026-11-10T00:50:00Z",
      "model_mean":57.5,"model_median":45.0,"model_source":"blended_sim"}


class PairedTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.db=Path(self.tmp.name)/"nfl.sqlite"
        with sqlite3.connect(self.db) as con:
            con.executescript("""
            CREATE TABLE games(game_id TEXT,home_score INTEGER,away_score INTEGER);
            CREATE TABLE player_games(game_id TEXT,team TEXT,player_id TEXT,
              player_name TEXT,season_type TEXT,receiving_yards REAL,
              rushing_yards REAL,passing_yards REAL);
            INSERT INTO games VALUES('2026_09_BUF_MIN',24,23);
            INSERT INTO player_games VALUES(
              '2026_09_BUF_MIN','MIN','GSIS-1','Justin Jefferson',
              'REG',86,0,0);
            """)
    def tearDown(self):
        self.tmp.cleanup()
    def test_unique_name_crosswalk_exact_game_and_cohort(self):
        r=compare(forecast(),[firstseen()],self.db)
        self.assertEqual(r["matched"],1)
        self.assertEqual(r["identity_methods"],
                         {"UNIQUE_SAME_GAME_TEAM_PLAYER_NAME":1})
        self.assertEqual(r["overall"]["v6_mean_mae"],28.0)
        self.assertEqual(r["overall"]["incumbent_mean_mae"],28.5)
        self.assertEqual(r["featured_incumbent_mean_ge_60"]["n"],0)
    def test_after_kickoff_not_promotable(self):
        r=compare(forecast(),[firstseen(
          recorded="2026-11-10T01:20:00Z")],self.db)
        self.assertEqual(r["matched"],0)
        self.assertEqual(r["excluded"],{"not_recorded_pregame":1})
    def test_no_cross_team_and_duplicate(self):
        wrong=firstseen()
        wrong["team"]="BUF";wrong["opponent"]="MIN"
        r=compare(forecast(),[wrong],self.db)
        self.assertEqual(r["matched"],0)
        self.assertIn("no_unique_v6_actual_match",r["excluded"])
        r2=compare(forecast(),[firstseen(),firstseen()],self.db)
        self.assertEqual(r2["matched"],1)
        self.assertEqual(r2["excluded"]["duplicate_firstseen"],1)
    def test_name_ambiguity_refused(self):
        f=forecast()
        duplicate=dict(f["games"][0]["projections"][0],player_id="GSIS-2")
        f["games"][0]["projections"].append(duplicate)
        with sqlite3.connect(self.db) as con:
            con.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?)",
              ("2026_09_BUF_MIN","MIN","GSIS-2","Justin Jefferson","REG",37,0,0))
        r=compare(f,[firstseen()],self.db)
        self.assertEqual(r["matched"],0)
        self.assertEqual(r["excluded"]["no_unique_v6_actual_match"],1)


if __name__=="__main__":
    unittest.main()
