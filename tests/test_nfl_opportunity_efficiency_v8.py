"""Proof that NFL V8 features are strictly pregame and markets use opportunity data."""
import sqlite3
import unittest
import math
from nfl_opportunity_efficiency_v8_research import (
    MARKETS,FEATURES,derive,build_rows,ASOF,ratio,ypo,mean,summarize
)

class FootballRoleV8Tests(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(":memory:")
        self.db.executescript("""
        CREATE TABLE player_games (
          player_id TEXT,player_name TEXT,position TEXT,team TEXT,opponent TEXT,
          season INT,week INT,game_id TEXT,is_home INT,season_type TEXT,
          targets REAL,receiving_yards REAL,carries REAL,rushing_yards REAL,
          attempts REAL,passing_yards REAL
        );
        CREATE TABLE games(
          game_id TEXT,season INT,week INT,home_team TEXT,away_team TEXT,
          spread_line REAL,total_line REAL
        );
        """)
        for w in range(1,10):
            g="G"+str(w)
            self.db.execute("INSERT INTO games VALUES(?,?,?,?,?,?,?)",
                            (g,2023,w,"KC","ARI",-3.5,46.5))
            self.db.executemany("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",[
             ("wr","Test Receiver","WR","KC","ARI",2023,w,g,1,"REG",7,41+w*2,0,0,0,0),
             ("rb","Test Runner","RB","KC","ARI",2023,w,g,1,"REG",0,0,12,60+w,0,0),
             ("qb","Test Passer","QB","KC","ARI",2023,w,g,1,"REG",0,0,0,0,34,225+w*3),
             ("oppqb","Opp Passer","QB","ARI","KC",2023,w,g,0,"REG",0,0,0,0,24,190+w),
            ])
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def test_all_three_markets_and_pregame_role_features(self):
        self.assertEqual(set(MARKETS),{"receiving_yards","rushing_yards","passing_yards"})
        self.assertEqual(MARKETS["receiving_yards"]["vol"],"targets")
        self.assertEqual(MARKETS["rushing_yards"]["vol"],"carries")
        self.assertEqual(MARKETS["passing_yards"]["vol"],"attempts")
        self.assertIn("last3_to_prev5_vol",FEATURES)
        self.assertIn("last3_yards_per_opportunity",FEATURES)

    def test_no_target_game_boxscore_leaks_into_pregame_forecast_features(self):
        for market,col in (("receiving_yards","receiving_yards"),
                           ("rushing_yards","rushing_yards"),
                           ("passing_yards","passing_yards")):
            initial=build_rows(self.db,market)
            fifth=[x for x in initial if x["week"]==5]
            sixth=[x for x in initial if x["week"]==6]
            self.assertTrue(fifth and sixth)
            f_before=dict(fifth[0]["features"])
            f6_before=dict(sixth[0]["features"])
            # Deliberately poison target game's actual yards.
            self.db.execute(f"UPDATE player_games SET {col}={col}+9999 WHERE week=5")
            after=build_rows(self.db,market)
            f_after=next(x for x in after if x["week"]==5)["features"]
            f6_after=next(x for x in after if x["week"]==6)["features"]
            for name in FEATURES:
                a,b=f_before[name],f_after[name]
                self.assertTrue(a==b or (isinstance(a,float) and isinstance(b,float) and
                    math.isnan(a) and math.isnan(b)),f"{market} target-week leakage: {name}")
            self.assertNotEqual(f6_before["last1_yds"],f6_after["last1_yds"])
            self.db.rollback()

    def test_usage_trend_responds_to_decline_and_volume_not_numeric_hype(self):
        seq=[{"season":2025,"week":i+1,"yds":65 if i<5 else 20,
              "vol":12 if i<5 else 4,"team_vol":40,"team":"KC"}
             for i in range(8)]
        original={"last3_avg_vol":4,"cur_avg_vol":9,"prior_avg_vol":None,
                  "last3_avg_yds":20,"cur_avg_yds":48.1}
        out=derive(seq,original,2025,9,"RB","KC")
        self.assertLess(out["last3_to_prev5_vol"],.5)
        self.assertLess(out["last3_to_prev5_yards"],.5)
        self.assertTrue(out["last3_yards_per_opportunity"]>0)

    def test_paired_calibration_report_is_not_an_invented_probability(self):
        examples=[{"season":2026,"week":w,"game_id":"g"+str(w),
                   "actual_yards":y} for w,y in ((1,50),(2,25),(3,40))]
        q=summarize(examples,[45,30,40],[35,35,35])
        self.assertEqual(q["n"],3)
        self.assertGreater(q["benefit_positive_yards"],0)
        self.assertEqual(q["n_game_clusters"],3)
        self.assertNotIn("probability_of_hit",q)

if __name__=="__main__":
    unittest.main()
