"""Deterministic identity, temporal ordering and previous miss features."""
import sqlite3
import unittest

from nfl_universal_opportunity_research import Observation
from nfl_universal_learned_role_challenger import (
    FEATURES,model_features,make_observations,metric
)

class LearnedRoleTests(unittest.TestCase):
    def test_error_history_is_prior_forecast_not_current_game(self):
        history=[Observation("id","Test WR","WR","KC",2025,w,"g"+str(w),
                             6,float(y)) for w,y in [(1,80),(2,75),(3,70),(4,20)]]
        zero=model_features(history,[],2025,5,"KC","receiving_yards")
        miss=model_features(history,[-30,-40,-45],2025,5,"KC","receiving_yards")
        self.assertEqual(zero["baseline"],miss["baseline"])
        self.assertEqual(miss["recent_3_overprediction_share"],1.)
        self.assertEqual(miss["recent_3_signed_error_mean"],-115/3)
        self.assertLess(miss["recent_3_signed_error_mean"],0)
        self.assertEqual(miss["recent_3_error_n"],3)
        self.assertTrue(set(FEATURES).issubset(miss))

    def test_no_future_game_or_same_week_yardage_leakage(self):
        con=sqlite3.connect(":memory:")
        con.execute("""CREATE TABLE player_games(
          player_id TEXT,player_name TEXT,position TEXT,team TEXT,season INT,
          week INT,game_id TEXT,season_type TEXT,
          targets REAL,receiving_yards REAL)""")
        con.executemany("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?)",[
          ("1","Test WR","WR","KC",2025,w,f"g{w}","REG",6,30+5*w)
          for w in range(1,8)
        ])
        con.commit()
        b=make_observations(con,"receiving_yards")
        wk4=next(r for r in b if r["week"]==4)
        wk5=next(r for r in b if r["week"]==5)
        con.execute("UPDATE player_games SET receiving_yards=9999 WHERE week=4")
        changed=make_observations(con,"receiving_yards")
        a4=next(r for r in changed if r["week"]==4)
        a5=next(r for r in changed if r["week"]==5)
        self.assertEqual(wk4["features"],a4["features"])
        self.assertEqual(wk4["baseline"],a4["baseline"])
        self.assertNotEqual(wk4["actual"],a4["actual"])
        self.assertNotEqual(wk5["features"]["last1_yards"],a5["features"]["last1_yards"])
        self.assertNotEqual(wk5["features"]["recent_3_signed_error_mean"],
                            a5["features"]["recent_3_signed_error_mean"])
        con.close()

    def test_metrics_fail_closed_on_unmatched_predictions(self):
        rows=[{"actual":50.0,"baseline":40.0},
              {"actual":20.0,"baseline":30.0}]
        result=metric(rows,[44,24])
        self.assertEqual(result["n"],2)
        self.assertEqual(result["baseline_mae"],10.)
        self.assertEqual(result["challenger_mae"],4.)
        with self.assertRaises(AssertionError):
            metric(rows,[20])

if __name__=="__main__":
    unittest.main()
