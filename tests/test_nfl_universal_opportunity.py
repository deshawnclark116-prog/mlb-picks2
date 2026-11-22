"""Deterministic NFL universal opportunity-model and serving parity tests.

No player names in eligibility or forecast formulas; as-of proof is verified
by perturbing the outcome for the CURRENT game and confirming its prediction
does not change (but it CAN affect a later game's forecast).
"""
import math
import sqlite3
import unittest

import nfl_yardage_v2 as v2
from nfl_universal_opportunity_research import (
    MARKETS,Observation,pregame_forecast,retrospective_rows,evaluate,
)

def sample(market, vols, ys, season=2025):
    return [Observation("p1","Example","QB" if market.startswith("passing") else
                        "RB" if "rushing" in market else "WR","TM",season,i+1,
                        "game"+str(i),float(vol),float(y))
            for i,(vol,y) in enumerate(zip(vols,ys))]

class UniversalModelTests(unittest.TestCase):
    def test_serving_uses_same_recent_volume_as_training_for_both_markets(self):
        for market,pos,minimum in (("receiving_yards","WR",3.0),
                                   ("rushing_yards","RB",5.0)):
            good=[{"vol":minimum+1,"pos":pos} for _ in range(3)]
            bad=[{"vol":0,"pos":pos},{"vol":0,"pos":pos},
                 {"vol":minimum,"pos":pos}]
            self.assertTrue(v2.serving_opportunity_eligible(good,market))
            self.assertFalse(v2.serving_opportunity_eligible(bad,market))
            self.assertFalse(v2.serving_opportunity_eligible(good[:2],market))
            self.assertFalse(v2.serving_opportunity_eligible([{**g,"pos":"QB"} for g in good],market))

    def test_all_six_market_types_present_and_definitions_explicit(self):
        self.assertEqual(set(MARKETS),
            {"rushing_yards","receiving_yards","passing_yards",
             "receptions","passing_completions","rushing_attempts"})
        self.assertEqual(MARKETS["passing_yards"][1],"attempts")
        self.assertEqual(MARKETS["receiving_yards"][1],"targets")
        self.assertEqual(MARKETS["rushing_yards"][1],"carries")

    def test_role_decline_falls_but_is_shrunk_marketwide(self):
        for market,volume in (("rushing_yards",10),("receiving_yards",7),
                              ("passing_yards",45),("receptions",7),
                              ("passing_completions",45),("rushing_attempts",10)):
            actuals=[80.0]*5+[20.0]*3
            h=sample(market,[volume]*5+[volume*.45]*3,actuals)
            baseline=pregame_forecast(h,2025,market,0,0)
            adjusted=pregame_forecast(h,2025,market,.75,.65)
            self.assertIsNotNone(baseline,market)
            self.assertIsNotNone(adjusted,market)
            self.assertLess(adjusted["forecast"],baseline["forecast"],market)
            self.assertEqual(adjusted["role_change"],"DECLINING")
            self.assertGreaterEqual(adjusted["forecast"],baseline["forecast"]*.5)
            self.assertEqual(adjusted["probability_source"],"NONE_RESEARCH_NUMERICAL_ONLY")

    def test_role_increase_applies_in_all_market_types(self):
        for market,volume in (("rushing_yards",5),("receiving_yards",4),
                              ("passing_yards",20),("receptions",4),
                              ("passing_completions",20),("rushing_attempts",5)):
            h=sample(market,[volume]*5+[volume*2]*3,[30.0]*5+[60.0]*3)
            baseline=pregame_forecast(h,2025,market,0,0)
            adjusted=pregame_forecast(h,2025,market,.55,.5)
            self.assertGreater(adjusted["forecast"],baseline["forecast"],market)

    def test_missing_opportunities_do_not_create_forecast(self):
        h=sample("receiving_yards",[2,1,0,0],[20,10,0,0])
        self.assertIsNone(pregame_forecast(h,2025,"receiving_yards",.75,.65))
        h=sample("passing_yards",[30,15,4],[100,80,50])
        self.assertIsNone(pregame_forecast(h,2025,"passing_yards",.75,.65))

    def db(self):
        con=sqlite3.connect(":memory:")
        con.executescript("""CREATE TABLE player_games (
          player_id TEXT,player_name TEXT,position TEXT,team TEXT,
          season INTEGER,week INTEGER,game_id TEXT,season_type TEXT,
          targets REAL,receiving_yards REAL,carries REAL,rushing_yards REAL,
          attempts REAL,passing_yards REAL,receptions REAL,completions REAL
        );""")
        values=[]
        # Six markets, four weeks. Outcome of week3 must only influence
        # week4 and not the forecast recorded before week3.
        for week in range(1,6):
            values.append(("QB","Q","QB","TM",2025,week,f"g{week}","REG",
                           0,0,0,0,30,180+week*10,0,20+week))
            values.append(("WR","W","WR","TM",2025,week,f"g{week}","REG",
                           7,30+week*5,0,0,0,0,4,0))
            values.append(("RB","R","RB","TM",2025,week,f"g{week}","REG",
                           0,0,10,50+week*5,0,0,0,0))
        con.executemany("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",values)
        con.commit()
        return con

    def test_no_target_week_data_leakage_for_all_markets(self):
        con=self.db()
        for market in MARKETS:
            before=retrospective_rows(con,market)
            target=[r for r in before if r["season"]==2025 and r["week"]==4]
            self.assertEqual(len(target),1,market)
            baseline=target[0]["baseline"]
            candidates=dict(target[0]["candidates"])
            metric=MARKETS[market][2]
            con.execute(f"UPDATE player_games SET {metric}={metric}+9999 WHERE week=4")
            after=retrospective_rows(con,market)
            corresponding=[r for r in after if r["season"]==2025 and r["week"]==4]
            self.assertEqual(len(corresponding),1,market)
            self.assertEqual(corresponding[0]["baseline"],baseline,market)
            self.assertEqual(corresponding[0]["candidates"],candidates,market)
            # restore before next market test
            con.rollback()
        con.close()

    def test_holdouts_reported_but_never_autopromote(self):
        con=self.db()
        report=evaluate(con)
        self.assertEqual(report["status"],"RESEARCH_ONLY_UNVALIDATED_DO_NOT_SHIP")
        self.assertTrue(report["no_postgame_feature_leakage"])
        for market,detail in report["markets"].items():
            self.assertFalse(detail["production_promoted"])
            self.assertEqual(detail["2025"]["n"],2,market)
            self.assertEqual(detail["2026"]["n"],0,market)
        con.close()

if __name__=="__main__":
    unittest.main()
