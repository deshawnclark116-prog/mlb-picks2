"""Future-only NHL V8 passing shadow ledger is immutable, exact-ID and no fake DNP."""
from datetime import datetime,timezone,timedelta
import sqlite3
import unittest
from nfl_v8_passing_paper import (
 SOURCE,SCHEMA,first_seen_pregame,grade,timestamp
)

NOW=datetime(2026,11,22,16,30,tzinfo=timezone.utc)
KICK="2026-11-22T18:00:00Z"
def base(pid="QB1",name="Quarterback",kickoff=KICK):
    return {
      "schema":SCHEMA,"source":SOURCE,"season":2026,"week":11,
      "market":"passing_yards","game_id":"2026_11_ARI_KC",
      "kickoff_utc":kickoff,
      "player_id":pid,"player":name,"team":"KC","opponent":"ARI",
      "prediction_yards":234.5,"baseline_blend_yards":223.2
    }
class PassingFirstSeenTests(unittest.TestCase):
    def setUp(self):
        self.con=sqlite3.connect(":memory:")
        self.con.executescript("""
          CREATE TABLE games (
            game_id TEXT,season INT,week INT,home_score REAL,away_score REAL);
          CREATE TABLE player_games (
            player_id TEXT,season INT,week INT,season_type TEXT,
            passing_yards REAL,attempts REAL);
        """)
        self.con.execute("INSERT INTO games VALUES (?,?,?,?,?)",
            ("2026_11_ARI_KC",2026,11,None,None))
        self.con.commit()
    def tearDown(self):
        self.con.close()
    def test_first_seen_once_and_a_second_pregame_run_cannot_replace_original(self):
        first=first_seen_pregame([], [base()], NOW)
        self.assertEqual(len(first),1)
        self.assertEqual(first[0]["prediction_yards"],234.5)
        after=first_seen_pregame(first,[{**base(),"prediction_yards":295.1}],
                                 NOW+timedelta(minutes=20))
        self.assertEqual(after,[])
        self.assertEqual(first[0]["prediction_yards"],234.5)

    def test_after_kickoff_never_firstseen_pregame(self):
        later=datetime(2026,11,22,19,0,tzinfo=timezone.utc)
        self.assertEqual(first_seen_pregame([],[base()],later),[])
        self.assertIsNone(timestamp("not-a-time"))

    def test_grading_needs_official_final_and_actual_game(self):
        frozen=first_seen_pregame([], [base()], NOW)
        after=datetime(2026,11,23,4,0,tzinfo=timezone.utc)
        self.assertEqual(grade(frozen,self.con,after)["graded_n"],0)
        self.con.execute("UPDATE games SET home_score=24,away_score=10")
        self.con.commit()
        self.assertEqual(grade(frozen,self.con,after)["graded_n"],0,
          "QB DNP / source missing cannot be zero")
        self.con.execute("INSERT INTO player_games VALUES (?,?,?,?,?,?)",
                         ("QB1",2026,11,"REG",250,30))
        self.con.commit()
        g=grade(frozen,self.con,after)
        self.assertEqual(g["graded_n"],1)
        self.assertEqual(g["records"][0]["actual_passing_yards"],250)
        self.assertEqual(g["records"][0]["absolute_error"],15.5)
        self.assertEqual(g["records"][0]["simple_average_error"],26.8)
        self.assertGreater(g["mean_benefit_yards"],0)

    def test_stale_after_kickoff_forecast_does_not_grade_even_after_final(self):
        bogus={**base(),"original_first_seen_at_utc":"2026-11-22T19:00:00Z"}
        self.con.execute("UPDATE games SET home_score=24,away_score=10")
        self.con.execute("INSERT INTO player_games VALUES (?,?,?,?,?,?)",
                         ("QB1",2026,11,"REG",250,30))
        self.con.commit()
        self.assertEqual(grade([bogus],self.con,datetime(2026,11,23,tzinfo=timezone.utc))["graded_n"],0)

    def test_no_data_from_opponent_player_or_other_week(self):
        frozen=first_seen_pregame([], [base()], NOW)
        self.con.execute("UPDATE games SET home_score=24,away_score=10")
        self.con.executemany("INSERT INTO player_games VALUES (?,?,?,?,?,?)",[
            ("QB1",2026,10,"REG",800,70),
            ("QB2",2026,11,"REG",12,8)
        ])
        self.con.commit()
        self.assertEqual(grade(frozen,self.con,datetime(2026,11,23,tzinfo=timezone.utc))["graded_n"],0)

if __name__=="__main__":unittest.main()
