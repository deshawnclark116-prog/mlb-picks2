"""Fail-closed regression tests for unpromoted NFL V6 joint-game engine."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import nfl_v6_joint_opportunity_shadow as v6


class AsOfTests(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(":memory:")
        self.db.executescript("""
          CREATE TABLE games(game_id TEXT,home_team TEXT,away_team TEXT,
            season INTEGER,week INTEGER,kickoff_utc TEXT,home_score INT,away_score INT);
          CREATE TABLE player_games(player_id TEXT,player_name TEXT,position TEXT,
            team TEXT,opponent TEXT,game_id TEXT,season INTEGER,week INTEGER,
            season_type TEXT,targets INT,receptions INT,receiving_yards INT,
            carries INT,rushing_yards INT,attempts INT,passing_yards INT);
        """)
    def add(self,game_id,week,kickoff,final=True):
        self.db.execute("INSERT INTO games VALUES(?,?,?,?,?,?,?,?)",
          (game_id,"BUF","MIN",2026,week,kickoff,
            24 if final else None,17 if final else None))
        for team,opp in (("BUF","MIN"),("MIN","BUF")):
            for pid,role,targets,carries,attempts in (
              ("wr","WR",9,0,0),("rb","RB",2,16,0),
              ("qb","QB",0,1,34)):
                self.db.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(
                  team+pid,team+role,role,team,opp,game_id,2026,week,"REG",
                  targets,int(targets*.6),targets*8,carries,carries*4,
                  attempts,attempts*7))
    def test_future_and_unfinal_games_cannot_enter_forecast(self):
        self.add("old",1,"2026-09-01T17:00:00Z")
        self.add("too_recent",2,"2026-09-08T17:00:00Z")
        self.add("unfinal",3,"2026-09-02T17:00:00Z",False)
        history=v6.asof_rows(self.db,"2026-09-09T16:00:00Z")
        self.assertEqual({x["game_id"] for x in history},{"old"})
    def test_last_seen_roster_and_team_conservation(self):
        for w in range(1,6):
            self.add(str(w),w,"2026-09-%02dT17:00:00Z"%w)
        rows=v6.asof_rows(self.db,"2026-09-08T17:00:00Z")
        teams,players=v6.game_history(rows)
        league=v6.league_rates(teams)
        game={"game_id":"future","home_team":"BUF","away_team":"MIN",
              "kickoff_utc":"2026-09-08T19:00:00Z"}
        a=v6.simulate_matchup(game,teams,players,league,10000,201)
        b=v6.simulate_matchup(game,teams,players,league,10000,201)
        self.assertEqual(a,b)
        self.assertEqual({p["market"] for p in a["projections"]},
          {"receiving_yards","rushing_yards","passing_yards"})
        self.assertEqual(len(a["projections"]),10)
        for p in a["projections"]:
            self.assertEqual(p["simulations"],10000)
            self.assertFalse(p["is_forward_calibrated"])
            self.assertFalse(p["active_status_verified"])
            self.assertTrue(0<=p["prob_reach_rounded_mean"]<=1)
            self.assertTrue(p["p10"]<=p["projected_median"]<=p["p90"])
        for team in ("BUF","MIN"):
            t=a["teams"][team]
            self.assertAlmostEqual(
              t["projected_pass_attempts"]+t["projected_rush_attempts"],
              t["projected_plays"],delta=.15)
    def test_strict_minimum_and_no_unknown_schedule(self):
        with self.assertRaisesRegex(ValueError,"ten thousand"):
            v6.simulate_matchup({"game_id":"x","home_team":"BUF",
              "away_team":"MIN"}, {}, {}, {}, 9999)


if __name__=="__main__":
    unittest.main()
