"""NFL V6 verified source and strict snap/player identity guards."""
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from nfl_v6_snap_role_evidence import read_snap_roles,observed_role


class SnapRoleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.dir=Path(self.tmp.name)
        self.db=self.dir/"nfl.sqlite"
        with sqlite3.connect(self.db) as con:
            con.executescript("""
            CREATE TABLE games(game_id TEXT,season INT,week INT,
                kickoff_utc TEXT,home_team TEXT,away_team TEXT,
                home_score INT,away_score INT);
            INSERT INTO games VALUES('2026_07_MIN_CHI',2026,7,
              '2026-10-25T17:00:00Z','MIN','CHI',24,20);
            INSERT INTO games VALUES('2026_08_MIN_DET',2026,8,
              '2026-11-01T18:00:00Z','MIN','DET',28,25);
            INSERT INTO games VALUES('2026_09_BUF_MIN',2026,9,
              '2026-11-10T01:15:00Z','MIN','BUF',NULL,NULL);
            """)
        self.csv=self.dir/"snap_counts_2026.csv"
        self.header=["game_id","season","week","game_type","team","player",
                     "offense_snaps","offense_pct"]
        self.samples=[
            ["2026_07_MIN_CHI","2026","7","REG","MIN","Justin Jefferson","47",".88"],
            ["2026_08_MIN_DET","2026","8","REG","MIN","Justin Jefferson","51",".96"],
            ["2026_09_BUF_MIN","2026","9","REG","MIN","Justin Jefferson","80",".99"],
            ["2026_08_MIN_DET","2026","8","REG","BUF","Justin Jefferson","50",".95"],
            ["2026_08_MIN_DET","2026","8","REG","DET","Other Wideout","0","0"],
        ]
        self.write()
    def write(self):
        with self.csv.open("w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(self.header)
            writer.writerows(self.samples)
    def tearDown(self):
        self.tmp.cleanup()
    def test_historical_cutoff_exact_game_and_team(self):
        source=read_snap_roles([self.csv],self.db,"2026-11-09T16:00:00Z")
        self.assertEqual(len(source["by_player_game"]),2)
        self.assertNotIn(("2026_08_MIN_DET","BUF","justin jefferson"),
                         source["by_player_game"])
        self.assertFalse(source["routes_run_available"])
        self.assertFalse(source["historic_asof_snap_publication_timing_verified"])
    def test_snap_role_uses_verified_denominators_not_routes(self):
        source=read_snap_roles([self.csv],self.db,"2026-11-09T16:00:00Z")
        hist=[
          {"game_id":"2026_07_MIN_CHI","team":"MIN","targets":8,"carries":0},
          {"game_id":"2026_08_MIN_DET","team":"MIN","targets":11,"carries":0}]
        role=observed_role(hist,"Justin Jefferson","MIN",source)
        self.assertEqual(role["matched"],2)
        self.assertEqual(role["status"],"OBSERVED_ASOF_OFFENSE_SNAP_ROLE_NOT_ROUTE_DATA")
        self.assertAlmostEqual(role["target_per_offensive_snap"],19/98,places=4)
        self.assertGreater(role["offense_snap_share"],.9)
        self.assertFalse(role["active_status_verified"])
        self.assertFalse(role["route_participation_verified"])
    def test_ambiguous_name_identity_is_refused(self):
        self.samples.append(self.samples[1].copy())
        self.write()
        source=read_snap_roles([self.csv],self.db,"2026-11-09T16:00:00Z")
        self.assertNotIn(("2026_08_MIN_DET","MIN","justin jefferson"),
                         source["by_player_game"])
        self.assertEqual(source["counts"]["ambiguous_player_game_names"],1)
        hist=[
          {"game_id":"2026_07_MIN_CHI","team":"MIN","targets":8,"carries":0},
          {"game_id":"2026_08_MIN_DET","team":"MIN","targets":11,"carries":0}]
        role=observed_role(hist,"Justin Jefferson","MIN",source)
        self.assertEqual(role["status"],"INSUFFICIENT_SNAP_SOURCE")
        self.assertEqual(role["matched"],1)
    def test_same_day_unavailable_and_invalid_schema(self):
        source=read_snap_roles([self.csv],self.db,"2026-11-01T20:00:00Z")
        self.assertEqual(len(source["by_player_game"]),1)
        self.csv.write_text("wrong,fields\n1,2\n")
        with self.assertRaisesRegex(ValueError,"schema"):
            read_snap_roles([self.csv],self.db,"2026-11-09T16:00:00Z")


if __name__=="__main__":
    unittest.main()
