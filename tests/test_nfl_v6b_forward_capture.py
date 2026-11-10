"""Only authentic unstarted NFL game can be frozen; never overwrite firstseen."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nfl_v6b_forward_capture as cap


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.dir=Path(self.tmp.name)
        self.db=self.dir/"source.sqlite"
        self.snap=self.dir/"snap_counts_2026.csv"
        self.snap.write_text("this is synthetic source for fixture tests\n"*50)
        self.root=self.dir/"firstseen"
        with sqlite3.connect(self.db) as con:
            con.executescript("""
              CREATE TABLE games(season INT,week INT,game_id TEXT,kickoff_utc TEXT);
              INSERT INTO games VALUES(2026,10,'2026_10_WAS_NYG',
                '2026-11-13T01:15:00Z');
              INSERT INTO games VALUES(2026,10,'2026_10_OLD_PAST',
                '2026-11-11T01:15:00Z');
              INSERT INTO games VALUES(2026,10,'2026_10_FUTURE_FAR',
                '2026-11-22T18:00:00Z');
            """)
    def tearDown(self):
        self.tmp.cleanup()
    def model(self,*args,**kwargs):
        self.assertTrue(kwargs["only_future"])
        self.assertEqual(kwargs["captured_at_utc"],"2026-11-12T15:00:00+00:00")
        self.assertGreaterEqual(kwargs["sims"],10000)
        return {"version":"NFL_V6B_SNAP_ROLE_SHADOW_A",
                "forward_source_cutoff_enforced":True,
                "only_future_game_snapshots":True,
                "asof_policy":"MIN(CAPTURE_TIME,KICKOFF_MINUS_2H)_24H_DATA_LAG",
                "snap_role_source_receipts":[{"game_id":"2026_10_WAS_NYG","source_files":[]}],
                "games":[{"game_id":"2026_10_WAS_NYG",
                  "kickoff_utc":"2026-11-13T01:15:00Z",
                  "teams":{"WAS":{"status":"SIMULATED_ROLE_UNVERIFIED"}},
                  "projections":[{"player":"Example","projected_mean":61.0}]}]}
    def test_strict_time_and_write_once(self):
        with patch.object(cap.v6,"run",side_effect=self.model) as mocked:
            r=cap.capture(self.db,[self.snap],self.root,
              now="2026-11-12T15:00:00Z")
            self.assertEqual(r["summary"]["captured"],["2026_10_WAS_NYG"])
            self.assertEqual(mocked.call_count,1)
        frozen=json.loads((self.root/"2026_10_WAS_NYG.json").read_text())
        self.assertEqual(frozen["projections"][0]["projected_mean"],61)
        self.assertFalse(frozen["team_roster_active_verified"])
        self.assertTrue(frozen["source_db_sha256"])
        with patch.object(cap.v6,"run") as mocked:
            r2=cap.capture(self.db,[self.snap],self.root,
              now="2026-11-12T15:00:00Z")
            self.assertEqual(r2["summary"]["captured"],[])
            self.assertEqual(r2["summary"]["already_frozen"],["2026_10_WAS_NYG"])
            mocked.assert_not_called()
        self.assertEqual(frozen,json.loads((self.root/"2026_10_WAS_NYG.json").read_text()))
    def test_after_kickoff_and_window_are_excluded(self):
        with patch.object(cap.v6,"run") as mocked:
            report=cap.capture(self.db,[self.snap],self.root,
              now="2026-11-13T03:00:00Z")
            self.assertEqual(report["summary"]["captured"],[])
            mocked.assert_not_called()
    def test_missing_source_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"missing actual"):
            cap.capture(self.db,[self.dir/"not_exists.csv"],self.root,
              now="2026-11-12T15:00:00Z")


if __name__=="__main__":
    unittest.main()
