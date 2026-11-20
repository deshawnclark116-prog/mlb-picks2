"""Tests for freezing authentic live NFL numerical V2/serving rows."""
import json
import tempfile
import unittest
from pathlib import Path

from nfl_v7_incumbent_snapshot import capture_incumbent

GAME={"game_id":"2026_11_BUF_MIA","home_team":"BUF","away_team":"MIA",
      "kickoff_utc":"2026-11-22T18:00:00Z","season":2026,"week":11}
OBS="2026-11-21T18:00:00Z"


def row(pid="00-0123456",market="rushing_yards",median=48.5):
    return {"player_id":pid,"player":"Fictional Back","team":"BUF",
            "opponent":"MIA","season":2026,"week":11,
            "kickoff_utc":GAME["kickoff_utc"],"market":market,
            "model_source":"v2_context","projected_median":median}


def document():
    return {"generated_at_utc":"2026-11-21T17:30:00Z",
            "season":2026,"week":11,"scheduled_games":[{
                "home_team":"BUF","away_team":"MIA",
                "kickoff_utc":GAME["kickoff_utc"]}],
            "picks":[row(),{**row(pid="other"),"team":"NYJ","opponent":"NE"}]}


class SourceReceiptTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.source=Path(self.tmp.name)/"published.json"
        self.write(document())

    def tearDown(self):
        self.tmp.cleanup()

    def write(self,j):
        self.source.write_text(json.dumps(j))

    def read(self,**kw):
        return capture_incumbent(self.source,GAME,observed_at_utc=OBS,**kw)

    def test_exact_published_frozen_numeric_original(self):
        r=self.read()
        self.assertEqual(r["status"],"ORIGINAL_SAME_GAME_PUBLISHED_PRODUCTION")
        self.assertTrue(r["matched_eligible"])
        self.assertEqual(r["numeric_row_count"],1)
        self.assertEqual(r["rows"][0]["projected_median"],48.5)
        self.assertEqual(r["rows"][0]["model_source"],"v2_context")
        self.assertEqual(r["age_hours"],.5)
        self.assertFalse(r["same_lead_time_as_v7"])
        self.assertEqual(len(r["source_sha256"]),64)

    def test_source_timestamp_future_or_stale_not_used(self):
        j=document();j["generated_at_utc"]="2026-11-22T01:00:00Z"
        self.write(j)
        self.assertEqual(self.read()["status"],"PRODUCTION_SOURCE_AFTER_CAPTURE")
        j=document();j["generated_at_utc"]="2026-11-18T00:00:00Z"
        self.write(j)
        self.assertEqual(self.read()["status"],"STALE_PRODUCTION_SOURCE")

    def test_schedule_and_week_exact_or_no_comparison(self):
        j=document();j["week"]=12
        self.write(j)
        self.assertEqual(self.read()["status"],"PRODUCTION_WRONG_SEASON_WEEK")
        j=document();j["scheduled_games"][0]["away_team"]="NYJ"
        self.write(j)
        self.assertEqual(self.read()["status"],"PRODUCTION_SCHEDULE_MISMATCH")

    def test_duplicate_player_market_suppresses_entire_baseline(self):
        j=document();j["picks"].append(row())
        self.write(j)
        self.assertEqual(self.read()["status"],"DUPLICATE_PRODUCTION_PLAYER_MARKET")

    def test_only_true_numeric_median_scored(self):
        j=document()
        j["picks"].append(row(pid="x",median=None))
        j["picks"].append(row(pid="y",median=float("nan")))
        j["picks"].append(row(pid="z",median=-1))
        self.write(j)
        self.assertEqual(self.read()["numeric_row_count"],1)

    def test_no_file_or_captured_after_kickoff(self):
        self.source.unlink()
        self.assertEqual(self.read()["status"],"NO_PRODUCTION_SOURCE_FILE")
        with self.assertRaisesRegex(ValueError,"AFTER_KICKOFF"):
            capture_incumbent(self.source,GAME,observed_at_utc=GAME["kickoff_utc"])


if __name__=="__main__":
    unittest.main()
