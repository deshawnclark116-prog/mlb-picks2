"""NFL V2 independent completion-aware grading coverage regression tests."""
import json
import tempfile
import unittest
from datetime import datetime,timezone,timedelta
from pathlib import Path

from nfl_v2_postgame_grade_coverage import build_status,finals_from_scoreboards

NOW=datetime(2026,10,20,13,0,tzinfo=timezone.utc)
GAME="2026_06_WAS_SF"
EVENT={"id":"401873008","date":"2026-10-20T00:15Z",
       "status":{"type":{"completed":True}},"week":{"number":6},
       "season":{"year":2026,"type":2},
       "competitions":[{"competitors":[
          {"homeAway":"home","team":{"abbreviation":"SF"},"score":"23"},
          {"homeAway":"away","team":{"abbreviation":"WSH"},"score":"17"}]}]}
SCORECARD={"schema":"NFL_V2_FIRST_SEEN_REAL_FINAL_NUMERIC_SCORECARD_V1",
           "first_seen_pre_cutoff_only":True,"automatic_model_promotion":False,
           "asof_utc":"2026-10-20T12:40:00+00:00",
           "graded_player_outcomes":10,"all_graded_rows":[]}

class Coverage(unittest.TestCase):
    def finals(self):
        result=finals_from_scoreboards([("2026-10-19",{"events":[EVENT]})],NOW)
        self.assertIn(GAME,result)
        return result

    def test_independent_final_completion(self):
        self.assertEqual(len(self.finals()),1)
        earlier=NOW-timedelta(hours=14)
        self.assertEqual(len(finals_from_scoreboards(
            [("2026-10-19",{"events":[EVENT]})],earlier)),0)

    def test_blank_forecast_not_silently_reported_graded(self):
        with tempfile.TemporaryDirectory() as td:
            status=build_status(SCORECARD,Path(td),self.finals(),NOW)
            self.assertEqual(status["recent_frozen_game_horizons_graded"],0)
            self.assertEqual(status["recent_game_horizons_missing_original_forecast"],2)
            self.assertEqual(status["recent_frozen_game_horizons_ungraded"],0)

    def test_frozen_ungraded_is_visible(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"forecasts"/"batches"
            root.mkdir(parents=True)
            (root/(f"2026_wk06_T24_{GAME}.jsonl")).write_text("immutable")
            d=build_status(SCORECARD,Path(td),self.finals(),NOW)
            self.assertEqual(d["status"],"PENDING_SOURCE_OR_GRADER")
            self.assertEqual(d["recent_frozen_game_horizons_ungraded"],1)
            self.assertEqual(d["recent_game_horizons_missing_original_forecast"],1)
            self.assertEqual(d["not_yet_graded"][0]["horizon"],"T24")

    def test_scored_t24_and_not_yet_t90_do_not_mix(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"forecasts"/"batches";root.mkdir(parents=True)
            for horizon in ("T24","T90"):
                (root/f"2026_wk06_{horizon}_{GAME}.jsonl").write_text("immutable")
            score={**SCORECARD,"all_graded_rows":[{
                "game_id":GAME,"horizon":"T24",
            }]}
            status=build_status(score,Path(td),self.finals(),NOW)
            self.assertEqual(status["recent_frozen_game_horizons_graded"],1)
            self.assertEqual(status["recent_frozen_game_horizons_ungraded"],1)
            self.assertEqual(status["not_yet_graded"][0]["horizon"],"T90")

    def test_invalid_scorecard_never_appears_healthy(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                build_status({**SCORECARD,"first_seen_pre_cutoff_only":False},
                             Path(td),self.finals(),NOW)
            with self.assertRaises(ValueError):
                build_status({**SCORECARD,"schema":"legacy"},
                             Path(td),self.finals(),NOW)

    def test_unfinished_games_cannot_be_graded_as_final(self):
        not_done={**EVENT,"status":{"type":{"completed":False}}}
        self.assertFalse(finals_from_scoreboards(
            [("2026-10-19",{"events":[not_done]})],NOW))
        with self.assertRaises(ValueError):
            finals_from_scoreboards([("2026-10-19",{})],NOW)

if __name__=="__main__":
    unittest.main()
