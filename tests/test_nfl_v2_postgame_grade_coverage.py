"""Coverage alarm independently verifies every archived forecast ID is graded or unverified."""
import hashlib
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
           "every_genuine_original_forecast_accounted":True,
           "asof_utc":"2026-10-20T12:40:00+00:00",
           "graded_player_outcomes":10,"all_graded_rows":[],
           "all_unverified_rows":[],"by_game_horizon":[]}

def archive(path,horizon,ids):
    path.parent.mkdir(parents=True,exist_ok=True)
    body=[json.dumps({"game_id":GAME,"horizon":horizon,"n":len(ids)})]
    body.extend(json.dumps({"id":i,"game_id":GAME,"horizon":horizon}) for i in ids)
    text="\n".join(body)+"\n"
    foot={"_footer":True,"n":len(ids),"sha256":hashlib.sha256(text.encode()).hexdigest()}
    path.write_text(text+json.dumps(foot)+"\n")

def score(graded_ids=(),unverified_ids=(),horizon="T24"):
    return {**SCORECARD,
      "all_graded_rows":[{"game_id":GAME,"horizon":horizon,
                          "forecast_id":i} for i in graded_ids],
      "all_unverified_rows":[{"game_id":GAME,"horizon":horizon,
                              "forecast_id":i} for i in unverified_ids],
      "by_game_horizon":[{
          "game_id":GAME,"horizon":horizon,
          "original_forecasts":len(graded_ids)+len(unverified_ids),
          "graded":len(graded_ids),"unverified":len(unverified_ids)}]}

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

    def test_missing_original_forecast_not_conflated_with_missing_grade(self):
        with tempfile.TemporaryDirectory() as td:
            status=build_status(SCORECARD,Path(td),self.finals(),NOW)
            self.assertEqual(status["recent_frozen_game_horizons_graded"],0)
            self.assertEqual(status["recent_game_horizons_missing_original_forecast"],2)
            self.assertEqual(status["recent_frozen_game_horizons_ungraded"],0)

    def test_complete_archived_game_missing_rows_fails_even_if_one_player_scored(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"forecasts"/"batches"
            archive(root/f"2026_wk06_T24_{GAME}.jsonl","T24",["id1","id2"])
            report=score(graded_ids=("id1",))
            d=build_status(report,Path(td),self.finals(),NOW)
            self.assertEqual(d["status"],"ORIGINAL_PREDICTIONS_MISSING_FROM_GRADING_REPORT")
            self.assertEqual(d["recent_frozen_game_horizons_ungraded"],1)
            self.assertEqual(d["not_yet_graded"][0]["expected_original_rows"],2)
            self.assertEqual(d["not_yet_graded"][0]["accounted_original_rows"],1)

    def test_every_forecast_accounted_one_actual_unknown_is_not_green_full_grade(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"forecasts"/"batches"
            archive(root/f"2026_wk06_T24_{GAME}.jsonl","T24",["id1","id2"])
            d=build_status(score(("id1",),("id2",)),Path(td),self.finals(),NOW)
            self.assertEqual(d["recent_frozen_game_horizons_graded"],1)
            self.assertEqual(d["recent_frozen_game_horizons_unverified"],1)
            self.assertEqual(d["recent_unverified_original_predictions"],1)
            self.assertEqual(d["recent_frozen_game_horizons_ungraded"],0)
            self.assertEqual(d["status"],"ALL_ORIGINAL_PREDICTIONS_ACCOUNTED_PARTIAL_ACTUAL_VERIFICATION")

    def test_complete_original_game_all_players_graded(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"forecasts"/"batches"
            archive(root/f"2026_wk06_T24_{GAME}.jsonl","T24",["id1","id2"])
            d=build_status(score(("id1","id2")),Path(td),self.finals(),NOW)
            self.assertEqual(d["recent_frozen_game_horizons_graded"],1)
            self.assertEqual(d["recent_frozen_game_horizons_unverified"],0)
            self.assertEqual(d["recent_frozen_game_horizons_ungraded"],0)

    def test_correct_count_wrong_ids_does_not_slip_through(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"forecasts"/"batches"
            archive(root/f"2026_wk06_T24_{GAME}.jsonl","T24",["id1"])
            d=build_status(score(("not_id1",)),Path(td),self.finals(),NOW)
            self.assertEqual(d["recent_frozen_game_horizons_ungraded"],1)

    def test_corrupt_original_footer_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"forecasts"/"batches";root.mkdir(parents=True)
            (root/f"2026_wk06_T24_{GAME}.jsonl").write_text("corrupt")
            d=build_status(SCORECARD,Path(td),self.finals(),NOW)
            self.assertEqual(d["not_yet_graded"][0]["reason"],
                             "ORIGINAL_FROZEN_BATCH_INVALID_OR_EMPTY")

    def test_invalid_scorecard_never_appears_healthy(self):
        with tempfile.TemporaryDirectory() as td:
            for override in ({"first_seen_pre_cutoff_only":False},
                             {"schema":"legacy"},
                             {"every_genuine_original_forecast_accounted":False}):
                with self.assertRaises(ValueError):
                    build_status({**SCORECARD,**override},Path(td),self.finals(),NOW)

    def test_unfinished_games_cannot_be_graded_as_final(self):
        not_done={**EVENT,"status":{"type":{"completed":False}}}
        self.assertFalse(finals_from_scoreboards(
            [("2026-10-19",{"events":[not_done]})],NOW))
        with self.assertRaises(ValueError):
            finals_from_scoreboards([("2026-10-19",{})],NOW)

if __name__=="__main__":
    unittest.main()
