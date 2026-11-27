"""Grade-only CBB neutral forward audit: no retrospective rewrite, no odds claim."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cbb_neutral_grade_only import grade_one,grade_state


def original():
    return {"schema":"CBB_NEUTRAL_FORWARD_ORIGINAL_V1",
            "game_id":"401900123","home_team_id":"5","away_team_id":"9",
            "first_seen_capture_utc":"2026-11-26T12:00:00+00:00",
            "event_tip_utc":"2026-11-26T14:00:00+00:00",
            "sha256":"example_sha_immutable",
            "selected_with_2025_development_only":"swap_plus_fixed_elo",
            "four_model_predictions":{
                "incumbent":{"home_pwin":.7,"margin":9.,"total":146.},
                "swap_plus_fixed_elo":{"home_pwin":.6,"margin":4.,"total":145.}
            }}


class GradeOnlyTests(unittest.TestCase):
    def test_grades_only_matching_original_source_and_final(self):
        r=original()
        official={"completed":True,"game_id":"401900123","home_team_id":"5",
                  "away_team_id":"9","home_score":73,"away_score":70}
        graded=grade_one(r,official)
        self.assertEqual(graded["status"],"GRADED")
        self.assertEqual(graded["model_grades"]["incumbent"]["margin_mae"],6.)
        self.assertEqual(graded["model_grades"]["swap_plus_fixed_elo"]["margin_mae"],1.)
        self.assertFalse(graded["sportsbook_roi_calculated"])
        invalid=dict(official,away_team_id="wrong")
        self.assertEqual(grade_one(r,invalid)["status"],"IDENTITY_MISMATCH")
        self.assertEqual(grade_one(r,{"completed":False})["status"],"PENDING")

    def test_original_cannot_be_overwritten_or_pretip_rewritten(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/"originals"/"2026-11-26"/"401900123.json"
            src.parent.mkdir(parents=True)
            src.write_text(json.dumps(original()))
            schedule=Path(d)/"schedule.csv"
            pd.DataFrame([{"game_id":401900123,"home_id":5,"away_id":9,
                           "home_score":73,"away_score":70,
                           "status_type_completed":True}]).to_csv(schedule,index=False)
            first=grade_state(str(src.parents[1]),str(Path(d)/"grades"),str(schedule))
            self.assertEqual(first["new_grades"],1)
            self.assertEqual(first["read"],1)
            second=grade_state(str(src.parents[1]),str(Path(d)/"grades"),str(schedule))
            self.assertEqual(second["new_grades"],0)
            self.assertEqual(second["already_graded"],1)
            edited=original()
            edited["first_seen_capture_utc"]="2026-11-26T15:00:00+00:00"
            src.write_text(json.dumps(edited))
            with self.assertRaisesRegex(ValueError,"NON_PRETIP"):
                grade_state(str(src.parents[1]),str(Path(d)/"grades"),str(schedule))

    def test_unavailable_results_cannot_be_synthesized(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"originals"/"2026-11-26"/"401900123.json"
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps(original()))
            schedule=Path(d)/"schedule.csv"
            pd.DataFrame([{"game_id":401900123,"home_id":5,"away_id":9,
                           "home_score":None,"away_score":None,
                           "status_type_completed":False}]).to_csv(schedule,index=False)
            stats=grade_state(str(p.parents[1]),str(Path(d)/"grades"),str(schedule))
            self.assertEqual(stats["pending"],1)
            self.assertEqual(stats["new_grades"],0)


if __name__=="__main__":
    unittest.main()
