"""CBB neutral challenger first-seen receipt and grading integrity."""
import json
import tempfile
import unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
import pandas as pd

from cbb_neutral_forward import freeze,grade_one,grade_frozen

NOW=datetime(2026,11,26,5,10,tzinfo=timezone.utc)
TIP=(NOW+timedelta(hours=7)).isoformat()


def sample():
    id="401900900"
    row={"game_id":id,"tip_utc":TIP,"home":"Sample Home","away":"Sample Away",
         "neutral_site":True,
         "incumbent_home_margin":9.,"incumbent_total":148.,
         "swap_home_margin":5.,"swap_total":147.,
         "specialist_home_margin":6.,"specialist_total":146.,
         "elo_blend_home_margin":4.5,"elo_blend_total":147.,
         "fixed_elo_home_win":.60,"model_uncertainty_sd":13.}
    audit={"schema":"CBB_NEUTRAL_COURT_CHALLENGER_V1",
           "selection_decided_2025_only":{"chosen_using_2025_only":"swap_plus_fixed_elo"},
           "upcoming_neutral_research_shadow":[row]}
    forecast={"schema":"CBB_GAME_INTELLIGENCE_V0_RESEARCH",
              "status":"RESEARCH_ONLY_NO_PRETIP_FIRST_SEEN_ARCHIVE",
              "generated_utc":(NOW-timedelta(minutes=2)).isoformat(),
              "game_predictions":[{"game_id":id,"tip_utc":TIP,
                                   "neutral_site":True,
                                   "home_team_id":"9","away_team_id":"7",
                                   "home":"Sample Home","away":"Sample Away"}]}
    return audit,forecast


class FreezeGrade(unittest.TestCase):
    def test_durable_original_and_replay_preserves_bytes(self):
        a,f=sample()
        with tempfile.TemporaryDirectory() as root:
            s=freeze(a,f,root,NOW.isoformat())
            self.assertEqual(s["new"],1)
            path=next(Path(root).rglob("*.json"))
            before=path.read_bytes()
            a["upcoming_neutral_research_shadow"][0]["elo_blend_home_margin"]=99.
            replay=freeze(a,f,root,(NOW+timedelta(minutes=3)).isoformat())
            self.assertEqual(replay["immutable_existing"],1)
            self.assertEqual(before,path.read_bytes())

    def test_cannot_save_future_scores_as_first_seen_after_tip(self):
        a,f=sample()
        with tempfile.TemporaryDirectory() as root:
            result=freeze(a,f,root,(NOW+timedelta(hours=8)).isoformat())
            self.assertEqual(result["new"],0)
            self.assertEqual(result["excluded_already_started"],1)
            self.assertEqual(list(Path(root).rglob("*.json")),[])

    def test_rejects_invalid_chosen_expert(self):
        a,f=sample()
        a["selection_decided_2025_only"]["chosen_using_2025_only"]="invented"
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,"invalid selected"):
                freeze(a,f,root,NOW.isoformat())

    def test_all_four_methods_grading_uses_only_real_same_id_finals(self):
        a,f=sample()
        with tempfile.TemporaryDirectory() as root:
            freeze(a,f,root,NOW.isoformat())
            rec=json.loads(next(Path(root).rglob("*.json")).read_text())
            self.assertEqual(set(rec["four_model_predictions"]),{
                "incumbent","swap_invariant","neutral_specialist","swap_plus_fixed_elo"
            })
            self.assertEqual(grade_one(rec,{"completed":False})["status"],"PENDING")
            grade=grade_one(rec,{"completed":True,"game_id":rec["game_id"],
                                "home_team_id":"9","away_team_id":"7",
                                "home_score":73,"away_score":68})
            self.assertEqual(grade["status"],"GRADED")
            self.assertTrue(grade["per_method"]["swap_plus_fixed_elo"]["winner_correct"])
            self.assertFalse(grade["no_real_market_odds_captured"] is False)
            self.assertEqual(
                grade_one(rec,{"completed":True,"game_id":"wrong",
                               "home_team_id":"9","away_team_id":"7",
                               "home_score":73,"away_score":68})["status"],"IDENTITY_MISMATCH")

    def test_official_result_grader_does_not_overwrite(self):
        a,f=sample()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            originals=root/"originals";grades=root/"grades"
            freeze(a,f,originals,NOW.isoformat())
            p=root/"schedule.csv"
            pd.DataFrame([{
                "game_id":401900900,"home_id":9,"away_id":7,
                "home_score":73,"away_score":68,
                "status_type_completed":True
            }]).to_csv(p,index=False)
            result=grade_frozen(originals,grades,str(p))
            self.assertEqual(result["new_grades"],1)
            later=grade_frozen(originals,grades,str(p))
            self.assertEqual(later["new_grades"],0)
            self.assertEqual(later["previously_graded"],1)


if __name__=="__main__":
    unittest.main()
