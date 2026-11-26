"""Neutral-court invariance and development-only selection contract."""
from __future__ import annotations

from datetime import datetime
import unittest
import numpy as np
from sklearn.linear_model import Ridge

from cbb_neutral_challenger import (
    ELO_BLEND_WEIGHT, MODEL_CANDIDATES, combined_with_elo,
    neutral_swap, symmetric_predict, selection_from_2025,
    neutral_specialist, eval_neutral, challenger_audit,
)
from cbb_game_engine_v0 import FEATURE_NAMES, vector, profile
from collections import defaultdict


class DummyModel:
    def predict(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x=x[None, :]
        # Explicit fake home-side label bias should cancel exactly.
        return np.column_stack((x[:,0] + 5. + .2*x[:,4],
                                x[:,1] + 4. + .1*x[:,5]))


def fixture(i, year=2025, margin=3, total=143, neutral=True):
    x=np.array([i+2.,143.,float(neutral),68.,
                106.,102.,100.,104.,
                .03,-.04, .02,.03,
                -.01,.02,.015,-.017,
                8.,9.,.05,-.03,
                5.,3.],dtype=float)
    return {"game_id":str(10000*i+year), "season":year,
            "home_team_id":str(i), "away_team_id":"999",
            "neutral_site":neutral, "x":x, "y":np.array([margin,total]),
            "home_prior_games":8, "away_prior_games":9,
            "tip_utc":f"{year-1}-11-20T19:00:00+00:00"}


class NeutralChallengerTests(unittest.TestCase):
    def test_swap_has_correct_22_feature_involution(self):
        self.assertEqual(len(FEATURE_NAMES), 22)
        a=fixture(1)["x"]
        b=neutral_swap(a)
        np.testing.assert_allclose(neutral_swap(b),a)
        self.assertEqual(b[0],-a[0])
        self.assertEqual(b[1],a[1])
        self.assertEqual(b[4],a[5])
        self.assertEqual(b[18],-a[18])

    def test_reject_non_neutral_as_unsafe_swap(self):
        a=fixture(1)["x"]
        a[2]=0.
        with self.assertRaisesRegex(ValueError,"neutral-court"):
            neutral_swap(a)
        with self.assertRaises(ValueError):
            neutral_swap(a[:21])

    def test_score_predictions_invariant_when_neutral_labels_swap(self):
        model=DummyModel()
        x=np.stack([fixture(i)["x"] for i in range(1,5)])
        original=model.predict(x)
        neutral=symmetric_predict(model,x)
        swapped=symmetric_predict(model,neutral_swap(x))
        np.testing.assert_allclose(neutral[:,0],-swapped[:,0],atol=1e-12)
        np.testing.assert_allclose(neutral[:,1],swapped[:,1],atol=1e-12)
        self.assertFalse(np.allclose(original,neutral))  # remove label bias

    def test_fixed_elo_composition_remains_antisimmetric_in_margin(self):
        native=np.array([[5.,144.],[-8.,138.]])
        initial=combined_with_elo(native,np.array([.6,.25]),12.)
        flipped=combined_with_elo(native*np.array([-1.,1.]),
                                  np.array([.4,.75]),12.)
        np.testing.assert_allclose(initial[:,0],-flipped[:,0],atol=1e-9)
        np.testing.assert_allclose(initial[:,1],flipped[:,1],atol=1e-9)
        self.assertEqual(ELO_BLEND_WEIGHT,.50)
        with self.assertRaisesRegex(ValueError,"Elo probabilities"):
            combined_with_elo(native,np.array([2.,.5]),12.)

    def test_specialist_training_has_no_non_neutral_and_cannot_use_future(self):
        train=[fixture(i,year=2023,margin=i-15) for i in range(1,110)]
        extra=fixture(200,year=2023,margin=-500,neutral=False)
        a=neutral_specialist(train)
        b=neutral_specialist(train+[extra])
        x=np.stack([r["x"] for r in train[:4]])
        np.testing.assert_allclose(a.predict(x),b.predict(x))
        np.testing.assert_allclose(symmetric_predict(a,x)[:,0],
                                   -symmetric_predict(a,neutral_swap(x))[:,0])
        with self.assertRaisesRegex(ValueError,"insufficient neutral"):
            neutral_specialist(train[:2])

    def test_development_selection_ignores_2026_and_enforces_loss_guard(self):
        base={"n":500,"moneyline_brier":.240, "margin_mae":12.,
              "winner_accuracy":.59,"total_mae":17.}
        metrics={name:dict(base) for name in MODEL_CANDIDATES}
        metrics["swap_invariant"]["moneyline_brier"] = .232
        metrics["swap_invariant"]["margin_mae"]=12.1
        metrics["swap_plus_fixed_elo"]["moneyline_brier"] = .225
        metrics["swap_plus_fixed_elo"]["margin_mae"]=13.5 # reject margin damage
        selection=selection_from_2025(metrics)
        self.assertEqual(selection["chosen_using_2025_only"],"swap_invariant")
        metrics["swap_invariant"]["winner_accuracy"]=.50  # won't accept lower wins
        again=selection_from_2025(metrics)
        self.assertEqual(again["chosen_using_2025_only"],"incumbent")
        metrics["swap_plus_fixed_elo"]["margin_mae"]=12.2
        eligible=selection_from_2025(metrics)
        self.assertEqual(eligible["chosen_using_2025_only"],"swap_plus_fixed_elo")

    def test_eval_neutral_is_graded_with_actual_results(self):
        rows=[fixture(1,margin=-10,total=150), fixture(2,margin=10,total=135)]
        x=np.array([[+6.,150.],[-6.,135.]])
        stats=eval_neutral(rows,x,12.)
        self.assertEqual(stats["n"],2)
        self.assertEqual(stats["winner_accuracy"],0.)
        self.assertGreater(stats["moneyline_brier"],.3)

    def test_validation_2025_selection_is_diagnostic_only_2026(self):
        # Full 2023+2024 neutral training pool required by neutral_specialist.
        train=[fixture(i,year=2023,margin=i%21-10,total=145) for i in range(1,140)]
        val=[fixture(i,year=2025,margin=i%15-7,total=150) for i in range(141,310)]
        test=[fixture(i,year=2026,margin=i%18-9,total=140) for i in range(320,340)]
        full=train+val+test
        model=DummyModel()
        # DummyModel predicts shape, independent ridge is learned for specialist.
        elo={r["game_id"]:.52 for r in full}
        # Use independent injection in place of Elo calculation: monkey-patch.
        from unittest.mock import patch
        with patch("cbb_neutral_challenger.elo_predict_all",return_value=elo):
            report=challenger_audit(full,train,val,test,model,None,model,None,12.)
        self.assertIn("validation_2025",report)
        self.assertIn("previously_inspected_2026_diagnostic",report)
        self.assertFalse(report["promotion_allowed"])
        self.assertFalse(report["upcoming_production_prediction_changes"])
        self.assertTrue(report["2026_is_not_untouched_for_new_hypotheses"])
        self.assertIn(report["selection_decided_2025_only"]["chosen_using_2025_only"],MODEL_CANDIDATES)


if __name__=="__main__":
    unittest.main()
