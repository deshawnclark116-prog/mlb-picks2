"""Test uncertainty challenger trained on older data, no look-ahead or hidden promotion."""
import unittest
import numpy as np
from cbb_uncertainty_diagnostic import GROUPS, group, learn_scales, gaussian_nll, experiment, to_markdown


def row(gid,n_home,n_away,margin=0.,total=140.):
    return {"game_id":str(gid), "home_prior_games":n_home, "away_prior_games":n_away,
            "y":np.array([margin,total],dtype=float)}


class ConditionalUncertaintyTests(unittest.TestCase):
    def test_bucket_logic_uses_only_prior_game_count(self):
        self.assertEqual(group(row(1,0,18)),GROUPS[0])
        self.assertEqual(group(row(1,3,4)),GROUPS[1])
        self.assertEqual(group(row(1,5,1)),GROUPS[2])
        with self.assertRaises(ValueError):
            group(row(1,-1,1))

    def test_scale_estimates_respond_to_observed_training_residuals_only(self):
        train=[
            row(1,0,1,50,200),row(2,0,2,-50,80),
            row(3,2,2,2,141),row(4,3,3,-2,139),
            row(5,10,10,6,140),row(6,7,8,-6,140),
        ]
        p=np.tile([0.,140.],(6,1))
        scales=learn_scales(train,p,(12.,19.))
        self.assertGreater(scales[GROUPS[0]]["sd_margin"],scales[GROUPS[1]]["sd_margin"])
        self.assertEqual(sum(v["training_n_2025"] for v in scales.values()),6)
        self.assertGreater(scales[GROUPS[2]]["sd_total"],18.5)
        self.assertLess(scales[GROUPS[2]]["sd_total"],19.)

    def test_no_holdout_leakage_into_candidate_scales(self):
        train=[
            row(1,0,1,50,200),row(2,0,2,-50,80),
            row(3,2,2,2,141),row(4,3,3,-2,139),
            row(5,10,10,6,140),row(6,7,8,-6,140),
        ]
        p=np.tile([0.,140.],(6,1))
        future=[row(7,0,3,200,300),row(8,6,7,-200,10)]
        h=np.tile([0.,140.],(2,1))
        initial=experiment(train,p,future,h,(12.,19.))
        future[0]["y"]=np.array([1000., 1000.])
        changed=experiment(train,p,future,h,(12.,19.))
        self.assertEqual(initial["scales_learned_using_2025_only"],changed["scales_learned_using_2025_only"])
        self.assertNotEqual(initial["previously_examined_2026_diagnostic"],
                            changed["previously_examined_2026_diagnostic"])
        self.assertFalse(initial["promotion_allowed"])
        self.assertFalse(initial["uncertainty_changes_live_predictions"])

    def test_reporting_and_coverage_scores_are_finite(self):
        train=[row(1,0,1,30,180),row(2,2,2,1,141),
               row(3,7,8,0,140)]
        pred=np.tile([0.,140.],(3,1))
        res=experiment(train,pred,train,pred,(12.,19.))
        md=to_markdown(res)
        self.assertEqual(res["schema"], "CBB_UNCERTAINTY_STRATIFICATION_EXPERIMENT_V1")
        self.assertIn("Previously inspected 2026",md)
        self.assertIn("not promoted",md)
        self.assertEqual(res["previously_examined_2026_diagnostic"]["overall"]["games"],3)
        self.assertTrue(np.isfinite(res["previously_examined_2026_diagnostic"]["overall"]["candidate_joint_nll_mean"]))

    def test_validation_refuses_mismatched_shapes_or_bad_sd(self):
        with self.assertRaisesRegex(ValueError,"mismatch"):
            learn_scales([row(1,0,2)],np.zeros((2,2)),(10,20))
        with self.assertRaisesRegex(ValueError,"invalid pooled"):
            learn_scales([row(1,0,2)],np.zeros((1,2)),(0,20))
        with self.assertRaisesRegex(ValueError,"invalid normal"):
            gaussian_nll(2.,0.)


if __name__=="__main__":
    unittest.main()
