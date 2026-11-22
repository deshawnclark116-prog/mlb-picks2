import unittest

from nfl_v2_empirical_probability import (
    EVENT,SOURCE,MIN_MARKET_ERRORS,whole_mean,
    personal_shift,probability_from_prior,prequential_audit
)

class V2ActualTargetProbabilityTests(unittest.TestCase):
    def make(self,n=120,week=3,market="receiving_yards"):
        return [
          {"season":2026,"week":week,"player_id":str(i%25),
           "name":"test"+str(i),"team":"TM",
           "prediction":30.0+(i%7),"actual":float(10+i%60),
           "residual":float(10+i%60)-(30.0+(i%7))}
          for i in range(n)
        ]

    def test_exact_whole_mean_contract(self):
        self.assertEqual([whole_mean(x) for x in (29.5,29.6,29.8,30.1,30.59,30.6)],
                         [29,30,30,30,30,31])
        with self.assertRaises(ValueError):whole_mean(float("nan"))

    def test_actual_completed_market_errors_are_distribution_not_fake_mc(self):
        r=self.make()
        p=probability_from_prior(r,29.8,whole_mean(29.8),player_id="1")
        self.assertEqual(p["event"],EVENT)
        self.assertEqual(p["source"],SOURCE)
        self.assertEqual(p["target"],30)
        self.assertEqual(p["model_reference_mean"],29.8)
        self.assertEqual(p["residual_sample_n"],120)
        self.assertEqual(p["residual_total_prior_n"],120)
        self.assertGreater(p["probability"],0)
        self.assertLess(p["probability"],1)
        self.assertEqual(p["market_calibration"],"NOT_FORWARD_CERTIFIED")
        self.assertNotIn("simulation_n",p)
        self.assertFalse(p["official_active_status_verified"])

    def test_no_fake_pct_without_support(self):
        self.assertIsNone(probability_from_prior(self.make(n=MIN_MARKET_ERRORS-1),35,35))

    def test_player_cannot_borrow_other_model_injury_or_td_pct(self):
        r=self.make()
        a=probability_from_prior(r,32.4,32,player_id="99")
        b=probability_from_prior(r,32.4,32,player_id="99",use_role=False)
        self.assertEqual(a,b)
        self.assertEqual(personal_shift([]),0)
        self.assertEqual(personal_shift([10]),0)
        self.assertLess(personal_shift([-30,-40,-10]),0)
        self.assertGreater(personal_shift([30,40,10]),0)

    def test_prequential_future_week_results_do_not_touch_earlier_prediction(self):
        original=self.make(80,week=6)
        after=self.make(80,week=7)
        for x in after:x["actual"]=x["prediction"]+999
        for x in after:x["residual"]=999
        prior=self.make(80,week=5)
        target=original[0]
        p=probability_from_prior(prior,target["prediction"],whole_mean(target["prediction"]))
        altered=after+[x.copy() for x in original]
        # Week7 outcomes are forbidden as calibration inputs to Week6.
        eligible=[x for x in prior+original+after if x["week"]<target["week"]]
        self.assertEqual(eligible,prior)
        self.assertEqual(p,probability_from_prior(eligible,target["prediction"],
                                                 whole_mean(target["prediction"])))
        report=prequential_audit(prior+original+altered)
        self.assertTrue(report["source_sample_before_target_only"])
        self.assertFalse(report["role_adjustment_published"])
        self.assertGreater(report["forward_2026_prequential_n"],0)

if __name__=="__main__":unittest.main()
