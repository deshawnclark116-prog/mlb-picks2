"""Independent falsification tests; no feature or odds selection on the 2026 cohort."""
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from cbb_probability_stress import (
    audit_cohort, block_bootstrap_brier_difference, elo_predict_all,
    format_markdown, make_audit, normcdf, score_subset, wilson,
)


def row(game, date, home, away, margin, total, neutral=False, season=2025,
        home_prior_games=8, away_prior_games=7):
    return {
        "game_id": str(game),
        "tip_utc": date + "T20:00:00+00:00",
        "home_team_id": str(home), "away_team_id": str(away),
        "season": season, "neutral_site": neutral,
        "home_prior_games": home_prior_games, "away_prior_games": away_prior_games,
        "player_box_coverage": True,
        "y": np.array([margin, total], dtype=float),
    }


class StressTests(unittest.TestCase):
    def test_elo_same_day_results_cannot_leak_between_games(self):
        early = row(1,"2025-02-11","A","B",70,140)
        later = row(2,"2025-02-11","A","C",-13,120)
        initial = elo_predict_all([early,later])
        self.assertAlmostEqual(initial["1"], initial["2"], places=12)
        # Changing one final is not allowed to influence another game same day.
        mutated = deepcopy(early)
        mutated["y"] = np.array([-75., 140.])
        variant = elo_predict_all([mutated,later])
        self.assertEqual(initial["2"], variant["2"])
        self.assertEqual(initial["1"], variant["1"])
        next_day = row(3,"2025-02-12","A","D",2,130)
        a, b = elo_predict_all([early, later, next_day]), elo_predict_all([mutated, later, next_day])
        self.assertNotEqual(a["3"], b["3"])

    def test_elo_has_no_home_bias_at_neutral_site(self):
        regular = row(1,"2025-03-01","A","B",4,145)
        neutral = deepcopy(regular)
        neutral["neutral_site"] = True
        self.assertGreater(elo_predict_all([regular])["1"], .5)
        self.assertAlmostEqual(elo_predict_all([neutral])["1"], .5, places=12)

    def test_year_reset_does_not_use_next_season_results_for_previous(self):
        earlier = row(1,"2025-03-01","A","B",10,130,season=2025)
        later = row(2,"2026-03-01","A","B",-10,130,season=2026)
        old = elo_predict_all([earlier])["1"]
        self.assertEqual(elo_predict_all([earlier, later])["1"],old)

    def test_high_confidence_bins_are_observed_not_assumed(self):
        rows = [
            row(i, f"2025-01-{i+1:02d}", "A",str(i), 1 if i % 2 else -1,
                150, neutral=(i%3==0), home_prior_games=i%9, away_prior_games=i%9)
            for i in range(1,22)
        ]
        preds = np.column_stack((
            np.array([-22 if i%2 else 22 for i in range(1,22)], dtype=float),
            np.full(21,150.)
        ))
        elo = {str(r["game_id"]): .5 for r in rows}
        result = audit_cohort(rows,preds,elo,(10.,20.))
        self.assertEqual(result["overall"]["games"],21)
        self.assertTrue(result["reliability"][-1]["games"]>0)
        # Actual winner direction intentionally opposite to high-confidence direction.
        self.assertEqual(result["reliability"][-1]["observed_favorite_win_rate"], 0.)
        self.assertLess(result["reliability"][-1]["observed_minus_expected_pp"], -90.)
        self.assertEqual(sum(v["games"] for v in result["reliability"]),21)
        self.assertEqual(result["subgroups"]["both_teams_fewer_than_5_current_games"]["games"],13)
        self.assertLess(result["overall"]["moneyline_accuracy"], .1)

    def test_uncertainty_coverage_tests_reference_actual_score(self):
        rows=[row(1,"2025-02-11","A","B",0,140),
              row(2,"2025-02-12","A","C",50,200)]
        x=score_subset(rows,[.5,.95],[.5,.5],[0,0],[140,140],10.,15.)
        self.assertEqual(x["margin_nominal_80pct_interval_coverage"],.5)
        self.assertEqual(x["total_nominal_80pct_interval_coverage"],.5)
        self.assertEqual(x["games"],2)
        self.assertIsNotNone(x["moneyline_accuracy_wilson95"])

    def test_wilson_and_blocked_bootstrap_handle_small_sample(self):
        self.assertIsNone(wilson(0,0))
        self.assertIsNone(block_bootstrap_brier_difference([row(1,"2025-02-11","A","B",4,140)], [.6], [.5]))
        self.assertGreaterEqual(wilson(0,10)[0],0.)
        self.assertLessEqual(wilson(10,10)[1],1.)

    def test_reject_cohort_length_and_bad_prob_scale(self):
        rows=[row(1,"2025-02-11","A","B",4,140)]
        with self.assertRaisesRegex(ValueError,"length mismatch"):
            audit_cohort(rows,np.array([[3.,140.],[0.,130.]]),{"1":.5},(15,20))
        with self.assertRaisesRegex(ValueError,"bad uncertainty"):
            audit_cohort(rows,np.array([[3.,140.]]),{"1":.5},(0,20))
        with self.assertRaisesRegex(ValueError,"missing independent Elo"):
            audit_cohort(rows,np.array([[3.,140.]]),{},(15,20))

    def test_historical_2026_is_explicitly_diagnostic_not_new_holdout(self):
        train=[row(i,f"2025-02-{i+1:02d}","A",f"B{i}",4,140)
               for i in range(1,10)]
        test=[row(i+20,f"2026-02-{i+1:02d}","A",f"B{i}",-1,130,season=2026)
              for i in range(1,10)]
        vpred=np.tile([2.,140.],(9,1))
        hpred=np.tile([1.,130.],(9,1))
        report=make_audit(train+test,train,vpred,test,hpred,(12.,20.))
        md=format_markdown(report)
        self.assertIn("Previously inspected historical cohort",md)
        self.assertTrue(report["forward_prospective_accuracy_not_yet_measured"])
        self.assertIn("Elo",md)
        self.assertIn("Model stress by game type",md)
        self.assertIn("not independent confirmation",md.lower())


if __name__=="__main__":
    unittest.main()
