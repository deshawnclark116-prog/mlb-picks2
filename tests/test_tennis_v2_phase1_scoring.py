"""Exact-mechanics tests with synthetic point inputs only (no season data)."""
import math
import unittest

from tennis_v2_phase1_guards import EvidenceBlocked, FormatEvidence
from tennis_v2_phase1_scoring import (
    hold_probability, match_win_probability, tiebreak_win_probability,
)

BO3 = FormatEvidence(3, "TB7_AT_6_ALL_SETS")
BO5 = FormatEvidence(5, "TB7_AT_6_ALL_SETS")
BO3_TB10 = FormatEvidence(3, "TB10_AT_6_FINAL")
BO5_TB10 = FormatEvidence(5, "TB10_AT_6_FINAL")
BO3_12ALL = FormatEvidence(3, "TB7_AT_12_FINAL")
BO5_12ALL = FormatEvidence(5, "TB7_AT_12_FINAL")
BO3_ADV = FormatEvidence(3, "ADVANTAGE_FINAL_SET")
BO5_ADV = FormatEvidence(5, "ADVANTAGE_FINAL_SET")


class ScoringMechanicsTests(unittest.TestCase):
    def test_fair_point_means_fair_game(self):
        self.assertAlmostEqual(hold_probability(0.5), 0.5, places=12)

    def test_service_advantage_changes_game_win(self):
        self.assertGreater(hold_probability(.7), .9)
        self.assertLess(hold_probability(.3), .1)

    def test_hold_symmetry(self):
        for p in (.03,.23,.45,.6,.91):
            with self.subTest(p=p):
                self.assertAlmostEqual(hold_probability(p)+hold_probability(1-p),
                                       1.0, places=10)

    def test_tiebreak_fairness(self):
        for first in (1,2):
            for race in (7,10):
                self.assertAlmostEqual(
                    tiebreak_win_probability(.5,.5,first_server=first,points_to_win=race),
                    .5,places=12)

    def test_tiebreak_player_swap_symmetry(self):
        for first in (1,2):
            for race in (7,10):
                a = tiebreak_win_probability(.65,.72,first_server=first,points_to_win=race)
                b = tiebreak_win_probability(.72,.65,first_server=3-first,points_to_win=race)
                self.assertAlmostEqual(a+b,1.0,places=10)

    def test_match_player_swap_symmetry_on_all_formats(self):
        for fmt in (BO3,BO5,BO3_TB10,BO5_TB10,BO3_ADV,BO5_ADV,BO3_12ALL,BO5_12ALL):
            with self.subTest(fmt=fmt):
                a = match_win_probability(.67,.61,fmt=fmt)
                b = match_win_probability(.61,.67,fmt=fmt)
                self.assertAlmostEqual(a+b,1.0,places=9)

    def test_marginal_match_of_identical_players_is_fair(self):
        for fmt in (BO3,BO5,BO3_TB10,BO5_TB10,BO3_ADV,BO5_ADV,BO3_12ALL,BO5_12ALL):
            with self.subTest(fmt=fmt):
                self.assertAlmostEqual(match_win_probability(.68,.68,fmt=fmt),
                                       .5,places=9)

    def test_service_edge_predicts_higher_match_win(self):
        for fmt in (BO3,BO5,BO3_TB10,BO5_TB10,BO3_ADV,BO5_ADV,BO3_12ALL,BO5_12ALL):
            with self.subTest(fmt=fmt):
                self.assertGreater(match_win_probability(.73,.61,fmt=fmt),.5)
                self.assertLess(match_win_probability(.61,.73,fmt=fmt),.5)

    def test_stronger_serve_skill_monotonic(self):
        for fmt in (BO3,BO5,BO3_ADV,BO3_TB10,BO3_12ALL):
            with self.subTest(fmt=fmt):
                low = match_win_probability(.58,.64,fmt=fmt)
                high = match_win_probability(.70,.64,fmt=fmt)
                self.assertGreater(high,low)

    def test_realistic_probabilities_in_unit_interval(self):
        for p,q in ((.58,.52),(.75,.8),(.9,.47),(.22,.44)):
            for fmt in (BO3,BO5,BO3_TB10,BO3_12ALL):
                v = match_win_probability(p,q,fmt=fmt)
                self.assertTrue(0<=v<=1 and math.isfinite(v))

    def test_no_premature_bestof5_as_bestof3(self):
        a = match_win_probability(.74,.60,fmt=BO3)
        b = match_win_probability(.74,.60,fmt=BO5)
        self.assertNotAlmostEqual(a,b,places=6)

    def test_stated_first_server_average(self):
        a = match_win_probability(.71,.64,fmt=BO3,first_server=1)
        b = match_win_probability(.71,.64,fmt=BO3,first_server=2)
        both = match_win_probability(.71,.64,fmt=BO3)
        self.assertAlmostEqual(both,.5*(a+b),places=12)

    def test_wta_deciding_10pt_tiebreak_is_supported(self):
        self.assertTrue(0<match_win_probability(.6,.62,fmt=BO3_TB10)<1)

    def test_old_wimbledon_12_all_format_is_supported(self):
        self.assertTrue(0<match_win_probability(.7,.66,fmt=BO5_12ALL)<1)

    def test_unsupported_format_is_blocked(self):
        with self.assertRaisesRegex(EvidenceBlocked,"BLOCKED_FORMAT"):
            match_win_probability(.6,.6,fmt=FormatEvidence(3,"MATCH_TB10_THIRD"))

    def test_invalid_probability_fails_closed(self):
        for bad in (0.0,1.0,-.1,1.1,float("nan"),float("inf"),True):
            with self.subTest(bad=bad):
                with self.assertRaises(EvidenceBlocked):
                    match_win_probability(bad,.62,fmt=BO3)

    def test_tiebreak_bad_serve_state_fails_closed(self):
        with self.assertRaises(EvidenceBlocked):
            tiebreak_win_probability(.65,.7,first_server=3)


if __name__=="__main__":
    unittest.main()
