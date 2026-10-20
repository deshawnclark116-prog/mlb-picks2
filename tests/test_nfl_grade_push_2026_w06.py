"""NFL grading: a half-sack equals the 0.5 prop line and must PUSH."""
import unittest
from nfl_grade_record_a import grade
class NflSackPushRegression(unittest.TestCase):
    def test_half_sack_equality_is_push_for_both_directions(self):
        self.assertEqual(grade({"pick":"OVER 0.5"},0.5),"push")
        self.assertEqual(grade({"pick":"UNDER 0.5"},0.5),"push")
    def test_strict_comparators_remain_unchanged(self):
        self.assertEqual(grade({"pick":"OVER 0.5"},1.0),"hit")
        self.assertEqual(grade({"pick":"OVER 0.5"},0.0),"miss")
        self.assertEqual(grade({"pick":"UNDER 0.5"},0.0),"hit")
        self.assertEqual(grade({"pick":"UNDER 0.5"},1.0),"miss")
        self.assertEqual(grade({"pick":"OVER 49.5"},50),"hit")
        self.assertEqual(grade({"pick":"UNDER 49.5"},49),"hit")
    def test_not_projections_or_bookless_targets(self):
        self.assertIsNone(grade({"pick":"Projected 66 yards"},66))
        self.assertIsNone(grade({"pick":"Scores a TD (OVER missing)"},1))
if __name__=="__main__":unittest.main()
