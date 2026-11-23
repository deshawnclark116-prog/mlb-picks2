import json, unittest
from pathlib import Path
import nfl_yardage_floor_table as t

TABLE = json.loads((Path(__file__).resolve().parent.parent / "nfl_models" / "nfl_yardage_floor_table.json").read_text())


class FloorTableTests(unittest.TestCase):
    def test_no_sportsbook_line_and_target(self):
        self.assertFalse(TABLE["sportsbook_line_used"])
        for m in ("rushing_yards", "receiving_yards"):
            for b in TABLE["markets"][m]["bands"]:
                self.assertGreaterEqual(b["n"], t.MIN_BAND_N)
                self.assertTrue(0.70 <= b["safe_probability"] <= 0.85)          # near the 75% target, never inflated
                self.assertTrue(0.40 <= b["reach_projection"] <= 0.60)          # a median projection is reached about half the time

    def test_lookup_floor_is_below_projection(self):
        for m in ("rushing_yards", "receiving_yards"):
            for proj in (12.0, 28.0, 44.0, 60.0, 85.0):
                reach, safe, prob = t.lookup(TABLE, m, proj)
                self.assertLess(safe, proj)
                self.assertGreaterEqual(safe, 0)

    def test_whole_rounding_rule(self):
        self.assertEqual(t.whole(23.59), 23)
        self.assertEqual(t.whole(23.6), 24)
        self.assertIsNone(t.lookup({"markets": {}}, "rushing_yards", 30.0))


if __name__ == "__main__":
    unittest.main()
