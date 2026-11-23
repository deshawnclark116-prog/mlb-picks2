import json, unittest
from pathlib import Path
import nfl_yardage_v2x as x

REPORT = json.loads((Path(__file__).resolve().parent.parent / "nfl_models" / "nfl_yardage_v2x_work" / "report.json").read_text())


class V2xTests(unittest.TestCase):
    def test_shipped_model_passed_its_gate_and_matches_features(self):
        self.assertTrue(REPORT["passed"])
        self.assertEqual(REPORT["features"], x.FEATURES)
        self.assertEqual(REPORT["train_start_season"], 2016)
        for m in x.MARKETS:
            r = REPORT["markets"][m]
            self.assertGreater(r["improvement_vs_naive_pct"], 0)
            self.assertLess(r["p_not_better_than_naive"], 0.05)
            self.assertGreater(r["n_train_final"], 5000)           # the point of v2x: many more training examples than the 2023+ model

    def test_extra_features_use_only_pregame_information(self):
        src = x.Sources()
        src.add_injuries([{"game_type": "REG", "gsis_id": "p1", "season": "2026", "week": "5", "report_status": "Questionable",
                           "practice_status": "Limited Participation in Practice"},
                          {"game_type": "REG", "gsis_id": "t1", "season": "2026", "week": "5", "report_status": "Out", "practice_status": ""}], 2026)
        src.add_snaps([{"game_type": "REG", "season": "2026", "week": "4", "player": "Pat One", "offense_pct": "0.8"}], 2026)
        cfg = x.MARKETS["receiving_yards"]
        regs = {"t1": (6.0, "WR"), "p1": (7.0, "WR"), "k1": (5.0, "QB")}
        e = x.extras(src, cfg, "p1", "Pat One", 2026, 5, regs, {"pat one": [0.5, 0.7, 0.9]})
        self.assertEqual(e["own_status"], 1)
        self.assertEqual(e["own_practice"], 1)
        self.assertEqual(e["tm_out_vol"], 6.0)                      # teammate listed Out; himself and other positions excluded
        self.assertEqual(e["tm_quest_vol"], 0.0)
        self.assertAlmostEqual(e["snap3"], (0.5 + 0.7 + 0.9) / 3)
        self.assertEqual(e["snap1"], 0.9)
        none = x.extras(src, cfg, "zz", "Nobody", 2026, 5, {}, {})
        self.assertEqual((none["own_status"], none["own_practice"]), (0, -1))   # unlisted players look like unlisted players in training
        self.assertIsNone(none["snap3"])

    def test_sources_must_cover_the_current_season(self):
        s = x.Sources()
        self.assertFalse(s.usable_for(2026))                         # no data -> builder falls back to v2
        s.years_inj.add(2026)
        s.years_snap.add(2025)
        self.assertTrue(s.usable_for(2026))


if __name__ == "__main__":
    unittest.main()
