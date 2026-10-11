"""Minimal immutable-artifact integrity suite. No network or model fitting."""
import hashlib
import json
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKED = ROOT / "cfb_models" / "cfb_asof_shadow_locked_v1"
FIRST = ROOT / "docs" / "cfb_asof_shadow_moneyline_first_seen.jsonl"
EXPECTED_SHA = "a1c96a2fd066b612ab89be3f6817edcf42c46130adb7cbb090789cdbfe6c91f5"
CORE = ("research_manifest.json", "production.json",
        "cfb_moneyline_context.json", "cfb_moneyline_context_columns.json")


def compute_locked_model_hash():
    hashes = []
    for name in CORE:
        data = (LOCKED / name).read_bytes()
        hashes.append(hashlib.sha256(data).hexdigest())
    return hashlib.sha256("".join(hashes).encode("ascii")).hexdigest()


class ImmutableCFBResearchModel(unittest.TestCase):
    def test_model_weight_hash_is_the_prospective_first_run(self):
        lock = json.loads((LOCKED / "MODEL_LOCK.json").read_text())
        self.assertEqual(lock["moneyline_model_sha256"], EXPECTED_SHA)
        self.assertEqual(compute_locked_model_hash(), EXPECTED_SHA)
        self.assertTrue(lock["never_auto_retrain"])
        self.assertFalse(lock["official_predictions_changed"])
        self.assertTrue(lock["not_a_proven_winning_model"])

    def test_all_initial_research_predictions_genuinely_precede_kickoff(self):
        rows = [json.loads(line) for line in FIRST.read_text().splitlines()
                if line.strip()]
        self.assertGreaterEqual(len(rows), 51)
        keys = set()
        for row in rows:
            key = (row["season"], row["week"], str(row["game_id"]))
            self.assertNotIn(key, keys)
            keys.add(key)
            self.assertEqual(row["frozen_challenger_model_sha256"], EXPECTED_SHA)
            self.assertEqual(row["schema"], "CFB_ASOF_MONEYLINE_FIRST_PREGAME_SHADOW_V1")
            self.assertFalse(row["official_prediction"])
            self.assertFalse(row["betting_edge_certified"])
            observed = datetime.fromisoformat(row["first_challenger_logged_at_utc"])
            earlier = datetime.fromisoformat(row["first_incumbent_logged_at_utc"])
            kickoff = datetime.fromisoformat(row["kickoff_utc"])
            self.assertLess(observed, kickoff)
            self.assertLess(earlier, kickoff)
            self.assertEqual(row["challenger_changed_side"],
                             row["challenger_selected_team"] != row["incumbent_selected_team"])
            self.assertAlmostEqual(row["challenger_home_final_probability"]
                                   + row["challenger_away_final_probability"], 1, places=6)

    def test_research_ratings_source_cannot_be_shortened_to_main_db(self):
        workflow = (ROOT / ".github/workflows/cfb_asof_forward_nightly.yml").read_text()
        self.assertIn("--season 2019 2020 2021 2022 2023 2024 2025", workflow)
        self.assertIn("set(range(2019,2027)) <= seasons", workflow)
        self.assertIn("github.ref == 'refs/heads/main'", workflow)
        self.assertIn("--model-dir cfb_models/cfb_asof_shadow_locked_v1", workflow)
        self.assertIn("--shadow-ledger docs/cfb_asof_shadow_moneyline_first_seen.jsonl", workflow)
        self.assertIn("--original-ledger docs/cfb_picks_log.jsonl", workflow)


if __name__ == "__main__":
    unittest.main()
