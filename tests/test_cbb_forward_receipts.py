"""CBB research-only first-seen receipt and verified-market grading safety tests."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from cbb_forward_receipts import (
    american_to_implied, first_seen_receipts, grade_frozen_folder,
    grade_receipt, market_quote, probs_with_market, write_immutable,
)

NOW = datetime(2026, 11, 25, 16, 0, tzinfo=timezone.utc)
TIP = (NOW + timedelta(hours=10)).isoformat()


def prediction():
    return {
        "game_id": "401900111", "tip_utc": TIP,
        "home": "Duke Blue Devils", "away": "Connecticut Huskies",
        "home_team_id": "150", "away_team_id": "41",
        "home_prior_games": 6, "away_prior_games": 5,
        "projected_home_margin": 3.7, "projected_total": 149.8,
        "projected_home_score": 76.8, "projected_away_score": 73.0,
        "p_home_win": 0.618, "p_away_win": .382
    }


def forecast():
    return {
        "schema": "CBB_GAME_INTELLIGENCE_V0_RESEARCH",
        "status": "RESEARCH_ONLY_NO_PRETIP_FIRST_SEEN_ARCHIVE",
        "generated_utc": (NOW - timedelta(minutes=1)).isoformat(),
        "report": {"residual_margin_sd_from_2025": 12.,
                   "residual_total_sd_from_2025": 18.},
        "game_predictions": [prediction()],
    }


def quote():
    return [{
        "id": "abcodds1",
        "home_team": "Duke Blue Devils", "away_team": "Connecticut Huskies",
        "commence_time": TIP,
        "bookmakers": [{
            "key": "fanduel", "last_update": (NOW - timedelta(minutes=1)).isoformat(),
            "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Duke Blue Devils", "price": -120},
                    {"name": "Connecticut Huskies", "price": +105},
                ]},
                {"key": "spreads", "outcomes": [
                    {"name": "Duke Blue Devils", "price": -110, "point": -2.5},
                    {"name": "Connecticut Huskies", "price": -110, "point": +2.5},
                ]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "price": -112, "point": 146.5},
                    {"name": "Under", "price": -108, "point": 146.5},
                ]},
            ]
        }]
    }]


class CbbReceiptTests(unittest.TestCase):
    def test_no_odds_never_invents_spread_or_total_probabilities(self):
        rec = first_seen_receipts(forecast(), [], NOW, "T24")
        self.assertEqual(len(rec), 1)
        self.assertIsNone(rec[0]["market"])
        self.assertIsNone(rec[0]["probabilities"]["home_cover"])
        self.assertIsNone(rec[0]["probabilities"]["over"])
        self.assertTrue(rec[0]["no_market_value_claim_if_quote_missing"])

    def test_strict_actual_odds_and_no_vig_probabilities(self):
        market = market_quote(prediction(), quote(), NOW)
        self.assertEqual(market["home_spread"], -2.5)
        self.assertEqual(market["total_line"], 146.5)
        self.assertEqual(market["home_ml_american"], -120)
        probs = probs_with_market(prediction(), market, 12, 18)
        self.assertAlmostEqual(probs["home_cover"] + probs["away_cover"], 1., places=5)
        self.assertGreater(probs["over"], .5)
        self.assertTrue(.50 < probs["home_win_no_vig_market"] < .57)
        self.assertAlmostEqual(american_to_implied(-110), 110 / 210, places=6)

    def test_wrong_opponent_tip_ambiguous_or_future_quote_rejected(self):
        other = quote()
        other[0]["away_team"] = "Kansas Jayhawks"
        self.assertIsNone(market_quote(prediction(), other, NOW))
        other = quote()
        other[0]["commence_time"] = (NOW + timedelta(hours=11)).isoformat()
        self.assertIsNone(market_quote(prediction(), other, NOW))
        other = quote()
        self.assertIsNone(market_quote(prediction(), other + other, NOW))
        other = quote()
        other[0]["bookmakers"][0]["last_update"] = (NOW + timedelta(hours=1)).isoformat()
        self.assertIsNone(market_quote(prediction(), other, NOW))

    def test_source_staleness_and_broken_market_no_fabricated_values(self):
        other = quote()
        other[0]["bookmakers"][0]["last_update"] = (NOW - timedelta(hours=7)).isoformat()
        self.assertIsNone(market_quote(prediction(), other, NOW))
        other = quote()
        other[0]["bookmakers"][0]["markets"][1]["outcomes"][1]["point"] = 3.5
        market = market_quote(prediction(), other, NOW)
        self.assertIsNone(market["home_spread"])
        self.assertIsNotNone(market["total_line"])

    def test_immutable_first_seen_and_identity_conflicts(self):
        rows = first_seen_receipts(forecast(), quote(), NOW, "T24")
        self.assertEqual(len(rows), 1)
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(write_immutable(temp, rows)["new"], 1)
            p = next(Path(temp).rglob("*.json"))
            saved = p.read_bytes()
            newer = json.loads(json.dumps(rows))
            newer[0]["projected_total"] = 205  # later model revision
            self.assertEqual(write_immutable(temp, newer)["unchanged_existing"], 1)
            self.assertEqual(p.read_bytes(), saved)
            newer[0]["home_team_id"] = "999"  # same game id, different team
            with self.assertRaisesRegex(ValueError, "IDENTITY_CONFLICT"):
                write_immutable(temp, newer)
            self.assertEqual(p.read_bytes(), saved)

    def test_t24_t90_do_not_capture_after_tip(self):
        self.assertEqual(len(first_seen_receipts(forecast(), quote(), NOW, "T24")), 1)
        self.assertEqual(len(first_seen_receipts(forecast(), quote(), NOW, "T90")), 0)
        after = NOW + timedelta(hours=11)
        # Historical model data cannot be relabeled as first seen after tip.
        self.assertEqual(first_seen_receipts(forecast(), quote(), after, "T24"), [])
        self.assertEqual(first_seen_receipts(forecast(), quote(), after, "T90"), [])

    def test_grading_is_real_and_push_is_neutral(self):
        row = first_seen_receipts(forecast(), quote(), NOW, "T24")[0]
        final = {"completed": True, "game_id": row["game_id"],
                 "home_team_id": row["home_team_id"], "away_team_id": row["away_team_id"],
                 "home_score": 75., "away_score": 72.5}
        # 72.5 isn't a real basketball score; test a push at exactly 146.5 instead.
        final["home_score"] = 80
        final["away_score"] = 66.5
        grade = grade_receipt(row, final)
        self.assertEqual(grade["status"], "GRADED")
        self.assertTrue(grade["markets"]["over"]["push"])
        self.assertEqual(grade["markets"]["over"]["realized_roi_per_unit"], 0)
        self.assertTrue(grade["markets"]["home_spread"]["push"])  # home wins exactly 2.5
        self.assertEqual(grade["markets"]["home_spread"]["realized_roi_per_unit"], 0)
        final["away_score"] = 71
        grade2 = grade_receipt(row, final)
        self.assertTrue(grade2["markets"]["over"]["won"])
        self.assertTrue(grade2["markets"]["home_spread"]["won"])
        self.assertEqual(grade2["markets"]["home_spread"]["realized_roi_per_unit"], 100 / 110)

    def test_invalid_identity_and_missing_final_never_grades(self):
        row = first_seen_receipts(forecast(), quote(), NOW, "T24")[0]
        self.assertEqual(grade_receipt(row, {"completed": False})["status"], "PENDING")
        self.assertEqual(grade_receipt(row, {"completed": True, "game_id": "wrong",
            "home_team_id": "150", "away_team_id": "41"})["status"], "IDENTITY_MISMATCH")

    def test_completed_schedule_grader_is_idempotent(self):
        row = first_seen_receipts(forecast(), quote(), NOW, "T24")[0]
        with tempfile.TemporaryDirectory() as tmp:
            receipts, grades = Path(tmp) / "receipts", Path(tmp) / "grades"
            write_immutable(receipts, [row])
            s = pd.DataFrame([{
                "game_id": int(row["game_id"]), "home_id": 150, "away_id": 41,
                "home_score": 80, "away_score": 71, "status_type_completed": True
            }])
            f = str(Path(tmp) / "schedule.parquet")
            s.to_parquet(f)
            first = grade_frozen_folder(str(receipts), f, str(grades))
            self.assertEqual(first["new_grades"], 1)
            second = grade_frozen_folder(str(receipts), f, str(grades))
            self.assertEqual(second["new_grades"], 0)
            self.assertEqual(second["already_graded"], 1)


if __name__ == "__main__":
    unittest.main()
