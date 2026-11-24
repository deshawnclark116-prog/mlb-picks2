"""Behavioral contracts for player/game-specific, pregame source-grounded scouting."""
import copy
import unittest
from nfl_game_scout import build, evidence_match, render_markdown

NOW = "2026-11-24T03:15:00Z"
KICK = "2026-11-26T01:00:00Z"
STAMP = "2026-11-24T03:04:57+00:00"


def fixture():
    board = {"season": 2026, "week": 12, "generated_at_utc": STAMP,
             "scheduled_games": [{"home_team": "LA", "away_team": "GB", "kickoff_utc": KICK}],
             "picks": [
                 {"player": "Example Receiver", "player_id": "WR1", "team": "LA", "opponent": "GB",
                  "market": "receiving_yards", "projected_median": 62, "kickoff_utc": KICK,
                  "confidence": "HIGH", "projection_probability_source": "UNAVAILABLE_NO_MATCHED_DISTRIBUTION"},
                 {"player": "Example Back", "player_id": "RB1", "team": "LA", "opponent": "GB",
                  "market": "rushing_yards", "projected_median": 72, "kickoff_utc": KICK},
             ]}
    role = {"schema": "nfl-role-evidence-coverage-v1", "season": 2026, "week": 12,
            "forecast_generated_at_utc": STAMP,
            "captured_at_utc": "2026-11-24T03:04:57.9Z",
            "player_market_evidence": [
                {"player": "Example Receiver", "player_id": "WR1", "team": "LA", "opponent": "GB",
                 "market": "receiving_yards", "kickoff_utc": KICK,
                 "observed_stat_rows": 3, "role_share_volume_weighted": .42,
                 "recent_team_game_observations": [
                     {"season": 2026, "week": 9, "stat_row_observed": True, "share_when_observed": .20},
                     {"season": 2026, "week": 10, "stat_row_observed": True, "share_when_observed": .32},
                     {"season": 2026, "week": 11, "stat_row_observed": True, "share_when_observed": .48},
                 ], "warnings": [], "roster_status": "ROSTER_LISTED_NOT_STARTER_VERIFIED",
                 "has_verified_starting_role": False},
                {"player": "Example Teammate", "player_id": "WR2", "team": "LA", "opponent": "GB",
                 "market": "receiving_yards", "kickoff_utc": KICK,
                 "observed_stat_rows": 3, "role_share_volume_weighted": .38,
                 "recent_team_game_observations": [], "warnings": []},
                {"player": "Example Back", "player_id": "RB1", "team": "LA", "opponent": "GB",
                 "market": "rushing_yards", "kickoff_utc": KICK,
                 "observed_stat_rows": 3, "role_share_volume_weighted": .61,
                 "recent_team_game_observations": [], "warnings": []},
            ]}
    injuries = {"captured_at_utc": "2026-11-24T00:48:41Z",
                "players": [{"player": "Opposing CB", "team": "GB", "injury_status": "questionable"}]}
    plays = []
    for w in (9, 10, 11):
        for i in range(5):
            plays.append({"season": 2026, "week": w, "season_type": "REG", "game_id": f"2026_{w}_LA_GB",
                          "posteam": "LA", "defteam": "GB", "play_type": "pass",
                          "receiver_player_id": "WR1", "air_yards": 24 if i < 3 else 7,
                          "yards_gained": 18 if i < 3 else 8})
        for i in range(5):
            plays.append({"season": 2026, "week": w, "season_type": "REG", "game_id": f"2026_{w}_LA_GB",
                          "posteam": "LA", "defteam": "GB", "play_type": "run",
                          "rusher_player_id": "RB1", "run_gap": "guard" if i < 3 else "end",
                          "yards_gained": 4 if i < 3 else 7})
    snaps = []
    for w in (9, 10, 11):
        snaps.append({"season": 2026, "week": w, "game_type": "REG", "team": "GB",
                      "player": "Opposing CB", "position": "CB", "defense_pct": .95})
        snaps.append({"season": 2026, "week": w, "game_type": "REG", "team": "LA",
                      "player": "Example Guard", "position": "G", "offense_pct": .98})
    return board, role, plays, snaps, injuries


class ScoutTests(unittest.TestCase):
    def test_real_mechanism_counts_named_personnel_and_opportunity_competition(self):
        args = fixture()
        frozen = copy.deepcopy(args[0])
        out = build(*args, now=NOW)
        self.assertEqual(out["summary"]["scouted_player_markets"], 2)
        cards = {c["market"]: c for c in out["player_cards"]}
        wr = cards["receiving_yards"]
        rb = cards["rushing_yards"]
        self.assertEqual(wr["published_projection_yards"], 62)
        self.assertEqual(wr["forecast_change_yards"], 0)
        self.assertEqual(wr["player_prior_play_profile"]["observed_opportunities"], 15)
        self.assertEqual(wr["opponent_prior_defensive_play_profile"]["games"], 3)
        self.assertGreater(wr["observed_role"]["recent_share_direction"]["change_percentage_points"], 20)
        self.assertEqual(wr["teammate_opportunity_competition"][0]["player"], "Example Teammate")
        self.assertIn("deep_20plus", [m["mechanism"] for m in wr["matched_play_mechanisms"]])
        self.assertEqual(wr["likely_defensive_personnel_from_previous_snaps"][0]["player"], "Opposing CB")
        self.assertEqual(wr["likely_defensive_personnel_from_previous_snaps"][0]["injury_source_status"], "questionable")
        self.assertFalse(wr["likely_defensive_personnel_from_previous_snaps"][0]["assignment_verified"])
        self.assertEqual(rb["likely_own_offensive_line_from_previous_snaps"][0]["player"], "Example Guard")
        self.assertTrue(wr["unverified_player_vs_corner_assignment"])
        self.assertTrue(rb["unverified_ol_vs_dl_blocking_assignment"])
        self.assertEqual(args[0], frozen, "no live forecast mutation")
        self.assertIn("PUBLISHED_HIGH_CONFIDENCE_WITHOUT_PLAYER_DISTRIBUTION", wr["data_warnings"])

    def test_named_defender_charting_attached_without_assuming_coverage_assignment(self):
        args = list(fixture())
        for sn in args[3]:
            if sn.get("player") == "Opposing CB":
                sn["pfr_player_id"] = "PFRCB1"
        prior = [
            {"pfr_player_id": "PFRCB1", "season": 2026, "week": w,
             "game_type": "REG", "def_targets": 4, "def_yards_allowed": 32,
             "def_completions_allowed": 2}
            for w in (9, 10, 11)
        ]
        out = build(*args, now=NOW, pfr_def=prior)
        wr = next(c for c in out["player_cards"] if c["market"] == "receiving_yards")
        cb = wr["likely_defensive_personnel_from_previous_snaps"][0]
        self.assertEqual(cb["prior_defender_charted_coverage"]["defender_attributed_targets"], 12)
        self.assertEqual(cb["prior_defender_charted_coverage"]["charted_yards_per_target_allowed"], 8.0)
        self.assertFalse(cb["assignment_verified"])
        self.assertTrue(cb["prior_defender_charted_coverage"]["not_individual_wr_assignment"])
        text = render_markdown(out)
        self.assertIn("Opposing CB prior defender charting", text)
        # A future weekly PFR result must not enter this forecast.
        future = {"pfr_player_id": "PFRCB1", "season": 2026, "week": 12,
                  "game_type": "REG", "def_targets": 100,
                  "def_yards_allowed": 9000, "def_completions_allowed": 100}
        out2 = build(*args, now=NOW, pfr_def=prior + [future])
        self.assertEqual(out["player_cards"], out2["player_cards"])

    def test_human_scout_notes_have_named_mechanisms_but_no_fake_assignments(self):
        report = build(*fixture(), now=NOW)
        text = render_markdown(report)
        self.assertIn("Example Receiver", text)
        self.assertIn("deep_20plus", text)
        self.assertIn("Opposing CB", text)
        self.assertIn("Game-day availability and individual assignments unverified", text)
        self.assertIn("Published projection: **62 yards**", text)
        self.assertIn("not adjusted", text)
        self.assertNotIn("will cover Example Receiver", text)

    def test_future_game_and_current_week_play_not_read(self):
        args = list(fixture())
        baseline = build(*args, now=NOW)
        args[2].append({"season": 2026, "week": 12, "season_type": "REG",
                        "game_id": "2026_12_LA_GB", "posteam": "LA", "defteam": "GB",
                        "play_type": "pass", "receiver_player_id": "WR1",
                        "air_yards": 60, "yards_gained": 1000})
        args[3].append({"season": 2026, "week": 12, "game_type": "REG",
                        "team": "GB", "player": "Future CB", "position": "CB", "defense_pct": 1.0})
        out = build(*args, now=NOW)
        self.assertEqual(out["player_cards"], baseline["player_cards"])

    def test_wrong_roster_snapshot_week_and_forecast_rejected(self):
        args = list(fixture())
        args[1] = copy.deepcopy(args[1])
        args[1]["forecast_generated_at_utc"] = "2026-11-24T01:00:00Z"
        with self.assertRaisesRegex(ValueError, "ROLE_FORECAST_MISMATCH"):
            build(*args, now=NOW)
        args = list(fixture())
        args[1] = copy.deepcopy(args[1])
        args[1]["week"] = 11
        with self.assertRaisesRegex(ValueError, "ROLE_WEEK_MISMATCH"):
            build(*args, now=NOW)

    def test_missing_role_and_coverage_never_invented(self):
        args = list(fixture())
        args[1] = copy.deepcopy(args[1])
        args[1]["player_market_evidence"] = []
        out = build(*args, now=NOW)
        self.assertEqual(out["summary"]["missing_role_evidence"], 2)
        self.assertIsNone(out["player_cards"][0]["observed_role"])
        self.assertTrue(all(c["forecast_change_yards"] == 0 for c in out["player_cards"]))

    def test_reject_kickoff_mismatch_and_duplicate_role(self):
        args = list(fixture())
        args[0] = copy.deepcopy(args[0])
        args[0]["picks"][0]["kickoff_utc"] = "2026-11-26T02:00:00Z"
        with self.assertRaisesRegex(ValueError, "PICK_KICKOFF_MISMATCH"):
            build(*args, now=NOW)
        args = list(fixture())
        args[1] = copy.deepcopy(args[1])
        args[1]["player_market_evidence"].append(args[1]["player_market_evidence"][0])
        with self.assertRaisesRegex(ValueError, "DUPLICATE_ROLE_EVIDENCE"):
            build(*args, now=NOW)

    def test_no_coverage_or_blocking_assignments_from_generic_team_stats(self):
        p = {"by_play_type": {"deep_20plus": {"n": 7, "observed_share": .4, "yards_per_observed_play": 9.1}}}
        d = {"by_play_type": {"deep_20plus": {"n": 18, "observed_share": .2, "yards_per_observed_play": 12.2}}}
        matches = evidence_match(p, d, "receiving_yards")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["interpretation"], "DESCRIPTIVE_MATCHUP_NOT_VALIDATED_CAUSAL_EFFECT")
        self.assertNotIn("cornerback", matches[0])

    def test_fail_closed_when_injury_receipt_is_in_future(self):
        args = list(fixture())
        args[4] = {"captured_at_utc": "2026-11-30T00:00:00Z", "players": []}
        with self.assertRaisesRegex(ValueError, "FUTURE_INJURY_TIMESTAMP"):
            build(*args, now=NOW)


if __name__ == "__main__":
    unittest.main()
