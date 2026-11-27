"""CBB scores, lines and strict historical-data chronology acceptance gates."""
import copy
import unittest
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from cbb_game_engine_v0 import (
    FEATURE_NAMES, build_rows, score_probabilities, profile, vector, LEAGUE_PRIOR,
)


def game(game_id, season, date, home, away, hs, aas, neutral=False, done=True):
    return {"game_id": game_id, "season": season, "game_date_time": date,
            "home_id": home, "away_id": away, "home_display_name": "Home",
            "away_display_name": "Away", "home_score": hs, "away_score": aas,
            "status_type_completed": done, "neutral_site": neutral}


def team_box(gid, team, points, opp_points):
    return {"game_id": gid, "team_id": team, "team_score": points,
            "opponent_team_score": opp_points, "field_goals_attempted": 60,
            "field_goals_made": 28, "three_point_field_goals_made": 9,
            "free_throws_attempted": 20, "offensive_rebounds": 10,
            "defensive_rebounds": 21, "total_turnovers": 12}


class GameModelTests(unittest.TestCase):
    def test_hybrid_preserves_neutral_and_combines_non_neutral(self):
        from cbb_game_engine_v0 import assemble_hybrid
        base=np.array([[5.,140.],[-3.,150.]])
        combined=np.array([[9.,145.],[8.,160.]])
        totals=np.array([[2.,148.],[4.,170.]])
        out,scales=assemble_hybrid(base,[13.,18.],combined,[11.,19.],totals,[12.,17.],[False,True])
        np.testing.assert_array_equal(out,[[9.,148.],[-3.,150.]])
        np.testing.assert_array_equal(scales,[[11.,17.],[13.,18.]])

    def test_moneyline_spread_and_total_consistency_from_one_score(self):
        r = score_probabilities(7., 143., 12., 18., home_spread=-5.5, total_line=140.5)
        self.assertAlmostEqual(r["projected_home_score"], 75.)
        self.assertAlmostEqual(r["projected_away_score"], 68.)
        self.assertGreater(r["p_home_win"], .5)
        self.assertGreater(r["p_home_cover"], .5)
        self.assertGreater(r["p_over"], .5)
        self.assertAlmostEqual(r["p_home_win"] + r["p_away_win"], 1, places=4)
        self.assertIsNone(score_probabilities(7., 143., 12., 18.)["p_home_cover"])
        # Stronger handicap on home side must reduce its cover probability.
        self.assertLess(score_probabilities(7., 143., 12., 18., home_spread=-12.)["p_home_cover"],
                        r["p_home_cover"])
        with self.assertRaises(ValueError):
            score_probabilities(0, 140, 0, 16)

    def test_future_final_row_and_same_day_outcomes_cannot_leak(self):
        dates = [
            "2026-11-18T23:00:00Z", "2026-11-20T23:00:00Z",
            "2026-11-25T18:00:00Z", "2026-11-26T20:00:00Z",
        ]
        sched = pd.DataFrame([
            game(1, 2027, dates[0], 10, 20, 71, 65),
            game(2, 2027, dates[1], 20, 10, 60, 55),
            game(3, 2027, dates[2], 10, 20, 90, 50, done=True),
            game(4, 2027, dates[3], 10, 20, None, None, done=False),
        ])
        box = pd.DataFrame([
            team_box(1, 10, 71, 65), team_box(1, 20, 65, 71),
            team_box(2, 20, 60, 55), team_box(2, 10, 55, 60),
            team_box(3, 10, 90, 50), team_box(3, 20, 50, 90),
        ])
        asof = datetime(2026, 11, 25, 22, 0, tzinfo=timezone.utc)
        hist, fut = build_rows(sched, box, asof=asof)
        self.assertEqual(len(hist), 3)
        self.assertEqual(len(fut), 1)
        snapshot = fut[0]["x"].copy()
        altered = box.copy()
        altered.loc[altered.game_id == 3, "field_goals_attempted"] = 70
        # Future game snapshot already includes only prior completed games;
        # changing its past 11/25 outcome changes it, as expected.
        _, fut2 = build_rows(sched, altered, asof=asof)
        self.assertFalse(np.array_equal(snapshot, fut2[0]["x"]))
        # But for a 11/25 game, its own final score was never used as feature.
        past = next(r for r in hist if r["game_id"] == "3")
        past2 = next(r for r in build_rows(sched, altered, asof=asof)[0] if r["game_id"] == "3")
        np.testing.assert_array_equal(past["x"], past2["x"])
        # Future games cannot enter today's features even with fake boxes.
        corrupted = pd.concat([box, pd.DataFrame([
            team_box(4, 10, 200, 0), team_box(4, 20, 0, 200)
        ])])
        _, fut3 = build_rows(sched, corrupted, asof=asof)
        np.testing.assert_array_equal(snapshot, fut3[0]["x"])

    def test_no_data_still_has_finite_prior_and_22_features(self):
        from collections import defaultdict
        h = defaultdict(list)
        a, b = profile(h, "newteam", 2027), profile(h, "other", 2027)
        self.assertEqual(a["n_current"], 0)
        self.assertEqual(a["pace"], LEAGUE_PRIOR["pace"])
        self.assertFalse(a["player_minutes_observed"])
        x, mechanical = vector(a, b, neutral=False)
        self.assertEqual(len(x), len(FEATURE_NAMES))
        self.assertTrue(np.isfinite(x).all())
        self.assertAlmostEqual(mechanical[0], 3.5)

    def test_results_from_same_calendar_date_not_mutually_visible(self):
        schedule = pd.DataFrame([
            game(1, 2027, "2026-11-20T18:00:00Z", 1, 2, 80, 60),
            game(2, 2027, "2026-11-20T23:00:00Z", 1, 3, 70, 60),
        ])
        boxes = pd.DataFrame([
            team_box(1, 1, 80, 60), team_box(1, 2, 60, 80),
            team_box(2, 1, 70, 60), team_box(2, 3, 60, 70)
        ])
        asof = datetime(2026, 11, 25, tzinfo=timezone.utc)
        rows, _ = build_rows(schedule, boxes, asof=asof)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["home_prior_games"], 0)
        self.assertEqual(rows[1]["home_prior_games"], 0)

    def test_player_minutes_dont_turn_into_confirmed_starters(self):
        schedule = pd.DataFrame([
            game(1, 2027, "2026-11-20T18:00:00Z", 1, 2, 80, 60),
            game(2, 2027, "2026-11-27T18:00:00Z", 1, 2, None, None, done=False),
        ])
        boxes = pd.DataFrame([team_box(1, 1, 80, 60), team_box(1, 2, 60, 80)])
        players = pd.DataFrame([
            {"game_id": 1, "team_id": 1, "athlete_id": 111, "minutes": 30, "points": 20},
            {"game_id": 1, "team_id": 2, "athlete_id": 222, "minutes": 32, "points": 16},
        ])
        _, future = build_rows(schedule, boxes, players,
                               asof=datetime(2026, 11, 25, tzinfo=timezone.utc))
        self.assertTrue(future[0]["player_box_coverage"])
        self.assertEqual(future[0]["home_prior_games"], 1)
        scouting = future[0]["game_intelligence"]
        self.assertFalse(scouting["confirmed_starting_fives"])
        self.assertFalse(scouting["injuries_source_verified"])
        self.assertTrue(scouting["evidence_not_validated_individual_matchup_effects"])
        self.assertEqual(len(scouting["home_offense_vs_away_defense"]["top_previous_rotation_players"]), 1)
        self.assertEqual(
            scouting["home_offense_vs_away_defense"]["top_previous_rotation_players"][0]["athlete_id"], "111")



if __name__ == "__main__":
    unittest.main()
