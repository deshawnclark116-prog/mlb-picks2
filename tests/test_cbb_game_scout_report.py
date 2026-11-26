"""Contract for source-backed, readable CBB team-vs-team matchup notes."""
import unittest
from cbb_game_scout_report import render


def example():
    return {"generated_utc": "2026-11-26T04:00:00Z",
            "report": {"untouched_2026_holdout": {"games": 6299,
                       "moneyline_accuracy": .7187, "moneyline_brier": .18099,
                       "margin_mae": 10.6162, "total_mae": 13.9873}},
            "game_predictions": [{
                "tip_utc": "2026-11-27T01:00:00+00:00",
                "home": "Duke Blue Devils", "away": "Connecticut Huskies",
                "projected_home_score": 74.5, "projected_away_score": 70.1,
                "projected_home_margin": 4.4, "projected_total": 144.6,
                "p_home_win": .64, "game_intelligence": {
                    "home_pace_prior": 69., "away_pace_prior": 64.,
                    "shared_pace_proxy": 66.5,
                    "home_offense_vs_away_defense": {
                        "offensive_points_per_100_possessions_prior": 113.,
                        "opponent_defensive_points_allowed_per_100_possessions_prior": 99.,
                        "shooting_efficiency_vs_defensive_allowance_proxy_pct_points": 1.2,
                        "turnover_rate_vs_opponent_forced_rate_pct_points": -2.4,
                        "offensive_rebound_rate_vs_opponent_allowed_pct_points": 3.1,
                        "top_previous_rotation_players": [
                            {"name": "Example Guard", "minutes_in_last_3_observed_team_games": 90.},
                        ],
                    },
                    "away_offense_vs_home_defense": {
                        "offensive_points_per_100_possessions_prior": 107.,
                        "opponent_defensive_points_allowed_per_100_possessions_prior": 100.,
                        "shooting_efficiency_vs_defensive_allowance_proxy_pct_points": -.8,
                        "turnover_rate_vs_opponent_forced_rate_pct_points": 1.2,
                        "offensive_rebound_rate_vs_opponent_allowed_pct_points": -1.1,
                        "top_previous_rotation_players": [],
                    }
                }
            }]}


class Scouts(unittest.TestCase):
    def test_named_players_but_no_lineups_fabricated(self):
        s=render(example())
        self.assertIn("Example Guard",s)
        self.assertIn("Not a confirmed lineup",s)
        self.assertIn("eFG%",s)
        self.assertIn("shared proxy 66.5",s)
        self.assertIn("64.0%",s)
        self.assertIn("No spread or O/U betting edge asserted",s)
        self.assertIn("Displayed 1 of 1",s)

    def test_missing_scout_does_not_fabricate_matchup(self):
        x=example()
        x["game_predictions"][0].pop("game_intelligence")
        s=render(x)
        self.assertIn("No source-grounded game intelligence snapshot supplied",s)
        self.assertNotIn("Example Guard",s)

    def test_invalid_limit_and_empty_list(self):
        with self.assertRaises(ValueError):
            render(example(),max_games=0)
        x=example()
        x["game_predictions"]=[]
        self.assertIn("Displayed 0 of 0",render(x))


if __name__=="__main__":
    unittest.main()
