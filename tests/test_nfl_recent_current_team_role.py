"""NFL release-blocking tests: no Tom-Brady-style retired/former-team forecasts."""
import unittest
from nfl_yardage_v2 import has_recent_current_team_role,serving_opportunity_eligible

def game(season,week,team,vol=8.0,pos="WR"):
    return {"season":season,"week":week,"team":team,
            "vol":vol,"yds":50,"pos":pos,"name":"Any Player"}

class NflTrueCurrentTeamRosterTests(unittest.TestCase):
    def test_archived_former_team_player_cannot_be_2026_game_prediction(self):
        history=[game(2022,w,"TB",9) for w in (14,15,16)]
        self.assertTrue(serving_opportunity_eligible(history,"receiving_yards"),
                        "The former serving bug could accept old usage")
        self.assertFalse(has_recent_current_team_role(history,"TB",2026,10))

    def test_current_team_participation_ending_before_latest_completed_game_rejected(self):
        h=[game(2026,w,"KC",6) for w in (5,6,7)]
        self.assertFalse(has_recent_current_team_role(h,"KC",2026,10))

    def test_active_week10_role_passes_and_team_bye_preserves_last_week(self):
        h=[game(2026,w,"KC",8) for w in (8,9,10)]
        self.assertTrue(has_recent_current_team_role(h,"KC",2026,10))
        h_bye=[game(2026,w,"CHI",7) for w in (6,8,9)]
        self.assertTrue(has_recent_current_team_role(h_bye,"CHI",2026,9))
        self.assertFalse(has_recent_current_team_role(h_bye,"CHI",2026,10))

    def test_one_game_after_trade_cannot_use_two_seasons_old_role(self):
        h=[game(2025,w,"ARI",10) for w in (15,16,17)]+[game(2026,10,"KC",6)]
        self.assertFalse(has_recent_current_team_role(h,"KC",2026,10))

    def test_recent_two_team_games_may_use_three_game_model_history(self):
        h=[game(2026,8,"ARI",7),game(2026,9,"KC",8),
           game(2026,10,"KC",9)]
        self.assertTrue(has_recent_current_team_role(h,"KC",2026,10))
        self.assertTrue(serving_opportunity_eligible(h,"receiving_yards"))

    def test_zero_last_team_game_source_missing_never_assume_active(self):
        h=[game(2026,w,"KC") for w in (8,9,10)]
        self.assertFalse(has_recent_current_team_role(h,"KC",2026,None))

if __name__=="__main__":
    unittest.main()
