import unittest
from nfl_phase1g_opportunity_grade import grade, MARKETS
class OpportunityGrade(unittest.TestCase):
    def test_market_actual_mapping(self):
        self.assertEqual(MARKETS["rec_yds"][0],"targets")
        self.assertEqual(MARKETS["rush_yds"][0],"carries")
        self.assertEqual(MARKETS["pass_yds"][0],"attempts")
    def test_missing_root_has_no_false_accuracy_claim(self):
        import tempfile,sqlite3
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            db=Path(d)/"test.db"
            with sqlite3.connect(db) as con:
                con.execute("CREATE TABLE player_games (game_id TEXT, team TEXT, player_id TEXT, season_type TEXT, targets INTEGER, carries INTEGER, attempts INTEGER)")
            out=grade(Path(d),db)
            self.assertEqual(out["matched_originals"]["n"],0)
            self.assertFalse(out["promotion_eligible"])
if __name__=="__main__":unittest.main()
