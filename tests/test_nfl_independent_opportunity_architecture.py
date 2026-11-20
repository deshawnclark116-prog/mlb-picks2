"""Independent NFL opportunity model source and chronological proof."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
import nfl_independent_opportunity_architecture_lab as lab

def hist(n=4,season=2025):
    return [{"season":season,"week":i+1,"carries":10,"targets":3,
        "rushing_yards":40,"receiving_yards":15} for i in range(n)]

class NFLArchitecturalTests(unittest.TestCase):
    def features(self,history=None):
        return lab.previous_only_features(history=hist() if history is None else history,
            team_games=[{"opportunities":25} for _ in range(4)],
            opponent_games=[{"yards_allowed":135} for _ in range(4)],
            week=6,season=2025,is_home=1,position="RB",market="rushing_yards")

    def test_features_uses_only_prior_history_and_unchanged_source(self):
        h=hist()
        frozen=self.features(h)
        self.assertEqual(len(frozen["features"]),len(lab.FEATURES))
        self.assertAlmostEqual(frozen["baseline"],40)
        self.assertAlmostEqual(frozen["prior_opps"],10)
        self.assertAlmostEqual(frozen["features"][10],10/25)
        self.assertEqual(frozen["source_history_rows"],4)
        self.assertFalse(frozen["opponent_coverage"]==0)
        self.assertEqual(h[-1]["rushing_yards"],40)

    def test_future_and_same_week_games_cannot_enter_features(self):
        for bad_week in (6,7):
            history=hist()+[{"season":2025,"week":bad_week,"carries":500,
                "targets":0,"rushing_yards":5000,"receiving_yards":0}]
            with self.assertRaisesRegex(ValueError,"LEAKAGE"):
                self.features(history)

    def test_history_missing_not_silently_filled_from_final(self):
        self.assertIsNone(self.features(hist(1)))
        empty=[{**r,"carries":0} for r in hist()]
        self.assertIsNone(self.features(empty))

    def test_all_market_roles_and_games_are_distinct(self):
        a=self.features()
        b=lab.previous_only_features(history=hist(),
          team_games=[],opponent_games=[],week=6,season=2025,
          is_home=False,position="WR",market="receiving_yards")
        self.assertNotEqual(a["features"],b["features"])
        self.assertEqual(a["eligible_high_volume"],True)
        self.assertEqual(b["eligible_high_volume"],False)

    @staticmethod
    def fixture(path,week4_yards):
        con=sqlite3.connect(path)
        con.execute("""CREATE TABLE games(game_id TEXT,season INT,week INT,season_type TEXT,
            home_score INT,away_score INT)""")
        con.execute("""CREATE TABLE player_games(player_id TEXT,player_name TEXT,
            team TEXT,opponent TEXT,position TEXT,season INT,week INT,
            game_id TEXT,season_type TEXT,is_home INT,
            carries REAL,targets REAL,rushing_yards REAL,receiving_yards REAL)""")
        for week in (1,2,3,4):
            gid=f"g{week}"
            con.execute("INSERT INTO games VALUES(?,2025,?,'REG',30,21)",(gid,week))
            for pid,team,opp,rush in [("A","ARI","BUF",40),
                                      ("B","BUF","ARI",50)]:
                if week==4 and pid=="A":rush=week4_yards
                con.execute("""INSERT INTO player_games VALUES
                    (?,?,?,?,'RB',2025,?,?,'REG',1,10,3,?,15)""",
                    (pid,"Player "+pid,team,opp,week,gid,rush))
        con.commit();con.close()

    def test_week_four_projections_same_when_future_outcome_changed(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/"a.sqlite";b=Path(d)/"b.sqlite"
            self.fixture(a,0);self.fixture(b,1000)
            def load(p):
                con=sqlite3.connect(p);con.row_factory=sqlite3.Row
                out=lab.build_pregame_dataset(con,seasons=(2025,))
                con.close();return out
            aa,bb=load(a),load(b)
            k=lambda rows:[r for r in rows if r["player_id"]=="A"
                   and r["week"]==4 and r["market"]=="rushing_yards"][0]
            self.assertEqual(k(aa)["features"],k(bb)["features"])
            self.assertEqual(k(aa)["baseline"],k(bb)["baseline"])
            self.assertNotEqual(k(aa)["actual"],k(bb)["actual"])
            self.assertEqual(k(aa)["source_history_rows"],3)

    def test_grade_uses_numerical_error_not_betting_hit(self):
        rows=[{"abs_error":5,"predicted":10,"actual":15},
              {"abs_error":15,"predicted":25,"actual":10}]
        scored=lab._stats(rows)
        self.assertEqual(scored["mae"],10)
        self.assertEqual(scored["within10"],.5)

if __name__=="__main__":
    unittest.main()
