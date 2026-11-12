#!/usr/bin/env python3
"""Synthetic, leak-proof tests for CFB matchup shadow reporting."""
import sqlite3
import unittest

from cfb_matchup_intelligence_shadow import build


def fixture():
    con = sqlite3.connect(":memory:")
    con.executescript("""
    CREATE TABLE games (
        game_id TEXT, season INTEGER, week INTEGER,
        home_team TEXT, away_team TEXT,
        home_points REAL, away_points REAL
    );
    CREATE TABLE player_games (
        game_id TEXT, team TEXT, season INTEGER, week INTEGER,
        player_id TEXT, player_name TEXT, position TEXT,
        carries REAL, pass_attempts REAL, passing_interceptions REAL
    );
    """)
    for gid, week, a, b in [("g1", 1, 24, 17), ("g2", 2, 42, 10),
                             ("g3", 3, 28, 31)]:
        con.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?)",
                    (gid, 2026, week, "Team A", "Team B", a, b))
        con.execute("INSERT INTO player_games VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (gid, "Team A", 2026, week, "a1", "A Quarterback", "QB",
                     3, 30 if week < 3 else 5, 0))
        if week == 3:
            con.execute("INSERT INTO player_games VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (gid, "Team A", 2026, week, "a2", "New Passer", "QB",
                         2, 26, 1))
        con.execute("INSERT INTO player_games VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (gid, "Team A", 2026, week, "ar", "A Rusher", "RB",
                     22, 0, 0))
        con.execute("INSERT INTO player_games VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (gid, "Team B", 2026, week, "b1", "B Quarterback", "QB",
                     6, 23, 0))
        con.execute("INSERT INTO player_games VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (gid, "Team B", 2026, week, "br", "B Rusher", "RB",
                     24, 0, 0))
    return con


def board(week=4):
    return {
        "season": 2026, "week": week, "generated_at_utc": "2026-11-12T20:00:00Z",
        "picks": [{"game_id": "future", "season": 2026, "week": week,
                   "market": "moneyline", "team": "Team A", "opponent": "Team B",
                   "pick": "Team A ML", "model_prob": 0.83,
                   "kickoff_utc": "2026-11-13T02:00:00Z"}]
    }


class CfbMatchupIntelligenceTests(unittest.TestCase):

    def setUp(self):
        self.con = fixture()

    def tearDown(self):
        self.con.close()

    def test_prior_week_observations_and_opportunity_coverage(self):
        result = build(self.con, board())
        self.assertEqual(result["game_count"], 1)
        self.assertFalse(result["model_changed"])
        self.assertFalse(result["official_pick_modified"])
        g = result["games"][0]
        self.assertEqual(g["team_context"]["prior_completed_games"], 3)
        self.assertEqual(g["team_context"]["opportunity_games_covered"], 3)
        self.assertAlmostEqual(g["team_context"]["last_five_points_for"], 31.333, 3)
        self.assertTrue(g["team_context"]["last_game_passer_changed"])
        self.assertTrue(g["context_ready"])
        self.assertFalse(g["prediction_produced"])

    def test_finished_target_week_cannot_poison_matchup(self):
        original = build(self.con, board())["games"][0]
        self.con.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?)",
                         ("future", 2026, 4, "Team A", "Team B", 200, 0))
        self.con.execute("INSERT INTO player_games VALUES (?,?,?,?,?,?,?,?,?,?)",
                         ("future", "Team A", 2026, 4, "a1", "A Quarterback",
                          "QB", 0, 100, 0))
        again = build(self.con, board())["games"][0]
        self.assertEqual(original["team_context"], again["team_context"])
        self.assertEqual(original["opponent_context"], again["opponent_context"])

    def test_game_earlier_in_same_week_never_leaks(self):
        a = build(self.con, board(4))["games"][0]
        self.con.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?)",
                         ("other_w4", 2026, 4, "Team A", "Team B", 250, 1))
        b = build(self.con, board(4))["games"][0]
        self.assertEqual(a["team_context"], b["team_context"])

    def test_duplicate_moneyline_game_rejected(self):
        b = board()
        b["picks"].append(dict(b["picks"][0]))
        with self.assertRaises(ValueError):
            build(self.con, b)

    def test_other_markets_not_included(self):
        b = board()
        b["picks"].append({"market": "rushing_yards", "game_id": "future"})
        self.assertEqual(build(self.con, b)["game_count"], 1)


if __name__ == "__main__":
    unittest.main()
