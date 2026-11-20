"""Source-time, incomplete-universe and numeric grading contract tests."""
import unittest
import nfl_v7_archived_week10_numeric_diagnostic as d

GAME={"home_team":"BUF","away_team":"NYJ","kickoff_utc":"2026-11-15T18:00:00Z"}
ROW={"player":"Josh Allen","team":"BUF","opponent":"NYJ",
    "player_id":"00-0034857","market":"rushing_yards",
    "kickoff_utc":"2026-11-15T18:00:00Z",
    "roster_status":"ROSTER_LISTED_NOT_STARTER_VERIFIED"}
ROW2={"player":"A. Receiver","team":"NYJ","opponent":"BUF",
    "player_id":"00-0039999","market":"receiving_yards",
    "kickoff_utc":"2026-11-15T18:00:00Z",
    "roster_status":"INJURY_CAUTION_NOT_CONFIRMED_ACTIVE"}


def snap():
    return {"schema":"nfl-role-evidence-coverage-v1","season":2026,"week":10,
        "captured_at_utc":"2026-11-15T09:15:53Z",
        "starter_status_verified":False,
        "roster_source_status":{"BUF":"ROSTER_CAPTURED","NYJ":"ROSTER_CAPTURED"},
        "player_market_evidence":[ROW, {**ROW,"player":"Different Back",
            "player_id":"00-0039998"}, ROW2, {**ROW2,"player":"Another Receiver",
            "player_id":"00-0039997"}]}


class RetroContracts(unittest.TestCase):
    def test_true_pregame_partial_roster_evidence_labeled(self):
        rosters=d.archived_rosters(snap(),GAME)
        self.assertEqual(rosters["BUF"]["items"],2)
        self.assertEqual(rosters["NYJ"]["items"],2)
        self.assertTrue(rosters["NYJ"]["no_game_day_active_certification"])
        self.assertEqual(rosters["NYJ"]["players"]["a receiver"][0]["status"],
                         "ROSTER_LISTED_CAUTION_NOT_CONFIRMED_ACTIVE")

    def test_future_roster_source_fails(self):
        x=snap();x["captured_at_utc"]="2026-11-15T19:00:00Z"
        with self.assertRaisesRegex(ValueError,"UNUSABLE_PRETIP_ROSTER"):
            d.archived_rosters(x,GAME)

    def test_unverified_identity_excluded(self):
        x=snap();x["player_market_evidence"][0]["roster_status"]="ROSTER_IDENTITY_NOT_VERIFIED"
        with self.assertRaisesRegex(ValueError,"INSUFFICIENT_PARTIAL_ROSTER"):
            d.archived_rosters(x,GAME)

    def test_mae_calculates_same_all_player_rows(self):
        rows=[{"actual":15,"v7":10,"v6b":20,"incumbent":16,"market":"rushing_yards"},
              {"actual":25,"v7":20,"v6b":18,"incumbent":23,"market":"receiving_yards"}]
        out=d.grades(rows)
        self.assertEqual(out["n"],2)
        self.assertEqual(out["v7_mae"],5)
        self.assertEqual(out["v6b_mae"],6)
        self.assertEqual(out["incumbent_mae"],1.5)


if __name__=="__main__":
    unittest.main()
