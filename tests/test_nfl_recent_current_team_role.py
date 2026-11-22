"""NFL all-market publisher prevents retired/former-team players without editing V2."""
import sqlite3
import unittest
from nfl_current_team_role_guard import (
    recent_team_supported,filter_current_team_picks,SOURCE_VERSION
)

def obs(season,week,team):
    return {"season":season,"week":week,"team":team}

class NflCurrentRosterTests(unittest.TestCase):
    def test_retired_and_old_team_histories_fail(self):
        self.assertFalse(recent_team_supported(
            [obs(2022,14,"TB"),obs(2022,15,"TB"),obs(2022,16,"TB")],
            "TB",2026,10))
        self.assertFalse(recent_team_supported(
            [obs(2026,w,"KC") for w in (5,6,7)],"KC",2026,10))

    def test_bye_week_and_two_current_team_games_valid(self):
        self.assertTrue(recent_team_supported(
            [obs(2026,w,"KC") for w in (8,9,10)],"KC",2026,10))
        self.assertTrue(recent_team_supported(
            [obs(2026,w,"CHI") for w in (6,8,9)],"CHI",2026,9))
        self.assertFalse(recent_team_supported(
            [obs(2026,8,"ARI"),obs(2026,10,"KC")],"KC",2026,10))
        self.assertTrue(recent_team_supported(
            [obs(2026,8,"ARI"),obs(2026,9,"KC"),obs(2026,10,"KC")],
            "KC",2026,10))

    def test_public_board_keeps_active_and_moneyline_but_rejects_old_veterans(self):
        con=sqlite3.connect(":memory:")
        con.execute("""CREATE TABLE player_games (
           player_id TEXT,player_name TEXT,team TEXT,season INTEGER,
           week INTEGER,season_type TEXT
        )""")
        records=[]
        for wk in (5,6,7):
            records.append(("OLD","Retired Veteran","TB",2022,wk,"REG"))
        for wk in (8,9,10):
            records.append(("NEW","Current Player","KC",2026,wk,"REG"))
        # Team has last played W10; player went missing after W9.
        for wk in (8,9):
            records.append(("MISSING","Missing Player","KC",2026,wk,"REG"))
        for wk in (8,9,10):
            records.append(("OTHER","Other Player","TB",2026,wk,"REG"))
        con.executemany("INSERT INTO player_games VALUES (?,?,?,?,?,?)",records)
        con.commit()
        picks=[
          {"market":"moneyline","team":"KC","player_id":"","pick":"KC ML"},
          {"market":"receiving_yards","team":"TB","player_id":"OLD",
           "player":"Retired Veteran","projected_mean":40.7},
          {"market":"anytime_touchdowns","team":"TB","player_id":"OLD",
           "player":"Retired Veteran","model_prob":.4},
          {"market":"receiving_yards","team":"KC","player_id":"NEW",
           "player":"Current Player","projected_mean":62.3},
          {"market":"rushing_yards","team":"KC","player_id":"MISSING",
           "player":"Missing Player","projected_mean":51.5},
          {"market":"passing_yards","team":"TB","player_id":"OTHER",
           "player":"Other Player","projected_mean":240.1},
          {"market":"receiving_yards","team":"KC","player_id":"UNRECOGNIZED",
           "player":"Unknown Player","projected_mean":30.3},
        ]
        original=str(picks)
        valid,audit=filter_current_team_picks(con,picks,2026,11)
        self.assertEqual(audit["version"],SOURCE_VERSION)
        self.assertEqual(audit["removed_count"],3)
        self.assertEqual(audit["source_current_team_verified"],2)
        self.assertEqual(audit["source_identity_unresolved_retained_unverified"],1)
        self.assertEqual(len(valid),4)
        self.assertEqual([x["market"] for x in valid],[
          "moneyline","receiving_yards","passing_yards","receiving_yards"])
        self.assertEqual(valid[1]["projected_mean"],62.3)
        self.assertEqual(valid[1]["starter_verified"],False)
        self.assertEqual(valid[-1]["current_team_role_source"],"UNVERIFIED_SOURCE_IDENTITY")
        self.assertEqual(str(picks),original)
        # W11 postgame row CANNOT affect pre-week11 safety decision.
        con.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?)",
                    ("MISSING","Missing Player","KC",2026,11,"REG"))
        with_future,_=filter_current_team_picks(con,picks,2026,11)
        self.assertEqual(valid,with_future)
        con.close()

    def test_one_new_team_game_cannot_replace_prior_year_workload(self):
        h=[obs(2025,16,"ARI"),obs(2025,17,"ARI"),obs(2026,10,"KC")]
        self.assertFalse(recent_team_supported(h,"KC",2026,10,min_team_games=2))
        self.assertTrue(recent_team_supported(h,"KC",2026,10,min_team_games=1))

if __name__=="__main__":
    unittest.main()
