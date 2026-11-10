"""V6B snap-role branch structural validity: separate from sealed V6A."""
import unittest
import nfl_v6_joint_opportunity_shadow as v6


class SnapV6BTests(unittest.TestCase):
    def test_role_priors_change_simulated_usage_with_same_game_opportunities(self):
        rows=[];source={"by_player_game":{}}
        for w in range(1,6):
            for team,opp in (("MIN","BUF"),("BUF","MIN")):
                gid=f"2026_0{w}_{team}_{opp}"
                for name,position,targets,carries,attempts,ry,cy,py,pct,snaps in (
                    ("Star Receiver","WR",11,0,0,88,0,0,.90,54),
                    ("Role Runningback","RB",2,15,0,14,66,0,.62,37),
                    ("Quarterback","QB",0,2,35,0,6,251,.98,59)):
                    pid=team+name
                    rows.append({"id":pid,"name":name,"pos":position,
                      "team":team,"opp":opp,"game_id":gid,"season":2026,
                      "week":w,"targets":targets,"receptions":int(targets*.7),
                      "rec_yards":ry,"carries":carries,"rush_yards":cy,
                      "attempts":attempts,"pass_yards":py,
                      "kickoff":f"2026-10-{w:02d}T17:00:00Z"})
                    source["by_player_game"][(gid,team,name.lower())]={
                      "offense_snaps":snaps,"offense_pct":pct}
        by_team,players=v6.game_history(rows)
        league=v6.league_rates(by_team)
        game={"game_id":"2026_09_BUF_MIN","home_team":"MIN",
              "away_team":"BUF","kickoff_utc":"2026-11-10T01:15:00Z"}
        b=v6.simulate_matchup(game,by_team,players,league,10000,402)
        c=v6.simulate_matchup(game,by_team,players,league,10000,402,role_source=source)
        self.assertEqual(b["game_id"],c["game_id"])
        self.assertNotIn("snap_source_role",b["teams"]["MIN"])
        observed=c["teams"]["MIN"]["snap_source_role"]
        self.assertEqual(observed["matched_candidates"],3)
        self.assertFalse(observed["route_data_included"])
        self.assertFalse(observed["active_status_verified"])
        for market in ("receiving_yards","rushing_yards","passing_yards"):
            self.assertTrue(any(p["market"]==market for p in c["projections"]))
        self.assertTrue(all(p["simulations"]==10000 for p in c["projections"]))
        m1=next(p for p in b["projections"] if p["team"]=="MIN" and
          p["player"]=="Star Receiver" and p["market"]=="receiving_yards")
        m2=next(p for p in c["projections"] if p["team"]=="MIN" and
          p["player"]=="Star Receiver" and p["market"]=="receiving_yards")
        self.assertNotEqual(m1["projected_mean"],m2["projected_mean"])
        self.assertEqual(m2["snap_role_source_status"],
            "OBSERVED_ASOF_OFFENSE_SNAP_ROLE_NOT_ROUTE_DATA")
        self.assertEqual(c,v6.simulate_matchup(
            game,by_team,players,league,10000,402,role_source=source))


if __name__=="__main__":
    unittest.main()
