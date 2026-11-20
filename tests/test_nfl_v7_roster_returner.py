"""Contract and model tests for the independent NFL V7 challenger."""
import unittest
from datetime import datetime, timedelta, timezone

import nfl_v6_joint_opportunity_shadow as v6
import nfl_v7_roster_returner as v7


KICK = "2026-11-22T18:00:00Z"
OBSERVED = "2026-11-20T18:00:00Z"


def fake_espn(names, injured=None):
    injured=injured or {}
    return {"athletes":[{"position":"offense","items":[
        {"displayName":name, "injuries":([{"status":injured[name]}]
           if name in injured else [])} for name in names]}]}


def source(team, names, injured=None):
    return v7.parse_roster(team,fake_espn(names,injured),OBSERVED,KICK)


def histories():
    allhist, by_team = {}, {}
    for team,opp in (("A","B"),("B","A")):
        games=[]
        for i in range(7):
            kick=(datetime(2026,9,1,tzinfo=timezone.utc)+
                  timedelta(days=i*7)).isoformat()
            roster=[
                ("p"+team,"Everyday "+team, 5, 0, 0),
                ("q"+team,"Quarterback "+team, 0, 2, 30),
            ]
            if i < 4:
                roster.append(("r"+team,"Returning "+team,2,4,0))
            game_rows=[]
            for pid, name, tgt, car, att in roster:
                row={"id":pid,"name":name,"pos":"QB" if att else "RB",
                    "team":team,"opp":opp,"kickoff":kick,
                    "targets":tgt,"carries":car,"attempts":att,
                    "rec_yards":tgt*7,"rush_yards":car*4,
                    "pass_yards":att*7}
                allhist.setdefault((pid,team),[]).append(row)
                game_rows.append(row)
            games.append({"team":team,"opp":opp,"kickoff":kick,
                "passes":30,"rushes":25,"targets":22,
                "pass_yards":220,"rush_yards":100,"rows":game_rows})
        by_team[team]=games
    return by_team,allhist


LEAGUE={"rec_ypt":7.0,"rush_ypc":4.0,"pass_ypa":7.0,
        "team_plays":55,"pass_rate":.55}


class V7Tests(unittest.TestCase):
    def test_requires_pregame_source_and_complete_roster(self):
        with self.assertRaisesRegex(ValueError,"ROSTER_NOT_OBSERVED_BEFORE_KICKOFF"):
            v7.parse_roster("A",fake_espn(["Name"]),KICK,KICK)
        with self.assertRaisesRegex(ValueError,"INVALID_ESPN_ROSTER_PAYLOAD"):
            v7.parse_roster("A",{"athletes":"oops"},OBSERVED,KICK)
        with self.assertRaisesRegex(ValueError,"EMPTY_ESPN_ROSTER"):
            v7.parse_roster("A",{"athletes":[]},OBSERVED,KICK)

    def test_v6_last_two_misses_returner_but_v7_can_restore(self):
        by,hist=histories()
        old=v6.participants("A",by["A"],hist,LEAGUE)
        self.assertNotIn("rA",{p["id"] for p in old})
        people,audit=v7.roster_candidates("A",by["A"],hist,LEAGUE,
            source("A",["Everyday A","Quarterback A","Returning A"]))
        self.assertIn("rA",{p["id"] for p in people})
        returning=next(p for p in people if p["id"]=="rA")
        self.assertEqual(returning["team_fixture_gap"],3)
        self.assertFalse(returning["game_day_active_verified"])
        self.assertEqual(audit["returners_admitted"],1)
        self.assertEqual(audit["challenger_candidates"],3)

    def test_out_and_missing_identity_never_resurrect(self):
        by,hist=histories()
        people,audit=v7.roster_candidates("A",by["A"],hist,LEAGUE,
            source("A",["Everyday A","Quarterback A","Returning A"],
                {"Returning A":"Out"}))
        self.assertNotIn("rA",{p["id"] for p in people})
        self.assertEqual(audit["listed_unavailable"],1)
        people,audit=v7.roster_candidates("A",by["A"],hist,LEAGUE,
            source("A",["Everyday A","Quarterback A"]))
        self.assertNotIn("rA",{p["id"] for p in people})
        self.assertEqual(audit["returners_admitted"],0)

    def test_ambiguous_name_does_not_count_as_verification(self):
        by,hist=histories()
        roster=source("A",["Everyday A","Quarterback A","Returning A",
                           "Returning A Jr."])
        people,audit=v7.roster_candidates("A",by["A"],hist,LEAGUE,roster)
        self.assertNotIn("rA",{p["id"] for p in people})
        self.assertGreaterEqual(audit["unmatched_or_ambiguous"],1)

    def test_wrong_team_fails_closed(self):
        by,hist=histories()
        with self.assertRaisesRegex(ValueError,"VERIFIED_TEAM_ROSTER_REQUIRED"):
            v7.roster_candidates("A",by["A"],hist,LEAGUE,
                source("B",["Everyday A","Returning A"]))

    def test_joint_simulation_has_returner_without_fabricated_active_prob(self):
        by,hist=histories()
        rosters={team:source(team,[f"Everyday {team}",
                       f"Quarterback {team}",f"Returning {team}"])
                 for team in ("A","B")}
        game={"game_id":"2026_11_A_B","home_team":"A","away_team":"B",
              "kickoff_utc":KICK}
        a=v7.simulate_game(game,by,hist,LEAGUE,rosters,sims=10000,seed=6)
        b=v7.simulate_game(game,by,hist,LEAGUE,rosters,sims=10000,seed=6)
        self.assertEqual(a,b)
        self.assertTrue(a["no_calibrated_probability_of_playing"])
        returners=[x for x in a["projections"]
                   if x.get("returner_added_to_player_universe")]
        self.assertTrue(any(x["player_id"]=="rA" for x in returners))
        self.assertTrue(any(x["player_id"]=="rB" for x in returners))
        self.assertTrue(all(x["simulations"]==10000 for x in a["projections"]))
        self.assertTrue(all(0<=x["prob_reach_rounded_mean"]<=1
                            for x in a["projections"]))
        self.assertTrue(all(a["teams"][team]["unknown_others_target_and_carry_bucket"]
                            for team in ("A","B")))


if __name__=="__main__":
    unittest.main()
