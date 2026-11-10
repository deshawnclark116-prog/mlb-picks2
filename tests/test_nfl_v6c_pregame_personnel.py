"""V6C source-timed personnel and actual opportunity-allocation tests.

V6A's legacy research outputs are unchanged when no source is supplied.
"""
import json
import tempfile
import unittest
from pathlib import Path

import nfl_v6_joint_opportunity_shadow as v6
from nfl_v6c_pregame_personnel import SCHEMA, read_personnel

GAME = {"game_id":"2026_10_BUF_MIN","home_team":"BUF","away_team":"MIN",
        "kickoff_utc":"2026-11-13T01:15:00Z"}
CAPTURE = "2026-11-12T19:00:00Z"
SOURCE = {"url":"https://www.buffalobills.com/news/test-official-report",
          "published_at_utc":"2026-11-12T18:30:00Z"}
OTHER_SOURCE = {"url":"https://www.vikings.com/news/test-official-report",
                "published_at_utc":"2026-11-12T18:30:00Z"}


def receipt():
    return {"schema":SCHEMA,"captured_at_utc":CAPTURE,
       "game_id":GAME["game_id"],"kickoff_utc":GAME["kickoff_utc"],
       "teams":[{"team":"BUF",
                 "starter_qb":{"player_id":"BUFqb","player":"BUFQB",
                               "sources":[SOURCE]},
                 "confirmed_out":[{"player_id":"BUFa","player":"BUFWRa",
                                    "sources":[SOURCE]}]},
                {"team":"MIN",
                 "starter_qb":None,"confirmed_out":[]}]}


class PersonnelContract(unittest.TestCase):
    def read(self, doc, *, captured_at=CAPTURE):
        with tempfile.TemporaryDirectory() as temp:
            f=Path(temp)/"role.json"
            f.write_text(json.dumps(doc))
            return read_personnel(f,GAME,CAPTURE,captured_at_utc=captured_at)

    def test_exact_game_and_source_timed_assertions(self):
        r=self.read(receipt())
        self.assertEqual(r["teams"]["BUF"]["starter_id"],"BUFqb")
        self.assertEqual(r["teams"]["BUF"]["inactive_ids"],{"BUFa"})
        self.assertIsNone(r["teams"]["MIN"]["starter_id"])
        self.assertTrue(r["role_assertions_are_not_automatically_source_verified"])
        self.assertEqual(len(r["sha256"]),64)

    def test_wrong_game_or_late_capture_or_source_rejected(self):
        d=receipt();d["game_id"]="OTHER"
        with self.assertRaisesRegex(ValueError,"GAME_ID"):
            self.read(d)
        d=receipt();d["captured_at_utc"]="2026-11-12T20:00:00Z"
        with self.assertRaisesRegex(ValueError,"CUTOFF"):
            self.read(d)
        d=receipt();d["teams"][0]["starter_qb"]["sources"]=[{
            "url":"https://www.buffalobills.com/news/test",
            "published_at_utc":"2026-11-12T19:01:00Z"}]
        with self.assertRaisesRegex(ValueError,"LATE"):
            self.read(d)
        d=receipt();d["teams"][0]["starter_qb"]["sources"]=[{
            "url":"https://not-an-official-news-site.example/news/fake",
            "published_at_utc":"2026-11-12T18:00:00Z"}]
        with self.assertRaisesRegex(ValueError,"NOT_OFFICIAL"):
            self.read(d)

    def test_conflicting_starter_and_out_rejected(self):
        d=receipt();d["teams"][0]["confirmed_out"][0]["player_id"]="BUFqb"
        with self.assertRaisesRegex(ValueError,"MARKED_OUT"):
            self.read(d)

    def test_duplicate_or_foreign_team_rejected(self):
        d=receipt();d["teams"][1]["team"]="BUF"
        with self.assertRaisesRegex(ValueError,"DUPLICATE"):
            self.read(d)
        d=receipt();d["teams"][1]["team"]="DAL"
        with self.assertRaisesRegex(ValueError,"FOREIGN_TEAM"):
            self.read(d)


def fixture():
    by_team={}
    hist={}
    for team,opp in (("BUF","MIN"),("MIN","BUF")):
        games=[]
        for week in range(1,6):
            rows=[]
            for suffix,pos,tar,car,att in (
                    ("a","WR",12,0,0),("b","WR",7,0,0),
                    ("rb","RB",2,18,0),("qb","QB",0,2,34)):
                p={"id":team+suffix,"name":team+pos+suffix,
                   "pos":pos,"team":team,"opp":opp,"game_id":str(week),
                   "kickoff":f"2026-09-{week:02d}T17:00:00Z",
                   "targets":tar,"receptions":int(tar*.7),
                   "rec_yards":tar*8,"carries":car,"rush_yards":car*4,
                   "attempts":att,"pass_yards":att*7}
                rows.append(p)
                hist.setdefault((p["id"],team),[]).append(p)
            games.append({"team":team,"opp":opp,
              "kickoff":f"2026-09-{week:02d}T17:00:00Z",
              "passes":34,"rushes":22,"pass_yards":238,"rush_yards":88,
              "targets":21,"rows":rows})
        by_team[team]=games
    return by_team,hist,v6.league_rates(by_team)


class PersonnelAllocation(unittest.TestCase):
    def test_starter_not_assumed_and_inactive_removed_from_real_draws(self):
        teams,history,league=fixture()
        no_source=v6.simulate_matchup(GAME,teams,history,league,10000,101)
        source=read_teams()
        conditioned=v6.simulate_matchup(GAME,teams,history,league,10000,101,
                                        personnel=source)
        orig={p["player"]:p for p in no_source["projections"] if p["market"]=="receiving_yards"}
        new={p["player"]:p for p in conditioned["projections"] if p["market"]=="receiving_yards"}
        self.assertIn("BUFWRa",orig)
        self.assertNotIn("BUFWRa",new)
        self.assertGreater(new["BUFWRb"]["projected_mean"],orig["BUFWRb"]["projected_mean"])
        self.assertEqual(conditioned["teams"]["BUF"]["source_asserted_out_removed"],["BUFWRa"])
        self.assertEqual(conditioned["teams"]["BUF"]["qb_role_status"],
                         "SOURCE_ASSERTED_STARTER_MATCHED_IN_HISTORY")
        self.assertEqual(conditioned["teams"]["MIN"]["qb_role_status"],
                         "NO_SOURCE_ASSERTED_STARTER")
        qb={p["player"] for p in conditioned["projections"] if p["market"]=="passing_yards"}
        self.assertIn("BUFQBqb",qb)
        self.assertNotIn("MINQBqb",qb)
        self.assertEqual(no_source["teams"]["BUF"].get("qb_role_status"),None)

    def test_unmatched_asserted_starter_never_falls_back_to_recent_qb(self):
        teams,history,league=fixture()
        r=read_teams()
        r["teams"]["BUF"]["starter_id"]="totally-new-qb"
        conditioned=v6.simulate_matchup(GAME,teams,history,league,10000,202,
                                        personnel=r)
        self.assertEqual(conditioned["teams"]["BUF"]["qb_role_status"],
                         "SOURCE_ASSERTED_STARTER_NOT_IN_CANDIDATE_HISTORY")
        self.assertFalse(any(p["market"]=="passing_yards" and p["team"]=="BUF"
                             for p in conditioned["projections"]))


def read_teams():
    return {"teams":{"BUF":{"starter_id":"BUFqb","inactive_ids":{"BUFa"}},
                     "MIN":{"starter_id":None,"inactive_ids":set()}}}


if __name__=="__main__":
    unittest.main()
