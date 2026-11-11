"""Original, strictly pregame Phase1G challenger comparison tests."""
import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from nfl_phase1g_forward_grade import compare

GID="2026_11_BUF_MIN"
KICK="2026-11-15T18:00:00Z"
CUT="2026-11-15T16:30:00Z"
RETRIEVED="2026-11-15T16:26:00Z"
FILE=f"2026_wk11_T90_{GID}.jsonl"


def champion(pid, y=61, content="valid-source"):
    return {"game_id":GID,"horizon":"T90","team":"BUF","opponent":"MIN",
      "player_id":pid,"player_name":"Test Player","outcome":"rec_yds",
      "median":y,"mean":y+2,"expected_opportunities":8.0,"quantiles":{"p10":24,"p90":100},
      "kickoff":KICK,"cutoff":CUT,
      "input_snapshots":{"content_id":content,"retrieval_ts":RETRIEVED,
                         "time_travel":False}}


def challenger(pid,y=58, content="valid-source"):
    x=champion(pid,y,content)
    x["model_version"]="p1g-prospective-v1"
    x["phase1g_evidence"]={"forecast_promoted":False}
    return x


def v2(pid,content="valid-source"):
    return {"game_id":GID,"horizon":"T90","team":"BUF","opponent":"MIN",
      "player_id":pid,"outcome":"rec_yds","kickoff":KICK,"cutoff":CUT,
      "as_of_retrieval_ts":RETRIEVED,
      "input_hashes":{"content_id":content},
      "kind":"v2_forward_forecast","eligibility":"eligible",
      "v2_frozen_comparator":{"point":60.0}}


class ProspectiveTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        for p in ("forecasts","player_challenger_forecasts","v2forecasts"):
            (self.root/p/"batches").mkdir(parents=True)
        self.db=self.root/"results.db"
        with sqlite3.connect(self.db) as c:
            c.executescript("""CREATE TABLE games(game_id TEXT,home_team TEXT,
              away_team TEXT,home_score INTEGER,away_score INTEGER);
              CREATE TABLE player_games(game_id TEXT,team TEXT,player_id TEXT,
              season_type TEXT,receiving_yards REAL,rushing_yards REAL,
              targets REAL,carries REAL);""")
            c.execute("INSERT INTO games VALUES(?,?,?,?,?)",(GID,"BUF","MIN",23,6))
            c.execute("INSERT INTO player_games VALUES(?,?,?,?,?,?,?,?)",
                      (GID,"BUF","p1","REG",58,0,10,0))

    def write(self,one=None,two=None,third=None):
        head={"_batch":FILE[:-6],"game_id":GID,
              "generated_at":"2026-11-15T16:35:00Z"}
        for name,rows in (("forecasts",[head]+(one or [champion("p1")])),
                          ("v2forecasts",two or [v2("p1")]),
                          ("player_challenger_forecasts",[head]+(third or [challenger("p1")]))):
            (self.root/name/"batches"/FILE).write_text(
                "\n".join(json.dumps(x) for x in rows)+"\n")

    def test_three_way_original_projection_accuracy(self):
        self.write()
        a=compare(self.root,self.db,weeks={11})
        self.assertEqual(a["matched"],1)
        self.assertEqual(a["all_verified_paired"]["phase1g_mae"],0)
        self.assertEqual(a["all_verified_paired"]["phase1e_mae"],3)
        self.assertEqual(a["all_verified_paired"]["frozen_v2_mae"],2)
        self.assertEqual(a["complete_player_by_player"][0]["actual"],58)
        opp=a["all_verified_paired"]["player_opportunity_comparison"]
        self.assertEqual(opp["n"],1)
        self.assertEqual(opp["phase1g_mae"],2.0)
        self.assertEqual(opp["phase1e_mae"],2.0)

    def test_missing_player_not_made_into_zero_yards(self):
        self.write(one=[champion("p1"),champion("p2")],
                   two=[v2("p1"),v2("p2")],
                   third=[challenger("p1"),challenger("p2")])
        a=compare(self.root,self.db,weeks={11})
        self.assertEqual(a["matched"],1)
        self.assertEqual(a["excluded"]["unverified_player_outcome_not_zero"],1)

    def test_different_source_versions_cannot_compare(self):
        self.write(third=[challenger("p1",content="different")])
        with self.assertRaisesRegex(ValueError,"NOT_SAME_PREGAME_SNAPSHOT"):
            compare(self.root,self.db,weeks={11})

    def test_late_model_generation_rejected(self):
        self.write()
        path=self.root/"player_challenger_forecasts"/"batches"/FILE
        z=path.read_text().splitlines()
        head=json.loads(z[0]);head["generated_at"]="2026-11-15T19:00:00Z"
        z[0]=json.dumps(head);path.write_text("\n".join(z)+"\n")
        with self.assertRaisesRegex(ValueError,"GENERATED_AFTER_KICKOFF"):
            compare(self.root,self.db,weeks={11})

    def test_unsupported_promotion_or_time_travel_rejected(self):
        c=challenger("p1")
        c["phase1g_evidence"]["forecast_promoted"]=True
        self.write(third=[c])
        with self.assertRaisesRegex(ValueError,"NOT_LABELED_RESEARCH"):
            compare(self.root,self.db,weeks={11})
        c=challenger("p1")
        c["input_snapshots"]["time_travel"]=True
        self.write(third=[c])
        with self.assertRaisesRegex(ValueError,"TIME_TRAVEL"):
            compare(self.root,self.db,weeks={11})

if __name__=="__main__":unittest.main()
