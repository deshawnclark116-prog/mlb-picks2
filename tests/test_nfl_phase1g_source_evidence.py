"""Phase1G strictly timestamped official personnel assertions change real role weights."""
import copy
import json
import tempfile
import unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path

import numpy as np

import nfl_phase1g_source_evidence as SRC
import nfl_phase1g_player_intelligence as G

UTC=timezone.utc
KICK=datetime(2026,11,15,18,tzinfo=UTC)
CUT=KICK-timedelta(minutes=90)
CAP=KICK-timedelta(minutes=100)
GAME="2026_11_BUF_MIN"
S,W=2026,11


def source():
    return {"schema":SRC.SCHEMA,"captured_at_utc":CAP.isoformat(),
      "kickoff_utc":KICK.isoformat(),"game_id":GAME,
      "teams":[{"team":"BUF",
                "starter_qb":{"player_id":"BUFqb1","player":"Test QB",
                  "sources":[{"url":"https://www.buffalobills.com/news/game-info",
                              "published_at_utc":(CAP-timedelta(minutes=15)).isoformat()}]},
                "confirmed_out":[{"player_id":"BUFwr2","player":"Test WR",
                  "sources":[{"url":"https://www.buffalobills.com/news/inactives",
                              "published_at_utc":(CAP-timedelta(minutes=15)).isoformat()}]}]},
               {"team":"MIN","starter_qb":None,"confirmed_out":[]}]}


def pack():
    return {"games":{
      (S,W,"BUF"):{"types":{
       "target":{"ids":["BUFwr1","BUFwr2"],
                 "P1":np.array([.45,.3]),"pact90":np.array([.9,.9])},
       "carry":{"ids":["BUFrb"],"P1":np.array([.75]),
                "pact90":np.array([.9])},
       "qb_att":{"ids":["BUFqb1","BUFqb2"],
                 "P1":np.array([.6,.25]),"pact90":np.array([.95,.7])}}},
      (S,W,"MIN"):{"types":{
       "target":{"ids":["MINwr"],"P1":np.array([.7]),
                 "pact90":np.array([.95])},
       "carry":{"ids":["MINrb"],"P1":np.array([.7]),
                 "pact90":np.array([.9])},
       "qb_att":{"ids":["MINqb"],"P1":np.array([.9]),
                 "pact90":np.array([.92])}}}}}


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.file=Path(self.folder.name)/"evidence.json"
        self.doc=source()
        self.game={"game_id":GAME,"home":"BUF","away":"MIN",
                   "kick":KICK,"season":S,"week":W,"horizon":"T90"}

    def run_source(self,doc=None):
        self.file.write_text(json.dumps(doc or self.doc))
        data=pack()
        got=G.apply_documented_personnel(
            data,self.file,self.game,CUT,CAP.isoformat())
        return data,got

    def test_confirmed_out_zero_and_qb_source_soft_prior(self):
        data,got=self.run_source()
        t=data["games"][(S,W,"BUF")]["types"]
        self.assertEqual(t["target"]["P1"][1],0)
        self.assertEqual(t["target"]["pact90"][1],0)
        self.assertGreater(t["qb_att"]["P1"][0],.6)
        self.assertLess(t["qb_att"]["P1"][0],1.)
        self.assertEqual(got["teams"]["BUF"]["qb_role_status"],
                         "SOURCE_ASSERTED_STARTER_SOFT_SCENARIO")
        self.assertEqual(len(got["source_sha256"]),64)
        self.assertIn("NOT_INDEPENDENTLY_VERIFIED",got["status"])

    def test_mismatched_game_time_and_unsupported_domain_fail(self):
        d=source();d["game_id"]="OTHER"
        with self.assertRaisesRegex(ValueError,"GAME_ID"):
            self.run_source(d)
        d=source()
        d["teams"][0]["starter_qb"]["sources"][0]["url"]="https://fake.example/start-qb"
        with self.assertRaisesRegex(ValueError,"NOT_OFFICIAL"):
            self.run_source(d)
        d=source()
        d["teams"][0]["starter_qb"]["sources"][0]["published_at_utc"]=KICK.isoformat()
        with self.assertRaisesRegex(ValueError,"LATE_OR_AFTER"):
            self.run_source(d)

    def test_qb_assertion_conflicting_with_out_is_rejected(self):
        d=source();d["teams"][0]["confirmed_out"][0]["player_id"]="BUFqb1"
        with self.assertRaisesRegex(ValueError,"STARTER_MARKED_OUT"):
            self.run_source(d)

    def test_missing_optional_receipt_never_invents_starter(self):
        data=pack()
        got=G.apply_documented_personnel(data,None,self.game,CUT,CAP.isoformat())
        self.assertEqual(got["status"],"NO_DOCUMENTED_PERSONNEL_ASSERTIONS")
        self.assertAlmostEqual(data["games"][(S,W,"BUF")]["types"]["qb_att"]["P1"][0],.6)


if __name__=="__main__":unittest.main()
