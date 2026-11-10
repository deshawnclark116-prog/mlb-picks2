"""Deterministic, leak-safe structural Phase1F challenger tests."""
import copy
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from nfl_phase1f_intelligence import (
    EvidenceError, VERSION, build_team_context, challenger_identity,
    joint_scripts, make_challenger_pack,
)

UTC=timezone.utc
KICK=datetime(2026,11,15,18,0,tzinfo=UTC)
CUTOFF=KICK-timedelta(minutes=90)


def unit(team, opp, *, coach="COACH-A", other=0):
    hist=[]
    opponent=[]
    for i in range(1,8):
        t=KICK-timedelta(days=7*i)
        hist.append({"kick":t,"s":2026,"w":11-i,"coach":"COACH-A" if i<4 else "OLD",
            "plays":61+i,"dropbacks":36+i,
            "rushes_designed":21+i,"rushes":25+i})
        opponent.append({"kick":t,"s":2026,"w":11-i,"coach":"OPP",
            "plays":65+i,"dropbacks":39+i,
            "rushes_designed":22+i,"rushes":27+i})
    return {"key":(2026,11,team),"kick":KICK,
            "last_info_ts":KICK-timedelta(days=6),
            "coach":coach,"team_hist":hist,"opp_allowed_hist":opponent,
            "players":[{"gid":team+"-qb1","pos":"QB",
                        "depth":("QB",1),"depth_ts":KICK-timedelta(days=2)},
                       {"gid":team+"-qb2","pos":"QB",
                        "depth":("QB",2),"depth_ts":KICK-timedelta(days=2)}],
            "rest":7,"opp_rest":6,"opp":opp}


def game(team):
    return {"key":(2026,11,team),"types":{
      "carry":{"ids":[team+"-rb"],"mu":25.0,"k":9.0,
               "pact24":np.array([.9]),"pact90":np.array([.98]),
               "P1":np.array([.6])},
      "target":{"ids":[team+"-wr"],"mu":30.,"k":10.0,
                "pact24":np.array([.95]),"pact90":np.array([.99]),
                "P1":np.array([.55])},
      "qb_att":{"ids":[team+"-qb1",team+"-qb2"],"mu":38.0,"k":12.,
                "P1":np.array([.42,.25]),"P0":np.array([.36,.19]),
                "pact24":np.array([.95,.6]),"pact90":np.array([.99,.4])}}}


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.teams=("BUF","MIN")
        self.pack={"games":{(2026,11,t):game(t) for t in self.teams},
                   "meta":{"target":{"other":.2,"alpha":1.0}}}
        self.units={ (2026,11,t):unit(t,self.teams[1-j])
                    for j,t in enumerate(self.teams)}

    def challenger(self):
        return make_challenger_pack(self.pack,self.units,2026,11,
                                    self.teams,"T90",CUTOFF)

    def test_original_pack_never_mutated_and_uses_depth_as_SOFT_prior(self):
        orig=copy.deepcopy(self.pack)
        out,contexts=self.challenger()
        self.assertEqual(contexts["BUF"]["qb_status"],
                         "RANK1_SOFT_QB_ALLOCATION_NOT_STARTER_VERIFICATION")
        self.assertEqual(out["games"][(2026,11,"BUF")]["types"]["qb_att"]
                         ["P1"][0], .42*1.30)
        self.assertEqual(self.pack["games"][(2026,11,"BUF")]
                         ["types"]["qb_att"]["P1"][0],.42)
        self.assertLessEqual(contexts["BUF"]["projected_plays"],
                             contexts["BUF"]["preselected_plays"]*1.18+1e-7)
        self.assertGreaterEqual(contexts["BUF"]["projected_plays"],
                             contexts["BUF"]["preselected_plays"]*.82-1e-7)
        self.assertEqual(self.pack["meta"],orig["meta"])
        self.assertEqual(out["meta"],orig["meta"])
        self.assertTrue(contexts["BUF"]["coach_status"].startswith("MATCHED_"))
        self.assertIn("NOT_IN_VERIFIED",contexts["BUF"]["weather_status"])

    def test_absent_starter_cannot_reappear_with_depth_rank(self):
        self.pack["games"][(2026,11,"BUF")]["types"]["qb_att"]["pact90"][0]=0.
        out,contexts=self.challenger()
        self.assertEqual(contexts["BUF"]["qb_status"],
                         "RANK1_QB_UNAVAILABLE_NOT_ASSIGNED")
        self.assertEqual(out["games"][(2026,11,"BUF")]
                         ["types"]["qb_att"]["P1"][0],.42)

    def test_no_depth_or_unknown_coach_is_explicit(self):
        self.units[(2026,11,"BUF")]["coach"]="NEW-COACH"
        self.units[(2026,11,"BUF")]["players"]=[]
        _,c=self.challenger()
        self.assertEqual(c["BUF"]["coach_status"],
                         "COACH_CONTINUITY_UNVERIFIED_FALLBACK_TO_TEAM")
        self.assertEqual(c["BUF"]["qb_status"],"UNVERIFIED_QB_DEPTH_CONTEXT")

    def test_future_and_after_cutoff_features_refused(self):
        u=self.units[(2026,11,"BUF")]
        u["team_hist"].append({"kick":KICK-timedelta(hours=2),"plays":99.,
            "dropbacks":90.,"rushes_designed":9.,"rushes":9.})
        with self.assertRaisesRegex(EvidenceError,"LEAKAGE"):
            self.challenger()
        u["team_hist"].pop()
        u["last_info_ts"]=KICK-timedelta(minutes=30)
        with self.assertRaisesRegex(EvidenceError,"LAST_INFO"):
            self.challenger()
        u["last_info_ts"]=KICK-timedelta(days=6)
        with self.assertRaisesRegex(EvidenceError,"AFTER_OR_AT_KICKOFF"):
            make_challenger_pack(self.pack,self.units,2026,11,
                                 self.teams,"T90",KICK)

    def test_missing_team_not_silently_imputed(self):
        self.units.pop((2026,11,"MIN"))
        with self.assertRaisesRegex(EvidenceError,"MISSING_AS_OF"):
            self.challenger()

    def test_joint_draws_share_game_pace_and_account_for_scrambles(self):
        cp,c=self.challenger()
        a=joint_scripts(c,self.teams,10000,123456,cp)
        b=joint_scripts(c,self.teams,10000,123456,cp)
        for t in self.teams:
            self.assertTrue(np.array_equal(a[t]["D"],b[t]["D"]))
            self.assertTrue(np.array_equal(a[t]["R"],b[t]["R"]))
            self.assertTrue(np.all(a[t]["R"]>=0))
            self.assertTrue(np.all(a[t]["D"]>=0))
            self.assertGreater(a[t]["D"].mean(),a[t]["R"].mean())
            self.assertAlmostEqual(c[t]["simulated_mean_plays"]+
                c[t]["simulated_mean_scrambles"],
                c[t]["simulated_mean_rushes"]+
                c[t]["simulated_mean_dropbacks"],places=6)
            self.assertAlmostEqual(c[t]["projected_plays"],
                c[t]["simulated_mean_plays"],delta=2)
        # Shared-game pace latent => positive cross-team correlation.
        x=a["BUF"]["D"]+a["BUF"]["R"]
        y=a["MIN"]["D"]+a["MIN"]["R"]
        self.assertGreater(float(np.corrcoef(x,y)[0,1]),.01)

    def test_unfit_simulation_and_identity_contract(self):
        cp,c=self.challenger()
        with self.assertRaisesRegex(EvidenceError,"INSUFFICIENT"):
            joint_scripts(c,self.teams,9999,1,cp)
        k=challenger_identity("frozen-phase1","snap-123")
        self.assertEqual(k,challenger_identity("frozen-phase1","snap-123"))
        self.assertNotEqual(k,challenger_identity("frozen-phase1","snap-124"))
        self.assertTrue(k.startswith("p1f-"))
        self.assertTrue(VERSION.startswith("nfl-phase1f-"))


if __name__=="__main__":
    unittest.main()
