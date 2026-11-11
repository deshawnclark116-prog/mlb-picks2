import copy
import unittest
from datetime import datetime,timedelta,timezone
import numpy as np
from nfl_phase1g_player_roles import adjust_team,RoleEvidenceError
UTC=timezone.utc
K=datetime(2026,11,15,18,tzinfo=UTC)
C=K-timedelta(minutes=90)

def fixture():
    g={"key":(2026,11,"BUF"),"types":{
       "target":{"ids":["WR1","WR2"],"P1":np.array([.40,.35]),
                 "pact90":np.array([1.,1.]),"pact24":np.array([1.,1.])},
       "carry":{"ids":["RB1","RB2"],"P1":np.array([.48,.32]),
                "pact90":np.array([1.,1.]),"pact24":np.array([1.,1.])}}}
    def player(pid,sh,key,role):
        return {"gid":pid,"max_info_ts_T90":C-timedelta(days=1),
            "hist":[{"kick":K-timedelta(days=7*i),"team":"BUF",
              "played":True,key:sh,"route_rt":role,"snap_pct":role}
              for i in range(1,7)]}
    u={"key":(2026,11,"BUF"),"kick":K,
       "players":[player("WR1",.65,"tgt_sh",.9),
                  player("WR2",.15,"tgt_sh",.5),
                  player("RB1",.65,"car_sh",.8),
                  player("RB2",.15,"car_sh",.35)]}
    return g,u

class Roles(unittest.TestCase):
    def test_role_shifts_are_nonzero_conservative_and_zero_sum(self):
        g,u=fixture();old=copy.deepcopy(g)
        out,receipts=adjust_team(g,u,"T90",C)
        self.assertEqual(len(receipts),4)
        for group in ("target","carry"):
            a=out["types"][group]["P1"]
            b=g["types"][group]["P1"]
            self.assertGreater(a[0],b[0])
            self.assertLess(a[1],b[1])
            self.assertAlmostEqual(a.sum(),b.sum())
            self.assertTrue(np.all(a>=0))
        self.assertTrue(np.array_equal(old["types"]["target"]["P1"],
                                       g["types"]["target"]["P1"]))

    def test_future_player_data_is_rejected(self):
        g,u=fixture()
        u["players"][0]["hist"][0]["kick"]=K+timedelta(days=1)
        with self.assertRaisesRegex(RoleEvidenceError,"LEAKAGE"):
            adjust_team(g,u,"T90",C)

    def test_after_cutoff_receipt_rejected(self):
        g,u=fixture()
        u["players"][0]["max_info_ts_T90"]=K
        with self.assertRaisesRegex(RoleEvidenceError,"AFTER_CUTOFF"):
            adjust_team(g,u,"T90",C)

    def test_unavailable_player_never_resurrected(self):
        g,u=fixture()
        g["types"]["target"]["pact90"][0]=0
        out,_=adjust_team(g,u,"T90",C)
        self.assertEqual(out["types"]["target"]["P1"][0],0.)
        self.assertAlmostEqual(out["types"]["target"]["P1"][1],
                               g["types"]["target"]["P1"].sum())
        self.assertAlmostEqual(out["types"]["target"]["P1"].sum(),
                               g["types"]["target"]["P1"].sum())

    def test_all_receivers_unavailable_keeps_unknown_bucket_unfilled(self):
        g,u=fixture()
        g["types"]["target"]["pact90"][:]=0.
        out,_=adjust_team(g,u,"T90",C)
        self.assertTrue(np.all(out["types"]["target"]["P1"]==0.))

    def test_no_postgame_game_result_in_role_history(self):
        g,u=fixture()
        u["players"][0]["hist"][0]["kick"]=C-timedelta(hours=3)
        with self.assertRaisesRegex(RoleEvidenceError,"LEAKAGE"):
            adjust_team(g,u,"T90",C)

    def test_missing_history_never_invents_share(self):
        g,u=fixture()
        u["players"]=[]
        out,receipts=adjust_team(g,u,"T90",C)
        self.assertEqual(receipts,[])
        self.assertTrue(np.array_equal(out["types"]["carry"]["P1"],
                                       g["types"]["carry"]["P1"]))

if __name__=="__main__":unittest.main()
