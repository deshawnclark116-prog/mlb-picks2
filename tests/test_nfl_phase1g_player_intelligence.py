"""Phase1G: real math tests for roster, teammate shares, defensive evidence, weather, game scripts.

They test how probabilities/volumes change in the simulator inputs, not just
whether a status flag was set. Frozen Phase1C math remains unchanged.
"""
import copy
import csv
import json
import math
import tempfile
import unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import nfl_phase1g_player_intelligence as G

UTC=timezone.utc
KICK=datetime(2026,11,15,18,tzinfo=UTC)
CUT=KICK-timedelta(minutes=90)
S=2026
W=11


def h(team,target,car,share,carryshare,snap):
    result=[]
    for i in range(8,0,-1):
        result.append({"team":team,"s":2026,"w":11-i,
                       "kick":KICK-timedelta(days=i*7),
                       "pos":"WR","tgt":target,"car":car,
                       "tgt_sh":share*(1.6 if i<3 else 1),
                       "car_sh":carryshare,"routes":None,
                       "route_rt":None,"snap_pct":snap,
                       "played":True})
    return result


def unit(team,opp):
    return {"key":(S,W,team),"kick":KICK,"opp":opp,
       "players":[
         {"gid":team+"wr1","pos":"WR","hist":h(team,9,0,.24,0,.94)},
         {"gid":team+"wr2","pos":"WR","hist":h(team,4,0,.14,0,.65)},
         {"gid":team+"rb","pos":"RB","hist":h(team,1,18,.04,.56,.70)}],
       "team_hist":[{"kick":KICK-timedelta(days=7*i),
                      "off_td":2.0+i*.1} for i in range(1,6)],
       "opp_allowed_hist":[{"kick":KICK-timedelta(days=7*i),
                      "off_td":2.5} for i in range(1,6)]}


def team(team):
    return {"key":(S,W,team),
     "_phase1f_horizon":"T90",
     "types":{
      "target":{"ids":[team+"wr1",team+"wr2",team+"rb"],
                "P1":np.array([.35,.23,.12],float),
                "pact90":np.array([.98,0.0,.92],float)},
      "carry":{"ids":[team+"rb",team+"qb"],
               "P1":np.array([.7,.10],float),
               "pact90":np.array([.94,.97],float)},
      "qb_att":{"ids":[team+"qb"],
                "P1":np.array([.95],float),
                "pact90":np.array([.98],float),
                "k":18.,"mu":39.0}}},
     "meta":"unused"}


class Phase1GRoles(unittest.TestCase):
    def setUp(self):
        self.teams=["BUF","MIN"]
        self.pack={"games":{(S,W,t):team(t) for t in self.teams},
                   "meta":{"carry":{"other":.1},"target":{"other":.2}}}
        self.units={(S,W,t):unit(t,"MIN" if t=="BUF" else "BUF")
                    for t in self.teams}

    def test_role_reallocation_and_actual_inactive_zero(self):
        original=copy.deepcopy(self.pack)
        result=G.apply_player_roles(self.pack,self.units,S,W,self.teams,"T90",CUT)
        t=self.pack["games"][(S,W,"BUF")]["types"]["target"]
        self.assertEqual(float(t["P1"][1]),0.)
        self.assertNotEqual(float(t["P1"][0]),float(original["games"][(S,W,"BUF")]["types"]["target"]["P1"][0]))
        self.assertGreater(float(t["P1"][0]),0.)
        self.assertGreater(len(result["BUF"]["target"]["player_evidence"]),0)
        self.assertEqual(result["BUF"]["target"]["inactive_count"],1)
        self.assertLessEqual(t["P1"].sum(),
                             1.15*original["games"][(S,W,"BUF")]["types"]["target"]["P1"].sum()+1e-6)
        self.assertEqual(float(original["games"][(S,W,"BUF")]["types"]["target"]["P1"][1]),.23)

    def test_player_history_contamination_fails_closed(self):
        self.units[(S,W,"BUF")]["players"][0]["hist"][0]["kick"]=KICK
        with self.assertRaisesRegex(G.EvidenceError,"AFTER_INFORMATION_CUTOFF"):
            G.apply_player_roles(self.pack,self.units,S,W,self.teams,"T90",CUT)

    def test_no_healthy_player_is_created_from_empty_carry_history(self):
        t=self.pack["games"][(S,W,"BUF")]["types"]["carry"]
        t["pact90"][1]=0.
        G.apply_player_roles(self.pack,self.units,S,W,self.teams,"T90",CUT)
        self.assertEqual(t["P1"][1],0.)

    def test_matchup_efficiency_not_posthoc_and_no_input_mutation(self):
        x={( "MIN","receiving","WR"):{"ratio":1.12,"events":99,"n_games":5},
           ( "MIN","rushing","RB"):{"ratio":.90,"events":80,"n_games":5}}
        base={"air":np.array([[.35,.45,.20],[.35,.45,.20]],float),
              "rush":np.array([[.25,.55,.20],[.25,.55,.20]],float),
              "catch":np.ones((2,4))*.62}
        idx={"rec":{(S,W,"BUFwr1"):0},"rush":{(S,W,"BUFrb"):0}}
        const=SimpleNamespace(air_v=np.array([-2.,5.,24.]),
                              rush_v=np.array([-4.,4.,16.]))
        prior=copy.deepcopy(base)
        out,receipt=G.matchup_event_efficiencies(base,idx,const,
                            self.units,self.teams,S,W,x)
        self.assertAlmostEqual(out["air"][0].sum(),1.,places=6)
        self.assertAlmostEqual(out["rush"][0].sum(),1.,places=6)
        self.assertGreater(out["air"][0,2],prior["air"][0,2])
        self.assertLess(out["rush"][0,2],prior["rush"][0,2])
        self.assertTrue(np.array_equal(base["air"],prior["air"]))
        self.assertIn("BUFwr1/receiving",receipt["BUF"])

    def test_defense_uses_only_pre_cutoff_opponent_games(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            data={"game":{}}
            lines=[]
            for team,opp,avg in (("DAL","MIN",10),("SF","BUF",5)):
                for i in range(1,5):
                    data["game"][(S,i,team)]={"kick":KICK-timedelta(days=(12-i)*7),
                                             "opp":opp}
                    lines.append({"season_type":"REG","week":i,"team":team,
                                  "position":"WR","targets":15,
                                  "receiving_yards":avg*15,
                                  "carries":0,"rushing_yards":0})
            # Add target game with huge misleading yards; no effect allowed.
            data["game"][(S,W,"DAL")]={"kick":KICK,"opp":"MIN"}
            lines.append({"season_type":"REG","week":W,"team":"DAL",
                          "position":"WR","targets":15,
                          "receiving_yards":9999,
                          "carries":0,"rushing_yards":0})
            with (folder/f"stats_player_week_{S}.csv").open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=lines[0]);w.writeheader();w.writerows(lines)
            d=G.defensive_rates(folder,SimpleNamespace(**data),CUT,S)
            self.assertIn(("MIN","receiving","WR"),d)
            self.assertGreater(d[("MIN","receiving","WR")]["ratio"],1.)
            self.assertLessEqual(d[("MIN","receiving","WR")]["ratio"],1.12)

    def test_only_authenticated_nws_weather_captured_before_cutoff(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"receipt.json"
            d={"game_id":"2026_11_BUF_MIN",
               "kickoff_utc":KICK.isoformat(),"venue_id":"BUF",
               "captured_at_utc":(CUT-timedelta(minutes=2)).isoformat(),
               "forecast_issued_at_utc":(CUT-timedelta(minutes=10)).isoformat(),
               "source_url":"https://api.weather.gov/gridpoints/BUF/10,20/forecast",
               "roof_status":"open","wind_mph":18,
               "precipitation_probability_pct":20}
            p.write_text(json.dumps(d))
            evidence,_=G.load_weather_evidence(p,d["game_id"],KICK,CUT,"BUF")
            self.assertTrue(evidence["outdoor_effect"])
            self.assertEqual(len(evidence["sha256"]),64)
            d["captured_at_utc"]=(CUT+timedelta(minutes=2)).isoformat()
            p.write_text(json.dumps(d))
            with self.assertRaisesRegex(G.EvidenceError,"NOT_ORIGINALLY_PREGAME"):
                G.load_weather_evidence(p,d["game_id"],KICK,CUT,"BUF")
            d["captured_at_utc"]=(CUT-timedelta(minutes=2)).isoformat()
            d["source_url"]="https://bad-site.example.com/forecasts"
            p.write_text(json.dumps(d))
            with self.assertRaisesRegex(G.EvidenceError,"WEATHER_SOURCE_NOT_NWS"):
                G.load_weather_evidence(p,d["game_id"],KICK,CUT,"BUF")

    def test_joint_script_conserves_real_plays_and_can_condition_weather(self):
        pack={"games":{(S,W,t):{
             "types":{"carry":{"k":12.},"qb_att":{"k":12.}}}
             for t in self.teams}}
        c={t:{"projected_plays":62.,"pass_fraction":.60,
              "scramble_rate_of_dropbacks":.08,
              "expected_offensive_tds":2.4} for t in self.teams}
        dry=G.correlated_script(copy.deepcopy(c),self.teams,10000,77,pack)
        wet=G.correlated_script(copy.deepcopy(c),self.teams,10000,77,pack,
          {"outdoor_effect":True,"wind_mph":30.,"precipitation_probability_pct":80})
        for t in self.teams:
            self.assertGreater(dry[t]["D"].mean(),wet[t]["D"].mean())
            self.assertGreater(dry[t]["D"].mean(),0)
            self.assertGreater(dry[t]["R"].mean(),0)
        self.assertEqual(G.model_identity("v0","snap","sha",10000),
                         G.model_identity("v0","snap","sha",10000))
        self.assertNotEqual(G.model_identity("v0","snap","sha",10000),
                            G.model_identity("v0","snap","sha2",10000))


if __name__=="__main__":unittest.main()
