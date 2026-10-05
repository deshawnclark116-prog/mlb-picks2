#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import nfl_v2_phase1e_integrated as E


B={
  "outcomes":{
    oc:{
      "selected_opportunity_config":{"x":1},
      "selected_efficiency_config":{"y":1}
    } for oc in E.OUTCOMES
  }
}
D={
  "outcomes":{
    "rec":{"selected_config":{"r":1}},
    "rec_yds":{"selected_config":{"r":1}},
    "rush_yds":{"selected_config":{"r":1}},
    "pass_yds":{"selected_config":{"bad":1}},
  }
}


def test_receiving_uses_surviving_role_allocator():
    old_o=E.p1b.opportunity_receipt; old_e=E.p1b.efficiency_receipt; old_a=E.p1d.allocation
    try:
        E.p1b.opportunity_receipt=lambda *args,**kwargs:{
            "team_opportunity_projection":30.0,
            "player_opportunity_projection":6.0,
            "projected_role_share":0.20,
        }
        E.p1b.efficiency_receipt=lambda *args,**kwargs:{
            "final_efficiency_projection":10.0,
            "player_shrunk_efficiency":9.0,
            "opponent_shrunk_allowed_efficiency":11.0,
        }
        E.p1d.allocation=lambda *args,**kwargs:{
            "predicted_shares":{"p":0.30},
            "share_sum":1.0,
            "roster_n":5,
        }
        r={"player_id":"p"}
        out=E.receipt([],{}, {},r,"rec_yds",B,D)
        assert out["role_source"]=="phase1d_surviving_coherent_allocator"
        assert abs(out["player_opportunity_projection"]-9.0)<1e-12
        assert abs(out["point_projection"]-90.0)<1e-12
    finally:
        E.p1b.opportunity_receipt=old_o; E.p1b.efficiency_receipt=old_e; E.p1d.allocation=old_a


def test_passing_keeps_phase1b_qb_path():
    old_o=E.p1b.opportunity_receipt; old_e=E.p1b.efficiency_receipt; old_a=E.p1d.allocation
    called={"allocation":False}
    try:
        E.p1b.opportunity_receipt=lambda *args,**kwargs:{
            "team_opportunity_projection":35.0,
            "player_opportunity_projection":32.0,
            "projected_role_share":0.9142857143,
        }
        E.p1b.efficiency_receipt=lambda *args,**kwargs:{
            "final_efficiency_projection":7.5,
            "player_shrunk_efficiency":7.4,
            "opponent_shrunk_allowed_efficiency":7.7,
        }
        def nope(*args,**kwargs):
            called["allocation"]=True
            raise AssertionError("generic QB allocator must not be called")
        E.p1d.allocation=nope
        out=E.receipt([],{}, {},{"player_id":"qb"},"pass_yds",B,D)
        assert not called["allocation"]
        assert out["role_source"]=="phase1b_qb_opportunity_path"
        assert abs(out["point_projection"]-240.0)<1e-12
    finally:
        E.p1b.opportunity_receipt=old_o; E.p1b.efficiency_receipt=old_e; E.p1d.allocation=old_a


if __name__=="__main__":
    test_receiving_uses_surviving_role_allocator()
    test_passing_keeps_phase1b_qb_path()
    print("NFL V2 Phase 1E integration tests: PASS")
