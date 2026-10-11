#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nfl_v2_phase1a_direct as D


def main():
    players=[]
    team={}
    oppmap={}
    # Three prior games for A/P vs changing opponents; B is target defense.
    for w,(c,y) in enumerate([(20,80),(24,120),(22,88)],1):
        r={"season":2025,"week":w,"player_id":"p","team":"A","opponent":"X","position":"RB",
           "carries":c/2,"rushing_yards":y/2,"targets":0.0,"receiving_yards":0.0,"receptions":0.0,"attempts":0.0,"passing_yards":0.0}
        players.append(r)
        team[(2025,w,"A")]={"carries":float(c),"rushing_yards":float(y),"targets":0.0,"receiving_yards":0.0,"receptions":0.0,"attempts":0.0,"passing_yards":0.0}
        oppmap[(2025,w,"A")]="X"
    # What defenses B and C allowed in their last three prior games.
    for w,c in enumerate([30,32,34],1):
        team[(2025,w,"Q"+str(w))]={"carries":float(c),"rushing_yards":float(c*5),"targets":0.0,"receiving_yards":0.0,"receptions":0.0,"attempts":0.0,"passing_yards":0.0}
        oppmap[(2025,w,"Q"+str(w))]="B"
    for w,c in enumerate([10,12,14],1):
        team[(2025,w,"R"+str(w))]={"carries":float(c),"rushing_yards":float(c*2),"targets":0.0,"receiving_yards":0.0,"receptions":0.0,"attempts":0.0,"passing_yards":0.0}
        oppmap[(2025,w,"R"+str(w))]="C"

    target={"season":2025,"week":4,"player_id":"p","team":"A","opponent":"B","position":"RB",
            "carries":99.0,"rushing_yards":999.0}
    rec=D.feature_receipt(players,team,oppmap,target,"rush_yds",3,3,5)
    assert rec is not None
    assert abs(rec["offense_recent_team_opportunities"]-22.0)<1e-12
    assert abs(rec["opponent_recent_allowed_opportunities"]-32.0)<1e-12
    assert abs(rec["player_recent_role_share"]-0.5)<1e-12
    p=D.project(rec,0.5,0.8)
    assert abs(p["team_opportunity_projection"]-27.0)<1e-12
    assert abs(p["player_opportunity_projection"]-13.5)<1e-12

    target_c={**target,"opponent":"C"}
    rec_c=D.feature_receipt(players,team,oppmap,target_c,"rush_yds",3,3,5)
    assert rec_c is not None
    pc=D.project(rec_c,0.5,0.8)
    assert pc["team_opportunity_projection"] < p["team_opportunity_projection"]
    assert pc["final_efficiency_projection"] < p["final_efficiency_projection"]
    assert pc["point_projection"] < p["point_projection"]

    # Target-week 999 yards never appears in prior history.
    assert D.prior_player_rows(players+[target],(2025,4),"p",None,8)[-1]["week"]==3
    print("NFL V2 Phase 1A direct-chain tests: PASS")


if __name__=="__main__":
    main()
