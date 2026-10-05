#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import nfl_v2_phase1g_receiving_mechanics as R


def seed():
    R._PLAYER_EVENTS.clear(); R._DEF_EVENTS.clear(); R._POS_EVENTS.clear(); R._CACHE.clear()
    # Prior WR history: 10 targets, 7 catches, 70 yards = 10 YPR, 7 YPT.
    evs=[]
    for i in range(10):
        comp=1.0 if i<7 else 0.0
        y=10.0 if comp else 0.0
        e={"key":(2024,1+i//5),"targets":1.0,"receptions":comp,"rec_yards":y,
           "completed_air":6.0 if comp else 0.0,"yac":4.0 if comp else 0.0}
        evs.append(e)
    R._PLAYER_EVENTS["p"]=list(evs)
    R._DEF_EVENTS[("B","WR")]=list(evs)
    R._POS_EVENTS["WR"]=list(evs)*20
    for d in (R._PLAYER_EVENTS,R._DEF_EVENTS,R._POS_EVENTS):
        for k in d: d[k].sort(key=lambda x:x["key"])


def cfg():
    return {"player_shrink_targets":10.0,"defense_shrink_targets":40.0,
            "defense_weight":0.15,"player_window_games":5,"defense_window_games":5}


def test_mechanical_identity():
    seed()
    r={"season":2024,"week":4,"player_id":"p","team":"A","opponent":"B","position":"WR"}
    q=R.efficiency_receipt(r,cfg())
    assert q is not None
    assert abs(q["projected_catch_rate"]-0.7)<1e-9
    assert abs(q["projected_air_per_catch"]-6.0)<1e-9
    assert abs(q["projected_yac_per_catch"]-4.0)<1e-9
    assert abs(q["projected_yards_per_target"]-7.0)<1e-9


def test_target_week_never_enters_prior():
    seed()
    R._PLAYER_EVENTS["p"].append({"key":(2024,4),"targets":1.0,"receptions":1.0,
                                   "rec_yards":1000.0,"completed_air":900.0,"yac":100.0})
    vals=R.prior_events(R._PLAYER_EVENTS,"p",(2024,4),window_games=5)
    assert all(e["key"] < (2024,4) for e in vals)
    assert max(e["rec_yards"] for e in vals) < 1000


def test_shrinkage_moves_cold_sample_toward_position_prior():
    prior={"catch_rate":0.7,"air_per_catch":6.0,"yac_per_catch":4.0,"yards_per_target":7.0,"targets":100,"catches":70}
    events=[{"key":(2024,1),"targets":1.0,"receptions":1.0,"rec_yards":30.0,"completed_air":20.0,"yac":10.0}]
    q=R.shrunk_components(events,prior,10.0)
    assert 0.7 < q["catch_rate"] < 1.0
    assert 6.0 < q["air_per_catch"] < 20.0
    assert 4.0 < q["yac_per_catch"] < 10.0


if __name__=="__main__":
    test_mechanical_identity()
    test_target_week_never_enters_prior()
    test_shrinkage_moves_cold_sample_toward_position_prior()
    print("NFL V2 Phase 1G receiving mechanics tests: PASS")
