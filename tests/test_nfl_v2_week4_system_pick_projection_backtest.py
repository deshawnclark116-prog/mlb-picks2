#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import nfl_v2_week4_system_pick_projection_backtest as B


def yard(outcome,prob=.70,pa=.90,name="p",rid="1"):
    # q[34] determines the highest candidate search ceiling.
    q=[0.0]*99
    # 70 of 99 entries >= 10 -> first index 29 -> 70%.
    for i in range(29,99):
        q[i]=10.0
    return {
        "game_id":"2026_04_A_B","player_name":name,"p_active":pa,
        "outcome":outcome,"quantile_grid_99":q,"cdf_lattice":None,
        "id":rid,
    }


def count(outcome,pa=.90,name="p",rid="2"):
    # P(X>=1) = 0.70
    return {
        "game_id":"2026_04_A_B","player_name":name,"p_active":pa,
        "outcome":outcome,"quantile_grid_99":None,
        "cdf_lattice":{"step":1,"cdf":[0.30,0.80,1.0]},
        "id":rid,
    }


def test_alt65_yardage_and_count():
    a=B.alt65(yard("rush_yds"))
    assert a["milestone"]==10 and abs(a["probability"]-.70)<1e-12
    c=B.alt65(count("rec"))
    assert c["milestone"]==1 and abs(c["probability"]-.70)<1e-12


def test_system_selection_caps_per_game_and_uses_pactive():
    rows=[]
    for i in range(12):
        r=count("rec",pa=.90+i*.001,name=f"p{i:02d}",rid=str(i))
        rows.append(r)
    rows.append(count("rec",pa=.74,name="excluded",rid="x"))
    picks=B.system_picks(rows)
    assert len(picks)==10
    assert all(p["player_name"]!="excluded" for p in picks)
    # Equal model probability: higher p_active ranks first.
    assert picks[0]["player_name"]=="p11"


def test_actual_atd_sums_rush_and_receiving_tds():
    assert B.actual_value("atd",{"rushing_tds":"1","receiving_tds":"2"})==3.0


if __name__=="__main__":
    test_alt65_yardage_and_count()
    test_system_selection_caps_per_game_and_uses_pactive()
    test_actual_atd_sums_rush_and_receiving_tds()
    print("NFL Week 4 system-pick projection backtest tests: PASS")
