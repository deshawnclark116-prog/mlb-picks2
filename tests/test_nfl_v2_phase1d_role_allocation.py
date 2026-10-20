#!/usr/bin/env python3
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import nfl_v2_phase1a_direct as p1a
import nfl_v2_phase1b_opportunity as p1b
import nfl_v2_phase1d_role_allocation as R


def build_fixture():
    players=[]
    team={}
    oppmap={}
    for w in (1,2,3):
        p1_targets=(6,7,8)[w-1]
        p2_targets=10-p1_targets
        for pid,tg in (("p1",p1_targets),("p2",p2_targets)):
            players.append({
                "season":2024,"week":w,"player_id":pid,"team":"A","opponent":"B",
                "position":"WR","targets":float(tg),"receiving_yards":float(tg*10),
                "receptions":float(max(0,tg-1)),"carries":0.0,"rushing_yards":0.0,
                "attempts":0.0,"passing_yards":0.0,
            })
        team[(2024,w,"A")]={
            "targets":10.0,"receiving_yards":100.0,"receptions":8.0,
            "carries":20.0,"rushing_yards":90.0,"attempts":30.0,"passing_yards":220.0,
        }
        oppmap[(2024,w,"A")]="B"
        # Add league history for position priors.
        players.append({
            "season":2024,"week":w,"player_id":f"x{w}","team":"C","opponent":"D",
            "position":"WR","targets":5.0,"receiving_yards":50.0,"receptions":4.0,
            "carries":0.0,"rushing_yards":0.0,"attempts":0.0,"passing_yards":0.0,
        })
        team[(2024,w,"C")]={
            "targets":10.0,"receiving_yards":100.0,"receptions":8.0,
            "carries":20.0,"rushing_yards":90.0,"attempts":30.0,"passing_yards":220.0,
        }
        oppmap[(2024,w,"C")]="D"
    target={
        "season":2024,"week":4,"player_id":"p1","team":"A","opponent":"B",
        "position":"WR","targets":9.0,"receiving_yards":90.0,"receptions":8.0,
        "carries":0.0,"rushing_yards":0.0,"attempts":0.0,"passing_yards":0.0,
    }
    players.append(target)
    team[(2024,4,"A")]={
        "targets":10.0,"receiving_yards":100.0,"receptions":9.0,
        "carries":20.0,"rushing_yards":90.0,"attempts":30.0,"passing_yards":220.0,
    }
    oppmap[(2024,4,"A")]="B"
    players.sort(key=lambda r:(r["season"],r["week"],r["team"],r["player_id"]))
    return players,team,oppmap,target


def write_roster(p2_status="ACT"):
    f=tempfile.NamedTemporaryFile("w",newline="",delete=False,suffix=".csv")
    with f:
        w=csv.DictWriter(f,fieldnames=["season","week","team","gsis_id","position","status","game_type"])
        w.writeheader()
        for week in (1,2,3):
            w.writerow({"season":2024,"week":week,"team":"A","gsis_id":"p1","position":"WR","status":"ACT","game_type":"REG"})
            w.writerow({"season":2024,"week":week,"team":"A","gsis_id":"p2","position":"WR","status":"ACT","game_type":"REG"})
        w.writerow({"season":2024,"week":4,"team":"A","gsis_id":"p1","position":"WR","status":"ACT","game_type":"REG"})
        w.writerow({"season":2024,"week":4,"team":"A","gsis_id":"p2","position":"WR","status":p2_status,"game_type":"REG"})
    return f.name


def cfg():
    return {
        "share_window":3,"decay":1.0,"global_weight":0.0,
        "trend_gain":0.0,"cold_start_scale":0.05,
        "concentration_power":1.0,
    }


def test_coherent_allocation():
    players,team,oppmap,target=build_fixture()
    p1a.build_indexes(players,team,oppmap)
    p1b.build_extra_indexes(players,team,oppmap)
    R.build_context(players,team)
    path=write_roster("ACT")
    R.load_rosters([path])
    a=R.allocation(players,team,target,"rec_yds",cfg())
    assert a is not None
    assert abs(a["share_sum"]-1.0)<1e-12
    assert a["predicted_shares"]["p1"]>a["predicted_shares"]["p2"]


def test_absent_teammate_redistributes_share():
    players,team,oppmap,target=build_fixture()
    p1a.build_indexes(players,team,oppmap)
    p1b.build_extra_indexes(players,team,oppmap)
    R.build_context(players,team)
    path=write_roster("RES")
    R.load_rosters([path])
    a=R.allocation(players,team,target,"rec_yds",cfg())
    assert a is not None
    assert set(a["predicted_shares"])=={"p1"}
    assert abs(a["predicted_shares"]["p1"]-1.0)<1e-12


def test_target_week_stats_do_not_enter_prior_share():
    players,team,oppmap,target=build_fixture()
    vals=R.prior_share_series(players,team,(2024,4),"p1","rec_yds",3,"A")
    assert len(vals)==3
    assert max(vals)<0.9


if __name__=="__main__":
    test_coherent_allocation()
    test_absent_teammate_redistributes_share()
    test_target_week_stats_do_not_enter_prior_share()
    print("NFL V2 Phase 1D-R role-allocation tests: PASS")
