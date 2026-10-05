#!/usr/bin/env python3
import csv, tempfile, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import nfl_v2_phase1a_direct as p1a
import nfl_v2_phase1b_opportunity as p1b
import nfl_v2_competent_human_baseline as H


def fixture():
    players=[]; team={}; opp={}
    for w in (1,2,3):
        for pid,tg in (("p1",6+w),("p2",4-w)):
            players.append({"season":2024,"week":w,"player_id":pid,"team":"A","opponent":"B","position":"WR",
                            "targets":float(tg),"receiving_yards":float(tg*10),"receptions":float(max(0,tg-1)),
                            "carries":0.0,"rushing_yards":0.0,"attempts":0.0,"passing_yards":0.0})
        team[(2024,w,"A")]={"targets":10.0,"receiving_yards":100.0,"receptions":8.0,"carries":20.0,"rushing_yards":90.0,"attempts":30.0,"passing_yards":220.0}
        opp[(2024,w,"A")]="B"
        players.append({"season":2024,"week":w,"player_id":f"x{w}","team":"C","opponent":"A","position":"WR",
                        "targets":5.0,"receiving_yards":45.0,"receptions":4.0,"carries":0.0,"rushing_yards":0.0,"attempts":0.0,"passing_yards":0.0})
        team[(2024,w,"C")]={"targets":10.0,"receiving_yards":90.0,"receptions":8.0,"carries":20.0,"rushing_yards":90.0,"attempts":30.0,"passing_yards":220.0}
        opp[(2024,w,"C")]="A"
    target={"season":2024,"week":4,"player_id":"p1","team":"A","opponent":"B","position":"WR",
            "targets":9.0,"receiving_yards":95.0,"receptions":8.0,"carries":0.0,"rushing_yards":0.0,"attempts":0.0,"passing_yards":0.0}
    players.append(target)
    team[(2024,4,"A")]={"targets":10.0,"receiving_yards":100.0,"receptions":9.0,"carries":20.0,"rushing_yards":90.0,"attempts":30.0,"passing_yards":220.0}
    opp[(2024,4,"A")]="B"
    players.sort(key=lambda r:(r["season"],r["week"],r["team"],r["player_id"]))
    return players,team,opp,target


def roster():
    f=tempfile.NamedTemporaryFile("w",newline="",delete=False,suffix=".csv")
    with f:
        w=csv.DictWriter(f,fieldnames=["season","week","team","gsis_id","position","status","game_type"])
        w.writeheader()
        w.writerow({"season":2024,"week":4,"team":"A","gsis_id":"p1","position":"WR","status":"RES","game_type":"REG"})
        w.writerow({"season":2024,"week":4,"team":"A","gsis_id":"p2","position":"WR","status":"ACT","game_type":"REG"})
    return f.name


def test_roster_status_is_ignored_for_historical_benchmark():
    H.load_roster_membership([roster()])
    ids={x["player_id"] for x in H._ROSTERS[(2024,4,"A")]}
    assert ids=={"p1","p2"}


def test_projection_is_prior_only_and_coherent():
    players,team,opp,target=fixture()
    p1a.build_indexes(players,team,opp); p1b.build_extra_indexes(players,team,opp); H.build_indexes(players,team)
    H.load_roster_membership([roster()])
    sh=H.coherent_share(players,team,target,"rec_yds")
    assert sh is not None and abs(sh["share_sum"]-1)<1e-12
    assert 0 < sh["player_share"] < 1
    p=H.receipt(players,team,opp,target,"rec_yds")
    assert p is not None
    assert p["point_projection"] < 200
    assert p["player_opportunity_projection"] < 10


if __name__=="__main__":
    test_roster_status_is_ignored_for_historical_benchmark()
    test_projection_is_prior_only_and_coherent()
    print("NFL competent-human baseline tests: PASS")
