#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nfl_v2_phase1b_opportunity import ewma, role_projection, opportunity_receipt, build_extra_indexes
import nfl_v2_phase1a_direct as p1a


def test_ewma_and_role_shift():
    assert abs(ewma([1.0, 2.0, 3.0], 1.0) - 2.0) < 1e-12
    base = role_projection([0.20, 0.20, 0.40], 0.75, 0.0)
    shifted = role_projection([0.20, 0.20, 0.40], 0.75, 0.50)
    assert shifted > base
    assert 0.0 <= shifted <= 1.0


def fixture():
    players=[]
    team={}
    team_opp={}
    for week in (1,2,3):
        # A offense, B defense. Player p1 owns half the carries.
        players.append({
            "season":2024,"week":week,"player_id":"p1","team":"A","opponent":"B","position":"RB",
            "carries":10.0,"rushing_yards":50.0,"targets":0.0,"receiving_yards":0.0,
            "receptions":0.0,"attempts":0.0,"passing_yards":0.0,
        })
        team[(2024,week,"A")]={"carries":20.0,"rushing_yards":100.0,"targets":30.0,"receiving_yards":220.0,"receptions":20.0,"attempts":32.0,"passing_yards":240.0}
        team_opp[(2024,week,"A")]="B"

        players.append({
            "season":2024,"week":week,"player_id":"q","team":"C","opponent":"A","position":"RB",
            "carries":8.0,"rushing_yards":32.0,"targets":0.0,"receiving_yards":0.0,
            "receptions":0.0,"attempts":0.0,"passing_yards":0.0,
        })
        team[(2024,week,"C")]={"carries":16.0,"rushing_yards":64.0,"targets":28.0,"receiving_yards":200.0,"receptions":18.0,"attempts":30.0,"passing_yards":220.0}
        team_opp[(2024,week,"C")]="A"

        # Teams D/E add league-history coverage and games against B.
        for idx,(tm,opp,carries) in enumerate((("D","B",18.0),("E","D",22.0),("F","E",19.0),("G","F",21.0),("H","G",20.0),("I","H",17.0))):
            pid=f"{tm}{week}"
            players.append({
                "season":2024,"week":week,"player_id":pid,"team":tm,"opponent":opp,"position":"RB",
                "carries":carries/2,"rushing_yards":carries*2.0,"targets":0.0,"receiving_yards":0.0,
                "receptions":0.0,"attempts":0.0,"passing_yards":0.0,
            })
            team[(2024,week,tm)]={"carries":carries,"rushing_yards":carries*4.0,"targets":25.0,"receiving_yards":180.0,"receptions":16.0,"attempts":27.0,"passing_yards":200.0}
            team_opp[(2024,week,tm)]=opp
    target={
        "season":2024,"week":4,"player_id":"p1","team":"A","opponent":"B","position":"RB",
        "carries":99.0,"rushing_yards":999.0,"targets":0.0,"receiving_yards":0.0,
        "receptions":0.0,"attempts":0.0,"passing_yards":0.0,
    }
    players.append(target)
    team[(2024,4,"A")]={"carries":99.0,"rushing_yards":999.0,"targets":99.0,"receiving_yards":999.0,"receptions":50.0,"attempts":99.0,"passing_yards":999.0}
    team_opp[(2024,4,"A")]="B"
    players.sort(key=lambda r:(r["season"],r["week"],r["team"],r["player_id"]))
    return players,team,team_opp,target


def test_opportunity_receipt_uses_prior_only():
    players,team,team_opp,target=fixture()
    p1a.build_indexes(players,team,team_opp)
    build_extra_indexes(players,team)
    cfg={
        "team_window":8,"league_window":96,"share_window":5,
        "team_decay":0.75,"offense_weight":0.60,"defense_weight":0.20,
        "role_decay":0.75,"shift_gain":0.25,
    }
    rec=opportunity_receipt(players,team,team_opp,target,"rush_yds",cfg)
    assert rec is not None
    # Current-week 99 carries must not leak into the pregame team projection.
    assert rec["team_opportunity_projection"] < 40
    assert rec["player_opportunity_projection"] < 25
    assert 0 <= rec["projected_role_share"] <= 1


if __name__ == "__main__":
    test_ewma_and_role_shift()
    test_opportunity_receipt_uses_prior_only()
    print("NFL V2 Phase 1B opportunity tests: PASS")