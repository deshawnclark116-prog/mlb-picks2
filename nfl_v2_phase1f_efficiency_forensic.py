#!/usr/bin/env python3
"""NFL V2 Phase 1F: efficiency forensic and fixed efficiency swap.

Research-only. No fitting, Monte Carlo, or sportsbook inputs.

Holding Phase1E predicted opportunities fixed, compare:
1) frozen Phase1B efficiency;
2) the fixed competent-human efficiency blend;
3) oracle opportunity (actual workload x predicted efficiency);
4) oracle efficiency (predicted workload x actual efficiency).

This directly answers whether remaining final-stat error is coming from workload
or production per opportunity.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import nfl_v2_phase1a_direct as p1a
import nfl_v2_phase1b_opportunity as p1b
import nfl_v2_phase1d_role_allocation as p1d
import nfl_v2_phase1e_integrated as p1e
import nfl_v2_competent_human_baseline as human

OUTCOMES=p1e.OUTCOMES
SPEC=p1a.SPEC


def mean(xs):
    xs=[x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None


def human_efficiency(players,team_totals,team_opp,r,outcome):
    if outcome=="pass_yds":
        q=human.qb_receipt(players,team_totals,team_opp,r)
        return None if q is None else q["final_efficiency_projection"]
    q=human.efficiency_receipt(players,team_totals,team_opp,r,outcome)
    return None if q is None else q["projection"]


def fixed_rows(players,season,outcome,max_week=18):
    source=[r for r in players if r["season"]==season and 1<=r["week"]<=max_week]
    return p1b.fixed_meaningful_rows(players,source,outcome)


def evaluate(players,team_totals,team_opp,rows,outcome,b,d):
    sp=SPEC[outcome]
    recs=[]
    for r in rows:
        base=p1e.receipt(players,team_totals,team_opp,r,outcome,b,d)
        he=human_efficiency(players,team_totals,team_opp,r,outcome)
        if base is None or he is None:
            continue
        y=r[sp["value"]]; ao=r[sp["opp"]]
        po=base["player_opportunity_projection"]
        be=base["final_efficiency_projection"]
        actual_eff=None if ao<=0 else y/ao

        bpoint=po*be
        hpoint=po*he
        oracle_opp_b=ao*be
        oracle_opp_h=ao*he
        oracle_eff=None if actual_eff is None else po*actual_eff

        recs.append({
            "season":r["season"],"week":r["week"],"player_id":r["player_id"],
            "team":r["team"],"opponent":r["opponent"],"position":r.get("position"),
            "outcome":outcome,"actual":y,"actual_opportunities":ao,
            "predicted_opportunities":po,
            "phase1b_efficiency":be,"human_efficiency":he,"actual_efficiency":actual_eff,
            "phase1e_point":bpoint,"human_eff_swap_point":hpoint,
            "phase1e_abs_error":abs(bpoint-y),
            "human_eff_swap_abs_error":abs(hpoint-y),
            "phase1b_efficiency_abs_error":None if actual_eff is None else abs(be-actual_eff),
            "human_efficiency_abs_error":None if actual_eff is None else abs(he-actual_eff),
            "oracle_opportunity_phase1b_eff_abs_error":abs(oracle_opp_b-y),
            "oracle_opportunity_human_eff_abs_error":abs(oracle_opp_h-y),
            "oracle_efficiency_abs_error":None if oracle_eff is None else abs(oracle_eff-y),
        })
    return recs


def summarize(rows,outcome):
    if not rows:
        return {"n":0}
    phase=[r["phase1e_abs_error"] for r in rows]
    hybrid=[r["human_eff_swap_abs_error"] for r in rows]
    out={
        "n":len(rows),
        "phase1e_mae":mean(phase),
        "human_eff_swap_mae":mean(hybrid),
        "human_eff_swap_minus_phase1e_mae":mean(hybrid)-mean(phase),
        "phase1b_efficiency_per_opp_mae":mean(r["phase1b_efficiency_abs_error"] for r in rows),
        "human_efficiency_per_opp_mae":mean(r["human_efficiency_abs_error"] for r in rows),
        "oracle_opportunity_phase1b_eff_mae":mean(r["oracle_opportunity_phase1b_eff_abs_error"] for r in rows),
        "oracle_opportunity_human_eff_mae":mean(r["oracle_opportunity_human_eff_abs_error"] for r in rows),
        "oracle_efficiency_mae":mean(r["oracle_efficiency_abs_error"] for r in rows),
        "hybrid_row_wins":sum(r["human_eff_swap_abs_error"]<r["phase1e_abs_error"] for r in rows),
        "phase1e_row_wins":sum(r["phase1e_abs_error"]<r["human_eff_swap_abs_error"] for r in rows),
    }
    if outcome.endswith("_yds"):
        out["hybrid_within"]={str(t):sum(e<=t for e in hybrid)/len(hybrid) for t in (5,10,15,20,25,30,40,50)}
    else:
        out["hybrid_within"]={str(t):sum(e<=t for e in hybrid)/len(hybrid) for t in (0,1,2,3)}
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stats",action="append",required=True)
    ap.add_argument("--roster",action="append",required=True)
    ap.add_argument("--phase1b-snapshot",required=True)
    ap.add_argument("--phase1d-snapshot",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    players,team_totals,team_opp=p1a.load_stats(a.stats)
    p1a.build_indexes(players,team_totals,team_opp)
    p1b.build_extra_indexes(players,team_totals,team_opp)
    p1d.build_context(players,team_totals)
    p1d.load_rosters(a.roster)
    human.build_indexes(players,team_totals,team_opp)

    b=json.loads(Path(a.phase1b_snapshot).read_text())
    d=json.loads(Path(a.phase1d_snapshot).read_text())

    report={
        "schema":"nfl-v2-phase1f-efficiency-forensic-v1",
        "status":"BURNED_RESEARCH_ONLY",
        "sportsbook_inputs_used":False,
        "monte_carlo_used":False,
        "new_fitting_used":False,
        "question":"Holding Phase1E opportunities fixed, which efficiency formulation is less wrong and how much error remains even with actual workload?",
        "outcomes":{},
    }

    for oc in OUTCOMES:
        r25=evaluate(players,team_totals,team_opp,fixed_rows(players,2025,oc),oc,b,d)
        r26=evaluate(players,team_totals,team_opp,fixed_rows(players,2026,oc,4),oc,b,d)
        report["outcomes"][oc]={
            "diagnostic_2025":summarize(r25,oc),
            "diagnostic_2026_wk1_4":summarize(r26,oc),
            "largest_2025_phase1e_misses":sorted(r25,key=lambda x:-x["phase1e_abs_error"])[:20],
            "largest_2025_hybrid_misses":sorted(r25,key=lambda x:-x["human_eff_swap_abs_error"])[:20],
        }

    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps({oc:{
        "2025":v["diagnostic_2025"],
        "2026_wk1_4":v["diagnostic_2026_wk1_4"],
    } for oc,v in report["outcomes"].items()},indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
