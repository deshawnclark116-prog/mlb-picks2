#!/usr/bin/env python3
"""NFL V2 Phase 1E: integrate surviving role allocation into direct projections.

No new fitting occurs here.

Frozen pieces:
- Phase1B team-volume + efficiency configs (selected on burned 2024);
- Phase1D-R target-share allocator for receptions/receiving yards;
- Phase1D-R carry-share allocator for rushing yards;
- Phase1B QB opportunity path for passing yards because generic QB share
  allocation was explicitly rejected.

The test is architectural: does a role model that honestly improved role-share
error also improve *predicted player opportunity* and then the final direct
stat projection when coupled to predicted team volume and frozen efficiency?

No Monte Carlo. No sportsbook inputs.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import nfl_v2_phase1a_direct as p1a
import nfl_v2_phase1b_opportunity as p1b
import nfl_v2_phase1d_role_allocation as p1d

OUTCOMES=("rush_yds","rec_yds","rec","pass_yds")
SPEC=p1a.SPEC


def mean(xs):
    xs=[x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None


def receipt(players,team_totals,team_opp,r,outcome,p1b_snap,p1d_snap):
    b=p1b_snap["outcomes"][outcome]
    opp_cfg=b["selected_opportunity_config"]
    eff_cfg=b["selected_efficiency_config"]

    base_opp=p1b.opportunity_receipt(players,team_totals,team_opp,r,outcome,opp_cfg)
    eff=p1b.efficiency_receipt(players,team_totals,team_opp,r,outcome,eff_cfg)
    if base_opp is None or eff is None:
        return None

    if outcome=="pass_yds":
        player_opp=base_opp["player_opportunity_projection"]
        role_share=base_opp["projected_role_share"]
        role_source="phase1b_qb_opportunity_path"
        role_meta=None
    else:
        role_cfg=p1d_snap["outcomes"][outcome].get("selected_config")
        if not role_cfg:
            return None
        alloc=p1d.allocation(players,team_totals,r,outcome,role_cfg)
        if alloc is None:
            return None
        role_share=alloc["predicted_shares"].get(r["player_id"])
        if role_share is None:
            return None
        player_opp=base_opp["team_opportunity_projection"]*role_share
        role_source="phase1d_surviving_coherent_allocator"
        role_meta={
            "share_sum":alloc["share_sum"],
            "roster_n":alloc["roster_n"],
        }

    point=player_opp*eff["final_efficiency_projection"]
    return {
        "team_opportunity_projection":base_opp["team_opportunity_projection"],
        "role_share_projection":role_share,
        "role_source":role_source,
        "role_meta":role_meta,
        "player_opportunity_projection":player_opp,
        "final_efficiency_projection":eff["final_efficiency_projection"],
        "player_shrunk_efficiency":eff["player_shrunk_efficiency"],
        "opponent_shrunk_allowed_efficiency":eff["opponent_shrunk_allowed_efficiency"],
        "point_projection":point,
    }


def fixed_rows(players,season,outcome,max_week=18):
    source=[r for r in players if r["season"]==season and 1<=r["week"]<=max_week]
    return p1b.fixed_meaningful_rows(players,source,outcome)


def evaluate(players,team_totals,team_opp,rows,outcome,p1b_snap,p1d_snap):
    sp=SPEC[outcome]
    errs=[]; signed=[]; opp_err=[]; role_err=[]; receipts=[]
    for r in rows:
        p=receipt(players,team_totals,team_opp,r,outcome,p1b_snap,p1d_snap)
        if p is None:
            continue
        y=r[sp["value"]]; ao=r[sp["opp"]]
        point=p["point_projection"]
        errs.append(abs(point-y)); signed.append(point-y)
        opp_err.append(abs(p["player_opportunity_projection"]-ao))
        actual_team=team_totals.get((r["season"],r["week"],r["team"]),{}).get(sp["opp"],0.0)
        ashare=None if actual_team<=0 else ao/actual_team
        if ashare is not None:
            role_err.append(abs(p["role_share_projection"]-ashare))
        receipts.append({
            "season":r["season"],"week":r["week"],"player_id":r["player_id"],
            "team":r["team"],"opponent":r["opponent"],"position":r.get("position"),
            "outcome":outcome,"actual":y,"actual_opportunities":ao,
            "actual_role_share":ashare,
            "abs_error":abs(point-y),"error":point-y,
            "opportunity_abs_error":abs(p["player_opportunity_projection"]-ao),
            **p,
        })
    return {
        "n":len(errs),"mae":mean(errs),"bias":mean(signed),
        "player_opportunity_mae":mean(opp_err),"role_share_mae":mean(role_err),
        "receipts":receipts,
    }


def summarize(ev,outcome):
    if not ev["n"]:
        return {"n":0}
    es=[r["abs_error"] for r in ev["receipts"]]
    out={
        "n":ev["n"],"mae":ev["mae"],"bias":ev["bias"],
        "median_absolute_error":statistics.median(es),
        "player_opportunity_mae":ev["player_opportunity_mae"],
        "role_share_mae":ev["role_share_mae"],
    }
    if outcome.endswith("_yds"):
        out["within"]={str(t):sum(e<=t for e in es)/len(es) for t in (5,10,15,20,25,30,40,50)}
        threshold=75 if outcome=="pass_yds" else 40
    else:
        out["within"]={str(t):sum(e<=t for e in es)/len(es) for t in (0,1,2,3)}
        threshold=3
    out["catastrophic_miss_rate"]=sum(e>threshold for e in es)/len(es)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stats",action="append",required=True)
    ap.add_argument("--roster",action="append",required=True)
    ap.add_argument("--phase1b-snapshot",required=True)
    ap.add_argument("--phase1d-snapshot",required=True)
    ap.add_argument("--human-snapshot",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    players,team_totals,team_opp=p1a.load_stats(a.stats)
    p1a.build_indexes(players,team_totals,team_opp)
    p1b.build_extra_indexes(players,team_totals,team_opp)
    p1d.build_context(players,team_totals)
    roster_meta=p1d.load_rosters(a.roster)

    b=json.loads(Path(a.phase1b_snapshot).read_text())
    d=json.loads(Path(a.phase1d_snapshot).read_text())
    h=json.loads(Path(a.human_snapshot).read_text())

    report={
        "schema":"nfl-v2-phase1e-integrated-direct-v1",
        "status":"BURNED_RESEARCH_ONLY",
        "sportsbook_inputs_used":False,
        "monte_carlo_used":False,
        "new_fitting_used":False,
        "selection_period":"none in Phase1E; composes components frozen from 2024 selection",
        "validation_period":"2025 regular season",
        "diagnostic_period":"2026 weeks 1-4 burned",
        "roster_meta":roster_meta,
        "outcomes":{},
    }

    for oc in OUTCOMES:
        val=evaluate(players,team_totals,team_opp,fixed_rows(players,2025,oc),oc,b,d)
        diag=evaluate(players,team_totals,team_opp,fixed_rows(players,2026,oc,4),oc,b,d)
        bv=b["outcomes"][oc]["burned_validation_2025"]
        hv=h["outcomes"][oc]["validation_2025"]
        report["outcomes"][oc]={
            "validation_2025":summarize(val,oc),
            "diagnostic_2026_wk1_4":summarize(diag,oc),
            "phase1b_validation_2025":{
                "n":bv["n"],"mae":bv["mae"],
                "player_opportunity_mae":bv["player_opportunity_mae"],
                "role_share_mae":bv["role_share_mae"],
            },
            "competent_human_validation_2025":hv,
            "delta_mae_vs_phase1b":None if val["mae"] is None else val["mae"]-bv["mae"],
            "delta_mae_vs_competent_human":None if val["mae"] is None else val["mae"]-hv["mae"],
            "largest_2025_misses":sorted(val["receipts"],key=lambda x:-x["abs_error"])[:20],
        }

    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps({
        oc:{
            "validation_2025":v["validation_2025"],
            "diagnostic_2026_wk1_4":v["diagnostic_2026_wk1_4"],
            "phase1b":v["phase1b_validation_2025"],
            "human":v["competent_human_validation_2025"],
            "delta_vs_phase1b":v["delta_mae_vs_phase1b"],
            "delta_vs_human":v["delta_mae_vs_competent_human"],
        } for oc,v in report["outcomes"].items()
    },indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
