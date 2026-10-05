#!/usr/bin/env python3
"""NFL V2 Phase 1A: direct, auditable football projection chain.

This is the first V2 predictive candidate. It intentionally has NO Monte Carlo.
For each player/outcome it predicts:

    team opportunities
  x player role share
  x production per opportunity
  = direct point projection

Team opportunities use both the offense's prior volume and the opponent's prior
allowed volume. Role uses the player's prior share on his current team.
Efficiency blends the player's prior efficiency with what the opponent has
allowed. Every input is strictly from earlier weeks.

Hyperparameters are chosen on burned 2025 only. 2026 weeks 1-4 are reported as
burned diagnostics, never as clean holdout evidence.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

OUTCOMES = ("rush_yds", "rec_yds", "rec", "pass_yds")
SPEC = {
    "rush_yds": {"opp":"carries", "value":"rushing_yards", "min_pred_opp":5.0},
    "rec_yds": {"opp":"targets", "value":"receiving_yards", "min_pred_opp":3.0},
    "rec": {"opp":"targets", "value":"receptions", "min_pred_opp":3.0},
    "pass_yds": {"opp":"attempts", "value":"passing_yards", "min_pred_opp":20.0},
}
PLAYER_ID_COLS=("player_id","gsis_id")
TEAM_COLS=("recent_team","team")
OPP_COLS=("opponent_team","opponent")
POS_COLS=("position","position_group")


def fnum(v):
    try:
        x=float(v)
        return None if math.isnan(x) else x
    except (TypeError,ValueError):
        return None


def first(row,names):
    for n in names:
        v=row.get(n)
        if v not in (None,""):
            return str(v)
    return None


def load_stats(paths):
    players=[]
    team=defaultdict(lambda: defaultdict(float))
    team_opp={}
    for path in paths:
        with open(path,newline="",encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r.get("season_type") or "REG")!="REG":
                    continue
                pid=first(r,PLAYER_ID_COLS); tm=first(r,TEAM_COLS); opp=first(r,OPP_COLS); pos=first(r,POS_COLS)
                if not pid or not tm or not opp:
                    continue
                try: key=(int(r["season"]),int(float(r["week"])))
                except (KeyError,TypeError,ValueError): continue
                rec={"season":key[0],"week":key[1],"player_id":pid,"team":tm,"opponent":opp,"position":pos}
                for col in {v for x in SPEC.values() for v in (x["opp"],x["value"])}:
                    aliases=(col,"interceptions") if col=="passing_interceptions" else (col,)
                    val=None
                    for a in aliases:
                        val=fnum(r.get(a))
                        if val is not None: break
                    rec[col]=0.0 if val is None else val
                players.append(rec)
                tk=(key[0],key[1],tm)
                team_opp[tk]=opp
                for col in ("carries","targets","attempts","rushing_yards","receiving_yards","receptions","passing_yards"):
                    team[tk][col]+=rec.get(col,0.0)
    players.sort(key=lambda r:(r["season"],r["week"],r["team"],r["player_id"]))
    return players,team,team_opp


def mean(xs):
    return sum(xs)/len(xs) if xs else None


def prior_team_values(team, team_opp, target, tm, col, window, allowed=False):
    season,week=target
    vals=[]
    for (s,w,t),stats in team.items():
        if (s,w)>=(season,week): continue
        if allowed:
            if team_opp.get((s,w,t))!=tm: continue
        elif t!=tm:
            continue
        vals.append(((s,w),stats.get(col,0.0)))
    vals.sort()
    return [x[1] for x in vals[-window:]]


def prior_player_rows(players,target,pid,team=None,window=8):
    s0,w0=target
    rows=[r for r in players if r["player_id"]==pid and (r["season"],r["week"])<(s0,w0) and (team is None or r["team"]==team)]
    rows.sort(key=lambda r:(r["season"],r["week"]))
    return rows[-window:]


def player_share_history(players, team_totals, target, pid, tm, opp_col, window):
    rows=prior_player_rows(players,target,pid,tm,window)
    vals=[]
    for r in rows:
        den=team_totals.get((r["season"],r["week"],tm),{}).get(opp_col,0.0)
        if den>0:
            vals.append(r[opp_col]/den)
    return vals


def player_eff_history(players,target,pid,value_col,opp_col,window):
    rows=prior_player_rows(players,target,pid,None,window)
    num=sum(r[value_col] for r in rows)
    den=sum(r[opp_col] for r in rows)
    return None if den<=0 else num/den


def opponent_eff_history(team_totals,team_opp,target,defense,value_col,opp_col,window):
    vals=[]
    s0,w0=target
    for (s,w,t),st in team_totals.items():
        if (s,w)>=(s0,w0) or team_opp.get((s,w,t))!=defense: continue
        if st.get(opp_col,0)>0:
            vals.append(((s,w),st[value_col],st[opp_col]))
    vals.sort()
    vals=vals[-window:]
    den=sum(x[2] for x in vals)
    return None if den<=0 else sum(x[1] for x in vals)/den


def actual_team_opp(team_totals,r,opp_col):
    return team_totals.get((r["season"],r["week"],r["team"]),{}).get(opp_col,0.0)


def feature_receipt(players,team_totals,team_opp,r,outcome,team_window,share_window,eff_window):
    sp=SPEC[outcome]; target=(r["season"],r["week"]); tm=r["team"]; opp=r["opponent"]
    own=prior_team_values(team_totals,team_opp,target,tm,sp["opp"],team_window,False)
    allowed=prior_team_values(team_totals,team_opp,target,opp,sp["opp"],team_window,True)
    shares=player_share_history(players,team_totals,target,r["player_id"],tm,sp["opp"],share_window)
    peff=player_eff_history(players,target,r["player_id"],sp["value"],sp["opp"],eff_window)
    deff=opponent_eff_history(team_totals,team_opp,target,opp,sp["value"],sp["opp"],eff_window)
    if len(own)<2 or len(allowed)<2 or len(shares)<2 or peff is None or deff is None:
        return None
    return {
        "offense_recent_team_opportunities":mean(own),
        "opponent_recent_allowed_opportunities":mean(allowed),
        "player_recent_role_share":mean(shares),
        "player_recent_efficiency":peff,
        "opponent_recent_allowed_efficiency":deff,
        "history_counts":{"team":len(own),"opponent":len(allowed),"share":len(shares)},
    }


def project(receipt,w_offense,w_player_eff):
    team_opp=w_offense*receipt["offense_recent_team_opportunities"]+(1-w_offense)*receipt["opponent_recent_allowed_opportunities"]
    eff=w_player_eff*receipt["player_recent_efficiency"]+(1-w_player_eff)*receipt["opponent_recent_allowed_efficiency"]
    player_opp=team_opp*receipt["player_recent_role_share"]
    point=player_opp*eff
    return {
        **receipt,
        "team_opportunity_projection":team_opp,
        "player_opportunity_projection":player_opp,
        "final_efficiency_projection":eff,
        "point_projection":point,
    }


def target_rows(players,season_weeks):
    wanted=set(season_weeks)
    return [r for r in players if (r["season"],r["week"]) in wanted]


def actual(r,outcome):
    return r[SPEC[outcome]["value"]]


def evaluate_config(players,team_totals,team_opp,rows,outcome,cfg):
    errs=[]; signed=[]; receipts=[]
    for r in rows:
        rec=feature_receipt(players,team_totals,team_opp,r,outcome,cfg["team_window"],cfg["share_window"],cfg["eff_window"])
        if rec is None: continue
        p=project(rec,cfg["w_offense"],cfg["w_player_eff"])
        if p["player_opportunity_projection"]<SPEC[outcome]["min_pred_opp"]: continue
        y=actual(r,outcome)
        errs.append(abs(p["point_projection"]-y)); signed.append(p["point_projection"]-y)
        receipts.append((r,p,y))
    return {
        "n":len(errs),
        "mae":mean(errs),
        "bias":mean(signed),
        "receipts":receipts,
    }


def configs():
    for tw,sw,ew,wo,we in itertools.product((3,5,8),(3,5),(5,8),(0.50,0.65,0.80),(0.60,0.80,1.0)):
        yield {"team_window":tw,"share_window":sw,"eff_window":ew,"w_offense":wo,"w_player_eff":we}


def select_on_2025(players,team_totals,team_opp,outcome):
    rows=target_rows(players,[(2025,w) for w in range(1,19)])
    best=None
    for cfg in configs():
        ev=evaluate_config(players,team_totals,team_opp,rows,outcome,cfg)
        # Enough evidence to prevent a tiny easy subset from winning.
        min_n={"rush_yds":250,"rec_yds":600,"rec":600,"pass_yds":200}[outcome]
        if ev["n"]<min_n or ev["mae"] is None: continue
        candidate=(ev["mae"],-ev["n"],json.dumps(cfg,sort_keys=True),cfg,ev)
        if best is None or candidate[:3]<best[:3]:
            best=candidate
    if best is None:
        raise RuntimeError(f"No eligible 2025 config for {outcome}")
    return best[3],best[4]


def summarize(ev,outcome):
    if not ev["n"]: return {"n":0}
    tols=(5,10,15,20,25) if outcome.endswith("_yds") else (0,1,2)
    errs=[abs(p["point_projection"]-y) for _,p,y in ev["receipts"]]
    return {
        "n":ev["n"],"mae":ev["mae"],"bias":ev["bias"],
        "within":{str(t):sum(e<=t for e in errs)/len(errs) for t in tols},
    }


def compare_week4_to_v1(players,team_totals,team_opp,audit,outcome,cfg):
    clean={(r["player_id"],r["outcome"]):r for r in audit["rows"] if r.get("clean_meaningful") and r["outcome"]==outcome}
    candidates=[r for r in players if r["season"]==2026 and r["week"]==4 and (r["player_id"],outcome) in clean]
    vals=[]
    for r in candidates:
        rec=feature_receipt(players,team_totals,team_opp,r,outcome,cfg["team_window"],cfg["share_window"],cfg["eff_window"])
        if rec is None: continue
        p=project(rec,cfg["w_offense"],cfg["w_player_eff"])
        a=clean[(r["player_id"],outcome)]
        vals.append((r,p,float(a["actual"]),float(a["median"])))
    if not vals: return {"n":0,"rows":[]}
    pe=[abs(p["point_projection"]-y) for _,p,y,_ in vals]
    ve=[abs(v-y) for _,p,y,v in vals]
    rows=[]
    for r,p,y,v in vals:
        rows.append({
            "player_id":r["player_id"],"player":clean[(r["player_id"],outcome)]["player"],"team":r["team"],"opponent":r["opponent"],
            "outcome":outcome,"actual":y,"v1_median":v,"v1_abs_error":abs(v-y),
            "v2_direct_point":p["point_projection"],"v2_abs_error":abs(p["point_projection"]-y),
            "receipt":p,
        })
    return {
        "n":len(vals),"v1_mae_same_rows":mean(ve),"v2_direct_mae":mean(pe),
        "v2_minus_v1_mae":mean(pe)-mean(ve),
        "v2_row_wins":sum(a<b for a,b in zip(pe,ve)),"v1_row_wins":sum(b<a for a,b in zip(pe,ve)),
        "rows":sorted(rows,key=lambda x:-x["v1_abs_error"]),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stats",action="append",required=True)
    ap.add_argument("--week4-audit",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    players,team_totals,team_opp=load_stats(a.stats)
    audit=json.loads(Path(a.week4_audit).read_text())
    report={
        "schema":"nfl-v2-phase1a-direct-chain-v1",
        "status":"BURNED_RESEARCH_ONLY",
        "sportsbook_inputs_used":False,
        "monte_carlo_used":False,
        "formula":"team opportunities x player role share x per-opportunity efficiency",
        "selection_period":"2025 regular season only",
        "diagnostic_period":"2026 weeks 1-4 burned",
        "outcomes":{},
    }
    for oc in OUTCOMES:
        cfg,dev=select_on_2025(players,team_totals,team_opp,oc)
        diag=evaluate_config(players,team_totals,team_opp,target_rows(players,[(2026,w) for w in range(1,5)]),oc,cfg)
        week4=compare_week4_to_v1(players,team_totals,team_opp,audit,oc,cfg)
        report["outcomes"][oc]={
            "selected_config":cfg,
            "selection_2025":summarize(dev,oc),
            "diagnostic_2026_wk1_4":summarize(diag,oc),
            "week4_exact_v1_clean_comparison":week4,
        }
    Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True))
    compact={oc:{
        "cfg":v["selected_config"],
        "dev":v["selection_2025"],
        "diag":v["diagnostic_2026_wk1_4"],
        "week4":{k:x for k,x in v["week4_exact_v1_clean_comparison"].items() if k!="rows"},
    } for oc,v in report["outcomes"].items()}
    print(json.dumps(compact,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
