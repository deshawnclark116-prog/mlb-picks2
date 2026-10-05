#!/usr/bin/env python3
"""NFL V2 Phase 1G-R: receiving mechanics candidate.

Rebuild receiving production from football components instead of predicting
receiving yards as one opaque number:

    predicted targets (Phase1E role/opportunity)
  x predicted catch probability
  x (predicted completed air yards per catch + predicted YAC per catch)
  = receiving-yard point projection

The receiving-efficiency components use only prior completed plays:
- the player's target/catch/air/YAC history;
- position priors;
- what the upcoming defense has allowed to the same position group.

A small shrinkage/opponent-weight grid is selected on burned 2024 using
*oracle-target receiving-yard MAE* so the efficiency layer is selected on its
own job. 2025 is untouched by selection and is the validation period.
2026 Weeks 1-4 are already-burned diagnostics.

No sportsbook input. No Monte Carlo.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import nfl_v2_phase1a_direct as p1a
import nfl_v2_phase1b_opportunity as p1b
import nfl_v2_phase1d_role_allocation as p1d
import nfl_v2_phase1e_integrated as p1e

GROUPS={"WR":"WR","TE":"TE","RB":"RB","FB":"RB","HB":"RB"}
OUTCOMES=("rec","rec_yds")

_PLAYER_EVENTS=defaultdict(list)
_DEF_EVENTS=defaultdict(list)
_POS_EVENTS=defaultdict(list)
_CACHE={}


def fnum(v):
    try:
        x=float(v)
        return None if math.isnan(x) else x
    except (TypeError,ValueError):
        return None


def mean(xs):
    xs=[x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None


def load_pbp(paths,position_map):
    _PLAYER_EVENTS.clear(); _DEF_EVENTS.clear(); _POS_EVENTS.clear(); _CACHE.clear()
    for path in paths:
        opener=gzip.open if str(path).endswith(".gz") else open
        with opener(path,"rt",newline="",encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r.get("season_type") or "REG")!="REG":
                    continue
                pid=str(r.get("receiver_player_id") or "").strip()
                defense=str(r.get("defteam") or "").strip()
                if not pid or not defense:
                    continue
                try:
                    s=int(float(r["season"])); w=int(float(r["week"]))
                except (KeyError,TypeError,ValueError):
                    continue
                pos=GROUPS.get((position_map.get((s,pid)) or "").upper())
                if pos is None:
                    continue
                comp=1.0 if str(r.get("complete_pass") or "")=="1" else 0.0
                yds=fnum(r.get("yards_gained")) or 0.0
                yac=fnum(r.get("yards_after_catch")) or 0.0
                if not comp:
                    yds=0.0; yac=0.0
                completed_air=(yds-yac) if comp else 0.0
                ev={"key":(s,w),"targets":1.0,"receptions":comp,"rec_yards":yds,
                    "completed_air":completed_air,"yac":yac}
                _PLAYER_EVENTS[pid].append(ev)
                _DEF_EVENTS[(defense,pos)].append(ev)
                _POS_EVENTS[pos].append(ev)
    for idx in (_PLAYER_EVENTS,_DEF_EVENTS,_POS_EVENTS):
        for k in idx:
            idx[k].sort(key=lambda x:x["key"])
    return {
        "player_ids":len(_PLAYER_EVENTS),
        "defense_position_series":len(_DEF_EVENTS),
        "events":sum(len(v) for v in _PLAYER_EVENTS.values()),
    }


def prior_events(index,key,target,window_games=None,max_events=None):
    arr=index.get(key,())
    vals=[e for e in arr if e["key"]<target]
    if window_games is not None and vals:
        games=[]
        seen=set()
        for e in reversed(vals):
            if e["key"] not in seen:
                if len(seen)>=window_games:
                    break
                seen.add(e["key"])
            games.append(e)
        vals=list(reversed(games))
    if max_events is not None:
        vals=vals[-max_events:]
    return vals


def aggregate(events):
    out=defaultdict(float)
    for e in events:
        for k in ("targets","receptions","rec_yards","completed_air","yac"):
            out[k]+=e[k]
    return out


def components_from_agg(a):
    t=a["targets"]; c=a["receptions"]
    if t<=0:
        return None
    return {
        "catch_rate":c/t,
        "air_per_catch":a["completed_air"]/c if c>0 else None,
        "yac_per_catch":a["yac"]/c if c>0 else None,
        "yards_per_target":a["rec_yards"]/t,
        "targets":t,"catches":c,
    }


def position_prior(target,pos):
    ck=("pos",target,pos)
    if ck in _CACHE:
        return _CACHE[ck]
    a=aggregate(prior_events(_POS_EVENTS,pos,target,max_events=1500))
    ans=components_from_agg(a)
    _CACHE[ck]=ans
    return ans


def shrunk_components(events,prior,k_targets):
    a=aggregate(events); c=components_from_agg(a)
    if prior is None:
        return None
    if c is None:
        return dict(prior)
    t=a["targets"]; catches=a["receptions"]
    # Catch rate shrinks by target-equivalents.
    catch=(a["receptions"]+k_targets*prior["catch_rate"])/(t+k_targets)
    # Air/YAC per catch shrink by expected catches implied by the target prior.
    k_catches=max(1.0,k_targets*prior["catch_rate"])
    air=(a["completed_air"]+k_catches*prior["air_per_catch"])/(catches+k_catches)
    yac=(a["yac"]+k_catches*prior["yac_per_catch"])/(catches+k_catches)
    return {"catch_rate":catch,"air_per_catch":air,"yac_per_catch":yac,
            "yards_per_target":catch*(air+yac),"targets":t,"catches":catches}


def efficiency_receipt(r,cfg):
    target=(r["season"],r["week"]); pid=r["player_id"]; defense=r["opponent"]
    pos=GROUPS.get((r.get("position") or "").upper())
    if pos is None:
        return None
    prior=position_prior(target,pos)
    if prior is None or prior["air_per_catch"] is None or prior["yac_per_catch"] is None:
        return None
    player=shrunk_components(
        prior_events(_PLAYER_EVENTS,pid,target,window_games=cfg["player_window_games"]),
        prior,cfg["player_shrink_targets"]
    )
    defense_allowed=shrunk_components(
        prior_events(_DEF_EVENTS,(defense,pos),target,window_games=cfg["defense_window_games"]),
        prior,cfg["defense_shrink_targets"]
    )
    if player is None or defense_allowed is None:
        return None
    w=cfg["defense_weight"]
    catch=max(0.0,min(1.0,(1-w)*player["catch_rate"]+w*defense_allowed["catch_rate"]))
    air=(1-w)*player["air_per_catch"]+w*defense_allowed["air_per_catch"]
    yac=(1-w)*player["yac_per_catch"]+w*defense_allowed["yac_per_catch"]
    return {
        "position_group":pos,
        "player_catch_rate":player["catch_rate"],
        "defense_allowed_catch_rate":defense_allowed["catch_rate"],
        "position_catch_rate":prior["catch_rate"],
        "projected_catch_rate":catch,
        "player_air_per_catch":player["air_per_catch"],
        "defense_air_per_catch":defense_allowed["air_per_catch"],
        "projected_air_per_catch":air,
        "player_yac_per_catch":player["yac_per_catch"],
        "defense_yac_per_catch":defense_allowed["yac_per_catch"],
        "projected_yac_per_catch":yac,
        "projected_yards_per_target":catch*(air+yac),
    }


def configs():
    for pk,dk,dw,pw,dw_games in itertools.product(
        (10.0,25.0),
        (40.0,80.0),
        (0.0,0.15,0.30),
        (5,8),
        (5,8),
    ):
        yield {
            "player_shrink_targets":pk,
            "defense_shrink_targets":dk,
            "defense_weight":dw,
            "player_window_games":pw,
            "defense_window_games":dw_games,
        }


def fixed_rows(players,season,max_week=18):
    source=[r for r in players if r["season"]==season and 1<=r["week"]<=max_week]
    return p1b.fixed_meaningful_rows(players,source,"rec_yds")


def eval_efficiency(rows,cfg):
    yerr=[]; cerr=[]; ypt_err=[]; receipts=[]
    for r in rows:
        q=efficiency_receipt(r,cfg)
        if q is None:
            continue
        targets=r["targets"]; rec=r["receptions"]; yds=r["receiving_yards"]
        if targets<=0:
            continue
        pred_rec=targets*q["projected_catch_rate"]
        pred_yds=targets*q["projected_yards_per_target"]
        actual_ypt=yds/targets
        yerr.append(abs(pred_yds-yds))
        cerr.append(abs(pred_rec-rec))
        ypt_err.append(abs(q["projected_yards_per_target"]-actual_ypt))
        receipts.append((r,q,pred_rec,pred_yds))
    return {"n":len(yerr),"oracle_target_rec_mae":mean(cerr),
            "oracle_target_rec_yds_mae":mean(yerr),"yards_per_target_mae":mean(ypt_err),
            "receipts":receipts}


def select_on_2024(players):
    rows=fixed_rows(players,2024)
    best=None
    for cfg in configs():
        ev=eval_efficiency(rows,cfg)
        if ev["n"]<1800:
            continue
        key=(ev["oracle_target_rec_yds_mae"],ev["yards_per_target_mae"],
             ev["oracle_target_rec_mae"],json.dumps(cfg,sort_keys=True))
        if best is None or key<best[0]:
            best=(key,cfg,ev)
    if best is None:
        raise RuntimeError("No eligible receiving mechanics config")
    return best[1],best[2]


def evaluate_full(players,team_totals,team_opp,rows,cfg,b,d):
    rec_err=[]; yds_err=[]; target_err=[]; receipts=[]
    for r in rows:
        base=p1e.receipt(players,team_totals,team_opp,r,"rec_yds",b,d)
        eff=efficiency_receipt(r,cfg)
        if base is None or eff is None:
            continue
        pt=base["player_opportunity_projection"]
        pred_rec=pt*eff["projected_catch_rate"]
        pred_yds=pt*eff["projected_yards_per_target"]
        rec_err.append(abs(pred_rec-r["receptions"]))
        yds_err.append(abs(pred_yds-r["receiving_yards"]))
        target_err.append(abs(pt-r["targets"]))
        receipts.append({
            "season":r["season"],"week":r["week"],"player_id":r["player_id"],
            "team":r["team"],"opponent":r["opponent"],"position":r.get("position"),
            "actual_targets":r["targets"],"predicted_targets":pt,
            "actual_receptions":r["receptions"],"predicted_receptions":pred_rec,
            "actual_receiving_yards":r["receiving_yards"],"predicted_receiving_yards":pred_yds,
            "rec_abs_error":abs(pred_rec-r["receptions"]),
            "rec_yds_abs_error":abs(pred_yds-r["receiving_yards"]),
            **eff,
        })
    return {"n":len(receipts),"rec_mae":mean(rec_err),"rec_yds_mae":mean(yds_err),
            "target_mae":mean(target_err),"receipts":receipts}


def summary_full(ev):
    if not ev["n"]:
        return {"n":0}
    rerr=[r["rec_abs_error"] for r in ev["receipts"]]
    yerr=[r["rec_yds_abs_error"] for r in ev["receipts"]]
    return {
        "n":ev["n"],"rec_mae":ev["rec_mae"],"rec_yds_mae":ev["rec_yds_mae"],"target_mae":ev["target_mae"],
        "rec_median_ae":statistics.median(rerr),"rec_yds_median_ae":statistics.median(yerr),
        "rec_within_1":sum(e<=1 for e in rerr)/len(rerr),
        "rec_within_2":sum(e<=2 for e in rerr)/len(rerr),
        "rec_yds_within_10":sum(e<=10 for e in yerr)/len(yerr),
        "rec_yds_within_20":sum(e<=20 for e in yerr)/len(yerr),
        "rec_yds_over_40_miss_rate":sum(e>40 for e in yerr)/len(yerr),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stats",action="append",required=True)
    ap.add_argument("--pbp",action="append",required=True)
    ap.add_argument("--roster",action="append",required=True)
    ap.add_argument("--phase1b-snapshot",required=True)
    ap.add_argument("--phase1d-snapshot",required=True)
    ap.add_argument("--phase1e-snapshot",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    players,team_totals,team_opp=p1a.load_stats(a.stats)
    p1a.build_indexes(players,team_totals,team_opp)
    p1b.build_extra_indexes(players,team_totals,team_opp)
    p1d.build_context(players,team_totals)
    p1d.load_rosters(a.roster)

    posmap={(r["season"],r["player_id"]):r.get("position") for r in players}
    pbp_meta=load_pbp(a.pbp,posmap)
    b=json.loads(Path(a.phase1b_snapshot).read_text())
    d=json.loads(Path(a.phase1d_snapshot).read_text())
    e=json.loads(Path(a.phase1e_snapshot).read_text())

    cfg,dev=select_on_2024(players)
    val_rows=fixed_rows(players,2025)
    diag_rows=fixed_rows(players,2026,4)
    val_eff=eval_efficiency(val_rows,cfg)
    diag_eff=eval_efficiency(diag_rows,cfg)
    val_full=evaluate_full(players,team_totals,team_opp,val_rows,cfg,b,d)
    diag_full=evaluate_full(players,team_totals,team_opp,diag_rows,cfg,b,d)

    report={
        "schema":"nfl-v2-phase1g-receiving-mechanics-v1",
        "status":"BURNED_RESEARCH_ONLY",
        "sportsbook_inputs_used":False,"monte_carlo_used":False,
        "selection_period":"2024 regular season only",
        "selection_metric":"oracle-target receiving-yard MAE",
        "validation_period":"2025 regular season; never used for selection",
        "diagnostic_period":"2026 weeks 1-4 burned",
        "pbp_meta":pbp_meta,
        "selected_config":cfg,
        "selection_2024":{
            "n":dev["n"],"oracle_target_rec_mae":dev["oracle_target_rec_mae"],
            "oracle_target_rec_yds_mae":dev["oracle_target_rec_yds_mae"],
            "yards_per_target_mae":dev["yards_per_target_mae"],
        },
        "validation_2025_efficiency":{
            "n":val_eff["n"],"oracle_target_rec_mae":val_eff["oracle_target_rec_mae"],
            "oracle_target_rec_yds_mae":val_eff["oracle_target_rec_yds_mae"],
            "yards_per_target_mae":val_eff["yards_per_target_mae"],
        },
        "diagnostic_2026_wk1_4_efficiency":{
            "n":diag_eff["n"],"oracle_target_rec_mae":diag_eff["oracle_target_rec_mae"],
            "oracle_target_rec_yds_mae":diag_eff["oracle_target_rec_yds_mae"],
            "yards_per_target_mae":diag_eff["yards_per_target_mae"],
        },
        "validation_2025_full":summary_full(val_full),
        "diagnostic_2026_wk1_4_full":summary_full(diag_full),
        "phase1e_benchmarks":{
            "rec_2025":e["outcomes"]["rec"]["validation_2025"]["mae"],
            "rec_yds_2025":e["outcomes"]["rec_yds"]["validation_2025"]["mae"],
            "rec_2026_wk1_4":e["outcomes"]["rec"]["diagnostic_2026_wk1_4"]["mae"],
            "rec_yds_2026_wk1_4":e["outcomes"]["rec_yds"]["diagnostic_2026_wk1_4"]["mae"]
        },
        "largest_2025_rec_yds_misses":sorted(val_full["receipts"],key=lambda x:-x["rec_yds_abs_error"])[:20],
    }

    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps({k:v for k,v in report.items() if k not in ("largest_2025_rec_yds_misses",)},indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())