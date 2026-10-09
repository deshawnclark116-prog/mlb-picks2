#!/usr/bin/env python3
"""NFL V2 Phase 1D-R: coherent player role / usage allocation research head.

This phase predicts *who gets the opportunities* before any yards are modeled.

Primary targets:
- carry share for rushing-yard players;
- target share for receiving-yard / reception players;
- pass-attempt share for passing-yard players.

The candidate is deliberately evaluated on role share and player opportunities
before final-stat efficiency is allowed to matter.

Pregame information only:
- prior completed-game player opportunity shares;
- prior completed-game role trend;
- current weekly roster membership / status at the target week;
- position as a weak cold-start prior.

Current-roster normalization makes opportunity allocation coherent: shares over
the eligible roster sum to one. If a previously high-share teammate is absent
from the current active roster, his share is not silently left in the model;
the remaining active players compete for the available opportunity.

Selection: burned 2024 only (2023 supplies prior history).
Validation: 2025, never used for selection.
Diagnostics: 2026 Weeks 1-4, already burned.
No Monte Carlo. No sportsbook inputs.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import nfl_v2_phase1a_direct as p1a
import nfl_v2_phase1b_opportunity as p1b

OUTCOMES=("rush_yds","rec_yds","rec","pass_yds")
SPEC=p1a.SPEC

ROSTER_ID_COLS=("gsis_id","player_id","smart_id")
ROSTER_TEAM_COLS=("team","recent_team")
ROSTER_POS_COLS=("position","position_group","ngs_position_group")
ROSTER_STATUS_COLS=("status","status_description_abbr","game_status")

TARGET_POSITIONS={
    "rush_yds":{"QB","RB","FB","HB","WR","TE"},
    "rec_yds":{"RB","FB","HB","WR","TE"},
    "rec":{"RB","FB","HB","WR","TE"},
    "pass_yds":{"QB"},
}

_ROSTERS=defaultdict(list)
_STATUS_COUNTS=Counter()
_POS_SHARE_CACHE={}
_ROLE_POS_INDEX=defaultdict(list)
_ALLOCATION_CACHE={}


def first(row,names):
    for n in names:
        v=row.get(n)
        if v not in (None,""):
            return str(v)
    return None


def fnum(v):
    try:
        x=float(v)
        return None if math.isnan(x) else x
    except (TypeError,ValueError):
        return None


def mean(xs):
    return sum(xs)/len(xs) if xs else None


def ewma(xs,decay):
    if not xs:
        return None
    if decay>=0.999999:
        return mean(xs)
    n=len(xs)
    ws=[decay**(n-1-i) for i in range(n)]
    den=sum(ws)
    return sum(x*w for x,w in zip(xs,ws))/den if den else None


def active_status(raw):
    """Conservative T90 roster-status interpretation.

    Unknown/blank statuses are retained rather than inventing an inactive flag.
    Clearly non-active roster states are excluded.
    """
    if raw in (None,""):
        return True
    s=str(raw).strip().upper()
    if s in {"ACT","ACTIVE"}:
        return True
    bad=("INA","INACTIVE","RES","IR","PUP","NFI","SUS","RET","CUT","DEV","PRACTICE")
    if any(s==x or s.startswith(x) for x in bad):
        return False
    return True


def load_rosters(paths):
    _ROSTERS.clear(); _STATUS_COUNTS.clear(); _POS_SHARE_CACHE.clear(); _ALLOCATION_CACHE.clear()
    seen=set()
    for path in paths:
        with open(path,newline="",encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    season=int(float(r.get("season") or 0))
                    week=int(float(r.get("week") or 0))
                except (TypeError,ValueError):
                    continue
                if season<=0 or week<=0:
                    continue
                gt=str(r.get("game_type") or r.get("season_type") or "REG").upper()
                if gt not in {"REG",""}:
                    continue
                pid=first(r,ROSTER_ID_COLS)
                team=first(r,ROSTER_TEAM_COLS)
                pos=(first(r,ROSTER_POS_COLS) or "").upper()
                status=first(r,ROSTER_STATUS_COLS)
                if not pid or not team:
                    continue
                key=(season,week,team,pid)
                if key in seen:
                    continue
                seen.add(key)
                _STATUS_COUNTS[(season,str(status or "BLANK").upper())]+=1
                _ROSTERS[(season,week,team)].append({
                    "player_id":pid,"team":team,"position":pos,
                    "status":status,"active":active_status(status),
                })
    for key in _ROSTERS:
        _ROSTERS[key].sort(key=lambda x:x["player_id"])
    return {
        "team_weeks":len(_ROSTERS),
        "rows":sum(len(v) for v in _ROSTERS.values()),
        "status_counts":{f"{s}:{st}":n for (s,st),n in sorted(_STATUS_COUNTS.items())},
    }


def prior_share_series(players,team_totals,target,pid,outcome,window,team=None):
    sp=SPEC[outcome]
    rows=p1a.prior_player_rows(players,target,pid,team,window)
    vals=[]
    for r in rows:
        den=team_totals.get((r["season"],r["week"],r["team"]),{}).get(sp["opp"],0.0)
        if den>0:
            vals.append(r.get(sp["opp"],0.0)/den)
    return vals[-window:]


_CTX_PLAYERS=None
_CTX_TEAM_TOTALS=None


def build_context(players,team_totals):
    global _CTX_PLAYERS,_CTX_TEAM_TOTALS
    _CTX_PLAYERS=players
    _CTX_TEAM_TOTALS=team_totals
    _ROLE_POS_INDEX.clear()
    for r in players:
        pos=(r.get("position") or "").upper()
        if pos:
            _ROLE_POS_INDEX[pos].append(r)
    for pos in _ROLE_POS_INDEX:
        _ROLE_POS_INDEX[pos].sort(key=lambda r:(r["season"],r["week"],r["player_id"]))
    _POS_SHARE_CACHE.clear()
    _ALLOCATION_CACHE.clear()


def position_share_prior(target,position,outcome,max_rows=500):
    key=(target,position,outcome,max_rows)
    if key in _POS_SHARE_CACHE:
        return _POS_SHARE_CACHE[key]
    if not position:
        _POS_SHARE_CACHE[key]=None
        return None
    sp=SPEC[outcome]
    vals=[]
    for r in _ROLE_POS_INDEX.get(position,()):
        if (r["season"],r["week"])>=target:
            break
        den=_CTX_TEAM_TOTALS.get((r["season"],r["week"],r["team"]),{}).get(sp["opp"],0.0)
        if den>0 and r.get(sp["opp"],0.0)>0:
            vals.append(r[sp["opp"]]/den)
    vals=vals[-max_rows:]
    ans=statistics.median(vals) if vals else None
    _POS_SHARE_CACHE[key]=ans
    return ans


def raw_role_score(players,team_totals,target,team,player,outcome,cfg):
    pid=player["player_id"]; pos=(player.get("position") or "").upper()
    team_hist=prior_share_series(players,team_totals,target,pid,outcome,cfg["share_window"],team)
    global_hist=prior_share_series(players,team_totals,target,pid,outcome,cfg["share_window"],None)

    t=ewma(team_hist,cfg["decay"]) if team_hist else None
    g=ewma(global_hist,cfg["decay"]) if global_hist else None

    if t is not None and g is not None:
        base=(1.0-cfg["global_weight"])*t+cfg["global_weight"]*g
    elif t is not None:
        base=t
    elif g is not None:
        base=g
    else:
        pp=position_share_prior(target,pos,outcome)
        base=(pp or 0.0)*cfg["cold_start_scale"]

    if len(team_hist)>=2 and cfg["trend_gain"]>0:
        prev=ewma(team_hist[:-1],cfg["decay"])
        shift=team_hist[-1]-(prev if prev is not None else team_hist[-1])
        base+=cfg["trend_gain"]*shift

    base=max(0.0,base)
    # Power >1 intentionally permits concentration around established leaders;
    # power <1 would spread usage. Both are tested honestly.
    return max(base,1e-6)**cfg["concentration_power"]


def eligible_roster(season,week,team,outcome):
    allowed=TARGET_POSITIONS[outcome]
    rows=[]
    for r in _ROSTERS.get((season,week,team),()):
        if not r["active"]:
            continue
        pos=(r.get("position") or "").upper()
        if pos in allowed:
            rows.append(r)
    return rows


def allocation(players,team_totals,r,outcome,cfg):
    key=(
        r["season"],r["week"],r["team"],outcome,
        cfg["share_window"],cfg["decay"],cfg["global_weight"],
        cfg["trend_gain"],cfg["cold_start_scale"],cfg["concentration_power"],
    )
    cached=_ALLOCATION_CACHE.get(key)
    if cached is not None:
        return cached
    roster=eligible_roster(r["season"],r["week"],r["team"],outcome)
    if not roster:
        _ALLOCATION_CACHE[key]=None
        return None
    target=(r["season"],r["week"])
    scores={x["player_id"]:raw_role_score(players,team_totals,target,r["team"],x,outcome,cfg) for x in roster}
    den=sum(scores.values())
    if den<=0:
        _ALLOCATION_CACHE[key]=None
        return None
    shares={pid:score/den for pid,score in scores.items()}
    rec={
        "predicted_shares":shares,
        "roster_n":len(roster),
        "share_sum":sum(shares.values()),
        "active_player_ids":sorted(shares),
    }
    _ALLOCATION_CACHE[key]=rec
    return rec


def transparent_share_baseline(players,team_totals,r,outcome,window=3):
    vals=prior_share_series(
        players,team_totals,(r["season"],r["week"]),r["player_id"],outcome,window,r["team"]
    )
    return mean(vals) if len(vals)>=2 else None


def phase1b_team_projection(players,team_totals,team_opp,r,outcome,phase1b_cfg):
    rec=p1b.opportunity_receipt(players,team_totals,team_opp,r,outcome,phase1b_cfg)
    return None if rec is None else rec["team_opportunity_projection"]


def fixed_rows(players,rows,outcome):
    return p1b.fixed_meaningful_rows(players,rows,outcome)


def evaluate(players,team_totals,team_opp,rows,outcome,cfg,phase1b_cfg=None):
    sp=SPEC[outcome]
    share_err=[]; oracle_opp_err=[]; coupled_opp_err=[]; baseline_err=[]
    coherent=[]; receipts=[]
    for r in rows:
        alloc=allocation(players,team_totals,r,outcome,cfg)
        if alloc is None:
            continue
        ps=alloc["predicted_shares"].get(r["player_id"])
        if ps is None:
            # Player passed the fixed pregame workload population but is not in
            # the active target-week roster: fail closed rather than call this 0.
            continue
        actual_team=team_totals.get((r["season"],r["week"],r["team"]),{}).get(sp["opp"],0.0)
        if actual_team<=0:
            continue
        ao=r.get(sp["opp"],0.0)
        ashare=ao/actual_team
        share_err.append(abs(ps-ashare))
        oracle=ps*actual_team
        oracle_opp_err.append(abs(oracle-ao))
        coherent.append(abs(alloc["share_sum"]-1.0))

        team_proj=None
        coupled=None
        if phase1b_cfg is not None:
            team_proj=phase1b_team_projection(players,team_totals,team_opp,r,outcome,phase1b_cfg)
            if team_proj is not None:
                coupled=team_proj*ps
                coupled_opp_err.append(abs(coupled-ao))

        b=transparent_share_baseline(players,team_totals,r,outcome,3)
        if b is not None:
            baseline_err.append(abs(b-ashare))

        receipts.append({
            "season":r["season"],"week":r["week"],"player_id":r["player_id"],
            "team":r["team"],"opponent":r["opponent"],"position":r.get("position"),
            "outcome":outcome,"actual_team_opportunities":actual_team,
            "actual_player_opportunities":ao,"actual_share":ashare,
            "predicted_share":ps,"oracle_player_opportunities":oracle,
            "phase1b_team_opportunity_projection":team_proj,
            "coupled_player_opportunity_projection":coupled,
            "roster_n":alloc["roster_n"],"share_sum":alloc["share_sum"],
        })
    return {
        "n":len(share_err),
        "role_share_mae":mean(share_err),
        "oracle_player_opportunity_mae":mean(oracle_opp_err),
        "coupled_player_opportunity_mae":mean(coupled_opp_err),
        "transparent_recent3_share_mae":mean(baseline_err),
        "max_coherence_error":max(coherent) if coherent else None,
        "rows":receipts,
    }


def configs():
    for sw,d,gw,tg,cs,pow_ in itertools.product(
        (3,5,8),
        (0.60,0.85,1.0),
        (0.0,0.25),
        (0.0,0.35),
        (0.05,0.15),
        (1.0,1.20),
    ):
        yield {
            "share_window":sw,"decay":d,"global_weight":gw,
            "trend_gain":tg,"cold_start_scale":cs,
            "concentration_power":pow_,
        }


def select_2024(players,team_totals,team_opp,outcome,phase1b_cfg):
    rows=fixed_rows(players,p1a.target_rows(players,[(2024,w) for w in range(1,19)]),outcome)
    min_n={"rush_yds":250,"rec_yds":600,"rec":600,"pass_yds":180}[outcome]
    best=None
    for cfg in configs():
        _ALLOCATION_CACHE.clear()
        ev=evaluate(players,team_totals,team_opp,rows,outcome,cfg,phase1b_cfg)
        if ev["n"]<min_n or ev["role_share_mae"] is None:
            continue
        key=(
            ev["role_share_mae"],
            ev["oracle_player_opportunity_mae"],
            ev["coupled_player_opportunity_mae"] if ev["coupled_player_opportunity_mae"] is not None else 1e9,
            json.dumps(cfg,sort_keys=True),
        )
        if best is None or key<best[0]:
            best=(key,cfg,ev)
    if best is None:
        raise RuntimeError(f"No eligible role config for {outcome}")
    return best[1],best[2]


def compact(ev):
    return {k:v for k,v in ev.items() if k!="rows"}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stats",action="append",required=True)
    ap.add_argument("--roster",action="append",required=True)
    ap.add_argument("--phase1b-snapshot",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    players,team_totals,team_opp=p1a.load_stats(a.stats)
    p1a.build_indexes(players,team_totals,team_opp)
    p1b.build_extra_indexes(players,team_totals,team_opp)
    build_context(players,team_totals)
    roster_meta=load_rosters(a.roster)
    phase1b=json.loads(Path(a.phase1b_snapshot).read_text())

    report={
        "schema":"nfl-v2-phase1d-role-allocation-v1",
        "status":"BURNED_RESEARCH_ONLY",
        "sportsbook_inputs_used":False,
        "monte_carlo_used":False,
        "selection_period":"2024 regular season only",
        "validation_period":"2025 regular season; never used for selection",
        "diagnostic_period":"2026 weeks 1-4 burned",
        "football_scope":"player opportunity share / concentration only",
        "primary_selection_metric":"role_share_mae",
        "roster_source_meta":roster_meta,
        "outcomes":{},
    }

    for oc in OUTCOMES:
        p1b_cfg=phase1b["outcomes"][oc]["selected_opportunity_config"]
        cfg,dev=select_2024(players,team_totals,team_opp,oc,p1b_cfg)
        _ALLOCATION_CACHE.clear()
        val_rows=fixed_rows(players,p1a.target_rows(players,[(2025,w) for w in range(1,19)]),oc)
        val=evaluate(players,team_totals,team_opp,val_rows,oc,cfg,p1b_cfg)
        _ALLOCATION_CACHE.clear()
        diag_source=[r for r in players if r["season"]==2026 and 1<=r["week"]<=4]
        diag_rows=fixed_rows(players,diag_source,oc)
        diag=evaluate(players,team_totals,team_opp,diag_rows,oc,cfg,p1b_cfg)
        report["outcomes"][oc]={
            "selected_config":cfg,
            "selection_2024":compact(dev),
            "validation_2025":compact(val),
            "diagnostic_2026_wk1_4":compact(diag),
            "phase1b_role_share_mae_2025":phase1b["outcomes"][oc]["burned_validation_2025"]["role_share_mae"],
            "phase1b_player_opportunity_mae_2025":phase1b["outcomes"][oc]["burned_validation_2025"]["player_opportunity_mae"],
            "largest_2025_role_misses":sorted(
                val["rows"],key=lambda x:-abs(x["predicted_share"]-x["actual_share"])
            )[:20],
        }

    Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps({
        oc:{
            "cfg":v["selected_config"],
            "dev2024":v["selection_2024"],
            "val2025":v["validation_2025"],
            "diag2026":v["diagnostic_2026_wk1_4"],
            "phase1b_role_mae_2025":v["phase1b_role_share_mae_2025"],
            "phase1b_opp_mae_2025":v["phase1b_player_opportunity_mae_2025"],
        } for oc,v in report["outcomes"].items()
    },indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())