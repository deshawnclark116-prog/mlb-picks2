#!/usr/bin/env python3
"""NFL V2 competent-human baseline.

Purpose
-------
Represent what a disciplined football researcher could calculate transparently
without Monte Carlo, sportsbook input, hyperparameter search, or black-box ML.

This is a *benchmark*, not a candidate model. The formula is frozen by football
reasoning before evaluation:

Receiving / rushing:
  recent team opportunity (70%)
  + recent opponent-allowed team opportunity (20%)
  + league team-opportunity prior (10%)
  x coherent recent player share among target-week roster members
  x recent player efficiency (70%)
  + opponent-allowed efficiency (20%)
  + position efficiency prior (10%)

Passing:
  recent QB attempts (75%)
  + opponent-allowed team attempts (15%)
  + league team-attempt prior (10%)
  x recent QB yards/attempt (75%)
  + opponent-allowed pass yards/attempt (15%)
  + QB league prior (10%)

Recent observations are exponentially weighted with the latest game receiving
the most weight. There is no parameter fitting.

Historical weekly roster data is used only for target-week team membership and
position when coherently normalizing RB/WR/TE usage. It is NOT used as a claim
that historical injury/game-status fields were known pregame.

No sportsbook lines. No Monte Carlo.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import nfl_v2_phase1a_direct as p1a
import nfl_v2_phase1b_opportunity as p1b

OUTCOMES=("rush_yds","rec_yds","rec","pass_yds")
SPEC=p1a.SPEC

TARGET_POSITIONS={
    "rush_yds":{"QB","RB","FB","HB","WR","TE"},
    "rec_yds":{"RB","FB","HB","WR","TE"},
    "rec":{"RB","FB","HB","WR","TE"},
}
ROSTER_ID_COLS=("gsis_id","player_id","smart_id")
ROSTER_TEAM_COLS=("team","recent_team")
ROSTER_POS_COLS=("position","position_group","ngs_position_group")

TEAM_WINDOW=3
OPP_WINDOW=3
ROLE_WINDOW=3
EFF_WINDOW=5
DECAY=0.65

TEAM_W=(0.70,0.20,0.10)  # own, opponent allowed, league
EFF_W=(0.70,0.20,0.10)   # player, opponent allowed, position
QB_ATTEMPT_W=(0.75,0.15,0.10)
QB_EFF_W=(0.75,0.15,0.10)

_ROSTERS=defaultdict(list)
_POS_INDEX=defaultdict(list)
_DEF_INDEX=defaultdict(list)
_ALL_TEAM=[]
_POS_EFF_CACHE={}
_POS_ROLE_CACHE={}
_OPP_EFF_CACHE={}
_LEAGUE_CACHE={}
_ALLOC_CACHE={}


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


def mean(xs):
    return sum(xs)/len(xs) if xs else None


def ewma(xs,decay=DECAY):
    if not xs:
        return None
    n=len(xs)
    ws=[decay**(n-1-i) for i in range(n)]
    den=sum(ws)
    return sum(x*w for x,w in zip(xs,ws))/den if den else None


def load_roster_membership(paths):
    _ROSTERS.clear()
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
                if gt not in {"","REG"}:
                    continue
                pid=first(r,ROSTER_ID_COLS); team=first(r,ROSTER_TEAM_COLS)
                pos=(first(r,ROSTER_POS_COLS) or "").upper()
                if not pid or not team:
                    continue
                key=(season,week,team,pid)
                if key in seen:
                    continue
                seen.add(key)
                _ROSTERS[(season,week,team)].append({"player_id":pid,"position":pos})
    for k in _ROSTERS:
        _ROSTERS[k].sort(key=lambda x:x["player_id"])
    return {"rows":sum(len(v) for v in _ROSTERS.values()),"team_weeks":len(_ROSTERS)}


def build_indexes(players,team_totals,team_opp):
    _POS_INDEX.clear(); _DEF_INDEX.clear(); _ALL_TEAM.clear(); _POS_EFF_CACHE.clear(); _POS_ROLE_CACHE.clear(); _OPP_EFF_CACHE.clear(); _LEAGUE_CACHE.clear(); _ALLOC_CACHE.clear()
    for r in players:
        pos=(r.get("position") or "").upper()
        if pos:
            _POS_INDEX[pos].append(r)
    for pos in _POS_INDEX:
        _POS_INDEX[pos].sort(key=lambda r:(r["season"],r["week"],r["player_id"]))
    for (s,w,team),st in team_totals.items():
        tkey=(s,w)
        _ALL_TEAM.append((tkey,team,st))
        defense=team_opp.get((s,w,team))
        if defense:
            _DEF_INDEX[defense].append((tkey,st))
    _ALL_TEAM.sort(key=lambda x:(x[0],x[1]))
    for defense in _DEF_INDEX:
        _DEF_INDEX[defense].sort(key=lambda x:x[0])


def league_team_prior(target,col,max_games=128):
    key=(target,col,max_games)
    if key in _LEAGUE_CACHE:
        return _LEAGUE_CACHE[key]
    vals=[]
    for tkey,_team,st in _ALL_TEAM:
        if tkey>=target:
            break
        vals.append(st.get(col,0.0))
    ans=mean(vals[-max_games:])
    _LEAGUE_CACHE[key]=ans
    return ans


def position_eff_prior(players,team_totals,target,position,value_col,opp_col,max_rows=500):
    key=(target,position,value_col,opp_col,max_rows)
    if key in _POS_EFF_CACHE:
        return _POS_EFF_CACHE[key]
    rows=[]
    for r in _POS_INDEX.get(position,()):
        if (r["season"],r["week"])>=target:
            break
        if r.get(opp_col,0.0)>0:
            rows.append(r)
    rows=rows[-max_rows:]
    den=sum(r[opp_col] for r in rows)
    ans=None if den<=0 else sum(r[value_col] for r in rows)/den
    _POS_EFF_CACHE[key]=ans
    return ans


def prior_role_shares(players,team_totals,target,pid,team,outcome):
    sp=SPEC[outcome]
    rows=p1a.prior_player_rows(players,target,pid,team,ROLE_WINDOW)
    vals=[]
    for r in rows:
        den=team_totals.get((r["season"],r["week"],team),{}).get(sp["opp"],0.0)
        if den>0:
            vals.append(r.get(sp["opp"],0.0)/den)
    return vals[-ROLE_WINDOW:]


def raw_role(players,team_totals,target,team,player,outcome):
    vals=prior_role_shares(players,team_totals,target,player["player_id"],team,outcome)
    if vals:
        return max(ewma(vals),1e-6)
    # Cold starts get a deliberately small cached position prior rather than zero.
    pos=(player.get("position") or "").upper()
    ckey=(target,pos,outcome)
    if ckey not in _POS_ROLE_CACHE:
        peers=[]
        sp=SPEC[outcome]
        for rr in _POS_INDEX.get(pos,()):
            if (rr["season"],rr["week"])>=target:
                break
            den=team_totals.get((rr["season"],rr["week"],rr["team"]),{}).get(sp["opp"],0.0)
            if den>0 and rr.get(sp["opp"],0.0)>0:
                peers.append(rr[sp["opp"]]/den)
        _POS_ROLE_CACHE[ckey]=statistics.median(peers[-300:]) if peers else 0.01
    return max(_POS_ROLE_CACHE[ckey]*0.10,1e-6)


def coherent_share(players,team_totals,r,outcome):
    key=(r["season"],r["week"],r["team"],outcome)
    alloc=_ALLOC_CACHE.get(key)
    if alloc is None:
        roster=[
            x for x in _ROSTERS.get((r["season"],r["week"],r["team"]),())
            if (x.get("position") or "").upper() in TARGET_POSITIONS[outcome]
        ]
        if not roster:
            _ALLOC_CACHE[key]={}
            return None
        target=(r["season"],r["week"])
        scores={x["player_id"]:raw_role(players,team_totals,target,r["team"],x,outcome) for x in roster}
        den=sum(scores.values())
        if den<=0:
            _ALLOC_CACHE[key]={}
            return None
        alloc={pid:v/den for pid,v in scores.items()}
        _ALLOC_CACHE[key]=alloc
    if not alloc or r["player_id"] not in alloc:
        return None
    return {
        "player_share":alloc[r["player_id"]],
        "share_sum":sum(alloc.values()),
        "roster_n":len(alloc),
    }


def team_opportunity_receipt(team_totals,team_opp,r,outcome):
    sp=SPEC[outcome]; target=(r["season"],r["week"])
    own=p1a.prior_team_values(team_totals,team_opp,target,r["team"],sp["opp"],TEAM_WINDOW,False)
    allowed=p1a.prior_team_values(team_totals,team_opp,target,r["opponent"],sp["opp"],OPP_WINDOW,True)
    league=league_team_prior(target,sp["opp"])
    if len(own)<2 or len(allowed)<2 or league is None:
        return None
    own_v=ewma(own); allowed_v=ewma(allowed)
    proj=TEAM_W[0]*own_v+TEAM_W[1]*allowed_v+TEAM_W[2]*league
    return {"own_recent":own_v,"opponent_allowed_recent":allowed_v,"league_prior":league,"projection":proj}


def player_efficiency(players,target,r,outcome):
    sp=SPEC[outcome]
    hist=p1a.prior_player_rows(players,target,r["player_id"],None,EFF_WINDOW)
    hist=[x for x in hist if x.get(sp["opp"],0.0)>0]
    den=sum(x[sp["opp"]] for x in hist)
    if den<=0:
        return None
    return sum(x[sp["value"]] for x in hist)/den


def opponent_efficiency(team_totals,team_opp,target,defense,outcome):
    key=(target,defense,outcome)
    if key in _OPP_EFF_CACHE:
        return _OPP_EFF_CACHE[key]
    sp=SPEC[outcome]
    vals=[]
    for tkey,st in _DEF_INDEX.get(defense,()):
        if tkey>=target:
            break
        den=st.get(sp["opp"],0.0)
        if den>0:
            vals.append((tkey,st.get(sp["value"],0.0),den))
    vals=vals[-EFF_WINDOW:]
    den=sum(x[2] for x in vals)
    ans=None if den<=0 else sum(x[1] for x in vals)/den
    _OPP_EFF_CACHE[key]=ans
    return ans


def efficiency_receipt(players,team_totals,team_opp,r,outcome):
    sp=SPEC[outcome]; target=(r["season"],r["week"])
    pe=player_efficiency(players,target,r,outcome)
    de=opponent_efficiency(team_totals,team_opp,target,r["opponent"],outcome)
    pp=position_eff_prior(players,team_totals,target,(r.get("position") or "").upper(),sp["value"],sp["opp"])
    if pe is None or de is None or pp is None:
        return None
    final=EFF_W[0]*pe+EFF_W[1]*de+EFF_W[2]*pp
    return {"player_recent_efficiency":pe,"opponent_allowed_efficiency":de,"position_prior_efficiency":pp,"projection":final}


def qb_receipt(players,team_totals,team_opp,r):
    target=(r["season"],r["week"])
    hist=p1a.prior_player_rows(players,target,r["player_id"],r["team"],3)
    attempts=[x["attempts"] for x in hist if x["attempts"]>0]
    if len(attempts)<2:
        return None

    allowed=p1a.prior_team_values(team_totals,team_opp,target,r["opponent"],"attempts",3,True)
    league=league_team_prior(target,"attempts")
    if len(allowed)<2 or league is None:
        return None
    qb_attempt=ewma(attempts)
    opp_attempt=ewma(allowed)
    attempt_proj=QB_ATTEMPT_W[0]*qb_attempt+QB_ATTEMPT_W[1]*opp_attempt+QB_ATTEMPT_W[2]*league

    ph=[x for x in p1a.prior_player_rows(players,target,r["player_id"],None,EFF_WINDOW) if x["attempts"]>0]
    den=sum(x["attempts"] for x in ph)
    if den<=0:
        return None
    player_ypa=sum(x["passing_yards"] for x in ph)/den
    opp_ypa=opponent_efficiency(team_totals,team_opp,target,r["opponent"],"pass_yds")
    pos_ypa=position_eff_prior(players,team_totals,target,"QB","passing_yards","attempts")
    if opp_ypa is None or pos_ypa is None:
        return None
    eff=QB_EFF_W[0]*player_ypa+QB_EFF_W[1]*opp_ypa+QB_EFF_W[2]*pos_ypa
    return {
        "player_recent_attempts":qb_attempt,
        "opponent_allowed_attempts":opp_attempt,
        "league_attempt_prior":league,
        "player_opportunity_projection":attempt_proj,
        "player_recent_efficiency":player_ypa,
        "opponent_allowed_efficiency":opp_ypa,
        "position_prior_efficiency":pos_ypa,
        "final_efficiency_projection":eff,
        "point_projection":attempt_proj*eff,
    }


def receipt(players,team_totals,team_opp,r,outcome):
    if outcome=="pass_yds":
        return qb_receipt(players,team_totals,team_opp,r)

    t=team_opportunity_receipt(team_totals,team_opp,r,outcome)
    sh=coherent_share(players,team_totals,r,outcome)
    e=efficiency_receipt(players,team_totals,team_opp,r,outcome)
    if t is None or sh is None or e is None:
        return None
    player_opp=t["projection"]*sh["player_share"]
    return {
        "team_opportunity_projection":t["projection"],
        "team_own_recent_opportunities":t["own_recent"],
        "team_opponent_allowed_opportunities":t["opponent_allowed_recent"],
        "team_league_prior_opportunities":t["league_prior"],
        "player_role_share":sh["player_share"],
        "role_share_sum":sh["share_sum"],
        "roster_n":sh["roster_n"],
        "player_opportunity_projection":player_opp,
        "player_recent_efficiency":e["player_recent_efficiency"],
        "opponent_allowed_efficiency":e["opponent_allowed_efficiency"],
        "position_prior_efficiency":e["position_prior_efficiency"],
        "final_efficiency_projection":e["projection"],
        "point_projection":player_opp*e["projection"],
    }


def fixed_rows(players,season,outcome,max_week=18):
    source=[r for r in players if r["season"]==season and 1<=r["week"]<=max_week]
    return p1b.fixed_meaningful_rows(players,source,outcome)


def evaluate(players,team_totals,team_opp,rows,outcome):
    sp=SPEC[outcome]
    errs=[]; signed=[]; opp_err=[]; role_err=[]; recs=[]
    for r in rows:
        p=receipt(players,team_totals,team_opp,r,outcome)
        if p is None:
            continue
        y=r[sp["value"]]; ao=r[sp["opp"]]
        point=p["point_projection"]; po=p["player_opportunity_projection"]
        errs.append(abs(point-y)); signed.append(point-y); opp_err.append(abs(po-ao))
        actual_team=team_totals.get((r["season"],r["week"],r["team"]),{}).get(sp["opp"],0.0)
        ashare=(ao/actual_team) if actual_team>0 and outcome!="pass_yds" else None
        if ashare is not None and p.get("player_role_share") is not None:
            role_err.append(abs(p["player_role_share"]-ashare))
        recs.append({
            "season":r["season"],"week":r["week"],"player_id":r["player_id"],
            "team":r["team"],"opponent":r["opponent"],"position":r.get("position"),
            "outcome":outcome,"actual":y,"point_projection":point,
            "abs_error":abs(point-y),"error":point-y,
            "actual_opportunities":ao,"player_opportunity_projection":po,
            "opportunity_abs_error":abs(po-ao),
            "actual_role_share":ashare,
            **p,
        })
    return {
        "n":len(errs),"mae":mean(errs),"bias":mean(signed),
        "player_opportunity_mae":mean(opp_err),"role_share_mae":mean(role_err),
        "rows":recs,
    }


def summarize(ev,outcome):
    if not ev["n"]:
        return {"n":0}
    es=[r["abs_error"] for r in ev["rows"]]
    out={
        "n":ev["n"],"mae":ev["mae"],"bias":ev["bias"],
        "median_absolute_error":statistics.median(es),
        "player_opportunity_mae":ev["player_opportunity_mae"],
        "role_share_mae":ev["role_share_mae"],
    }
    if outcome.endswith("_yds"):
        out["within"]={str(t):sum(e<=t for e in es)/len(es) for t in (5,10,15,20,25,30,40,50)}
    else:
        out["within"]={str(t):sum(e<=t for e in es)/len(es) for t in (0,1,2,3)}
    out["catastrophic_miss_rate"]=sum(e>(75 if outcome=="pass_yds" else 40 if outcome.endswith("_yds") else 3) for e in es)/len(es)
    return out


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
    build_indexes(players,team_totals,team_opp)
    roster_meta=load_roster_membership(a.roster)
    p1b_snapshot=json.loads(Path(a.phase1b_snapshot).read_text())

    report={
        "schema":"nfl-v2-competent-human-baseline-v1",
        "status":"FROZEN_TRANSPARENT_BENCHMARK",
        "sportsbook_inputs_used":False,
        "monte_carlo_used":False,
        "hyperparameter_search_used":False,
        "formula":{
            "team_weights":{"own_recent":TEAM_W[0],"opponent_allowed":TEAM_W[1],"league":TEAM_W[2]},
            "efficiency_weights":{"player_recent":EFF_W[0],"opponent_allowed":EFF_W[1],"position_prior":EFF_W[2]},
            "qb_attempt_weights":{"player_recent":QB_ATTEMPT_W[0],"opponent_allowed":QB_ATTEMPT_W[1],"league":QB_ATTEMPT_W[2]},
            "qb_efficiency_weights":{"player_recent":QB_EFF_W[0],"opponent_allowed":QB_EFF_W[1],"qb_prior":QB_EFF_W[2]},
            "team_window":TEAM_WINDOW,"role_window":ROLE_WINDOW,"efficiency_window":EFF_WINDOW,"ewma_decay":DECAY,
        },
        "historical_roster_note":"Weekly roster is used for team membership/position only; historical game-status fields are intentionally ignored.",
        "roster_meta":roster_meta,
        "outcomes":{},
    }

    for oc in OUTCOMES:
        ev2025=evaluate(players,team_totals,team_opp,fixed_rows(players,2025,oc),oc)
        ev2026=evaluate(players,team_totals,team_opp,fixed_rows(players,2026,oc,4),oc)
        report["outcomes"][oc]={
            "validation_2025":summarize(ev2025,oc),
            "diagnostic_2026_wk1_4":summarize(ev2026,oc),
            "phase1b_validation_2025":{
                "mae":p1b_snapshot["outcomes"][oc]["burned_validation_2025"]["mae"],
                "player_opportunity_mae":p1b_snapshot["outcomes"][oc]["burned_validation_2025"]["player_opportunity_mae"],
                "role_share_mae":p1b_snapshot["outcomes"][oc]["burned_validation_2025"]["role_share_mae"],
            },
            "largest_2025_misses":sorted(ev2025["rows"],key=lambda x:-x["abs_error"])[:20],
        }

    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps({
        oc:{
            "validation_2025":v["validation_2025"],
            "diagnostic_2026_wk1_4":v["diagnostic_2026_wk1_4"],
            "phase1b_validation_2025":v["phase1b_validation_2025"],
        } for oc,v in report["outcomes"].items()
    },indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())