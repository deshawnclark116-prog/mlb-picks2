"""Phase1G research: pregame player-specific opportunity allocation.

Separate from frozen Phase1A/1C. No sportsbook, postgame, unsupported
weather, assumed confirmed starter, or outcome-informed parameters.
"""
from __future__ import annotations
import copy
from datetime import timedelta
import numpy as np

VERSION="phase1g-player-role-a"
MIN_GAMES=3
HALF_LIFE=3.5
PRIOR_STRENGTH=5.0
MAX_SHARE_MULTIPLIER=1.45
MIN_SHARE_MULTIPLIER=0.60

class RoleEvidenceError(ValueError):pass

def _history(p,cutoff):
    valid=[]
    for h in p.get("hist",[]):
        if h["kick"]+timedelta(hours=24)>cutoff:
            raise RoleEvidenceError("HISTORICAL_ROLE_LEAKAGE")
        if h.get("team")!=p.get("_target_team"):
            continue
        if h.get("played"):
            valid.append(h)
    return valid[-12:]

def _recent(hist,key):
    a=[float(h[key]) for h in hist if h.get(key) is not None
       and np.isfinite(float(h[key]))]
    if not a:return None,0
    weights=np.exp2(-np.arange(len(a)-1,-1,-1)/HALF_LIFE)
    return float(np.dot(a,weights)/weights.sum()),len(a)

def adjust_team(g,unit,horizon,cutoff):
    """Change target/carry P1 only when observed pregame role evidence exists.

    Missing routes or missing active roster are uncertainty, not zero usage.
    Keep outside-player allocation and probability conservation. Return
    complete feature/adjustment receipt for prospective component grading.
    """
    if horizon not in ("T24","T90"):raise RoleEvidenceError("BAD_HORIZON")
    if cutoff.tzinfo is None:raise RoleEvidenceError("NAIVE_CUTOFF")
    if unit["kick"]<=cutoff:raise RoleEvidenceError("AFTER_KICKOFF")
    if tuple(unit["key"])!=tuple(g["key"]):
        raise RoleEvidenceError("WRONG_TEAM_SOURCE")
    out=copy.deepcopy(g)
    players={p["gid"]:{**p,"_target_team":unit["key"][2]}
             for p in unit.get("players",[])}
    pk="pact24" if horizon=="T24" else "pact90"
    receipts=[]
    for group,share_key,role_key in (("target","tgt_sh","route_rt"),
                                      ("carry","car_sh","snap_pct")):
        t=out["types"][group]
        p0=np.asarray(t["P1"],float)
        ids=list(t["ids"])
        if len(ids)!=len(p0):raise RoleEvidenceError("ALLOCATION_SHAPE_MISMATCH")
        if np.any(~np.isfinite(p0)) or np.any(p0<0) or p0.sum()>1.000001:
            raise RoleEvidenceError("BAD_FROZEN_ALLOCATION")
        new=p0.copy()
        locked=np.asarray(t[pk],float)<=0
        for i,pid in enumerate(ids):
            p=players.get(pid)
            if p is None or float(t[pk][i])<=0:
                continue
            if p.get("max_info_ts_"+horizon) is not None and p["max_info_ts_"+horizon]>cutoff:
                raise RoleEvidenceError("PLAYER_SOURCE_AFTER_CUTOFF")
            hist=_history(p,cutoff)
            if len(hist)<MIN_GAMES:continue
            recent,n=_recent(hist,share_key)
            deployment,nd=_recent(hist,role_key)
            if recent is None or n<MIN_GAMES:continue
            # Deployment is a separate observable, not another yardage mean.
            # If route/snap data missing, retain a weak role observation.
            reliability=min(1.,n/8.)*(0.5 if deployment is None or nd<3 else 1.)
            if deployment is not None:
                deployment=float(np.clip(deployment,0,1))
            if not 0<=recent<=1:continue
            # Prior retained; evidence is a *bounded* change, not a reset.
            post=(PRIOR_STRENGTH*p0[i]+n*reliability*recent)/(
                PRIOR_STRENGTH+n*reliability)
            # Recent route/snap deployment supplies an explicit warning
            # against extrapolating rare low-deployment usage.
            if deployment is not None and deployment<0.15:
                post=min(post,p0[i])
            new[i]=float(np.clip(post,p0[i]*MIN_SHARE_MULTIPLIER,
                                    p0[i]*MAX_SHARE_MULTIPLIER))
            receipts.append({"group":group,"player_id":pid,"games":n,
                "baseline_share":float(p0[i]),"recent_share":recent,
                "role_deployment":deployment,"reliability":reliability,
                "candidate_share":float(new[i]),"status":"RESEARCH_PRIOR"})
        # Preserve frozen outside-player share, rather than manufacturing
        # new opportunities. Reallocation is zero-sum within listed players.
        total=float(p0.sum())
        free=~locked
        free_total=float(p0[free].sum())
        if free_total>0 and new[free].sum()>0:
            new[free]*=free_total/new[free].sum()
        new[locked]=p0[locked]
        if np.any(new<0) or new.sum()>1.000001:
            raise RoleEvidenceError("BAD_NEW_ALLOCATION")
        t["P1"]=new
        for row in receipts:
            if row["group"]==group:
                row["normalized_share"]=float(new[ids.index(row["player_id"])])
    return out,receipts

def apply(pack,units,season,week,teams,horizon,cutoff):
    result={"games":copy.deepcopy(pack["games"]),"meta":copy.deepcopy(pack["meta"])}
    report={}
    for team in teams:
        key=(season,week,team)
        if key not in units or key not in result["games"]:
            raise RoleEvidenceError("MISSING_PLAYER_PREGAME_SOURCE")
        result["games"][key],report[team]=adjust_team(
            result["games"][key],units[key],horizon,cutoff)
    return result,report
