"""Phase 1F: opponent / coaching / personnel-aware JOINT opportunity challenger.

RESEARCH SHADOW ONLY. This is a pre-registered structural hypothesis, NOT
a promoted model. Produces finite, coupled team offensive volume (rushes +
dropbacks = plays in every draw), while existing Phase 1C still simulates each
individual carry/target/air-yard/YAC/TD. Existing selected Phase 1A prefit
means remain strong priors. No sportsbook input, no article-driven inference,
no target-game result, and no source invented for weather or cornerback.
All inputs must come from pre-cutoff Phase 1 Data units and frozen CAS pack.
HYPERPARAMETERS BELOW ARE RESEARCH PRIOR CHOICES, NOT VALIDATED COEFFICIENTS.
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import timezone
from math import exp, isfinite
import numpy as np

VERSION = "nfl-phase1f-joint-context-a"
# A priori bounded blending; no Week 4-9 outcome tuning.
PRIOR_WEIGHT = 0.60
RECENT_WEIGHT = 0.25
OPP_WEIGHT = 0.15
HALF_LIFE = 4.0
MAX_RELATIVE_SHIFT = 0.18
PASS_RATE_RANGE = (0.26, 0.83)
PACE_SHAPE = 24.0  # common within-game uncertainty, tested for numerical stability
MIN_QB_RESEARCH_PROB = 0.0


class EvidenceError(ValueError):
    pass


def _num(v):
    try:
        f=float(v)
        return f if isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _ewma(hist, key, *, coach=None):
    rows=[r for r in hist if _num(r.get(key)) is not None and (
        coach is None or r.get("coach") == coach)]
    if not rows:
        return None,0
    a=np.array([float(r[key]) for r in rows[-10:]],float)
    k=np.arange(len(a))[::-1]
    w=np.exp2(-k/HALF_LIFE)
    return float(w@a/w.sum()),len(a)


def _time_ok(unit, cutoff):
    """Do not allow a target game (or too-recent old game) into feature history."""
    if cutoff.tzinfo is None:
        raise EvidenceError("CUTOFF_MUST_BE_TIMEZONE_AWARE")
    if unit["kick"] <= cutoff:
        raise EvidenceError("FORECAST_AFTER_OR_AT_KICKOFF")
    if unit.get("last_info_ts") is not None and unit["last_info_ts"] > cutoff:
        raise EvidenceError("LEAKAGE_LAST_INFO_AFTER_CUTOFF")
    for key in ("team_hist","opp_allowed_hist"):
        for r in unit.get(key,()):
            if r["kick"] >= unit["kick"] or r["kick"] >= cutoff:
                raise EvidenceError("LEAKAGE_CURRENT_OR_FUTURE_HISTORY")
            # nfl_phase1_data.build enforces 24-hour availability of completed games.
            from datetime import timedelta
            if r["kick"] + timedelta(hours=24) > cutoff:
                raise EvidenceError("LEAKAGE_GAME_NOT_YET_AVAILABLE")


def _blend(selected, own, opp, *, own_n, opp_n):
    """Shrink contextual evidence when there are few real, as-of observations."""
    if selected is None or selected <= 0:
        raise EvidenceError("INVALID_PREFIT_TEAM_PRIOR")
    own_rel=min(1.0,own_n/4.0) if own is not None else 0.0
    opp_rel=min(1.0,opp_n/4.0) if opp is not None else 0.0
    ws=[PRIOR_WEIGHT,RECENT_WEIGHT*own_rel,OPP_WEIGHT*opp_rel]
    vals=[selected,own if own is not None else selected,
          opp if opp is not None else selected]
    mixed=sum(w*v for w,v in zip(ws,vals))/sum(ws)
    return float(np.clip(mixed, selected*(1-MAX_RELATIVE_SHIFT),
                             selected*(1+MAX_RELATIVE_SHIFT)))


def build_team_context(unit, g, cutoff):
    """Team pace, pass mix, coach continuity, opponent allowed; as-of only."""
    _time_ok(unit,cutoff)
    if tuple(unit["key"])!=tuple(g["key"]):
        raise EvidenceError("TEAM_UNIT_DOES_NOT_MATCH_PREFIT_PACK")
    t=g["types"]
    if not {"carry","qb_att","target"}<=set(t):
        raise EvidenceError("MISSING_COMPLETE_PHASE1_OPPORTUNITY_GROUPS")
    old_r=float(t["carry"]["mu"]);old_d=float(t["qb_att"]["mu"])
    old_plays=old_r+old_d
    if old_plays<=0:raise EvidenceError("EMPTY_TEAM_PREFIT_PRIOR")
    team_hist=unit.get("team_hist") or []
    allowed_hist=unit.get("opp_allowed_hist") or []
    # Coaching tendencies: only use a coach-specific history when it is
    # explicitly present and has >=2 observed games. Unknown != unchanged.
    coach=unit.get("coach")
    has_coach=bool(coach) and sum(x.get("coach")==coach
                        for x in team_hist)>=2
    own=([x for x in team_hist if x.get("coach")==coach]
         if has_coach else team_hist)
    recent_plays,n1=_ewma(own,"plays")
    opp_plays,n2=_ewma(allowed_hist,"plays")
    if recent_plays is not None and recent_plays<=0: recent_plays=None;n1=0
    if opp_plays is not None and opp_plays<=0: opp_plays=None;n2=0
    plays=_blend(old_plays,recent_plays,opp_plays,own_n=n1,opp_n=n2)
    own_db,nd=_ewma(own,"dropbacks"); own_r,nr=_ewma(own,"rushes")
    opp_db,nod=_ewma(allowed_hist,"dropbacks");opp_r,nor_=_ewma(allowed_hist,"rushes")
    own_pass=own_db/(own_db+own_r) if own_db is not None and own_r is not None and own_db+own_r>0 else None
    opp_pass=opp_db/(opp_db+opp_r) if opp_db is not None and opp_r is not None and opp_db+opp_r>0 else None
    selected_rate=old_d/old_plays
    pass_rate=_blend(selected_rate,own_pass,opp_pass,
                    own_n=min(nd,nr),opp_n=min(nod,nor_))
    pass_rate=float(np.clip(pass_rate,*PASS_RATE_RANGE))
    pk="pact24" if g.get("_phase1f_horizon")=="T24" else "pact90"
    q=t["qb_att"]
    starter_status="UNVERIFIED_QB_DEPTH_CONTEXT"
    rank1_id=None
    qb_hist={p["gid"]:p for p in unit.get("players",()) if p.get("pos")=="QB"}
    # Rank-one depth-chart information is only a *soft prior*.
    for gid in q["ids"]:
        p=qb_hist.get(gid)
        if not p:continue
        dep=p.get("depth")
        ts=p.get("depth_ts")
        if dep and len(dep)>1 and str(dep[0]).upper()=="QB" and float(dep[1])==1 and ts is not None and ts<=cutoff:
            rank1_id=gid
            starter_status="RANK1_DEPTH_CHART_PRIOR_NOT_CONFIRMED_STARTER"
            break
    unavailable=[str(gid) for gid,p in zip(q["ids"],q[pk]) if float(p)<=0.0]
    return {"team":unit["key"][2],"coach_status":(
            "MATCHED_RECENT_COACH_TENDENCIES" if has_coach
            else "COACH_CONTINUITY_UNVERIFIED_FALLBACK_TO_TEAM"),
        "recent_offensive_games":n1,"recent_opponent_defense_games":n2,
        "preselected_plays":old_plays,"projected_plays":plays,
        "preselected_pass_fraction":selected_rate,"pass_fraction":pass_rate,
        "qb_depth_rank1_id":rank1_id,"qb_status":starter_status,
        "unavailable_qb_ids":unavailable,
        "rest_days":_num(unit.get("rest")),
        "weather_status":"NOT_IN_VERIFIED_PHASE1_SNAPSHOT_NO_WEATHER_OVERRIDE",
        "specific_corner_matchup_status":"NOT_VERIFIABLE_FROM_CURRENT_TRACKING_FEED",
        "source":"AS_OF_PHASE1_HISTORIES_PLUS_FROZEN_PREFIT_PACK"}


def add_soft_qb_prior(g, context):
    """Modify actual QB allocation probabilities, never fabricate a starter.

    Only a timestamp-valid rank-1 depth-chart QB gets a predeclared, bounded
    preference among listed candidates; zero-probability unavailable QBs stay
    excluded. Unlisted QB outside-bucket stays possible. No role claim made.
    """
    qt=g["types"]["qb_att"]
    gid=context["qb_depth_rank1_id"]
    if gid is None:return
    ids=list(qt["ids"])
    if gid not in ids:return
    ix=ids.index(gid)
    pk="pact24" if g["_phase1f_horizon"]=="T24" else "pact90"
    if float(qt[pk][ix])<=MIN_QB_RESEARCH_PROB:
        context["qb_status"]="RANK1_QB_UNAVAILABLE_NOT_ASSIGNED"
        return
    p=np.array(qt["P1"],float).copy()
    # The learned role prior remains dominant; boost max 30% with no certainty.
    p[ix]=min(p[ix]*1.30,1.0)
    qt["P1"]=p
    context["qb_status"]="RANK1_SOFT_QB_ALLOCATION_NOT_STARTER_VERIFICATION"


def make_challenger_pack(pack, context_units, season, week, teams, horizon, cutoff):
    if horizon not in ("T24","T90"):raise EvidenceError("INVALID_FORECAST_HORIZON")
    if len(teams)!=2 or teams[0]==teams[1]:raise EvidenceError("INVALID_GAME_TEAMS")
    if cutoff.tzinfo is None: raise EvidenceError("MISSING_CUTOFF_TIMEZONE")
    view={"games":{},"meta":copy.deepcopy(pack["meta"])}
    report={}
    for team in teams:
        k=(season,week,team)
        u=context_units.get(k)
        g=pack["games"].get(k)
        if u is None or g is None:
            raise EvidenceError("MISSING_AS_OF_CONTEXT_OR_PACK_FOR_"+team)
        x=copy.deepcopy(g)
        x["_phase1f_horizon"]=horizon
        c=build_team_context(u,x,cutoff)
        add_soft_qb_prior(x,c)
        view["games"][k]=x
        report[team]=c
    return view,report


def joint_scripts(report, teams, N, seed, original_pack):
    """Simulated team plays are *not* independent pass and rush draws.

    A common within-matchup tempo latent affects both sides. Each team then
    draws NB plays and beta-binomial passing fraction. R+D=plays exactly.
    Game scores or game-day box-score facts are NEVER read here.
    """
    if N<10000: raise EvidenceError("INSUFFICIENT_SIM_DRAWS")
    rng=np.random.default_rng(seed)
    common=rng.gamma(PACE_SHAPE,1.0/PACE_SHAPE,size=N)
    scripts={}
    for team in teams:
        c=report[team]
        mu=float(c["projected_plays"])
        kpack=original_pack["games"]
        gs=[v for key,v in kpack.items() if key[2]==team]
        if len(gs)!=1: raise EvidenceError("AMBIGUOUS_TEAM_PACK")
        t=gs[0]["types"]
        kk=max(3.0,float(min(t["carry"]["k"],t["qb_att"]["k"])))
        # independent residual variation, shared tempo; no negative totals.
        lam=rng.gamma(kk,mu*common/kk)
        plays=rng.poisson(lam).astype(np.int64)
        frac=float(c["pass_fraction"])
        # No verified starter => deliberately larger uncertainty in pass/run mix.
        concentration=32.0 if c["qb_status"].startswith("UNVERIFIED") else 48.0
        a=max(frac*concentration,1e-5)
        b=max((1-frac)*concentration,1e-5)
        pass_rate=rng.beta(a,b,size=N)
        db=rng.binomial(plays,pass_rate).astype(np.int64)
        rushes=plays-db
        assert np.array_equal(rushes+db,plays)
        scripts[team]={"R":rushes,"D":db}
        c["simulated_mean_plays"]=float(plays.mean())
        c["simulated_mean_dropbacks"]=float(db.mean())
        c["simulated_mean_rushes"]=float(rushes.mean())
    return scripts


def challenger_identity(base_version, original_content_id, config=None):
    """Stable model identity; never label Phase1F as original Phase1C champion."""
    b=json.dumps({"version":VERSION,"base_version":base_version,
                  "config":config or {},"snapshot":original_content_id},sort_keys=True)
    return "p1f-"+hashlib.sha256(b.encode()).hexdigest()[:20]
