"""Phase 1G individual NFL opportunity and matchup challenger (RESEARCH ONLY).

Adds *real calculations* to Phase 1F:
 - as-of per-player target/carry + snap-role trends, with shrinkage and
   bounded movement from the frozen Phase 1A prior;
 - dropout/inactive gates and finite redistribution across teammates;
 - documented per-position opponent yards/target or yards/carry rates,
   shrinkage to league and controlled event-efficiency PMF adjustments;
 - coupled score-scenario and weather-conditioned offensive volume;
 - exact source/decision receipts for every player.

No sportsbook lines, outcome hindsight, unsupported route tracking, inferred
confirmed starter, or forged weather are allowed. Tuning constants below are
hypotheses, NOT validated predictive gains. Never touch the frozen predictor.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime,timedelta,timezone
from pathlib import Path

import numpy as np
import nfl_phase1f_intelligence as F

VERSION = "NFL_PHASE1G_PERSONNEL_ROLE_MATCHUP_JOINT_RESEARCH_A"
# Fixed ex ante; older 2026 games are already a burned research set.
ROLE_BASE_PRIOR = 0.70
ROLE_RECENT_MIX = 0.65
MAX_ROLE_SHIFT = 0.30
MAX_EFFICIENCY_RATIO = 0.12
MAX_SCRIPT_PASS_SHIFT = 0.06
MIN_DEF_GAMES = 3
QB_UNKNOWN = "NO_VERIFIED_GAME_STARTER"


class EvidenceError(ValueError):
    pass


def _asof(row, cutoff):
    if row.get("kick") is None or row["kick"]+timedelta(hours=24)>cutoff:
        return False
    return True


def _weighted(seq, key, window=8):
    rows=[x for x in seq if x.get(key) is not None and math.isfinite(float(x[key]))]
    rows=rows[-window:]
    if not rows:return None,0
    w=np.exp2(-np.arange(len(rows))[::-1]/3.0)
    return float(np.dot(w,[float(x[key]) for x in rows])/w.sum()),len(rows)


def _role_target(g, player, kind):
    hist=[z for z in player.get("hist",[]) if z.get("team")==g["key"][2]]
    if not hist:return None,{"n_games":0,"source":"NO_MATCHED_TEAM_HISTORY"}
    col={"target":"tgt_sh","carry":"car_sh"}[kind]
    share,n=_weighted(hist,col)
    old=[z for z in hist[:-3] if z.get(col) is not None]
    oldmean,m=_weighted(old,col,window=10)
    recent=[z for z in hist[-3:] if z.get(col) is not None]
    recentmean,rm=_weighted(recent,col,window=3)
    if share is None:return None,{"n_games":0,"source":"NO_TEAM_DENOMINATOR"}
    changed=share
    if m>=2 and rm>=2:
        changed=ROLE_RECENT_MIX*recentmean+(1-ROLE_RECENT_MIX)*oldmean
    snap,n_snap=_weighted(hist,"snap_pct")
    past_snap,m_snap=_weighted(hist[:-3],"snap_pct")
    if n_snap>=3 and m_snap>=2 and snap is not None and past_snap and past_snap>0.1:
        # Snap share is independent of targets, but overlap may be large:
        # only 15% of a supported snap trend enters the usage candidate.
        changed*=float(np.clip(1+0.15*(snap/past_snap-1),0.88,1.12))
    route_games=[z for z in hist if z.get("routes") is not None]
    route_quality="UNAVAILABLE"
    if len(route_games)>=4:
        # Use historical real routes, not projected 2026 tracking.
        tprr=(sum(z["tgt"] for z in route_games)+10.2)/(
            sum(z["routes"] for z in route_games)+60.0)
        rr,nr=_weighted(route_games,"route_rt")
        if nr>=4 and rr is not None and 0<=rr<=1:
            route_quality="HISTORICAL_ROUTE_DATA_ONLY"
            # Relative route volume tracks future availability, not target rate
            # certainty. Cannot use routes from future/target game.
            changed=0.9*changed+0.1*rr*tprr
    return max(0.,changed),{
        "n_games":n,"share_all":share,"share_last3":recentmean,
        "share_earlier":oldmean,"recent_snap_pct":snap,
        "route_status":route_quality,"source":"COMPLETED_PLAYER_GAMES_AS_OF"}


def apply_player_roles(pack,units,season,week,teams,horizon,cutoff):
    """Change the *inputs* to Phase 1C teammate multinomial allocations."""
    k="pact24" if horizon=="T24" else "pact90"
    changes={}
    for team in teams:
        key=(season,week,team)
        g=pack["games"][key]
        u=units[key]
        if u["kick"]<=cutoff:raise EvidenceError("TARGET_GAME_NOT_IN_FUTURE")
        byid={x["gid"]:x for x in u.get("players",[])}
        team_receipt={}
        for kind in ("target","carry"):
            t=g["types"][kind]
            ids=list(t["ids"])
            old=np.asarray(t["P1"],float)
            active=np.asarray(t[k],float)
            if len(old)!=len(ids) or len(active)!=len(ids):
                raise EvidenceError("ROLE_VECTOR_ID_LENGTH_MISMATCH")
            new=old.copy()
            evidence=[]
            baseline=float(old.sum())
            for j,gid in enumerate(ids):
                p=byid.get(gid)
                oldp=max(float(old[j]),0.)
                if float(active[j])<=0.0:
                    new[j]=0.
                    evidence.append({"player_id":gid,"status":"SOURCE_GATED_UNAVAILABLE",
                                     "original_weight":oldp,"new_weight":0.})
                    continue
                if p is None:
                    evidence.append({"player_id":gid,"status":"ROLE_HISTORY_MISSING",
                                     "original_weight":oldp,"new_weight":oldp})
                    continue
                hist=[z for z in p.get("hist",[]) if z.get("team")==team]
                if any(not _asof(z,cutoff) for z in hist):
                    raise EvidenceError("PLAYER_HISTORY_AFTER_INFORMATION_CUTOFF")
                share,rec=_role_target(g,p,kind)
                if share is None or rec["n_games"]<3:
                    evidence.append({"player_id":gid,"status":"INSUFFICIENT_HISTORY_PRIOR_ONLY",
                                     "original_weight":oldp,"new_weight":oldp,**rec})
                    continue
                peers=[_role_target(g,x,kind)[0] for x in u.get("players",[])
                       if x["gid"] in ids and x.get("pos") in ("WR","TE","RB","QB")]
                supported=sum(v for v in peers if v is not None and v>0)
                baseline_share=oldp/max(baseline,1e-9)
                if supported<=0 or baseline_share<=0:
                    evidence.append({"player_id":gid,"status":"NO_STABLE_PEER_SHARE",
                                     "original_weight":oldp,"new_weight":oldp,**rec})
                    continue
                current_share=share/supported
                ratio=float(np.clip(current_share/baseline_share,
                                    1-MAX_ROLE_SHIFT,1+MAX_ROLE_SHIFT))
                reliability=min(1.,rec["n_games"]/8.0)
                new[j]=oldp*(1+(1-ROLE_BASE_PRIOR)*reliability*(ratio-1))
                evidence.append({"player_id":gid,"status":"RECENT_ROLE_CONDITIONED",
                                 "original_weight":oldp,"new_weight":float(new[j]),
                                 "applied_multiplier":float(new[j]/oldp)
                                 if oldp else None,**rec})
            # Inactive weights never come back. Named-target share stays
            # close to the frozen prior; outside bucket remains independent
            # and absorbs missed/newly signed candidates.
            new=np.maximum(new,0.)
            for j,gid in enumerate(ids):
                if active[j]<=0 and new[j]!=0:raise EvidenceError("INACTIVE_RESURRECTED")
            if new.sum()>0 and baseline>0:
                # Bound overall named propensity mass rather than chasing
                # a postgame target-volume factor.
                target=np.clip(new.sum(),baseline*.85,baseline*1.15)
                new*=target/new.sum()
            t["P1"]=new
            for i,item in enumerate(evidence):
                item["new_weight"]=float(new[i])
            team_receipt[kind]={"player_evidence":evidence,
                  "named_weight_before":baseline,
                  "named_weight_after":float(new.sum()),
                  "outside_weight_preserved":True,
                  "inactive_count":int(np.sum(active<=0))}
        # QB distribution controlled separately with depth and source gates.
        team_receipt["qb"]={"status":g.get("_phase1g_qb_status",
            QB_UNKNOWN),"no_inferred_confirmed_starter":True}
        changes[team]=team_receipt
    return changes


def defensive_rates(asof_root,data,cutoff,season):
    """Raw 2025-26 same-snapshot weekly stats, only played-and-available games.

    Position-specific defensive receiving-yards/target and rushing-yards/
    carry; opponent aggregates and league denominator from exactly the same
    pre-cutoff raw games. Never read target-game results.
    """
    vals=defaultdict(lambda:{"events":0.,"yards":0.,"games":set()})
    league=defaultdict(lambda:{"events":0.,"yards":0.,"games":set()})
    for year in (season-1,season):
        src=Path(asof_root)/f"stats_player_week_{year}.csv"
        if not src.exists():continue
        with src.open(newline="",encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("season_type","REG")!="REG":continue
                try: key=(year,int(r["week"]),r["team"])
                except (ValueError,KeyError):continue
                ctx=data.game.get(key)
                if not ctx or ctx["kick"]+timedelta(hours=24)>cutoff:continue
                if ctx["kick"]>=cutoff:raise EvidenceError("DEFENSE_FUTURE_GAME_LEAK")
                pos=r.get("position","")
                if pos=="FB":pos="RB"
                if pos not in ("WR","TE","RB","QB"):continue
                for typ,cnt,yds in (
                    ("receiving",r.get("targets"),r.get("receiving_yards")),
                    ("rushing",r.get("carries"),r.get("rushing_yards"))):
                    try: n=float(cnt or 0);v=float(yds or 0)
                    except ValueError:continue
                    if not (math.isfinite(n) and math.isfinite(v) and n>0):
                        continue
                    opp=ctx["opp"]
                    for dest,dkey in ((vals,(opp,typ,pos)),(league,(typ,pos))):
                        d=dest[dkey];d["events"]+=n;d["yards"]+=v
                        d["games"].add((year,key[1],r["team"]))
    out={}
    for (defteam,typ,pos),a in vals.items():
        b=league[(typ,pos)]
        ref=b["yards"]/b["events"] if b["events"]>0 else None
        n=a["events"]
        threshold=40. if typ=="receiving" else 50.
        if not ref or ref<=0 or n<threshold or len(a["games"])<MIN_DEF_GAMES:
            continue
        # 100-pseudo-event prior. Extremely volatile team rank is not used.
        posterior=(a["yards"]+100.*ref)/(n+100.)
        ratio=float(np.clip(posterior/ref,
                            1-MAX_EFFICIENCY_RATIO,
                            1+MAX_EFFICIENCY_RATIO))
        out[(defteam,typ,pos)]={"ratio":ratio,"events":n,
            "n_games":len(a["games"]),"opp_yards_per_event":a["yards"]/n,
            "league_yards_per_event":ref,
            "source":"SNAPSHOT_STATS_ONLY_24H_LAG_AND_BAYES_SHRINK"}
    return out


def _tilt_pmf(row, bins, ratio):
    """Small, normalized within-model shift without eliminating negative plays."""
    p=np.asarray(row,float)
    v=np.asarray(bins,float)
    if len(p)!=len(v) or p.min()<0 or p.sum()<=0:
        raise EvidenceError("INCOMPATIBLE_EFFICIENCY_BINS")
    mean=float(np.dot(p,v)/p.sum())
    sd=max(float(np.sqrt(np.dot(p,(v-mean)**2)/p.sum())),1.0)
    shift=float(np.log(ratio)*.5)
    z=np.clip(shift*(v-mean)/sd,-2.,2.)
    q=p*np.exp(z-np.max(z))
    return q/q.sum()


def matchup_event_efficiencies(base_eff,idx,constants,units,teams,season,week,defense):
    """Create an isolated per-player rush + receiving air/catch efficiency set.

    Changes actual event PMFs read by Phase1C. Preserves marginal
    normalization, all supports, all original arrays, and team QB/receiver
    accounting. This is a research-only prior, not a validated efficiency head.
    """
    out=dict(base_eff)
    out["rush"]=np.asarray(base_eff["rush"]).copy()
    out["air"]=np.asarray(base_eff["air"]).copy()
    out["catch"]=np.asarray(base_eff["catch"]).copy()
    receipts={}
    for team in teams:
        u=units[(season,week,team)]
        own={}
        for p in u.get("players",[]):
            gid=p["gid"];pos=p.get("pos")
            if pos not in ("WR","TE","RB","QB"):continue
            for typ,idxkind,arr,values in (
                ("receiving","rec","air",constants.air_v),
                ("rushing","rush","rush",constants.rush_v)):
                key=idx[idxkind].get((season,week,gid))
                e=defense.get((u["opp"],typ,pos))
                if key is None or e is None:continue
                ratio=e["ratio"]
                if arr=="air":
                    out[arr][key]=_tilt_pmf(out[arr][key],values,ratio)
                    # The opponent's receiving-yards/target proxy also
                    # contains catch-rate information. Apply a bounded
                    # conditional likelihood shift; do not overwrite QB skill.
                    c=out["catch"][key]
                    scale=float(np.clip(np.sqrt(ratio),.94,1.06))
                    out["catch"][key]=np.clip(c*scale,.001,.995)
                else:
                    out[arr][key]=_tilt_pmf(out[arr][key],values,ratio)
                own[f"{gid}/{typ}"]={"ratio":ratio,"opponent":u["opp"],
                    "position":pos,"n_events":e["events"],
                    "n_defense_games":e["n_games"],
                    "applied_to":arr}
        receipts[team]=own
    return out,receipts


def load_weather_evidence(path,game_id,kickoff,cutoff,team_venue):
    """Optional frozen evidence only; cannot construct weather after kickoff.

    Source is an NWS forecast receipt captured before model cutoff. A game
    without an exact recorded source has **no applied weather adjustment**.
    """
    if path is None:return None,{"status":"UNAVAILABLE_NO_CHANGE"}
    raw=Path(path).read_bytes();d=json.loads(raw)
    date=lambda v:datetime.fromisoformat(v.replace("Z","+00:00"))
    capture=date(d["captured_at_utc"])
    issued=date(d["forecast_issued_at_utc"])
    if d.get("game_id")!=game_id or date(d["kickoff_utc"])!=kickoff:
        raise EvidenceError("WEATHER_WRONG_GAME")
    if not (issued<=capture<=cutoff<kickoff):
        raise EvidenceError("WEATHER_NOT_ORIGINALLY_PREGAME")
    from urllib.parse import urlparse
    url=urlparse(d.get("source_url",""))
    if url.scheme!="https" or url.hostname not in ("api.weather.gov","forecast.weather.gov"):
        raise EvidenceError("WEATHER_SOURCE_NOT_NWS")
    if d.get("venue_id")!=team_venue:
        raise EvidenceError("WEATHER_VENUE_MISMATCH")
    indoor=d.get("roof_status") in ("closed","indoor")
    wind=float(d.get("wind_mph",0.))
    precip=float(d.get("precipitation_probability_pct",0.))
    if not (0<=wind<=100 and 0<=precip<=100):
        raise EvidenceError("WEATHER_INVALID_VALUE")
    state={"status":"NWS_SOURCED_PRE_CUTOFF",
        "roof_status":d["roof_status"],"wind_mph":wind,
        "precipitation_probability_pct":precip,
        "outdoor_effect":not indoor,
        "sha256":hashlib.sha256(raw).hexdigest()}
    return state,state


def correlated_script(base_context,teams,N,seed,pack,weather=None):
    """Same-game score scenario, pace and QB scramble accounting.

    Shared tempo latent and opposing score-pressure latent. This approximates
    game scripts, NOT a validated possession-by-possession simulator.
    """
    if N<10000:raise EvidenceError("INSUFFICIENT_GAME_DRAWS")
    rng=np.random.default_rng(seed)
    common=rng.gamma(F.PACE_SHAPE,1/F.PACE_SHAPE,size=N)
    pressure=rng.standard_normal(N)
    base_adv={}
    for team in teams:
        c=base_context[team]
        # As-of offensive TD rate and opponent allowed TD rate, when present.
        pts=c.get("expected_offensive_tds")
        base_adv[team]=float(pts) if pts is not None else 2.
    edge=float(np.clip((base_adv[teams[0]]-base_adv[teams[1]])/3.,-.6,.6))
    script={}
    for i,team in enumerate(teams):
        c=base_context[team]
        g=pack["games"][next(k for k in pack["games"] if k[2]==team)]
        t=g["types"]
        shape=max(3.,float(min(t["carry"]["k"],t["qb_att"]["k"])))
        rate=float(c["projected_plays"])
        lam=rng.gamma(shape,rate*common/shape)
        plays=rng.poisson(lam).astype(np.int64)
        # A positive score advantage reduces passing; opposite team gets
        # the opposite sign in the SAME scenario draw.
        score_scenario=(edge+pressure) * (1. if i==0 else -1.)
        p=float(c["pass_fraction"])-np.clip(score_scenario*.032,
                                  -MAX_SCRIPT_PASS_SHIFT,MAX_SCRIPT_PASS_SHIFT)
        if weather and weather.get("outdoor_effect"):
            # Optional wind/precip risk, only if genuine NWS pregame receipt.
            impact=min(.045,max(weather["wind_mph"]-15,0)*.0015+
                        weather["precipitation_probability_pct"]*.0001)
            p-=impact
        p=np.clip(p,.20,.88)
        concentration=40.
        rates=rng.beta(p*concentration,(1-p)*concentration)
        db=rng.binomial(plays,rates).astype(np.int64)
        design=plays-db
        scr=rng.binomial(db,float(c["scramble_rate_of_dropbacks"]))
        rush=design+scr
        if not np.array_equal(db+design,plays):
            raise EvidenceError("SCORE_SCRIPT_SNAPS_DONT_BALANCE")
        script[team]={"D":db,"R":rush}
        c["score_script"]={
            "estimated_own_offensive_td_rate":base_adv[team],
            "expected_dropbacks":float(db.mean()),
            "expected_carries":float(rush.mean()),
            "score_pressure_linked_to_opponent":True,
            "weather_adjusted":bool(weather and weather.get("outdoor_effect"))}
    return script


def build_context_team_tds(units,teams,season,week):
    """Pre-cutoff TDs per team game, tempered with opposing defense allowed."""
    x={}
    for team in teams:
        u=units[(season,week,team)]
        own,n=F._ewma(u.get("team_hist",[]),"off_td")
        opp,m=F._ewma(u.get("opp_allowed_hist",[]),"off_td")
        val=2.0
        if own is not None and opp is not None:
            val=(2*val+max(0.,own)+max(0.,opp))/4
        elif own is not None:val=(val+max(0.,own))/2
        x[team]=float(np.clip(val,.5,5))
    return x


def asof_context_with_data(asof_dir, targets):
    """Read source only once; annotate coach in a COPY of frozen Phase1 units."""
    import nfl_phase1_data as P1
    data=P1.Data(str(asof_dir))
    all_units=P1.build(data,depth_universe=True,targets=targets)
    out={}
    for u in all_units:
        if u["key"] not in set(targets):continue
        x=copy.deepcopy(u)
        team=x["key"][2]
        for row in x.get("team_hist",()):
            g=data.game.get((row["s"],row["w"],team))
            row["coach"]=(g or {}).get("coach")
        out[x["key"]]=x
    if len(out)!=len(set(targets)):
        raise EvidenceError("MISSING_AS_OF_PLAYER_TEAM_UNITS")
    return out,data


def apply_documented_personnel(pack, document, game, cutoff, capture):
    """Optional V6C exact-game primary-source receipts, never inferred starters.

    The external source content is not automatically independently verified by
    URL/timestamp validation. Label as evidence assertion, not verified fact.
    """
    if document is None:return {"status":"NO_DOCUMENTED_PERSONNEL_ASSERTIONS"}
    from nfl_v6c_pregame_personnel import read_personnel
    teams=[game["home"],game["away"]]
    normalized={"home_team":teams[0],"away_team":teams[1],
        "kickoff_utc":game["kick"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "game_id":game["game_id"]}
    rec=read_personnel(document,normalized,cutoff,
                       captured_at_utc=capture)
    applied={}
    horizon=game["horizon"]
    pk="pact24" if horizon=="T24" else "pact90"
    for team in teams:
        source=rec["teams"][team]
        g=pack["games"][(game["season"],game["week"],team)]
        out_ids=source["inactive_ids"]
        for kind,t in g["types"].items():
            if kind=="def_snap":continue
            for j,gid in enumerate(t["ids"]):
                if str(gid) in out_ids:
                    t[pk][j]=0.
                    t["P1"][j]=0.
        qb=source["starter_id"]
        qbstatus="NO_SOURCE_ASSERTED_STARTER"
        if qb:
            t=g["types"]["qb_att"]
            if qb not in list(map(str,t["ids"])):
                qbstatus="SOURCE_QB_NOT_IN_MODELED_CANDIDATES"
            else:
                j=list(map(str,t["ids"])).index(qb)
                if float(t[pk][j])<=0.0:
                    raise EvidenceError("SOURCE_QB_CONFLICTS_WITH_INACTIVE")
                # Verified source URL/timing, not verified source *content*.
                # Upgrade the role scenario but retain backup probability.
                t["P1"][j]=min(float(t["P1"][j])*1.6,1.)
                qbstatus="SOURCE_ASSERTED_STARTER_SOFT_SCENARIO"
        g["_phase1g_qb_status"]=qbstatus
        applied[team]={"removed_ids":sorted(out_ids),
                       "qb_role_status":qbstatus}
    return {"status":"SOURCE_ASSERTIONS_APPLIED_NOT_INDEPENDENTLY_VERIFIED",
            "source_sha256":rec["sha256"],"teams":applied,
            "captured_at_utc":rec["captured_at_utc"]}


def model_identity(base_version, snapshot_content_id, engine_sha, n_draws,
                   weather_sha=None, personnel_sha=None):
    """Genuine model change creates a new immutable identifier and batch."""
    data={"version":VERSION,"base_model":base_version,
          "source_snapshot":snapshot_content_id,
          "research_source_sha256":engine_sha,"draws":n_draws,
          "weather_evidence":weather_sha,"personnel_evidence":personnel_sha}
    return "p1g-"+hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:20]
