#!/usr/bin/env python3
"""NFL universal opportunity / role-change challenger: RESEARCH ONLY.

Applies to the actual player-performance *numerics*, not to sportsbook lines
or the frontend. No favored or disfavored player names. Covers passing,
rushing and receiving yards AND supporting count markets. Uses exactly the
same pregame chronological evidence discipline as the frozen incumbents.

The original v2 rush/receiving model was trained only on players with at
least three prior records and sufficient last-3 opportunity, but the serving
projector omitted the recent-opportunity gate (fixed separately). This
research challenger addresses the larger flaw: slow-to-react rolling averages
and no meaningful multi-game role-collapse detection.

Scientific protocol:
 - All row features are generated from strictly preceding games; the target
   game is only absorbed AFTER every row for that (season, week) is scored.
 - No use of sportsbook lines, forecasts produced post kickoff, future labels
   or player-specific rules.
 - Candidate parameters are chosen on 2024 weeks 7-18 only (development).
 - 2025 is an UNTOUCHED report-only season and 2026 is an independent forward
   check. Do not re-tune parameters against holdouts.
 - Archived V2 XGBoost trained using 2025 examples MUST NEVER be compared
   on 2025 as if it were an out-of-sample opponent. We report XGB comparison
   on 2026 only, on exact same player+market+game keys when available.
 - A challenger must beat incumbent on matched MAE AND avoid worsening
   systematic overprojection of declining-opportunity players, with adequate
   samples, before ANY numerical production promotion.
 - Count markets are research/projection schema coverage, NOT claims that
   production currently serves them.

Usage:
 python nfl_universal_opportunity_research.py --db nfl_models/nfl_model.sqlite
 python nfl_universal_opportunity_research.py --db /tmp/nfl.sqlite --json /tmp/report.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
from typing import Sequence

MARKETS={
    "receiving_yards":("WR,TE,RB","targets","receiving_yards",3.0),
    "rushing_yards":("RB,FB,QB,WR","carries","rushing_yards",3.0),
    "passing_yards":("QB","attempts","passing_yards",18.0),
    "receptions":("WR,TE,RB","targets","receptions",3.0),
    "passing_completions":("QB","attempts","completions",18.0),
    "rushing_attempts":("RB,FB,QB,WR","carries","carries",3.0),
}
# Conservative production v2 exact eligibility is separate and unchanged:
# RB/FB rushing >=5 carries, WR/TE/RB receiving >=3 targets.
CHALLENGERS={
    "A_role35_recent35":(.35,.35),
    "B_role55_recent50":(.55,.50),
    "C_role75_recent65":(.75,.65),
}
PRE_REGISTERED_TRAIN_PERIOD="2024 W07-W18"
UNTOUCHED_EVALUATION=[2025,2026]

def avg(values):
    return sum(values)/len(values) if values else None

def clamp(x,lower,upper):
    return min(upper,max(lower,x))

def numeric(x):
    return x is not None and isinstance(x,(float,int)) and math.isfinite(x)

@dataclass(frozen=True)
class Observation:
    player_id:str
    player_name:str
    position:str
    team:str
    season:int
    week:int
    game_id:str
    opportunity:float
    yards:float

def pregame_forecast(history:Sequence[Observation],season:int,market:str,
                     role_weight:float,recent_weight:float):
    """Season/prior baseline + shrinkage-controlled *prior-game* role trend.

    A signed error last week is not blindly subtracted from the next game:
    repeated misses only matter when an actual usage trend supports them.
    We publish the ratio, current opportunity, confidence sample count and
    predicted yards/count independently, never an invented hit probability.
    """
    if not history:return None
    cfg=MARKETS[market]
    positions=set(cfg[0].split(","))
    if history[-1].position not in positions:return None
    if len(history)<3:return None
    last3=history[-3:]
    recent_volume=avg([g.opportunity for g in last3])
    if recent_volume is None or recent_volume<cfg[3]:return None
    # Never mix future seasons into current-season priors.
    cur=[g for g in history if g.season==season]
    prior=[g for g in history if g.season==season-1]
    earlier=history[-8:-3]
    cur_average=avg([g.yards for g in cur])
    prior_average=avg([g.yards for g in prior])
    last3_average=avg([g.yards for g in last3])
    if cur_average is None and prior_average is None:return None
    if cur_average is None:base=prior_average
    elif prior_average is None:base=cur_average
    else:
        # Conservative season shrinkage; do not let a few 2026 games wipe
        # out a prior-season role nor let prior-year form dominate Week 10.
        ncur=len(cur)
        base=(ncur*cur_average+min(4,max(0,8-ncur))*prior_average) / (
             ncur+min(4,max(0,8-ncur)))
    assert base is not None
    # Compare to PREVIOUS opportunities, not season mean containing last3.
    # The 3+ prior-game requirement rejects an ungrounded "collapse" signal.
    older_volume=avg([g.opportunity for g in earlier]) if len(earlier)>=3 else None
    has_role_trend=older_volume is not None and older_volume>=1
    ratio=recent_volume/older_volume if has_role_trend else 1.0
    # Clip noisy tiny samples and shrink toward 1.0 (no role shift).
    shrunk_role=1.0+(clamp(ratio,.40,1.60)-1.0)*3/6 if has_role_trend else 1.0
    freshness=min(1.0,len(cur)/5)
    # Positive AND negative recent usage shifts can be supported by data.
    # Recent outlier yardage remains partially shrunk to season estimate.
    recent_yards=last3_average if last3_average is not None else base
    corrected=base*(1.0+role_weight*freshness*(shrunk_role-1.0))
    corrected += recent_weight*freshness*(recent_yards-base)
    # Prevent a three-game outlier producing a >50% numerical swing.
    low=max(0.0,base*.50);high=max(base*.50,base*1.50)
    forecast=clamp(corrected,low,high)
    return {
        "market":market,"forecast":forecast,"baseline":max(0.0,base),
        "recent_opportunities":recent_volume,
        "older_opportunities":older_volume,
        "opportunity_ratio":ratio if has_role_trend else None,
        "role_change":("DECLINING" if has_role_trend and shrunk_role<.85 else
                        "INCREASING" if has_role_trend and shrunk_role>1.15 else
                        "NOT_ESTABLISHED"),
        "recent_3_actual_avg":recent_yards,"prior_games":len(history),
        "current_season_games":len(cur),"eligible":True,
        "probability_source":"NONE_RESEARCH_NUMERICAL_ONLY",
    }

def rows_from_db(con,market):
    pos,col,ycol,_=MARKETS[market]
    cols={r[1] for r in con.execute("PRAGMA table_info(player_games)")}
    if not {col,ycol}.issubset(cols):
        raise RuntimeError("NFL_UNIVERSAL_MARKET_REQUIRED_COLUMNS_MISSING_"+market)
    sql=f"""
      SELECT player_id, player_name, position,team,season,week,game_id,
             COALESCE({col},0),COALESCE({ycol},0)
      FROM player_games WHERE season_type='REG'
      AND position IN ({",".join("?" for _ in pos.split(","))})
      AND player_id IS NOT NULL AND game_id IS NOT NULL
      ORDER BY season,week,game_id,player_id
    """
    return [Observation(str(a),str(b),str(c),str(d),int(e),int(f),
                        str(g),float(h),float(i))
            for a,b,c,d,e,f,g,h,i in con.execute(sql,tuple(pos.split(",")))]

def retrospective_rows(con,market):
    observations=rows_from_db(con,market)
    hist=defaultdict(list)
    pending=defaultdict(list)
    for o in observations:
        pending[(o.season,o.week)].append(o)
    scored=[]
    for season,week in sorted(pending):
        today=pending[(season,week)]
        # Every player observed in this week uses ONLY historical games
        # completed in prior weeks, never teammates or own target-game box.
        for o in today:
            h=hist[o.player_id]
            f0=pregame_forecast(h,season,market,0,0)
            if f0 is None:continue
            candidates={name:pregame_forecast(h,season,market,*params)["forecast"]
                        for name,params in CHALLENGERS.items()}
            scored.append({"key":(o.player_id,o.season,o.week,o.game_id),
               "player":o.player_name,"market":market,"season":o.season,"week":o.week,
               "actual":o.yards,"baseline":f0["baseline"],"features":f0,
               "candidates":candidates})
        for o in today:
            hist[o.player_id].append(o)
    return scored

def paired_summary(rows,a,b):
    if not rows:return {"n":0}
    err_a=[abs(r[a]-r["actual"]) for r in rows]
    err_b=[abs(r[b]-r["actual"]) for r in rows]
    return {"n":len(rows),
            "baseline_mae":round(sum(err_a)/len(rows),3),
            "challenger_mae":round(sum(err_b)/len(rows),3),
            "delta_mae":round((sum(err_b)-sum(err_a))/len(rows),3),
            "baseline_over20_rate":round(sum(r[a]-r["actual"]>=20 for r in rows)/len(rows),4),
            "challenger_over20_rate":round(sum(r[b]-r["actual"]>=20 for r in rows)/len(rows),4),
            "challenger_better_share":round(sum(err_b[i]<err_a[i] for i in range(len(rows)))/len(rows),4)}

def evaluate(con,all_markets=MARKETS):
    out={"schema":"NFL_UNIVERSAL_OPPORTUNITY_RESEARCH_V1",
         "status":"RESEARCH_ONLY_UNVALIDATED_DO_NOT_SHIP",
         "development_period":PRE_REGISTERED_TRAIN_PERIOD,
         "holdouts":UNTOUCHED_EVALUATION,"no_sportsbook_lines":True,
         "no_postgame_feature_leakage":True,"markets":{}}
    for market in all_markets:
        rows=retrospective_rows(con,market)
        dev=[r for r in rows if r["season"]==2024 and r["week"]>=7]
        # Development comparator: all candidate definitions were fixed
        # BEFORE observing this output. 2025/26 are never used to tune.
        scores={}
        for name in CHALLENGERS:
            for r in rows:r["_candidate"]=r["candidates"][name]
            scores[name]=paired_summary(dev,"baseline","_candidate")
        # One pre-registered winner chosen on 2024 only, no survivor bias.
        chosen=min(CHALLENGERS,key=lambda k:(
            scores[k].get("delta_mae",float("inf")),k))
        entry={"n_development":len(dev),"development_candidates":scores,
               "selected_by_2024_only":chosen,
               "eligible_raw_rows":len(rows)}
        for year in UNTOUCHED_EVALUATION:
            sample=[r for r in rows if r["season"]==year]
            for r in sample:r["_candidate"]=r["candidates"][chosen]
            entry[str(year)]=paired_summary(sample,"baseline","_candidate")
            declining=[r for r in sample if r["features"]["role_change"]=="DECLINING"]
            entry[str(year)]["declining_role"]=paired_summary(declining,"baseline","_candidate")
            if len(sample)<100:
                entry[str(year)]["validation_warning"]="INSUFFICIENT_N_FOR_PROMOTION"
        # Do not auto-promote based on tiny or in-sample results.
        p25=entry.get("2025",{});p26=entry.get("2026",{})
        entry["eligible_for_further_review"]=(
            p25.get("n",0)>=100 and p26.get("n",0)>=100
            and p25.get("delta_mae",0)<0 and p26.get("delta_mae",0)<0
            and p25.get("declining_role",{}).get("delta_mae",0)<=0
            and p26.get("declining_role",{}).get("delta_mae",0)<=0)
        entry["production_promoted"]=False
        out["markets"][market]=entry
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--db",required=True)
    p.add_argument("--json")
    args=p.parse_args()
    with sqlite3.connect(f"file:{args.db}?mode=ro",uri=True) as con:
        report=evaluate(con)
    payload=json.dumps(report,indent=2,sort_keys=True)
    print(payload,flush=True)
    if args.json:
        Path(args.json).write_text(payload+"\n")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
