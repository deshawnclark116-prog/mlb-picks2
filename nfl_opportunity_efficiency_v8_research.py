#!/usr/bin/env python3
"""NFL V8 role-and-efficiency challengers against actual frozen V2 predictions.

PURE RESEARCH: no production forecast or sportsbook probabilities changed.

Why: V2 is a 16-feature one-stage yardage regressor. We need to establish
whether separately modeling opportunities, changing workload and efficiency
significantly improves forward NFL projections BEFORE replacing V2.

Models (fixed in advance, chosen ONLY by 2024 validation):
  role_direct: Extra role/availability/efficiency features directly -> yards.
  two_stage: E[targets/carries/attempts] * E[yards/opportunity], independently
             modeled, with current + earlier opportunity and efficiency.
  incumbent: exact deployed V2 frozen artifact on the same 2026 player-game
             and strictly pregame information (only rec/rush).
  simple_blend: current-season/prior-season average.

TRAIN: 2023 and 2024 weeks 1-12.
SELECT ROUNDS: 2024 weeks 13-18 (no 2025/2026 feedback).
REFIT: 2023-2025 at FIXED 2024 rounds, compared on <=2026 week10.
  2026 rows were previously consulted, so metrics are a DIAGNOSTIC and
  do NOT authorize promotion without later untouched forward validation.
  2025 rows are scored only using models trained through 2024.
KEYS: same actual player ID, market, season, week; no player-name cherry pick.
Unplayed games and same-week results NEVER feed earlier-game features.
Inputs: nflverse play-by-game player_games/schedule; no invented snap counts,
defensive coverage, confirmed injuries, QB info or sportsbook probability.

A real algorithmic advance must win paired MAE over frozen incumbent by
2% with a positive 95% cluster-bootstrap lower CI, not worsen declining-role
errors or drop meaningful sample coverage. Only report a pass as RESEARCH.

Run: python nfl_opportunity_efficiency_v8_research.py --db /tmp/nfl_model.sqlite
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import json
import math
import sqlite3

import numpy as np
import xgboost as xgb
import nfl_yardage_v2 as v2

MARKETS={
    "receiving_yards":{"vol":"targets","yds":"receiving_yards",
                       "positions":("WR","TE","RB"),"min_last3_vol":3.0,"max_eff":40.},
    "rushing_yards":{"vol":"carries","yds":"rushing_yards",
                       "positions":("RB","FB"),"min_last3_vol":5.0,"max_eff":20.},
    "passing_yards":{"vol":"attempts","yds":"passing_yards",
                       "positions":("QB",),"min_last3_vol":18.0,"max_eff":18.},
}
DEVELOP=(2024,12)
ASOF=(2026,11)
TRAIN_YEARS=(2023,2024,2025)
PARAMS_DIRECT={
    "objective":"reg:absoluteerror","eval_metric":"mae",
    "max_depth":3,"eta":.04,"subsample":.85,"colsample_bytree":.85,
    "min_child_weight":16,"reg_lambda":10.,"tree_method":"hist",
    "seed":20261122,"nthread":2,
}
PARAMS_VOLUME={
    "objective":"reg:squarederror","eval_metric":"mae",
    "max_depth":3,"eta":.04,"subsample":.85,"colsample_bytree":.85,
    "min_child_weight":18,"reg_lambda":15.,"tree_method":"hist",
    "seed":20261122,"nthread":2,
}
PARAMS_EFF={
    "objective":"reg:squarederror","eval_metric":"mae",
    "max_depth":2,"eta":.04,"subsample":.85,"colsample_bytree":.85,
    "min_child_weight":24,"reg_lambda":15.,"tree_method":"hist",
    "seed":20261122,"nthread":2,
}
EXTRA=[
    "position_qb","position_rb","position_wr","position_te",
    "last1_yds","last2_yds","last5_yds","last8_yds",
    "last1_vol","last2_vol","last5_vol","last8_vol",
    "prev5_yds","prev5_vol","last3_to_prev5_vol",
    "last3_to_season_vol","last3_to_prior_vol",
    "last3_to_prev5_yards","last3_to_season_yards",
    "last3_vol_slope","last3_yds_slope",
    "last3_vol_std","last3_yds_std",
    "last3_zero_opportunities","last3_vol_per_game_share",
    "last3_yards_per_opportunity","last8_yards_per_opportunity",
    "last1_yards_per_opportunity",
    "previous5_yards_per_opportunity",
    "last3_change_from_prior_share",
    "same_team_last3","weeks_since_last_appearance",
    "season_games","prior_games",
]
FEATURES=v2.FEATURES+EXTRA

def mean(items):
    return float(np.mean(items)) if items else float("nan")

def valid(x):
    return x is not None and isinstance(x,(float,int)) and math.isfinite(x)

def ratio(n,d):
    return float(n/d) if valid(n) and valid(d) and d>0 else float("nan")

def ypo(h):
    v=sum(x["vol"] for x in h)
    return sum(x["yds"] for x in h)/v if v>0 else float("nan")

def normalized_market_history(con,market):
    """Use exact original V2 replay code for each market, including as-of
    defense state, training eligibility, player identity and game spread.
    Passing extends the same private config in this RESEARCH PROCESS only."""
    if market not in v2.MARKETS:
        v2.MARKETS[market]={k:val for k,val in MARKETS[market].items() if k!="max_eff"}
    history=v2.History(con,market)
    original,_,_=history.replay(until=ASOF)
    return history,original

def derive(ph,base,season,week,pos,team):
    recent=ph[-3:]
    last8=ph[-8:]
    prev5=ph[-8:-3]
    cur=[x for x in ph if x["season"]==season]
    prior=[x for x in ph if x["season"]==season-1]
    vols=[x["vol"] for x in recent]
    yds=[x["yds"] for x in recent]
    older_share=mean([x["vol"]/x["team_vol"] for x in prev5 if x["team_vol"]>0])
    recent_share=mean([x["vol"]/x["team_vol"] for x in recent if x["team_vol"]>0])
    older_vol=mean([x["vol"] for x in prev5]) if len(prev5)>=3 else float("nan")
    older_yd=mean([x["yds"] for x in prev5]) if len(prev5)>=3 else float("nan")
    seq={
       "position_qb":float(pos=="QB"),"position_rb":float(pos in ("RB","FB")),
       "position_wr":float(pos=="WR"),"position_te":float(pos=="TE"),
       "last1_yds":float(ph[-1]["yds"]),
       "last2_yds":mean([x["yds"] for x in ph[-2:]]),
       "last5_yds":mean([x["yds"] for x in ph[-5:]]),
       "last8_yds":mean([x["yds"] for x in last8]),
       "last1_vol":float(ph[-1]["vol"]),
       "last2_vol":mean([x["vol"] for x in ph[-2:]]),
       "last5_vol":mean([x["vol"] for x in ph[-5:]]),
       "last8_vol":mean([x["vol"] for x in last8]),
       "prev5_yds":older_yd,"prev5_vol":older_vol,
       "last3_to_prev5_vol":ratio(base["last3_avg_vol"],older_vol),
       "last3_to_season_vol":ratio(base["last3_avg_vol"],base["cur_avg_vol"]),
       "last3_to_prior_vol":ratio(base["last3_avg_vol"],base["prior_avg_vol"]),
       "last3_to_prev5_yards":ratio(base["last3_avg_yds"],older_yd),
       "last3_to_season_yards":ratio(base["last3_avg_yds"],base["cur_avg_yds"]),
       "last3_vol_slope":float(vols[-1]-vols[0])/2,
       "last3_yds_slope":float(yds[-1]-yds[0])/2,
       "last3_vol_std":float(np.std(vols)),
       "last3_yds_std":float(np.std(yds)),
       "last3_zero_opportunities":float(sum(x==0 for x in vols)),
       "last3_vol_per_game_share":recent_share,
       "last3_yards_per_opportunity":ypo(recent),
       "last8_yards_per_opportunity":ypo(last8),
       "last1_yards_per_opportunity":ypo(ph[-1:]),
       "previous5_yards_per_opportunity":ypo(prev5) if len(prev5)>=3 else float("nan"),
       "last3_change_from_prior_share":recent_share-older_share,
       "same_team_last3":float(sum(x["team"]==team for x in recent)),
       "weeks_since_last_appearance":
          float(week-ph[-1]["week"] if ph[-1]["season"]==season else 10),
       "season_games":float(len(cur)),"prior_games":float(len(prior))
    }
    # Pre-game source features also never include current game's targets,
    # catches, team result or actual efficiency.
    return {**base,**seq}

def build_rows(con,market):
    history,original=normalized_market_history(con,market)
    indexed={(str(m["player_id"]),int(m["season"]),int(m["week"])):(m,f,y)
             for m,f,y in original}
    hist=defaultdict(list)
    output=[]
    by_week=defaultdict(list)
    for row in history.rows:
        if (row[5],row[6])>=ASOF:continue
        by_week[(row[5],row[6])].append(row)
    for (season,week) in sorted(by_week):
        today=by_week[(season,week)]
        # read only prior weeks
        for (pid,name,pos,team,opp,s,w,gid,is_home,vol,yds) in today:
            key=(str(pid),int(season),int(week))
            existing=indexed.get(key)
            if existing is None:continue
            m,f,y=existing
            ph=hist[pid]
            assert len(ph)>=3
            features=derive(ph,f,season,week,pos,team)
            output.append({
                "key":key,"player":name,"player_id":str(pid),
                "team":team,"opp":opp,"market":market,
                "season":season,"week":week,"game_id":str(gid),
                "actual_yards":float(y),"actual_volume":float(vol),
                "actual_eff":float(yds/vol) if vol>0 else float("nan"),
                "features":features,
                "base":v2.naive_preds(f)[1],
                "role_collapse":valid(features["last3_to_prev5_vol"]) and
                                 features["last3_to_prev5_vol"]<.75,
            })
        for (pid,name,pos,team,opp,s,w,gid,is_home,vol,yds) in today:
            hist[pid].append({"season":season,"week":week,
                              "yds":float(yds),"vol":float(vol),
                              "team_vol":float(history.team_vol[(gid,team)]),
                              "team":team})
    assert len(output)==len(original),"canonical V2 replay must match reconstructed pregame keys"
    return output

def matrix(rows):
    return np.array([[r["features"].get(key,float("nan"))
                      if valid(r["features"].get(key)) else np.nan
                      for key in FEATURES] for r in rows],dtype=np.float32)

def old_matrix(rows):
    return np.array([[r["features"].get(key,float("nan"))
                      if valid(r["features"].get(key)) else np.nan
                      for key in v2.FEATURES] for r in rows],dtype=np.float32)

def dmat(rows,label=None,features=FEATURES):
    X=matrix(rows) if features==FEATURES else old_matrix(rows)
    return xgb.DMatrix(X,label=label,feature_names=features)

def fit_model(rows,validation,params,target,nmax=650):
    getter=(lambda r:float(r[target])) if target!="actual_eff" else (
        lambda r:min(MARKETS[r["market"]]["max_eff"],max(-4.,float(r["actual_eff"]))))
    tr=[r for r in rows if target!="actual_eff" or valid(r["actual_eff"])]
    va=[r for r in validation if target!="actual_eff" or valid(r["actual_eff"])]
    if len(tr)<120 or len(va)<35:
        return None
    labels=[getter(r) for r in tr]
    vlabels=[getter(r) for r in va]
    if target=="actual_eff":
        # Model needs E[efficiency], not just a noisy median. In very low
        # volume games the y/opportunity can explode; cap intentionally.
        pass
    booster=xgb.train(params,dmat(tr,labels),num_boost_round=nmax,
                      evals=[(dmat(va,vlabels),"dev_2024")],
                      early_stopping_rounds=45,verbose_eval=False)
    return booster.best_iteration+1

def train_final(rows,params,target,rounds):
    if rounds is None:return None
    tr=[r for r in rows if target!="actual_eff" or valid(r["actual_eff"])]
    getter=(lambda r:float(r[target])) if target!="actual_eff" else (
        lambda r:min(MARKETS[r["market"]]["max_eff"],max(-4.,float(r["actual_eff"]))))
    return xgb.train(params,dmat(tr,[getter(r) for r in tr]),
                     num_boost_round=rounds)

def predict(boost,rows):
    if boost is None:return None
    return np.maximum(0.,boost.predict(dmat(rows)))

def summarize(rows,pred,comparator):
    n=len(rows)
    if not n:return {"n":0}
    y=np.array([x["actual_yards"] for x in rows])
    e=np.abs(np.asarray(pred)-y)
    c=np.abs(np.asarray(comparator)-y)
    delta=c-e
    # CI is bootstrap over weekly game clusters (players sharing matchup
    # are not independent); repeatability anchored to fixed RNG seed.
    buckets=defaultdict(list)
    for idx,r in enumerate(rows):
        buckets[(r["season"],r["week"],r["game_id"])].append(idx)
    groups=list(buckets.values())
    rng=np.random.default_rng(20261122)
    bootstrap=[]
    for j in range(1500):
        draw=rng.integers(0,len(groups),len(groups))
        idx=[i for k in draw for i in groups[int(k)]]
        bootstrap.append(float(delta[idx].mean()))
    ci=np.quantile(bootstrap,[.025,.975])
    return {
       "n":n,"n_game_clusters":len(groups),
       "challenger_mae":round(float(e.mean()),3),
       "incumbent_mae":round(float(c.mean()),3),
       "benefit_positive_yards":round(float(delta.mean()),3),
       "benefit_pct":round(float(delta.mean()/c.mean()*100),2),
       "paired_ci95_benefit_yards":[round(float(q),3) for q in ci],
       "game_clusters_not_independent_players":True,
       "challenger_mean_error":round(float((np.asarray(pred)-y).mean()),3),
       "incumbent_mean_error":round(float((np.asarray(comparator)-y).mean()),3),
    }

def market_report(con,market):
    rows=build_rows(con,market)
    development=[r for r in rows if r["season"]==2023 or
                 (r["season"]==2024 and r["week"]<=12)]
    validation=[r for r in rows if r["season"]==2024 and r["week"]>12]
    years={y:[r for r in rows if r["season"]==y] for y in (2025,2026)}
    if not development or not validation or len(years[2026])<70:
        return {"status":"INSUFFICIENT_SOURCE_COVERAGE","n":{str(y):len(z) for y,z in years.items()}}
    rounds={
      "role_direct":fit_model(development,validation,PARAMS_DIRECT,"actual_yards"),
      "opportunity_volume":fit_model(development,validation,PARAMS_VOLUME,"actual_volume"),
      "conditional_efficiency":fit_model(development,validation,PARAMS_EFF,"actual_eff"),
    }
    devfull=development+validation
    out={"rounds_chosen_2024_only":rounds,
         "deployed_frozen_v2_fitted_through_2025":market in ("receiving_yards","rushing_yards") and
           (v2.MODEL_DIR/f"nfl_{market}_v2.json").exists(),
         "market":market,"status":"RESEARCH_ONLY",
         "2026_ends_before_week":ASOF[1]}
    for year in (2025,2026):
        training=devfull if year==2025 else [r for r in rows if r["season"] in TRAIN_YEARS]
        sample=years[year]
        fitted={
         "role_direct":train_final(training,PARAMS_DIRECT,"actual_yards",rounds["role_direct"]),
         "opportunity_volume":train_final(training,PARAMS_VOLUME,"actual_volume",rounds["opportunity_volume"]),
         "conditional_efficiency":train_final(training,PARAMS_EFF,"actual_eff",rounds["conditional_efficiency"]),
        }
        rd=predict(fitted["role_direct"],sample)
        vo=predict(fitted["opportunity_volume"],sample)
        eff=predict(fitted["conditional_efficiency"],sample)
        if rd is None or vo is None or eff is None:
            out[str(year)]={"status":"CHALLENGER_MISSING_TRAIN_COVERAGE"}
            continue
        # Two-stage makes a predicted expected number of opportunities
        # multiplied by predicted efficiency. No current-game volume leakage.
        two=vo*eff
        base=np.array([r["base"] for r in sample],dtype=float)
        if market in ("receiving_yards","rushing_yards"):
            if year==2026:
                champion=xgb.Booster()
                champion.load_model(str(v2.MODEL_DIR/f"nfl_{market}_v2.json"))
                incumbent=np.maximum(0.,champion.predict(dmat(sample,features=v2.FEATURES)))
            else:
                incumbent_model=xgb.train(v2.PARAMS,
                    dmat(devfull,[r["actual_yards"] for r in devfull],features=v2.FEATURES),
                    num_boost_round=json.loads((v2.MODEL_DIR/"report.json").read_text())
                      ["markets"][market]["rounds"])
                incumbent=np.maximum(0.,incumbent_model.predict(dmat(sample,features=v2.FEATURES)))
        else:
            incumbent=base
        assessment={}
        for key,p in [("role_direct",rd),("two_stage",two),("simple_blend",base)]:
            assessment[key]=summarize(sample,p,incumbent)
            crows=[r for r in sample if r["role_collapse"]]
            indices=[i for i,r in enumerate(sample) if r["role_collapse"]]
            assessment[key]["declining_role"]=summarize(crows,np.asarray(p)[indices],
                                                         incumbent[indices])
            if key!="simple_blend":
                assessment[key]["promotion_review_eligible"]=(
                   year==2026 and len(sample)>=100 and
                   assessment[key]["benefit_pct"]>=2. and
                   assessment[key]["paired_ci95_benefit_yards"][0]>0 and
                   (assessment[key]["declining_role"]["n"]<20 or
                    assessment[key]["declining_role"]["benefit_positive_yards"]>=0)
                )
        out[str(year)]=assessment
    out["production_promoted"]=False
    out["untouched_future_holdout_required_before_promotion"]=True
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--db",required=True)
    ap.add_argument("--out")
    args=ap.parse_args()
    con=sqlite3.connect(f"file:{args.db}?mode=ro",uri=True)
    document={
      "schema":"NFL_ROLE_OPPORTUNITY_EFFICIENCY_V8_SHADOW_V1",
      "status":"RESEARCH_ONLY_NO_PRODUCTION_CHANGE",
      "asof_source_cutoff":"2026_W11_OPEN",
      "markets":{}
    }
    try:
        for market in MARKETS:
            print("TESTING",market,flush=True)
            document["markets"][market]=market_report(con,market)
            print("MARKET_RESULT",market,json.dumps(document["markets"][market]),flush=True)
    finally:
        con.close()
    s=json.dumps(document,indent=2,allow_nan=False)
    print("FINAL_SHADOW",s,flush=True)
    if args.out:Path(args.out).write_text(s+"\n")
    return 0

if __name__=="__main__":raise SystemExit(main())
