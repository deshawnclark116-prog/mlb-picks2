#!/usr/bin/env python3
"""Independent V2 P(reach exact forecast) learned probability head.

V2's XGBoost numerical champion is frozen. On the COMPLETED 2026 games
strictly BEFORE the target week, construct a second binary classifier for:
   real_yards >= whole_0_6(original V2 forecast)
with features frozen at the player's original pregame input + the original
V2 center + ONLY earlier completed V2 forecast errors for that player.

Training and validation discipline:
- 2025 must NOT be used (V2 was refit on it).
- Prequential 2026 Week 7-10 forecasts train on prior weeks only,
  never current week's results, no player-name/race/game label features.
- A NEW probability head is not a pass simply for outputting percentages.
  Require prequential Brier better than 50% baseline AND the market-specific
  empirical CDF on the *same* exact 2026 games.
- All hyperparams and feature names were declared BEFORE testing.
- Source information includes number of real outcomes and certified status:
  it is NOT odds, lineup verification, or a betting success guarantee.
"""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
import xgboost as xgb

from nfl_v2_empirical_probability import (
    EVENT,SOURCE,MIN_MARKET_ERRORS,whole_mean,
    probability_from_prior
)

SOURCE_LEARNED="V2_CONTEXT_2026_ASOF_LEARNED_EXACT_MEAN_PHEAD_V1"
TRAIN_MIN_N=170
VALIDATION_MIN_N=80
MIN_WEEK_FOR_ASOF_TEST=7
FEATURES=[
    "v2_forecast","v2_target","v2_current_season_games",
    "v2_last3_avg_yds","v2_last3_avg_vol","v2_season_avg_yds",
    "v2_season_avg_vol","v2_prior_avg_yds","v2_prior_avg_vol",
    "v2_eff16","v2_team_spread","v2_team_implied","v2_opp_allowed_pg",
    "v2_is_home","v2_last3_share","v2_last3_vs_season_yards",
    "v2_last3_vs_prior_yards","v2_last3_vs_season_vol",
    "prior_v2_miss_count","prior_v2_last3_signed_error_avg",
    "prior_v2_last3_absolute_error_avg","prior_v2_last1_error",
    "prior_v2_over10_count","prior_v2_under10_count",
]
PARAMS={
 "objective":"binary:logistic","eval_metric":"logloss",
 "max_depth":2,"eta":.035,"subsample":.82,"colsample_bytree":.87,
 "min_child_weight":22,"lambda":18.,"alpha":1.0,
 "tree_method":"hist","seed":20261122,"nthread":2
}
N_ROUNDS=55

def normal_number(x):
    if x is None:return float("nan")
    try:
        v=float(x)
        return v if math.isfinite(v) else float("nan")
    except (ValueError,TypeError):return float("nan")

def features(row,prev_misses):
    f=row["v2_features"]
    last3=prev_misses[-3:]
    num=lambda key:normal_number(f.get(key))
    avg=lambda xs:float(np.mean(xs)) if xs else float("nan")
    difference=lambda a,b:a-b if math.isfinite(a) and math.isfinite(b) else float("nan")
    last3yd=num("last3_avg_yds");seasonyd=num("cur_avg_yds")
    prior_yd=num("prior_avg_yds");last3vol=num("last3_avg_vol");seasonvol=num("cur_avg_vol")
    return {
      "v2_forecast":row["prediction"],
      "v2_target":whole_mean(row["prediction"]),
      "v2_current_season_games":num("cur_n"),
      "v2_last3_avg_yds":last3yd,
      "v2_last3_avg_vol":last3vol,
      "v2_season_avg_yds":seasonyd,
      "v2_season_avg_vol":seasonvol,
      "v2_prior_avg_yds":prior_yd,
      "v2_prior_avg_vol":num("prior_avg_vol"),
      "v2_eff16":num("eff_16"),
      "v2_team_spread":num("team_spread"),
      "v2_team_implied":num("implied_team_total"),
      "v2_opp_allowed_pg":num("opp_allowed_pg"),
      "v2_is_home":num("is_home"),
      "v2_last3_share":num("last3_share"),
      "v2_last3_vs_season_yards":difference(last3yd,seasonyd),
      "v2_last3_vs_prior_yards":difference(last3yd,prior_yd),
      "v2_last3_vs_season_vol":difference(last3vol,seasonvol),
      "prior_v2_miss_count":len(prev_misses),
      "prior_v2_last3_signed_error_avg":avg(last3),
      "prior_v2_last3_absolute_error_avg":avg([abs(x) for x in last3]),
      "prior_v2_last1_error":prev_misses[-1] if prev_misses else float("nan"),
      "prior_v2_over10_count":sum(x<=-10 for x in last3),
      "prior_v2_under10_count":sum(x>=10 for x in last3),
    }

def with_asof_features(observations):
    """Strict week-batches avoid same-week outcome -> prior-error feature."""
    by_week=defaultdict(list)
    for x in observations:by_week[(x["season"],x["week"])].append(x)
    history=defaultdict(list)
    records=[]
    for key in sorted(by_week):
        today=by_week[key]
        for r in today:
            enriched=dict(r)
            enriched["phead_features"]=features(r,history[r["player_id"]])
            enriched["binary_outcome"]=int(r["actual"]>=whole_mean(r["prediction"]))
            records.append(enriched)
        for r in today:
            history[r["player_id"]].append(r["residual"])
    return records,history

def design(rows):
    return np.asarray([[r["phead_features"][key] for key in FEATURES] for r in rows],dtype=np.float32)

def fit(rows):
    if len(rows)<TRAIN_MIN_N:return None
    y=[r["binary_outcome"] for r in rows]
    if min(sum(y),len(y)-sum(y))<30:return None
    model=xgb.train(PARAMS,xgb.DMatrix(design(rows),label=y,feature_names=FEATURES),
                   num_boost_round=N_ROUNDS,verbose_eval=False)
    return model

def evaluate_model(observations):
    """2026 walk-forward only, ONE fixed design, same-period baseline pairs."""
    rows,hist=with_asof_features(observations)
    by_week=defaultdict(list)
    for x in rows:by_week[x["week"]].append(x)
    learned,empirical,half,outcomes=[],[],[],[]
    for week in sorted(by_week):
        if week<MIN_WEEK_FOR_ASOF_TEST:continue
        prior=[x for x in rows if x["week"]<week]
        if len(prior)<TRAIN_MIN_N:continue
        booster=fit(prior)
        if booster is None:continue
        candidates=by_week[week]
        yp=np.asarray(booster.predict(xgb.DMatrix(design(candidates),feature_names=FEATURES)),dtype=float)
        for r,yhat in zip(candidates,yp):
            e=probability_from_prior(prior,r["prediction"],whole_mean(r["prediction"]),r["player_id"])
            if e is None:continue
            y=r["binary_outcome"]
            learned.append(float((yhat-y)**2))
            empirical.append(float((e["probability"]-y)**2))
            half.append(float((.5-y)**2))
            outcomes.append({
               "week":week,"pred":float(yhat),"actual":y,
               "empirical":e["probability"],
               "prior_v2_over10_count":r["phead_features"]["prior_v2_over10_count"]
            })
    n=len(learned)
    metrics={
       "version":SOURCE_LEARNED,
       "n_forward":n,"prequential_week_min":MIN_WEEK_FOR_ASOF_TEST,
       "v2_frozen_through_2025":True,"market_line_not_used":True,
       "independent_forward_certification":False,
       "learned_brier":round(float(np.mean(learned)),4) if n else None,
       "empirical_brier":round(float(np.mean(empirical)),4) if n else None,
       "constant50_brier":round(float(np.mean(half)),4) if n else None,
       "repeated_v2_over10_brier":None
    }
    repeat=[o for o in outcomes if o["prior_v2_over10_count"]>=2]
    if repeat:
        metrics["repeated_v2_over10_brier"]=round(
           float(np.mean([(o["pred"]-o["actual"])**2 for o in repeat])),4)
        metrics["n_repeated_v2_over10"]=len(repeat)
    metrics["promotion_eligible"]=(
       n>=VALIDATION_MIN_N and
       metrics["learned_brier"] is not None and
       metrics["learned_brier"]+0.0005 < metrics["empirical_brier"] and
       metrics["learned_brier"] < .25
    )
    return metrics

def predict_one(booster,history,current_player,current_features,residual_n):
    """Serve a previously trained, independently gated probability head.
    Do NOT train the classifier once per current player."""
    if booster is None:return None
    priors=history.get(current_player["player_id"],[])
    item={
      "prediction":float(current_player["prediction"]),
      "v2_features":current_features
    }
    f=features(item,priors)
    row={**item,"phead_features":f}
    prob=float(booster.predict(xgb.DMatrix(design([row]),feature_names=FEATURES))[0])
    target=whole_mean(item["prediction"])
    return {
       "source":SOURCE_LEARNED,"event":EVENT,
       "target":target,"probability":round(prob,4),
       "model_reference_mean":round(item["prediction"],1),
       "residual_sample_n":residual_n,"residual_total_prior_n":residual_n,
       "market_calibration":"NOT_FORWARD_CERTIFIED",
       "official_active_status_verified":False
    }
