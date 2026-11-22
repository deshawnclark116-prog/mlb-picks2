#!/usr/bin/env python3
"""Independent NFL market-wide role change + repeated forecast-error ML challenger.

NO production writes. Train: 2023 + 2024 Weeks 1-12. Early-stop: 2024
Weeks 13-18. Untouched chronological evaluation: full 2025 and completed
2026. No reading source/game results until after each prediction snapshot.

Markets: rushing, receiving, passing yards, plus carry, reception and pass
completion counts. Learn whether reduced opportunity and repeated past signed
forecast errors imply lower NEXT-game production rather than naively subtract
last week's shortfall. Include drift, efficiency, role, as-of forecast errors.

The incumbent V2 receiving/rushing XGBoost models were refit through 2025
and so CANNOT be validly graded as holdout on 2025. Compare their original
frozen artifact to this independent model on 2026 exclusively, exact same
player/market/week rows. Grade current 2025 without treating the incumbent
as independent evidence. Missing prior data -> no pick, not forced zero.

A model with good 2025/26 historical metrics is only a candidate for forward
paper-trading, not automatically eligible for live 2026 Week11 promotion.
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

from nfl_universal_opportunity_research import (
    MARKETS,avg,clamp,rows_from_db,pregame_forecast,
)

FEATURES=(
    "baseline","last1_yards","last2_yards","last3_yards","last5_yards",
    "last8_yards","current_season_yards","prior_season_yards",
    "last1_vol","last2_vol","last3_vol","last5_vol","last8_vol",
    "previous5_vol","current_season_vol","prior_season_vol",
    "recent_vs_previous5_ratio","last3_eff","last8_eff",
    "season_eff","prior_eff","current_n","prior_n","all_n",
    "week_of_season","last_seen_gap","team_switch",
    "last3_zero_opportunity","recent_2_overprediction_share",
    "recent_3_overprediction_share","recent_3_signed_error_mean",
    "recent_3_absolute_error_mean","recent_3_error_n",
)
PARAMS={
    "objective":"reg:absoluteerror","eval_metric":"mae",
    "max_depth":3,"eta":.035,"subsample":.8,"colsample_bytree":.85,
    "min_child_weight":18,"reg_lambda":7.0,
    "tree_method":"hist","seed":20261122,
}
TRAIN_PERIOD="2023 + 2024 W01-W12"
VAL_PERIOD="2024 W13-W18"
TEST_PERIODS=("2025","2026")

def n_avg(seq):
    a=[v for v in seq if isinstance(v,(float,int)) and math.isfinite(v)]
    return sum(a)/len(a) if a else float("nan")

def n_eff(hist):
    v=sum(x.opportunity for x in hist)
    return sum(x.yards for x in hist)/v if v>0 else float("nan")

def model_features(history, errors,season,week,team,market):
    """Strictly PRIOR game features, the same for TRAIN and SERVE."""
    if not history:return None
    f0=pregame_forecast(history,season,market,0,0)
    if f0 is None:return None
    last3=history[-3:];last8=history[-8:];prev5=history[-8:-3]
    cur=[x for x in history if x.season==season]
    prior=[x for x in history if x.season==season-1]
    err3=errors[-3:]
    return {
        "baseline":f0["baseline"],
        "last1_yards":history[-1].yards,
        "last2_yards":n_avg([x.yards for x in history[-2:]]),
        "last3_yards":n_avg([x.yards for x in last3]),
        "last5_yards":n_avg([x.yards for x in history[-5:]]),
        "last8_yards":n_avg([x.yards for x in last8]),
        "current_season_yards":n_avg([x.yards for x in cur]),
        "prior_season_yards":n_avg([x.yards for x in prior]),
        "last1_vol":history[-1].opportunity,
        "last2_vol":n_avg([x.opportunity for x in history[-2:]]),
        "last3_vol":n_avg([x.opportunity for x in last3]),
        "last5_vol":n_avg([x.opportunity for x in history[-5:]]),
        "last8_vol":n_avg([x.opportunity for x in last8]),
        "previous5_vol":n_avg([x.opportunity for x in prev5]) if len(prev5)>=3 else float("nan"),
        "current_season_vol":n_avg([x.opportunity for x in cur]),
        "prior_season_vol":n_avg([x.opportunity for x in prior]),
        "recent_vs_previous5_ratio":(
           clamp(f0["opportunity_ratio"],0.1,4.0) if f0["opportunity_ratio"] is not None
           else float("nan")),
        "last3_eff":n_eff(last3),"last8_eff":n_eff(last8),
        "season_eff":n_eff(cur),"prior_eff":n_eff(prior),
        "current_n":len(cur),"prior_n":len(prior),"all_n":len(history),
        "week_of_season":week,
        "last_seen_gap":week-history[-1].week if history[-1].season==season else 8,
        "team_switch":1.0 if history[-1].team!=team else 0.0,
        "last3_zero_opportunity":sum(x.opportunity==0 for x in last3),
        "recent_2_overprediction_share":n_avg([int(e<0) for e in errors[-2:]]),
        "recent_3_overprediction_share":n_avg([int(e<0) for e in err3]),
        "recent_3_signed_error_mean":n_avg(err3),
        "recent_3_absolute_error_mean":n_avg([abs(e) for e in err3]),
        "recent_3_error_n":len(err3),
    }

def make_observations(con,market):
    raw=rows_from_db(con,market)
    by_week=defaultdict(list)
    for o in raw:by_week[(o.season,o.week)].append(o)
    history=defaultdict(list)
    errors=defaultdict(list)
    out=[]
    for (season,week) in sorted(by_week):
        pending=[]
        for row in by_week[(season,week)]:
            f=model_features(history[row.player_id],errors[row.player_id],
                             season,week,row.team,market)
            if f is None:continue
            record={"player_id":row.player_id,"player":row.player_name,
                    "team":row.team,"season":season,"week":week,
                    "game_id":row.game_id,"market":market,
                    "actual":row.yards,"baseline":f["baseline"],"features":f}
            pending.append(record)
            out.append(record)
        # Target is not permitted to influence ANY pregame feature in
        # the same week, even if same player somehow appears twice.
        for row in pending:
            errors[row["player_id"]].append(row["actual"]-row["baseline"])
        for row in by_week[(season,week)]:
            history[row.player_id].append(row)
    return out

def matrix(rows):
    return np.asarray([[row["features"][k] for k in FEATURES] for row in rows],
                      dtype=np.float32)

def metric(rows,predictions):
    if not rows:return {"n":0}
    assert len(rows)==len(predictions)
    n=len(rows)
    truth=np.array([r["actual"] for r in rows])
    baseline=np.array([r["baseline"] for r in rows])
    out=np.maximum(0.0,np.asarray(predictions,dtype=float))
    if not np.isfinite(out).all():
        raise RuntimeError("NFL_ROLE_MODEL_NONFINITE_OUTPUT")
    return {
       "n":n,
       "baseline_mae":round(float(np.abs(baseline-truth).mean()),3),
       "challenger_mae":round(float(np.abs(out-truth).mean()),3),
       "delta_mae":round(float((np.abs(out-truth)-np.abs(baseline-truth)).mean()),3),
       "baseline_bias":round(float((baseline-truth).mean()),3),
       "challenger_bias":round(float((out-truth).mean()),3),
       "baseline_overpredict_by_20_rate":round(float(((baseline-truth)>=20).mean()),4),
       "challenger_overpredict_by_20_rate":round(float(((out-truth)>=20).mean()),4),
       "challenger_better_fraction":round(float((np.abs(out-truth)<np.abs(baseline-truth)).mean()),4)
    }

def do_market(con,market):
    rows=make_observations(con,market)
    train=[r for r in rows if r["season"]==2023 or (r["season"]==2024 and r["week"]<=12)]
    val=[r for r in rows if r["season"]==2024 and r["week"]>12]
    if len(train)<100 or len(val)<30:
        return {"status":"INSUFFICIENT_DEVELOPMENT_ROWS","train":len(train),"validation":len(val)}
    b=xgb.train(PARAMS,xgb.DMatrix(matrix(train),label=[r["actual"] for r in train],
                                  feature_names=list(FEATURES)),
                num_boost_round=650,evals=[(xgb.DMatrix(matrix(val),
                label=[r["actual"] for r in val],feature_names=list(FEATURES)),"val")],
                early_stopping_rounds=55,verbose_eval=False)
    best=b.best_iteration+1
    def predict(rows):
        if not rows:return np.zeros(0)
        return b.predict(xgb.DMatrix(matrix(rows),feature_names=list(FEATURES)),
                         iteration_range=(0,best))
    rep={"model":"independent_pre_2025_xgb_role_error_v1",
         "model_market":market,"train_n":len(train),"val_n":len(val),
         "best_rounds":best,
         "selection_only_2024_validation_mae":metric(val,predict(val)),
         "2025_model_never_trained_on_holdout":True,
         "2026_model_never_trained_on_holdout":True,
         "promotion_status":"RESEARCH_ONLY_NEEDS_FORWARD_ASOF_LINEUP_AND_MATCHED_PROOF"}
    for y in (2025,2026):
        subset=[r for r in rows if r["season"]==y]
        pred=predict(subset)
        summary=metric(subset,pred)
        declining=[(r,pr) for r,pr in zip(subset,pred) if
                   r["features"]["recent_vs_previous5_ratio"]<.75]
        summary["declining_opportunity"]=metric([r for r,_ in declining],
                                                [p for _,p in declining])
        previously_overpredicted=[(r,pr) for r,pr in zip(subset,pred) if
                                  r["features"]["recent_3_overprediction_share"]>=.67
                                  and r["features"]["recent_3_error_n"]>=3]
        summary["repeated_prior_misses"]=metric([r for r,_ in previously_overpredicted],
                                               [p for _,p in previously_overpredicted])
        rep[str(y)]=summary
    # Critical unbiased incumbent compare: frozen original XGB model is
    # refit using ALL 2025 -> only 2026 can be genuinely forward relative
    # to that saved fit. Compare exact same (player ID, season, week) and
    # observed yardage (no name matching, no hindsight roster selection).
    if market in ("rushing_yards","receiving_yards"):
        try:
            import nfl_yardage_v2 as v2
            saved=v2.MODEL_DIR / f"nfl_{market}_v2.json"
            if saved.exists():
                earlier,_,_=v2.History(con,market).replay()
                old=[r for r in earlier if r[0]["season"]==2026]
                oldmat=v2.to_matrix(old)
                model=xgb.Booster();model.load_model(str(saved))
                saved_pred=model.predict(xgb.DMatrix(oldmat,feature_names=v2.FEATURES))
                pm={(str(r[0]["player_id"]),r[0]["season"],r[0]["week"]):float(pr)
                    for r,pr in zip(old,saved_pred)}
                comparable=[(r,p) for r,p in zip(
                   [r for r in rows if r["season"]==2026],
                   predict([r for r in rows if r["season"]==2026]))
                   if (r["player_id"],r["season"],r["week"]) in pm]
                challenger_err=[abs(p-r["actual"]) for r,p in comparable]
                incumbent_err=[abs(pm[(r["player_id"],r["season"],r["week"])]-r["actual"])
                               for r,_ in comparable]
                rep["2026_saved_incumbent_v2_comparison"]={
                    "n":len(comparable),
                    "incumbent_mae":round(sum(incumbent_err)/len(comparable),3) if comparable else None,
                    "challenger_mae":round(sum(challenger_err)/len(comparable),3) if comparable else None,
                    "challenger_delta_mae":round((sum(challenger_err)-sum(incumbent_err))/len(comparable),3) if comparable else None,
                    "incumbent_trained_through_2025":True,
                    "historical_schedule_market_timestamp_unverified":True,
                }
            else:
                rep["2026_saved_incumbent_v2_comparison"]={"status":"INCUMBENT_ARTIFACT_MISSING"}
        except Exception as err:
            rep["2026_saved_incumbent_v2_comparison"]={"status":"SOURCE_COMPARISON_ERROR",
                                                       "detail":str(err)}
    rep["production_promoted"]=False
    return rep

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--db",required=True)
    p.add_argument("--out")
    a=p.parse_args()
    con=sqlite3.connect(f"file:{a.db}?mode=ro",uri=True)
    try:
        doc={"schema":"NFL_UNIVERSAL_LEARNED_ROLE_V1",
             "status":"RESEARCH_ONLY_NO_PRODUCTION_CHANGE",
             "train_period":TRAIN_PERIOD,"validation_period":VAL_PERIOD,
             "held_out":TEST_PERIODS,"markets":{}}
        for market in MARKETS:
            print("EVALUATING",market,flush=True)
            doc["markets"][market]=do_market(con,market)
    finally:
        con.close()
    def json_scalar(value):
        if isinstance(value,np.generic):
            return value.item()
        raise TypeError("NON_JSON_MODEL_METRIC_"+str(type(value)))
    payload=json.dumps(doc,indent=2,default=json_scalar,allow_nan=False)
    print(payload,flush=True)
    if a.out:Path(a.out).write_text(payload+"\n")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
