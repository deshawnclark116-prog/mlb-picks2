#!/usr/bin/env python3
"""Independent NFL numerical model competition: NOT a V7 patch.

Compare 3 independently defined predictive architectures:
  (1) frozen empirical recent-five baseline;
  (2) direct gradient boosted expected yardage;
  (3) factorized: gradient boosted opportunities * Bayesian shrunk
      yards/opportunity with prior-opponent context.

2024 train, 2025 validation selects architecture, 2024+25 final train,
2026 descriptive test (2026 was already consulted by prior research).
Every feature is calculated STRICTLY from completed previous NFL weeks;
same-week games are withheld until the whole week has been predicted.
Player selection is retrospective PLAYED population: cannot claim a
lineup-aware pregame or unconditional betting win. Do not auto-promote.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sqlite3
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

MARKETS = {
    "rushing_yards": ("carries", 4.20, 135.0),
    "receiving_yards": ("targets", 7.80, 215.0),
}
FEATURES = (
    "played_prior", "last_game_gap", "opp_3", "opp_5", "opp_8",
    "yards_3", "yards_5", "yards_8", "yards_sd5",
    "team_opportunity_avg5", "team_share_5",
    "opponent_def_allowed_avg5", "home", "week",
    "is_rb", "is_wr", "is_te", "is_qb",
)
SEED = 20261120
VERSION = "NFL_INDEPENDENT_OPPORTUNITY_HEAD_ARCHITECTURE_BATTLE_A"
PARAMS = dict(max_iter=90,max_leaf_nodes=12,min_samples_leaf=35,
              learning_rate=.045,l2_regularization=9.0,random_state=SEED)


def previous_only_features(*,history,team_games,opponent_games,week,season,
                           is_home,position,market):
    """Pure, measurable as-of transform; never receives today's actual."""
    op,rate,allowed_norm=MARKETS[market]
    recent=list(history)
    if len(recent)<3 or sum(max(0,float(g[op])) for g in recent[-5:])<=0:
        return None
    hs=recent[-8:]
    ms=lambda n,k:mean(float(g[k]) for g in hs[-n:])
    opp_3,opp_5,opp_8=(ms(n,op) for n in (3,5,8))
    y3,y5,y8=(ms(n,market) for n in (3,5,8))
    y5_samples=[float(g[market]) for g in hs[-5:]]
    ysd=float(np.std(y5_samples))
    team_before=list(team_games)[-5:]
    defense_before=list(opponent_games)[-5:]
    tavg=mean(float(g["opportunities"]) for g in team_before) if team_before else 0.0
    davg=mean(float(g["yards_allowed"]) for g in defense_before) if defense_before else allowed_norm
    gap=(season-int(hs[-1]["season"]))*20 + (week-int(hs[-1]["week"]))
    if gap<=0:
        raise ValueError("LEAKAGE_SAME_OR_FUTURE_WEEK_USED")
    share=min(1.0,max(0.,opp_5/max(1.,tavg)))
    feat=(len(history),min(40,gap),opp_3,opp_5,opp_8,y3,y5,y8,ysd,
          tavg,share,davg,float(is_home or 0),int(week),
          *[float(position==p) for p in ("RB","WR","TE","QB")])
    if len(feat)!=len(FEATURES) or any(not math.isfinite(v) for v in feat):
        raise ValueError("INVALID_PREGAME_FEATURE")
    total_op=sum(max(0,float(g[op])) for g in hs)
    total_y=sum(float(g[market]) for g in hs)
    # A prior from league-wide rates, NOT the player's future actual.
    eff=(max(-.5*total_op,total_y)+20*rate)/(total_op+20)
    eff=max(0.5,eff)
    opponent_factor=min(1.20,max(.80,davg/allowed_norm))
    return {"features":feat,"prior_opps":opp_5,
            "baseline":max(0.,y5),
            "prior_efficiency":eff*opponent_factor,
            "eligible_high_volume":opp_5 >= (5 if market=="rushing_yards" else 4),
            "opponent_coverage":len(defense_before),
            "team_coverage":len(team_before),
            "source_history_rows":len(history)}


def build_pregame_dataset(conn,*,seasons=(2024,2025,2026)):
    """Predict all players as of the start of each week before updating state."""
    sql="""SELECT p.player_id,p.player_name,p.team,p.opponent,
                   p.position,p.season,p.week,p.game_id,p.is_home,
                   p.carries,p.targets,p.rushing_yards,p.receiving_yards
             FROM player_games p JOIN games g ON g.game_id=p.game_id
             WHERE p.season_type='REG' AND g.season_type='REG'
                   AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
                   AND p.season BETWEEN ? AND ?
             ORDER BY p.season,p.week,p.game_id,p.player_id"""
    cur=conn.execute(sql,(min(seasons),max(seasons)))
    by_week=defaultdict(list)
    for row in cur:
        d=dict(zip([q[0] for q in cur.description],row))
        if d["season"] in seasons and d["week"]<=18:
            by_week[(d["season"],d["week"])].append(d)
    player=defaultdict(lambda:deque(maxlen=16))
    team_last=defaultdict(lambda:deque(maxlen=6))
    defense_last=defaultdict(lambda:deque(maxlen=6))
    output=[]
    for season_week in sorted(by_week):
        season,week=season_week
        week_rows=by_week[season_week]
        # Strict batch barrier: no predictions in this week can see any game
        # outcomes, snaps, roster participation, or score from the SAME week.
        for d in week_rows:
            if not d["player_id"] or not d["team"] or not d["opponent"]:
                continue
            for market,(opp,_,_) in MARKETS.items():
                if d[opp] is None or d[market] is None:
                    continue
                raw=previous_only_features(
                    history=player[d["player_id"]],
                    team_games=team_last[(d["team"],market)],
                    opponent_games=defense_last[(d["opponent"],market)],
                    week=week,season=season,is_home=d["is_home"],
                    position=d["position"],market=market)
                if raw is None:
                    continue
                output.append({"season":season,"week":week,
                    "game_id":d["game_id"],"player_id":d["player_id"],
                    "player":d["player_name"],"team":d["team"],
                    "opponent":d["opponent"],"position":d["position"],
                    "market":market,"actual":float(d[market]),
                    "actual_opportunities":float(d[opp]),
                    **raw})
        # Update only AFTER producing all predictions from this week.
        agg=defaultdict(lambda:{"opportunities":0,"yards_allowed":0})
        for d in week_rows:
            if d["player_id"]:
                hist={"season":season,"week":week}
                for k in ("carries","targets","rushing_yards","receiving_yards"):
                    hist[k]=float(d[k] or 0)
                player[d["player_id"]].append(hist)
            if d["game_id"] is None:
                continue
            for market,(op,_,_) in MARKETS.items():
                if d[op] is None or d[market] is None:
                    continue
                agg[(d["game_id"],d["team"],market)]["opportunities"]+=max(0,float(d[op]))
                agg[(d["game_id"],d["opponent"],market)]["yards_allowed"]+=max(0,float(d[market]))
        # Only one aggregate record per team/game, never 1 per player.
        for (gid,team,market),value in sorted(agg.items()):
            if "opportunities" in value:
                team_last[(team,market)].append({"opportunities":value["opportunities"]})
            if "yards_allowed" in value:
                defense_last[(team,market)].append({"yards_allowed":value["yards_allowed"]})
    return output


def _estimate(train,validate,market):
    tr=[r for r in train if r["market"]==market]
    te=[r for r in validate if r["market"]==market]
    if len(tr)<250 or len(te)<80:
        raise ValueError("NOT_ENOUGH_CHRONOLOGICALLY_PREPARED_ROWS_"+market)
    X=np.asarray([r["features"] for r in tr],float)
    Y=np.asarray([r["actual"] for r in tr],float)
    V=np.asarray([r["features"] for r in te],float)
    op=np.asarray([r["actual_opportunities"] for r in tr],float)
    direct=HistGradientBoostingRegressor(**PARAMS).fit(X,Y)
    count=HistGradientBoostingRegressor(**{**PARAMS,"loss":"poisson"}).fit(X,op)
    return {r["player_id"]+"|"+r["game_id"]+"|"+market:{
        "baseline":r["baseline"],
        "direct":max(0.,float(pred)),
        "factorized":max(0.,float(npred)*r["prior_efficiency"])
      }for r,pred,npred in zip(te,direct.predict(V),count.predict(V))}


def _make_grades(rows,pred,arm):
    g=[]
    for r in rows:
        k=r["player_id"]+"|"+r["game_id"]+"|"+r["market"]
        if k not in pred:
            raise ValueError("MISSING_ORIGINAL_SAME_ROW_FORECAST")
        g.append({**r,"predicted":float(pred[k][arm]),
                  "abs_error":abs(float(pred[k][arm])-r["actual"])})
    return g


def _stats(rows):
    if not rows:return {"n":0,"mae":None,"within10":None}
    return {"n":len(rows),"mae":round(mean(r["abs_error"] for r in rows),4),
            "within10":round(mean(r["abs_error"]<=10 for r in rows),4),
            "bias":round(mean(r["predicted"]-r["actual"] for r in rows),4)}


def _paired_gain_ci(rows,challenger,baseline,*,seed=SEED,iters=1200):
    pairs=defaultdict(list)
    for row in rows:
        k=row["player_id"]+"|"+row["game_id"]+"|"+row["market"]
        pairs[row["game_id"]].append(
            abs(row["actual"]-baseline[k])-abs(row["actual"]-challenger[k]))
    if len(pairs)<12:return None
    rng=random.Random(seed)
    groups=list(pairs.values())
    vals=[]
    for _ in range(iters):
        sample=[groups[rng.randrange(len(groups))] for _ in groups]
        flat=[q for g in sample for q in g]
        vals.append(mean(flat))
    vals.sort()
    return [round(vals[int(.025*(iters-1))],4),round(vals[int(.975*(iters-1))],4)]


def run(db):
    con=sqlite3.connect("file:"+str(Path(db).resolve())+"?mode=ro",uri=True)
    con.row_factory=sqlite3.Row
    try:
        rows=build_pregame_dataset(con)
    finally:
        con.close()
    year_counts={str(year):len([r for r in rows if r["season"]==year])
                 for year in (2024,2025,2026)}
    report={"schema":VERSION,"input_db":str(db),"source_population":
        "POSTGAME_OBSERVED_PLAYER_ROWS_ONLY_NOT_AUTHENTIC_INACTIVE_OR_ROSTER",
        "source_date_policy":"STRICT_PRIOR_COMPLETED_NFL_WEEKS",
        "model_train":"2024_ONLY","architecture_select_validation":"2025_ONLY",
        "final_train":"2024_AND_2025",
        "descriptive_2026":"CONSULTED_RESEARCH_PERIOD_NOT_PRISTINE_HOLDOUT",
        "production_model_changed":False,"official_bets":0,
        "player_participation_prediction":False,
        "same_roster_source_not_verified":True,
        "eligible_to_promote":False,"counts":year_counts,"markets":{}}
    for market in MARKETS:
        tr=[r for r in rows if r["season"]==2024 and r["market"]==market]
        val=[r for r in rows if r["season"]==2025 and r["market"]==market]
        test=[r for r in rows if r["season"]==2026 and r["market"]==market]
        validation_predictions=_estimate(tr,val,market)
        val_scores={arm:_stats(_make_grades(val,validation_predictions,arm))
                    for arm in ("baseline","direct","factorized")}
        # Selection on 2025 VALIDATION ONLY; tie -> simpler baseline.
        winner=min(("baseline","direct","factorized"),
                   key=lambda a:(val_scores[a]["mae"],
                     ("baseline","direct","factorized").index(a)))
        future_preds=_estimate(tr+val,test,market)
        test_scores={arm:_stats(_make_grades(test,future_preds,arm))
                     for arm in ("baseline","direct","factorized")}
        rows_p=[r for r in test if r["market"]==market]
        eval_map={arm:{r["player_id"]+"|"+r["game_id"]+"|"+market:
                         float(future_preds[r["player_id"]+"|"+r["game_id"]+"|"+market][arm])
                         for r in rows_p}for arm in ("baseline","direct","factorized")}
        high=[r for r in test if r["eligible_high_volume"]]
        report["markets"][market]={
            "2024_train_rows":len(tr),"2025_validated_rows":len(val),
            "2026_descriptive_rows":len(test),
            "chosen_on_2025_only":winner,"2025_validation_mae":val_scores,
            "2026_descriptive_mae":test_scores,
            "2026_high_prior_volume":{arm:_stats(_make_grades(high,future_preds,arm))
                for arm in ("baseline","direct","factorized")},
            "2026_paired_gain_vs_recent5_ci_by_game":{
                arm:_paired_gain_ci(rows_p,eval_map[arm],eval_map["baseline"])
                for arm in ("direct","factorized")},
            "2026_games":len({r["game_id"] for r in rows_p}),
            "no_lineup_unconditional_promotion":True,
            "model_promoted":False}
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--db",default="nfl_models/nfl_model.sqlite")
    p.add_argument("--out",required=True)
    a=p.parse_args()
    report=run(a.db)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("NFL_INDEPENDENT_OPPORTUNITY_ARCHITECTURES_REAL_NUMERIC",json.dumps({
        "seasons":report["counts"],"markets":report["markets"],
        "promotion":False}))
