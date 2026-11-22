#!/usr/bin/env python3
"""NFL V8 PASSING YARDS: forward-only, self-grading shadow/paper engine.

No live betting picks, no fabricated hit percentage, no frontend changes.
The 2026 W01-W10 matched diagnostic indicated a meaningful learned passing
yardage improvement over a naive blend, but these 2026 games have already
been consulted during research. The ONLY honest path to promotion is to
freeze genuinely NEW pre-kickoff model forecasts, then grade them after
results become available. This ledger makes that automatic.

The trained model uses 2023-2025 only and FIXED 92 trees selected on
2024 weeks13-18, then predicts the next scheduled 2026 game from strictly
prior-game history. Every record is stamped first-seen and immutable.
No historical postgame snapshot is manufactured as an original prediction.
No opponent/player name weights; no sportsbook line required.

Use:
 python nfl_v8_passing_paper.py --db nfl_models/nfl_model.sqlite \
  --ledger docs/nfl_v8_passing_first_seen.jsonl \
  --out docs/nfl_v8_passing_shadow.json \
  --grade-out docs/nfl_v8_passing_forward_grade.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import xgboost as xgb
import nfl_yardage_v2 as v2
from nfl_opportunity_efficiency_v8_research import (
  MARKETS,PARAMS_DIRECT,FEATURES,ASOF,derive,build_rows,
  dmat,train_final,mean
)

SOURCE="NFL_V8_PASSING_ROLE_DIRECT_2024_SELECTED_2025_FIT_SHADOW"
SCHEMA="NFL_V8_PASSING_FIRST_SEEN_PREGAME_V1"
FIXED_DEV_ROUNDS=92
UTC=timezone.utc
def timestamp(x):
    try:
        t=datetime.fromisoformat(str(x).replace("Z","+00:00"))
        return t.astimezone(UTC) if t.tzinfo is not None else None
    except (ValueError,TypeError,OverflowError):
        return None

def now_iso(now):
    return now.astimezone(UTC).isoformat().replace("+00:00","Z")

def trained_model(con):
    rows=build_rows(con,"passing_yards")
    past=[r for r in rows if 2023<=r["season"]<=2025]
    if len(past)<500:
        raise RuntimeError("V8_PASSING_TRAINING_HISTORY_MISSING")
    model=train_final(past,PARAMS_DIRECT,"actual_yards",FIXED_DEV_ROUNDS)
    if model is None:
        raise RuntimeError("V8_PASSING_MODEL_DID_NOT_FIT")
    return model,len(past)

def current_predictions(con,season,week,now,model,trained_n):
    if season!=2026 or not (1<=week<=22):
        raise RuntimeError("V8_UNREVIEWED_SEASON_ASOF")
    # Future revision must validate a newly frozen model, not quietly
    # change the fitted years or the hard-coded number of trees.
    v2.MARKETS.setdefault("passing_yards",{
        "vol":"attempts","yds":"passing_yards","positions":("QB",),
        "min_last3_vol":18.
    })
    hist_obj=v2.History(con,"passing_yards")
    _,hist,defense=hist_obj.replay(until=(season,week))
    upcoming=con.execute("""
      SELECT game_id,home_team,away_team,kickoff_utc
      FROM games WHERE season=? AND week=? AND season_type='REG'
    """,(season,week)).fetchall()
    upcoming_by_team={}
    for gid,home,away,kickoff in upcoming:
        dt=timestamp(kickoff)
        if dt is None or now>=dt:continue
        upcoming_by_team[str(home)]=(str(gid),str(away),True,kickoff)
        upcoming_by_team[str(away)]=(str(gid),str(home),False,kickoff)
    features,profiles=[],[]
    for pid,ph in hist.items():
        if len(ph)<3 or ph[-1]["pos"]!="QB":continue
        team=ph[-1]["team"]
        scheduled=upcoming_by_team.get(team)
        if scheduled is None:continue
        prior_vol=mean([g["vol"] for g in ph[-3:]])
        if not math.isfinite(prior_vol) or prior_vol<18.0:continue
        gid,opp,is_home,kickoff=scheduled
        f=hist_obj.features(ph,season,week,team,opp,is_home,defense)
        f=derive(ph,f,season,week,"QB",team)
        features.append({
            "features":f,"market":"passing_yards",
            "season":season,"week":week,
            "game_id":gid,"player_id":str(pid),
            "player":ph[-1]["name"],"team":team,
            "opponent":opp,"kickoff_utc":kickoff,
            "prior_games":len(ph),"recent_attempts":round(prior_vol,2),
            "baseline_blend":float(f["blend_yds"])
        })
    if not features:return []
    pred=model.predict(dmat(features))
    output=[]
    for item,p in zip(features,pred):
        mean_yd=max(0.,float(p))
        if not math.isfinite(mean_yd):continue
        output.append({
          "source":SOURCE,
          "schema":SCHEMA,
          "season":season,"week":week,
          "market":"passing_yards",
          "game_id":item["game_id"],
          "kickoff_utc":item["kickoff_utc"],
          "player_id":item["player_id"],
          "player":item["player"],
          "team":item["team"],
          "opponent":item["opponent"],
          "prediction_yards":round(mean_yd,2),
          "baseline_blend_yards":round(item["baseline_blend"],2),
          "recent_prior_attempts_pg":item["recent_attempts"],
          "prior_player_games":item["prior_games"],
          "trained_game_rows":trained_n,
          "number_of_trees":FIXED_DEV_ROUNDS,
          "starter_verified":False,
          "hit_probability":None,
          "status":"RESEARCH_ONLY_NO_OFFICIAL_NFL_PICKS"
        })
    return output

def load_ledger(path):
    rows=[]
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():continue
            try:row=json.loads(line)
            except ValueError:continue
            if isinstance(row,dict) and row.get("schema")==SCHEMA:rows.append(row)
    return rows

def first_seen_pregame(existing,current,now):
    """Immutable by player and scheduled game, with strict pregame timestamps."""
    keys={(int(x["season"]),int(x["week"]),str(x["game_id"]),
           str(x["player_id"])) for x in existing}
    new=[]
    for item in current:
        kickoff=timestamp(item["kickoff_utc"])
        if kickoff is None or now>=kickoff:continue
        key=(item["season"],item["week"],item["game_id"],item["player_id"])
        if key in keys:continue
        stamped={**item,"original_first_seen_at_utc":now_iso(now)}
        new.append(stamped)
        keys.add(key)
    return new

def grade(ledger,con,now):
    outcomes=[]
    # Strictly require official completed game score AND player boxscore.
    # Otherwise never assume a DNP counted zero passing yards.
    for r in ledger:
        kickoff=timestamp(r.get("kickoff_utc"))
        source=timestamp(r.get("original_first_seen_at_utc"))
        if kickoff is None or source is None or not source<kickoff<now:
            continue
        final=con.execute("""
           SELECT home_score,away_score FROM games
           WHERE game_id=? AND season=? AND week=?
        """,(r["game_id"],r["season"],r["week"])).fetchone()
        if final is None or final[0] is None or final[1] is None:continue
        actual=con.execute("""
            SELECT passing_yards,attempts FROM player_games
            WHERE player_id=? AND season=? AND week=? AND season_type='REG'
        """,(r["player_id"],r["season"],r["week"])).fetchall()
        if len(actual)!=1 or actual[0][0] is None:continue
        y=float(actual[0][0])
        if not math.isfinite(y):continue
        baseline=r["baseline_blend_yards"]
        forecast=r["prediction_yards"]
        outcomes.append({
          "season":r["season"],"week":r["week"],"game_id":r["game_id"],
          "player_id":r["player_id"],"player":r["player"],
          "prediction_yards":forecast,
          "original_first_seen_at_utc":r["original_first_seen_at_utc"],
          "actual_passing_yards":y,
          "absolute_error":round(abs(forecast-y),3),
          "simple_average_error":round(abs(baseline-y),3),
          "yards_better_than_simple_average":round(abs(baseline-y)-abs(forecast-y),3),
          "starter_verified":False,
          "graded_from_completed_official_score":True
        })
    out={
        "schema":"NFL_V8_PASSING_FORWARD_GRADE_V1",
        "status":"PAPER_ONLY_NOT_APPROVED_BETTING",
        "graded_at_utc":now_iso(now),
        "firstseen_total":len(ledger),"graded_n":len(outcomes),
        "grade_only_completed_games_with_original_pretip_receipt":True,
        "records":outcomes
    }
    if outcomes:
        out["mae_model"]=round(sum(x["absolute_error"] for x in outcomes)/len(outcomes),3)
        out["mae_simple_average"]=round(sum(x["simple_average_error"] for x in outcomes)/len(outcomes),3)
        out["mean_benefit_yards"]=round(sum(x["yards_better_than_simple_average"] for x in outcomes)/len(outcomes),3)
    return out

def execute(con,season,week,now,ledger_path):
    now=now.astimezone(UTC)
    existing=load_ledger(ledger_path)
    model,trained_n=trained_model(con)
    current=current_predictions(con,season,week,now,model,trained_n)
    additions=first_seen_pregame(existing,current,now)
    ledger=existing+additions
    pending=[r for r in current if r.get("source")==SOURCE]
    return {
       "schema":"NFL_V8_PASSING_PREGAME_SHADOW_V1",
       "generated_at_utc":now_iso(now),
       "season":season,"week":week,
       "source":SOURCE,
       "status":"RESEARCH_ONLY_NOT_MODEL_BOARD",
       "new_pretip_firstseen":len(additions),
       "total_first_seen":len(ledger),
       "projections":pending,
       "no_calibrated_hit_percentage":True,
       "original_frozen_forecast_preserved":True,
       "trained_through_year":2025,
       "training_selected_2024_only":True,
    }, ledger, grade(ledger,con,now)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--db",required=True)
    parser.add_argument("--forecast",default="docs/nfl_predictions.json")
    parser.add_argument("--ledger",default="docs/nfl_v8_passing_first_seen.jsonl")
    parser.add_argument("--out",default="docs/nfl_v8_passing_shadow.json")
    parser.add_argument("--grade-out",default="docs/nfl_v8_passing_forward_grade.json")
    options=parser.parse_args()
    board=json.loads(Path(options.forecast).read_text())
    season=board.get("season");week=board.get("week")
    if not (isinstance(season,int) and isinstance(week,int)):
        raise RuntimeError("FROZEN_FORECAST_SEASON_WEEK_REQUIRED")
    db=sqlite3.connect(f"file:{options.db}?mode=ro",uri=True)
    try:
        now=datetime.now(UTC)
        doc,ledger,report=execute(db,season,week,now,Path(options.ledger))
    finally:
        db.close()
    out=Path(options.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(doc,indent=2,allow_nan=False)+"\n")
    g=Path(options.grade_out)
    g.parent.mkdir(parents=True,exist_ok=True)
    g.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    ledger_path=Path(options.ledger)
    ledger_path.parent.mkdir(parents=True,exist_ok=True)
    ledger_path.write_text("".join(json.dumps(x,sort_keys=True,allow_nan=False)+"\n" for x in ledger))
    print("V8_PASSING_SHADOW_PUBLISHED",json.dumps({
        "now":doc["generated_at_utc"],"season":season,"week":week,
        "models":len(doc["projections"]),
        "new_first_seen":doc["new_pretip_firstseen"],
        "grading":report["graded_n"],
        "no_betting_probability":True
    }),flush=True)
    return 0

if __name__=="__main__":raise SystemExit(main())
