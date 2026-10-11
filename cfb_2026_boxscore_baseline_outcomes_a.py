#!/usr/bin/env python3
"""Grade FIRST frozen pregame Week6 last3 box-score benchmark after FINAL games.

Never regenerate today's forecasts for grading. Enforce as-of input artifacts,
the original successful GitHub Actions execution, source checksum and player
identity. No model fit, probability claims, historical backfill or promotion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime,timezone
from pathlib import Path

import cfb_2026_shadow_outcomes_a as integrity

SCHEMA="CFB_2026_LAST3_POINT_BENCHMARK_OUTCOMES_V1"
INPUT="CFB_2026_BOX_SCORE_LAST3_PROSPECTIVE_BASELINE_V1"
MARKETS={
    "rushing_yards":("rushing_yards",),
    "passing_touchdowns":("passing_touchdowns",),
    "anytime_touchdowns":("rushing_touchdowns","receiving_touchdowns"),
}
FINAL={"STATUS_FINAL","STATUS_FINAL_OVERTIME"}


class BaselineGradeError(ValueError):
    pass


def utc(value):
    return integrity.utc(value)


def evaluate(paper, db, source_run, *, original_sha=None):
    if paper.get("schema")!=INPUT or paper.get("status")!="FROZEN_PREGAME_NAIVE_POINT_BASELINE_ONLY":
        raise BaselineGradeError("WRONG_ORIGIN_BASELINE_SCHEMA")
    if paper.get("historical_forecasts_backfilled")!=0 or paper.get("production_model_modified") is not False:
        raise BaselineGradeError("MODIFIED_PRODUCTION_OR_BACKFILLED_PREDS")
    if not isinstance(source_run,dict) or source_run.get("conclusion")!="success" or source_run.get("status")!="completed":
        raise BaselineGradeError("UNVERIFIED_GITHUB_ORIGINAL_RUN")
    if source_run.get("name")!="CFB 2026 event data qualification (research, no serving)":
        raise BaselineGradeError("NOT_ORIGINAL_RESEARCH_WORKFLOW")
    if not isinstance(source_run.get("head_sha"),str) or len(source_run["head_sha"])!=40:
        raise BaselineGradeError("MISSING_ORIGIN_COMMIT")
    made=utc(paper.get("generated_at_utc"))
    started=utc(source_run.get("run_started_at"))
    created=utc(source_run.get("created_at"))
    updated=utc(source_run.get("updated_at"))
    if not created<=started<=made<=updated:
        raise BaselineGradeError("RUN_VERSUS_ARTIFACT_TIMELINE_MISMATCH")
    rows=paper.get("forecasts")
    if not isinstance(rows,list) or paper.get("baseline_rows")!=len(rows):
        raise BaselineGradeError("ORIGINAL_BASELINE_ROW_COUNT_MISMATCH")
    games={str(x["game_id"]):x for x in db.execute(
        "SELECT game_id,season,week,kickoff_utc,home_team,away_team,home_points,away_points "
        "FROM games WHERE season=2026")}
    status={str(gid):s for gid,s in db.execute(
        "SELECT game_id,espn_status FROM schedule_snapshot WHERE season=2026")}
    seen=set(); graded=[]; ungraded=Counter()
    for pred in rows:
        market=pred.get("market")
        if market not in MARKETS:
            raise BaselineGradeError("UNKNOWN_PLAYER_STAT_MARKET")
        gid,pid=str(pred.get("game_id")),str(pred.get("player_id"))
        key=(gid,pid,market)
        if key in seen:
            raise BaselineGradeError("DUPLICATE_ORIGINAL_PLAYER_STAT_FORECAST")
        seen.add(key)
        kickoff=utc(pred.get("kickoff_utc"))
        if (utc(pred.get("research_generated_at_utc"))!=made or
            utc(pred.get("board_generated_at_utc"))>made or
            started>=kickoff or created>=kickoff or made>=kickoff):
            raise BaselineGradeError("NO_GENUINE_ORIGINAL_PREGAME_PROOF")
        vals=pred.get("last3_values")
        ids=pred.get("last3_game_ids")
        if not isinstance(vals,list) or len(vals)!=3 or len(ids or [])!=3 or len(set(ids))!=3:
            raise BaselineGradeError("MISSING_THREE_ORIGINAL_BASELINE_GAMES")
        if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in vals):
            raise BaselineGradeError("NON_FINITE_ORIGINAL_BASELINE_STATS")
        mu=float(pred.get("point_mean_last3"))
        if not math.isfinite(mu) or abs(mu-sum(vals)/3)>0.00006:
            raise BaselineGradeError("MEAN_CHANGED_AFTER_PREGAME")
        med=float(pred.get("point_median_last3"))
        if not math.isfinite(med) or med!=sorted(vals)[1]:
            raise BaselineGradeError("MEDIAN_CHANGED_AFTER_PREGAME")
        game=games.get(gid)
        if game is None:
            ungraded["UNKNOWN_CURRENT_GAME"]+=1
            continue
        if (pred.get("team") not in (game["home_team"],game["away_team"]) or
            game["week"]!=pred.get("week") or
            abs((utc(game["kickoff_utc"])-kickoff).total_seconds())>120):
            raise BaselineGradeError("ORIGINAL_GAME_ID_TEAM_KICKOFF_MISMATCH")
        # Any revision to a previously observed source game must be detected,
        # not silently applied retroactively to a saved pregame baseline.
        for hist_id, hist_value in zip(ids,vals):
            histgame=games.get(str(hist_id))
            if histgame is None or histgame["week"]>=game["week"] or status.get(str(hist_id)) not in FINAL:
                raise BaselineGradeError("ORIGINAL_PRIOR_GAME_NOT_VERIFIED")
            historical=db.execute(
                "SELECT team,position,rushing_yards,passing_touchdowns,"
                "rushing_touchdowns,receiving_touchdowns FROM player_games "
                "WHERE player_id=? AND game_id=? AND season=2026",(pid,str(hist_id))).fetchall()
            if len(historical)!=1 or historical[0]["team"]!=pred["team"] or historical[0]["position"]!=pred["position"]:
                raise BaselineGradeError("ORIGINAL_PRIOR_PLAYER_TEAM_AMBIGUOUS")
            fields=MARKETS[market]
            if any(historical[0][f] is None for f in fields):
                raise BaselineGradeError("ORIGINAL_PRIOR_PLAYER_STAT_NOW_MISSING")
            total=sum(float(historical[0][f]) for f in fields)
            if abs(total-float(hist_value))>1e-6:
                raise BaselineGradeError("ORIGINAL_SOURCE_REVISION_CHANGED_BASELINE")
        if status.get(gid) not in FINAL or game["home_points"] is None or game["away_points"] is None:
            ungraded["GAME_NOT_FINAL"]+=1
            continue
        found=db.execute(
            "SELECT team,position,rushing_yards,passing_touchdowns,"
            "rushing_touchdowns,receiving_touchdowns FROM player_games "
            "WHERE player_id=? AND game_id=? AND season=2026",(pid,gid)).fetchall()
        if len(found)>1:
            raise BaselineGradeError("AMBIGUOUS_FINISHED_PLAYER_ID")
        if not found:
            ungraded["NO_OFFICIAL_PLAYER_LINE"]+=1
            continue
        row=found[0]
        if row["team"]!=pred["team"] or row["position"]!=pred["position"]:
            raise BaselineGradeError("FINISHED_PLAYER_TEAM_POSITION_CHANGED")
        fields=MARKETS[market]
        if any(row[f] is None for f in fields):
            ungraded["UNKNOWN_ACTUAL_STAT_NOT_ZERO"]+=1
            continue
        actual=sum(float(row[f]) for f in fields)
        graded.append({
            "game_id":gid,"player_id":pid,"market":market,"team":pred["team"],
            "actual":actual,"original_mean":mu,"original_median":med,
            "abs_error_mean":abs(mu-actual),"abs_error_median":abs(med-actual),
            "error_mean":mu-actual,
        })
    by=defaultdict(list)
    for x in graded:
        by[x["market"]].append(x)
    def metrics(xs):
        n=len(xs)
        if not n:
            return {"n":0}
        return {"n":n,
                "mean_baseline_MAE":round(sum(x["abs_error_mean"] for x in xs)/n,4),
                "median_baseline_MAE":round(sum(x["abs_error_median"] for x in xs)/n,4),
                "mean_baseline_bias":round(sum(x["error_mean"] for x in xs)/n,4),
                "note":"Observational selected player stats; no model edge or profits inferred."}
    return {
        "schema":SCHEMA,
        "source_run_id":source_run.get("id"),
        "source_sha":source_run["head_sha"],
        "original_source_artifact_sha256":original_sha,
        "original_prediction_generated_at":paper["generated_at_utc"],
        "n_original_pregame_baselines":len(rows),
        "graded_n":len(graded),"not_yet_graded_n":len(rows)-len(graded),
        "ungraded_reasons":dict(ungraded),"overall":metrics(graded),
        "by_market":{m:metrics(v) for m,v in sorted(by.items())},
        "graded_rows":graded,"no_model_promotions_or_past_predictions_rewritten":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original-paper",type=Path,required=True)
    p.add_argument("--original-actions-metadata",type=Path,required=True)
    p.add_argument("--results-db",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    data=a.original_paper.read_bytes()
    with sqlite3.connect(f"file:{a.results_db}?mode=ro",uri=True) as conn:
        conn.row_factory=sqlite3.Row
        r=evaluate(json.loads(data),conn,json.loads(a.original_actions_metadata.read_text()),
                   original_sha=hashlib.sha256(data).hexdigest())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"original_n":r["n_original_pregame_baselines"],
                     "graded_n":r["graded_n"],"by_market":r["by_market"]},sort_keys=True))


if __name__=="__main__":
    main()
