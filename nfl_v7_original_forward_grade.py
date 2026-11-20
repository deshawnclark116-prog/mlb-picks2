#!/usr/bin/env python3
"""Honest, original-only NFL V7/V6B numeric grades; research, never promotion.

All candidate players fixed before kickoff. The same game's eventual box score
is used ONLY for grading. A missing player stat row is UNRESOLVED, not a DNP
or zero. Pair V6B by exact game/team/GSIS player/market, without matching
on actual performance. Source capture times can differ; do not declare a
fair deployment victory on these exploratory pairs.
"""
from __future__ import annotations
import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import nfl_v6_joint_opportunity_shadow as v6

MARKETS={"receiving_yards","rushing_yards","passing_yards"}
SCHEMA="NFL_V7_ROSTER_RETURNER_ORIGINAL_NUMERIC_AUDIT_V1"


def validate(receipt):
    if receipt.get("schema")!="NFL_V7_PREGAME_ROSTER_RETURNER_JOINT_SHADOW_A":
        raise ValueError("NOT_FROZEN_V7_FIRSTSEEN")
    if receipt.get("status")!="RESEARCH_ONLY_NOT_PROMOTED":
        raise ValueError("V7_SOURCE_STATUS_INVALID")
    if v6.utc(receipt["capture_observed_at_utc"]) >= v6.utc(receipt["kickoff_utc"]):
        raise ValueError("V7_FORECAST_NOT_PRETIP")
    if v6.utc(receipt["source_cutoff_utc"]) > v6.utc(receipt["capture_observed_at_utc"]):
        raise ValueError("V7_HISTORY_AFTER_SOURCE_RECEIPT")
    if receipt.get("official_bets")!=0 or not receipt.get("conditional_player_predictions_not_calibrated"):
        raise ValueError("V7_MISREPRESENTS_CONDITIONAL_PROBABILITY")
    g=receipt["model"]
    if g["game_id"]!=receipt["game_id"] or g["kickoff_utc"]!=receipt["kickoff_utc"]:
        raise ValueError("V7_GAME_IDENTITY_MISMATCH")
    seen=set()
    for row in g["projections"]:
        if row["market"] not in MARKETS:
            continue
        key=(row["team"],str(row["player_id"]),row["market"])
        if key in seen:
            raise ValueError("V7_DUPLICATE_PLAYER_MARKET")
        seen.add(key)
        if row["team"] not in (g["home"],g["away"]) or \
           row["opponent"] not in (g["home"],g["away"]) or \
           row["team"]==row["opponent"]:
            raise ValueError("V7_WRONG_GAME_TEAM")
        if row["simulations"]<10000 or not math.isfinite(float(row["projected_median"])):
            raise ValueError("V7_INVALID_DISTRIBUTION")
    return g


def grade(v7_dir,v6_dir,db,*,asof=None):
    now=v6.utc(asof) if asof else datetime.now(timezone.utc)
    report={"schema":SCHEMA,"graded_at_utc":now.isoformat(),
        "model_promoted":False,"eligible_for_production_promotion":False,
        "source_time_leads_not_identical":True,
        "no_missing_player_row_assumed_zero":True,
        "games":[],"excluded":{},"pairs":[],"published_incumbent_pairs":[]}
    excluded=Counter()
    con=sqlite3.connect("file:"+str(Path(db).absolute())+"?mode=ro",uri=True)
    try:
        originals=sorted(Path(v7_dir).glob("2026_*.json"))
        if not originals:
            raise ValueError("NO_ORIGINAL_V7_FORWARD_FILES")
        for file in originals:
            v7=json.loads(file.read_text())
            model=validate(v7)
            gid=v7["game_id"]
            if gid!=file.stem:
                raise ValueError("V7_RECEIPT_FILENAME_MISMATCH")
            if v6.utc(v7["kickoff_utc"]) >= now:
                excluded["not_started_yet"] += len(model["projections"])
                continue
            game=con.execute("SELECT home_score,away_score FROM games WHERE game_id=?",
                             (gid,)).fetchall()
            if len(game)!=1 or None in game[0]:
                excluded["no_independent_final_game_result"] += len(model["projections"])
                continue
            truth=defaultdict(list)
            for row in con.execute("""SELECT player_id,team,receiving_yards,
                                     rushing_yards,passing_yards
                                     FROM player_games WHERE game_id=? AND
                                     season_type='REG'""",(gid,)):
                truth[(row[1],str(row[0]))].append(dict(zip(
                    ("id","team","receiving_yards","rushing_yards","passing_yards"),row)))
            p6=Path(v6_dir)/(gid+".json")
            previous={}
            if p6.exists():
                v6b=json.loads(p6.read_text())
                if (v6b.get("game_id")!=gid or
                    v6.utc(v6b["source_snapshot_utc"])>=v6.utc(v7["kickoff_utc"])):
                    raise ValueError("INVALID_OR_LATE_V6B_COMPARATOR")
                for row in v6b["projections"]:
                    key=(row["team"],str(row["player_id"]),row["market"])
                    if key in previous:
                        raise ValueError("DUPLICATE_V6B_COMPARATOR")
                    previous[key]=row
            else:
                excluded["no_pre_game_v6b_comparator"]+=1
            # The published V2 production baseline was frozen with the
            # actual original challenger, never recomputed after results.
            live_receipt=v7.get("frozen_live_incumbent") or {}
            live={}
            if live_receipt.get("status")=="ORIGINAL_SAME_GAME_PUBLISHED_PRODUCTION":
                if (live_receipt.get("game_id")!=gid or
                    live_receipt.get("kickoff_utc")!=v7["kickoff_utc"] or
                    v6.utc(live_receipt["generated_at_utc"])>
                    v6.utc(live_receipt["observed_at_utc"]) or
                    v6.utc(live_receipt["observed_at_utc"])>
                    v6.utc(v7["capture_observed_at_utc"])):
                    raise ValueError("INVALID_OR_LATE_LIVE_INCUMBENT_RECEIPT")
                for record in live_receipt["rows"]:
                    key=(record["team"],str(record["player_id"]),record["market"])
                    if key in live:
                        raise ValueError("DUPLICATE_ORIGINAL_INCUMBENT_COMPARATOR")
                    live[key]=record
            else:
                excluded["no_valid_same_game_published_incumbent"]+=1
            case={"game_id":gid,"kickoff_utc":v7["kickoff_utc"],
                "v7_source_observed_utc":v7["capture_observed_at_utc"],
                "v6b_source_observed_utc":(v6b["source_snapshot_utc"]
                    if p6.exists() else None),"original_count":0,
                "graded":0,"unresolved":0,"paired":0,"paired_live_incumbent":0,
                "live_incumbent_source_status":live_receipt.get("status",
                    "NO_FROZEN_INCUMBENT_RECEIPT"),
                "returner_graded":0}
            for row in model["projections"]:
                if row["market"] not in MARKETS:
                    continue
                case["original_count"]+=1
                key=(row["team"],str(row["player_id"]),row["market"])
                tr=truth.get(key[:2],[])
                if len(tr)!=1 or tr[0].get(row["market"]) is None:
                    case["unresolved"]+=1
                    excluded["no_unique_verified_player_stat_row"]+=1
                    continue
                actual=float(tr[0][row["market"]])
                prediction=float(row["projected_median"])
                if not math.isfinite(actual):
                    case["unresolved"]+=1
                    excluded["nonfinite_actual"]+=1
                    continue
                case["graded"]+=1
                ret=bool(row.get("returner_added_to_player_universe"))
                case["returner_graded"]+=int(ret)
                prior=previous.get(key)
                paired=prior is not None
                case["paired"]+=int(paired)
                live_row=live.get(key)
                live_paired=live_row is not None
                case["paired_live_incumbent"]+=int(live_paired)
                record={"game_id":gid,"team":row["team"],"player_id":key[1],
                    "player":row["player"],"market":row["market"],
                    "actual":actual,"v7_median":prediction,
                    "v7_absolute_error":round(abs(actual-prediction),4),
                    "v7_within_10":abs(actual-prediction)<=10,
                    "v7_within_20":abs(actual-prediction)<=20,
                    "returner":ret,"v6b_paired":paired,
                    "published_incumbent_paired":live_paired,
                    "published_incumbent_median":(
                        float(live_row["projected_median"]) if live_paired else None),
                    "published_incumbent_model_source":(
                        live_row["model_source"] if live_paired else None),
                    "published_incumbent_absolute_error":(
                        round(abs(actual-float(live_row["projected_median"])),4)
                        if live_paired else None),
                    "v6b_median":(float(prior["projected_median"]) if paired else None),
                    "v6b_absolute_error":(round(abs(actual-float(prior["projected_median"])),4)
                                           if paired else None)}
                if live_paired:
                    report["published_incumbent_pairs"].append(record)
                else:
                    excluded["no_same_player_published_production_original"]+=1
                if paired:
                    report["pairs"].append(record)
                else:
                    excluded["new_v7_population_no_v6b_same_player"]+=1
                # Include every resolved original, not only paired winners.
                report.setdefault("all_original_grades",[]).append(record)
            report["games"].append(case)
    finally:
        con.close()
    def summary(rows):
        if not rows:
            return {"n":0,"v7_mae":None,"v6b_mae":None,"v7_within_10":None,
                    "v6b_within_10":None}
        return {"n":len(rows),
            "v7_mae":round(sum(x["v7_absolute_error"] for x in rows)/len(rows),4),
            "v6b_mae":round(sum(x["v6b_absolute_error"] for x in rows)/len(rows),4),
            "v7_within_10":round(sum(x["v7_within_10"] for x in rows)/len(rows),4),
            "v6b_within_10":round(sum(x["v6b_absolute_error"]<=10 for x in rows)/len(rows),4)}
    report["paired_summary"]=summary(report["pairs"])
    report["paired_by_market"]={
        m:summary([x for x in report["pairs"] if x["market"]==m])
        for m in sorted(MARKETS)}
    def live_summary(rows):
        if not rows:
            return {"n":0,"v7_mae":None,"published_incumbent_mae":None,
                    "v7_within_10":None,"published_incumbent_within_10":None}
        n=len(rows)
        return {"n":n,
                "v7_mae":round(sum(x["v7_absolute_error"] for x in rows)/n,4),
                "published_incumbent_mae":round(sum(
                    x["published_incumbent_absolute_error"] for x in rows)/n,4),
                "v7_within_10":round(sum(x["v7_within_10"] for x in rows)/n,4),
                "published_incumbent_within_10":round(sum(
                    x["published_incumbent_absolute_error"]<=10 for x in rows)/n,4)}
    report["published_incumbent_paired_summary"]=live_summary(
        report["published_incumbent_pairs"])
    report["published_incumbent_paired_by_market"]={
        m:live_summary([x for x in report["published_incumbent_pairs"]
                        if x["market"]==m]) for m in sorted(MARKETS)}
    allrows=report.get("all_original_grades",[])
    report["v7_all_original_graded"]=len(allrows)
    report["v7_returner_only_graded"]=sum(x["returner"] for x in allrows)
    report["v7_all_original_median_mae"]=(
        round(sum(x["v7_absolute_error"] for x in allrows)/len(allrows),4)
        if allrows else None)
    report["excluded"]=dict(excluded)
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--v7-dir",default="docs/nfl_v7_firstseen")
    ap.add_argument("--v6-dir",default="docs/nfl_v6b_firstseen")
    ap.add_argument("--db",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    result=grade(args.v7_dir,args.v6_dir,args.db)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n")
    print("NFL_V7_ALL_ORIGINAL_FORWARD_GRADE",json.dumps({
        "paired":result["paired_summary"],
        "graded":result["v7_all_original_graded"],
        "returners":result["v7_returner_only_graded"],
        "excluded":result["excluded"]}))
