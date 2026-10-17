#!/usr/bin/env python3
"""Diagnose and expose genuine, *conditional* QB forecasts after starter news.

Important: This is NOT a corrected immutable Phase1E forecast or a calibrated
full-game estimate.  The original P(active) mixture includes zeros in inactive
draws: E[Y] = P(active) * E[Y | active].  Recover that conditional component
without fabricating new simulated game outcomes.  The frozen record remains
unchanged and available for honest postgame grading.

Sources must affirm the exact game/team starting QB, predate forecast cutoff,
and be published before kickoff.  Missing/late/conflicting data fails closed.
"""
from __future__ import annotations
import argparse
import json
import math
from datetime import datetime,timezone
from pathlib import Path

SCHEMA="NFL_V2_SOURCE_CONFIRMED_QB_ROLE_CONDITIONAL_V1"
OUTCOMES=("pass_yds","pass_td","int","rush_yds")
MIN_CANDIDATE_P=0.0001

def utc(value):
    if not isinstance(value,str) or not value:
        raise ValueError("TIMESTAMP_MISSING")
    t=datetime.fromisoformat(value.replace("Z","+00:00"))
    if t.tzinfo is None:
        raise ValueError("TIMESTAMP_NAIVE")
    return t.astimezone(timezone.utc)


def normalized(value):
    return " ".join(str(value or "").casefold().strip().split())


def _rows(board,game,horizon):
    fields=board["row_fields"]
    expected=("game_id","player","team","opp","pos","outcome","p_active",
              "exp_opp","mean","median","id")
    if not set(expected).issubset(fields):
        raise ValueError("SOURCE_BOARD_SCHEMA_INCOMPLETE")
    if len(set(fields))!=len(fields):
        raise ValueError("DUPLICATED_BOARD_COLUMN_NAMES")
    by={name:fields.index(name) for name in expected}
    return [{k:r[i] for k,i in by.items()} for r in board["forecasts"].get(horizon,[])
            if r[by["game_id"]]==game]


def _sources(assertion,cutoff,now):
    sources=assertion.get("official_sources")
    if not isinstance(sources,list) or not sources:
        raise ValueError("NO_PRIMARY_SOURCE")
    supports=set()
    for source in sources:
        url=source.get("url","")
        if not url.startswith("https://www.philadelphiaeagles.com/news/"):
            raise ValueError("NOT_SUPPORTED_OFFICIAL_EAGLES_SOURCE")
        published=utc(source["published_at_utc"])
        if published>now or published>cutoff:
            raise ValueError("SOURCE_NOT_PUBLIC_BEFORE_MODEL_CUTOFF")
        supports.update(source.get("supports") or [])
    if f'{assertion["verified_starter"]} will start' not in supports:
        raise ValueError("SOURCE_DOES_NOT_VERIFY_STARTER")
    for player in assertion["out_players"]:
        if not any(player.lower() in s.lower() and
                   ("out" in s.lower() or "ruled out" in s.lower())
                   for s in supports):
            raise ValueError("SOURCE_DOES_NOT_VERIFY_OUT_PLAYER")
    return sources


def build(board,evidence,*,now):
    if now.tzinfo is None:
        raise ValueError("NAIVE_ASOF_TIME")
    now=now.astimezone(timezone.utc)
    if evidence.get("schema")!="nfl-confirmed-starting-quarterback-source-evidence-v1":
        raise ValueError("WRONG_EVIDENCE_SCHEMA")
    if board.get("schema")!="nfl-new-engine-shadow-v1":
        raise ValueError("WRONG_FROZEN_V2_SOURCE")
    if board.get("status",{}).get("publication_state")!="PUBLISHED":
        raise ValueError("NO_PUBLISHED_ORIGINAL_MODEL")
    gs={g["game_id"]:g for g in board.get("games",[])}
    scenarios=[]
    for assertion in evidence.get("assertions",[]):
        game_id=assertion["game_id"]
        if game_id not in gs:
            raise ValueError("SOURCE_GAME_NOT_IN_V2_BOARD")
        game=gs[game_id]
        if {assertion["team"],assertion["opponent"]}!={game["home"],game["away"]}:
            raise ValueError("SOURCE_GAME_TEAMS_MISMATCH")
        kickoff=utc(game["kickoff_utc"])
        if utc(assertion["kickoff_utc"])!=kickoff:
            raise ValueError("SOURCE_KICKOFF_MISMATCH")
        if now>=kickoff:
            raise ValueError("GAME_ALREADY_STARTED_NO_NEW_PREGAME_SCENARIO")
        for horizon in ("T24","T90"):
            meta=game.get("horizons",{}).get(horizon)
            if meta is None:
                continue
            cutoff=utc(meta["cutoff"])
            sources=_sources(assertion,cutoff,now)
            by=_rows(board,game_id,horizon)
            qbs=[x for x in by if x["team"]==assertion["team"] and
                 x["pos"]=="QB" and x["outcome"]=="pass_yds"]
            chosen=[x for x in qbs if normalized(x["player"])==
                    normalized(assertion["verified_starter"])]
            if len(chosen)!=1:
                raise ValueError("STARTER_ABSENT_OR_AMBIGUOUS_IN_ORIGINAL_QB_FORECAST")
            current=chosen[0]
            p=current["p_active"]
            if not isinstance(p,(float,int)) or not MIN_CANDIDATE_P<=p<=1:
                raise ValueError("STARTER_PLAY_PROBABILITY_INVALID")
            if not isinstance(current["mean"],(float,int)) or current["mean"]<0:
                raise ValueError("STARTER_MEAN_INVALID")
            if not isinstance(current["exp_opp"],(float,int)) or current["exp_opp"]<0:
                raise ValueError("STARTER_OPPORTUNITIES_INVALID")
            competing=[]
            for player in assertion["out_players"]:
                out=[x for x in qbs if normalized(x["player"])==normalized(player)]
                if len(out)!=1:
                    raise ValueError("OUT_QB_NOT_IN_ORIGINAL_QB_CANDIDATES")
                competing.append({"player":player,
                    "original_p_active":out[0]["p_active"],
                    "original_unconditional_pass_yards_mean":out[0]["mean"]})
            results={}
            for metric in OUTCOMES:
                matches=[x for x in by if x["team"]==assertion["team"] and
                          x["pos"]=="QB" and x["outcome"]==metric and
                          normalized(x["player"])==normalized(assertion["verified_starter"])]
                if len(matches)!=1:
                    raise ValueError("MISSING_OR_AMBIGUOUS_STARTER_METRIC_"+metric)
                row=matches[0]
                if row["p_active"]!=p:
                    raise ValueError("INCONSISTENT_STARTER_PLAY_PROB_ACROSS_MARKETS")
                mean=float(row["mean"])
                if not math.isfinite(mean) or mean<0:
                    raise ValueError("INVALID_METRIC_MEAN")
                results[metric]={"original_unconditional_mean":round(mean,4),
                    "conditional_mean_if_active":round(mean/p,2),
                    "original_median_including_not_playing":row["median"],
                    "original_forecast_id":row["id"]}
            scenarios.append({
                "game_id":game_id,"team":assertion["team"],
                "opponent":assertion["opponent"],"horizon":horizon,
                "kickoff_utc":game["kickoff_utc"],
                "original_forecast_cutoff_utc":meta["cutoff"],
                "official_verified_starter":assertion["verified_starter"],
                "starter_original_p_active":round(p,6),
                "original_player_mean_pass_opportunities":current["exp_opp"],
                "conditional_mean_pass_opportunities_if_active":
                    round(current["exp_opp"]/p,2),
                "qb_stat_scenarios":results,
                "out_qb_evidence":competing,
                "official_sources":[s["url"] for s in sources],
                "source_public_before_original_cutoff":True,
                "model_saw_source_at_original_cutoff":False,
                "status":"ROLE_CONFIRMED_MODEL_PROBABILITY_CONTRADICTION" if p<.75
                         else "ROLE_CONFIRMED_DIAGNOSTIC",
                "not_original_v2_prediction":True,
                "not_calibrated_starter_replacement_model":True,
                "requires_full_team_resimulation":True,
                "no_betting_prediction_or_quote":True
            })
    return {"schema":SCHEMA,"created_at_utc":now.isoformat(),
            "status":"RESEARCH_ONLY_VERIFIED_STARTER_SCENARIO_NOT_VALIDATED_FOR_BETTING",
            "original_immutable_v2_board_unchanged":True,
            "method":"E[QB STAT | ACTIVE] = original E[QB STAT] / original P(active). "+
             "Conditional mean is NOT an updated full-game median or team-coherent model.",
            "sources_were_public_before_cutoff_not_consumed_by_original_model":True,
            "scenarios":scenarios}


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--board",type=Path,required=True)
    a.add_argument("--evidence",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    args=a.parse_args()
    result=build(json.loads(args.board.read_text()),
                 json.loads(args.evidence.read_text()),
                 now=datetime.now(timezone.utc))
    if not result["scenarios"]:
        raise SystemExit("NO_VERIFIED_QB_STARTER_SCENARIO")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":result["status"],"scenarios":[{
       "game":x["game_id"],"horizon":x["horizon"],"starter":x["official_verified_starter"],
       "model_play_prob":x["starter_original_p_active"],
       "model_original_pass_mean":x["qb_stat_scenarios"]["pass_yds"]["original_unconditional_mean"],
       "conditional_pass_mean_if_active":x["qb_stat_scenarios"]["pass_yds"]["conditional_mean_if_active"],
       "conditional_expected_pass_opportunities":x["conditional_mean_pass_opportunities_if_active"]
       } for x in result["scenarios"]]},sort_keys=True))


if __name__=="__main__":
    main()
