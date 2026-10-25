#!/usr/bin/env python3
"""Fast NHL game-date projection delivery. Derive, never refit.

The trusted source is the original immutable-lock, hash-chained NHL V2 shot
research artifact (may be >14 MB). Serving ALL historical horizons from that
artifact for today's 4-game slate is unnecessary and can timeout on Render.

Publish a small source-equivalent artifact with ONLY the newest two verified
dates and ONE best *actually existing frozen pretip horizon* per player/game
(T30 > T90 > T24H); original mean, median, P10-P90 and p_ge1..5 remain
byte-for-byte numeric values from source. This changes delivery, not modeling
or certification. No bookmaker and no retrospectively manufactured output.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from datetime import datetime, timezone
import re

SOURCE_SCHEMA = "nhl-v2-sog-nfl-style-shadow-v1"
REQUIRED=("game_id","player_id","team","opponent","game_date",
          "scheduled_start","mean","median","generated_at","cutoff_at",
          "meaningful_candidate","original_row_hash",
          "p_ge1","p_ge2","p_ge3","p_ge4","p_ge5")
HORIZONS=("T30","T90","T24H")

def utc(x):
    if not isinstance(x,str):
        raise ValueError("NHL_COMPACT_UNVERIFIED_CLOCK")
    dt=datetime.fromisoformat(x.replace("Z","+00:00"))
    if dt.tzinfo is None:raise ValueError("NHL_COMPACT_NAIVE_CLOCK")
    return dt.astimezone(timezone.utc)

def make_compact(source,dates_to_keep=2):
    if (not isinstance(source,dict)
        or source.get("schema")!=SOURCE_SCHEMA
        or source.get("status")!="RESEARCH_ONLY_NOT_VALIDATED_FOR_BETTING"
        or source.get("verified_original_hash_chain") is not True
        or source.get("verified_frozen_code_and_model_lock") is not True
        or source.get("not_a_betting_board") is not True
        or source.get("dressed_lineups_verified") is not False
        or source.get("actual_player_book_lines_verified") is not False
        or source.get("historical_corrected_probabilities_were_not_published_pregame") is not True):
        raise ValueError("NHL_COMPACT_SOURCE_NOT_LOCKED_RESEARCH")
    cols=source.get("row_fields")
    if not isinstance(cols,list) or len(set(cols))!=len(cols) or any(k not in cols for k in REQUIRED):
        raise ValueError("NHL_COMPACT_SOURCE_ROW_SCHEMA")
    col={name:i for i,name in enumerate(cols)}
    if not isinstance(source.get("forecasts"),dict) or not isinstance(source.get("dates"),list):
        raise ValueError("NHL_COMPACT_NO_VERIFIED_SCHEDULE")
    dates=sorted(set(source["dates"]))
    if not dates or not all(isinstance(d,str) and re.fullmatch(r"20\d{2}-\d\d-\d\d",d) for d in dates):
        raise ValueError("NHL_COMPACT_INVALID_GAME_DATES")
    selected=dates[-dates_to_keep:]
    winners={}
    seen_source=set()
    for h in HORIZONS:
        if h not in source.get("horizons",[]):
            continue
        for r in source["forecasts"].get(h,[]):
            if not isinstance(r,list) or len(r)!=len(cols):
                raise ValueError("NHL_COMPACT_BAD_SOURCE_ROW_LENGTH")
            if r[col["game_date"]] not in selected or r[col["meaningful_candidate"]] is not True:
                continue
            if r[col["mean"]]<0 or r[col["median"]]<0:raise ValueError("NHL_COMPACT_NEGATIVE_COUNT")
            if not utc(r[col["generated_at"]])<=utc(r[col["cutoff_at"]])<utc(r[col["scheduled_start"]]):
                raise ValueError("NHL_COMPACT_NOT_ORIGINAL_PREGAME")
            if r[col["forecast_horizon"]]!=h:
                raise ValueError("NHL_COMPACT_FORECAST_HORIZON_MISMATCH")
            key=(str(r[col["game_id"]]),str(r[col["player_id"]]))
            if not key[0] or not key[1] or key[0]=="None" or key[1]=="None":
                raise ValueError("NHL_COMPACT_PLAYER_ID_UNVERIFIED")
            orig=(h,*key)
            if orig in seen_source:raise ValueError("NHL_COMPACT_DUPLICATE_SOURCE_PLAYER_HORIZON")
            seen_source.add(orig)
            if key not in winners:
                winners[key]=(h,r)
            else:
                prev=winners[key][1]
                if (prev[col["scheduled_start"]]!=r[col["scheduled_start"]]
                    or prev[col["team"]]!=r[col["team"]]
                    or prev[col["opponent"]]!=r[col["opponent"]]):
                    raise ValueError("NHL_COMPACT_CONFLICTING_PLAYER_GAME_IDENTITY")
    derived={h:[] for h in HORIZONS if h in source.get("horizons",[])}
    for h,r in winners.values():
        derived[h].append(r.copy())
    for h,rows in derived.items():
        rows.sort(key=lambda r:(r[col["game_date"]],
               str(r[col["game_id"]]),str(r[col["player_id"]])))
    total=sum(len(v) for v in derived.values())
    if not total:raise ValueError("NHL_COMPACT_NO_VALID_NEWEST_PLAYER_COUNTS")
    out={k:copy.deepcopy(v) for k,v in source.items()
         if k not in {"forecasts","horizons","dates","total_valid_forecasts"}}
    out.update({
        "forecasts":derived,"horizons":list(derived),"dates":selected,
        "total_valid_forecasts":total,
        "compact_delivery_schema":"NHL_V2_ORIGINAL_COUNT_BEST_HORIZON_DATE_DELIVERY_V1",
        "compact_generation_policy":"T30_THEN_T90_THEN_T24H_FIRST_REAL_ORIGINAL",
        "compact_source_historical_population_retained":False,
        "original_v2_numerical_forecasts_unchanged":True,
        "model_probabilities_prospectively_calibrated":False,
        "delivered_meaningful_player_forecasts":total,
        "no_goal_market_in_this_shot_only_artifact":True,
    })
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,default=Path("docs/nhl_v2_shadow.json"))
    p.add_argument("--out",type=Path,default=Path("docs/nhl_v2_shots_compact.json"))
    a=p.parse_args()
    source=json.loads(a.source.read_text())
    out=make_compact(source)
    if a.out.exists():
        old=json.loads(a.out.read_text())
        if old==out:
            print("NHL_COMPACT_NO_CHANGE",out["dates"],out["total_valid_forecasts"])
            return
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,separators=(",",":"),ensure_ascii=False)+"\n")
    print("NHL_COMPACT_SOURCE_VERIFIED",out["dates"],out["total_valid_forecasts"])

if __name__=="__main__":main()
