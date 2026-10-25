#!/usr/bin/env python3
"""Create lightweight, source-preserving DAILY NHL shot-count research feeds.

The upstream frozen multi-date 14MB NHL-V2 shadow is valid but far too large
for the time-sensitive /api/nhl-count-research endpoint. These files are
exact ORIGINAL rows partitioned by source-record NHL game_date; NO model rerun,
new prediction, scoring adjustment, recalibration or bookmaker prices.
"""
from __future__ import annotations
import argparse
import json
import re
from datetime import date,timedelta
from pathlib import Path

SCHEMA="nhl-v2-sog-nfl-style-shadow-v1"
PARTITION_SCHEMA="NHL_V2_ORIGINAL_SHOT_COUNT_DAILY_PARTITION_V1"
HORIZONS=("T24H","T90","T30")
MIN_HISTORY_DAYS=2
FORWARD_DAYS=5

def subset(raw:dict, day:str)->dict:
    if not re.fullmatch(r'20\d{2}-\d{2}-\d{2}',day):
        raise ValueError("NHL_INVALID_DAILY_GAME_DATE")
    date.fromisoformat(day)
    if(raw.get("schema")!=SCHEMA or raw.get("verified_original_hash_chain") is not True or
       raw.get("verified_frozen_code_and_model_lock") is not True or
       raw.get("status")!="RESEARCH_ONLY_NOT_VALIDATED_FOR_BETTING" or
       raw.get("not_a_betting_board") is not True or
       raw.get("historical_corrected_probabilities_were_not_published_pregame") is not True):
        raise ValueError("NHL_DAILY_UNVERIFIED_OR_UNCALIBRATED_SOURCE_MISMATCH")
    fields=raw.get("row_fields")
    if not isinstance(fields,list) or len(set(fields))!=len(fields) or "game_date" not in fields:
        raise ValueError("NHL_DAILY_INVALID_ORIGINAL_FIELDS")
    fdate=fields.index("game_date")
    fgid=fields.index("game_id")
    fplayer=fields.index("player_id")
    fstart=fields.index("scheduled_start")
    horizons={}
    sumrows=0
    for h in HORIZONS:
        orig=raw.get("forecasts",{}).get(h,[])
        if not isinstance(orig,list):
            raise ValueError("NHL_DAILY_INVALID_ORIGINAL_HORIZON")
        rows=[r for r in orig if r[fdate]==day]
        if any(len(r)!=len(fields) for r in rows):
            raise ValueError("NHL_DAILY_INVALID_PARTITION_ROW")
        ids=[(str(r[fgid]),str(r[fplayer]),r[fstart]) for r in rows]
        if len(ids)!=len(set(ids)):
            raise ValueError("NHL_DAILY_DUPLICATE_ORIGINAL_PLAYER_HORIZON")
        horizons[h]=rows
        sumrows+=len(rows)
    # Metadata is faithful to locked full source. The original full document
    # remains independently readable: we do not replace it with a new model.
    output={**raw,"dates":[day] if sumrows else [],
            "forecasts":horizons,
            "total_valid_forecasts":sumrows,
            "daily_partition_schema":PARTITION_SCHEMA,
            "daily_partition_game_date":day,
            "full_source_total_valid_forecasts":raw["total_valid_forecasts"],
            "all_rows_exactly_preserved_from_original":True,
            "no_new_model_scoring_or_probability_calibration":True,
            "not_a_betting_board":True}
    return output

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,default=Path("docs/nhl_v2_shadow.json"))
    p.add_argument("--out-dir",type=Path,default=Path("docs/nhl_sog_daily"))
    p.add_argument("--today-et",required=True)
    a=p.parse_args()
    today=date.fromisoformat(a.today_et)
    source=json.loads(a.source.read_text())
    a.out_dir.mkdir(parents=True,exist_ok=True)
    generated=0
    for offset in range(-MIN_HISTORY_DAYS,FORWARD_DAYS+1):
        day=(today+timedelta(days=offset)).isoformat()
        x=subset(source,day)
        path=a.out_dir/(day+".json")
        if not x["total_valid_forecasts"]:
            if path.exists():path.unlink()
            continue
        path.write_text(json.dumps(x,separators=(',',':'),allow_nan=False)+"\n")
        generated+=1
        print("NHL_DAILY_ORIGINAL_SOG_SOURCE",day,x["total_valid_forecasts"],
              len(path.read_bytes()))
    print("NHL_DAILY_ORIGINAL_SHOT_COUNT_PARTITIONS_WRITTEN",generated)

if __name__=="__main__":
    main()
