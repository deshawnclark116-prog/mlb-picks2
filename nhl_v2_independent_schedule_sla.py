#!/usr/bin/env python3
"""Independent NHL V2 deadline SLA: schedule expected vs ACTUAL frozen receipts.

No NHL model code imports, no sportsbook inputs, no retrospective forecasts.
A green collector job, an explicit MISSED row, or a research page containing
old predictions cannot satisfy an expected game/horizon deadline. Only an
on-time frozen FORECAST receipt for the identical game/start/horizon counts.
Schedule is fetched from the NHL, not inferred from collector's own ledger.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nhl_v2_secondary_cutoff_gate import NHL_SCHEDULE, load_json, utc

UTC=timezone.utc
HORIZONS={"T24H":1440,"T90":90,"T30":30}
STATUS="NHL_V2_INDEPENDENT_SCHEDULE_CUTOFF_SLA_V1"


def verify_ledger(path):
    data=Path(path).read_bytes()
    prev="0"*64
    rows=[]
    for i,line in enumerate(data.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        row=json.loads(line)
        h=hashlib.sha256(json.dumps(
            {k:v for k,v in row.items() if k!="row_hash"},
            sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
        if row.get("prev_hash")!=prev or row.get("row_hash")!=h or row.get("seq")!=len(rows):
            raise ValueError(f"Frozen NHL hash-chain invalid at row {i}")
        rows.append(row)
        prev=h
    return rows,hashlib.sha256(data).hexdigest()


def expected(schedule_payloads,now,lock_eligible,*,hours=60,grace_minutes=10):
    by_game={}
    for data in schedule_payloads:
        if not isinstance(data,dict) or not isinstance(data.get("gameWeek"),list):
            raise ValueError("Official NHL schedule unavailable/malformed")
        for day in data["gameWeek"]:
            for g in day.get("games",[]):
                if (g.get("gameType")!=2 or not g.get("startTimeUTC")
                        or g.get("gameScheduleState","OK")!="OK"):
                    continue
                start=utc(g["startTimeUTC"])
                gid=str(g["id"])
                key=(gid,start.isoformat())
                by_game[key]=g
    expected_keys=[]
    for (gid,start_txt),g in sorted(by_game.items()):
        start=utc(start_txt)
        for horizon,minutes in HORIZONS.items():
            cutoff=start-timedelta(minutes=minutes)
            if cutoff<lock_eligible or cutoff<now-timedelta(hours=hours) or cutoff>now-timedelta(minutes=grace_minutes):
                continue
            expected_keys.append({
                "game_id":gid,"horizon":horizon,"scheduled_start":start.isoformat(),
                "cutoff_at":cutoff.isoformat(),
                "schedule_state":g.get("gameState"),
            })
    return expected_keys


def evaluate(schedule_payloads,ledger,now,lock_eligible,*,hours=60):
    exp=expected(schedule_payloads,now,lock_eligible,hours=hours)
    recorded={}
    for row in ledger:
        kind=row.get("record_type")
        if kind not in ("FORECAST","MISSED_CUTOFF","MISSED_REVISED_CUTOFF",
                        "INVALID_LATE","SOURCE_FETCH_FAILED_NO_FORECAST",
                        "SOURCE_INCOMPLETE_NO_FORECAST"):
            continue
        ident=(str(row.get("game_id")),row.get("forecast_horizon"),
               utc(row["scheduled_start"]))
        recorded.setdefault(ident,[]).append(row)
    missing=[]
    successful=[]
    for item in exp:
        key=(item["game_id"],item["horizon"],utc(item["scheduled_start"]))
        rows=recorded.get(key,[])
        cutoff=utc(item["cutoff_at"])
        timed=[r for r in rows
               if r["record_type"]=="FORECAST"
               and r.get("player_id") is not None
               and cutoff-timedelta(minutes=15)<=utc(r["generated_at"])<=cutoff]
        counts=Counter(str(r["player_id"]) for r in timed)
        valid=[r for r in timed if counts[str(r["player_id"])]==1]
        # The frozen B2 model performs separate source-manifest integrity
        # checks. This checker independently audits the exact game/start,
        # cutoff, timestamp and non-ambiguous player delivery.
        if valid:
            successful.append({**item,"unique_player_rows":len(
                {str(r["player_id"]) for r in valid})})
        else:
            failures=Counter(r["record_type"] for r in rows)
            missing.append({**item,"reason":"MISSED_OR_FAILED" if failures
                            else "NO_LEDGER_DECISION_RECORDED",
                            "recorded_failure_types":dict(sorted(failures.items()))})
    # Preserve every miss from the full audit window, but alert only on
    # recently elapsed deadlines. Historic misses are immutable evidence, not
    # newly broken forecasts on every successive monitoring run.
    recent_missing=[x for x in missing
                    if now-timedelta(hours=24)<utc(x["cutoff_at"])<=now]
    return {
        "schema":STATUS,"asof_utc":now.isoformat(),
        "new_missing_or_failed_last_24h":len(recent_missing),
        "older_missing_or_failed_in_audit_window":len(missing)-len(recent_missing),
        "new_missed_examples":recent_missing[:12],
        "monitoring_alert_policy":"DAILY_NEW_MISSES_ONLY_KEEP_HISTORICAL_MISSES_RECORDED",
        "observation_hours":hours,
        "expected_game_horizon_decisions":len(exp),
        "genuine_pregame_game_horizon_forecasts":len(successful),
        "missing_or_failed":len(missing),
        "per_horizon":{h:{
            "expected":sum(x["horizon"]==h for x in exp),
            "forecasted":sum(x["horizon"]==h for x in successful),
            "missing":sum(x["horizon"]==h for x in missing)
        } for h in HORIZONS},
        "missed_examples":missing[:30],
        "status":"FAILED_REAL_PREGAME_CUTOFF_SLA" if missing else
                 ("PASS_ALL_OFFICIAL_SCHEDULED_CUTOFFS_OBSERVED"
                  if exp else "NO_DUE_GAMES_IN_OBSERVATION_WINDOW"),
        "caveat":"Official schedule retrieved at audit time may reflect revisions; missed historical decisions are never repaired",
        "non_betting_research_only":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ledger",type=Path,required=True)
    p.add_argument("--lock",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--hours",type=int,default=60)
    args=p.parse_args()
    if not 1<=args.hours<=168:
        p.error("hours must be 1..168")
    now=datetime.now(UTC)
    # Days on both sides of today's week boundary. Deduplicated by real
    # NHL game ID and scheduled start; NOT dependent on collector's ledger.
    payloads=[]
    for offset in (-3,-1,0,1):
        day=(now+timedelta(days=offset)).strftime("%Y-%m-%d")
        payloads.append(load_json(f"https://api-web.nhle.com/v1/schedule/{day}"))
    rows,sha=verify_ledger(args.ledger)
    lock=json.loads(args.lock.read_text())
    if not lock.get("eligible_from_cutoff_utc"):
        raise ValueError("original model lock has no eligible start")
    result=evaluate(payloads,rows,now,utc(lock["eligible_from_cutoff_utc"]),hours=args.hours)
    result["original_verified_ledger_sha256"]=sha
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("NHL_INDEPENDENT_SCHEDULE_VS_ACTUAL_FORECAST_RECEIPTS",
          json.dumps(result,sort_keys=True),flush=True)
    # One daily run emits a failing Actions notification only when a
    # deadline was newly missed in the preceding 24h. The report still
    # marks older missing forecasts as FAILED_REAL_PREGAME_CUTOFF_SLA.
    if result["new_missing_or_failed_last_24h"]:
        print("::error::NEW NHL V2 pregame forecast deadlines missed in last 24h",flush=True)
        return 2
    if result["missing_or_failed"]:
        print("HISTORICAL_NHL_FORECAST_MISS_RETAINED_NO_DUPLICATE_FAILURE_EMAIL",
              result["missing_or_failed"],flush=True)
    return 0
    return 0


if __name__=="__main__":
    sys.exit(main())
