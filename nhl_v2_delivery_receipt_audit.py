#!/usr/bin/env python3
"""NHL B2 genuine pregame delivery audit -- observable receipts, never synthetic uptime.

Works on the original verified hash-chained model ledger. A FORECAST must
have been issued before its immutable cutoff. A consumed missed/failure
decision stays missed. A completed Actions run is not proof of forecasts.

This reports only *observed decisions*, not total scheduled NHL games:
without a separately verified schedule, an absent game is unknowable here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA="NHL_V2_FIRST_PREGAME_DELIVERY_DIAGNOSTIC_V1"
FAILURES=("MISSED_CUTOFF","MISSED_REVISED_CUTOFF","INVALID_LATE",
          "SOURCE_INCOMPLETE_NO_FORECAST","SOURCE_FETCH_FAILED_NO_FORECAST")
KINDS=("FORECAST",)+FAILURES
HORIZONS=("T24H","T90","T30")


def timestamp(s):
    if not isinstance(s,str):
        raise ValueError("missing timestamp")
    t=datetime.fromisoformat(s.replace("Z","+00:00"))
    if t.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return t.astimezone(timezone.utc)


def analyze(rows, now, *, window_hours=72):
    if now.tzinfo is None:
        raise ValueError("audit time must be timezone-aware")
    now=now.astimezone(timezone.utc)
    cutoff_window=now-timedelta(hours=window_hours)
    observed=defaultdict(list)
    for row in rows:
        typ=row.get("record_type")
        if typ not in KINDS:
            continue
        if row.get("forecast_horizon") not in HORIZONS:
            raise ValueError("unsupported actual decision horizon")
        start=timestamp(row.get("scheduled_start"))
        cutoff=timestamp(row.get("cutoff_at"))
        if not cutoff<start:
            raise ValueError("decision cutoff at/after game start")
        identity=(str(row.get("game_id")),row["forecast_horizon"],
                  start.isoformat())
        if typ=="FORECAST":
            generated=timestamp(row.get("generated_at"))
            if generated>cutoff or generated>=start:
                raise ValueError("postcutoff model candidate cannot be a pregame forecast")
        else:
            timestamp(row.get("observed_at"))
        observed[identity].append(row)
    summaries=[]
    for key,records in sorted(observed.items()):
        types={r["record_type"] for r in records}
        if len(types)!=1:
            raise ValueError("same decision has both a valid forecast and an invalid/missed status")
        typ=next(iter(types))
        if typ!="FORECAST" and len(records)!=1:
            raise ValueError("duplicate immutable missed/failure decision for one game/horizon")
        if typ=="FORECAST":
            players={r.get("player_id") for r in records}
            if None in players or len(players)!=len(records):
                raise ValueError("duplicate or null player identity within a game forecast")
        else:
            players=set()
        summaries.append({
            "game_id":key[0],"horizon":key[1],
            "scheduled_start":key[2],
            "cutoff_at":timestamp(records[0]["cutoff_at"]).isoformat(),
            "decision":typ,
            "forecast_players":len(players),
            "observed_at":max(
                timestamp(r.get("generated_at") if typ=="FORECAST"
                          else r.get("observed_at")) for r in records).isoformat(),
        })
    recent=[r for r in summaries
            if cutoff_window<=timestamp(r["cutoff_at"])<=now]
    counts=Counter(x["decision"] for x in summaries)
    recent_counts=Counter(x["decision"] for x in recent)
    passed=sum(x["decision"]=="FORECAST" for x in recent)
    missed=sum(recent_counts.get(x,0) for x in FAILURES)
    by_horizon={}
    for h in HORIZONS:
        rr=[r for r in recent if r["horizon"]==h]
        by_horizon[h]={
            "genuine_observed_decision_games":len(rr),
            "forecast_game_decisions":sum(r["decision"]=="FORECAST" for r in rr),
            "failed_or_missed_decisions":sum(r["decision"] in FAILURES for r in rr),
            "model_forecast_player_rows":sum(r["forecast_players"] for r in rr),
        }
    if missed:
        status="GAPS_PROVEN_RECENT_UNMISSED_WINDOW_RECOVERY_FORBIDDEN"
    elif passed:
        status="SOME_VALID_CAPTURED_FORECASTS_SCHEDULE_COMPLETENESS_UNPROVEN"
    else:
        status="NO_RECENT_OBSERVED_DECISIONS_SCHEDULE_COMPLETENESS_UNKNOWN"
    latest_fc=max((r["observed_at"] for r in summaries if r["decision"]=="FORECAST"),
                  default=None)
    latest_event=max((r["observed_at"] for r in summaries),default=None)
    return {
        "schema":SCHEMA,"status":status,
        "report_created_at_utc":now.isoformat(),
        "recent_window_hours":window_hours,
        "total_ledger_decision_games":len(summaries),
        "all_recorded_decision_statuses":dict(sorted(counts.items())),
        "recent_observed_decision_games":len(recent),
        "recent_statuses":dict(sorted(recent_counts.items())),
        "by_horizon":by_horizon,
        "last_genuine_frozen_pregame_forecast_utc":latest_fc,
        "last_observed_cutoff_decision_utc":latest_event,
        "recent_missed_or_failed_decisions":missed,
        "recent_forecast_game_decisions":passed,
        "recent_missed_examples":[r for r in recent
                                  if r["decision"] in FAILURES][:12],
        "stated_limitations":[
            "Only decisions already recorded in immutable research ledger are counted.",
            "A scheduled NHL matchup missing entirely from the ledger is NOT counted here.",
            "GitHub cron trigger liveness requires independent Actions run audit.",
            "Finished games must NEVER have a retroactively invented pregame receipt.",
            "Original frozen model, past outcomes, player roles, and sportsbook lines unchanged."
        ],
        "production_betting_picks_added":0,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ledger",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--now",default=None)
    args=p.parse_args()
    raw=args.ledger.read_bytes()
    rows=[json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    now=timestamp(args.now) if args.now else datetime.now(timezone.utc)
    report=analyze(rows,now)
    report["original_hash_chained_ledger_sha256"]=hashlib.sha256(raw).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in
                      ("status","total_ledger_decision_games",
                       "recent_missed_or_failed_decisions",
                       "recent_forecast_game_decisions",
                       "last_genuine_frozen_pregame_forecast_utc",
                       "by_horizon")},sort_keys=True))


if __name__=="__main__":
    main()
