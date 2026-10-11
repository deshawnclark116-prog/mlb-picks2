#!/usr/bin/env python3
"""Operational fallback gate for frozen NHL B2's first-pregame scheduler.

Designed for successful completions of the *existing* NHL Phase0 watchdog,
which is separately proven to receive GitHub schedule events. Gate checks
official upcoming NHL regular-season starts, never fabricates forecasts.
This file imports NO scoring engine, loads NO 2026 final game stats, makes
NO outcomes/lineup assumptions and cannot open an official betting card.
"""
import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

HORIZON_MIN=(1440,90,30)
WINDOW_SECONDS=900
NHL_SCHEDULE="https://api-web.nhle.com/v1/schedule/now"


def utc(value):
    t=datetime.fromisoformat(value.replace("Z","+00:00"))
    if t.tzinfo is None:
        raise ValueError("timezone absent")
    return t.astimezone(timezone.utc)


def due(now, schedule):
    """True only while an actual 15-minute pre-cutoff window is still open."""
    if now.tzinfo is None:
        raise ValueError("naive scheduler time")
    if not isinstance(schedule,dict) or not isinstance(schedule.get("gameWeek"),list):
        raise ValueError("NHL schedule payload missing gameWeek")
    qualified=[]
    seen=set()
    for day in schedule["gameWeek"]:
        for game in day.get("games",[]):
            if (game.get("gameType")!=2
                    or game.get("gameState") not in ("FUT","PRE")
                    or game.get("gameScheduleState","OK")!="OK"):
                continue
            gid=game.get("id")
            if gid is None or not game.get("startTimeUTC"):
                continue
            start=utc(game["startTimeUTC"])
            for horizon in HORIZON_MIN:
                cutoff=start-timedelta(minutes=horizon)
                seconds=(cutoff-now).total_seconds()
                identity=(str(gid),horizon,start.isoformat())
                if 0<=seconds<=WINDOW_SECONDS and identity not in seen:
                    seen.add(identity)
                    qualified.append({"game_id":str(gid),"cutoff":cutoff.isoformat(),
                                      "horizon_minutes":horizon,"seconds_until_cutoff":seconds})
    return sorted(qualified,key=lambda r:(r["cutoff"],r["game_id"],r["horizon_minutes"]))


def primary_just_succeeded(runs, now):
    """Only confirmed *scheduled* successes can suppress a backup."""
    for r in runs:
        if r.get("event")!="schedule" or r.get("conclusion")!="success":
            continue
        created=r.get("created_at")
        if not created:
            continue
        gap=(now-utc(created)).total_seconds()
        if 0<=gap<=12*60:
            return True
    return False


def load_json(url,token=None,timeout=12):
    headers={"User-Agent":"NHL-B2-frozen-first-pregame-ops","Accept":"application/json"}
    if token:
        headers["Authorization"]="Bearer "+token
        headers["X-GitHub-Api-Version"]="2022-11-28"
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=timeout) as response:
        return json.load(response)


def decision(now,fetch_schedule,fetch_runs=None):
    try:
        candidates=due(now,fetch_schedule())
    except Exception as e:
        return {"should_run":False,"reason":"SCHEDULE_SOURCE_UNAVAILABLE",
                "error":str(e)[:160],"due":0}
    if not candidates:
        return {"should_run":False,"reason":"NO_VERIFIED_OPEN_NHL_B2_CUTOFF",
                "due":0}
    if fetch_runs is not None:
        try:
            recently=fetch_runs()
            if not isinstance(recently,dict) or not isinstance(recently.get("workflow_runs"),list):
                raise ValueError("invalid workflow run response")
            if primary_just_succeeded(recently["workflow_runs"],now):
                return {"should_run":False,"reason":"PRIMARY_GITHUB_SCHEDULE_SUCCEEDED_RECENTLY",
                        "due":len(candidates),"examples":candidates[:4]}
        except Exception as ex:
            # An unavailable Actions read API is not proof that the primary
            # ran. The shared named job concurrency handles duplicate runs.
            return {"should_run":True,"reason":"DUE_WITH_UNVERIFIED_PRIMARY",
                    "due":len(candidates),"examples":candidates[:4],
                    "github_check_error":str(ex)[:160]}
    return {"should_run":True,"reason":"VERIFIED_OPEN_NHL_B2_CUTOFF",
            "due":len(candidates),"examples":candidates[:4]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--github-output",default=os.getenv("GITHUB_OUTPUT"))
    p.add_argument("--repo",default=os.getenv("GITHUB_REPOSITORY","deshawnclark116-prog/mlb-picks2"))
    p.add_argument("--token",default=os.getenv("GH_TOKEN"))
    args=p.parse_args()
    now=datetime.now(timezone.utc)
    endpoint=f"https://api.github.com/repos/{args.repo}/actions/workflows/nhl_v2_prospective_delivery.yml/runs?event=schedule&per_page=8"
    result=decision(now,lambda:load_json(NHL_SCHEDULE),
                    lambda:load_json(endpoint,args.token))
    print(json.dumps(result,sort_keys=True))
    if args.github_output:
        with open(args.github_output,"a",encoding="utf-8") as out:
            out.write(f"should_run={'true' if result['should_run'] else 'false'}\n")
    return 0


if __name__=="__main__":
    sys.exit(main())
