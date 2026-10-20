#!/usr/bin/env python3
"""Postgame coverage guard for the EXISTING NFL V2 true-stats grader.

Never substitute re-created forecasts, legacy picks or hypothetical lines.
Observe independently finalized games; compare their original T24/T90
frozen batch existence with rows already graded against actual player stats.
A completed game with a frozen forecast but no grade is an explicit failure,
rather than a green workflow showing some other, older week's grades.
"""
import argparse
import json
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

SCHEMA="NFL_V2_POSTGAME_GRADING_COVERAGE_V1"
BASE="https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
TEAMS={"LAR":"LA","WSH":"WAS","JAC":"JAX"}
def team(x):return TEAMS.get(x,x)

def finals_from_scoreboards(responses,now):
    """Only events objectively COMPLETED and at least 6h after kickoff."""
    out={}
    for day,obj in responses:
        if not isinstance(obj.get("events"),list):
            raise ValueError("ESPN did not return a valid scoreboard for "+day)
        for event in obj["events"]:
            if not (event.get("status") or {}).get("type",{}).get("completed"):
                continue
            stage=((event.get("season") or {}).get("type") or
                   (obj.get("season") or {}).get("type"))
            if stage is not None and int(stage)!=2: # NFL regular-season
                continue
            comp=(event.get("competitions") or [{}])[0]
            sides=comp.get("competitors") or []
            if len(sides)!=2:raise ValueError("Expected two teams in completed NFL game")
            home=next((x for x in sides if x.get("homeAway")=="home"),None)
            away=next((x for x in sides if x.get("homeAway")=="away"),None)
            if not home or not away:raise ValueError("No explicit ESPN home/away")
            kickoff=datetime.fromisoformat(event["date"].replace("Z","+00:00"))
            if kickoff.tzinfo is None:raise ValueError("ESPN kickoff timezone absent")
            if now-kickoff<timedelta(hours=6):
                continue
            # ESPN event-level week varies by API shape. Top-level week
            # is safe only when all scoreboard events belong to that week.
            week=(event.get("week") or {}).get("number")
            if week is None:week=(obj.get("week") or {}).get("number")
            if week is None:raise ValueError("ESPN scoreboard lacks week for completed game")
            season=(event.get("season") or {}).get("year")
            if season is None:season=(obj.get("season") or {}).get("year")
            if season is None:raise ValueError("ESPN scoreboard lacks season year")
            gid=f'{int(season)}_{int(week):02d}_{team(away["team"]["abbreviation"])}_{team(home["team"]["abbreviation"])}'
            if gid in out:
                if out[gid]["espn_id"]!=str(event["id"]):
                    raise ValueError("Conflicting completed ESPN game ID")
                continue
            out[gid]={"game_id":gid,"espn_id":str(event["id"]),
                 "kickoff_utc":kickoff.astimezone(timezone.utc).isoformat(),
                 "away":team(away["team"]["abbreviation"]),
                 "home":team(home["team"]["abbreviation"]),
                 "away_score":int(away["score"]),"home_score":int(home["score"]),
                 "week":int(week)}
    return out

def build_status(scorecard,state,finals,now):
    if scorecard.get("schema")!="NFL_V2_FIRST_SEEN_REAL_FINAL_NUMERIC_SCORECARD_V1":
        raise ValueError("Not the genuine original NFL V2 grade schema")
    if not scorecard.get("first_seen_pre_cutoff_only") or scorecard.get("automatic_model_promotion"):
        raise ValueError("Forecast first-seen integrity not verified")
    graded={}
    for row in scorecard.get("all_graded_rows") or []:
        if row.get("horizon") not in ("T24","T90"):
            raise ValueError("Unknown V2 horizon in grade")
        graded.setdefault((row["game_id"],row["horizon"]),0)
        graded[(row["game_id"],row["horizon"])]+=1
    latest=[]
    pending=[]
    missing_forecast=[]
    for gid,g in sorted(finals.items()):
        for h in ("T24","T90"):
            # Immutable filenames: 2026_wk06_T24_2026_06_HOU_JAX.jsonl
            path=Path(state)/"forecasts"/"batches"/(
                 f'{gid[:4]}_wk{g["week"]:02d}_{h}_{gid}.jsonl')
            num=graded.get((gid,h),0)
            if not path.is_file():
                missing_forecast.append({"game_id":gid,"horizon":h,
                   "reason":"NO_GENUINE_FROZEN_PREGAME_BATCH_NOT_A_GRADING_FAILURE"})
            elif num<1:
                pending.append({"game_id":gid,"horizon":h,
                   "reason":"COMPLETED_GAME_FROZEN_FORECAST_NOT_GRADED"})
            else:
                latest.append({"game_id":gid,"horizon":h,"graded_player_outcomes":num})
    outcome="PENDING_SOURCE_OR_GRADER" if pending else "UP_TO_DATE_FOR_AVAILABLE_FROZEN_BATCHES"
    return {"schema":SCHEMA,"asof_utc":now.astimezone(timezone.utc).isoformat(),
        "status":outcome,"source":"ESPN independently completed game scoreboards; original nfl-shadow-state batch files; existing V2 true-stats scorecard",
        "grade_source_asof_utc":scorecard.get("asof_utc"),
        "original_scorecard_outcomes":scorecard["graded_player_outcomes"],
        "recent_completed_games":list(finals.values()),
        "recent_completed_game_count":len(finals),
        "recent_frozen_game_horizons_graded":len(latest),
        "recent_frozen_game_horizons_ungraded":len(pending),
        "recent_game_horizons_missing_original_forecast":len(missing_forecast),
        "graded":latest,"not_yet_graded":pending,
        "no_original_forecast":missing_forecast,
        "research_only_no_bookmaker_wagers":True}

def fetch_scoreboard(day):
    url=BASE+"?dates="+day.strftime("%Y%m%d")+"&limit=50"
    req=urllib.request.Request(url,headers={"User-Agent":"PropEdgeNFLV2SourceCoverage/1.0",
        "Accept":"application/json"})
    with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scorecard",type=Path,required=True)
    p.add_argument("--state",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    now=datetime.now(timezone.utc)
    local=now.astimezone(ZoneInfo("America/New_York")).date()
    days=[local-timedelta(days=n) for n in range(5)]
    responses=[(d.isoformat(),fetch_scoreboard(d)) for d in days]
    finals=finals_from_scoreboards(responses,now)
    scorecard=json.loads(a.scorecard.read_text())
    status=build_status(scorecard,a.state,finals,now)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(status,sort_keys=True,indent=2)+"\n")
    print("NFL_V2_POSTGAME_GRADE_COVERAGE",json.dumps({
        k:status[k] for k in ("asof_utc","status","recent_completed_game_count",
        "recent_frozen_game_horizons_graded","recent_frozen_game_horizons_ungraded",
        "recent_game_horizons_missing_original_forecast",
        "original_scorecard_outcomes")},sort_keys=True),flush=True)
    for missing in status["not_yet_graded"]:
        print("::warning::COMPLETED_FROZEN_NFL_V2_UNGRADED "+missing["game_id"]+" "+missing["horizon"],flush=True)
    for missing in status["no_original_forecast"]:
        print("::notice::ORIGINAL_NFL_V2_FORECAST_MISSING "+missing["game_id"]+" "+missing["horizon"],flush=True)

if __name__=="__main__":main()
