#!/usr/bin/env python3
"""Pregame NFL role receipts for the existing published numeric projections.

Produces a separate, as-of artifact without changing or retraining the
champion. Do not represent this source-coverage screen as a calibrated
sportsbook edge, lineup confirmation or a new projection model.
"""
import argparse
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import nfl_analyst_role_preflight as role

ALLOWED_MARKETS={"rushing_yards","receiving_yards"}
SNAPSHOT_SCHEMA="nfl-role-evidence-coverage-v1"
REQUIRED_RUSH=5.0
REQUIRED_TARGET=3.0

def capture(forecast,db,*,now=None,roster_fetch=role.roster_evidence):
    clock=now or datetime.now(timezone.utc)
    clock=clock.astimezone(timezone.utc)
    kickoff_by_game={}
    for g in forecast["scheduled_games"]:
        kickoff=datetime.fromisoformat(g["kickoff_utc"].replace("Z","+00:00"))
        for team,opponent in ((g["home_team"],g["away_team"]),(g["away_team"],g["home_team"])):
            kickoff_by_game[(team,opponent)]=(kickoff,g)
    generated=datetime.fromisoformat(forecast["generated_at_utc"].replace("Z","+00:00"))
    if generated>clock:
        raise ValueError("FORECAST_GENERATED_IN_FUTURE")
    teams=set()
    for p in forecast["picks"]:
        if p.get("market") not in ALLOWED_MARKETS:
            continue
        entry=kickoff_by_game.get((p.get("team"),p.get("opponent")))
        if entry and generated<entry[0] and clock<entry[0]:
            teams.add(p["team"])
    checked=clock.isoformat().replace("+00:00","Z")
    # One roster request per team, not one per player; requests run concurrently
    # and failure is represented as unknown, never as verified-active.
    with ThreadPoolExecutor(max_workers=8) as pool:
        tasks={t:pool.submit(roster_fetch,t,checked) for t in sorted(teams)}
        rosters={}
        for t,future in tasks.items():
            try:rosters[t]=future.result()
            except Exception:
                rosters[t]={"status":"ROSTER_SOURCE_UNAVAILABLE","players":{},
                            "checked_at_utc":checked}
    rows=[]
    for p in forecast["picks"]:
        market=p.get("market")
        if market not in ALLOWED_MARKETS or p.get("projected_median") is None:
            continue
        team=p.get("team"); opponent=p.get("opponent")
        match=kickoff_by_game.get((team,opponent))
        if not match or generated>=match[0] or clock>=match[0]:
            continue
        usage=role.observed_usage(db,season=forecast["season"],week=forecast["week"],
                                 team=team,player=p["player"],market=market)
        roster=rosters.get(team,{"status":"ROSTER_SOURCE_UNAVAILABLE","players":{}})
        current=role.roster_match(roster,p["player"])
        status=(current["status"] if roster["status"]=="ROSTER_CAPTURED"
                else "ROSTER_SOURCE_UNAVAILABLE")
        recent=usage.get("opportunities_per_observed_game")
        min_attempt=REQUIRED_RUSH if market=="rushing_yards" else REQUIRED_TARGET
        flags=[]
        if status!="ROSTER_LISTED_NOT_STARTER_VERIFIED":
            flags.append("PLAYER_CURRENT_ROSTER_STATUS_UNCERTAIN")
        if usage.get("observed_stat_rows")!=3 or usage.get("unknown_stat_appearances")!=0:
            flags.append("THREE_PRIOR_TEAM_GAME_ROWS_UNVERIFIED")
        if recent is None or recent<min_attempt:
            flags.append("LOW_OR_UNKNOWN_PRIOR_OPPORTUNITY")
        # Prior stats do not prove next game's snap assignment. All releases
        # remain "no validated bet" regardless of this evidence gate.
        featured_role_evidence_complete=not flags
        rows.append({
            "player":p["player"],"team":team,"opponent":opponent,
            "market":market,"kickoff_utc":match[1]["kickoff_utc"],
            "player_id":p.get("player_id"),
            "published_model_median":p["projected_median"],
            "published_model_source":p.get("model_source"),
            "roster_status":status,"recent_opportunities_per_observed_game":recent,
            "recent_team_share":usage.get("share_per_observed_game"),
            "observed_stat_rows":usage.get("observed_stat_rows",0),
            "role_status":usage.get("status"),
            "role_evidence_complete_for_showcase":featured_role_evidence_complete,
            "warnings":flags,
            "has_verified_starting_role":False,
            "has_calibrated_book_bet_probability":False,
        })
    counts=Counter(x["market"] for x in rows if x["role_evidence_complete_for_showcase"])
    return {
        "schema":SNAPSHOT_SCHEMA,"status":"PREGAME_RESEARCH_ROLE_SNAPSHOT",
        "season":forecast["season"],"week":forecast["week"],
        "forecast_generated_at_utc":forecast["generated_at_utc"],
        "captured_at_utc":checked,
        "roster_source_status":{t:r["status"] for t,r in rosters.items()},
        "player_market_evidence":rows,
        "n_forecast_rows":len(rows),"n_with_complete_historical_role_evidence":sum(counts.values()),
        "is_betting_probability_calibrated":False,
        "starter_status_verified":False,
        "warnings":"Historical opportunities and ESPN roster presence do not certify starter/inactives. "
                   "Only use to avoid advertising unsupported past roles in the featured card; "
                   "no bet expected-value validation is supplied."
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--forecast",default="docs/nfl_predictions.json")
    parser.add_argument("--db",default="nfl_models/nfl_model.sqlite")
    parser.add_argument("--out",default="docs/nfl_role_snapshot.json")
    args=parser.parse_args()
    fc=json.loads(Path(args.forecast).read_text())
    db=sqlite3.connect(f"file:{args.db}?mode=ro",uri=True)
    try:
        result=capture(fc,db)
    finally:
        db.close()
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"schema":result["schema"],
                      "forecast_rows":result["n_forecast_rows"],
                      "role_evidence_complete":result["n_with_complete_historical_role_evidence"],
                      "roster_sources":result["roster_source_status"]},sort_keys=True))
if __name__=="__main__":
    main()
