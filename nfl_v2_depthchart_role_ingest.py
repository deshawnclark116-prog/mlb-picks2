#!/usr/bin/env python3
"""Read-only, inexpensive team depth-chart + roster QUARTERBACK ROLE evidence.

ESPN rank=1 IS NOT an official game-day starting lineup. This capture cannot
 certify a starter, create sportsbook picks, or backdate a frozen T24/T90 run.
Only presents a conditional *if depth-chart QB1 starts* role hypothesis
where identity matches the published V2 game and the player is not listed
out / IR / questionable. Stale/missing/contradictory data fail closed.
"""
from __future__ import annotations
import argparse,hashlib,json,re,unicodedata
from collections import defaultdict
from datetime import datetime,timezone,timedelta
from pathlib import Path
import requests

SCHEMA="NFL_V2_ASOF_ESPN_DEPTH_CHART_ROLE_EVIDENCE_V1"
TEAMS={"LA":"lar","WAS":"wsh"}
ESPN="https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
UNAVAILABLE={"out","injured reserve","injuredreserve","reserve","suspended",
             "physically unable to perform","pup","reserve/injured"}
CAUTION={"questionable","doubtful","limited","day-to-day","probable"}
DEFERRAL="NO_VERIFIED_GAME_DAY_STARTER"


def utc(x):
    r=datetime.fromisoformat(str(x).replace("Z","+00:00"))
    if r.tzinfo is None:raise ValueError("NAIVE_TIMESTAMP")
    return r.astimezone(timezone.utc)


def norm(x):
    s=unicodedata.normalize("NFKD",str(x or ""))
    s="".join(c for c in s if not unicodedata.combining(c)).lower()
    s=re.sub(r"\b(jr|sr|ii|iii|iv)\.?\b","",s)
    return " ".join(re.sub("[^a-z ]"," ",s).split())


def roster_status(roster,name):
    found=[]
    for group in roster.get("athletes",[]):
        for r in group.get("items") or []:
            if norm(r.get("displayName") or r.get("fullName"))==norm(name):
                found.append(r)
    if len(found)!=1:return "ROSTER_IDENTITY_MISSING_OR_AMBIGUOUS"
    r=found[0]
    statuses={str(r.get("status") or "").lower().strip()}
    statuses|={str(x.get("status") or "").lower().strip() for x in r.get("injuries") or []
                if isinstance(x,dict)}
    statuses|={str(x.get("type") or "").lower().strip() for x in r.get("injuries") or []
                if isinstance(x,dict)}
    if any(any(token in item for token in UNAVAILABLE) for item in statuses):
        return "ROSTER_LISTED_UNAVAILABLE"
    if any(any(token in item for token in CAUTION) for item in statuses):
        return "ROSTER_INJURY_CAUTION_NOT_STARTER_VERIFIED"
    return "ROSTER_PRESENT_NOT_STARTER_VERIFIED"


def ranked_qbs(chart):
    ret=[]
    for group in chart.get("depthCharts") or []:
        for key,position in (group.get("positions") or {}).items():
            p=position.get("position") or {}
            if key.lower() not in ("qb","quarterback") and str(p.get("abbreviation","")).upper()!="QB":
                continue
            for item in position.get("athletes") or []:
                a=item.get("athlete") or {}
                try:
                    rank=int(item["rank"])
                except (KeyError,ValueError,TypeError):continue
                name=a.get("displayName") or a.get("fullName")
                if name and rank>0:ret.append({"name":name,"rank":rank,
                    "espn_athlete_id":a.get("id")})
    # duplicate QB roles in more than one chart group are ambiguous
    return sorted(ret,key=lambda x:(x["rank"],norm(x["name"])))


def fetch_one(team,*,getter=requests.get,at):
    url=f"{ESPN}/{TEAMS.get(team,team.lower())}"
    data={}
    for name,suffix in (("depthchart","depthcharts"),("roster","roster")):
        u=url+"/"+suffix
        resp=getter(u,timeout=18)
        resp.raise_for_status()
        js=resp.json()
        if not isinstance(js,dict):raise ValueError("ESPN_NOT_OBJECT")
        data[name]=js
    qb=ranked_qbs(data["depthchart"])
    rank1=[r for r in qb if r["rank"]==1]
    if len(rank1)!=1 or len({norm(x["name"]) for x in qb})!=len(qb):
        return {"team":team,"status":"NO_UNAMBIGUOUS_DEPTH_CHART_QB1",
                "depthchart_url":url+"/depthcharts","captured_at_utc":at.isoformat(),
                "qb_rankings":qb}
    primary=rank1[0]
    status=roster_status(data["roster"],primary["name"])
    accepted=status=="ROSTER_PRESENT_NOT_STARTER_VERIFIED"
    return {"team":team,"status":"CONDITIONAL_QB1_ROLE_CANDIDATE" if accepted else
            "DEPTH_CHART_CANNOT_RESOLVE_STARTER",
            "depthchart_url":url+"/depthcharts",
            "roster_url":url+"/roster",
            "captured_at_utc":at.isoformat(),
            "qb_rankings":qb[:6],"primary":primary,"primary_roster_status":status,
            "source_tier":"THIRD_PARTY_DEPTH_CHART_NOT_CONFIRMED_GAME_DAY_STARTER",
            "has_verified_game_day_starter":False,
            "source_hash_sha256":hashlib.sha256(json.dumps(
                {"depthchart":data["depthchart"],"roster":data["roster"]},
                sort_keys=True).encode()).hexdigest()}


def capture(board,*,now,getter=requests.get):
    if board.get("schema")!="nfl-new-engine-shadow-v1" or board.get(
            "status",{}).get("publication_state")!="PUBLISHED":
        raise ValueError("NO_VERIFIED_IMMUTABLE_V2_BOARD")
    futures=[g for g in board.get("games",[]) if utc(g["kickoff_utc"])>now]
    teams=sorted({t for g in futures for t in (g["home"],g["away"])})
    report={};errors={}
    for team in teams:
        try:
            report[team]=fetch_one(team,getter=getter,at=now)
        except (requests.RequestException,ValueError,KeyError) as e:
            # Do not leak URL query or auth into logs. ESPN endpoints are
            # keyless. A missing capture is never an active designation.
            errors[team]=type(e).__name__
            report[team]={"team":team,"status":"ESPN_SOURCE_UNAVAILABLE_NO_STARTER_INFERENCE",
                          "captured_at_utc":now.isoformat()}
    return {"schema":SCHEMA,"generated_at_utc":now.isoformat(),
            "source":"ESPN_TEAM_DEPTH_CHART_AND_ROSTER_KEYLESS",
            "not_official_starter_confirmation":True,
            "not_original_v2_frozen_forecast":True,
            "no_betting_picks":True,
            "team_data":report,"source_errors":errors,
            "upcoming_game_ids":[g["game_id"] for g in futures]}


def provisional_targets(board,depth,*,now,horizons=("T24","T90")):
    if depth.get("schema")!=SCHEMA or not depth.get("not_official_starter_confirmation"):
        raise ValueError("UNVERIFIABLE_DEPTHCHART_SNAPSHOT")
    captured=utc(depth["generated_at_utc"])
    if captured>now or now-captured>timedelta(hours=4):
        raise ValueError("STALE_OR_FUTURE_QB_ROLE_SNAPSHOT")
    fields=board["row_fields"]
    keys=("game_id","player","team","pos","outcome","median","p_active")
    ix={k:fields.index(k) for k in keys}
    output=[]
    for game in board["games"]:
        if utc(game["kickoff_utc"])<=now:continue
        for team in (game["home"],game["away"]):
            record=depth["team_data"].get(team) or {}
            if record.get("status")!="CONDITIONAL_QB1_ROLE_CANDIDATE":continue
            primary=record["primary"]["name"]
            for horizon in horizons:
                meta=game.get("horizons",{}).get(horizon)
                if not meta:continue
                qbs=[r for r in board["forecasts"].get(horizon,[]) if
                      r[ix["game_id"]]==game["game_id"] and r[ix["team"]]==team
                      and r[ix["pos"]]=="QB" and r[ix["outcome"]]=="pass_yds"]
                matches=[r for r in qbs if norm(r[ix["player"]])==norm(primary)]
                if len(matches)!=1:continue
                # Expensive full-event resimulation is restricted to a
                # concrete role contradiction, not every ordinary QB1.
                highest=max((float(r[ix["p_active"]]) for r in qbs),default=0)
                n_high=sum(float(r[ix["median"]])>=10 for r in qbs)
                if highest>=.5 and n_high<=1 and float(matches[0][ix["p_active"]])>=.5:
                    continue
                output.append({
                    "game_id":game["game_id"],
                    "team":team,"opponent":game["away"] if team==game["home"] else game["home"],
                    "season":int(game["game_id"].split("_")[0]),
                    "week":int(game["game_id"].split("_")[1]),
                    "horizon":horizon,"kickoff_utc":game["kickoff_utc"],
                    "original_forecast_cutoff_utc":meta["cutoff"],
                    "official_verified_starter":primary, # protocol field only, proof below says NOT verified
                    "source_tier":"PROVISIONAL_DEPTH_CHART_QB1_NOT_CONFIRMED_STARTER",
                    "source_was_known_at_original_cutoff":False,
                    "latest_official_source_utc":None,
                    "official_sources":[],
                    "provisional_depthchart_source":record["depthchart_url"],
                    "captured_at_utc":record["captured_at_utc"],
                    "original_model_p_active":matches[0][ix["p_active"]],
                    "new_pregame_scenario_generated_after_original_cutoff":now>utc(meta["cutoff"]),
                    "must_not_relabel_as_original_T24_T90":True,
                    "not_game_day_verified_starter":True})
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--board",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    now=datetime.now(timezone.utc)
    r=capture(json.loads(a.board.read_text()),now=now)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    print("NFL_V2_ESPN_DEPTH_CHART_RESEARCH",json.dumps({
        "teams":len(r["team_data"]),
        "conditional_candidates":sum(x["status"]=="CONDITIONAL_QB1_ROLE_CANDIDATE"
                                     for x in r["team_data"].values()),
        "error_teams":len(r["source_errors"])},sort_keys=True))


if __name__=="__main__":main()
