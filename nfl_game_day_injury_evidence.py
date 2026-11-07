"""NFL pregame league injury evidence: independent corroboration, NOT fabricated inactives.

A public ESPN league injury response supplements the existing ESPN team-roster
sweep. It cannot certify the 53-man active list or a starting role. A newly
reported OUT can suppress a pregame pick; DOUBTFUL/QUESTIONABLE remain
availability risks rather than logical zero-play events.

Every exclusion requires a same-season, named-team, fresh as-of injury record,
one unambiguous athlete ID and an original report date not in the future.
No injury report is permitted to rewrite earlier original model predictions.
Source outages, stale reports, conflicting reports and missing effective times
never become a "cleared/active" claim.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from datetime import datetime,timedelta,timezone
from typing import Any,Callable

import requests

URL="https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
SOURCE_SCHEMA="NFL_ESPN_LEAGUE_INJURY_ASOF_V1"
TEAM_ALIAS={"LAR":"LA","WSH":"WAS"}
OUT={
    "out","injured reserve","injuredreserve","reserve/injured",
    "suspended","suspension","commissioner exempt list","exempt",
    "physically unable to perform","pup","non-football injury",
    "reserve/retired",
}
CAUTION={"doubtful","questionable","limited","day-to-day","day to day"}
FRESH_DAYS=7

def normalized(value):
    val=unicodedata.normalize("NFKD",str(value or ""))
    val="".join(c for c in val if not unicodedata.combining(c))
    val=re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b","",val.casefold())
    return " ".join(re.sub(r"[^a-z0-9 ]"," ",val).split())

def _time(value):
    if not isinstance(value,str) or not value.strip():
        return None
    try:
        ts=datetime.fromisoformat(value.strip().replace("Z","+00:00"))
        if ts.tzinfo is None:
            # Date-only reports have no precise issuance time; fail closed
            # rather than assume they were public before our model cutoff.
            return None
        return ts.astimezone(timezone.utc)
    except ValueError:
        return None

def _empty(now,reason):
    return {
        "schema":SOURCE_SCHEMA,"status":"SOURCE_UNAVAILABLE",
        "checked_at_utc":now.isoformat(),
        "source":URL,"reason":reason,
        "season_verified":False,"records":[],
    }

def capture_league_injuries(season:int,now:datetime,http_get:Callable=requests.get,clock:Callable=lambda:datetime.now(timezone.utc)):
    if now.tzinfo is None or not isinstance(season,int):
        raise ValueError("NFL_INJURY_INVALID_ASOF_OR_SEASON")
    now=now.astimezone(timezone.utc)
    try:
        r=http_get(URL,timeout=15)
        r.raise_for_status()
        payload=r.json()
        received=clock().astimezone(timezone.utc)  # AFTER last source byte arrived
    except (requests.RequestException,OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
        return _empty(clock().astimezone(timezone.utc),"HTTP_OR_JSON_"+type(exc).__name__)
    if received<now or not isinstance(payload,dict):
        return _empty(received,"INVALID_SOURCE_CLOCK_OR_SCHEMA")
    league_year=(payload.get("season") or {}).get("year")
    if str(league_year)!=str(season):
        return _empty(received,"WRONG_OR_MISSING_SEASON")
    sections=payload.get("injuries")
    if not isinstance(sections,list):
        return _empty(received,"NO_LEAGUE_INJURY_SECTIONS")
    records=[]
    for group in sections:
        if not isinstance(group,dict):continue
        team=group.get("team") or {}
        abbr=TEAM_ALIAS.get(str(team.get("abbreviation") or "").upper(),
                            str(team.get("abbreviation") or "").upper())
        if not re.fullmatch(r"[A-Z]{2,3}",abbr):
            continue
        entries=group.get("injuries")
        if not isinstance(entries,list):continue
        for row in entries:
            if not isinstance(row,dict):continue
            athlete=row.get("athlete") or {}
            if not isinstance(athlete,dict):continue
            name=athlete.get("displayName") or athlete.get("fullName")
            key=normalized(name)
            if not key:continue
            stat=row.get("status")
            if not isinstance(stat,str) or not stat.strip():continue
            status=stat.strip().casefold()
            report_time=_time(row.get("date"))
            # Provider's date can refer to an ancient injury even though
            # the league response was fetched today. Preserve it as a
            # diagnostic but never hard-exclude from stale/unknown time.
            fresh=(bool(report_time) and
                   received-timedelta(days=FRESH_DAYS)<=report_time<=received)
            identity=str(athlete.get("id") or "").strip()
            records.append({
                "team":abbr,"player":str(name),"normalized_player":key,
                "espn_athlete_id":identity or None,
                "status":status,
                "report_date_utc":report_time.isoformat() if report_time else None,
                "fresh_asof":fresh,
                "source":URL,
            })
    if not records:
        return {
            "schema":SOURCE_SCHEMA,
            "status":"EMPTY_SOURCE_NO_INJURY_REPORTS",
            "checked_at_utc":received.isoformat(),
            "source":URL,"season":season,"season_verified":True,
            "records":[],"raw_source_records":0,
            "reason":"Current ESPN league injury endpoint returned no usable player injury entries; never infer everyone healthy."
        }
    return {
        "schema":SOURCE_SCHEMA,"status":"CAPTURED",
        "checked_at_utc":received.isoformat(),
        "requested_at_utc":now.isoformat(),
        "source":URL,"season":season,
        "season_verified":True,
        "records":records,
        "raw_source_records":len(records),
        "caution_statuses_are_not_inactives":True,
        "provider_report_time_can_be_missing":True,
    }

def classify_for_pick(snapshot:dict,team:str,player:str,now:datetime,kickoff:datetime):
    """One player verdict. No negative inference if source/time/identity weak."""
    if now.tzinfo is None or kickoff.tzinfo is None:
        raise ValueError("NFL_INJURY_CLOCK_NAIVE")
    now=now.astimezone(timezone.utc)
    kickoff=kickoff.astimezone(timezone.utc)
    if now>=kickoff:
        return {"status":"GAME_STARTED","hard_out":False}
    if snapshot.get("schema")!=SOURCE_SCHEMA or snapshot.get("status")!="CAPTURED" or not snapshot.get("season_verified"):
        return {"status":"SOURCE_UNAVAILABLE","hard_out":False}
    check=_time(snapshot.get("checked_at_utc"))
    if check is None or check>now or check>=kickoff or (now-check).total_seconds()>1800:
        return {"status":"STALE_SOURCE_CAPTURE","hard_out":False}
    key=normalized(player)
    rows=[x for x in snapshot.get("records",[])
          if x.get("team")==team and x.get("normalized_player")==key
          and x.get("fresh_asof")]
    if not rows:
        return {"status":"NO_CURRENT_REPORT_NOT_CLEARED","hard_out":False}
    names={x.get("espn_athlete_id") for x in rows}
    if None in names or len(names)!=1:
        return {"status":"AMBIGUOUS_ATHLETE_ID","hard_out":False}
    with_times=[(ts,x) for x in rows
                if (ts:=_time(x.get("report_date_utc"))) is not None and ts<=now]
    if not with_times:
        return {"status":"REPORT_TIME_UNVERIFIED","hard_out":False}
    last=max(ts for ts,_ in with_times)
    statuses={x.get("status") for ts,x in with_times if ts==last}
    if len(statuses)!=1:
        return {"status":"CONFLICTING_LATEST_REPORTS","hard_out":False}
    status=next(iter(statuses))
    common={
        "source":URL,"report_date_utc":last.isoformat(),
        "espn_athlete_id":next(iter(names)),
        "reported_status":status,
    }
    if status in OUT:
        return {"status":"RECENT_VERIFIED_REPORTED_OUT","hard_out":True,**common}
    if status in CAUTION:
        return {"status":"INJURY_PARTICIPATION_UNCERTAIN","hard_out":False,**common}
    return {"status":"UNVERIFIED_OR_CLEARED_REPORT","hard_out":False,**common}

def apply_to_serving_picks(picks:list,snapshot:dict,scheduled_games:list,now:datetime):
    """Suppression applies only prekickoff. Preserve all unsuppressed medians."""
    games={}
    for g in scheduled_games:
        home,away=g["home_team"],g["away_team"]
        t=_time(g["kickoff_utc"])
        if t is None:raise ValueError("NFL_INJURY_SCHEDULE_KICKOFF_INVALID")
        for t1,t2 in ((home,away),(away,home)):
            if (t1,t2) in games:
                raise ValueError("NFL_INJURY_DUPLICATE_SCHEDULE_PAIR")
            games[(t1,t2)]=t
    result=[];evidence=[];removed=[]
    for p in picks:
        pair=(p.get("team"),p.get("opponent"))
        kick=games.get(pair)
        if kick is None:
            raise ValueError("NFL_INJURY_PICK_WITHOUT_SCHEDULED_GAME")
        rec=classify_for_pick(snapshot,pair[0],p.get("player"),now,kick)
        if rec["hard_out"]:
            removed.append({
                "player":p.get("player"),"team":pair[0],
                "opponent":pair[1],"market":p.get("market"),
                "reported_status":rec.get("reported_status"),
                "report_date_utc":rec.get("report_date_utc"),
            })
            continue
        # Preserve original numbers; optional metadata only.
        if rec["status"]=="INJURY_PARTICIPATION_UNCERTAIN":
            p={**p,"game_day_availability_uncertain":True,
               "game_day_availability_source":"ESPN league injury report",
               "game_day_availability_reported_status":rec["reported_status"]}
            evidence.append({
                "player":p.get("player"),"team":pair[0],"market":p.get("market"),
                "report_date_utc":rec.get("report_date_utc"),
                "status":rec.get("reported_status"),
            })
        result.append(p)
    return result,{
        "schema":SOURCE_SCHEMA,"status":snapshot.get("status"),
        "checked_at_utc":snapshot.get("checked_at_utc"),
        "source":URL,"season_verified":snapshot.get("season_verified",False),
        "source_records":snapshot.get("raw_source_records",0),
        "confirmed_out_suppressed":len(removed),
        "uncertain_participation_picks":len(evidence),
        "suppressed":removed,
        "uncertainties":evidence,
        "no_source_report_is_not_active_certification":True,
        "does_not_modify_model_medians_or_probabilities":True,
    }
