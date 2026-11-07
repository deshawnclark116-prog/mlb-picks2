"""Daily-cached Sleeper NFL injury evidence (once/day provider request).

Source: https://api.sleeper.app/v1/players/nfl
Sleeper's documentation says the large player map is not intended to be
polled more than ONCE PER DAY. GitHub's single daily capture workflow caches
only small source-timestamped injury metadata in docs/. The 8x/day serving
workflow READS the cached file; it never re-fetches Sleeper.

A Sleeper injury_status such as Out describes a current platform designation,
NOT a verified NFL game-day inactive list, starting QB, or official game report.
The source does NOT give a reliable injury-status issuance timestamp; do not
substitute injury_start_date or news_updated for that timestamp.
A high-confidence, current, ID-matched OUT/IR report may suppress featured
model picks, but the original projection is never altered or backfilled.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime,timedelta,timezone
from pathlib import Path
from typing import Callable

import requests

URL="https://api.sleeper.app/v1/players/nfl"
SCHEMA="NFL_SLEEPER_DAILY_INJURY_SNAPSHOT_V1"
INJURY_OUT={"out","ir","injured reserve","pup","suspended","suspension"}
INJURY_CAUTION={"doubtful","questionable","limited","day-to-day","day to day"}
TEAM_MAP={"LAR":"LA","WSH":"WAS"}
TTL_HOURS=20

def _t(value):
    if not isinstance(value,str) or not value:return None
    try:
        t=datetime.fromisoformat(value.replace("Z","+00:00"))
        return t.astimezone(timezone.utc) if t.tzinfo else None
    except ValueError:return None

def _norm(value):
    s=unicodedata.normalize("NFKD",str(value or ""))
    s="".join(c for c in s if not unicodedata.combining(c))
    s=re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b","",s.casefold())
    return " ".join(re.sub("[^a-z0-9 ]"," ",s).split())

def read(path:Path):
    try:
        d=json.loads(Path(path).read_text())
    except (OSError,ValueError,TypeError):
        return {"schema":SCHEMA,"status":"SOURCE_NOT_CACHED","players":[]}
    if (d.get("schema")!=SCHEMA or
        not isinstance(d.get("players"),list) or
        d.get("status")!="CAPTURED"):
        return {"schema":SCHEMA,"status":"CACHE_SCHEMA_INVALID","players":[]}
    return d

def capture_once_per_day(cache_path:Path, now:datetime, *,
                         http_get:Callable=requests.get, clock:Callable=lambda:datetime.now(timezone.utc)):
    if now.tzinfo is None:raise ValueError("NFL_SLEEPER_NAIVE_CAPTURE_TIME")
    now=now.astimezone(timezone.utc)
    old=read(cache_path)
    prev=_t(old.get("captured_at_utc"))
    if (old.get("status")=="CAPTURED" and prev is not None and
        prev.date()==now.date()):
        # One request per UTC date, including subsequent manual dispatch.
        return old,False
    response=http_get(URL,timeout=28,headers={
        "User-Agent":"PropEdge-NFL-Injury-Provenance/1.0",
        "Accept":"application/json"})
    response.raise_for_status()
    # Do not retain the full 5MB+ raw roster file in the Git repository.
    # Insist on a plausible complete dictionary rather than an empty
    # 200 response or a partial/filtered source.
    if len(response.content)>22_000_000:
        raise ValueError("NFL_SLEEPER_FULL_MAP_UNEXPECTEDLY_LARGE")
    data=response.json()
    receipt=clock().astimezone(timezone.utc)
    if receipt<now or not isinstance(data,dict) or len(data)<1500:
        raise ValueError("NFL_SLEEPER_UNVERIFIABLE_SOURCE_OR_CLOCK")
    players=[]; statuses=Counter()
    for sleeper_id,r in data.items():
        if not isinstance(r,dict):continue
        st=str(r.get("injury_status") or "").strip().casefold()
        if not st:continue
        statuses[st]+=1
        team=str(r.get("team") or r.get("team_abbr") or "").upper()
        team=TEAM_MAP.get(team,team)
        name=r.get("full_name") or " ".join(str(r.get(k) or "") for k in ("first_name","last_name")).strip()
        if not re.fullmatch("[A-Z]{2,3}",team) or not _norm(name):continue
        gsis=str(r.get("gsis_id") or "").strip()
        espn=str(r.get("espn_id") or "").strip()
        players.append({
            "sleeper_id":str(sleeper_id),
            "gsis_id":gsis or None,"espn_id":espn or None,
            "team":team,"player":name,"norm_player":_norm(name),
            "injury_status":st,"player_status":str(r.get("status") or ""),
            "injury_start_date":r.get("injury_start_date"),
        })
    if len(players)<25:
        raise ValueError("NFL_SLEEPER_TOO_FEW_INJURY_ROWS")
    return {
        "schema":SCHEMA,"status":"CAPTURED",
        "captured_at_utc":receipt.isoformat(),
        "requested_at_utc":now.isoformat(),
        "source":URL,
        "source_issue_time_verified":False,
        "injury_start_date_is_not_status_report_time":True,
        "source_player_count":len(data),
        "injured_player_count":len(players),
        "injury_status_counts":dict(statuses),
        "provider_license":"Sleeper API public free non-commercial; obtain licensing for commercial deployment.",
        "players":sorted(players,key=lambda p:(p["team"],p["norm_player"],p["sleeper_id"])),
    },True

def verdict(snapshot:dict, pick:dict, now:datetime, kickoff:datetime):
    if now.tzinfo is None or kickoff.tzinfo is None:
        raise ValueError("NFL_SLEEPER_NAIVE_ASOF_KICKOFF")
    now=now.astimezone(timezone.utc)
    kickoff=kickoff.astimezone(timezone.utc)
    if now>=kickoff:return {"status":"GAME_ALREADY_STARTED","suppress":False}
    if snapshot.get("status")!="CAPTURED" or snapshot.get("schema")!=SCHEMA:
        return {"status":"SOURCE_NOT_CACHED","suppress":False}
    seen=_t(snapshot.get("captured_at_utc"))
    if seen is None or seen>now or seen>=kickoff or now-seen>timedelta(hours=TTL_HOURS):
        return {"status":"CACHED_SOURCE_STALE_OR_AFTER_KICKOFF","suppress":False}
    key=_norm(pick.get("player"))
    group=[p for p in snapshot.get("players",[])
           if p.get("norm_player")==key and p.get("team")==pick.get("team")]
    if not group:return {"status":"NO_REPORTED_INJURY_NOT_CLEARED","suppress":False}
    # Require ONE mapped physical player. Prefer GSIS ID where available,
    # preventing a same-name/other-team error and a name collision.
    ids={p.get("sleeper_id") for p in group}
    if len(ids)!=1 or None in ids:
        return {"status":"INJURY_IDENTITY_AMBIGUOUS","suppress":False}
    row=group[0]
    if not row.get("gsis_id") and not row.get("espn_id"):
        return {"status":"INJURY_PERSON_ID_UNVERIFIED","suppress":False}
    nfl_id=str(pick.get("player_id") or "")
    if row.get("gsis_id") and nfl_id.startswith("00-") and row["gsis_id"]!=nfl_id:
        return {"status":"GSIS_PLAYER_ID_CONFLICT","suppress":False}
    status=row["injury_status"]
    base={"reported_status":status,"source":URL,
          "captured_at_utc":seen.isoformat(),
          "sleeper_id":row["sleeper_id"]}
    if status in INJURY_OUT:
        return {"status":"CURRENT_OUT_OR_IR_PLATFORM_DESIGNATION","suppress":True,**base}
    if status in INJURY_CAUTION:
        return {"status":"INJURY_PARTICIPATION_UNCERTAIN","suppress":False,**base}
    return {"status":"OTHER_INJURY_DESIGNATION_UNVERIFIED","suppress":False,**base}

def apply(picks:list,snapshot:dict,scheduled_games:list,now:datetime):
    lookup={}
    for g in scheduled_games:
        tip=_t(g.get("kickoff_utc"))
        if tip is None:raise ValueError("NFL_SLEEPER_SCHEDULE_HAS_NO_VALID_TIME")
        for pair in ((g["home_team"],g["away_team"]),
                     (g["away_team"],g["home_team"])):
            if pair in lookup:raise ValueError("NFL_SLEEPER_DUPLICATE_GAME_PAIR")
            lookup[pair]=tip
    retained=[];suppressed=[];risks=[]
    for p in picks:
        pair=(p.get("team"),p.get("opponent"))
        if pair not in lookup:raise ValueError("NFL_SLEEPER_UNSCHEDULED_PLAYER_PICK")
        status=verdict(snapshot,p,now,lookup[pair])
        if status["suppress"]:
            suppressed.append({"player":p.get("player"),"team":p.get("team"),
                "market":p.get("market"),"injury_status":status.get("reported_status"),
                "snapshot_captured_at_utc":status.get("captured_at_utc")})
            continue
        if status["status"]=="INJURY_PARTICIPATION_UNCERTAIN":
            p={**p,"game_day_availability_uncertain":True,
                   "game_day_availability_source":"Sleeper current player injury snapshot",
                   "game_day_availability_reported_status":status["reported_status"]}
            risks.append({"player":p.get("player"),"team":p.get("team"),
                          "market":p.get("market"),"status":status["reported_status"]})
        retained.append(p)
    audit={
        "schema":SCHEMA,"source_status":snapshot.get("status"),
        "source":URL,"captured_at_utc":snapshot.get("captured_at_utc"),
        "source_player_count":snapshot.get("source_player_count"),
        "injured_player_count":snapshot.get("injured_player_count"),
        "suppressed_current_out_or_ir":len(suppressed),
        "uncertain_participation_picks":len(risks),
        "suppressed":suppressed,"uncertainties":risks,
        "source_does_not_verify_official_inactives":True,
        "source_status_reported_at_unknown":True,
        "no_numerical_model_change":True
    }
    return retained,audit

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="docs/nfl_sleeper_injury_snapshot.json")
    a=p.parse_args()
    path=Path(a.out)
    data,new=capture_once_per_day(path,datetime.now(timezone.utc))
    if new:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
    print("NFL_SLEEPER_DAILY_INJURY_CAPTURE",json.dumps({
        "new_fetch":new,"captured_at_utc":data.get("captured_at_utc"),
        "source_players":data.get("source_player_count"),
        "injured_players":data.get("injured_player_count"),
        "status_counts":data.get("injury_status_counts"),
    },sort_keys=True))
if __name__=="__main__":main()
