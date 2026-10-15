#!/usr/bin/env python3
"""NHL V2 Odds-API.io FanDuel main-SOG source feasibility: capped, fail-closed.

Provider is the separate Odds-API.io (/v3), NOT The Odds API (/v4).
This checks actual future events with a frozen already-published SOG forecast.
It does not assume Player Props labels, alternate-line eligibility, dressed
players, market value, licensed redistribution or official betting picks.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from nhl_v2_live_fanduel_sog_probe import future_model_games
from nhl_sog_fanduel_mainline_reference import NAME_TO_TEAM, namekey, timestamp

SCHEMA="NHL_V2_ODDS_API_IO_FANDUEL_SOG_SOURCE_FEASIBILITY_V1"
BASE="https://api.odds-api.io/v3"
SHOT_SUFFIX=re.compile(r"^(.+?)\s*\((?:Total )?Shots on Goal\)$",re.I)
SHOT_DASH=re.compile(r"^(.+?)\s+-\s+(?:Total )?Shots on Goal$",re.I)
MAX_PAID_EVENTS=2


class ProviderUnavailable(RuntimeError):
    pass


def request(path, args):
    url=BASE+path+"?"+urllib.parse.urlencode(args)
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={
            "User-Agent":"nhl-v2-verified-source-audit","Accept":"application/json"
        }),timeout=16) as response:
            return json.loads(response.read(2_000_000))
    except urllib.error.HTTPError as exc:
        # The URL embeds an API key; never print the exception, response body
        # or full URL into public logs.
        raise ProviderUnavailable("HTTP_"+str(exc.code)) from None
    except (urllib.error.URLError,TimeoutError):
        raise ProviderUnavailable("TRANSPORT_UNAVAILABLE") from None
    except (UnicodeError,ValueError):
        raise ProviderUnavailable("INVALID_JSON") from None


def _real_event_identity(event):
    if not isinstance(event,dict):
        return None
    home=event.get("home") or event.get("homeTeam")
    away=event.get("away") or event.get("awayTeam")
    pair={NAME_TO_TEAM.get(str(x or "").casefold()) for x in (home,away)}
    if None in pair or len(pair)!=2:
        return None
    try:
        date=timestamp(event.get("date") or event.get("startTime"))
    except (TypeError,ValueError):
        return None
    return (frozenset(pair),date)


def _shot_mainline_candidates(event_odds):
    books=event_odds.get("bookmakers") if isinstance(event_odds,dict) else None
    if not isinstance(books,dict):
        return {},{"NO_FANDUEL_BOOKMAKER":1}
    fd=[v for k,v in books.items() if k.casefold()=="fanduel"]
    if len(fd)!=1 or not isinstance(fd[0],list):
        return {},{"NO_UNIQUE_FANDUEL_BOOKMAKER":1}
    found=defaultdict(list)
    reasons=Counter()
    for market in fd[0]:
        if not isinstance(market,dict) or market.get("name")!="Player Props":
            continue
        for outcome in market.get("odds",[]) or []:
            if not isinstance(outcome,dict):
                continue
            label=outcome.get("label")
            if not isinstance(label,str):
                continue
            match=SHOT_SUFFIX.fullmatch(label.strip()) or SHOT_DASH.fullmatch(label.strip())
            if not match:
                continue
            player=namekey(match.group(1))
            line=outcome.get("hdp")
            if not isinstance(line,(float,int)) or isinstance(line,bool) or line<0 or line>10.5 or line%1!=.5:
                reasons["MISSING_OR_UNVERIFIED_MAIN_LINE"]+=1
                continue
            over=outcome.get("over")
            under=outcome.get("under")
            try:
                if not 1<float(over)<50 or not 1<float(under)<50:
                    raise ValueError("invalid quoted decimal price")
            except (ValueError,TypeError):
                reasons["MISSING_PAIRED_DECIMAL_PRICES"]+=1
                continue
            found[player].append({"player":player,"line":line,"over_decimal":str(over),
                                  "under_decimal":str(under)})
    result={}
    for player,items in found.items():
        if len(items)!=1:
            reasons["MULTIPLE_LINES_OR_ALTERNATES_QUARANTINED"]+=1
        else:
            result[player]=items[0]
    return result,dict(reasons)


def coverage(board, *, key, now, fetch=request, max_events=MAX_PAID_EVENTS):
    if now.tzinfo is None or not 1<=max_events<=MAX_PAID_EVENTS:
        raise ValueError("invalid capture clock or bounded event cap")
    games=future_model_games(board,now)
    result={"schema":SCHEMA,"source":"odds-api.io","bookmaker":"FanDuel",
            "observed_at_utc":now.astimezone(timezone.utc).isoformat(),
            "status":"UNKNOWN","model_games":len(games),"matched_provider_games":0,
            "paid_event_requests_attempted":0,"verified_main_shot_lines":0,
            "matched_model_players_with_real_line":0,"rejections":{},
            "lineup_certified":False,"official_picks":0,
            "individual_quotes_published":False,
            "provider_market_last_updated_certified":False}
    if not key:
        result["status"]="PROVIDER_KEY_NOT_AVAILABLE_IN_GITHUB_SECRETS"
        return result
    if not games:
        result["status"]="NO_VERIFIED_FUTURE_MODEL_GAMES"
        return result
    try:
        events=fetch("/events",{"apiKey":key,"sport":"nhl"})
    except ProviderUnavailable as exc:
        result["status"]="EVENT_DISCOVERY_"+str(exc)
        return result
    if not isinstance(events,list):
        result["status"]="INVALID_EVENTS_PAYLOAD"
        return result
    chosen=[]
    for ev in events:
        ident=_real_event_identity(ev)
        if ident is None:
            continue
        for k,rr in games.items():
            if frozenset([rr[0]["team"],rr[0]["opponent"]])==ident[0] and abs(
                (timestamp(k[1])-ident[1]).total_seconds())<=600:
                if sum(1 for k2,rows in games.items()
                       if frozenset([rows[0]["team"],rows[0]["opponent"]])==ident[0]
                       and abs((timestamp(k2[1])-ident[1]).total_seconds())<=600)!=1:
                    continue
                chosen.append((k,ev,rr))
                break
    chosen.sort(key=lambda x:timestamp(x[0][1]))
    reasons=Counter()
    seen=set()
    for k,event,rr in chosen:
        if len(seen)>=max_events:
            break
        if k in seen:
            continue
        seen.add(k)
        result["paid_event_requests_attempted"]+=1
        try:
            odds=fetch("/odds",{"apiKey":key,"eventId":event["id"],
                                "bookmakers":"FanDuel"})
        except ProviderUnavailable as exc:
            reasons["ODDS_"+str(exc)]+=1
            if str(exc).startswith(("HTTP_401","HTTP_403","HTTP_429")):
                result["status"]="SOURCE_BLOCKED_"+str(exc)
                break
            continue
        result["matched_provider_games"]+=1
        lines,reject=_shot_mainline_candidates(odds)
        reasons.update(reject)
        result["verified_main_shot_lines"]+=len(lines)
        # Match to one exact player ID only. We do not infer game-day dressing.
        unique=defaultdict(set)
        for r in rr:
            unique[namekey(r.get("player"))].add((str(r["player_id"]),r["team"]))
        result["matched_model_players_with_real_line"]+=sum(
            len(identities)==1 for player,identities in unique.items() if player in lines)
    result["rejections"]=dict(sorted(reasons.items()))
    if result["status"]=="UNKNOWN":
        result["status"]=("REAL_FANDUEL_SOG_MAINLINES_OBSERVED_RESEARCH_ONLY"
                          if result["matched_model_players_with_real_line"]
                          else "NO_QUALIFIED_NHL_SHOT_LINES_VERIFIED")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--board",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--max-events",type=int,default=MAX_PAID_EVENTS)
    a=p.parse_args()
    board=json.loads(a.board.read_text())
    result=coverage(board,key=os.getenv("ODDS_API_IO_KEY"),
                    now=datetime.now(timezone.utc),max_events=a.max_events)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))
    return 0 if result["status"] in (
        "REAL_FANDUEL_SOG_MAINLINES_OBSERVED_RESEARCH_ONLY",
        "NO_VERIFIED_FUTURE_MODEL_GAMES","NO_QUALIFIED_NHL_SHOT_LINES_VERIFIED",
        "PROVIDER_KEY_NOT_AVAILABLE_IN_GITHUB_SECRETS") else 1


if __name__=="__main__":
    sys.exit(main())
