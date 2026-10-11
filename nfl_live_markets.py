"""Read-only live NFL player market quotes. Never manufactures an executable bet.

FanDuel only; require matching NFL event, player, BOTH prices at identical
main-line point and provider-issued recent update timestamps. The Odds API v4
event endpoint is used only after one specific game is selected by the viewer.
Cache by event for 15 minutes to conserve limited quota; no background sweep.
"""
from datetime import datetime, timezone, timedelta
from threading import Lock
import math
import requests

TEAM_NAMES={
    "ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens",
    "BUF":"Buffalo Bills","CAR":"Carolina Panthers","CHI":"Chicago Bears",
    "CIN":"Cincinnati Bengals","CLE":"Cleveland Browns","DAL":"Dallas Cowboys",
    "DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers",
    "HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars",
    "KC":"Kansas City Chiefs","LV":"Las Vegas Raiders","LAC":"Los Angeles Chargers",
    "LAR":"Los Angeles Rams","MIA":"Miami Dolphins","MIN":"Minnesota Vikings",
    "NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants",
    "NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers",
    "SF":"San Francisco 49ers","SEA":"Seattle Seahawks","TB":"Tampa Bay Buccaneers",
    "TEN":"Tennessee Titans","WAS":"Washington Commanders"}
MARKETS={"player_rush_yds":"rushing_yards","player_reception_yds":"receiving_yards"}
BASE="https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
TTL=900
MAX_MARKET_AGE=1800
CACHE={}
GUARD=Lock()


def utc(value):
    if not isinstance(value,str) or not value.strip():
        raise ValueError("timestamp missing")
    result=datetime.fromisoformat(value.replace("Z","+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone missing")
    return result.astimezone(timezone.utc)


def american(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and (value>=100 or value<=-100) and abs(value)<=20000


def parse_event(payload, *, home, away, kickoff, clock):
    """Fail closed if market names, player lines, sides or timestamps ambiguous."""
    current=clock.astimezone(timezone.utc)
    if (payload.get("home_team")!=TEAM_NAMES[home] or
        payload.get("away_team")!=TEAM_NAMES[away] or
        abs((utc(payload.get("commence_time"))-kickoff).total_seconds())>300):
        raise ValueError("event_not_confirmed")
    if kickoff<=current:
        raise ValueError("game_already_started")
    by_key={}
    for bookmaker in payload.get("bookmakers") or []:
        if bookmaker.get("key")!="fanduel":
            continue
        try:
            if not 0<=(current-utc(bookmaker.get("last_update"))).total_seconds()<=MAX_MARKET_AGE:
                continue
        except (TypeError,ValueError):
            continue
        for market in bookmaker.get("markets") or []:
            m=MARKETS.get(market.get("key"))
            if not m:
                continue
            try:
                age=(current-utc(market.get("last_update"))).total_seconds()
                if age<0 or age>MAX_MARKET_AGE:
                    continue
            except (TypeError,ValueError):
                continue
            for outcome in market.get("outcomes") or []:
                player=outcome.get("description")
                side=str(outcome.get("name") or "").lower()
                point=outcome.get("point")
                price=outcome.get("price")
                if (not isinstance(player,str) or len(player.strip())<3 or side not in ("over","under")
                    or not isinstance(point,(int,float)) or not math.isfinite(point) or point<0
                    or not american(price)):
                    continue
                key=(player.casefold().strip(),m)
                rows=by_key.setdefault(key,[])
                rows.append((side,float(point),int(price),player.strip(),market["last_update"]))
    lines=[]
    for (normal,market),records in by_key.items():
        # No fallback to another book, no unpaired sides, no joining mismatched
        # alternate points, and no choice by best price across multiple lines.
        overs=[x for x in records if x[0]=="over"]
        unders=[x for x in records if x[0]=="under"]
        if len(overs)!=1 or len(unders)!=1 or overs[0][1]!=unders[0][1]:
            continue
        a,b=overs[0],unders[0]
        lines.append({"player":a[3],"market":market,"line":a[1],
             "over_price":a[2],"under_price":b[2],"book":"fanduel",
             "market_updated_at_utc":a[4],"home":home,"away":away,
             "kickoff_utc":kickoff.isoformat().replace("+00:00","Z")})
    return sorted(lines,key=lambda x:(x["market"],x["player"].lower()))


def collect(home,away,kickoff_utc,api_key,now=None,http_get=requests.get):
    current=now or datetime.now(timezone.utc)
    start=utc(kickoff_utc)
    if home not in TEAM_NAMES or away not in TEAM_NAMES or home==away or start<=current:
        return {"status":"NOT_PREGAME","lines":[],"checked_at_utc":current.isoformat()}
    if not api_key:
        return {"status":"ODDS_KEY_NOT_CONFIGURED","lines":[],"checked_at_utc":current.isoformat()}
    key=(home,away,start.isoformat())
    # Lock includes network fetch: parallel callers cannot multiply provider
    # quota usage. Does not store API keys or leak token-bearing exceptions.
    with GUARD:
        existing=CACHE.get(key)
        if existing and (current-existing[0]).total_seconds()<TTL:
            return existing[1]
        def empty(status):
            result={"status":status,"lines":[],"checked_at_utc":current.isoformat()}
            CACHE[key]=(current,result)
            return result
        try:
            resp=http_get(BASE+"/events",params={"apiKey":api_key},timeout=12)
            if resp.status_code!=200:
                return empty("ODDS_PROVIDER_UNAVAILABLE")
            matches=[x for x in resp.json() if x.get("home_team")==TEAM_NAMES[home]
               and x.get("away_team")==TEAM_NAMES[away] and
               abs((utc(x.get("commence_time"))-start).total_seconds())<=300]
            if len(matches)!=1 or not matches[0].get("id"):
                return empty("EVENT_NOT_FOUND")
            url=BASE+"/events/"+str(matches[0]["id"])+"/odds"
            quotes=http_get(url,params={"apiKey":api_key,"regions":"us",
                "bookmakers":"fanduel","markets":",".join(MARKETS),"oddsFormat":"american"},timeout=12)
            if quotes.status_code!=200:
                return empty("ODDS_PROVIDER_UNAVAILABLE")
            lines=parse_event(quotes.json(),home=home,away=away,kickoff=start,clock=current)
            result={"status":"VERIFIED_QUOTES" if lines else "NO_VERIFIED_PLAYER_MARKETS",
                    "book":"fanduel","lines":lines,"checked_at_utc":current.isoformat()}
            CACHE[key]=(current,result)
            return result
        except (requests.RequestException,ValueError,TypeError,KeyError):
            return empty("ODDS_PROVIDER_UNAVAILABLE")
