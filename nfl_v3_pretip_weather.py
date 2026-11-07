"""NFL V3 free pretip weather capture (research evidence, NOT a projection modifier).

Sources: ESPN public NFL scoreboard for REAL scheduled game/venue city/state;
Open-Meteo public geocoding + free hourly forecast (no API key). Venue/city
centre is a geocoding approximation, not an exact stadium sensor. Weather
retrieval time is AFTER full response bytes. Missing/ambiguous venue, roof
status or provider data are explicit unknowns, not fabricated dry conditions.

Weather variables are captured as a separate, immutable-in-time proof object;
the existing certified Phase1C player forecasts are NEVER overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime,timedelta,timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SCHEMA="NFL_V3_PRETIP_WEATHER_SOURCE_RECEIPT_V1"
ZONE=ZoneInfo("America/New_York")
TEAM_ALIASES={"WSH":"WAS","LAR":"LA","LARams":"LA"}
STATES={
 "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
 "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia",
 "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas",
 "KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts",
 "MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana",
 "NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico",
 "NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma",
 "OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina",
 "SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont",
 "VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"
}
FIELDS=("temperature_2m","wind_speed_10m","wind_gusts_10m",
        "precipitation","precipitation_probability")


def as_utc(value):
    ts=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if ts.tzinfo is None:raise ValueError("NFL_V3_WEATHER_NAIVE_CLOCK")
    return ts.astimezone(timezone.utc)


def json_get(url,timeout=15):
    request=urllib.request.Request(url,headers={
        "User-Agent":"prop-edge-nfl-v3-weather-research/1.0",
        "Accept":"application/json"})
    with urllib.request.urlopen(request,timeout=timeout) as response:
        if not response.geturl().startswith("https://"):
            raise RuntimeError("NFL_V3_WEATHER_PROVIDER_REDIRECT_NOT_HTTPS")
        raw=response.read()
    if len(raw)>6_000_000:raise RuntimeError("NFL_V3_WEATHER_OVERSIZED_PROVIDER_RESPONSE")
    return json.loads(raw)


def game_venue(event,home,away,kickoff):
    competitions=event.get("competitions") or []
    if len(competitions)!=1:return None
    comp=competitions[0]
    participants=comp.get("competitors") or []
    if len(participants)!=2:return None
    teams={}
    for p in participants:
        alias=TEAM_ALIASES.get(str(p.get("team",{}).get("abbreviation") or ""),str(p.get("team",{}).get("abbreviation") or ""))
        teams[p.get("homeAway")]=alias
    if teams!={"home":home,"away":away}:return None
    if abs((as_utc(comp["date"])-kickoff).total_seconds())>600:return None
    venue=comp.get("venue") or {}
    address=venue.get("address") or {}
    city=str(address.get("city") or "").strip()
    state=str(address.get("state") or "").strip()
    if not city or not state or state not in STATES:return None
    return {
        "provider_event_id":str(event.get("id") or ""),
        "name":str(venue.get("fullName") or "UNKNOWN_VENUE"),
        "city":city,"state":state,
        "is_indoor_venue":venue.get("indoor") if isinstance(venue.get("indoor"),bool) else None,
        "roof_status":"NOT_VERIFIED_BY_VENUE_TYPE"
    }


def place_coordinates(payload,city,state):
    target=STATES.get(state)
    if not target or not isinstance(payload.get("results"),list):
        raise ValueError("NFL_V3_WEATHER_GEOCODE_SOURCE_UNAVAILABLE")
    matches=[p for p in payload["results"] if
             str(p.get("country_code","")).upper()=="US" and
             str(p.get("name","")).casefold()==city.casefold() and
             str(p.get("admin1","")).casefold()==target.casefold()]
    if len(matches)!=1:
        raise ValueError("NFL_V3_WEATHER_GEOCODE_NOT_UNIQUE_CITY_STATE")
    pt=matches[0];lat=pt.get("latitude");lon=pt.get("longitude")
    if not all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in (lat,lon)):
        raise ValueError("NFL_V3_WEATHER_COORDINATES_MISSING")
    if not 24<=lat<=50 or not -127<=lon<=-66:
        raise ValueError("NFL_V3_WEATHER_COORDINATES_OUTSIDE_CONUS")
    return float(lat),float(lon)


def extract_hourly_forecast(doc,kickoff):
    # Explicit UTC timezone prevents daylight-saving offsets shifting games.
    if doc.get("utc_offset_seconds")!=0 or doc.get("timezone") not in ("GMT","UTC"):
        raise ValueError("NFL_V3_WEATHER_PROVIDER_TIMEZONE_NOT_UTC")
    hours=doc.get("hourly") or {}
    timestamps=hours.get("time") or []
    if not isinstance(timestamps,list) or not timestamps:
        raise ValueError("NFL_V3_WEATHER_HOURLY_TIMES_MISSING")
    for key in FIELDS:
        if not isinstance(hours.get(key),list) or len(hours[key])!=len(timestamps):
            raise ValueError("NFL_V3_WEATHER_HOURLY_FIELD_MISSING")
    selected=[]
    for j,day_hour in enumerate(timestamps):
        t=datetime.fromisoformat(day_hour).replace(tzinfo=timezone.utc)
        if kickoff-timedelta(minutes=31)<=t<=kickoff+timedelta(hours=3):
            vals={k:hours[k][j] for k in FIELDS}
            if any(not isinstance(v,(int,float)) or isinstance(v,bool) for v in vals.values()):
                raise ValueError("NFL_V3_WEATHER_NON_NUMERIC_HOURLY_DATA")
            if not 0<=vals["precipitation_probability"]<=100 or min(vals["precipitation"],vals["wind_speed_10m"],vals["wind_gusts_10m"])<0:
                raise ValueError("NFL_V3_WEATHER_INVALID_HOURLY_VALUE")
            selected.append({"valid_time_utc":t.isoformat(),**vals})
    if len(selected)<3:
        raise ValueError("NFL_V3_WEATHER_KICKOFF_HOURLY_COVERAGE_MISSING")
    return selected


def capture(schedule,now=None,fetch=json_get):
    observed=as_utc(now or datetime.now(timezone.utc))
    dates=sorted({as_utc(g["kickoff_utc"]).astimezone(ZONE).strftime("%Y%m%d")
                  for g in schedule if as_utc(g["kickoff_utc"])>observed})
    boards={}
    for day in dates:
        url="https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates="+day
        try:boards[day]=fetch(url).get("events",[])
        except (ValueError,KeyError,RuntimeError,TimeoutError,OSError) as err:
            boards[day]=None
    geocodes={}
    rows=[]
    for g in schedule:
        kickoff=as_utc(g["kickoff_utc"])
        if kickoff<=observed:continue
        home,away=g["home_team"],g["away_team"]
        date=kickoff.astimezone(ZONE).strftime("%Y%m%d")
        rec={"home":home,"away":away,"kickoff_utc":kickoff.isoformat(),
             "research_only":True,"model_coefficients_applied":False,
             "source":"ESPN scheduled venue; Open-Meteo city-centre hourly forecast"}
        try:
            if boards.get(date) is None:
                raise ValueError("NFL_V3_WEATHER_ESPN_SCHEDULE_UNAVAILABLE")
            possibles=[v for event in boards[date] if
                       (v:=game_venue(event,home,away,kickoff)) is not None]
            if len(possibles)!=1:
                raise ValueError("NFL_V3_WEATHER_EXACT_GAME_VENUE_NOT_FOUND")
            venue=possibles[0]
            rec["venue"]=venue
            key=(venue["city"],venue["state"])
            if key not in geocodes:
                url="https://geocoding-api.open-meteo.com/v1/search?"+urllib.parse.urlencode({
                    "name":venue["city"],"count":20,"language":"en","format":"json","countryCode":"US"})
                geocodes[key]=place_coordinates(fetch(url),*key)
            lat,lon=geocodes[key]
            url="https://api.open-meteo.com/v1/forecast?"+urllib.parse.urlencode({
                "latitude":lat,"longitude":lon,"hourly":",".join(FIELDS),
                "start_date":kickoff.date().isoformat(),
                "end_date":(kickoff+timedelta(days=1)).date().isoformat(),
                "timezone":"UTC","temperature_unit":"fahrenheit",
                "wind_speed_unit":"mph","precipitation_unit":"inch"})
            forecasts=extract_hourly_forecast(fetch(url),kickoff)
            rec.update({
                "status":"CITY_CENTRE_WEATHER_CAPTURED_NOT_STADIUM_SENSOR",
                "location_precision":"CITY_CENTRE_APPROXIMATION_NOT_VENUE_COORDINATE",
                "geocoded_city_latitude":lat,"geocoded_city_longitude":lon,
                "hourly_forecast":forecasts,
                "last_provider_response_received_at_utc":datetime.now(timezone.utc).isoformat(),
                "weather_effect_on_model":"NOT_TRAINED_OR_APPLIED",
                "weather_effect_confidence":"UNKNOWN_UNVALIDATED",
            })
        except (ValueError,KeyError,RuntimeError,TimeoutError,OSError) as err:
            rec["status"]="WEATHER_EVIDENCE_UNAVAILABLE"
            rec["reason"]=str(err)[:250]
        rows.append(rec)
    return {
        "schema":SCHEMA,"observed_at_utc":observed.isoformat(),
        "n_upcoming_games":len(rows),
        "n_city_weather_captured":sum(r["status"]=="CITY_CENTRE_WEATHER_CAPTURED_NOT_STADIUM_SENSOR" for r in rows),
        "weather_model_not_promoted":True,
        "fixed_pretip_horizon_preserved":True,
        "no_bookmaker_data_used":True,
        "no_numerical_model_projection_changed":True,
        "license_notice":"Open-Meteo free API is for non-commercial use; verify applicable terms before commercial product deployment. Attribution required.",
        "games":rows
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--serving",default="docs/nfl_predictions.json")
    p.add_argument("--out",required=True)
    args=p.parse_args()
    doc=json.loads(Path(args.serving).read_text())
    res=capture(doc["scheduled_games"])
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(res,indent=2,sort_keys=True)+"\n")
    print("NFL_V3_PRETIP_WEATHER_CAPTURE",json.dumps({
        "upcoming":res["n_upcoming_games"],
        "weather_city_captured":res["n_city_weather_captured"],
        "unavailable":res["n_upcoming_games"]-res["n_city_weather_captured"],
        "model_changed":False
    },sort_keys=True))


if __name__=="__main__":
    main()
