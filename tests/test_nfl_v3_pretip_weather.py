"""Free weather source proof: ESPN venue, Open-Meteo UTC hours, no bets."""
from datetime import datetime,timezone
import pytest

from nfl_v3_pretip_weather import (
    SCHEMA,game_venue,place_coordinates,extract_hourly_forecast,capture,
)

NOW=datetime(2026,11,7,22,0,tzinfo=timezone.utc)
KICK="2026-11-08T18:00:00Z"


def make_event():
    return {
        "id":"401","competitions":[{
            "date":KICK,
            "competitors":[
                {"homeAway":"home","team":{"abbreviation":"ATL"}},
                {"homeAway":"away","team":{"abbreviation":"CIN"}}],
            "venue":{"fullName":"Example Dome","indoor":True,
                     "address":{"city":"Atlanta","state":"GA"}}
        }]
    }


def hourly():
    labels=[f"2026-11-08T{h:02d}:00" for h in range(16,23)]
    return {
        "timezone":"GMT","utc_offset_seconds":0,
        "hourly":{"time":labels,
                  "temperature_2m":[62]*len(labels),
                  "wind_speed_10m":[10]*len(labels),
                  "wind_gusts_10m":[16]*len(labels),
                  "precipitation":[0.0]*len(labels),
                  "precipitation_probability":[30]*len(labels)}
    }


def fetch(url):
    if "espn.com" in url:return {"events":[make_event()]}
    if "geocoding-api" in url:
        return {"results":[{"name":"Atlanta","admin1":"Georgia","country_code":"US",
                            "latitude":33.749,"longitude":-84.387}]}
    if "api.open-meteo.com" in url:return hourly()
    raise AssertionError("Unexpected network endpoint "+url)


def test_real_schedule_venue_and_keyless_city_weather_pre_game_only():
    rows=[{"home_team":"ATL","away_team":"CIN","kickoff_utc":KICK}]
    report=capture(rows,now=NOW,fetch=fetch)
    assert report["schema"]==SCHEMA
    assert report["n_upcoming_games"]==1
    assert report["n_city_weather_captured"]==1
    assert report["weather_model_not_promoted"]
    assert report["no_numerical_model_projection_changed"]
    game=report["games"][0]
    assert game["venue"]["name"]=="Example Dome"
    assert game["venue"]["is_indoor_venue"] is True
    assert game["venue"]["roof_status"]=="NOT_VERIFIED_BY_VENUE_TYPE"
    assert game["status"]=="CITY_CENTRE_WEATHER_CAPTURED_NOT_STADIUM_SENSOR"
    assert game["location_precision"]=="CITY_CENTRE_APPROXIMATION_NOT_VENUE_COORDINATE"
    assert len(game["hourly_forecast"])>=3
    assert game["hourly_forecast"][0]["wind_speed_10m"]==10
    assert all(x["valid_time_utc"].endswith("+00:00") for x in game["hourly_forecast"])


def test_wrong_teams_or_started_game_never_generate_pretip_weather():
    event=make_event()
    assert game_venue(event,"TB","ATL",datetime(2026,11,8,18,tzinfo=timezone.utc)) is None
    actual=capture([{"home_team":"ATL","away_team":"CIN","kickoff_utc":KICK}],
                   now=datetime(2026,11,8,18,1,tzinfo=timezone.utc),
                   fetch=lambda url: (_ for _ in ()).throw(RuntimeError("network called after kickoff")))
    assert actual["n_upcoming_games"]==0


def test_missing_venue_or_geocode_never_uses_fake_stadium_weather():
    def absent(url):
        if "espn.com" in url:return {"events":[]}
        raise AssertionError("Weather called despite absent exact-match venue")
    r=capture([{"home_team":"ATL","away_team":"CIN","kickoff_utc":KICK}],
              now=NOW,fetch=absent)
    assert r["n_city_weather_captured"]==0
    assert r["games"][0]["status"]=="WEATHER_EVIDENCE_UNAVAILABLE"
    with pytest.raises(ValueError,match="NOT_UNIQUE"):
        place_coordinates({"results":[{"name":"Atlanta","admin1":"Georgia",
             "country_code":"US","latitude":33.75,"longitude":-84.39},
             {"name":"Atlanta","admin1":"Georgia","country_code":"US",
              "latitude":33.75,"longitude":-84.39}]},"Atlanta","GA")


def test_timezone_and_missing_weather_values_are_not_interpreted_as_clear_sky():
    doc=hourly()
    doc["utc_offset_seconds"]=-5*3600
    with pytest.raises(ValueError,match="TIMEZONE"):
        extract_hourly_forecast(doc,datetime(2026,11,8,18,tzinfo=timezone.utc))
    doc=hourly()
    doc["hourly"]["precipitation_probability"][2]=None
    with pytest.raises(ValueError,match="NON_NUMERIC"):
        extract_hourly_forecast(doc,datetime(2026,11,8,18,tzinfo=timezone.utc))
    doc=hourly()
    doc["hourly"]["wind_gusts_10m"][2]=-1
    with pytest.raises(ValueError,match="INVALID_HOURLY"):
        extract_hourly_forecast(doc,datetime(2026,11,8,18,tzinfo=timezone.utc))
