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
    report=capture(rows,now=NOW,fetch=fetch,clock=lambda:NOW)
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


def test_weather_response_after_kickoff_not_counted_as_pregame_evidence():
    # Even if retrieval started before kickoff, a long provider response
    # cannot claim a completed pre-kickoff receipt once the game begins.
    start=datetime(2026,11,8,17,59,tzinfo=timezone.utc)
    end=datetime(2026,11,8,18,1,tzinfo=timezone.utc)
    report=capture([{"home_team":"ATL","away_team":"CIN","kickoff_utc":KICK}],
                   now=start,fetch=fetch,clock=lambda:end)
    assert report["n_upcoming_games"]==1
    assert report["n_city_weather_captured"]==0
    assert "RETRIEVED_AFTER_KICKOFF" in report["games"][0]["reason"]


def test_major_miami_gardens_source_place_beats_small_duplicate_without_guessing():
    chosen=place_coordinates({"results":[
        {"id":4164167,"name":"Miami Gardens","admin1":"Florida",
         "country_code":"US","population":113187,
         "latitude":25.94204,"longitude":-80.2456},
        {"id":4164166,"name":"Miami Gardens","admin1":"Florida",
         "country_code":"US","population":2706,
         "latitude":25.97981,"longitude":-80.20255}
    ]},"Miami Gardens","FL")
    assert chosen==(25.94204,-80.2456)
    # Without population or overwhelming separation, source stays unresolved.
    with pytest.raises(ValueError,match="NOT_UNIQUE"):
        place_coordinates({"results":[
            {"name":"Miami Gardens","admin1":"Florida",
             "country_code":"US","population":110000,
             "latitude":25.94204,"longitude":-80.2456},
            {"name":"Miami Gardens","admin1":"Florida",
             "country_code":"US","population":22000,
             "latitude":25.97981,"longitude":-80.20255}
        ]},"Miami Gardens","FL")


def test_madrid_international_game_never_uses_falcons_atlanta_stadium():
    MadridKick="2026-11-08T14:30:00Z"
    event={"id":"Madrid","competitions":[{
        "date":MadridKick,
        "competitors":[{"homeAway":"home","team":{"abbreviation":"ATL"}},
                       {"homeAway":"away","team":{"abbreviation":"CIN"}}],
        "venue":{"fullName":"Santiago Bernabéu",
                 "address":{"city":"Madrid","country":"Spain"}}}]}
    venue=game_venue(event,"ATL","CIN",
                     datetime(2026,11,8,14,30,tzinfo=timezone.utc))
    assert venue["city"]=="Madrid"
    assert venue["state"] is None
    assert venue["country_code"]=="ES"
    assert venue["name"]=="Santiago Bernabéu"
    result=place_coordinates({"results":[{
        "name":"Madrid","admin1":"Madrid","country_code":"ES",
        "latitude":40.4168,"longitude":-3.7038}]},"Madrid",None,"ES")
    assert result==(40.4168,-3.7038)
    assert venue["roof_status"]=="NOT_VERIFIED_BY_VENUE_TYPE"


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
