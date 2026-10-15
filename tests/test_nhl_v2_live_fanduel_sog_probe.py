"""Real NHL bookline source adapter tests never use purchased API credits."""
from datetime import datetime,timezone
import json
import nhl_v2_live_fanduel_sog_probe as P
from nhl_sog_fanduel_mainline_reference import SCHEMA

NOW=datetime(2026,10,11,20,0,tzinfo=timezone.utc)
AFTER=datetime(2026,10,11,20,1,tzinfo=timezone.utc)

def board():
    fields=["game_id","player_id","player","team","opponent",
        "scheduled_start","forecast_horizon","generated_at","cutoff_at",
        "mean","p_ge1","p_ge2","p_ge3","p_ge4","p_ge5"]
    row=["2026020084","8471234","Test Forward","BOS","NYR",
        "2026-10-11T21:00:00Z","T90","2026-10-11T19:19:00Z",
        "2026-10-11T19:30:00Z",2.9,.91,.72,.52,.33,.19]
    return {
      "schema":"nhl-v2-sog-nfl-style-shadow-v1",
      "status":"RESEARCH_ONLY_NOT_VALIDATED_FOR_BETTING",
      "not_a_betting_board":True,
      "verified_original_hash_chain":True,
      "verified_frozen_code_and_model_lock":True,
      "actual_player_book_lines_verified":False,
      "dressed_lineups_verified":False,
      "historical_corrected_probabilities_were_not_published_pregame":True,
      "source_ledger_sha256":"b"*64,
      "row_fields":fields,
      "forecasts":{"T24H":[],"T90":[row],"T30":[]}
    }

def event(quote=False):
    e={"id":"realtest84","sport_key":"icehockey_nhl",
       "commence_time":"2026-10-11T21:00:00Z",
       "home_team":"Boston Bruins","away_team":"New York Rangers"}
    if quote:
        e["bookmakers"]=[{"key":"fanduel","markets":[{
          "key":"player_shots_on_goal",
          "last_update":"2026-10-11T19:56:00Z",
          "outcomes":[
          {"name":"Over","description":"Test Forward","point":2.5,"price":-112},
          {"name":"Under","description":"Test Forward","point":2.5,"price":-108}
          ]}]}]
    return e

def transport(url,params):
    assert params["apiKey"]=="not-a-real-key"
    if url.endswith("/events"):
        return [event()],{"remaining":"100","response_sha256":"1"*64}
    assert "/events/realtest84/odds" in url
    assert params["bookmakers"]=="fanduel"
    assert params["markets"]=="player_shots_on_goal"
    assert params["oddsFormat"]=="american"
    return event(quote=True),{
        "remaining":"99","response_sha256":"f"*64,"used":"1"}

def test_live_event_mainline_only_once_bounded_and_no_official_bet():
    d=P.collect(board(),api_key="not-a-real-key",now=NOW,fetch=transport,
                clock=lambda:AFTER,max_events=1)
    assert d["status"]=="VERIFIED_REAL_FANDUEL_MAINLINES_RESEARCH_ONLY",d
    assert d["matched_events"]==1
    assert d["actual_paid_event_calls"]==1
    assert d["qualified_player_main_lines"]==1,d
    assert d["records"][0]["fan_duel_main_shots_line"]==2.5
    assert d["records"][0]["model_posthoc_p_over_this_line"]==.52
    assert d["official_picks"]==0 and not d["lineup_certified"]
    assert d["records"][0]["official_prediction"] is False
    assert d["records"][0]["pick"] is None
    published=P.public_status(d)
    assert "records" not in published
    assert "retrieval_hashes" not in published
    assert "price" not in json.dumps(published)


def test_missing_credentials_consumes_zero_real_request_credits():
    calls=[]
    d=P.collect(board(),api_key=None,now=NOW,
                fetch=lambda *a,**kw:calls.append(a))
    assert d["status"]=="MISSING_EXISTING_GITHUB_ODDS_API_KEY"
    assert d["actual_paid_event_calls"]==0
    assert not calls


def test_no_model_game_and_no_provider_calls_after_puck_drop():
    calls=[]
    d=P.collect(board(),api_key="not-a-real-key",
                now=datetime(2026,10,11,23,tzinfo=timezone.utc),
                fetch=lambda *a,**kw:calls.append(a))
    assert d["status"]=="NO_FUTURE_VERIFIED_MODEL_GAMES_NO_PAID_API_CALLS"
    assert d["actual_paid_event_calls"]==0
    assert not calls


def test_unlocked_research_or_fabricated_board_refused_even_with_valid_key():
    invalid=board()
    invalid["verified_original_hash_chain"]=False
    import pytest
    with pytest.raises(ValueError):
        P.collect(invalid,api_key="not-a-real-key",now=NOW,fetch=transport)


def test_three_game_cap_never_shops_alt_lines_or_spends_without_match():
    b=board()
    events=[event(),event()]
    events[1]=dict(events[1],id="other",home_team="Toronto Maple Leafs")
    count=[]
    def get(url,params):
        count.append(url)
        if url.endswith("/events"):
            return events,{"remaining":"20","response_sha256":"1"*64}
        return event(quote=True),{"remaining":"19","response_sha256":"f"*64}
    result=P.collect(b,api_key="not-a-real-key",now=NOW,fetch=get,
                     clock=lambda:AFTER,max_events=3)
    assert len(count)==2
    assert result["actual_paid_event_calls"]==1
    assert result["qualified_player_main_lines"]==1


def test_incomplete_book_market_yields_no_fake_line():
    def missing(url,params):
        if url.endswith("/events"):
            return [event()],{"remaining":"20","response_sha256":"1"*64}
        return event(),{"remaining":"19","response_sha256":"f"*64}
    d=P.collect(board(),api_key="not-a-real-key",now=NOW,fetch=missing,
                clock=lambda:AFTER)
    assert d["qualified_player_main_lines"]==0
    assert d["official_picks"]==0
    assert d["provider_rejections"]["NO_UNIQUE_FANDUEL_EVENT"]==1
    assert d["status"]=="NO_ADMISSIBLE_REAL_FANDUEL_MAINLINES"


def test_real_provider_401_is_access_denied_and_no_more_events_are_queried():
    inputs=board()
    # Two eligible real NHL games at the same time, but a 401 on the first
    # must stop the whole provider cascade immediately.
    dup=list(inputs["forecasts"]["T90"][0])
    dup[0]="2026020085"
    dup[1]="8471244"
    dup[2]="Other Forward"
    dup[3]="TOR"
    dup[4]="MTL"
    inputs["forecasts"]["T90"].append(dup)
    calls=[]
    def auth(url,params):
        calls.append(url)
        if url.endswith("/events"):
            first=event()
            second=dict(first,id="realtest85",
                        home_team="Toronto Maple Leafs",
                        away_team="Montreal Canadiens")
            return [first,second],{"remaining":"50","response_sha256":"1"*64}
        raise P.QuoteCaptureFailure("PROVIDER_HTTP_STATUS_401")
    d=P.collect(inputs,api_key="not-a-real-key",now=NOW,fetch=auth,
                clock=lambda:AFTER,max_events=3)
    assert len(calls)==2  # free event list + exactly one denied paid endpoint
    assert d["actual_paid_event_calls"]==1
    assert d["provider_rejections"]["PROVIDER_HTTP_STATUS_401"]==1
    assert d["provider_access_blocked"]=="PROVIDER_HTTP_STATUS_401"
    assert d["status"]=="PROVIDER_AUTH_OR_PLAYER_PROP_ENTITLEMENT_BLOCKED"
    assert d["qualified_player_main_lines"]==0
    assert d["official_picks"]==0


def test_real_provider_rate_limit_stops_without_faking_no_market():
    def quota(url,params):
        if url.endswith("/events"):
            return [event()],{"remaining":"0","response_sha256":"1"*64}
        raise P.QuoteCaptureFailure("PROVIDER_HTTP_STATUS_429")
    d=P.collect(board(),api_key="not-a-real-key",now=NOW,fetch=quota,
                clock=lambda:AFTER,max_events=3)
    assert d["status"]=="PROVIDER_RATE_LIMITED"
    assert d["actual_paid_event_calls"]==1
    assert d["provider_access_blocked"]=="PROVIDER_HTTP_STATUS_429"
