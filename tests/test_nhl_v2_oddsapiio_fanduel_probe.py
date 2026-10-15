"""Quota-capped real NHL provider market discovery never invents sportsbook lines."""
from datetime import datetime,timezone
import nhl_v2_oddsapiio_fanduel_probe as P

NOW=datetime(2026,10,11,20,0,tzinfo=timezone.utc)

def board():
    fields=["game_id","player_id","player","team","opponent",
            "scheduled_start","forecast_horizon","generated_at","cutoff_at",
            "mean","p_ge1","p_ge2","p_ge3","p_ge4","p_ge5"]
    row=["2026020084","8471234","Test Forward","BOS","NYR",
         "2026-10-11T21:00:00Z","T90","2026-10-11T19:19:00Z",
         "2026-10-11T19:30:00Z",2.9,.91,.72,.52,.33,.19]
    return {"schema":"nhl-v2-sog-nfl-style-shadow-v1",
            "status":"RESEARCH_ONLY_NOT_VALIDATED_FOR_BETTING",
            "verified_original_hash_chain":True,
            "verified_frozen_code_and_model_lock":True,
            "not_a_betting_board":True,
            "actual_player_book_lines_verified":False,
            "dressed_lineups_verified":False,
            "historical_corrected_probabilities_were_not_published_pregame":True,
            "row_fields":fields,
            "forecasts":{"T90":[row],"T30":[],"T24H":[]}}


def event():
    return {"id":777,"home":"Boston Bruins",
            "away":"New York Rangers","date":"2026-10-11T21:00:00Z"}


def odds():
    return {"id":777,"home":"Boston Bruins",
            "away":"New York Rangers","date":"2026-10-11T21:00:00Z",
            "bookmakers":{"FanDuel":[
              {"name":"Player Props","odds":[
                {"label":"Test Forward (Total Shots on Goal)","hdp":2.5,
                 "over":"1.91","under":"1.83"}]}]}}


def transport(path,params):
    assert params["apiKey"]=="demo"
    if path=="/events":
        assert params["sport"]=="nhl"
        return [event()]
    assert path=="/odds" and params["eventId"]==777
    assert params["bookmakers"]=="FanDuel"
    return odds()


def test_main_shots_have_real_provider_match_but_never_claim_picks_or_dressed():
    d=P.coverage(board(),key="demo",now=NOW,fetch=transport)
    assert d["status"]=="REAL_FANDUEL_SOG_MAINLINES_OBSERVED_RESEARCH_ONLY",d
    assert d["paid_event_requests_attempted"]==1
    assert d["matched_provider_games"]==1
    assert d["verified_main_shot_lines"]==1
    assert d["matched_model_players_with_real_line"]==1
    assert d["official_picks"]==0 and d["lineup_certified"] is False
    assert d["individual_quotes_published"] is False


def test_ambiguous_shot_prop_lines_are_not_main():
    b=odds()
    b["bookmakers"]["FanDuel"][0]["odds"].append(
        {"label":"Test Forward (Total Shots on Goal)","hdp":3.5,
         "over":"2.3","under":"1.5"})
    lines,rejected=P._shot_mainline_candidates(b)
    assert lines=={}
    assert rejected["MULTIPLE_LINES_OR_ALTERNATES_QUARANTINED"]==1


def test_other_props_cannot_be_mistaken_for_shots():
    b=odds()
    b["bookmakers"]["FanDuel"][0]["odds"][0]["label"]="Test Forward (Points)"
    lines,rejected=P._shot_mainline_candidates(b)
    assert not lines


def test_no_secret_means_no_paid_or_free_provider_calls():
    calls=[]
    d=P.coverage(board(),key=None,now=NOW,
                 fetch=lambda *args:calls.append(args))
    assert d["status"]=="PROVIDER_KEY_NOT_AVAILABLE_IN_GITHUB_SECRETS"
    assert calls==[]
    assert d["paid_event_requests_attempted"]==0


def test_only_future_games_with_pre_cutoff_forecasts_qualify():
    calls=[]
    future=datetime(2026,10,11,23,0,tzinfo=timezone.utc)
    d=P.coverage(board(),key="demo",now=future,
                 fetch=lambda *args:calls.append(args))
    assert d["status"]=="NO_VERIFIED_FUTURE_MODEL_GAMES"
    assert calls==[]


def test_http_401_blocks_any_further_provider_calls():
    calls=[]
    def blocked(path,params):
        calls.append(path)
        if path=="/events":
            return [event()]
        raise P.ProviderUnavailable("HTTP_401")
    d=P.coverage(board(),key="demo",now=NOW,fetch=blocked)
    assert d["status"]=="SOURCE_BLOCKED_HTTP_401"
    assert d["paid_event_requests_attempted"]==1
    assert calls==["/events","/odds"]


def test_bad_provider_payload_never_verified():
    def empty(path,params):
        if path=="/events":
            return [event()]
        return {"bookmakers":{"FanDuel":[]}}
    d=P.coverage(board(),key="demo",now=NOW,fetch=empty)
    assert d["verified_main_shot_lines"]==0
    assert d["matched_model_players_with_real_line"]==0
    assert d["official_picks"]==0
