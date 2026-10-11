"""NHL real-FanDuel-main-line shape test: never invent prices or bet sides."""
from copy import deepcopy
import pytest
import nhl_sog_fanduel_mainline_reference as r

NOW="2026-10-11T20:00:00Z"
START="2026-10-11T21:00:00Z"


def original_model_board():
    fields=["game_id","player_id","player","team","opponent",
            "scheduled_start","forecast_horizon","generated_at","cutoff_at",
            "mean","p_ge1","p_ge2","p_ge3","p_ge4","p_ge5"]
    return {
       "schema":"nhl-v2-sog-nfl-style-shadow-v1",
       "status":"RESEARCH_ONLY_NOT_VALIDATED_FOR_BETTING",
       "not_a_betting_board":True,
       "verified_original_hash_chain":True,
       "verified_frozen_code_and_model_lock":True,
       "actual_player_book_lines_verified":False,
       "dressed_lineups_verified":False,
       "historical_corrected_probabilities_were_not_published_pregame":True,
       "source_ledger_sha256":"a"*64,
       "row_fields":fields,
       "forecasts":{"T90":[["2026020084","8471234","Test Forward","BOS","NYR",
                 START,"T90","2026-10-11T19:19:00Z",
                 "2026-10-11T19:30:00Z",2.9,.91,.72,.52,.33,.19]],
                    "T30":[],"T24H":[]}
    }


def real_provider_event():
    return {
      "id":"theodds_event_84","sport_key":"icehockey_nhl",
      "commence_time":START,
      "home_team":"Boston Bruins","away_team":"New York Rangers",
      "bookmakers":[{"key":"fanduel","markets":[{
         "key":"player_shots_on_goal",
         "last_update":"2026-10-11T19:56:00Z",
         "outcomes":[
           {"name":"Over","description":"Test Forward","point":2.5,"price":-112},
           {"name":"Under","description":"Test Forward","point":2.5,"price":-108}
         ]}]}]
    }


def test_real_main_line_matches_only_one_game_and_original_forecast():
    d=r.compare_original_model_to_real_market(
        original_model_board(),[real_provider_event()],captured_at_utc=NOW)
    assert d["fanduel_player_main_lines_matched_to_real_model"]==1
    p=d["records"][0]
    assert p["fan_duel_main_shots_line"]==2.5
    assert p["model_posthoc_p_over_this_line"]==.52
    assert p["model_posthoc_p_under_this_line"]==.48
    assert p["player_id"]=="8471234"
    assert p["official_prediction"] is False
    assert p["betting_edge_validated"] is False
    assert p["pick"] is None
    assert d["lineup_certified"] is False
    assert d["no_provider_call_made_by_this_adapter"] is True


def test_real_line_changes_derived_probability_without_rewriting_model():
    model=original_model_board()
    frozen=deepcopy(model)
    event=real_provider_event()
    for outcome in event["bookmakers"][0]["markets"][0]["outcomes"]:
        outcome["point"]=1.5
    d=r.compare_original_model_to_real_market(model,[event],captured_at_utc=NOW)
    assert d["records"][0]["model_posthoc_p_over_this_line"]==.72
    assert model==frozen


def test_multiple_main_or_alternate_lines_rejected_not_best_price_shopped():
    e=real_provider_event()
    e["bookmakers"][0]["markets"][0]["outcomes"].extend([
        {"name":"Over","description":"Test Forward","point":3.5,"price":125},
        {"name":"Under","description":"Test Forward","point":3.5,"price":-148}
    ])
    d=r.compare_original_model_to_real_market(
        original_model_board(),[e],captured_at_utc=NOW)
    assert not d["records"]
    assert d["rejections"]["MULTIPLE_ALTERNATE_OR_UNPAIRED_MAIN_LINES"]==1


def test_stale_unverified_and_wrong_bookmaker_do_not_fabricate_lines():
    e=real_provider_event()
    e["bookmakers"][0]["markets"][0]["last_update"]="2026-10-11T17:00:00Z"
    d=r.compare_original_model_to_real_market(original_model_board(),[e],captured_at_utc=NOW)
    assert d["records"]==[]
    assert d["rejections"]["STALE_BOOK_MAIN_MARKET"]==1
    e=real_provider_event()
    e["bookmakers"][0]["key"]="draftkings"
    d=r.compare_original_model_to_real_market(original_model_board(),[e],captured_at_utc=NOW)
    assert d["records"]==[]
    assert d["rejections"]["NO_UNIQUE_FANDUEL_EVENT"]==1


def test_wrong_game_start_and_model_after_puck_drop_quarantined():
    e=real_provider_event()
    e["commence_time"]="2026-10-11T23:00:00Z"
    d=r.compare_original_model_to_real_market(original_model_board(),[e],captured_at_utc=NOW)
    assert not d["records"]
    assert d["rejections"]["NO_UNIQUE_MODEL_GAME_WITH_MATCHED_REAL_START"]==1
    e=real_provider_event()
    model=original_model_board()
    model["forecasts"]["T90"][0][7]="2026-10-11T21:01:00Z"
    d=r.compare_original_model_to_real_market(model,[e],captured_at_utc=NOW)
    assert not d["records"]


def test_two_different_nhl_ids_share_name_in_same_game_rejected():
    model=original_model_board()
    extra=list(model["forecasts"]["T90"][0])
    extra[1]="8479999"
    model["forecasts"]["T90"].append(extra)
    d=r.compare_original_model_to_real_market(
        model,[real_provider_event()],captured_at_utc=NOW)
    assert not d["records"]
    assert d["rejections"]["AMBIGUOUS_PLAYER_NHL_ID"]==2


def test_bad_historical_or_unsupported_model_not_book_integrated():
    model=original_model_board()
    model["verified_original_hash_chain"]=False
    with pytest.raises(r.UnqualifiedSOGSource,match="NOT_FROZEN"):
        r.compare_original_model_to_real_market(model,[real_provider_event()],
                                                captured_at_utc=NOW)
