"""Fail-closed NHL daily source extraction: no lost, fabricated or future-fitted shots."""
from nhl_v2_daily_serve import subset,PARTITION_SCHEMA
import copy
import pytest

FIELDS=["game_id","player_id","game_date","scheduled_start"]
def source():
    a=["2026020176","3001","2026-10-24","2026-10-25T00:00:00Z"]
    b=["2026020180","3001","2026-10-25","2026-10-25T19:00:00Z"]
    return {"schema":"nhl-v2-sog-nfl-style-shadow-v1",
       "verified_original_hash_chain":True,
       "verified_frozen_code_and_model_lock":True,
       "status":"RESEARCH_ONLY_NOT_VALIDATED_FOR_BETTING",
       "not_a_betting_board":True,
       "historical_corrected_probabilities_were_not_published_pregame":True,
       "source_snapshot_commit_sha":"a"*40,
       "source_ledger_sha256":"b"*64,
       "row_fields":FIELDS,"horizons":["T24H","T90","T30"],
       "dates":["2026-10-24","2026-10-25"],
       "forecasts":{"T24H":[a,b],"T90":[a],"T30":[a]},
       "total_valid_forecasts":4}

def test_daily_exact_original_rows_and_source_provenance():
    full=source();frozen=copy.deepcopy(full)
    a=subset(full,"2026-10-24")
    assert a["daily_partition_schema"]==PARTITION_SCHEMA
    assert a["daily_partition_game_date"]=="2026-10-24"
    assert a["total_valid_forecasts"]==3
    assert a["full_source_total_valid_forecasts"]==4
    assert a["forecasts"]["T24H"][0] is full["forecasts"]["T24H"][0]
    assert a["source_ledger_sha256"]==full["source_ledger_sha256"]
    assert a["historical_corrected_probabilities_were_not_published_pregame"]
    assert a["not_a_betting_board"] and a["all_rows_exactly_preserved_from_original"]
    assert full==frozen
    tomorrow=subset(full,"2026-10-25")
    assert tomorrow["total_valid_forecasts"]==1
    assert tomorrow["forecasts"]["T90"]==[]
    assert tomorrow["forecasts"]["T30"]==[]

def test_source_research_truth_gate_does_not_promote_stale_or_unsupported_goals():
    bad=source();bad["not_a_betting_board"]=False
    with pytest.raises(ValueError,match="UNVERIFIED"):
        subset(bad,"2026-10-24")
    bad=source();bad["historical_corrected_probabilities_were_not_published_pregame"]=False
    with pytest.raises(ValueError,match="UNCALIBRATED"):
        subset(bad,"2026-10-24")
    with pytest.raises(ValueError,match="DATE"):
        subset(source(),"October 24")
    empty=subset(source(),"2026-10-27")
    assert empty["total_valid_forecasts"]==0 and empty["dates"]==[]

def test_duplicate_original_source_not_silently_deduplicated():
    bad=source();bad["forecasts"]["T90"].append(list(bad["forecasts"]["T90"][0]))
    with pytest.raises(ValueError,match="DUPLICATE"):
        subset(bad,"2026-10-24")
