"""Delivery evidence: success means a verified on-time forecast, not a green workflow."""
from datetime import datetime,timezone
import pytest
from nhl_v2_delivery_receipt_audit import analyze

NOW=datetime(2026,10,11,20,0,tzinfo=timezone.utc)

def fc(player=1):
    return {
        "record_type":"FORECAST","game_id":202602001,
        "forecast_horizon":"T30",
        "scheduled_start":"2026-10-11T20:30:00Z",
        "cutoff_at":"2026-10-11T20:00:00Z",
        "generated_at":"2026-10-11T19:59:01Z",
        "player_id":player
    }

def miss(gid=2):
    return {
        "record_type":"MISSED_CUTOFF","game_id":gid,
        "forecast_horizon":"T90",
        "scheduled_start":"2026-10-11T20:30:00Z",
        "cutoff_at":"2026-10-11T19:00:00Z",
        "observed_at":"2026-10-11T19:15:00Z"
    }


def test_valid_forecast_and_a_miss_are_distinct_immutable_decisions():
    out=analyze([fc(),fc(2),miss()],NOW)
    assert out["total_ledger_decision_games"]==2
    assert out["recent_forecast_game_decisions"]==1
    assert out["recent_missed_or_failed_decisions"]==1
    assert out["by_horizon"]["T30"]["model_forecast_player_rows"]==2
    assert out["by_horizon"]["T90"]["failed_or_missed_decisions"]==1
    assert out["last_genuine_frozen_pregame_forecast_utc"] is not None
    assert out["status"].startswith("GAPS_PROVEN")


def test_no_ledger_does_not_claim_there_was_no_schedule():
    report=analyze([],NOW)
    assert report["recent_observed_decision_games"]==0
    assert "SCHEDULE_COMPLETENESS_UNKNOWN" in report["status"]


def test_postcutoff_late_forecast_never_counted_as_a_valid_capture():
    late=fc()
    late["generated_at"]="2026-10-11T20:00:11Z"
    with pytest.raises(ValueError,match="postcutoff"):
        analyze([late],NOW)


def test_exact_same_decision_cannot_be_forecast_and_missed():
    failed=miss(gid=202602001)
    failed["forecast_horizon"]="T30"
    failed["cutoff_at"]="2026-10-11T20:00:00Z"
    with pytest.raises(ValueError,match="valid forecast and an invalid"):
        analyze([fc(),failed],NOW)


def test_duplicate_synthetic_misses_or_players_rejected():
    with pytest.raises(ValueError,match="duplicate immutable"):
        analyze([miss(),miss()],NOW)
    with pytest.raises(ValueError,match="duplicate or null"):
        analyze([fc(),fc()],NOW)


def test_old_missed_cannot_be_rebranded_as_new_forecast():
    old=miss()
    old["scheduled_start"]="2026-09-09T21:00:00Z"
    old["cutoff_at"]="2026-09-09T19:30:00Z"
    old["observed_at"]="2026-09-09T19:34:00Z"
    x=analyze([old],NOW)
    assert x["recent_missed_or_failed_decisions"]==0
    assert x["all_recorded_decision_statuses"]["MISSED_CUTOFF"]==1
    assert x["recent_forecast_game_decisions"]==0
