"""NHL V2 projection research contract: actual shot-count distributions FIRST.

Synthetic tests deliberately include historical shifted P1..P5 to prove the
adapter ignores them, plus cross-team identity, timestamp and duplicate gates.
"""
import copy
import math

import pytest

from nhl_sog_projection_contract_a import (
    ProjectionIntegrityError, count_projection, probability_over_half_line,
    research_board, SCHEMA,
)


def forecast(pid=1001, game=2026020001, team="NYR", horizon="T24H",
             issued="2026-10-09T14:00:00Z", cutoff="2026-10-09T16:00:00Z",
             start="2026-10-10T16:00:00Z", mu=1.0, alpha=1.0):
    return {
        "record_type": "FORECAST",
        "forecast_id": f"{game}-{horizon}-{pid}",
        "player_id": pid, "game_id": game, "team": team,
        "opponent": "BOS" if team == "NYR" else "NYR",
        "forecast_horizon": horizon,
        "scheduled_start": start, "generated_at": issued, "cutoff_at": cutoff,
        "source_manifest_sha256": "a"*64,
        "expected_sog": mu, "dispersion": alpha,
        "median_sog": 99, "P1": 1.0, "P2": .50, "P3": .25,
        "P4": .125, "P5": .0625,  # intentionally SHIFTED and mislabeled legacy fields
        "candidate_status": "CANDIDATE_MEANINGFUL",
        "availability_state": "NO_ROSTER_DATA_PUBLISHED_YET",
        "receipt": {"player_name": "Player One"},
    }


def test_nb2_expected_shots_and_correct_p_ge_k_independent_of_any_book_line():
    values = count_projection(1.0, 1.0)
    assert values["expected_sog"] == 1
    assert values["median_sog"] == 0  # geometric dist: P(0) = .5
    assert values["P1"] == 0.5
    assert values["P2"] == 0.25
    assert values["P3"] == 0.125
    assert values["P4"] == 0.0625
    assert values["P5"] == 0.03125
    assert values["p10_sog"] == 0
    assert values["p90_sog"] == 3
    assert all(values[f"P{k}"] >= values[f"P{k+1}"] for k in range(1,5))


def test_true_player_specific_line_can_change_probability_without_changing_forecast():
    a = probability_over_half_line(1., 1., 1.5)
    b = probability_over_half_line(1., 1., 2.5)
    c = probability_over_half_line(1., 1., 3.5)
    assert a["over_probability"] == 0.25
    assert b["over_probability"] == 0.125
    assert c["over_probability"] == 0.0625
    assert all(math.isclose(x["over_probability"]+x["under_probability"],1)
               for x in (a,b,c))


@pytest.mark.parametrize("mu,alpha", [(float("nan"),1.0),(1.0,0.0),(-1.0,.5),
                                       (2.0,float("inf")),(True,1.0),
                                       (2.0,True),(51.0,1.0)])
def test_bad_count_parameters_must_be_rejected_not_clipped_to_fake_confidence(mu,alpha):
    with pytest.raises(ProjectionIntegrityError):
        count_projection(mu,alpha)


@pytest.mark.parametrize("line",[2.0,1.25,13.5,float("nan"),True,-0.5])
def test_sportsbook_line_helper_cannot_infer_unseen_or_unsupported_main_line(line):
    with pytest.raises(ProjectionIntegrityError):
        probability_over_half_line(2.0,0.5,line)


def test_unverified_v2_forecast_shows_expected_shots_but_no_market_pick():
    rows = [forecast()]
    original = copy.deepcopy(rows)
    board = research_board(rows,as_of_utc="2026-10-10T12:00:00Z")
    assert rows == original
    assert board["schema"] == SCHEMA
    assert board["projection_count"] == 1
    row = board["projections"][0]
    assert row["expected_sog"] == 1.0
    assert row["P1"] == .5  # NOT legacy P1 == 1
    assert row["P3"] == .125
    assert row["median_sog"] == 0  # NOT old median == 99
    assert row["original_legacy_P1_through_P5_used"] is False
    assert row["research_only"] and not row["model_promoted"]
    assert row["availability_confirmed_for_game"] is False
    assert row["betting_pick"] is None and row["book_main_line"] is None
    assert board["official_shots_on_goal_picks"] == 0


def test_cross_team_same_player_same_game_is_quarantined_not_confidently_selected():
    rows = [forecast(),forecast(team="BOS",horizon="T90",
                              issued="2026-10-10T10:00:00Z",
                              cutoff="2026-10-10T11:00:00Z")]
    result = research_board(rows, as_of_utc="2026-10-10T12:00:00Z")
    assert result["projection_count"] == 0
    assert result["excluded"]["CROSS_TEAM_OR_SCHEDULE_CONFLICT_QUARANTINED"] == 2


def test_two_valid_forecast_horizons_one_visible_player_projection_no_blend():
    rows = [forecast(), forecast(horizon="T90",mu=2.7,alpha=.25,
                                issued="2026-10-10T10:00:00Z",
                                cutoff="2026-10-10T11:00:00Z")]
    result = research_board(rows,as_of_utc="2026-10-10T12:00:00Z")
    assert result["projection_count"] == 1
    assert result["projections"][0]["forecast_horizon"] == "T90"
    assert result["projections"][0]["expected_sog"] == 2.7
    assert result["excluded"]["SECONDARY_HORIZON_NOT_SELECTED_FOR_DISPLAY"] == 1


def test_post_cutoff_and_already_started_rows_cannot_be_promoted_retroactively():
    late = forecast(issued="2026-10-09T17:00:00Z")
    after = forecast(pid=2002,start="2026-10-10T11:00:00Z",
                     cutoff="2026-10-09T16:00:00Z")
    board = research_board([late,after],as_of_utc="2026-10-10T12:00:00Z")
    assert board["projection_count"] == 0
    assert board["excluded"]["INVALID_OR_POST_CUTOFF_RECEIPT"] == 1
    assert board["excluded"]["GAME_STARTED_NOT_A_PREMATCH_PROJECTION"] == 1


def test_unverified_source_or_duplicate_identical_horizon_rows_not_published():
    broken = forecast()
    broken["source_manifest_sha256"] = None
    duplicate1=forecast(pid=3010)
    duplicate2=copy.deepcopy(duplicate1)
    result=research_board([broken,duplicate1,duplicate2],as_of_utc="2026-10-10T12:00:00Z")
    assert result["projection_count"]==0
    assert result["excluded"]["MISSING_SOURCE_VINTAGE_HASH"]==1
    assert result["excluded"]["DUPLICATE_FORECAST_SAME_HORIZON_AND_TIME"]==2


def test_all_forecasts_after_as_of_time_blocked():
    future = forecast(issued="2026-10-10T13:00:00Z",
                      cutoff="2026-10-10T15:00:00Z")
    result = research_board([future],as_of_utc="2026-10-10T12:00:00Z")
    assert result["projection_count"]==0
    assert result["excluded"]["INVALID_OR_POST_CUTOFF_RECEIPT"]==1
