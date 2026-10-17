"""NHL NFL-style research page: genuine NB2 forecasts only, fail closed."""
import copy
import json
from pathlib import Path

import pytest
from scipy.stats import nbinom

import nhl_v2_projection_board_publish as B


def example(**kw):
    row = {
        "record_type": "FORECAST",
        "forecast_id": "2026020001-T24H-20261010190000-8470001",
        "row_hash": "a" * 64,
        "game_id": 2026020001,
        "player_id": 8470001,
        "schedule_date": "2026-10-10",
        "scheduled_start": "2026-10-10T19:00:00Z",
        "forecast_horizon": "T24H",
        "cutoff_at": "2026-10-09T19:00:00Z",
        "generated_at": "2026-10-09T18:59:30Z",
        "source_manifest_sha256": "1" * 64,
        "team": "BOS", "opponent": "PHI",
        "schedule_state": "OK",
        "engine_version": "B2-locked-test",
        "availability_used_in_forecast": False,
        "availability_confidence": "NOT_CERTIFIED",
        "availability_state": "NOT_CAPTURED",
        "expected_sog": 2.7, "median_sog": 2, "dispersion": .35,
        "P1": 1, "P2": .75, "P3": .56, "P4": .30, "P5": .15,
        "meaningful_expected_participant": True,
        "receipt": {
            "player_name": "Example Skater", "position": "F",
            "recent_sog_history_last10_appearances": [0, 2, 1, 4],
            "toi_seconds": {"recent3": 950},
            "pp_toi_recent3_seconds": 180,
            "opp_sog_allowed_mean5": 29.5,
            "rest_hours": 48,
        },
    }
    row.update(kw)
    return row


def extract(row,field):
    return row[B.FIELDS.index(field)]


def test_actual_count_distribution_not_fixed_line_classification():
    rows=B.derive([example()],verified_chain=True,verified_lock=True,source_sha="e"*64)
    assert rows["status"]==B.BLOCKED
    assert rows["not_a_betting_board"]
    assert rows["actual_player_book_lines_verified"] is False
    assert rows["dressed_lineups_verified"] is False
    assert rows["historical_corrected_probabilities_were_not_published_pregame"] is True
    assert rows["total_valid_forecasts"]==1
    r=rows["forecasts"]["T24H"][0]
    alpha=.35;mu=2.7;nbp=1/alpha/(1/alpha+mu)
    assert extract(r,"mean")==2.7
    assert extract(r,"p_ge1")==pytest.approx(nbinom.sf(0,1/alpha,nbp),abs=.0001)
    assert extract(r,"p_ge3")==pytest.approx(nbinom.sf(2,1/alpha,nbp),abs=.0001)
    assert extract(r,"p0")+extract(r,"p_ge1")==pytest.approx(1,abs=.00011)
    assert extract(r,"p_ge1")>=extract(r,"p_ge2")>=extract(r,"p_ge3")
    assert extract(r,"p10")<=extract(r,"median")<=extract(r,"p90")
    assert extract(r,"recent10_sog")==[0,2,1,4]
    assert "line" not in B.FIELDS and "pick" not in B.FIELDS


def test_missing_lock_and_ledger_proof_refuses_to_publish():
    for x,y in [(False,True),(True,False),(False,False)]:
        with pytest.raises(B.BoardSafetyError, match="verification required"):
            B.derive([example()],verified_chain=x,verified_lock=y)


def test_two_teams_same_player_game_horizon_are_both_quarantined():
    original=example()
    disputed=example(team="PHI",opponent="BOS",row_hash="b"*64)
    d=B.derive([original,disputed],verified_chain=True,verified_lock=True)
    assert d["total_valid_forecasts"]==0
    assert d["excluded"]["quarantined_duplicate_identity_groups"]==1
    assert d["excluded"]["quarantined_duplicate_identity_rows"]==2
    assert d["excluded"]["cross_team_identity_conflicts"]==1
    assert d["forecasts"]["T24H"]==[]


def test_nonforecast_rows_not_rendered_as_model_predictions():
    source=example()
    late=dict(source,record_type="MISSED_CUTOFF",row_hash="f"*64)
    d=B.derive([source,late],verified_chain=True,verified_lock=True)
    assert d["total_valid_forecasts"]==1
    assert d["nonforecast_decisions"]["MISSED_CUTOFF"]==1


@pytest.mark.parametrize("changes",[
    {"source_manifest_sha256":None},
    {"availability_confidence":"CONFIRMED_DRESSED"},
    {"availability_used_in_forecast":True},
    {"schedule_state":"POSTPONED"},
    {"generated_at":"2026-10-09T19:00:01Z"},
    {"team":"BOS","opponent":"BOS"},
    {"expected_sog":-1},
    {"expected_sog":float("nan")},
    {"dispersion":0},
    {"cutoff_at":"2026-10-10T19:00:00Z"},
    {"row_hash":None},
])
def test_quarantine_unsafe_forecast_without_changing_frozen_row(changes):
    source=example(**changes)
    before=copy.deepcopy(source)
    if changes.get("row_hash") is None and "row_hash" in changes:
        with pytest.raises(B.BoardSafetyError,match="duplicate or missing original row hash"):
            B.derive([source],verified_chain=True,verified_lock=True)
    else:
        r=B.derive([source],verified_chain=True,verified_lock=True)
        assert r["total_valid_forecasts"]==0
        assert r["excluded"]["quarantined_invalid_forecasts"]==1
    assert source==before


def test_horizons_not_combined_and_games_expose_cutoffs():
    rows=[
        example(),
        example(forecast_horizon="T90",forecast_id="g-T90-p",
                generated_at="2026-10-10T17:28:00Z",cutoff_at="2026-10-10T17:30:00Z",
                row_hash="b"*64),
        example(forecast_horizon="T30",forecast_id="g-T30-p",
                generated_at="2026-10-10T18:28:00Z",cutoff_at="2026-10-10T18:30:00Z",
                row_hash="c"*64),
    ]
    d=B.derive(rows,verified_chain=True,verified_lock=True)
    assert {h:len(d["forecasts"][h]) for h in B.HORIZONS}=={"T24H":1,"T90":1,"T30":1}
    assert len(d["games"])==1
    assert set(d["games"][0]["horizons"])==set(B.HORIZONS)
    assert len(d["dates"])==1
    assert all(r[B.FIELDS.index("forecast_horizon")]==h
               for h,items in d["forecasts"].items() for r in items)


def test_ui_has_nfl_equivalent_projections_and_honest_empty_state():
    h=(Path(__file__).resolve().parents[1]/"docs"/"nhl.html").read_text()
    for required in ('nhl_v2_shadow.json','T24H','T90','T30','"mean"','"median"',
                     '"p10"','"p90"','"p_ge3"','"p0"','id="player"',
                     'id="team"','id="date"','id="game"','NOT CERTIFIED'):
        assert required in h
    assert 'fetch("nhl_predictions.json"' not in h
    assert 'pickCard(p)' not in h
    assert 'No legacy 2.5-shot UNDER picks' in h


def test_no_fake_mean_from_original_fixed_line_threshold():
    row=example(expected_sog=1.4,dispersion=.5)
    d=B.derive([row],verified_chain=True,verified_lock=True)
    assert extract(d["forecasts"]["T24H"][0],"mean")==1.4
    assert "2.5" not in d["row_fields"]
