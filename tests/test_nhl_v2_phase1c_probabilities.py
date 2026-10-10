"""Focused Phase1C probability tests; no frozen engine mutation or model fitting."""
import copy
import math

import pytest
from scipy.stats import nbinom

import nhl_v2_phase1c_probabilities as P


def row(mu=2.2, alpha=0.4):
    r = 1 / alpha
    p = r / (r + mu)
    return {
        "record_type": "FORECAST", "forecast_id": "immutable-row-A", "game_id": 2026020001,
        "forecast_horizon": "T90", "cutoff_at": "2026-10-10T20:00:00Z",
        "expected_sog": mu, "dispersion": alpha,
        **{f"P{k}": 1.0 if k == 1 else float(nbinom.sf(k - 2, r, p)) for k in range(1, 6)},
    }


def test_exact_independent_nb2_thresholds():
    mu, alpha = 2.2, 0.4
    r = 1 / alpha
    p = r / (r + mu)
    result = P.nb2_thresholds(mu, alpha)
    assert result["P1"] == pytest.approx(1 - p ** r, abs=1e-13)
    assert result["P2"] == pytest.approx(1 - p ** r - r * (1 - p) * p ** r, abs=1e-13)
    assert result["P3"] == pytest.approx(nbinom.sf(2, r, p), abs=1e-13)
    assert all(0 <= result[f"P{k+1}"] <= result[f"P{k}"] <= 1 for k in range(1, 5))


def test_old_frozen_fields_identified_without_mutation():
    original = row()
    saved = copy.deepcopy(original)
    derived = P.interpretation(original)
    assert derived["legacy_mismatch_thresholds"] == [1, 2, 3, 4, 5]
    assert derived["legacy_fields_policy"] == P.UNSAFE_LEGACY
    assert derived["historical_pregame_publication_claimed"] is False
    assert derived["corrected_posthoc"]["P1"] < 1
    assert original == saved


def test_corrected_fields_already_correct_do_not_trigger_false_defect():
    original = row()
    original.update(P.nb2_thresholds(original["expected_sog"], original["dispersion"]))
    assert P.interpretation(original)["legacy_mismatch_thresholds"] == []


def test_zero_mean_makes_all_survival_probabilities_zero():
    assert set(P.nb2_thresholds(0.0, 0.4).values()) == {0.0}


@pytest.mark.parametrize("mu,alpha", [(float("nan"), 0.4), (2., 0.),
                                        (2., -1.), (-1., 0.3), (2., float("inf")),
                                        (True, .2), (2., True)])
def test_reject_invalid_distribution_inputs(mu, alpha):
    with pytest.raises(P.ProbabilityIntegrityError):
        P.nb2_thresholds(mu, alpha)


def test_refuse_non_forecast_and_missing_required_fields():
    with pytest.raises(P.ProbabilityIntegrityError):
        P.interpretation({"record_type": "MISSED_WINDOW"})
    broken = row()
    broken.pop("dispersion")
    with pytest.raises(P.ProbabilityIntegrityError):
        P.interpretation(broken)


def test_cross_team_collisions_preserved_and_quarantined_not_deduplicated():
    a = row()
    a.update(row_hash="f" * 64, team="BOS", player_id=1001)
    b = copy.deepcopy(a)
    b.update(row_hash="e" * 64, team="NYR")
    r = P.audit([a, b, {"record_type": "MISSED_WINDOW"}])
    assert r["n_forecasts_checked"] == 2
    assert r["n_unique_forecast_ids"] == 1
    assert r["n_forecast_id_collision_groups"] == 1
    assert r["n_collision_rows"] == 2
    assert len(r["forecast_id_collisions_quarantined_from_unique_player_joins"][a["forecast_id"]]) == 2
    assert r["legacy_shifted_threshold_mismatches"] == {f"P{k}": 2 for k in range(1, 6)}
    assert r["frozen_engine_or_ledger_modified"] is False


def test_duplicate_immutable_row_hash_refused():
    a = row()
    a["row_hash"] = "f" * 64
    b = copy.deepcopy(a)
    b["forecast_id"] = "distinct-player-key"
    with pytest.raises(P.ProbabilityIntegrityError, match="duplicate immutable row_hash"):
        P.audit([a, b])
