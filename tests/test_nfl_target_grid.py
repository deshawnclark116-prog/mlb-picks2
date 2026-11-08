"""Frozen, meaningful, bookmaker-independent NFL simulation target contract."""
import numpy as np
import nfl_target_grid as G
import nfl_sim
import nfl_serving_builder_a as serving


def test_hardest_meaningful_target_with_conservative_lower_bound():
    # 75% of identical player-game draws finish at 50 yards; this is
    # a deliberately discrete synthetic distribution, not fitted odds.
    samples = np.array([50.] * 7500 + [10.] * 2500)
    grid = G.build_target_grid(samples, "receiving_yards", 40.0)
    assert grid["schema"] == G.VERSION
    assert grid["chosen_target"]["yards"] == 50
    assert grid["chosen_target"]["probability"] == .75
    assert grid["chosen_target"]["wilson_lower_95"] > .65
    assert grid["grid"][0]["yards"] >= 25
    assert all(e["yards"] >= .55 * 40 for e in grid["grid"])
    assert all(e["yards"] >= 25 for e in grid["grid"])
    assert all(e["yards"] % 5 == 0 for e in grid["grid"])
    assert grid["conditional_on_modeled_opportunities"]
    assert not grid["forward_calibrated"]


def test_reject_trivial_high_probability_thresholds_and_borderline_monte_carlo():
    low = np.array([10.] * 9900 + [25.] * 100)
    r = G.build_target_grid(low, "receiving_yards", 10.0)
    assert r["grid"][0]["yards"] == 25
    assert r["chosen_target"] is None
    # A raw 65% frequency has Wilson lower bound below 65%.
    borderline = np.array([50.] * 6500 + [10.] * 3500)
    r = G.build_target_grid(borderline, "rushing_yards", 40.0)
    assert r["chosen_target"] is None
    assert r["grid"][0]["yards"] >= 30
    assert G.wilson_lower(6500, 10000) < .65


def test_model_target_completely_independent_of_sportsbook_price_or_line():
    a = serving.make_real_odds_pick("receiving_yards", "Original Model", "GSIS1",
        "HOU", "LAC", 2026, 9, [7, 8, 8, 10], [0., 2., 8., 15., 19.],
        None, 5)
    b = serving.make_real_odds_pick("receiving_yards", "Original Model", "GSIS1",
        "HOU", "LAC", 2026, 9, [7, 8, 8, 10], [0., 2., 8., 15., 19.],
        {"line": 29.5, "over_price":-110,"under_price":-110,"book":"FanDuel"}, 5)
    assert a["nfl_model_target_grid"] == b["nfl_model_target_grid"]
    assert a["nfl_model_target"] == b["nfl_model_target"]
    assert a["nba_style_reach_probability"] == b["nba_style_reach_probability"]


def test_grid_requires_original_matched_ten_thousand_simulations():
    for values in (np.zeros(9999), np.array([1., float("nan")]+[1.]*10000)):
        try:
            G.build_target_grid(values, "receiving_yards", 70.)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid simulation accepted")


def test_probability_monotonic_and_grid_never_rounds_up_a_side():
    s = np.array([10.] * 2000 + [40.] * 4000 + [90.] * 4000)
    g = G.build_target_grid(s, "receiving_yards", 54.)
    probabilities = [x["probability"] for x in g["grid"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0 <= e["wilson_lower_95"] <= e["probability"] <= 1 for e in g["grid"])
