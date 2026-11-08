"""NFL research-event contract: no invented exact-yard probabilities or book dependence."""
import ast
from pathlib import Path

import numpy as np
import nfl_sim


def test_own_model_projection_event_and_band_are_distinct():
    samples = np.array([0., 10., 20., 30., 40.], dtype=float)
    r = nfl_sim._summarize(samples, line=5.5)
    assert r["median"] == 20.
    assert r["whole_projection"] == 20
    assert r["projection_reach_probability"] == 0.6
    assert r["projection_within_10_probability"] == 0.6
    assert r["projection_simulation_n"] == 5
    assert r["projection_probability_source"] == "UNCALIBRATED_HISTORICAL_USAGE_BOOTSTRAP"
    # Sportsbook line may alter directional odds, NEVER model-only events.
    other = nfl_sim._summarize(samples, line=35.5)
    for field in ("whole_projection", "projection_reach_probability",
                  "projection_within_10_probability", "projection_simulation_n"):
        assert r[field] == other[field]


def test_nba_style_point_six_rounding_is_explicit_not_exact_yard_event():
    samples = np.array([13., 13.2, 14., 14.], dtype=float)
    result = nfl_sim._summarize(samples, line=100.)
    assert result["median"] == 13.6
    assert result["whole_projection"] == 14
    assert result["projection_reach_probability"] == 0.5
    # This is the chance of a TEN-YARD BAND, not P(exactly 14 yards).
    assert result["projection_within_10_probability"] == 1.0


def test_independent_nfl_simulator_runs_at_least_ten_thousand():
    source = ast.parse(Path("nfl_serving_builder_a.py").read_text())
    sizes = [a.value for a in source.body
             if isinstance(a, ast.Assign)
             for target in a.targets if isinstance(target, ast.Name)
             and target.id == "SIMS_PER_PICK"]
    assert len(sizes) == 1 and isinstance(sizes[0], ast.Constant)
    assert sizes[0].value >= 10_000


def test_simulation_distribution_fields_are_nonnegative_and_bounded():
    r = nfl_sim.simulate([0, 3, 5], [0, 2, 5, 7], 35.5,
                         sims=10_000, rng=np.random.RandomState(7))
    assert r["projection_simulation_n"] == 10_000
    assert r["mean"] >= 0 and r["median"] >= 0
    assert 0 <= r["projection_reach_probability"] <= 1
    assert 0 <= r["projection_within_10_probability"] <= 1


def test_unpriced_and_real_priced_lines_share_identical_deterministic_draws():
    args = ([1, 4, 6, 10], [0.0, 1.0, 3.0, 11.0, 0.0])
    a = nfl_sim.simulate(*args, line=0.0, sims=10_000)
    b = nfl_sim.simulate(*args, line=42.5, sims=10_000)
    c = nfl_sim.simulate(*args, line=42.5, sims=10_000)
    assert np.array_equal(a["samples"], b["samples"])
    assert np.array_equal(b["samples"], c["samples"])
    for k in ("median", "mean", "p10", "p90", "whole_projection",
              "projection_reach_probability", "projection_within_10_probability"):
        assert a[k] == b[k] == c[k]
    assert a["prob_over"] != b["prob_over"], "Only line-specific bet event can move"


def test_repeated_blended_models_stable_but_new_evidence_changes_draws():
    args=([8,10], [2.0,3.0,4.0], .6, [5,6], [1.0,0.0,11.0])
    a=nfl_sim.simulate_blended(*args, line=50.5, sims=10_000)
    b=nfl_sim.simulate_blended(*args, line=60.5, sims=10_000)
    assert np.array_equal(a["samples"], b["samples"])
    changed=nfl_sim.simulate_blended([8,12], args[1], args[2], args[3],
                                     args[4], line=50.5, sims=10_000)
    assert not np.array_equal(a["samples"], changed["samples"])


def test_nba_mean_event_is_not_median_or_ten_yard_band():
    # Skewed yards demonstrate why the NBA-style rounded MEAN must use
    # a DIFFERENT probability from the ~50% chance of reaching a median.
    samples = np.array([0., 0., 0., 0., 30.], dtype=float)
    r = nfl_sim._summarize(samples, line=24.5)
    assert r["mean"] == 6.0
    assert r["median"] == 0.0
    assert r["whole_projection"] == 0
    assert r["nba_style_whole_mean_projection"] == 6
    assert r["nba_style_reach_probability"] == 0.2
    assert r["projection_reach_probability"] == 1.0
    assert r["nba_style_probability_event"] == "P_YARDS_AT_LEAST_ROUNDED_PUBLISHED_MEAN"
    assert r["projection_within_10_probability"] == .8


def test_nba_mean_event_has_same_rounding_rule_as_app():
    samples = np.array([13., 13.2, 14., 14.], dtype=float)
    r = nfl_sim._summarize(samples, line=100.0)
    assert r["mean"] == 13.6
    assert r["nba_style_whole_mean_projection"] == 14
    assert r["nba_style_reach_probability"] == .5
    # No substitution of offered sportsbook threshold.
    other = nfl_sim._summarize(samples, line=0.5)
    assert other["nba_style_whole_mean_projection"] == 14
    assert other["nba_style_reach_probability"] == .5


def test_10k_nba_model_reach_event_independent_of_sportsbook_line():
    counts, yards = [1, 4, 6, 10], [0., 1., 3., 11., 0.]
    x = nfl_sim.simulate(counts, yards, line=1., sims=10_000)
    y = nfl_sim.simulate(counts, yards, line=99., sims=10_000)
    assert x["nba_style_whole_mean_projection"] == y["nba_style_whole_mean_projection"]
    assert x["nba_style_reach_probability"] == y["nba_style_reach_probability"]
    assert x["projection_simulation_n"] == 10_000
    assert x["nba_style_whole_mean_projection"] >= 0
    assert 0 <= x["nba_style_reach_probability"] <= 1
