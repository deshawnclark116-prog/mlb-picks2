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
