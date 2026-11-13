"""Research-only NFL model-derived yardage targets from unchanged Monte Carlo draws.

Never infer P(hit) from a mean, median, sportsbook odds or a hand-tuned
normal distribution. This operates on the SAME sampled numeric outcomes
as the weekly player's projected moments.

A meaningful event has a predeclared absolute floor (receiving 25 yards,
rushing 30 yards) AND a proportional floor (>=55% of projected mean).
The selected event is the HARDEST target whose one-sided Wilson 95%
sampling lower bound clears 60%. This avoids selecting arbitrarily easy
5-yard 'high-confidence' outcomes, and prevents 10k Monte Carlo noise
from deciding a near-threshold pass. All rules are fixed BEFORE seeing
future results. Not forward calibrated. No DNP/active model.
"""
from __future__ import annotations

import math
import numpy as np

VERSION = "nfl_meaningful_yardage_targets_v1"
MIN_YARDS = {"receiving_yards": 25, "rushing_yards": 30}
GRID_STEP = 5
RELATIVE_MIN = 0.55
MODEL_MIN_P = 0.60
Z_ONE_SIDED_95 = 1.6448536269514722
MAX_GRID_YARDS = 300


def wilson_lower(successes: int, trials: int) -> float:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid Monte Carlo count")
    n = float(trials)
    z = Z_ONE_SIDED_95
    p = successes / n
    return max(0.0, (p + z * z / (2 * n) -
                      z * math.sqrt((p * (1-p) + z*z/(4*n)) / n)) /
               (1 + z*z/n))


def build_target_grid(results, market: str, published_mean: float) -> dict:
    if market not in MIN_YARDS:
        raise ValueError("unsupported NFL yardage market")
    arr = np.asarray(results, dtype=float)
    if arr.ndim != 1 or len(arr) < 10_000 or not np.all(np.isfinite(arr)):
        raise ValueError("10000+ finite original model draws required")
    if not math.isfinite(published_mean) or published_mean < 0:
        raise ValueError("invalid original model mean")
    n = int(len(arr))
    min_allowed = max(MIN_YARDS[market], math.ceil(RELATIVE_MIN * published_mean))
    start = GRID_STEP * math.ceil(min_allowed / GRID_STEP)
    # Publicly fixed finite horizon avoids accidental unbounded output.
    # The observed P90 only limits useless extreme-grid entries.
    p90 = float(np.percentile(arr, 90))
    limit = min(MAX_GRID_YARDS, GRID_STEP * math.ceil(max(start, p90) / GRID_STEP))
    grid = []
    chosen = None
    for target in range(start, limit + 1, GRID_STEP):
        successes = int(np.count_nonzero(arr >= target))
        observed = successes / n
        lower = wilson_lower(successes, n)
        event = {"yards": target,
                 "probability": round(observed, 4),
                 "wilson_lower_95": round(lower, 4)}
        grid.append(event)
        # Evaluate full-precision values rather than rounded display fields.
        if lower >= MODEL_MIN_P:
            chosen = event

    return {
        "schema": VERSION,
        "simulation_n": n,
        "minimum_absolute_yards": MIN_YARDS[market],
        "minimum_relative_mean_fraction": RELATIVE_MIN,
        "required_wilson_lower_95": MODEL_MIN_P,
        "grid_step_yards": GRID_STEP,
        "conditional_on_modeled_opportunities": True,
        "forward_calibrated": False,
        "official_active_status_verified": False,
        "chosen_target": chosen,
        "grid": grid,
    }
