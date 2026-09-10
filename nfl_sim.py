"""
nfl_sim.py - Volatility-aware Monte Carlo rushing/receiving-yards engine, NFL.

Identical design and math to cfb_rush_sim.py (this repo's proven,
already-shipped CFB Monte Carlo simulator) -- copied rather than
cross-imported, matching this repo's convention of each sport keeping
its own model code even when the underlying math is shared.

Two-stage bootstrap per simulated game:
  1. Event COUNT (carries, or targets) for the simulated game is drawn
     from one of the player's own recent real games.
  2. Each of those events independently draws ONE yardage value, with
     replacement, from the POOLED set of the player's own recent real
     per-event outcomes (a carry's real rushing_yards, or a target's
     real receiving_yards -- 0 for an incompletion, a real recorded
     empirical outcome either way -- see nfl_pbp_foundation_a.py).

No distributional assumption for either stage -- both are real,
recorded per-play NFL data (nflverse's play-by-play release).
"""
import numpy as np


def simulate(recent_game_event_counts, recent_event_yards_pool, line, sims=10000, rng=None):
    """
    recent_game_event_counts: list of ints, this player's carries (or
      targets) in each of their recent real games (as-of, strictly
      prior to the game being projected).
    recent_event_yards_pool: list of floats, ALL individual real
      per-event yardage values from those same recent games pooled
      together.
    line: the prop line.
    sims: number of simulated games.

    Returns mean/median/IQR/prob_over/prob_under, same shape as
    cfb_rush_sim.py's simulate() so downstream code is identical.
    """
    if rng is None:
        rng = np.random.RandomState()

    counts = np.asarray(recent_game_event_counts, dtype=np.int64)
    pool = np.asarray(recent_event_yards_pool, dtype=np.float64)
    if len(counts) == 0 or len(pool) == 0:
        return None

    results = np.empty(sims, dtype=np.float64)
    for i in range(sims):
        n_events = int(counts[rng.randint(len(counts))])
        if n_events <= 0:
            results[i] = 0.0
            continue
        draws = pool[rng.randint(0, len(pool), size=n_events)]
        results[i] = draws.sum()

    mean_yds = float(np.mean(results))
    p10, p25, p50, p75, p90 = np.percentile(results, [10, 25, 50, 75, 90])
    iqr = float(p75 - p25)
    prob_over = float(np.mean(results > line))
    prob_under = float(np.mean(results < line))
    if prob_over >= prob_under:
        side, side_prob = "OVER", prob_over
    else:
        side, side_prob = "UNDER", prob_under

    decisiveness = side_prob
    if decisiveness >= 0.70:
        confidence = "HIGH"; no_bet = False
    elif decisiveness >= 0.64:
        confidence = "MEDIUM"; no_bet = False
    elif decisiveness >= 0.59:
        confidence = "LOW"; no_bet = False
    else:
        confidence = "NO_BET"; no_bet = True

    return {
        "mean": round(mean_yds, 1),
        "median": float(p50),
        "iqr": round(iqr, 1),
        "p10": float(p10), "p90": float(p90),
        "line": line,
        "side": side,
        "side_prob": round(side_prob, 3),
        "prob_over": round(prob_over, 3),
        "prob_under": round(prob_under, 3),
        "confidence": confidence,
        "no_bet": no_bet,
        "samples": results,
    }
