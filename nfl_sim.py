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


def _summarize(results, line):
    """Shared post-processing for simulate()/simulate_blended() -- same
    sims-array-in, stats-dict-out shape either way."""
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


def _draw_one_game(counts, pool, rng):
    n_events = int(counts[rng.randint(len(counts))])
    if n_events <= 0:
        return 0.0
    draws = pool[rng.randint(0, len(pool), size=n_events)]
    return float(draws.sum())


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
        results[i] = _draw_one_game(counts, pool, rng)

    return _summarize(results, line)


def simulate_blended(current_counts, current_pool, current_weight,
                      prior_counts, prior_pool, line, sims=10000, rng=None):
    """
    Same two-stage bootstrap as simulate(), but each simulated game first
    picks which real season to draw its game-count/pool from: the
    player's current season (probability current_weight) or their prior
    season (probability 1 - current_weight) -- rather than the old
    all-or-nothing switch (100% prior season through week 3, then
    instantly 100% current season at week 4, whatever the current
    season's sample size).

    current_weight should typically be min(1.0, n_current_games /
    RECENT_GAMES_WINDOW) at the call site: 0 with no current-season
    games yet (pure prior season, same as simulate() on prior_counts/
    prior_pool alone), rising toward 1.0 as real current-season games
    accumulate, reaching pure current-season once there's a full
    window's worth -- so a real, brand-new signal (this year's opener)
    gets partial weight immediately instead of either being ignored
    entirely or, a few weeks later, suddenly outweighing an entire real
    prior season on the strength of 3 games.

    Falls back to a plain simulate() on whichever side actually has data
    when the other side is completely empty (a rookie with no prior
    season, or a prior-season player cut and not yet appearing this
    season) rather than dividing by a mix that doesn't exist.
    """
    if rng is None:
        rng = np.random.RandomState()

    have_current = current_weight > 0 and len(current_counts) > 0 and len(current_pool) > 0
    have_prior = current_weight < 1 and len(prior_counts) > 0 and len(prior_pool) > 0

    if not have_current and not have_prior:
        return None
    if not have_current:
        return simulate(prior_counts, prior_pool, line, sims=sims, rng=rng)
    if not have_prior:
        return simulate(current_counts, current_pool, line, sims=sims, rng=rng)

    cur_counts = np.asarray(current_counts, dtype=np.int64)
    cur_pool = np.asarray(current_pool, dtype=np.float64)
    pri_counts = np.asarray(prior_counts, dtype=np.int64)
    pri_pool = np.asarray(prior_pool, dtype=np.float64)

    coin = rng.random_sample(sims) < current_weight
    results = np.empty(sims, dtype=np.float64)
    for i in range(sims):
        if coin[i]:
            results[i] = _draw_one_game(cur_counts, cur_pool, rng)
        else:
            results[i] = _draw_one_game(pri_counts, pri_pool, rng)

    return _summarize(results, line)
