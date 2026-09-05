"""
cfb_rush_sim.py - Volatility-aware Monte Carlo rushing-yards engine, CFB.

Mirrors ksim.py's actual design (two-level simulation: sample this game's
workload, then sample per-touch outcomes from a real recent-outcome pool)
but adapted to what CFB's raw data actually supports: cfbfastR-data's
player_stats is genuinely PLAY-LEVEL (rush_yds = exact yards on that one
specific carry), so instead of ksim's assumed-Bernoulli-rate model (the
right choice for strikeouts, which really are one binary event per plate
appearance), this bootstrap-resamples real recorded per-carry yardage
values directly -- no distributional assumption needed for the part that
matters most.

Two-stage bootstrap per simulated game:
  1. Carry COUNT for the simulated game is drawn from one of the player's
     own recent real games (picked uniformly at random from the pool of
     recent games) -- captures real game-to-game volume swings directly,
     the same "sample from real recent variability instead of a flat
     average" idea as ksim's start_k_rates pooling.
  2. Each of those carries independently draws ONE yardage value, with
     replacement, from the POOLED set of the player's own recent
     individual carry outcomes (across all recent games, not just the
     one sampled for the count) -- a real empirical per-carry outcome
     distribution, capturing boom/bust tendency directly from what
     actually happened, no parametric shape assumed.

A player with wildly variable recent volume or per-carry outcomes sims
out genuinely wide (matching ksim's volatility-aware philosophy) without
any separate "volatility" parameter -- it falls straight out of what
real recent games actually looked like.
"""
import numpy as np


def context_adjusted_counts(counts, recent3_avg_volume, opp_allowed, projected_margin, coef):
    """Scales real recent event counts for the CURRENT matchup using a
    fitted linear model of volume ~ own recent rate + opponent context +
    game script (see cfb_rush_sim_opponent_context_a.py / cfb_volume_
    context_test_a.py for how `coef` was fit and validated per market --
    only wire this in for a market whose context-adjusted simulator
    actually passed its own CRPS/bootstrap/AUC gate against the plain
    simulator; it is NOT a generic assumption that context always helps
    -- receiving_yards and passing_yards were tested the same way and
    did NOT clear the bar).

    The per-event outcome pool (yards, or a 0/1 touchdown flag) is left
    untouched -- only how many events get simulated for this matchup
    changes, exactly as validated."""
    if recent3_avg_volume is None or recent3_avg_volume <= 0 or opp_allowed is None or projected_margin is None:
        return counts
    predicted = (coef["intercept"] + coef["recent3_avg_volume"] * recent3_avg_volume
                 + coef["opp_allowed"] * opp_allowed + coef["projected_margin"] * projected_margin)
    ratio = predicted / recent3_avg_volume
    lo, hi = coef.get("ratio_bounds", (0.4, 2.0))
    ratio = max(lo, min(hi, ratio))
    return [max(0, int(round(c * ratio))) for c in counts]


def simulate(recent_game_carry_counts, recent_carry_yards_pool, line, sims=10000, rng=None):
    """
    recent_game_carry_counts: list of ints, this player's carries in each
      of their recent real games (as-of, strictly prior to the game being
      projected) -- e.g. [15, 18, 6, 24, 12]. Drives simulated workload.
    recent_carry_yards_pool: list of ints, ALL individual real carry
      yardage values from those same recent games pooled together --
      e.g. every real carry from the last 5 games. Drives simulated
      per-carry outcomes.
    line: the prop line (e.g. 69.5).
    sims: number of simulated games.

    Returns mean/median/IQR/prob_over/prob_under, same shape as ksim.py's
    output so the two markets can share a serving-layer contract.
    """
    if rng is None:
        rng = np.random.RandomState()

    counts = np.asarray(recent_game_carry_counts, dtype=np.int64)
    pool = np.asarray(recent_carry_yards_pool, dtype=np.float64)
    if len(counts) == 0 or len(pool) == 0:
        return None

    results = np.empty(sims, dtype=np.float64)
    for i in range(sims):
        n_carries = int(counts[rng.randint(len(counts))])
        if n_carries <= 0:
            results[i] = 0.0
            continue
        draws = pool[rng.randint(0, len(pool), size=n_carries)]
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
        "samples": results,  # kept for CRPS scoring against a real holdout
    }


if __name__ == "__main__":
    # Consistent workhorse: steady carries, steady per-carry outcomes
    steady_counts = [18, 20, 17, 19, 22]
    steady_pool = [4, 3, 5, 2, 6, 4, 3, 8, 1, 4, 5, 3, 2, 9, 4] * 6
    print("=== Steady workhorse RB at line 69.5 ===")
    r = simulate(steady_counts, steady_pool, 69.5)
    print(f"  mean={r['mean']} IQR={r['iqr']} {r['side']} "
          f"{int(r['side_prob']*100)}% [{r['confidence']}] no_bet={r['no_bet']}")

    # Volatile boom/bust back: wild carry counts, boom/bust per-carry outcomes
    volatile_counts = [4, 22, 6, 26, 3]
    volatile_pool = [1, 0, -2, 1, 55, 2, 1, 0, 38, 1, -1, 2, 0, 61, 1]
    print("\n=== Volatile boom/bust RB at line 69.5 ===")
    r2 = simulate(volatile_counts, volatile_pool, 69.5)
    print(f"  mean={r2['mean']} IQR={r2['iqr']} {r2['side']} "
          f"{int(r2['side_prob']*100)}% [{r2['confidence']}] no_bet={r2['no_bet']}")
