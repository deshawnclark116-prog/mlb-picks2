"""
ksim.py - Volatility-aware Monte Carlo strikeout engine.
Simulates a start 10,000 times plate-by-plate (TTO decay included), then uses
the SPREAD of the outcome distribution to gate confidence. Volatile pitchers
(wide distribution, near coin-flip on the line) are flagged no-bet.

Real inputs only: pitcher's K-per-batter-faced and expected batters faced —
both already computed in api.py's pitcher_feature_row.
"""
import numpy as np

LEAGUE_AVG_K = 0.22


def _simulate_start(pitcher_k_rate, expected_bf, lineup_k_rates=None, rng=None):
    """One simulated start. Returns total strikeouts.
    If lineup_k_rates given (9 values), uses Log5 per batter; otherwise uses the
    pitcher's flat rate (we proved lineup K-rate adds ~no signal, so flat is the
    honest default, but the hook stays for future use)."""
    if rng is None:
        rng = np.random
    # batters faced varies start to start — normal around expected, clamped
    bf = int(round(rng.normal(expected_bf, 2.5)))
    bf = max(9, min(30, bf))
    ks = 0
    for pa_idx in range(bf):
        times_through = pa_idx // 9
        # TTO decay: pitcher loses effectiveness each time through the order
        if times_through == 0:
            decay = 1.0
        elif times_through == 1:
            decay = 0.94
        else:
            decay = 0.85
        p_k = pitcher_k_rate * decay
        if lineup_k_rates:
            # Log5 matchup vs this lineup slot (kept available, not used by default)
            b = lineup_k_rates[pa_idx % 9]
            num = (p_k * b) / LEAGUE_AVG_K
            den = num + ((1 - p_k) * (1 - b)) / (1 - LEAGUE_AVG_K)
            p_k = num / den if den else p_k
        p_k = min(0.6, max(0.02, p_k))
        if rng.rand() < p_k:
            ks += 1
    return ks


def simulate(pitcher_k_rate, expected_bf, line, lineup_k_rates=None, sims=10000):
    """Run the sim and return a dict with the distribution, over/under prob,
    and a volatility-based confidence / no-bet flag."""
    rng = np.random.RandomState()  # fresh stream
    results = np.empty(sims, dtype=np.int16)
    for i in range(sims):
        results[i] = _simulate_start(pitcher_k_rate, expected_bf, lineup_k_rates, rng)

    mean_ks = float(np.mean(results))
    p25, p50, p75 = np.percentile(results, [25, 50, 75])
    iqr = float(p75 - p25)

    prob_over = float(np.mean(results > line))
    prob_under = float(np.mean(results < line))
    # the side the sim favors, and how strongly
    if prob_over >= prob_under:
        side, side_prob = "OVER", prob_over
    else:
        side, side_prob = "UNDER", prob_under

    # ── volatility gate ──
    # how decisive is the sim about which side of the line? near 0.5 = coin flip.
    # Also factor the raw spread: a wide IQR relative to the mean = volatile arm.
    decisiveness = side_prob                      # 0.5 (coin flip) .. 1.0 (certain)
    rel_spread = iqr / mean_ks if mean_ks > 0 else 1.0

    # confidence tiers from the distribution itself
    if decisiveness >= 0.68 and rel_spread <= 0.55:
        confidence = "HIGH"; no_bet = False
    elif decisiveness >= 0.62 and rel_spread <= 0.70:
        confidence = "MEDIUM"; no_bet = False
    elif decisiveness >= 0.58:
        confidence = "LOW"; no_bet = False
    else:
        confidence = "NO_BET"; no_bet = True      # too close to a coin flip

    return {
        "mean": round(mean_ks, 2),
        "median": float(p50),
        "iqr": round(iqr, 2),
        "rel_spread": round(rel_spread, 3),
        "line": line,
        "side": side,
        "side_prob": round(side_prob, 3),
        "prob_over": round(prob_over, 3),
        "prob_under": round(prob_under, 3),
        "confidence": confidence,
        "no_bet": no_bet,
    }


if __name__ == "__main__":
    # TEST on real Luzardo numbers: 0.266 K/BF, 24.3 BF
    print("=== Luzardo (0.266 K/BF, 24.3 BF) ===")
    for line in (5.5, 6.5, 7.5):
        r = simulate(0.266, 24.3, line)
        print(f"line {line}: mean={r['mean']} IQR={r['iqr']} "
              f"rel_spread={r['rel_spread']} -> {r['side']} "
              f"{int(r['side_prob']*100)}% [{r['confidence']}]")
    # a STABLE low-K contact pitcher for contrast
    print("\n=== Stable low-K pitcher (0.15 K/BF, 22 BF) ===")
    for line in (3.5, 4.5):
        r = simulate(0.15, 22, line)
        print(f"line {line}: mean={r['mean']} IQR={r['iqr']} "
              f"rel_spread={r['rel_spread']} -> {r['side']} "
              f"{int(r['side_prob']*100)}% [{r['confidence']}]")
