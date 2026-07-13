"""
ksim.py - Volatility-aware Monte Carlo strikeout engine v2.
Each simulated game samples a K-rate from the pitcher's ACTUAL distribution of
recent per-start K-rates — so a volatile arm (Baz: 0.04 night to 0.32 night)
sims out genuinely wide and gets flagged NO_BET. Consistent pitchers stay tight
and bettable. Confidence comes from how decisive the distribution is about the
line (coin flips -> NO_BET). Real inputs only.
"""
import numpy as np

LEAGUE_AVG_K = 0.22

# v8.21: strikeout rate calibration. The champion systematically under-projected
# K by ~5% (mean bias -0.24). Factor derived on 2025 (development) and validated
# on the untouched 2026 holdout via pitcher_k_rate_calibration_gate_a: mean bias
# -0.2403 -> -0.0013, mean CRPS 1.29753 -> 1.29344, bootstrap P(better)=0.904,
# coverage[q10,q90] 0.864 -> 0.872 (within guard). Scales the rate entering the
# sim (season anchor + per-start pool), exactly as the validated challenger did.
K_RATE_CALIBRATION = 1.0504


def _simulate_start(k_rate_sample, expected_bf, rng):
    bf = int(round(rng.normal(expected_bf, 2.5)))
    bf = max(9, min(30, bf))
    ks = 0
    for pa_idx in range(bf):
        times_through = pa_idx // 9
        if times_through == 0:
            decay = 1.0
        elif times_through == 1:
            decay = 0.94
        else:
            decay = 0.85
        p_k = min(0.6, max(0.02, k_rate_sample * decay))
        if rng.rand() < p_k:
            ks += 1
    return ks


def simulate(k_per_bf, expected_bf, line, start_k_rates=None, sims=10000):
    """
    k_per_bf: pitcher's season K per batter faced (center / fallback).
    start_k_rates: list of recent per-start K-rates (so/bf each start). If given,
      each sim draws one — capturing real night-to-night volatility. If None,
      falls back to flat k_per_bf.
    """
    rng = np.random.RandomState()

    # v8.21: apply the validated rate calibration at the sim boundary, to both
    # the season anchor and the per-start pool (matches the formal-gate challenger).
    if K_RATE_CALIBRATION != 1.0:
        k_per_bf = k_per_bf * K_RATE_CALIBRATION
        if start_k_rates:
            start_k_rates = [r * K_RATE_CALIBRATION for r in start_k_rates]

    if start_k_rates and len(start_k_rates) >= 4:
        pool = np.array(start_k_rates, dtype=float)
        # stabilize tiny samples: blend each start toward the season mean
        pool = 0.7 * pool + 0.3 * k_per_bf
    else:
        pool = np.array([k_per_bf], dtype=float)

    results = np.empty(sims, dtype=np.int16)
    for i in range(sims):
        k_sample = pool[rng.randint(len(pool))]
        results[i] = _simulate_start(k_sample, expected_bf, rng)

    mean_ks = float(np.mean(results))
    p25, p50, p75 = np.percentile(results, [25, 50, 75])
    iqr = float(p75 - p25)
    prob_over = float(np.mean(results > line))
    prob_under = float(np.mean(results < line))
    if prob_over >= prob_under:
        side, side_prob = "OVER", prob_over
    else:
        side, side_prob = "UNDER", prob_under

    # confidence purely from decisiveness about THIS line (coin flips -> NO_BET)
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
        "mean": round(mean_ks, 2),
        "median": float(p50),
        "iqr": round(iqr, 2),
        "line": line,
        "side": side,
        "side_prob": round(side_prob, 3),
        "prob_over": round(prob_over, 3),
        "prob_under": round(prob_under, 3),
        "confidence": confidence,
        "no_bet": no_bet,
    }


if __name__ == "__main__":
    # Baz: real recent per-start K-rates (volatile: ranges 0.04 to 0.32)
    baz_rates = [0.115, 0.042, 0.321, 0.20, 0.24, 0.16, 0.22, 0.15, 0.25, 0.16]
    print("=== Baz (volatile, K/BF 0.198) at line 6.5 ===")
    r = simulate(0.198, 25.5, 6.5, start_k_rates=baz_rates)
    print(f"  mean={r['mean']} IQR={r['iqr']} {r['side']} "
          f"{int(r['side_prob']*100)}% [{r['confidence']}] no_bet={r['no_bet']}")

    steady = [0.26, 0.25, 0.27, 0.26, 0.28, 0.25, 0.26, 0.27]
    print("\n=== Steady pitcher (K/BF 0.26) at line 6.5 ===")
    r2 = simulate(0.26, 24, 6.5, start_k_rates=steady)
    print(f"  mean={r2['mean']} IQR={r2['iqr']} {r2['side']} "
          f"{int(r2['side_prob']*100)}% [{r2['confidence']}] no_bet={r2['no_bet']}")
