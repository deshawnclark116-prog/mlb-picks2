"""
ksim.py - Volatility-aware Monte Carlo strikeout engine.
Simulates a start 10,000 times plate-by-plate (TTO decay). Confidence comes from
how DECISIVE the distribution is about which side of the line we land on —
coin-flip lines get flagged NO_BET. Real inputs only (K/BF + expected BF).
"""
import numpy as np

LEAGUE_AVG_K = 0.22


def _simulate_start(pitcher_k_rate, expected_bf, lineup_k_rates, rng):
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
        p_k = pitcher_k_rate * decay
        if lineup_k_rates:
            b = lineup_k_rates[pa_idx % 9]
            num = (p_k * b) / LEAGUE_AVG_K
            den = num + ((1 - p_k) * (1 - b)) / (1 - LEAGUE_AVG_K)
            p_k = num / den if den else p_k
        p_k = min(0.6, max(0.02, p_k))
        if rng.rand() < p_k:
            ks += 1
    return ks


def simulate(pitcher_k_rate, expected_bf, line, lineup_k_rates=None, sims=10000):
    rng = np.random.RandomState()
    results = np.empty(sims, dtype=np.int16)
    for i in range(sims):
        results[i] = _simulate_start(pitcher_k_rate, expected_bf, lineup_k_rates, rng)

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
    print("=== Luzardo (0.266 K/BF, 24.3 BF) ===")
    for line in (5.5, 6.5, 7.5):
        r = simulate(0.266, 24.3, line)
        print(f"line {line}: mean={r['mean']} -> {r['side']} "
              f"{int(r['side_prob']*100)}% [{r['confidence']}]")
    print("\n=== Stable low-K pitcher (0.15 K/BF, 22 BF) ===")
    for line in (3.5, 4.5):
        r = simulate(0.15, 22, line)
        print(f"line {line}: mean={r['mean']} -> {r['side']} "
              f"{int(r['side_prob']*100)}% [{r['confidence']}]")
