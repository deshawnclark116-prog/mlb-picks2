"""
marginsim.py - Margin-of-victory Monte Carlo for run lines.
Simulates each game's run margin 10,000x using negative-binomial run scoring
(correct for overdispersed baseball runs), then reads the cover probability off
the distribution. Volatility-gated: near-coin-flip covers -> NO_BET.
"""
import numpy as np

# Negative binomial dispersion for MLB team runs. Variance/mean ~1.5-1.6 for
# runs per game empirically. We parameterize NB by (r, p): mean = r(1-p)/p.
# Given a target mean and a variance multiplier v (var = v*mean), solve for r.
def _nb_params(mean, var_mult=1.55):
    mean = max(mean, 0.3)
    var = var_mult * mean
    if var <= mean:           # safety: fall back near-Poisson
        var = mean * 1.2
    p = mean / var            # 0<p<1
    r = mean * p / (1 - p)    # number of successes
    return r, p


def _sim_runs(mean, size, rng, var_mult=1.55):
    r, p = _nb_params(mean, var_mult)
    # numpy negative_binomial(n, p) counts failures before n successes; its
    # mean is n*(1-p)/p — matches our r,p parameterization.
    return rng.negative_binomial(r, p, size=size)


def simulate_runline(home_exp, away_exp, home_line, sims=10000, var_mult=1.55):
    """home_line is the home team's spread (e.g. -1.5 if home favored, +1.5 if dog).
    Returns cover probabilities for both sides + volatility-gated confidence."""
    rng = np.random.RandomState()
    home_runs = _sim_runs(home_exp, sims, rng, var_mult)
    away_runs = _sim_runs(away_exp, sims, rng, var_mult)
    # break ties (baseball has no ties): give the higher-expected team the edge
    # by adding a tiny extra-innings sim — simplest: re-roll ties as a coin flip
    margin = home_runs - away_runs            # positive = home wins by that много
    ties = margin == 0
    if ties.any():
        # extra innings: whoever has higher expected runs more likely wins
        ph = home_exp / (home_exp + away_exp)
        coin = rng.rand(int(ties.sum())) < ph
        margin[ties] = np.where(coin, 1, -1)

    # home covers home_line if (home_margin) > -home_line
    # e.g. home -1.5: covers if margin > 1.5 -> win by 2+
    # e.g. home +1.5: covers if margin > -1.5 -> lose by 1 or win
    home_cover = float(np.mean(margin > -home_line))
    away_cover = 1.0 - home_cover

    if home_cover >= away_cover:
        side, side_prob = "home", home_cover
    else:
        side, side_prob = "away", away_cover

    # volatility gate: decisiveness about the cover
    if side_prob >= 0.62:
        confidence = "HIGH"; no_bet = False
    elif side_prob >= 0.58:
        confidence = "MEDIUM"; no_bet = False
    elif side_prob >= 0.55:
        confidence = "LOW"; no_bet = False
    else:
        confidence = "NO_BET"; no_bet = True

    return {
        "home_cover": round(home_cover, 3),
        "away_cover": round(away_cover, 3),
        "side": side,
        "side_prob": round(side_prob, 3),
        "mean_margin": round(float(np.mean(margin)), 2),
        "confidence": confidence,
        "no_bet": no_bet,
    }


def expected_runs(home_team, away_team, run_table):
    """Blend offense and opponent defense for each side's expected runs."""
    h = run_table.get(home_team); a = run_table.get(away_team)
    if not h or not a:
        return None
    home_exp = (h["rs_pg"] + a["ra_pg"]) / 2.0
    away_exp = (a["rs_pg"] + h["ra_pg"]) / 2.0
    return home_exp, away_exp


if __name__ == "__main__":
    # TEST: Dodgers (home, big favorite) vs Giants, home -1.5
    import gamelines
    rt = gamelines.team_run_table()
    exp = expected_runs("Los Angeles Dodgers", "San Francisco Giants", rt)
    print("Expected runs (Dodgers, Giants):", [round(x,2) for x in exp])
    print("\n-- Dodgers -1.5 (favorite covering by 2+) --")
    r = simulate_runline(exp[0], exp[1], -1.5)
    print(r)
    print("\n-- Giants +1.5 (dog, lose by 1 or win) --")
    print(f"away_cover (Giants +1.5) = {r['away_cover']}")
    print("\n-- Even matchup test: Rays vs Pirates, home -1.5 --")
    exp2 = expected_runs("Pittsburgh Pirates", "Tampa Bay Rays", rt)
    r2 = simulate_runline(exp2[0], exp2[1], -1.5)
    print("Expected:", [round(x,2) for x in exp2], "->", r2)
