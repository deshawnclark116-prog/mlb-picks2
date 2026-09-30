"""
NFL_PHASE1C_METRICS  (Phase 1C, shadow research)

Distribution summaries, accuracy curves, PIT / reliability, calibration maps and the predictability score.  Pure numpy; no data access.
"""
import numpy as np

QS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
GRID19 = tuple(np.round(np.arange(0.05, 0.951, 0.05), 2))
TOL = {"rush_yds": (5, 10, 15, 20, 25, 30, 35, 40), "rec_yds": (5, 10, 15, 20, 25, 30, 35, 40), "pass_yds": (10, 20, 30, 40, 50, 60, 75, 100),
       "rush_td": (0, 1), "rec_td": (0, 1), "pass_td": (0, 1, 2), "int": (0, 1, 2), "rec": (0, 1, 2, 3), "tackles": (0, 1, 2, 3, 4), "sacks": (0, 0.5, 1), "def_int": (0, 1)}
CONTINUOUS = ("rush_yds", "rec_yds", "pass_yds")


def summary_row(S):
    """S [n, N] draws -> dict of per-row arrays: mean, median, sd, quantiles."""
    out = {"mean": S.mean(1), "sd": S.std(1), "median": np.median(S, 1)}
    q = np.quantile(S, QS, axis=1)
    for p, v in zip(QS, q):
        out[f"p{int(round(p * 100)):02d}"] = v
    return out


def pit_randomized(S, y, seed=3):
    """Randomized PIT for discrete / continuous forecasts represented by draws."""
    rng = np.random.default_rng(seed)
    hi = (S <= y[:, None]).mean(1)
    lo = (S < y[:, None]).mean(1)
    return lo + rng.random(len(y)) * (hi - lo)


def coverage_from_pit(u, level):
    a = (1 - level) / 2
    return float(np.mean((u >= a) & (u <= 1 - a)))


def pit_ks(u):
    xs = np.linspace(0.1, 0.9, 9)
    return float(max(abs(np.mean(u <= x) - x) for x in xs))


def accuracy_curve(pred, y, tols):
    err = np.abs(pred - y)
    return {str(t): round(float((err <= t + 1e-9).mean()), 4) for t in tols}


def interval_stats(S, y, levels=(0.5, 0.8, 0.9), seed=3):
    """Discreteness-correct coverage (randomized PIT) plus the naive quantile-interval coverage and mean width."""
    u = pit_randomized(S, y, seed)
    out = {}
    for lv in levels:
        a = (1 - lv) / 2
        lo, hi = np.quantile(S, a, axis=1), np.quantile(S, 1 - a, axis=1)
        out[str(lv)] = {"pit_coverage": round(coverage_from_pit(u, lv), 4), "interval_coverage": round(float(((y >= lo) & (y <= hi)).mean()), 4),
                        "mean_width": round(float((hi - lo).mean()), 4)}
    out["pit_ks"] = round(pit_ks(u), 4)
    return out


def brier_ece(p, y, bins=10):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    br = float(((p - y) ** 2).mean())
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9
    ece, rel = 0.0, []
    for i in range(bins):
        m = (p >= q[i]) & (p < q[i + 1])
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
        rel.append({"pred": round(float(p[m].mean()), 4), "obs": round(float(y[m].mean()), 4), "n": int(m.sum())})
    return {"logloss": round(ll, 4), "brier": round(br, 4), "ece": round(float(ece), 4), "reliability": rel}


# ------------------------------------------------------------------ calibration maps applied to draws
def fit_pit_map(u, knots=101):
    xs = np.linspace(0, 1, knots)
    G = np.maximum.accumulate(np.array([np.mean(u <= x) for x in xs]))
    G[0], G[-1] = 0.0, 1.0
    return {"xs": xs, "G": G}


def apply_pit_map(S, m, seed=5):
    """Draw from the recalibrated distribution F' = G(F): u = G^{-1}(v), x = empirical quantile of the row's draws at u."""
    n, N = S.shape
    rng = np.random.default_rng(seed)
    v = (np.arange(N) + 0.5) / N
    Ginv = np.interp(v, m["G"], m["xs"])                       # monotone inverse (G is non-decreasing)
    Ss = np.sort(S, 1)
    idx = np.minimum((Ginv * N).astype(int), N - 1)
    out = Ss[:, idx]
    return out[:, rng.permutation(N)]


def fit_scale_map(S, y, target=0.8, integer=False):
    """Dispersion scaling about the row median: choose tau so that PIT-free interval coverage of the scaled draws equals the target."""
    med = np.median(S, 1)[:, None]
    lo, hi = 0.3, 3.0
    for _ in range(30):
        tau = 0.5 * (lo + hi)
        Z = med + tau * (S - med)
        a = (1 - target) / 2
        cov = float(((y >= np.quantile(Z, a, axis=1)) & (y <= np.quantile(Z, 1 - a, axis=1))).mean())
        lo, hi = (lo, tau) if cov > target else (tau, hi)
    return {"tau": 0.5 * (lo + hi)}


def apply_scale_map(S, m, nonneg=True):
    med = np.median(S, 1)[:, None]
    Z = med + m["tau"] * (S - med)
    return np.maximum(Z, 0) if nonneg else Z


def fit_conformal_map(S, y, level=0.8):
    """Split-conformal widening: nonconformity = max(q_lo - y, y - q_hi); add its (1-alpha)-quantile to both ends (applied to draws as a shift of the tails)."""
    a = (1 - level) / 2
    lo, hi = np.quantile(S, a, axis=1), np.quantile(S, 1 - a, axis=1)
    score = np.maximum(lo - y, y - hi)
    n = len(score)
    q = float(np.quantile(score, min(1.0, level * (1 + 1.0 / n))))
    return {"q": q, "level": level}


def apply_conformal_map(S, m, nonneg=True):
    med = np.median(S, 1)[:, None]
    Z = S + np.sign(S - med) * max(m["q"], 0.0)                  # push draws away from the median by the conformal margin
    return np.maximum(Z, 0) if nonneg else Z


# ------------------------------------------------------------------ predictability score
def predictability(S, pact, opp_cv, scale0, role_shift=None):
    """Uncertainty value U in [0, 1) (higher = less predictable) and reason codes, from the simulated distribution and pregame quantities only.

    U = the engine's OWN expected relative error, E|X - median| / (|mean| + scale0), mapped through x/(1+x). The simulation already integrates availability,
    opportunity volatility, efficiency variance and role competition, so U needs no ad-hoc weights. Reason codes rank the pregame drivers
    (availability risk, opportunity volatility, role change, spread) for each row. The first version of this score (a weighted sum of components) was
    ANTI-informative on development data (Spearman -0.53 vs relative error) and was replaced before any adoption; see README."""
    med = np.median(S, 1); mean = S.mean(1)
    rel = np.abs(S - med[:, None]).mean(1) / (np.abs(mean) + scale0)
    U = rel / (1.0 + rel)
    comp = {"availability_risk": 1.0 - np.clip(pact, 0, 1), "opportunity_volatility": np.clip(opp_cv / (1.0 + opp_cv), 0, 1), "spread": U}
    if role_shift is not None:
        comp["role_change"] = np.clip(role_shift / (0.1 + role_shift), 0, 1)
    names = list(comp)
    M = np.column_stack([comp[k] for k in names])
    top = M.argsort(1)[:, ::-1][:, :2]
    reasons = [[names[j] for j in row if M[i, j] > 0.1] for i, row in enumerate(top)]
    return U, reasons


def decile_table(U, err_rel, n_bins=10):
    q = np.quantile(U, np.linspace(0, 1, n_bins + 1)); q[-1] += 1e-9
    rows = []
    for i in range(n_bins):
        m = (U >= q[i]) & (U < q[i + 1])
        if m.sum():
            rows.append({"decile": i + 1, "n": int(m.sum()), "mean_U": round(float(U[m].mean()), 4), "mean_rel_abs_error": round(float(err_rel[m].mean()), 4)})
    vals = [r["mean_rel_abs_error"] for r in rows]
    viol = sum(1 for a, b in zip(vals, vals[1:]) if b < a - 1e-9)
    from math import sqrt
    rk = lambda a: np.argsort(np.argsort(a)).astype(float)
    rho = float(np.corrcoef(rk(U), rk(err_rel))[0, 1])
    return {"deciles": rows, "monotonic_violations": viol, "spearman_U_vs_error": round(rho, 4)}


def spread_skill(sd, err_abs, n_bins=10):
    """Predicted sd deciles vs realized mean absolute error of the median (absolute units)."""
    q = np.quantile(sd, np.linspace(0, 1, n_bins + 1)); q[-1] += 1e-9
    rows = []
    for i in range(n_bins):
        m = (sd >= q[i]) & (sd < q[i + 1])
        if m.sum():
            rows.append({"decile": i + 1, "n": int(m.sum()), "mean_pred_sd": round(float(sd[m].mean()), 4), "mean_abs_error": round(float(err_abs[m].mean()), 4)})
    vals = [r["mean_abs_error"] for r in rows]
    rk = lambda a: np.argsort(np.argsort(a)).astype(float)
    return {"deciles": rows, "monotonic_violations": sum(1 for a, b in zip(vals, vals[1:]) if b < a - 1e-9), "spearman_sd_vs_abs_error": round(float(np.corrcoef(rk(sd), rk(err_abs))[0, 1]), 4)}
