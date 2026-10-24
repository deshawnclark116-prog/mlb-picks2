"""NHL V2 Phase0 pure metric helpers (numpy only, deterministic). Research-only."""
import math

import numpy as np

EPS = 1e-6


def R(x, nd=6):
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd)


def brier(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    n1 = float(y.sum()); n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return None
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg_rank = csum - (cnt - 1) / 2.0
    ranks = avg_rank[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pr_auc(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    if y.sum() == 0:
        return None
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def reliability(p, y, bins=10):
    p = np.asarray(p, float); y = np.asarray(y, float)
    out = []; ece = 0.0; n = len(y)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo)
        c = int(m.sum())
        if c == 0:
            continue
        mp, my = float(p[m].mean()), float(y[m].mean())
        ece += abs(mp - my) * c / n
        out.append({"bin": "%.1f-%.1f" % (lo, hi), "n": c, "mean_pred": R(mp), "outcome_rate": R(my)})
    return out, float(ece)


def calib_slope_intercept(p, y, iters=50):
    """Logistic recalibration y ~ a + b*logit(p) by Newton; None when degenerate."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); y = np.asarray(y, float)
    if y.sum() == 0 or y.sum() == len(y) or np.std(p) < 1e-9:
        return None, None
    x = np.log(p / (1 - p))
    a, b = 0.0, 1.0
    for _ in range(iters):
        z = np.clip(a + b * x, -30.0, 30.0)
        q = 1 / (1 + np.exp(-z))
        w = q * (1 - q)
        g = np.array([np.sum(y - q), np.sum((y - q) * x)])
        H = np.array([[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None, None
        a += step[0]; b += step[1]
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(a), float(b)


def prob_metrics(p, y, base_rate_fit):
    p = np.asarray(p, float); y = np.asarray(y, float)
    rel, ece = reliability(p, y)
    a, b = calib_slope_intercept(p, y)
    br = brier(p, y)
    br_const = brier(np.full(len(y), base_rate_fit), y)
    qs = np.quantile(p, [0, .1, .25, .5, .75, .9, 1.0])
    return {
        "n": int(len(y)), "event_base_rate": R(y.mean()), "fit_base_rate": R(base_rate_fit),
        "brier": R(br), "brier_const_fit_rate": R(br_const),
        "brier_skill_vs_fit_rate": R(1 - br / br_const) if br_const > 0 else None,
        "log_loss": R(logloss(p, y)), "ece_10bin": R(ece),
        "calibration_intercept": R(a), "calibration_slope": R(b),
        "roc_auc_secondary": R(auc(p, y)), "pr_auc": R(pr_auc(p, y)),
        "pred_mean": R(p.mean()),
        "pred_quantiles_min_p10_p25_p50_p75_p90_max": [R(v) for v in qs],
        "sharpness_var_of_pred": R(np.var(p)),
        "reliability_table": rel,
    }


def count_metrics(mu, actual, within=(0.5, 1, 2), miss=(2, 3, 4)):
    mu = np.asarray(mu, float); a = np.asarray(actual, float)
    e = mu - a; ae = np.abs(e); n = len(a)
    d = {"n": int(n), "mae": R(ae.mean()), "bias_pred_minus_actual": R(e.mean()),
         "median_ae": R(np.median(ae)), "rmse": R(math.sqrt(float(np.mean(e ** 2))))}
    for w in within:
        d["within_%s" % w] = R((ae <= w).mean())
    for m in miss:
        d["miss_gt_%s" % m] = R((ae > m).mean())
        d["miss_gt_%s_n" % m] = int((ae > m).sum())
    return d


def nb_tail_ge(mu, k, r=None):
    """P(X>=k) for Poisson(mu) (r None) or NegBin(mean mu, size r). Closed form, no sampling."""
    mu = float(mu)
    if mu <= 0:
        return 0.0 if k > 0 else 1.0
    if r is None:
        lp = -mu; term = math.exp(lp); cdf = 0.0
        for j in range(0, k):
            cdf += term
            term *= mu / (j + 1)
        return float(min(1.0, max(0.0, 1.0 - cdf)))
    pr = r / (r + mu)
    term = pr ** r; cdf = 0.0
    for j in range(0, k):
        cdf += term
        term *= (j + r) / (j + 1) * (1 - pr)
    return float(min(1.0, max(0.0, 1.0 - cdf)))


def dispersion(mu, y):
    """Returns (index, r): index = sum((y-mu)^2)/sum(mu); r = moment NB size or None."""
    mu = np.asarray(mu, float); y = np.asarray(y, float)
    idx = float(np.sum((y - mu) ** 2) / np.sum(mu))
    den = float(np.sum((y - mu) ** 2 - mu))
    r = float(np.sum(mu ** 2) / den) if den > 1e-9 else None
    return idx, r


def map_mu_to_prob(mu, k, dist):
    r = dist.get("r") if dist.get("family") == "negbin" else None
    return np.array([nb_tail_ge(m, k, r) for m in mu], float)


def paired_block_bootstrap(diff, blocks, n_boot=2000, seed=20261007):
    """95% CI of mean(diff) resampling whole blocks (game dates)."""
    diff = np.asarray(diff, float)
    _, inv = np.unique(np.asarray(blocks), return_inverse=True)
    nb = int(inv.max()) + 1
    s = np.zeros(nb); c = np.zeros(nb)
    np.add.at(s, inv, diff); np.add.at(c, inv, 1.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(n_boot, nb))
    m = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return {"mean": R(diff.mean()), "ci95_lo": R(np.quantile(m, 0.025)), "ci95_hi": R(np.quantile(m, 0.975)),
            "n_blocks": nb, "n_boot": n_boot}
