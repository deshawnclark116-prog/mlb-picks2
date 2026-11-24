"""Tennis V2 Phase0 metrics (research only; closed forms, no simulation). Bootstrap = paired resampling of whole matches, 2000 resamples, seed 20261008."""
import math

import numpy as np

SEED = 20261008
B = 2000
EPS = 1e-12
_CACHE = {}


def brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(tot)


def bucket_table(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return [{"bin": "%.1f-%.1f" % (b / bins, (b + 1) / bins), "n": int((idx == b).sum()), "mean_pred": float(p[idx == b].mean()), "obs": float(y[idx == b].mean())}
            for b in range(bins) if (idx == b).any()]


def auc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv)); i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1; i = j + 1
    r = np.empty(len(allv)); r[order] = ranks
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def calib_slope(p, y, iters=50):
    """logistic recalibration y ~ sigmoid(a + b*logit p) by Newton; closed-form iteration."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); y = np.asarray(y, float)
    x = np.log(p / (1 - p)); a, b = 0.0, 1.0
    for _ in range(iters):
        z = a + b * x; q = 1 / (1 + np.exp(-z)); w = q * (1 - q) + 1e-12
        g = np.array([np.sum(y - q), np.sum((y - q) * x)])
        H = np.array([[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]])
        try:
            d = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        a, b = a + d[0], b + d[1]
        if np.abs(d).max() < 1e-9:
            break
    return {"intercept": float(a), "slope": float(b)}


def prob_metrics(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    return {"n": int(len(p)), "brier": brier(p, y), "log_loss": log_loss(p, y), "ece_10bin": ece(p, y), "auc": auc(p, y), "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
            "calibration": calib_slope(p, y), "buckets": bucket_table(p, y)}


def count_metrics(pred, actual, tiers=True):
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    e = pred - actual; ae = np.abs(e)
    out = {"n": int(len(e)), "mae": float(ae.mean()), "bias": float(e.mean()), "median_ae": float(np.median(ae))}
    if tiers:
        for k in (2, 4, 6, 8):
            out["within_%d" % k] = float((ae <= k).mean())
        out["miss_gt_10"] = float((ae > 10).mean()); out["miss_gt_15"] = float((ae > 15).mean())
    return out


def paired_boot(loss_a, loss_b, b=B, seed=SEED):
    """mean(loss_a - loss_b) with 95% percentile interval; resamples whole matches. Negative => A better (lower loss)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    means = d[idx].mean(axis=1)
    return {"mean_diff": float(d.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)), "n": int(n)}


def verdict(loss_model, losses_simple, ece_val=None, prob=False):
    """Protocol native verdict vs the STRONGEST simple baseline (lowest mean loss on identical rows)."""
    if not losses_simple:
        return {"verdict": "BLOCKED_DATA"}
    best = min(losses_simple, key=lambda k: float(np.mean(losses_simple[k])))
    bl = np.asarray(losses_simple[best], float); ml = np.asarray(loss_model, float)
    bt = paired_boot(ml, bl)
    rel = (bl.mean() - ml.mean()) / bl.mean() if bl.mean() else 0.0
    if (bt["lo"] > 0) or (prob and ece_val is not None and ece_val > 0.05):
        v = "REJECTED"
    elif rel >= 0.02 and bt["hi"] < 0 and (not prob or (ece_val is not None and ece_val <= 0.03)):
        v = "SURVIVES"
    elif bt["hi"] >= 0 or True:
        v = "WEAK"
    return {"verdict": v, "strongest_simple": best, "relative_improvement": float(rel), "paired_bootstrap": bt}


def multiclass(dists, actuals):
    """dists: list of dict class->prob; actuals: list of class."""
    ll = br = 0.0; hit = top2 = 0
    for d, a in zip(dists, actuals):
        pa = max(d.get(a, 0.0), EPS)
        ll += -math.log(pa)
        br += sum((v - (1.0 if k == a else 0.0)) ** 2 for k, v in d.items())
        ranked = sorted(d, key=d.get, reverse=True)
        hit += ranked[0] == a; top2 += a in ranked[:2]
    n = len(actuals)
    return {"n": n, "exact_accuracy": hit / n, "log_loss": ll / n, "brier": br / n, "top2_coverage": top2 / n}


def per_row_multiclass_ll(dists, actuals):
    return np.array([-math.log(max(d.get(a, 0.0), EPS)) for d, a in zip(dists, actuals)])
