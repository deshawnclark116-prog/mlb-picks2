"""Shared splits, metrics and distribution scoring for Phase 1A (shadow research)."""
import math

import numpy as np
from scipy import stats

TRAIN = lambda s, w: s == 2023 or (s == 2024 and w <= 12)
VALID = lambda s, w: s == 2024 and w > 12
DEV = lambda s, w: s == 2025 or (s == 2026 and w <= 3)          # BURNED development evaluation
DEV_LABEL = "DEVELOPMENT (2025 + 2026 wk1-3; burned, not holdout)"
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
XGB = {"max_depth": 4, "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20,
       "reg_lambda": 5.0, "seed": 20260929, "nthread": 4}


def cont(pred, y):
    e = pred - y; a = np.abs(e)
    return {"n": int(len(y)), "mae": round(float(a.mean()), 4), "rmse": round(float(np.sqrt((e ** 2).mean())), 4),
            "bias": round(float(e.mean()), 4), "median_ae": round(float(np.median(a)), 4)}


def binary(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    bins = np.minimum((p * 10).astype(int), 9)
    rel = [{"bin": b, "n": int((bins == b).sum()), "mean_p": round(float(p[bins == b].mean()), 4),
            "obs": round(float(y[bins == b].mean()), 4)} for b in range(10) if (bins == b).any()]
    return {"n": int(len(y)), "base_rate": round(float(y.mean()), 4),
            "logloss": round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 5),
            "brier": round(float(((p - y) ** 2).mean()), 5),
            "ece": round(float(sum(abs(r["mean_p"] - r["obs"]) * r["n"] for r in rel) / len(y)), 5), "reliability": rel}


def nb_params(mu, k):
    """scipy nbinom (n, p) for mean mu, dispersion k (var = mu + mu^2/k)."""
    mu = np.maximum(mu, 1e-6)
    return k, k / (k + mu)


def crps_nb(mu, k, y, top=None):
    """Mean exact CRPS of a negative binomial on the integer grid."""
    return float(np.mean(crps_nb_rows(mu, k, y, top)))


def crps_nb_rows(mu, k, y, top=None):
    """Per-row exact CRPS of a negative binomial on the integer grid."""
    mu = np.asarray(mu, float); y = np.asarray(y, float)
    top = int(top or max(np.max(y), np.max(mu) * 4) + 20)
    grid = np.arange(0, top + 1)
    n, p = nb_params(mu, k)
    F = stats.nbinom.cdf(grid[None, :], n, p[:, None]) if np.ndim(n) == 0 else stats.nbinom.cdf(grid[None, :], n[:, None], p[:, None])
    H = (grid[None, :] >= y[:, None]).astype(float)
    return np.sum((F - H) ** 2, axis=1)


def nb_quantiles(mu, k, qs=QS):
    n, p = nb_params(np.asarray(mu, float), k)
    return {q: stats.nbinom.ppf(q, n, p) for q in qs}


def crps_samples(S, y):
    """CRPS from samples S [n, m] (energy form, sorted-sample O(m log m))."""
    S = np.sort(S, axis=1); m = S.shape[1]
    t1 = np.mean(np.abs(S - y[:, None]), axis=1)
    i = np.arange(1, m + 1)
    t2 = np.sum((2 * i - m - 1)[None, :] * S, axis=1) / (m * m)
    return float(np.mean(t1 - t2))


def sample_summary(S):
    return {"mean": S.mean(1), "p10": np.quantile(S, 0.10, 1), "p25": np.quantile(S, 0.25, 1),
            "median": np.quantile(S, 0.5, 1), "p75": np.quantile(S, 0.75, 1), "p90": np.quantile(S, 0.90, 1)}


def coverage(lo, hi, y):
    return round(float(((y >= lo) & (y <= hi)).mean()), 4)


def poisson_dev(mu, y):
    mu = np.maximum(mu, 1e-6)
    return round(float(2 * np.mean(np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / mu), 0.0) - (y - mu))), 4)


def block_boot(err_new, err_ref, blocks, n=2000, seed=7):
    """Paired week-block bootstrap. Returns (mean improvement, p_not_better)."""
    blocks = np.asarray(blocks)
    ub = np.unique(blocks)
    d = err_ref - err_new
    sums = np.array([d[blocks == b].sum() for b in ub]); cnt = np.array([(blocks == b).sum() for b in ub])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ub), (n, len(ub)))
    bs = sums[idx].sum(1) / cnt[idx].sum(1)
    return round(float(d.mean()), 5), round(float((bs <= 0).mean()), 4)


def fit_xgb(Xtr, ytr, Xva, yva, cols, obj, extra=None):
    import xgboost as xgb
    b = xgb.train({**XGB, "objective": obj, **(extra or {})}, xgb.DMatrix(Xtr, label=ytr, feature_names=cols), 2000,
                  evals=[(xgb.DMatrix(Xva, label=yva, feature_names=cols), "va")], early_stopping_rounds=60, verbose_eval=False)
    return b


def xgb_pred(b, X, cols):
    import xgboost as xgb
    return b.predict(xgb.DMatrix(X, feature_names=cols), iteration_range=(0, b.best_iteration + 1))


def matrix(rows, cols):
    return np.array([[r.get(c) if r.get(c) is not None else np.nan for c in cols] for r in rows], dtype=np.float64)


def safe(x, d=np.nan):
    return d if x is None or (isinstance(x, float) and math.isnan(x)) else x


# ------------------------------------------------------------------ fit audit
FIT_AUDIT = []
LAST_BURNED = (2026, 3)          # last burned development week; nothing later may enter any architecture-selection fit
DEV_WINDOW_LABELS = ("depth_ablation",)   # fits allowed to use burned-development weeks (labelled DEVELOPMENT)


def audit_fit(label, tr, va=()):
    """Register the row windows of a fit; raise if forward data or evaluation rows leak in."""
    seen = set()
    for r in list(tr) + list(va):
        sw = (r["s"], r["w"])
        if sw > LAST_BURNED:
            raise AssertionError(f"fit '{label}' contains post-burned data {sw}")
        tag = "train" if TRAIN(*sw) else "valid" if VALID(*sw) else "dev"
        if tag == "dev" and label not in DEV_WINDOW_LABELS:
            raise AssertionError(f"fit '{label}' contains development-evaluation rows {sw}")
        seen.add((r["s"], tag))
    FIT_AUDIT.append({"label": label, "windows": sorted(f"{s}:{t}" for s, t in seen), "n": len(tr) + len(va)})
