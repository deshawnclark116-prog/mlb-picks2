"""
CFB_PHASE1_EFFICIENCY -- per-event yardage distributions (per carry / per completion / per reception) with hierarchical shrinkage and exact compound distributions by FFT convolution (CFB Outcome Engine v1, RESEARCH / SHADOW ONLY).
Event tables: cfb_phase1_data.load_events (dev: seasons <= 2024). All histories are AS-OF: only events of strictly earlier weeks. Yards are clipped to [YMIN, YMAX] and indexed y - YMIN.
"""
import math
from collections import defaultdict

import numpy as np
from scipy import optimize

import cfb_phase1_common as C

YMIN, YMAX = -10, 99
NB = YMAX - YMIN + 1
FFT_L = 8192
WINDOW = 150                                  # most recent events of a player used for his empirical counts


def clip_idx(y):
    return int(min(max(y, YMIN), YMAX)) - YMIN


class EventIndex:
    """Per-key chronological event arrays: key -> (week_index[], yard_idx[]) for as-of histograms."""

    def __init__(self, events, key_fn, yard_fn="yards", filt=None):
        tmp = defaultdict(list)
        for e in events:
            if filt and not filt(e):
                continue
            k = key_fn(e)
            if k:
                tmp[k].append((C.week_index(e["season"], e["week"]), clip_idx(e[yard_fn])))
        self.w, self.y = {}, {}
        for k, v in tmp.items():
            v.sort()
            self.w[k] = np.array([a for a, _ in v]); self.y[k] = np.array([b for _, b in v])

    def counts_before(self, key, widx, window=WINDOW):
        w = self.w.get(key)
        if w is None:
            return np.zeros(NB), 0
        hi = int(np.searchsorted(w, widx, side="left"))
        lo = max(0, hi - window)
        c = np.bincount(self.y[key][lo:hi], minlength=NB).astype(float)
        return c, hi - lo


def pooled_pmf(index, keys_group, group_of_key, train_max_widx, smooth=0.5):
    """Position pmfs from TRAINING events only (week index <= train_max_widx): {group: pmf[NB]} + league pmf."""
    acc = defaultdict(lambda: np.zeros(NB))
    for k, w in index.w.items():
        m = w <= train_max_widx
        if not m.any():
            continue
        g = group_of_key(k)
        acc[g] += np.bincount(index.y[k][m], minlength=NB)
    league = sum(acc.values(), np.zeros(NB)) if acc else np.ones(NB)
    out = {g: (c + smooth * league / league.sum()) / (c.sum() + smooth) for g, c in acc.items()}
    return out, (league + smooth) / (league.sum() + smooth * NB)


def shrink_rows(index, keys, widx, base_pmf_rows, kappa):
    """Hierarchical player pmf: (player counts + kappa * base) / (n + kappa), as-of widx. base_pmf_rows [n, NB]."""
    out = np.empty_like(base_pmf_rows); ns = np.zeros(len(keys))
    for i, (k, w) in enumerate(zip(keys, widx)):
        c, n = index.counts_before(k, w)
        out[i] = (c + kappa * base_pmf_rows[i]) / (n + kappa); ns[i] = n
    return out, ns


def tilt_rows(pmf_rows, z, theta):
    """Exponential tilt p_k ~ p_k * exp(theta * z * (k - mean_k)): opponent-adjusts the yardage distribution by a standardized opponent allowed-yardage index z."""
    ks = np.arange(NB)[None, :]
    mu = (pmf_rows * ks).sum(axis=1, keepdims=True)
    z = np.clip(z, -3.0, 3.0)                                                   # development amendment 2: standardized opponent index clipped (FCS / tiny-sample outliers broke the tilt)
    t = pmf_rows * np.exp(np.clip(theta * z[:, None] * (ks - mu) / 10.0, -20, 20))
    return t / t.sum(axis=1, keepdims=True)


def fit_tilt(pmf_rows, z, y_idx):
    """Training-only 1-parameter maximum likelihood of the opponent tilt on event-level data (rows = events)."""
    def nll(theta):
        t = tilt_rows(pmf_rows, z, theta)
        return -float(np.log(np.maximum(t[np.arange(len(y_idx)), y_idx], 1e-12)).sum())
    res = optimize.minimize_scalar(nll, bounds=(-3, 3), method="bounded", options={"xatol": 1e-3})
    return float(res.x)


class CompoundPmf:
    """Lazy exact compound distribution of S = sum of c iid draws from a per-row pmf over NB bins (index domain = S - c*YMIN). `.score` streams chunks so the n x FFT_L matrix is never held."""

    def __init__(self, pmf_rows, c):
        self.p, self.c = pmf_rows, np.asarray(c).astype(int)

    def chunks(self, size=400):
        for a in range(0, len(self.c), size):
            sl = slice(a, min(a + size, len(self.c)))
            P = np.zeros((sl.stop - sl.start, FFT_L)); P[:, :NB] = self.p[sl]
            phi = np.fft.rfft(P, axis=1)
            out = np.fft.irfft(np.power(phi, self.c[sl][:, None]), n=FFT_L, axis=1)
            yield sl, np.clip(out, 0.0, None)

    def score(self, model_id, y_total, keys, support=None):
        """y_total: realized total yards; observed index = y_total - c*YMIN. Returns (summary, crps[n], nll[n])."""
        n = len(self.c); idx = np.asarray(y_total).astype(int) - self.c * YMIN
        cr, nl, pit, mean = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        v = C.pit_v(model_id, keys); cov = {c: np.zeros(n, bool) for c in (0.5, 0.8, 0.9)}; med = np.zeros(n); tail = np.zeros(n)
        for sl, pm in self.chunks():
            k = np.clip(idx[sl], 0, FFT_L - 1)
            tail[sl] = 1.0 - pm.sum(axis=1)
            cdf = np.cumsum(pm, axis=1)
            ind = (k[:, None] <= np.arange(FFT_L)[None, :])
            cr[sl] = ((cdf - ind) ** 2).sum(axis=1)
            py = pm[np.arange(len(k)), k]
            nl[sl] = -np.log(np.maximum(py, C.NLL_FLOOR))
            ar = np.arange(len(k)); below = np.where(k > 0, cdf[ar, np.maximum(k - 1, 0)], 0.0)
            pit[sl] = below + v[sl] * py
            mean[sl] = pm @ np.arange(FFT_L)
            med[sl] = (cdf >= 0.5).argmax(axis=1)
            for c in cov:
                lo, hi = (cdf >= (1 - c) / 2).argmax(axis=1), (cdf >= (1 + c) / 2).argmax(axis=1)
                cov[c][sl] = (k >= lo) & (k <= hi)
        y = idx
        s = {"crps": float(cr.mean()), "nll": float(nl.mean()), "pit_ks": C.ks_uniform(pit), "mean_pred": float((mean + self.c * YMIN).mean()), "mean_obs": float(np.asarray(y_total).mean()), "rel_mean_bias": float(((mean + self.c * YMIN).mean() - np.mean(y_total)) / max(abs(np.mean(y_total)), 1e-9)),
             "rmse_mean": float(np.sqrt(((mean - y) ** 2).mean())), "mae_median": float(np.abs(med - y).mean()), "coverage": {str(int(c * 100)): float(m.mean()) for c, m in cov.items()}, "n": int(n), "max_tail_mass": float(tail.max())}
        return s, cr, nl
