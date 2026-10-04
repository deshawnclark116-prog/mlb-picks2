"""
CFB_PHASE1_OPPORTUNITY -- coherent opportunity allocation components (CFB Outcome Engine v1, RESEARCH / SHADOW ONLY).
A Dirichlet-multinomial allocation of a team total N among candidates (+ an explicit OTHER bucket) has beta-binomial marginals BB(N, alpha_i, alpha_0 - alpha_i); evaluation uses those marginals given the REALIZED team total
(allocation-only), the simulator draws the full coherent multinomial. Participation enters as an optional activity layer (inactive => zero opportunity, weight removed).
Candidates: share model 'base' (recent shrunk share) vs 'logit' (binomial-likelihood logistic on role state), kappa (concentration) by training-only maximum likelihood per group.
"""
import math

import numpy as np
from scipy import optimize, special, stats
from sklearn.linear_model import LogisticRegression

import cfb_phase1_team_environment as TE

KAPPA_BOUNDS = (0.3, 800.0)
LOGISTIC_C = 1.0


def bb_pmf_rows(N, s, kappa, K):
    """BetaBinomial(N_i, s_i * kappa_i, (1 - s_i) * kappa_i) pmf on 0..K (zero above N_i)."""
    N = np.asarray(N).astype(int); s = np.clip(np.asarray(s, float), 1e-5, 1 - 1e-5); kappa = np.broadcast_to(np.asarray(kappa, float), s.shape)
    ks = np.arange(K + 1)[None, :]
    return stats.betabinom.pmf(ks, N[:, None], (s * kappa)[:, None], ((1 - s) * kappa)[:, None])


def fit_kappa(y, N, s):
    y = np.asarray(y); N = np.asarray(N).astype(int); s = np.clip(np.asarray(s, float), 1e-5, 1 - 1e-5)
    def nll(t):
        k = math.exp(t)
        return -float(stats.betabinom.logpmf(y, N, s * k, (1 - s) * k).sum())
    lo, hi = math.log(KAPPA_BOUNDS[0]), math.log(KAPPA_BOUNDS[1])
    res = optimize.minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-5})
    return {"kappa": float(math.exp(res.x)), "at_bound": bool(res.x - lo < 1e-3 or hi - res.x < 1e-3)}


def binomial_logit_fit(X, y, N):
    """Binomial-likelihood logistic regression: each row replicated with label 1 (weight y) and label 0 (weight N - y)."""
    Xr = np.vstack([X, X]); lab = np.concatenate([np.ones(len(X)), np.zeros(len(X))]); w = np.concatenate([np.asarray(y, float), np.asarray(N, float) - np.asarray(y, float)])
    keep = w > 0
    return LogisticRegression(C=LOGISTIC_C, penalty="l2", solver="lbfgs", max_iter=4000).fit(Xr[keep], lab[keep], sample_weight=w[keep])


def arr(rows, names):
    return {n: np.array([r[n] for r in rows], dtype=float) for n in names}


class ShareAlloc:
    """Allocation share model for a set of candidate rows. kind: 'base' | 'logit'. group_fn(rows)-> int group index array (kappa and priors per group)."""

    def __init__(self, kind, share_col, active_share_col, feature_names, y_key, n_key, groups, prior_weight=1.0, use_activity=False, activity_y=None):
        self.kind, self.share_col, self.act_col, self.names, self.y_key, self.n_key, self.groups = kind, share_col, active_share_col, feature_names, y_key, n_key, groups
        self.prior_weight, self.use_activity, self.activity_y = prior_weight, use_activity, activity_y

    def _g(self, rows):
        return np.array([self.groups(r) for r in rows])

    def fit(self, rows):
        y = np.array([r[self.y_key] for r in rows], float); N = np.array([r[self.n_key] for r in rows], float); g = self._g(rows)
        ok = N > 0
        if self.use_activity:
            ya = np.array([self.activity_y(r) for r in rows], float); ok = ok & (ya == 1)                   # share model is fitted on ACTIVE rows only (activity is its own layer)
        self.pi = {k: (float(y[ok & (g == k)].sum() / N[ok & (g == k)].sum()) if (ok & (g == k)).any() and N[ok & (g == k)].sum() > 0 else 0.0) for k in sorted(set(g.tolist()))}
        if self.kind == "logit":
            a = arr(rows, self.names); self.prep = TE.Prep(self.names).fit(a)
            X = self.prep.transform(a)
            self.lg = binomial_logit_fit(X[ok], y[ok], N[ok])
        s = self.share(rows)
        self.kappa = {}
        for k in self.pi:
            m = ok & (g == k)
            self.kappa[k] = fit_kappa(y[m], N[m], s[m]) if m.sum() > 40 else {"kappa": 5.0, "fallback": True}
        if self.use_activity:
            a = arr(rows, self.names)
            self.aprep = TE.Prep(self.names).fit(a)
            self.alg = LogisticRegression(C=LOGISTIC_C, penalty="l2", solver="lbfgs", max_iter=4000).fit(self.aprep.transform(a), ya)
        return self

    def share(self, rows):
        g = self._g(rows)
        if self.kind == "base":
            n5 = np.array([r["P_APPS_L5"] for r in rows], float); sh = np.array([r[self.share_col] for r in rows], float); sh = np.where(np.isfinite(sh), sh, 0.0)
            pi = np.array([self.pi.get(k, 0.0) for k in g])
            return (sh * n5 + pi * self.prior_weight) / (n5 + self.prior_weight)
        a = arr(rows, self.names)
        return self.lg.predict_proba(self.prep.transform(a))[:, 1]

    def activity(self, rows):
        a = arr(rows, self.names)
        return self.alg.predict_proba(self.aprep.transform(a))[:, 1]

    def pmf(self, rows, K, N_override=None):
        N = np.array([r[self.n_key] for r in rows], float) if N_override is None else N_override
        g = self._g(rows); s = self.share(rows)
        kap = np.array([self.kappa[k]["kappa"] if k in self.kappa else 5.0 for k in g])
        pmf = bb_pmf_rows(N, s, kap, K)
        if self.use_activity:
            a = self.activity(rows)[:, None]
            pmf = a * pmf; pmf[:, 0] += 1.0 - a[:, 0]
        return pmf

    def artifact(self):
        return {"kind": self.kind, "prior_share_by_group": self.pi, "kappa": self.kappa, "activity_layer": self.use_activity, "features": self.names if self.kind == "logit" else None}
