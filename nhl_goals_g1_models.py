"""
NHL_GOALS_G1_MODELS -- G0 position-conversion compound, G1 hierarchical player-conversion compound (beta-binomial), G2 direct count comparator (protocol: phase_goals_g1_protocol.json).
The SOG distribution of G0 / G1 is the FROZEN Phase 1A B2 architecture (nhl_sog_phase1b_probe.fit_architecture, refit on the fold's training rows). The compound PMFs are exact analytic mixtures over the full B2 SOG PMF (no Monte Carlo).
"""
import math

import numpy as np
from scipy import optimize, special, stats

import nhl_goals_g1_data as GD
import nhl_sog_phase1a_data as D
import nhl_sog_phase1a_metrics as M
import nhl_sog_phase1a_models as MD
import nhl_sog_phase1b_probe as PB

GMAX = GD.GMAX
CLASSES = ("F", "D", "U")
MIN_CLASS_PLAYERS = 30
HYPER_LOG_BOUNDS = (math.log(0.05), math.log(5000.0))
START_CONCENTRATION = 50.0
CHUNK = 2000
G2_ALPHA = 0.001


def pos_class_of(tab):
    return np.where(tab["POS_F"] == 1, "F", np.where(tab["POS_D"] == 1, "D", "U"))


# ------------------------------------------------------------------ B2 SOG distribution
def fit_sog(train):
    return PB.fit_architecture(train, list(D.FEATURES))


def sog_params(sog_arch, tab):
    kind, params, mu = PB.predict_architecture(sog_arch, tab)
    return params, mu


def sog_pmf_chunks(params, n, chunk=CHUNK):
    """Yield (slice, pmf_S[c, K+1], survival_at_K[c]) of the NB2 SOG distribution (K extended until survival < 1e-10)."""
    y0 = np.zeros(n, dtype=np.int64)
    yield from M.pmf_matrix("nb2", params, y0, chunk=chunk)


# ------------------------------------------------------------------ exact compound PMFs
def binom_matrix(p, K, G=GMAX):
    s = np.arange(K + 1)[:, None]; y = np.arange(G + 1)[None, :]
    return stats.binom.pmf(y, s, p)                                            # zero where y > s


def betabinom_pmf_rows(a, b, K):
    """[c, K+1, G+1] beta-binomial pmf with per-row (a, b); zero where y > s."""
    s = np.arange(K + 1)[None, :, None].astype(float); y = np.arange(GMAX + 1)[None, None, :].astype(float)
    a = a[:, None, None]; b = b[:, None, None]
    lp = (special.gammaln(s + 1) - special.gammaln(y + 1) - special.gammaln(np.maximum(s - y, 0) + 1)
          + special.betaln(y + a, np.maximum(s - y, 0) + b) - special.betaln(a, b))
    out = np.exp(lp)
    return np.where(y <= s, out, 0.0)


def compound_betabinom_pmf(params, a_row, b_row, n):
    out = np.zeros((n, GMAX + 1)); sf_max = 0.0; Kmax = 0
    for sl, pm, sf in sog_pmf_chunks(params, n):
        K = pm.shape[1] - 1; Kmax = max(Kmax, K); sf_max = max(sf_max, float(np.max(sf)))
        c = pm.shape[0]
        a, b = a_row[sl], b_row[sl]
        for j in range(0, c, 250):
            t = slice(j, min(j + 250, c))
            bb = betabinom_pmf_rows(a[t], b[t], K)
            out[sl.start + j: sl.start + min(j + 250, c)] = np.einsum("ck,ckg->cg", pm[t], bb)
    return out, {"max_sog_survival_at_K": sf_max, "max_K": Kmax}


# ------------------------------------------------------------------ G0
class G0:
    name = "G0"

    def fit(self, train, sog_arch=None):
        self.sog = sog_arch or fit_sog(train)
        cls = pos_class_of(train); pl = train["played"] == 1
        g, e = train["goals"], train["eff_sog"]
        tot_e = float(e[pl].sum())
        self.p_pool = float(g[pl].sum() / tot_e)
        self.p = {}; self.n_shots = {}
        for c in CLASSES:
            m = pl & (cls == c); es = float(e[m].sum())
            self.p[c] = float(g[m].sum() / es) if es > 0 else self.p_pool
            self.n_shots[c] = es
        return self

    def predict(self, tab):
        n = len(tab["game_id"]); cls = pos_class_of(tab)
        p_row = np.array([self.p[c] for c in cls])
        params, mu = sog_params(self.sog, tab)
        pmf = np.zeros((n, GMAX + 1)); sf_max = 0.0; Kmax = 0
        for sl, pm, sf in sog_pmf_chunks(params, n):
            K = pm.shape[1] - 1; Kmax = max(Kmax, K); sf_max = max(sf_max, float(np.max(sf)))
            pr = p_row[sl]; block = np.zeros((pm.shape[0], GMAX + 1))
            for c in CLASSES:
                m = cls[sl] == c
                if m.any():
                    block[m] = pm[m] @ binom_matrix(self.p[c], K)
            pmf[sl] = block
        return {"pmf": pmf, "implied_mean_sog": mu, "implied_p": p_row, "diagnostics": {"max_sog_survival_at_K": sf_max, "max_K": Kmax}}

    def artifact(self):
        return {"model": "G0", "conversion_by_class": self.p, "pooled_conversion": self.p_pool, "training_effective_shots_by_class": self.n_shots, "sog_nb2_alpha": self.sog["nb"]["alpha"], "sog_fit": self.sog["fit"]}


# ------------------------------------------------------------------ G1
def betabinom_marginal_nll(theta, g, n):
    a, b = math.exp(theta[0]), math.exp(theta[1])
    ll = special.gammaln(n + 1) - special.gammaln(g + 1) - special.gammaln(n - g + 1) + special.betaln(g + a, n - g + b) - special.betaln(a, b)
    return -float(np.sum(ll))


def fit_hyperprior(g, n):
    """Maximum-likelihood beta-binomial marginal likelihood over per-player aggregated (goals g, effective shots n) with n >= 1. Returns dict with a, b, convergence and boundary diagnostics."""
    g = np.asarray(g, float); n = np.asarray(n, float)
    ok = n >= 1
    g, n = g[ok], n[ok]
    m = float(g.sum() / n.sum())
    m = min(max(m, 1e-4), 1 - 1e-4)
    start = [math.log(m * START_CONCENTRATION), math.log((1 - m) * START_CONCENTRATION)]
    res = optimize.minimize(betabinom_marginal_nll, start, args=(g, n), method="L-BFGS-B", bounds=[HYPER_LOG_BOUNDS, HYPER_LOG_BOUNDS])
    lo, hi = HYPER_LOG_BOUNDS
    at_bound = {"log_a_at_lower": bool(res.x[0] - lo < 1e-3), "log_a_at_upper": bool(hi - res.x[0] < 1e-3), "log_b_at_lower": bool(res.x[1] - lo < 1e-3), "log_b_at_upper": bool(hi - res.x[1] < 1e-3)}
    a, b = math.exp(res.x[0]), math.exp(res.x[1])
    return {"a": a, "b": b, "concentration": a + b, "mean": a / (a + b), "n_players": int(len(g)), "total_shots": float(n.sum()), "total_goals": float(g.sum()), "converged": bool(res.success), "n_iter": int(res.nit),
            "neg_loglik": float(res.fun), "boundary_hits": at_bound, "any_boundary_hit": bool(any(at_bound.values()))}


class G1:
    name = "G1"

    def fit(self, train, sog_arch=None):
        self.sog = sog_arch or fit_sog(train)
        cls = pos_class_of(train); pl = train["played"] == 1
        agg = {}                                                                 # (player, class) -> [goals, eff shots] over TRAINING played rows
        for p, c, g, e in zip(train["player_id"][pl], cls[pl], train["goals"][pl], train["eff_sog"][pl]):
            a = agg.setdefault((int(p), c), [0, 0]); a[0] += int(g); a[1] += int(e)
        self.hyper, self.hyper_source = {}, {}
        allg = np.array([v[0] for v in agg.values()]); alln = np.array([v[1] for v in agg.values()])
        self.pooled = fit_hyperprior(allg, alln)
        for c in CLASSES:
            vs = [v for (p, cc), v in agg.items() if cc == c and v[1] >= 1]
            if len(vs) >= MIN_CLASS_PLAYERS:
                self.hyper[c] = fit_hyperprior([v[0] for v in vs], [v[1] for v in vs]); self.hyper_source[c] = "class"
            else:
                self.hyper[c] = self.pooled; self.hyper_source[c] = "pooled_fallback(<30 players)"
        return self

    def posterior(self, tab):
        cls = pos_class_of(tab)
        a0 = np.array([self.hyper[c]["a"] for c in cls]); b0 = np.array([self.hyper[c]["b"] for c in cls])
        g = np.nan_to_num(tab["GOALS_SUM_CUM"], nan=0.0); e = np.nan_to_num(tab["EFFSOG_SUM_CUM"], nan=0.0)
        return a0 + g, b0 + (e - g), a0 + b0, e

    def predict(self, tab):
        n = len(tab["game_id"])
        a, b, conc, e_prior = self.posterior(tab)
        params, mu = sog_params(self.sog, tab)
        pmf, diag = compound_betabinom_pmf(params, a, b, n)
        pm = a / (a + b); psd = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        return {"pmf": pmf, "implied_mean_sog": mu, "implied_p": pm, "posterior_sd": psd, "prior_dominated": conc > e_prior, "diagnostics": diag}

    def artifact(self):
        return {"model": "G1", "hyperprior": self.hyper, "hyperprior_source": self.hyper_source, "pooled_hyperprior": self.pooled, "sog_nb2_alpha": self.sog["nb"]["alpha"], "sog_fit": self.sog["fit"]}


# ------------------------------------------------------------------ G2
class G2:
    name = "G2"

    def fit(self, train):
        prep = MD.Preprocessor(GD.G2_FEATURES).fit(train)
        X = prep.transform(train)
        m, info = MD.fit_poisson(X, train["goals"], G2_ALPHA)
        nb = MD.fit_nb_alpha(train["goals"], m.predict(X))
        self.arch = {"prep": prep, "model": m, "nb": nb, "fit": info}
        return self

    def predict(self, tab):
        mu = self.arch["model"].predict(self.arch["prep"].transform(tab))
        n = len(mu)
        out = np.zeros((n, GMAX + 1)); sf_max = 0.0
        for sl, pm, sf in M.pmf_matrix("nb2", {"mu": mu, "alpha": self.arch["nb"]["alpha"]}, np.zeros(n, dtype=np.int64)):
            sf_max = max(sf_max, float(np.max(sf)))
            k = min(pm.shape[1], GMAX + 1)
            out[sl, :k] = pm[:, :k]
        return {"pmf": out, "mu": mu, "diagnostics": {"max_goal_survival_at_K": sf_max}}

    def artifact(self):
        return {"model": "G2", "features": list(GD.G2_FEATURES), "alpha_poisson": G2_ALPHA, "nb2_dispersion": self.arch["nb"], "fit": self.arch["fit"]}


# ------------------------------------------------------------------ coherence
def coherence_report(pmf):
    s = pmf.sum(axis=1)
    return {"min_row_sum": float(s.min()), "max_row_sum": float(s.max()), "max_missing_tail_mass": float(np.max(1.0 - s)), "min_probability": float(pmf.min()), "finite": bool(np.isfinite(pmf).all())}


def pmf_mean(pmf):
    return pmf @ np.arange(pmf.shape[1])
