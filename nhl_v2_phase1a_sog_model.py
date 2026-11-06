"""NHL V2 Phase1A-SOG: fixed B2 count engine (regularised Poisson-regression mean + NB2 dispersion) and distribution metrics.

Ported as FIXED ARCHITECTURE from the burned V1 corpus (nhl_sog_phase1a_models.py / nhl_sog_phase1a_metrics.py). No architecture search, no
hyper-parameter tuning (alpha = 0.001 is frozen), no simulation: the NB2 distribution is evaluated in closed form."""
import hashlib
import json
import math

import numpy as np
from scipy import optimize, special, stats

import nhl_v2_phase1a_sog_data as D

POISSON_ALPHA = 0.001
NB_BOUNDS = (1e-6, 20.0)
SF_TOL = 1e-10
K_CEIL = 200
NLL_FLOOR = 1e-15
BOOT_SEED = 20261002
BOOT_REPS = 10000
ENGINE_VERSION = "nhl-v2-sog-b2-1.0"


def sha_json(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=float).encode()).hexdigest()


def select_rows(tab, mask):
    return {k: v[mask] for k, v in tab.items()}


def population_hash(tab):
    h = hashlib.sha256()
    for k in ("game_id", "team_id", "player_id"):
        h.update(np.ascontiguousarray(tab[k]).tobytes())
    return h.hexdigest()


class Preprocessor:
    def __init__(self, names):
        self.names = list(names)
        self.cont = [n for n in self.names if n not in D.BINARY]

    @staticmethod
    def raw(tab, name):
        x = np.asarray(tab[name]).astype(float)
        return np.minimum(x, D.REST_CAP) if name == "TEAM_REST_HOURS" else x

    def fit(self, tab):
        self.median, self.mean, self.std = {}, {}, {}
        for n in self.cont:
            x = self.raw(tab, n)
            med = float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0
            xi = np.where(np.isnan(x), med, x)
            self.median[n] = med; self.mean[n] = float(xi.mean()); sd = float(xi.std()); self.std[n] = sd if sd > 0 else 1.0
        self.train_hash = population_hash(tab)
        return self

    def transform(self, tab):
        cols = []
        for n in self.names:
            x = self.raw(tab, n)
            if n in D.BINARY:
                cols.append(np.where(np.isnan(x), 0.0, x))
            else:
                miss = np.isnan(x)
                cols.append((np.where(miss, self.median[n], x) - self.mean[n]) / self.std[n])
        for n in self.cont:
            cols.append(np.isnan(self.raw(tab, n)).astype(float))
        return np.column_stack(cols)

    def schema(self):
        return {"feature_order": self.names, "continuous": self.cont, "median": self.median, "mean": self.mean, "std": self.std, "training_population_hash": self.train_hash}

    @classmethod
    def from_schema(cls, s):
        p = cls(s["feature_order"]); p.median, p.mean, p.std = s["median"], s["mean"], s["std"]; p.train_hash = s["training_population_hash"]
        return p


def nb2_loglik(alpha, y, mu):
    r = 1.0 / alpha
    return float(np.sum(special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))))


def fit_nb_alpha(y, mu):
    y = np.asarray(y, float); mu = np.maximum(np.asarray(mu, float), 1e-12)
    lo, hi = math.log(NB_BOUNDS[0]), math.log(NB_BOUNDS[1])
    res = optimize.minimize_scalar(lambda t: -nb2_loglik(math.exp(t), y, mu), bounds=(lo, hi), method="bounded", options={"xatol": 1e-9})
    return {"alpha": math.exp(res.x), "at_lower_bound": bool(res.x - lo < 1e-3), "at_upper_bound": bool(hi - res.x < 1e-3), "loglik": -float(res.fun)}


def fit_b2(train, horizon):
    """Fit the fixed B2 on a training table. Returns a JSON-serialisable artifact."""
    from sklearn.linear_model import PoissonRegressor
    prep = Preprocessor(D.FEATURES).fit(train)
    X = prep.transform(train)
    m = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=1000, tol=1e-6)
    m.fit(X, train["sog"])
    mu = m.predict(X)
    nb = fit_nb_alpha(train["sog"], mu)
    art = {"engine_version": ENGINE_VERSION, "architecture": "B2_FIXED", "horizon": horizon, "poisson_alpha": POISSON_ALPHA, "coef": [float(c) for c in m.coef_], "intercept": float(m.intercept_),
           "n_iter": int(m.n_iter_), "converged": bool(m.n_iter_ < 1000), "preprocessing": prep.schema(), "nb2": nb, "n_train_rows": int(len(train["sog"])), "train_seasons": sorted({int(s) for s in np.unique(train["season"])})}
    art["sha256"] = sha_json({k: v for k, v in art.items() if k != "sha256"})
    return art


def predict_mu(art, tab):
    prep = Preprocessor.from_schema(art["preprocessing"])
    return np.exp(prep.transform(tab) @ np.array(art["coef"]) + art["intercept"])


# ------------------------------------------------------------------ distributions
def pmf_matrix(kind, params, y, chunk=20000):
    n = len(y)
    for a in range(0, n, chunk):
        sl = slice(a, min(a + chunk, n))
        mu = params["mu"][sl]
        K = max(40, int(np.max(y[sl])) + 1)
        while True:
            ks = np.arange(K + 1)[None, :]
            if kind == "poisson":
                pm = stats.poisson.pmf(ks, mu[:, None]); sf = stats.poisson.sf(K, mu)
            else:
                r = 1.0 / params["alpha"]; pp = r / (r + mu)
                pm = stats.nbinom.pmf(ks, r, pp[:, None]); sf = stats.nbinom.sf(K, r, pp)
            if np.max(sf) < SF_TOL:
                break
            if K >= K_CEIL:
                raise RuntimeError("tail probability %.3e >= %s at k=%d: distribution invalid" % (float(np.max(sf)), SF_TOL, K_CEIL))
            K = min(K * 2, K_CEIL)
        yield sl, pm, sf


def pit_v(model_id, game_ids, player_ids):
    out = np.empty(len(game_ids))
    for i, (g, p) in enumerate(zip(game_ids, player_ids)):
        out[i] = int.from_bytes(hashlib.sha256(("%s|%d|%d" % (model_id, int(g), int(p))).encode()).digest()[:8], "big") / 2.0 ** 64
    return out


def row_metrics(kind, params, y, model_id, game_ids, player_ids):
    n = len(y)
    out = {"crps": np.zeros(n), "nll": np.zeros(n), "pit": np.zeros(n), "floor_hits": 0, "max_K": 0, "p_ge": np.zeros((n, 5)), "median": np.zeros(n), "lo": {}, "hi": {}}
    for c in (0.5, 0.8, 0.9):
        out["lo"][c] = np.zeros(n, dtype=np.int32); out["hi"][c] = np.zeros(n, dtype=np.int32)
    v = pit_v(model_id, game_ids, player_ids)
    for sl, pm, sf in pmf_matrix(kind, params, y):
        yy = y[sl]; K = pm.shape[1] - 1
        out["max_K"] = max(out["max_K"], K)
        cdf = np.cumsum(pm, axis=1)
        ind = (yy[:, None] <= np.arange(K + 1)[None, :]).astype(float)
        out["crps"][sl] = ((cdf - ind) ** 2).sum(axis=1)
        py = pm[np.arange(len(yy)), yy]
        out["floor_hits"] += int((py < NLL_FLOOR).sum())
        out["nll"][sl] = -np.log(np.maximum(py, NLL_FLOOR))
        below = np.where(yy > 0, cdf[np.arange(len(yy)), np.maximum(yy - 1, 0)], 0.0)
        out["pit"][sl] = below + v[sl] * py
        for j in range(5):
            out["p_ge"][sl, j] = 1.0 - cdf[:, min(j, K)]
        out["median"][sl] = (cdf >= 0.5 - 1e-12).argmax(axis=1)
        for c in (0.5, 0.8, 0.9):
            out["lo"][c][sl] = (cdf >= (1 - c) / 2 - 1e-12).argmax(axis=1)
            out["hi"][c][sl] = (cdf >= (1 + c) / 2 - 1e-12).argmax(axis=1)
    return out


def distribution_summary(mu, alpha):
    """Closed-form summary of NB2(mu, alpha): mean, median, variance, P(SOG>=1..5)."""
    mu = np.asarray(mu, float)
    r = 1.0 / alpha; pp = r / (r + mu)
    cdf = np.stack([stats.nbinom.cdf(k, r, pp) for k in range(0, 5)], axis=1)
    med = stats.nbinom.ppf(0.5, r, pp)
    return {"mean": mu, "median": med, "variance": mu + alpha * mu ** 2, "p_ge": 1.0 - np.concatenate([np.zeros((len(mu), 1)), cdf[:, :4]], axis=1)}


def game_macro(values, game_ids):
    order = np.argsort(game_ids, kind="stable")
    g = np.asarray(game_ids)[order]; v = np.asarray(values)[order]
    uniq, start = np.unique(g, return_index=True)
    sums = np.add.reduceat(v, start); cnt = np.diff(np.append(start, len(g)))
    per_game = sums / cnt
    return float(per_game.mean()), uniq, per_game


def ks_uniform(u):
    u = np.sort(u); n = len(u); i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def central_metrics(pred, y, median=None):
    pred = np.asarray(pred, float); y = np.asarray(y, float)
    e = pred - y; ae = np.abs(e)
    d = {"n": int(len(y)), "mae_mean": float(ae.mean()), "bias": float(e.mean()), "median_ae": float(np.median(ae)), "within_0.5": float((ae <= 0.5).mean()), "within_1": float((ae <= 1).mean()), "within_2": float((ae <= 2).mean())}
    for k in (2, 3, 4):
        d["miss_gt_%d" % k] = float((ae > k).mean()); d["miss_gt_%d_n" % k] = int((ae > k).sum())
    if median is not None:
        d["mae_median_forecast"] = float(np.mean(np.abs(np.asarray(median, float) - y)))
    return d


def calib_fit(p, e, iters=50):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); e = np.asarray(e, float)
    if e.sum() == 0 or e.sum() == len(e) or np.std(p) < 1e-9:
        return None, None
    x = np.log(p / (1 - p)); a, b = 0.0, 1.0
    for _ in range(iters):
        z = np.clip(a + b * x, -30, 30); q = 1 / (1 + np.exp(-z)); w = q * (1 - q)
        g = np.array([np.sum(e - q), np.sum((e - q) * x)])
        H = np.array([[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]])
        try:
            st = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None, None
        a += st[0]; b += st[1]
        if np.max(np.abs(st)) < 1e-9:
            break
    return float(a), float(b)


def threshold_report(p_ge, y, base_rates, nbins=10):
    """P(SOG>=k) k=1..5: Brier, Brier skill vs frozen base-rate comparator, calibration, ECE, observed vs predicted, reliability."""
    out = {}
    for j in range(5):
        p = p_ge[:, j]; e = (y >= j + 1).astype(float)
        order = np.argsort(p, kind="stable"); bins = np.array_split(order, nbins)
        ece = sum(len(b) / len(p) * abs(p[b].mean() - e[b].mean()) for b in bins if len(b))
        b0 = base_rates[:, j] if np.ndim(base_rates) == 2 else base_rates[j]
        br = float(np.mean((p - e) ** 2)); br0 = float(np.mean((b0 - e) ** 2))
        a, b = calib_fit(p, e)
        out["P(SOG>=%d)" % (j + 1)] = {"brier": br, "brier_base_rate_comparator": br0, "brier_skill": (1 - br / br0) if br0 > 0 else None, "ece_10_equal_count_bins": float(ece), "mean_pred": float(p.mean()), "observed_rate": float(e.mean()),
                                       "calibration_intercept": a, "calibration_slope": b,
                                       "reliability": [{"n": int(len(b)), "mean_pred": float(p[b].mean()), "observed": float(e[b].mean())} for b in bins if len(b)]}
    return out


def summarize(kind, params, y, model_id, game_ids, player_ids, mean_pred, base_rates):
    m = row_metrics(kind, params, y, model_id, game_ids, player_ids)
    crps_g, uniq, crps_pg = game_macro(m["crps"], game_ids)
    nll_g, _, nll_pg = game_macro(m["nll"], game_ids)
    cov = {str(int(c * 100)): float(np.mean((y >= m["lo"][c]) & (y <= m["hi"][c]))) for c in (0.5, 0.8, 0.9)}
    res = {"model_id": model_id, "n_rows": int(len(y)), "n_games": int(len(uniq)), "crps_macro_game": crps_g, "crps_row_mean": float(m["crps"].mean()), "nll_macro_game": nll_g, "nll_row_mean": float(m["nll"].mean()),
           "nll_floor_hits": m["floor_hits"], "central": central_metrics(mean_pred, y, m["median"]), "mean_pred": float(np.mean(mean_pred)), "mean_obs": float(np.mean(y)), "median_pred_mean": float(m["median"].mean()),
           "interval_coverage": cov, "interval_coverage_error": {k: abs(v - int(k) / 100) for k, v in cov.items()}, "pit_ks": ks_uniform(m["pit"]), "thresholds": threshold_report(m["p_ge"], y, base_rates), "support_ceiling_used_K": m["max_K"]}
    return res, {"crps": m["crps"], "nll": m["nll"], "pit": m["pit"], "crps_pg": crps_pg, "games": uniq, "median": m["median"], "p_ge": m["p_ge"]}


def blocked_bootstrap(game_delta, game_week, reps=BOOT_REPS, seed=BOOT_SEED):
    game_delta = np.asarray(game_delta, float); game_week = np.asarray(game_week)
    weeks = np.unique(game_week); W = len(weeks)
    pos = {int(w): i for i, w in enumerate(weeks)}
    S = np.zeros(W); N = np.zeros(W)
    for d, w in zip(game_delta, game_week):
        S[pos[int(w)]] += d; N[pos[int(w)]] += 1
    starts = np.array([i for i in range(W - 1) if weeks[i + 1] == weeks[i] + 1])
    nb = int(math.ceil(W / 2))
    rng = np.random.default_rng(seed)
    pick = starts[rng.integers(0, len(starts), size=(reps, nb))]
    idx = np.stack([pick, pick + 1], axis=2).reshape(reps, nb * 2)[:, :W]
    boot = S[idx].sum(axis=1) / N[idx].sum(axis=1)
    return {"observed_mean_delta": float(np.mean(game_delta)), "ci95_lo": float(np.percentile(boot, 2.5)), "ci95_hi": float(np.percentile(boot, 97.5)), "replicates": reps, "seed": seed}
