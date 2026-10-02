"""
NHL_SOG_PHASE1A_MODELS -- B0 empirical-shrinkage mixture, B1 Poisson regression, B2 NB2 wrapper, B3 structural participation mixture (registered in phase1a_protocol.json).
Learned models: sklearn PoissonRegressor / LogisticRegression only; NB2 dispersion by deterministic bounded maximum likelihood with mu fixed. No Monte Carlo, no tree / neural models.
"""
import hashlib
import json
import math

import numpy as np
from scipy import optimize, special, stats
from sklearn.linear_model import LogisticRegression, PoissonRegressor

import nhl_sog_phase1a_data as D

KAPPA_PLAY = 5.0
KAPPA_SOG = 5.0
DIRICHLET_EPS = 0.01
SUPPORT = 60
ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
C_GRID = [0.01, 0.1, 1.0, 10.0]
NB_BOUNDS = (1e-6, 20.0)
B3_AVAIL = ["POS_F", "POS_D", "POS_UNKNOWN", "PLAYED_LAST1", "PLAY_RATE_TG3", "PLAY_RATE_TG10", "PLAY_DEN_TG3", "PLAY_DEN_TG10", "TEAM_GAMES_SINCE_APPEARANCE", "DAYS_SINCE_LAST_APPEARANCE", "N_CURRENT_SEASON_TEAM_GAMES_OBS",
            "N_CURRENT_SEASON_PLAYER_APPEARANCES", "TOI_MEAN_CT_APP3", "TOI_DELTA_CT_3_10", "PP_TOI_MEAN_CT_APP3", "PP_ALLOC_SHARE_MEAN_CT_APP3", "PP_ALLOC_SHARE_DELTA_CT_3_10", "SHIFT_MEAN_CT_APP3", "SHIFT_DELTA_CT_3_10",
            "TEAM_REST_HOURS", "BACK_TO_BACK"]
B3_COND = ["POS_F", "POS_D", "POS_UNKNOWN", "SOG_MEAN_APP5", "SOG_MEAN_APP10", "SOG_SD_APP10", "SOG_PER60_APP10", "N_SKILL_APPEARANCES_10", "TOI_MEAN_CT_APP3", "TOI_MEAN_CT_APP10", "TOI_DELTA_CT_3_10", "PP_TOI_MEAN_CT_APP3",
           "PP_ALLOC_SHARE_MEAN_CT_APP3", "PP_ALLOC_SHARE_MEAN_CT_APP10", "PP_ALLOC_SHARE_DELTA_CT_3_10", "SHIFT_MEAN_CT_APP3", "SHIFT_MEAN_CT_APP10", "SHIFT_DELTA_CT_3_10", "N_ROLE_APPEARANCES_10", "TEAM_SOG_FOR_MEAN5",
           "OPP_SOG_ALLOWED_MEAN5", "IS_HOME", "TEAM_REST_HOURS", "BACK_TO_BACK", "N_CURRENT_SEASON_TEAM_GAMES_OBS"]
B1_FEATURES = list(D.FEATURES)


def sha_json(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=float).encode()).hexdigest()


def select_rows(tab, mask):
    return {k: v[mask] for k, v in tab.items()}


def population_hash(tab):
    h = hashlib.sha256()
    for k in ("game_id", "team_id", "player_id"):
        h.update(np.ascontiguousarray(tab[k]).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ preprocessing (training-only)
class Preprocessor:
    def __init__(self, names):
        self.names = list(names)
        self.cont = [n for n in self.names if n not in D.BINARY]

    @staticmethod
    def raw(tab, name):
        x = tab[name].astype(float)
        return np.minimum(x, D.REST_CAP) if name == "TEAM_REST_HOURS" else x          # cap for regression input only

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
        for n in self.cont:                                                            # explicit missing indicator for every continuous feature
            cols.append(np.isnan(self.raw(tab, n)).astype(float))
        return np.column_stack(cols)

    def schema(self):
        return {"feature_order": self.names, "continuous": self.cont, "missing_indicator_columns": [n + "__missing" for n in self.cont], "median": self.median, "mean": self.mean, "std": self.std, "training_population_hash": self.train_hash}


# ------------------------------------------------------------------ NB2
def nb2_loglik(alpha, y, mu):
    r = 1.0 / alpha
    return float(np.sum(special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))))


def fit_nb_alpha(y, mu):
    """Deterministic bounded MLE of the NB2 dispersion with mu fixed, on log alpha within [1e-6, 20]."""
    y = np.asarray(y, float); mu = np.maximum(np.asarray(mu, float), 1e-12)
    lo, hi = math.log(NB_BOUNDS[0]), math.log(NB_BOUNDS[1])
    res = optimize.minimize_scalar(lambda t: -nb2_loglik(math.exp(t), y, mu), bounds=(lo, hi), method="bounded", options={"xatol": 1e-9})
    a = math.exp(res.x)
    return {"alpha": a, "at_lower_bound": bool(res.x - lo < 1e-3), "at_upper_bound": bool(hi - res.x < 1e-3), "loglik": -float(res.fun)}


# ------------------------------------------------------------------ B0
class B0:
    def fit(self, tab):
        pos = np.where(tab["POS_F"] == 1, 0, np.where(tab["POS_D"] == 1, 1, 2))
        played = tab["played"] == 1
        self.base = {}; self.q = {}
        gp = float(played.mean())
        counts_all = np.bincount(tab["sog"][played], minlength=SUPPORT + 1).astype(float)
        if tab["sog"].max() > SUPPORT:
            raise RuntimeError(f"observed SOG {tab['sog'].max()} > {SUPPORT}: BLOCK")
        for ci, name in enumerate(("F", "D", "U")):
            m = pos == ci
            self.base[name] = float(played[m].mean()) if m.any() else gp
            c = np.bincount(tab["sog"][m & played], minlength=SUPPORT + 1).astype(float) if (m & played).any() else counts_all
            q = c + DIRICHLET_EPS * 1.0                                                  # DIRICHLET_EPS per support cell
            self.q[name] = q / q.sum()
        self.global_play = gp
        self.hash = sha_json({"base": self.base, "q": {k: v.tolist() for k, v in self.q.items()}})
        return self

    def predict(self, tab):
        pos = np.where(tab["POS_F"] == 1, "F", np.where(tab["POS_D"] == 1, "D", "U"))
        n = len(pos)
        base = np.array([self.base[p] for p in pos]); Q = np.stack([self.q[p] for p in pos]) if n else np.zeros((0, SUPPORT + 1))
        p_play = (tab["plays10"] + KAPPA_PLAY * base) / (tab["den10"] + KAPPA_PLAY)
        counts = np.zeros((n, SUPPORT + 1))
        h = tab["hist10"]
        for j in range(10):
            ok = h[:, j] >= 0
            if (h[ok, j] > SUPPORT).any():
                raise RuntimeError("history SOG > 60: BLOCK")
            np.add.at(counts, (np.where(ok)[0], h[ok, j]), 1.0)
        napp = counts.sum(axis=1, keepdims=True)
        cond = (counts + KAPPA_SOG * Q) / (napp + KAPPA_SOG)
        pmf = p_play[:, None] * cond; pmf[:, 0] += 1.0 - p_play
        mean_cond = (cond * np.arange(SUPPORT + 1)).sum(axis=1)
        return {"pmf": pmf, "p_play": p_play, "cond_pmf": cond, "mean": p_play * mean_cond, "mean_cond": mean_cond}

    def artifact(self):
        return {"model": "B0", "constants": {"KAPPA_PLAY": KAPPA_PLAY, "KAPPA_SOG": KAPPA_SOG, "DIRICHLET_EPS": DIRICHLET_EPS, "SUPPORT": SUPPORT}, "position_base_play_rate": self.base, "position_pmf": {k: v.tolist() for k, v in self.q.items()}, "sha256": self.hash}


# ------------------------------------------------------------------ B1 / B2
def fit_poisson(X, y, alpha):
    m = PoissonRegressor(alpha=alpha, max_iter=1000, tol=1e-6)
    m.fit(X, y)
    return m, {"n_iter": int(m.n_iter_), "converged": bool(m.n_iter_ < 1000)}


def poisson_artifact(m, prep, alpha, extra=None):
    a = {"coef": m.coef_.tolist(), "intercept": float(m.intercept_), "alpha": alpha, "preprocessing": prep.schema(), **(extra or {})}
    a["sha256"] = sha_json(a)
    return a


def predict_poisson_from_artifact(a, X):
    return np.exp(X @ np.array(a["coef"]) + a["intercept"])


# ------------------------------------------------------------------ B3
def fit_logistic(X, y, C):
    m = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=2000)
    m.fit(X, y)
    return m, {"n_iter": int(m.n_iter_[0]), "converged": bool(m.n_iter_[0] < 2000)}


def logistic_artifact(m, prep, C):
    a = {"coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]), "C": C, "preprocessing": prep.schema()}
    a["sha256"] = sha_json(a)
    return a
