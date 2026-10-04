"""
CFB_PHASE1B_COMPONENTS -- component implementations + experiment wiring for the burned-development core engine (2019-2024; D1-D4 scoring frame). Nothing here reads 2025.
Allocation components (carries O2, QB attempts, receptions), completion, per-event yardage compounds (FFT convolution), event hazards, team volume V2, team scoring.
"""
import math
from collections import defaultdict

import numpy as np
from scipy import optimize, special, stats
from sklearn.linear_model import LogisticRegression, PoissonRegressor

import cfb_phase1_common as C
import cfb_phase1_opportunity as OP
import cfb_phase1_role_state as RS
import cfb_phase1_team_environment as TE

K_CAR, K_ATT, K_REC = 70, 80, 70
PRIMARY = ("QB", "RB", "FB")
ROLE_NAMES = RS.ROLE_FEATURES + RS.L20_FEATURES + ["T_PASS_RATE_B0", "T_PLAYS_B0", "T_RUSH_B0", "T_PASS_B0", "RATING_GAP", "ABS_RATING_GAP", "IS_HOME", "O_IS_FCS", "T_N_SEASON"]


def attach_team_labels(rows, player_games, team_games):
    """Target-game team totals (labels only, read after features): player-line sums and the non-primary (gadget) carries."""
    tot = defaultdict(lambda: {"car": 0.0, "gad": 0.0, "att": 0.0, "rec": 0.0, "comp_lines": 0.0})
    for r in player_games:
        t = tot[(r["game_id"], r["team"])]
        t["car"] += r["carries"] or 0; t["att"] += r["pass_attempts"] or 0; t["rec"] += r["receptions"] or 0; t["comp_lines"] += r["completions"] or 0
        if r["position"] not in PRIMARY:
            t["gad"] += r["carries"] or 0
    tgl = {(r["game_id"], r["team"]): r for r in team_games}
    for r in rows:
        t = tot[(r["game_id"], r["team"])]; g = tgl[(r["game_id"], r["team"])]
        r["n_car"], r["g_true"], r["n_att"] = t["car"], t["gad"], t["att"]
        r["n_rec"] = max(t["rec"], float(g["completions"] or 0)); r["n_prim"] = t["car"] - t["gad"]
    return rows


def g_zero(r):
    return 0


def g_qb_vs_rb(r):
    return 0 if r["POS_QB"] == 1 else 1


def act_carry(r):
    return int(r["y_carries"] >= 1)


def act_att(r):
    return int(r["y_pass_att"] >= 1)


def act_rec(r):
    return int(r["y_receptions"] >= 1)


def pos_group(r):
    return 0 if r["POS_QB"] == 1 else 1 if r["POS_RB"] == 1 else 2 if r["POS_WR"] == 1 else 3


def player_slices(va, tr):
    g = lambda n: np.array([r[n] for r in va], float)
    ret, trn = g("P_RETURNING") == 1, g("P_TRANSFER_NEWCOMER") == 1
    return {"QB": g("POS_QB") == 1, "RB": g("POS_RB") == 1, "WR": g("POS_WR") == 1, "TE": g("POS_TE") == 1, "returning": ret & ~trn, "transfer_newcomer": trn, "other": ~ret & ~trn, "early(<=3 games)": g("T_N_SEASON") <= 3, "established": g("T_N_SEASON") > 3,
            "home": g("IS_HOME") == 1, "away": (g("IS_HOME") == 0), "high_coverage": (g("T_N_SEASON") > 3) & (g("O_RATING_N") > 3), "low_coverage": (g("T_N_SEASON") <= 3) | (g("O_RATING_N") <= 3)}


# ------------------------------------------------------------------ O2 role-family carry allocation
class O2Carries:
    """Primary rushers (QB / RB / FB): pool N - G allocated by share x (optional activity); WR / TE / other carries G | N ~ BetaBinomial and allocated among WR / TE candidates by a Dirichlet with weights alpha_i = lam * (carries in the last 20 appearances + a0)."""
    A0, A_OTHER = 0.25, 1.0
    GMAX = 14

    def __init__(self, activity=False, logit=True):
        self.activity, self.logit = activity, logit

    def fit(self, rows):
        prim = [r for r in rows if r["position"] in PRIMARY and r["n_prim"] > 0]
        self.share = OP.ShareAlloc("logit" if self.logit else "base", "P_CARRY_SHARE_L5", "P_CARRY_SHARE_ACTIVE_L5" if self.activity else "P_CARRY_SHARE_L5", ROLE_NAMES, "y_carries", "n_prim", g_qb_vs_rb, prior_weight=1.0,
                                   use_activity=self.activity, activity_y=act_carry).fit(prim)
        # team-level gadget process: one row per team-game
        tg = {}
        for r in rows:
            tg.setdefault((r["game_id"], r["team"]), (r["n_car"], r["g_true"]))
        N = np.array([v[0] for v in tg.values()]); G = np.array([v[1] for v in tg.values()])
        self.rho = float(G.sum() / N.sum())
        self.kg = OP.fit_kappa(G.astype(int), N, np.full(len(N), self.rho))
        # gadget allocation concentration scale lam by ML on WR / TE candidate rows (single parameter)
        gad_rows = [r for r in rows if r["position"] not in PRIMARY]
        self.lam = self._fit_lam(gad_rows)
        return self

    GAD_FEATURES = ("bias", "log1p_car_l20", "car_share_l5x10", "apps_l5_over5", "transfer_newcomer", "rec_share_l5x10", "returning", "TE:bias", "TE:log1p_car_l20", "TE:car_share_l5x10", "TE:apps_l5_over5", "TE:transfer_newcomer", "TE:rec_share_l5x10", "TE:returning")

    @staticmethod
    def _gx(rows):
        g = lambda n: np.array([r[n] for r in rows], float)
        sh = np.nan_to_num(g("P_CARRY_SHARE_L5")); rs = np.nan_to_num(g("P_REC_SHARE_L5"))
        base = np.column_stack([np.ones(len(rows)), np.log1p(g("P_CARRIES_L20")), sh * 10, g("P_APPS_L5") / 5.0, g("P_TRANSFER_NEWCOMER"), rs * 10, g("P_RETURNING")])
        te = g("POS_TE")[:, None]
        return np.hstack([base, base * te])                                                                  # position-specific coefficients: WR / other vs TE

    def _group_alpha(self, rows, theta):
        """Dirichlet weights alpha_i = exp(theta . x_i) (log-linear), OTHER weight exp(theta[-1]); returns (alpha_i, W_team)."""
        X = self._gx(rows); alpha = np.exp(np.clip(X @ theta[:-1], -12, 8))
        keys = {}; gi = np.array([keys.setdefault((r["game_id"], r["team"]), len(keys)) for r in rows])
        Wg = np.zeros(len(keys)); np.add.at(Wg, gi, alpha)
        return alpha, Wg[gi] + math.exp(np.clip(theta[-1], -12, 8))

    def _fit_lam(self, rows):
        """Training-only maximum likelihood of the log-linear Dirichlet weights (and the OTHER weight) of the gadget allocation from WR / TE / other candidate rows, given the true gadget mass of the game."""
        if not rows:
            return np.zeros(len(self.GAD_FEATURES) + 1)
        y = np.array([r["y_carries"] for r in rows]); gt = np.array([r["g_true"] for r in rows]).astype(int)
        ok = gt >= y
        def nll(theta):
            a, W = self._group_alpha(rows, theta)
            return -float(stats.betabinom.logpmf(y[ok], gt[ok], np.maximum(a[ok], 1e-9), np.maximum(W[ok] - a[ok], 1e-9)).sum()) + 1e-3 * float(np.sum(theta[1:-1] ** 2))
        t0 = np.zeros(len(self.GAD_FEATURES) + 1); t0[0] = -2.0; t0[-1] = 0.0
        res = optimize.minimize(nll, t0, method="L-BFGS-B")
        return res.x

    def pmf(self, rows):
        n = len(rows); N = np.array([r["n_car"] for r in rows]).astype(int)
        prim = np.array([r["position"] in PRIMARY for r in rows])
        out = np.zeros((n, K_CAR + 1))
        # P(g | N) for every row
        gs = np.arange(self.GMAX + 1)
        pg = stats.betabinom.pmf(gs[None, :], N[:, None], self.rho * self.kg["kappa"], (1 - self.rho) * self.kg["kappa"])             # [n, G+1]
        pg = pg / np.maximum(pg.sum(axis=1, keepdims=True), 1e-12)
        if prim.any():
            idx = np.where(prim)[0]; sub = [rows[i] for i in idx]
            s = self.share.share(sub); gk = np.array([0 if r["POS_QB"] == 1 else 1 for r in sub]); kap = np.array([self.share.kappa[k]["kappa"] if k in self.share.kappa else 5.0 for k in gk])
            acc = np.zeros((len(idx), K_CAR + 1))
            for g in gs:
                pool = np.maximum(N[idx] - g, 0)
                acc += pg[idx, g][:, None] * OP.bb_pmf_rows(pool, s, kap, K_CAR)
            if self.activity:
                a = self.share.activity(sub)[:, None]; acc = a * acc; acc[:, 0] += 1.0 - a[:, 0]
            out[idx] = acc
        if (~prim).any():
            idx = np.where(~prim)[0]; sub = [rows[i] for i in idx]
            a, W = self._group_alpha(sub, self.lam)
            acc = np.zeros((len(idx), K_CAR + 1))
            for g in gs:
                acc += pg[idx, g][:, None] * stats.betabinom.pmf(np.arange(K_CAR + 1)[None, :], g, np.maximum(a, 1e-6)[:, None], np.maximum(W - a, 1e-6)[:, None])
            out[idx] = acc
        return out

    def artifact(self):
        return {"component": "O2_role_family_carry_allocation", "activity_layer": self.activity, "primary": self.share.artifact(), "gadget_rho": self.rho, "gadget_kappa": self.kg, "gadget_dirichlet_theta": [float(x) for x in self.lam], "gadget_features": list(self.GAD_FEATURES) + ["OTHER_log_weight"]}


def o1_variant(kind):
    """Phase 1 reference arms on the SAME rows: O1 B0 (recent shrunk share) / O1 C1 (single binomial-logistic model for every position), given realized team carries."""
    def fp(tr, va):
        m = OP.ShareAlloc(kind, "P_CARRY_SHARE_L5", "P_CARRY_SHARE_L5", RS.ROLE_FEATURES + ["T_PASS_RATE_B0", "T_PLAYS_B0"], "y_carries", "n_car", pos_group, prior_weight=1.0).fit([r for r in tr if r["n_car"] > 0])
        return m.pmf(va, K_CAR)
    return fp


def o2_variant(activity, logit=True):
    def fp(tr, va):
        return O2Carries(activity=activity, logit=logit).fit(tr).pmf(va)
    return fp


# ------------------------------------------------------------------ allocation experiments for QB attempts / receptions
def alloc_variant(kind, share_col, act_col, y_key, n_key, K, groups, activity=False, act_y=None, names=None):
    def fp(tr, va):
        m = OP.ShareAlloc(kind, share_col, act_col if activity else share_col, names or ROLE_NAMES, y_key, n_key, groups, prior_weight=1.0, use_activity=activity, activity_y=act_y).fit([r for r in tr if r[n_key] > 0])
        return m.pmf(va, K)
    return fp


# ------------------------------------------------------------------ completion / hazards (binomial processes given the realized opportunity)
def _arr(rows, names):
    return {n: np.array([r[n] for r in rows], dtype=float) for n in names}


def binom_pmf_rows(n, p, K):
    n = np.asarray(n).astype(int)
    return stats.binom.pmf(np.arange(K + 1)[None, :], n[:, None], np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)[:, None])


def bbin_pmf_rows(n, p, kappa, K):
    return OP.bb_pmf_rows(n, np.clip(p, 1e-5, 1 - 1e-5), kappa, K)


class RateComponent:
    """Opportunity-conditioned event rate: Y | n ~ BetaBinomial(n, p_i, kappa). p_i: 'base' = as-of shrunk player rate (events_L20 + k*p_group)/(opp_L20 + k) ; 'logit' = binomial-logistic on [logit(shrunk rate), context]. kappa by training ML."""

    def __init__(self, kind, ev_col, opp_col, y_key, n_key, prior_k, group_fn, ctx_names=(), kappa_fixed=None):
        self.kind, self.ev, self.opp, self.y_key, self.n_key, self.k0, self.group_fn, self.ctx, self.kappa_fixed = kind, ev_col, opp_col, y_key, n_key, prior_k, group_fn, list(ctx_names), kappa_fixed

    def _shrunk(self, rows):
        g = np.array([self.group_fn(r) for r in rows])
        ev = np.array([r[self.ev] for r in rows], float); op = np.array([r[self.opp] for r in rows], float)
        pg = np.array([self.pg.get(k, self.pall) for k in g])
        return (ev + self.k0 * pg) / (op + self.k0)

    def fit(self, rows):
        rows = [r for r in rows if r[self.n_key] >= 1]
        y = np.array([r[self.y_key] for r in rows], float); n = np.array([r[self.n_key] for r in rows], float); g = np.array([self.group_fn(r) for r in rows])
        self.pall = float(y.sum() / n.sum()); self.pg = {k: float(y[g == k].sum() / max(n[g == k].sum(), 1)) for k in set(g.tolist())}
        p0 = self._shrunk(rows)
        if self.kind == "logit":
            sh = np.log(p0 / (1 - p0)); X = np.column_stack([sh] + [np.array([r[c] for r in rows], float) for c in self.ctx])
            self.prep = TE.Prep(["sh"] + self.ctx).fit({**{"sh": sh}, **{c: np.array([r[c] for r in rows], float) for c in self.ctx}})
            self.lg = OP.binomial_logit_fit(self.prep.transform({**{"sh": sh}, **{c: np.array([r[c] for r in rows], float) for c in self.ctx}}), y, n)
        p = self._p(rows)
        self.kappa = self.kappa_fixed or OP.fit_kappa(y.astype(int), n, p)["kappa"]
        return self

    def _p(self, rows):
        p0 = np.clip(self._shrunk(rows), 1e-5, 1 - 1e-5)
        if self.kind == "base":
            return p0
        sh = np.log(p0 / (1 - p0))
        d = {"sh": sh, **{c: np.array([r[c] for r in rows], float) for c in self.ctx}}
        return self.lg.predict_proba(self.prep.transform(d))[:, 1]

    def pmf(self, rows, K):
        return bbin_pmf_rows(np.array([r[self.n_key] for r in rows]), self._p(rows), self.kappa, K)

    def artifact(self):
        return {"kind": self.kind, "prior_rate_by_group": self.pg, "prior_k": self.k0, "kappa": self.kappa, "context": self.ctx}


# ------------------------------------------------------------------ V2 coherent team volume: plays N -> dropbacks D | N -> rush R = N - D ; sacks S | D ; attempts A = D - S
K_N = 150


def _nb2_rows(mu, alpha, K):
    pm, _ = C.nb2_pmf_matrix(mu, alpha, K)
    return pm


class V2Volume:
    """N ~ NB2 (B0 blend or GLM on the shared state); D | N ~ BetaBinomial(N, pi, kappa) with pi from a binomial-logistic model; S | D ~ Binomial(D, s). Exact joint evaluation (no Monte Carlo)."""

    def __init__(self, n_kind="c1", pi_kind="logit"):
        self.n_kind, self.pi_kind = n_kind, pi_kind

    def fit(self, team):
        tab = TE.table_arrays(team, TE.TEAM_FEATURES_V2)
        yN = np.array([r["y_plays"] for r in team], float)
        self.nm = TE.C1Count(TE.TEAM_FEATURES_V2).fit(tab, yN) if self.n_kind == "c1" else TE.B0Count("T_PLAYS_B0").fit(tab, yN)
        D_ = np.array([r["y_pass"] + r["y_sacks"] for r in team], float); S_ = np.array([r["y_sacks"] for r in team], float)
        self.pi0 = float(D_.sum() / yN.sum()); self.s = float(S_.sum() / D_.sum())
        if self.pi_kind == "logit":
            self.prep = TE.Prep(TE.TEAM_FEATURES_V2).fit(tab); self.lg = OP.binomial_logit_fit(self.prep.transform(tab), D_, yN)
        pi = self._pi(team, tab)
        self.kappa = OP.fit_kappa(D_.astype(int), yN.astype(int), pi)["kappa"]
        return self

    def _pi(self, team, tab=None):
        tab = tab or TE.table_arrays(team, TE.TEAM_FEATURES_V2)
        if self.pi_kind == "const":
            return np.full(len(team), self.pi0)
        return self.lg.predict_proba(self.prep.transform(tab))[:, 1]

    def pmfs(self, team):
        """-> dict of pmf matrices: 'N' plays, 'D' dropbacks, 'R' rush plays (= N - D), 'A' pass attempts (= D - S)."""
        tab = TE.table_arrays(team, TE.TEAM_FEATURES_V2); n = len(team)
        pN = _nb2_rows(self.nm.mean(tab), self.nm.alpha(), K_N); pi = np.clip(self._pi(team, tab), 1e-4, 1 - 1e-4)
        R = np.zeros((n, K_N + 1)); Dm = np.zeros((n, K_N + 1)); A = np.zeros((n, K_N + 1))
        ns = np.arange(K_N + 1)
        dd = np.arange(K_N + 1)
        B = stats.binom.pmf((dd[:, None] - dd[None, :]), dd[:, None], self.s)                                  # [d, a] = P(sacks = d - a | D = d)
        B = np.where(dd[:, None] >= dd[None, :], B, 0.0)
        for a in range(0, n, 40):
            sl = slice(a, min(a + 40, n)); c = sl.stop - sl.start
            bb = stats.betabinom.pmf(dd[None, None, :], ns[None, :, None], (pi[sl] * self.kappa)[:, None, None], ((1 - pi[sl]) * self.kappa)[:, None, None])        # [c, n, d]
            J = pN[sl][:, :, None] * bb
            Dm[sl] = J.sum(axis=1)
            for i in range(c):                                                                               # R = n - d
                R[sl.start + i] = _antidiag(J[i])
            A[sl] = Dm[sl] @ B
        return {"N": pN, "D": Dm, "R": R, "A": A}

    def artifact(self):
        return {"component": "V2_team_volume", "plays": self.nm.nb, "plays_model": self.n_kind, "pass_share_model": self.pi_kind, "pass_share_prior": self.pi0, "kappa": self.kappa, "sack_rate": self.s}


def _antidiag(J):
    """R[r] = sum_n J[n, n - r] (rush plays = plays - dropbacks)."""
    K = J.shape[0] - 1
    out = np.zeros(K + 1)
    for r in range(K + 1):
        n = np.arange(r, K + 1)
        out[r] = J[n, n - r].sum()
    return out


class IndependentVolume:
    def __init__(self, rush_kind="c1", pass_kind="b0"):
        self.rk, self.pk = rush_kind, pass_kind

    def fit(self, team):
        tab = TE.table_arrays(team, TE.TEAM_FEATURES_V2)
        yR = np.array([r["y_rush"] for r in team], float); yA = np.array([r["y_pass"] for r in team], float)
        self.rm = TE.C1Count(TE.TEAM_FEATURES_V2).fit(tab, yR) if self.rk == "c1" else TE.B0Count("T_RUSH_B0").fit(tab, yR)
        self.am = TE.C1Count(TE.TEAM_FEATURES_V2).fit(tab, yA) if self.pk == "c1" else TE.B0Count("T_PASS_B0").fit(tab, yA)
        self.sack_mean = float(np.mean([r["y_sacks"] for r in team]))
        return self

    def pmfs(self, team):
        tab = TE.table_arrays(team, TE.TEAM_FEATURES_V2)
        R = _nb2_rows(self.rm.mean(tab), self.rm.alpha(), K_N); A = _nb2_rows(self.am.mean(tab), self.am.alpha(), K_N)
        S = stats.poisson.pmf(np.arange(K_N + 1), self.sack_mean)
        L = 512
        N = np.fft.irfft(np.fft.rfft(R, L, axis=1) * np.fft.rfft(A, L, axis=1) * np.fft.rfft(S, L)[None, :], L, axis=1)[:, :K_N + 1]
        return {"R": R, "A": A, "N": np.clip(N, 0, None)}


def volume_variants(target):
    """target in {'R','A','N'}: pmf of the V2 construction vs the independent constructions for the same marginal."""
    def v2(n_kind, pi_kind):
        return lambda tr, va: V2Volume(n_kind, pi_kind).fit(tr).pmfs(va)[target]
    def ind(rk, pk):
        return lambda tr, va: IndependentVolume(rk, pk).fit(tr).pmfs(va)[target]
    return [("IND_T1B0_T2B0", ind("b0", "b0")), ("IND_T1C1_T2B0", ind("c1", "b0")), ("V2_N_B0_pi_const", v2("b0", "const")), ("V2_N_C1_pi_const", v2("c1", "const")), ("V2_N_C1_pi_logit", v2("c1", "logit"))]
