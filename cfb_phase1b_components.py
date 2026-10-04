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
        self.share = OP.ShareAlloc("logit" if self.logit else "base", "P_CARRY_SHARE_L5", "P_CARRY_SHARE_ACTIVE_L5" if self.activity else "P_CARRY_SHARE_L5", ROLE_NAMES, "y_carries", "n_prim", lambda r: 0 if r["POS_QB"] == 1 else 1, prior_weight=1.0,
                                   use_activity=self.activity, activity_y=lambda r: int(r["y_carries"] >= 1)).fit(prim)
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

    def _group_alpha(self, rows, lam):
        alpha = np.array([lam * (r["P_CARRIES_L20"] + self.A0) for r in rows])
        key = [(r["game_id"], r["team"]) for r in rows]
        W = defaultdict(float)
        for k, a in zip(key, alpha):
            W[k] += a
        Wt = np.array([W[k] + lam * self.A_OTHER for k in key])
        return alpha, Wt

    def _fit_lam(self, rows):
        if not rows:
            return 1.0
        y = np.array([r["y_carries"] for r in rows]); gt = np.array([r["g_true"] for r in rows]).astype(int)
        def nll(t):
            lam = math.exp(t); a, W = self._group_alpha(rows, lam)
            ok = gt >= y
            return -float(stats.betabinom.logpmf(y[ok], gt[ok], np.maximum(a[ok], 1e-6), np.maximum(W[ok] - a[ok], 1e-6)).sum())
        res = optimize.minimize_scalar(nll, bounds=(math.log(0.05), math.log(50.0)), method="bounded", options={"xatol": 1e-4})
        return float(math.exp(res.x))

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
        return {"component": "O2_role_family_carry_allocation", "activity_layer": self.activity, "primary": self.share.artifact(), "gadget_rho": self.rho, "gadget_kappa": self.kg, "gadget_lambda": self.lam, "a0": self.A0, "a_other": self.A_OTHER}


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
