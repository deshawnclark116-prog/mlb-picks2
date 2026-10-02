"""
NFL_PHASE1_EFFICIENCY  (Phase 1B generic engine, shadow research)

Distribution models for WHAT HAPPENS ON an opportunity (a carry, a target, a dropback). Phase 1A supplies the number of
opportunities; nothing here recomputes them.

Model family (one engine for every component):
  1. hierarchical shrunk empirical pmf   league -> position -> player   (decayed, as-of counts; kappa tuned on validation)
  2. exponential-tilt adjustment          p_k  ∝  base_k * exp( (x·W) · phi_k )       (few basis functions phi over the bins)
     - K=2 bins, phi=[0,1]  -> logistic hazard with an offset (catch, sack, INT, TD ...)
     - K=40 yard bins, phi = {negative, explosive, linear yards} -> parametric matchup tilt over the baseline pmf
     Features are z-scored on TRAIN; L2 shrinkage on W is tuned on VALIDATION; nested feature families are fit by
     dropping columns (no importance-based evidence anywhere).
  3. scoring: per-opportunity log score, ordinal CRPS over bin values, game-level CRPS via common-random-number sampling of
     sums over the actual opportunity count (isolates the efficiency layer), tail-bin log scores and reliability.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

EPS = 1e-9


# ------------------------------------------------------------------ hierarchical pmf
def shrink(counts, prior, kappa):
    """(counts + kappa * prior) / (sum(counts) + kappa); counts [n,K] or [K], prior [n,K] or [K] pmf."""
    counts = np.asarray(counts, float)
    n = counts.sum(-1, keepdims=True)
    return (counts + kappa * prior) / (n + kappa)


def hier_pmf(c_player, c_pos, c_lg, kappa_player, kappa_pos, floor=1e-6):
    """Player decayed counts shrunk to position, position shrunk to league. All arrays [n,K] (or broadcastable)."""
    c_lg = np.asarray(c_lg, float)
    p_lg = (c_lg + 1.0 / c_lg.shape[-1]) / (c_lg.sum(-1, keepdims=True) + 1.0)
    p_pos = shrink(c_pos, p_lg, kappa_pos)
    p = shrink(c_player, p_pos, kappa_player)
    p = np.maximum(p, floor)
    return p / p.sum(-1, keepdims=True)


# ------------------------------------------------------------------ tilt model
class Tilt:
    """Softmax tilt over a fixed baseline pmf.  logits = log(base) + (X W) Phi^T ;  W [d,b]; intercept column is unpenalized."""

    def __init__(self, phi, l2=1.0, unpenalized=(0,)):
        self.phi = np.asarray(phi, float)              # [K,b]
        self.l2 = l2
        self.unpen = set(unpenalized)
        self.W = None
        self.mu = None
        self.sd = None

    def _design(self, X, fit=False):
        X = np.asarray(X, float)
        if fit:
            self.mu = X.mean(0); self.sd = X.std(0); self.sd[self.sd < 1e-9] = 1.0
        Z = (X - self.mu) / self.sd
        return np.column_stack([np.ones(len(Z)), Z])

    def fit(self, C, logB, X, w=None):
        C = np.asarray(C, float); n, K = C.shape
        Z = self._design(X, fit=True)
        d, b = Z.shape[1], self.phi.shape[1]
        N = C.sum(1)
        pen = np.array([0.0 if j in self.unpen else 1.0 for j in range(d)])[:, None]
        ntot = max(N.sum(), 1.0)

        def f(v):
            W = v.reshape(d, b)
            L = logB + (Z @ W) @ self.phi.T
            lse = logsumexp(L, axis=1, keepdims=True)
            P = np.exp(L - lse)
            nll = (-(C * (L - lse)).sum() + 0.5 * self.l2 * (pen * W ** 2).sum()) / ntot
            G = (Z.T @ ((N[:, None] * P - C) @ self.phi) + self.l2 * pen * W) / ntot
            return nll, G.ravel()

        r = minimize(f, np.zeros(d * b), jac=True, method="L-BFGS-B", options={"maxiter": 200})
        self.W = r.x.reshape(d, b)
        self.n_fit = float(ntot)
        return self

    def predict(self, logB, X):
        Z = self._design(X)
        L = logB + (Z @ self.W) @ self.phi.T
        return np.exp(L - logsumexp(L, axis=1, keepdims=True))


# ------------------------------------------------------------------ scoring
def logscore(P, C, floor=1e-9):
    """Per-record total negative log likelihood, normalised later by opportunity count."""
    return -(np.asarray(C) * np.log(np.maximum(P, floor))).sum(1)


def ordinal_crps_records(P, C, values):
    """Sum over the record's opportunities of the ordinal CRPS of a single-opportunity pmf with bin values `values` (sorted)."""
    P = np.asarray(P); C = np.asarray(C)
    dv = np.diff(values)
    FP = np.cumsum(P, 1)[:, :-1]
    tot = np.zeros(len(P))
    cum = np.cumsum(C, 1)[:, :-1]                       # sum of indicators across opportunities: sum_i 1[y_i <= k]
    N = C.sum(1, keepdims=True)
    # sum_i (F(k) - 1[y_i<=k])^2 = N F^2 - 2 F cum + cum   (indicator squared == indicator)
    tot = ((N * FP ** 2 - 2 * FP * cum + cum) * dv).sum(1)
    return tot


def bin_values(hist_by_grid, M, grid):
    """League train mean of the underlying integer value within each bin (bin -> real value used for sampling and CRPS)."""
    num = (M * (hist_by_grid[:, None] * grid[:, None])).sum(0)
    den = (M * hist_by_grid[:, None]).sum(0)
    mid = np.array([grid[M[:, k] > 0].mean() for k in range(M.shape[1])])
    return np.where(den > 0, num / np.maximum(den, 1e-9), mid)


def sample_sums(P, N, vals, draws, seed_keys, rng_base=7, seed_fn=None):
    """Distribution of sum over N_i iid draws from P_i, `draws` samples per record with common random numbers keyed per record.
    Returns [n, draws] of sums using bin values `vals`."""
    n = len(P)
    out = np.zeros((n, draws))
    for i in range(n):
        ni = int(N[i])
        if ni == 0:
            continue
        rng = np.random.default_rng([rng_base, seed_keys[i]])
        m = rng.multinomial(ni, P[i] / P[i].sum(), size=draws)
        out[i] = m @ vals
    return out


def crps_samples(S, y):
    """Sample CRPS per row: E|X-y| - 0.5 E|X-X'|  (S [n,m])."""
    n, m = S.shape
    t1 = np.abs(S - y[:, None]).mean(1)
    Ss = np.sort(S, 1)
    idx = np.arange(1, m + 1)
    t2 = (2 * (Ss * (2 * idx - m - 1)).sum(1)) / (m * m)     # E|X-X'| = 2/m^2 sum (2i-m-1) x_(i)
    return t1 - 0.5 * t2


# ================================================================== component orchestration (shared by all Phase 1B modules)
import nfl_phase1_common as C  # noqa: E402

FAMILIES = ["team", "opp", "scheme", "personnel"]           # B1 = +team, B2 = +opp, B3 = +scheme, B4 = +personnel, B5 = +interaction
NESTED = {"B0": [], "B1": ["team"], "B2": ["team", "opp"], "B3": ["team", "opp", "scheme"],
          "B4": ["team", "opp", "scheme", "personnel"], "B5": ["team", "opp", "scheme", "personnel", "interaction"]}


def masks(R):
    tr = np.array([C.TRAIN(s, w) for s, w in zip(R.s, R.w)])
    va = np.array([C.VALID(s, w) for s, w in zip(R.s, R.w)])
    dv = np.array([C.DEV(s, w) for s, w in zip(R.s, R.w)])
    act = getattr(R, "active", None)
    if act is not None:
        tr, va, dv = tr & act, va & act, dv & act
    return tr, va, dv


def impute_stack(R, fams, fit_mask, extra=None):
    """Concatenate the requested families; NaN -> train mean of the column (column-wise), + extra blocks."""
    blocks = [R.fam[f] for f in fams if f != "interaction"]
    if extra is not None and "interaction" in fams:
        blocks.append(extra)
    if not blocks:
        return np.zeros((R.n, 0))
    X = np.column_stack(blocks).astype(float)
    mu = np.nanmean(X[fit_mask], 0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    ix = np.where(np.isnan(X))
    X[ix] = mu[ix[1]]
    return X


def hier_base(R, key_pl, key_pp, key_lg, gi, kp, kpos):
    return hier_pmf(R.base[key_pl][:, gi, :], R.base[key_pp][:, gi, :], R.base[key_lg][:, gi, :], kp, kpos)


def tune_hier(R, cnt, key_pl, key_pp, key_lg, tr, va, gis=(0, 1, 2), kps=(5, 10, 20, 40, 80, 160, 1e9), kpos=(20, 100)):
    """Choose (decay, kappa_player, kappa_pos) by validation log score of the baseline pmf; returns dict + table."""
    best, table = None, []
    for gi in gis:
        for kp in kps:
            for kq in kpos:
                P = hier_base(R, key_pl, key_pp, key_lg, gi, kp, kq)
                ls = logscore(P[va], cnt[va]).sum() / max(cnt[va].sum(), 1.0)
                table.append({"gamma_idx": gi, "kappa_player": kp, "kappa_pos": kq, "valid_logscore": float(ls)})
                if best is None or ls < best[0]:
                    best = (ls, gi, kp, kq)
    return {"gamma_idx": best[1], "kappa_player": best[2], "kappa_pos": best[3], "valid_logscore": float(best[0])}, table


def fit_family(logB, cnt, X, phi, tr, va, dv, l2_grid=(30.0, 300.0, 3000.0)):
    """Fit a tilt on TRAIN, choose l2 on VALID, refit on TRAIN+VALID (audited) and return predictions for every row."""
    if X.shape[1] == 0:
        P = np.exp(logB); P = P / P.sum(1, keepdims=True)
        return P, {"l2": None, "n_features": 0, "valid_logscore_train_only_fit": float(logscore(P[va], cnt[va]).sum() / max(cnt[va].sum(), 1.0)), "Pv": P[va]}
    best = None
    for l2 in l2_grid:
        m = Tilt(phi, l2=l2).fit(cnt[tr], logB[tr], X[tr])
        Pv = m.predict(logB[va], X[va])
        ls = logscore(Pv, cnt[va]).sum() / max(cnt[va].sum(), 1.0)
        if best is None or ls < best[0]:
            best = (ls, l2, Pv)
    fit_mask = tr | va
    m = Tilt(phi, l2=best[1]).fit(cnt[fit_mask], logB[fit_mask], X[fit_mask])
    return m.predict(logB, X), {"l2": best[1], "n_features": int(X.shape[1]), "valid_logscore_train_only_fit": float(best[0]), "Pv": best[2]}


def score_rows(P, cnt, vals):
    n_opp = np.maximum(cnt.sum(1), 1e-9)
    return {"ls": logscore(P, cnt), "crps": ordinal_crps_records(P, cnt, vals), "n": cnt.sum(1)}


def period_summary(sc, R, sub_mask, ref=None):
    """Aggregate per-opportunity log score and CRPS for 2025 / 2026 wk1-3 / combined (development)."""
    out = {}
    for tag, m in (("2025", sub_mask & (R.s == 2025)), ("2026_wk1_3", sub_mask & (R.s == 2026)), ("combined", sub_mask)):
        if m.sum() == 0:
            continue
        n = sc["n"][m].sum()
        out[tag] = {"n_records": int(m.sum()), "n_opps": float(n), "logscore": float(sc["ls"][m].sum() / n), "crps": float(sc["crps"][m].sum() / n)}
    if ref is not None:
        m = sub_mask
        blocks = np.array([f"{a}-{b}" for a, b in zip(R.s[m], R.w[m])])
        n = sc["n"][m]
        # record-level contribution divided by the same normaliser: paired week-block bootstrap of the difference in totals
        imp_ls, p_ls = C.block_boot(sc["ls"][m], ref["ls"][m], blocks)
        imp_cr, p_cr = C.block_boot(sc["crps"][m], ref["crps"][m], blocks)
        tot = n.sum()
        out["vs_ref"] = {"logscore_improvement_per_opp": float(imp_ls * m.sum() / tot) if tot else None, "p_logscore_not_better": p_ls,
                         "crps_improvement_per_opp": float(imp_cr * m.sum() / tot) if tot else None, "p_crps_not_better": p_cr}
    return out


def group_calibration(P, cnt, groups, mask):
    """Tail calibration: for each named group of bins, mean predicted probability vs observed frequency, and binary log loss."""
    out = {}
    N = cnt[mask].sum()
    for name, idx in groups.items():
        pg = P[mask][:, idx].sum(1)
        obs = cnt[mask][:, idx].sum(1)
        n = cnt[mask].sum(1)
        pred_mean = float((pg * n).sum() / N); obs_rate = float(obs.sum() / N)
        pc = np.clip(pg, 1e-6, 1 - 1e-6)
        ll = float(-(obs * np.log(pc) + (n - obs) * np.log(1 - pc)).sum() / N)
        out[name] = {"pred": round(pred_mean, 5), "obs": round(obs_rate, 5), "logloss": round(ll, 5)}
    return out


def nested_study(R, cnt, vals, predict_fn, extra=None, groups=None, ref_name="B0"):
    """Run the nested feature-family ablation B0..B5.

    predict_fn(X, level) -> (P [n,K], info). Selection uses ONLY train-fit VALID log score (a level is kept when it improves
    the previous kept level by >= 1e-4 nats per opportunity); development scores are reported for every level.
    Returns (report, {level: P})."""
    tr, va, dv = masks(R)
    rep, preds, valid_ls = {}, {}, {}
    refsc = None
    for lvl, fams in NESTED.items():
        X = impute_stack(R, fams, tr | va, extra)
        P, info = predict_fn(X, lvl)
        preds[lvl] = P
        sc = score_rows(P, cnt, vals)
        if lvl == ref_name:
            refsc = sc
        entry = {"families": fams, "fit": {k: v for k, v in info.items() if k != "Pv"}, "development": period_summary(sc, R, dv, None if lvl == ref_name else refsc)}
        if groups:
            entry["tail_calibration"] = {"combined": group_calibration(P, cnt, groups, dv)}
        rep[lvl] = entry
        valid_ls[lvl] = info.get("valid_logscore_train_only_fit")
    # leave-one-family-out from B4 (does each family earn its place given the others?). Reported, not used for selection.
    loo = {}
    sc4 = score_rows(preds["B4"], cnt, vals)
    for fam_out in ("team", "opp", "scheme", "personnel"):
        fams = [f for f in NESTED["B4"] if f != fam_out]
        X = impute_stack(R, fams, tr | va, extra)
        P, _ = predict_fn(X, f"B4-{fam_out}")
        loo[fam_out] = {"development_B4_vs_without": period_summary(sc4, R, dv, score_rows(P, cnt, vals)).get("vs_ref")}
    kept = ref_name
    for lvl in list(NESTED)[1:]:
        if valid_ls[lvl] is not None and valid_ls[kept] is not None and valid_ls[kept] - valid_ls[lvl] >= 1e-4:
            kept = lvl
    return {"levels": rep, "selected_level": kept, "leave_one_family_out_from_B4": loo, "selection_rule": "highest nested level whose TRAIN-fit VALID log score improves the previous kept level by >= 1e-4 nats/opportunity",
            "valid_logscore_by_level": valid_ls}, preds
