"""
NHL_SOG_PHASE1A_METRICS -- discrete CRPS / NLL / PIT / interval coverage / threshold diagnostics / game-level aggregation / blocked bootstrap (registered in phase1a_protocol.json).
Distributions are represented by an exact PMF matrix over 0..K per row (K extended dynamically for parametric families until survival < 1e-10; hard ceiling 200).
"""
import hashlib
import math

import numpy as np
from scipy import stats

SF_TOL = 1e-10
K_CEIL = 200
NLL_FLOOR = 1e-15
BOOT_SEED = 20261002
BOOT_REPS = 10000


class InvalidDistribution(RuntimeError):
    pass


# ------------------------------------------------------------------ distributions -> PMF matrices
def pmf_matrix(kind, params, y, chunk=20000):
    """Yield (slice, pmf[n, K+1], sf_at_K) per chunk. kind: 'pmf' (params['pmf'] fixed support, rows x K+1), 'poisson' (mu), 'nb2' (mu, alpha), 'mix_nb2' (p, mu, alpha) with P(0)=(1-p)+p*NB(0), P(k>0)=p*NB(k)."""
    n = len(y)
    for a in range(0, n, chunk):
        sl = slice(a, min(a + chunk, n))
        if kind == "pmf":
            pm = params["pmf"][sl]
            yield sl, pm, np.zeros(pm.shape[0])
            continue
        mu = params["mu"][sl]
        K = max(40, int(np.max(y[sl])) + 1)
        while True:
            ks = np.arange(K + 1)[None, :]
            if kind == "poisson":
                pm = stats.poisson.pmf(ks, mu[:, None]); sf = stats.poisson.sf(K, mu)
            else:
                al = params["alpha"]
                r = 1.0 / al; pp = r / (r + mu)
                pm = stats.nbinom.pmf(ks, r, pp[:, None]); sf = stats.nbinom.sf(K, r, pp)
                if kind == "mix_nb2":
                    p = params["p"][sl]
                    pm = p[:, None] * pm; pm[:, 0] += 1.0 - p; sf = p * sf
            if np.max(sf) < SF_TOL:
                break
            if K >= K_CEIL:
                raise InvalidDistribution(f"tail probability {float(np.max(sf)):.3e} >= {SF_TOL} at k={K_CEIL}: distribution invalid")
            K = min(K * 2, K_CEIL)
        yield sl, pm, sf


def row_metrics(kind, params, y, model_id, game_ids, player_ids):
    """Per-row CRPS, NLL, PIT u, central-interval bounds, threshold probabilities, floor hits."""
    n = len(y)
    out = {"crps": np.zeros(n), "nll": np.zeros(n), "pit": np.zeros(n), "floor_hits": 0, "max_K": 0, "p_ge": np.zeros((n, 5)), "lo": {}, "hi": {}, "max_sf": 0.0}
    for c in (0.5, 0.8, 0.9):
        out["lo"][c] = np.zeros(n, dtype=np.int32); out["hi"][c] = np.zeros(n, dtype=np.int32)
    v = pit_v(model_id, game_ids, player_ids)
    for sl, pm, sf in pmf_matrix(kind, params, y):
        yy = y[sl]; K = pm.shape[1] - 1
        out["max_K"] = max(out["max_K"], K); out["max_sf"] = max(out["max_sf"], float(np.max(sf)))
        cdf = np.cumsum(pm, axis=1)
        ind = (yy[:, None] <= np.arange(K + 1)[None, :]).astype(float)
        out["crps"][sl] = ((cdf - ind) ** 2).sum(axis=1)
        py = pm[np.arange(len(yy)), yy]
        fl = py < NLL_FLOOR
        out["floor_hits"] += int(fl.sum())
        out["nll"][sl] = -np.log(np.maximum(py, NLL_FLOOR))
        below = np.where(yy > 0, cdf[np.arange(len(yy)), np.maximum(yy - 1, 0)], 0.0)
        out["pit"][sl] = below + v[sl] * py
        for j in range(5):
            out["p_ge"][sl, j] = 1.0 - cdf[:, min(j, K)]
        for c in (0.5, 0.8, 0.9):
            out["lo"][c][sl] = (cdf >= (1 - c) / 2 - 1e-12).argmax(axis=1)
            out["hi"][c][sl] = (cdf >= (1 + c) / 2 - 1e-12).argmax(axis=1)
    return out


def pit_v(model_id, game_ids, player_ids):
    """Deterministic v in [0,1): first 8 bytes of SHA256(model_id|game_id|player_id) / 2^64 (no global random draw)."""
    out = np.empty(len(game_ids))
    for i, (g, p) in enumerate(zip(game_ids, player_ids)):
        out[i] = int.from_bytes(hashlib.sha256(f"{model_id}|{int(g)}|{int(p)}".encode()).digest()[:8], "big") / 2.0 ** 64
    return out


def ks_uniform(u):
    u = np.sort(u); n = len(u)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def crps_point(F_vals, y):
    """Discrete CRPS from an explicit CDF over 0..K (test helper)."""
    ks = np.arange(len(F_vals))
    return float(((np.asarray(F_vals) - (y <= ks)) ** 2).sum())


# ------------------------------------------------------------------ aggregation
def game_macro(values, game_ids):
    """Mean over games of the mean of `values` within each game."""
    order = np.argsort(game_ids, kind="stable")
    g = np.asarray(game_ids)[order]; v = np.asarray(values)[order]
    uniq, start = np.unique(g, return_index=True)
    sums = np.add.reduceat(v, start); cnt = np.diff(np.append(start, len(g)))
    per_game = sums / cnt
    return float(per_game.mean()), uniq, per_game


def coverage(lo, hi, y):
    return float(np.mean((y >= lo) & (y <= hi)))


def threshold_diag(p_ge, y, nbins=10):
    out = {}
    for j in range(5):
        p = p_ge[:, j]; e = (y >= j + 1).astype(float)
        order = np.argsort(p, kind="stable")
        bins = np.array_split(order, nbins)
        ece = sum(len(b) / len(p) * abs(p[b].mean() - e[b].mean()) for b in bins if len(b))
        out[f"P(SOG>={j + 1})"] = {"brier": float(np.mean((p - e) ** 2)), "ece_10_equal_count_bins": float(ece), "mean_pred": float(p.mean()), "base_rate": float(e.mean())}
    return out


def summarize(kind, params, y, model_id, game_ids, player_ids, mean_pred, extra=None):
    m = row_metrics(kind, params, y, model_id, game_ids, player_ids)
    crps_g, uniq, crps_per_game = game_macro(m["crps"], game_ids)
    nll_g, _, nll_per_game = game_macro(m["nll"], game_ids)
    cov = {f"{int(c * 100)}": coverage(m["lo"][c], m["hi"][c], y) for c in (0.5, 0.8, 0.9)}
    res = {"model_id": model_id, "n_rows": int(len(y)), "n_games": int(len(uniq)), "crps_macro_game": crps_g, "crps_row_mean": float(m["crps"].mean()), "nll_macro_game": nll_g, "nll_row_mean": float(m["nll"].mean()), "nll_floor_hits": m["floor_hits"],
           "mean_pred": float(np.mean(mean_pred)), "mean_obs": float(np.mean(y)), "mae_mean_prediction": float(np.mean(np.abs(mean_pred - y))), "bias_mean_prediction": float(np.mean(mean_pred - y)),
           "coverage": cov, "coverage_error": {k: abs(v - int(k) / 100) for k, v in cov.items()}, "pit_ks": ks_uniform(m["pit"]), "threshold_diagnostics": threshold_diag(m["p_ge"], y), "support_ceiling_used_K": m["max_K"], "max_survival_at_K": m["max_sf"]}
    if extra:
        res.update(extra)
    return res, {"crps": m["crps"], "nll": m["nll"], "pit": m["pit"], "game_ids": np.asarray(game_ids), "crps_per_game": crps_per_game, "games": uniq, "nll_per_game": nll_per_game}


# ------------------------------------------------------------------ slices
def slice_crps(crps_rows, mask):
    return float(crps_rows[mask].mean()) if mask.sum() else None


def slice_gate(ch_crps, ref_crps, masks, min_rows=500, tol=0.05):
    """No eligible slice (>= min_rows) with CRPS more than `tol` worse than the reference. Returns (pass, detail)."""
    detail = {}
    ok = True
    for name, m in masks.items():
        n = int(m.sum())
        if n < min_rows:
            detail[name] = {"rows": n, "eligible": False}
            continue
        c, r = float(ch_crps[m].mean()), float(ref_crps[m].mean())
        rel = (c - r) / r
        detail[name] = {"rows": n, "eligible": True, "challenger": c, "reference": r, "relative_worse": rel, "pass": rel <= tol}
        ok = ok and rel <= tol
    return ok, detail


# ------------------------------------------------------------------ 2-calendar-week moving-block bootstrap
def blocked_bootstrap(game_delta, game_week, reps=BOOT_REPS, seed=BOOT_SEED):
    """game_delta: per-game (challenger - reference) mean CRPS; game_week: calendar-week index of each game (Monday-based). Moving blocks of 2 consecutive calendar weeks."""
    game_delta = np.asarray(game_delta, float); game_week = np.asarray(game_week)
    weeks = np.unique(game_week)
    W = len(weeks)
    pos = {int(w): i for i, w in enumerate(weeks)}
    S = np.zeros(W); N = np.zeros(W)
    for d, w in zip(game_delta, game_week):
        S[pos[int(w)]] += d; N[pos[int(w)]] += 1
    # valid block starts: weeks i, i+1 both present AND consecutive calendar weeks
    starts = np.array([i for i in range(W - 1) if weeks[i + 1] == weeks[i] + 1])
    if len(starts) == 0:
        raise ValueError("no valid 2-week blocks")
    nb = int(math.ceil(W / 2))
    rng = np.random.default_rng(seed)
    pick = starts[rng.integers(0, len(starts), size=(reps, nb))]
    idx = np.stack([pick, pick + 1], axis=2).reshape(reps, nb * 2)[:, :W]
    boot = S[idx].sum(axis=1) / N[idx].sum(axis=1)
    return boot


def bootstrap_report(game_delta, game_week, ref_mean, reps=BOOT_REPS, seed=BOOT_SEED):
    boot = blocked_bootstrap(game_delta, game_week, reps, seed)
    obs = float(np.mean(game_delta))
    return {"observed_mean_delta": obs, "relative_crps_improvement": float(-obs / ref_mean), "one_sided_95_upper_bound": float(np.percentile(boot, 95)), "two_sided_2_5": float(np.percentile(boot, 2.5)), "two_sided_97_5": float(np.percentile(boot, 97.5)),
            "replicates": int(reps), "seed": int(seed), "n_games": int(len(game_delta)), "n_calendar_weeks": int(len(np.unique(game_week)))}
