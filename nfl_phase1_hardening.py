"""
NFL_PHASE1_HARDENING  (Phase 1A hardening for Phase 1B; shadow research, DEVELOPMENT data only)

C. Opportunity interval calibration
   Count distributions are discrete, so a central [q10, q90] interval covers MORE than 80% even for a perfectly
   calibrated forecast. The right yardsticks are (1) the coverage the forecast itself implies, and (2) the
   randomized-PIT coverage P(0.1 <= u <= 0.9), which equals 0.80 exactly for a calibrated discrete forecast.
   Candidates (fit on the 2024 wk13-18 validation weeks, compared on development):
     R0  as-is (Dirichlet-multinomial, alpha tuned for CRPS)
     R1  alpha re-tuned with a calibration-aware objective
     R2  PIT recalibration (monotone map of the predictive CDF fit on validation PITs)
     R3  empirical residual distribution (residuals of the mean forecast by forecast-size bucket)
     R4  temperature (flatten/sharpen the pmf, exponent tuned on validation)
     R5  zero-mass recalibration (isotonic map of P(y=0), positive part rescaled)
   Selection: lowest CRPS among candidates whose PIT-coverage is within 0.03 of nominal at 80% and 50% and whose
   median MAE is not worse; otherwise the best-calibrated candidate.

D. Named-player mass deficit
   Decomposes actual and forecast shares of the team total (who carries the ball that named players do not) and
   tests outside-bucket policies on development data: fixed (Phase 1A), proportional to active named weight, and
   as-of rolling estimates of the bucket size.
"""
import numpy as np

import nfl_phase1_common as C
import nfl_phase1_opportunity as O

COMPOSITIONAL = ("carry", "target", "qb_att", "rz_carry", "rz_target")
PPKEY = {"carry": "car", "target": "tgt", "qb_att": "att", "rz_carry": "rz_car", "rz_target": "rz_tgt"}
WINDOWS = (64, 128, 256, 512, None)


# ------------------------------------------------------------------ discrete distribution tools
def pmf_from_samples(S, cap):
    n, m = S.shape
    Sc = np.minimum(S, cap).astype(np.int64)
    idx = (np.arange(n)[:, None] * (cap + 1) + Sc).ravel()
    return np.bincount(idx, minlength=n * (cap + 1)).reshape(n, cap + 1) / m


def cdf_of(H):
    F = np.cumsum(H, axis=1)
    F[:, -1] = 1.0
    return F


def quantile_idx(F, p):
    return np.argmax(F >= p - 1e-12, axis=1)


def randomized_pit(F, H, y, seed):
    rng = np.random.default_rng(seed)
    yi = np.minimum(y.astype(int), F.shape[1] - 1)
    hi = F[np.arange(len(y)), yi]
    lo = hi - H[np.arange(len(y)), yi]
    return lo + rng.random(len(y)) * (hi - lo)


def crps_pmf(F, y):
    grid = np.arange(F.shape[1])[None, :]
    ind = (grid >= y[:, None]).astype(float)
    return np.sum((F - ind) ** 2, axis=1)


def summarize_pmf(H, y, seed=11):
    F = cdf_of(H)
    n = len(y)
    q10, q50, q90, q25, q75 = (quantile_idx(F, p) for p in (0.10, 0.50, 0.90, 0.25, 0.75))
    ar = np.arange(n)
    pred_cov80 = F[ar, q90] - np.where(q10 > 0, F[ar, np.maximum(q10 - 1, 0)], 0.0)
    pred_cov50 = F[ar, q75] - np.where(q25 > 0, F[ar, np.maximum(q25 - 1, 0)], 0.0)
    grid = np.arange(H.shape[1])[None, :]
    mean = (H * grid).sum(1)
    u = randomized_pit(F, H, y, seed)
    deciles = [float(np.mean(u <= t)) for t in np.linspace(0.1, 0.9, 9)]
    cr = crps_pmf(F, y)
    return {"crps": round(float(cr.mean()), 4), "mae_median": round(float(np.abs(q50 - y).mean()), 4),
            "rmse_mean": round(float(np.sqrt(((mean - y) ** 2).mean())), 4), "bias_mean": round(float((mean - y).mean()), 4),
            "cov80_empirical_interval": round(float(((y >= q10) & (y <= q90)).mean()), 4),
            "cov80_implied_by_forecast": round(float(pred_cov80.mean()), 4),
            "cov50_empirical_interval": round(float(((y >= q25) & (y <= q75)).mean()), 4),
            "cov50_implied_by_forecast": round(float(pred_cov50.mean()), 4),
            "pit_cov80": round(float(np.mean((u >= 0.1) & (u <= 0.9))), 4), "pit_cov50": round(float(np.mean((u >= 0.25) & (u <= 0.75))), 4),
            "pit_ks": round(float(np.max(np.abs(np.array(deciles) - np.linspace(0.1, 0.9, 9)))), 4),
            "mean_width80": round(float((q90 - q10).mean()), 3), "zero_rate_forecast": round(float(H[:, 0].mean()), 4),
            "zero_rate_actual": round(float((y == 0).mean()), 4)}, cr, np.abs(q50 - y)


# ------------------------------------------------------------------ recalibration candidates
def fit_pit_map(u_val):
    xs = np.linspace(0, 1, 101)
    G = np.array([np.mean(u_val <= x) for x in xs])
    G = np.maximum.accumulate(G)
    G[0], G[-1] = 0.0, 1.0
    return xs, G


def apply_pit_map(H, xs, G):
    F = cdf_of(H)
    F2 = np.interp(F, xs, G)
    F2[:, -1] = 1.0
    F2 = np.maximum.accumulate(F2, axis=1)
    H2 = np.diff(np.concatenate([np.zeros((len(F2), 1)), F2], axis=1), axis=1)
    return np.maximum(H2, 0.0)


def apply_temperature(H, tau):
    P = np.where(H > 0, H, 0.0) ** (1.0 / tau)
    return P / P.sum(1, keepdims=True)


def fit_temperature(H, y):
    best = None
    for tau in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0):
        cr = crps_pmf(cdf_of(apply_temperature(H, tau)), y).mean()
        best = (cr, tau) if best is None or cr < best[0] else best
    return best[1]


def bucket_edges(mean):
    return np.unique(np.quantile(mean, [0.2, 0.4, 0.6, 0.8]))


def empirical_residual_pmf(Hv, yv, H):
    """Predictive pmf = forecast mean + validation residuals of forecasts in the same size bucket (floored at 0)."""
    grid = np.arange(Hv.shape[1])
    mv = (Hv * grid).sum(1); m = (H * grid).sum(1)
    edges = bucket_edges(mv)
    bv = np.searchsorted(edges, mv); b = np.searchsorted(edges, m)
    out = np.zeros_like(H)
    cap = H.shape[1] - 1
    for k in np.unique(b):
        res = yv[bv == k] - mv[bv == k]
        if len(res) < 20:
            res = yv - mv
        rows = np.where(b == k)[0]
        vals = np.clip(np.round(m[rows][:, None] + res[None, :]), 0, cap).astype(int)
        for i, r in enumerate(rows):
            out[r] = np.bincount(vals[i], minlength=cap + 1)[:cap + 1] / len(res)
    return out


def apply_zero_map(H, p0_val, z_val):
    """Isotonic-style (binned) map from forecast P(y=0) to observed zero frequency, fit on validation."""
    edges = np.unique(np.quantile(p0_val, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return H
    b = np.clip(np.searchsorted(edges, p0_val, side="right") - 1, 0, len(edges) - 2)
    xs = np.array([p0_val[b == k].mean() for k in range(len(edges) - 1) if (b == k).any()])
    ys = np.array([z_val[b == k].mean() for k in range(len(edges) - 1) if (b == k).any()])
    ys = np.maximum.accumulate(ys)
    p0 = H[:, 0]
    p0n = np.clip(np.interp(p0, xs, ys), 1e-6, 1 - 1e-6)
    H2 = H.copy()
    pos = 1.0 - p0
    H2[:, 1:] = H[:, 1:] * (1 - p0n)[:, None] / np.where(pos > 0, pos, 1.0)[:, None]
    H2[:, 0] = p0n
    H2[pos <= 0, 1:] = 0.0
    return H2 / H2.sum(1, keepdims=True)


def calibration_study(name, rep_type, frames_valid, frames_dev, cache_dev, other, k0, comp, omode):
    """All candidate calibrations for one opportunity type; returns a report dict (development)."""
    cfg = rep_type["selected_config"]; alpha = rep_type["dirichlet_alpha"]
    yv, Sv, mv = O.run_cfg(frames_valid, cfg, other, alpha, k0, comp, tag="val_cal", other_mode=omode)
    yd, Sd, md = cache_dev["selected"]
    cap = int(min(max(Sv.max(), Sd.max(), yv.max(), yd.max()) + 1, 140))
    Hv, Hd = pmf_from_samples(Sv, cap), pmf_from_samples(Sd, cap)
    blocks = np.array([f"{a}-{b}" for a, b, _, _, _ in md])
    cands = {"R0_as_is": Hd}
    # R1: alpha re-tuned by a calibration-aware objective on validation (CRPS + PIT distance), then re-simulated on dev
    if comp and alpha is not None:
        best = None
        for a in O.ALPHA_GRID + (160.0, 320.0):
            y1, S1, _ = O.run_cfg(frames_valid, cfg, other, a, k0, comp, tag="val_alpha", other_mode=omode)
            H1 = pmf_from_samples(S1, cap)
            m1, cr1, _ = summarize_pmf(H1, y1)
            obj = m1["crps"] * (1 + 2.0 * max(m1["pit_ks"], abs(m1["pit_cov80"] - 0.8), abs(m1["pit_cov50"] - 0.5)))
            best = (obj, a) if best is None or obj < best[0] else best
        a1 = best[1]
        yy, S1d, _ = O.run_cfg(frames_dev, cfg, other, a1, k0, comp, tag="eval", other_mode=omode)
        cands["R1_alpha_calibration_aware"] = pmf_from_samples(S1d, cap)
    # R2 PIT map
    uv = randomized_pit(cdf_of(Hv), Hv, yv, 5)
    xs, G = fit_pit_map(uv)
    cands["R2_pit_recalibration"] = apply_pit_map(Hd, xs, G)
    # R3 empirical residuals
    cands["R3_empirical_residual"] = empirical_residual_pmf(Hv, yv, Hd)
    # R4 temperature
    tau = fit_temperature(Hv, yv)
    cands["R4_temperature"] = apply_temperature(Hd, tau)
    # R5 zero mass
    cands["R5_zero_mass"] = apply_zero_map(Hd, Hv[:, 0], (yv == 0).astype(float))
    res = {"n_valid": int(len(yv)), "n_dev": int(len(yd)), "pit_map_knots": 101, "temperature_tau": tau, "candidates": {}}
    base_cr = base_ae = None
    for nme, H in cands.items():
        m, cr, ae = summarize_pmf(H, yd)
        for tag, sel in (("2025", lambda t: t == 2025), ("2026_wk1_3", lambda t: t == 2026)):
            ix = np.array([sel(a) for a, _, _, _, _ in md])
            if ix.sum() > 30:
                mm, _, _ = summarize_pmf(H[ix], yd[ix])
                m[tag] = {k: mm[k] for k in ("crps", "pit_cov80", "pit_cov50", "cov80_empirical_interval", "cov80_implied_by_forecast")}
        if nme == "R0_as_is":
            base_cr, base_ae = cr, ae
        else:
            imp, p = C.block_boot(ae, base_ae, blocks)
            impc, pc = C.block_boot(cr, base_cr, blocks)
            m["vs_R0"] = {"mae_improvement": imp, "p_mae": p, "crps_improvement": impc, "p_crps": pc}
        res["candidates"][nme] = m
    ok = [n for n, m in res["candidates"].items() if abs(m["pit_cov80"] - 0.8) <= 0.03 and abs(m["pit_cov50"] - 0.5) <= 0.03
          and (n == "R0_as_is" or m["mae_median"] <= res["candidates"]["R0_as_is"]["mae_median"] + 1e-9 or m["vs_R0"]["p_mae"] > 0.5)]
    pool = ok or [min(res["candidates"], key=lambda n: max(abs(res["candidates"][n]["pit_cov80"] - 0.8), abs(res["candidates"][n]["pit_cov50"] - 0.5)))]
    res["selected"] = min(pool, key=lambda n: res["candidates"][n]["crps"])
    res["selected_reason"] = ("lowest CRPS among candidates with randomized-PIT coverage within 0.03 of nominal at 80% and 50% and median MAE not worse"
                              if ok else "no candidate met the calibration band; best-calibrated candidate chosen")
    res["_pmf"] = {"dev": cands[res["selected"]], "valid": Hv, "cap": cap, "y_dev": yd, "meta_dev": md}
    return res


# ------------------------------------------------------------------ named-player mass decomposition
def expected_named_share(f, cfg, other, omode, pact_mode, N=120):
    """Mean over sampled worlds of (named counts / team total), for counterfactual availability settings."""
    rng = np.random.default_rng(O.stable_seed(f["key"], ("mass", pact_mode)))
    n = len(f["y"])
    prop = f["P1"]
    if pact_mode == "all_active":
        pact = np.ones(n)
    elif pact_mode == "oracle_availability":
        pact = np.array([1.0 if r["ref"]["actual"]["played"] else 0.0 for r in f["rows"]])
    else:
        pact = f["pact_sel"]
    mu, kk = (f["mu_sel"], f["k_sel"]) if cfg["team"] else (f["mu_b0"], f["k_b0"])
    counts, T, _, _ = O.sample_alloc(pact, prop, f.get("other_override", other), mu, kk, None, N, rng, omode)
    ok = T > 0
    return float(np.mean(counts.sum(1)[ok] / T[ok])) if ok.any() else np.nan


def mass_decomposition(D, name, frames_dev_all, rep_type, other, omode):
    """Decompose the named-player share for one allocation type (carry): who the outside players are, and how much of the
    forecast deficit is bucket size, double counting, availability shrinkage, or a stale (train-period) outside share."""
    cfg = rep_type["selected_config"]
    by_season = {}
    cats_total = {}
    reasons = {}
    prior_n = _prior_game_counts(D)
    rows_out = []
    F_model, F_all, F_oracle, actual_named, T_all = [], [], [], [], []
    role_bias = []
    for f in frames_dev_all:
        s, w, team = f["key"]
        T = f["T_act"]
        if not T:
            continue
        cand_ids = {r["id"] for r in f["rows"]}
        named = float(np.nansum(f["y"]))
        cats = {"named_candidates": named}
        for gid, st in D.stat.get((s, w, team), {}).items():
            if gid in cand_ids:
                continue
            c = D.pp.get((s, w, gid), {}).get(PPKEY[name], 0.0)
            if not c:
                continue
            pos = st["pos"]
            if name in ("carry", "rz_carry"):
                grp = "RB/FB not in candidate set" if pos in ("RB", "FB") else "QB not in candidate set" if pos == "QB" else "WR/TE/other position (outside by design)"
            elif name == "qb_att":
                grp = "QB not in candidate set" if pos == "QB" else "non-QB passer (outside by design)"
            else:
                grp = "RB/WR/TE not in candidate set" if pos in ("RB", "FB", "WR", "TE") else "QB/other position (outside by design)"
            if "not in candidate set" in grp:
                n0 = prior_n.get((s, w), {}).get(gid, 0)
                why = "no prior game (debut)" if n0 == 0 else "<3 prior games (eligibility filter)" if n0 < 3 else ">=3 prior games but not on recent team roster/history window"
                reasons[why] = reasons.get(why, 0.0) + c
            cats[grp] = cats.get(grp, 0.0) + c
        for k, v in cats.items():
            cats_total[k] = cats_total.get(k, 0.0) + v
            by_season.setdefault(s, {})
            by_season[s][k] = by_season[s].get(k, 0.0) + v
        by_season[s]["_T"] = by_season[s].get("_T", 0.0) + T
        T_all.append(T); actual_named.append(named / T)
        F_model.append(expected_named_share(f, cfg, other, omode, "model"))
        F_all.append(expected_named_share(f, cfg, other, omode, "all_active"))
        F_oracle.append(expected_named_share(f, cfg, other, omode, "oracle_availability"))
        played = [(r["P1"], y) for r, y in zip(f["rows"], f["y"]) if r["ref"]["actual"]["played"] and not np.isnan(y)]
        if played:
            role_bias.append(sum(p for p, _ in played) - sum(y for _, y in played) / T)
    tot = sum(T_all)
    out = {"n_team_games": len(T_all), "type": name,
           "actual_share_of_team_total": {k: round(v / tot, 4) for k, v in cats_total.items()},
           "actual_share_by_season": {int(s): {k: round(v / d["_T"], 4) for k, v in d.items() if k != "_T"} for s, d in by_season.items()},
           "why_actual_outside_RB_QB_were_not_candidates(share of team total)": {k: round(v / tot, 4) for k, v in reasons.items()},
           "forecast_named_share": {"model_sampled_availability": round(float(np.nanmean(F_model)), 4),
                                    "all_candidates_active": round(float(np.nanmean(F_all)), 4),
                                    "oracle_availability": round(float(np.nanmean(F_oracle)), 4)},
           "actual_named_share_mean_of_ratios": round(float(np.mean(actual_named)), 4),
           "role_model_bias_sum_of_propensities_minus_actual_share_for_players_who_played": round(float(np.mean(role_bias)), 4),
           "bucket_weight_used": round(float(other), 4), "bucket_mode": omode}
    m, a, o_ = out["forecast_named_share"]["model_sampled_availability"], out["forecast_named_share"]["all_candidates_active"], out["forecast_named_share"]["oracle_availability"]
    A = out["actual_named_share_mean_of_ratios"]
    out["deficit_decomposition_actual_minus_forecast"] = {
        "total_deficit": round(A - m, 4),
        "availability_shrinkage(all_active_minus_model)": round(a - m, 4),
        "oracle_vs_sampled_availability(oracle_minus_model)": round(o_ - m, 4),
        "remaining_after_oracle_availability(actual_minus_oracle)": round(A - o_, 4)}
    return out


def _prior_game_counts(D):
    """(s, w) -> {gid: number of earlier weeks in which the player has a stats row}."""
    weeks = sorted({(k[0], k[1]) for k in D.stat})
    cum, out = {}, {}
    for sw in weeks:
        out[sw] = dict(cum)
        for team_key in [k for k in D.stat if (k[0], k[1]) == sw]:
            for gid in D.stat[team_key]:
                cum[gid] = cum.get(gid, 0) + 1
    return out


# ------------------------------------------------------------------ outside-bucket policy variants
def run_variant(frames, name, rep_type, mode, window, akey, k0, comp=True):
    """Re-derive bucket weight (as-of), re-tune alpha on validation, run the selected config on development."""
    other0 = O.other_weight(frames, akey)
    roll = O.rolling_outside(frames, window)
    for f in frames:
        o_ = roll[f["key"]]
        f["other_override"] = (o_ / max(1.0 - o_, 1e-6)) if mode == "prop" else o_
    other = (other0 / max(1.0 - other0, 1e-6)) if mode == "prop" else other0
    fv = [f for f in frames if C.VALID(f["s"], f["w"])]
    fd = [f for f in frames if C.DEV(f["s"], f["w"])]
    alpha, _ = O.tune_alpha(fv, other, comp, k0, mode)
    cfg = rep_type["selected_config"]
    yv, Sv, _ = O.run_cfg(fv, cfg, other, alpha, k0, comp, tag="pol_val", other_mode=mode)
    yd, Sd, md = O.run_cfg(fd, cfg, other, alpha, k0, comp, tag="eval", other_mode=mode)
    return {"other_weight": other, "alpha": alpha, "valid": (yv, Sv), "dev": (yd, Sd, md), "frames_dev": fd, "cfg": cfg}


def policy_metrics(v, ref=None):
    yd, Sd, md = v["dev"]
    m, cr, ae = O.metrics_from(yd, Sd, md)
    tot, act = {}, {}
    for (s_, w_, t_, _, _), yy, mu in zip(md, yd, Sd.mean(1)):
        tot[(s_, w_, t_)] = tot.get((s_, w_, t_), 0.0) + mu; act[(s_, w_, t_)] = act.get((s_, w_, t_), 0.0) + yy
    Tm = {f["key"]: (f["mu_sel"] if v["cfg"]["team"] else f["mu_b0"]) for f in v["frames_dev"]}
    fc = float(np.mean([tot[k] / Tm[k] for k in tot if Tm.get(k)])); ac = float(np.mean([act[k] / Tm[k] for k in act if Tm.get(k)]))
    out = {"mae": m["mae"], "crps": m["crps"], "mean_rmse": m["mean_forecast"]["rmse"], "mean_bias": m["mean_forecast"]["bias"],
           "named_share_forecast": round(fc, 4), "named_share_actual": round(ac, 4), "named_share_error": round(fc - ac, 4),
           "alpha": v["alpha"], "cov80": m["cov80"]}
    blocks = np.array([f"{a}-{b}" for a, b, _, _, _ in md])
    if ref is not None:
        _, cr0, ae0 = O.metrics_from(*ref["dev"])
        imp, p = C.block_boot(ae, ae0, blocks)
        impc, pc = C.block_boot(cr, cr0, blocks)
        out["vs_fixed_train_mean"] = {"mae_improvement": imp, "p_mae": p, "crps_improvement": impc, "p_crps": pc}
    return out


def outside_policy_study(frames, name, rep_type, akey, k0):
    """Compare bucket policies on development data; returns (report, selected policy dict)."""
    variants = {"V0_fixed_train_mean(Phase1A)": ("fixed", None), "V1_proportional_train_mean": ("prop", None)}
    for w in WINDOWS[:-1]:
        variants[f"V2_fixed_rolling_{w}"] = ("fixed", w)
        variants[f"V3_proportional_rolling_{w}"] = ("prop", w)
    res, runs = {}, {}
    for nm, (mode, window) in variants.items():
        runs[nm] = run_variant(frames, name, rep_type, mode, window, akey, k0)
    ref = runs["V0_fixed_train_mean(Phase1A)"]
    for nm, v in runs.items():
        # validation-period named-share error decides the window; development is only used to compare the finalists
        yv, Sv = v["valid"]
        res[nm] = policy_metrics(v, None if nm.startswith("V0") else ref)
    return res, runs
