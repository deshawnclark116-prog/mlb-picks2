"""
NFL_PHASE1_OPPORTUNITY  (Phase 1A, shadow research)

Team-coherent opportunity allocation. Teammates compete for a finite team
total; nothing is forecast independently and allowed to total 137%.

  team total ~ NegBin(mu, k)                       (team environment layer)
  active_i   ~ Bernoulli(P(active_i))              (availability layer; T-24h)
  w_i        = active_i * propensity_i             (role/share layer, conditional-on-active share)
  w_other    = mean share of players outside the candidate set (train mean)
  shares     ~ Dirichlet(alpha * w / sum w)  (alpha = inf: deterministic)
  counts     ~ Multinomial(total, shares)          -> player counts, sum = total exactly

Inactive players receive zero in their inactive branch, and their weight
disappears from the normalisation, so their opportunity is redistributed to the
active teammates (proportionally to propensity) instead of vanishing.

Compositional allocations: carry (RB+QB), target (RB/WR/TE), qb_att (QB),
rz_carry, rz_target. Non-compositional (a play has ~4-5 receivers on the field,
~11 defenders): route (dropbacks x route rate) and def_snap (team snaps x snap
share); for these the module reports reconciliation, not a sum-to-one constraint.

Ablation chain (each step changes exactly one thing; all scored on the pregame
universe, non-participants = 0):
  A0 dumb baseline   NB(team blend x last-8 share x status-lookup P(active))
  A1 + coherence     multinomial allocation, last-8 propensities, blend total, lookup P(active)
  A2 + role model    learned mean-share propensity replaces last-8
  A3 + availability  selected P(active) (T-24h) replaces the lookup
  A4 + team env      selected team-total model replaces the blend
  A5 + dispersion    Dirichlet concentration alpha (tuned on validation)
"""
import math
from collections import defaultdict

import numpy as np

import nfl_phase1_common as C
import nfl_phase1_role_state as R
import nfl_phase1_team_environment as TE

ALLOC = {
    # name: (role type, team total key, actual key on player record, positions, compositional)
    "carry": ("carry", "rushes", "car", ("RB", "QB"), True),
    "target": ("target", "targets", "tgt", ("RB", "WR", "TE"), True),
    "qb_att": ("qb_att", "dropbacks", "att", ("QB",), True),
    "rz_carry": ("rz_carry", "rz_rushes", "rz_car", ("RB", "QB"), True),
    "rz_target": ("rz_target", "rz_targets", "rz_tgt", ("RB", "WR", "TE"), True),
    "route": ("route", "dropbacks", "routes", ("RB", "WR", "TE"), False),
    "def_snap": ("def_snap", "def_snaps", "snaps", ("DL", "LB", "DB"), False),
}
NSAMP = 200
ALPHA_GRID = (5.0, 10.0, 20.0, 40.0, 80.0)
POS_FLAGS = ("pos_QB", "pos_RB", "pos_WR", "pos_TE", "grp_DL", "grp_LB", "grp_DB")


def stable_seed(key, tag):
    import zlib
    return zlib.crc32(repr((key, tag)).encode())


# ------------------------------------------------------------------ sampling core
def sample_alloc(pact, prop, other, mu_T, k_T, alpha, N, rng):
    """Return counts [N, n] for the named candidates and totals [N].
    pact, prop: arrays [n]; other: scalar weight of the outside-candidate bucket."""
    n = len(prop)
    p_n, r_n = k_T, k_T / (k_T + max(mu_T, 1e-6))
    T = rng.negative_binomial(p_n, r_n, size=N)
    active = rng.random((N, n)) < pact[None, :]
    w = active * prop[None, :]
    tot = w.sum(1) + other
    tot = np.where(tot <= 0, 1.0, tot)
    full = np.column_stack([w, np.full(N, other)]) / tot[:, None]
    zero = (w.sum(1) + other) <= 0
    full[zero, :] = 0.0; full[zero, -1] = 1.0
    if alpha is not None and np.isfinite(alpha):
        g = rng.gamma(np.maximum(alpha * full, 1e-9))
        g[full == 0] = 0.0
        s = g.sum(1, keepdims=True)
        full = np.where(s > 0, g / np.maximum(s, 1e-300), full)
    counts = np.empty((N, n + 1), dtype=np.int64)
    for i in range(N):
        counts[i] = rng.multinomial(int(T[i]), full[i])
    return counts[:, :n], T, active, full


# ------------------------------------------------------------------ propensity models
def add_pos(row):
    F = row["ref"].get("_F", {})
    lf = row["LF"]
    for k in POS_FLAGS:
        lf[k] = F.get(k)
    return lf


def fit_propensity(rows, tname, families=None):
    """Mean-objective share model (squared error), conditional on playing. `families` = role-state feature
    families that earned their place in the leave-one-family-out test for this share type."""
    for r in rows:
        r["LF"] = R.learned_features(r, tname)
        add_pos(r)
    fams = families or list(R.FAMILIES)
    cols = [c for f in fams for c in R.FAMILIES[f]] + [c for c in POS_FLAGS if any(r["LF"].get(c) is not None for r in rows[:500])]
    lab = [r for r in rows if r["y"] is not None]
    tr = [r for r in lab if C.TRAIN(r["s"], r["w"])]; va = [r for r in lab if C.VALID(r["s"], r["w"])]
    C.audit_fit(f"opportunity_propensity_{tname}", tr, va)
    m = lambda rr: C.matrix([r["LF"] for r in rr], cols)
    y = lambda rr: np.array([r["y"] for r in rr])
    b = C.fit_xgb(m(tr), y(tr), m(va), y(va), cols, "reg:squarederror")
    p = np.clip(C.xgb_pred(b, m(rows), cols), 0.0, 1.0)
    for r, x in zip(rows, p):
        r["P1"] = float(x)
    return {"families": fams, "cols": cols, "rounds": int(b.best_iteration + 1), "n_train": len(tr), "n_valid": len(va)}


# ------------------------------------------------------------------ per-unit frames
def frames_for(rows_by_type, units, te_pred, te_b0, pact_sel, pact_lookup):
    """For each allocation type: list of frames, one per team-game (unit), holding
    candidate arrays for every step of the ablation."""
    out = {}
    for name, (rtype, tkey, akey, poss, comp) in ALLOC.items():
        rows = rows_by_type[rtype]
        by_unit = defaultdict(list)
        for r in rows:
            by_unit[r["unit"]["key"]].append(r)
        frames = []
        for u in units:
            rs = by_unit.get(u["key"])
            if not rs:
                continue
            k = u["key"]
            a = [r["ref"]["actual"] for r in rs]
            if name == "def_snap":
                act = np.array([x["snaps"] for x in a])
                other = 0.0
            else:
                act = np.array([x[akey] if x[akey] is not None else np.nan for x in a], float)
            T_act = u["team_actual"][tkey if tkey != "def_snaps" else "def_snaps"]
            fr = {"key": k, "kick": u["kick"], "rows": rs, "y": act, "T_act": T_act, "s": k[0], "w": k[1], "team": k[2],
                  "P0": np.array([r["R0"] for r in rs]), "P1": np.array([r["P1"] for r in rs]),
                  "pact_sel": np.array([pact_sel(r["ref"]) for r in rs]), "pact_lookup": np.array([pact_lookup(r["ref"]) for r in rs]),
                  "mu_sel": te_pred[(k[0], k[1], k[2], tkey)][0], "k_sel": te_pred[(k[0], k[1], k[2], tkey)][1],
                  "mu_b0": te_b0[(k[0], k[1], k[2], tkey)][0], "k_b0": te_b0[(k[0], k[1], k[2], tkey)][1]}
            frames.append(fr)
        out[name] = frames
    return out


def other_weight(frames, tname_key):
    """Mean share of the team total NOT carried by named candidates, from TRAIN units only."""
    v = []
    for f in frames:
        if not C.TRAIN(f["s"], f["w"]) or not f["T_act"]:
            continue
        v.append(max(0.0, 1.0 - np.nansum(f["y"]) / f["T_act"]))
    return float(np.mean(v)) if v else 0.03


# ------------------------------------------------------------------ evaluation
def crps_rows(S, y):
    Ss = np.sort(S, axis=1); m = Ss.shape[1]
    t1 = np.mean(np.abs(Ss - y[:, None]), axis=1)
    i = np.arange(1, m + 1)
    t2 = np.sum((2 * i - m - 1)[None, :] * Ss, axis=1) / (m * m)
    return t1 - t2


CONFIGS = {   # cumulative ablation from the dumb baseline (each step changes one thing)
    "A0": None,
    "A1": {"coherent": True, "role": False, "avail": False, "team": False, "dispersion": False},
    "A2": {"coherent": True, "role": True, "avail": False, "team": False, "dispersion": False},
    "A3": {"coherent": True, "role": True, "avail": True, "team": False, "dispersion": False},
    "A4": {"coherent": True, "role": True, "avail": True, "team": True, "dispersion": False},
    "A5": {"coherent": True, "role": True, "avail": True, "team": True, "dispersion": True},
}
FULL = CONFIGS["A5"]
COMPONENTS = ("coherent", "role", "avail", "team", "dispersion")


def run_cfg(frames, cfg, other, alpha, k0, comp, N=NSAMP, tag="eval"):
    """cfg=None is the dumb baseline: NB(team blend x last-8 share x status-lookup P(active)) with a
    per-stat dispersion fit on train. Returns (y, S, meta) concatenated over frames."""
    ys, Ss, meta = [], [], []
    for f in frames:
        rng = np.random.default_rng(stable_seed(f["key"], ("crn", tag)))   # common random numbers: identical parts of two configs give identical samples
        y = f["y"]; ok = ~np.isnan(y)
        if not ok.any():
            continue
        n = len(y)
        c = cfg or {"coherent": False, "role": False, "avail": False, "team": False, "dispersion": False}
        prop = f["P1"] if c["role"] else f["P0"]
        pact = f["pact_sel"] if c["avail"] else f["pact_lookup"]
        mu, kk = (f["mu_sel"], f["k_sel"]) if c["team"] else (f["mu_b0"], f["k_b0"])
        if comp and c["coherent"]:
            counts, T, _, _ = sample_alloc(pact, prop, other, mu, kk, alpha if c["dispersion"] else None, N, rng)
            S = counts.T
        elif comp:
            mu_i = np.maximum(mu * prop * pact, 1e-6)
            S = rng.negative_binomial(k0, k0 / (k0 + mu_i)[:, None], size=(n, N))
        else:
            T = rng.negative_binomial(kk, kk / (kk + max(mu, 1e-6)), size=N)
            act = rng.random((n, N)) < pact[:, None]
            sd = f.get("share_sd", 0.12) if c["dispersion"] else 0.0
            sh = np.clip(prop[:, None] + sd * rng.standard_normal((n, N)), 0, 1)
            S = np.where(act, np.round(T[None, :] * sh), 0)
        ys.append(y[ok]); Ss.append(S[ok])
        meta.append([(f["s"], f["w"], f["team"], f["rows"][i]["id"], f["rows"][i]) for i in np.where(ok)[0]])
    return np.concatenate(ys), np.concatenate(Ss, axis=0), [m for mm in meta for m in mm]


def metrics_from(y, S, meta):
    sm = {"mean": S.mean(1), "median": np.median(S, axis=1)}
    q = {p: np.quantile(S, p, axis=1) for p in (0.1, 0.25, 0.75, 0.9)}
    cr = crps_rows(S, y)
    out = {**C.cont(sm["median"], y), "mean_forecast": C.cont(sm["mean"], y), "crps": round(float(cr.mean()), 4),
           "p10_p90_cov80": C.coverage(q[0.1], q[0.9], y), "p25_p75_cov50": C.coverage(q[0.25], q[0.75], y),
           "poisson_deviance_mean": C.poisson_dev(sm["mean"], y)}
    out["cov80"], out["cov50"] = out["p10_p90_cov80"], out["p25_p75_cov50"]
    return out, cr, np.abs(sm["median"] - y)


def fit_k0(frames, other):
    """Per-stat NB dispersion of the dumb baseline, method of moments on TRAIN + VALID frames only."""
    m, y = [], []
    for f in frames:
        if not (C.TRAIN(f["s"], f["w"]) or C.VALID(f["s"], f["w"])):
            continue
        ok = ~np.isnan(f["y"])
        m.append((f["mu_b0"] * f["P0"] * f["pact_lookup"])[ok]); y.append(f["y"][ok])
    m = np.maximum(np.concatenate(m), 1e-6); y = np.concatenate(y)
    excess = float(np.sum((y - m) ** 2 - m))
    return float(max(np.sum(m ** 2) / excess, 0.5)) if excess > 0 else 1e4


def tune_alpha(frames_valid, other, comp, k0):
    best = None
    for a in ALPHA_GRID:
        y, S, meta = run_cfg(frames_valid, FULL, other, a, k0, comp, N=120, tag="alpha")
        cr = crps_rows(S, y).mean()
        best = (cr, a) if best is None or cr < best[0] else best
    return best[1], best[0]


def tune_share_sd(frames_valid):
    res = [r["y"] - r["P1"] for f in frames_valid for r in f["rows"] if r["y"] is not None]
    return float(np.std(res)) if res else 0.12


def paired(cache_new, cache_ref):
    y, S, meta = cache_new
    yr, Sr, _ = cache_ref
    m, cr, ae = metrics_from(y, S, meta)
    _, cr0, ae0 = metrics_from(yr, Sr, None)
    blocks = np.array([f"{a}-{b}" for a, b, _, _, _ in meta])
    imp, p = C.block_boot(ae, ae0, blocks)
    impc, pc = C.block_boot(cr, cr0, blocks)
    return {"mae_improvement": imp, "p_mae_not_better": p, "crps_improvement": impc, "p_crps_not_better": pc}


def summarize(cache_item, ref=None):
    y, S, meta = cache_item
    m, cr, ae = metrics_from(y, S, meta)
    m["n"] = int(len(y))
    for tag, sel in (("2025", lambda t: t == 2025), ("2026_wk1_3", lambda t: t == 2026)):
        ix = np.array([sel(a) for a, _, _, _, _ in meta])
        if ix.sum():
            mm, _, _ = metrics_from(y[ix], S[ix], None)
            m[tag] = {k: mm[k] for k in ("mae", "rmse", "bias", "crps", "cov80")}
    if ref is not None:
        m["vs_A0"] = paired(cache_item, ref)
    return m


def evaluate(rows_by_type, units, te_pred, te_b0, pact_sel, pact_lookup, families_by_type=None):
    rep = {"label": C.DEV_LABEL, "types": {}, "propensity_models": {}}
    families_by_type = families_by_type or {}
    fitted_types = {}
    for name, (rtype, tkey, akey, poss, comp) in ALLOC.items():
        if rtype not in fitted_types:
            fitted_types[rtype] = fit_propensity(rows_by_type[rtype], rtype, families_by_type.get(rtype))
        rep["propensity_models"][name] = fitted_types[rtype]
    frames_all = frames_for(rows_by_type, units, te_pred, te_b0, pact_sel, pact_lookup)
    rep["_cache"], rep["_frames"] = {}, {}
    for name, (rtype, tkey, akey, poss, comp) in ALLOC.items():
        frames = frames_all[name]
        if name == "route":
            frames = [f for f in frames if f["s"] <= 2025]
        if not frames:
            continue
        fv = [f for f in frames if C.VALID(f["s"], f["w"])]
        other = other_weight(frames, akey) if comp else 0.0
        sd = tune_share_sd(fv)
        for f in frames:
            f["share_sd"] = sd
        k0 = fit_k0(frames, other)
        alpha, acrps = tune_alpha(fv, other, comp, k0) if comp else (None, None)
        fd = [f for f in frames if C.DEV(f["s"], f["w"])]
        res = {"n_dev_team_games": len(fd), "other_weight": round(other, 4), "dirichlet_alpha": alpha, "baseline_nb_k": round(k0, 3),
               "share_noise_sd": round(sd, 4), "alpha_valid_crps": None if acrps is None else round(float(acrps), 4), "compositional": comp}
        cache = {"A0": run_cfg(fd, None, other, alpha, k0, comp)}
        res["steps"] = {"A0": summarize(cache["A0"])}
        for step, cfg in CONFIGS.items():
            if step == "A0" or (not comp and step == "A1"):
                continue
            cache[step] = run_cfg(fd, cfg, other, alpha, k0, comp)
            res["steps"][step] = summarize(cache[step], cache["A0"])
        # leave-one-component-out against the full chain (does each component earn its place given the others?)
        res["leave_one_out"] = {}
        comps = [c for c in COMPONENTS if comp or c != "coherent"]
        for c_ in comps:
            cfg = dict(FULL); cfg[c_] = False
            item = run_cfg(fd, cfg, other, alpha, k0, comp)
            m = summarize(item)
            # improvement of FULL over the variant without the component (positive = component helps)
            m["full_vs_without"] = paired(cache["A5"], item)
            res["leave_one_out"][c_] = m
        earns = {c_: (v["full_vs_without"]["crps_improvement"] > 0 and v["full_vs_without"]["p_crps_not_better"] < 0.10)
                 for c_, v in res["leave_one_out"].items()}
        cfg_sel = {c_: (earns.get(c_, False) if c_ in earns else False) for c_ in COMPONENTS}
        res["components_earning_place"] = earns
        res["selected_config"] = cfg_sel
        cache["selected"] = run_cfg(fd, cfg_sel, other, alpha, k0, comp) if cfg_sel != FULL else cache["A5"]
        res["selected_metrics"] = summarize(cache["selected"], cache["A0"])
        res["selected_vs_full"] = paired(cache["selected"], cache["A5"]) if cfg_sel != FULL else None
        y_, S_, meta_ = cache["selected"]
        tot, act = defaultdict(float), defaultdict(float)
        for (s0, w0, t0, _, _), yy, mu in zip(meta_, y_, S_.mean(1)):
            tot[(s0, w0, t0)] += mu; act[(s0, w0, t0)] += yy
        tm = {f["key"]: (f["mu_sel"] if cfg_sel["team"] else f["mu_b0"]) for f in fd}
        res["reconciliation"] = {
            "per_sample_identity": "named counts + outside-candidate bucket = team total exactly in every simulated sample (asserted in tests)",
            "forecast_named_share_of_team_total": round(float(np.mean([tot[k] / tm[k] for k in tot if tm.get(k)])), 4),
            "actual_named_share_of_team_total": round(float(np.mean([act[k] / tm[k] for k in act if tm.get(k)])), 4),
            "train_mean_named_share": round(1 - other, 4) if comp else None}
        res["regimes"] = regime_table(cache, "selected", unit_regimes(fd))
        rep["types"][name] = res
        rep["_cache"][name] = cache
        rep["_frames"][name] = fd
    return rep, frames_all


# ------------------------------------------------------------------ regimes
def unit_regimes(frames):
    """Extra unit-level tags: a top-propensity teammate is Out/Doubtful on the T-24h report."""
    tags = {}
    for f in frames:
        rs = f["rows"]
        if not rs:
            continue
        top = int(np.argmax(f["P0"]))
        st = (rs[top]["ref"].get("inj") or ("",))[0]
        if st in ("Out", "Doubtful") and f["P0"][top] >= 0.15:
            for i, r in enumerate(rs):
                if i != top:
                    tags[(f["key"], r["id"])] = "starter_out(top-share teammate Out/Doubtful)"
    return tags


def regime_table(cache, best, extra=None):
    out = {}
    extra = extra or {}
    for step in ("A0", best):
        y, S, meta = cache[step]
        ae = np.abs(np.median(S, axis=1) - y); cr = crps_rows(S, y)
        by = defaultdict(list)
        for i, (s_, w_, t_, pid, row) in enumerate(meta):
            tags = list(R.regimes(row))
            x = extra.get(((s_, w_, t_), pid))
            if x:
                tags.append(x)
            for g in tags:
                by[g].append(i)
        for g, ix in by.items():
            if len(ix) >= 25:
                out.setdefault(g, {"n": len(ix)})[step] = {"mae": round(float(ae[ix].mean()), 4), "crps": round(float(cr[ix].mean()), 4)}
    return {g: v for g, v in sorted(out.items()) if "A0" in v and best in v}


# ------------------------------------------------------------------ routes -> TPRR -> targets vs target allocation
def tprr_of(ref):
    h = [g for g in ref["hist"] if g.get("routes") is not None][-16:]
    routes = sum(g["routes"] for g in h); tgt = sum(g["tgt"] for g in h)
    return (tgt + 0.17 * 60.0) / (routes + 60.0)


def structural_vs_fallback(rep, frames_all):
    """Compare (structural) dropbacks -> route rate -> TPRR -> targets with (fallback) team targets -> allocation,
    on targets, over candidates present in both. Development data only."""
    out = {"label": C.DEV_LABEL, "definition": "structural: routes ~ route model (selected config), targets = round(routes x TPRR); "
                                               "fallback: target allocation (selected config). Both forecast before kickoff."}
    route_cache = rep["_cache"]["route"]["selected"]; tgt_cache = rep["_cache"]["target"]["selected"]
    ry, rS, rmeta = route_cache
    ty, tS, tmeta = tgt_cache
    ridx = {(m[0], m[1], m[2], m[3]): i for i, m in enumerate(rmeta)}
    pairs = [(i, ridx[(m[0], m[1], m[2], m[3])]) for i, m in enumerate(tmeta) if (m[0], m[1], m[2], m[3]) in ridx]
    if not pairs:
        return out
    ti = np.array([p[0] for p in pairs]); ri = np.array([p[1] for p in pairs])
    tp = np.array([tprr_of(tmeta[i][4]["ref"]) for i in ti])
    rng = np.random.default_rng(20260929)
    struct = np.round(rS[ri] * tp[:, None])
    y = ty[ti]; fall = tS[ti]
    blocks = np.array([f"{tmeta[i][0]}-{tmeta[i][1]}" for i in ti])
    res = {}
    for tag, S in (("structural", struct), ("fallback_allocation", fall)):
        m, cr, ae = metrics_from(y, S, None)
        res[tag] = {**m, "n": int(len(y))}
        res[tag]["_cr"], res[tag]["_ae"] = cr, ae
    imp, p = C.block_boot(res["structural"]["_ae"], res["fallback_allocation"]["_ae"], blocks)
    impc, pc = C.block_boot(res["structural"]["_cr"], res["fallback_allocation"]["_cr"], blocks)
    for t in res:
        res[t].pop("_cr"); res[t].pop("_ae")
    out.update({"n": int(len(y)), "seasons": "2025 only (2026 target-game routes unavailable; route history stale)", **res,
                "structural_vs_fallback": {"mae_improvement": imp, "p_not_better": p, "crps_improvement": impc, "p_crps_not_better": pc},
                "official_forward_path": "structural" if imp > 0 and impc > 0 and p < 0.10 and pc < 0.10 else "fallback_allocation"})
    return out


# ------------------------------------------------------------------ protocol-eligible universe check
ELIGIBLE = {   # protocol v1.1 prior-usage eligibility (mean over the last 3 game rows)
    "carry": ("carry", ("RB",), "car", 5.0),
    "target": ("target", ("WR", "TE", "RB"), "tgt", 3.0),
    "qb_att": ("qb_att", ("QB",), "att", 15.0),
}


def eligible_universe_check(rep):
    """Selected vs dumb baseline restricted to players meeting the protocol's prior-usage eligibility (removes the
    easy zeros of bench / practice-squad candidates that the broader development universe contains)."""
    out = {}
    for name, (_, poss, key, thr) in ELIGIBLE.items():
        if name not in rep["_cache"]:
            continue
        c = rep["_cache"][name]
        y0, S0, m0 = c["A0"]; y1, S1, m1 = c["selected"]
        keep = np.array([(row["grp"] in poss) and len(row["ref"]["hist"]) >= 3 and
                         (sum(g[key] for g in row["ref"]["hist"][-3:]) / 3.0) >= thr for (_, _, _, _, row) in m0])
        if keep.sum() < 50:
            continue
        a0, cr0, ae0 = metrics_from(y0[keep], S0[keep], None)
        a1, cr1, ae1 = metrics_from(y1[keep], S1[keep], None)
        blocks = np.array([f"{a}-{b}" for (a, b, _, _, _) in np.array(m0, dtype=object)[keep]])
        imp, p = C.block_boot(ae1, ae0, blocks)
        impc, pc = C.block_boot(cr1, cr0, blocks)
        share_zero = float((y0[keep] == 0).mean())
        out[name] = {"n": int(keep.sum()), "share_actual_zero": round(share_zero, 4), "baseline": {k: a0[k] for k in ("mae", "crps", "cov80")},
                     "selected": {k: a1[k] for k in ("mae", "crps", "cov80")},
                     "vs_baseline": {"mae_improvement": imp, "p": p, "crps_improvement": impc, "p_crps": pc}}
    return out
