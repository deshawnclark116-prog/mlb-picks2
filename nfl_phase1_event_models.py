"""
NFL_PHASE1_EVENT_MODELS  (Phase 1B, shadow research)

Generic component runners built on nfl_phase1_efficiency:
  run_pmf     categorical outcome per opportunity (air yards, YAC, ...): hierarchical pmf + tilt, nested B0..B5
  run_binary  hazard per opportunity (catch, sack, INT, TD ...): hierarchical rate + offset logistic, nested B0..B5
  Stacked     several strata (e.g. 4 air-yard buckets) stacked as separate rows so one model serves all strata
Every component reports development scores for 2025 / 2026 wk1-3 / combined, and week-block bootstrap deltas vs B0 (hierarchical).
"""
import numpy as np

import nfl_phase1_efficiency as F


class Stacked:
    """Rows of `R` repeated once per stratum; aux one-hot columns are appended to the 'team' family (used from B1 up)."""

    def __init__(self, R, n_strata):
        self.n = R.n * n_strata
        rep = lambda a: np.concatenate([a] * n_strata)
        self.s, self.w = rep(R.s), rep(R.w)
        self.key = R.key * n_strata
        self.pos = rep(R.pos)
        self.active = rep(R.active)
        eye = np.repeat(np.eye(n_strata)[:, 1:], R.n, axis=0)
        self.fam = {f: rep(R.fam[f]) for f in R.fam}
        self.fam["team"] = np.column_stack([self.fam["team"], eye])
        self.base = {}
        self.cnt = {}


def _logit_tendency(P):
    p = np.clip(P[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def inter_binary(P0, R, opp_i, scheme_i):
    tr = F.masks(R)[0]
    t = _logit_tendency(P0); t = t - t[tr].mean()
    def col(name, i):
        x = R.fam[name][:, i].astype(float); m = np.nanmean(x[tr]); return np.where(np.isnan(x), m, x)
    return np.column_stack([t * col("opp", opp_i)] + [t * col("scheme", j) for j in scheme_i])


def run_binary(R, cnt2, base, opp_i, scheme_i, name, tune_kappas=(5, 10, 20, 40, 80, 160, 1e9)):
    """cnt2 [n,2] = (fail, success); base = dict(pl, pp, lg) arrays [n,NG,2] of decayed counts (fail, success)."""
    tr, va, dv = F.masks(R)
    F.C.audit_fit(f"phase1b_{name}_hier", [{"s": s, "w": w} for s, w in zip(R.s[tr | va], R.w[tr | va])])
    R.base = base
    tuned, table = F.tune_hier(R, cnt2, "pl", "pp", "lg", tr, va, kps=tune_kappas, kpos=(20, 100, 500))
    P0 = F.hier_base(R, "pl", "pp", "lg", tuned["gamma_idx"], tuned["kappa_player"], tuned["kappa_pos"])
    P_lg = F.hier_base(R, "pl", "pp", "lg", 0, 1e12, 1e12)
    P_pos = F.hier_base(R, "pl", "pp", "lg", tuned["gamma_idx"], 1e12, 1e-6)
    vals = np.array([0.0, 1.0])
    logP0 = np.log(P0)
    ex = inter_binary(P0, R, opp_i, scheme_i)
    phi = np.array([[0.0], [1.0]])

    def fn(X, lvl):
        return F.fit_family(logP0, cnt2, X, phi, tr, va, dv)
    rep, preds = F.nested_study(R, cnt2, vals, fn, ex, {"success": np.array([1])})
    sc_lg = F.score_rows(P_lg, cnt2, vals)
    out = {"hier_tuning": tuned, "n_dev_opps": float(cnt2[dv].sum()), "dev_success_rate": float(cnt2[dv][:, 1].sum() / cnt2[dv].sum()),
           "lookup_baselines": {"league_rate": F.period_summary(sc_lg, R, dv),
                                "position_rate": F.period_summary(F.score_rows(P_pos, cnt2, vals), R, dv, sc_lg),
                                "naive_player_specific_kappa5": F.period_summary(F.score_rows(F.hier_base(R, "pl", "pp", "lg", tuned["gamma_idx"], 5.0, 100.0), cnt2, vals), R, dv, sc_lg),
                                "hierarchical_B0": F.period_summary(F.score_rows(P0, cnt2, vals), R, dv, sc_lg)},
           "player_persistence_B0_vs_position": F.period_summary(F.score_rows(P0, cnt2, vals), R, dv, F.score_rows(P_pos, cnt2, vals)),
           "nested": rep, "reliability": reliability(preds[rep["selected_level"]][:, 1], cnt2, dv)}
    return out, {"P0": P0, "P_pos": P_pos, "P_lg": P_lg, "preds": preds, "sel": preds[rep["selected_level"]]}


def reliability(p, cnt2, mask, bins=8):
    p = p[mask]; n = cnt2[mask].sum(1); y = cnt2[mask][:, 1]
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9
    rows, ece = [], 0.0
    N = n.sum()
    for i in range(bins):
        m = (p >= q[i]) & (p < q[i + 1])
        if m.sum() == 0:
            continue
        pm = float((p[m] * n[m]).sum() / n[m].sum()); om = float(y[m].sum() / n[m].sum())
        rows.append({"pred": round(pm, 4), "obs": round(om, 4), "n": float(n[m].sum())})
        ece += n[m].sum() / N * abs(pm - om)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    brier = float((y * (1 - pc) ** 2 + (n - y) * pc ** 2).sum() / N)
    return {"ece": round(float(ece), 5), "brier": round(brier, 5), "bins": rows}


def run_pmf(R, cnt, keys, vals, phi, groups, name, tendency_fn, opp_i, scheme_i, kappas=(5, 10, 20, 40, 80, 160, 1e9)):
    tr, va, dv = F.masks(R)
    F.C.audit_fit(f"phase1b_{name}_hier", [{"s": s, "w": w} for s, w in zip(R.s[tr | va], R.w[tr | va])])
    tuned, table = F.tune_hier(R, cnt, keys[0], keys[1], keys[2], tr, va, kps=kappas, kpos=(20, 100, 500))
    P0 = F.hier_base(R, keys[0], keys[1], keys[2], tuned["gamma_idx"], tuned["kappa_player"], tuned["kappa_pos"])
    P_lg = F.hier_base(R, keys[0], keys[1], keys[2], 0, 1e12, 1e12)
    P_pos = F.hier_base(R, keys[0], keys[1], keys[2], tuned["gamma_idx"], 1e12, 1e-6)
    logP0 = np.log(P0)
    t = tendency_fn(P0)
    t = t - t[tr].mean()
    def col(nm, i):
        x = R.fam[nm][:, i].astype(float); m = np.nanmean(x[tr]); return np.where(np.isnan(x), m, x)
    ex = np.column_stack([t * col("opp", opp_i)] + [t * col("scheme", j) for j in scheme_i])

    def fn(X, lvl):
        return F.fit_family(logP0, cnt, X, phi, tr, va, dv)
    rep, preds = F.nested_study(R, cnt, vals, fn, ex, groups)
    sc_lg = F.score_rows(P_lg, cnt, vals)
    out = {"hier_tuning": tuned, "hier_tuning_grid_top5": sorted(table, key=lambda r: r["valid_logscore"])[:5],
           "lookup_baselines": {"league_pmf": F.period_summary(sc_lg, R, dv),
                                "position_pmf": F.period_summary(F.score_rows(P_pos, cnt, vals), R, dv, sc_lg),
                                "naive_player_specific_kappa5": F.period_summary(F.score_rows(F.hier_base(R, keys[0], keys[1], keys[2], tuned["gamma_idx"], 5.0, 100.0), cnt, vals), R, dv, sc_lg),
                                "hierarchical_B0": F.period_summary(F.score_rows(P0, cnt, vals), R, dv, sc_lg)},
           "player_persistence_B0_vs_position": F.period_summary(F.score_rows(P0, cnt, vals), R, dv, F.score_rows(P_pos, cnt, vals)),
           "nested": rep}
    return out, {"P0": P0, "P_pos": P_pos, "P_lg": P_lg, "preds": preds, "sel": preds[rep["selected_level"]], "vals": vals}


def run_td(Rr, Rc):
    """TD conversion hazards. Rush TD per carry and receiving TD per target, stratified red zone (<=20) / outside; strongly shrunk
    (kappa tuned on validation, position-average and naive player-specific baselines reported explicitly). Non-red-zone
    explosive TDs are the outside stratum (their yards are governed by the explosive-rush / explosive-YAC processes)."""
    out, aux = {}, {}
    for nm, R, key in (("rush_td", Rr, "ev"), ("rec_td", Rc, "tev")):
        S = Stacked(R, 2)
        ev = R.cnt[key]
        cnt2 = np.concatenate([np.column_stack([ev[:, 0] - ev[:, 1], ev[:, 1]]), np.column_stack([ev[:, 2] - ev[:, 3], ev[:, 3]])])
        bkey = {"ev": ("evp", "evpos", "evlg"), "tev": ("tv_pl", "tv_pp", "tv_lg")}[key]
        bp = {}
        for lvl, k in zip(("pl", "pp", "lg"), bkey):
            a = R.base[k]
            bp[lvl] = np.concatenate([np.stack([a[:, :, 0] - a[:, :, 1], a[:, :, 1]], axis=2), np.stack([a[:, :, 2] - a[:, :, 3], a[:, :, 3]], axis=2)])
        out[nm], aux[nm] = run_binary(S, cnt2, bp, opp_i=0 if nm == "rush_td" else 0, scheme_i=(0,) if nm == "rush_td" else (0, 7), name=nm)
    return out, aux


def kmeans(Z, k, seed=0, iters=50):
    rng = np.random.default_rng(seed)
    cen = Z[rng.choice(len(Z), k, replace=False)]
    for _ in range(iters):
        lab = ((Z[:, None, :] - cen[None]) ** 2).sum(2).argmin(1)
        new = np.array([Z[lab == j].mean(0) if (lab == j).any() else cen[j] for j in range(k)])
        if np.allclose(new, cen):
            break
        cen = new
    return cen


def archetype_test(R, cnt, vals, P0, phi, traits, level_fams, extra, name, ks=(3, 5)):
    """Do data-derived archetypes (k-means on as-of traits, fit on TRAIN only) improve out-of-time distributions beyond the selected
    feature level?  Archetype dummies are appended to the selected level's features; compared on DEVELOPMENT with a paired bootstrap."""
    tr, va, dv = F.masks(R)
    logP0 = np.log(P0)
    X0 = F.impute_stack(R, level_fams, tr | va, extra)
    ref_P, _ = F.fit_family(logP0, cnt, X0, phi, tr, va, dv)
    ref_sc = F.score_rows(ref_P, cnt, vals)
    mu, sd = traits[tr].mean(0), traits[tr].std(0) + 1e-9
    Z = (traits - mu) / sd
    out = {"reference_level_features": level_fams, "reference_valid_logscore_train_only_fit": None}
    preds = {}
    _, ref_info = F.fit_family(logP0, cnt, X0, phi, tr, va, dv)
    out["reference_valid_logscore_train_only_fit"] = ref_info["valid_logscore_train_only_fit"]
    for k in ks:
        cen = kmeans(Z[tr], k)
        lab = ((Z[:, None, :] - cen[None]) ** 2).sum(2).argmin(1)
        D = np.eye(k)[lab][:, 1:]
        P, info = F.fit_family(logP0, cnt, np.column_stack([X0, D]), phi, tr, va, dv)
        sc = F.score_rows(P, cnt, vals)
        out[f"k{k}"] = {"development": F.period_summary(sc, R, dv, ref_sc), "valid_logscore_train_only_fit": info["valid_logscore_train_only_fit"]}
        preds[k] = P
    # adoption: validation log score improves >= 1e-4 nats on the reference AND development improvement has p < 0.10 and >= 1e-3 nats
    best = None
    for k in ks:
        v = out[f"k{k}"]
        vi = out["reference_valid_logscore_train_only_fit"] - v["valid_logscore_train_only_fit"]
        dvs = v["development"]["vs_ref"]
        ok = vi >= 1e-4 and dvs["p_logscore_not_better"] < 0.10 and dvs["logscore_improvement_per_opp"] >= 1e-3
        v["valid_improvement"] = vi; v["adopt"] = bool(ok)
        if ok and (best is None or vi > best[1]):
            best = (k, vi)
    out["adopted_k"] = None if best is None else best[0]
    return out, preds
