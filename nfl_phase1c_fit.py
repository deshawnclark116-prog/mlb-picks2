"""
NFL_PHASE1C_FIT  (Phase 1C, shadow research)

Fit the ADJUDICATED Phase 1B component versions on an expanding window ending at `fit_end` (a season*100+week key) and return per-record
predictions (all rows, including forecast-only candidates) in the arrays the joint simulator consumes.

variants: 'adjudicated' (final), 'B0' (hierarchical player pmf/rate, no tilt), 'simple' (position-average pmf/rate; no player, no features).
"""
import numpy as np

import nfl_phase1_defense_events as DE
import nfl_phase1_efficiency as F
import nfl_phase1_receiving_efficiency as RC
import nfl_phase1_rushing_efficiency as RU
import nfl_phase1b_data as B
import nfl_phase1c_adjudicate as AJ


def _windows(R, fit_end):
    t = R.s * 100 + R.w
    tr_all = R.active & (t <= fit_end)
    weeks = np.unique(t[tr_all]); cut = weeks[int(len(weeks) * 0.75)]
    return tr_all & (t < cut), tr_all & (t >= cut)


def fit_predict(name, Rs, drecs, level, variant, fit_end, structure="single"):
    """Returns predictions for every row of the component's (possibly stacked) record set: pmf [n,K] or hazard [n,2]."""
    spec = AJ.spec_for(name, Rs, drecs)
    R, cnt = spec["R"], spec["cnt"]
    tr, va = _windows(R, fit_end)
    tuned, _ = F.tune_hier(R, cnt, *spec["keys"], tr, va, kps=AJ.KAPPAS, kpos=(20, 100, 500))
    P0 = F.hier_base(R, *spec["keys"], tuned["gamma_idx"], tuned["kappa_player"], tuned["kappa_pos"])
    if variant == "simple":
        return F.hier_base(R, *spec["keys"], tuned["gamma_idx"], 1e12, 1e-6), tuned
    if variant == "B0" or level == "B0" and structure == "single":
        return P0, tuned
    ex = spec["extra_fn"](P0, R, tr | va)
    X = F.impute_stack(R, F.NESTED[level], tr | va, ex)
    if structure.startswith("two"):
        T = int(structure[3:])
        te = np.zeros(R.n, bool)
        fn = RU.two_process_fn(P0, cnt, tr, va, te, T)
        P, info = fn(X, level)
    else:
        P, info = F.fit_family(np.log(P0), cnt, X, spec["phi"], tr, va, np.zeros(R.n, bool))
    tuned = {**{k: tuned[k] for k in ("gamma_idx", "kappa_player", "kappa_pos")}, "l2": info.get("l2") if isinstance(info, dict) else None, "n_features": int(X.shape[1])}
    return P, tuned


def fit_defense(target, drecs, level, variant, fit_end):
    y = np.array([r[target] for r in drecs], float); snaps = np.array([r["snaps"] for r in drecs], float)
    act = np.array([not r.get("extra", False) for r in drecs], bool)
    t = np.array([r["s"] * 100 + r["w"] for r in drecs])
    base = {lvl: np.stack([r["base"][target][lvl] for r in drecs]) for lvl in ("p", "pos", "lg")}
    fam = {f: np.array([r["fam"][f] for r in drecs], float) for f in ("team", "opp", "scheme", "personnel")}
    tr_all = act & (t <= fit_end)
    weeks = np.unique(t[tr_all]); cut = weeks[int(len(weeks) * 0.75)]
    tr, va = tr_all & (t < cut), tr_all & (t >= cut)
    from scipy.special import gammaln
    nll = lambda y_, mu: -(y_ * np.log(np.maximum(mu, 1e-12)) - mu - gammaln(y_ + 1))
    best = None
    for gi in (0, 1, 2):
        for kp in DE.KAPPA_GRID:
            for kq in (300.0, 1500.0):
                v = nll(y[va], AJ._vec_rate(base, gi, kp, kq)[va] * snaps[va]).mean()
                if best is None or v < best[0]:
                    best = (v, gi, kp, kq)
    _, gi, kp, kq = best
    hyper = {"gamma_idx": gi, "kappa_player_snaps": kp, "kappa_pos_snaps": kq, "distribution": "poisson given snaps"}
    if variant == "simple":
        return AJ._vec_rate(base, gi, 1e12, kq), hyper
    rate0 = AJ._vec_rate(base, gi, kp, kq)
    if variant == "B0" or level == "B0":
        return rate0, hyper
    z = np.log(np.maximum(rate0, 1e-6)); z = z - z[tr].mean()
    blocks = [fam[f] for f in F.NESTED[level] if f != "interaction"]
    if "interaction" in F.NESTED[level]:
        sc0 = fam["scheme"][:, 0]; sc0 = np.where(np.isnan(sc0), np.nanmean(sc0[tr]), sc0)
        blocks.append(np.column_stack([z * fam["opp"][:, 0], z * sc0]))
    X = np.column_stack(blocks); m_ = np.nanmean(X[tr | va], 0); ix = np.where(np.isnan(X)); X[ix] = m_[ix[1]]
    bl = None
    for l2 in (100.0, 1000.0, 10000.0):
        m = DE.PoissonTilt(l2).fit(y[tr], (rate0 * snaps)[tr], X[tr])
        v = nll(y[va], m.predict((rate0 * snaps)[va], X[va])).mean()
        if bl is None or v < bl[0]:
            bl = (v, l2)
    m = DE.PoissonTilt(bl[1]).fit(y[tr | va], (rate0 * snaps)[tr | va], X[tr | va])
    hyper["l2"] = bl[1]; hyper["n_features"] = int(X.shape[1])
    return m.predict(rate0, X), hyper


COMPONENTS = ["rush", "rush_td", "rec_air", "rec_catch", "rec_yac", "rec_td", "pass_sack", "pass_int", "pass_completion", "pass_air", "pass_yac", "pass_td"]


def fit_all(Rs, drecs, config, variant, fit_end):
    """config: {component: {"level": "B2", "structure": "single"}}; defenders: {"def_tackles": {"level": ...}, ...}"""
    E, hyper = {}, {}
    for c in COMPONENTS:
        cfg = config[c]
        P, tuned = fit_predict(c, Rs, drecs, cfg["level"], variant, fit_end, cfg.get("structure", "single"))
        E[c] = P; hyper[c] = tuned
    n_r, n_e, n_p = Rs["rush"].n, Rs["rec"].n, Rs["pass"].n
    out = {"rush": E["rush"], "rush_td": (E["rush_td"][:n_r, 1], E["rush_td"][n_r:, 1]),
           "air": E["rec_air"], "catch": E["rec_catch"][:, 1].reshape(4, n_e).T, "yac": E["rec_yac"].reshape(4, n_e, -1).transpose(1, 0, 2),
           "rec_td": (E["rec_td"][:n_e, 1], E["rec_td"][n_e:, 1]),
           "qb_sack": E["pass_sack"][:, 1], "qb_int": E["pass_int"][:, 1], "qb_comp": E["pass_completion"][:, 1].reshape(4, n_p).T}
    out["air_p"] = E["pass_air"]; out["yac_p"] = E["pass_yac"].reshape(4, n_p, -1).transpose(1, 0, 2); out["pass_td"] = E["pass_td"][:, 1]
    out["comp_p"] = out["qb_comp"]
    # league (as-of, position-free) completion baseline for the QB adjustment: all-league rate via kappa=inf
    Pl, _ = fit_predict("pass_completion", Rs, drecs, "B0", "simple", fit_end)
    out["qb_comp_league"] = Pl[:, 1].reshape(4, n_p).T
    dres = {t: fit_defense(t, drecs, config["def_" + t]["level"], variant, fit_end) for t in DE.TARGETS}
    out["def_rate"] = {t: dres[t][0] for t in DE.TARGETS}
    for t in DE.TARGETS:
        hyper["def_" + t] = dres[t][1]
    out["hyper"] = hyper
    return out
