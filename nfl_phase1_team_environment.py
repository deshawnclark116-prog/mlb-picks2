"""
NFL_PHASE1_TEAM_ENVIRONMENT  (Phase 1A, shadow research)

Pre-game distributions for team offensive volume (plays, dropbacks, rush
attempts, designed rushes, targets, red-zone plays/rushes/targets, goal-line
plays/rushes) and team defensive snaps. No sportsbook inputs.

Candidates, compared on burned development data against the Phase 0B blend:
  B0  blend       team season-to-date mean, prior season blended at 3 games' weight
  C1  opp_adjust  B0 + a * (opponent-allowed blend - league mean)  (a fit on train)
  C2  poisson_glm log-link GLM: log B0, log opponent-allowed ratio, home, rest
                  difference, week, starting-QB-out flag
  C3  xgb_poisson gradient boosting (count:poisson) on the C2 inputs + EWMA/last-3
All means are wrapped in a negative binomial with dispersion k fit on the
training split (method of moments), which gives quantiles and exact CRPS.
"""
import math
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

import nfl_phase1_common as C

TARGETS = ["plays", "dropbacks", "rushes", "rushes_designed", "targets", "rz_plays", "rz_rushes", "rz_targets",
           "gl_plays", "gl_rushes", "def_snaps"]


def blend(hist, s, key):
    cur = [g[key] for g in hist if g["s"] == s]; pri = [g[key] for g in hist if g["s"] == s - 1]
    c = sum(cur) / len(cur) if cur else None; p = sum(pri) / len(pri) if pri else None
    if c is None:
        return p
    if p is None:
        return c
    return (c * len(cur) + p * 3) / (len(cur) + 3)


def ewma(hist, key, hl=4.0):
    if not hist:
        return None
    w = np.array([0.5 ** ((len(hist) - 1 - i) / hl) for i in range(len(hist))])
    return float(np.dot(w, [g[key] for g in hist]) / w.sum())


def team_rows(units, qb_out, lg=None):
    """One row per team-game with pre-game features (T-24 information). lg: fixed league means per target (Phase 1D serving of serialized models);
    None = mean over TRAIN rows of the units passed (research behaviour)."""
    rows = []
    league = defaultdict(list)          # season -> list of per-game team values (as absorbed)
    for u in units:
        s, w, team = u["key"]
        th, oa = u["team_hist"], u["opp_allowed_hist"]
        oo, da = u["opp_off_hist"], u["def_allowed_hist"]
        r = {"s": s, "w": w, "team": team, "opp": u["opp"], "home": float(u["home"]),
             "rest_diff": (C.safe(u["rest"], 7) - C.safe(u["opp_rest"], 7)), "week": float(w),
             "qb_out": qb_out.get(u["key"], 0.0), "n_cur": float(sum(1 for g in th if g["s"] == s))}
        for k in TARGETS:
            if k == "def_snaps":
                r[f"b0_{k}"] = blend(th, s, k)
                r[f"opp_{k}"] = blend(oo, s, "plays")            # opponent offense volume
                r[f"ewma_{k}"] = ewma(th[-12:], k)
            else:
                r[f"b0_{k}"] = blend(th, s, k)
                r[f"opp_{k}"] = blend(oa, s, k)                   # what the opponent's defense allows
                r[f"ewma_{k}"] = ewma(th[-12:], k)
            r[f"l3_{k}"] = (sum(g[k] for g in th[-3:]) / len(th[-3:])) if th else None
            r[f"y_{k}"] = u["team_actual"][k]
        rows.append(r)
    # league means per season from training rows only (fixed constants)
    for k in TARGETS:
        vals = [r[f"y_{k}"] for r in rows if C.TRAIN(r["s"], r["w"])] if lg is None else None
        for r in rows:
            r[f"lg_{k}"] = float(np.mean(vals)) if lg is None else float(lg[k])
    return rows


def _feat_glm(rows, k):
    X = []
    for r in rows:
        b0 = C.safe(r[f"b0_{k}"], r[f"lg_{k}"]); op = C.safe(r[f"opp_{k}"], r[f"lg_{k}"])
        X.append([1.0, math.log(max(b0, 0.5)), math.log(max(op, 0.5) / max(r[f"lg_{k}"], 0.5)), r["home"],
                  r["rest_diff"] / 7.0, r["qb_out"]])
    return np.array(X)


def fit_glm(X, y, l2=1e-3):
    def nll(b):
        eta = X @ b; mu = np.exp(eta)
        return float(np.sum(mu - y * eta)) / len(y) + l2 * float(b[1:] @ b[1:])

    def grad(b):
        mu = np.exp(X @ b)
        g = X.T @ (mu - y) / len(y); g[1:] += 2 * l2 * b[1:]
        return g
    b0 = np.zeros(X.shape[1]); b0[1] = 1.0
    return minimize(nll, b0, jac=grad, method="L-BFGS-B").x


def nb_k(mu, y):
    """Method-of-moments dispersion: var = mu + mu^2/k."""
    resid2 = (y - mu) ** 2
    excess = np.sum(resid2 - mu)
    return float(max(np.sum(mu ** 2) / excess, 1.0)) if excess > 0 else 1e4


XGB_COLS = lambda k: ["home", "rest_diff", "week", "qb_out", "n_cur", f"b0_{k}", f"opp_{k}", f"ewma_{k}", f"l3_{k}", f"lg_{k}"]


def qb_out_flags(units):
    """1.0 if the team's most-used recent QB is Out/Doubtful on the report available at T-24h."""
    out = {}
    for u in units:
        qbs = [r for r in u["players"] if r["pos"] == "QB" and r["hist"]]
        if not qbs:
            continue
        top = max(qbs, key=lambda r: sum(g["att"] for g in r["hist"][-3:]))
        out[u["key"]] = 1.0 if top["inj"] and top["inj"][0] in ("Out", "Doubtful") else 0.0
    return out


def evaluate(units, qb_out=None):
    qb_out = qb_out if qb_out is not None else qb_out_flags(units)
    rows = team_rows(units, qb_out)
    tr = [r for r in rows if C.TRAIN(r["s"], r["w"])]
    va = [r for r in rows if C.VALID(r["s"], r["w"])]
    dv = [r for r in rows if C.DEV(r["s"], r["w"])]
    out = {"label": C.DEV_LABEL, "targets": {}}
    fitted = {}
    for k in TARGETS:
        ok = lambda rr: [r for r in rr if r[f"b0_{k}"] is not None]
        tr_, va_, dv_ = ok(tr), ok(va), ok(dv)
        y = lambda rr: np.array([r[f"y_{k}"] for r in rr])
        preds = {}
        # B0
        b0 = lambda rr: np.array([r[f"b0_{k}"] for r in rr])
        preds["B0_blend"] = (b0(tr_), b0(va_), b0(dv_))
        # C1 opponent adjustment
        adj = lambda rr: np.array([C.safe(r[f"opp_{k}"], r[f"lg_{k}"]) - r[f"lg_{k}"] for r in rr])
        a = float(np.dot(adj(tr_), y(tr_) - b0(tr_)) / max(np.dot(adj(tr_), adj(tr_)), 1e-9))
        preds["C1_opp_adjust"] = tuple(b0(rr) + a * adj(rr) for rr in (tr_, va_, dv_))
        # C2 GLM
        beta = fit_glm(_feat_glm(tr_, k), y(tr_))
        preds["C2_poisson_glm"] = tuple(np.exp(_feat_glm(rr, k) @ beta) for rr in (tr_, va_, dv_))
        # C3 xgb poisson
        cols = XGB_COLS(k)
        C.audit_fit(f"team_{k}_xgb", tr_, va_)
        bst = C.fit_xgb(C.matrix(tr_, cols), y(tr_), C.matrix(va_, cols), y(va_), cols, "count:poisson")
        preds["C3_xgb_poisson"] = tuple(C.xgb_pred(bst, C.matrix(rr, cols), cols) for rr in (tr_, va_, dv_))
        res = {"n_train": len(tr_), "n_dev": len(dv_), "opp_adjust_a": round(a, 4), "glm_beta": [round(float(x), 4) for x in beta],
               "candidates": {}}
        blocks = np.array([f"{r['s']}-{r['w']}" for r in dv_])
        yd = y(dv_)
        ref_err = np.abs(preds["B0_blend"][2] - yd)
        kd = {}
        for name, (ptr, pva, pdv) in preds.items():
            kd[name] = nb_k(np.concatenate([ptr, pva]), np.concatenate([y(tr_), y(va_)]))
            m = {}
            for tag, sel in (("2025", lambda r: r["s"] == 2025), ("2026_wk1_3", lambda r: r["s"] == 2026), ("combined", lambda r: True)):
                idx = np.array([sel(r) for r in dv_])
                if idx.sum() == 0:
                    continue
                q = C.nb_quantiles(pdv[idx], kd[name])
                m[tag] = {**C.cont(pdv[idx], yd[idx]), "crps": round(C.crps_nb(pdv[idx], kd[name], yd[idx]), 4),
                          "cov80": C.coverage(q[0.10], q[0.90], yd[idx]), "cov50": C.coverage(q[0.25], q[0.75], yd[idx])}
            imp, p = C.block_boot(np.abs(pdv - yd), ref_err, blocks)
            crps_new = C.crps_nb_rows(pdv, kd[name], yd)
            if name == "B0_blend":
                crps_ref = crps_new
            imp_c, p_c = C.block_boot(crps_new, crps_ref, blocks)
            m["vs_B0"] = {"mae_improvement": imp, "p_mae_not_better": p, "crps_improvement": imp_c, "p_crps_not_better": p_c}
            m["nb_k"] = round(kd[name], 2)
            res["candidates"][name] = m
        # selection on development data: must beat B0 on MAE and CRPS with p < 0.10
        best = "B0_blend"
        for name, m in res["candidates"].items():
            if name == "B0_blend":
                continue
            v = m["vs_B0"]
            if v["mae_improvement"] > 0 and v["crps_improvement"] > 0 and v["p_mae_not_better"] < 0.10 and v["p_crps_not_better"] < 0.10:
                if best == "B0_blend" or m["combined"]["crps"] < res["candidates"][best]["combined"]["crps"]:
                    best = name
        res["selected"] = best
        out["targets"][k] = res
        C.audit_fit(f"team_{k}_glm", tr_)
        fitted[k] = {"selected": best, "a": a, "beta": beta, "k": kd[best], "k_b0": kd["B0_blend"], "xgb": bst}
    # derived pass rate from selected dropbacks / plays
    return out, fitted, rows


def predict(fitted, rows):
    """Selected-model mean and NB dispersion per team-game key and target."""
    out = {}
    for k, f in fitted.items():
        ok = [r for r in rows if r[f"b0_{k}"] is not None]
        if f["selected"] == "B0_blend":
            mu = np.array([r[f"b0_{k}"] for r in ok])
        elif f["selected"] == "C1_opp_adjust":
            mu = np.array([r[f"b0_{k}"] + f["a"] * (C.safe(r[f"opp_{k}"], r[f"lg_{k}"]) - r[f"lg_{k}"]) for r in ok])
        elif f["selected"] == "C2_poisson_glm":
            mu = np.exp(_feat_glm(ok, k) @ f["beta"])
        else:
            cols = XGB_COLS(k)
            mu = C.xgb_pred(f["xgb"], C.matrix(ok, cols), cols)
        for r, m in zip(ok, mu):
            out[(r["s"], r["w"], r["team"], k)] = (max(float(m), 0.1), f["k"])
        for r in rows:
            out.setdefault((r["s"], r["w"], r["team"], k), (r[f"lg_{k}"], f["k"]))
    return out
