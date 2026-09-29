"""
NFL_PHASE1_ROLE_STATE  (Phase 1A, shadow research)

Estimates a player's current role from strictly prior games and forecasts
his NEXT-game share (conditional on playing) plus P(role expands / stable /
contracts), for:
  rb_carry   RB share of team rush attempts        qb_rush  QB share of team rush attempts
  target     WR/TE/RB share of team targets        route    WR/TE/RB route rate (2023-2025 truth only)
  qb_att     QB share of team dropbacks (attempts) rz_carry / rz_target  red-zone shares
  def_snap   defensive snap share

Formulations compared on burned development data:
  R0 last8      mean of last 8 played games (baseline)
  R1 ewma       exponentially weighted mean, half-life tuned on train
  R2 kalman     local-level state-space filter (random-walk role + observation
                noise; extra drift per missed game), q/r tuned on train by
                predictive log-likelihood
  R3 bocpd      Bayesian online changepoint detection (constant hazard, normal
                segments), hazard tuned on train
  R4 hmm        discrete role states (K=5 share levels), transition matrix and
                emission sd estimated on train, forward filtering
  R5 learned    gradient boosting on usage history + availability + teammate
                absence + organisational priors + R2-R4 filter outputs
                ("continuous latent role score"); family ablations below

Role change classes: next share minus last-8 level > +delta (expand),
< -delta (contract), else stable; delta = 0.05 (0.10 for snap shares).
"""
import math
from collections import defaultdict

import numpy as np
from scipy import stats

import nfl_phase1_common as C

TYPES = {
    "rb_carry": {"key": "car_sh", "pos": ("RB",), "delta": 0.05},
    "qb_rush": {"key": "car_sh", "pos": ("QB",), "delta": 0.05},
    "target": {"key": "tgt_sh", "pos": ("WR", "TE", "RB"), "delta": 0.05},
    "route": {"key": "route_rt", "pos": ("WR", "TE", "RB"), "delta": 0.10},
    "qb_att": {"key": "att_sh", "pos": ("QB",), "delta": 0.10},
    "rz_carry": {"key": "rz_car_sh", "pos": ("RB", "QB"), "delta": 0.05},
    "rz_target": {"key": "rz_tgt_sh", "pos": ("WR", "TE", "RB"), "delta": 0.05},
    "def_snap": {"key": "pct", "pos": ("DL", "LB", "DB"), "delta": 0.10},
}


# ------------------------------------------------------------------ filters
def seq_of(hist, key):
    """(values, gaps before each value) of played games with a defined share."""
    vals, gaps = [], []
    for g in hist:
        if key == "pct":
            vals.append(g["pct"]); gaps.append(0)
        elif g.get("played") and g.get(key) is not None:
            vals.append(g[key]); gaps.append(0)
    return vals


def r_last8(v, prior):
    return (float(np.mean(v[-8:])), None) if v else (prior, None)


def r_ewma(v, prior, hl):
    if not v:
        return prior, None
    x = np.array(v[-12:]); w = 0.5 ** (np.arange(len(x))[::-1] / hl)
    return float(np.dot(w, x) / w.sum()), None


def r_kalman(v, prior, q, r, p0=0.02, gap=0):
    m, P = prior, p0
    for y in v:
        P = P + q
        K = P / (P + r); m = m + K * (y - m); P = (1 - K) * P
    P = P + q * (1 + gap)
    return m, P + r


def r_bocpd(v, prior, hazard, r, tau2=0.02, maxrun=16):
    # run-length distribution with normal-normal segments (known obs variance r)
    probs = np.array([1.0]); means = np.array([prior]); vars_ = np.array([tau2])
    for y in v:
        pred_sd = np.sqrt(vars_ + r)
        lik = stats.norm.pdf(y, means, pred_sd)
        growth = probs * lik * (1 - hazard)
        cp = np.sum(probs * lik * hazard)
        probs = np.append(cp, growth)
        # posterior update of segment means
        k = vars_ / (vars_ + r)
        new_means = means + k * (y - means); new_vars = (1 - k) * vars_
        means = np.append(prior, new_means); vars_ = np.append(tau2, new_vars)
        probs = probs / probs.sum()
        if len(probs) > maxrun:
            probs, means, vars_ = probs[:maxrun], means[:maxrun], vars_[:maxrun]
            probs = probs / probs.sum()
    # predictive: next obs may start a new segment
    mix_p = np.append(hazard, (1 - hazard) * probs) if False else probs
    mu = float(np.dot(mix_p, means)); var = float(np.dot(mix_p, vars_ + r + (means - mu) ** 2))
    return mu, var


class HMM:
    def __init__(self, train_seqs, K=5):
        allv = np.concatenate([np.array(s) for s in train_seqs if s]) if train_seqs else np.array([0.0])
        qs = np.quantile(allv, np.linspace(0.1, 0.9, K))
        self.mu = np.unique(qs) if len(np.unique(qs)) == K else np.linspace(allv.min(), allv.max(), K)
        self.K = len(self.mu)
        T = np.ones((self.K, self.K)); res = []
        for s in train_seqs:
            st = [int(np.argmin(np.abs(self.mu - x))) for x in s]
            res += [x - self.mu[i] for x, i in zip(s, st)]
            for a, b in zip(st[:-1], st[1:]):
                T[a, b] += 1
        self.T = T / T.sum(1, keepdims=True)
        self.sd = float(np.std(res)) if res else 0.05
        c = np.bincount([int(np.argmin(np.abs(self.mu - x))) for s in train_seqs for x in s[:1]], minlength=self.K) + 1.0
        self.pi = c / c.sum()

    def predict(self, v):
        a = self.pi.copy()
        for y in v:
            a = a * stats.norm.pdf(y, self.mu, self.sd); a = a / max(a.sum(), 1e-300)
            a = a @ self.T
        mu = float(a @ self.mu); var = float(a @ (self.sd ** 2 + (self.mu - mu) ** 2))
        return mu, var


# ------------------------------------------------------------------ rows
def type_rows(units, tname):
    """One row per (candidate who PLAYED the target game, share type)."""
    cfg = TYPES[tname]; key = cfg["key"]
    rows = []
    for u in units:
        s, w, team = u["key"]
        cands = u["defenders"] if tname == "def_snap" else u["players"]
        # teammate information for absence features (T-24 statuses)
        mates = []
        for r in cands:
            grp = r.get("grp") or r.get("pos")
            if grp not in cfg["pos"]:
                continue
            v = seq_of(r["hist"], key)
            l8 = float(np.mean(v[-8:])) if v else 0.0
            st = (r["inj"] or ("",))[0] if r.get("inj") else ""
            mates.append((r, l8, st, v))
        T = u["team_actual"]
        for r, l8, st, v in mates:
            if len(v) < 1:
                continue
            a = r["actual"]
            y = None
            if a["played"]:
                if tname == "def_snap":
                    y = a["pct"]
                else:
                    den = {"car_sh": T["rushes"], "tgt_sh": T["targets"], "att_sh": T["dropbacks"], "route_rt": T["dropbacks"],
                           "rz_car_sh": T["rz_rushes"], "rz_tgt_sh": T["rz_targets"]}[key]
                    num = {"car_sh": a["car"], "tgt_sh": a["tgt"], "att_sh": a["att"], "route_rt": a["routes"],
                           "rz_car_sh": a["rz_car"], "rz_tgt_sh": a["rz_tgt"]}[key]
                    y = num / den if (num is not None and den) else None
            vac = sum(ml8 for m, ml8, mst, _ in mates if m is not r and mst in ("Out", "Doubtful"))
            ret = sum(ml8 for m, ml8, mst, _ in mates if m is not r and (m.get("_F") or {}).get("returning"))
            rows.append({"s": s, "w": w, "team": team, "id": r.get("gid") or r.get("pfr"), "grp": r.get("grp") or r.get("pos"),
                         "v": v, "y": y, "l8": float(np.mean(v[-8:])), "vac": vac, "ret": ret, "ref": r, "unit": u})
    return rows


def tune_and_predict(rows, tname):
    tr = [r for r in rows if r["y"] is not None and (C.TRAIN(r["s"], r["w"]) or C.VALID(r["s"], r["w"]))]
    prior_by = defaultdict(list)
    for r in tr:
        prior_by[r["grp"]].append(r["y"])
    prior = {g: float(np.mean(v)) for g, v in prior_by.items()}
    for r in rows:
        r["prior"] = prior.get(r["grp"], float(np.mean([x["y"] for x in tr])))
    y = lambda rr: np.array([r["y"] for r in rr])
    ytr = y(tr)
    # R1 half-life
    best = None
    for hl in (1.0, 2.0, 3.0, 5.0, 8.0):
        e = np.mean(np.abs(np.array([r_ewma(r["v"], r["prior"], hl)[0] for r in tr]) - ytr))
        best = (e, hl) if best is None or e < best[0] else best
    hl = best[1]
    # R2 kalman q, r by predictive log-lik
    best = None
    for q in (0.0005, 0.001, 0.002, 0.004, 0.008):
        for rr_ in (0.002, 0.005, 0.01, 0.02):
            mv = [r_kalman(r["v"], r["prior"], q, rr_) for r in tr]
            ll = np.mean(stats.norm.logpdf(ytr, [m for m, _ in mv], np.sqrt([vv for _, vv in mv])))
            best = (ll, q, rr_) if best is None or ll > best[0] else best
    _, kq, kr = best
    # R3 bocpd hazard
    best = None
    for hz in (0.02, 0.05, 0.1, 0.2):
        mv = [r_bocpd(r["v"][-12:], r["prior"], hz, kr) for r in tr]
        ll = np.mean(stats.norm.logpdf(ytr, [m for m, _ in mv], np.sqrt([vv for _, vv in mv])))
        best = (ll, hz) if best is None or ll > best[0] else best
    hz = best[1]
    hmm = HMM([r["v"][-12:] for r in tr])
    for r in rows:
        r["R0"] = r_last8(r["v"], r["prior"])[0]
        r["R1"] = r_ewma(r["v"], r["prior"], hl)[0]
        r["R2"], r["R2v"] = r_kalman(r["v"], r["prior"], kq, kr)
        r["R3"], r["R3v"] = r_bocpd(r["v"][-12:], r["prior"], hz, kr)
        r["R4"], r["R4v"] = hmm.predict(r["v"][-12:])
    return {"ewma_half_life": hl, "kalman_q": kq, "kalman_r": kr, "bocpd_hazard": hz, "hmm_levels": [round(float(x), 4) for x in hmm.mu]}


def learned_features(r, tname):
    ref = r["ref"]; F = ref.get("_F", {})
    v = r["v"]
    out = {"l1": v[-1], "l3": float(np.mean(v[-3:])), "l8": r["l8"], "sd5": float(np.std(v[-5:])) if len(v) >= 3 else None,
           "n": float(len(v)), "vac": r["vac"], "ret": r["ret"], "R2": r["R2"], "R3": r["R3"], "R4": r["R4"],
           "R2v": r["R2v"], "prior": r["prior"]}
    for k in ("gap", "returning", "team_change", "rookie", "years_exp", "draft_round", "draft_pick", "depth_rank", "depth_listed",
              "st_Questionable", "st_Doubtful", "pr_limited", "pr_dnp", "snap_l3", "snap_l8", "snap_trend", "p_active_T24"):
        out[k] = F.get(k) if k != "p_active_T24" else ref.get("p_active_T24")
    out["opp_env"] = F.get("_opp_env")
    return out


FAMILIES = {
    "usage_only": ["l1", "l3", "l8", "sd5", "n"],
    "+own_availability": ["st_Questionable", "st_Doubtful", "pr_limited", "pr_dnp", "gap", "returning", "p_active_T24", "snap_l3", "snap_l8", "snap_trend"],
    "+teammate_absence": ["vac", "ret"],
    "+organizational_intent": ["rookie", "years_exp", "draft_round", "draft_pick", "team_change"],
    "+opponent": ["opp_env"],
    "+role_state_filters": ["R2", "R3", "R4", "R2v"],
}


def learned_models(rows, tname):
    for r in rows:
        r["LF"] = learned_features(r, tname)
    lab = [r for r in rows if r["y"] is not None]
    tr = [r for r in lab if C.TRAIN(r["s"], r["w"])]; va = [r for r in lab if C.VALID(r["s"], r["w"])]
    dv = [r for r in lab if C.DEV(r["s"], r["w"])]
    y = lambda rr: np.array([r["y"] for r in rr])
    M = lambda rr, cols: C.matrix([r["LF"] for r in rr], cols)
    abl = {}
    cols = []
    preds = {}
    for fam, add in FAMILIES.items():
        cols = cols + add
        if len(tr) < 200 or len(va) < 50:
            continue
        b = C.fit_xgb(M(tr, cols), y(tr), M(va, cols), y(va), cols, "reg:absoluteerror")
        p = np.clip(C.xgb_pred(b, M(rows, cols), cols), 0, 1)
        preds[fam] = p
        pd = np.array([p[i] for i, r in enumerate(rows) if r["y"] is not None and C.DEV(r["s"], r["w"])])
        abl[fam] = C.cont(pd, y(dv))
    # leave-one-family-out from the full set (does each family earn its place?)
    full = [c for f in FAMILIES.values() for c in f]
    loo = {}
    if len(tr) >= 200 and len(va) >= 50:
        bfull = preds.get("+role_state_filters")
        efull = np.abs(np.array([bfull[i] for i, r in enumerate(rows) if r["y"] is not None and C.DEV(r["s"], r["w"])]) - y(dv))
        blocks = np.array([f"{r['s']}-{r['w']}" for r in dv])
        for fam, add in FAMILIES.items():
            if fam == "usage_only":
                continue
            cols2 = [c for c in full if c not in add]
            b = C.fit_xgb(M(tr, cols2), y(tr), M(va, cols2), y(va), cols2, "reg:absoluteerror")
            p2 = np.clip(C.xgb_pred(b, M(dv, cols2), cols2), 0, 1)
            imp, pv = C.block_boot(efull, np.abs(p2 - y(dv)), blocks)
            loo[fam] = {"mae_without_family": round(float(np.abs(p2 - y(dv)).mean()), 5),
                        "mae_full": round(float(efull.mean()), 5), "family_contribution": imp, "p_family_not_helpful": pv}
    return preds, abl, loo


def class_probs(mu, var, l8, delta):
    sd = np.sqrt(np.maximum(var, 1e-6))
    pe = 1 - stats.norm.cdf(l8 + delta, mu, sd); pc = stats.norm.cdf(l8 - delta, mu, sd)
    return np.column_stack([pc, np.clip(1 - pe - pc, 1e-6, 1), pe])


def role_change_scores(rows, tname, resid_sd):
    """3-class logloss of P(contract, stable, expand)."""
    d = TYPES[tname]["delta"]
    tr = [r for r in rows if r["y"] is not None and (C.TRAIN(r["s"], r["w"]) or C.VALID(r["s"], r["w"]))]
    dv = [r for r in rows if r["y"] is not None and C.DEV(r["s"], r["w"])]
    cls = lambda r: 2 if r["y"] - r["l8"] > d else 0 if r["y"] - r["l8"] < -d else 1
    freq = np.bincount([cls(r) for r in tr], minlength=3) + 1.0; freq = freq / freq.sum()
    yc = np.array([cls(r) for r in dv])
    out = {"delta": d, "dev_class_rates": [round(float(x), 4) for x in np.bincount(yc, minlength=3) / len(yc)]}
    ll = lambda P: float(-np.mean(np.log(np.clip(P[np.arange(len(yc)), yc], 1e-6, 1))))
    out["B0_class_frequency"] = ll(np.tile(freq, (len(dv), 1)))
    l8 = np.array([r["l8"] for r in dv])
    for name in ("R2", "R3", "R4"):
        P = class_probs(np.array([r[name] for r in dv]), np.array([r[name + "v"] for r in dv]), l8, d)
        P = P / P.sum(1, keepdims=True)
        out[name] = ll(P)
    if "R5" in dv[0]:
        P = class_probs(np.array([r["R5"] for r in dv]), np.full(len(dv), resid_sd ** 2), l8, d)
        P = P / P.sum(1, keepdims=True)
        out["R5"] = ll(P)
    return out


def regimes(r):
    F = r["ref"].get("_F", {})
    v = r["v"]
    out = []
    if r["vac"] >= 0.15:
        out.append("teammate_injury(vacated>=0.15)")
    if r["ret"] >= 0.15:
        out.append("teammate_returning")
    if F.get("returning"):
        out.append("player_returning")
    if F.get("rookie"):
        out.append("rookie")
    if F.get("team_change"):
        out.append("team_change")
    ref = r["ref"]
    if ref.get("depth") and ref.get("depth_prev"):
        if ref["depth"][1] < ref["depth_prev"][1]:
            out.append("depth_promotion")
        elif ref["depth"][1] > ref["depth_prev"][1]:
            out.append("depth_demotion")
    if len(v) >= 3:
        if v[-1] - r["l8"] > 0.10:
            out.append("abrupt_expansion(l1-l8>0.10)")
        if v[-1] - r["l8"] < -0.10:
            out.append("abrupt_contraction(l1-l8<-0.10)")
    return out


def evaluate(units):
    rep = {"label": C.DEV_LABEL, "types": {}}
    selected = {}
    for tname in TYPES:
        rows = type_rows(units, tname)
        if not rows:
            continue
        params = tune_and_predict(rows, tname)
        preds, abl, loo = learned_models(rows, tname)
        if "+role_state_filters" in preds:
            for r, p in zip(rows, preds["+role_state_filters"]):
                r["R5"] = float(p)
        dv = [r for r in rows if r["y"] is not None and C.DEV(r["s"], r["w"])]
        if not dv:
            continue
        y = np.array([r["y"] for r in dv]); blocks = np.array([f"{r['s']}-{r['w']}" for r in dv])
        e0 = np.abs(np.array([r["R0"] for r in dv]) - y)
        res = {"n_dev": len(dv), "n_train_valid": sum(1 for r in rows if r["y"] is not None and (C.TRAIN(r["s"], r["w"]) or C.VALID(r["s"], r["w"]))),
               "params": params, "models": {}}
        names = ["R0", "R1", "R2", "R3", "R4"] + (["R5"] if "R5" in dv[0] else [])
        for n in names:
            p = np.array([r[n] for r in dv])
            m = {"combined": C.cont(p, y)}
            for t2, sel in (("2025", lambda r: r["s"] == 2025), ("2026_wk1_3", lambda r: r["s"] == 2026)):
                ix = np.array([sel(r) for r in dv])
                if ix.sum():
                    m[t2] = C.cont(p[ix], y[ix])
            imp, pv = C.block_boot(np.abs(p - y), e0, blocks)
            m["vs_R0"] = {"mae_improvement": imp, "p_not_better": pv}
            res["models"][n] = m
        # regimes (next-game share MAE, each model)
        reg = defaultdict(list)
        for i, r in enumerate(dv):
            for g in regimes(r):
                reg[g].append(i)
        vols = np.array([np.std(r["v"][-5:]) if len(r["v"]) >= 3 else 0 for r in dv])
        hi = np.quantile(vols, 2 / 3)
        reg["high_volatility(top tercile sd5)"] = [i for i in range(len(dv)) if vols[i] >= hi]
        res["regimes"] = {}
        for g, ix in sorted(reg.items()):
            if len(ix) < 15:
                continue
            ix = np.array(ix)
            res["regimes"][g] = {"n": int(len(ix)), **{n: round(float(np.abs(np.array([dv[i][n] for i in ix]) - y[ix]).mean()), 5) for n in names}}
        resid_sd = float(np.std([r["y"] - r.get("R5", r["R2"]) for r in rows if r["y"] is not None and C.VALID(r["s"], r["w"])] or [0.1]))
        res["role_change_logloss"] = role_change_scores(rows, tname, resid_sd)
        res["learned_family_ablation_forward"] = abl
        res["learned_leave_one_family_out"] = loo
        # selection on development MAE with block bootstrap vs R0
        cand = [n for n in names if n != "R0"]
        best = min(cand, key=lambda n: res["models"][n]["combined"]["mae"])
        v = res["models"][best]["vs_R0"]
        res["selected"] = best if v["mae_improvement"] > 0 and v["p_not_better"] < 0.10 else "R0"
        selected[tname] = res["selected"]
        for r in rows:
            r["ref"].setdefault("_role", {})[tname] = {"pred": r[res["selected"]], "R0": r["R0"],
                                                       "var": r.get("R2v")}
        rep["types"][tname] = res
    return rep, selected
