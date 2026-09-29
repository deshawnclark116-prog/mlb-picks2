"""
NFL_PHASE1_AVAILABILITY  (Phase 1A, shadow research)

P(active) at T-24h and T-90m, plus the conditional snap-share distribution
given active.

Three-state (inactive / active-limited / active-normal) is NOT built: nflverse
has no "limited" label, and any snap-share threshold defining "limited" would be
arbitrary. Instead: P(active) + a separate conditional snap-share distribution
(quantiles), which carries the same information without a manufactured label.

T-24h inputs: injury report (assumption A2), previous week's roster status,
games missed, prior snap participation and trend, position, prior role, rookie /
experience, team change, depth-chart rank from the latest snapshot <= cutoff
(2025+ only). T-90m adds the game-day roster status (A3). The T-24h model never
sees game-day roster status (invariant).

Candidates: status lookup (Phase 0B baseline), L2 logistic regression, gradient
boosting. Selection on burned development data by logloss with a week-block
bootstrap (p < 0.10) versus the baseline.
"""
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

import nfl_phase1_common as C

STATUSES = ["none", "Questionable", "Doubtful", "Out", "other"]
PRACTICE = {"Full Participation in Practice": "full", "Limited Participation in Practice": "limited",
            "Did Not Participate In Practice": "dnp"}
ROSTER = ["ACT", "INA", "RES", "DEV", "CUT", "missing", "other"]


def _status(inj):
    if not inj or not inj[0]:
        return "none"
    return inj[0] if inj[0] in STATUSES else "other"


def player_features(u, r):
    """Pre-game features for an offensive candidate (T-24h information)."""
    s, w, team = u["key"]
    h = [g for g in r["hist"] if g["played"]]
    F = {"pos_" + p: 1.0 if r["pos"] == p else 0.0 for p in ("QB", "RB", "WR", "TE")}
    for k in ("car_sh", "tgt_sh", "route_rt", "att_sh", "rz_car_sh", "rz_tgt_sh", "snap"):
        vals = [g[k] for g in h if g[k] is not None]
        F[f"{k}_l1"] = vals[-1] if vals else None
        F[f"{k}_l3"] = float(np.mean(vals[-3:])) if vals else None
        F[f"{k}_l8"] = float(np.mean(vals[-8:])) if vals else None
        F[f"{k}_sd5"] = float(np.std(vals[-5:])) if len(vals) >= 3 else None
        F[f"{k}_trend"] = (F[f"{k}_l3"] - F[f"{k}_l8"]) if vals else None
        if vals:
            wts = 0.5 ** (np.arange(len(vals[-8:]))[::-1] / 2.0)
            F[f"{k}_ewm"] = float(np.dot(wts, vals[-8:]) / wts.sum())
        else:
            F[f"{k}_ewm"] = None
        F[f"{k}_n"] = float(len(vals))
    last_kick = h[-1]["kick"] if h else None
    F["gap"] = float(sum(1 for g in u["team_hist"] if last_kick is None or g["kick"] > last_kick)) if last_kick else 99.0
    F["gap"] = min(F["gap"], 20.0)
    F["returning"] = 1.0 if h and F["gap"] >= 1 else 0.0
    F["team_change"] = 1.0 if h and h[-1]["team"] != team else 0.0
    F["no_history"] = 0.0 if h else 1.0
    F["n_cur"] = float(r.get("n_cur", 0))
    st = _status(r["inj"])
    for x in STATUSES:
        F["st_" + x] = 1.0 if st == x else 0.0
    pr = PRACTICE.get((r["inj"] or ("", ""))[1], "none")
    for x in ("full", "limited", "dnp", "none"):
        F["pr_" + x] = 1.0 if pr == x else 0.0
    pv = r["prev_roster"] if r["prev_roster"] in ROSTER else "other"
    for x in ROSTER:
        F["prev_" + x] = 1.0 if pv == x else 0.0
    d = r["draft"] or {}
    F["rookie"] = 1.0 if d.get("rookie") == s else 0.0
    F["years_exp"] = float(s - d["rookie"]) if d.get("rookie") else None
    F["draft_round"] = d.get("round") if d.get("round") else 8.0
    F["draft_pick"] = d.get("pick") if d.get("pick") else 300.0
    dep = r.get("depth")
    F["depth_rank"] = dep[1] if dep else (9.0 if r.get("depth_listed_any") else None)
    F["depth_listed"] = (1.0 if dep else 0.0) if r.get("depth_listed_any") else None
    F["usage_l3"] = (C.safe(F["car_sh_l3"], 0) + C.safe(F["tgt_sh_l3"], 0) + C.safe(F["att_sh_l3"], 0))
    return F


def t90_features(r):
    g = r["game_roster_T90"] if r["game_roster_T90"] in ROSTER else "other"
    return {"t90_" + x: 1.0 if g == x else 0.0 for x in ROSTER}


def defender_features(u, r):
    s, w, team = u["key"]
    h = r["hist"]
    vals = [g["pct"] for g in h]
    F = {"grp_" + g: 1.0 if r["grp"] == g else 0.0 for g in ("DL", "LB", "DB")}
    F.update({"snap_l1": vals[-1] if vals else None, "snap_l3": float(np.mean(vals[-3:])) if vals else None,
              "snap_l8": float(np.mean(vals[-8:])) if vals else None,
              "snap_sd5": float(np.std(vals[-5:])) if len(vals) >= 3 else None})
    F["snap_trend"] = F["snap_l3"] - F["snap_l8"] if vals else None
    last_kick = h[-1]["kick"] if h else None
    F["gap"] = min(float(sum(1 for g in u["team_hist"] if last_kick is None or g["kick"] > last_kick)), 20.0)
    F["returning"] = 1.0 if F["gap"] >= 1 else 0.0
    st = _status(r["inj"])
    for x in STATUSES:
        F["st_" + x] = 1.0 if st == x else 0.0
    pr = PRACTICE.get((r["inj"] or ("", ""))[1], "none")
    for x in ("full", "limited", "dnp", "none"):
        F["pr_" + x] = 1.0 if pr == x else 0.0
    pv = r["prev_roster"] if r["prev_roster"] in ROSTER else "other"
    for x in ROSTER:
        F["prev_" + x] = 1.0 if pv == x else 0.0
    F["n_hist"] = float(len(h))
    return F


OFF_T24 = (["pos_QB", "pos_RB", "pos_WR", "pos_TE", "snap_l1", "snap_l3", "snap_l8", "snap_trend", "snap_n", "gap", "returning",
            "team_change", "no_history", "n_cur", "rookie", "years_exp", "draft_round", "usage_l3"]
           + ["st_" + x for x in STATUSES] + ["pr_" + x for x in ("full", "limited", "dnp", "none")] + ["prev_" + x for x in ROSTER])
OFF_T24_DEPTH = OFF_T24 + ["depth_rank", "depth_listed"]
DEF_T24 = (["grp_DL", "grp_LB", "grp_DB", "snap_l1", "snap_l3", "snap_l8", "snap_trend", "gap", "returning", "n_hist"]
           + ["st_" + x for x in STATUSES] + ["pr_" + x for x in ("full", "limited", "dnp", "none")] + ["prev_" + x for x in ROSTER])
T90_EXTRA = ["t90_" + x for x in ROSTER]
LOGIT_NUMERIC_FILL = 0.0


def fit_logit(X, y, l2=1.0):
    mu = np.nanmean(X, 0); sd = np.nanstd(X, 0) + 1e-6
    Z = np.nan_to_num((X - mu) / sd, nan=0.0)
    Z = np.hstack([np.ones((len(Z), 1)), Z])

    def f(b):
        z = Z @ b; p = 1 / (1 + np.exp(-z))
        ll = -np.sum(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))
        return (ll + l2 * np.sum(b[1:] ** 2)) / len(y), (Z.T @ (p - y) + 2 * l2 * np.r_[0, b[1:]]) / len(y)
    b = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
    return {"b": b, "mu": mu, "sd": sd}


def pred_logit(m, X):
    Z = np.nan_to_num((X - m["mu"]) / m["sd"], nan=0.0)
    return 1 / (1 + np.exp(-(m["b"][0] + Z @ m["b"][1:])))


def status_table(rows, y, t90=False):
    """Phase 0B baseline: P(played | final status) (T-24h) or | game-day roster (T-90m)."""
    key = (lambda r: r["_t90status"]) if t90 else (lambda r: r["_status"])
    tab = defaultdict(lambda: [0.0, 0.0])
    for r, v in zip(rows, y):
        tab[key(r)][0] += v; tab[key(r)][1] += 1
    base = float(np.mean(y))
    return {k: (a + base * 5) / (n + 5) for k, (a, n) in tab.items()}, base, key


def evaluate_block(rows, label, cols_t24, cols_t90):
    tr = [r for r in rows if C.TRAIN(r["s"], r["w"])]
    va = [r for r in rows if C.VALID(r["s"], r["w"])]
    dv = [r for r in rows if C.DEV(r["s"], r["w"])]
    y = lambda rr: np.array([r["y"] for r in rr], float)
    out = {"label": C.DEV_LABEL, "n_train": len(tr), "n_dev": len(dv)}
    preds = {}
    for tag, cols, t90 in (("T24", cols_t24, False), ("T90", cols_t90, True)):
        C.audit_fit(f"availability_{label}_{tag}", tr, va)
        tab, base, key = status_table(tr + va, y(tr + va), t90=t90)
        pb = np.array([tab.get(key(r), base) for r in dv])
        lg = fit_logit(C.matrix(tr + va, cols), y(tr + va))
        pl = pred_logit(lg, C.matrix(dv, cols))
        bst = C.fit_xgb(C.matrix(tr, cols), y(tr), C.matrix(va, cols), y(va), cols, "binary:logistic")
        px = C.xgb_pred(bst, C.matrix(dv, cols), cols)
        yd = y(dv); blocks = np.array([f"{r['s']}-{r['w']}" for r in dv])
        ll = lambda p: -(yd * np.log(np.clip(p, 1e-4, 1)) + (1 - yd) * np.log(np.clip(1 - p, 1e-4, 1)))
        res = {}
        for name, p in (("B0_status_lookup", pb), ("C1_logistic", pl), ("C2_xgb", px)):
            m = {"combined": C.binary(p, yd)}
            for t2, sel in (("2025", lambda r: r["s"] == 2025), ("2026_wk1_3", lambda r: r["s"] == 2026)):
                ix = np.array([sel(r) for r in dv])
                m[t2] = {k: v for k, v in C.binary(p[ix], yd[ix]).items() if k != "reliability"}
            imp, pv = C.block_boot(ll(p), ll(pb), blocks)
            m["vs_B0_logloss"] = {"improvement": imp, "p_not_better": pv}
            # reliability by designation and position
            m["by_status"] = {}
            for st in STATUSES:
                ix = np.array([r["_status"] == st for r in dv])
                if ix.sum() >= 20:
                    m["by_status"][st] = {"n": int(ix.sum()), "mean_p": round(float(p[ix].mean()), 4), "obs": round(float(yd[ix].mean()), 4),
                                          "logloss": round(float(ll(p)[ix].mean()), 4)}
            m["by_group"] = {}
            for g in sorted({r["_grp"] for r in dv}):
                ix = np.array([r["_grp"] == g for r in dv])
                m["by_group"][g] = {"n": int(ix.sum()), "mean_p": round(float(p[ix].mean()), 4), "obs": round(float(yd[ix].mean()), 4),
                                    "logloss": round(float(ll(p)[ix].mean()), 4)}
            res[name] = m
        best = min(("C1_logistic", "C2_xgb"), key=lambda n: res[n]["combined"]["logloss"])
        sel = best if res[best]["vs_B0_logloss"]["p_not_better"] < 0.10 and res[best]["vs_B0_logloss"]["improvement"] > 0 else "B0_status_lookup"
        res["selected"] = sel
        out[tag] = res
        allp = {"B0_status_lookup": np.array([tab.get(key(r), base) for r in rows]),
                "C1_logistic": pred_logit(lg, C.matrix(rows, cols)),
                "C2_xgb": C.xgb_pred(bst, C.matrix(rows, cols), cols)}
        preds[tag] = allp[sel]
    return out, preds


def snap_share_block(rows, cols):
    """Conditional snap share given active: baseline last-8 mean vs XGB quantiles."""
    act = [r for r in rows if r["y"] == 1 and r.get("snap_l8") is not None]
    tr = [r for r in act if C.TRAIN(r["s"], r["w"])]; va = [r for r in act if C.VALID(r["s"], r["w"])]
    dv = [r for r in act if C.DEV(r["s"], r["w"])]
    C.audit_fit("availability_conditional_snap_share", tr, va)
    y = lambda rr: np.array([r["y_snap"] for r in rr])
    # baseline: last-8 mean + training residual quantiles
    base = lambda rr: np.array([r["snap_l8"] for r in rr])
    resq = np.quantile(y(tr) - base(tr), C.QS)
    qb = np.clip(base(dv)[:, None] + resq[None, :], 0, 1)
    bq = {}
    for q in C.QS:
        bq[q] = C.fit_xgb(C.matrix(tr, cols), y(tr), C.matrix(va, cols), y(va), cols, "reg:quantileerror", {"quantile_alpha": q})
    qx = np.sort(np.clip(np.column_stack([C.xgb_pred(bq[q], C.matrix(dv, cols), cols) for q in C.QS]), 0, 1), axis=1)
    yd = y(dv); blocks = np.array([f"{r['s']}-{r['w']}" for r in dv])

    def score(Q):
        pin = np.mean([np.maximum(q * (yd - Q[:, i]), (q - 1) * (yd - Q[:, i])) for i, q in enumerate(C.QS)], axis=0)
        return {**C.cont(Q[:, 2], yd), "quantile_score": round(float(pin.mean()), 5),
                "cov80": C.coverage(Q[:, 0], Q[:, 4], yd), "cov50": C.coverage(Q[:, 1], Q[:, 3], yd)}, pin
    sb, pb_ = score(qb); sx, px_ = score(qx)
    imp, p = C.block_boot(np.abs(qx[:, 2] - yd), np.abs(qb[:, 2] - yd), blocks)
    impq, pq = C.block_boot(px_, pb_, blocks)
    sel = "C1_xgb_quantiles" if imp > 0 and impq > 0 and p < 0.10 and pq < 0.10 else "B0_last8_plus_residuals"
    return {"label": C.DEV_LABEL, "n_dev": len(dv), "B0_last8_plus_residuals": sb, "C1_xgb_quantiles": sx,
            "vs_B0": {"median_mae_improvement": imp, "p_not_better": p, "quantile_score_improvement": impq, "p_q_not_better": pq},
            "selected": sel}


def build_rows(units):
    off, dfn = [], []
    for u in units:
        s, w, team = u["key"]
        for r in u["players"]:
            F = player_features(u, r); F.update(t90_features(r)); r["_F"] = F
            F.update({"s": s, "w": w, "team": team, "gid": r["gid"], "y": 1.0 if r["actual"]["played"] else 0.0,
                      "y_snap": r["actual"]["snap_pct"], "_status": _status(r["inj"]), "_t90status": r["game_roster_T90"],
                      "_grp": r["pos"], "_ref": r, "_unit": u})
            off.append(F)
        for r in u["defenders"]:
            F = defender_features(u, r); F.update(t90_features(r)); r["_F"] = F
            F.update({"s": s, "w": w, "team": team, "pfr": r["pfr"], "y": 1.0 if r["actual"]["played"] else 0.0,
                      "y_snap": r["actual"]["pct"], "_status": _status(r["inj"]), "_t90status": r["game_roster_T90"],
                      "_grp": r["grp"], "_ref": r, "_unit": u})
            dfn.append(F)
    return off, dfn


def evaluate(units):
    off, dfn = build_rows(units)
    rep, preds = {}, {}
    rep["offense"], po = evaluate_block(off, "offense", OFF_T24, OFF_T24 + T90_EXTRA)
    rep["defense"], pd = evaluate_block(dfn, "defense", DEF_T24, DEF_T24 + T90_EXTRA)
    # depth-chart ablation (2025+ only: train 2025 wk1-9, evaluate 2025 wk10-18 + 2026 wk1-3)
    rep["offense_depth_ablation"] = depth_ablation(off)
    rep["offense_conditional_snap_share"] = snap_share_block(off, [c for c in OFF_T24 if c not in ("snap_n",)])
    rep["defense_conditional_snap_share"] = snap_share_block(dfn, DEF_T24)
    for tag in ("T24", "T90"):
        for r, p in zip(off, po[tag]):
            r["_ref"]["p_active_" + tag] = float(p)
        for r, p in zip(dfn, pd[tag]):
            r["_ref"]["p_active_" + tag] = float(p)
    rep["three_state"] = ("not built: no source labels 'limited'; any snap-share cutoff would be arbitrary. "
                          "Modelled as P(active) + conditional snap-share quantiles instead.")
    return rep, off, dfn


def depth_ablation(off):
    rows = [r for r in off if r["s"] >= 2025 and r.get("depth_listed") is not None]
    tr = [r for r in rows if r["s"] == 2025 and r["w"] <= 6]; va = [r for r in rows if r["s"] == 2025 and 7 <= r["w"] <= 9]
    dv = [r for r in rows if (r["s"] == 2025 and r["w"] >= 10) or r["s"] == 2026]
    y = lambda rr: np.array([r["y"] for r in rr])
    C.audit_fit("depth_ablation", tr, va)
    out = {"split": "train 2025 wk1-6, early-stop 2025 wk7-9, evaluate 2025 wk10-18 + 2026 wk1-3 (DEVELOPMENT)",
           "n_train": len(tr), "n_dev": len(dv)}
    yd = y(dv); blocks = np.array([f"{r['s']}-{r['w']}" for r in dv])
    res = {}
    for name, cols in (("without_depth", OFF_T24), ("with_depth", OFF_T24_DEPTH)):
        lg = fit_logit(C.matrix(tr + va, cols), y(tr + va))
        p = pred_logit(lg, C.matrix(dv, cols))
        res[name] = (p, C.binary(p, yd))
    ll = lambda p: -(yd * np.log(np.clip(p, 1e-4, 1)) + (1 - yd) * np.log(np.clip(1 - p, 1e-4, 1)))
    imp, p = C.block_boot(ll(res["with_depth"][0]), ll(res["without_depth"][0]), blocks)
    out.update({"logistic_" + k: {kk: vv for kk, vv in v[1].items() if kk != "reliability"} for k, v in res.items()})
    out["logistic_with_vs_without"] = {"logloss_improvement": imp, "p_not_better": p}
    # apples-to-apples with the selected family: XGBoost trained on 2023 - 2025 wk6 (depth missing before 2025),
    # early-stopped on 2025 wk7-9, evaluated on the same 2025 wk10+ / 2026 rows
    tr2 = [r for r in off if C.TRAIN(r["s"], r["w"]) or C.VALID(r["s"], r["w"]) or (r["s"] == 2025 and r["w"] <= 6)]
    C.audit_fit("depth_ablation", tr2, va)
    for r in tr2:
        if r["s"] < 2025:
            r["depth_rank"], r["depth_listed"] = None, None
    xres = {}
    for name, cols in (("without_depth", OFF_T24), ("with_depth", OFF_T24_DEPTH)):
        b = C.fit_xgb(C.matrix(tr2, cols), y(tr2), C.matrix(va, cols), y(va), cols, "binary:logistic")
        px = C.xgb_pred(b, C.matrix(dv, cols), cols)
        xres[name] = (px, C.binary(px, yd))
    imp2, p2 = C.block_boot(ll(xres["with_depth"][0]), ll(xres["without_depth"][0]), blocks)
    out.update({"xgb_" + k: {kk: vv for kk, vv in v[1].items() if kk != "reliability"} for k, v in xres.items()})
    out["xgb_with_vs_without"] = {"logloss_improvement": imp2, "p_not_better": p2}
    out["xgb_train_split"] = "2023 - 2025 wk6 (depth features missing before 2025), early stop 2025 wk7-9"
    return out
