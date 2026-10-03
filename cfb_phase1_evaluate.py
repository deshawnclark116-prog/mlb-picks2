"""
CFB_PHASE1_EVALUATE -- development comparison (D1-D4) of the first CFB Outcome Engine components against their baselines, exactly as registered in phase1_protocol.json (+ amendment 1).
  T1 team rush attempts | T2 team pass attempts | P1 player participation | O1 carry-share allocation (given the realized team carry total)
2025 (late confirmation), 2026 Weeks 1-4 and 2026 Week >= 5 are never read. Refuses to run unless the protocol + amendment + code are committed and clean.
  python cfb_phase1_evaluate.py --work DIR
"""
import argparse
import json
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy import optimize, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import cfb_phase1_common as C
import cfb_phase1_data as D
import cfb_phase1_role_state as RS
import cfb_phase1_team_environment as TE

REPO = Path(__file__).resolve().parent
OUT = REPO / "cfb_models" / "cfb_outcome_engine"
PROTOCOL, AMEND = OUT / "phase1_protocol.json", OUT / "phase1_protocol_amendment_1.json"
CODE = ["cfb_phase1_common.py", "cfb_phase1_data.py", "cfb_phase1_team_environment.py", "cfb_phase1_role_state.py", "cfb_phase1_evaluate.py", "tests/test_cfb_phase1.py"]
K_TEAM, K_PLAYER = 110, 90
KAPPA_BOUNDS = (0.5, 500.0)
LOGISTIC_C = 1.0
TH = {"rel_gain": 0.005, "nll": 0.005, "ks": 0.02, "ece": 0.01, "slice": 0.05, "min_team": 200, "min_player": 500}
P1_FEATURES = ["POS_QB", "POS_RB", "POS_WR", "POS_TE", "P_APPS_L3", "P_APPS_L5", "P_APPS_L12", "P_GAMES_SINCE_APP", "P_RETURNING", "P_TRANSFER_NEWCOMER", "P_CARRY_SHARE_L5", "P_REC_SHARE_L5", "P_ATT_SHARE_L5", "P_CARRY_SHARE_ACTIVE_L5", "P_REC_SHARE_ACTIVE_L5",
               "P_ATT_SHARE_ACTIVE_L5", "P_CARRY_SHARE_DECAY", "P_REC_SHARE_DECAY", "P_ATT_SHARE_DECAY", "TEAM_N_ACTIVE_CARRIERS_L5", "TEAM_N_ACTIVE_RECEIVERS_L5", "T_PLAYS_B0", "RATING_GAP", "ABS_RATING_GAP", "IS_HOME", "O_IS_FCS", "T_N_SEASON"]
O1_FEATURES = RS.ROLE_FEATURES + ["T_PASS_RATE_B0", "T_PLAYS_B0"]


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def committed_and_clean(path):
    rel = str(Path(path).resolve().relative_to(REPO))
    return bool(git("log", "--format=%H", "-1", "--", rel)) and git("status", "--porcelain", "--", rel) == ""


def jsn(o):
    if isinstance(o, dict):
        return {str(k): jsn(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsn(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def sel(rows, seasons):
    s = set(seasons)
    return [r for r in rows if r["season"] in s]


# ------------------------------------------------------------------ decision rules (pure, unit-tested)
def gate_decision(*, rel_gain, upper95, rel_secondary, calib_delta, slices_ok, kind):
    """kind='count': calib_delta = PIT-KS delta (<= 0.02); kind='binary': calib_delta = ECE delta (<= 0.01); rel_secondary = NLL (count) / Brier (binary) relative change (<= +0.5%)."""
    cal_lim = TH["ks"] if kind == "count" else TH["ece"]
    g = {"G1": {"pass": bool(rel_gain >= TH["rel_gain"] or upper95 < 0), "rel_gain": rel_gain, "boot_upper95": upper95}, "G2": {"pass": bool(rel_secondary <= TH["nll"]), "value": rel_secondary},
         "G3": {"pass": bool(calib_delta <= cal_lim), "value": calib_delta}, "G4": {"pass": bool(slices_ok)}}
    return g, all(v["pass"] for v in g.values())


# ------------------------------------------------------------------ count components (distribution scoring)
def score_count(model_id, pmf, y, keys, weeks):
    cr, nl = C.crps_rows(pmf, y), C.nll_rows(pmf, y)
    v = C.pit_v(model_id, keys); pit = C.randomized_pit(pmf, y, v)
    mean = C.pmf_mean(pmf); med = C.pmf_quantile(pmf, 0.5)
    cov = {}
    for c in (0.5, 0.8, 0.9):
        lo, hi = C.pmf_quantile(pmf, (1 - c) / 2), C.pmf_quantile(pmf, (1 + c) / 2)
        cov[str(int(c * 100))] = float(np.mean((y >= lo) & (y <= hi)))
    return {"crps": float(cr.mean()), "nll": float(nl.mean()), "pit_ks": C.ks_uniform(pit), "mean_bias": float((mean - y).mean()), "mae_median": float(np.abs(med - y).mean()), "rmse_mean": float(np.sqrt(((mean - y) ** 2).mean())),
            "coverage": cov, "mean_pred": float(mean.mean()), "mean_obs": float(y.mean()), "n": int(len(y))}, {"crps": cr, "nll": nl, "pit_ks": C.ks_uniform(pit)}


def team_slices(tab, ref_mean, train_median):
    return {"fbs_vs_fbs": tab["O_IS_FCS"] == 0, "fbs_vs_fcs": tab["O_IS_FCS"] == 1, "home": tab["IS_HOME"] == 1, "away": (tab["IS_HOME"] == 0) & (tab["IS_NEUTRAL"] == 0), "neutral": tab["IS_NEUTRAL"] == 1, "early(<=3 games)": tab["T_N_SEASON"] <= 3,
            "established": tab["T_N_SEASON"] > 3, "gap<100": tab["ABS_RATING_GAP"] < 100, "gap100-250": (tab["ABS_RATING_GAP"] >= 100) & (tab["ABS_RATING_GAP"] <= 250), "gap>250": tab["ABS_RATING_GAP"] > 250,
            "high_expected": ref_mean >= train_median, "low_expected": ref_mean < train_median, "low_coverage": (tab["T_N_SEASON"] <= 3) | (tab["O_RATING_N"] <= 3), "high_coverage": (tab["T_N_SEASON"] > 3) & (tab["O_RATING_N"] > 3)}


def run_team_component(rows, target, b0_mean, log):
    names = {"y_rush": "T1_team_rush_attempts", "y_pass": "T2_team_pass_attempts"}[target]
    pooled = {a: {"crps": [], "nll": [], "pit_ks": [], "sc": {}, "masks": {}} for a in ("B0", "C1")}
    pooled_w, pooled_ref = [], []
    folds = {}
    for fid, tr_s, va_s in C.DEV_FOLDS:
        tr, va = sel(rows, tr_s), sel(rows, [va_s])
        ta, va_a = TE.table_arrays(tr), TE.table_arrays(va)
        ytr = np.array([r[target] for r in tr], float); yva = np.array([r[target] for r in va], int)
        models = {"B0": TE.B0Count(b0_mean).fit(ta, ytr), "C1": TE.C1Count().fit(ta, ytr)}
        keys = [r["key"] for r in va]
        wk = np.array([C.week_index(r["season"], r["week"]) for r in va])
        med = float(np.median(models["B0"].mean(ta)))
        folds[fid] = {"n_train": len(tr), "n_validate": len(va), "models": {}, "hyper": {}}
        refm = models["B0"].mean(va_a)
        for a, m in models.items():
            mu = m.mean(va_a)
            pmf, sf = C.nb2_pmf_matrix(mu, m.alpha(), K_TEAM)
            summ, rr = score_count(f"{names}_{a}_{fid}", pmf, yva, keys, wk)
            summ["max_tail_mass"] = float(sf.max())
            folds[fid]["models"][a] = summ
            folds[fid]["hyper"][a] = {"alpha": m.alpha(), **{k: v for k, v in m.nb.items() if k != "alpha"}}
            p = pooled[a]; p["crps"].append(rr["crps"]); p["nll"].append(rr["nll"]); p["pit_ks"].append(rr["pit_ks"])
        pooled_w.append(wk)
        for k, mk in team_slices(va_a, refm, med).items():
            for a in pooled:
                pooled[a]["masks"].setdefault(k, []).append(mk)
        log(f"  {names} {fid} validate {va_s}: " + " ".join(f"{a} crps={folds[fid]['models'][a]['crps']:.4f} nll={folds[fid]['models'][a]['nll']:.4f}" for a in ("B0", "C1")))
    mean = {a: {"crps": float(np.mean([f["models"][a]["crps"] for f in folds.values()])), "nll": float(np.mean([f["models"][a]["nll"] for f in folds.values()])), "pit_ks": float(np.mean([f["models"][a]["pit_ks"] for f in folds.values()]))} for a in ("B0", "C1")}
    cr0, cr1 = np.concatenate(pooled["B0"]["crps"]), np.concatenate(pooled["C1"]["crps"]); wk = np.concatenate(pooled_w)
    bs = C.bootstrap_report(cr1 - cr0, wk, float(cr0.mean()))
    masks = {k: np.concatenate(v) for k, v in pooled["C1"]["masks"].items()}
    ok, det = C.slice_gate(cr1, cr0, masks, TH["min_team"], TH["slice"])
    g, passed = gate_decision(rel_gain=(mean["B0"]["crps"] - mean["C1"]["crps"]) / mean["B0"]["crps"], upper95=bs["one_sided_95_upper_bound"], rel_secondary=(mean["C1"]["nll"] - mean["B0"]["nll"]) / mean["B0"]["nll"],
                              calib_delta=mean["C1"]["pit_ks"] - mean["B0"]["pit_ks"], slices_ok=ok, kind="count")
    return {"component": names, "folds": folds, "mean_over_folds": mean, "bootstrap_pooled": bs, "slices": det, "gates": g, "challenger_promoted": passed, "retained": "C1" if passed else "B0"}


# ------------------------------------------------------------------ P1 participation (binary)
def ece(p, y, bins=10):
    order = np.argsort(p, kind="stable")
    return float(sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in np.array_split(order, bins) if len(b)))


def binary_scores(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return {"logloss": float(ll.mean()), "brier": float(np.mean((p - y) ** 2)), "ece": ece(p, y), "auc_diagnostic": float(roc_auc_score(y, p)), "mean_pred": float(p.mean()), "base_rate": float(y.mean()), "n": int(len(y))}, ll


def player_slices(tab, ref_mean, train_median):
    ret = tab["P_RETURNING"] == 1; trn = tab["P_TRANSFER_NEWCOMER"] == 1
    return {"QB": tab["POS_QB"] == 1, "RB": tab["POS_RB"] == 1, "WR": tab["POS_WR"] == 1, "TE": tab["POS_TE"] == 1, "returning": ret & ~trn, "transfer_newcomer": trn, "other": ~ret & ~trn, "early(<=3 games)": tab["T_N_SEASON"] <= 3, "established": tab["T_N_SEASON"] > 3,
            "home": tab["IS_HOME"] == 1, "away": (tab["IS_HOME"] == 0) & (tab["IS_NEUTRAL"] == 0), "neutral": tab["IS_NEUTRAL"] == 1, "high_expected": ref_mean >= train_median, "low_expected": ref_mean < train_median,
            "low_coverage": (tab["T_N_SEASON"] <= 3) | (tab["O_RATING_N"] <= 3), "high_coverage": (tab["T_N_SEASON"] > 3) & (tab["O_RATING_N"] > 3)}


def pos_idx(tab):
    return np.where(tab["POS_QB"] == 1, 0, np.where(tab["POS_RB"] == 1, 1, np.where(tab["POS_WR"] == 1, 2, 3)))


def p1_b0(tr, ytr, tab):
    pos = pos_idx(tr); pi = np.array([ytr[pos == k].mean() if (pos == k).any() else ytr.mean() for k in range(4)])
    return (tab["P_APPS_L5"] + 2.0 * pi[pos_idx(tab)]) / (5.0 + 2.0), pi


def p1_c1_matrix(tab):
    t = dict(tab)
    t["P_APPS_L3"] = tab["P_APPS_L3"] / 3.0; t["P_APPS_L5"] = tab["P_APPS_L5"] / 5.0; t["P_APPS_L12"] = tab["P_APPS_L12"] / 12.0
    return t


def run_participation(rows, log):
    pooled = {a: {"ll": [], "mask": {}} for a in ("B0", "C1")}; wks = []
    folds = {}
    for fid, tr_s, va_s in C.DEV_FOLDS:
        tr, va = sel(rows, tr_s), sel(rows, [va_s])
        ta, vt = RS.arrays(tr), RS.arrays(va)
        ytr = np.array([r["y_part"] for r in tr], float); yva = np.array([r["y_part"] for r in va], float)
        pb_tr, _ = p1_b0(ta, ytr, ta); pb, pi = p1_b0(ta, ytr, vt)
        names = P1_FEATURES
        prep = TE.Prep(names).fit(p1_c1_matrix(ta)); X = prep.transform(p1_c1_matrix(ta))
        lg = LogisticRegression(C=LOGISTIC_C, penalty="l2", solver="lbfgs", max_iter=3000).fit(X, ytr)
        pc = lg.predict_proba(prep.transform(p1_c1_matrix(vt)))[:, 1]
        folds[fid] = {"n_train": len(tr), "n_validate": len(va), "models": {}, "hyper": {"B0_position_participation_rates_QB_RB_WR_TE": pi, "C1_converged": bool(lg.n_iter_[0] < 3000)}}
        med = float(np.median(pb_tr)); wk = np.array([C.week_index(r["season"], r["week"]) for r in va]); wks.append(wk)
        for a, p in (("B0", pb), ("C1", pc)):
            s, ll = binary_scores(p, yva); folds[fid]["models"][a] = s; pooled[a]["ll"].append(ll)
        for k, mk in player_slices(vt, pb, med).items():
            for a in pooled:
                pooled[a]["mask"].setdefault(k, []).append(mk)
        log(f"  P1 {fid} validate {va_s}: " + " ".join(f"{a} logloss={folds[fid]['models'][a]['logloss']:.4f}" for a in ("B0", "C1")))
    mean = {a: {k: float(np.mean([f["models"][a][k] for f in folds.values()])) for k in ("logloss", "brier", "ece")} for a in ("B0", "C1")}
    l0, l1 = np.concatenate(pooled["B0"]["ll"]), np.concatenate(pooled["C1"]["ll"]); wk = np.concatenate(wks)
    bs = C.bootstrap_report(l1 - l0, wk, float(l0.mean()))
    masks = {k: np.concatenate(v) for k, v in pooled["C1"]["mask"].items()}
    ok, det = C.slice_gate(l1, l0, masks, TH["min_player"], TH["slice"])
    g, passed = gate_decision(rel_gain=(mean["B0"]["logloss"] - mean["C1"]["logloss"]) / mean["B0"]["logloss"], upper95=bs["one_sided_95_upper_bound"], rel_secondary=(mean["C1"]["brier"] - mean["B0"]["brier"]) / mean["B0"]["brier"],
                              calib_delta=mean["C1"]["ece"] - mean["B0"]["ece"], slices_ok=ok, kind="binary")
    return {"component": "P1_player_participation", "folds": folds, "mean_over_folds": mean, "bootstrap_pooled": bs, "slices": det, "gates": g, "challenger_promoted": passed, "retained": "C1" if passed else "B0"}


# ------------------------------------------------------------------ O1 carry-share allocation (beta-binomial)
def bb_pmf(N, s, kappa, K=K_PLAYER):
    s = np.clip(s, 1e-4, 1 - 1e-4)
    a = (s * kappa)[:, None]; b = ((1 - s) * kappa)[:, None]
    ks = np.arange(K + 1)[None, :]
    return stats.betabinom.pmf(ks, N[:, None].astype(int), a, b)


def fit_kappa(y, N, s):
    y = np.asarray(y); N = np.asarray(N).astype(int); s = np.clip(s, 1e-4, 1 - 1e-4)
    def nll(t):
        k = np.exp(t)
        return -float(stats.betabinom.logpmf(y, N, s * k, (1 - s) * k).sum())
    res = optimize.minimize_scalar(nll, bounds=(np.log(KAPPA_BOUNDS[0]), np.log(KAPPA_BOUNDS[1])), method="bounded", options={"xatol": 1e-6})
    k = float(np.exp(res.x))
    return {"kappa": k, "at_lower_bound": bool(abs(res.x - np.log(KAPPA_BOUNDS[0])) < 1e-3), "at_upper_bound": bool(abs(res.x - np.log(KAPPA_BOUNDS[1])) < 1e-3)}


def o1_b0_share(tab, pi_share):
    n5 = tab["P_APPS_L5"]; s = tab["P_CARRY_SHARE_L5"]
    s = np.where(np.isfinite(s), s, 0.0)
    return (s * n5 + pi_share[pos_idx(tab)] * 1.0) / (n5 + 1.0)


def run_carry_share(rows, log):
    pooled = {a: {"crps": [], "nll": [], "pit_ks": [], "mask": {}} for a in ("B0", "C1")}; wks = []
    folds = {}
    for fid, tr_s, va_s in C.DEV_FOLDS:
        tr = [r for r in sel(rows, tr_s) if r["team_carries_target"] > 0]; va = [r for r in sel(rows, [va_s]) if r["team_carries_target"] > 0]
        ta, vt = RS.arrays(tr), RS.arrays(va)
        ytr = np.array([r["y_carries"] for r in tr], int); yva = np.array([r["y_carries"] for r in va], int)
        Ntr = np.array([r["team_carries_target"] for r in tr]); Nva = np.array([r["team_carries_target"] for r in va])
        pos_tr = pos_idx(ta)
        pi_share = np.array([(ytr[pos_tr == k].sum() / Ntr[pos_tr == k].sum()) if (pos_tr == k).any() else 0.0 for k in range(4)])
        s_b0_tr, s_b0_va = o1_b0_share(ta, pi_share), o1_b0_share(vt, pi_share)
        prep = TE.Prep(O1_FEATURES).fit(ta); X = prep.transform(ta)
        Xrep = np.vstack([X, X]); lab = np.concatenate([np.ones(len(X)), np.zeros(len(X))]); w = np.concatenate([ytr.astype(float), (Ntr - ytr).astype(float)])
        keep = w > 0
        lg = LogisticRegression(C=LOGISTIC_C, penalty="l2", solver="lbfgs", max_iter=3000).fit(Xrep[keep], lab[keep], sample_weight=w[keep])
        s_c1_tr, s_c1_va = lg.predict_proba(X)[:, 1], lg.predict_proba(prep.transform(vt))[:, 1]
        kap = {"B0": {}, "C1": {}}
        for a, s_tr in (("B0", s_b0_tr), ("C1", s_c1_tr)):
            for k, nm in enumerate(("QB", "RB", "WR", "TE")):
                m = pos_tr == k
                kap[a][nm] = fit_kappa(ytr[m], Ntr[m], s_tr[m]) if m.sum() > 50 else {"kappa": 5.0, "fallback": True}
        pos_va = pos_idx(vt)
        folds[fid] = {"n_train": len(tr), "n_validate": len(va), "models": {}, "hyper": {"B0_pi_share_QB_RB_WR_TE": pi_share, "kappa": kap, "C1_converged": bool(lg.n_iter_[0] < 3000)}}
        keys = [r["key"] for r in va]; wk = np.array([C.week_index(r["season"], r["week"]) for r in va]); wks.append(wk)
        med = float(np.median(s_b0_tr * Ntr))
        for a, s_va in (("B0", s_b0_va), ("C1", s_c1_va)):
            pmf = np.zeros((len(va), K_PLAYER + 1))
            for k, nm in enumerate(("QB", "RB", "WR", "TE")):
                m = pos_va == k
                if m.any():
                    pmf[m] = bb_pmf(Nva[m], s_va[m], kap[a][nm]["kappa"])
            summ, rr = score_count(f"O1_{a}_{fid}", pmf, yva, keys, wk)
            summ["max_tail_mass"] = float(np.max(1 - pmf.sum(axis=1)))
            folds[fid]["models"][a] = summ
            p = pooled[a]; p["crps"].append(rr["crps"]); p["nll"].append(rr["nll"]); p["pit_ks"].append(rr["pit_ks"])
        for k, mk in player_slices(vt, s_b0_va * Nva, med).items():
            for a in pooled:
                pooled[a]["mask"].setdefault(k, []).append(mk)
        log(f"  O1 {fid} validate {va_s}: " + " ".join(f"{a} crps={folds[fid]['models'][a]['crps']:.4f} nll={folds[fid]['models'][a]['nll']:.4f}" for a in ("B0", "C1")))
    mean = {a: {"crps": float(np.mean([f["models"][a]["crps"] for f in folds.values()])), "nll": float(np.mean([f["models"][a]["nll"] for f in folds.values()])), "pit_ks": float(np.mean([f["models"][a]["pit_ks"] for f in folds.values()]))} for a in ("B0", "C1")}
    c0, c1 = np.concatenate(pooled["B0"]["crps"]), np.concatenate(pooled["C1"]["crps"]); wk = np.concatenate(wks)
    bs = C.bootstrap_report(c1 - c0, wk, float(c0.mean()))
    masks = {k: np.concatenate(v) for k, v in pooled["C1"]["mask"].items()}
    ok, det = C.slice_gate(c1, c0, masks, TH["min_player"], TH["slice"])
    g, passed = gate_decision(rel_gain=(mean["B0"]["crps"] - mean["C1"]["crps"]) / mean["B0"]["crps"], upper95=bs["one_sided_95_upper_bound"], rel_secondary=(mean["C1"]["nll"] - mean["B0"]["nll"]) / mean["B0"]["nll"],
                              calib_delta=mean["C1"]["pit_ks"] - mean["B0"]["pit_ks"], slices_ok=ok, kind="count")
    return {"component": "O1_carry_share_allocation", "folds": folds, "mean_over_folds": mean, "bootstrap_pooled": bs, "slices": det, "gates": g, "challenger_promoted": passed, "retained": "C1" if passed else "B0"}


# ------------------------------------------------------------------ driver
def load_tables(work):
    p = Path(work) / "phase1_tables.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    tg, pg = D.load_frozen()
    team_rows = TE.build_team_table(tg)
    cand, cov = D.build_candidates(tg, pg)
    prow = RS.build_feature_table(tg, pg, cand)
    pickle.dump((team_rows, prow, cov), open(p, "wb"), protocol=4)
    return team_rows, prow, cov


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); a = ap.parse_args()
    Path(a.work).mkdir(parents=True, exist_ok=True)
    assert committed_and_clean(PROTOCOL) and committed_and_clean(AMEND), "protocol + amendment must be committed before any performance is computed"
    assert all(committed_and_clean(REPO / c) for c in CODE), "code and tests must be committed (clean) before any performance is computed"
    assert not (OUT / "phase1_dev_results.json").exists(), "development already scored"
    team_rows, prow, cov = load_tables(a.work)
    assert max(r["season"] for r in team_rows + prow) <= 2025 and all(r["season"] <= 2024 or False for r in sel(team_rows, [2019, 2020, 2021, 2022, 2023, 2024]))
    t0 = time.time()
    res = {"protocol_sha": git("log", "--format=%H", "-1", "--", str(PROTOCOL.relative_to(REPO))), "amendment_sha": git("log", "--format=%H", "-1", "--", str(AMEND.relative_to(REPO))), "code_head": git("rev-parse", "HEAD"),
           "periods": "development D1-D4 only (validate 2021-2024); 2025 / 2026 untouched", "candidate_universe_coverage": cov, "components": {}}
    res["components"]["T1_team_rush_attempts"] = run_team_component(team_rows, "y_rush", "T_RUSH_B0", print)
    res["components"]["T2_team_pass_attempts"] = run_team_component(team_rows, "y_pass", "T_PASS_B0", print)
    res["components"]["P1_player_participation"] = run_participation(prow, print)
    res["components"]["O1_carry_share_allocation"] = run_carry_share(prow, print)
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "phase1_dev_results.json").write_text(json.dumps(jsn(res), indent=1, sort_keys=True) + "\n")
    print(json.dumps(jsn({k: {"retained": v["retained"], "promoted": v["challenger_promoted"], "gates": {g: x["pass"] for g, x in v["gates"].items()}, "mean": v["mean_over_folds"]} for k, v in res["components"].items()}), indent=1))


if __name__ == "__main__":
    main()
