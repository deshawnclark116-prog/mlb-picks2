"""
NFL_PHASE1C_ADJUDICATE  (Phase 1C step 0; burned data only; rule recorded in adjudication_rule.json BEFORE any result)

Rolling-origin (strictly chronological) model-selection study for every Phase 1B component. See adjudication_rule.json.

  python -u nfl_phase1c_adjudicate.py --records records_full.pkl --out DIR  rush rec_air rec_catch ...
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

import nfl_phase1_defense_events as DE
import nfl_phase1_efficiency as F
import nfl_phase1_event_models as EM
import nfl_phase1_passing_efficiency as PA
import nfl_phase1_receiving_efficiency as RC
import nfl_phase1_rushing_efficiency as RU
import nfl_phase1b_data as B

FOLDS = {"F1": (202401, 202406), "F2": (202407, 202412), "F3": (202413, 202418), "F4": (202501, 202506),
         "F5": (202507, 202512), "F6": (202513, 202518), "F7": (202601, 202603)}
LEVELS = list(F.NESTED)
MATERIAL = 0.0005            # 0.05% of the kept candidate's mean score
MIN_FOLDS_WIN = 5
KAPPAS = (5, 10, 20, 40, 80, 160, 1e9)


# ------------------------------------------------------------------ component specs
def mask_warmup(Rs, drecs):
    """Amendment 1: 2022 week 1 has no as-of history (features identically zero) -> excluded from every fit and test."""
    for R in Rs.values():
        R.active = R.active & ~((R.s == 2022) & (R.w == 1))
    for r in drecs:
        if r["s"] == 2022 and r["w"] == 1:
            r["extra"] = True
    return Rs, drecs


def spec_for(name, Rs, drecs):
    if name == "rush":
        R = Rs["rush"]
        return dict(kind="pmf", R=R, cnt=R.cnt["y"], keys=("pl", "pp", "lg"), vals=RU.BIN_VALS, phi=RU.PHI_SINGLE, extra_fn=_rush_extra, rush=True)
    if name in ("rec_air", "pass_air"):
        R = Rs["rec"] if name == "rec_air" else Rs["pass"]
        return dict(kind="pmf", R=R, cnt=R.cnt["air"], keys=("air_pl", "air_pp", "air_lg"), vals=RC.AIR_V, phi=RC.PHI_AIR,
                    extra_fn=_generic_extra(lambda P0: P0[:, RC.AIR_V >= 20].sum(1), 1, (6,)))
    if name in ("rec_yac", "pass_yac"):
        R = Rs["rec"] if name == "rec_yac" else Rs["pass"]
        S = EM.Stacked(R, 4)
        S.base = {f"yac_{l}": np.concatenate([R.base[f"yac_{l}"][:, :, b * B.K_Y:(b + 1) * B.K_Y] for b in range(4)]) for l in ("pl", "pp", "lg")}
        cnt = np.concatenate([R.cnt["yac"][:, b * B.K_Y:(b + 1) * B.K_Y] for b in range(4)])
        return dict(kind="pmf", R=S, cnt=cnt, keys=("yac_pl", "yac_pp", "yac_lg"), vals=RC.YAC_V, phi=RC.PHI_YAC,
                    extra_fn=_generic_extra(lambda P0: P0[:, RC.YAC_V >= 15].sum(1), 2, (0,)))
    if name in ("rec_catch", "pass_completion"):
        R = Rs["rec"] if name == "rec_catch" else Rs["pass"]
        S = EM.Stacked(R, 4)
        cnt2, bp = [], {"pl": [], "pp": [], "lg": []}
        for b in range(4):
            tgt, cat = R.cnt["cb"][:, 2 * b], R.cnt["cb"][:, 2 * b + 1]
            cnt2.append(np.column_stack([tgt - cat, cat]))
            for l in ("pl", "pp", "lg"):
                a = R.base[f"cb_{l}"]
                bp[l].append(np.stack([a[:, :, 2 * b] - a[:, :, 2 * b + 1], a[:, :, 2 * b + 1]], axis=2))
        S.base = {k: np.concatenate(v) for k, v in bp.items()}
        return dict(kind="bin", R=S, cnt=np.concatenate(cnt2), keys=("pl", "pp", "lg"), vals=np.array([0.0, 1.0]), phi=np.array([[0.0], [1.0]]),
                    extra_fn=_bin_extra(0, (0, 7)))
    if name in ("pass_sack", "pass_int", "pass_td"):
        R = Rs["pass"]; num, den, oi, si = {"pass_sack": (1, 0, 3, (7, 6)), "pass_int": (4, 2, 4, (7,)), "pass_td": (5, 2, 5, (6,))}[name]
        S = EM.Stacked(R, 1)
        pt = R.cnt["pt"]
        S.base = {l: np.stack([R.base[f"pt_{l}"][:, :, den] - R.base[f"pt_{l}"][:, :, num], R.base[f"pt_{l}"][:, :, num]], axis=2) for l in ("pl", "pp", "lg")}
        return dict(kind="bin", R=S, cnt=np.column_stack([pt[:, den] - pt[:, num], pt[:, num]]), keys=("pl", "pp", "lg"), vals=np.array([0.0, 1.0]),
                    phi=np.array([[0.0], [1.0]]), extra_fn=_bin_extra(oi, si))
    if name in ("rush_td", "rec_td"):
        R = Rs["rush"] if name == "rush_td" else Rs["rec"]
        key = "ev" if name == "rush_td" else "tev"
        S = EM.Stacked(R, 2)
        ev = R.cnt[key]
        cnt2 = np.concatenate([np.column_stack([ev[:, 0] - ev[:, 1], ev[:, 1]]), np.column_stack([ev[:, 2] - ev[:, 3], ev[:, 3]])])
        bk = {"ev": ("evp", "evpos", "evlg"), "tev": ("tv_pl", "tv_pp", "tv_lg")}[key]
        bp = {}
        for l, k in zip(("pl", "pp", "lg"), bk):
            a = R.base[k]
            bp[l] = np.concatenate([np.stack([a[:, :, 0] - a[:, :, 1], a[:, :, 1]], axis=2), np.stack([a[:, :, 2] - a[:, :, 3], a[:, :, 3]], axis=2)])
        S.base = bp
        return dict(kind="bin", R=S, cnt=cnt2, keys=("pl", "pp", "lg"), vals=np.array([0.0, 1.0]), phi=np.array([[0.0], [1.0]]),
                    extra_fn=_bin_extra(0, (0,) if name == "rush_td" else (0, 7)))
    raise KeyError(name)


def _cols(R, tr, nm, i):
    x = R.fam[nm][:, i].astype(float); m = np.nanmean(x[tr]); return np.where(np.isnan(x), m, x)


def _generic_extra(tend_fn, opp_i, scheme_i):
    def f(P0, R, tr):
        t = tend_fn(P0); t = t - t[tr].mean()
        return np.column_stack([t * _cols(R, tr, "opp", opp_i)] + [t * _cols(R, tr, "scheme", j) for j in scheme_i])
    return f


def _bin_extra(opp_i, scheme_i):
    def f(P0, R, tr):
        p = np.clip(P0[:, 1], 1e-6, 1 - 1e-6); t = np.log(p / (1 - p)); t = t - t[tr].mean()
        return np.column_stack([t * _cols(R, tr, "opp", opp_i)] + [t * _cols(R, tr, "scheme", j) for j in scheme_i])
    return f


def _rush_extra(P0, R, tr):
    t_expl = P0[:, RU.BIN_VALS >= 15].sum(1); t_neg = P0[:, RU.BIN_VALS < 0].sum(1)
    ce, cn = t_expl - t_expl[tr].mean(), t_neg - t_neg[tr].mean()
    return np.column_stack([ce * _cols(R, tr, "opp", 0), ce * _cols(R, tr, "scheme", 0), cn * _cols(R, tr, "scheme", 1)])


# ------------------------------------------------------------------ fold machinery
def fold_masks(t, act, start, end):
    tr_all = act & (t < start)
    te = act & (t >= start) & (t <= end)
    weeks = np.unique(t[tr_all])
    cut = weeks[int(len(weeks) * 0.75)]
    return tr_all & (t < cut), tr_all & (t >= cut), te


def run_pmf_like(spec, want_two_process=False):
    R, cnt, vals = spec["R"], spec["cnt"], spec["vals"]
    t = R.s * 100 + R.w; act = R.active
    out = {"levels": {l: {} for l in LEVELS}, "crps": {l: {} for l in LEVELS}, "hier": {}, "n_test_opps": {}}
    if want_two_process:
        for T in (15, 20, 25):
            for l in LEVELS:
                out["levels"][f"two{T}_{l}"] = {}
    for fn, (a, b) in FOLDS.items():
        tr, va, te = fold_masks(t, act, a, b)
        if te.sum() == 0:
            continue
        tuned, _ = F.tune_hier(R, cnt, *spec["keys"], tr, va, kps=KAPPAS, kpos=(20, 100, 500))
        P0 = F.hier_base(R, *spec["keys"], tuned["gamma_idx"], tuned["kappa_player"], tuned["kappa_pos"])
        logP0 = np.log(P0)
        ex = spec["extra_fn"](P0, R, tr | va)
        out["hier"][fn] = {k: tuned[k] for k in ("gamma_idx", "kappa_player", "kappa_pos")}
        n_te = cnt[te].sum(); out["n_test_opps"][fn] = float(n_te)
        for lvl, fams in F.NESTED.items():
            X = F.impute_stack(R, fams, tr | va, ex)
            P, _ = F.fit_family(logP0, cnt, X, spec["phi"], tr, va, te)
            out["levels"][lvl][fn] = float(F.logscore(P[te], cnt[te]).sum() / n_te)
            if spec["kind"] == "pmf":
                out["crps"][lvl][fn] = float(F.ordinal_crps_records(P[te], cnt[te], vals).sum() / n_te)
            if want_two_process:
                for T in (15, 20, 25):
                    fn2 = RU.two_process_fn(P0, cnt, tr, va, te, T)
                    P2, _ = fn2(X, lvl)
                    out["levels"][f"two{T}_{lvl}"][fn] = float(F.logscore(P2[te], cnt[te]).sum() / n_te)
    return out


def _vec_rate(base, gi, kp, kq, floor=1e-4):
    lg, pos, pl = base["lg"][:, gi, :], base["pos"][:, gi, :], base["p"][:, gi, :]
    r_lg = np.maximum(lg[:, 1] / np.maximum(lg[:, 0], 1.0), floor)
    r_pos = (pos[:, 1] + kq * r_lg) / (pos[:, 0] + kq)
    return (pl[:, 1] + kp * r_pos) / (pl[:, 0] + kp)


def run_defense(target, drecs):
    y = np.array([r[target] for r in drecs], float); snaps = np.array([r["snaps"] for r in drecs], float)
    act = np.array([not r.get("extra", False) for r in drecs], bool)
    s = np.array([r["s"] for r in drecs]); w = np.array([r["w"] for r in drecs]); t = s * 100 + w
    base = {lvl: np.stack([r["base"][target][lvl] for r in drecs]) for lvl in ("p", "pos", "lg")}
    fam = {f: np.array([r["fam"][f] for r in drecs], float) for f in ("team", "opp", "scheme", "personnel")}
    out = {"levels": {l: {} for l in LEVELS}, "hier": {}, "n_test": {}}
    nll = lambda y_, mu: -(y_ * np.log(np.maximum(mu, 1e-12)) - mu - __import__("scipy.special", fromlist=["gammaln"]).gammaln(y_ + 1))
    for fn, (a, b) in FOLDS.items():
        tr, va, te = fold_masks(t, act, a, b)
        if te.sum() == 0:
            continue
        best = None
        for gi in (0, 1, 2):
            for kp in DE.KAPPA_GRID:
                for kq in (300.0, 1500.0):
                    mu = _vec_rate(base, gi, kp, kq) * snaps
                    v = nll(y[va], mu[va]).mean()
                    if best is None or v < best[0]:
                        best = (v, gi, kp, kq)
        _, gi, kp, kq = best
        rate0 = _vec_rate(base, gi, kp, kq)
        out["hier"][fn] = {"gamma_idx": gi, "kappa_player": kp, "kappa_pos": kq}
        z = np.log(np.maximum(rate0, 1e-6)); z = z - z[tr].mean()
        out["n_test"][fn] = int(te.sum())
        for lvl, fl in F.NESTED.items():
            blocks = [fam[f] for f in fl if f != "interaction"]
            if "interaction" in fl:
                sc0 = fam["scheme"][:, 0]; sc0 = np.where(np.isnan(sc0), np.nanmean(sc0[tr]), sc0)
                blocks.append(np.column_stack([z * fam["opp"][:, 0], z * sc0]))
            if not blocks:
                mu = rate0 * snaps
            else:
                X = np.column_stack(blocks); m_ = np.nanmean(X[tr | va], 0); ix = np.where(np.isnan(X)); X[ix] = m_[ix[1]]
                bl = None
                for l2 in (100.0, 1000.0, 10000.0):
                    m = DE.PoissonTilt(l2).fit(y[tr], (rate0 * snaps)[tr], X[tr])
                    v = nll(y[va], m.predict((rate0 * snaps)[va], X[va])).mean()
                    if bl is None or v < bl[0]:
                        bl = (v, l2)
                m = DE.PoissonTilt(bl[1]).fit(y[tr | va], (rate0 * snaps)[tr | va], X[tr | va])
                mu = m.predict(rate0 * snaps, X)
            out["levels"][lvl][fn] = float(nll(y[te], mu[te]).mean())
    return out


# ------------------------------------------------------------------ adjudication rule
def adjudicate(scores, ladder):
    """scores: {candidate: {fold: score (lower better)}}; ladder: candidates in complexity order. Rule from adjudication_rule.json."""
    folds = sorted(set.intersection(*[set(scores[c]) for c in ladder]))
    mean = lambda c: float(np.mean([scores[c][f] for f in folds]))
    kept = ladder[0]; steps = []
    for c in ladder[1:]:
        d = np.array([scores[kept][f] - scores[c][f] for f in folds])          # positive = c better
        thr = MATERIAL * mean(kept)
        wins = int((d > 0).sum())
        ok = d.mean() >= thr and wins >= MIN_FOLDS_WIN and d.min() >= -thr
        steps.append({"candidate": c, "vs": kept, "mean_improvement": float(d.mean()), "materiality_threshold": float(thr), "folds_won": wins, "n_folds": len(folds),
                      "worst_fold_delta": float(d.min()), "per_fold_delta": {f: float(x) for f, x in zip(folds, d)}, "replaces": bool(ok)})
        if ok:
            kept = c
    return {"selected": kept, "mean_score_by_candidate": {c: mean(c) for c in ladder}, "folds": folds, "steps": steps}


def assemble(adj_dir, out_json, out_md, run1_dir=None):
    """Merge per-group outputs into adjudication_results.json + a markdown table; also keep run 1 (2022 wk1 included) for transparency."""
    res = {}
    for f in sorted(Path(adj_dir).glob("adj_*.json")):
        res.update(json.load(open(f)))
    final = {}
    for k, e in res.items():
        final[k] = {"selected": e["adjudication"]["selected"], "structure": e.get("structure_adjudication", {}).get("selected"),
                    "mean_score_by_candidate": e["adjudication"]["mean_score_by_candidate"], "steps": e["adjudication"]["steps"],
                    "structure_steps": e.get("structure_adjudication", {}).get("steps"), "hier_by_fold": e["raw"].get("hier"),
                    "fold_scores": {l: v for l, v in e["raw"]["levels"].items()}, "crps_secondary_mean": e.get("crps_secondary_mean")}
    Path(out_json).write_text(json.dumps({"rule": "adjudication_rule.json (with amendment_1)", "final": final}, indent=1, default=float))
    if run1_dir:
        r1 = {}
        for f in sorted(Path(run1_dir).glob("adj_*.json")):
            r1.update(json.load(open(f)))
        Path(out_json).with_name("adjudication_run1_partial.json").write_text(json.dumps(
            {"note": "run 1 INCLUDED 2022 week-1 rows with all-zero as-of features; partial (interrupted); superseded by amendment 1; kept for transparency",
             "selected": {k: {"selected": e["adjudication"]["selected"], "mean": e["adjudication"]["mean_score_by_candidate"]} for k, e in r1.items()}}, indent=1, default=float))
    L = ["| component | selected | B0 mean score | selected mean score | decisive step(s) |", "|---|---|---|---|---|"]
    for k, e in final.items():
        sel = e["selected"]; ms = e["mean_score_by_candidate"]
        steps = "; ".join(f"{s['candidate']} vs {s['vs']}: +{s['mean_improvement']:.5f} (thr {s['materiality_threshold']:.5f}), {s['folds_won']}/{s['n_folds']} folds, worst {s['worst_fold_delta']:+.5f}, {'ADOPT' if s['replaces'] else 'keep simpler'}"
                          for s in e["steps"] if s["replaces"] or s["folds_won"] >= 5)
        L.append(f"| {k} | {sel}{' / ' + e['structure'] if e.get('structure') else ''} | {ms['B0']:.5f} | {ms[sel]:.5f} | {steps} |")
    Path(out_md).write_text("\n".join(L) + "\n")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("components", nargs="+")
    a = ap.parse_args()
    rec = pickle.load(open(a.records, "rb"))
    Rs, drecs = mask_warmup(rec["Rs"], rec["drecs"])
    res = {}
    for name in a.components:
        t0 = time.time()
        if name.startswith("def_"):
            raw = run_defense(name[4:], drecs)
        else:
            raw = run_pmf_like(spec_for(name, Rs, drecs), want_two_process=(name == "rush"))
        entry = {"raw": raw, "adjudication": adjudicate({l: raw["levels"][l] for l in LEVELS}, LEVELS)}
        if name == "rush":
            k = entry["adjudication"]["selected"]
            cands = [k] + [f"two{T}_{k}" for T in (15, 20, 25)]
            entry["structure_adjudication"] = adjudicate({c: raw["levels"][c] for c in cands}, cands)
        if raw.get("crps") and raw["crps"].get("B0"):
            entry["crps_secondary_mean"] = {l: float(np.mean(list(v.values()))) for l, v in raw["crps"].items() if v}
        res[name] = entry
        Path(a.out).mkdir(parents=True, exist_ok=True)
        json.dump(res, open(Path(a.out) / f"adj_{'_'.join(a.components)}.json", "w"), default=float)
        print(name, entry["adjudication"]["selected"], entry.get("structure_adjudication", {}).get("selected"), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
