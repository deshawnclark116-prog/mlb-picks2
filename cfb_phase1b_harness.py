"""
CFB_PHASE1B_HARNESS -- generic development-experiment harness for the Phase 1B core-engine components (burned development 2019-2024 only; 2025 is never loaded: `assert_dev_seasons`).
A component experiment is an ordered list of variants (simplest first). Each variant: fit_predict(train_rows, val_rows) -> pmf[n, K+1] (count components) or p[n] (binary). The harness scores every variant on D1-D4 (validate 2021-2024), runs the Phase 1 promotion gates of each
variant against the CURRENT incumbent in complexity order, and returns a complete record (also written to the failed-experiment registry for every rejected variant).
"""
import numpy as np

import cfb_phase1_common as C

DEV_MAX_SEASON = 2024
TH = {"rel_gain": 0.005, "nll": 0.005, "ks": 0.02, "ece": 0.01, "slice": 0.05, "min_player": 500, "min_team": 200}


def assert_dev_seasons(rows):
    mx = max(r["season"] for r in rows)
    if mx > DEV_MAX_SEASON:
        raise RuntimeError(f"development code may not load season {mx} (2025 is reserved for the one integrated confirmation)")


def sel(rows, seasons):
    s = set(seasons)
    return [r for r in rows if r["season"] in s]


def count_metrics(model_id, pmf, y, keys):
    y = np.asarray(y).astype(int)
    pmf = np.asarray(pmf)
    cr, nl = C.crps_rows(pmf, y), C.nll_rows(pmf, y)
    pit = C.randomized_pit(pmf, y, C.pit_v(model_id, keys))
    mean = C.pmf_mean(pmf)
    cov = {}
    for c in (0.5, 0.8, 0.9):
        lo, hi = C.pmf_quantile(pmf, (1 - c) / 2), C.pmf_quantile(pmf, (1 + c) / 2)
        cov[str(int(c * 100))] = float(np.mean((y >= lo) & (y <= hi)))
    summ = {"crps": float(cr.mean()), "nll": float(nl.mean()), "pit_ks": C.ks_uniform(pit), "mean_pred": float(mean.mean()), "mean_obs": float(y.mean()), "rel_mean_bias": float((mean.mean() - y.mean()) / max(y.mean(), 1e-9)),
            "rmse_mean": float(np.sqrt(((mean - y) ** 2).mean())), "mae_median": float(np.abs(C.pmf_quantile(pmf, 0.5) - y).mean()), "coverage": cov, "n": int(len(y)), "max_tail_mass": float(np.max(1 - pmf.sum(axis=1)))}
    return summ, cr, nl


def binary_metrics(p, y):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9); y = np.asarray(y, float)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    order = np.argsort(p, kind="stable")
    ece = float(sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in np.array_split(order, 10) if len(b)))
    return {"logloss": float(ll.mean()), "brier": float(np.mean((p - y) ** 2)), "ece": ece, "mean_pred": float(p.mean()), "base_rate": float(y.mean()), "n": int(len(y))}, ll


def run_experiment(name, rows, y_fn, variants, slice_fn, kind="count", min_rows=None, log=print, keys_fn=None):
    """variants: list of (variant_name, fit_predict). Returns the full experiment record."""
    assert_dev_seasons(rows)
    min_rows = min_rows or TH["min_player"]
    per = {v: {"loss": [], "nll": [], "ks": [], "masks": {}, "folds": {}, "wk": []} for v, _ in variants}
    for fid, tr_s, va_s in C.DEV_FOLDS:
        tr, va = sel(rows, tr_s), sel(rows, [va_s])
        y = np.asarray(y_fn(va)); keys = [r["key"] for r in va] if keys_fn is None else keys_fn(va)
        wk = np.array([C.week_index(r["season"], r["week"]) for r in va])
        masks = slice_fn(va, tr)
        for v, fp in variants:
            out = fp(tr, va)
            if kind == "count":
                s, cr, nl = out.score(f"{name}_{v}_{fid}", y, keys) if hasattr(out, "score") else count_metrics(f"{name}_{v}_{fid}", out, y, keys); loss = cr; sec = nl
            else:
                s, ll = binary_metrics(out, y); loss = ll; sec = np.array([s["brier"]])
            d = per[v]; d["folds"][fid] = s; d["loss"].append(loss); d["nll"].append(sec if kind == "count" else s["brier"]); d["ks"].append(s["pit_ks"] if kind == "count" else s["ece"]); d["wk"].append(wk)
            for k, m in masks.items():
                d["masks"].setdefault(k, []).append(m)
        log(f"  {name} {fid}: " + " ".join(f"{v}={per[v]['folds'][fid]['crps' if kind == 'count' else 'logloss']:.4f}" for v, _ in variants))
    key = "crps" if kind == "count" else "logloss"
    sec_key = "nll" if kind == "count" else "brier"
    cal_key = "pit_ks" if kind == "count" else "ece"
    mean = {v: {key: float(np.mean([f[key] for f in per[v]["folds"].values()])), sec_key: float(np.mean([f[sec_key] for f in per[v]["folds"].values()])), cal_key: float(np.mean([f[cal_key] for f in per[v]["folds"].values()]))} for v, _ in variants}
    def gates(ch, inc):
        l_ch, l_inc = np.concatenate(per[ch]["loss"]), np.concatenate(per[inc]["loss"]); wk = np.concatenate(per[ch]["wk"])
        bs = C.bootstrap_report(l_ch - l_inc, wk, float(l_inc.mean()))
        masks = {k: np.concatenate(v) for k, v in per[ch]["masks"].items()}
        ok, det = C.slice_gate(l_ch, l_inc, masks, min_rows, TH["slice"])
        rel = (mean[inc][key] - mean[ch][key]) / mean[inc][key]
        g = {"G1": {"pass": bool(rel >= TH["rel_gain"] or bs["one_sided_95_upper_bound"] < 0), "rel_gain": rel, "boot_upper95": bs["one_sided_95_upper_bound"]},
             "G2": {"pass": bool((mean[ch][sec_key] - mean[inc][sec_key]) / mean[inc][sec_key] <= TH["nll"]), "value": (mean[ch][sec_key] - mean[inc][sec_key]) / mean[inc][sec_key]},
             "G3": {"pass": bool(mean[ch][cal_key] - mean[inc][cal_key] <= (TH["ks"] if kind == "count" else TH["ece"])), "value": mean[ch][cal_key] - mean[inc][cal_key]}, "G4": {"pass": bool(ok)}}
        return {"gates": g, "promoted": all(x["pass"] for x in g.values()), "bootstrap": bs, "slices": det}
    incumbent, history = variants[0][0], []
    for v, _ in variants[1:]:
        r = gates(v, incumbent); history.append({"challenger": v, "incumbent": incumbent, **r})
        if r["promoted"]:
            incumbent = v
    return {"experiment": name, "kind": kind, "variants": [v for v, _ in variants], "fold_summaries": {v: per[v]["folds"] for v, _ in variants}, "mean_over_folds": mean, "promotion_history": history, "selected": incumbent,
            "best_by_primary_loss": min(mean, key=lambda v: mean[v][key]), "rejected": [v for v, _ in variants if v != incumbent and v != variants[0][0]]}


# ------------------------------------------------------------------ single-split scoring from a FROZEN bundle (used by the ONE 2025 integrated confirmation)
PASS = {"primary_worse": 0.005, "secondary_worse": 0.005, "abs_rel_bias": 0.05, "coverage_error_vs_baseline": 0.05, "slice_worse": 0.10}


def score_split(name, tr, va, y_fn, variants, baseline, selected, slice_fn, kind="count", min_rows=None, keys_fn=None):
    """Fit every variant on `tr`, score on `va` ONCE; evaluate the registered component pass rule of `selected` against `baseline` (phase1b_core_engine_protocol.json 'component_pass_rule' + development amendment 3 coverage convention)."""
    min_rows = min_rows or TH["min_player"]
    y = np.asarray(y_fn(va)); keys = [r["key"] for r in va] if keys_fn is None else keys_fn(va)
    wk = np.array([C.week_index(r["season"], r["week"]) for r in va]); masks = slice_fn(va, tr)
    res, loss = {}, {}
    for v, fp in variants:
        out = fp(tr, va)
        if kind == "count":
            s, cr, nl = out.score(f"{name}_{v}", y, keys) if hasattr(out, "score") else count_metrics(f"{name}_{v}", out, y, keys); loss[v] = (cr, nl)
        else:
            s, ll = binary_metrics(out, y); loss[v] = (ll, None)
        res[v] = s
    key = "crps" if kind == "count" else "logloss"; sec = "nll" if kind == "count" else "brier"
    b, s_ = res[baseline], res[selected]
    primary = (s_[key] - b[key]) / b[key]; secondary = (s_[sec] - b[sec]) / b[sec]
    if kind == "count":
        bias = abs(s_["rel_mean_bias"]); cov_ok = all(abs(s_["coverage"][c] - int(c) / 100) - abs(b["coverage"][c] - int(c) / 100) <= PASS["coverage_error_vs_baseline"] for c in ("80", "90"))
    else:
        bias = abs(s_["mean_pred"] - s_["base_rate"]) / max(s_["base_rate"], 1e-9); cov_ok = True
    ok_sl, det = C.slice_gate(loss[selected][0], loss[baseline][0], masks, min_rows, PASS["slice_worse"])
    bs = C.bootstrap_report(loss[selected][0] - loss[baseline][0], wk, float(loss[baseline][0].mean()))
    rule = {"primary_not_worse_0.5pct": {"pass": bool(primary <= PASS["primary_worse"]), "value": primary}, "secondary_not_worse_0.5pct": {"pass": bool(secondary <= PASS["secondary_worse"]), "value": secondary},
            "abs_rel_mean_bias_le_5pct": {"pass": bool(bias <= PASS["abs_rel_bias"]), "value": bias}, "coverage_not_worse_than_baseline_by_0.05": {"pass": bool(cov_ok)}, "no_slice_more_than_10pct_worse": {"pass": bool(ok_sl), "detail": det}}
    return {"component": name, "kind": kind, "n": int(len(y)), "metrics": res, "baseline": baseline, "selected": selected, "pass_rule": rule, "PASSES_CONFIRMATION": bool(all(v["pass"] for v in rule.values())), "bootstrap_selected_vs_baseline": bs}
