#!/usr/bin/env python3
"""
HITS_CONTEXT_STABILITY_CONFIRMATION_A

Rung after the challenger gate. Confirms the batter_hits context challenger's
2026-holdout win is REAL and STABLE, not a single-number fluke -- before any
production wiring is considered. Predictions-first: no odds.

Method
------
Rebuilds the exact strict-D-1 dataset (import from hits_context_challenger_gate_a),
loads the two models saved by the gate run (retrained_base + challenger), scores
both on the 2026 holdout, then stress-tests the difference three ways:

1. Date-block bootstrap (resample DATES, not rows; B replicates): confidence
   intervals and P(challenger better) for delta log-loss, delta Brier, delta AUC.
2. Monthly stability: per 2026 month, does the challenger win log-loss / AUC?
3. Slice stability: by lineup-spot bucket, platoon advantage, and park type --
   is the challenger non-worse across the board (no slice it quietly hurts)?

Pre-registered STABLE verdict (all must hold):
  - P(challenger log-loss < base) >= 0.90
  - AUC delta 95% CI lower bound > 0
  - challenger wins log-loss in >= half of eligible months (>= 800 rows)
  - challenger non-worse (delta log-loss >= -0.002) in >= 60% of eligible slices

Read-only on hr_model.sqlite. Loads models from the gate work dir (retrains if
absent). Writes only a report. Touches no production model or code.

Run (Render, after the gate)
----------------------------
python -u hits_context_stability_confirmation_a.py 2>&1 | tee /data/hr_model/hits_context_stability_confirmation_a.log
"""

import argparse
import json
import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np

import hits_context_challenger_gate_a as g

WORKDIR = g.WORKDIR
B_BOOTSTRAP = 2000
SEED = 20260712
MONTH_MIN_ROWS = 800
SLICE_NONWORSE_TOL = 0.002


def logloss(prob, label):
    eps = 1e-12
    p = np.clip(prob, eps, 1 - eps)
    return float(-np.mean(label * np.log(p) + (1 - label) * np.log(1 - p)))


def brier(prob, label):
    return float(np.mean((prob - label) ** 2))


def np_auc(scores, labels):
    labels = np.asarray(labels)
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s = np.asarray(scores)[order]
    i = 0
    n = len(s)
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def train_arm(feats, tr, va, name, work):
    import xgboost as xgb
    saved = work / f"{name}.json"
    bst = xgb.Booster()
    if saved.exists():
        bst.load_model(str(saved))
        return bst, feats
    # fallback: retrain with identical params
    def mat(rows):
        X = np.array([[r["f"].get(k, g.NAN) for k in feats] for r in rows], dtype=np.float32)
        y = np.array([r["y"] for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=feats)
    bst = xgb.train(g.__dict__.get("params") or {
        "objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 4,
        "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 5, "seed": 13},
        mat(tr), num_boost_round=800, evals=[(mat(va), "val")],
        early_stopping_rounds=40, verbose_eval=False)
    return bst, feats


def predict(bst, feats, rows):
    import xgboost as xgb
    X = np.array([[r["f"].get(k, g.NAN) for k in feats] for r in rows], dtype=np.float32)
    it = getattr(bst, "best_iteration", None)
    rng = (0, it + 1) if it is not None else (0, 0)
    try:
        return bst.predict(xgb.DMatrix(X, feature_names=feats), iteration_range=rng)
    except Exception:
        return bst.predict(xgb.DMatrix(X, feature_names=feats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=g.SOURCE)
    ap.add_argument("--workdir", default=str(WORKDIR))
    args = ap.parse_args()
    work = Path(args.workdir)

    import sqlite3
    con = sqlite3.connect(f"file:{args.source}?mode=ro", uri=True)
    print("HITS_CONTEXT_STABILITY_CONFIRMATION_A", flush=True)
    print("=====================================", flush=True)
    print("rebuilding dataset ...", flush=True)
    data = g.build_dataset(con)
    con.close()
    dev, hol = data["2025"], data["2026"]
    dates = sorted({r["game_date"] for r in dev})
    cut = dates[int(len(dates) * 0.8)] if len(dates) > 5 else dates[-1]
    tr = [r for r in dev if r["game_date"] < cut]
    va = [r for r in dev if r["game_date"] >= cut]
    print(f"  holdout rows: {len(hol)}", flush=True)

    base_bst, bf = train_arm(g.BASE_FEATURES, tr, va, "retrained_base", work)
    chal_bst, cf = train_arm(g.BASE_FEATURES + g.CONTEXT_FEATURES, tr, va, "challenger", work)

    pb = np.asarray(predict(base_bst, bf, hol), dtype=np.float64)
    pc = np.asarray(predict(chal_bst, cf, hol), dtype=np.float64)
    y = np.array([r["y"] for r in hol], dtype=np.float64)
    gdate = np.array([r["game_date"] for r in hol])
    month = np.array([d[:7] for d in gdate])

    print("\nPOINT ESTIMATES (2026 holdout)")
    print("------------------------------")
    b_ll, c_ll = logloss(pb, y), logloss(pc, y)
    b_au, c_au = np_auc(pb, y), np_auc(pc, y)
    b_br, c_br = brier(pb, y), brier(pc, y)
    print(f"  log-loss  base={b_ll:.5f}  chal={c_ll:.5f}  delta={b_ll-c_ll:+.5f}")
    print(f"  AUC       base={b_au:.4f}  chal={c_au:.4f}  delta={c_au-b_au:+.4f}")
    print(f"  Brier     base={b_br:.5f}  chal={c_br:.5f}  delta={b_br-c_br:+.5f}")

    # ---- 1. date-block bootstrap ----
    print(f"\nDATE-BLOCK BOOTSTRAP (B={B_BOOTSTRAP}, resampling dates)")
    print("-------------------------------------------------------")
    uniq = np.array(sorted(set(gdate.tolist())))
    idx_by_date = {d: np.where(gdate == d)[0] for d in uniq}
    rng = np.random.default_rng(SEED)
    d_ll = np.empty(B_BOOTSTRAP); d_au = np.empty(B_BOOTSTRAP); d_br = np.empty(B_BOOTSTRAP)
    for b in range(B_BOOTSTRAP):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_date[d] for d in samp])
        yy = y[idx]
        d_ll[b] = logloss(pb[idx], yy) - logloss(pc[idx], yy)   # >0 chal better
        d_br[b] = brier(pb[idx], yy) - brier(pc[idx], yy)       # >0 chal better
        d_au[b] = np_auc(pc[idx], yy) - np_auc(pb[idx], yy)     # >0 chal better

    def summ(arr, name):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        pbetter = float(np.mean(arr > 0))
        print(f"  {name:12s} mean={arr.mean():+.5f}  95%CI=[{lo:+.5f},{hi:+.5f}]  P(chal better)={pbetter:.3f}")
        return {"mean": float(arr.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "p_better": pbetter}
    boot = {"delta_log_loss": summ(d_ll, "log-loss"),
            "delta_auc": summ(d_au, "AUC"),
            "delta_brier": summ(d_br, "Brier")}

    # ---- 2. monthly stability ----
    print("\nMONTHLY STABILITY")
    print("-----------------")
    months = sorted(set(month.tolist()))
    monthly = []
    ll_month_wins = 0; ll_month_elig = 0
    for m in months:
        mi = np.where(month == m)[0]
        if len(mi) < MONTH_MIN_ROWS:
            print(f"  {m}: n={len(mi)} (skip, <{MONTH_MIN_ROWS})"); continue
        yy = y[mi]
        mb_ll, mc_ll = logloss(pb[mi], yy), logloss(pc[mi], yy)
        mb_au, mc_au = np_auc(pb[mi], yy), np_auc(pc[mi], yy)
        win = mc_ll < mb_ll
        ll_month_elig += 1; ll_month_wins += int(win)
        monthly.append({"month": m, "n": int(len(mi)), "d_logloss": mb_ll-mc_ll, "d_auc": mc_au-mb_au, "chal_wins_ll": win})
        print(f"  {m}: n={len(mi):5d}  d_logloss={mb_ll-mc_ll:+.5f}  d_auc={mc_au-mb_au:+.4f}  {'chal' if win else 'BASE'}")

    # ---- 3. slice stability ----
    print("\nSLICE STABILITY (challenger non-worse = d_logloss >= -%.3f)" % SLICE_NONWORSE_TOL)
    print("--------------------------------------------------------")
    def bucket_lineup(r):
        s = r["f"]["batting_order"]
        if s is None: return "spot_unknown"
        s = int(s)
        return "spot_1_2" if s <= 2 else "spot_3_5" if s <= 5 else "spot_6_9"
    def bucket_platoon(r):
        v = r["f"]["platoon_advantage"]
        return "platoon_unknown" if (v is None or (isinstance(v, float) and math.isnan(v))) else ("platoon_adv" if v == 1.0 else "platoon_no")
    def bucket_park(r):
        f = r["f"]
        if f.get("hitterish_park") == 1: return "park_hitter"
        if f.get("pitcherish_park") == 1: return "park_pitcher"
        return "park_neutral"
    slices = {}
    for i, r in enumerate(hol):
        for key in (bucket_lineup(r), bucket_platoon(r), bucket_park(r)):
            slices.setdefault(key, []).append(i)
    slice_rows = []; nonworse = 0; elig = 0
    for key in sorted(slices):
        ii = np.array(slices[key])
        if len(ii) < 300:
            print(f"  {key}: n={len(ii)} (skip, <300)"); continue
        yy = y[ii]
        dl = logloss(pb[ii], yy) - logloss(pc[ii], yy)
        ok = dl >= -SLICE_NONWORSE_TOL
        elig += 1; nonworse += int(ok)
        slice_rows.append({"slice": key, "n": int(len(ii)), "d_logloss": dl, "non_worse": ok})
        print(f"  {key:16s} n={len(ii):5d}  d_logloss={dl:+.5f}  {'ok' if ok else 'WORSE'}")

    # ---- verdict ----
    cond_boot_ll = boot["delta_log_loss"]["p_better"] >= 0.90
    cond_boot_auc = boot["delta_auc"]["ci_lo"] > 0.0
    cond_month = ll_month_elig > 0 and ll_month_wins >= math.ceil(ll_month_elig / 2)
    cond_slice = elig > 0 and (nonworse / elig) >= 0.60
    stable = cond_boot_ll and cond_boot_auc and cond_month and cond_slice
    verdict = ("HITS_CONTEXT_CHALLENGER_STABLE_ON_2026_HOLDOUT_READY_FOR_"
               "IMPLEMENTATION_PARITY_PLANNING_NO_AUTO_PROMOTION"
               if stable else
               "HITS_CONTEXT_CHALLENGER_NOT_YET_STABLE_DO_NOT_PRODUCTIONIZE")

    print("\n================ STABILITY VERDICT ================")
    print(f"  P(chal better log-loss) >= 0.90 : {boot['delta_log_loss']['p_better']:.3f}  -> {cond_boot_ll}")
    print(f"  AUC delta 95% CI lower > 0       : {boot['delta_auc']['ci_lo']:+.5f}  -> {cond_boot_auc}")
    print(f"  monthly log-loss wins >= half    : {ll_month_wins}/{ll_month_elig}  -> {cond_month}")
    print(f"  slices non-worse >= 60%          : {nonworse}/{elig}  -> {cond_slice}")
    print(f"  VERDICT: {verdict}")

    report = {"script": "HITS_CONTEXT_STABILITY_CONFIRMATION_A", "holdout": "2026",
              "point": {"base_log_loss": b_ll, "chal_log_loss": c_ll,
                        "base_auc": b_au, "chal_auc": c_au,
                        "base_brier": b_br, "chal_brier": c_br},
              "bootstrap": boot, "monthly": monthly, "slices": slice_rows,
              "verdict": verdict, "stable": stable}
    out_dir = Path("/data/hr_model") if Path("/data/hr_model").exists() else work
    (out_dir / "hits_context_stability_confirmation_a_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {out_dir/'hits_context_stability_confirmation_a_report.json'}")
    print("Read-only. No production model or code changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
