#!/usr/bin/env python3
"""
HITS_EASY_STABILITY_CONFIRMATION_B

Confirms the base+easy batter_hits model (the feature-discovery winner: base 8 +
live-cheap features, park/weather rejected) is a REAL, STABLE improvement over
base on the untouched 2026 holdout -- before any production wiring.
Predictions-first: no odds.

Reuses hits_feature_discovery_b.build for the dataset and its exact features,
retrains base and base+easy deterministically, then stress-tests base+easy vs
base three ways: date-block bootstrap (B=2000), monthly, and slice stability.

Pre-registered STABLE verdict (all must hold):
  P(base+easy better log-loss) >= 0.90; AUC delta 95% CI lower > 0;
  base+easy wins log-loss in >= half of eligible months (>= 800 rows);
  base+easy non-worse (d_logloss >= -0.002) in >= 60% of eligible slices.

Read-only on hr_model.sqlite. Writes only a report. No production touch.

Run (Render, after hits_feature_discovery_b)
--------------------------------------------
python -u hits_easy_stability_confirmation_b.py 2>&1 | tee /data/hr_model/hits_easy_stability_confirmation_b.log
"""

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

import hits_feature_discovery_b as fd
import hits_context_stability_confirmation_a as stab  # np_auc, logloss, brier

B_BOOTSTRAP = 2000
SEED = 20260712
MONTH_MIN_ROWS = 800
SLICE_TOL = 0.002


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=fd.SOURCE)
    ap.add_argument("--workdir", default=str(fd.WORKDIR))
    args = ap.parse_args()
    import xgboost as xgb
    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(f"file:{args.source}?mode=ro", uri=True)
    print("HITS_EASY_STABILITY_CONFIRMATION_B\n==================================", flush=True)
    print("rebuilding dataset ...", flush=True)
    data = fd.build(con); con.close()
    dev, hol = data["2025"], data["2026"]
    dates = sorted({r["game_date"] for r in dev})
    cut = dates[int(len(dates) * 0.8)]
    tr = [r for r in dev if r["game_date"] < cut]
    va = [r for r in dev if r["game_date"] >= cut]
    print(f"  holdout rows: {len(hol)}", flush=True)

    # keep the same easy set feature-discovery used (leak-checked; expected_pa_v1 kept)
    ep = np.array([r["f"]["expected_pa_v1"] for r in hol], dtype=float)
    apv = np.array([r["actual_pa"] for r in hol], dtype=float)
    okp = ~np.isnan(ep)
    easy = list(fd.EASY) if (okp.sum() == 0 or np.mean(np.round(ep[okp]) == apv[okp]) <= 0.95) \
        else [f for f in fd.EASY if f != "expected_pa_v1"]
    feat_base, feat_easy = fd.BASE, fd.BASE + easy

    def mat(rows, feats):
        X = np.array([[r["f"].get(k, fd.NAN) for k in feats] for r in rows], dtype=np.float32)
        y = np.array([r["y"] for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=feats)

    def train(feats):
        b = xgb.train(fd.PARAMS, mat(tr, feats), num_boost_round=800,
                      evals=[(mat(va, feats), "val")], early_stopping_rounds=40,
                      verbose_eval=False)
        return b.predict(mat(hol, feats), iteration_range=(0, b.best_iteration + 1))

    print("training base and base+easy ...", flush=True)
    pb = np.asarray(train(feat_base), dtype=np.float64)
    pe = np.asarray(train(feat_easy), dtype=np.float64)
    y = np.array([r["y"] for r in hol], dtype=np.float64)
    gd = np.array([r["game_date"] for r in hol]); month = np.array([d[:7] for d in gd])

    print("\nPOINT (2026 holdout)")
    print(f"  log-loss base={stab.logloss(pb,y):.5f} easy={stab.logloss(pe,y):.5f} delta={stab.logloss(pb,y)-stab.logloss(pe,y):+.5f}")
    print(f"  AUC      base={stab.np_auc(pb,y):.4f} easy={stab.np_auc(pe,y):.4f} delta={stab.np_auc(pe,y)-stab.np_auc(pb,y):+.4f}")

    # bootstrap
    print(f"\nDATE-BLOCK BOOTSTRAP (B={B_BOOTSTRAP})")
    uniq = np.array(sorted(set(gd.tolist())))
    idx_by = {d: np.where(gd == d)[0] for d in uniq}
    rng = np.random.default_rng(SEED)
    dll = np.empty(B_BOOTSTRAP); dau = np.empty(B_BOOTSTRAP); dbr = np.empty(B_BOOTSTRAP)
    for b in range(B_BOOTSTRAP):
        idx = np.concatenate([idx_by[d] for d in rng.choice(uniq, len(uniq), replace=True)])
        yy = y[idx]
        dll[b] = stab.logloss(pb[idx], yy) - stab.logloss(pe[idx], yy)
        dbr[b] = stab.brier(pb[idx], yy) - stab.brier(pe[idx], yy)
        dau[b] = stab.np_auc(pe[idx], yy) - stab.np_auc(pb[idx], yy)

    def summ(a, name):
        lo, hi = np.percentile(a, [2.5, 97.5]); pb_ = float(np.mean(a > 0))
        print(f"  {name:9s} mean={a.mean():+.5f} 95%CI=[{lo:+.5f},{hi:+.5f}] P(better)={pb_:.3f}")
        return {"mean": float(a.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "p_better": pb_}
    boot = {"log_loss": summ(dll, "log-loss"), "auc": summ(dau, "AUC"), "brier": summ(dbr, "Brier")}

    print("\nMONTHLY")
    mwins = melig = 0; monthly = []
    for m in sorted(set(month.tolist())):
        mi = np.where(month == m)[0]
        if len(mi) < MONTH_MIN_ROWS:
            print(f"  {m}: n={len(mi)} skip"); continue
        yy = y[mi]; d = stab.logloss(pb[mi], yy) - stab.logloss(pe[mi], yy)
        da = stab.np_auc(pe[mi], yy) - stab.np_auc(pb[mi], yy)
        w = d > 0; melig += 1; mwins += int(w)
        monthly.append({"month": m, "n": int(len(mi)), "d_logloss": d, "d_auc": da, "win": w})
        print(f"  {m}: n={len(mi):5d} d_logloss={d:+.5f} d_auc={da:+.4f} {'easy' if w else 'BASE'}")

    print("\nSLICES")
    def blin(r):
        s = int(r["f"]["batting_order"]); return "spot_1_2" if s <= 2 else "spot_3_5" if s <= 5 else "spot_6_9"
    def bplat(r):
        v = r["f"]["platoon_advantage"]
        return "platoon_unk" if (v is None or (isinstance(v, float) and math.isnan(v))) else ("platoon_adv" if v == 1.0 else "platoon_no")
    def bpark(r):
        f = r["f"]
        return "park_hitter" if f.get("hitterish_park") == 1 else "park_pitcher" if f.get("pitcherish_park") == 1 else "park_neutral"
    sl = {}
    for i, r in enumerate(hol):
        for k in (blin(r), bplat(r), bpark(r)):
            sl.setdefault(k, []).append(i)
    nonworse = elig = 0; slrows = []
    for k in sorted(sl):
        ii = np.array(sl[k])
        if len(ii) < 300:
            print(f"  {k}: n={len(ii)} skip"); continue
        yy = y[ii]; d = stab.logloss(pb[ii], yy) - stab.logloss(pe[ii], yy)
        ok = d >= -SLICE_TOL; elig += 1; nonworse += int(ok)
        slrows.append({"slice": k, "n": int(len(ii)), "d_logloss": d, "ok": ok})
        print(f"  {k:14s} n={len(ii):5d} d_logloss={d:+.5f} {'ok' if ok else 'WORSE'}")

    c1 = boot["log_loss"]["p_better"] >= 0.90
    c2 = boot["auc"]["ci_lo"] > 0.0
    c3 = melig > 0 and mwins >= math.ceil(melig / 2)
    c4 = elig > 0 and nonworse / elig >= 0.60
    stable = c1 and c2 and c3 and c4
    verdict = ("HITS_EASY_MODEL_STABLE_ON_2026_HOLDOUT_READY_FOR_LIVE_FEATURE_WIRING_AND_PARITY"
               if stable else "HITS_EASY_MODEL_NOT_YET_STABLE")
    print("\n================ VERDICT ================")
    print(f"  P(better logloss)>=0.90: {boot['log_loss']['p_better']:.3f} -> {c1}")
    print(f"  AUC CI lower>0:          {boot['auc']['ci_lo']:+.5f} -> {c2}")
    print(f"  monthly wins>=half:      {mwins}/{melig} -> {c3}")
    print(f"  slices non-worse>=60%:   {nonworse}/{elig} -> {c4}")
    print(f"  VERDICT: {verdict}")

    report = {"script": "HITS_EASY_STABILITY_CONFIRMATION_B", "holdout": "2026",
              "features_easy": easy, "bootstrap": boot, "monthly": monthly,
              "slices": slrows, "stable": stable, "verdict": verdict}
    out = Path("/data/hr_model") if Path("/data/hr_model").exists() else work
    (out / "hits_easy_stability_confirmation_b_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out/'hits_easy_stability_confirmation_b_report.json'}")
    print("Read-only. No production model or code changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
