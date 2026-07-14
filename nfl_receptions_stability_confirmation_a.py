#!/usr/bin/env python3
"""
NFL_RECEPTIONS_STABILITY_CONFIRMATION_A

Rung 4 of the NFL pipeline. Confirms the receptions champion (trained and
gated in nfl_receptions_champion_gate_a) is not a lucky fit to a particular
stretch of the 2024 holdout, and does not quietly fail for any position
subgroup -- before any live wiring. Predictions-first: no odds.

Loads the model + columns already saved by the champion gate (no retraining),
re-scores the 2024 holdout, then:
  1. Date-block bootstrap (B=2000, resampling WEEKS not rows): 95% CI on AUC.
  2. Weekly breakdown: AUC/log-loss per week of the 2024 holdout.
  3. Slice breakdown: WR / TE / RB and home / away, each independently --
     catches the case (like the 0.6-0.7 reliability bin) where aggregate
     calibration looks fine but one subgroup is quietly miscalibrated.

Pre-registered STABLE verdict (all must hold):
  - bootstrap AUC 95% CI lower bound >= 0.65
  - AUC >= 0.65 in >= 80% of individual weeks (min 100 rows/week)
  - every position slice (min 300 rows): AUC >= 0.55 and ECE <= 0.04

Read-only. Writes only a report. No production change.

Run (Render, after the champion gate)
--------------------------------------
python -u nfl_receptions_stability_confirmation_a.py 2>&1 | tee /data/nfl_model/nfl_receptions_stability_confirmation_a.log
"""

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import nfl_receptions_champion_gate_a as g

MODEL_WORKDIR = Path("/data/nfl_model/nfl_receptions_champion_gate_a_work")
B_BOOTSTRAP = 2000
SEED = 20260714
MIN_WEEK_ROWS = 100
MIN_SLICE_ROWS = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=g.BASELINE_DEFAULT)
    ap.add_argument("--model-dir", default=str(MODEL_WORKDIR))
    args = ap.parse_args()
    import xgboost as xgb

    print("NFL_RECEPTIONS_STABILITY_CONFIRMATION_A\n========================================")
    con = sqlite3.connect(f"file:{args.baseline}?mode=ro", uri=True)
    cols = ["season", "week", "position", "is_home"] + g.FEATURES + ["over_0_5"]
    hol = con.execute(
        f"SELECT {', '.join(cols)} FROM nfl_receptions_baseline WHERE season=2024").fetchall()
    con.close()
    print(f"2024 holdout rows: {len(hol)}")

    mdir = Path(args.model_dir)
    bst = xgb.Booster(); bst.load_model(str(mdir / "nfl_receptions.json"))
    feat_cols = json.loads((mdir / "nfl_receptions_columns.json").read_text())
    assert feat_cols == g.FEATURES

    n_meta = 4  # season, week, position, is_home
    X = np.array([[r[n_meta + i] if r[n_meta + i] is not None else g.NAN
                   for i in range(len(g.FEATURES))] for r in hol], dtype=np.float32)
    y = np.array([r[-1] for r in hol], dtype=np.float64)
    probs = bst.predict(xgb.DMatrix(X, feature_names=g.FEATURES))
    week = np.array([r[1] for r in hol])
    position = np.array([r[2] for r in hol])
    is_home = np.array([r[3] for r in hol])

    print(f"\nPOINT: AUC={g.auc(probs,y):.4f}  n={len(y)}")

    # ---- 1. bootstrap ----
    print(f"\nDATE-BLOCK BOOTSTRAP (B={B_BOOTSTRAP}, resampling weeks)")
    uniq = np.array(sorted(set(week.tolist())))
    idx_by = {w: np.where(week == w)[0] for w in uniq}
    rng = np.random.default_rng(SEED)
    aucs = np.empty(B_BOOTSTRAP)
    for b in range(B_BOOTSTRAP):
        idx = np.concatenate([idx_by[w] for w in rng.choice(uniq, len(uniq), replace=True)])
        aucs[b] = g.auc(probs[idx], y[idx])
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    print(f"  AUC mean={aucs.mean():.4f}  95%CI=[{lo:.4f},{hi:.4f}]")

    # ---- 2. weekly ----
    print("\nWEEKLY (AUC per week, min {} rows)".format(MIN_WEEK_ROWS))
    weeks_ok = weeks_elig = 0
    weekly_rows = []
    for w in uniq:
        idx = np.where(week == w)[0]
        if len(idx) < MIN_WEEK_ROWS:
            print(f"  week {int(w):>2}: n={len(idx)} (skip)"); continue
        a = g.auc(probs[idx], y[idx])
        weeks_elig += 1
        ok = a >= 0.65
        weeks_ok += int(ok)
        weekly_rows.append({"week": int(w), "n": int(len(idx)), "auc": a, "ok": ok})
        print(f"  week {int(w):>2}: n={len(idx):>4}  AUC={a:.4f}  {'ok' if ok else 'LOW'}")

    # ---- 3. slices ----
    print(f"\nSLICES (min {MIN_SLICE_ROWS} rows)")
    slices = {}
    for i in range(len(y)):
        slices.setdefault(position[i], []).append(i)
        slices.setdefault("home" if is_home[i] == 1 else "away", []).append(i)
    slice_rows = []; slices_ok = slices_elig = 0
    for key in sorted(slices):
        idx = np.array(slices[key])
        if len(idx) < MIN_SLICE_ROWS:
            print(f"  {key:8s} n={len(idx)} (skip)"); continue
        a = g.auc(probs[idx], y[idx])
        m = g.metrics(probs[idx], y[idx])
        ok = a >= 0.55 and m["ece"] <= 0.04
        slices_elig += 1
        slices_ok += int(ok)
        slice_rows.append({"slice": key, "n": int(len(idx)), "auc": a, "ece": m["ece"], "ok": ok})
        print(f"  {key:8s} n={len(idx):>4}  AUC={a:.4f}  ECE={m['ece']:.4f}  {'ok' if ok else 'FAIL'}")

    c1 = lo >= 0.65
    c2 = weeks_elig > 0 and (weeks_ok / weeks_elig) >= 0.80
    c3 = slices_elig > 0 and slices_ok == slices_elig
    stable = c1 and c2 and c3
    verdict = ("NFL_RECEPTIONS_STABLE_READY_FOR_LIVE_WIRING" if stable
               else "NFL_RECEPTIONS_NOT_YET_STABLE")

    print("\n================ VERDICT ================")
    print(f"  bootstrap AUC CI lower >= 0.65: {lo:.4f} -> {c1}")
    print(f"  weeks AUC>=0.65 >= 80%:          {weeks_ok}/{weeks_elig} -> {c2}")
    print(f"  all slices pass:                 {slices_ok}/{slices_elig} -> {c3}")
    print(f"  VERDICT: {verdict}")

    report = {"script": "NFL_RECEPTIONS_STABILITY_CONFIRMATION_A",
              "bootstrap": {"mean": float(aucs.mean()), "ci_lo": float(lo), "ci_hi": float(hi)},
              "weekly": weekly_rows, "slices": slice_rows, "stable": stable, "verdict": verdict}
    out = Path("/data/nfl_model") if Path("/data/nfl_model").exists() else mdir
    (out / "nfl_receptions_stability_confirmation_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out/'nfl_receptions_stability_confirmation_a_report.json'}")
    print("Read-only. No production change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
