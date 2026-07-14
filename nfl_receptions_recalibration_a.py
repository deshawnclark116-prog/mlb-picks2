#!/usr/bin/env python3
"""
NFL_RECEPTIONS_RECALIBRATION_A

Both receptions lines (1.5, 2.5) failed their pre-registered champion gate on
ECE alone (1.5: 0.0222 vs 0.02 bar; 2.5: 0.0242 vs 0.02 bar, and TE slice
0.0524) while AUC was strong and stable (bootstrap CI floors 0.78/0.78, 16/16
weeks). That specific pattern -- good ranking, imperfect absolute probability
-- is what post-hoc calibration (Platt scaling) exists to fix, without
retraining or touching the ranking model at all.

Method (never touches the 2024 holdout during fitting):
  1. Load the already-trained champion for --line, and its ORIGINAL internal
     validation slice from champion_gate (2023 weeks >= the 80% cut) -- this
     was already held out from training weights, so it's a legitimate
     calibration-fitting set, and it is NOT the 2024 holdout.
  2. Fit a 2-parameter logistic recalibration: recal = sigmoid(a*logit(p) + b)
     via gradient descent on binary cross-entropy, on that validation slice.
  3. Apply (a, b) -- fixed, frozen -- to the RAW model predictions on the
     untouched 2024 holdout. Recompute AUC/log-loss/Brier/ECE.
  4. Re-run the exact same pre-registered gate from champion_gate on the
     recalibrated holdout probabilities.

AUC is mathematically unchanged by this (sigmoid(a*x+b) with a>0 is monotonic
in x when a>0 -- verified below), which isolates the fix to calibration only.

Read-only on the baseline + saved model. Writes only a report + a small
calibration.json (a, b) for the line, if useful for later serving.

Run (Render, after both champion gate runs)
--------------------------------------------
python -u nfl_receptions_recalibration_a.py --line 1.5 2>&1 | tee /data/nfl_model/nfl_receptions_recalibration_a_1_5.log
python -u nfl_receptions_recalibration_a.py --line 2.5 2>&1 | tee /data/nfl_model/nfl_receptions_recalibration_a_2_5.log
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


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_platt(raw_probs, labels, iters=2000, lr=0.1):
    """Fit recal = sigmoid(a*logit(p) + b) by gradient descent on log-loss.
    Initialized at a=1, b=0 (identity) so it can only improve or match, never
    make a well-behaved input worse by a bad init."""
    x = logit(np.asarray(raw_probs, dtype=float))
    y = np.asarray(labels, dtype=float)
    a, b = 1.0, 0.0
    n = len(y)
    for _ in range(iters):
        z = a * x + b
        pred = sigmoid(z)
        grad = pred - y
        ga = float(np.mean(grad * x))
        gb = float(np.mean(grad))
        a -= lr * ga
        b -= lr * gb
    return a, b


def apply_platt(raw_probs, a, b):
    return sigmoid(a * logit(np.asarray(raw_probs, dtype=float)) + b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=g.BASELINE_DEFAULT)
    ap.add_argument("--model-dir", default=str(MODEL_WORKDIR))
    ap.add_argument("--line", type=float, choices=[1.5, 2.5], required=True)
    args = ap.parse_args()
    import xgboost as xgb

    target_col = g.LINE_TO_COLUMN[args.line]
    mdir = Path(args.model_dir) / f"line_{args.line}"
    model_stem = f"nfl_receptions_over_{str(args.line).replace('.', '_')}"
    print(f"NFL_RECEPTIONS_RECALIBRATION_A  [line={args.line}]\n" + "=" * 40)

    bst = xgb.Booster(); bst.load_model(str(mdir / f"{model_stem}.json"))
    feat_cols = json.loads((mdir / f"{model_stem}_columns.json").read_text())
    assert feat_cols == g.FEATURES

    dev, hol = g.load(args.baseline, target_col)
    weeks = sorted({r[1] for r in dev})
    cut = weeks[int(len(weeks) * 0.8)] if len(weeks) > 5 else weeks[-1]
    va = [r for r in dev if r[1] >= cut]   # same internal-val slice champion_gate used
    print(f"calibration-fit slice: 2023 weeks >= {cut}, n={len(va)}  (never touches 2024 holdout)")

    def mat(rows):
        X = np.array([[r[2 + i] if r[2 + i] is not None else g.NAN for i in range(len(g.FEATURES))]
                      for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, feature_names=g.FEATURES)

    va_raw = bst.predict(mat(va))
    va_y = [r[-1] for r in va]
    a, b = fit_platt(va_raw, va_y)
    assert a > 0, f"fitted slope a={a} <= 0 -- recalibration would invert ranking, refusing to apply"
    print(f"fitted Platt params: a={a:.4f}  b={b:.4f}")

    hol_raw = bst.predict(mat(hol))
    hol_y = [r[-1] for r in hol]
    hol_recal = apply_platt(hol_raw, a, b)

    before = g.metrics(list(map(float, hol_raw)), hol_y)
    after = g.metrics(list(map(float, hol_recal)), hol_y)

    # AUC must be unchanged (monotonic transform) -- verify, don't assume
    auc_delta = abs(before["auc"] - after["auc"])
    print(f"\nAUC before={before['auc']:.4f}  after={after['auc']:.4f}  "
          f"delta={auc_delta:.5f}  (should be ~0, monotonic transform)")

    print("\n============ 2024 HOLDOUT: raw vs recalibrated ============")
    print(f"  {'arm':14s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'raw':14s} {before['auc']:>7.4f}  {before['log_loss']:>9.5f}  {before['brier']:>7.5f} {before['ece']:>7.4f}")
    print(f"  {'recalibrated':14s} {after['auc']:>7.4f}  {after['log_loss']:>9.5f}  {after['brier']:>7.5f} {after['ece']:>7.4f}")

    print("\nrecalibrated reliability (pred -> actual):")
    for r in after["reliability"]:
        print(f"   {r['bin']}  n={r['n']:>5}  pred={r['pred']:.3f}  actual={r['actual']:.3f}")

    train_rate = float(np.mean([r[-1] for r in dev if r[1] < cut]))
    constant = g.metrics([train_rate] * len(hol), hol_y)
    d_ll = constant["log_loss"] - after["log_loss"]

    c1 = after["auc"] >= g.GATE["min_auc"]
    c2 = after["ece"] <= g.GATE["max_ece"]
    c3 = d_ll >= g.GATE["min_logloss_gain"]
    c4 = after["brier"] < constant["brier"]
    passed = c1 and c2 and c3 and c4
    verdict = (f"NFL_RECEPTIONS_LINE_{args.line}_RECALIBRATED_PASSES_GATE"
               if passed else f"NFL_RECEPTIONS_LINE_{args.line}_RECALIBRATED_STILL_FAILS_GATE")

    print("\n============ PRE-REGISTERED GATE (on recalibrated probabilities) ============")
    print(f"  AUC >= {g.GATE['min_auc']}:            {after['auc']:.4f}  -> {c1}")
    print(f"  ECE <= {g.GATE['max_ece']}:              {after['ece']:.4f}  -> {c2}")
    print(f"  logloss gain >= {g.GATE['min_logloss_gain']}:  {d_ll:+.5f}  -> {c3}")
    print(f"  Brier better than constant:  {after['brier']:.5f} < {constant['brier']:.5f}  -> {c4}")
    print(f"  VERDICT: {verdict}")

    if passed:
        (mdir / f"{model_stem}_calibration.json").write_text(json.dumps({"a": a, "b": b}))
        print(f"\ncalibration params written: {mdir / (model_stem + '_calibration.json')}")
        print("Next: re-run stability confirmation using these recalibrated probabilities before wiring live.")

    report = {"script": "NFL_RECEPTIONS_RECALIBRATION_A", "line": args.line,
              "platt": {"a": a, "b": b}, "before": before, "after": after,
              "auc_delta": auc_delta, "gate": g.GATE, "passed": passed, "verdict": verdict}
    out = Path("/data/nfl_model") if Path("/data/nfl_model").exists() else mdir
    report_name = f"nfl_receptions_recalibration_a_line_{args.line}_report.json"
    (out / report_name).write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out/report_name}")
    print("Read-only on the model + baseline (writes only calibration.json on a pass). No production change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
