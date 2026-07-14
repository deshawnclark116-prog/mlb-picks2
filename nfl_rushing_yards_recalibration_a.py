#!/usr/bin/env python3
"""
NFL_RUSHING_YARDS_RECALIBRATION_A

The rushing-yards champion (nfl_rushing_yards_champion_gate_a) cleared AUC
(0.6290), log-loss gain (+0.01889), and Brier, but failed ECE badly (0.1218
vs the 0.02 bar). The reliability table pinpoints it: the 0.5-0.6 predicted
bin (236 of 350 holdout rows, two-thirds of the data) predicts 0.553 but
actually resolves at 0.720 -- a 17-point gap. That lines up exactly with the
year-over-year base-rate shift already flagged in the clean baseline (2023
dev over-rate 54.67% vs 2024 holdout 65.71%): the model, trained mostly on
2023's cooler rate, systematically underpredicts 2024. Good ranking,
miscalibrated absolute probability -- exactly the pattern post-hoc Platt
calibration exists to fix, without retraining or touching the ranking model
at all (same method already validated on NFL receptions line 1.5).

Method (never touches the 2024 holdout during fitting):
  1. Load the already-trained champion and its ORIGINAL internal validation
     slice from champion_gate (2023 rows from pick_val_cut's row-count-based
     cut) -- already held out from training weights, a legitimate
     calibration-fitting set, and NOT the 2024 holdout.
  2. Fit a 2-parameter logistic recalibration: recal = sigmoid(a*logit(p)+b)
     via gradient descent on binary cross-entropy, on that validation slice.
  3. Apply (a, b) -- fixed, frozen -- to the RAW model predictions on the
     untouched 2024 holdout. Recompute AUC/log-loss/Brier/ECE.
  4. Re-run the exact same pre-registered gate from champion_gate on the
     recalibrated holdout probabilities.

AUC is mathematically unchanged by this (sigmoid(a*x+b) with a>0 is monotonic
in x), which isolates the fix to calibration only -- verified below, not
assumed.

Read-only on the baseline + saved model. Writes only a report + a small
calibration.json (a, b), if useful for later serving.

Run (Render, after the champion gate)
--------------------------------------
python -u nfl_rushing_yards_recalibration_a.py 2>&1 | tee /data/nfl_model/nfl_rushing_yards_recalibration_a.log
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import nfl_rushing_yards_champion_gate_a as g


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
    ap.add_argument("--model-dir", default=g.WORKDIR_DEFAULT)
    args = ap.parse_args()
    import xgboost as xgb

    mdir = Path(args.model_dir)
    print("NFL_RUSHING_YARDS_RECALIBRATION_A\n" + "=" * 34)

    bst = xgb.Booster(); bst.load_model(str(mdir / "nfl_rushing_yards.json"))
    feat_cols = json.loads((mdir / "nfl_rushing_yards_columns.json").read_text())
    assert feat_cols == g.FEATURES

    dev, hol = g.load(args.baseline)
    cut = g.pick_val_cut(dev)
    va = [r for r in dev if r[1] >= cut]   # same internal-val slice champion_gate used
    print(f"calibration-fit slice: 2023 weeks >= {cut}, n={len(va)}  (never touches 2024 holdout)")

    def mat(rows):
        X = np.array([[r[2 + i] if r[2 + i] is not None else g.NAN for i in range(len(g.FEATURES))]
                      for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, feature_names=g.FEATURES)

    # CRITICAL: predict without iteration_range silently uses ALL saved boosted
    # rounds, not the early-stopping-optimal count champion_gate trained to and
    # gated on. Must be passed explicitly on every predict call.
    itr = (0, bst.best_iteration + 1)
    print(f"scoring with iteration_range={itr} (best_iteration={bst.best_iteration}, "
          f"num_boosted_rounds={bst.num_boosted_rounds()})")

    va_raw = bst.predict(mat(va), iteration_range=itr)
    va_y = [r[-1] for r in va]
    a, b = fit_platt(va_raw, va_y)
    assert a > 0, f"fitted slope a={a} <= 0 -- recalibration would invert ranking, refusing to apply"
    print(f"fitted Platt params: a={a:.4f}  b={b:.4f}")

    hol_raw = bst.predict(mat(hol), iteration_range=itr)
    hol_y = [r[-1] for r in hol]
    hol_recal = apply_platt(hol_raw, a, b)

    before = g.metrics(list(map(float, hol_raw)), hol_y)
    after = g.metrics(list(map(float, hol_recal)), hol_y)

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
    verdict = ("NFL_RUSHING_YARDS_RECALIBRATED_PASSES_GATE"
               if passed else "NFL_RUSHING_YARDS_RECALIBRATED_STILL_FAILS_GATE")

    print("\n============ PRE-REGISTERED GATE (on recalibrated probabilities) ============")
    print(f"  AUC >= {g.GATE['min_auc']}:            {after['auc']:.4f}  -> {c1}")
    print(f"  ECE <= {g.GATE['max_ece']}:              {after['ece']:.4f}  -> {c2}")
    print(f"  logloss gain >= {g.GATE['min_logloss_gain']}:  {d_ll:+.5f}  -> {c3}")
    print(f"  Brier better than constant:  {after['brier']:.5f} < {constant['brier']:.5f}  -> {c4}")
    print(f"  VERDICT: {verdict}")

    if passed:
        (mdir / "nfl_rushing_yards_calibration.json").write_text(json.dumps({"a": a, "b": b}))
        print(f"\ncalibration params written: {mdir / 'nfl_rushing_yards_calibration.json'}")
        print("Next: re-run stability confirmation using these recalibrated probabilities before wiring live.")

    report = {"script": "NFL_RUSHING_YARDS_RECALIBRATION_A",
              "platt": {"a": a, "b": b}, "before": before, "after": after,
              "auc_delta": auc_delta, "gate": g.GATE, "passed": passed, "verdict": verdict}
    out = Path("/data/nfl_model") if Path("/data/nfl_model").exists() else mdir
    report_name = "nfl_rushing_yards_recalibration_a_report.json"
    (out / report_name).write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out/report_name}")
    print("Read-only on the model + baseline (writes only calibration.json on a pass). No production change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
