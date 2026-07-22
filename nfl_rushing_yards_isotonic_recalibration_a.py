#!/usr/bin/env python3
"""
NFL_RUSHING_YARDS_ISOTONIC_RECALIBRATION_A

The 2-parameter Platt recalibration (nfl_rushing_yards_recalibration_a) cut
ECE more than half (0.1218 -> 0.0592) but still failed the 0.02 bar -- the
recalibrated reliability table showed a real residual pattern a single
logistic transform can't reach (0.6-0.7 predicted resolving at 0.520, 0.8-0.9
resolving at 0.727 -- not a uniform shift). Isotonic regression fits an
arbitrary non-decreasing step function instead of a 2-parameter shape, so it
can match the empirical calibration curve more closely than Platt can.

Real risk, stated up front: the calibration-fit slice is only n=75 (2023
weeks >= the internal val cut). Isotonic regression via PAVA has far more
effective degrees of freedom than Platt's 2 parameters -- with this little
data it can easily overfit noise in the fitting slice rather than learn a
real pattern. This is exactly why the result is judged on the untouched 2024
holdout gate, not assumed to work because it's a fancier method.

Method (same anti-overfit discipline as the Platt attempt):
  1. Load the same champion model + the same calibration-fit slice
     (2023 weeks >= pick_val_cut) already used by the Platt attempt --
     never touches the 2024 holdout during fitting.
  2. Fit isotonic regression via PAVA (pool-adjacent-violators) on
     (raw_prob, label) pairs from that slice -- no sklearn dependency,
     hand-rolled, standard algorithm.
  3. Apply the fitted step function to RAW model predictions on the
     untouched 2024 holdout via right-continuous step lookup (extrapolate
     flat at the ends, which is standard for isotonic calibration).
  4. Re-run the exact same pre-registered gate from champion_gate.

AUC is rank-preserving under isotonic (non-decreasing map) up to ties --
verified below, not assumed.

Read-only on the model + baseline. Writes only a report + calibration.json
(the fitted step function) on a pass.

Run
---
python -u nfl_rushing_yards_isotonic_recalibration_a.py \
    --baseline nfl_models/nfl_rushing_yards_clean_baseline_a_work/baseline.sqlite \
    --model-dir nfl_models/nfl_rushing_yards_champion_gate_a_work
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


def fit_isotonic(raw_probs, labels):
    """Pool-adjacent-violators algorithm. Returns (x_sorted, y_step) --
    the fitted non-decreasing step function as parallel arrays, ready for
    step-function lookup on new x values."""
    order = np.argsort(raw_probs, kind="mergesort")
    x = np.asarray(raw_probs, dtype=float)[order]
    y = np.asarray(labels, dtype=float)[order]

    # PAVA: maintain a stack of (value, weight) blocks; merge back-to-front
    # whenever the newest block's mean would violate monotonicity.
    block_val = []
    block_w = []
    block_start = []
    for i in range(len(y)):
        block_val.append(y[i])
        block_w.append(1.0)
        block_start.append(i)
        while len(block_val) > 1 and block_val[-2] > block_val[-1]:
            v2, w2 = block_val.pop(), block_w.pop()
            v1, w1 = block_val.pop(), block_w.pop()
            block_start.pop()
            new_w = w1 + w2
            new_v = (v1 * w1 + v2 * w2) / new_w
            block_val.append(new_v)
            block_w.append(new_w)

    # expand blocks back to per-point fitted values (still sorted by x)
    y_fit = np.empty(len(y))
    idx = 0
    for bi, start in enumerate(block_start):
        end = block_start[bi + 1] if bi + 1 < len(block_start) else len(y)
        y_fit[start:end] = block_val[bi]
    return x, y_fit


def apply_isotonic(x_fit, y_fit, new_x):
    """Step-function lookup: for each new_x, find its position among the
    fitted x's (searchsorted) and take the corresponding y. Flat
    extrapolation beyond the fitted range (standard for isotonic calibration
    -- we have no information past the ends of the fitting data)."""
    new_x = np.asarray(new_x, dtype=float)
    idx = np.searchsorted(x_fit, new_x, side="right") - 1
    idx = np.clip(idx, 0, len(y_fit) - 1)
    out = y_fit[idx]
    below = new_x < x_fit[0]
    out[below] = y_fit[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=g.BASELINE_DEFAULT)
    ap.add_argument("--model-dir", default=g.WORKDIR_DEFAULT)
    args = ap.parse_args()
    import xgboost as xgb

    mdir = Path(args.model_dir)
    print("NFL_RUSHING_YARDS_ISOTONIC_RECALIBRATION_A\n" + "=" * 43)

    bst = xgb.Booster(); bst.load_model(str(mdir / "nfl_rushing_yards.json"))
    feat_cols = json.loads((mdir / "nfl_rushing_yards_columns.json").read_text())
    assert feat_cols == g.FEATURES

    dev, hol = g.load(args.baseline)
    cut = g.pick_val_cut(dev)
    va = [r for r in dev if r[1] >= cut]
    print(f"calibration-fit slice: 2023 weeks >= {cut}, n={len(va)}  (never touches 2024 holdout)")
    print(f"NOTE: n={len(va)} is small -- isotonic has far more degrees of freedom than Platt's 2 "
          f"params, real overfit risk. Judged on the holdout gate, not assumed to work.")

    def mat(rows):
        X = np.array([[r[2 + i] if r[2 + i] is not None else g.NAN for i in range(len(g.FEATURES))]
                      for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, feature_names=g.FEATURES)

    itr = (0, bst.best_iteration + 1)
    print(f"scoring with iteration_range={itr} (best_iteration={bst.best_iteration}, "
          f"num_boosted_rounds={bst.num_boosted_rounds()})")

    va_raw = bst.predict(mat(va), iteration_range=itr)
    va_y = [r[-1] for r in va]
    x_fit, y_fit = fit_isotonic(va_raw, va_y)
    n_blocks = len(set(np.round(y_fit, 6)))
    print(f"fitted isotonic step function: {n_blocks} distinct pooled levels from {len(va)} points")

    hol_raw = bst.predict(mat(hol), iteration_range=itr)
    hol_y = [r[-1] for r in hol]
    hol_recal = apply_isotonic(x_fit, y_fit, hol_raw)

    before = g.metrics(list(map(float, hol_raw)), hol_y)
    after = g.metrics(list(map(float, hol_recal)), hol_y)

    # AUC should be ~unchanged (rank-preserving map) modulo tie-breaking
    # differences introduced by pooling -- check, don't assume.
    auc_delta = abs(before["auc"] - after["auc"])
    print(f"\nAUC before={before['auc']:.4f}  after={after['auc']:.4f}  "
          f"delta={auc_delta:.5f}  (should be small -- pooling can introduce ties Platt didn't)")

    print("\n============ 2024 HOLDOUT: raw vs isotonic-recalibrated ============")
    print(f"  {'arm':22s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'raw':22s} {before['auc']:>7.4f}  {before['log_loss']:>9.5f}  {before['brier']:>7.5f} {before['ece']:>7.4f}")
    print(f"  {'isotonic recalibrated':22s} {after['auc']:>7.4f}  {after['log_loss']:>9.5f}  {after['brier']:>7.5f} {after['ece']:>7.4f}")

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
    verdict = ("NFL_RUSHING_YARDS_ISOTONIC_RECALIBRATED_PASSES_GATE"
               if passed else "NFL_RUSHING_YARDS_ISOTONIC_RECALIBRATED_STILL_FAILS_GATE")

    print("\n============ PRE-REGISTERED GATE (on isotonic-recalibrated probabilities) ============")
    print(f"  AUC >= {g.GATE['min_auc']}:            {after['auc']:.4f}  -> {c1}")
    print(f"  ECE <= {g.GATE['max_ece']}:              {after['ece']:.4f}  -> {c2}")
    print(f"  logloss gain >= {g.GATE['min_logloss_gain']}:  {d_ll:+.5f}  -> {c3}")
    print(f"  Brier better than constant:  {after['brier']:.5f} < {constant['brier']:.5f}  -> {c4}")
    print(f"  VERDICT: {verdict}")

    if passed:
        calib_path = mdir / "nfl_rushing_yards_isotonic_calibration.json"
        calib_path.write_text(json.dumps({"x": x_fit.tolist(), "y": y_fit.tolist()}))
        print(f"\nisotonic calibration written: {calib_path}")
        print("Next: re-run stability confirmation using these recalibrated probabilities before wiring live.")

    report = {"script": "NFL_RUSHING_YARDS_ISOTONIC_RECALIBRATION_A",
              "calibration_fit_n": len(va), "n_pooled_levels": n_blocks,
              "before": before, "after": after, "auc_delta": auc_delta,
              "gate": g.GATE, "passed": passed, "verdict": verdict}
    report_path = mdir / "nfl_rushing_yards_isotonic_recalibration_a_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {report_path}")
    print("Read-only on the model + baseline (writes only calibration.json on a pass). No production change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
