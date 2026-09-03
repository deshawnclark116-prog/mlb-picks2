#!/usr/bin/env python3
"""
CFB_FCS_RUSHING_YARDS_ISOTONIC_RECALIBRATION_A

Three plain fixes already tried and all fell short: gate_a (plain, ECE
0.0437), gate_b (Platt, ECE 0.0463 -- WORSE), gate_c (regularization,
reusing the FBS rushing_yards market's own actual fix, ECE 0.0541 --
also WORSE), gate_d (pooled 2024+2025 holdout, ECE 0.0722 -- worse
again, and the reliability tables across ALL of these show the SAME
shape: a consistent OVER-prediction concentrated in the 0.4-0.7 raw-
probability range, not season-to-season drift or plain overfitting).
AUC is consistently strong (0.65-0.68) across every attempt -- real,
stable ranking signal -- so the problem is specifically probability
calibration, and specifically a curve shape a single 2-parameter Platt
sigmoid can't reach (same reasoning as nfl_rushing_yards_isotonic_
recalibration_a.py, reused directly here). Isotonic regression fits an
arbitrary non-decreasing step function instead, which can match this
kind of localized over-prediction where Platt can't.

Uses gate_a's model (n=796 train, n=452 val, plain params -- gate_a had
the least-bad ECE of the four attempts, so it's the most reasonable
base to recalibrate rather than starting from a worse one) and fits the
isotonic map on the SAME 2024 internal-val slice gate_a/b already used
for validation-only purposes -- never touches the 2025 holdout for
fitting. Real risk, stated up front: n=452 is a real fitting slice
(more than the NFL script's n=75 that this technique was originally
proven on) but isotonic's degrees of freedom still exceed Platt's, so
this is judged strictly by the untouched 2025 holdout gate, not assumed
to work because it's a fancier method.

Run
---
python -u cfb_fcs_rushing_yards_isotonic_recalibration_a.py
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import cfb_fcs_rushing_yards_champion_gate_a as g


def fit_isotonic(raw_probs, labels):
    order = np.argsort(raw_probs, kind="mergesort")
    x = np.asarray(raw_probs, dtype=float)[order]
    y = np.asarray(labels, dtype=float)[order]

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

    y_step = np.empty(len(y))
    starts = block_start + [len(y)]
    for k in range(len(block_val)):
        y_step[starts[k]:starts[k + 1]] = block_val[k]
    return x, y_step


def apply_isotonic(x_fit, y_fit, new_x):
    new_x = np.asarray(new_x, dtype=float)
    idx = np.searchsorted(x_fit, new_x, side="right") - 1
    idx = np.clip(idx, 0, len(y_fit) - 1)
    out = y_fit[idx]
    below = new_x < x_fit[0]
    out[below] = y_fit[0]
    return out


def main():
    import xgboost as xgb
    work = Path(g.WORKDIR_DEFAULT); work.mkdir(parents=True, exist_ok=True)
    print("CFB_FCS_RUSHING_YARDS_ISOTONIC_RECALIBRATION_A\n" + "=" * 47)
    tr, va, hol = g.load(g.BASELINE_DEFAULT)

    def mat(rows):
        X = np.array([[r[2 + i] if r[2 + i] is not None else g.NAN for i in range(len(g.FEATURES))] for r in rows], dtype=np.float32)
        y = np.array([r[-1] for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=g.FEATURES)

    bst = xgb.train(g.PARAMS, mat(tr), num_boost_round=800, evals=[(mat(va), "val")],
                    early_stopping_rounds=40, verbose_eval=False)
    itr = (0, bst.best_iteration + 1)
    print(f"best_iteration={bst.best_iteration}")

    probs_va_raw = bst.predict(mat(va), iteration_range=itr)
    labels_va = [r[-1] for r in va]
    x_fit, y_fit = fit_isotonic(probs_va_raw, labels_va)
    va_before = g.metrics(list(map(float, probs_va_raw)), labels_va)
    va_after = g.metrics(list(map(float, apply_isotonic(x_fit, y_fit, probs_va_raw))), labels_va)
    print(f"internal val (n={len(va)}) ECE before={va_before['ece']:.4f}  after={va_after['ece']:.4f}")

    probs_hol_raw = bst.predict(mat(hol), iteration_range=itr)
    labels_hol = [r[-1] for r in hol]
    probs_hol_cal = apply_isotonic(x_fit, y_fit, probs_hol_raw)
    challenger_raw = g.metrics(list(map(float, probs_hol_raw)), labels_hol)
    challenger = g.metrics(list(map(float, probs_hol_cal)), labels_hol)

    # AUC must be rank-preserving under a non-decreasing map -- verify, not assume
    auc_diff = abs(challenger["auc"] - challenger_raw["auc"])
    print(f"AUC raw={challenger_raw['auc']:.4f}  calibrated={challenger['auc']:.4f}  "
          f"(diff={auc_diff:.4f}, should be ~0 up to ties)")

    train_rate = float(np.mean([r[-1] for r in tr]))
    constant = g.metrics([train_rate] * len(hol), labels_hol)

    print(f"\n============ {g.HOLDOUT_SEASON} HOLDOUT ============")
    print(f"  {'arm':16s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'constant':16s} {'n/a':>7s} {constant['log_loss']:>9.5f}  {constant['brier']:>7.5f} {constant['ece']:>7.4f}")
    print(f"  {'challenger(raw)':16s} {challenger_raw['auc']:>7.4f}  {challenger_raw['log_loss']:>9.5f}  {challenger_raw['brier']:>7.5f} {challenger_raw['ece']:>7.4f}")
    print(f"  {'challenger(iso)':16s} {challenger['auc']:>7.4f}  {challenger['log_loss']:>9.5f}  {challenger['brier']:>7.5f} {challenger['ece']:>7.4f}")

    print("\nchallenger(iso) reliability:")
    for b in challenger["reliability"]:
        print(f"   {b['bin']}  n={b['n']:>5}  pred={b['pred']:.3f}  actual={b['actual']:.3f}")

    d_ll = constant["log_loss"] - challenger["log_loss"]
    c1 = challenger["auc"] >= g.GATE["min_auc"]
    c2 = challenger["ece"] <= g.GATE["max_ece"]
    c3 = d_ll >= g.GATE["min_logloss_gain"]
    c4 = challenger["brier"] < constant["brier"]
    passed = c1 and c2 and c3 and c4
    verdict = ("CFB_FCS_RUSHING_YARDS_CHAMPION_PASSES_GATE_READY_FOR_STABILITY_CONFIRMATION"
               if passed else "CFB_FCS_RUSHING_YARDS_CHAMPION_DOES_NOT_CLEAR_GATE")

    print("\n============ PRE-REGISTERED GATE (isotonic arm) ============")
    print(f"  AUC >= {g.GATE['min_auc']}:            {challenger['auc']:.4f}  -> {c1}")
    print(f"  ECE <= {g.GATE['max_ece']}:              {challenger['ece']:.4f}  -> {c2}")
    print(f"  logloss gain >= {g.GATE['min_logloss_gain']}:  {d_ll:+.5f}  -> {c3}")
    print(f"  Brier better than constant:  {challenger['brier']:.5f} < {constant['brier']:.5f}  -> {c4}")
    print(f"  VERDICT: {verdict}")

    if passed:
        bst.save_model(str(work / "cfb_fcs_rushing_yards.json"))
        (work / "cfb_fcs_rushing_yards_columns.json").write_text(json.dumps(g.FEATURES))
        (work / "cfb_fcs_rushing_yards_isotonic.json").write_text(
            json.dumps({"x": x_fit.tolist(), "y": y_fit.tolist()}))
    report = {"script": "CFB_FCS_RUSHING_YARDS_ISOTONIC_RECALIBRATION_A", "holdout": g.HOLDOUT_SEASON,
              "constant": constant, "challenger_raw": challenger_raw, "challenger_isotonic": challenger,
              "gate": g.GATE, "passed": passed, "verdict": verdict, "best_iteration": bst.best_iteration}
    (work / "cfb_fcs_rushing_yards_isotonic_recalibration_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
