#!/usr/bin/env python3
"""
CFB_FCS_RUSHING_YARDS_CHAMPION_GATE_B

gate_a passed AUC/logloss/Brier (AUC 0.6526, real signal -- comparable to
the FBS market's own 0.6364) but failed ECE (0.0437 vs 0.02 bar). Same
playbook as the FBS market's own gate_b: try Platt scaling fit on the
2024 internal-val slice before reaching for anything more invasive
(population re-scoping, pooling more holdout seasons, etc).

Platt map fit on the 2024 INTERNAL VAL slice only (never touches the 2025
holdout for fitting) -- same rule as everywhere else in this repo.

Run
---
python -u cfb_fcs_rushing_yards_champion_gate_b.py
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


def fit_platt(raw_probs, labels, iters=2000, lr=0.1):
    p = np.clip(np.asarray(raw_probs, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    y = np.asarray(labels, dtype=float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        z = a * x + b
        pred = 1.0 / (1.0 + np.exp(-z))
        grad_a = np.mean((pred - y) * x)
        grad_b = np.mean(pred - y)
        a -= lr * grad_a
        b -= lr * grad_b
    return a, b


def apply_platt(raw_probs, a, b):
    p = np.clip(np.asarray(raw_probs, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-(a * x + b)))


def main():
    import xgboost as xgb
    work = Path(g.WORKDIR_DEFAULT); work.mkdir(parents=True, exist_ok=True)
    print("CFB_FCS_RUSHING_YARDS_CHAMPION_GATE_B\n======================================")
    tr, va, hol = g.load(g.BASELINE_DEFAULT)

    def mat(rows):
        X = np.array([[r[2 + i] if r[2 + i] is not None else g.NAN for i in range(len(g.FEATURES))] for r in rows], dtype=np.float32)
        y = np.array([r[-1] for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=g.FEATURES)

    bst = xgb.train(g.PARAMS, mat(tr), num_boost_round=800, evals=[(mat(va), "val")],
                    early_stopping_rounds=40, verbose_eval=False)
    itr = (0, bst.best_iteration + 1)
    print(f"best_iteration={bst.best_iteration}  scoring with iteration_range={itr}")

    probs_va_raw = bst.predict(mat(va), iteration_range=itr)
    labels_va = [r[-1] for r in va]
    a, b = fit_platt(probs_va_raw, labels_va)
    print(f"\nPlatt scaling fit on internal val (n={len(va)}): a={a:.4f} b={b:.4f}")
    va_before = g.metrics(list(map(float, probs_va_raw)), labels_va)
    va_after = g.metrics(list(map(float, apply_platt(probs_va_raw, a, b))), labels_va)
    print(f"  internal val ECE before={va_before['ece']:.4f}  after={va_after['ece']:.4f}")

    probs_hol_raw = bst.predict(mat(hol), iteration_range=itr)
    labels_hol = [r[-1] for r in hol]
    probs_hol_cal = apply_platt(probs_hol_raw, a, b)
    challenger_raw = g.metrics(list(map(float, probs_hol_raw)), labels_hol)
    challenger = g.metrics(list(map(float, probs_hol_cal)), labels_hol)

    train_rate = float(np.mean([r[-1] for r in tr]))
    constant = g.metrics([train_rate] * len(hol), labels_hol)

    print(f"\n============ {g.HOLDOUT_SEASON} HOLDOUT ============")
    print(f"  {'arm':16s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'constant':16s} {'n/a':>7s} {constant['log_loss']:>9.5f}  {constant['brier']:>7.5f} {constant['ece']:>7.4f}")
    print(f"  {'challenger(raw)':16s} {challenger_raw['auc']:>7.4f}  {challenger_raw['log_loss']:>9.5f}  {challenger_raw['brier']:>7.5f} {challenger_raw['ece']:>7.4f}")
    print(f"  {'challenger(cal)':16s} {challenger['auc']:>7.4f}  {challenger['log_loss']:>9.5f}  {challenger['brier']:>7.5f} {challenger['ece']:>7.4f}")

    print("\nchallenger(cal) reliability:")
    for bexp in challenger["reliability"]:
        print(f"   {bexp['bin']}  n={bexp['n']:>5}  pred={bexp['pred']:.3f}  actual={bexp['actual']:.3f}")

    d_ll = constant["log_loss"] - challenger["log_loss"]
    c1 = challenger["auc"] >= g.GATE["min_auc"]
    c2 = challenger["ece"] <= g.GATE["max_ece"]
    c3 = d_ll >= g.GATE["min_logloss_gain"]
    c4 = challenger["brier"] < constant["brier"]
    passed = c1 and c2 and c3 and c4
    verdict = ("CFB_FCS_RUSHING_YARDS_CHAMPION_PASSES_GATE_READY_FOR_STABILITY_CONFIRMATION"
               if passed else "CFB_FCS_RUSHING_YARDS_CHAMPION_DOES_NOT_CLEAR_GATE")

    print("\n============ PRE-REGISTERED GATE (calibrated arm) ============")
    print(f"  AUC >= {g.GATE['min_auc']}:            {challenger['auc']:.4f}  -> {c1}")
    print(f"  ECE <= {g.GATE['max_ece']}:              {challenger['ece']:.4f}  -> {c2}")
    print(f"  logloss gain >= {g.GATE['min_logloss_gain']}:  {d_ll:+.5f}  -> {c3}")
    print(f"  Brier better than constant:  {challenger['brier']:.5f} < {constant['brier']:.5f}  -> {c4}")
    print(f"  VERDICT: {verdict}")

    bst.save_model(str(work / "cfb_fcs_rushing_yards.json"))
    (work / "cfb_fcs_rushing_yards_columns.json").write_text(json.dumps(g.FEATURES))
    (work / "cfb_fcs_rushing_yards_platt.json").write_text(json.dumps({"a": a, "b": b}))
    report = {"script": "CFB_FCS_RUSHING_YARDS_CHAMPION_GATE_B", "holdout": g.HOLDOUT_SEASON,
              "constant": constant, "challenger_raw": challenger_raw, "challenger_calibrated": challenger,
              "platt": {"a": a, "b": b}, "gate": g.GATE,
              "passed": passed, "verdict": verdict, "best_iteration": bst.best_iteration}
    (work / "cfb_fcs_rushing_yards_champion_gate_b_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nmodel + report written to {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
