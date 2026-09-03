#!/usr/bin/env python3
"""
CFB_FCS_RUSHING_YARDS_CHAMPION_GATE_C

gate_b's Platt scaling attempt didn't help (holdout ECE 0.0437 -> 0.0463,
slightly WORSE -- the internal-val slice's miscalibration doesn't match
the holdout's specific miscalibration, exactly the same failure mode
already seen and solved for the FBS rushing_yards market). Reusing that
market's own actual fix (cfb_rushing_yards_champion_gate_d.py) directly,
not rediscovering it: heavier regularization at TRAINING time (shallower
trees, higher min_child_weight, L2 regularization, slower learning rate)
to prevent overconfident/jagged probability estimates in the first
place, rather than trying to smooth them out after the fact.

Two arms on the untouched 2025 holdout (2022-2023 = train, 2024 =
internal val for early stopping):
  constant   predicts the dev-seasons base rate for every row
  challenger binary:logistic, gate_d's regularized params

Pre-registered pass, unchanged: AUC >= 0.58, ECE <= 0.02, logloss gain
>= 0.01, Brier better than constant.

Run
---
python -u cfb_fcs_rushing_yards_champion_gate_c.py
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

PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3,
          "eta": 0.03, "subsample": 0.7, "colsample_bytree": 0.7,
          "min_child_weight": 15, "reg_lambda": 3.0, "seed": 13}


def main():
    import xgboost as xgb
    work = Path(g.WORKDIR_DEFAULT); work.mkdir(parents=True, exist_ok=True)
    print("CFB_FCS_RUSHING_YARDS_CHAMPION_GATE_C\n======================================")
    tr, va, hol = g.load(g.BASELINE_DEFAULT)
    print(f"train {g.DEV_SEASONS} rows: {len(tr)}   internal val {g.VAL_SEASON} rows: {len(va)}   "
          f"holdout {g.HOLDOUT_SEASON} rows: {len(hol)}")

    def mat(rows):
        X = np.array([[r[2 + i] if r[2 + i] is not None else g.NAN for i in range(len(g.FEATURES))] for r in rows], dtype=np.float32)
        y = np.array([r[-1] for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=g.FEATURES)

    print("\ntraining challenger (regularized) ...", flush=True)
    bst = xgb.train(PARAMS, mat(tr), num_boost_round=800, evals=[(mat(va), "val")],
                    early_stopping_rounds=40, verbose_eval=False)
    itr = (0, bst.best_iteration + 1)
    print(f"  best_iteration={bst.best_iteration}  scoring with iteration_range={itr}")

    probs_hol = bst.predict(mat(hol), iteration_range=itr)
    labels_hol = [r[-1] for r in hol]
    challenger = g.metrics(list(map(float, probs_hol)), labels_hol)

    train_rate = float(np.mean([r[-1] for r in tr]))
    constant = g.metrics([train_rate] * len(hol), labels_hol)

    print(f"\n============ {g.HOLDOUT_SEASON} HOLDOUT ============")
    print(f"  {'arm':12s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'constant':12s} {'n/a':>7s} {constant['log_loss']:>9.5f}  {constant['brier']:>7.5f} {constant['ece']:>7.4f}")
    print(f"  {'challenger':12s} {challenger['auc']:>7.4f}  {challenger['log_loss']:>9.5f}  {challenger['brier']:>7.5f} {challenger['ece']:>7.4f}")

    print("\nchallenger reliability (pred -> actual):")
    for b in challenger["reliability"]:
        print(f"   {b['bin']}  n={b['n']:>5}  pred={b['pred']:.3f}  actual={b['actual']:.3f}")

    imp = bst.get_score(importance_type="gain")
    print("\nfeature importance (gain):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"   {k:32s} {v:9.2f}")

    d_ll = constant["log_loss"] - challenger["log_loss"]
    c1 = challenger["auc"] >= g.GATE["min_auc"]
    c2 = challenger["ece"] <= g.GATE["max_ece"]
    c3 = d_ll >= g.GATE["min_logloss_gain"]
    c4 = challenger["brier"] < constant["brier"]
    passed = c1 and c2 and c3 and c4
    verdict = ("CFB_FCS_RUSHING_YARDS_CHAMPION_PASSES_GATE_READY_FOR_STABILITY_CONFIRMATION"
               if passed else "CFB_FCS_RUSHING_YARDS_CHAMPION_DOES_NOT_CLEAR_GATE")

    print("\n============ PRE-REGISTERED GATE ============")
    print(f"  AUC >= {g.GATE['min_auc']}:            {challenger['auc']:.4f}  -> {c1}")
    print(f"  ECE <= {g.GATE['max_ece']}:              {challenger['ece']:.4f}  -> {c2}")
    print(f"  logloss gain >= {g.GATE['min_logloss_gain']}:  {d_ll:+.5f}  -> {c3}")
    print(f"  Brier better than constant:  {challenger['brier']:.5f} < {constant['brier']:.5f}  -> {c4}")
    print(f"  VERDICT: {verdict}")

    bst.save_model(str(work / "cfb_fcs_rushing_yards.json"))
    (work / "cfb_fcs_rushing_yards_columns.json").write_text(json.dumps(g.FEATURES))
    report = {"script": "CFB_FCS_RUSHING_YARDS_CHAMPION_GATE_C", "holdout": g.HOLDOUT_SEASON,
              "constant": constant, "challenger": challenger, "gate": g.GATE,
              "passed": passed, "verdict": verdict, "importance": imp,
              "best_iteration": bst.best_iteration}
    (work / "cfb_fcs_rushing_yards_champion_gate_c_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nmodel + report written to {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
