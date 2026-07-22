#!/usr/bin/env python3
"""
NFL_RUSHING_YARDS_CALIBRATION_POWER_GATE_A

Diagnosis that motivated this script: a Monte Carlo check showed that on
this market's 350-row holdout, a PERFECTLY calibrated model still fails the
project's standard ECE <= 0.02 bar 92.7% of the time -- the bar was
calibrated on MLB-sized holdouts (thousands of rows), and at n=350 the ECE
statistic's pure sampling noise floor (median ~0.038) sits above the bar
itself. Both recalibration attempts were therefore fighting a test that a
perfect model would fail 9 times in 10. That is a broken test, not a broken
model -- but the fix must not quietly become "grade our own homework with a
looser rubric," so this script replaces the fixed numeric bar with a proper
significance test, pre-registered here before it is first run:

  CALIBRATION CRITERION (replaces "ECE <= 0.02" for this market):
    Parametric bootstrap goodness-of-fit test (Hosmer-Lemeshow-style,
    simulation-based). Null hypothesis: the candidate's per-row
    probabilities are the true outcome probabilities. Simulate outcomes
    from those probabilities B=10,000 times, recompute ECE each time with
    the exact metric function the real gate uses, and compute
      p = fraction of null simulations with ECE >= the observed ECE.
    PASS requires p >= 0.10 -- the observed miscalibration must NOT be in
    the worst 10% of what a genuinely calibrated model would produce at
    this exact n and probability distribution. (Stricter than the usual
    0.05 convention, on purpose: the burden of proof stays on the model.)

  POWER CONTROL (the test must be shown to have teeth, not assumed):
    The same test is run on the RAW (un-recalibrated) champion, whose ECE
    0.1218 reflects the real, diagnosed 2023->2024 base-rate shift. The
    test is only trusted if it FAILS the raw model decisively (p < 0.01).
    If the control does not fail, the test has no power at this n and no
    calibration conclusion is drawn at all.

  All other gate criteria are UNCHANGED from champion_gate and must still
  hold on the Platt-recalibrated probabilities:
    AUC >= 0.58, log-loss gain vs constant >= 0.01, Brier < constant.

Everything runs on the same untouched 2024 holdout; the Platt parameters
are refit on the same 2023 internal-validation slice as before (fitting
never touches the holdout).

Read-only on the model + baseline. Writes a report, and on a full pass the
Platt calibration.json that recalibration_a would have written.

Run
---
python -u nfl_rushing_yards_calibration_power_gate_a.py \
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
from nfl_rushing_yards_recalibration_a import fit_platt, apply_platt

B_SIMS = 10_000
PASS_MIN_P = 0.10      # candidate must not be in the worst 10% of a calibrated model's ECEs
CONTROL_MAX_P = 0.01   # raw model must fail this decisively, or the test has no power
SEED = 13


def null_ece_distribution(probs, rng, b_sims):
    """ECE sampling distribution under the null 'probs are the truth':
    simulate outcomes from probs, recompute ECE with the exact same metric
    function the real gate uses."""
    out = np.empty(b_sims)
    plist = list(map(float, probs))
    for i in range(b_sims):
        sim_y = rng.binomial(1, probs).astype(float)
        out[i] = g.metrics(plist, list(sim_y))["ece"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=g.BASELINE_DEFAULT)
    ap.add_argument("--model-dir", default=g.WORKDIR_DEFAULT)
    args = ap.parse_args()
    import xgboost as xgb

    mdir = Path(args.model_dir)
    print("NFL_RUSHING_YARDS_CALIBRATION_POWER_GATE_A\n" + "=" * 43)

    bst = xgb.Booster(); bst.load_model(str(mdir / "nfl_rushing_yards.json"))
    feat_cols = json.loads((mdir / "nfl_rushing_yards_columns.json").read_text())
    assert feat_cols == g.FEATURES

    dev, hol = g.load(args.baseline)
    cut = g.pick_val_cut(dev)
    va = [r for r in dev if r[1] >= cut]
    print(f"calibration-fit slice: 2023 weeks >= {cut}, n={len(va)}  (never touches 2024 holdout)")

    def mat(rows):
        X = np.array([[r[2 + i] if r[2 + i] is not None else g.NAN for i in range(len(g.FEATURES))]
                      for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, feature_names=g.FEATURES)

    itr = (0, bst.best_iteration + 1)
    print(f"scoring with iteration_range={itr}")

    va_raw = bst.predict(mat(va), iteration_range=itr)
    va_y = [r[-1] for r in va]
    a, b = fit_platt(va_raw, va_y)
    assert a > 0
    print(f"Platt params (refit, same slice as recalibration_a): a={a:.4f}  b={b:.4f}")

    hol_raw = bst.predict(mat(hol), iteration_range=itr)
    hol_y = [r[-1] for r in hol]
    hol_recal = apply_platt(hol_raw, a, b)

    raw_m = g.metrics(list(map(float, hol_raw)), hol_y)
    recal_m = g.metrics(list(map(float, hol_recal)), hol_y)
    print(f"\nholdout n={len(hol_y)}")
    print(f"  raw:   AUC={raw_m['auc']:.4f}  ECE={raw_m['ece']:.4f}")
    print(f"  platt: AUC={recal_m['auc']:.4f}  ECE={recal_m['ece']:.4f}")

    rng = np.random.default_rng(SEED)

    # -------- POWER CONTROL first: the test must catch the raw model --------
    print(f"\n[control] simulating {B_SIMS} null ECEs with the RAW model's probs as truth ...")
    null_raw = null_ece_distribution(np.asarray(hol_raw, dtype=float), rng, B_SIMS)
    p_raw = float((null_raw >= raw_m["ece"]).mean())
    control_ok = p_raw < CONTROL_MAX_P
    print(f"[control] raw ECE={raw_m['ece']:.4f}  null median={np.median(null_raw):.4f}  "
          f"p={p_raw:.4f}  -> {'test HAS power (raw decisively fails)' if control_ok else 'TEST HAS NO POWER -- no conclusion drawn'}")

    # -------- candidate test: Platt-recalibrated --------
    print(f"\n[candidate] simulating {B_SIMS} null ECEs with the PLATT model's probs as truth ...")
    null_recal = null_ece_distribution(np.asarray(hol_recal, dtype=float), rng, B_SIMS)
    p_recal = float((null_recal >= recal_m["ece"]).mean())
    implied_bar = float(np.percentile(null_recal, 100 * (1 - PASS_MIN_P)))
    print(f"[candidate] platt ECE={recal_m['ece']:.4f}  null median={np.median(null_recal):.4f}  "
          f"p={p_recal:.4f}  (sample-size-implied bar at p={PASS_MIN_P}: ECE <= {implied_bar:.4f})")

    # -------- unchanged criteria --------
    train_rate = float(np.mean([r[-1] for r in dev if r[1] < cut]))
    constant = g.metrics([train_rate] * len(hol), hol_y)
    d_ll = constant["log_loss"] - recal_m["log_loss"]

    c_auc = recal_m["auc"] >= g.GATE["min_auc"]
    c_ll = d_ll >= g.GATE["min_logloss_gain"]
    c_brier = recal_m["brier"] < constant["brier"]
    c_control = control_ok
    c_calib = p_recal >= PASS_MIN_P

    passed = c_auc and c_ll and c_brier and c_control and c_calib
    verdict = ("NFL_RUSHING_YARDS_PLATT_PASSES_POWER_ADJUSTED_GATE"
               if passed else "NFL_RUSHING_YARDS_PLATT_FAILS_POWER_ADJUSTED_GATE")

    print("\n============ PRE-REGISTERED POWER-ADJUSTED GATE ============")
    print(f"  AUC >= {g.GATE['min_auc']}:                 {recal_m['auc']:.4f}  -> {c_auc}")
    print(f"  logloss gain >= {g.GATE['min_logloss_gain']}:       {d_ll:+.5f}  -> {c_ll}")
    print(f"  Brier better than constant:       {recal_m['brier']:.5f} < {constant['brier']:.5f}  -> {c_brier}")
    print(f"  power control (raw p < {CONTROL_MAX_P}):     p={p_raw:.4f}  -> {c_control}")
    print(f"  calibration test (p >= {PASS_MIN_P}):      p={p_recal:.4f}  -> {c_calib}")
    print(f"  VERDICT: {verdict}")

    if passed:
        (mdir / "nfl_rushing_yards_calibration.json").write_text(json.dumps({"a": a, "b": b}))
        print(f"\ncalibration params written: {mdir / 'nfl_rushing_yards_calibration.json'}")
        print("Next: stability confirmation on the recalibrated probabilities before any wiring.")

    report = {"script": "NFL_RUSHING_YARDS_CALIBRATION_POWER_GATE_A",
              "b_sims": B_SIMS, "pass_min_p": PASS_MIN_P, "control_max_p": CONTROL_MAX_P,
              "platt": {"a": a, "b": b},
              "raw": raw_m, "recalibrated": recal_m,
              "p_raw": p_raw, "p_recal": p_recal,
              "null_recal_median": float(np.median(null_recal)),
              "implied_ece_bar_at_this_n": implied_bar,
              "control_has_power": control_ok,
              "unchanged_criteria": {"auc": c_auc, "logloss_gain": c_ll, "brier": c_brier},
              "passed": passed, "verdict": verdict}
    report_path = mdir / "nfl_rushing_yards_calibration_power_gate_a_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {report_path}")
    print("Read-only on the model + baseline (writes calibration.json only on a full pass).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
