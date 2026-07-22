#!/usr/bin/env python3
"""
NFL_RUSHING_YARDS_WALKFORWARD_CALIBRATION_A

FINAL calibration attempt against this holdout, pre-committed as such: this
is the 5th evaluation touching 2024 outcomes (raw champion, Platt, isotonic,
power-gate, now this). If this design fails, the rushing-yards verdict is
"wait for 2025 data" -- no further calibration attempts against this
holdout, to stop eroding it with repeated looks.

Why this design: every prior attempt fit calibration on 2023 data only, but
the diagnosed problem IS a 2023->2024 shift (base rate 54.67% -> 65.71%).
A static map fit on last season cannot track this season by construction.
The deployment-shaped fix is ROLLING IN-SEASON RECALIBRATION -- the same
philosophy as MLB's weekly retrain: as each 2024 week finishes, refit the
Platt correction on all 2024 weeks seen so far (plus the 2023 internal-val
slice as a prior/warmup), then predict the NEXT week with it. Every week's
predictions use strictly-earlier data only -- walk-forward, no peeking.
This evaluates the system as it would actually run during a live season,
which is the honest question for a preseason deployment decision.

The underlying champion model is FROZEN (no retraining) -- only the
2-parameter Platt map updates weekly. Platt (not isotonic) per the isotonic
attempt's lesson: at these calibration-set sizes, 2 parameters is the right
capacity.

Pre-registered gate (identical criteria to the power-adjusted gate):
  1. AUC >= 0.58 on the pooled walk-forward predictions
  2. log-loss gain vs the constant arm >= 0.01
  3. Brier better than the constant arm
  4. power control: the same bootstrap test must decisively fail the raw
     champion (p < 0.01), else no conclusion is drawn
  5. calibration: bootstrap goodness-of-fit p >= 0.10 on the pooled
     walk-forward predictions

Read-only on the model + baseline. Writes only a report.

Run
---
python -u nfl_rushing_yards_walkforward_calibration_a.py \
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
PASS_MIN_P = 0.10
CONTROL_MAX_P = 0.01
SEED = 13


def null_ece_distribution(probs, rng, b_sims):
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
    print("NFL_RUSHING_YARDS_WALKFORWARD_CALIBRATION_A\n" + "=" * 44)
    print("(pre-committed: FINAL calibration attempt against this holdout)")

    bst = xgb.Booster(); bst.load_model(str(mdir / "nfl_rushing_yards.json"))
    feat_cols = json.loads((mdir / "nfl_rushing_yards_columns.json").read_text())
    assert feat_cols == g.FEATURES

    dev, hol = g.load(args.baseline)
    cut = g.pick_val_cut(dev)
    va = [r for r in dev if r[1] >= cut]

    def mat(rows):
        X = np.array([[r[2 + i] if r[2 + i] is not None else g.NAN for i in range(len(g.FEATURES))]
                      for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, feature_names=g.FEATURES)

    itr = (0, bst.best_iteration + 1)
    print(f"scoring with iteration_range={itr}")

    va_raw = np.asarray(bst.predict(mat(va), iteration_range=itr), dtype=float)
    va_y = np.asarray([r[-1] for r in va], dtype=float)

    hol_raw = np.asarray(bst.predict(mat(hol), iteration_range=itr), dtype=float)
    hol_y = np.asarray([r[-1] for r in hol], dtype=float)
    hol_weeks = np.asarray([r[1] for r in hol])

    weeks = sorted(set(hol_weeks.tolist()))
    print(f"2024 holdout: n={len(hol_y)} across weeks {weeks[0]}..{weeks[-1]}")
    print(f"warmup calibration pool: 2023 weeks >= {cut}, n={len(va)}")

    wf_pred = np.empty(len(hol_y))
    print("\nweekly walk-forward (pool = 2023 val slice + 2024 weeks < w):")
    for w in weeks:
        seen = hol_weeks < w
        pool_x = np.concatenate([va_raw, hol_raw[seen]])
        pool_y = np.concatenate([va_y, hol_y[seen]])
        a, b = fit_platt(pool_x, pool_y)
        if a <= 0:
            a, b = 1.0, 0.0
        mask = hol_weeks == w
        wf_pred[mask] = apply_platt(hol_raw[mask], a, b)
        print(f"  week {w:>2}: pool n={len(pool_y):>3} (2024 rows seen: {int(seen.sum()):>3})  "
              f"a={a:.3f} b={b:+.3f}  scored n={int(mask.sum())}")

    raw_m = g.metrics(list(map(float, hol_raw)), list(hol_y))
    wf_m = g.metrics(list(map(float, wf_pred)), list(hol_y))

    print(f"\n============ 2024 HOLDOUT: raw vs walk-forward recalibrated ============")
    print(f"  {'arm':24s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'raw':24s} {raw_m['auc']:>7.4f}  {raw_m['log_loss']:>9.5f}  {raw_m['brier']:>7.5f} {raw_m['ece']:>7.4f}")
    print(f"  {'walk-forward platt':24s} {wf_m['auc']:>7.4f}  {wf_m['log_loss']:>9.5f}  {wf_m['brier']:>7.5f} {wf_m['ece']:>7.4f}")

    print("\nwalk-forward reliability (pred -> actual):")
    for r in wf_m["reliability"]:
        print(f"   {r['bin']}  n={r['n']:>5}  pred={r['pred']:.3f}  actual={r['actual']:.3f}")

    rng = np.random.default_rng(SEED)

    print(f"\n[control] {B_SIMS} null sims, raw probs as truth ...")
    null_raw = null_ece_distribution(hol_raw, rng, B_SIMS)
    p_raw = float((null_raw >= raw_m["ece"]).mean())
    control_ok = p_raw < CONTROL_MAX_P
    print(f"[control] p={p_raw:.4f}  -> {'has power' if control_ok else 'NO POWER -- no conclusion'}")

    print(f"[candidate] {B_SIMS} null sims, walk-forward probs as truth ...")
    null_wf = null_ece_distribution(wf_pred, rng, B_SIMS)
    p_wf = float((null_wf >= wf_m["ece"]).mean())
    implied_bar = float(np.percentile(null_wf, 100 * (1 - PASS_MIN_P)))
    print(f"[candidate] wf ECE={wf_m['ece']:.4f}  null median={np.median(null_wf):.4f}  "
          f"p={p_wf:.4f}  (implied bar at p={PASS_MIN_P}: ECE <= {implied_bar:.4f})")

    train_rate = float(np.mean([r[-1] for r in dev if r[1] < cut]))
    constant = g.metrics([train_rate] * len(hol_y), list(hol_y))
    d_ll = constant["log_loss"] - wf_m["log_loss"]

    c_auc = wf_m["auc"] >= g.GATE["min_auc"]
    c_ll = d_ll >= g.GATE["min_logloss_gain"]
    c_brier = wf_m["brier"] < constant["brier"]
    c_control = control_ok
    c_calib = p_wf >= PASS_MIN_P
    passed = c_auc and c_ll and c_brier and c_control and c_calib
    verdict = ("NFL_RUSHING_YARDS_WALKFORWARD_PASSES_GATE"
               if passed else "NFL_RUSHING_YARDS_WALKFORWARD_FAILS_GATE_WAIT_FOR_2025_DATA")

    print("\n============ PRE-REGISTERED GATE (walk-forward, final attempt) ============")
    print(f"  AUC >= {g.GATE['min_auc']}:                 {wf_m['auc']:.4f}  -> {c_auc}")
    print(f"  logloss gain >= {g.GATE['min_logloss_gain']}:       {d_ll:+.5f}  -> {c_ll}")
    print(f"  Brier better than constant:       {wf_m['brier']:.5f} < {constant['brier']:.5f}  -> {c_brier}")
    print(f"  power control (raw p < {CONTROL_MAX_P}):     p={p_raw:.4f}  -> {c_control}")
    print(f"  calibration test (p >= {PASS_MIN_P}):      p={p_wf:.4f}  -> {c_calib}")
    print(f"  VERDICT: {verdict}")

    report = {"script": "NFL_RUSHING_YARDS_WALKFORWARD_CALIBRATION_A",
              "final_attempt_against_this_holdout": True,
              "b_sims": B_SIMS, "pass_min_p": PASS_MIN_P, "control_max_p": CONTROL_MAX_P,
              "raw": raw_m, "walkforward": wf_m,
              "p_raw": p_raw, "p_walkforward": p_wf,
              "implied_ece_bar_at_this_n": implied_bar,
              "constant": {"log_loss": constant["log_loss"], "brier": constant["brier"]},
              "passed": passed, "verdict": verdict}
    report_path = mdir / "nfl_rushing_yards_walkforward_calibration_a_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {report_path}")
    print("Read-only on the model + baseline. No production change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
