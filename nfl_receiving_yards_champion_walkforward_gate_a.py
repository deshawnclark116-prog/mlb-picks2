#!/usr/bin/env python3
"""
NFL_RECEIVING_YARDS_CHAMPION_WALKFORWARD_GATE_A

Trains and formally validates the FIRST receiving-yards model (WR-only,
line from the baseline's 2023-dev diagnostic) -- and evaluates it the way
it would actually be deployed, in ONE pre-registered look at the 2024
holdout. Every lesson from the first two markets is built in from day one
instead of discovered afterward:

  - walk-forward weekly Platt recalibration (rushing's lesson: static
    calibration cannot track a season-over-season shift; the deployment
    shape is a frozen model + weekly refit on the season in progress)
  - power-adjusted calibration testing (a fixed ECE bar has no defined
    power at arbitrary n; the criterion is a parametric-bootstrap
    goodness-of-fit p-value)
  - SYNTHETIC power control (receptions' lesson: using the raw model as
    the control proves nothing when the raw model happens to be decently
    calibrated -- instead, a deliberately distorted copy of the candidate
    (logit shift +0.5, a ~12-point miscalibration at p=0.5) must be
    decisively rejected, or the test is declared powerless and no
    calibration conclusion is drawn)

Pre-registered gate (written before this script has ever been run):
  1. AUC >= 0.58 on pooled walk-forward 2024 predictions
  2. log-loss gain vs the constant arm >= 0.01
  3. Brier better than the constant arm
  4. power control: bootstrap must reject the logit+0.5 distorted copy
     of the candidate (p < 0.01)
  5. calibration: bootstrap goodness-of-fit p >= 0.10 on the pooled
     walk-forward predictions

Read-only on the baseline. Writes the trained model + report to its own
work dir. No production wiring.

Run
---
python -u nfl_receiving_yards_champion_walkforward_gate_a.py \
    --baseline nfl_models/nfl_receiving_yards_clean_baseline_a_work/baseline.sqlite \
    --workdir nfl_models/nfl_receiving_yards_champion_walkforward_gate_a_work
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# identical math/discipline to the rushing pipeline, imported not re-derived
import nfl_rushing_yards_champion_gate_a as g
from nfl_rushing_yards_recalibration_a import fit_platt, apply_platt, logit, sigmoid
from nfl_receiving_yards_clean_baseline_a import MODEL_COLUMNS as FEATURES

BASELINE_DEFAULT = "nfl_models/nfl_receiving_yards_clean_baseline_a_work/baseline.sqlite"
WORKDIR_DEFAULT = "nfl_models/nfl_receiving_yards_champion_walkforward_gate_a_work"

B_SIMS = 10_000
PASS_MIN_P = 0.10
CONTROL_MAX_P = 0.01
CONTROL_LOGIT_SHIFT = 0.5
SEED = 13


def load(baseline_path):
    con = sqlite3.connect(f"file:{baseline_path}?mode=ro", uri=True)
    cols = ["season", "week"] + FEATURES + ["over_line"]
    rows = con.execute(
        f"SELECT {', '.join(cols)} FROM nfl_receiving_yards_baseline").fetchall()
    con.close()
    dev = [r for r in rows if r[0] == 2023]
    hol = [r for r in rows if r[0] == 2024]
    return dev, hol


def ece_only(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    n = len(y)
    total = 0.0
    for b in range(10):
        m = (p >= b / 10) & (p < (b + 1) / 10) if b < 9 else (p >= 0.9)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        total += abs(float(p[m].mean()) - float(y[m].mean())) * cnt / n
    return total


def bootstrap_p(probs, observed_ece, rng, b_sims=B_SIMS):
    probs = np.asarray(probs, dtype=float)
    worse = 0
    for _ in range(b_sims):
        sim_y = rng.binomial(1, probs).astype(float)
        if ece_only(probs, sim_y) >= observed_ece:
            worse += 1
    return worse / b_sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=BASELINE_DEFAULT)
    ap.add_argument("--workdir", default=WORKDIR_DEFAULT)
    args = ap.parse_args()
    import xgboost as xgb

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    print("NFL_RECEIVING_YARDS_CHAMPION_WALKFORWARD_GATE_A\n" + "=" * 47)
    print("(one pre-registered look at the 2024 holdout -- train + walk-forward eval)")

    dev, hol = load(args.baseline)
    cut = g.pick_val_cut(dev)
    tr = [r for r in dev if r[1] < cut]
    va = [r for r in dev if r[1] >= cut]
    print(f"2023 dev: train weeks < {cut}: {len(tr)}   internal val weeks >= {cut}: {len(va)}")
    print(f"2024 holdout: n={len(hol)}")

    def mat(rows, with_label=True):
        X = np.array([[r[2 + i] if r[2 + i] is not None else g.NAN
                       for i in range(len(FEATURES))] for r in rows], dtype=np.float32)
        d = xgb.DMatrix(X, feature_names=FEATURES)
        if with_label:
            d.set_label(np.array([r[-1] for r in rows], dtype=float))
        return d

    dtr, dva = mat(tr), mat(va)
    bst = xgb.train(g.PARAMS, dtr, num_boost_round=800,
                    evals=[(dva, "val")], early_stopping_rounds=30, verbose_eval=False)
    itr = (0, bst.best_iteration + 1)
    print(f"trained: best_iteration={bst.best_iteration}  scoring with iteration_range={itr}")

    va_raw = np.asarray(bst.predict(dva, iteration_range=itr), dtype=float)
    va_y = np.asarray([r[-1] for r in va], dtype=float)
    hol_raw = np.asarray(bst.predict(mat(hol, with_label=False), iteration_range=itr), dtype=float)
    hol_y = np.asarray([r[-1] for r in hol], dtype=float)
    hol_weeks = np.asarray([r[1] for r in hol])

    weeks = sorted(set(hol_weeks.tolist()))
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
        print(f"  week {w:>2}: pool n={len(pool_y):>4}  a={a:.3f} b={b:+.3f}  scored n={int(mask.sum())}")

    raw_m = g.metrics(list(map(float, hol_raw)), list(hol_y))
    wf_m = g.metrics(list(map(float, wf_pred)), list(hol_y))
    assert abs(round(ece_only(wf_pred, hol_y), 4) - wf_m["ece"]) < 1e-9

    print(f"\n============ 2024 HOLDOUT ============")
    print(f"  {'arm':20s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'raw (info only)':20s} {raw_m['auc']:>7.4f}  {raw_m['log_loss']:>9.5f}  {raw_m['brier']:>7.5f} {raw_m['ece']:>7.4f}")
    print(f"  {'walk-forward platt':20s} {wf_m['auc']:>7.4f}  {wf_m['log_loss']:>9.5f}  {wf_m['brier']:>7.5f} {wf_m['ece']:>7.4f}")

    print("\nwalk-forward reliability (pred -> actual):")
    for r in wf_m["reliability"]:
        print(f"   {r['bin']}  n={r['n']:>5}  pred={r['pred']:.3f}  actual={r['actual']:.3f}")

    rng = np.random.default_rng(SEED)

    # synthetic power control: a known ~12-point distortion of the candidate
    distorted = sigmoid(logit(wf_pred) + CONTROL_LOGIT_SHIFT)
    d_ece = ece_only(distorted, hol_y)
    print(f"\n[control] distorted copy (logit+{CONTROL_LOGIT_SHIFT}): ECE={d_ece:.4f}  "
          f"bootstrapping {B_SIMS} sims ...")
    p_control = bootstrap_p(distorted, d_ece, rng)
    control_ok = p_control < CONTROL_MAX_P
    print(f"[control] p={p_control:.4f}  -> {'test has power' if control_ok else 'NO POWER -- no conclusion'}")

    print(f"[candidate] bootstrapping walk-forward ({B_SIMS} sims) ...")
    p_wf = bootstrap_p(wf_pred, wf_m["ece"], rng)
    implied_bar = None
    print(f"[candidate] wf ECE={wf_m['ece']:.4f}  p={p_wf:.4f}")

    train_rate = float(np.mean([r[-1] for r in tr]))
    constant = g.metrics([train_rate] * len(hol_y), list(hol_y))
    d_ll = constant["log_loss"] - wf_m["log_loss"]

    c_auc = wf_m["auc"] >= g.GATE["min_auc"]
    c_ll = d_ll >= g.GATE["min_logloss_gain"]
    c_brier = wf_m["brier"] < constant["brier"]
    c_control = control_ok
    c_calib = p_wf >= PASS_MIN_P
    passed = c_auc and c_ll and c_brier and c_control and c_calib
    verdict = ("NFL_RECEIVING_YARDS_WALKFORWARD_PASSES_GATE"
               if passed else "NFL_RECEIVING_YARDS_WALKFORWARD_FAILS_GATE")

    print("\n============ PRE-REGISTERED GATE ============")
    print(f"  AUC >= {g.GATE['min_auc']}:                 {wf_m['auc']:.4f}  -> {c_auc}")
    print(f"  logloss gain >= {g.GATE['min_logloss_gain']}:       {d_ll:+.5f}  -> {c_ll}")
    print(f"  Brier better than constant:       {wf_m['brier']:.5f} < {constant['brier']:.5f}  -> {c_brier}")
    print(f"  power control (distorted p < {CONTROL_MAX_P}): p={p_control:.4f}  -> {c_control}")
    print(f"  calibration test (p >= {PASS_MIN_P}):      p={p_wf:.4f}  -> {c_calib}")
    print(f"  VERDICT: {verdict}")

    bst.save_model(str(work / "nfl_receiving_yards.json"))
    (work / "nfl_receiving_yards_columns.json").write_text(json.dumps(FEATURES))
    report = {"script": "NFL_RECEIVING_YARDS_CHAMPION_WALKFORWARD_GATE_A",
              "features": FEATURES, "params": g.PARAMS,
              "b_sims": B_SIMS, "pass_min_p": PASS_MIN_P, "control_max_p": CONTROL_MAX_P,
              "control_logit_shift": CONTROL_LOGIT_SHIFT,
              "best_iteration": int(bst.best_iteration),
              "raw": raw_m, "walkforward": wf_m,
              "p_control": p_control, "p_walkforward": p_wf,
              "constant": {"log_loss": constant["log_loss"], "brier": constant["brier"]},
              "passed": passed, "verdict": verdict}
    (work / "nfl_receiving_yards_champion_walkforward_gate_a_report.json").write_text(
        json.dumps(report, indent=2))
    print(f"\nmodel + report written to {work}")
    print("No production wiring. Read-only on the baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
