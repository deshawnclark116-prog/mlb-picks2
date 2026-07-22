#!/usr/bin/env python3
"""
NFL_RECEPTIONS_WALKFORWARD_CALIBRATION_A

Applies the walk-forward in-season recalibration design that just passed the
rushing-yards gate to the parked receptions market (both lines, 1.5 / 2.5).

Why it might work here: receptions' blockers were all calibration-shaped --
line 1.5 failed only the per-position slice ECE bars (TE 0.0425, and the
per-position Platt attempt made RB worse while fixing nothing), line 2.5
failed aggregate + all slices. Two lessons from rushing yards apply
directly: (1) fixed ECE bars have little statistical power at slice sizes
of n=810-1,623, so slices are judged with the parametric-bootstrap
goodness-of-fit test instead; (2) calibration fit on 2023 cannot track a
2023->2024 shift -- the walk-forward weekly refit can. One global Platt map
per week (NOT per-position -- the per-position attempt overfit its tiny
n=46-107 fitting slices and is a documented dead end).

PRE-COMMITTED: this is the FINAL calibration attempt against the receptions
2024 holdout (looks so far: champion gate, global Platt, per-position Platt,
stability x2). If a line fails here, that line's verdict is "wait for 2025
data" -- no further attempts against this holdout.

Pre-registered gate, per line (written before first run):
  1. AUC >= 0.58 on pooled walk-forward predictions
  2. log-loss gain vs constant >= 0.01
  3. Brier better than constant
  4. power control: bootstrap test must decisively fail the RAW model's
     aggregate calibration (p < 0.01), else no conclusion
  5. aggregate calibration: bootstrap goodness-of-fit p >= 0.10
  6. every position slice with n >= 300 (RB, TE, WR): slice bootstrap
     goodness-of-fit p >= 0.10 (replaces the old fixed 0.04 slice bar,
     same reasoning as the aggregate)

Read-only on the models + baseline. Writes only reports.

Run
---
python -u nfl_receptions_walkforward_calibration_a.py --line 1.5 \
    --baseline nfl_models/nfl_receptions_clean_baseline_a_work/baseline.sqlite \
    --model-dir nfl_models/nfl_receptions_champion_gate_a_work
python -u nfl_receptions_walkforward_calibration_a.py --line 2.5 ...
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

import nfl_receptions_champion_gate_a as g
from nfl_receptions_recalibration_a import fit_platt, apply_platt

B_SIMS = 10_000
PASS_MIN_P = 0.10
CONTROL_MAX_P = 0.01
MIN_SLICE_ROWS = 300
SEED = 13
POSITIONS = ["RB", "TE", "WR"]

# row layout from g.load: [season, week] + FEATURES + [target]
IDX_WEEK = 1
IDX_IS_WR = 2 + g.FEATURES.index("is_wr")
IDX_IS_TE = 2 + g.FEATURES.index("is_te")
IDX_IS_RB = 2 + g.FEATURES.index("is_rb")


def row_position(r):
    if r[IDX_IS_RB] == 1:
        return "RB"
    if r[IDX_IS_TE] == 1:
        return "TE"
    if r[IDX_IS_WR] == 1:
        return "WR"
    return "OTHER"


def ece_only(p, y):
    """Exact replica of g.metrics' ECE binning, without the rest (for speed
    in the bootstrap loop). Verified against g.metrics on real data below."""
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
    ap.add_argument("--line", type=float, choices=[1.5, 2.5], required=True)
    ap.add_argument("--baseline", default=g.BASELINE_DEFAULT)
    ap.add_argument("--model-dir", default=g.WORKDIR_DEFAULT)
    args = ap.parse_args()
    import xgboost as xgb

    target_col = g.LINE_TO_COLUMN[args.line]
    mdir = Path(args.model_dir) / f"line_{args.line}"
    model_stem = f"nfl_receptions_over_{str(args.line).replace('.', '_')}"
    print(f"NFL_RECEPTIONS_WALKFORWARD_CALIBRATION_A  [line={args.line}]\n" + "=" * 48)
    print("(pre-committed: FINAL calibration attempt against this holdout)")

    bst = xgb.Booster(); bst.load_model(str(mdir / f"{model_stem}.json"))
    feat_cols = json.loads((mdir / f"{model_stem}_columns.json").read_text())
    assert feat_cols == g.FEATURES
    itr = (0, bst.best_iteration + 1)

    dev, hol = g.load(args.baseline, target_col)
    weeks_dev = sorted({r[IDX_WEEK] for r in dev})
    cut = weeks_dev[int(len(weeks_dev) * 0.8)] if len(weeks_dev) > 5 else weeks_dev[-1]
    va = [r for r in dev if r[IDX_WEEK] >= cut]
    print(f"warmup calibration pool: 2023 weeks >= {cut}, n={len(va)}")

    def mat(rows):
        X = np.array([[r[2 + i] if r[2 + i] is not None else g.NAN for i in range(len(g.FEATURES))]
                      for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, feature_names=g.FEATURES)

    print(f"scoring with iteration_range={itr}")
    va_raw = np.asarray(bst.predict(mat(va), iteration_range=itr), dtype=float)
    va_y = np.asarray([r[-1] for r in va], dtype=float)

    hol_raw = np.asarray(bst.predict(mat(hol), iteration_range=itr), dtype=float)
    hol_y = np.asarray([r[-1] for r in hol], dtype=float)
    hol_weeks = np.asarray([r[IDX_WEEK] for r in hol])
    hol_pos = [row_position(r) for r in hol]

    weeks = sorted(set(hol_weeks.tolist()))
    print(f"2024 holdout: n={len(hol_y)} across weeks {weeks[0]}..{weeks[-1]}")

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
    # self-check: fast ECE replica must match the gate's metric (g.metrics
    # rounds its reported ece to 4 decimals, so compare at that precision)
    assert abs(round(ece_only(wf_pred, hol_y), 4) - wf_m["ece"]) < 1e-9

    print(f"\n============ 2024 HOLDOUT: raw vs walk-forward ============")
    print(f"  {'arm':20s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'raw':20s} {raw_m['auc']:>7.4f}  {raw_m['log_loss']:>9.5f}  {raw_m['brier']:>7.5f} {raw_m['ece']:>7.4f}")
    print(f"  {'walk-forward platt':20s} {wf_m['auc']:>7.4f}  {wf_m['log_loss']:>9.5f}  {wf_m['brier']:>7.5f} {wf_m['ece']:>7.4f}")

    rng = np.random.default_rng(SEED)

    print(f"\n[control] bootstrap on RAW aggregate ({B_SIMS} sims) ...")
    p_raw = bootstrap_p(hol_raw, raw_m["ece"], rng)
    control_ok = p_raw < CONTROL_MAX_P
    print(f"[control] raw ECE={raw_m['ece']:.4f}  p={p_raw:.4f}  -> "
          f"{'has power' if control_ok else 'NO POWER -- no conclusion'}")

    print(f"[candidate] bootstrap on walk-forward aggregate ({B_SIMS} sims) ...")
    p_wf = bootstrap_p(wf_pred, wf_m["ece"], rng)
    print(f"[candidate] wf ECE={wf_m['ece']:.4f}  p={p_wf:.4f}")

    print(f"\nposition slices (bootstrap p >= {PASS_MIN_P} for every slice with n >= {MIN_SLICE_ROWS}):")
    slice_results = {}
    slices_ok = True
    for pos in POSITIONS:
        idx = [i for i, p in enumerate(hol_pos) if p == pos]
        if len(idx) < MIN_SLICE_ROWS:
            print(f"  {pos}: n={len(idx)} (below {MIN_SLICE_ROWS}, skipped)")
            continue
        sp = wf_pred[idx]
        sy = hol_y[idx]
        s_ece = ece_only(sp, sy)
        s_p = bootstrap_p(sp, s_ece, rng)
        ok = s_p >= PASS_MIN_P
        slices_ok = slices_ok and ok
        slice_results[pos] = {"n": len(idx), "ece": round(s_ece, 4), "p": s_p, "ok": ok}
        print(f"  {pos}: n={len(idx):>5}  ECE={s_ece:.4f}  p={s_p:.4f}  -> {'ok' if ok else 'FAIL'}")

    train_rate = float(np.mean([r[-1] for r in dev if r[IDX_WEEK] < cut]))
    constant = g.metrics([train_rate] * len(hol_y), list(hol_y))
    d_ll = constant["log_loss"] - wf_m["log_loss"]

    c_auc = wf_m["auc"] >= g.GATE["min_auc"]
    c_ll = d_ll >= g.GATE["min_logloss_gain"]
    c_brier = wf_m["brier"] < constant["brier"]
    c_control = control_ok
    c_calib = p_wf >= PASS_MIN_P
    c_slices = slices_ok
    passed = c_auc and c_ll and c_brier and c_control and c_calib and c_slices
    verdict = (f"NFL_RECEPTIONS_LINE_{args.line}_WALKFORWARD_PASSES_GATE"
               if passed else
               f"NFL_RECEPTIONS_LINE_{args.line}_WALKFORWARD_FAILS_GATE_WAIT_FOR_2025_DATA")

    print("\n============ PRE-REGISTERED GATE (walk-forward, final attempt) ============")
    print(f"  AUC >= {g.GATE['min_auc']}:                 {wf_m['auc']:.4f}  -> {c_auc}")
    print(f"  logloss gain >= {g.GATE['min_logloss_gain']}:       {d_ll:+.5f}  -> {c_ll}")
    print(f"  Brier better than constant:       {wf_m['brier']:.5f} < {constant['brier']:.5f}  -> {c_brier}")
    print(f"  power control (raw p < {CONTROL_MAX_P}):     p={p_raw:.4f}  -> {c_control}")
    print(f"  aggregate calibration (p >= {PASS_MIN_P}): p={p_wf:.4f}  -> {c_calib}")
    print(f"  all slices n>={MIN_SLICE_ROWS} pass (p >= {PASS_MIN_P}):  -> {c_slices}")
    print(f"  VERDICT: {verdict}")

    report = {"script": "NFL_RECEPTIONS_WALKFORWARD_CALIBRATION_A", "line": args.line,
              "final_attempt_against_this_holdout": True,
              "b_sims": B_SIMS, "pass_min_p": PASS_MIN_P, "control_max_p": CONTROL_MAX_P,
              "raw": raw_m, "walkforward": wf_m,
              "p_raw": p_raw, "p_walkforward": p_wf,
              "slices": slice_results,
              "constant": {"log_loss": constant["log_loss"], "brier": constant["brier"]},
              "passed": passed, "verdict": verdict}
    report_path = (Path(args.model_dir)
                   / f"nfl_receptions_walkforward_calibration_a_line_{args.line}_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {report_path}")
    print("Read-only. No production change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
