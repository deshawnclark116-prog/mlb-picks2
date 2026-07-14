#!/usr/bin/env python3
"""
NFL_RECEPTIONS_POSITION_RECALIBRATION_A

Global Platt recalibration (nfl_receptions_recalibration_a) fixed line 1.5's
AGGREGATE ECE (0.0222 -> 0.0192) but the TE slice still failed stability
(0.0425 vs the 0.04 bar) -- and TE was line 2.5's failure point too (0.0524).
A single global affine correction cannot fix a subgroup-specific miscalibration
-- this fits a SEPARATE Platt correction per position (RB, TE, WR) instead of
one global (a, b), directly targeting the diagnosed problem.

Method (same anti-overfit discipline: fit on 2023 dev's held-out internal
validation slice ONLY, never the 2024 holdout):
  For each position in (RB, TE, WR): fit (a_pos, b_pos) via Platt scaling on
  that position's subset of the internal validation slice (2023 weeks >= the
  80% cut). Apply each row's own position-specific correction on 2024.

Note the calibration-fitting slice is small (n~211 total, ~50-90 per
position) -- this is exactly why the result is judged on the same real
2024 holdout gate + slice checks, not assumed to work.

Read-only on the model + baseline. Writes only a report and, on a pass, a
per-position calibration.json.

Run (Render, per line)
-----------------------
python -u nfl_receptions_position_recalibration_a.py --line 1.5 2>&1 | tee /data/nfl_model/nfl_receptions_position_recalibration_a_1_5.log
python -u nfl_receptions_position_recalibration_a.py --line 2.5 2>&1 | tee /data/nfl_model/nfl_receptions_position_recalibration_a_2_5.log
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
from nfl_receptions_recalibration_a import fit_platt, apply_platt, MODEL_WORKDIR

POSITIONS = ["RB", "TE", "WR"]
MIN_SLICE_ROWS = 300
SLICE_ECE_BAR = 0.04


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
    print(f"NFL_RECEPTIONS_POSITION_RECALIBRATION_A  [line={args.line}]\n" + "=" * 44)

    bst = xgb.Booster(); bst.load_model(str(mdir / f"{model_stem}.json"))
    feat_cols = json.loads((mdir / f"{model_stem}_columns.json").read_text())
    assert feat_cols == g.FEATURES
    itr = (0, bst.best_iteration + 1)

    con = sqlite3.connect(f"file:{args.baseline}?mode=ro", uri=True)
    cols = ["season", "week", "position"] + g.FEATURES + [target_col]
    all_rows = con.execute(f"SELECT {', '.join(cols)} FROM nfl_receptions_baseline").fetchall()
    con.close()
    dev = [r for r in all_rows if r[0] == 2023]
    hol = [r for r in all_rows if r[0] == 2024]
    weeks = sorted({r[1] for r in dev})
    cut = weeks[int(len(weeks) * 0.8)] if len(weeks) > 5 else weeks[-1]
    va = [r for r in dev if r[1] >= cut]
    print(f"calibration-fit slice: 2023 weeks >= {cut}, n={len(va)} (never touches 2024 holdout)")

    def mat(rows):
        X = np.array([[r[3 + i] if r[3 + i] is not None else g.NAN for i in range(len(g.FEATURES))]
                      for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, feature_names=g.FEATURES)

    params = {}
    for pos in POSITIONS:
        va_pos = [r for r in va if r[2] == pos]
        if len(va_pos) < 20:
            print(f"  {pos}: n={len(va_pos)} too small, falling back to identity (a=1,b=0)")
            params[pos] = (1.0, 0.0)
            continue
        raw = bst.predict(mat(va_pos), iteration_range=itr)
        y = [r[-1] for r in va_pos]
        a, b = fit_platt(raw, y)
        if a <= 0:
            print(f"  {pos}: n={len(va_pos)} fitted slope a={a:.3f} <= 0, refusing, using identity")
            a, b = 1.0, 0.0
        params[pos] = (a, b)
        print(f"  {pos}: n={len(va_pos)}  a={a:.4f}  b={b:.4f}")

    hol_raw = bst.predict(mat(hol), iteration_range=itr)
    hol_recal = np.empty_like(hol_raw)
    for i, r in enumerate(hol):
        a, b = params[r[2]]
        hol_recal[i] = apply_platt(np.array([hol_raw[i]]), a, b)[0]
    hol_y = [r[-1] for r in hol]

    before = g.metrics(list(map(float, hol_raw)), hol_y)
    after = g.metrics(list(map(float, hol_recal)), hol_y)
    print(f"\nAUC before={before['auc']:.4f}  after={after['auc']:.4f}  "
          f"delta={abs(before['auc']-after['auc']):.5f}")

    print("\n============ 2024 HOLDOUT: raw vs per-position recalibrated ============")
    print(f"  {'arm':22s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'raw':22s} {before['auc']:>7.4f}  {before['log_loss']:>9.5f}  {before['brier']:>7.5f} {before['ece']:>7.4f}")
    print(f"  {'per-pos recalibrated':22s} {after['auc']:>7.4f}  {after['log_loss']:>9.5f}  {after['brier']:>7.5f} {after['ece']:>7.4f}")

    print(f"\nSLICE ECE after per-position recalibration (bar <= {SLICE_ECE_BAR}):")
    slice_ok = True
    for pos in POSITIONS:
        idx = [i for i, r in enumerate(hol) if r[2] == pos]
        if len(idx) < MIN_SLICE_ROWS:
            print(f"  {pos}: n={len(idx)} (skip)"); continue
        pm = g.metrics([float(hol_recal[i]) for i in idx], [hol_y[i] for i in idx])
        ok = pm["ece"] <= SLICE_ECE_BAR
        slice_ok = slice_ok and ok
        print(f"  {pos}: n={len(idx):>5}  AUC={pm['auc']:.4f}  ECE={pm['ece']:.4f}  {'ok' if ok else 'FAIL'}")

    train_rate = float(np.mean([r[-1] for r in dev if r[1] < cut]))
    constant = g.metrics([train_rate] * len(hol), hol_y)
    d_ll = constant["log_loss"] - after["log_loss"]

    c1 = after["auc"] >= g.GATE["min_auc"]
    c2 = after["ece"] <= g.GATE["max_ece"]
    c3 = d_ll >= g.GATE["min_logloss_gain"]
    c4 = after["brier"] < constant["brier"]
    c5 = slice_ok
    passed = c1 and c2 and c3 and c4 and c5
    verdict = (f"NFL_RECEPTIONS_LINE_{args.line}_POSITION_RECALIBRATED_PASSES_GATE_AND_SLICES"
               if passed else f"NFL_RECEPTIONS_LINE_{args.line}_POSITION_RECALIBRATED_STILL_FAILS")

    print("\n============ PRE-REGISTERED GATE (aggregate + per-position slices) ============")
    print(f"  AUC >= {g.GATE['min_auc']}:            {after['auc']:.4f}  -> {c1}")
    print(f"  ECE <= {g.GATE['max_ece']}:              {after['ece']:.4f}  -> {c2}")
    print(f"  logloss gain >= {g.GATE['min_logloss_gain']}:  {d_ll:+.5f}  -> {c3}")
    print(f"  Brier better than constant:  {after['brier']:.5f} < {constant['brier']:.5f}  -> {c4}")
    print(f"  all position slices ECE <= {SLICE_ECE_BAR}:  -> {c5}")
    print(f"  VERDICT: {verdict}")

    if passed:
        out_path = mdir / f"{model_stem}_position_calibration.json"
        out_path.write_text(json.dumps({pos: {"a": a, "b": b} for pos, (a, b) in params.items()}))
        print(f"\nper-position calibration written: {out_path}")
        print("Next: re-run stability confirmation with per-position recalibration applied.")

    report = {"script": "NFL_RECEPTIONS_POSITION_RECALIBRATION_A", "line": args.line,
              "params": {pos: {"a": a, "b": b} for pos, (a, b) in params.items()},
              "before": before, "after": after, "passed": passed, "verdict": verdict}
    out = Path("/data/nfl_model") if Path("/data/nfl_model").exists() else mdir
    report_name = f"nfl_receptions_position_recalibration_a_line_{args.line}_report.json"
    (out / report_name).write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out/report_name}")
    print("Read-only. No production change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
