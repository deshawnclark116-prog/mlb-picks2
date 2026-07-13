#!/usr/bin/env python3
"""
PITCHER_K_RATE_CALIBRATION_GATE_A

Formal validation of the K rate-calibration challenger on the untouched 2026
holdout. Predictions-first: no odds.

Challenger (frozen isolated change): multiply the strikeout rate entering the
simulation by a factor DERIVED FROM 2025 (default 1.0504 from
pitcher_k_2025_calibration_derive_a) -- both the blended k_per_bf and the
per-outing volatility pool, so the whole projected K distribution scales
uniformly. Everything else in ksim (workload, BF_SD, TTO, blend weights, lineup
activation, sims, seeds) stays identical to the champion.

The factor was set on 2025; this gate tests it on 2026. That is proper
out-of-sample validation, not fitting the holdout.

Method: read the champion's 2026 predictions from the existing formal_rows.sqlite
(same seeds), then re-run build_2026_rows with the rate-scaled ksim and compare
row-paired.

Pre-registered pass (all):
  1. |challenger mean bias| < |champion mean bias|  (bias reduced)
  2. challenger mean CRPS <= champion mean CRPS + 1e-4 (not worse; ideally better)
  3. date-block bootstrap P(challenger CRPS < champion CRPS) >= 0.90
  4. interval coverage[q10,q90] not worse by more than 0.03 (guard the spread)

A pass authorizes NOTHING automatically -- it is evidence for an implementation
patch to ksim, promoted through the same discipline as D0.

Read-only on the baseline + formal_rows; writes only its own scored copy.

Run (Render)
------------
python -u pitcher_k_rate_calibration_gate_a.py 2>&1 | tee /data/hr_model/pitcher_k_rate_calibration_gate_a.log
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

import pitcher_k_d0_2026_formal_gate_a as gate

WORK = Path("/data/hr_model/pitcher_k_rate_calibration_gate_a_work")
CHALL = WORK / "challenger_2026.sqlite"
CHAMP_DB = Path("/data/hr_model/pitcher_k_d0_2026_formal_gate_a_work/formal_rows.sqlite")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=float, default=1.0504)
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    factor = args.factor

    if not CHAMP_DB.exists():
        print(f"FAIL: champion formal_rows not found: {CHAMP_DB}")
        return 1

    print("PITCHER_K_RATE_CALIBRATION_GATE_A\n=================================")
    print(f"2025-derived rate factor under test: {factor:.4f}")

    # champion 2026 predictions (already scored, same seeds)
    cc = sqlite3.connect(f"file:{CHAMP_DB}?mode=ro", uri=True)
    champ = {(r[0], r[1]): r for r in cc.execute(
        "SELECT game_id, pitcher_id, actual_k, d0_mean, d0_crps, d0_actual_in_10_90, game_date "
        "FROM formal_rows WHERE year=2026")}
    cc.close()
    print(f"champion 2026 rows: {len(champ)}")

    # challenger: rate-scaled ksim, same machinery/seeds
    _orig = gate.vectorized_ksim

    def calibrated_ksim(k_per_bf, expected_bf, start_k_rates, **kw):
        scaled_rates = [r * factor for r in start_k_rates] if start_k_rates else start_k_rates
        return _orig(k_per_bf * factor, expected_bf, scaled_rates, **kw)

    gate.vectorized_ksim = calibrated_ksim
    gate.HOLDOUT_YEAR = 2026
    print("scoring 2026 with rate-scaled ksim ...", flush=True)
    base = sqlite3.connect(f"file:{gate.BASELINE_DB}?mode=ro", uri=True)
    if CHALL.exists():
        CHALL.unlink()
    out = sqlite3.connect(str(CHALL))
    gate.build_2026_rows(base, out)
    out.commit(); base.close()
    chal = {(r[0], r[1]): r for r in out.execute(
        "SELECT game_id, pitcher_id, actual_k, d0_mean, d0_crps, d0_actual_in_10_90 FROM formal_rows WHERE year=2026")}
    out.close()
    gate.vectorized_ksim = _orig  # restore

    keys = sorted(set(champ) & set(chal))
    n = len(keys)
    print(f"paired rows: {n}")
    if n == 0:
        print("FAIL: no paired rows.")
        return 1

    act = np.array([champ[k][2] for k in keys], float)
    cm = np.array([champ[k][3] for k in keys], float)   # champion mean
    xm = np.array([chal[k][3] for k in keys], float)    # challenger mean
    cc_crps = np.array([champ[k][4] for k in keys], float)
    xc_crps = np.array([chal[k][4] for k in keys], float)
    cov_c = np.array([champ[k][5] for k in keys], float)
    cov_x = np.array([chal[k][5] for k in keys], float)
    dates = np.array([champ[k][6] for k in keys])

    champ_bias, chal_bias = float((cm - act).mean()), float((xm - act).mean())
    champ_crps, chal_crps = float(cc_crps.mean()), float(xc_crps.mean())
    champ_cov, chal_cov = float(cov_c.mean()), float(cov_x.mean())
    champ_mae, chal_mae = float(np.abs(cm - act).mean()), float(np.abs(xm - act).mean())

    print("\n================ 2026 HOLDOUT: champion vs rate-calibrated challenger ================")
    print(f"  {'metric':16s} {'champion':>10s} {'challenger':>11s}")
    print(f"  {'mean bias':16s} {champ_bias:>+10.4f} {chal_bias:>+11.4f}")
    print(f"  {'MAE':16s} {champ_mae:>10.4f} {chal_mae:>11.4f}")
    print(f"  {'mean CRPS':16s} {champ_crps:>10.5f} {chal_crps:>11.5f}")
    print(f"  {'cov[q10,q90]':16s} {champ_cov:>10.3f} {chal_cov:>11.3f}")

    # date-block bootstrap on CRPS delta (champion - challenger; >0 = challenger better)
    uniq = np.array(sorted(set(dates.tolist())))
    idx_by = {d: np.where(dates == d)[0] for d in uniq}
    rng = np.random.default_rng(20260713)
    B = 2000
    d = np.empty(B)
    for b in range(B):
        idx = np.concatenate([idx_by[x] for x in rng.choice(uniq, len(uniq), replace=True)])
        d[b] = cc_crps[idx].mean() - xc_crps[idx].mean()
    p_better = float((d > 0).mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n  CRPS delta (champ-chal) mean={d.mean():+.5f} 95%CI=[{lo:+.5f},{hi:+.5f}] P(chal better)={p_better:.3f}")

    c1 = abs(chal_bias) < abs(champ_bias)
    c2 = chal_crps <= champ_crps + 1e-4
    c3 = p_better >= 0.90
    c4 = chal_cov <= champ_cov + 0.03
    passed = c1 and c2 and c3 and c4
    verdict = ("RATE_CALIBRATION_CHALLENGER_PASSES_2026_GATE_READY_FOR_KSIM_PATCH"
               if passed else "RATE_CALIBRATION_CHALLENGER_DOES_NOT_CLEAR_GATE")
    print("\n================ PRE-REGISTERED GATE ================")
    print(f"  |bias| reduced:              {abs(champ_bias):.4f} -> {abs(chal_bias):.4f}  -> {c1}")
    print(f"  CRPS not worse:              {champ_crps:.5f} -> {chal_crps:.5f}  -> {c2}")
    print(f"  bootstrap P(better CRPS)>=.9: {p_better:.3f}  -> {c3}")
    print(f"  coverage not worse >0.03:    {champ_cov:.3f} -> {chal_cov:.3f}  -> {c4}")
    print(f"  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
