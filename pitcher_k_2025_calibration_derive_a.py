#!/usr/bin/env python3
"""
PITCHER_K_2025_CALIBRATION_DERIVE_A

Derives the K rate-calibration factor from 2025 (development), so the ksim
challenger's fix is set on dev and VALIDATED on the untouched 2026 holdout --
never fitted to the holdout. Predictions-first: no odds.

The 2026 calibration audit showed the champion under-projects strikeouts by a
~4.6% rate error (predicted K/BF 0.209 vs actual 0.219, bias -0.24). To fix that
honestly we must measure the SAME rate error on 2025 and set the correction from
there. If a 2025-derived factor also removes the 2026 bias (checked later by the
formal gate), the fix is real and generalizes.

Method: reuse the formal gate's exact scoring machinery unchanged -- import the
module and point HOLDOUT_YEAR at 2025, then run build_2026_rows against the same
clean baseline. This scores 2025 starts (2024 = warm-up) with the identical
pitcher_profile / lineup blend / vectorized_ksim the champion uses. Then measure
the rate gap and emit the calibration factor.

Outputs the factor to apply to the pitcher k_per_bf anchor in the ksim challenger.

Read-only on the baseline; writes only its own scored copy. Changes no
production code or the champion.

Run (Render)
------------
python -u pitcher_k_2025_calibration_derive_a.py 2>&1 | tee /data/hr_model/pitcher_k_2025_calibration_derive_a.log
"""

import sqlite3
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import pitcher_k_d0_2026_formal_gate_a as gate

WORK = Path("/data/hr_model/pitcher_k_2025_calibration_derive_a_work")
SCORED = WORK / "scored_2025.sqlite"


def band_bf(e):
    return "bf<20" if e < 20 else "bf_20_24" if e < 24.0001 else "bf_25plus"


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    if not Path(gate.BASELINE_DB).exists():
        print(f"FAIL: baseline not found: {gate.BASELINE_DB}")
        return 1

    # Reuse the champion's exact scoring, pointed at 2025 (dev).
    gate.HOLDOUT_YEAR = 2025
    print("PITCHER_K_2025_CALIBRATION_DERIVE_A\n===================================")
    print(f"scoring 2025 with the champion machinery (HOLDOUT_YEAR={gate.HOLDOUT_YEAR}) ...", flush=True)

    base = sqlite3.connect(f"file:{gate.BASELINE_DB}?mode=ro", uri=True)
    if SCORED.exists():
        SCORED.unlink()
    out = sqlite3.connect(str(SCORED))
    gate.build_2026_rows(base, out)
    out.commit()
    base.close()

    rows = list(out.execute(
        "SELECT actual_k, actual_bf, expected_bf, d0_mean, d0_actual_in_10_90, "
        "d0_actual_in_iqr FROM formal_rows WHERE year=2025"))
    out.close()
    n = len(rows)
    if n == 0:
        print("FAIL: no 2025 rows scored.")
        return 1

    sum_actk = sum(r[0] for r in rows)
    sum_pred = sum(r[3] for r in rows)
    mean_bias = (sum_pred - sum_actk) / n
    mae = sum(abs(r[3] - r[0]) for r in rows) / n
    pred_rate = sum(r[3] / r[2] for r in rows if r[2]) / n
    act_rate = sum(r[0] / r[1] for r in rows if r[1]) / n
    cov_1090 = sum(r[4] for r in rows) / n
    cov_iqr = sum(r[5] for r in rows) / n

    # calibration factor: scale the projected mean so 2025 bias -> 0
    factor = sum_actk / sum_pred if sum_pred else 1.0

    print(f"\n2025 DEV SCORE  n={n}")
    print(f"  mean pred {sum_pred/n:.3f} vs actual {sum_actk/n:.3f}  (bias {mean_bias:+.3f}, MAE {mae:.3f})")
    print(f"  predicted K/BF {pred_rate:.4f} vs actual K/BF {act_rate:.4f}  (rate bias {pred_rate-act_rate:+.4f})")
    print(f"  coverage[q10,q90] {cov_1090:.3f} (target 0.80)   coverage[IQR] {cov_iqr:.3f} (target 0.50)")

    print("\nby expected_bf band (bias):")
    bands = {}
    for r in rows:
        bands.setdefault(band_bf(r[2]), []).append(r)
    for b in ("bf<20", "bf_20_24", "bf_25plus"):
        if b in bands:
            g = bands[b]
            bi = sum(x[3] - x[0] for x in g) / len(g)
            print(f"   {b:10s} n={len(g):4d}  bias={bi:+.3f}")

    print("\n================ DERIVED CALIBRATION (from 2025 dev) ================")
    print(f"  k rate calibration factor = {factor:.4f}")
    print(f"  (multiply the champion's k_per_bf anchor by this in the ksim challenger,")
    print(f"   then validate on the 2026 holdout via the formal gate -- NOT fitted to 2026)")
    if abs(factor - 1.0) < 0.005:
        print("  NOTE: factor ~1.0 -> 2025 shows little rate bias; the 2026 gap may be")
        print("        holdout-specific (league drift) and should NOT be hard-coded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
