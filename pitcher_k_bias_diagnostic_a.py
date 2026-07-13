#!/usr/bin/env python3
"""
PITCHER_K_BIAS_DIAGNOSTIC_A

Localizes the two calibration findings from pitcher_k_calibration_audit_a on the
K champion (D0): a small mean under-projection (bias ~ -0.24) and slightly
over-wide intervals (coverage[q10,q90] ~ 0.86 vs 0.80 target). Predictions-first.

Read-only on formal_rows.sqlite (the formal gate's per-2026-start output). It
breaks the bias, MAE, and interval coverage down by subset so we know whether a
ksim challenger should make a GLOBAL change (bias everywhere) or a TARGETED one
(bias concentrated in, say, the intervened subset or high-workload starts).

Breakdowns: intervention_status, d0_lineup_used, prior_qualifying_outings,
expected_bf band, and actual_k band, plus by month.

Run (Render)
------------
python -u pitcher_k_bias_diagnostic_a.py 2>&1 | tee /data/hr_model/pitcher_k_bias_diagnostic_a.log
"""

import sqlite3
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

FORMAL_DB = Path("/data/hr_model/pitcher_k_d0_2026_formal_gate_a_work/formal_rows.sqlite")


def stats(rows):
    n = len(rows)
    if n == 0:
        return None
    bias = sum(r["d0_mean"] - r["actual_k"] for r in rows) / n
    mae = sum(abs(r["d0_mean"] - r["actual_k"]) for r in rows) / n
    cov = sum(r["d0_actual_in_10_90"] for r in rows) / n
    return {"n": n, "bias": bias, "mae": mae, "cov_10_90": cov}


def breakdown(rows, name, keyfn, order=None):
    groups = {}
    for r in rows:
        groups.setdefault(keyfn(r), []).append(r)
    print(f"\n{name}")
    print(f"  {'bucket':22s} {'n':>6s} {'bias':>8s} {'MAE':>7s} {'cov[q10,q90]':>13s}")
    keys = order if order else sorted(groups, key=lambda k: str(k))
    for k in keys:
        if k not in groups:
            continue
        s = stats(groups[k])
        flag = "" if abs(s["cov_10_90"] - 0.80) <= 0.05 else "  <- wide" if s["cov_10_90"] > 0.85 else "  <- narrow"
        print(f"  {str(k):22s} {s['n']:6d} {s['bias']:+8.3f} {s['mae']:7.3f} {s['cov_10_90']:13.3f}{flag}")


def band_bf(r):
    e = r["expected_bf"]
    return "bf<20" if e < 20 else "bf_20_24" if e < 24.0001 else "bf_25plus"


def band_outings(r):
    o = r["prior_qualifying_outings"]
    return "out_3_5" if o <= 5 else "out_6_10" if o <= 10 else "out_11plus"


def band_actual(r):
    a = r["actual_k"]
    return "k_0_3" if a <= 3 else "k_4_6" if a <= 6 else "k_7plus"


def main():
    if not FORMAL_DB.exists():
        print(f"FAIL: {FORMAL_DB} not found.")
        return 1
    con = sqlite3.connect(f"file:{FORMAL_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT game_date, actual_k, expected_bf, prior_qualifying_outings, "
        "intervention_status, d0_lineup_used, d0_mean, d0_actual_in_10_90 FROM formal_rows")]
    con.close()

    print("PITCHER_K_BIAS_DIAGNOSTIC_A\n===========================")
    ov = stats(rows)
    print(f"overall  n={ov['n']}  bias={ov['bias']:+.3f}  MAE={ov['mae']:.3f}  cov[q10,q90]={ov['cov_10_90']:.3f}")

    breakdown(rows, "by intervention_status", lambda r: r["intervention_status"])
    breakdown(rows, "by d0_lineup_used", lambda r: f"lineup_used={r['d0_lineup_used']}", order=["lineup_used=0", "lineup_used=1"])
    breakdown(rows, "by prior_qualifying_outings", band_outings, order=["out_3_5", "out_6_10", "out_11plus"])
    breakdown(rows, "by expected_bf band", band_bf, order=["bf<20", "bf_20_24", "bf_25plus"])
    breakdown(rows, "by actual_k band (post-hoc)", band_actual, order=["k_0_3", "k_4_6", "k_7plus"])
    breakdown(rows, "by month", lambda r: r["game_date"][:7])

    print("\nREAD:")
    print("  - If bias is roughly constant across buckets -> global under-projection;")
    print("    a small mean recalibration (e.g. anchor / k_per_bf) is the ksim challenger.")
    print("  - If bias concentrates in one bucket (e.g. bf_25plus or k_7plus) -> targeted.")
    print("  - cov[q10,q90] >> 0.80 marks where the sim distribution is too wide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
