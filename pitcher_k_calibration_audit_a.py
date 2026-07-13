#!/usr/bin/env python3
"""
PITCHER_K_CALIBRATION_AUDIT_A

Confirms the pitcher-strikeout champion (the D0 model that passed the formal
2026 CRPS gate) is well-calibrated in its DECISION distribution, not just in
aggregate CRPS. Predictions-first: no odds.

Zero reconstruction: the formal gate already scored every 2026 starter and
stored the predicted quantiles (d0_q10 / d0_median / d0_q90), the realized
actual_k, and interval-coverage indicators in formal_rows.sqlite. This reads
those and checks distributional calibration directly.

A well-calibrated forecast distribution satisfies:
  P(actual < q10)            ~ 0.10
  P(actual <= median)        ~ 0.50
  P(actual < q90)            ~ 0.90
  coverage[q10,q90]          ~ 0.80
  coverage[IQR]              ~ 0.50
and PIT bins below q10 / q10-median / median-q90 / above q90 ~ 0.10/0.40/0.40/0.10.

Also reports mean bias and MAE, and a monthly breakdown (calibration stability).
a0 (old incumbent) is shown alongside d0 (deployed champion) for reference.

Read-only. Changes nothing.

Run (Render)
------------
python -u pitcher_k_calibration_audit_a.py 2>&1 | tee /data/hr_model/pitcher_k_calibration_audit_a.log
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


def frac(rows, pred):
    return sum(1 for r in rows if pred(r)) / len(rows) if rows else float("nan")


def summarize(rows, tag):
    n = len(rows)
    mean_pred = sum(r[f"{tag}_mean"] for r in rows) / n
    mean_act = sum(r["actual_k"] for r in rows) / n
    mae = sum(abs(r[f"{tag}_mean"] - r["actual_k"]) for r in rows) / n
    below_q10 = frac(rows, lambda r: r["actual_k"] < r[f"{tag}_q10"])
    le_median = frac(rows, lambda r: r["actual_k"] <= r[f"{tag}_median"])
    below_q90 = frac(rows, lambda r: r["actual_k"] < r[f"{tag}_q90"])
    cov_1090 = sum(r[f"{tag}_actual_in_10_90"] for r in rows) / n
    cov_iqr = sum(r[f"{tag}_actual_in_iqr"] for r in rows) / n
    # PIT bins
    b1 = frac(rows, lambda r: r["actual_k"] < r[f"{tag}_q10"])
    b2 = frac(rows, lambda r: r[f"{tag}_q10"] <= r["actual_k"] <= r[f"{tag}_median"])
    b3 = frac(rows, lambda r: r[f"{tag}_median"] < r["actual_k"] <= r[f"{tag}_q90"])
    b4 = frac(rows, lambda r: r["actual_k"] > r[f"{tag}_q90"])
    return {"n": n, "mean_pred": mean_pred, "mean_actual": mean_act, "bias": mean_pred - mean_act,
            "mae": mae, "P(<q10)": below_q10, "P(<=median)": le_median, "P(<q90)": below_q90,
            "cov_10_90": cov_1090, "cov_iqr": cov_iqr, "pit": (b1, b2, b3, b4)}


def show(s, tag):
    print(f"\n[{tag}] n={s['n']}")
    print(f"  mean pred {s['mean_pred']:.2f} vs actual {s['mean_actual']:.2f}  "
          f"(bias {s['bias']:+.3f}, MAE {s['mae']:.3f})")
    print(f"  P(actual < q10)     {s['P(<q10)']:.3f}   (target 0.10)")
    print(f"  P(actual <= median) {s['P(<=median)']:.3f}   (target 0.50)")
    print(f"  P(actual < q90)     {s['P(<q90)']:.3f}   (target 0.90)")
    print(f"  coverage [q10,q90]  {s['cov_10_90']:.3f}   (target 0.80)")
    print(f"  coverage [IQR]      {s['cov_iqr']:.3f}   (target 0.50)")
    b1, b2, b3, b4 = s["pit"]
    print(f"  PIT bins  <q10={b1:.3f} q10-med={b2:.3f} med-q90={b3:.3f} >q90={b4:.3f}  "
          f"(target 0.10/0.40/0.40/0.10)")


def main():
    if not FORMAL_DB.exists():
        print(f"FAIL: {FORMAL_DB} not found (run the formal gate first).")
        return 1
    con = sqlite3.connect(f"file:{FORMAL_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cols = ("game_date, actual_k, "
            "d0_mean, d0_median, d0_q10, d0_q90, d0_actual_in_iqr, d0_actual_in_10_90, "
            "a0_mean, a0_median, a0_q10, a0_q90, a0_actual_in_iqr, a0_actual_in_10_90")
    rows = [dict(r) for r in con.execute(f"SELECT {cols} FROM formal_rows")]
    con.close()
    print("PITCHER_K_CALIBRATION_AUDIT_A\n=============================")
    print(f"formal_rows: {len(rows)} 2026 starts")

    d0 = summarize(rows, "d0")
    a0 = summarize(rows, "a0")
    print("\n================ DISTRIBUTIONAL CALIBRATION (2026 holdout) ================")
    show(d0, "d0 = DEPLOYED champion")
    show(a0, "a0 = old incumbent (reference)")

    # monthly stability for d0
    print("\nMONTHLY (d0 coverage[q10,q90], target 0.80):")
    months = {}
    for r in rows:
        months.setdefault(r["game_date"][:7], []).append(r)
    for m in sorted(months):
        mr = months[m]
        cov = sum(x["d0_actual_in_10_90"] for x in mr) / len(mr)
        print(f"   {m}: n={len(mr):4d}  cov_10_90={cov:.3f}")

    # verdict
    def near(x, t, tol=0.05):
        return abs(x - t) <= tol
    ok = (near(d0["cov_10_90"], 0.80) and near(d0["cov_iqr"], 0.50)
          and near(d0["P(<=median)"], 0.50) and near(d0["P(<q10)"], 0.10, 0.04)
          and near(d0["P(<q90)"], 0.90, 0.04))
    verdict = ("PITCHER_K_CHAMPION_WELL_CALIBRATED_ON_2026_HOLDOUT"
               if ok else "PITCHER_K_CHAMPION_CALIBRATION_OFF_INVESTIGATE")
    print(f"\n================ VERDICT ================\n  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
