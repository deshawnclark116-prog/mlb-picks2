#!/usr/bin/env python3
"""
NFL_RECEPTIONS_THRESHOLD_DIAGNOSTIC_A

Corrects a real flaw: the flagship line was set to over 0.5 (at least one
catch), which turned out to be a near-lock for WR/TE (~90% base rate) -- not a
genuine prediction market, and not what real reception props look like (books
set lines at 3.5/4.5/5.5 for a reason: they want the outcome in doubt).

This measures the actual_receptions distribution already stored in the clean
baseline and reports the over-rate at each candidate threshold, overall and by
position, so the next threshold is chosen from real uncertainty, not another
guess. Computed on 2023 (DEV) ONLY -- deliberately never touches the 2024
holdout, since picking a threshold is a market-design decision and must follow
the same "derive on dev, evaluate on holdout" discipline as everything else.

Read-only. Changes nothing.

Run (Render)
------------
python -u nfl_receptions_threshold_diagnostic_a.py 2>&1 | tee /data/nfl_model/nfl_receptions_threshold_diagnostic_a.log
"""

import sqlite3
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

BASELINE = "/data/nfl_model/nfl_receptions_clean_baseline_a_work/baseline.sqlite"
THRESHOLDS = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]


def main():
    con = sqlite3.connect(f"file:{BASELINE}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT position, actual_receptions FROM nfl_receptions_baseline WHERE season=2023").fetchall()
    con.close()
    print("NFL_RECEPTIONS_THRESHOLD_DIAGNOSTIC_A\n======================================")
    print(f"2023 (dev) eligible rows: {len(rows)}\n")

    def over_rate(subset, t):
        if not subset:
            return None
        over = sum(1 for _, r in subset if r >= (t + 0.5))
        return over / len(subset)

    positions = {"ALL": rows}
    for pos in ("WR", "TE", "RB"):
        positions[pos] = [r for r in rows if r[0] == pos]

    print(f"{'threshold':>10s}" + "".join(f"{p:>10s}" for p in positions))
    for t in THRESHOLDS:
        line = f"{'>' + str(t):>10s}"
        for p, subset in positions.items():
            rate = over_rate(subset, t)
            line += f"{rate:>10.3f}" if rate is not None else f"{'n/a':>10s}"
        print(line)

    # find the threshold whose overall rate is closest to 50% -- maximum
    # genuine uncertainty, matching how a real book would price a coin-flip-ish
    # market rather than a near-lock or near-impossible one.
    best_t, best_dist = None, 1.0
    for t in THRESHOLDS:
        rate = over_rate(rows, t)
        dist = abs(rate - 0.5)
        if dist < best_dist:
            best_dist, best_t = dist, t

    print(f"\nclosest-to-50% overall threshold: over {best_t} "
          f"(rate={over_rate(rows, best_t):.3f})")
    print("\nREAD: pick a threshold with real uncertainty (rate roughly 0.35-0.65),")
    print("  not one so low it's a near-lock or so high it's a near-impossibility.")
    print("  A single fixed line keeps parity with every other market in this system")
    print("  (batter_hits >0.5, batter_total_bases >1.5, batter_home_runs >0.5 --")
    print("  all chosen because the population is genuinely split near that line).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
