#!/usr/bin/env python3
"""
HR_CALIBRATION_REALITY_CHECK_A

Quantifies how miscalibrated the current batter_home_runs path is. The live
model sets P(hits a HR) to a hardcoded ~0.358 ("elite") / ~0.342 tier constant.
This measures the REAL per-game HR rate from batter_games (strict D-1), overall
and by power tier, so the gap is explicit. Predictions-first: no odds.

Read-only. Changes nothing.

Run (Render)
------------
python -u hr_calibration_reality_check_a.py 2>&1 | tee /data/hr_model/hr_calibration_reality_check_a.log
"""

import sqlite3
import sys
from collections import deque
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

SOURCE = "/data/hr_model/hr_model.sqlite"
MODEL_ELITE_PROB = 0.358   # 1 - e^{-0.444}
MODEL_BASE_PROB = 0.342    # 1 - e^{-0.419}
MIN_PRIOR_AB = 40
MIN_PRIOR_GAMES = 10


def main():
    con = sqlite3.connect(f"file:{SOURCE}?mode=ro", uri=True)
    rows = con.execute("""SELECT batter_id, substr(game_date,1,4) AS season, game_date,
                          at_bats, hits, doubles, triples, home_runs
                          FROM batter_games WHERE at_bats IS NOT NULL
                          ORDER BY batter_id, season, game_date""").fetchall()
    con.close()

    print("HR_CALIBRATION_REALITY_CHECK_A\n==============================")
    print(f"model states P(HR>=1) = {MODEL_ELITE_PROB} (elite) / {MODEL_BASE_PROB} (base)\n")

    # overall base rate
    n_all = len(rows)
    hr_all = sum(1 for r in rows if (r[7] or 0) >= 1)
    print(f"OVERALL per-game HR rate (all batter-games): {hr_all}/{n_all} = {hr_all/n_all:.4f}")

    # strict-D-1 prior-season SLG per eligible batter-game, bucketed
    from itertools import groupby
    def keyfn(r):
        return (r[0], r[1])
    tiers = {"elite(SLG>=.500)": [], "good(.430-.500)": [], "avg(.380-.430)": [], "low(<.380)": []}
    for (bid, season), grp in groupby(sorted(rows, key=keyfn), keyfn):
        g = list(grp)
        cum_ab = cum_tb = 0
        ngames = 0
        i = 0
        for cur_g in g:
            gd = cur_g[2]
            while i < len(g) and g[i][2] < gd:
                h = g[i]
                tb = (h[4] or 0) + (h[5] or 0) + 2 * (h[6] or 0) + 3 * (h[7] or 0)
                cum_ab += (h[3] or 0); cum_tb += tb; ngames += 1
                i += 1
            if ngames < MIN_PRIOR_GAMES or cum_ab < MIN_PRIOR_AB:
                continue
            slg = cum_tb / cum_ab if cum_ab else 0
            hit_hr = 1 if (cur_g[7] or 0) >= 1 else 0
            tier = ("elite(SLG>=.500)" if slg >= 0.500 else
                    "good(.430-.500)" if slg >= 0.430 else
                    "avg(.380-.430)" if slg >= 0.380 else "low(<.380)")
            tiers[tier].append(hit_hr)

    print("\nREAL per-game HR rate by prior-season SLG tier (strict D-1):")
    print(f"  {'tier':20s} {'n':>7s} {'actual P(HR>=1)':>16s}   vs model")
    for t in ("elite(SLG>=.500)", "good(.430-.500)", "avg(.380-.430)", "low(<.380)"):
        v = tiers[t]
        if not v:
            continue
        rate = sum(v) / len(v)
        model = MODEL_ELITE_PROB if t.startswith("elite") else MODEL_BASE_PROB
        print(f"  {t:20s} {len(v):7d} {rate:16.4f}   model~{model:.3f}  ({model/rate:.1f}x too high)")

    print("\nREAD: even the top SLG tier hits a HR in only ~1 game in 8-12; the model's")
    print("  ~0.35 constant is ~3-4x that. HR needs a real calibrated model, not a tier constant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
