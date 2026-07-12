#!/usr/bin/env python3
"""
TOTAL_BASES_CHAMPION_PREFLIGHT_A

Rung 3 of the total-bases pipeline (mirrors pitcher_k_champion_baseline_preflight).

Purpose
-------
Score the ACTUAL current champion model (/data/models/batter_total_bases.json)
on the strict-D-1 clean baseline built by rung 2, and report -- honestly, on an
untouched 2026 holdout -- whether the champion has real, usable signal for TB
OVER 1.5, and where (if anywhere) it clears a betting benchmark.

It does NOT retrain, tune, or change production. Read-only evaluation.

Champion probability replication
--------------------------------
proj      = xgb batter_total_bases prediction on the 11 clean features
prob_over = 1 - poisson_cdf(floor(1.5), proj)   [api.py::prob_over, exact]
(The live power nudge +/-0.03..0.04 needs live BvP and is omitted here; it does
 not affect ranking/discrimination materially. Noted as a known simplification.)

Evaluation
----------
Per season (2025 development, 2026 HOLDOUT):
  - base rate of over_1_5
  - projection accuracy (mean proj vs mean actual TB, MAE)
  - AUC of prob_over vs realized over_1_5 (rank method, tie-corrected)
  - calibration deciles (mean predicted prob vs actual rate)
  - threshold table: for each prob cutoff, selected count, actual over-rate,
    lift vs base, and the fair American odds that rate implies -- so it can be
    compared against real TB OVER 1.5 book prices.

Run (Render)
------------
python -u total_bases_champion_preflight_a.py 2>&1 | tee /data/hr_model/total_bases_champion_preflight_a.log
"""

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

MODEL_COLUMNS = [
    "season_avg", "tb_per_pa", "rbi_per_pa", "runs_per_pa", "hr_rate",
    "bb_rate", "so_rate", "recent5_target", "recent15_target",
    "batting_order", "games_played",
]
LINE = 1.5
MODEL_DIR = Path("/data/models")
DEFAULT_BASELINE = "/data/hr_model/total_bases_clean_baseline_a_work/baseline.sqlite"


# ---- exact replication of api.py Poisson ----
def poisson_cdf(k, lam):
    if lam <= 0:
        return 1.0
    s, term = 0.0, math.exp(-lam)
    for i in range(k + 1):
        if i:
            term *= lam / i
        s += term
    return min(s, 1.0)


def prob_over(expected, line=LINE):
    return 1 - poisson_cdf(int(math.floor(line)), max(expected, 1e-6))


def fair_american(rate):
    """American odds implied by a win probability."""
    r = min(max(rate, 1e-6), 1 - 1e-6)
    if r >= 0.5:
        return -round(100 * r / (1 - r))
    return round(100 * (1 - r) / r)


def auc(scores, labels):
    """Tie-corrected AUC via rank sum (Mann-Whitney)."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0  # 1-based average rank
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    pos = sum(1 for l in labels if l == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return None
    sum_pos = sum(ranks[i] for i in range(n) if labels[i] == 1)
    return (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def evaluate(rows):
    probs = [r["prob"] for r in rows]
    labels = [r["over_1_5"] for r in rows]
    projs = [r["proj"] for r in rows]
    acts = [r["actual_tb"] for r in rows]
    n = len(rows)
    base = sum(labels) / n if n else 0

    res = {
        "n": n,
        "base_rate": round(base, 4),
        "mean_proj": round(sum(projs) / n, 3) if n else None,
        "mean_actual_tb": round(sum(acts) / n, 3) if n else None,
        "proj_mae": round(sum(abs(p - a) for p, a in zip(projs, acts)) / n, 3) if n else None,
        "auc": round(auc(probs, labels), 4) if n else None,
    }

    # calibration deciles by predicted prob
    idx = sorted(range(n), key=lambda i: probs[i])
    deciles = []
    for d in range(10):
        lo = d * n // 10
        hi = (d + 1) * n // 10
        seg = idx[lo:hi]
        if not seg:
            continue
        mp = sum(probs[i] for i in seg) / len(seg)
        ar = sum(labels[i] for i in seg) / len(seg)
        deciles.append({"decile": d + 1, "n": len(seg),
                        "mean_pred": round(mp, 4), "actual_rate": round(ar, 4)})
    res["calibration_deciles"] = deciles

    # threshold table
    tbl = []
    for t in (0.35, 0.40, 0.45, 0.50, 0.524, 0.55, 0.60, 0.63, 0.70):
        sel = [i for i in range(n) if probs[i] >= t]
        if not sel:
            tbl.append({"thresh": t, "n_selected": 0}); continue
        ar = sum(labels[i] for i in sel) / len(sel)
        tbl.append({
            "thresh": t,
            "n_selected": len(sel),
            "pct_of_pool": round(len(sel) / n, 3),
            "actual_over_rate": round(ar, 4),
            "lift_vs_base": round(ar - base, 4),
            "implied_fair_odds": fair_american(ar),
            "beats_-110_bet": ar >= 0.5238,
        })
    res["threshold_table"] = tbl
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=DEFAULT_BASELINE)
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    args = ap.parse_args()

    import numpy as np
    import xgboost as xgb

    mdir = Path(args.model_dir)
    mp = mdir / "batter_total_bases.json"
    cp = mdir / "batter_total_bases_columns.json"
    if not mp.exists() or not cp.exists():
        raise SystemExit(f"model or columns not found in {mdir}")
    cols = json.loads(cp.read_text())
    if cols != MODEL_COLUMNS:
        print(f"WARNING: model columns differ from expected.\n  model={cols}\n  expected={MODEL_COLUMNS}")
    booster = xgb.Booster()
    booster.load_model(str(mp))

    con = sqlite3.connect(f"file:{args.baseline}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    raw = con.execute(f"SELECT season, over_1_5, actual_tb, {', '.join(cols)} FROM tb_baseline").fetchall()
    con.close()
    print(f"loaded {len(raw)} baseline rows from {args.baseline}", flush=True)

    X = np.array([[r[c] for c in cols] for r in raw], dtype=np.float32)
    preds = booster.predict(xgb.DMatrix(X, feature_names=cols))

    by_season = {}
    for r, proj in zip(raw, preds):
        rec = {"season": r["season"], "over_1_5": int(r["over_1_5"]),
               "actual_tb": int(r["actual_tb"]),
               "proj": float(proj), "prob": prob_over(float(proj))}
        by_season.setdefault(r["season"], []).append(rec)

    report = {"script": "TOTAL_BASES_CHAMPION_PREFLIGHT_A", "line": LINE,
              "model": str(mp), "by_season": {}}

    for season in sorted(by_season):
        rows = by_season[season]
        res = evaluate(rows)
        report["by_season"][season] = res
        tag = "HOLDOUT" if season == "2026" else "development"
        print(f"\n================ SEASON {season} ({tag}) ================")
        print(f" n={res['n']}  base_rate(over1.5)={res['base_rate']}  AUC={res['auc']}")
        print(f" projection: mean_proj={res['mean_proj']} vs mean_actual_tb={res['mean_actual_tb']}  MAE={res['proj_mae']}")
        print(" calibration deciles (by predicted prob):")
        print("   decile     n   mean_pred  actual_rate")
        for d in res["calibration_deciles"]:
            print(f"   {d['decile']:>4}  {d['n']:>6}    {d['mean_pred']:.4f}     {d['actual_rate']:.4f}")
        print(" threshold table:")
        print("   thresh  n_sel  pool%   over_rate  lift   fair_odds  beats -110")
        for t in res["threshold_table"]:
            if t["n_selected"] == 0:
                print(f"   {t['thresh']:.3f}   0"); continue
            print(f"   {t['thresh']:.3f}  {t['n_selected']:>5}  {t['pct_of_pool']:.3f}   "
                  f"{t['actual_over_rate']:.4f}  {t['lift_vs_base']:+.4f}  "
                  f"{t['implied_fair_odds']:>+6}     {'YES' if t['beats_-110_bet'] else 'no'}")

    out_dir = Path("/data/hr_model") if Path("/data/hr_model").exists() else Path.cwd()
    (out_dir / "total_bases_champion_preflight_a_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {out_dir/'total_bases_champion_preflight_a_report.json'}")
    print("Read-only evaluation. No production state changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
