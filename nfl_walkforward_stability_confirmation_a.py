#!/usr/bin/env python3
"""
NFL_WALKFORWARD_STABILITY_CONFIRMATION_A

Stability rung for the two markets that passed their walk-forward gates
(rushing yards, receiving yards). Confirms the SHIPPING probabilities --
frozen champion + weekly walk-forward Platt, reproduced exactly as gated --
are not a lucky stretch of the 2024 holdout and don't quietly fail for a
subgroup or a stretch of the season, before any live wiring.

Why not reuse nfl_receptions_stability_confirmation_a's bars: they were
tuned to a 0.79-AUC market (CI lower >= 0.65, weekly AUC >= 0.65 with 100+
rows/week). These markets run AUC 0.62-0.66 with 20-60 rows/week -- those
bars are mechanically unpassable here regardless of stability, and its
fixed slice-ECE bar (0.04) is the same no-power fixed bar this pipeline
already replaced. Bars below are pre-registered for THESE markets, written
before this script's first run:

  STABLE verdict (all must hold, per market):
  1. Week-block bootstrap (B=2000, resampling weeks with replacement):
     AUC 95% CI lower bound >= 0.55 -- convincingly better than chance
     even in unlucky week-resamples.
  2. Season quarters (holdout weeks split into 4 contiguous blocks
     balanced by row count): AUC >= 0.50 in at least 3 of 4 quarters AND
     no quarter below 0.45 -- catches a real regime collapse (e.g. model
     works early season, dies late) without failing on tiny-block noise.
  3. Home/away slices (min 100 rows): AUC >= 0.52 AND calibration
     bootstrap goodness-of-fit p >= 0.10 -- no quietly broken subgroup,
     judged with the power-adjusted test, not a fixed ECE bar.

Note on repeated holdout use: this is the pipeline's designed next rung
(same role stability confirmation played for receptions), re-examining the
already-validated probabilities for consistency -- not a new modeling
attempt trying to force a pass out of the same data.

Read-only. Writes only a report per market.

Run
---
python -u nfl_walkforward_stability_confirmation_a.py --market rushing_yards
python -u nfl_walkforward_stability_confirmation_a.py --market receiving_yards
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

import nfl_rushing_yards_champion_gate_a as g  # metrics/auc/pick_val_cut/NAN shared
from nfl_rushing_yards_recalibration_a import fit_platt, apply_platt
from nfl_receiving_yards_clean_baseline_a import MODEL_COLUMNS as RECV_FEATURES

MARKETS = {
    "rushing_yards": {
        "baseline": "nfl_models/nfl_rushing_yards_clean_baseline_a_work/baseline.sqlite",
        "table": "nfl_rushing_yards_baseline",
        "features": g.FEATURES,
        "model_dir": "nfl_models/nfl_rushing_yards_champion_gate_a_work",
        "stem": "nfl_rushing_yards",
    },
    "receiving_yards": {
        "baseline": "nfl_models/nfl_receiving_yards_clean_baseline_a_work/baseline.sqlite",
        "table": "nfl_receiving_yards_baseline",
        "features": RECV_FEATURES,
        "model_dir": "nfl_models/nfl_receiving_yards_champion_walkforward_gate_a_work",
        "stem": "nfl_receiving_yards",
    },
}

B_BOOTSTRAP = 2000
B_CALIB = 10_000
SEED = 20260722
MIN_SLICE_ROWS = 100
CI_LO_BAR = 0.55
QUARTER_AUC_BAR = 0.50
QUARTER_MIN_PASS = 3
QUARTER_FLOOR = 0.45
SLICE_AUC_BAR = 0.52
SLICE_CALIB_MIN_P = 0.10


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


def bootstrap_calib_p(probs, y, rng, b_sims=B_CALIB):
    observed = ece_only(probs, y)
    probs = np.asarray(probs, dtype=float)
    worse = 0
    for _ in range(b_sims):
        sim_y = rng.binomial(1, probs).astype(float)
        if ece_only(probs, sim_y) >= observed:
            worse += 1
    return observed, worse / b_sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=sorted(MARKETS), required=True)
    args = ap.parse_args()
    import xgboost as xgb

    cfg = MARKETS[args.market]
    feats = cfg["features"]
    mdir = Path(cfg["model_dir"])
    print(f"NFL_WALKFORWARD_STABILITY_CONFIRMATION_A  [{args.market}]\n" + "=" * 52)

    con = sqlite3.connect(f"file:{cfg['baseline']}?mode=ro", uri=True)
    cols = ["season", "week"] + feats + ["over_line"]
    rows = con.execute(f"SELECT {', '.join(cols)} FROM {cfg['table']}").fetchall()
    con.close()
    dev = [r for r in rows if r[0] == 2023]
    hol = [r for r in rows if r[0] == 2024]

    bst = xgb.Booster(); bst.load_model(str(mdir / f"{cfg['stem']}.json"))
    feat_cols = json.loads((mdir / f"{cfg['stem']}_columns.json").read_text())
    assert feat_cols == feats
    itr = (0, bst.best_iteration + 1)
    print(f"scoring with iteration_range={itr}")

    def mat(rr):
        X = np.array([[r[2 + i] if r[2 + i] is not None else g.NAN
                       for i in range(len(feats))] for r in rr], dtype=np.float32)
        return xgb.DMatrix(X, feature_names=feats)

    # reproduce the exact walk-forward probabilities the gate validated
    cut = g.pick_val_cut(dev)
    va = [r for r in dev if r[1] >= cut]
    va_raw = np.asarray(bst.predict(mat(va), iteration_range=itr), dtype=float)
    va_y = np.asarray([r[-1] for r in va], dtype=float)
    hol_raw = np.asarray(bst.predict(mat(hol), iteration_range=itr), dtype=float)
    y = np.asarray([r[-1] for r in hol], dtype=float)
    week = np.asarray([r[1] for r in hol])

    probs = np.empty(len(y))
    for w in sorted(set(week.tolist())):
        seen = week < w
        a, b = fit_platt(np.concatenate([va_raw, hol_raw[seen]]),
                         np.concatenate([va_y, y[seen]]))
        if a <= 0:
            a, b = 1.0, 0.0
        mask = week == w
        probs[mask] = apply_platt(hol_raw[mask], a, b)

    point_auc = g.auc(probs, y)
    print(f"holdout n={len(y)}  walk-forward point AUC={point_auc:.4f}  "
          f"ECE={ece_only(probs, y):.4f}")

    rng = np.random.default_rng(SEED)

    # ---- 1. week-block bootstrap ----
    uniq = np.array(sorted(set(week.tolist())))
    idx_by = {w: np.where(week == w)[0] for w in uniq}
    aucs = np.empty(B_BOOTSTRAP)
    for b in range(B_BOOTSTRAP):
        idx = np.concatenate([idx_by[w] for w in rng.choice(uniq, len(uniq), replace=True)])
        aucs[b] = g.auc(probs[idx], y[idx])
    ci_lo, ci_hi = (float(x) for x in np.percentile(aucs, [2.5, 97.5]))
    print(f"\nWEEK-BLOCK BOOTSTRAP (B={B_BOOTSTRAP}): AUC mean={aucs.mean():.4f}  "
          f"95%CI=[{ci_lo:.4f},{ci_hi:.4f}]")

    # ---- 2. season quarters (contiguous, balanced by rows) ----
    print("\nSEASON QUARTERS (contiguous week blocks, balanced by row count):")
    order = np.argsort(week, kind="mergesort")
    boundaries = [int(len(y) * f) for f in (0.25, 0.5, 0.75)]
    quarter_of_rank = np.searchsorted(boundaries, np.arange(len(y)), side="right")
    quarter = np.empty(len(y), dtype=int)
    quarter[order] = quarter_of_rank
    q_rows = []
    q_pass = 0
    q_floor_ok = True
    for qi in range(4):
        idx = np.where(quarter == qi)[0]
        wks = sorted(set(week[idx].tolist()))
        a = g.auc(probs[idx], y[idx])
        ok = a >= QUARTER_AUC_BAR
        q_pass += int(ok)
        if a < QUARTER_FLOOR:
            q_floor_ok = False
        q_rows.append({"quarter": qi + 1, "weeks": f"{wks[0]}-{wks[-1]}",
                       "n": int(len(idx)), "auc": round(float(a), 4), "ok": bool(ok)})
        print(f"  Q{qi+1} (weeks {wks[0]:>2}-{wks[-1]:>2}): n={len(idx):>4}  AUC={a:.4f}  "
              f"{'ok' if ok else 'LOW'}")

    # ---- 3. home/away slices ----
    print(f"\nSLICES (home/away, min {MIN_SLICE_ROWS} rows; AUC >= {SLICE_AUC_BAR} "
          f"and calibration p >= {SLICE_CALIB_MIN_P}):")
    ih = 2 + feats.index("is_home")
    home_mask = np.asarray([r[ih] == 1 for r in hol])
    slice_rows = []
    slices_ok = slices_elig = 0
    for name, mask in (("home", home_mask), ("away", ~home_mask)):
        idx = np.where(mask)[0]
        if len(idx) < MIN_SLICE_ROWS:
            print(f"  {name:5s} n={len(idx)} (skip)"); continue
        a = g.auc(probs[idx], y[idx])
        s_ece, s_p = bootstrap_calib_p(probs[idx], y[idx], rng)
        ok = a >= SLICE_AUC_BAR and s_p >= SLICE_CALIB_MIN_P
        slices_elig += 1
        slices_ok += int(ok)
        slice_rows.append({"slice": name, "n": int(len(idx)), "auc": round(float(a), 4),
                           "ece": round(s_ece, 4), "calib_p": s_p, "ok": bool(ok)})
        print(f"  {name:5s} n={len(idx):>4}  AUC={a:.4f}  ECE={s_ece:.4f}  "
              f"calib_p={s_p:.4f}  {'ok' if ok else 'FAIL'}")

    c1 = bool(ci_lo >= CI_LO_BAR)
    c2 = bool(q_pass >= QUARTER_MIN_PASS and q_floor_ok)
    c3 = bool(slices_elig > 0 and slices_ok == slices_elig)
    stable = c1 and c2 and c3
    up = args.market.upper()
    verdict = (f"NFL_{up}_WALKFORWARD_STABLE_READY_FOR_LIVE_WIRING" if stable
               else f"NFL_{up}_WALKFORWARD_NOT_YET_STABLE")

    print("\n================ PRE-REGISTERED VERDICT ================")
    print(f"  bootstrap AUC CI lower >= {CI_LO_BAR}:  {ci_lo:.4f} -> {c1}")
    print(f"  quarters >= {QUARTER_AUC_BAR} in >= {QUARTER_MIN_PASS}/4, none < {QUARTER_FLOOR}: "
          f"{q_pass}/4 pass, floor_ok={q_floor_ok} -> {c2}")
    print(f"  all slices pass:                {slices_ok}/{slices_elig} -> {c3}")
    print(f"  VERDICT: {verdict}")

    report = {"script": "NFL_WALKFORWARD_STABILITY_CONFIRMATION_A", "market": args.market,
              "point_auc": round(float(point_auc), 4),
              "bootstrap": {"B": B_BOOTSTRAP, "mean": round(float(aucs.mean()), 4),
                             "ci_lo": ci_lo, "ci_hi": ci_hi, "bar": CI_LO_BAR},
              "quarters": q_rows, "slices": slice_rows,
              "stable": stable, "verdict": verdict}
    report_path = mdir / f"nfl_walkforward_stability_confirmation_a_{args.market}_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {report_path}")
    print("Read-only. No production change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
