#!/usr/bin/env python3
"""
HITS_CONTEXT_PRODUCTION_BUILDER_A

Turns the validated base+easy hits research win into deployable production
artifacts, using ONLY zero-train/serve-skew features so live == training by
construction. Predictions-first: no odds.

Production feature set (13) = champion base (8) + zero-skew easy (5):
    platoon_advantage, pitcher_is_R, is_home, expected_pa_v1, recent_xbh_avg
The opposing-pitcher rates (weakest in importance, and the only features with
batter_games-vs-live-API skew risk) are intentionally EXCLUDED.

Two stages:
1. VALIDATION (train 2025 -> score 2026 holdout): confirm base+zero-skew retains
   the gain vs base, and is not materially worse than base+all-easy. If zero-skew
   keeps ~the win, we ship it and skip the skew entirely.
2. PRODUCTION: retrain the frozen 13-feature model on ALL data (2025+2026) and
   export the deployable artifacts, plus the expected_pa lookup table.

Exports (to work dir; deploying to /data/models is a separate explicit step):
    batter_hits_context.json           (xgboost model)
    batter_hits_context_columns.json   (feature order)
    expected_pa_lookup.json            ({"<spot>|<side>": avg_pa})

Read-only on hr_model.sqlite. Touches no production model or code.

Run (Render)
------------
python -u hits_context_production_builder_a.py 2>&1 | tee /data/hr_model/hits_context_production_builder_a.log
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

import hits_feature_discovery_b as fd

WORKDIR = Path("/data/hr_model/hits_context_production_builder_a_work")
ZERO_SKEW = ["platoon_advantage", "pitcher_is_R", "is_home", "expected_pa_v1", "recent_xbh_avg"]
PROD_FEATURES = fd.BASE + ZERO_SKEW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=fd.SOURCE)
    ap.add_argument("--workdir", default=str(WORKDIR))
    args = ap.parse_args()
    import xgboost as xgb
    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(f"file:{args.source}?mode=ro", uri=True)
    print("HITS_CONTEXT_PRODUCTION_BUILDER_A\n=================================", flush=True)
    print("building dataset ...", flush=True)
    data = fd.build(con)
    dev, hol = data["2025"], data["2026"]

    def mat(rows, feats):
        X = np.array([[r["f"].get(k, fd.NAN) for k in feats] for r in rows], dtype=np.float32)
        y = np.array([r["y"] for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=feats)

    # ---- Stage 1: validation on the 2026 holdout ----
    dates = sorted({r["game_date"] for r in dev})
    cut = dates[int(len(dates) * 0.8)]
    tr = [r for r in dev if r["game_date"] < cut]
    va = [r for r in dev if r["game_date"] >= cut]

    def train_val(feats):
        b = xgb.train(fd.PARAMS, mat(tr, feats), num_boost_round=800,
                      evals=[(mat(va, feats), "val")], early_stopping_rounds=40,
                      verbose_eval=False)
        p = b.predict(mat(hol, feats), iteration_range=(0, b.best_iteration + 1))
        return fd.metrics(list(map(float, p)), [r["y"] for r in hol])

    print("\nSTAGE 1: VALIDATION (train 2025 -> 2026 holdout)")
    arms = {"base": fd.BASE, "base+zeroskew (PROD)": PROD_FEATURES, "base+all_easy": fd.BASE + fd.EASY}
    v = {name: train_val(feats) for name, feats in arms.items()}
    print(f"  {'arm':22s} {'AUC':>7s} {'logloss':>9s} {'Brier':>9s} {'ECE':>7s}")
    for name in arms:
        r = v[name]
        print(f"  {name:22s} {r['auc']:.4f}  {r['log_loss']:.5f}  {r['brier']:.5f}  {r['ece']:.4f}")
    d_auc = v["base+zeroskew (PROD)"]["auc"] - v["base"]["auc"]
    d_ll = v["base"]["log_loss"] - v["base+zeroskew (PROD)"]["log_loss"]
    retain = d_auc >= 0.005 and d_ll >= 0.001
    lost_vs_full = v["base+all_easy"]["auc"] - v["base+zeroskew (PROD)"]["auc"]
    print(f"\n  zero-skew vs base:      dAUC={d_auc:+.4f}  dlogloss={d_ll:+.5f}  -> {'RETAINS GAIN' if retain else 'WEAK'}")
    print(f"  gain given up vs all-easy (incl. opp-pitcher): dAUC={lost_vs_full:+.4f}")

    # ---- expected_pa lookup (matches hr_expected_pa_a semantics: avg PA by spot x side) ----
    cur = con.execute("""SELECT lineup_spot, side, AVG(plate_appearances)
                         FROM batter_games
                         WHERE lineup_spot BETWEEN 1 AND 9 AND side IN ('home','away')
                           AND plate_appearances IS NOT NULL
                         GROUP BY lineup_spot, side""")
    lookup = {f"{int(sp)}|{sd}": round(float(avg), 4) for sp, sd, avg in cur.fetchall()}
    con.close()
    print(f"\nexpected_pa lookup: {len(lookup)} (spot|side) entries")
    for k in sorted(lookup, key=lambda x: (x.split('|')[1], int(x.split('|')[0])))[:4]:
        print(f"    {k} -> {lookup[k]}")

    # ---- Stage 2: production model on ALL data ----
    print("\nSTAGE 2: PRODUCTION (retrain frozen 13-feature model on ALL data)")
    allrows = dev + hol
    adates = sorted({r["game_date"] for r in allrows})
    acut = adates[int(len(adates) * 0.9)]
    atr = [r for r in allrows if r["game_date"] < acut]
    ava = [r for r in allrows if r["game_date"] >= acut]
    final = xgb.train(fd.PARAMS, mat(atr, PROD_FEATURES), num_boost_round=800,
                      evals=[(mat(ava, PROD_FEATURES), "val")], early_stopping_rounds=40,
                      verbose_eval=False)
    print(f"  trained on {len(allrows)} rows, best_iteration={final.best_iteration}")

    final.save_model(str(work / "batter_hits_context.json"))
    (work / "batter_hits_context_columns.json").write_text(json.dumps(PROD_FEATURES))
    (work / "expected_pa_lookup.json").write_text(json.dumps(lookup, indent=2))

    imp = final.get_score(importance_type="gain")
    print("\n  final model feature importance (gain):")
    for k, val in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"    {k:24s} {val:9.2f}")

    print("\nARTIFACTS WRITTEN (to work dir; NOT yet deployed to /data/models):")
    for f in ("batter_hits_context.json", "batter_hits_context_columns.json", "expected_pa_lookup.json"):
        print(f"    {work / f}")
    report = {"script": "HITS_CONTEXT_PRODUCTION_BUILDER_A",
              "prod_features": PROD_FEATURES, "validation": v,
              "zero_skew_retains_gain": bool(retain),
              "gain_given_up_vs_all_easy_auc": float(lost_vs_full),
              "expected_pa_lookup_entries": len(lookup),
              "final_importance": imp}
    out = Path("/data/hr_model") if Path("/data/hr_model").exists() else work
    (out / "hits_context_production_builder_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out/'hits_context_production_builder_a_report.json'}")
    print("Read-only on hr_model.sqlite. No production model or code changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
