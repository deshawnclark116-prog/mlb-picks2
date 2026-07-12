#!/usr/bin/env python3
"""
HITS_CONTEXT_DEPLOY_VERIFY_A

Verifies the DEPLOYED artifacts in /data/models (batter_hits_context.json +
columns + expected_pa_lookup.json) are the validated model and reproduce the
research numbers, scored exactly the way api.py will serve them (expected_pa
from the lookup). Predictions-first: no odds.

This is the go/no-go check before flipping HITS_CONTEXT_ENABLED=1.

Confirms:
  1. The three artifacts load and the model's columns match what was built.
  2. Scoring the deployed model on the 2026 holdout with lookup-served
     expected_pa reproduces the validated metrics (AUC ~0.60, calibrated ECE).
  3. The lookup covers every (spot, side) the live path will request.

Read-only. Changes nothing.

Run (Render, after deploying the artifacts to /data/models)
-----------------------------------------------------------
python -u hits_context_deploy_verify_a.py 2>&1 | tee /data/hr_model/hits_context_deploy_verify_a.log
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

import hits_feature_discovery_b as fd

MODELS = Path("/data/models")


def main():
    import xgboost as xgb
    mp = MODELS / "batter_hits_context.json"
    cp = MODELS / "batter_hits_context_columns.json"
    lp = MODELS / "expected_pa_lookup.json"
    print("HITS_CONTEXT_DEPLOY_VERIFY_A\n============================", flush=True)
    for p in (mp, cp, lp):
        print(f"  {'OK ' if p.exists() else 'MISSING'} {p}")
    if not (mp.exists() and cp.exists() and lp.exists()):
        print("\nFAIL: deploy the artifacts to /data/models first.")
        return 1

    cols = json.loads(cp.read_text())
    lookup = json.loads(lp.read_text())
    booster = xgb.Booster(); booster.load_model(str(mp))
    print(f"\nmodel columns ({len(cols)}): {cols}")
    print(f"expected_pa lookup entries: {len(lookup)}")

    con = sqlite3.connect(f"file:{fd.SOURCE}?mode=ro", uri=True)
    data = fd.build(con); con.close()
    hol = data["2026"]

    # serve-consistent expected_pa (same override api.py applies via the lookup)
    missing_keys = set()
    for r in hol:
        f = r["f"]
        side = "home" if f.get("is_home") == 1.0 else "away"
        try:
            key = f"{int(f['batting_order'])}|{side}"
        except Exception:
            key = None
        lk = lookup.get(key) if key else None
        if lk is None and key:
            missing_keys.add(key)
        if lk is not None:
            f["expected_pa_v1"] = lk

    print(f"\nlookup coverage: {'COMPLETE' if not missing_keys else 'MISSING '+str(sorted(missing_keys))}")

    X = np.array([[r["f"].get(c, fd.NAN) for c in cols] for r in hol], dtype=np.float32)
    probs = booster.predict(xgb.DMatrix(X, feature_names=cols))
    res = fd.metrics(list(map(float, probs)), [r["y"] for r in hol])

    print("\nDEPLOYED MODEL ON 2026 HOLDOUT (served exactly as api.py will)")
    print(f"  n={res['n']}  base_rate={res['base_rate']}")
    print(f"  AUC={res['auc']}  log_loss={res['log_loss']}  Brier={res['brier']}  ECE={res['ece']}")
    print("  reliability (pred -> actual):")
    for b in res["reliability"]:
        print(f"    {b['bin']}  n={b['n']:>6}  pred={b['pred']:.3f}  actual={b['actual']:.3f}")

    # go/no-go: deployed model must be sharp (AUC well above base ~0.593) and calibrated
    ok = (res["auc"] is not None and res["auc"] >= 0.598 and res["ece"] <= 0.03 and not missing_keys)
    verdict = ("DEPLOYED_MODEL_VERIFIED_REPRODUCES_VALIDATION_SAFE_TO_ENABLE_HITS_CONTEXT"
               if ok else "DEPLOYED_MODEL_DID_NOT_REPRODUCE_VALIDATION_DO_NOT_ENABLE")
    print(f"\n  AUC>=0.598: {res['auc'] is not None and res['auc']>=0.598}   "
          f"ECE<=0.03: {res['ece']<=0.03}   lookup complete: {not missing_keys}")
    print(f"  VERDICT: {verdict}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
