#!/usr/bin/env python3
"""
NFL_RECEPTIONS_PRODUCTION_BUILDER_A

Rung 5 of the NFL pipeline. Turns the validated receptions champion (gate
passed, stability confirmed, population representativeness confirmed clean)
into deployable artifacts, mirroring hits_context_production_builder_a.py.

Retrains the frozen 12-feature spec on ALL data (2023+2024, last 10% of weeks
held out internally for early stopping) and exports:
    nfl_receptions.json           (xgboost model)
    nfl_receptions_columns.json   (feature order)

Deploying to /data/nfl_models is a separate, explicit step (same discipline as
every MLB model: nothing ships without a human copying the file).

Read-only on the clean baseline. Writes only its own work dir.

Run (Render)
------------
python -u nfl_receptions_production_builder_a.py 2>&1 | tee /data/nfl_model/nfl_receptions_production_builder_a.log
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

import nfl_receptions_champion_gate_a as g

WORKDIR_DEFAULT = "/data/nfl_model/nfl_receptions_production_builder_a_work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=g.BASELINE_DEFAULT)
    ap.add_argument("--workdir", default=WORKDIR_DEFAULT)
    args = ap.parse_args()
    import xgboost as xgb

    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    print("NFL_RECEPTIONS_PRODUCTION_BUILDER_A\n====================================")

    con = sqlite3.connect(f"file:{args.baseline}?mode=ro", uri=True)
    cols = ["season", "week"] + g.FEATURES + ["over_0_5"]
    rows = con.execute(f"SELECT {', '.join(cols)} FROM nfl_receptions_baseline "
                       f"ORDER BY season, week").fetchall()
    con.close()
    print(f"total rows (2023+2024): {len(rows)}")

    # chronological split across (season, week) so the internal validation
    # slice is the most recent weeks, matching every other builder in this project.
    order = sorted(set((r[0], r[1]) for r in rows))
    cut = order[int(len(order) * 0.9)]
    tr = [r for r in rows if (r[0], r[1]) < cut]
    va = [r for r in rows if (r[0], r[1]) >= cut]
    print(f"  train: {len(tr)}   internal val (>= {cut}): {len(va)}")

    def mat(rowset):
        X = np.array([[r[2 + i] if r[2 + i] is not None else g.NAN for i in range(len(g.FEATURES))]
                      for r in rowset], dtype=np.float32)
        y = np.array([r[-1] for r in rowset], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=g.FEATURES)

    bst = xgb.train(g.PARAMS, mat(tr), num_boost_round=800, evals=[(mat(va), "val")],
                    early_stopping_rounds=40, verbose_eval=False)
    print(f"  best_iteration={bst.best_iteration}")

    bst.save_model(str(work / "nfl_receptions.json"))
    (work / "nfl_receptions_columns.json").write_text(json.dumps(g.FEATURES))

    imp = bst.get_score(importance_type="gain")
    print("\nfinal model feature importance (gain):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"   {k:32s} {v:9.2f}")

    print(f"\nARTIFACTS WRITTEN (not yet deployed to /data/nfl_models):")
    print(f"   {work / 'nfl_receptions.json'}")
    print(f"   {work / 'nfl_receptions_columns.json'}")
    print("\nRead-only on the baseline. No production state changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
