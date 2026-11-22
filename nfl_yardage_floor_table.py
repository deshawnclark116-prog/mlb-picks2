#!/usr/bin/env python3
"""NFL yardage: measured "safe number" table (no sportsbook line anywhere).

The live yardage projection (nfl_yardage_v2.py) is a median-type projection, so the player reaches it about half the time at every
size. That is honest but carries no pick. This script measures, on past games the model had not seen, how large a fraction of a
projection players actually reach, by market and projection size, and writes a small table the serving builder uses to attach:

  * projection_reach_probability_measured  - measured chance of reaching the rounded projection itself (about 50%);
  * safe_number / safe_number_probability  - a lower number that was reached about TARGET of the time in that projection size band.

Walk-forward: each test week is predicted by a model trained only on strictly earlier weeks (2025 week 6 onwards and all of 2026),
using exactly the live feature set. Rates are measured from those out-of-sample predictions only.

Run:  python -u nfl_yardage_floor_table.py --db nfl_models/nfl_model.sqlite --out nfl_models/nfl_yardage_floor_table.json
"""
import argparse
import json
import sqlite3

import numpy as np

import nfl_yardage_v2 as v2

TARGET = 0.75
BANDS = [(0, 20), (20, 35), (35, 50), (50, 70), (70, 10_000)]
MIN_BAND_N = 60
ROUNDS = 140


def whole(x):
    b = int(np.floor(x))
    return b + (1 if x - b >= 0.6 - 1e-10 else 0)


def walk_forward(con, market):
    rows, _, _ = v2.History(con, market).replay()
    keys = sorted({(m["season"], m["week"]) for m, _, _ in rows})
    test = [k for k in keys if (k[0] == 2025 and k[1] >= 6) or k[0] >= 2026]
    params = dict(v2.PARAMS)
    params["nthread"] = 4
    act, pred = [], []
    for k in test:
        tr = [r for r in rows if (r[0]["season"], r[0]["week"]) < k]
        te = [r for r in rows if (r[0]["season"], r[0]["week"]) == k]
        if len(tr) < 300 or not te:
            continue
        import xgboost as xgb
        b = xgb.train(params, xgb.DMatrix(v2.to_matrix(tr), label=[r[2] for r in tr], feature_names=v2.FEATURES),
                      num_boost_round=ROUNDS)
        pred += list(b.predict(xgb.DMatrix(v2.to_matrix(te), feature_names=v2.FEATURES)))
        act += [r[2] for r in te]
    return np.array(act, float), np.array(pred, float)


def table_for(act, pred):
    out = []
    for lo, hi in BANDS:
        i = (pred >= lo) & (pred < hi)
        n = int(i.sum())
        if n < MIN_BAND_N:
            continue
        a, p = act[i], pred[i]
        ratios = a / np.maximum(p, 1.0)
        r = float(np.quantile(ratios, 1.0 - TARGET))            # reached by TARGET of players
        k = np.array([whole(x) for x in p])
        out.append({"proj_min": lo, "proj_max": hi, "n": n,
                    "reach_projection": round(float(np.mean(a >= k)), 3),
                    "safe_ratio": round(min(max(r, 0.0), 1.0), 3),
                    "safe_probability": round(float(np.mean(a >= np.floor(np.minimum(np.maximum(r, 0.0), 1.0) * p))), 3)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="nfl_models/nfl_model.sqlite")
    ap.add_argument("--out", default="nfl_models/nfl_yardage_floor_table.json")
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    doc = {"schema": "NFL_YARDAGE_SAFE_NUMBER_TABLE_V1", "target": TARGET, "sportsbook_line_used": False,
           "method": "walk-forward out-of-sample predictions of the live v2 yardage model; each week uses only earlier weeks",
           "markets": {}}
    for m in v2.MARKETS:
        act, pred = walk_forward(con, m)
        doc["markets"][m] = {"n": int(len(act)), "bands": table_for(act, pred)}
        print(m, len(act), json.dumps(doc["markets"][m]["bands"]))
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    return 0


def lookup(table, market, proj):
    """(reach_probability, safe_number, safe_probability) for a projection, or None if the table has no band for it."""
    bands = ((table or {}).get("markets", {}).get(market) or {}).get("bands") or []
    for b in bands:
        if b["proj_min"] <= proj < b["proj_max"]:
            return b["reach_projection"], int(np.floor(b["safe_ratio"] * proj)), b["safe_probability"]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
