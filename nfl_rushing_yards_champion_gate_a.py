#!/usr/bin/env python3
"""
NFL_RUSHING_YARDS_CHAMPION_GATE_A

Rung 3 of the NFL rushing-yards pipeline. Builds and formally validates the
FIRST honest rushing-yards model directly as a calibrated binary:logistic
classifier -- there is no prior champion, this market never had one.
Predictions-first: no odds. RB-only population (see clean_baseline for why).

Two arms on the untouched 2024 holdout (2023 = train, last 20% of 2023 weeks
held out internally for early stopping):
  constant   predicts the 2023 base rate for every row
  challenger binary:logistic on the 9 clean-baseline features

Pre-registered pass (written before this script has ever been run):
  1. AUC >= 0.58 on the 2024 holdout (real discrimination)
  2. ECE <= 0.02 (well calibrated -- same bar every shipped model cleared)
  3. log-loss beats the constant arm by >= 0.01
  4. Brier score beats the constant arm

CRITICAL (learned from the receptions bug): every predict() call after
reloading a saved model MUST pass iteration_range=(0, best_iteration+1), or
xgboost silently scores with all saved boosted rounds instead of the
early-stopping-optimal count the model was actually trained/gated to. This
script gets it right from the first version -- downstream scripts (stability,
recalibration if needed) must match it exactly.

Read-only on the clean baseline. Writes only a report + the trained model to
its own work dir.

Run (Render)
------------
python -u nfl_rushing_yards_champion_gate_a.py 2>&1 | tee /data/nfl_model/nfl_rushing_yards_champion_gate_a.log
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

BASELINE_DEFAULT = "/data/nfl_model/nfl_rushing_yards_clean_baseline_a_work/baseline.sqlite"
WORKDIR_DEFAULT = "/data/nfl_model/nfl_rushing_yards_champion_gate_a_work"

FEATURES = [
    "season_avg_rush_yards", "recent3_avg_rush_yards", "recent5_avg_rush_yards",
    "season_avg_carries", "recent3_avg_carries", "yards_per_carry",
    "opp_rush_yards_allowed_per_game", "is_home", "games_played",
]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 4,
          "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
          "min_child_weight": 5, "seed": 13}
NAN = float("nan")

GATE = {"min_auc": 0.58, "max_ece": 0.02, "min_logloss_gain": 0.01}


def auc(scores, labels):
    labels = np.asarray(labels)
    pos = labels.sum(); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    s = np.asarray(scores)[order]; i = 0; n = len(s)
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def metrics(probs, labels):
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(labels, dtype=float)
    n = len(y)
    brier = float(np.mean((p - y) ** 2))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    ece = 0.0; rel = []
    for b in range(10):
        m = (p >= b / 10) & (p < (b + 1) / 10) if b < 9 else (p >= 0.9)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        mp = float(p[m].mean()); ar = float(y[m].mean())
        ece += abs(mp - ar) * cnt / n
        rel.append({"bin": f"{b/10:.1f}-{(b+1)/10:.1f}", "n": cnt,
                    "pred": round(mp, 4), "actual": round(ar, 4)})
    return {"n": n, "base_rate": round(float(y.mean()), 4), "auc": round(auc(probs, labels), 4),
            "log_loss": round(ll, 5), "brier": round(brier, 5), "ece": round(ece, 4),
            "reliability": rel}


MIN_VAL_ROWS = 60
VAL_FRAC = 0.2


def pick_val_cut(dev):
    """Row-count-based internal val cut, not week-index-based.

    Eligible-row density is NOT uniform across weeks: it ramps up early in
    the season (players need 3+ prior games before they're eligible) and
    craters in the postseason (far fewer games survive each week). A plain
    80th-percentile-of-WEEKS cut can land the internal val slice entirely on
    the tiny playoff tail -- exactly what happened here (weeks >= 19, n=25,
    almost pure noise, causing early stopping to fire after ~1 round). Walk
    backward from the latest week accumulating ROWS until hitting a real
    floor, so the val slice is always big enough to guide early stopping.
    """
    from collections import Counter
    weeks = sorted({r[1] for r in dev})
    counts = Counter(r[1] for r in dev)
    total = len(dev)
    target = max(MIN_VAL_ROWS, int(total * VAL_FRAC))
    cum = 0
    cut = weeks[-1]
    for w in reversed(weeks):
        cum += counts[w]
        cut = w
        if cum >= target:
            break
    return cut


def load(baseline_path):
    con = sqlite3.connect(f"file:{baseline_path}?mode=ro", uri=True)
    cols = ["season", "week"] + FEATURES + ["over_line"]
    rows = con.execute(f"SELECT {', '.join(cols)} FROM nfl_rushing_yards_baseline").fetchall()
    con.close()
    dev = [r for r in rows if r[0] == 2023]
    hol = [r for r in rows if r[0] == 2024]
    return dev, hol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=BASELINE_DEFAULT)
    ap.add_argument("--workdir", default=WORKDIR_DEFAULT)
    args = ap.parse_args()
    import xgboost as xgb

    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    print("NFL_RUSHING_YARDS_CHAMPION_GATE_A\n=================================")
    dev, hol = load(args.baseline)
    print(f"2023 (dev) rows: {len(dev)}   2024 (holdout) rows: {len(hol)}")

    cut = pick_val_cut(dev)
    tr = [r for r in dev if r[1] < cut]
    va = [r for r in dev if r[1] >= cut]
    print(f"  train weeks < {cut}: {len(tr)}   internal val weeks >= {cut}: {len(va)}")

    def mat(rows):
        X = np.array([[r[2 + i] if r[2 + i] is not None else NAN for i in range(len(FEATURES))] for r in rows], dtype=np.float32)
        y = np.array([r[-1] for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=FEATURES)

    print("\ntraining challenger (binary:logistic) ...", flush=True)
    bst = xgb.train(PARAMS, mat(tr), num_boost_round=800, evals=[(mat(va), "val")],
                    early_stopping_rounds=40, verbose_eval=False)
    itr = (0, bst.best_iteration + 1)
    print(f"  best_iteration={bst.best_iteration}  scoring with iteration_range={itr}")

    probs_hol = bst.predict(mat(hol), iteration_range=itr)
    labels_hol = [r[-1] for r in hol]
    challenger = metrics(list(map(float, probs_hol)), labels_hol)

    train_rate = float(np.mean([r[-1] for r in tr]))
    constant = metrics([train_rate] * len(hol), labels_hol)

    print("\n============ 2024 HOLDOUT ============")
    print(f"  {'arm':12s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'constant':12s} {'n/a':>7s} {constant['log_loss']:>9.5f}  {constant['brier']:>7.5f} {constant['ece']:>7.4f}")
    print(f"  {'challenger':12s} {challenger['auc']:>7.4f}  {challenger['log_loss']:>9.5f}  {challenger['brier']:>7.5f} {challenger['ece']:>7.4f}")

    print("\nchallenger reliability (pred -> actual):")
    for b in challenger["reliability"]:
        print(f"   {b['bin']}  n={b['n']:>5}  pred={b['pred']:.3f}  actual={b['actual']:.3f}")

    imp = bst.get_score(importance_type="gain")
    print("\nfeature importance (gain):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"   {k:32s} {v:9.2f}")

    d_ll = constant["log_loss"] - challenger["log_loss"]
    c1 = challenger["auc"] >= GATE["min_auc"]
    c2 = challenger["ece"] <= GATE["max_ece"]
    c3 = d_ll >= GATE["min_logloss_gain"]
    c4 = challenger["brier"] < constant["brier"]
    passed = c1 and c2 and c3 and c4
    verdict = ("NFL_RUSHING_YARDS_CHAMPION_PASSES_GATE_READY_FOR_STABILITY_CONFIRMATION"
               if passed else "NFL_RUSHING_YARDS_CHAMPION_DOES_NOT_CLEAR_GATE")

    print("\n============ PRE-REGISTERED GATE ============")
    print(f"  AUC >= {GATE['min_auc']}:            {challenger['auc']:.4f}  -> {c1}")
    print(f"  ECE <= {GATE['max_ece']}:              {challenger['ece']:.4f}  -> {c2}")
    print(f"  logloss gain >= {GATE['min_logloss_gain']}:  {d_ll:+.5f}  -> {c3}")
    print(f"  Brier better than constant:  {challenger['brier']:.5f} < {constant['brier']:.5f}  -> {c4}")
    print(f"  VERDICT: {verdict}")

    bst.save_model(str(work / "nfl_rushing_yards.json"))
    (work / "nfl_rushing_yards_columns.json").write_text(json.dumps(FEATURES))
    report = {"script": "NFL_RUSHING_YARDS_CHAMPION_GATE_A", "holdout": 2024,
              "constant": constant, "challenger": challenger, "gate": GATE,
              "passed": passed, "verdict": verdict, "importance": imp,
              "best_iteration": bst.best_iteration}
    (work / "nfl_rushing_yards_champion_gate_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nmodel + report written to {work}")
    print("No production wiring yet. Read-only on the baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
