#!/usr/bin/env python3
"""
MONEYLINE_CHAMPION_GATE_A

Rung 2 of the moneyline pipeline. Trains the first honest moneyline model
as a calibrated binary:logistic classifier and formally validates it
against the untouched 2025 holdout. Predictions-first: no odds used in
training or gating.

THREE arms on the 2025 holdout (2024 = train, row-count-based internal
val slice for early stopping -- see pick_val_cut, the fix already learned
the hard way on the NFL rushing-yards pipeline: a week/date-index-based
cut can land the val slice on an unrepresentative tail):
  constant    predicts the 2024 base rate for every row
  heuristic   reproduction of gamelines.py's actual live formula:
              log5(pythag_home, pythag_away) + HOME_EDGE, using the same
              Pythagorean win expectation already stored in the baseline
              -- this is the REAL bar, not a strawman. It's a legitimate,
              well-established sabermetric approach with no training, and
              moneyline has run on exactly this formula unmodified in
              production. A new model must beat it, not just clear an
              absolute floor.
  challenger  binary:logistic on the 14 clean-baseline features

Pre-registered pass (written before this script has ever been run):
  1. AUC >= heuristic's AUC on the 2025 holdout (must out-rank the formula
     already in production, not just discriminate at all)
  2. ECE <= 0.02 (same bar every shipped model has cleared)
  3. log-loss beats BOTH the constant arm and the heuristic arm
  4. Brier score beats BOTH the constant arm and the heuristic arm

CRITICAL (learned from the receptions bug): every predict() call after
reloading a saved model MUST pass iteration_range=(0, best_iteration+1).
This script gets it right from the first version.

CRITICAL (learned from the rushing-yards bug): the internal validation
split for early stopping must be row-count-based, not week/date-index
based, or it can land entirely on an unrepresentative tail and starve
early stopping of a usable signal. pick_val_cut() does this from the
start here too.

Read-only on the clean baseline. Writes only a report + the trained model
to its own work dir.

Run (Render or locally)
------------------------
python -u moneyline_champion_gate_a.py 2>&1 | tee moneyline_champion_gate_a.log
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

BASELINE_DEFAULT = "moneyline_clean_baseline_a_work/baseline.sqlite"
WORKDIR_DEFAULT = "moneyline_champion_gate_a_work"

FEATURES = [
    "home_win_pct", "home_run_diff_pg", "home_pythag_win_pct",
    "home_recent10_win_pct", "home_recent10_run_diff", "home_win_streak", "home_rest_days",
    "away_win_pct", "away_run_diff_pg", "away_pythag_win_pct",
    "away_recent10_win_pct", "away_recent10_run_diff", "away_win_streak", "away_rest_days",
]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 4,
          "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
          "min_child_weight": 5, "seed": 13}
NAN = float("nan")
HOME_EDGE = 0.035  # matches gamelines.py exactly -- reproducing the live formula

MIN_VAL_ROWS = 100
VAL_FRAC = 0.2

GATE = {"max_ece": 0.02}  # min_auc set dynamically to the heuristic's own AUC (see main)


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


def log5(p_a, p_b):
    den = p_a + p_b - 2 * p_a * p_b
    if den == 0:
        return 0.5
    return (p_a - p_a * p_b) / den


def heuristic_prob(home_pyth, away_pyth):
    base = log5(home_pyth, away_pyth)
    return min(0.95, max(0.05, base + HOME_EDGE))


def pick_val_cut(dev_dates_sorted_with_counts, total, min_val_rows=MIN_VAL_ROWS, val_frac=VAL_FRAC):
    """Row-count-based internal val cut over DATES (not weeks -- moneyline
    has ~15 games/day, so date is the natural grouping unit). Walk backward
    from the latest date accumulating rows until hitting a real floor, so
    the val slice is always big enough to guide early stopping regardless
    of how row density is shaped across the season."""
    target = max(min_val_rows, int(total * val_frac))
    cum = 0
    cut = dev_dates_sorted_with_counts[-1][0]
    for d, cnt in reversed(dev_dates_sorted_with_counts):
        cum += cnt
        cut = d
        if cum >= target:
            break
    return cut


def load(baseline_path):
    con = sqlite3.connect(f"file:{baseline_path}?mode=ro", uri=True)
    cols = ["season", "date"] + FEATURES + ["home_pythag_win_pct", "away_pythag_win_pct", "home_win"]
    rows = con.execute(f"SELECT {', '.join(cols)} FROM moneyline_baseline").fetchall()
    con.close()
    dev = [r for r in rows if r[0] == 2024]
    hol = [r for r in rows if r[0] == 2025]
    return dev, hol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=BASELINE_DEFAULT)
    ap.add_argument("--workdir", default=WORKDIR_DEFAULT)
    args = ap.parse_args()
    import xgboost as xgb

    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    print("MONEYLINE_CHAMPION_GATE_A\n=========================")
    dev, hol = load(args.baseline)
    print(f"2024 (dev) rows: {len(dev)}   2025 (holdout) rows: {len(hol)}")

    from collections import Counter
    date_counts = Counter(r[1] for r in dev)
    dates_sorted = sorted(date_counts.items())
    cut = pick_val_cut(dates_sorted, len(dev))
    tr = [r for r in dev if r[1] < cut]
    va = [r for r in dev if r[1] >= cut]
    print(f"  train dates < {cut}: {len(tr)}   internal val dates >= {cut}: {len(va)}")

    n_meta = 2  # season, date

    def mat(rows):
        X = np.array([[r[n_meta + i] if r[n_meta + i] is not None else NAN for i in range(len(FEATURES))]
                      for r in rows], dtype=np.float32)
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

    # heuristic arm: reproduce gamelines.py's live formula exactly, from the
    # same stored Pythagorean values (idx -3, -2 before home_win at idx -1)
    heur_probs = [heuristic_prob(r[-3], r[-2]) for r in hol]
    heuristic = metrics(heur_probs, labels_hol)

    train_rate = float(np.mean([r[-1] for r in tr]))
    constant = metrics([train_rate] * len(hol), labels_hol)

    print("\n============ 2025 HOLDOUT ============")
    print(f"  {'arm':12s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'constant':12s} {'n/a':>7s} {constant['log_loss']:>9.5f}  {constant['brier']:>7.5f} {constant['ece']:>7.4f}")
    print(f"  {'heuristic':12s} {heuristic['auc']:>7.4f}  {heuristic['log_loss']:>9.5f}  {heuristic['brier']:>7.5f} {heuristic['ece']:>7.4f}")
    print(f"  {'challenger':12s} {challenger['auc']:>7.4f}  {challenger['log_loss']:>9.5f}  {challenger['brier']:>7.5f} {challenger['ece']:>7.4f}")

    print("\nchallenger reliability (pred -> actual):")
    for b in challenger["reliability"]:
        print(f"   {b['bin']}  n={b['n']:>5}  pred={b['pred']:.3f}  actual={b['actual']:.3f}")

    imp = bst.get_score(importance_type="gain")
    print("\nfeature importance (gain):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"   {k:32s} {v:9.2f}")

    min_auc = heuristic["auc"]
    c1 = challenger["auc"] >= min_auc
    c2 = challenger["ece"] <= GATE["max_ece"]
    c3 = challenger["log_loss"] < constant["log_loss"] and challenger["log_loss"] < heuristic["log_loss"]
    c4 = challenger["brier"] < constant["brier"] and challenger["brier"] < heuristic["brier"]
    passed = c1 and c2 and c3 and c4
    verdict = ("MONEYLINE_CHAMPION_PASSES_GATE_READY_FOR_STABILITY_CONFIRMATION"
               if passed else "MONEYLINE_CHAMPION_DOES_NOT_CLEAR_GATE")

    print("\n============ PRE-REGISTERED GATE ============")
    print(f"  AUC >= heuristic's {min_auc}:      {challenger['auc']:.4f}  -> {c1}")
    print(f"  ECE <= {GATE['max_ece']}:              {challenger['ece']:.4f}  -> {c2}")
    print(f"  logloss beats constant AND heuristic:  {challenger['log_loss']:.5f}  -> {c3}")
    print(f"  Brier beats constant AND heuristic:     {challenger['brier']:.5f}  -> {c4}")
    print(f"  VERDICT: {verdict}")

    bst.save_model(str(work / "moneyline.json"))
    (work / "moneyline_columns.json").write_text(json.dumps(FEATURES))
    report = {"script": "MONEYLINE_CHAMPION_GATE_A", "holdout": 2025,
              "constant": constant, "heuristic": heuristic, "challenger": challenger,
              "gate": {**GATE, "min_auc": min_auc},
              "passed": passed, "verdict": verdict, "importance": imp,
              "best_iteration": bst.best_iteration}
    (work / "moneyline_champion_gate_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nmodel + report written to {work}")
    print("No production wiring yet. Read-only on the baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
