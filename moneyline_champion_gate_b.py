#!/usr/bin/env python3
"""
MONEYLINE_CHAMPION_GATE_B

Rung 2b: gates Challenger B (clean_baseline_b -- adds starting pitcher
quality, the single isolated change from the "a" feature set) against the
exact same pre-registered criteria "a" failed on. "a" beat the heuristic
on calibration (ECE 0.0133 vs 0.0621) but LOST on AUC (0.5408 vs 0.5611)
-- the hypothesis being tested here is that starting-pitcher quality (the
single biggest real driver of an MLB game's outcome, and the thing real
sportsbooks price moneylines off, which "a" completely omitted) recovers
the missing discrimination without sacrificing the calibration edge.

Same three arms, same holdout, same discipline as gate_a (row-count-based
pick_val_cut, correct iteration_range from the first version):
  constant    2024 base rate
  heuristic   gamelines.py's live log5(pythag)+HOME_EDGE formula
  challenger  binary:logistic on the 19 clean_baseline_b features (14 team
              features + 5 new starting-pitcher features per side... note:
              5 pitcher features x 2 sides = 10, + 7 team features x 2 = 14,
              total 24 -- see FEATURES list for the exact count)

Pre-registered pass (identical bar to gate_a, unchanged after seeing "a"'s
result -- not loosened to make this pass):
  1. AUC >= heuristic's AUC on the 2025 holdout
  2. ECE <= 0.02
  3. log-loss beats BOTH the constant arm and the heuristic arm
  4. Brier score beats BOTH the constant arm and the heuristic arm

Read-only on the clean baseline. Writes only a report + the trained model
to its own work dir.

Run (Render or locally)
------------------------
python -u moneyline_champion_gate_b.py 2>&1 | tee moneyline_champion_gate_b.log
"""

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

BASELINE_DEFAULT = "moneyline_clean_baseline_b_work/baseline.sqlite"
WORKDIR_DEFAULT = "moneyline_champion_gate_b_work"

TEAM_COLS = ["win_pct", "run_diff_pg", "pythag_win_pct",
             "recent10_win_pct", "recent10_run_diff", "win_streak", "rest_days"]
PITCHER_COLS = ["pitcher_season_era", "pitcher_season_k9", "pitcher_season_bb9",
                "pitcher_recent3_era", "pitcher_starts"]
FEATURES = ([f"home_{c}" for c in TEAM_COLS] + [f"away_{c}" for c in TEAM_COLS]
            + [f"home_{c}" for c in PITCHER_COLS] + [f"away_{c}" for c in PITCHER_COLS])

PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 4,
          "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
          "min_child_weight": 5, "seed": 13}
NAN = float("nan")
HOME_EDGE = 0.035

MIN_VAL_ROWS = 100
VAL_FRAC = 0.2

GATE = {"max_ece": 0.02}


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
    cols = ["season", "date"] + FEATURES + ["existing_heuristic_home_pythag", "existing_heuristic_away_pythag", "home_win"]
    rows = con.execute(f"SELECT {', '.join(cols)} FROM moneyline_baseline_b").fetchall()
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
    print("MONEYLINE_CHAMPION_GATE_B\n=========================")
    print(f"features ({len(FEATURES)}): {FEATURES}")
    dev, hol = load(args.baseline)
    print(f"2024 (dev) rows: {len(dev)}   2025 (holdout) rows: {len(hol)}")

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
    verdict = ("MONEYLINE_CHAMPION_B_PASSES_GATE_READY_FOR_STABILITY_CONFIRMATION"
               if passed else "MONEYLINE_CHAMPION_B_DOES_NOT_CLEAR_GATE")

    print("\n============ PRE-REGISTERED GATE (unchanged from gate_a) ============")
    print(f"  AUC >= heuristic's {min_auc}:      {challenger['auc']:.4f}  -> {c1}")
    print(f"  ECE <= {GATE['max_ece']}:              {challenger['ece']:.4f}  -> {c2}")
    print(f"  logloss beats constant AND heuristic:  {challenger['log_loss']:.5f}  -> {c3}")
    print(f"  Brier beats constant AND heuristic:     {challenger['brier']:.5f}  -> {c4}")
    print(f"  VERDICT: {verdict}")

    bst.save_model(str(work / "moneyline.json"))
    (work / "moneyline_columns.json").write_text(json.dumps(FEATURES))
    report = {"script": "MONEYLINE_CHAMPION_GATE_B", "holdout": 2025,
              "constant": constant, "heuristic": heuristic, "challenger": challenger,
              "gate": {**GATE, "min_auc": min_auc},
              "passed": passed, "verdict": verdict, "importance": imp,
              "best_iteration": bst.best_iteration}
    (work / "moneyline_champion_gate_b_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nmodel + report written to {work}")
    print("No production wiring yet. Read-only on the baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
