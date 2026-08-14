#!/usr/bin/env python3
"""
NFL_PRESEASON_CHAMPION_GATE_A

First honest validation pass on the preseason rushing/receiving yards
baselines. Same discipline as nfl_rushing_yards_champion_gate_a.py
(constant vs challenger on an untouched holdout, pre-registered gate),
adapted for a much smaller, multi-season-pooled population:

  - dev = seasons 2021-2024 pooled (not a single season -- preseason
    rows per season are too few on their own), holdout = 2025
    (untouched, most complete-coverage labeled season)
  - internal train/val split within dev is by chronological game_date
    across seasons (not week-within-season, since eligibility here is
    already cross-season)

Handles both markets in one script since the populations are small
enough that per-market files would be near-duplicate boilerplate.

Pre-registered pass (written before this script has ever been run):
  1. AUC >= 0.58 on the 2025 holdout
  2. ECE <= 0.05 (loosened from the regular-season 0.02 bar -- holdout
     here is ~140-260 rows, an order of magnitude smaller, so binned
     calibration error is inherently noisier; still a real bar)
  3. log-loss beats the constant arm by >= 0.01
  4. Brier score beats the constant arm

An honest FAIL here (no real signal in this population) is an expected,
useful possible outcome, not a bug to fix.

Read-only on the clean baseline. Writes only a report + trained model.

Run:
    python -u nfl_preseason_champion_gate_a.py
"""
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

BASELINE_WORKDIR = Path("nfl_models/nfl_preseason_clean_baseline_a_work")
WORKDIR = Path("nfl_models/nfl_preseason_champion_gate_a_work")

MARKETS = {
    "preseason_rushing_yards": {"stem": "preseason_rushing_yards"},
    "preseason_receiving_yards": {"stem": "preseason_receiving_yards"},
}
FEATURES = ["career_avg_yards", "recent3_avg_yards", "recent5_avg_yards",
            "career_avg_rate", "recent3_avg_rate", "yards_per_unit",
            "is_home", "games_played"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3,
          "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
          "min_child_weight": 5, "seed": 13}
NAN = float("nan")
DEV_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUT_SEASON = 2025

GATE = {"min_auc": 0.58, "max_ece": 0.05, "min_logloss_gain": 0.01}
MIN_VAL_ROWS = 40
VAL_FRAC = 0.2


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


def pick_val_cut(dev_sorted_dates):
    """Row-count-based internal val cut on chronological game_date, mirroring
    the regular-season script's row-count logic (density isn't uniform)."""
    total = len(dev_sorted_dates)
    target = max(MIN_VAL_ROWS, int(total * VAL_FRAC))
    cut_idx = max(0, total - target)
    return dev_sorted_dates[cut_idx]


def load(baseline_path, stem):
    con = sqlite3.connect(f"file:{baseline_path}?mode=ro", uri=True)
    cols = ["season", "game_date"] + FEATURES + ["over_line"]
    rows = con.execute(f"SELECT {', '.join(cols)} FROM {stem}_baseline ORDER BY game_date").fetchall()
    con.close()
    dev = [r for r in rows if r[0] in DEV_SEASONS]
    hol = [r for r in rows if r[0] == HOLDOUT_SEASON]
    return dev, hol


def mat(rows, xgb):
    n_feat = len(FEATURES)
    X = np.array([[r[2 + i] if r[2 + i] is not None else NAN for i in range(n_feat)] for r in rows], dtype=np.float32)
    y = np.array([r[-1] for r in rows], dtype=np.float32)
    return xgb.DMatrix(X, label=y, feature_names=FEATURES)


def run_market(mkt_key, stem, workdir):
    import xgboost as xgb
    print(f"\n{'='*70}\n{mkt_key}\n{'='*70}")
    baseline_path = BASELINE_WORKDIR / f"{stem}_baseline.sqlite"
    dev, hol = load(baseline_path, stem)
    print(f"dev {DEV_SEASONS} rows: {len(dev)}   holdout {HOLDOUT_SEASON} rows: {len(hol)}")
    if len(dev) < 60 or len(hol) < 30:
        print("  TOO FEW ROWS to validate honestly -- skipping market.")
        return {"skipped": True, "reason": "insufficient rows", "n_dev": len(dev), "n_holdout": len(hol)}

    dates = [r[1] for r in dev]
    cut = pick_val_cut(dates)
    tr = [r for r in dev if r[1] < cut]
    va = [r for r in dev if r[1] >= cut]
    print(f"  train (game_date < {cut}): {len(tr)}   internal val (>= {cut}): {len(va)}")

    print("\ntraining challenger (binary:logistic) ...", flush=True)
    bst = xgb.train(PARAMS, mat(tr, xgb), num_boost_round=800, evals=[(mat(va, xgb), "val")],
                    early_stopping_rounds=40, verbose_eval=False)
    itr = (0, bst.best_iteration + 1)
    print(f"  best_iteration={bst.best_iteration}  scoring with iteration_range={itr}")

    probs_hol = bst.predict(mat(hol, xgb), iteration_range=itr)
    labels_hol = [r[-1] for r in hol]
    challenger = metrics(list(map(float, probs_hol)), labels_hol)

    train_rate = float(np.mean([r[-1] for r in tr]))
    constant = metrics([train_rate] * len(hol), labels_hol)

    print(f"\n============ {HOLDOUT_SEASON} HOLDOUT ============")
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
    verdict = f"{mkt_key.upper()}_{'PASSES_GATE' if passed else 'DOES_NOT_CLEAR_GATE'}"

    print("\n============ PRE-REGISTERED GATE ============")
    print(f"  AUC >= {GATE['min_auc']}:            {challenger['auc']:.4f}  -> {c1}")
    print(f"  ECE <= {GATE['max_ece']}:              {challenger['ece']:.4f}  -> {c2}")
    print(f"  logloss gain >= {GATE['min_logloss_gain']}:  {d_ll:+.5f}  -> {c3}")
    print(f"  Brier better than constant:  {challenger['brier']:.5f} < {constant['brier']:.5f}  -> {c4}")
    print(f"  VERDICT: {verdict}")

    bst.save_model(str(workdir / f"{stem}.json"))
    (workdir / f"{stem}_columns.json").write_text(json.dumps(FEATURES))
    return {"skipped": False, "holdout_season": HOLDOUT_SEASON,
            "constant": constant, "challenger": challenger, "gate": GATE,
            "passed": passed, "verdict": verdict, "importance": imp,
            "best_iteration": bst.best_iteration, "n_dev": len(dev), "n_holdout": len(hol)}


def main():
    workdir = WORKDIR
    workdir.mkdir(parents=True, exist_ok=True)
    print("NFL_PRESEASON_CHAMPION_GATE_A\n=============================")

    full_report = {"script": "NFL_PRESEASON_CHAMPION_GATE_A", "markets": {}}
    for mkt_key, cfg in MARKETS.items():
        result = run_market(mkt_key, cfg["stem"], workdir)
        full_report["markets"][mkt_key] = result

    (workdir / "nfl_preseason_champion_gate_a_report.json").write_text(json.dumps(full_report, indent=2))
    print(f"\nreport: {workdir / 'nfl_preseason_champion_gate_a_report.json'}")

    any_pass = any(r.get("passed") for r in full_report["markets"].values())
    return 0 if any_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
