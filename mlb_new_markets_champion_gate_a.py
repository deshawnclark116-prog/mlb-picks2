#!/usr/bin/env python3
"""
MLB_NEW_MARKETS_CHAMPION_GATE_A

Real validation for the 7 new candidate markets baselined by
mlb_new_markets_clean_baseline_a.py: constant vs. trained-XGBoost on the
untouched 2025 holdout, same discipline as every other champion gate in
this repo (nfl_rushing_yards_champion_gate_a.py, total_bases's, etc.).

Pre-registered pass (written before this script has ever been run):
  1. AUC >= 0.58
  2. ECE <= 0.03
  3. log-loss beats the constant arm by >= 0.01
  4. Brier score beats the constant arm

Internal train/val split within 2024 dev is by chronological date
(row-count-based cut, mirroring the NFL scripts' pick_val_cut -- density
isn't uniform across a season).

Read-only on the clean baseline. Writes only a report + trained models.

Run:
    python -u mlb_new_markets_champion_gate_a.py
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

BASELINE_WORKDIR = Path("/data/mlb_new_markets_clean_baseline_a_work")
WORKDIR = Path("/data/mlb_new_markets_champion_gate_a_work")

BATTER_MARKETS = ["batter_walks", "batter_doubles", "batter_singles", "batter_strikeouts"]
PITCHER_MARKETS = ["pitcher_walks", "pitcher_outs", "pitcher_hits_allowed"]
FEATURE_COLS_BATTER = ["season_rate", "recent5_avg", "recent15_avg",
                        "season_avg_pa", "batting_order", "games_played"]
FEATURE_COLS_PITCHER = ["season_rate", "recent5_avg", "recent15_avg",
                         "season_avg_bf", "games_played"]

PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 4,
          "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
          "min_child_weight": 5, "seed": 13}
NAN = float("nan")
DEV_SEASON = 2024
HOLDOUT_SEASON = 2025
GATE = {"min_auc": 0.58, "max_ece": 0.03, "min_logloss_gain": 0.01}
MIN_VAL_ROWS = 200
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


def pick_val_cut(dates_sorted):
    total = len(dates_sorted)
    target = max(MIN_VAL_ROWS, int(total * VAL_FRAC))
    cut_idx = max(0, total - target)
    return dates_sorted[cut_idx]


def load(mkt_key):
    base_db = BASELINE_WORKDIR / f"{mkt_key}_baseline.sqlite"
    con = sqlite3.connect(f"file:{base_db}?mode=ro", uri=True)
    manifest = json.loads((BASELINE_WORKDIR / "manifest.json").read_text())
    feat_cols = manifest["markets"][mkt_key]["feature_cols"]
    cols = ["season", "date"] + feat_cols + ["over_line"]
    rows = con.execute(f"SELECT {', '.join(cols)} FROM {mkt_key}_baseline ORDER BY date").fetchall()
    con.close()
    dev = [r for r in rows if r[0] == DEV_SEASON]
    hol = [r for r in rows if r[0] == HOLDOUT_SEASON]
    return dev, hol, feat_cols, manifest["markets"][mkt_key]["line"]


def mat(rows, feat_cols, xgb):
    n_feat = len(feat_cols)
    X = np.array([[r[2 + i] if r[2 + i] is not None else NAN for i in range(n_feat)] for r in rows], dtype=np.float32)
    y = np.array([r[-1] for r in rows], dtype=np.float32)
    return xgb.DMatrix(X, label=y, feature_names=feat_cols)


def run_market(mkt_key, workdir):
    import xgboost as xgb
    print(f"\n{'='*70}\n{mkt_key}\n{'='*70}")
    dev, hol, feat_cols, line = load(mkt_key)
    print(f"line={line}  dev {DEV_SEASON} rows: {len(dev)}   holdout {HOLDOUT_SEASON} rows: {len(hol)}")

    dates = [r[1] for r in dev]
    cut = pick_val_cut(dates)
    tr = [r for r in dev if r[1] < cut]
    va = [r for r in dev if r[1] >= cut]
    print(f"  train (date < {cut}): {len(tr)}   internal val (>= {cut}): {len(va)}")

    print("training challenger (binary:logistic) ...", flush=True)
    bst = xgb.train(PARAMS, mat(tr, feat_cols, xgb), num_boost_round=800,
                     evals=[(mat(va, feat_cols, xgb), "val")],
                     early_stopping_rounds=40, verbose_eval=False)
    itr = (0, bst.best_iteration + 1)
    print(f"  best_iteration={bst.best_iteration}  scoring with iteration_range={itr}")

    probs_hol = bst.predict(mat(hol, feat_cols, xgb), iteration_range=itr)
    labels_hol = [r[-1] for r in hol]
    challenger = metrics(list(map(float, probs_hol)), labels_hol)

    train_rate = float(np.mean([r[-1] for r in tr]))
    constant = metrics([train_rate] * len(hol), labels_hol)

    print(f"\n============ {HOLDOUT_SEASON} HOLDOUT ============")
    print(f"  {'arm':12s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'constant':12s} {'n/a':>7s} {constant['log_loss']:>9.5f}  {constant['brier']:>7.5f} {constant['ece']:>7.4f}")
    print(f"  {'challenger':12s} {challenger['auc']:>7.4f}  {challenger['log_loss']:>9.5f}  {challenger['brier']:>7.5f} {challenger['ece']:>7.4f}")

    imp = bst.get_score(importance_type="gain")
    print("\nfeature importance (gain):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"   {k:24s} {v:9.2f}")

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

    workdir.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(workdir / f"{mkt_key}.json"))
    (workdir / f"{mkt_key}_columns.json").write_text(json.dumps(feat_cols))
    return {"line": line, "constant": constant, "challenger": challenger, "gate": GATE,
            "passed": passed, "verdict": verdict, "importance": imp,
            "best_iteration": bst.best_iteration, "n_dev": len(dev), "n_holdout": len(hol)}


def main():
    workdir = WORKDIR
    print("MLB_NEW_MARKETS_CHAMPION_GATE_A\n================================")

    report = {"script": "MLB_NEW_MARKETS_CHAMPION_GATE_A", "markets": {}}
    for mkt_key in BATTER_MARKETS + PITCHER_MARKETS:
        report["markets"][mkt_key] = run_market(mkt_key, workdir)

    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "mlb_new_markets_champion_gate_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n\n{'#'*70}\nSUMMARY\n{'#'*70}")
    for mkt, r in report["markets"].items():
        print(f"  {mkt:24s} AUC={r['challenger']['auc']:.4f}  {r['verdict']}")
    print(f"\nreport: {workdir / 'mlb_new_markets_champion_gate_a_report.json'}")

    any_pass = any(r["passed"] for r in report["markets"].values())
    return 0 if any_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
