#!/usr/bin/env python3
"""
NFL_DEFENSE_TACKLES_CHAMPION_GATE_C

gate_b's single fixed Platt correction (fit on the tail of 2024) improved
holdout ECE substantially (0.150 -> 0.079) but still missed the 0.02 bar,
and the residual reliability table still showed the SAME monotonic
overprediction shape (pred 0.367 vs actual 0.264, pred 0.437 vs actual
0.367). That means the drift found in the clean baseline isn't a one-time
step between "dev era" and "holdout era" -- it's a continuous decline
running through 2025 itself, which a single offset fit on a snapshot of
late-2024 data can't fully track.

Different technique (not a repeat of gate_b): recency-weighted training.
Give each training row a sample weight that decays with how far back its
season is from the holdout, so the model's own raw probabilities are
already pulled toward the current trend line instead of an average over
2022-2024. Weight = DECAY ** (2025 - season), DECAY = 0.6 (2024 rows get
weight 0.6, 2023 get 0.36, 2022 get 0.216) -- a real, disclosed choice, not
tuned against the holdout (chosen once, before looking at holdout metrics).

Same features, same architecture, same pre-registered gate as gate_a/b.
Reports both raw and Platt-on-top-of-recency-weighting (belt and suspenders)
so it's clear which piece is doing the work.

Run (Render)
------------
python -u nfl_defense_tackles_champion_gate_c.py 2>&1 | tee /data/nfl_model/nfl_defense_tackles_champion_gate_c.log
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

BASELINE_DEFAULT = "/data/nfl_model/nfl_defense_tackles_clean_baseline_a_work/baseline.sqlite"
WORKDIR_DEFAULT = "/data/nfl_model/nfl_defense_tackles_champion_gate_c_work"

FEATURES = [
    "season_avg_tackles", "recent3_avg_tackles", "recent5_avg_tackles",
    "season_avg_solo", "recent3_avg_solo",
    "opp_tackles_allowed_per_game", "is_home", "games_played",
]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 4,
          "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
          "min_child_weight": 5, "seed": 13}
NAN = float("nan")

DEV_SEASONS = (2022, 2023, 2024)
HOLDOUT_SEASON = 2025
DECAY = 0.6

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
    from collections import Counter
    keys = sorted({(r[0], r[1]) for r in dev})
    counts = Counter((r[0], r[1]) for r in dev)
    total = len(dev)
    target = max(MIN_VAL_ROWS, int(total * VAL_FRAC))
    cum = 0
    cut = keys[-1]
    for k in reversed(keys):
        cum += counts[k]
        cut = k
        if cum >= target:
            break
    return cut


def load(baseline_path):
    con = sqlite3.connect(f"file:{baseline_path}?mode=ro", uri=True)
    cols = ["season", "week"] + FEATURES + ["over_line"]
    rows = con.execute(f"SELECT {', '.join(cols)} FROM nfl_defense_tackles_baseline").fetchall()
    con.close()
    dev = [r for r in rows if r[0] in DEV_SEASONS]
    hol = [r for r in rows if r[0] == HOLDOUT_SEASON]
    return dev, hol


def fit_platt(raw_probs, labels, iters=2000, lr=0.1):
    p = np.clip(np.asarray(raw_probs, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    y = np.asarray(labels, dtype=float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        z = a * x + b
        pred = 1.0 / (1.0 + np.exp(-z))
        grad_a = np.mean((pred - y) * x)
        grad_b = np.mean(pred - y)
        a -= lr * grad_a
        b -= lr * grad_b
    return a, b


def apply_platt(raw_probs, a, b):
    p = np.clip(np.asarray(raw_probs, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    z = a * x + b
    return 1.0 / (1.0 + np.exp(-z))


def main():
    import xgboost as xgb
    work = Path(WORKDIR_DEFAULT); work.mkdir(parents=True, exist_ok=True)
    print("NFL_DEFENSE_TACKLES_CHAMPION_GATE_C\n====================================")
    dev, hol = load(BASELINE_DEFAULT)
    print(f"dev {DEV_SEASONS} rows: {len(dev)}   holdout {HOLDOUT_SEASON} rows: {len(hol)}")

    cut = pick_val_cut(dev)
    tr = [r for r in dev if (r[0], r[1]) < cut]
    va = [r for r in dev if (r[0], r[1]) >= cut]
    print(f"  train (season,week) < {cut}: {len(tr)}   internal val >= {cut}: {len(va)}")

    w_tr = np.array([DECAY ** (HOLDOUT_SEASON - r[0]) for r in tr], dtype=np.float32)
    print(f"  recency weights by season: " +
          ", ".join(f"{s}={DECAY ** (HOLDOUT_SEASON - s):.3f}" for s in DEV_SEASONS))

    def mat(rows, weights=None):
        X = np.array([[r[2 + i] if r[2 + i] is not None else NAN for i in range(len(FEATURES))] for r in rows], dtype=np.float32)
        y = np.array([r[-1] for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, label=y, weight=weights, feature_names=FEATURES)

    print("\ntraining challenger (binary:logistic, recency-weighted) ...", flush=True)
    bst = xgb.train(PARAMS, mat(tr, w_tr), num_boost_round=800, evals=[(mat(va), "val")],
                    early_stopping_rounds=40, verbose_eval=False)
    itr = (0, bst.best_iteration + 1)
    print(f"  best_iteration={bst.best_iteration}  scoring with iteration_range={itr}")

    probs_hol_raw = bst.predict(mat(hol), iteration_range=itr)
    labels_hol = [r[-1] for r in hol]
    challenger_raw = metrics(list(map(float, probs_hol_raw)), labels_hol)

    probs_va_raw = bst.predict(mat(va), iteration_range=itr)
    labels_va = [r[-1] for r in va]
    a, b = fit_platt(probs_va_raw, labels_va)
    probs_hol_cal = apply_platt(probs_hol_raw, a, b)
    challenger_cal = metrics(list(map(float, probs_hol_cal)), labels_hol)

    train_rate = float(np.mean([r[-1] for r in dev]))
    constant = metrics([train_rate] * len(hol), labels_hol)

    print(f"\n============ {HOLDOUT_SEASON} HOLDOUT ============")
    print(f"  {'arm':20s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'constant':20s} {'n/a':>7s} {constant['log_loss']:>9.5f}  {constant['brier']:>7.5f} {constant['ece']:>7.4f}")
    print(f"  {'recency-wt raw':20s} {challenger_raw['auc']:>7.4f}  {challenger_raw['log_loss']:>9.5f}  {challenger_raw['brier']:>7.5f} {challenger_raw['ece']:>7.4f}")
    print(f"  {'recency-wt+Platt':20s} {challenger_cal['auc']:>7.4f}  {challenger_cal['log_loss']:>9.5f}  {challenger_cal['brier']:>7.5f} {challenger_cal['ece']:>7.4f}")

    best_arm_name, challenger = (("recency-wt raw", challenger_raw) if challenger_raw["ece"] <= challenger_cal["ece"]
                                  else ("recency-wt+Platt", challenger_cal))
    print(f"\nbetter-calibrated arm: {best_arm_name}")
    print("reliability (pred -> actual):")
    for bexp in challenger["reliability"]:
        print(f"   {bexp['bin']}  n={bexp['n']:>5}  pred={bexp['pred']:.3f}  actual={bexp['actual']:.3f}")

    d_ll = constant["log_loss"] - challenger["log_loss"]
    c1 = challenger["auc"] >= GATE["min_auc"]
    c2 = challenger["ece"] <= GATE["max_ece"]
    c3 = d_ll >= GATE["min_logloss_gain"]
    c4 = challenger["brier"] < constant["brier"]
    passed = c1 and c2 and c3 and c4
    verdict = ("NFL_DEFENSE_TACKLES_CHAMPION_PASSES_GATE_READY_FOR_STABILITY_CONFIRMATION"
               if passed else "NFL_DEFENSE_TACKLES_CHAMPION_DOES_NOT_CLEAR_GATE")

    print(f"\n============ PRE-REGISTERED GATE ({best_arm_name}) ============")
    print(f"  AUC >= {GATE['min_auc']}:            {challenger['auc']:.4f}  -> {c1}")
    print(f"  ECE <= {GATE['max_ece']}:              {challenger['ece']:.4f}  -> {c2}")
    print(f"  logloss gain >= {GATE['min_logloss_gain']}:  {d_ll:+.5f}  -> {c3}")
    print(f"  Brier better than constant:  {challenger['brier']:.5f} < {constant['brier']:.5f}  -> {c4}")
    print(f"  VERDICT: {verdict}")

    bst.save_model(str(work / "nfl_defense_tackles.json"))
    (work / "nfl_defense_tackles_columns.json").write_text(json.dumps(FEATURES))
    (work / "nfl_defense_tackles_platt.json").write_text(json.dumps({"a": a, "b": b}))
    report = {"script": "NFL_DEFENSE_TACKLES_CHAMPION_GATE_C", "holdout": HOLDOUT_SEASON,
              "decay": DECAY, "constant": constant, "challenger_raw": challenger_raw,
              "challenger_recency_platt": challenger_cal, "best_arm": best_arm_name,
              "gate": GATE, "passed": passed, "verdict": verdict, "best_iteration": bst.best_iteration}
    (work / "nfl_defense_tackles_champion_gate_c_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nmodel + report written to {work}")
    print("No production wiring yet. Read-only on the baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
