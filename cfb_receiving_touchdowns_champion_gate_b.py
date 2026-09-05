#!/usr/bin/env python3
"""
CFB_RECEIVING_TOUCHDOWNS_CHAMPION_GATE_B

gate_a (all-division population) fell short: AUC 0.5732 (bar 0.58),
logloss gain +0.00852 (bar 0.01). This uses clean_baseline_b's Power4-
vs-Power4-only population instead -- the same fix already validated for
receiving_yards, tested here rather than assumed to transfer.

Same proven configuration as the other CFB TD markets (margin-context
features, regularized hyperparameters, power-adjusted bootstrap
calibration test).

DEV=2018-2022, VAL=2023, HOLDOUT=2024 -- 2025 excluded, same
touchdown_player_id completeness caveat as every other TD market here.

Pre-registered pass bar (all):
  1. AUC >= 0.58
  2. calibration bootstrap goodness-of-fit p >= 0.10
  3. logloss gain >= 0.01 vs constant
  4. Brier better than constant

Run
---
python -u cfb_receiving_touchdowns_champion_gate_a.py
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

BASELINE_DEFAULT = "/data/cfb_model/cfb_receiving_touchdowns_clean_baseline_b_work/baseline.sqlite"
WORKDIR_DEFAULT = "/data/cfb_model/cfb_receiving_touchdowns_champion_gate_b_work"

FEATURES = [
    "season_avg_rec_td", "recent3_avg_rec_td", "recent5_avg_rec_td",
    "season_avg_receptions", "recent3_avg_receptions", "td_per_reception",
    "opp_rec_td_allowed_per_game", "is_home", "games_played",
    "team_net_margin", "opp_net_margin", "projected_margin",
]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3,
          "eta": 0.03, "subsample": 0.7, "colsample_bytree": 0.7,
          "min_child_weight": 15, "reg_lambda": 3.0, "seed": 13}
NAN = float("nan")

DEV_SEASONS = (2018, 2019, 2020, 2021, 2022)
VAL_SEASON = 2023
HOLDOUT_SEASON = 2024

GATE = {"min_auc": 0.58, "min_logloss_gain": 0.01}
CALIB_MIN_P = 0.10
B_CALIB = 10_000
SEED = 20260829


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


def bootstrap_calib_p(probs, y, rng, b_sims=B_CALIB):
    def ece_only(p, yy):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        n = len(yy)
        total = 0.0
        for b in range(10):
            m = (p >= b / 10) & (p < (b + 1) / 10) if b < 9 else (p >= 0.9)
            cnt = int(m.sum())
            if cnt == 0:
                continue
            total += abs(float(p[m].mean()) - float(yy[m].mean())) * cnt / n
        return total
    observed = ece_only(probs, y)
    worse = 0
    for _ in range(b_sims):
        sim_y = rng.binomial(1, probs).astype(float)
        if ece_only(probs, sim_y) >= observed:
            worse += 1
    return observed, worse / b_sims


def load(baseline_path):
    con = sqlite3.connect(f"file:{baseline_path}?mode=ro", uri=True)
    cols = ["season", "week"] + FEATURES + ["over_line"]
    rows = con.execute(f"SELECT {', '.join(cols)} FROM cfb_receiving_touchdowns_baseline").fetchall()
    con.close()
    tr = [r for r in rows if r[0] in DEV_SEASONS]
    va = [r for r in rows if r[0] == VAL_SEASON]
    hol = [r for r in rows if r[0] == HOLDOUT_SEASON]
    return tr, va, hol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=BASELINE_DEFAULT)
    ap.add_argument("--workdir", default=WORKDIR_DEFAULT)
    args = ap.parse_args()
    import xgboost as xgb

    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    print("CFB_RECEIVING_TOUCHDOWNS_CHAMPION_GATE_B\n=========================================")
    tr, va, hol = load(args.baseline)
    print(f"train {DEV_SEASONS} rows: {len(tr)}   internal val {VAL_SEASON} rows: {len(va)}   "
          f"holdout {HOLDOUT_SEASON} rows: {len(hol)}")

    def mat(rows):
        X = np.array([[r[2 + i] if r[2 + i] is not None else NAN for i in range(len(FEATURES))] for r in rows], dtype=np.float32)
        y = np.array([r[-1] for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=FEATURES)

    print("\ntraining challenger (binary:logistic, regularized) ...", flush=True)
    bst = xgb.train(PARAMS, mat(tr), num_boost_round=800, evals=[(mat(va), "val")],
                    early_stopping_rounds=40, verbose_eval=False)
    itr = (0, bst.best_iteration + 1)
    print(f"  best_iteration={bst.best_iteration}  scoring with iteration_range={itr}")

    probs_hol = bst.predict(mat(hol), iteration_range=itr)
    labels_hol = np.array([r[-1] for r in hol])
    challenger = metrics(list(map(float, probs_hol)), labels_hol.tolist())

    train_rate = float(np.mean([r[-1] for r in tr]))
    constant = metrics([train_rate] * len(hol), labels_hol.tolist())

    rng = np.random.default_rng(SEED)
    observed_ece, calib_p = bootstrap_calib_p(probs_hol, labels_hol, rng)

    print(f"\n============ {HOLDOUT_SEASON} HOLDOUT ============")
    print(f"  {'arm':12s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'constant':12s} {'n/a':>7s} {constant['log_loss']:>9.5f}  {constant['brier']:>7.5f} {constant['ece']:>7.4f}")
    print(f"  {'challenger':12s} {challenger['auc']:>7.4f}  {challenger['log_loss']:>9.5f}  {challenger['brier']:>7.5f} {challenger['ece']:>7.4f}")
    print(f"  calibration bootstrap goodness-of-fit p = {calib_p:.4f} (bar >= {CALIB_MIN_P})")

    print("\nchallenger reliability (pred -> actual):")
    for b in challenger["reliability"]:
        print(f"   {b['bin']}  n={b['n']:>5}  pred={b['pred']:.3f}  actual={b['actual']:.3f}")

    imp = bst.get_score(importance_type="gain")
    print("\nfeature importance (gain):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"   {k:32s} {v:9.2f}")

    d_ll = constant["log_loss"] - challenger["log_loss"]
    c1 = challenger["auc"] >= GATE["min_auc"]
    c2 = calib_p >= CALIB_MIN_P
    c3 = d_ll >= GATE["min_logloss_gain"]
    c4 = challenger["brier"] < constant["brier"]
    passed = c1 and c2 and c3 and c4
    verdict = ("CFB_RECEIVING_TOUCHDOWNS_CHAMPION_PASSES_GATE_READY_FOR_STABILITY_CONFIRMATION"
               if passed else "CFB_RECEIVING_TOUCHDOWNS_CHAMPION_DOES_NOT_CLEAR_GATE")

    print("\n============ PRE-REGISTERED GATE (power-adjusted calibration) ============")
    print(f"  AUC >= {GATE['min_auc']}:            {challenger['auc']:.4f}  -> {c1}")
    print(f"  calib_p >= {CALIB_MIN_P}:              {calib_p:.4f}  -> {c2}")
    print(f"  logloss gain >= {GATE['min_logloss_gain']}:  {d_ll:+.5f}  -> {c3}")
    print(f"  Brier better than constant:  {challenger['brier']:.5f} < {constant['brier']:.5f}  -> {c4}")
    print(f"  VERDICT: {verdict}")

    bst.save_model(str(work / "cfb_receiving_touchdowns.json"))
    (work / "cfb_receiving_touchdowns_columns.json").write_text(json.dumps(FEATURES))
    report = {"script": "CFB_RECEIVING_TOUCHDOWNS_CHAMPION_GATE_B", "holdout": HOLDOUT_SEASON,
              "constant": constant, "challenger": challenger, "calib_p": calib_p,
              "gate": {**GATE, "calib_min_p": CALIB_MIN_P},
              "passed": passed, "verdict": verdict, "importance": imp,
              "best_iteration": bst.best_iteration}
    (work / "cfb_receiving_touchdowns_champion_gate_b_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nmodel + report written to {work}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
