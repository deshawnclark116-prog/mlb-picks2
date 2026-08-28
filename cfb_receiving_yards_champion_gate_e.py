#!/usr/bin/env python3
"""
CFB_RECEIVING_YARDS_CHAMPION_GATE_E

PRE-REGISTERED BEFORE RUNNING (per explicit user decision after gate_d's
inconclusive result: "redesign the holdout test" rather than add more
training data, since gate_d showed more dev-season data alone doesn't
resolve a holdout-SIZE problem -- AUC point estimate was flat at 0.5767
vs gate_c's 0.5789, bootstrap CI still straddled the bar either way).

WHAT CHANGES vs gate_d: only the season split. Same underlying eligible-row
pool (loaded from clean_baseline_d's baseline.sqlite -- same features,
same Power4-vs-Power4 scoping, same eligibility floor). Same model class,
hyperparameters, and gate bars -- nothing about the pass/fail criteria is
touched.

    DEV_SEASONS (train)      = 2018, 2019, 2020, 2021, 2022   (5 seasons)
    VAL_SEASON (early stop)  = 2023                            (1 season)
    HOLDOUT_SEASONS (gate)   = 2024, 2025  POOLED               (~561 rows,
                                roughly 2x gate_d's 267-row holdout)

Rationale: gate_c/gate_d's problem was never "not enough training rows" --
1321 dev rows already trained a stable challenger with near-identical
point estimates each time. The problem was a holdout too small (n=267) to
separate "true AUC ~0.58" from "true AUC ~0.50" with any power (bootstrap
CI was [0.51, 0.64] both times). Pooling two holdout seasons instead of
one is the direct fix for that -- not a loosened bar, a bigger sample to
test the SAME bar against.

Pre-registered pass bar (identical to gate_c/gate_d, NOT loosened):
  1. AUC >= 0.58
  2. calibration bootstrap goodness-of-fit p >= 0.10
  3. logloss gain >= 0.01 vs constant
  4. Brier better than constant

If this still lands inconclusive, the honest read is that CFB
receiving_yards does not have enough real signal in this feature set to
clear the bar the other markets cleared, and it stays unshipped -- this
script does not get a follow-up "gate_f" that pools yet more seasons
into the same test after seeing this result.

Run
---
python -u cfb_receiving_yards_champion_gate_e.py
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

BASELINE_DEFAULT = "/data/cfb_model/cfb_receiving_yards_clean_baseline_d_work/baseline.sqlite"
WORKDIR_DEFAULT = "/data/cfb_model/cfb_receiving_yards_champion_gate_e_work"

FEATURES = [
    "season_avg_rec_yards", "recent3_avg_rec_yards", "recent5_avg_rec_yards",
    "season_avg_receptions", "recent3_avg_receptions", "yards_per_reception",
    "opp_rec_yards_allowed_per_game", "is_home", "games_played",
    "team_net_margin", "opp_net_margin", "projected_margin",
]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3,
          "eta": 0.03, "subsample": 0.7, "colsample_bytree": 0.7,
          "min_child_weight": 15, "reg_lambda": 3.0, "seed": 13}
NAN = float("nan")

DEV_SEASONS = (2018, 2019, 2020, 2021, 2022)
VAL_SEASON = 2023
HOLDOUT_SEASONS = (2024, 2025)

GATE = {"min_auc": 0.58, "min_logloss_gain": 0.01}
CALIB_MIN_P = 0.10
B_CALIB = 10_000
B_BOOTSTRAP = 5000
SEED = 20260827


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
    rows = con.execute(f"SELECT {', '.join(cols)} FROM cfb_receiving_yards_baseline").fetchall()
    con.close()
    tr = [r for r in rows if r[0] in DEV_SEASONS]
    va = [r for r in rows if r[0] == VAL_SEASON]
    hol = [r for r in rows if r[0] in HOLDOUT_SEASONS]
    return tr, va, hol


def main():
    import xgboost as xgb
    work = Path(WORKDIR_DEFAULT); work.mkdir(parents=True, exist_ok=True)
    print("CFB_RECEIVING_YARDS_CHAMPION_GATE_E\n====================================")
    tr, va, hol = load(BASELINE_DEFAULT)
    print(f"train {DEV_SEASONS} rows: {len(tr)}   internal val {VAL_SEASON} rows: {len(va)}   "
          f"holdout {HOLDOUT_SEASONS} (pooled) rows: {len(hol)}")

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

    print(f"\n============ {HOLDOUT_SEASONS} HOLDOUT (POOLED) ============")
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

    # Bootstrap AUC CI on the pooled holdout -- reported alongside the gate
    # verdict (not a substitute for it), same diagnostic used on gate_c/d.
    boot_rng = np.random.default_rng(SEED + 1)
    n_hol = len(labels_hol)
    aucs = np.empty(B_BOOTSTRAP)
    for b in range(B_BOOTSTRAP):
        idx = boot_rng.integers(0, n_hol, n_hol)
        aucs[b] = auc(probs_hol[idx], labels_hol[idx])
    ci_lo, ci_hi = (float(x) for x in np.percentile(aucs, [2.5, 97.5]))
    p_below_50 = float((aucs <= 0.50).mean())
    print(f"\nbootstrap AUC 95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]  (B={B_BOOTSTRAP})")
    print(f"P(true AUC <= 0.50) = {p_below_50:.4f}")

    d_ll = constant["log_loss"] - challenger["log_loss"]
    c1 = challenger["auc"] >= GATE["min_auc"]
    c2 = calib_p >= CALIB_MIN_P
    c3 = d_ll >= GATE["min_logloss_gain"]
    c4 = challenger["brier"] < constant["brier"]
    passed = c1 and c2 and c3 and c4
    verdict = ("CFB_RECEIVING_YARDS_CHAMPION_PASSES_GATE_READY_FOR_STABILITY_CONFIRMATION"
               if passed else "CFB_RECEIVING_YARDS_CHAMPION_DOES_NOT_CLEAR_GATE")

    print("\n============ PRE-REGISTERED GATE (power-adjusted calibration) ============")
    print(f"  AUC >= {GATE['min_auc']}:            {challenger['auc']:.4f}  -> {c1}")
    print(f"  calib_p >= {CALIB_MIN_P}:              {calib_p:.4f}  -> {c2}")
    print(f"  logloss gain >= {GATE['min_logloss_gain']}:  {d_ll:+.5f}  -> {c3}")
    print(f"  Brier better than constant:  {challenger['brier']:.5f} < {constant['brier']:.5f}  -> {c4}")
    print(f"  VERDICT: {verdict}")

    bst.save_model(str(work / "cfb_receiving_yards.json"))
    (work / "cfb_receiving_yards_columns.json").write_text(json.dumps(FEATURES))
    report = {"script": "CFB_RECEIVING_YARDS_CHAMPION_GATE_E",
              "dev_seasons": list(DEV_SEASONS), "val_season": VAL_SEASON,
              "holdout_seasons": list(HOLDOUT_SEASONS),
              "constant": constant, "challenger": challenger, "calib_p": calib_p,
              "bootstrap_auc_ci": {"lo": ci_lo, "hi": ci_hi, "p_true_auc_le_050": p_below_50},
              "gate": {**GATE, "calib_min_p": CALIB_MIN_P},
              "passed": passed, "verdict": verdict, "importance": imp,
              "best_iteration": bst.best_iteration}
    (work / "cfb_receiving_yards_champion_gate_e_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nmodel + report written to {work}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
