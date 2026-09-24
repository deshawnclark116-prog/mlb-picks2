#!/usr/bin/env python3
"""
NHL_GOALIE_SAVES_CHAMPION_GATE_A

Real goalie-saves OVER/UNDER classifier, same champion-gate process as
nhl_points_champion_gate_a.py (see that file for the shared auc/metrics/
bootstrap_calib_p import reasoning).

DEV=2018-2022, VAL=2023, HOLDOUT=2024. Pre-registered pass bar (all):
AUC >= 0.58, calib_p >= 0.10, logloss gain >= 0.01, Brier better than
constant.

Run
---
python -u nhl_goalie_saves_champion_gate_a.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nhl_moneyline_champion_gate_a import auc, metrics, bootstrap_calib_p

BASELINE_DEFAULT = "nhl_models/nhl_goalie_saves_clean_baseline_a_work/baseline.sqlite"
WORKDIR_DEFAULT = "nhl_models/nhl_goalie_saves_champion_gate_a_work"

FEATURES = [
    "season_avg_saves", "recent3_avg_saves", "recent5_avg_saves",
    "season_avg_shots_against", "recent3_avg_shots_against", "save_pct",
    "opp_shots_for_per_game", "is_home", "games_played",
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
SEED = 20260923


def load(baseline_path):
    con = sqlite3.connect(f"file:{baseline_path}?mode=ro", uri=True)
    cols = ["season", "week"] + FEATURES + ["over_line"]
    rows = con.execute(f"SELECT {', '.join(cols)} FROM nhl_goalie_saves_baseline").fetchall()
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
    print("NHL_GOALIE_SAVES_CHAMPION_GATE_A\n=================================")
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
    verdict = ("NHL_GOALIE_SAVES_CHAMPION_PASSES_GATE_READY_FOR_STABILITY_CONFIRMATION"
               if passed else "NHL_GOALIE_SAVES_CHAMPION_DOES_NOT_CLEAR_GATE")

    print("\n============ PRE-REGISTERED GATE (power-adjusted calibration) ============")
    print(f"  AUC >= {GATE['min_auc']}:            {challenger['auc']:.4f}  -> {c1}")
    print(f"  calib_p >= {CALIB_MIN_P}:              {calib_p:.4f}  -> {c2}")
    print(f"  logloss gain >= {GATE['min_logloss_gain']}:  {d_ll:+.5f}  -> {c3}")
    print(f"  Brier better than constant:  {challenger['brier']:.5f} < {constant['brier']:.5f}  -> {c4}")
    print(f"  VERDICT: {verdict}")

    bst.save_model(str(work / "nhl_goalie_saves.json"))
    (work / "nhl_goalie_saves_columns.json").write_text(json.dumps(FEATURES))
    report = {"script": "NHL_GOALIE_SAVES_CHAMPION_GATE_A", "holdout": HOLDOUT_SEASON,
              "constant": constant, "challenger": challenger, "calib_p": calib_p,
              "gate": {**GATE, "calib_min_p": CALIB_MIN_P},
              "passed": passed, "verdict": verdict, "importance": imp,
              "best_iteration": bst.best_iteration}
    (work / "nhl_goalie_saves_champion_gate_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nmodel + report written to {work}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
