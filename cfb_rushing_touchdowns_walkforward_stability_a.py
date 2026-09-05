#!/usr/bin/env python3
"""
CFB_RUSHING_TOUCHDOWNS_WALKFORWARD_STABILITY_A

Final rung before live wiring, mirroring cfb_passing_touchdowns_
walkforward_stability_a.py exactly (same two-rung structure, same bars).
Uses the exact model/features/hyperparameters cfb_rushing_touchdowns_
champion_gate_a already validated -- this script does not retrain a
different model, it re-examines the already-passing predictions for
consistency.

DEV=2018-2022, VAL=2023, HOLDOUT=2024 -- not 2025, per the documented
data-quality finding in cfb_rushing_touchdowns_clean_baseline_a.py
(cfbfastR-data's touchdown_player_id field is measurably incomplete for
2025 -- confirmed directly for rushing TDs too, not just passing).

RUNG 1 -- walk-forward calibration (deployment-shaped test): as each
holdout-season week finishes, refit a 2-parameter Platt map on the
internal-val slice + holdout weeks seen so far (strictly earlier only),
predict the NEXT week with it -- exactly how a live 'growing'
calibration_policy would behave. Note: the champion gate's RAW
(uncalibrated) probabilities already passed the bootstrap calibration
test cleanly (p=0.18) -- this rung checks whether adding weekly Platt on
top holds up too, not because raw failed.

RUNG 2 -- stability confirmation (same bars as every other market this
repo has run through this rung): week-block bootstrap AUC 95% CI lower >=
0.55; season quarters AUC >= 0.50 in >= 3/4 quarters, none < 0.45; home/
away slices (min 100 rows) AUC >= 0.52 and calibration p >= 0.10.

Read-only. Writes only a report.

Run
---
python -u cfb_rushing_touchdowns_walkforward_stability_a.py
"""
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import cfb_rushing_touchdowns_champion_gate_a as g
from cfb_rushing_yards_champion_gate_b import fit_platt, apply_platt

B_CALIB = 10_000
B_BOOTSTRAP = 2000
SEED = 20260828
CALIB_MIN_P = 0.10

CI_LO_BAR = 0.55
QUARTER_AUC_BAR = 0.50
QUARTER_MIN_PASS = 3
QUARTER_FLOOR = 0.45
MIN_SLICE_ROWS = 100
SLICE_AUC_BAR = 0.52
SLICE_CALIB_MIN_P = 0.10

WORKDIR = Path("/data/cfb_model/cfb_rushing_touchdowns_walkforward_stability_a_work")


def ece_only(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    n = len(y)
    total = 0.0
    for b in range(10):
        m = (p >= b / 10) & (p < (b + 1) / 10) if b < 9 else (p >= 0.9)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        total += abs(float(p[m].mean()) - float(y[m].mean())) * cnt / n
    return total


def bootstrap_calib_p(probs, y, rng, b_sims=B_CALIB):
    observed = ece_only(probs, y)
    probs = np.asarray(probs, dtype=float)
    worse = 0
    for _ in range(b_sims):
        sim_y = rng.binomial(1, probs).astype(float)
        if ece_only(probs, sim_y) >= observed:
            worse += 1
    return observed, worse / b_sims


def main():
    import xgboost as xgb
    WORKDIR.mkdir(parents=True, exist_ok=True)
    print("CFB_RUSHING_TOUCHDOWNS_WALKFORWARD_STABILITY_A\n" + "=" * 42)

    tr, va, hol = g.load(g.BASELINE_DEFAULT)

    def mat(rows):
        X = np.array([[r[2 + i] if r[2 + i] is not None else g.NAN for i in range(len(g.FEATURES))] for r in rows], dtype=np.float32)
        y = np.array([r[-1] for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=g.FEATURES)

    bst = xgb.train(g.PARAMS, mat(tr), num_boost_round=800, evals=[(mat(va), "val")],
                    early_stopping_rounds=40, verbose_eval=False)
    itr = (0, bst.best_iteration + 1)
    print(f"best_iteration={bst.best_iteration}  scoring with iteration_range={itr}")

    va_raw = np.asarray(bst.predict(mat(va), iteration_range=itr), dtype=float)
    va_y = np.asarray([r[-1] for r in va], dtype=float)

    hol_raw = np.asarray(bst.predict(mat(hol), iteration_range=itr), dtype=float)
    hol_y = np.asarray([r[-1] for r in hol], dtype=float)
    hol_weeks = np.asarray([r[1] for r in hol])
    weeks = sorted(set(hol_weeks.tolist()))
    print(f"{g.HOLDOUT_SEASON} holdout: n={len(hol_y)} across weeks {weeks[0]}..{weeks[-1]}")
    print(f"warmup calibration pool: {g.VAL_SEASON} internal val, n={len(va)}")

    # ---- RUNG 1: weekly walk-forward Platt ----
    wf_pred = np.empty(len(hol_y))
    for w in weeks:
        seen = hol_weeks < w
        pool_x = np.concatenate([va_raw, hol_raw[seen]])
        pool_y = np.concatenate([va_y, hol_y[seen]])
        a, b = fit_platt(pool_x, pool_y)
        if a <= 0:
            a, b = 1.0, 0.0
        mask = hol_weeks == w
        wf_pred[mask] = apply_platt(hol_raw[mask], a, b)

    wf_metrics = g.metrics(list(map(float, wf_pred)), list(hol_y))
    train_rate = float(np.mean([r[-1] for r in tr]))
    constant = g.metrics([train_rate] * len(hol_y), list(hol_y))
    d_ll = constant["log_loss"] - wf_metrics["log_loss"]

    rng = np.random.default_rng(SEED)
    observed_ece, calib_p = bootstrap_calib_p(wf_pred, hol_y, rng)

    print(f"\n============ RUNG 1: WALK-FORWARD CALIBRATION ({g.HOLDOUT_SEASON}) ============")
    print(f"  {'arm':12s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'constant':12s} {'n/a':>7s} {constant['log_loss']:>9.5f}  {constant['brier']:>7.5f} {constant['ece']:>7.4f}")
    print(f"  {'walk-fwd':12s} {wf_metrics['auc']:>7.4f}  {wf_metrics['log_loss']:>9.5f}  {wf_metrics['brier']:>7.5f} {wf_metrics['ece']:>7.4f}")
    print(f"  calibration bootstrap goodness-of-fit p = {calib_p:.4f} (bar >= {CALIB_MIN_P})")

    r1a = wf_metrics["auc"] >= 0.58
    r1b = d_ll >= 0.01
    r1c = wf_metrics["brier"] < constant["brier"]
    r1d = calib_p >= CALIB_MIN_P
    rung1_pass = r1a and r1b and r1c and r1d
    print(f"  AUC>=0.58: {r1a}  logloss_gain>=0.01: {r1b} ({d_ll:+.5f})  "
          f"Brier<constant: {r1c}  calib_p>={CALIB_MIN_P}: {r1d}")
    print(f"  RUNG 1: {'PASS' if rung1_pass else 'FAIL'}")

    # ---- RUNG 2: stability confirmation ----
    print(f"\n============ RUNG 2: STABILITY CONFIRMATION ============")
    probs, y, week = wf_pred, hol_y, hol_weeks

    uniq = np.array(weeks)
    idx_by = {w: np.where(week == w)[0] for w in uniq}
    aucs = np.empty(B_BOOTSTRAP)
    for b in range(B_BOOTSTRAP):
        idx = np.concatenate([idx_by[w] for w in rng.choice(uniq, len(uniq), replace=True)])
        aucs[b] = g.auc(probs[idx], y[idx])
    ci_lo, ci_hi = (float(x) for x in np.percentile(aucs, [2.5, 97.5]))
    print(f"WEEK-BLOCK BOOTSTRAP (B={B_BOOTSTRAP}): AUC mean={aucs.mean():.4f}  "
          f"95%CI=[{ci_lo:.4f},{ci_hi:.4f}]")

    order = np.argsort(week, kind="mergesort")
    boundaries = [int(len(y) * f) for f in (0.25, 0.5, 0.75)]
    quarter_of_rank = np.searchsorted(boundaries, np.arange(len(y)), side="right")
    quarter = np.empty(len(y), dtype=int)
    quarter[order] = quarter_of_rank
    q_pass = 0
    q_floor_ok = True
    print("SEASON QUARTERS:")
    for qi in range(4):
        idx = np.where(quarter == qi)[0]
        wks = sorted(set(week[idx].tolist()))
        a = g.auc(probs[idx], y[idx])
        ok = a >= QUARTER_AUC_BAR
        q_pass += int(ok)
        if a < QUARTER_FLOOR:
            q_floor_ok = False
        print(f"  Q{qi+1} (weeks {wks[0]:>2}-{wks[-1]:>2}): n={len(idx):>4}  AUC={a:.4f}  "
              f"{'ok' if ok else 'LOW'}")

    print(f"SLICES (home/away, min {MIN_SLICE_ROWS} rows):")
    ih = 2 + g.FEATURES.index("is_home")
    home_mask = np.asarray([r[ih] == 1 for r in hol])
    slices_ok = slices_elig = 0
    for name, mask in (("home", home_mask), ("away", ~home_mask)):
        idx = np.where(mask)[0]
        if len(idx) < MIN_SLICE_ROWS:
            print(f"  {name:5s} n={len(idx)} (skip)"); continue
        a = g.auc(probs[idx], y[idx])
        s_ece, s_p = bootstrap_calib_p(probs[idx], y[idx], rng)
        ok = a >= SLICE_AUC_BAR and s_p >= SLICE_CALIB_MIN_P
        slices_elig += 1
        slices_ok += int(ok)
        print(f"  {name:5s} n={len(idx):>4}  AUC={a:.4f}  ECE={s_ece:.4f}  "
              f"calib_p={s_p:.4f}  {'ok' if ok else 'FAIL'}")

    r2a = bool(ci_lo >= CI_LO_BAR)
    r2b = bool(q_pass >= QUARTER_MIN_PASS and q_floor_ok)
    r2c = bool(slices_elig > 0 and slices_ok == slices_elig)
    rung2_pass = r2a and r2b and r2c
    print(f"  bootstrap CI lower>={CI_LO_BAR}: {r2a}  quarters {q_pass}/4 (floor_ok={q_floor_ok}): {r2b}  "
          f"slices {slices_ok}/{slices_elig}: {r2c}")
    print(f"  RUNG 2: {'PASS' if rung2_pass else 'FAIL'}")

    stable = rung1_pass and rung2_pass
    verdict = ("CFB_RUSHING_TOUCHDOWNS_WALKFORWARD_STABLE_READY_FOR_LIVE_WIRING" if stable
               else "CFB_RUSHING_TOUCHDOWNS_WALKFORWARD_NOT_YET_STABLE")
    print(f"\n================ FINAL VERDICT ================\n  {verdict}")

    bst.save_model(str(WORKDIR / "cfb_rushing_touchdowns.json"))
    (WORKDIR / "cfb_rushing_touchdowns_columns.json").write_text(json.dumps(g.FEATURES))
    report = {
        "script": "CFB_RUSHING_TOUCHDOWNS_WALKFORWARD_STABILITY_A",
        "rung1_walkforward": {"auc": wf_metrics["auc"], "logloss": wf_metrics["log_loss"],
                                "brier": wf_metrics["brier"], "ece": wf_metrics["ece"],
                                "logloss_gain": round(d_ll, 5), "calib_p": calib_p, "pass": rung1_pass},
        "rung2_stability": {"bootstrap_ci_lo": ci_lo, "bootstrap_ci_hi": ci_hi,
                              "quarters_pass": q_pass, "quarters_floor_ok": q_floor_ok,
                              "slices_ok": slices_ok, "slices_elig": slices_elig, "pass": rung2_pass},
        "stable": stable, "verdict": verdict,
    }
    (WORKDIR / "cfb_rushing_touchdowns_walkforward_stability_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport + model written to {WORKDIR}")
    return 0 if stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
