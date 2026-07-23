#!/usr/bin/env python3
"""
NFL_RECEIVING_YARDS_ROLLING_CALIBRATION_A

Head-to-head test: does a ROLLING window of recent current-season weeks
calibrate receiving_yards better than the GROWING pool (warmup + all
current-season weeks so far) that's already in production?

Motivation: the 2025 fresh-holdout run showed receiving_yards keeps real
discrimination (AUC 0.6371) but is overconfident (calibration p=0.0422,
fails the p>=0.10 bar) under the growing-pool design. A narrower window of
only the most recent weeks might track a mid-season calibration drift more
tightly -- but it also means fitting Platt's 2 parameters on far fewer
points, which is exactly the failure mode that broke isotonic regression
earlier in this project (n=75 -> catastrophic overfit). Rather than assume
either way, this runs both policies through the identical power-adjusted
bootstrap gate used everywhere else in this project and reports both.

GROWING (baseline, already in production):
  pool for week w = warmup (2024 tail, fixed) + every completed 2025 week < w

ROLLING (candidate):
  if fewer than MIN_ROLLING_N current-season rows have accumulated yet,
  falls back to the growing-pool behavior (warmup + everything so far) --
  there's nothing to roll a window over yet.
  Once MIN_ROLLING_N is reached, pool for week w = only the last
  WINDOW_WEEKS completed weeks (2025 rows only, warmup dropped entirely).

Same raw XGBoost scores feed both policies (computed once); only the Platt
fit's input pool differs. Uses the exact same eligible rows, model, and
metrics/bootstrap machinery as nfl_2025_fresh_holdout_validation_a.py so
the two are directly comparable to that already-run baseline.

Run (GitHub runner -- needs nflverse + the frozen receiving_yards model)
---
python -u nfl_receiving_yards_rolling_calibration_a.py
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

import nfl_rushing_yards_champion_gate_a as g
from nfl_rushing_yards_recalibration_a import fit_platt, apply_platt, logit, sigmoid
from nfl_serving_builder_a import MARKETS, SeasonEngine, score, DB_DEFAULT
from nfl_2025_fresh_holdout_validation_a import ece_only, bootstrap_p, B_SIMS, PASS_MIN_P, CONTROL_MAX_P, CONTROL_LOGIT_SHIFT, SEED

MKT = "receiving_yards"
HOLDOUT_SEASON = 2025
WINDOW_WEEKS = 5
MIN_ROLLING_N = 50


def run_policy(policy, by_week, warm_raw, warm_y, bst, cfg, xgb, line):
    """Walk forward through by_week, fitting Platt per the given policy,
    and return (probs, ys, platt_trail)."""
    weeks_sorted = sorted(by_week)
    probs, ys, platt_trail = [], [], []
    seen_weeks = []  # list of (week, X_rows, y_array) in chronological order

    for w in weeks_sorted:
        if policy == "growing":
            pool_raw_parts = [warm_raw] + [pr for (_, pr, _) in seen_weeks]
            pool_y_parts = [warm_y] + [py for (_, _, py) in seen_weeks]
        elif policy == "rolling":
            total_current_n = sum(len(py) for (_, _, py) in seen_weeks)
            if total_current_n < MIN_ROLLING_N:
                pool_raw_parts = [warm_raw] + [pr for (_, pr, _) in seen_weeks]
                pool_y_parts = [warm_y] + [py for (_, _, py) in seen_weeks]
            else:
                recent = seen_weeks[-WINDOW_WEEKS:]
                pool_raw_parts = [pr for (_, pr, _) in recent]
                pool_y_parts = [py for (_, _, py) in recent]
        else:
            raise ValueError(policy)

        pool_raw = np.concatenate(pool_raw_parts) if pool_raw_parts else np.empty(0)
        pool_y = np.concatenate(pool_y_parts) if pool_y_parts else np.empty(0)
        a, b = fit_platt(pool_raw, pool_y)
        if a <= 0:
            a, b = 1.0, 0.0

        wk_rows = by_week[w]
        raw = score(bst, cfg["features"], [r[5] for r in wk_rows], xgb)
        y = np.array([1.0 if r[6] >= line + 0.5 else 0.0 for r in wk_rows])
        probs.extend(apply_platt(raw, a, b).tolist())
        ys.extend(y.tolist())
        platt_trail.append({"week": int(w), "pool_n": int(len(pool_y)),
                             "a": round(float(a), 4), "b": round(float(b), 4),
                             "scored": len(wk_rows)})
        seen_weeks.append((w, raw, y))

    return np.asarray(probs), np.asarray(ys), platt_trail


def evaluate(probs, ys, warm_full_rate, rng):
    m = g.metrics(list(map(float, probs)), list(ys))
    distorted = sigmoid(logit(probs) + CONTROL_LOGIT_SHIFT)
    p_control = bootstrap_p(distorted, ece_only(distorted, ys), rng)
    control_ok = p_control < CONTROL_MAX_P
    p_cal = bootstrap_p(probs, m["ece"], rng)

    constant = g.metrics([warm_full_rate] * len(ys), list(ys))
    d_ll = constant["log_loss"] - m["log_loss"]

    c_auc = m["auc"] >= g.GATE["min_auc"]
    c_ll = d_ll >= g.GATE["min_logloss_gain"]
    c_brier = m["brier"] < constant["brier"]
    c_calib = p_cal >= PASS_MIN_P
    passed = c_auc and c_ll and c_brier and control_ok and c_calib

    return {
        "metrics": m, "p_control": p_control, "control_ok": control_ok,
        "p_calibration": p_cal, "d_logloss": d_ll,
        "constant_brier": constant["brier"], "gate": {
            "auc": c_auc, "logloss_gain": c_ll, "brier": c_brier,
            "power_control": control_ok, "calibration": c_calib,
        },
        "passed": bool(passed),
    }


def main():
    import xgboost as xgb

    print("NFL_RECEIVING_YARDS_ROLLING_CALIBRATION_A\n" + "=" * 43)
    con = sqlite3.connect(f"file:{DB_DEFAULT}?mode=ro", uri=True)
    seasons = sorted(r[0] for r in con.execute("SELECT DISTINCT season FROM player_games"))
    print(f"player_games seasons in db: {seasons}")

    cfg = MARKETS[MKT]
    line = cfg["line"]
    warm_season = HOLDOUT_SEASON - 1

    bst = xgb.Booster(); bst.load_model(str(cfg["model_dir"] / f"{cfg['stem']}.json"))
    feat_cols = json.loads((cfg["model_dir"] / f"{cfg['stem']}_columns.json").read_text())
    assert feat_cols == cfg["features"]

    warm = SeasonEngine(con, MKT, warm_season).replay()
    cut = g.pick_val_cut([(None, r[4]) for r in warm])
    warm_slice = [r for r in warm if r[4] >= cut]
    warm_raw = score(bst, cfg["features"], [r[5] for r in warm_slice], xgb)
    warm_y = np.array([1.0 if r[6] >= line + 0.5 else 0.0 for r in warm_slice])
    warm_full_rate = float(np.mean([1.0 if r[6] >= line + 0.5 else 0.0 for r in warm]))
    print(f"warmup: {warm_season} weeks >= {cut}, n={len(warm_slice)}  "
          f"({warm_season} full-season over-rate {warm_full_rate:.4f})")

    hol = SeasonEngine(con, MKT, HOLDOUT_SEASON).replay()
    if not hol:
        raise RuntimeError(f"{MKT}: no eligible {HOLDOUT_SEASON} rows in db")
    by_week = {}
    for r in hol:
        by_week.setdefault(r[4], []).append(r)
    print(f"holdout: n={len(hol)} eligible rows across {len(by_week)} weeks")
    print(f"rolling policy: window={WINDOW_WEEKS} weeks, min_rolling_n={MIN_ROLLING_N}")

    results = {}
    for policy in ("growing", "rolling"):
        probs, ys, trail = run_policy(policy, by_week, warm_raw, warm_y, bst, cfg, xgb, line)
        rng = np.random.default_rng(SEED)
        ev = evaluate(probs, ys, warm_full_rate, rng)
        ev["platt_trail"] = trail
        results[policy] = ev
        m = ev["metrics"]
        print(f"\n============ POLICY: {policy} ============")
        print(f"  AUC={m['auc']:.4f}  logloss={m['log_loss']:.5f}  "
              f"Brier={m['brier']:.5f}  ECE={m['ece']:.4f}")
        print(f"  AUC >= {g.GATE['min_auc']}: {ev['gate']['auc']}")
        print(f"  logloss gain >= {g.GATE['min_logloss_gain']}: {ev['d_logloss']:+.5f} -> {ev['gate']['logloss_gain']}")
        print(f"  Brier better than constant: {m['brier']:.5f} < {ev['constant_brier']:.5f} -> {ev['gate']['brier']}")
        print(f"  power control (p < {CONTROL_MAX_P}): p={ev['p_control']:.4f} -> {ev['control_ok']}")
        print(f"  calibration test (p >= {PASS_MIN_P}): p={ev['p_calibration']:.4f} -> {ev['gate']['calibration']}")
        print(f"  PASSED: {ev['passed']}")

    print("\n================ HEAD-TO-HEAD VERDICT ================")
    g_ev, r_ev = results["growing"], results["rolling"]
    print(f"  growing: calibration p={g_ev['p_calibration']:.4f}  passed={g_ev['passed']}")
    print(f"  rolling: calibration p={r_ev['p_calibration']:.4f}  passed={r_ev['passed']}")
    if r_ev["passed"] and not g_ev["passed"]:
        verdict = "ROLLING_WINDOW_FIXES_CALIBRATION"
    elif r_ev["passed"] and g_ev["passed"]:
        verdict = "BOTH_PASS_KEEP_GROWING_SIMPLER"
    elif not r_ev["passed"] and not g_ev["passed"]:
        verdict = "NEITHER_PASSES_CALIBRATION_NOT_THE_FIX"
    else:
        verdict = "ROLLING_WINDOW_WORSE_REJECTED"
    print(f"  VERDICT: {verdict}")

    report = {
        "script": "NFL_RECEIVING_YARDS_ROLLING_CALIBRATION_A",
        "market": MKT, "holdout_season": HOLDOUT_SEASON,
        "window_weeks": WINDOW_WEEKS, "min_rolling_n": MIN_ROLLING_N,
        "growing": g_ev,
        "rolling": r_ev,
        "verdict": verdict,
    }
    rp = cfg["model_dir"] / "nfl_receiving_yards_rolling_calibration_a_report.json"
    rp.write_text(json.dumps(report, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o))
    print(f"\nreport: {rp}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
