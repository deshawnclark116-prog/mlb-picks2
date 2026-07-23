#!/usr/bin/env python3
"""
NFL_2025_FRESH_HOLDOUT_VALIDATION_A

The strongest pre-deployment check available: run the EXACT serving
configuration (frozen champions + weekly walk-forward Platt, computed by
the serving engine itself -- nfl_serving_builder_a.SeasonEngine, the very
code that will serve 2026) against the completed 2025 season, which no
model, calibration design, threshold choice, or analysis in this project
has ever touched. One look, pre-registered here before the first run.

Both markets were validated on 2024 (dev=2023). 2025 answers: does the
same frozen-model-plus-rolling-calibration design hold up one MORE season
away from its training data, evaluated by the production code path?

Pre-registered gate, per market (identical criteria to the walk-forward
gates that admitted these markets):
  1. AUC >= 0.58 on pooled walk-forward 2025 predictions
  2. log-loss gain >= 0.01 vs the constant arm (constant predicts the
     warmup season's (2024) full-season base rate -- strictly prior
     information, the analog of the original gates' train-rate arm)
  3. Brier better than the constant arm
  4. synthetic power control: a logit+0.5-distorted copy of the candidate
     must be rejected (bootstrap p < 0.01), else the test is declared
     powerless and no calibration conclusion is drawn
  5. calibration: parametric-bootstrap goodness-of-fit p >= 0.10

Walk-forward pool for 2025 week w = 2024's internal-val slice (pick_val_cut,
the exact role 2023's slice played in the original gates) + 2025 completed
weeks < w. Line stays 49.5 (the frozen serving config being validated).

--holdout-season 2024 exists purely as a mechanics check: with it, this
script must reproduce the original gated numbers (AUC 0.6172 / 0.6637)
exactly, proving the harness before the one real 2025 run.

Read-only on the foundation db + frozen models. Writes one report per
market into the market's model dir.

Run (GitHub runner via .github/workflows/nfl_2025_validation.yml, or
anywhere the foundation db contains 2025 player stats)
---
python -u nfl_2025_fresh_holdout_validation_a.py
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

import nfl_rushing_yards_champion_gate_a as g
from nfl_rushing_yards_recalibration_a import fit_platt, apply_platt, logit, sigmoid
from nfl_serving_builder_a import MARKETS, SeasonEngine, score, DB_DEFAULT, build_platt_pool

B_SIMS = 10_000
PASS_MIN_P = 0.10
CONTROL_MAX_P = 0.01
CONTROL_LOGIT_SHIFT = 0.5
SEED = 13


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


def bootstrap_p(probs, observed_ece, rng, b_sims=B_SIMS):
    probs = np.asarray(probs, dtype=float)
    worse = 0
    for _ in range(b_sims):
        sim_y = rng.binomial(1, probs).astype(float)
        if ece_only(probs, sim_y) >= observed_ece:
            worse += 1
    return worse / b_sims


def validate_market(con, mkt, holdout_season, xgb):
    cfg = MARKETS[mkt]
    line = cfg["line"]
    warm_season = holdout_season - 1
    print(f"\n================ {mkt}  (holdout {holdout_season}, warmup {warm_season}) ================")

    bst = xgb.Booster(); bst.load_model(str(cfg["model_dir"] / f"{cfg['stem']}.json"))
    feat_cols = json.loads((cfg["model_dir"] / f"{cfg['stem']}_columns.json").read_text())
    assert feat_cols == cfg["features"]

    warm = SeasonEngine(con, mkt, warm_season).replay()
    cut = g.pick_val_cut([(None, r[4]) for r in warm])
    warm_slice = [r for r in warm if r[4] >= cut]
    warm_raw = score(bst, cfg["features"], [r[5] for r in warm_slice], xgb)
    warm_y = np.array([1.0 if r[6] >= line + 0.5 else 0.0 for r in warm_slice])
    warm_full_rate = float(np.mean([1.0 if r[6] >= line + 0.5 else 0.0 for r in warm]))
    print(f"warmup: {warm_season} weeks >= {cut}, n={len(warm_slice)}  "
          f"({warm_season} full-season over-rate {warm_full_rate:.4f})")

    hol = SeasonEngine(con, mkt, holdout_season).replay()
    if not hol:
        raise RuntimeError(f"{mkt}: no eligible {holdout_season} rows in db -- "
                           f"was the foundation refreshed with {holdout_season} stats?")
    hol_y = np.array([1.0 if r[6] >= line + 0.5 else 0.0 for r in hol])
    print(f"holdout: n={len(hol)} eligible rows, over-rate {hol_y.mean():.4f}")

    by_week = {}
    for r in hol:
        by_week.setdefault(r[4], []).append(r)

    policy = cfg.get("calibration_policy", "growing")
    probs, ys = [], []
    seen_weeks = []
    platt_trail = []
    for w in sorted(by_week):
        pool_raw, pool_y = build_platt_pool(
            policy, warm_raw, warm_y, seen_weeks,
            window_weeks=cfg.get("calibration_window_weeks"),
            min_rolling_n=cfg.get("calibration_min_rolling_n"))
        a, b = fit_platt(pool_raw, pool_y)
        if a <= 0:
            a, b = 1.0, 0.0
        wk = by_week[w]
        raw = score(bst, cfg["features"], [r[5] for r in wk], xgb)
        y = np.array([1.0 if r[6] >= line + 0.5 else 0.0 for r in wk])
        probs.extend(apply_platt(raw, a, b).tolist())
        ys.extend(y.tolist())
        platt_trail.append({"week": int(w), "policy": policy, "pool_n": int(len(pool_y)),
                             "a": round(float(a), 4), "b": round(float(b), 4),
                             "scored": len(wk)})
        seen_weeks.append((raw, y))

    probs = np.asarray(probs)
    ys = np.asarray(ys)
    m = g.metrics(list(map(float, probs)), list(ys))
    print(f"\nwalk-forward: AUC={m['auc']:.4f}  logloss={m['log_loss']:.5f}  "
          f"Brier={m['brier']:.5f}  ECE={m['ece']:.4f}")
    print("reliability (pred -> actual):")
    for r in m["reliability"]:
        print(f"   {r['bin']}  n={r['n']:>5}  pred={r['pred']:.3f}  actual={r['actual']:.3f}")

    rng = np.random.default_rng(SEED)
    distorted = sigmoid(logit(probs) + CONTROL_LOGIT_SHIFT)
    p_control = bootstrap_p(distorted, ece_only(distorted, ys), rng)
    control_ok = p_control < CONTROL_MAX_P
    print(f"[control] distorted copy: p={p_control:.4f} -> "
          f"{'test has power' if control_ok else 'NO POWER'}")
    p_cal = bootstrap_p(probs, m["ece"], rng)
    print(f"[candidate] calibration p={p_cal:.4f}")

    constant = g.metrics([warm_full_rate] * len(ys), list(ys))
    d_ll = constant["log_loss"] - m["log_loss"]

    c_auc = m["auc"] >= g.GATE["min_auc"]
    c_ll = d_ll >= g.GATE["min_logloss_gain"]
    c_brier = m["brier"] < constant["brier"]
    c_calib = p_cal >= PASS_MIN_P
    passed = c_auc and c_ll and c_brier and control_ok and c_calib
    up = mkt.upper()
    verdict = (f"NFL_{up}_HOLDS_ON_FRESH_{holdout_season}_HOLDOUT" if passed
               else f"NFL_{up}_FAILS_ON_FRESH_{holdout_season}_HOLDOUT")

    print(f"\n============ PRE-REGISTERED GATE ({holdout_season}) ============")
    print(f"  AUC >= {g.GATE['min_auc']}:                 {m['auc']:.4f}  -> {c_auc}")
    print(f"  logloss gain >= {g.GATE['min_logloss_gain']}:       {d_ll:+.5f}  -> {c_ll}")
    print(f"  Brier better than constant:       {m['brier']:.5f} < {constant['brier']:.5f}  -> {c_brier}")
    print(f"  power control (p < {CONTROL_MAX_P}):        p={p_control:.4f}  -> {control_ok}")
    print(f"  calibration test (p >= {PASS_MIN_P}):      p={p_cal:.4f}  -> {c_calib}")
    print(f"  VERDICT: {verdict}")

    report = {"script": "NFL_2025_FRESH_HOLDOUT_VALIDATION_A", "market": mkt,
              "holdout_season": holdout_season, "warmup_season": warm_season,
              "line": line, "n": int(len(ys)), "base_rate": round(float(ys.mean()), 4),
              "warmup_full_season_rate": round(warm_full_rate, 4),
              "metrics": m, "p_control": p_control, "p_calibration": p_cal,
              "constant": {"log_loss": constant["log_loss"], "brier": constant["brier"]},
              "platt_trail": platt_trail, "passed": bool(passed), "verdict": verdict}
    rp = cfg["model_dir"] / (f"nfl_2025_fresh_holdout_validation_a_{mkt}_"
                              f"{holdout_season}_report.json")
    rp.write_text(json.dumps(report, indent=2))
    print(f"report: {rp}")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_DEFAULT))
    ap.add_argument("--holdout-season", type=int, default=2025,
                    help="2024 = mechanics check (must reproduce the gated numbers)")
    args = ap.parse_args()
    import xgboost as xgb

    print("NFL_2025_FRESH_HOLDOUT_VALIDATION_A\n" + "=" * 35)
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    seasons = sorted(r[0] for r in con.execute("SELECT DISTINCT season FROM player_games"))
    print(f"player_games seasons in db: {seasons}")

    results = {}
    for mkt in MARKETS:
        results[mkt] = validate_market(con, mkt, args.holdout_season, xgb)
    con.close()

    print("\n================ SUMMARY ================")
    for mkt, ok in results.items():
        print(f"  {mkt}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
