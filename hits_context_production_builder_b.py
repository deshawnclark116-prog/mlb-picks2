#!/usr/bin/env python3
"""
HITS_CONTEXT_PRODUCTION_BUILDER_B

Supersedes hits_context_production_builder_a.py, which had a real gap: its
Stage 2 (the actual retrain-on-all-data run that gets exported and deployed)
trains against a genuine held-out recency slice for early stopping, but never
computes or reports that slice's AUC/calibration -- the artifact that
actually ships was never itself measured. Only Stage 1 (a differently-mixed
train/holdout split) got checked and reported. That gap is exactly why a
model whose ARCHITECTURE tested at AUC 0.60 on a proper holdout
(hits_absolute_significance_gate_a.py, n=22,069) could still ship as a
specific artifact that performs at AUC ~0.50 live -- the recipe was proven,
the actual exported binary never was.

Fix: apply the same two-boss significance gate from
hits_absolute_significance_gate_a.py directly to Stage 2's own held-out
slice, BEFORE allowing export. If the specific artifact about to be shipped
doesn't clear both bosses on its own terms, this refuses to export it --
no promotion, no guessing that "the method worked before so this run must
be fine too."

  BOSS 1: 95% bootstrap CI lower bound for the artifact's own AUC on its
          held-out slice must be > 0.50.
  BOSS 2: paired bootstrap significance vs an empirical-Bayes shrunk
          season_avg baseline (same shrinkage as the significance gate)
          fit on the SAME training rows -- diff CI lower bound must be > 0.

Same feature set as before (base 8 + zero-skew context 8), same
expected_pa lookup construction. Read-only on hr_model.sqlite until the
gate passes; only then are artifacts written.

Run (Render)
------------
python -u hits_context_production_builder_b.py 2>&1 | tee /data/hr_model/hits_context_production_builder_b.log
"""

import argparse
import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

import hits_feature_discovery_b as fd
from hits_absolute_significance_gate_a import (
    auc, bootstrap_ci, paired_bootstrap_diff_ci, SHRINKAGE_PRIOR_AB, GATE,
    N_BOOTSTRAP,
)

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

WORKDIR = Path("/data/hr_model/hits_context_production_builder_b_work")
ZERO_SKEW = ["platoon_advantage", "pitcher_is_R", "is_home", "expected_pa_v1", "recent_xbh_avg",
             "opp_pitcher_h_per_pa", "opp_pitcher_k_per_pa", "opp_pitcher_pa_seen"]
PROD_FEATURES = fd.BASE + ZERO_SKEW

# Final certification slice: the most recent CERT_DAYS of available data,
# never touched by training or by early stopping's own validation slice --
# a genuinely untouched holdout for the specific artifact about to ship,
# not the recipe in the abstract.
CERT_DAYS = 21


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=fd.SOURCE)
    ap.add_argument("--workdir", default=str(WORKDIR))
    args = ap.parse_args()
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(f"file:{args.source}?mode=ro", uri=True)
    print("HITS_CONTEXT_PRODUCTION_BUILDER_B\n=================================", flush=True)
    print("building dataset ...", flush=True)
    data = fd.build(con)
    dev, hol = data["2025"], data["2026"]

    # expected_pa lookup -- unchanged from builder_a, zero train/serve skew.
    cur = con.execute("""SELECT lineup_spot, side, AVG(plate_appearances)
                         FROM batter_games
                         WHERE lineup_spot BETWEEN 1 AND 9 AND side IN ('home','away')
                           AND plate_appearances IS NOT NULL
                         GROUP BY lineup_spot, side""")
    lookup = {f"{int(sp)}|{sd}": round(float(avg), 4) for sp, sd, avg in cur.fetchall()}
    con.close()

    allrows = dev + hol
    for r in allrows:
        f = r["f"]
        side = "home" if f.get("is_home") == 1.0 else "away"
        try:
            key = f"{int(f['batting_order'])}|{side}"
        except Exception:
            key = None
        lk = lookup.get(key) if key else None
        if lk is not None:
            f["expected_pa_v1"] = lk

    def mat(rows, feats):
        X = np.array([[r["f"].get(k, fd.NAN) for k in feats] for r in rows], dtype=np.float32)
        y = np.array([r["y"] for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=feats)

    # ---- Reserve a genuinely untouched final certification slice ----
    adates = sorted({r["game_date"] for r in allrows})
    last_date = date.fromisoformat(adates[-1])
    cert_cutoff = (last_date - timedelta(days=CERT_DAYS)).isoformat()
    pre_cert = [r for r in allrows if r["game_date"] < cert_cutoff]
    cert = [r for r in allrows if r["game_date"] >= cert_cutoff]
    print(f"total rows: {len(allrows)}  cert_cutoff: {cert_cutoff}  "
          f"pre_cert: {len(pre_cert)}  cert (final, untouched): {len(cert)}")

    # ---- Train on pre_cert, early-stopping against ITS OWN tail slice ----
    pre_dates = sorted({r["game_date"] for r in pre_cert})
    ecut = pre_dates[int(len(pre_dates) * 0.9)]
    atr = [r for r in pre_cert if r["game_date"] < ecut]
    ava = [r for r in pre_cert if r["game_date"] >= ecut]
    print(f"train: {len(atr)}  early_stop_val: {len(ava)}  (ecut {ecut})")

    final = xgb.train(fd.PARAMS, mat(atr, PROD_FEATURES), num_boost_round=800,
                      evals=[(mat(ava, PROD_FEATURES), "val")], early_stopping_rounds=40,
                      verbose_eval=False)
    best = final.best_iteration + 1
    print(f"trained on {len(atr)} rows, best_iteration={final.best_iteration}")

    # ---- Certify THIS SPECIFIC artifact on the untouched cert slice ----
    cert_probs = final.predict(mat(cert, PROD_FEATURES), iteration_range=(0, best))
    cert_y = np.array([r["y"] for r in cert], dtype=float)

    # Shrunk season_avg baseline for the cert slice. fd.build()'s returned
    # records don't carry game_id/batter_id forward (only season_avg's
    # already-computed ratio), so cum_ab can't be re-joined here -- approximate
    # it from games_played * league-average ABs/game. Conservative: if
    # anything this under-shrinks a small sample relative to its true AB
    # count, which only makes Boss 2 a harder bar to clear, not an easier one.
    train_avgs = [r["f"]["season_avg"] for r in atr]
    league_avg = float(np.mean(train_avgs))
    print(f"league_avg (train slice): {league_avg:.4f}")

    def shrunk(r):
        n = max(r["f"].get("games_played", 0), 1)
        raw = r["f"]["season_avg"]
        # approximate AB count from games_played * league-wide AB/game (~3.8)
        ab_est = n * 3.8
        return (raw * ab_est + SHRINKAGE_PRIOR_AB * league_avg) / (ab_est + SHRINKAGE_PRIOR_AB)

    x_tr = np.array([[shrunk(r)] for r in atr])
    y_tr = np.array([r["y"] for r in atr], dtype=float)
    x_cert = np.array([[shrunk(r)] for r in cert])

    scaler = StandardScaler().fit(x_tr)
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(scaler.transform(x_tr), y_tr)
    baseline_probs = clf.predict_proba(scaler.transform(x_cert))[:, 1]

    a_final = auc(cert_probs, cert_y)
    _, lo_final, hi_final = bootstrap_ci(cert_probs, cert_y, n_boot=N_BOOTSTRAP)
    a_base = auc(baseline_probs, cert_y)
    _, lo_base, hi_base = bootstrap_ci(baseline_probs, cert_y, n_boot=N_BOOTSTRAP)
    diff_mean, diff_lo, diff_hi = paired_bootstrap_diff_ci(cert_probs, baseline_probs, cert_y, n_boot=N_BOOTSTRAP)

    print(f"\n============ CERTIFICATION (n={len(cert)}, dates {cert_cutoff}..{adates[-1]}) ============")
    print(f"  {'arm':28s} {'AUC':>7s} {'95% CI'}")
    print(f"  {'boss2_season_avg_shrunk':28s} {a_base:.4f}  [{lo_base:.4f}, {hi_base:.4f}]")
    print(f"  {'THIS ARTIFACT':28s} {a_final:.4f}  [{lo_final:.4f}, {hi_final:.4f}]")
    print(f"  paired diff (artifact - boss2): {diff_mean:+.4f}  95% CI [{diff_lo:+.4f}, {diff_hi:+.4f}]")

    boss1_pass = lo_final > GATE["min_auc_ci_lower"]
    boss2_pass = diff_lo > GATE["min_diff_ci_lower"]
    passed = boss1_pass and boss2_pass
    print(f"\n  BOSS 1 (own AUC CI lower bound > {GATE['min_auc_ci_lower']}): {lo_final:.4f} -> {'PASS' if boss1_pass else 'FAIL'}")
    print(f"  BOSS 2 (beats shrunk season_avg, diff CI lower bound > {GATE['min_diff_ci_lower']}): {diff_lo:+.4f} -> {'PASS' if boss2_pass else 'FAIL'}")
    print(f"  VERDICT: {'PASS -- exporting artifact' if passed else 'FAIL -- NOT exporting, do not promote'}")

    report = {
        "script": "HITS_CONTEXT_PRODUCTION_BUILDER_B",
        "cert_days": CERT_DAYS, "cert_cutoff": cert_cutoff, "n_cert": len(cert),
        "n_train": len(atr), "n_early_stop_val": len(ava), "best_iteration": final.best_iteration,
        "artifact_auc": a_final, "artifact_ci": [lo_final, hi_final],
        "boss2_auc": a_base, "boss2_ci": [lo_base, hi_base],
        "paired_diff": {"mean": diff_mean, "ci_lower": diff_lo, "ci_upper": diff_hi},
        "boss1_pass": boss1_pass, "boss2_pass": boss2_pass, "passed": passed,
        "prod_features": PROD_FEATURES,
    }
    out = Path("/data/hr_model") if Path("/data/hr_model").exists() else work
    (out / "hits_context_production_builder_b_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out/'hits_context_production_builder_b_report.json'}")

    if not passed:
        print("\nNOT writing artifacts -- this specific model failed its own certification.")
        return 2

    final.save_model(str(work / "batter_hits_context.json"))
    (work / "batter_hits_context_columns.json").write_text(json.dumps(PROD_FEATURES))
    (work / "expected_pa_lookup.json").write_text(json.dumps(lookup, indent=2))
    print("\nARTIFACTS WRITTEN (to work dir; promoting to /data/models is a separate explicit step):")
    for f in ("batter_hits_context.json", "batter_hits_context_columns.json", "expected_pa_lookup.json"):
        print(f"    {work / f}")
    print("\nRead-only on hr_model.sqlite until the gate passed. No production model or code changed yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
