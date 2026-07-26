#!/usr/bin/env python3
"""
HITS_ABSOLUTE_SIGNIFICANCE_GATE_A

Root cause of today's investigation: hits_context_challenger_gate_a.py (the
gate that shipped the currently-deployed batter_hits_context model) only
ever tested a RELATIVE delta -- "does adding context features beat a
retrained base model by >=0.005 AUC" -- with no absolute floor and no
significance test. That let a model with near-zero real discrimination
(live AUC 0.505 on 1,036 graded picks, 0.493 on a properly matched
same-population comparison) clear the gate and ship.

This gate closes both gaps. Every challenger must clear TWO bosses on the
SAME untouched 2026 holdout used everywhere else in this repo:

  BOSS 1 (real discrimination, not a coin flip):
      95% bootstrap CI lower bound for the challenger's own AUC must be
      strictly > 0.50.

  BOSS 2 (beats its own simplest input, with statistical significance,
          not just a bigger point estimate):
      A season_avg-only baseline, WITH empirical-Bayes shrinkage toward
      the league mean (so a guy with 8 at-bats doesn't get a wild raw
      average), is fit as a single-feature logistic regression on the
      SAME train/holdout split. Challenger vs this baseline is compared
      via PAIRED bootstrap resampling (same resampled rows scored by
      both arms, not independently resampled arms): the 95% CI of
      (AUC_challenger - AUC_baseline) must have a lower bound strictly
      > 0.0.

If either boss is not cleared: hard FAIL, no artifact exported, no
auto-promotion. This mirrors exactly what the ad hoc afternoon
investigation found by hand (season_avg AUC 0.535 [0.495, 0.576],
challenger AUC 0.493 [0.454, 0.533], diff CI touching 0.000) -- codifying
that discipline into the pipeline itself instead of relying on someone
running it by hand again next time.

Read-only on hr_model.sqlite. Writes only a report + arm models to its
own work dir. Touches no production model or code. No auto-promotion.

Run (Render)
------------
python -u hits_absolute_significance_gate_a.py 2>&1 | tee /data/hr_model/hits_absolute_significance_gate_a.log
"""

import argparse
import json
import math
import sqlite3
import sys
from collections import deque
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

SOURCE = "/data/hr_model/hr_model.sqlite"
WORKDIR = Path("/data/hr_model/hits_absolute_significance_gate_a_work")
MIN_PRIOR_AB = 20
MIN_PRIOR_GAMES = 5
NAN = float("nan")
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 13

# The actual currently-deployed feature set (fd.BASE + ZERO_SKEW from
# hits_context_production_builder_a.py) -- the challenger tested here IS
# the real thing, not a stand-in for it.
BASE_FEATURES = ["season_avg", "recent15_avg", "recent5_avg", "hr_rate",
                 "bb_rate", "so_rate", "batting_order", "games_played"]
CONTEXT_FEATURES = ["platoon_advantage", "pitcher_is_R", "is_home", "expected_pa_v1",
                     "recent_xbh_avg", "opp_pitcher_h_per_pa", "opp_pitcher_k_per_pa",
                     "opp_pitcher_pa_seen"]
CHALLENGER_FEATURES = BASE_FEATURES + CONTEXT_FEATURES

PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 4,
          "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
          "min_child_weight": 5, "seed": 13}

# Empirical-Bayes shrinkage prior strength, in pseudo-at-bats. Chosen as a
# round number representing roughly 3 weeks of a everyday player's at-bats
# -- strong enough to tame a 10-AB sample, light enough not to flatten a
# real 300+ AB season sample toward the mean.
SHRINKAGE_PRIOR_AB = 50

GATE = {"min_auc_ci_lower": 0.50, "min_diff_ci_lower": 0.0}


def platoon_advantage(bh, ph):
    if bh in (None, "") or ph in (None, ""):
        return NAN
    if bh == "S":
        return 1.0
    if (bh == "L" and ph == "R") or (bh == "R" and ph == "L"):
        return 1.0
    return 0.0


def fnum(v):
    return float(v) if v is not None else NAN


def auc(scores, labels):
    import numpy as np
    scores = np.asarray(scores, dtype=float); labels = np.asarray(labels, dtype=float)
    pos = labels.sum(); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]; i = 0; n = len(s)
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def build_dataset(conn):
    """Same strict-D-1 dataset as hits_context_challenger_gate_a.py, plus
    cum_h/cum_ab carried through per row (not just the ratio) so a proper
    empirical-Bayes shrunk season average can be computed for Boss 2 --
    the ratio alone isn't enough to shrink correctly toward the mean."""
    cols = ["game_id", "game_date", "batter_id", "lineup_spot", "batter_hand",
            "pitcher_hand", "opposing_pitcher_id", "side", "expected_pa_v1",
            "plate_appearances", "at_bats", "hits", "doubles", "triples",
            "home_runs", "walks", "strikeouts"]
    rows = conn.execute(
        f"SELECT {','.join(cols)} FROM batter_games WHERE at_bats IS NOT NULL "
        f"ORDER BY batter_id, game_date, game_id").fetchall()
    ix = {c: i for i, c in enumerate(cols)}

    def N(r, k):
        v = r[ix[k]]
        return v if v is not None else 0

    feat = {}
    cur = None; group = []

    def flush(group):
        cum = dict(pa=0, ab=0, h=0, hr=0, bb=0, so=0)
        rec_h = deque(maxlen=15); rec_xbh = deque(maxlen=15)
        n_prior = 0; i = 0
        for g in group:
            gd = g[ix["game_date"]]
            while i < len(group) and group[i][ix["game_date"]] < gd:
                h = group[i]
                cum["pa"] += N(h, "plate_appearances"); cum["ab"] += N(h, "at_bats")
                cum["h"] += N(h, "hits"); cum["hr"] += N(h, "home_runs")
                cum["bb"] += N(h, "walks"); cum["so"] += N(h, "strikeouts")
                rec_h.append(N(h, "hits"))
                rec_xbh.append(N(h, "doubles") + N(h, "triples") + N(h, "home_runs"))
                n_prior += 1; i += 1
            if n_prior < MIN_PRIOR_GAMES or cum["ab"] < MIN_PRIOR_AB:
                continue
            pa = cum["pa"] or 1
            r5 = list(rec_h)[-5:]; r15 = list(rec_h)
            side = g[ix["side"]]
            f = {
                "season_avg": cum["h"] / cum["ab"] if cum["ab"] else 0.0,
                "recent5_avg": sum(r5) / len(r5) if r5 else 0.0,
                "recent15_avg": sum(r15) / len(r15) if r15 else 0.0,
                "hr_rate": cum["hr"] / pa, "bb_rate": cum["bb"] / pa, "so_rate": cum["so"] / pa,
                "batting_order": g[ix["lineup_spot"]] if g[ix["lineup_spot"]] is not None else 9,
                "games_played": n_prior,
                "recent_xbh_avg": sum(rec_xbh) / len(rec_xbh) if rec_xbh else 0.0,
                "platoon_advantage": platoon_advantage(g[ix["batter_hand"]], g[ix["pitcher_hand"]]),
                "pitcher_is_R": 1.0 if g[ix["pitcher_hand"]] == "R" else (0.0 if g[ix["pitcher_hand"]] in ("L", "S") else NAN),
                "is_home": 1.0 if side == "home" else (0.0 if side == "away" else NAN),
                "expected_pa_v1": fnum(g[ix["expected_pa_v1"]]),
            }
            feat[(g[ix["game_id"]], g[ix["batter_id"]])] = {
                "season": gd[:4], "game_date": gd, "f": f,
                "cum_h": cum["h"], "cum_ab": cum["ab"],
                "y": 1 if N(g, "hits") >= 1 else 0,
            }
        return

    for r in rows:
        if r[ix["batter_id"]] != cur:
            if group:
                flush(group)
            group = []; cur = r[ix["batter_id"]]
        group.append(r)
    if group:
        flush(group)

    # opposing pitcher as-of hits/K allowed -- from pitcher_games (each
    # pitcher's own precise per-start line), matching the live serving
    # definition exactly (see hits_feature_discovery_b.py's build() for
    # the concrete Eduardo Rivera proof this fixed a real dilution bug).
    pgs = conn.execute(
        "SELECT pitcher_id, game_date, hits_allowed, batters_faced, strikeouts "
        "FROM pitcher_games WHERE batters_faced >= 12 ORDER BY pitcher_id, game_date"
    ).fetchall()
    pitcher_hist = {}
    for pid, gd, hits, bf, so in pgs:
        pitcher_hist.setdefault(pid, []).append((gd, hits or 0, bf or 0, so or 0))

    for r in rows:
        k = (r[ix["game_id"]], r[ix["batter_id"]])
        if k not in feat:
            continue
        opp_pid = r[ix["opposing_pitcher_id"]]
        game_date = r[ix["game_date"]]
        cum_hits = cum_bf = cum_so = 0
        for gd, hits, bf, so in pitcher_hist.get(opp_pid, []):
            if gd >= game_date:
                break
            cum_hits += hits; cum_bf += bf; cum_so += so
        if cum_bf > 0:
            feat[k]["f"]["opp_pitcher_h_per_pa"] = cum_hits / cum_bf
            feat[k]["f"]["opp_pitcher_pa_seen"] = float(cum_bf)
            feat[k]["f"]["opp_pitcher_k_per_pa"] = cum_so / cum_bf
        else:
            feat[k]["f"]["opp_pitcher_h_per_pa"] = NAN
            feat[k]["f"]["opp_pitcher_pa_seen"] = 0.0
            feat[k]["f"]["opp_pitcher_k_per_pa"] = NAN

    data = {"2025": [], "2026": []}
    for rec in feat.values():
        if rec["season"] in data:
            data[rec["season"]].append(rec)
    return data


def bootstrap_ci(scores, labels, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """95% percentile bootstrap CI for a single arm's AUC."""
    import numpy as np
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores); labels = np.asarray(labels)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y = labels[idx]
        if y.sum() == 0 or y.sum() == n:
            continue
        vals.append(auc(scores[idx], y))
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_diff_ci(scores_a, scores_b, labels, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """95% CI for (AUC_a - AUC_b), resampling the SAME row indices for both
    arms each draw (paired, not independent) -- this is the actual
    significance test a relative-only point-estimate delta never was."""
    import numpy as np
    rng = np.random.default_rng(seed)
    sa = np.asarray(scores_a); sb = np.asarray(scores_b); y = np.asarray(labels)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == n:
            continue
        diffs.append(auc(sa[idx], yy) - auc(sb[idx], yy))
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--workdir", default=str(WORKDIR))
    args = ap.parse_args()

    import numpy as np
    import xgboost as xgb

    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{args.source}?mode=ro", uri=True)
    print("HITS_ABSOLUTE_SIGNIFICANCE_GATE_A\n==================================", flush=True)
    print("building strict-D-1 dataset ...", flush=True)
    data = build_dataset(conn)
    conn.close()
    dev, hol = data["2025"], data["2026"]
    print(f"  2025 (dev) rows: {len(dev)}   2026 (holdout) rows: {len(hol)}")

    dates = sorted({r["game_date"] for r in dev})
    cut = dates[int(len(dates) * 0.8)] if len(dates) > 5 else dates[-1]
    tr = [r for r in dev if r["game_date"] < cut]
    va = [r for r in dev if r["game_date"] >= cut]
    print(f"  train {len(tr)}  internal_val {len(va)}  holdout {len(hol)}  (val cut {cut})")

    def mat(rowset, feats):
        X = np.array([[r["f"].get(k, NAN) for k in feats] for r in rowset], dtype=np.float32)
        y = np.array([r["y"] for r in rowset], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=feats)

    # ---- Challenger: the actual deployed feature set ----
    print("\ntraining challenger (deployed feature set: base 8 + zero-skew context 8) ...")
    booster = xgb.train(PARAMS, mat(tr, CHALLENGER_FEATURES), num_boost_round=800,
                        evals=[(mat(va, CHALLENGER_FEATURES), "val")],
                        early_stopping_rounds=40, verbose_eval=False)
    best = booster.best_iteration + 1
    chal_probs = booster.predict(mat(hol, CHALLENGER_FEATURES), iteration_range=(0, best))
    hol_y = np.array([r["y"] for r in hol], dtype=float)

    # ---- Boss 2 baseline: empirical-Bayes shrunk season_avg, single-feature
    # logistic regression (fit on train, scored on holdout -- same split
    # discipline as the challenger, not a shortcut). ----
    print("training Boss 2 baseline (empirical-Bayes shrunk season_avg, 1-feature logistic) ...")
    league_avg = float(np.mean([r["cum_h"] / r["cum_ab"] for r in tr if r["cum_ab"] > 0]))
    print(f"  league_avg (train, {SHRINKAGE_PRIOR_AB}-AB shrinkage prior): {league_avg:.4f}")

    def shrunk_avg(r):
        return (r["cum_h"] + SHRINKAGE_PRIOR_AB * league_avg) / (r["cum_ab"] + SHRINKAGE_PRIOR_AB)

    x_tr = np.array([[shrunk_avg(r)] for r in tr], dtype=np.float64)
    y_tr = np.array([r["y"] for r in tr], dtype=np.float64)
    x_hol = np.array([[shrunk_avg(r)] for r in hol], dtype=np.float64)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x_tr)
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(scaler.transform(x_tr), y_tr)
    baseline_probs = clf.predict_proba(scaler.transform(x_hol))[:, 1]

    # ---- Constant, for reference only (not a boss, just context) ----
    const_rate = float(np.mean(y_tr))
    const_probs = np.full(len(hol), const_rate)

    def report_arm(name, probs):
        a = auc(probs, hol_y)
        mean, lo, hi = bootstrap_ci(probs, hol_y)
        print(f"  {name:22s} AUC={a:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  n={len(hol_y)}")
        return {"auc": a, "ci_mean": mean, "ci_lower": lo, "ci_upper": hi}

    print(f"\n============ 2026 HOLDOUT (n={len(hol)}) ============")
    r_const = report_arm("constant", const_probs)
    r_base = report_arm("boss2_season_avg_shrunk", baseline_probs)
    r_chal = report_arm("challenger (deployed)", chal_probs)

    diff_mean, diff_lo, diff_hi = paired_bootstrap_diff_ci(chal_probs, baseline_probs, hol_y)
    print(f"\npaired bootstrap: AUC(challenger) - AUC(boss2_season_avg_shrunk)")
    print(f"  mean={diff_mean:+.4f}  95% CI [{diff_lo:+.4f}, {diff_hi:+.4f}]")

    boss1_pass = r_chal["ci_lower"] > GATE["min_auc_ci_lower"]
    boss2_pass = diff_lo > GATE["min_diff_ci_lower"]
    passed = boss1_pass and boss2_pass

    print(f"\n============ GATE ============")
    print(f"  BOSS 1 (challenger AUC 95% CI lower bound > {GATE['min_auc_ci_lower']}): "
          f"{r_chal['ci_lower']:.4f} -> {'PASS' if boss1_pass else 'FAIL'}")
    print(f"  BOSS 2 (challenger beats shrunk season_avg, diff CI lower bound > {GATE['min_diff_ci_lower']}): "
          f"{diff_lo:+.4f} -> {'PASS' if boss2_pass else 'FAIL'}")
    print(f"  VERDICT: {'PASS -- safe to promote' if passed else 'FAIL -- do not promote, do not auto-ship'}")

    report = {
        "script": "HITS_ABSOLUTE_SIGNIFICANCE_GATE_A",
        "holdout": "2026", "n_holdout": len(hol),
        "n_train": len(tr), "n_internal_val": len(va),
        "shrinkage_prior_ab": SHRINKAGE_PRIOR_AB, "league_avg_train": league_avg,
        "arms": {"constant": r_const, "boss2_season_avg_shrunk": r_base, "challenger": r_chal},
        "paired_diff_challenger_minus_boss2": {"mean": diff_mean, "ci_lower": diff_lo, "ci_upper": diff_hi},
        "gate": GATE, "boss1_pass": boss1_pass, "boss2_pass": boss2_pass, "passed": passed,
        "challenger_features": CHALLENGER_FEATURES,
    }
    (work / "hits_absolute_significance_gate_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {work / 'hits_absolute_significance_gate_a_report.json'}")
    print("Read-only on hr_model.sqlite. No production model or code changed. No auto-promotion.")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
