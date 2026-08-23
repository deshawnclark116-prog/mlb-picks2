#!/usr/bin/env python3
"""
RBI_RUNS_CLEAN_BASELINE_AND_GATE_A

Real validation for batter_rbi and batter_runs. Both already had a trained
model sitting in models/ (batter_rbi.json, batter_runs.json), but neither
had ever been through this repo's real validation discipline -- unlike
every other market here, there's no champion_gate script for them.
Checked how they were actually made (train_extra.py): count:poisson
objective, trained in incremental per-season CHUNKS with NO early stopping
and NO held-out validation, run manually per season ("python train_extra.py
rbi 2024", "... rbi 2025", ...). Provenance is ambiguous enough that 2025
may already be baked into the saved artifact -- scoring it as a "holdout"
would not be honest. So this retrains both from scratch instead of trying
to validate the existing artifacts.

Feature set is NOT reinvented -- it's copied directly from the existing
columns files (models/batter_rbi_columns.json, batter_runs_columns.json),
which matches the validated batter_total_bases feature set exactly (both
were trained by the same train_extra.py against the same feature builder).
Strict D-1, same as total_bases_clean_baseline_a.py.

Split: 2024 = dev, 2025 = untouched holdout (real, complete seasons, from
backfill.py's season_{year}.jsonl -- same source as the other new-market
validations run today).

Objective: binary:logistic on over_line (not count:poisson regression like
the original) -- matches this repo's gate methodology (AUC/ECE/logloss vs
a constant), the same target reframing total_bases uses (prob_over via its
own gate, not a raw count regression scored directly).

Pre-registered pass: AUC >= 0.58, ECE <= 0.03, logloss gain >= 0.01,
Brier better than constant. Same bar as every other market here.

Read-only on season_{year}.jsonl and the existing models (never modifies
them). Writes only its own workdir.

Run:
    python -u rbi_runs_clean_baseline_and_gate_a.py
"""
import json
import sqlite3
import sys
from collections import deque
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

DATA_DIR = Path("/data")
WORKDIR = Path("/data/rbi_runs_clean_baseline_and_gate_a_work")
DEV_SEASON = 2024
HOLDOUT_SEASON = 2025

FEATURE_COLS = ["season_avg", "tb_per_pa", "rbi_per_pa", "runs_per_pa", "hr_rate",
                "bb_rate", "so_rate", "recent_target5", "recent_target15",
                "batting_order", "games_played"]

MARKETS = {
    "batter_rbi": {"target": "rbi", "threshold_candidates": [0.5, 1.5]},
    "batter_runs": {"target": "runs", "threshold_candidates": [0.5, 1.5]},
}
MIN_AB = 20
MIN_GAMES = 5
NAN = float("nan")
GATE = {"min_auc": 0.58, "max_ece": 0.03, "min_logloss_gain": 0.01}
MIN_VAL_ROWS = 200
VAL_FRAC = 0.2


def _n(v):
    return v if isinstance(v, (int, float)) and v is not None else 0


def load_batters(season):
    out = []
    with open(DATA_DIR / f"season_{season}.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") == "batter":
                out.append(r)
    return out


def build_rows(rows, target_field):
    by_player = {}
    for r in rows:
        by_player.setdefault(r["player_id"], []).append(r)

    out = []
    for pid, games in by_player.items():
        games.sort(key=lambda r: r["date"])
        cum_ab = cum_h = cum_pa = cum_tb = cum_rbi = cum_runs = cum_hr = cum_bb = cum_so = 0
        recent_target = deque(maxlen=15)
        n = 0
        for g in games:
            ab, pa, h = _n(g.get("ab")), _n(g.get("pa")), _n(g.get("h"))
            tb, rbi, runs = _n(g.get("tb")), _n(g.get("rbi")), _n(g.get("runs"))
            y = _n(g.get(target_field))

            if cum_ab >= MIN_AB and n >= MIN_GAMES:
                r5 = list(recent_target)[-5:]
                r15 = list(recent_target)
                feat = {
                    "season_avg": cum_h / cum_ab if cum_ab else 0.0,
                    "tb_per_pa": cum_tb / cum_pa if cum_pa else 0.0,
                    "rbi_per_pa": cum_rbi / cum_pa if cum_pa else 0.0,
                    "runs_per_pa": cum_runs / cum_pa if cum_pa else 0.0,
                    "hr_rate": cum_hr / cum_pa if cum_pa else 0.0,
                    "bb_rate": cum_bb / cum_pa if cum_pa else 0.0,
                    "so_rate": cum_so / cum_pa if cum_pa else 0.0,
                    "recent_target5": sum(r5) / len(r5) if r5 else 0.0,
                    "recent_target15": sum(r15) / len(r15) if r15 else 0.0,
                    "batting_order": g.get("batting_order") or 9,
                    "games_played": n,
                }
                out.append({
                    "player_id": pid, "name": g.get("name"), "date": g["date"],
                    **feat, "actual": y,
                })

            cum_ab += ab; cum_h += h; cum_pa += pa
            cum_tb += tb; cum_rbi += rbi; cum_runs += runs
            cum_hr += _n(g.get("hr")); cum_bb += _n(g.get("bb")); cum_so += _n(g.get("so"))
            recent_target.append(y)
            n += 1
    return out


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
        rel.append({"bin": f"{b/10:.1f}-{(b+1)/10:.1f}", "n": cnt, "pred": round(mp, 4), "actual": round(ar, 4)})
    return {"n": n, "base_rate": round(float(y.mean()), 4), "auc": round(auc(probs, labels), 4),
            "log_loss": round(ll, 5), "brier": round(brier, 5), "ece": round(ece, 4), "reliability": rel}


def pick_val_cut(dates_sorted):
    total = len(dates_sorted)
    target = max(MIN_VAL_ROWS, int(total * VAL_FRAC))
    return dates_sorted[max(0, total - target)]


def mat(rows, xgb):
    X = np.array([[r.get(c, NAN) if r.get(c) is not None else NAN for c in FEATURE_COLS] for r in rows], dtype=np.float32)
    y = np.array([r["over_line"] for r in rows], dtype=np.float32)
    return xgb.DMatrix(X, label=y, feature_names=FEATURE_COLS)


def run_market(mkt_key, cfg, dev_rows, hol_rows, workdir):
    import xgboost as xgb
    print(f"\n{'='*70}\n{mkt_key}\n{'='*70}")
    print(f"dev {DEV_SEASON} rows: {len(dev_rows)}   holdout {HOLDOUT_SEASON} rows: {len(hol_rows)}")

    dev_actual = [r["actual"] for r in dev_rows]
    best_t, best_dist = None, 1.0
    print(f"\nTHRESHOLD DIAGNOSTIC (dev only, n={len(dev_actual)})")
    for t in cfg["threshold_candidates"]:
        rate = sum(1 for y in dev_actual if y >= t + 0.5) / len(dev_actual)
        dist = abs(rate - 0.5)
        if dist < best_dist:
            best_dist, best_t = dist, t
        print(f"  >{t:>6.1f}   over_rate={rate:.3f}")
    LINE = best_t
    print(f"  closest-to-50% threshold: over {LINE}")

    for r in dev_rows:
        r["over_line"] = 1 if r["actual"] >= (LINE + 0.5) else 0
    for r in hol_rows:
        r["over_line"] = 1 if r["actual"] >= (LINE + 0.5) else 0

    dates = [r["date"] for r in dev_rows]
    cut = pick_val_cut(sorted(dates))
    tr = [r for r in dev_rows if r["date"] < cut]
    va = [r for r in dev_rows if r["date"] >= cut]
    print(f"  train (date < {cut}): {len(tr)}   internal val (>= {cut}): {len(va)}")

    print("training challenger (binary:logistic) ...", flush=True)
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 4,
              "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 5, "seed": 13}
    bst = xgb.train(params, mat(tr, xgb), num_boost_round=800, evals=[(mat(va, xgb), "val")],
                     early_stopping_rounds=40, verbose_eval=False)
    itr = (0, bst.best_iteration + 1)
    print(f"  best_iteration={bst.best_iteration}  scoring with iteration_range={itr}")

    probs_hol = bst.predict(mat(hol_rows, xgb), iteration_range=itr)
    labels_hol = [r["over_line"] for r in hol_rows]
    challenger = metrics(list(map(float, probs_hol)), labels_hol)
    train_rate = float(np.mean([r["over_line"] for r in tr]))
    constant = metrics([train_rate] * len(hol_rows), labels_hol)

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
    (workdir / f"{mkt_key}_columns.json").write_text(json.dumps(FEATURE_COLS))
    return {"line": LINE, "constant": constant, "challenger": challenger, "gate": GATE,
            "passed": passed, "verdict": verdict, "importance": imp,
            "n_dev": len(dev_rows), "n_holdout": len(hol_rows)}


def main():
    print("RBI_RUNS_CLEAN_BASELINE_AND_GATE_A\n===================================")
    dev_batters = load_batters(DEV_SEASON)
    hol_batters = load_batters(HOLDOUT_SEASON)
    print(f"raw batter rows: dev={len(dev_batters)} holdout={len(hol_batters)}")

    report = {"script": "RBI_RUNS_CLEAN_BASELINE_AND_GATE_A", "markets": {}}
    for mkt_key, cfg in MARKETS.items():
        dev_rows = build_rows(dev_batters, cfg["target"])
        hol_rows = build_rows(hol_batters, cfg["target"])
        report["markets"][mkt_key] = run_market(mkt_key, cfg, dev_rows, hol_rows, WORKDIR)

    WORKDIR.mkdir(parents=True, exist_ok=True)
    (WORKDIR / "rbi_runs_gate_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n\n{'#'*70}\nSUMMARY\n{'#'*70}")
    for mkt, r in report["markets"].items():
        print(f"  {mkt:16s} AUC={r['challenger']['auc']:.4f}  {r['verdict']}")
    print(f"\nreport: {WORKDIR / 'rbi_runs_gate_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
