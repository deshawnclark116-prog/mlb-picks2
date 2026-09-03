#!/usr/bin/env python3
"""
CFB_PRIOR_SEASON_EARLY_GATE_A

The live CFB pipeline (cfb_serving_builder_a.py) requires 3+ CURRENT-season
games before it will score a player -- weeks 1-3 of every season write an
empty board. Exact same situation NFL's rushing_yards market was in before
nfl_preseason_to_regular_season_gate_a.py tested whether prior real games
(there: same-year preseason; here: last season's full-season stats, since
CFB has no separate preseason slate) give the model anything for those
early weeks. That test passed for NFL rushing_yards (AUC 0.7027) and
honestly failed for NFL receiving_yards (AUC 0.5676, did not clear gate) --
same two-arm design and same pre-registered bar, replicated here across
all four CFB markets. An honest fail is an expected, useful outcome, not
something to route around.

Data
----
Local /data/cfb_model/cfb_model.sqlite equivalent is stale (pre-dates the
passing columns); using cfb_models/cfb_model.sqlite instead, which has all
four markets' stat columns for 2022-2025 (same cfbfastR source, same
player_id namespace within this file -- unlike the live ESPN 2026 feed,
there is NO cross-source ID mismatch here, so prior-season matching is a
direct player_id join, not name-matching).

Design
------
For each market, weeks 1-3 of season S (S in {2023, 2024, 2025}) are the
scored population; PRIOR_SEASON = S-1 supplies the bootstrap features for
players who appear in it. dev = weeks 1-3 of 2023+2024 (prior seasons
2022+2023), holdout = weeks 1-3 of 2025 (prior season 2024) -- untouched
until final scoring, same rule as every other gate in this repo.

Two arms, scored on holdout:
  constant     train-set base rate (== today's live behavior: an empty
               board, equivalent prediction-value to guessing the rate)
  challenger   XGBoost on prior_season_avg_<stat>, prior_season_games,
               prior_season_avg_rate (carries/receptions/attempts per
               game, matching each market's existing "recent volume"
               gate). Players with no prior-season row (true freshmen,
               transfers into FBS, etc.) get NaN features -- XGBoost
               falls back toward the base rate for them, which is honest:
               no fabricated signal for someone with no real history.

Eligibility: position-gated same as the live per-market baseline
(RB/WR/QB), but NO current-season game requirement -- that's the whole
point, this is specifically the population the live pipeline leaves at 0.

Pre-registered pass (written before this script has ever been run,
identical bar to the NFL version): AUC >= 0.58, logloss beats constant by
>= 0.01, Brier beats constant. Markets that don't clear it don't ship,
same as NFL receiving_yards didn't.

Read-only on cfb_models/cfb_model.sqlite. Writes only its own workdir.

Run
---
python -u cfb_prior_season_early_gate_a.py
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

DB_DEFAULT = "cfb_models/cfb_model.sqlite"
WORKDIR = Path("/data/cfb_model/cfb_prior_season_early_gate_a_work")
MAX_WEEK = 3
LINES = {
    "rushing_yards": 69.5,
    "receiving_yards": 59.5,
    "passing_yards": 214.5,
    "passing_touchdowns": 1.5,
}
DEV_SEASONS = [2023, 2024]
HOLDOUT_SEASON = 2025
NAN = float("nan")

MARKETS = {
    "rushing_yards": {"position": "RB", "stat": "rushing_yards", "rate": "carries"},
    "receiving_yards": {"position": "WR", "stat": "receiving_yards", "rate": "receptions"},
    "passing_yards": {"position": "QB", "stat": "passing_yards", "rate": "pass_attempts"},
    "passing_touchdowns": {"position": "QB", "stat": "passing_touchdowns", "rate": "pass_attempts"},
}

FEATURE_COLS = ["prior_season_avg_stat", "prior_season_games", "prior_season_avg_rate"]


def load_weeks123(conn, season, position):
    rows = conn.execute("""
        SELECT player_id, player_name, week, carries, rushing_yards,
               receptions, receiving_yards, pass_attempts, passing_yards,
               passing_touchdowns
        FROM player_games
        WHERE season = ? AND week <= ? AND position = ?
    """, (season, MAX_WEEK, position)).fetchall()
    return rows


def load_full_season_by_player(conn, season, position):
    """player_id -> list of per-game stat dicts for the WHOLE prior season
    (all weeks) -- this is real last-season production, not a subset."""
    rows = conn.execute("""
        SELECT player_id, carries, rushing_yards, receptions, receiving_yards,
               pass_attempts, passing_yards, passing_touchdowns
        FROM player_games
        WHERE season = ? AND position = ?
    """, (season, position)).fetchall()
    by_pid = {}
    for r in rows:
        pid = r[0]
        by_pid.setdefault(pid, []).append({
            "carries": r[1] or 0, "rushing_yards": r[2] or 0,
            "receptions": r[3] or 0, "receiving_yards": r[4] or 0,
            "pass_attempts": r[5] or 0, "passing_yards": r[6] or 0,
            "passing_touchdowns": r[7] or 0,
        })
    return by_pid


def prior_season_features(pid, prior_by_pid, cfg):
    games = prior_by_pid.get(pid)
    if not games:
        return None
    stat_field, rate_field = cfg["stat"], cfg["rate"]
    stat_vals = [g[stat_field] for g in games]
    rate_vals = [g[rate_field] for g in games]
    n = len(games)
    return {
        "prior_season_avg_stat": sum(stat_vals) / n,
        "prior_season_games": n,
        "prior_season_avg_rate": sum(rate_vals) / n,
    }


def build_rows(cur_rows, prior_by_pid, cfg, line):
    stat_field = cfg["stat"]
    idx = {"carries": 3, "rushing_yards": 4, "receptions": 5, "receiving_yards": 6,
           "pass_attempts": 7, "passing_yards": 8, "passing_touchdowns": 9}
    out = []
    for r in cur_rows:
        pid, pname, week = r[0], r[1], r[2]
        actual = r[idx[stat_field]]
        actual = actual if actual is not None else 0
        feat = prior_season_features(pid, prior_by_pid, cfg)
        out.append({
            "player_id": pid, "player_name": pname, "week": week,
            "prior_season_avg_stat": feat["prior_season_avg_stat"] if feat else NAN,
            "prior_season_games": feat["prior_season_games"] if feat else NAN,
            "prior_season_avg_rate": feat["prior_season_avg_rate"] if feat else NAN,
            "actual": actual,
            "over_line": 1 if actual >= (line + 0.5) else 0,
            "matched": feat is not None,
        })
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
    ece = 0.0
    for b in range(10):
        m = (p >= b / 10) & (p < (b + 1) / 10) if b < 9 else (p >= 0.9)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        ece += abs(float(p[m].mean()) - float(y[m].mean())) * cnt / n
    return {"n": n, "base_rate": round(float(y.mean()), 4), "auc": round(auc(probs, labels), 4),
            "log_loss": round(ll, 5), "brier": round(brier, 5), "ece": round(ece, 4)}


def mat(rows, xgb):
    X = np.array([[r.get(c, NAN) for c in FEATURE_COLS] for r in rows], dtype=np.float32)
    y = np.array([r["over_line"] for r in rows], dtype=np.float32)
    return xgb.DMatrix(X, label=y, feature_names=FEATURE_COLS)


def run_market(mkt_key, cfg, conn):
    import xgboost as xgb
    line = LINES[mkt_key]
    print(f"\n{'='*70}\n{mkt_key} (weeks 1-{MAX_WEEK} only, line={line})\n{'='*70}")

    dev_rows = []
    for season in DEV_SEASONS:
        cur = load_weeks123(conn, season, cfg["position"])
        prior = load_full_season_by_player(conn, season - 1, cfg["position"])
        dev_rows += build_rows(cur, prior, cfg, line)
    cur_hol = load_weeks123(conn, HOLDOUT_SEASON, cfg["position"])
    prior_hol = load_full_season_by_player(conn, HOLDOUT_SEASON - 1, cfg["position"])
    hol_rows = build_rows(cur_hol, prior_hol, cfg, line)

    matched_dev = sum(1 for r in dev_rows if r["matched"])
    matched_hol = sum(1 for r in hol_rows if r["matched"])
    print(f"dev {DEV_SEASONS}: {len(dev_rows)} rows ({matched_dev} matched to a prior season, "
          f"{matched_dev/len(dev_rows)*100:.1f}%)")
    print(f"holdout {HOLDOUT_SEASON}: {len(hol_rows)} rows ({matched_hol} matched, "
          f"{matched_hol/len(hol_rows)*100:.1f}%)")

    n = len(dev_rows)
    cut = int(n * 0.8)
    tr, va = dev_rows[:cut], dev_rows[cut:]
    print(f"  train={len(tr)}  internal val={len(va)}")

    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3,
              "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 5, "seed": 13}
    bst = xgb.train(params, mat(tr, xgb), num_boost_round=800, evals=[(mat(va, xgb), "val")],
                     early_stopping_rounds=40, verbose_eval=False)
    itr = (0, bst.best_iteration + 1)
    probs_hol = bst.predict(mat(hol_rows, xgb), iteration_range=itr)
    labels_hol = [r["over_line"] for r in hol_rows]
    challenger = metrics(list(map(float, probs_hol)), labels_hol)

    train_rate = float(np.mean([r["over_line"] for r in tr]))
    constant = metrics([train_rate] * len(hol_rows), labels_hol)

    print(f"\n  {'arm':12s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'constant':12s} {'n/a':>7s} {constant['log_loss']:>9.5f}  {constant['brier']:>7.5f} {constant['ece']:>7.4f}")
    print(f"  {'challenger':12s} {challenger['auc']:>7.4f}  {challenger['log_loss']:>9.5f}  {challenger['brier']:>7.5f} {challenger['ece']:>7.4f}")

    imp = bst.get_score(importance_type="gain")
    print("\n  feature importance (gain):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"    {k:24s} {v:9.2f}")

    d_ll = constant["log_loss"] - challenger["log_loss"]
    c1 = challenger["auc"] >= 0.58
    c2 = d_ll >= 0.01
    c3 = challenger["brier"] < constant["brier"]
    passed = c1 and c2 and c3
    verdict = f"{mkt_key.upper()}_{'PASSES_GATE' if passed else 'DOES_NOT_CLEAR_GATE'}"
    print(f"\n  GATE: AUC>=0.58 -> {c1}   logloss gain>=0.01 -> {c2}   Brier better -> {c3}")
    print(f"  VERDICT: {verdict}")

    return {
        "n_dev": len(dev_rows), "n_holdout": len(hol_rows),
        "matched_dev_pct": round(matched_dev / len(dev_rows) * 100, 1),
        "matched_holdout_pct": round(matched_hol / len(hol_rows) * 100, 1),
        "constant": constant, "challenger": challenger,
        "importance": imp, "passed": passed, "verdict": verdict,
    }


def main():
    print("CFB_PRIOR_SEASON_EARLY_GATE_A\n==============================")
    WORKDIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{DB_DEFAULT}?mode=ro", uri=True)

    report = {"script": "CFB_PRIOR_SEASON_EARLY_GATE_A", "markets": {}}
    for mkt_key, cfg in MARKETS.items():
        report["markets"][mkt_key] = run_market(mkt_key, cfg, conn)
    conn.close()

    (WORKDIR / "cfb_prior_season_early_gate_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n\n{'#'*70}\nSUMMARY\n{'#'*70}")
    for mkt, r in report["markets"].items():
        print(f"  {mkt:20s} AUC={r['challenger']['auc']:.4f}  matched(holdout)={r['matched_holdout_pct']}%  {r['verdict']}")
    print(f"\nreport: {WORKDIR / 'cfb_prior_season_early_gate_a_report.json'}")

    any_pass = any(r["passed"] for r in report["markets"].values())
    return 0 if any_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
