#!/usr/bin/env python3
"""
NFL_PRIOR_SEASON_ANYTIME_TOUCHDOWNS_GATE_A

Same design as cfb_prior_season_anytime_touchdowns_gate_a.py (last
season's real full-season production bootstraps weeks 1-3), adapted for
NFL: nfl_anytime_touchdowns_champion_gate_a.py's in-season model
correctly requires 3 real current-season games before trusting a pick,
so weeks 1-3 of every season show zero anytime-TD picks -- but almost
every returning RB/WR has a real, complete prior season on record.

Population/eligibility mirrors the in-season market exactly: RB with
prior-season avg carries >= 12, OR WR with prior-season avg receptions
>= 5 -- each position's own already-validated volume floor, not an
invented combined-touches number.

DEV=2021-2024 (prior seasons 2020-2023), HOLDOUT=2025 (prior season
2024) -- matches nfl_anytime_touchdowns_champion_gate_a.py's own
HOLDOUT_SEASON=2025 so both variants are validated on the same
real, most-recent unseen season.

Two arms, scored on holdout:
  constant     train-set base rate (== today's live behavior: an empty
               board)
  challenger   XGBoost on prior_season_avg_stat (total TDs per game last
               season), prior_season_games, prior_season_avg_rate
               (carries/game for RB, receptions/game for WR, last
               season) -- identical feature shape to CFB's prior-season
               anytime-TD gate.

Pre-registered pass (written before this script has ever been run):
AUC >= 0.58, logloss beats constant by >= 0.01, Brier beats constant.

Read-only. Writes only its own workdir.

Run
---
python -u nfl_prior_season_anytime_touchdowns_gate_a.py --db /path/to/nfl_model.sqlite
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

DB_DEFAULT = "nfl_models/nfl_model.sqlite"
WORKDIR = Path("nfl_models/nfl_prior_season_anytime_touchdowns_gate_a_work")
MAX_WEEK = 3
LINE = 0.5
DEV_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUT_SEASON = 2025
NAN = float("nan")

MIN_RECENT_CARRIES_RB = 12
MIN_RECENT_RECEPTIONS_WR = 5
FEATURE_COLS = ["prior_season_avg_stat", "prior_season_games", "prior_season_avg_rate"]


def load_weeks123(conn, season):
    return conn.execute("""
        SELECT player_id, player_name, position, week, carries, receptions,
               rushing_tds, receiving_tds
        FROM player_games
        WHERE season = ? AND week <= ? AND position IN ('RB', 'WR')
    """, (season, MAX_WEEK)).fetchall()


def load_full_season_by_player(conn, season):
    rows = conn.execute("""
        SELECT player_id, position, carries, receptions, rushing_tds, receiving_tds
        FROM player_games WHERE season = ? AND position IN ('RB', 'WR')
    """, (season,)).fetchall()
    by_pid = {}
    for pid, pos, carries, receptions, rtd, rectd in rows:
        by_pid.setdefault(pid, {"position": pos, "games": []})
        by_pid[pid]["games"].append({
            "carries": carries or 0, "receptions": receptions or 0,
            "total_td": (rtd or 0) + (rectd or 0),
        })
    return by_pid


def prior_season_features(pid, prior_by_pid):
    # No volume floor here, deliberately -- matches CFB's prior-season
    # anytime-TD gate's own reasoning: a serving-time volume floor
    # belongs in the production builder, not training/validation.
    entry = prior_by_pid.get(pid)
    if not entry:
        return None
    pos = entry["position"]
    games = entry["games"]
    n = len(games)
    rate_field = "carries" if pos == "RB" else "receptions"
    avg_rate = sum(g[rate_field] for g in games) / n
    return {
        "prior_season_avg_stat": sum(g["total_td"] for g in games) / n,
        "prior_season_games": n,
        "prior_season_avg_rate": avg_rate,
    }


def build_rows(cur_rows, prior_by_pid):
    out = []
    for pid, pname, pos, week, carries, receptions, rtd, rectd in cur_rows:
        actual = (rtd or 0) + (rectd or 0)
        feat = prior_season_features(pid, prior_by_pid)
        out.append({
            "player_id": pid, "player_name": pname, "week": week,
            "prior_season_avg_stat": feat["prior_season_avg_stat"] if feat else NAN,
            "prior_season_games": feat["prior_season_games"] if feat else NAN,
            "prior_season_avg_rate": feat["prior_season_avg_rate"] if feat else NAN,
            "actual": actual,
            "over_line": 1 if actual >= (LINE + 0.5) else 0,
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


def main():
    import xgboost as xgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_DEFAULT)
    args = ap.parse_args()

    print("NFL_PRIOR_SEASON_ANYTIME_TOUCHDOWNS_GATE_A\n============================================")
    WORKDIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    dev_rows = []
    for season in DEV_SEASONS:
        cur = load_weeks123(conn, season)
        prior = load_full_season_by_player(conn, season - 1)
        dev_rows += build_rows(cur, prior)
    cur_hol = load_weeks123(conn, HOLDOUT_SEASON)
    prior_hol = load_full_season_by_player(conn, HOLDOUT_SEASON - 1)
    hol_rows = build_rows(cur_hol, prior_hol)
    conn.close()

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
    verdict = f"NFL_PRIOR_SEASON_ANYTIME_TOUCHDOWNS_{'PASSES_GATE' if passed else 'DOES_NOT_CLEAR_GATE'}"
    print(f"\n  GATE: AUC>=0.58 -> {c1}   logloss gain>=0.01 -> {c2}   Brier better -> {c3}")
    print(f"  VERDICT: {verdict}")

    bst.save_model(str(WORKDIR / "nfl_prior_season_anytime_touchdowns.json"))
    (WORKDIR / "nfl_prior_season_anytime_touchdowns_columns.json").write_text(json.dumps(FEATURE_COLS))
    report = {
        "script": "NFL_PRIOR_SEASON_ANYTIME_TOUCHDOWNS_GATE_A",
        "dev_seasons": DEV_SEASONS, "holdout_season": HOLDOUT_SEASON,
        "n_dev": len(dev_rows), "n_holdout": len(hol_rows),
        "matched_dev_pct": round(matched_dev / len(dev_rows) * 100, 1),
        "matched_holdout_pct": round(matched_hol / len(hol_rows) * 100, 1),
        "constant": constant, "challenger": challenger,
        "importance": imp, "passed": passed, "verdict": verdict,
    }
    (WORKDIR / "nfl_prior_season_anytime_touchdowns_gate_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport + model: {WORKDIR}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
