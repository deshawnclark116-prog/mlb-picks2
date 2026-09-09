#!/usr/bin/env python3
"""
NFL_PRIOR_SEASON_EARLY_GATE_A

A third, distinct hypothesis for weeks 1-3 (which the live regular-season
model leaves empty -- it needs 3+ CURRENT-season games). Already tested
and resolved:
  1. preseason predicting NEXT preseason (nfl_preseason_champion_gate_a.py)
     -- dead end, AUC 0.41/0.49.
  2. SAME-YEAR preseason predicting weeks 1-3 of the regular season
     (nfl_preseason_to_regular_season_gate_a.py) -- passed for
     rushing_yards (AUC 0.7027), shipped as rushing_yards_early_season.
     Real gap found in production (2026-09-09, Week 1): real veteran
     starters (James Conner, Jahmyr Gibbs, Bijan Robinson, Derrick Henry,
     etc.) are rested for the ENTIRE preseason -- normal practice for
     established players -- so they have zero preseason rows and never
     appear in that model's population at all. The board ends up
     entirely backups/camp bodies who logged the preseason snaps, which
     is exactly backwards from what a bettor wants to see.

This one: does last season's REAL, FULL regular-season performance
predict weeks 1-3 of the following regular season? Exactly the CFB
pattern (cfb_prior_season_early_gate_a.py) applied to NFL -- unlike
hypothesis #2 above, no cross-source name-matching is needed here: both
"current weeks 1-3" and "prior full season" come from the same nflverse
stats_player_week_{year}.csv source (this sandbox's proxy blocks the
api.github.com asset-resolution call nfl_player_games_foundation_a.py
needs, so these were fetched directly via their known download URLs and
cached at /data/nflverse_csv/ -- same real files that script would
otherwise produce), so it's a direct gsis player_id join, no fuzzy
matching, no missed-format risk.

This directly targets the production gap: an established veteran with
real 2025 tape (Conner, Gibbs, Henry, etc.) but zero 2026 preseason
snaps gets a real, informed pick here even though hypothesis #2's model
has nothing for him.

Split: dev = 2023+2024 regular season weeks 1-3 (priors 2022+2023),
holdout = 2025 regular season weeks 1-3 (prior 2024) -- untouched until
final scoring, same rule as every other gate in this repo.

Two arms, scored on holdout:
  constant     train-set base rate (== today's live behavior for a
               player this model would otherwise skip: an empty board)
  challenger   XGBoost on prior_season_avg_yards, prior_season_games_
               played, prior_season_avg_rate (carries or targets per
               game -- matches the position's existing early-season
               feature set exactly). Players with no prior-season row
               (rookies, first-year-in-league) get NaN features --
               XGBoost falls back toward the base rate for them, honest:
               no fabricated signal for someone with no real history.

Eligibility: RB (rushing_yards) / WR (receiving_yards) weeks 1-3 of the
target season, REG only -- no current-season game requirement, that's
the whole point.

Pre-registered pass (written before this script has ever been run,
same bar as every other gate in this repo): AUC >= 0.58, logloss beats
constant by >= 0.01, Brier beats constant. A market that doesn't clear
it doesn't ship.

Read-only on local nflverse CSVs. Writes only its own workdir.

Run
---
python -u nfl_prior_season_early_gate_a.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

CSV_DIR = Path("/data/nflverse_csv")
WORKDIR = Path("/data/nfl_prior_season_early_gate_a_work")

MAX_WEEK = 3
LINE = 49.5  # matches rushing_yards_early_season's live line
DEV_SEASONS = [2023, 2024]
HOLDOUT_SEASON = 2025
NAN = float("nan")

MARKETS = {
    "rushing_yards": {"position": "RB", "stat": "rushing_yards", "rate": "carries"},
    "receiving_yards": {"position": "WR", "stat": "receiving_yards", "rate": "targets"},
}

FEATURE_COLS = ["prior_season_avg_yards", "prior_season_games_played", "prior_season_avg_rate"]


def _f(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def load_season_csv(season):
    path = CSV_DIR / f"stats_player_week_{season}.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_weeks123(rows, position):
    out = []
    for r in rows:
        if r.get("season_type") != "REG" or r.get("position") != position:
            continue
        try:
            week = int(r["week"])
        except Exception:
            continue
        if week > MAX_WEEK:
            continue
        out.append(r)
    return out


def load_full_season_by_player(rows, position):
    """player_id -> list of per-game stat dicts for the WHOLE prior
    season (every week, REG only) -- real last-season production."""
    by_pid = {}
    for r in rows:
        if r.get("season_type") != "REG" or r.get("position") != position:
            continue
        pid = r.get("player_id")
        if not pid:
            continue
        by_pid.setdefault(pid, []).append({
            "carries": _f(r.get("carries")), "rushing_yards": _f(r.get("rushing_yards")),
            "targets": _f(r.get("targets")), "receiving_yards": _f(r.get("receiving_yards")),
        })
    return by_pid


def prior_season_features(pid, prior_by_pid, cfg):
    games = prior_by_pid.get(pid)
    if not games:
        return None
    stat_field, rate_field = cfg["stat"], cfg["rate"]
    n = len(games)
    return {
        "prior_season_avg_yards": sum(g[stat_field] for g in games) / n,
        "prior_season_games_played": n,
        "prior_season_avg_rate": sum(g[rate_field] for g in games) / n,
    }


def build_rows(cur_rows, prior_by_pid, cfg):
    out = []
    for r in cur_rows:
        pid = r.get("player_id")
        actual = _f(r.get(cfg["stat"]))
        feat = prior_season_features(pid, prior_by_pid, cfg)
        out.append({
            "player_id": pid, "name": r.get("player_display_name") or r.get("player_name"),
            "prior_season_avg_yards": feat["prior_season_avg_yards"] if feat else NAN,
            "prior_season_games_played": feat["prior_season_games_played"] if feat else NAN,
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


def run_market(mkt_key, cfg, seasons_raw):
    import xgboost as xgb
    print(f"\n{'='*70}\n{mkt_key} (weeks 1-{MAX_WEEK} only, line={LINE})\n{'='*70}")

    dev_rows = []
    for season in DEV_SEASONS:
        cur = load_weeks123(seasons_raw[season], cfg["position"])
        prior = load_full_season_by_player(seasons_raw[season - 1], cfg["position"])
        dev_rows += build_rows(cur, prior, cfg)
    cur_hol = load_weeks123(seasons_raw[HOLDOUT_SEASON], cfg["position"])
    prior_hol = load_full_season_by_player(seasons_raw[HOLDOUT_SEASON - 1], cfg["position"])
    hol_rows = build_rows(cur_hol, prior_hol, cfg)

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

    bst.save_model(str(WORKDIR / f"nfl_prior_season_{mkt_key}.json"))
    (WORKDIR / f"nfl_prior_season_{mkt_key}_columns.json").write_text(json.dumps(FEATURE_COLS))

    return {
        "n_dev": len(dev_rows), "n_holdout": len(hol_rows),
        "matched_dev_pct": round(matched_dev / len(dev_rows) * 100, 1),
        "matched_holdout_pct": round(matched_hol / len(hol_rows) * 100, 1),
        "constant": constant, "challenger": challenger,
        "importance": imp, "passed": passed, "verdict": verdict,
    }


def main():
    print("NFL_PRIOR_SEASON_EARLY_GATE_A\n==============================")
    WORKDIR.mkdir(parents=True, exist_ok=True)

    needed_seasons = sorted({s for s in DEV_SEASONS + [HOLDOUT_SEASON]} |
                             {s - 1 for s in DEV_SEASONS + [HOLDOUT_SEASON]})
    print(f"loading real nflverse weekly stats for seasons {needed_seasons}...")
    seasons_raw = {s: load_season_csv(s) for s in needed_seasons}
    for s in needed_seasons:
        print(f"  {s}: {len(seasons_raw[s])} rows")

    report = {"script": "NFL_PRIOR_SEASON_EARLY_GATE_A", "markets": {}}
    for mkt_key, cfg in MARKETS.items():
        report["markets"][mkt_key] = run_market(mkt_key, cfg, seasons_raw)

    (WORKDIR / "nfl_prior_season_early_gate_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n\n{'#'*70}\nSUMMARY\n{'#'*70}")
    for mkt, r in report["markets"].items():
        print(f"  {mkt:18s} AUC={r['challenger']['auc']:.4f}  matched(holdout)={r['matched_holdout_pct']}%  {r['verdict']}")
    print(f"\nreport: {WORKDIR / 'nfl_prior_season_early_gate_a_report.json'}")

    any_pass = any(r["passed"] for r in report["markets"].values())
    return 0 if any_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
