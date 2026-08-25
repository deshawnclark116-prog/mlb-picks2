#!/usr/bin/env python3
"""
NFL_PRESEASON_TO_REGULAR_SEASON_GATE_A

New hypothesis, distinct from the already-failed one (preseason predicting
next-preseason performance, see nfl_preseason_champion_gate_a.py -- AUC
0.41/0.49, real dead end, not revisited here). This tests whether a
player's REAL PRESEASON usage that same year gives the regular-season
model anything useful specifically for weeks 1-3, which the live pipeline
(nfl_serving_builder_a.py) deliberately leaves EMPTY right now because it
requires 3+ prior REGULAR-SEASON games before it'll speak -- there is no
in-season history yet at week 1. Preseason already happened by then, so
using it isn't a data availability problem, only a question of whether it
actually helps.

Data
----
Regular season weeks 1-3, REG only, 2022-2025: pulled directly from
nflverse's stats_player_week_{year}.csv + games.csv (same real source
nfl_player_games_foundation_a.py uses) -- this sandbox's proxy blocks the
api.github.com call that script's asset-resolution step needs (not an
Akamai/UA issue like the ESPN case, a different block: this session's
GitHub scope doesn't cover the nflverse repo), so the CSVs were fetched
directly via their known download URLs and read locally instead of
running that script end-to-end.

Preseason: local nfl_preseason.sqlite, already pulled and position-tagged
earlier (2021-2026, ESPN box scores).

Player identity: nflverse uses gsis_id, ESPN preseason data uses ESPN
athlete_id -- no shared key between the two sources. Joined by normalized
full name + position + season instead (strip accents/punctuation/suffixes,
lowercase). Real, disclosed approximation -- will miss a player whose
name is formatted very differently between the two sources, but won't
silently mismatch two different people at a meaningfully high rate for
common name formats.

Design
------
Two arms, scored on regular season weeks 1-3 only:
  constant     train-set base rate for every row (what the live pipeline
               effectively gives you today -- nothing, an empty board,
               which is equivalent prediction-value to guessing the base
               rate)
  challenger   XGBoost on that same year's preseason performance:
               preseason_avg_yards, preseason_games_played,
               preseason_avg_rate (carries or targets per game).
               Players with no preseason match get NaN features
               (XGBoost handles missing values; effectively falls back
               toward the base rate for that player, which is honest --
               no fabricated signal for a player who didn't play
               preseason).

Split: dev = 2022-2024 regular season weeks 1-3, holdout = 2025 regular
season weeks 1-3 (untouched).

Pre-registered pass: AUC >= 0.58, logloss beats constant by >= 0.01,
Brier beats constant -- same bar as every other gate in this repo.
An honest FAIL (preseason doesn't help early regular season either) is
an expected, useful possible outcome.

Read-only on local CSVs and nfl_preseason.sqlite. Writes only its own
workdir.

Run:
    python -u nfl_preseason_to_regular_season_gate_a.py
"""
import csv
import json
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

CSV_DIR = Path("/data/nflverse_csv")
PRESEASON_DB = Path("nfl_models/nfl_preseason.sqlite")
WORKDIR = Path("/data/nfl_preseason_to_regular_season_gate_a_work")

DEV_SEASONS = [2022, 2023, 2024]
HOLDOUT_SEASON = 2025
LINE = 49.5  # matches the live regular-season model's line for both markets
NAN = float("nan")

MARKETS = {
    "rushing_yards": {"position": "RB", "stat": "rushing_yards", "rate": "carries"},
    "receiving_yards": {"position": "WR", "stat": "receiving_yards", "rate": "targets"},
}


def norm_name(name):
    name = unicodedata.normalize("NFKD", name or "")
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", name)
    name = re.sub(r"[^a-z ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _f(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def load_regular_weeks123(seasons):
    seasons = set(seasons)
    rows = []
    for season in sorted(seasons):
        path = CSV_DIR / f"stats_player_week_{season}.csv"
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("season_type") != "REG":
                    continue
                try:
                    week = int(r["week"])
                except Exception:
                    continue
                if week > 3:
                    continue
                rows.append(r)
    return rows


def load_preseason_by_key(seasons):
    con = sqlite3.connect(f"file:{PRESEASON_DB}?mode=ro", uri=True)
    seasons_sql = ",".join(str(s) for s in seasons)
    rows = con.execute(f"""
        SELECT player_name, position, season, carries, rushing_yards, targets, receiving_yards
        FROM player_games WHERE season IN ({seasons_sql}) AND position IS NOT NULL
    """).fetchall()
    con.close()
    by_key = {}
    for name, pos, season, carries, ry, targets, rey in rows:
        key = (norm_name(name), pos, season)
        by_key.setdefault(key, []).append({
            "carries": carries or 0, "rushing_yards": ry or 0,
            "targets": targets or 0, "receiving_yards": rey or 0,
        })
    return by_key


def preseason_features(key, by_key, mkt_key, cfg):
    games = by_key.get(key)
    if not games:
        return None
    stat_field = cfg["stat"]
    rate_field = cfg["rate"]
    yards = [g[stat_field] for g in games]
    rate = [g[rate_field] for g in games]
    n = len(games)
    return {
        "preseason_avg_yards": sum(yards) / n,
        "preseason_games_played": n,
        "preseason_avg_rate": sum(rate) / n,
    }


FEATURE_COLS = ["preseason_avg_yards", "preseason_games_played", "preseason_avg_rate"]


def build_rows(reg_rows, preseason_by_key, mkt_key, cfg):
    out = []
    for r in reg_rows:
        if r.get("position") != cfg["position"]:
            continue
        try:
            season = int(r["season"])
        except Exception:
            continue
        name = r.get("player_display_name") or r.get("player_name")
        key = (norm_name(name), cfg["position"], season)
        feat = preseason_features(key, preseason_by_key, mkt_key, cfg)
        actual = _f(r.get(cfg["stat"]))
        row = {
            "season": season, "name": name,
            "preseason_avg_yards": feat["preseason_avg_yards"] if feat else NAN,
            "preseason_games_played": feat["preseason_games_played"] if feat else NAN,
            "preseason_avg_rate": feat["preseason_avg_rate"] if feat else NAN,
            "actual": actual,
            "over_line": 1 if actual >= (LINE + 0.5) else 0,
            "matched": feat is not None,
        }
        out.append(row)
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


def run_market(mkt_key, cfg, reg_dev, reg_hol, preseason_dev, preseason_hol, workdir):
    import xgboost as xgb
    print(f"\n{'='*70}\n{mkt_key} (weeks 1-3 only)\n{'='*70}")

    dev_rows = build_rows(reg_dev, preseason_dev, mkt_key, cfg)
    hol_rows = build_rows(reg_hol, preseason_hol, mkt_key, cfg)
    matched_dev = sum(1 for r in dev_rows if r["matched"])
    matched_hol = sum(1 for r in hol_rows if r["matched"])
    print(f"dev {DEV_SEASONS}: {len(dev_rows)} rows ({matched_dev} matched to a preseason game, "
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
    print("NFL_PRESEASON_TO_REGULAR_SEASON_GATE_A\n=======================================")
    WORKDIR.mkdir(parents=True, exist_ok=True)

    print("loading regular season weeks 1-3 (real nflverse data)...")
    reg_dev = load_regular_weeks123(DEV_SEASONS)
    reg_hol = load_regular_weeks123([HOLDOUT_SEASON])
    print(f"  dev rows: {len(reg_dev)}   holdout rows: {len(reg_hol)}")

    print("loading local preseason data...")
    preseason_dev = load_preseason_by_key(DEV_SEASONS)
    preseason_hol = load_preseason_by_key([HOLDOUT_SEASON])
    print(f"  dev preseason player-season keys: {len(preseason_dev)}")
    print(f"  holdout preseason player-season keys: {len(preseason_hol)}")

    report = {"script": "NFL_PRESEASON_TO_REGULAR_SEASON_GATE_A", "markets": {}}
    for mkt_key, cfg in MARKETS.items():
        report["markets"][mkt_key] = run_market(
            mkt_key, cfg, reg_dev, reg_hol, preseason_dev, preseason_hol, WORKDIR)

    (WORKDIR / "nfl_preseason_to_regular_season_gate_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n\n{'#'*70}\nSUMMARY\n{'#'*70}")
    for mkt, r in report["markets"].items():
        print(f"  {mkt:18s} AUC={r['challenger']['auc']:.4f}  matched(holdout)={r['matched_holdout_pct']}%  {r['verdict']}")
    print(f"\nreport: {WORKDIR / 'nfl_preseason_to_regular_season_gate_a_report.json'}")

    any_pass = any(r["passed"] for r in report["markets"].values())
    return 0 if any_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
