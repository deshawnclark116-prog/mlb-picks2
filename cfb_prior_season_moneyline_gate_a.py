#!/usr/bin/env python3
"""
CFB_PRIOR_SEASON_MONEYLINE_GATE_A

Same design as cfb_prior_season_anytime_touchdowns_gate_a.py (last
season's real full-season production bootstraps weeks 1-3), adapted for
moneyline: instead of leaving weeks 1-3 with zero picks until a team has
3 real current-season games, feed each team's PRIOR season's real,
complete final record (win rate, net scoring margin, points for/against)
-- exactly the real history a bettor already has in week 1 of a new
season, whether or not the model does.

Population: every real FBS game in weeks 1-3, symmetric two-rows-per-
game design (matches cfb_moneyline_clean_baseline_a.py exactly) -- both
teams' PRIOR season stats, whatever they are; no volume/games floor here
by design, same reasoning cfb_prior_season_anytime_touchdowns_gate_a.py
documents (a serving-time floor belongs in the production builder, not
training/validation).

DEV=2019-2023 (prior seasons 2018-2022), HOLDOUT=2024 (prior season
2023) -- same seasons as the anytime_touchdowns prior-season gate, for
consistency; final scores have no 2025 completeness issue the way
touchdown attribution does, so this isn't excluding 2025 for a data
reason, just matching the rest of this repo's convention.

Pre-registered pass (written before this script has ever been run):
AUC >= 0.58, logloss beats constant by >= 0.01, Brier beats constant.

Read-only. Writes only its own workdir.

Run
---
python -u cfb_prior_season_moneyline_gate_a.py
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
WORKDIR = Path("cfb_models/cfb_prior_season_moneyline_gate_a_work")
MAX_WEEK = 3
DEV_SEASONS = [2019, 2020, 2021, 2022, 2023]
HOLDOUT_SEASON = 2024
NAN = float("nan")

FEATURE_COLS = [
    "prior_net_margin", "prior_win_rate", "prior_avg_points_for", "prior_avg_points_against",
    "prior_games", "opp_prior_net_margin", "opp_prior_win_rate", "opp_prior_avg_points_for",
    "opp_prior_avg_points_against", "opp_prior_games", "prior_projected_margin",
    "is_home", "is_neutral_site",
]


def load_weeks123(conn, season):
    return conn.execute("""
        SELECT game_id, week, home_team, away_team, home_points, away_points, neutral_site
        FROM games
        WHERE season = ? AND week <= ? AND home_points IS NOT NULL AND away_points IS NOT NULL
    """, (season, MAX_WEEK)).fetchall()


def load_full_season_team_stats(conn, season):
    """Full real final season stats per team -- unlike the in-season
    engine's asof tracker, this is a completed season, so it's just one
    pass: team -> {games, win_rate, net_margin, avg_points_for,
    avg_points_against}."""
    games = conn.execute("""
        SELECT home_team, away_team, home_points, away_points
        FROM games WHERE season = ? AND home_points IS NOT NULL AND away_points IS NOT NULL
    """, (season,)).fetchall()
    state = {}  # team -> [wins, points_for, points_against, games]
    for home, away, hp, ap in games:
        hst = state.setdefault(home, [0, 0, 0, 0])
        hst[0] += 1 if hp > ap else 0
        hst[1] += hp; hst[2] += ap; hst[3] += 1
        ast = state.setdefault(away, [0, 0, 0, 0])
        ast[0] += 1 if ap > hp else 0
        ast[1] += ap; ast[2] += hp; ast[3] += 1
    out = {}
    for team, (wins, pf, pa, n) in state.items():
        if n == 0:
            continue
        out[team] = {"games": n, "win_rate": wins / n, "net_margin": (pf - pa) / n,
                     "avg_points_for": pf / n, "avg_points_against": pa / n}
    return out


def team_features(team, opp, prior_stats, is_home, is_neutral):
    own = prior_stats.get(team)
    opp_s = prior_stats.get(opp)
    if not own or not opp_s:
        return None
    return {
        "prior_net_margin": own["net_margin"], "prior_win_rate": own["win_rate"],
        "prior_avg_points_for": own["avg_points_for"], "prior_avg_points_against": own["avg_points_against"],
        "prior_games": own["games"],
        "opp_prior_net_margin": opp_s["net_margin"], "opp_prior_win_rate": opp_s["win_rate"],
        "opp_prior_avg_points_for": opp_s["avg_points_for"], "opp_prior_avg_points_against": opp_s["avg_points_against"],
        "opp_prior_games": opp_s["games"],
        "prior_projected_margin": own["net_margin"] - opp_s["net_margin"],
        "is_home": 1.0 if is_home else 0.0, "is_neutral_site": 1.0 if is_neutral else 0.0,
    }


def build_rows(cur_games, prior_stats):
    out = []
    for gid, week, home, away, hp, ap, neutral in cur_games:
        home_won = 1 if hp > ap else 0
        for team, opp, is_home, won in ((home, away, True, home_won), (away, home, False, 1 - home_won)):
            feat = team_features(team, opp, prior_stats, is_home, neutral)
            if feat is None:
                continue
            row = dict(feat)
            row["team_won"] = won
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
    y = np.array([r["team_won"] for r in rows], dtype=np.float32)
    return xgb.DMatrix(X, label=y, feature_names=FEATURE_COLS)


def main():
    import xgboost as xgb
    print("CFB_PRIOR_SEASON_MONEYLINE_GATE_A\n==================================")
    WORKDIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{DB_DEFAULT}?mode=ro", uri=True)

    dev_rows = []
    for season in DEV_SEASONS:
        cur = load_weeks123(conn, season)
        prior = load_full_season_team_stats(conn, season - 1)
        dev_rows += build_rows(cur, prior)
    cur_hol = load_weeks123(conn, HOLDOUT_SEASON)
    prior_hol = load_full_season_team_stats(conn, HOLDOUT_SEASON - 1)
    hol_rows = build_rows(cur_hol, prior_hol)
    conn.close()

    print(f"dev {DEV_SEASONS}: {len(dev_rows)} rows")
    print(f"holdout {HOLDOUT_SEASON}: {len(hol_rows)} rows")

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
    labels_hol = [r["team_won"] for r in hol_rows]
    challenger = metrics(list(map(float, probs_hol)), labels_hol)

    train_rate = float(np.mean([r["team_won"] for r in tr]))
    constant = metrics([train_rate] * len(hol_rows), labels_hol)

    print(f"\n  {'arm':12s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'constant':12s} {'n/a':>7s} {constant['log_loss']:>9.5f}  {constant['brier']:>7.5f} {constant['ece']:>7.4f}")
    print(f"  {'challenger':12s} {challenger['auc']:>7.4f}  {challenger['log_loss']:>9.5f}  {challenger['brier']:>7.5f} {challenger['ece']:>7.4f}")

    imp = bst.get_score(importance_type="gain")
    print("\n  feature importance (gain):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"    {k:28s} {v:9.2f}")

    d_ll = constant["log_loss"] - challenger["log_loss"]
    c1 = challenger["auc"] >= 0.58
    c2 = d_ll >= 0.01
    c3 = challenger["brier"] < constant["brier"]
    passed = c1 and c2 and c3
    verdict = f"MONEYLINE_PRIOR_SEASON_{'PASSES_GATE' if passed else 'DOES_NOT_CLEAR_GATE'}"
    print(f"\n  GATE: AUC>=0.58 -> {c1}   logloss gain>=0.01 -> {c2}   Brier better -> {c3}")
    print(f"  VERDICT: {verdict}")

    bst.save_model(str(WORKDIR / "cfb_prior_season_moneyline.json"))
    (WORKDIR / "cfb_prior_season_moneyline_columns.json").write_text(json.dumps(FEATURE_COLS))
    report = {
        "script": "CFB_PRIOR_SEASON_MONEYLINE_GATE_A",
        "dev_seasons": DEV_SEASONS, "holdout_season": HOLDOUT_SEASON,
        "n_dev": len(dev_rows), "n_holdout": len(hol_rows),
        "constant": constant, "challenger": challenger,
        "importance": imp, "passed": passed, "verdict": verdict,
    }
    (WORKDIR / "cfb_prior_season_moneyline_gate_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport + model: {WORKDIR}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
