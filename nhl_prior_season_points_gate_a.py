#!/usr/bin/env python3
"""
NHL_PRIOR_SEASON_POINTS_GATE_A

Same design as nhl_prior_season_moneyline_gate_a.py, adapted to the
player level: instead of leaving the first weeks of a new season with
zero anytime-point picks until a skater has 5 real current-season games
(nhl_points_champion_gate_a.py's own eligibility floor), feed each
skater's PRIOR season's real, complete rate stats (points/60, games
played, season average) -- exactly the real history a bettor already
has on opening night, whether or not the model does.

Population: every real skater-game row in weeks 1-3 (see nhl_live_
foundation_a.py's docstring for what "week" means here) with a prior-
season record on file -- no volume/games floor here by design, same
reasoning the moneyline prior-season gate documents (a serving-time
floor belongs in the production builder, not training/validation).

DEV=2019-2023 (prior seasons 2018-2022), HOLDOUT=2024 (prior season
2023) -- same seasons as the moneyline prior-season gate, for
consistency.

Pre-registered pass (written before this script has ever been run):
AUC >= 0.58, logloss beats constant by >= 0.01, Brier beats constant.

Read-only. Writes only its own workdir.

Run
---
python -u nhl_prior_season_points_gate_a.py
"""
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

DB_DEFAULT = "nhl_models/nhl_model.sqlite"
WORKDIR = Path("nhl_models/nhl_prior_season_points_gate_a_work")
MAX_WEEK = 3
DEV_SEASONS = [2019, 2020, 2021, 2022, 2023]
HOLDOUT_SEASON = 2024
NAN = float("nan")
STAT_COL = "points"
LINE = 0.5
TABLE = "skater_games"

FEATURE_COLS = [
    "prior_avg", "prior_games", "prior_per60", "opp_prior_allowed",
    "is_home", "team_net_margin", "opp_net_margin", "projected_margin",
]


def load_weeks123_rows(conn, season):
    return conn.execute(f"""
        SELECT sg.player_id, sg.player_name, sg.team, sg.opponent, sg.is_home, sg.{STAT_COL}
        FROM {TABLE} sg JOIN games g ON sg.game_id = g.game_id
        WHERE sg.season = ? AND g.week <= ?
    """, (season, MAX_WEEK)).fetchall()


def load_prior_season_player_stats(conn, season):
    rows = conn.execute(f"SELECT player_id, team, {STAT_COL}, toi_seconds FROM {TABLE} WHERE season=?",
                         (season,)).fetchall()
    state = {}
    for pid, team, stat, toi in rows:
        d = state.setdefault(pid, {"team": team, "total_stat": 0.0, "total_toi": 0.0, "games": 0})
        d["team"] = team
        d["total_stat"] += stat or 0
        d["total_toi"] += toi or 0
        d["games"] += 1
    out = {}
    for pid, d in state.items():
        if d["games"] == 0:
            continue
        total_toi_hr = d["total_toi"] / 3600.0
        out[pid] = {"games": d["games"], "avg": d["total_stat"] / d["games"],
                    "per60": (d["total_stat"] / total_toi_hr) if total_toi_hr > 0 else 0.0}
    return out


def load_prior_season_opp_allowed(conn, season):
    rows = conn.execute(f"SELECT game_id, team, SUM({STAT_COL}) FROM {TABLE} WHERE season=? GROUP BY game_id, team",
                         (season,)).fetchall()
    totals = {(gid, team): (total or 0) for gid, team, total in rows}
    game_teams = defaultdict(list)
    for gid, team, _ in rows:
        game_teams[gid].append(team)
    s, n = defaultdict(float), defaultdict(int)
    for gid, teams in game_teams.items():
        if len(teams) != 2:
            continue
        a, b = teams
        s[a] += totals[(gid, b)]; n[a] += 1
        s[b] += totals[(gid, a)]; n[b] += 1
    return {t: s[t] / n[t] for t in s if n[t] > 0}


def load_prior_season_team_stats(conn, season):
    games = conn.execute(
        "SELECT home_abbrev, away_abbrev, home_score, away_score FROM games "
        "WHERE season=? AND home_score IS NOT NULL AND away_score IS NOT NULL", (season,)).fetchall()
    state = {}
    for ha, aa, hs, aws in games:
        hst = state.setdefault(ha, [0, 0, 0, 0])
        hst[0] += 1 if hs > aws else 0
        hst[1] += hs; hst[2] += aws; hst[3] += 1
        ast = state.setdefault(aa, [0, 0, 0, 0])
        ast[0] += 1 if aws > hs else 0
        ast[1] += aws; ast[2] += hs; ast[3] += 1
    out = {}
    for team, (wins, gf, ga, n) in state.items():
        if n == 0:
            continue
        out[team] = {"net_margin": (gf - ga) / n}
    return out


def build_rows(cur_rows, prior_player_stats, prior_opp_allowed, prior_team_stats):
    out = []
    for pid, pname, team, opp, is_home, stat_val in cur_rows:
        pst = prior_player_stats.get(pid)
        if pst is None:
            continue
        team_margin = (prior_team_stats.get(team) or {}).get("net_margin")
        opp_margin = (prior_team_stats.get(opp) or {}).get("net_margin")
        proj_margin = (team_margin - opp_margin) if (team_margin is not None and opp_margin is not None) else None
        out.append({
            "prior_avg": pst["avg"], "prior_games": pst["games"], "prior_per60": pst["per60"],
            "opp_prior_allowed": prior_opp_allowed.get(opp),
            "is_home": 1.0 if is_home else 0.0,
            "team_net_margin": team_margin, "opp_net_margin": opp_margin, "projected_margin": proj_margin,
            "target": 1 if stat_val >= LINE + 0.5 else 0,
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
    X = np.array([[r.get(c, NAN) if r.get(c) is not None else NAN for c in FEATURE_COLS] for r in rows], dtype=np.float32)
    y = np.array([r["target"] for r in rows], dtype=np.float32)
    return xgb.DMatrix(X, label=y, feature_names=FEATURE_COLS)


def main():
    import xgboost as xgb
    print("NHL_PRIOR_SEASON_POINTS_GATE_A\n===============================")
    WORKDIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{DB_DEFAULT}?mode=ro", uri=True)

    dev_rows = []
    for season in DEV_SEASONS:
        cur = load_weeks123_rows(conn, season)
        prior_player = load_prior_season_player_stats(conn, season - 1)
        prior_allowed = load_prior_season_opp_allowed(conn, season - 1)
        prior_team = load_prior_season_team_stats(conn, season - 1)
        dev_rows += build_rows(cur, prior_player, prior_allowed, prior_team)

    cur_hol = load_weeks123_rows(conn, HOLDOUT_SEASON)
    prior_player_hol = load_prior_season_player_stats(conn, HOLDOUT_SEASON - 1)
    prior_allowed_hol = load_prior_season_opp_allowed(conn, HOLDOUT_SEASON - 1)
    prior_team_hol = load_prior_season_team_stats(conn, HOLDOUT_SEASON - 1)
    hol_rows = build_rows(cur_hol, prior_player_hol, prior_allowed_hol, prior_team_hol)
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
    labels_hol = [r["target"] for r in hol_rows]
    challenger = metrics(list(map(float, probs_hol)), labels_hol)

    train_rate = float(np.mean([r["target"] for r in tr]))
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
    verdict = f"NHL_PRIOR_SEASON_POINTS_{'PASSES_GATE' if passed else 'DOES_NOT_CLEAR_GATE'}"
    print(f"\n  GATE: AUC>=0.58 -> {c1}   logloss gain>=0.01 -> {c2}   Brier better -> {c3}")
    print(f"  VERDICT: {verdict}")

    bst.save_model(str(WORKDIR / "nhl_prior_season_points.json"))
    (WORKDIR / "nhl_prior_season_points_columns.json").write_text(json.dumps(FEATURE_COLS))
    report = {
        "script": "NHL_PRIOR_SEASON_POINTS_GATE_A",
        "dev_seasons": DEV_SEASONS, "holdout_season": HOLDOUT_SEASON,
        "n_dev": len(dev_rows), "n_holdout": len(hol_rows),
        "constant": constant, "challenger": challenger,
        "importance": imp, "passed": passed, "verdict": verdict,
    }
    (WORKDIR / "nhl_prior_season_points_gate_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport + model: {WORKDIR}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
