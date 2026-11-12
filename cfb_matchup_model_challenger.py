#!/usr/bin/env python3
"""Research-only CFB moneyline matchup challenger.

STRICT AS-OF CONTRACT: every candidate's features come from completed games
in weeks strictly less than its target week, including for historical rows.
The model trains on 2019-2023, early-stops on 2024 and evaluates 2025. It does
not use sportsbook prices, injuries, confirmed starters, weather or lineups.
The 2025 season was consulted by earlier research and is NOT a blind holdout.
Never automatically promote to production based on these results.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from cfb_matchup_intelligence_shadow import query_games, query_boxscores, summary

TRAIN_SEASONS = (2019, 2020, 2021, 2022, 2023)
STOP_SEASON = 2024
EVAL_SEASON = 2025
FEATURES = [
    "home_last5_points_for", "away_last5_points_for",
    "home_last5_points_allowed", "away_last5_points_allowed",
    "home_last5_margin", "away_last5_margin",
    "home_margin_trend", "away_margin_trend",
    "home_margin_sd", "away_margin_sd",
    "home_pass_rate", "away_pass_rate",
    "home_carries", "away_carries",
    "home_pass_attempts", "away_pass_attempts",
    "home_recent_qb_changed", "away_recent_qb_changed",
    "home_prior_games", "away_prior_games",
    "home_offense_vs_away_defense", "away_offense_vs_home_defense",
    "margin_difference", "pass_rate_difference",
    "neutral_site",
]


def number(x):
    return float(x) if x is not None and math.isfinite(float(x)) else np.nan


def difference(a, b):
    x, y = number(a), number(b)
    return x - y if math.isfinite(x) and math.isfinite(y) else np.nan


def make_features(home, away, neutral):
    h, a = home, away
    def v(z, name):
        return z.get(name)
    hm, am = v(h, "last_five_margin"), v(a, "last_five_margin")
    ht, at = v(h, "last_three_margin"), v(a, "last_three_margin")
    hf, af = v(h, "last_five_points_for"), v(a, "last_five_points_for")
    ha, aa = v(h, "last_five_points_allowed"), v(a, "last_five_points_allowed")
    hp, ap = v(h, "last_five_pass_rate"), v(a, "last_five_pass_rate")
    vals = [
        hf, af, ha, aa, hm, am,
        difference(ht, hm), difference(at, am),
        v(h, "last_five_margin_sd"), v(a, "last_five_margin_sd"),
        hp, ap,
        v(h, "last_five_carries"), v(a, "last_five_carries"),
        v(h, "last_five_pass_attempts"), v(a, "last_five_pass_attempts"),
        float(bool(h.get("last_game_passer_changed"))),
        float(bool(a.get("last_game_passer_changed"))),
        h["prior_completed_games"], a["prior_completed_games"],
        difference(hf, aa), difference(af, ha),
        difference(hm, am), difference(hp, ap),
        int(bool(neutral)),
    ]
    return np.asarray([number(x) for x in vals], dtype=np.float32)


def load_historical_games(con, season):
    rows = con.execute(
        """SELECT game_id, week, home_team, away_team, home_points,
                  away_points, neutral_site
           FROM games WHERE season=? AND home_points IS NOT NULL
             AND away_points IS NOT NULL ORDER BY week, game_id""",
        (season,)
    ).fetchall()
    return [dict(zip(("game_id", "week", "home", "away", "hp", "ap", "neutral"), row))
            for row in rows if row[4] != row[5]]


def historical_rows(con, season):
    games = load_historical_games(con, season)
    by_week = defaultdict(list)
    for g in games:
        by_week[int(g["week"])].append(g)
    out = []
    for week, scheduled in sorted(by_week.items()):
        histories = query_games(con, season, week)
        boxes = query_boxscores(con, season, week)
        cache = {}
        for g in scheduled:
            home, away = g["home"], g["away"]
            for team in (home, away):
                if team not in cache:
                    cache[team] = summary(team, histories, boxes)
            h, a = cache[home], cache[away]
            if min(h["prior_completed_games"], a["prior_completed_games"]) < 3:
                continue
            if min(h["opportunity_games_covered"], a["opportunity_games_covered"]) < 2:
                continue
            out.append({
                "game_id": str(g["game_id"]), "season": season, "week": week,
                "label": int(g["hp"] > g["ap"]),
                "features": make_features(h, a, g["neutral"]),
                "margin_diff": difference(h["last_five_margin"], a["last_five_margin"])
            })
    return out


def array(rows):
    return np.stack([r["features"] for r in rows]), np.asarray(
        [r["label"] for r in rows], dtype=np.float32)


def auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    pos = y == 1
    neg = y == 0
    if not pos.any() or not neg.any():
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # Average ranks for tied probabilities.
    for v in np.unique(p):
        t = np.flatnonzero(p == v)
        if len(t) > 1:
            ranks[t] = np.mean(ranks[t])
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) /
                 (pos.sum() * neg.sum()))


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return {
        "n": int(len(y)), "wins_for_home": int(y.sum()),
        "accuracy": round(float(np.mean((p >= .5) == y)), 4),
        "brier": round(float(np.mean((p - y) ** 2)), 5),
        "logloss": round(float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))), 5),
        "auc": None if auc(y, p) is None else round(auc(y, p), 5),
        "mean_prediction": round(float(p.mean()), 4),
        "actual_home_rate": round(float(y.mean()), 4)
    }


def run(db):
    import xgboost as xgb
    con = sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro", uri=True)
    try:
        seasons = {}
        for s in (*TRAIN_SEASONS, STOP_SEASON, EVAL_SEASON):
            seasons[s] = historical_rows(con, s)
    finally:
        con.close()
    train = [r for s in TRAIN_SEASONS for r in seasons[s]]
    validation, evaluation = seasons[STOP_SEASON], seasons[EVAL_SEASON]
    if min(len(train), len(validation), len(evaluation)) < 100:
        raise ValueError({
            "message": "insufficient independent historical matchup coverage",
            "train": len(train), "validation": len(validation),
            "evaluation": len(evaluation),
        })
    xt, yt = array(train)
    xv, yv = array(validation)
    xe, ye = array(evaluation)
    dtrain = xgb.DMatrix(xt, label=yt, feature_names=FEATURES)
    dstop = xgb.DMatrix(xv, label=yv, feature_names=FEATURES)
    deval = xgb.DMatrix(xe, feature_names=FEATURES)
    params = {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "max_depth": 3, "eta": 0.03, "min_child_weight": 20,
        "subsample": 0.8, "colsample_bytree": 0.85,
        "reg_lambda": 6.0, "seed": 20261112, "nthread": 2,
    }
    booster = xgb.train(params, dtrain, num_boost_round=700,
                        evals=[(dstop, "2024_early_stop")],
                        early_stopping_rounds=90, verbose_eval=False)
    best_range = (0, booster.best_iteration + 1)
    challenger = booster.predict(deval, iteration_range=best_range)
    naive = np.asarray([1 / (1 + math.exp(-max(-60, min(60,
                            (r["margin_diff"] if math.isfinite(r["margin_diff"]) else 0) / 14))))
                        for r in evaluation])
    result = {
        "status": "RESEARCH_ONLY_NO_PROMOTION",
        "model": "CFB_MATCHUP_CHALLENGER_V1_XGBOOST",
        "train_seasons": list(TRAIN_SEASONS), "early_stop_season": STOP_SEASON,
        "evaluated_season": EVAL_SEASON, "eval_season_previously_consulted": True,
        "market_odds_used": False,
        "forward_2026_test": False,
        "official_prediction_modified": False,
        "feature_contract": "matchup shadow: games/week < target week only",
        "coverage": {
            "train": len(train), "early_stop": len(validation),
            "2025_eval": len(evaluation)
        },
        "features": FEATURES,
        "best_iteration": booster.best_iteration,
        "candidate": metrics(ye, challenger),
        "margin_only_heuristic": metrics(ye, naive),
        "must_beat_incumbent_on_same_games_and_true_future_data": True,
        "promote": False,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="cfb_models/cfb_model.sqlite")
    parser.add_argument("--out", default="/tmp/cfb_matchup_challenger_2025_research.json")
    args = parser.parse_args()
    report = run(args.db)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
