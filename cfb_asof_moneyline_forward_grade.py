#!/usr/bin/env python3
"""Read-only paired FIRST-PREGAME moneyline grading: incumbent vs as-of shadow.

Use exactly one matched genuine pre-kickoff forecast per game, and the same
completed final result for both models. No threshold selection, retraining,
backfills, promoted picks, or sportsbook ROI claims.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from cfb_asof_forward_moneyline import SCHEMA, NO_PROMOTION, utc
from cfb_grade_record_a import CANCELLED_STATUSES, load_games

AUDIT = "CFB_ASOF_SHADOW_FIRST_PREGAME_PAIRED_FORWARD_AUDIT_V1"
EPSILON = 1e-6


def _logloss(p, y):
    p = min(max(float(p), EPSILON), 1 - EPSILON)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def evaluate(rows, completed_games, statuses):
    games = {str(g["game_id"]): g for g in completed_games}
    seen = set()
    graded = []
    pending = Counter()
    for row in rows:
        if row.get("schema") != SCHEMA or row.get("status") != NO_PROMOTION:
            raise ValueError("unsupported or corrupted shadow provenance")
        key = (row.get("season"), row.get("week"), str(row.get("game_id")))
        if key in seen:
            raise ValueError("duplicate immutable shadow game identity")
        seen.add(key)
        try:
            shadow_at, incumbent_at = (utc(row["first_challenger_logged_at_utc"]),
                                      utc(row["first_incumbent_logged_at_utc"]))
            kickoff = utc(row["kickoff_utc"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("unverifiable first-pregame timestamp") from exc
        if max(shadow_at, incumbent_at) >= kickoff:
            raise ValueError("cannot grade post-kickoff disguised as pregame")
        if row.get("official_prediction") is not False:
            raise ValueError("shadow cannot be marked an official pick")
        g = games.get(key[2])
        if g is None:
            pending["game_not_in_current_results"] += 1
            continue
        if (g["season"] != row["season"] or g["week"] != row["week"] or
            g["home_team"] != row["team_home"] or
            g["away_team"] != row["team_away"]):
            raise ValueError("incompatible game schedule identity")
        if statuses.get(key[2]) in CANCELLED_STATUSES:
            pending["cancelled_or_excluded"] += 1
            continue
        h, a = g["home_points"], g["away_points"]
        if h is None or a is None:
            pending["not_final"] += 1
            continue
        if h == a:
            pending["tied_excluded"] += 1
            continue
        ih = (row.get("original_incumbent_first_probability")
              if row.get("incumbent_selected_team") == g["home_team"]
              else 1 - row.get("original_incumbent_first_probability"))
        ch = row.get("challenger_home_final_probability")
        ca = row.get("challenger_away_final_probability")
        if not all(isinstance(p, (int, float)) and not isinstance(p, bool)
                   and math.isfinite(p) and 0 <= p <= 1 for p in (ih, ch, ca)):
            raise ValueError("invalid first-pregame original probability")
        if abs(ch + ca - 1) > 1e-6:
            raise ValueError("shadow pair probabilities inconsistent")
        y = int(h > a)
        inc_brier, ch_brier = (ih - y)**2, (ch - y)**2
        inc_ll, ch_ll = _logloss(ih, y), _logloss(ch, y)
        winner = g["home_team"] if y else g["away_team"]
        graded.append({
            "season": row["season"], "week": row["week"], "game_id": key[2],
            "first_incumbent_at": incumbent_at.isoformat(),
            "first_shadow_at": shadow_at.isoformat(),
            "kickoff": kickoff.isoformat(),
            "winner": winner,
            "incumbent_selected": row["incumbent_selected_team"],
            "challenger_selected": row["challenger_selected_team"],
            "incumbent_correct": winner == row["incumbent_selected_team"],
            "challenger_correct": winner == row["challenger_selected_team"],
            "home_actual": y, "incumbent_home_probability": ih,
            "challenger_home_probability": ch,
            "incumbent_brier": inc_brier, "challenger_brier": ch_brier,
            "incumbent_logloss": inc_ll, "challenger_logloss": ch_ll,
            "brier_gain_positive_helps_challenger": inc_brier - ch_brier,
            "logloss_gain_positive_helps_challenger": inc_ll - ch_ll
        })
    return graded, dict(pending)


def summarize(rows):
    if not rows:
        return {"n": 0, "status": "NO_FINISHED_FIRST_PREGAME_MATCHED_GAMES"}
    return {
        "n": len(rows),
        "incumbent_winners": sum(r["incumbent_correct"] for r in rows),
        "challenger_winners": sum(r["challenger_correct"] for r in rows),
        "incumbent_brier": round(mean(r["incumbent_brier"] for r in rows), 5),
        "challenger_brier": round(mean(r["challenger_brier"] for r in rows), 5),
        "incumbent_logloss": round(mean(r["incumbent_logloss"] for r in rows), 5),
        "challenger_logloss": round(mean(r["challenger_logloss"] for r in rows), 5),
        "brier_gain_positive_helps_challenger": round(
            mean(r["brier_gain_positive_helps_challenger"] for r in rows), 5),
        "logloss_gain_positive_helps_challenger": round(
            mean(r["logloss_gain_positive_helps_challenger"] for r in rows), 5),
    }


def paired_game_bootstrap(graded, seed=20261011, draws=4000):
    if len(graded) < 12:
        return {"available": False, "reason": "n<12 paired completed games"}
    rng = random.Random(seed)
    result = {}
    for metric in ("brier", "logloss"):
        differences = [r[f"{metric}_gain_positive_helps_challenger"] for r in graded]
        simulated = sorted(
            mean(differences[rng.randrange(len(differences))]
                 for _ in range(len(differences)))
            for _ in range(draws))
        result[f"{metric}_gain_ci95"] = [
            round(simulated[int(draws*.025)], 6),
            round(simulated[min(draws-1, int(draws*.975))], 6)]
    result["available"] = True
    result["resampling_unit"] = "one_game"
    return result


def audit(rows, con):
    con.row_factory = sqlite3.Row
    games, snap = load_games(con)
    graded, pending = evaluate(rows, games, snap)
    weeks = sorted({(r["season"], r["week"]) for r in graded})
    return {
        "audit": AUDIT, "mode": "OBSERVATIONAL_RESEARCH_ONLY_NEVER_AUTO_PROMOTE",
        "original_shadows_seen": len(rows), "matched_final_games": len(graded),
        "pending_or_excluded": pending,
        "overall": summarize(graded),
        "by_week": {
            f"{season}-W{week}": summarize([
                r for r in graded if r["season"] == season and r["week"] == week])
            for season, week in weeks},
        "paired_game_bootstrap": paired_game_bootstrap(graded),
        "only_future_first_pregame_forecasts_count": True,
        "no_prediction_revisions": True, "no_betting_ROI_claim": True,
        "promotion_decision": "NONE_NOT_YET_PROVEN",
        "limitations": [
            "A research-model candidate with a true future forecast is not an independent sportsbook bet.",
            "Holdout cannot contain predictions produced retrospectively after kickoff.",
            "Do not tune threshold, side choice, coefficients or probabilities on graded upcoming weeks.",
            "Game-level log loss/Brier are proper probability scores, not point-spread or player-prop errors."
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ledger", default="docs/cfb_asof_shadow_moneyline_first_seen.jsonl")
    ap.add_argument("--out", default="docs/cfb_asof_moneyline_forward_grade.json")
    a = ap.parse_args()
    path = Path(a.ledger)
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()] if path.exists() else []
    con = sqlite3.connect(f"file:{Path(a.db).resolve()}?mode=ro", uri=True)
    try:
        result = audit(rows, con)
    finally:
        con.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, sort_keys=True, indent=2)+"\n")
    print(json.dumps({k:result[k] for k in
                      ("audit","original_shadows_seen","matched_final_games",
                       "overall","pending_or_excluded")},sort_keys=True))


if __name__ == "__main__":
    main()
