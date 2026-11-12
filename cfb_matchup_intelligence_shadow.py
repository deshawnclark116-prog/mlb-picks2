#!/usr/bin/env python3
"""Strictly pre-week CFB matchup intelligence evidence (research only).

Provides observed team scoring / offensive opportunity / quarterback-role
context for each live moneyline game without revising the champion's original
probability or leaking target-week outcomes.  Possessions, sacks, pass rush,
coverage, injuries, confirmed starters and weather are NOT inferred from this
box-score-only feed.  No betting pick is generated.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

SCHEMA = "CFB_MATCHUP_INTELLIGENCE_SHADOW_V1"
N_RECENT = 5


def safe_round(x, digits=3):
    return round(float(x), digits) if x is not None and math.isfinite(float(x)) else None


def safe_mean(xs):
    return mean(xs) if xs else None


def query_games(con, season, week):
    """An entire target week is held out, including games completed early."""
    rows = con.execute(
        """SELECT game_id, week, home_team, away_team, home_points, away_points
           FROM games WHERE season=? AND week<? AND
           home_points IS NOT NULL AND away_points IS NOT NULL
           ORDER BY week, game_id""", (season, week)
    ).fetchall()
    histories = defaultdict(list)
    for gid, w, home, away, hp, ap in rows:
        histories[home].append({
            "game_id": str(gid), "week": int(w), "opponent": away,
            "points_for": float(hp), "points_against": float(ap)
        })
        histories[away].append({
            "game_id": str(gid), "week": int(w), "opponent": home,
            "points_for": float(ap), "points_against": float(hp)
        })
    return histories


def query_boxscores(con, season, week):
    rows = con.execute(
        """SELECT game_id, team, player_id, player_name, position,
                  COALESCE(carries,0), COALESCE(pass_attempts,0),
                  COALESCE(passing_interceptions,0)
           FROM player_games WHERE season=? AND week<?""",
        (season, week)
    ).fetchall()
    totals = defaultdict(lambda: {"carries": 0., "pass_attempts": 0.,
                                   "interceptions": 0., "qbs": []})
    for gid, team, pid, name, pos, carries, attempts, ints in rows:
        z = totals[(str(gid), team)]
        z["carries"] += float(carries)
        z["pass_attempts"] += float(attempts)
        z["interceptions"] += float(ints)
        if attempts and float(attempts) > 0:
            z["qbs"].append({
                "player_id": str(pid), "name": name,
                "attempts": float(attempts), "position": pos
            })
    return totals


def summary(team, hist, box):
    history = hist.get(team, [])
    last = history[-N_RECENT:]
    recent3 = history[-3:]
    points_for = [x["points_for"] for x in last]
    points_against = [x["points_against"] for x in last]
    score_margins = [x["points_for"] - x["points_against"] for x in last]
    opportunities = []
    for g in last:
        z = box.get((g["game_id"], team))
        if z and z["carries"] + z["pass_attempts"] > 0:
            opportunities.append({
                "game_id": g["game_id"],
                "carries": z["carries"], "pass_attempts": z["pass_attempts"],
                "pass_rate": z["pass_attempts"] / (z["carries"] + z["pass_attempts"])
            })
    # All observed completed games, rather than the live roster, determine
    # earlier passer deployment. Never call a historical leader a confirmed starter.
    qb_sequence = []
    for g in recent3:
        players = (box.get((g["game_id"], team)) or {}).get("qbs", [])
        total = sum(p["attempts"] for p in players)
        if not total:
            continue
        qb = max(players, key=lambda x: x["attempts"])
        qb_sequence.append({
            "game_id": g["game_id"], "player_id": qb["player_id"],
            "name": qb["name"], "attempt_share": safe_round(qb["attempts"] / total)
        })
    qb_changed = len(qb_sequence) >= 2 and (
        qb_sequence[-1]["player_id"] != qb_sequence[-2]["player_id"]
    )
    game_count = len(history)
    flags = []
    if game_count < 3:
        flags.append("FEWER_THAN_THREE_PRIOR_TEAM_GAMES")
    if len(opportunities) < min(len(last), 3):
        flags.append("INCOMPLETE_OFFENSIVE_OPPORTUNITY_BOX_SCORES")
    if qb_changed:
        flags.append("RECENT_LEADING_PASSER_CHANGED_NOT_A_STARTER_CONFIRMATION")
    if len(score_margins) >= 3 and pstdev(score_margins) >= 21:
        flags.append("HIGH_SCORE_MARGIN_VOLATILITY")
    return {
        "team": team,
        "prior_completed_games": game_count,
        "observed_game_ids_recent": [g["game_id"] for g in last],
        "last_game_week": last[-1]["week"] if last else None,
        "last_five_points_for": safe_round(safe_mean(points_for)),
        "last_five_points_allowed": safe_round(safe_mean(points_against)),
        "last_five_margin": safe_round(safe_mean(score_margins)),
        "last_three_margin": safe_round(safe_mean(
            [g["points_for"] - g["points_against"] for g in recent3])),
        "last_five_margin_sd": safe_round(pstdev(score_margins) if len(score_margins) >= 2 else None),
        "opportunity_games_covered": len(opportunities),
        "last_five_pass_rate": safe_round(safe_mean([z["pass_rate"] for z in opportunities])),
        "last_five_carries": safe_round(safe_mean([z["carries"] for z in opportunities])),
        "last_five_pass_attempts": safe_round(safe_mean([z["pass_attempts"] for z in opportunities])),
        "last_three_leading_passers": qb_sequence,
        "last_game_passer_changed": qb_changed,
        "risk_flags": flags
    }


def build(con, board):
    season, week = int(board["season"]), int(board["week"])
    histories = query_games(con, season, week)
    boxes = query_boxscores(con, season, week)
    games = []
    visited = set()
    for pick in board.get("picks", []):
        if pick.get("market") != "moneyline":
            continue
        if int(pick.get("season", -1)) != season or int(pick.get("week", -1)) != week:
            raise ValueError("moneyline pick belongs to a different season/week")
        gid = str(pick.get("game_id") or "")
        if not gid or gid in visited:
            raise ValueError("missing or duplicate moneyline game id")
        visited.add(gid)
        team, opponent = pick["team"], pick["opponent"]
        own = summary(team, histories, boxes)
        other = summary(opponent, histories, boxes)
        games.append({
            "game_id": gid, "team": team, "opponent": opponent,
            "kickoff_utc": pick.get("kickoff_utc"),
            "champion_pick": pick.get("pick"),
            "champion_original_probability_display_only": pick.get("model_prob"),
            "team_context": own, "opponent_context": other,
            "scoring_matchup_last_five": {
                "team_offense_vs_opponent_points_allowed": (
                    [own["last_five_points_for"], other["last_five_points_allowed"]]),
                "opponent_offense_vs_team_points_allowed": (
                    [other["last_five_points_for"], own["last_five_points_allowed"]])
            },
            "context_ready": min(own["prior_completed_games"], other["prior_completed_games"]) >= 3
                and min(own["opportunity_games_covered"], other["opportunity_games_covered"]) >= 3,
            "prediction_produced": False
        })
    return {
        "schema": SCHEMA, "season": season, "week": week,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "board_generated_at_utc": board.get("generated_at_utc"),
        "status": "RESEARCH_ONLY_NO_AUTOMATIC_PROMOTION",
        "model_changed": False,
        "official_pick_modified": False,
        "game_count": len(games),
        "source": "ESPN completed final scores and player box scores in local cfb_model.sqlite",
        "asof_rule": "Only games and player_games rows whose week is STRICTLY LESS than forecast target week",
        "limitations": [
            "Rushes plus pass attempts are an observed opportunity proxy, NOT true possessions or snaps",
            "No opponent-adjusted trench, pressure, coverage or play-by-play metrics in this feed",
            "Past leading passer is not a verified current starter",
            "No certified injuries, inactives, weather, market prices, or betting edge",
            "Descriptive context is NOT a validated winner model and produces no probability"
        ],
        "games": games
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="cfb_models/cfb_model.sqlite")
    ap.add_argument("--board", default="docs/cfb_predictions.json")
    ap.add_argument("--out", default="docs/cfb_matchup_intelligence_shadow.json")
    args = ap.parse_args()
    board = json.loads(Path(args.board).read_text())
    con = sqlite3.connect(f"file:{Path(args.db).resolve()}?mode=ro", uri=True)
    try:
        result = build(con, board)
    finally:
        con.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"CFB matchup intelligence (shadow): {result['game_count']} games; no live picks altered")


if __name__ == "__main__":
    main()
