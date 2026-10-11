#!/usr/bin/env python3
"""CFB Week 6 benchmark-only pregame player-stat forecasts, using ESPN game totals.

Purpose: avoid mistaking 69.5 yards or a binary classifier's confidence for
an expected yardage total. This is a naive, explicitly labeled comparator,
not a promoted CFB model and not a replacement for real per-play distributions.
Run ONLY against the current slate and timestamp before kickoff. Never backfill.
"""
import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

SCHEMA = "CFB_2026_BOX_SCORE_LAST3_PROSPECTIVE_BASELINE_V1"
MARKETS = {
    "rushing_yards": ("rushing_yards", ("RB",)),
    "passing_touchdowns": ("passing_touchdowns", ("QB",)),
    "anytime_touchdowns": (None, ("RB", "WR", "TE", "QB")),
}
FINALS = {"STATUS_FINAL", "STATUS_FINAL_OVERTIME"}


class BaselineError(ValueError):
    pass


def utc(raw):
    try:
        d = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if d.tzinfo is None:
            raise ValueError("unaware")
        return d.astimezone(timezone.utc)
    except (ValueError, TypeError):
        raise BaselineError("INVALID_TIMESTAMP")


def snapshot(db, board, *, at, sha256):
    now = utc(at)
    if board.get("season") != 2026 or board.get("week") != 6:
        raise BaselineError("NOT_ORIGINAL_2026_WEEK6_BOARD")
    generated = utc(board.get("generated_at_utc"))
    if generated > now:
        raise BaselineError("BOARD_CREATED_IN_FUTURE")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise BaselineError("MISSING_SOURCE_DB_HASH")
    games = {}
    for row in db.execute("SELECT game_id,season,week,kickoff_utc,home_team,away_team,home_points,away_points FROM games WHERE season=2026"):
        gid = str(row["game_id"])
        if gid in games:
            raise BaselineError("DUPLICATE_GAME")
        games[gid] = row
    statuses = {}
    for gid, status in db.execute("SELECT game_id,espn_status FROM schedule_snapshot WHERE season=2026"):
        statuses[str(gid)] = status
    # Lookup by actual ID and game. A name-only fuzzy player merge is forbidden.
    prior = {}
    for row in db.execute(
        "SELECT player_id,game_id,team,position,season,week,rushing_yards,"
        "passing_touchdowns,rushing_touchdowns,receiving_touchdowns "
        "FROM player_games WHERE season=2026"):
        key = (str(row["player_id"]),str(row["game_id"]))
        if key in prior:
            raise BaselineError("DUPLICATE_PLAYER_GAME_ID")
        prior[key] = row
    skipped, forecasts, seen = Counter(), [], set()
    for pick in board.get("picks", []):
        market = pick.get("market")
        if market not in MARKETS:
            skipped["MARKET_NOT_STATISTICAL_PLAYER_FORECAST"] += 1
            continue
        gid, pid, team = str(pick.get("game_id")), str(pick.get("player_id")), pick.get("team")
        key = (gid,pid,market)
        if key in seen:
            raise BaselineError("DUPLICATE_CURRENT_BOARD_PLAYER_MARKET")
        seen.add(key)
        game = games.get(gid)
        if game is None:
            skipped["GAME_NOT_FOUND"] += 1
            continue
        target_kick = utc(game["kickoff_utc"])
        board_kick = utc(pick.get("kickoff_utc"))
        if abs((target_kick-board_kick).total_seconds()) > 120:
            skipped["KICKOFF_DIFFERENCE_NEEDS_MANUAL_REVIEW"] += 1
            continue
        if now >= target_kick or generated >= target_kick:
            skipped["NOT_PROVABLY_PREGAME"] += 1
            continue
        if team not in (game["home_team"],game["away_team"]):
            skipped["PLAYER_TEAM_NOT_PARTICIPATING"] += 1
            continue
        if statuses.get(gid) in FINALS or game["home_points"] is not None:
            skipped["GAME_ALREADY_FINAL"] += 1
            continue
        if pick.get("roster_verification") != "ON_CURRENT_ROSTER_SNAPSHOT":
            skipped["ROSTER_NOT_CONFIRMED_IN_BOARD"] += 1
            continue
        stat, allowed_pos = MARKETS[market]
        games_for_player = []
        for (player_id, game_id), row in prior.items():
            if player_id != pid or row["team"] != team or row["position"] not in allowed_pos:
                continue
            past_game = games.get(game_id)
            if (past_game is None or past_game["season"] != 2026 or
                past_game["week"] >= game["week"] or
                statuses.get(game_id) not in FINALS or
                past_game["home_points"] is None or past_game["away_points"] is None):
                continue
            if utc(past_game["kickoff_utc"]) >= now:
                continue
            fields = (["rushing_touchdowns","receiving_touchdowns"]
                      if market=="anytime_touchdowns" else [stat])
            if any(row[field] is None for field in fields):
                continue
            actual = sum(float(row[field]) for field in fields)
            games_for_player.append((int(past_game["week"]),game_id,actual))
        games_for_player.sort(key=lambda r:(r[0],r[1]))
        tail = games_for_player[-3:]
        if len(tail) < 3:
            skipped["INSUFFICIENT_PRIOR_3_SAME_TEAM_FINAL_GAME_STATS"] += 1
            continue
        # Pre-registered zero-fitting baseline: arithmetic average of three
        # observed recent games, not an inferential distribution or sportsbook line.
        vals = [r[2] for r in tail]
        forecast = {
            "market":market,"game_id":gid,"player_id":pid,
            "player":pick.get("player"),"team":team,
            "opponent":pick.get("opponent"),"season":2026,"week":6,
            "position":prior[(pid,tail[-1][1])]["position"],
            "kickoff_utc":target_kick.isoformat().replace("+00:00","Z"),
            "research_generated_at_utc":now.isoformat().replace("+00:00","Z"),
            "board_generated_at_utc":generated.isoformat().replace("+00:00","Z"),
            "point_mean_last3":round(mean(vals),4),
            "point_median_last3":round(median(vals),4),
            "last3_values":vals,
            "last3_game_ids":[r[1] for r in tail],
            "n_recent_final_games":3,
            "unit":"yards" if market=="rushing_yards" else "touchdowns",
            "not_a_calibrated_interval":True,
            "not_a_betting_recommendation":True,
            "original_classifier_probability_not_used":True,
            "research_only":True,
        }
        forecasts.append(forecast)
    return {
        "schema":SCHEMA,"status":"FROZEN_PREGAME_NAIVE_POINT_BASELINE_ONLY",
        "generated_at_utc":now.isoformat().replace("+00:00","Z"),
        "source_database_sha256":sha256,
        "source_board_sha256":None,
        "baseline_rows":len(forecasts),
        "by_market":dict(Counter(x["market"] for x in forecasts)),
        "excluded":dict(sorted(skipped.items())),
        "forecasts":sorted(forecasts,key=lambda r:(r["kickoff_utc"],r["game_id"],r["market"],r["player_id"])),
        "historical_forecasts_backfilled":0,"production_model_modified":False,
        "claim":"Not a new champion; compare to future actuals on the same player-game population.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--db",type=Path,required=True)
    p.add_argument("--board",type=Path,default=Path("docs/cfb_predictions.json"))
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    raw=a.board.read_bytes()
    h=hashlib.sha256()
    with a.db.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    with sqlite3.connect(f"file:{a.db}?mode=ro",uri=True) as db:
        db.row_factory=sqlite3.Row
        data=snapshot(db,json.loads(raw),at=datetime.now(timezone.utc),sha256=h.hexdigest())
    data["source_board_sha256"]=hashlib.sha256(raw).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":data["status"],"baseline_rows":data["baseline_rows"],
                      "by_market":data["by_market"],"excluded":data["excluded"]},sort_keys=True))


if __name__=="__main__":
    main()
