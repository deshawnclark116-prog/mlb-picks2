#!/usr/bin/env python3
"""Prospective NFL high-probability model targets: FIRST SEEN, never backfilled.

Freeze before kickoff after independent three-game role receipt. Grade only
post-game player_games, with strict team/opponent identity and no assumed DNP.
Model probabilities are uncalibrated and conditional on simulated opportunity.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from nfl_precision_forward import normalized_name
from nfl_target_grid import VERSION

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
LOG = DOCS / "nfl_model_target_firstseen.jsonl"
REPORT = DOCS / "nfl_model_target_forward_grade.json"
DB = ROOT / "nfl_models" / "nfl_model.sqlite"
SCHEMA = "NFL_MODEL_TARGET_FIRSTSEEN_V1"


def instant(s):
    x = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if x.tzinfo is None:
        raise ValueError("must be UTC-aware")
    return x.astimezone(timezone.utc)


def identity(x):
    return "|".join(str(x[k]) for k in (
        "season", "week", "kickoff_utc", "team", "player_id", "market"))


def valid_target(p):
    t, g = p.get("nfl_model_target"), p.get("nfl_model_target_grid")
    market = p.get("market")
    floor = {"receiving_yards": 25, "rushing_yards": 30}.get(market)
    if (floor is None or not isinstance(t, dict) or not isinstance(g, dict)
        or g.get("schema") != VERSION or
        p.get("projection_probability_source") != "UNCALIBRATED_HISTORICAL_USAGE_BOOTSTRAP" or
        p.get("nfl_model_target_event") != "P_YARDS_AT_LEAST_FIXED_MEANINGFUL_THRESHOLD" or
        p.get("model_source") == "v2_context" or
        p.get("projection_simulation_n", 0) < 10000 or
        g.get("simulation_n") != p["projection_simulation_n"] or
        g.get("minimum_absolute_yards") != floor or
        g.get("minimum_relative_mean_fraction") != .55 or
        g.get("required_wilson_lower_95") != .65 or
        g.get("grid_step_yards") != 5 or
        g.get("conditional_on_modeled_opportunities") is not True or
        g.get("forward_calibrated") is not False or
        g.get("official_active_status_verified") is not False or
        not isinstance(g.get("grid"), list) or not g["grid"] or
        g.get("chosen_target") != t):
        return False
    mean = p.get("projected_mean", -1)
    if not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean <= 0:
        return False
    yards, probability, lower = t.get("yards"), t.get("probability"), t.get("wilson_lower_95")
    if (not isinstance(yards, int) or yards % 5 != 0 or
        yards < max(floor, math.ceil(.55 * mean)) or
        not all(isinstance(v, (int, float)) and math.isfinite(v)
                for v in (probability, lower)) or
        not .65 <= lower <= probability <= 1):
        return False
    grid = g["grid"]
    if any(grid[i]["yards"] >= grid[i+1]["yards"] or
           grid[i]["probability"] < grid[i+1]["probability"]
           for i in range(len(grid)-1)):
        return False
    valid = [row for row in grid if row["wilson_lower_95"] >= .65]
    return bool(valid and valid[-1] == t)


def capture_rows(board, role, at):
    generated = instant(board["generated_at_utc"])
    captured = instant(role["captured_at_utc"])
    if (role.get("schema") != "nfl-role-evidence-coverage-v1" or
        role.get("status") != "PREGAME_RESEARCH_ROLE_SNAPSHOT" or
        instant(role["forecast_generated_at_utc"]) != generated or
        role.get("season") != board.get("season") or
        role.get("week") != board.get("week") or
        captured < generated or captured > at or generated > at or
        (at-generated).total_seconds() > 4*3600):
        raise ValueError("unverified pregame role/data receipt")
    verified = set()
    for r in role.get("player_market_evidence", []):
        if (r.get("role_evidence_complete_for_showcase") is True and
            r.get("observed_stat_rows") == 3 and
            r.get("roster_status") == "ROSTER_LISTED_NOT_STARTER_VERIFIED" and
            r.get("warnings") == [] and
            r.get("has_verified_starting_role") is False and
            r.get("has_calibrated_book_bet_probability") is False):
            verified.add((r.get("player_id"), r.get("team"), r.get("opponent"),
                          r.get("market"), r.get("kickoff_utc"),
                          r.get("published_model_median"),
                          r.get("published_model_source")))
    games = defaultdict(list)
    for p in board.get("picks", []):
        if not valid_target(p) or instant(p["kickoff_utc"]) <= at or instant(p["kickoff_utc"]) <= generated:
            continue
        k = (p.get("player_id"), p.get("team"), p.get("opponent"),
             p.get("market"), p.get("kickoff_utc"),
             p.get("projected_median"), p.get("model_source"))
        if k not in verified:
            continue
        game = (p["kickoff_utc"], *sorted([p["team"], p["opponent"]]))
        games[game].append(p)
    output = []
    for _, picks in sorted(games.items()):
        picks.sort(key=lambda p: (-p["nfl_model_target"]["probability"],
                                  -p["nfl_model_target"]["wilson_lower_95"],
                                  -p["nfl_model_target"]["yards"],
                                  p["player"]))
        for rank, p in enumerate(picks[:10], 1):
            t = p["nfl_model_target"]
            output.append({
                "schema": SCHEMA, "season": board["season"], "week": board["week"],
                "kickoff_utc": p["kickoff_utc"], "team": p["team"],
                "opponent": p["opponent"], "player_id": p["player_id"],
                "player": p["player"], "market": p["market"], "rank": rank,
                "target_yards": t["yards"], "model_hit_probability": t["probability"],
                "mc_wilson_lower_95": t["wilson_lower_95"],
                "model_mean": p["projected_mean"], "model_median": p["projected_median"],
                "model_source": p["model_source"], "simulation_n": p["projection_simulation_n"],
                "forecast_generated_at_utc": board["generated_at_utc"],
                "role_verified_at_utc": role["captured_at_utc"],
                "firstseen_at_utc": at.isoformat(),
                "game_day_active_verified": False,
                "forward_calibrated": False,
                "source_event": "P_YARDS_AT_LEAST_FIXED_MEANINGFUL_THRESHOLD",
            })
    return output


def capture(board, role, now, log_path=LOG):
    selected = capture_rows(board, role, now)
    previous = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()] if log_path.exists() else []
    seen = {identity(x) for x in previous}
    if len(seen) != len(previous):
        raise ValueError("duplicate original firstseen record")
    fresh = [x for x in selected if identity(x) not in seen]
    if fresh:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            for x in fresh:
                f.write(json.dumps(x, sort_keys=True) + "\n")
    return {"selected_now": len(selected), "new_firstseen": len(fresh),
            "total_firstseen": len(previous)+len(fresh)}


def grade(log_rows, database, now):
    if not database.exists():
        return {"schema": "NFL_MODEL_TARGET_FORWARD_GRADE_V1",
                "summary": {"total": len(log_rows), "graded": 0, "unresolved": len(log_rows)},
                "rows": []}
    con = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    by_id, by_name = {}, {}
    periods = {(x["season"], x["week"]) for x in log_rows}
    for season, week in periods:
        for x in con.execute("SELECT player_id,player_name,team,opponent,rushing_yards,receiving_yards "
                             "FROM player_games WHERE season=? AND week=?", (season,week)):
            a = (season,week,str(x["player_id"]),x["team"],x["opponent"])
            b = (season,week,normalized_name(x["player_name"]),x["team"],x["opponent"])
            by_id.setdefault(a, []).append(x)
            by_name.setdefault(b, []).append(x)
    con.close()
    rows = []
    for p in log_rows:
        if now < instant(p["kickoff_utc"]).timestamp() + 6*3600:
            rows.append({"key": identity(p), "status": "PENDING_FULL_GAME"})
            continue
        a=(p["season"],p["week"],str(p["player_id"]),p["team"],p["opponent"])
        b=(p["season"],p["week"],normalized_name(p["player"]),p["team"],p["opponent"])
        matches=by_id.get(a,[])
        if len(matches)!=1:
            matches=by_name.get(b,[])
        if len(matches)!=1 or matches[0][p["market"]] is None:
            rows.append({"key": identity(p), "status": "UNRESOLVED", "market": p["market"]})
            continue
        actual = float(matches[0][p["market"]])
        hit = int(actual >= p["target_yards"])
        rows.append({"key": identity(p), "status": "GRADED",
                     "market": p["market"], "player": p["player"], "week": p["week"],
                     "target": p["target_yards"], "actual": actual,
                     "hit": hit, "probability": p["model_hit_probability"],
                     "brier": (hit-p["model_hit_probability"])**2})
    graded = [x for x in rows if x["status"]=="GRADED"]
    return {"schema": "NFL_MODEL_TARGET_FORWARD_GRADE_V1",
            "conditional_model_probabilities_not_yet_calibrated": True,
            "summary": {
                "total": len(rows), "graded": len(graded),
                "pending": sum(x["status"]=="PENDING_FULL_GAME" for x in rows),
                "unresolved": sum(x["status"]=="UNRESOLVED" for x in rows),
                "observed_hit_rate": round(sum(x["hit"] for x in graded)/len(graded),4) if graded else None,
                "mean_predicted_probability": round(sum(x["probability"] for x in graded)/len(graded),4) if graded else None,
                "brier": round(sum(x["brier"] for x in graded)/len(graded),4) if graded else None,
            }, "rows": rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("operation",choices=("capture","grade"))
    parser.add_argument("--board",type=Path,default=DOCS/"nfl_predictions.json")
    parser.add_argument("--role",type=Path,default=DOCS/"nfl_role_snapshot.json")
    parser.add_argument("--log",type=Path,default=LOG)
    parser.add_argument("--out",type=Path,default=REPORT)
    parser.add_argument("--db",type=Path,default=DB)
    a=parser.parse_args()
    if a.operation=="capture":
        result=capture(json.loads(a.board.read_text()),json.loads(a.role.read_text()),
                       datetime.now(timezone.utc),a.log)
    else:
        log=[json.loads(x) for x in a.log.read_text().splitlines() if x.strip()] if a.log.exists() else []
        result=grade(log,a.db,datetime.now(timezone.utc).timestamp())
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result.get("summary",result),sort_keys=True))


if __name__=="__main__":
    main()
