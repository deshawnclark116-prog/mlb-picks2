#!/usr/bin/env python3
"""Immutable forward evaluation for the NFL numerical precision shortlist.

Capture: first genuine PRE-KICKOFF 10k+ matched player-bootstrap distributions,
after matching the independent recent-role receipt. Rank by P(within ±10
yards), not the ~50% chance of attaining a distribution's own median.
No sportsbook data, retrospective reconstruction or implied bet calibration.

Grade: join only real postgame nflverse player_games box scores. Missing
player rows are UNRESOLVED, NOT silently a miss or an invented DNP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "NFL_NUMERIC_PRECISION_FIRSTSEEN_V1"
YARDS = {"rushing_yards", "receiving_yards"}
SOURCE = "UNCALIBRATED_HISTORICAL_USAGE_BOOTSTRAP"
ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
LOG = DOCS / "nfl_precision_firstseen.jsonl"
OUT = DOCS / "nfl_precision_forward_grade.json"
DB = ROOT / "nfl_models" / "nfl_model.sqlite"


def utc(value):
    v = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if v.tzinfo is None:
        raise ValueError("naive timestamp")
    return v.astimezone(timezone.utc)


def whole(x):
    return int(math.floor(float(x)) + (float(x) % 1 >= .6 - 1e-10))


def key(row):
    return "|".join(str(row[k]) for k in (
        "season", "week", "kickoff_utc", "team", "market", "player_id"))


def normalized_name(name):
    name = unicodedata.normalize("NFKD", str(name or ""))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"\\b(jr|sr|ii|iii|iv|v)\\.?\\b", "", name)
    name = re.sub(r"[^a-z ]", "", name)
    return re.sub(r"\\s+", " ", name).strip()


def valid_num(v):
    return isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)


def candidate(row):
    if row.get("market") not in YARDS or row.get("projection_probability_source") != SOURCE:
        return False
    if not all(valid_num(row.get(k)) for k in (
        "projected_median", "projected_mean", "projected_low", "projected_high",
        "projection_reach_probability", "projection_within_10_probability")):
        return False
    if row["projected_median"] <= 0 or row["projected_low"] < 0:
        return False
    if not (row["projected_low"] <= row["projected_median"] <= row["projected_high"]):
        return False
    if any(not 0 <= row[k] <= 1 for k in (
        "projection_reach_probability", "projection_within_10_probability")):
        return False
    if row.get("whole_projection") != whole(row["projected_median"]):
        return False
    if not isinstance(row.get("projection_simulation_n"), int) or row["projection_simulation_n"] < 10000:
        return False
    return all(isinstance(row.get(k), str) and row[k].strip() for k in (
        "player", "player_id", "team", "opponent", "model_source", "kickoff_utc"))


def role_key(row):
    return (str(row.get("player_id")), row.get("team"), row.get("opponent"),
            row.get("market"), row.get("kickoff_utc"), row.get("published_model_median"),
            row.get("published_model_source"))


def capture_rows(board, role, at):
    generated = utc(board["generated_at_utc"])
    captured = utc(role["captured_at_utc"])
    if generated > at or captured > at or captured < generated:
        raise ValueError("future, stale or unverified source capture")
    if (at - generated).total_seconds() > 4 * 3600:
        raise ValueError("source older than 4h: refuse retrospective firstseen")
    if role.get("schema") != "nfl-role-evidence-coverage-v1" or role.get("status") != "PREGAME_RESEARCH_ROLE_SNAPSHOT":
        raise ValueError("role receipt unverified")
    if utc(role["forecast_generated_at_utc"]) != generated:
        raise ValueError("role receipt does not match forecast")
    if (role["season"], role["week"]) != (board["season"], board["week"]):
        raise ValueError("wrong NFL week")
    verified = set()
    for row in role["player_market_evidence"]:
        if (row.get("role_evidence_complete_for_showcase") is True
            and row.get("observed_stat_rows") == 3
            and row.get("roster_status") == "ROSTER_LISTED_NOT_STARTER_VERIFIED"
            and row.get("has_verified_starting_role") is False
            and row.get("has_calibrated_book_bet_probability") is False
            and row.get("warnings") == []):
            verified.add(role_key(row))

    groups = defaultdict(list)
    for row in board["picks"]:
        if not candidate(row) or utc(row["kickoff_utc"]) <= at or utc(row["kickoff_utc"]) <= generated:
            continue
        rkey = (row["player_id"], row["team"], row["opponent"],
                row["market"], row["kickoff_utc"], row["projected_median"],
                row["model_source"])
        if rkey not in verified:
            continue
        game_key = (row["kickoff_utc"], *sorted([row["team"], row["opponent"]]))
        groups[game_key].append(row)

    digest = hashlib.sha256(json.dumps(
        board, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    selected = []
    for game, rows in sorted(groups.items()):
        rows.sort(key=lambda r: (-r["projection_within_10_probability"],
                                 -r["projection_reach_probability"],
                                 -int(r.get("games_played") or 0),
                                 r["player"], r["market"]))
        for rank, row in enumerate(rows[:10], 1):
            selected.append({
                "schema": SCHEMA, "source_sha256": digest, "season": board["season"],
                "week": board["week"], "kickoff_utc": row["kickoff_utc"],
                "player": row["player"], "player_id": row["player_id"],
                "team": row["team"], "opponent": row["opponent"],
                "market": row["market"], "model_source": row["model_source"],
                "model_median": row["projected_median"],
                "whole_projection": row["whole_projection"],
                "model_mean": row["projected_mean"],
                "projection_reach_probability": row["projection_reach_probability"],
                "projection_within_10_probability": row["projection_within_10_probability"],
                "projection_simulation_n": row["projection_simulation_n"],
                "source_generated_at_utc": board["generated_at_utc"],
                "role_captured_at_utc": role["captured_at_utc"],
                "recorded_at_utc": at.isoformat(), "frontend_research_rank": rank,
                "bet_probability_calibrated": False,
                "official_active_status_verified": False,
                "selection_basis": "TEN_YARD_NUMERICAL_ACCURACY_BAND_UNCALIBRATED",
            })
    return selected


def capture(board, role, at, log_path):
    incoming = capture_rows(board, role, at)
    existing = []
    if log_path.exists():
        existing = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    seen = {key(r) for r in existing}
    if len(seen) != len(existing):
        raise ValueError("duplicate existing immutable keys")
    new = []
    for row in incoming:
        if key(row) not in seen:
            new.append(row)
            seen.add(key(row))
    if new:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as file:
            for row in new:
                file.write(json.dumps(row, sort_keys=True) + "\n")
    return {"available_qualified": len(incoming), "new_firstseen": len(new),
            "total_immutable": len(seen), "source_generated_at_utc": board["generated_at_utc"]}


def grade(log_rows, db):
    # SQL contains only genuine postgame player stats. A missing player row
    # is ambiguous (injury, absent, provider lag); NEVER assume zero.
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    by_week = defaultdict(list)
    for row in log_rows:
        by_week[(row["season"], row["week"])].append(row)
    # Event-safe ID matching followed by a STRICT, unique name+team+opponent
    # crosswalk for ESPN-vs-GSIS ID namespaces. Never join by name alone,
    # choose an arbitrary duplicate, or conflate a player's prior week.
    by_id, by_name = {}, {}
    for (season, week) in by_week:
        found = con.execute(
            "SELECT player_id, player_name, team, opponent, rushing_yards, receiving_yards "
            "FROM player_games WHERE season=? AND week=?", (season, week)).fetchall()
        for row in found:
            idkey = (season, week, str(row["player_id"]), row["team"], row["opponent"])
            namekey = (season, week, normalized_name(row["player_name"]), row["team"], row["opponent"])
            by_id.setdefault(idkey, []).append(row)
            by_name.setdefault(namekey, []).append(row)
    con.close()
    results = []
    for p in log_rows:
        matches = by_id.get((p["season"], p["week"], str(p["player_id"]), p["team"], p["opponent"]), [])
        match_method = "EXACT_PLAYER_ID_GAME_TEAM"
        if len(matches) != 1:
            matches = by_name.get((p["season"], p["week"], normalized_name(p["player"]),
                                   p["team"], p["opponent"]), [])
            match_method = "UNIQUE_PLAYER_NAME_GAME_TEAM"
        if len(matches) != 1:
            results.append({"key": key(p), "result": "UNRESOLVED_ID_OR_MISSING_BOX",
                            "market": p["market"]})
            continue
        value = matches[0][p["market"]]
        if value is None:
            results.append({"key": key(p), "result": "UNRESOLVED_MISSING_STAT", "market": p["market"]})
            continue
        err = abs(float(value) - p["whole_projection"])
        reach = int(float(value) >= p["whole_projection"])
        near = int(err <= 10)
        results.append({
            "key": key(p), "result": "GRADED", "market": p["market"],
            "player": p["player"], "week": p["week"], "actual": float(value),
            "identity_method": match_method,
            "forecast": p["whole_projection"], "abs_error": err,
            "within_10": near, "reach_own_projection": reach,
            "p_within_10": p["projection_within_10_probability"],
            "p_reach": p["projection_reach_probability"],
            "brier_within_10": (near-p["projection_within_10_probability"])**2,
            "brier_reach": (reach-p["projection_reach_probability"])**2,
        })
    graded = [x for x in results if x["result"] == "GRADED"]
    summary = {"n_firstseen": len(log_rows), "n_graded": len(graded),
               "n_unresolved": len(results)-len(graded),
               "n_graded_name_crosswalk": sum(
                   r["identity_method"] == "UNIQUE_PLAYER_NAME_GAME_TEAM" for r in graded),
               "mean_abs_error_yards": round(sum(x["abs_error"] for x in graded)/len(graded), 3) if graded else None,
               "observed_within_10_rate": round(sum(x["within_10"] for x in graded)/len(graded), 4) if graded else None,
               "predicted_within_10_mean": round(sum(x["p_within_10"] for x in graded)/len(graded), 4) if graded else None,
               "within_10_brier": round(sum(x["brier_within_10"] for x in graded)/len(graded), 4) if graded else None}
    return {"schema": "NFL_NUMERIC_PRECISION_FORWARD_GRADE_V1",
            "not_a_calibrated_betting_system": True, "summary": summary, "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("operation", choices=("capture", "grade"))
    ap.add_argument("--board", type=Path, default=DOCS/"nfl_predictions.json")
    ap.add_argument("--role", type=Path, default=DOCS/"nfl_role_snapshot.json")
    ap.add_argument("--log", type=Path, default=LOG)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    if args.operation == "capture":
        result = capture(json.loads(args.board.read_text()),
                         json.loads(args.role.read_text()), now, args.log)
    else:
        rows = [json.loads(line) for line in args.log.read_text().splitlines()
                if line.strip()] if args.log.exists() else []
        result = grade(rows, args.db)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result.get("summary", result), sort_keys=True))


if __name__ == "__main__":
    main()
