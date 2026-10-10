#!/usr/bin/env python3
"""Prospective CFB 2026 empirical play-level SHADOW only.

Never serve to the user-facing board. Derive an independent stat distribution
from verified *same-player, same-team, already-completed* 2026 play histories.
Every included game must reconcile counts + totals with the independently
refreshed ESPN player-game boxscore. Forecast creation uses the real process
clock and MUST precede kickoff. No prior 2026 archived pick is backfilled.
This is a non-promoted empirical benchmark, not a certified calibrated model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cfb_rush_sim as SIM
import cfb_2026_event_source_qualify_a as Q

SHADOW_SCHEMA = "CFB_2026_PER_EVENT_SHADOW_V1"
RESEARCH_STATUS = "NEW_PREKICKOFF_RESEARCH_ONLY_NOT_PRODUCTION"
CFG = {
    "rushing_yards": {"position": "RB", "table": "rush_carries",
                      "value_col": "yards", "order_col": "carry_index",
                      "min_volume": 12, "stat_unit": "yards"},
    "passing_touchdowns": {"position": "QB", "table": "pass_attempts_log",
                           "value_col": "is_touchdown", "order_col": "attempt_index",
                           "min_volume": 15, "stat_unit": "passing_touchdowns"},
}


def _verified_game_history(model_con, event_con, market, player, team, week,
                           finals, source_games):
    cfg = CFG[market]
    table, value_col, volume_col, target_col, position = Q.EVENT_TABLES[market]
    eligible = set(finals) & source_games
    event, source_warning = Q._event_stats(event_con, table, value_col, 2026, eligible)
    espn = Q._model_stats(model_con, market, volume_col, target_col, position, 2026, eligible)
    owners = {(str(pid), str(gid)): str(owner) for pid, gid, owner in model_con.execute(
        "SELECT player_id, game_id, team FROM player_games WHERE season=2026"
    )}
    valid = []
    for (pid, gid), event_summary in event.items():
        if pid != str(player) or espn.get((pid, gid)) != event_summary:
            continue
        if owners.get((pid, gid)) != str(team):
            continue
        if event_summary["week"] >= week:
            continue
        valid.append((event_summary["week"], gid, event_summary["count"], event_summary["value"]))
    valid.sort(key=lambda r: (r[0], r[1]))
    newest = valid[-8:]
    if len(newest) < 3:
        return None, "FEWER_THAN_3_INDEPENDENTLY_RECONCILED_SAME_TEAM_PRIOR_GAMES"
    counts = [row[2] for row in newest]
    if sum(counts[-3:]) / 3 < cfg["min_volume"]:
        return None, "RECENT3_EVENT_WORKLOAD_BELOW_ORIGINAL_PILOT_GATE"
    pool = []
    for _, gid, count, _ in newest:
        samples = event_con.execute(
            f"SELECT {cfg['value_col']} FROM {cfg['table']} WHERE season=? AND player_id=? "
            f"AND game_id=? ORDER BY {cfg['order_col']}", (2026, str(player), str(gid)),
        ).fetchall()
        if len(samples) != count:
            return None, "EVENT_COUNT_DOES_NOT_MATCH_VERIFIED_GROUP"
        pool.extend(float(x[0]) for x in samples)
    if not pool:
        return None, "NO_PRIOR_EVENT_OUTCOME_POOL"
    # The original team's identity is from ESPN player_games, not from
    # cfbfastR's mixed possession/defense 'team' labels on interceptions.
    return {"counts": counts, "pool": pool, "games": [x[1] for x in newest],
            "mixed_source_team_groups_in_population": len(source_warning)}, None


def shadow(model_con, event_con, board, qualification, schedule_csv, *,
           player_csv=None, generated_at=None, n_simulations=4000):
    at = Q.utc(generated_at or datetime.now(timezone.utc))
    if qualification.get("schema") != Q.SCHEMA:
        raise Q.QualificationError("SOURCE_QUALIFICATION_SCHEMA_REQUIRED")
    received = Q.utc(qualification.get("captured_at_utc"))
    files = qualification.get("raw_source_files") or {}
    if Q.digest(schedule_csv) != files.get("schedule", {}).get("sha256"):
        raise Q.QualificationError("SOURCE_SCHEDULE_HASH_MISMATCH")
    if player_csv is not None and Q.digest(player_csv) != files.get("player_stats", {}).get("sha256"):
        raise Q.QualificationError("SOURCE_PLAYER_EVENTS_HASH_MISMATCH")
    if received > at:
        raise Q.QualificationError("SOURCE_ATTESTED_IN_THE_FUTURE")
    if n_simulations < 100 or n_simulations > 250000:
        raise Q.QualificationError("INVALID_SHADOW_SIM_COUNT")
    source_games, kickoffs = Q.source_games(schedule_csv)
    _, finals = Q._final_games(model_con, 2026, at)
    schedule = {str(row["game_id"]): dict(row) for row in model_con.execute(
        "SELECT game_id, week, season, kickoff_utc, home_team, away_team "
        "FROM games WHERE season=2026")}
    live_states = {str(gid): state for gid, state in model_con.execute(
        "SELECT game_id, espn_state FROM schedule_snapshot WHERE season=2026")}
    rows, skipped, seen = [], Counter(), set()
    for pick in board.get("picks", []):
        market = pick.get("market")
        if market not in CFG or pick.get("season") != 2026:
            continue
        pid, gid = str(pick.get("player_id")), str(pick.get("game_id"))
        k = (gid, pid, market)
        if k in seen:
            skipped["DUPLICATE_CURRENT_BOARD_KEY"] += 1
            continue
        seen.add(k)
        game = schedule.get(gid)
        if game is None or gid not in source_games:
            skipped["GAME_NOT_VERIFIED_FBS_FBS"] += 1
            continue
        kickoff = Q.utc(game["kickoff_utc"])
        src_kick = kickoffs.get(gid)
        if not src_kick or abs((kickoff - src_kick).total_seconds()) > 120:
            skipped["SOURCE_AND_ESPN_KICKOFF_DISAGREE"] += 1
            continue
        if at >= kickoff or received >= kickoff or live_states.get(gid) != "pre":
            skipped["NOT_GENUINELY_PREGAME"] += 1
            continue
        if pick.get("team") not in (game["home_team"], game["away_team"]):
            skipped["TEAM_NOT_IN_GAME"] += 1
            continue
        if not isinstance(game["week"], int):
            skipped["NO_WEEK_IDENTITY"] += 1
            continue
        history, why = _verified_game_history(model_con, event_con, market,
                                              pid, pick["team"], game["week"],
                                              finals, source_games)
        if history is None:
            skipped[why] += 1
            continue
        line = pick.get("line")
        if not isinstance(line, (int, float)) or isinstance(line, bool) or not math.isfinite(line):
            skipped["INVALID_FIXED_LINE"] += 1
            continue
        seed_key = "|".join(map(str, (gid, pid, market, qualification["raw_source_files"]["player_stats"]["sha256"])))
        seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
        result = SIM.simulate(history["counts"], history["pool"], float(line),
                              sims=n_simulations, rng=np.random.RandomState(seed))
        if result is None:
            skipped["NO_SIMULATION_RESULT"] += 1
            continue
        rows.append({
            "game_id": gid, "player_id": pid, "team": pick["team"],
            "market": market, "unit": CFG[market]["stat_unit"],
            "week": game["week"], "kickoff_utc": kickoff.isoformat().replace("+00:00", "Z"),
            "forecast_generated_at_utc": at.isoformat().replace("+00:00", "Z"),
            "model_name": "EXISTING_EMPIRICAL_2025_HOLDOUT_PLAIN_SIM_SHADOW",
            "projected_mean": result["mean"], "projected_median": result["median"],
            "p10": result["p10"], "p90": result["p90"],
            "p_over_fixed_line": result["prob_over"],
            "original_classifier_line": line,
            "n_simulations": n_simulations,
            "prior_verified_same_team_game_ids": history["games"],
            "source_sha256": qualification["raw_source_files"]["player_stats"]["sha256"],
            "decision_source": "EMPIRICAL_EVENT_DISTRIBUTION_NOT_CLASSIFIER_SIDE",
            "not_a_betting_recommendation": True,
            "historical_original_prediction": False,
            "research_only": True,
        })
    return {
        "schema": SHADOW_SCHEMA, "status": RESEARCH_STATUS,
        "generated_at_utc": at.isoformat().replace("+00:00", "Z"),
        "source_qualification_status": qualification["operational_source_status"],
        "forward_original_ledger_modified": False,
        "suspension_or_champion_changed": False,
        "historical_backfills": 0,
        "shadow_predictions": len(rows),
        "shadow": sorted(rows, key=lambda r: (r["kickoff_utc"], r["game_id"], r["market"], r["player_id"])),
        "excluded": dict(skipped),
        "scientific_caveat": "Paper-only new pregame distributions, never calibrated/promoted, no 2026 outcome feedback used for parameters.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-db", type=Path, required=True)
    p.add_argument("--event-db", type=Path, required=True)
    p.add_argument("--board", type=Path, default=Path("docs/cfb_predictions.json"))
    p.add_argument("--schedule-csv", type=Path, required=True)
    p.add_argument("--player-csv", type=Path, required=True)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    with sqlite3.connect(f"file:{a.model_db}?mode=ro", uri=True) as model, sqlite3.connect(
        f"file:{a.event_db}?mode=ro", uri=True
    ) as event:
        model.row_factory = sqlite3.Row
        event.row_factory = sqlite3.Row
        result = shadow(model, event, json.loads(a.board.read_text()),
                        json.loads(a.qualification.read_text()), a.schedule_csv,
                        player_csv=a.player_csv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "shadow_predictions": result["shadow_predictions"],
                      "excluded": result["excluded"], "historical_backfills": 0}))
    # A future slate with qualifying market candidates and zero research
    # forecasts is an explicit blocked outcome, not a successful pilot.
    if not result["shadow"] and any(
        p.get("market") in CFG and p.get("season") == 2026
        for p in json.loads(a.board.read_text()).get("picks", [])
    ):
        raise SystemExit("SHADOW_ZERO_FORECASTS_WITH_PREGAME_CANDIDATES_BLOCKED")


if __name__ == "__main__":
    main()
