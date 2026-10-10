#!/usr/bin/env python3
"""CFB 2026 per-play data *qualification*, NOT a prediction or model promotion.

Inspect a timestamped contemporary cfbfastR player_stats snapshot against a
separately refreshed ESPN game/box-score SQLite snapshot. Both ultimately use
ESPN information: these are independent *transformations*, not independent
underlying observational sources.

The existing extractors construct rush_carries, pass_attempts_log, and
recv_catches in a TEMPORARY event DB. This module refuses to infer pregame
availability of data at any historical 2026 forecast cutoff from a snapshot
downloaded today. It does not rewrite a single canonical forecast or
retroactively grade simulated point projections.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "CFB_2026_EVENT_QUALIFICATION_V1"
SOURCE_BASE = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main"
NO_HISTORICAL_REPLAY = "CURRENT_FETCH_HAS_NO_VERIFIED_HISTORICAL_PUBLICATION_VINTAGE"

EVENT_TABLES = {
    "rushing_yards": ("rush_carries", "yards", "carries", "rushing_yards", "RB"),
    "passing_touchdowns": ("pass_attempts_log", "is_touchdown", "pass_attempts", "passing_touchdowns", "QB"),
    "passing_yards": ("pass_attempts_log", "yards", "pass_attempts", "passing_yards", "QB"),
    "receiving_yards": ("recv_catches", "yards", "receptions", "receiving_yards", "WR"),
}

class QualificationError(ValueError):
    pass


def utc(ts):
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            raise QualificationError("NAIVE_RECEIPT_CLOCK")
        return ts.astimezone(timezone.utc)
    if not isinstance(ts, str):
        raise QualificationError("MISSING_TIME")
    try:
        x = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError as exc:
        raise QualificationError("INVALID_TIME") from exc
    if x.tzinfo is None:
        raise QualificationError("NAIVE_RECEIPT_CLOCK")
    return x.astimezone(timezone.utc)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def source_games(schedule_csv, season=2026):
    """Only regular season FBS-vs-FBS; do not widen a frozen pilot's scope."""
    ids, kickoff = set(), {}
    with open(schedule_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("home_division") or "").lower() != "fbs" or (row.get("away_division") or "").lower() != "fbs":
                continue
            if row.get("season_type") != "regular" or str(row.get("season")) != str(season):
                continue
            gid = str(row.get("game_id") or "")
            if not gid or gid in ids:
                raise QualificationError("MISSING_OR_DUPLICATE_SOURCE_GAME")
            ids.add(gid)
            if row.get("start_date"):
                kickoff[gid] = utc(row["start_date"])
    if not ids:
        raise QualificationError("NO_FBS_FBS_SOURCE_GAMES")
    return ids, kickoff


def _final_games(con, season, receipt):
    try:
        columns = {x[1] for x in con.execute("PRAGMA table_info(schedule_snapshot)")}
    except sqlite3.Error:
        columns = set()
    status = {}
    if {"game_id", "espn_status"} <= columns:
        status = {str(g): s for g, s in con.execute(
            "SELECT game_id, espn_status FROM schedule_snapshot WHERE season=?", (season,))}
    all_games, finished = {}, {}
    for r in con.execute(
        "SELECT game_id, season, week, kickoff_utc, home_team, away_team, home_points, away_points "
        "FROM games WHERE season=?", (season,)):
        gid = str(r["game_id"])
        all_games[gid] = dict(r)
        if r["home_points"] is None or r["away_points"] is None:
            continue
        if status and status.get(gid) not in ("STATUS_FINAL", "STATUS_FINAL_OVERTIME"):
            continue
        kickoff = utc(r["kickoff_utc"]) if r["kickoff_utc"] else None
        if kickoff is None or kickoff >= receipt:
            continue
        finished[gid] = dict(r)
    return all_games, finished


def _event_stats(con, table, value_col, season, eligible_games):
    if table not in {"rush_carries", "recv_catches", "pass_attempts_log"} or value_col not in {"yards", "is_touchdown"}:
        raise QualificationError("UNTRUSTED_EVENT_TABLE")
    out = {}
    for player_id, gid, week, count, total, minimum, maximum, distinct_teams in con.execute(
        f"""SELECT player_id, game_id, week, COUNT(*), SUM({value_col}),
            MIN({value_col}), MAX({value_col}), COUNT(DISTINCT team)
            FROM {table} WHERE season=?
            GROUP BY player_id, game_id, week""", (season,)):
        gid = str(gid)
        if gid not in eligible_games:
            continue
        key = (str(player_id), gid)
        if key in out:
            raise QualificationError("DUPLICATE_PLAYER_GAME_WEEK_IN_EVENT_SOURCE")
        if minimum is None or maximum is None or minimum < -120 and value_col == "yards":
            raise QualificationError("INVALID_EVENT_STAT")
        if value_col == "is_touchdown" and (minimum < 0 or maximum > 1):
            raise QualificationError("NON_BINARY_PASS_TOUCHDOWN_EVENT")
        if distinct_teams != 1:
            raise QualificationError("CROSS_TEAM_SAME_PLAYER_GAME_EVENTS")
        out[key] = {"week": int(week), "count": int(count), "value": int(total)}
    return out


def _model_stats(con, market, volume_field, target_field, position, season, eligible_games):
    out = {}
    for r in con.execute(
        f"""SELECT player_id, game_id, week, {volume_field} AS volume,
            {target_field} AS target FROM player_games
            WHERE season=? AND position=?""", (season, position)):
        gid = str(r["game_id"])
        if gid not in eligible_games:
            continue
        key = (str(r["player_id"]), gid)
        if key in out:
            raise QualificationError("DUPLICATE_ESPN_PLAYER_GAME")
        out[key] = {
            "week": int(r["week"]),
            "count": int(r["volume"] or 0),
            "value": int(r["target"] or 0),
        }
    return out


def _market_comparison(event_rows, model_rows):
    eligible = {key: row for key, row in model_rows.items() if row["count"] > 0}
    exact = 0
    missing, mismatches = [], []
    for key, e in sorted(eligible.items()):
        o = event_rows.get(key)
        if o is None:
            missing.append(key)
        elif e != o:
            mismatches.append({"player_id": key[0], "game_id": key[1],
                               "event": o, "espn": e})
        else:
            exact += 1
    orphan = sorted(key for key in event_rows if key not in model_rows)
    total = len(eligible)
    return {
        "espn_player_games_with_positive_volume": total,
        "event_matched_player_games": exact,
        "event_missing_player_games": len(missing),
        "event_mismatched_player_games": len(mismatches),
        "event_unmatched_espn_player_games": len(orphan),
        "exact_match_fraction": round(exact / total, 6) if total else None,
        "missing_samples": [{"player_id": p, "game_id": g} for p, g in missing[:10]],
        "mismatch_samples": mismatches[:10],
        "unmatched_event_samples": [{"player_id": p, "game_id": g} for p, g in orphan[:10]],
    }


def qualifying_prior_event_games(ledger, event_con, finished_games, receipt, season):
    """Research-only future opportunity coverage; never backfill logged projections.

    A contemporary source snapshot may support a future player forecast only
    when RECEIVED before kickoff and each source game was independently final.
    """
    results = {"rushing_yards": {"picks": 0, "source_eligible": 0},
               "passing_touchdowns": {"picks": 0, "source_eligible": 0}}
    recent = {}
    for mkt in results:
        table, val, _, _, _ = EVENT_TABLES[mkt]
        by_player = defaultdict(set)
        for pid, gid, wk in event_con.execute(
            f"SELECT player_id, game_id, week FROM {table} WHERE season=?", (season,)):
            if str(gid) in finished_games:
                by_player[str(pid)].add((int(wk), str(gid)))
        recent[mkt] = by_player
    for p in ledger:
        mkt = p.get("market")
        if mkt not in results or int(p.get("season", -1)) != season:
            continue
        results[mkt]["picks"] += 1
        # No historical cutoff replays from a presently downloaded file.
        kickoff = utc(p.get("kickoff_utc")) if p.get("kickoff_utc") else None
        if kickoff is None or kickoff <= receipt:
            continue
        pid, week = str(p.get("player_id")), p.get("week")
        if not isinstance(week, int):
            continue
        if len({gid for w, gid in recent[mkt].get(pid, ()) if w < week}) >= 3:
            results[mkt]["source_eligible"] += 1
    return results


def qualify(model_con, event_con, schedule_csv, player_csv, ledger, *, received_at=None):
    receipt = utc(received_at or datetime.now(timezone.utc))
    files = {
        "player_stats": {"sha256": digest(player_csv), "size_bytes": player_csv.stat().st_size,
                         "url": SOURCE_BASE + "/player_stats/csv/player_stats_2026.csv"},
        "schedule": {"sha256": digest(schedule_csv), "size_bytes": schedule_csv.stat().st_size,
                     "url": SOURCE_BASE + "/schedules/csv/cfb_schedules_2026.csv"},
    }
    source_ids, source_kicks = source_games(schedule_csv)
    all_games, finals = _final_games(model_con, 2026, receipt)
    both = set(finals) & source_ids
    comparisons = {}
    for market, (table, value_col, volume_col, target_col, position) in EVENT_TABLES.items():
        es = _event_stats(event_con, table, value_col, 2026, both)
        ms = _model_stats(model_con, market, volume_col, target_col, position, 2026, both)
        comparisons[market] = _market_comparison(es, ms)
    c = qualifying_prior_event_games(ledger, event_con, finals, receipt, 2026)
    overlap = bool(both)
    mismatched = any(x["event_missing_player_games"] or x["event_mismatched_player_games"]
                     for x in comparisons.values())
    return {
        "schema": SCHEMA,
        "captured_at_utc": receipt.isoformat().replace("+00:00", "Z"),
        "raw_source_files": files,
        "source_data_author": "sportsdataverse/cfbfastR-data",
        "upstream_independence": "NOT_INDEPENDENT_BOTH_ESPN_DERIVED",
        "source_game_scope": "2026_FBS_VS_FBS_REGULAR_ONLY",
        "source_games_total": len(source_ids),
        "espn_games_total": len(all_games),
        "espn_confirmed_final_games": len(finals),
        "overlapping_completed_games": len(both),
        "overlap_game_ids_first10": sorted(both)[:10],
        "comparisons": comparisons,
        "potential_future_picks_only": c,
        "historic_forecast_replay": "BLOCKED: " + NO_HISTORICAL_REPLAY,
        "historical_source_publication_vintage_verified": False,
        "production_predictions_edited": False,
        "live_model_changed": False,
        "promoted": False,
        "operational_source_status": ("NO_COMPLETED_GAME_OVERLAP" if not overlap
                                      else "EVENT_RECONCILIATION_MISMATCH" if mismatched
                                      else "MATCHED_OVERLAP_NOT_HISTORICALLY_VINTAGED"),
        "research_only": True,
        "eligible_to_overwrite_original_pregame_forecasts": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-db", type=Path, required=True)
    parser.add_argument("--event-db", type=Path, required=True)
    parser.add_argument("--schedule-csv", type=Path, required=True)
    parser.add_argument("--player-csv", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=Path("docs/cfb_picks_log.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with sqlite3.connect(f"file:{args.model_db}?mode=ro", uri=True) as model, sqlite3.connect(
        f"file:{args.event_db}?mode=ro", uri=True
    ) as event:
        model.row_factory = sqlite3.Row
        event.row_factory = sqlite3.Row
        entries = [json.loads(line) for line in args.ledger.read_text().splitlines() if line.strip()]
        data = qualify(model, event, args.schedule_csv, args.player_csv, entries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "source_status": data["operational_source_status"],
        "overlapping_completed_games": data["overlapping_completed_games"],
        "comparisons": {m: {"exact": x["event_matched_player_games"],
                            "missing": x["event_missing_player_games"],
                            "mismatch": x["event_mismatched_player_games"]}
                        for m, x in data["comparisons"].items()},
        "historical_replay": "BLOCKED",
        "new_predictions": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
