#!/usr/bin/env python3
"""
NHL_GRADE_RECORD_A

Grades every logged NHL pick (docs/nhl_picks_log.jsonl, appended to by
nhl_serving_builder_a.py's PICKS_LOG_PATH -- logged BEFORE the finished-
game filter drops a pick from the live board) against real outcomes, and
writes docs/nhl_record.json in the same summary/results shape every
other sport's record.json uses.

Two real, different grading paths -- NOT one shared "did the team win"
rule (an earlier version of this file only had the team-level rule,
which would have silently mis-graded every points/shots_on_goal/
goalie_saves pick as if it were a moneyline bet):

  moneyline / moneyline_early_season   team-level: hit iff the picked
                                        team actually won, looked up by
                                        (game_date, team, opponent)
                                        against games.home_team/
                                        away_team + home_score/away_score.
  points / shots_on_goal / goalie_saves   player-level: hit iff the
                                        player's real stat line that game
                                        clears the pick's own OVER/UNDER
                                        (or, for points' one-sided
                                        "OVER 0.5" anytime design, iff
                                        actual points >= 1), looked up by
                                        (player_id, game_date) against
                                        skater_games/goalie_games.

A pick is gradable once its real game/stat-line row exists with a final
result -- ungraded (not yet played/ingested) picks are silently skipped,
not counted as a miss.

Read-only against the DB and the ledger; only ever writes
docs/nhl_record.json.

Run
---
python -u nhl_grade_record_a.py
"""
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

REPO = Path(__file__).resolve().parent
DB_DEFAULT = REPO / "nhl_models" / "nhl_model.sqlite"
DOCS = REPO / "docs"
LOG_DEFAULT = DOCS / "nhl_picks_log.jsonl"
OUT_DEFAULT = DOCS / "nhl_record.json"

TEAM_LEVEL_MARKETS = {"moneyline", "moneyline_early_season"}
# An "_early_season" pick is graded against the exact same real-world stat
# as its in-season counterpart -- only the model that produced the pick
# differs -- so both variants map to the same (table, stat_col) here.
# Real bug found and fixed here: an earlier version of this dict only had
# the in-season keys, so every points_early_season/shots_on_goal_early_
# season pick silently vanished from grading entirely (filtered out of
# BOTH team_picks and player_picks, never counted as graded OR ungraded)
# instead of just waiting for its real game to be played.
PLAYER_STAT_COLUMN = {"points": ("skater_games", "points"),
                       "points_early_season": ("skater_games", "points"),
                       "shots_on_goal": ("skater_games", "shots"),
                       "shots_on_goal_early_season": ("skater_games", "shots"),
                       "goalie_saves": ("goalie_games", "saves")}


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_ledger(path):
    picks = []
    if not path.exists():
        return picks
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            picks.append(json.loads(line))
        except Exception:
            continue
    return picks


def grade_over_under(pick_str, actual):
    m = re.search(r"(OVER|UNDER)\s+(-?\d+(?:\.\d+)?)", pick_str.upper())
    if not m:
        return None
    side, line = m.group(1), float(m.group(2))
    return "hit" if (actual > line if side == "OVER" else actual < line) else "miss"


def main():
    print("NHL_GRADE_RECORD_A\n==================")
    ledger = load_ledger(LOG_DEFAULT)
    print(f"ledger: {len(ledger)} logged picks")
    if not ledger:
        OUT_DEFAULT.write_text(json.dumps({
            "summary": {"total": 0, "hits": 0, "misses": 0, "hit_rate": 0},
            "by_market": {}, "results": [], "last_updated": now_utc(),
        }, indent=2))
        print(f"no ledger entries yet -- wrote empty record to {OUT_DEFAULT}")
        return 0

    con = sqlite3.connect(f"file:{DB_DEFAULT}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    team_picks = [p for p in ledger if p.get("market") in TEAM_LEVEL_MARKETS]
    player_picks = [p for p in ledger if p.get("market") in PLAYER_STAT_COLUMN]

    dates = {p["game_date"] for p in team_picks}
    final_by_key = {}
    if dates:
        placeholders = ",".join("?" for _ in dates)
        rows = con.execute(
            f"SELECT * FROM games WHERE game_date IN ({placeholders}) "
            f"AND home_score IS NOT NULL AND away_score IS NOT NULL",
            tuple(dates)).fetchall()
        for r in rows:
            final_by_key[(r["game_date"], r["home_team"], r["away_team"])] = r

    # Player-level actual stats, keyed by (table, player_id, game_date) --
    # both tables share the same shape closely enough to query generically.
    actual_by_key = {}
    for table in {"skater_games", "goalie_games"} & {t for t, _ in PLAYER_STAT_COLUMN.values()}:
        pids = {p["player_id"] for p in player_picks
                if PLAYER_STAT_COLUMN.get(p.get("market"), (None,))[0] == table}
        if not pids:
            continue
        placeholders = ",".join("?" for _ in pids)
        rows = con.execute(
            f"SELECT * FROM {table} WHERE player_id IN ({placeholders})", tuple(pids)).fetchall()
        for r in rows:
            actual_by_key[(table, r["player_id"], r["game_date"])] = r
    con.close()

    results = []
    ungraded = 0

    for p in team_picks:
        team, opp, gdate = p.get("team"), p.get("opponent"), p.get("game_date")
        row = final_by_key.get((gdate, team, opp)) or final_by_key.get((gdate, opp, team))
        if row is None:
            ungraded += 1
            continue
        team_won = ((row["home_team"] == team and row["home_score"] > row["away_score"]) or
                    (row["away_team"] == team and row["away_score"] > row["home_score"]))
        results.append({
            "season": p["season"], "game_date": gdate, "market": p["market"],
            "team": team, "opponent": opp, "pick": p.get("pick"),
            "actual": f"{row['home_team']} {row['home_score']}-{row['away_score']} {row['away_team']}",
            "result": "hit" if team_won else "miss",
            "model_prob": p.get("model_prob"), "model_source": p.get("model_source"),
        })

    for p in player_picks:
        table, stat_col = PLAYER_STAT_COLUMN[p["market"]]
        row = actual_by_key.get((table, p.get("player_id"), p.get("game_date")))
        if row is None:
            ungraded += 1
            continue
        actual = row[stat_col]
        result = grade_over_under(p.get("pick", ""), actual)
        if result is None:
            continue
        results.append({
            "season": p["season"], "game_date": p["game_date"], "market": p["market"],
            "player": p.get("player"), "player_id": p.get("player_id"),
            "team": p.get("team"), "opponent": p.get("opponent"),
            "pick": p.get("pick"), "line": p.get("line"), "actual": actual, "result": result,
            "model_prob": p.get("model_prob"), "model_source": p.get("model_source"),
        })

    print(f"graded: {len(results)}  ungraded (game/stat not yet played/ingested): {ungraded}")

    total = len(results)
    hits = sum(1 for r in results if r["result"] == "hit")
    by_market = {}
    for r in results:
        mk = r["market"]
        by_market.setdefault(mk, {"hits": 0, "total": 0})
        by_market[mk]["total"] += 1
        by_market[mk]["hits"] += 1 if r["result"] == "hit" else 0

    record = {
        "summary": {"total": total, "hits": hits, "misses": total - hits,
                     "hit_rate": round(hits / total * 100, 1) if total else 0},
        "by_market": {
            k: {**v, "hit_rate": round(v["hits"] / v["total"] * 100, 1) if v["total"] else 0}
            for k, v in by_market.items()
        },
        "results": sorted(results, key=lambda r: r["game_date"], reverse=True),
        "last_updated": now_utc(),
    }
    OUT_DEFAULT.write_text(json.dumps(record, indent=2))
    print(f"\n{hits}/{total} ({record['summary']['hit_rate']}%) written to {OUT_DEFAULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
