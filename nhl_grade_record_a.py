#!/usr/bin/env python3
"""
NHL_GRADE_RECORD_A

Grades every logged NHL pick (docs/nhl_picks_log.jsonl, appended to by
nhl_serving_builder_a.py's PICKS_LOG_PATH -- logged BEFORE the finished-
game filter drops a pick from the live board) against real outcomes in
nhl_model.sqlite's games table, and writes docs/nhl_record.json in the
same summary/results shape every other sport's record.json uses.

Moneyline is team-level (not a per-player OVER/UNDER line), so grading
is simpler than CFB's regex-on-"OVER/UNDER N" parse: a pick of
"{team} ML" is a hit iff that team actually won the real game, looked up
by (season, game_date, team, opponent) against games.home_team/away_team
+ home_score/away_score. A pick is gradable once that real game has a
final score in the DB -- ungraded (not yet played/ingested) picks are
silently skipped, not counted as a miss.

Read-only against the DB and the ledger; only ever writes
docs/nhl_record.json.

Run
---
python -u nhl_grade_record_a.py
"""
import json
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

    dates = {p["game_date"] for p in ledger}
    placeholders = ",".join("?" for _ in dates)
    rows = con.execute(
        f"SELECT * FROM games WHERE game_date IN ({placeholders}) "
        f"AND home_score IS NOT NULL AND away_score IS NOT NULL",
        tuple(dates)).fetchall()
    con.close()
    final_by_key = {}
    for r in rows:
        final_by_key[(r["game_date"], r["home_team"], r["away_team"])] = r

    results = []
    ungraded = 0
    for p in ledger:
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

    print(f"graded: {len(results)}  ungraded (game not yet played/ingested): {ungraded}")

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
