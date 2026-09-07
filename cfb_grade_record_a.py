#!/usr/bin/env python3
"""
CFB_GRADE_RECORD_A

Grades every logged CFB pick (docs/cfb_picks_log.jsonl, appended to by
cfb_serving_builder_a.py's PICKS_LOG_PATH -- logged BEFORE the finished-
game filter drops a pick from the live board, see that file's docstring
for why the per-week archive alone can't be used for this) against real
outcomes in cfb_model.sqlite's player_games table, and writes
docs/cfb_record.json in the same summary/results shape MLB's record.json
uses (api.py's update_record()), adapted for CFB: segmented by market
instead of by_prop, no book-odds or bvp/confidence buckets since CFB
picks don't carry those fields (predictions-first, no odds).

A pick is gradable once player_games has a row for (player_id, season,
week) -- that row only exists once the real game has been played and
ingested, so no separate schedule-final check is needed here (unlike
MLB, which has to poll a live API and ask explicitly). Ungraded picks
(game not yet played/ingested) are silently skipped, not counted as a
miss.

Read-only against the DB and the ledger; only ever writes
docs/cfb_record.json.

Run
---
python -u cfb_grade_record_a.py
"""
import argparse
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
DB_DEFAULT = REPO / "cfb_models" / "cfb_model.sqlite"
DOCS = REPO / "docs"
LOG_DEFAULT = DOCS / "cfb_picks_log.jsonl"
OUT_DEFAULT = DOCS / "cfb_record.json"

# Actual-outcome column per base market (an "_early_season" pick is
# graded against the exact same real-world stat as its in-season
# counterpart -- only the model that produced the pick differs).
# anytime_touchdowns sums two raw columns, same as the in-season
# AnytimeTouchdownEngine/prior-season bootstrap both do.
MARKET_STAT_COLUMN = {
    "rushing_yards": "rushing_yards",
    "passing_touchdowns": "passing_touchdowns",
    "receiving_yards": "receiving_yards",
    "passing_yards": "passing_yards",
    "rushing_touchdowns": "rushing_touchdowns",
    "receiving_touchdowns": "receiving_touchdowns",
}


def actual_stat(market, row):
    base = market.replace("_early_season", "")
    if base == "anytime_touchdowns":
        return (row["rushing_touchdowns"] or 0) + (row["receiving_touchdowns"] or 0)
    col = MARKET_STAT_COLUMN.get(base)
    return row[col] if col else None


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


def grade(pick, actual):
    p = str(pick.get("pick", "")).upper().strip()
    m = re.search(r"(OVER|UNDER)\s+(-?\d+(?:\.\d+)?)", p)
    if not m:
        return None
    side, line = m.group(1), float(m.group(2))
    return "hit" if (actual > line if side == "OVER" else actual < line) else "miss"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_DEFAULT))
    ap.add_argument("--log", default=str(LOG_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    print("CFB_GRADE_RECORD_A\n==================")
    ledger = load_ledger(Path(args.log))
    print(f"ledger: {len(ledger)} logged picks")
    if not ledger:
        Path(args.out).write_text(json.dumps({
            "summary": {"total": 0, "hits": 0, "misses": 0, "hit_rate": 0},
            "by_market": {}, "results": [], "last_updated": now_utc(),
        }, indent=2))
        print(f"no ledger entries yet -- wrote empty record to {args.out}")
        return 0

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    # One query per (season, week) instead of per pick -- a ledger with
    # thousands of entries would otherwise mean thousands of round trips.
    by_sw = {}
    for p in ledger:
        by_sw.setdefault((p["season"], p["week"]), set()).add(p["player_id"])
    actual_by_key = {}
    for (season, week), pids in by_sw.items():
        placeholders = ",".join("?" for _ in pids)
        rows = con.execute(
            f"SELECT * FROM player_games WHERE season=? AND week=? AND player_id IN ({placeholders})",
            (season, week, *pids)).fetchall()
        for r in rows:
            actual_by_key[(r["player_id"], season, week)] = r
    con.close()

    results = []
    ungraded = 0
    for p in ledger:
        row = actual_by_key.get((p["player_id"], p["season"], p["week"]))
        if row is None:
            ungraded += 1
            continue
        actual = actual_stat(p["market"], row)
        if actual is None:
            continue
        result = grade(p, actual)
        if result is None:
            continue
        results.append({
            "season": p["season"], "week": p["week"], "market": p["market"],
            "player": p.get("player"), "player_id": p.get("player_id"),
            "team": p.get("team"), "opponent": p.get("opponent"),
            "pick": p.get("pick"), "line": p.get("line"),
            "actual": actual, "result": result,
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
        "results": sorted(results, key=lambda r: (r["season"], r["week"]), reverse=True),
        "last_updated": now_utc(),
    }
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(f"\n{hits}/{total} ({record['summary']['hit_rate']}%) written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
