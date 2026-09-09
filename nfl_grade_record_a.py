#!/usr/bin/env python3
"""
NFL_GRADE_RECORD_A

Grades every logged NFL pick (docs/nfl_picks_log.jsonl, appended to by
nfl_serving_builder_a.py's PICKS_LOG_PATH) against real outcomes in
nfl_model.sqlite's player_games table, and writes docs/nfl_record.json
in the same summary/results shape MLB's record.json uses (api.py's
update_record()), adapted for NFL: segmented by market instead of by_prop,
no book-odds or bvp/confidence buckets since NFL picks don't carry those
fields (predictions-first, no odds). Mirrors cfb_grade_record_a.py
exactly -- see that file for the fuller design rationale.

A pick is gradable once player_games has a row for (player_id, season,
week) -- that row only exists once the real game has been played and
ingested, so no separate schedule-final check is needed here. Ungraded
picks (game not yet played/ingested) are silently skipped, not counted
as a miss.

"_early_season" picks are a real exception to the id match above:
they're sourced from ESPN (this year's preseason box scores, or ESPN's
live team roster for the prior-season-informed arm -- see
nfl_serving_builder_a.py's build_preseason_rushing_picks /
build_prior_season_picks), so their logged player_id is an ESPN athlete
id, not nflverse's gsis_id that player_games is keyed on -- the two
namespaces have no shared crosswalk (same real limitation documented in
nfl_preseason_to_regular_season_gate_a.py). Falls back to a normalized-
name match against player_games for exactly these markets so they're
still gradable at all, instead of silently sitting at 0 forever.

Read-only against the DB and the ledger; only ever writes
docs/nfl_record.json.

Run
---
python -u nfl_grade_record_a.py
"""
import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

REPO = Path(__file__).resolve().parent
DB_DEFAULT = REPO / "nfl_models" / "nfl_model.sqlite"
DOCS = REPO / "docs"
LOG_DEFAULT = DOCS / "nfl_picks_log.jsonl"
OUT_DEFAULT = DOCS / "nfl_record.json"

# Actual-outcome column per base market (an "_early_season" pick is
# graded against the exact same real-world stat as its in-season
# counterpart -- only the model that produced the pick differs). "sacks"
# is served with a human-facing display_line (0.5) baked directly into
# the pick string, so grading it needs no special-casing here -- the
# generic OVER/UNDER-line parse in grade() already uses that same string.
MARKET_STAT_COLUMN = {
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "sacks": "def_sacks",
}


def actual_stat(market, row):
    base = market.replace("_early_season", "")
    col = MARKET_STAT_COLUMN.get(base)
    return row[col] if col else None


def norm_player_name(name):
    name = unicodedata.normalize("NFKD", name or "")
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", name)
    name = re.sub(r"[^a-z ]", "", name)
    return re.sub(r"\s+", " ", name).strip()


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

    print("NFL_GRADE_RECORD_A\n==================")
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

    by_sw = {}
    for p in ledger:
        by_sw.setdefault((p["season"], p["week"]), set()).add(p["player_id"])
    actual_by_key = {}
    # Name-based fallback index, built alongside the id-based one from the
    # exact same rows -- only consulted for "_early_season" picks, whose
    # logged player_id is an ESPN athlete id with no crosswalk to
    # player_games' gsis_id (see module docstring).
    actual_by_name_key = {}
    for (season, week), pids in by_sw.items():
        placeholders = ",".join("?" for _ in pids)
        rows = con.execute(
            f"SELECT * FROM player_games WHERE season=? AND week=? AND player_id IN ({placeholders})",
            (season, week, *pids)).fetchall()
        for r in rows:
            actual_by_key[(r["player_id"], season, week)] = r
        name_rows = con.execute(
            "SELECT * FROM player_games WHERE season=? AND week=?", (season, week)).fetchall()
        for r in name_rows:
            key = (norm_player_name(r["player_name"]), season, week)
            actual_by_name_key.setdefault(key, r)
    con.close()

    results = []
    ungraded = 0
    for p in ledger:
        row = actual_by_key.get((p["player_id"], p["season"], p["week"]))
        if row is None and str(p["market"]).endswith("_early_season"):
            row = actual_by_name_key.get(
                (norm_player_name(p.get("player")), p["season"], p["week"]))
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
