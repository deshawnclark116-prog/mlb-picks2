#!/usr/bin/env python3
"""
TENNIS_GRADE_RECORD_A

Grades every logged tennis pick (docs/tennis_picks_log.jsonl, appended to
by tennis_serving_builder_a.py's PICKS_LOG_PATH) against real final
results from ESPN's public tennis scoreboard API, and writes
docs/tennis_record.json in the same summary/results shape MLB/NFL/CFB's
record.json files use (api.py's update_record()/nfl_record()/cfb_record()),
segmented by market instead of by_prop -- mirrors nfl_grade_record_a.py's
overall design; see that file for the fuller rationale.

Both served tennis markets are graded here:

  total_games   real market line the pick was made against (logged
                verbatim) vs. the real final total games (sum of every
                completed set's games, both players). Same OVER/UNDER-
                line parse NFL/CFB use.

  set_betting   an unagraded model PROJECTION (no real market exists to
                grade it against -- see tennis_serving_builder_a.py's
                docstring), but it's still a concrete, checkable claim
                ("Player X 3-1"): a hit means the model's single most
                likely exact score was the actual final score, winner
                and set count both. Graded here so the projection's real
                track record is visible even though no book line backs
                it -- "unagraded" (no confidence tier, no odds) is not
                the same as "not gradable".

A pick is gradable once ESPN's scoreboard for its logged espn_date shows
that competition_id as STATUS_FINAL ("post"/completed) -- that only
happens once the real match has actually finished, so no separate
schedule check is needed. Not-yet-final picks (match still scheduled/in
progress, or ESPN hasn't posted a result yet) are silently skipped, not
counted as a miss. One ESPN scoreboard fetch per (tour, espn_date) pair
actually present in the ledger, not one fetch per pick.

Read-only against ESPN's API and the ledger; only ever writes
docs/tennis_record.json.

Run
---
python -u tennis_grade_record_a.py
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from tennis_serving_builder_a import fetch_espn_schedule, SINGLES_GROUPING

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

REPO = Path(__file__).resolve().parent
DOCS = REPO / "docs"
LOG_DEFAULT = DOCS / "tennis_picks_log.jsonl"
OUT_DEFAULT = DOCS / "tennis_record.json"

_EMPTY_RECORD = {
    "summary": {"total": 0, "hits": 0, "misses": 0, "hit_rate": 0},
    "by_market": {}, "results": [],
}


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


def fetch_finalized_by_competition_id(tour, espn_date):
    """Re-queries the exact same ESPN scoreboard call the builder made
    (same tour, same dates param) and returns {competition_id: comp} for
    every FINAL match in the correct singles grouping -- same grouping
    filter as extract_actionable_matches, since ESPN's per-tour endpoints
    both return the full shared bracket for combined events."""
    wanted_grouping = SINGLES_GROUPING[tour]
    out = {}
    try:
        espn_json = fetch_espn_schedule(tour, espn_date)
    except Exception as e:
        print(f"  {tour}/{espn_date}: ESPN fetch failed: {e}")
        return out
    for event in espn_json.get("events", []):
        for grouping in event.get("groupings", []):
            slug = (grouping.get("grouping", {}).get("displayName") or "").strip().lower()
            if slug != wanted_grouping:
                continue
            for comp in grouping.get("competitions", []):
                status = comp.get("status", {}).get("type", {})
                if status.get("state") != "post" or not status.get("completed"):
                    continue
                out[str(comp.get("id"))] = comp
    return out


def actual_from_competition(comp):
    """Returns (winner_name, winner_sets, loser_sets, total_games) from a
    finalized ESPN competition. Each linescore entry's own "winner" flag
    marks who won THAT SET (confirmed against a real match: Michelsen's
    linescores carried winner=true only for the 2 sets he actually won,
    even though he lost the match) -- summing it per competitor gives
    real sets won, no need to trust the top-level "winner" flag for the
    set count, only for who ultimately won."""
    total_games = 0
    winner_name = winner_sets = loser_sets = None
    for c in comp.get("competitors", []):
        linescores = c.get("linescores", [])
        sets_won = sum(1 for ls in linescores if ls.get("winner"))
        total_games += sum(ls.get("value") or 0 for ls in linescores)
        name = c.get("athlete", {}).get("displayName")
        if c.get("winner"):
            winner_name, winner_sets = name, sets_won
        else:
            loser_sets = sets_won
    if winner_name is None or winner_sets is None or loser_sets is None:
        return None
    return winner_name, winner_sets, loser_sets, total_games


def grade_total_games(pick, actual_total_games):
    p = str(pick.get("pick", "")).upper().strip()
    m = re.search(r"(OVER|UNDER)\s+(-?\d+(?:\.\d+)?)", p)
    if not m:
        return None
    side, line = m.group(1), float(m.group(2))
    hit = actual_total_games > line if side == "OVER" else actual_total_games < line
    return "hit" if hit else "miss"


def grade_set_betting(pick, winner_name, winner_sets, loser_sets):
    p = str(pick.get("pick", "")).strip()
    m = re.match(r"^(.*)\s+(\d+)-(\d+)$", p)
    if not m:
        return None, None
    picked_name, picked_w, picked_l = m.group(1).strip(), int(m.group(2)), int(m.group(3))
    winner_correct = picked_name == winner_name
    exact_score_correct = winner_correct and picked_w == winner_sets and picked_l == loser_sets
    return ("hit" if exact_score_correct else "miss"), winner_correct


def main():
    print("TENNIS_GRADE_RECORD_A\n======================")
    ledger = load_ledger(LOG_DEFAULT)
    print(f"ledger: {len(ledger)} logged picks")
    if not ledger:
        OUT_DEFAULT.parent.mkdir(parents=True, exist_ok=True)
        OUT_DEFAULT.write_text(json.dumps({**_EMPTY_RECORD, "last_updated": now_utc()}, indent=2))
        print(f"no ledger entries yet -- wrote empty record to {OUT_DEFAULT}")
        return 0

    groups = {(p.get("tour"), p.get("espn_date")) for p in ledger}
    comps_by_group = {}
    for tour, espn_date in sorted(groups):
        if not tour or not espn_date:
            continue
        comps_by_group[(tour, espn_date)] = fetch_finalized_by_competition_id(tour, espn_date)
    total_final = sum(len(v) for v in comps_by_group.values())
    print(f"fetched {len(comps_by_group)} (tour, date) scoreboards, "
          f"{total_final} finalized singles matches total")

    results = []
    ungraded = 0
    for p in ledger:
        comp = comps_by_group.get((p.get("tour"), p.get("espn_date")), {}).get(
            str(p.get("competition_id")))
        if comp is None:
            ungraded += 1
            continue
        actual = actual_from_competition(comp)
        if actual is None:
            ungraded += 1
            continue
        winner_name, winner_sets, loser_sets, total_games = actual

        market = p.get("market")
        base = {
            "tour": p.get("tour"), "tourney": p.get("tourney"),
            "surface": p.get("surface"), "market": market,
            "player1": p.get("player1"), "player2": p.get("player2"),
            "pick": p.get("pick"), "model_prob": p.get("model_prob"),
            "start_time_utc": p.get("start_time_utc"),
            "actual": f"{winner_name} {winner_sets}-{loser_sets} ({total_games} games)",
        }
        if market == "total_games":
            result = grade_total_games(p, total_games)
            if result is None:
                continue
            results.append({**base, "line": p.get("line"), "result": result})
        elif market == "set_betting":
            result, winner_correct = grade_set_betting(p, winner_name, winner_sets, loser_sets)
            if result is None:
                continue
            results.append({**base, "unagraded": True, "winner_correct": winner_correct,
                             "result": result})
        else:
            continue

    print(f"graded: {len(results)}  ungraded (not yet final/ESPN result missing): {ungraded}")

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
        "results": sorted(results, key=lambda r: r.get("start_time_utc") or "", reverse=True),
        "last_updated": now_utc(),
    }
    OUT_DEFAULT.parent.mkdir(parents=True, exist_ok=True)
    OUT_DEFAULT.write_text(json.dumps(record, indent=2))
    print(f"\n{hits}/{total} ({record['summary']['hit_rate']}%) written to {OUT_DEFAULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
