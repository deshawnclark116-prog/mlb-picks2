#!/usr/bin/env python3
"""
PITCHER_DATASET_BUILDER_A

Builds the pitcher-side counterpart to hr_dataset_builder_a.py's
batter_games table -- one row per PITCHER START, with that pitcher's own
precise boxscore pitching line (hits allowed, batters faced, strikeouts,
innings pitched), not a diluted whole-team total.

Why this exists: opp_pitcher_h_per_pa / opp_pitcher_k_per_pa (excluded
from batter_hits_context, and the likely reason the model looks
insensitive to who's actually pitching) were computed by reverse-
aggregating batter_games via opposing_pitcher_id -- which tags every
batter's ENTIRE game line to whichever pitcher started, including at-bats
that actually happened against relievers after the starter left. Verified
concretely: Eduardo Rivera's own line in game 824737 was 3 hits / 12 BF
(25.0%), but the team's full-game total attributed to him by the old
method was 7 hits / 35 PA (20.0%) -- 6 relievers finished that game. That
isn't a live-vs-training skew, it's a feature-definition problem, and it
exists in the training data regardless of how it's served.

This builds a clean pitcher_games table so opp_pitcher_h_per_pa can be
recomputed from each pitcher's own precise starts -- the same quantity
pitcher_feature_row() already serves live (and which the parity check
confirmed is internally consistent across MLB API surfaces).

One row = one pitcher start in one completed game (gamesStarted == 1
only -- relief appearances are a different question, not needed for this
feature).

Reuses hr_dataset_builder_a.py's schedule/feed/caching plumbing so this
follows the exact same fetch pattern already proven in production.

Output: /data/hr_model/pitcher_game_dataset.csv (or ./hr_model/ locally)

Run small first:
    python pitcher_dataset_builder_a.py --days 7 --max-games 20

Then bigger (match batter_games' actual training window):
    python pitcher_dataset_builder_a.py --start-date 2025-01-01 --end-date 2026-07-22
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import hr_dataset_builder_a as hb

FIELDNAMES = [
    "game_id", "game_date", "pitcher_id", "pitcher_name", "team", "opponent",
    "side", "pitcher_hand", "innings_pitched", "batters_faced",
    "hits_allowed", "strikeouts", "walks", "earned_runs", "games_started",
]


def pitcher_start_rows(feed: Dict[str, Any]) -> List[Dict[str, Any]]:
    gid = hb.safe_int(((feed.get("gamePk")) or (feed.get("gameData") or {}).get("game", {}).get("pk")))
    game_date = hb.official_date(feed)
    if not gid or not game_date:
        return []

    rows = []
    for side, opp_side in (("home", "away"), ("away", "home")):
        team = hb.team_name(feed, side)
        opponent = hb.team_name(feed, opp_side)
        box = hb.box_team(feed, side)
        pitcher_ids = box.get("pitchers") or []
        players = box.get("players") or {}

        for pid in pitcher_ids:
            pid = hb.safe_int(pid)
            if not pid:
                continue
            p = players.get(f"ID{pid}") or players.get(str(pid))
            if not p:
                continue
            pstat = ((p.get("stats") or {}).get("pitching") or {})
            if not pstat or str(pstat.get("gamesStarted")) != "1":
                continue  # starts only -- this feature is about starting pitchers

            person = p.get("person") or {}
            hand = hb.player_hand(feed, pid, "pitch")
            rows.append({
                "game_id": gid,
                "game_date": game_date.isoformat(),
                "pitcher_id": pid,
                "pitcher_name": person.get("fullName"),
                "team": team,
                "opponent": opponent,
                "side": side,
                "pitcher_hand": hand,
                "innings_pitched": pstat.get("inningsPitched"),
                "batters_faced": hb.safe_int(pstat.get("battersFaced"), 0),
                "hits_allowed": hb.safe_int(pstat.get("hits"), 0),
                "strikeouts": hb.safe_int(pstat.get("strikeOuts"), 0),
                "walks": hb.safe_int(pstat.get("baseOnBalls"), 0),
                "earned_runs": hb.safe_int(pstat.get("earnedRuns"), 0),
                "games_started": 1,
            })
    return rows


def append_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def dedupe_csv(path: Path) -> None:
    if not path.exists():
        return
    seen = set()
    kept = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r.get("game_id"), r.get("pitcher_id"))
            if key in seen:
                continue
            seen.add(key)
            kept.append(r)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in kept:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-games", type=int, default=20)
    args = ap.parse_args()

    out_path = Path(args.output) if args.output else hb.DATA_DIR / "pitcher_game_dataset.csv"

    end = dt.date.fromisoformat(args.end_date) if args.end_date else dt.date.today() - dt.timedelta(days=1)
    start = dt.date.fromisoformat(args.start_date) if args.start_date else end - dt.timedelta(days=max(0, args.days - 1))

    print("PITCHER_DATASET_BUILDER_A\n" + "=" * 25)
    print("One row = one pitcher START in one completed game (own precise boxscore line).")
    print(f"date_range: {start} to {end}")
    print(f"output: {out_path}")

    all_rows: List[Dict[str, Any]] = []
    games_used = 0
    errors: Counter = Counter()

    for day in hb.daterange(start, end):
        games = hb.get_schedule_for_date(day)
        for game in games:
            if args.max_games and games_used >= args.max_games:
                break
            if not hb.is_final_game(game):
                continue
            gid = hb.safe_int(game.get("gamePk"))
            if not gid:
                continue

            feed = hb.get_game_feed(gid)
            if not feed:
                errors["missing_feed"] += 1
                continue
            try:
                rows = pitcher_start_rows(feed)
                all_rows.extend(rows)
                games_used += 1
                print(f"  game_id={gid} {day}  starts_added={len(rows)}  total_rows={len(all_rows)}")
                append_csv(out_path, rows)
            except Exception as e:
                errors[f"game_error:{type(e).__name__}"] += 1
        if args.max_games and games_used >= args.max_games:
            break

    dedupe_csv(out_path)
    print(f"\ngames_used={games_used}  total_rows={len(all_rows)}  errors={dict(errors)}")
    print(f"output written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
