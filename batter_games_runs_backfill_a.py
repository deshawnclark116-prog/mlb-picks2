#!/usr/bin/env python3
"""
BATTER_GAMES_RUNS_BACKFILL_A

Unblocks the total_bases / rbi / runs models by backfilling the `runs` column
into hr_model.sqlite::batter_games from the MLB StatsAPI hitting gameLog, matched
exactly on game id (gamePk). Those three production models consume runs_per_pa
(and batter_runs's target IS runs), which batter_games currently does not store,
so they cannot be honestly evaluated or retrained until this is done.

Safe by construction:
- Only ADDS a `runs` column and fills it; never alters existing columns.
- Idempotent: re-running updates the same values.
- Dry run by default (reports what it would fill). Pass --apply to write.

Run (Render) -- preview first, then apply
-----------------------------------------
python -u batter_games_runs_backfill_a.py
python -u batter_games_runs_backfill_a.py --apply 2>&1 | tee /data/hr_model/batter_games_runs_backfill_a.log
"""

import argparse
import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

MLB = "https://statsapi.mlb.com/api/v1"
DEFAULT_SOURCE = "/data/hr_model/hr_model.sqlite"


def fetch_json(url, tries=3, timeout=20):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "runs-backfill/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if i == tries - 1:
                return {"_error": str(e)}
            time.sleep(1.5 * (i + 1))
    return {}


def gamelog_runs(pid, season):
    """Return {game_id(str): runs} for a batter-season from the hitting gameLog."""
    url = f"{MLB}/people/{pid}/stats?stats=gameLog&group=hitting&season={season}"
    d = fetch_json(url)
    out = {}
    try:
        for sp in d["stats"][0]["splits"]:
            gpk = sp.get("game", {}).get("gamePk")
            if gpk is None:
                continue
            runs = int(sp.get("stat", {}).get("runs", 0) or 0)
            out[str(gpk)] = runs
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--limit", type=int, default=0, help="limit batter-seasons (testing)")
    args = ap.parse_args()

    mode = "APPLY" if args.apply else "DRY_RUN"
    print(f"BATTER_GAMES_RUNS_BACKFILL_A  [{mode}]\n{'='*38}", flush=True)

    conn = sqlite3.connect(args.source)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(batter_games)")}
    have_runs = "runs" in cols
    print(f"runs column present: {have_runs}")
    if args.apply and not have_runs:
        conn.execute("ALTER TABLE batter_games ADD COLUMN runs INTEGER")
        conn.commit()
        print("added `runs` column")

    pairs = conn.execute("""SELECT DISTINCT batter_id, substr(game_date,1,4) AS season
                            FROM batter_games WHERE batter_id IS NOT NULL
                            ORDER BY season, batter_id""").fetchall()
    if args.limit:
        pairs = pairs[:args.limit]
    total_rows = conn.execute("SELECT COUNT(*) FROM batter_games").fetchone()[0]
    print(f"batter-seasons to process: {len(pairs)}   (batter_games rows: {total_rows})")

    filled = matched = missing = 0
    sample = []
    for n, (pid, season) in enumerate(pairs, 1):
        runs_by_game = gamelog_runs(pid, season)
        rows = conn.execute(
            "SELECT game_id FROM batter_games WHERE batter_id=? AND substr(game_date,1,4)=?",
            (pid, season)).fetchall()
        for (gid,) in rows:
            r = runs_by_game.get(str(gid))
            if r is None:
                missing += 1
                continue
            matched += 1
            if len(sample) < 8:
                sample.append((pid, gid, r))
            if args.apply:
                conn.execute("UPDATE batter_games SET runs=? WHERE batter_id=? AND game_id=?",
                             (r, pid, gid))
                filled += 1
        if args.apply and n % 50 == 0:
            conn.commit()
        if n % 100 == 0:
            print(f"  {n}/{len(pairs)} batter-seasons  (matched={matched} missing={missing})", flush=True)
        time.sleep(0.05)

    if args.apply:
        conn.commit()

    print("\nSAMPLE (batter_id, game_id, runs):")
    for s in sample:
        print(f"   {s}")
    print(f"\nmatched game rows: {matched}   unmatched: {missing}")
    if args.apply:
        cov = conn.execute("SELECT COUNT(*) FROM batter_games WHERE runs IS NOT NULL").fetchone()[0]
        print(f"batter_games rows with runs now: {cov}/{total_rows} ({cov/total_rows:.1%})")
        print("APPLIED. `runs` column populated.")
    else:
        print(f"DRY RUN: would fill ~{matched} rows. Re-run with --apply to write.")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
