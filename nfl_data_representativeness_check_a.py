#!/usr/bin/env python3
"""
NFL_DATA_REPRESENTATIVENESS_CHECK_A

Sanity check prompted by the receptions champion's unusually high AUC (0.78 --
well above every MLB market's 0.55-0.63 range). "Too good" is usually a leak,
not a miracle. This checks the specific, most likely explanation: does
nfl_model.sqlite::player_games include a row for a WR/TE/RB who was active but
recorded ZERO targets that game, or does the source data only emit a row when a
player records some statistical event? If true zero-target games are missing
from the source, "will he catch anything" becomes artificially easy, because
the hardest, most genuinely uncertain cases never appear to be predicted.

This is NOT a holdout-discipline violation -- it's a question about whether the
POPULATION being scored is representative of "every pass-catcher who played,"
which is prior to and separate from the train/holdout split.

Read-only. Changes nothing.

Run (Render)
------------
python -u nfl_data_representativeness_check_a.py 2>&1 | tee /data/nfl_model/nfl_data_representativeness_check_a.log
"""

import sqlite3
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

SOURCE = "/data/nfl_model/nfl_model.sqlite"


def main():
    con = sqlite3.connect(f"file:{SOURCE}?mode=ro", uri=True)
    print("NFL_DATA_REPRESENTATIVENESS_CHECK_A\n====================================")

    # 1. targets distribution among WR/TE/RB rows -- if targets=0 is rare or
    #    absent, that's the smoking gun for population truncation.
    print("\ntargets distribution (WR/TE/RB rows, by season):")
    for season in (2023, 2024):
        rows = con.execute(
            "SELECT targets, COUNT(*) FROM player_games "
            "WHERE position IN ('WR','TE','RB') AND season=? "
            "GROUP BY targets ORDER BY targets", (season,)).fetchall()
        total = sum(n for _, n in rows)
        zero = sum(n for t, n in rows if t == 0 or t is None)
        print(f"  {season}: total={total}  targets=0 rows={zero}  ({zero/total:.1%})")
        print(f"    targets histogram (first 8 buckets): {rows[:8]}")

    # 2. team-week roster size: how many WR/TE/RB rows appear per team per week?
    #    A typical NFL team dresses ~7-9 skill-position pass-catchers on gameday
    #    (WR/TE/RB combined). If we consistently see far fewer rows per team-week
    #    than that, the source is likely dropping zero-involvement players.
    print("\nWR/TE/RB rows per team per week (sample distribution):")
    per_team_week = con.execute("""
        SELECT team, season, week, COUNT(*) as n
        FROM player_games
        WHERE position IN ('WR','TE','RB')
        GROUP BY team, season, week
    """).fetchall()
    counts = [r[3] for r in per_team_week]
    counts.sort()
    n = len(counts)
    if n:
        print(f"  team-weeks sampled: {n}")
        print(f"  min={counts[0]}  p25={counts[n//4]}  median={counts[n//2]}  "
              f"p75={counts[3*n//4]}  max={counts[-1]}")

    # 3. explicit zero-catch, zero-target rows: do they exist at all?
    z = con.execute(
        "SELECT COUNT(*) FROM player_games WHERE position IN ('WR','TE','RB') "
        "AND targets=0 AND receptions=0").fetchone()[0]
    print(f"\nrows with targets=0 AND receptions=0 (true zero-involvement games): {z}")

    con.close()

    print("\nREAD:")
    print("  If targets=0 is a near-zero fraction (<5%) and rows-per-team-week is")
    print("  consistently low (well under a realistic ~7-9 pass-catchers/team), the")
    print("  source likely omits zero-involvement players -- the receptions AUC is")
    print("  inflated by population truncation, not genuine predictability, and the")
    print("  model should NOT be wired live as-is.")
    print("  If targets=0 is well represented and rows-per-team-week looks like a")
    print("  real roster, the strong AUC is more likely genuine: reception rate is")
    print("  simply a more volume-driven, lower-variance signal than a single at-bat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
