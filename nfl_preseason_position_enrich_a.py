#!/usr/bin/env python3
"""
NFL_PRESEASON_POSITION_ENRICH_A

ESPN's box-score athlete objects (used by nfl_preseason_data_foundation_a.py)
don't carry a position field -- confirmed by inspecting the raw payload live.
Position matters here: the regular-season rushing_yards model is deliberately
scoped to RB only (QB scrambles / WR jet sweeps are a different, noisier
population -- see nfl_rushing_yards_clean_baseline_a.py), and this should
hold the same line rather than fall back to "anyone who touched the ball."

ESPN's per-team roster endpoint DOES carry position, but it's a CURRENT
snapshot -- players no longer on any NFL roster (retired, out of the league)
won't appear, so this necessarily has better coverage for recent seasons
than for 2021-2022. Documented gap, not hidden: rows for players who can't
be matched are left with position=NULL and simply excluded downstream by
any position-scoped query, not mislabeled.

Run:
    python -u nfl_preseason_position_enrich_a.py --db nfl_models/nfl_preseason.sqlite
"""
import argparse
import json
import sqlite3
import subprocess
import time
from pathlib import Path

UA = "curl/8.5.0"
TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster"


def curl_json(url, timeout=20, retries=4):
    for attempt in range(retries):
        try:
            r = subprocess.run(["curl", "-s", "--max-time", str(timeout), "-A", UA, url],
                                capture_output=True, timeout=timeout + 10)
            if r.returncode != 0:
                raise RuntimeError(f"curl exit {r.returncode}")
            return json.loads(r.stdout.decode("utf-8"))
        except Exception as e:
            print(f"  retry ({attempt+1}/{retries}) {url}: {e}")
            time.sleep(2 * (attempt + 1))
    return {}


def fetch_team_ids():
    data = curl_json(TEAMS_URL)
    teams = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
    return [(t["team"]["id"], t["team"]["abbreviation"]) for t in teams]


def fetch_roster_positions(team_id):
    data = curl_json(ROSTER_URL.format(team_id=team_id))
    out = {}
    for group in data.get("athletes", []):
        for ath in group.get("items", []):
            aid = ath.get("id")
            pos = (ath.get("position") or {}).get("abbreviation")
            if aid and pos:
                out[aid] = pos
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    args = ap.parse_args()

    con = sqlite3.connect(str(args.db))
    cols = {r[1] for r in con.execute("PRAGMA table_info(player_games)")}
    if "position" not in cols:
        con.execute("ALTER TABLE player_games ADD COLUMN position TEXT")
        con.commit()

    teams = fetch_team_ids()
    print(f"{len(teams)} teams")
    position_map = {}
    for team_id, abbr in teams:
        pos = fetch_roster_positions(team_id)
        position_map.update(pos)
        print(f"  {abbr}: {len(pos)} rostered players")
        time.sleep(0.1)

    print(f"\ntotal distinct current-roster athletes with position: {len(position_map)}")

    rows = con.execute("SELECT DISTINCT athlete_id FROM player_games").fetchall()
    matched = 0
    for (aid,) in rows:
        pos = position_map.get(aid)
        if pos:
            con.execute("UPDATE player_games SET position = ? WHERE athlete_id = ?", (pos, aid))
            matched += 1
    con.commit()

    total_athletes = len(rows)
    print(f"matched {matched}/{total_athletes} distinct athletes to a current position "
          f"({matched/total_athletes*100:.1f}%)")

    by_pos = con.execute("""
        SELECT position, COUNT(*) FROM player_games
        WHERE position IS NOT NULL GROUP BY position ORDER BY COUNT(*) DESC
    """).fetchall()
    print("\nrows by position (matched only):")
    for pos, c in by_pos:
        print(f"  {pos:6s} {c}")

    con.close()


if __name__ == "__main__":
    main()
