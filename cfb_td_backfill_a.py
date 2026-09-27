#!/usr/bin/env python3
"""
CFB_TD_BACKFILL_A

cfbfastR-data's 2025 player stats are missing almost every rushing and
receiving TD from week 9 on (~40 TDs/week recorded vs ~340 in real
games; yards are fine). That corrupts every TD label and every
"last season's TDs" feature built on 2025. ESPN box scores have the real
numbers, and cfbfastR's player_id/game_id ARE ESPN ids, so this patches
rushing/receiving/passing TDs in place for the affected weeks.

Results are cached (cfb_models/cfb_espn_td_cache_{season}.json) so the
daily pipeline -- which rebuilds 2025 from cfbfastR every run -- can
re-apply the fix without refetching hundreds of box scores.

Run
---
python -u cfb_td_backfill_a.py --db cfb_models/cfb_model.sqlite --season 2025 --from-week 9
"""
import argparse
import json
import sqlite3
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
HEADERS = {  # same as cfb_espn_live_foundation_a.py -- a bare UA gets 403s
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={gid}"


def fetch_tds(gid):
    req = urllib.request.Request(SUMMARY.format(gid=gid), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode("utf-8"))
    out = {}
    for team in (d.get("boxscore") or {}).get("players") or []:
        for cat in team.get("statistics") or []:
            col = {"rushing": "rushing_touchdowns", "receiving": "receiving_touchdowns",
                   "passing": "passing_touchdowns"}.get(cat.get("name"))
            if not col:
                continue
            labels = cat.get("labels") or []
            for a in cat.get("athletes") or []:
                row = dict(zip(labels, a.get("stats") or []))
                try:
                    td = int(float(row.get("TD")))
                except (TypeError, ValueError):
                    continue
                out.setdefault(a["athlete"]["id"], {})[col] = td
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--from-week", type=int, default=9)
    args = ap.parse_args()
    cache_path = REPO / "cfb_models" / f"cfb_espn_td_cache_{args.season}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    con = sqlite3.connect(args.db)
    gids = [r[0] for r in con.execute(
        "SELECT game_id FROM games WHERE season=? AND week>=? AND home_points IS NOT NULL",
        (args.season, args.from_week))]
    fetched = 0
    for gid in gids:
        if str(gid) in cache:
            continue
        try:
            cache[str(gid)] = fetch_tds(gid)
            fetched += 1
        except Exception as e:
            print(f"  {gid}: fetch failed ({e})")
        time.sleep(0.15)
        if fetched and fetched % 100 == 0:
            cache_path.write_text(json.dumps(cache))
            print(f"  fetched {fetched} box scores ...", flush=True)
    cache_path.write_text(json.dumps(cache))
    updated = 0
    for gid in gids:
        for pid, tds in cache.get(str(gid), {}).items():
            sets = ", ".join(f"{c} = ?" for c in tds)
            cur = con.execute(f"UPDATE player_games SET {sets} WHERE game_id = ? AND player_id = ?",
                              (*tds.values(), str(gid), pid))
            updated += cur.rowcount
    con.commit()
    tot = con.execute("""SELECT SUM(COALESCE(rushing_touchdowns,0)+COALESCE(receiving_touchdowns,0))
                         FROM player_games WHERE season=? AND week>=?""", (args.season, args.from_week)).fetchone()[0]
    print(f"{len(gids)} games ({fetched} newly fetched), {updated} player-game rows updated; "
          f"season {args.season} weeks {args.from_week}+ now have {tot} rushing+receiving TDs")
    con.close()


if __name__ == "__main__":
    raise SystemExit(main())
