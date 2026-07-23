"""
One-off diagnostic: verify the claim that nflverse's 'schedules' release
(games.csv -- the exact asset nfl_player_games_foundation_a.py already
pulls) includes spread_line/total_line/moneyline columns, before building
any game-script features on top of an assumption. Also checks whether
those fields are populated for upcoming (not-yet-played) games, since a
serving-time feature needs the line to exist BEFORE kickoff, not just in
historical archives.
"""
import csv
import gzip
import io
import json
import urllib.request

RELEASES_API = "https://api.github.com/repos/nflverse/nflverse-data/releases/tags/{tag}"
UA = {"User-Agent": "nfl-schedules-odds-check/1.0"}


def _http_get_bytes(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def find_asset(tag, name):
    data = json.loads(_http_get_bytes(RELEASES_API.format(tag=tag)).decode("utf-8"))
    for a in data.get("assets", []):
        if a["name"] == name:
            return a["browser_download_url"], a["name"].lower().endswith(".gz")
    names = [a["name"] for a in data.get("assets", [])]
    raise SystemExit(f"asset {name} not found in tag {tag}. Assets: {names[:20]}")


def load_csv(url, is_gz):
    raw = _http_get_bytes(url)
    if is_gz:
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def main():
    url, is_gz = find_asset("schedules", "games.csv")
    rows = load_csv(url, is_gz)
    print(f"games.csv: {len(rows)} total rows")
    cols = sorted(rows[0].keys())
    print(f"columns: {cols}")
    print()

    odds_cols = [c for c in ["spread_line", "total_line", "home_moneyline",
                              "away_moneyline", "away_spread_odds",
                              "home_spread_odds", "over_odds", "under_odds"]
                 if c in rows[0]]
    print(f"odds-related columns present: {odds_cols}")
    print()

    if not odds_cols:
        print("NONE of the expected odds columns exist in this asset. "
              "The claim that nflverse's schedules release bundles Vegas "
              "lines does not hold for THIS asset ('games.csv' under the "
              "'schedules' tag) -- would need a different source/asset.")
        return

    # Check population rate for a recent completed season and for the
    # current/future season -- a serving-time feature needs the line
    # BEFORE kickoff, not just in historical archives.
    for season in ["2024", "2025", "2026"]:
        season_rows = [r for r in rows if r.get("season") == season]
        if not season_rows:
            print(f"season {season}: 0 rows in games.csv")
            continue
        for col in odds_cols[:3]:
            non_empty = sum(1 for r in season_rows if r.get(col, "").strip())
            print(f"season {season} {col}: {non_empty}/{len(season_rows)} populated")
        # sample a couple of rows
        for r in season_rows[:2]:
            sample = {c: r.get(c) for c in odds_cols}
            print(f"  sample game_id={r.get('game_id')} week={r.get('week')}: {sample}")
        print()


if __name__ == "__main__":
    main()
