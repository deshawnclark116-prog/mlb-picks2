"""
One-off diagnostic: before concluding the 2025 fresh-holdout failure means
the models don't generalize, rule out a confound -- 2025 data now comes from
a different nflverse release tag ('stats_player') than 2023/2024 data (the
old frozen 'player_stats' comprehensive file). Compares column headers and
basic per-season summary stats across both sources to check the newer tag
isn't silently computing fields differently.
"""
import csv
import gzip
import io
import json
import urllib.request

RELEASES_API = "https://api.github.com/repos/nflverse/nflverse-data/releases/tags/{tag}"
UA = {"User-Agent": "nfl-schema-check/1.0"}


def _http_get_bytes(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def find_asset(tag, name):
    data = json.loads(_http_get_bytes(RELEASES_API.format(tag=tag)).decode("utf-8"))
    for a in data.get("assets", []):
        if a["name"] == name:
            return a["browser_download_url"], a["name"].lower().endswith(".gz")
    raise SystemExit(f"asset {name} not found in tag {tag}")


def load_csv(url, is_gz):
    raw = _http_get_bytes(url)
    if is_gz:
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def summarize(rows, season, label, position_filter=None):
    rows = [r for r in rows if r.get("season") == str(season)]
    if position_filter:
        rows = [r for r in rows if r.get("position") in position_filter]
    n = len(rows)
    if n == 0:
        print(f"  {label} season={season}: 0 rows")
        return
    def avg(field):
        vals = []
        for r in rows:
            v = r.get(field, "")
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                pass
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    weeks = sorted(set(r.get("week") for r in rows))
    print(f"  {label} season={season}: n={n} weeks={weeks[:3]}...{weeks[-3:]} (n_weeks={len(weeks)})")
    for field in ["carries", "rushing_yards", "targets", "receptions", "receiving_yards"]:
        m, c = avg(field)
        print(f"    {field}: mean={m} (non-empty={c}/{n})")


def main():
    print("Comparing 'player_stats' (frozen, 2023/2024 source) vs 'stats_player' (current, 2025 source)")
    print()

    old_url, old_gz = find_asset("player_stats", "player_stats.csv")
    old_rows = load_csv(old_url, old_gz)
    print(f"player_stats.csv: {len(old_rows)} total rows, columns: {sorted(old_rows[0].keys())}")
    print()
    summarize(old_rows, 2024, "player_stats.csv")
    print()

    new_url, new_gz = find_asset("stats_player", "stats_player_week_2025.csv")
    new_rows = load_csv(new_url, new_gz)
    print(f"stats_player_week_2025.csv: {len(new_rows)} total rows, columns: {sorted(new_rows[0].keys())}")
    print()
    summarize(new_rows, 2025, "stats_player_week_2025.csv")
    print()

    # Also pull stats_player_week_2024 (if present) to compare same-season,
    # different-tag -- the cleanest possible apples-to-apples check.
    try:
        cmp_url, cmp_gz = find_asset("stats_player", "stats_player_week_2024.csv")
        cmp_rows = load_csv(cmp_url, cmp_gz)
        print(f"stats_player_week_2024.csv (same season, new tag): {len(cmp_rows)} total rows")
        summarize(cmp_rows, 2024, "stats_player_week_2024.csv")
    except SystemExit as e:
        print(f"  (skipped same-season cross-tag check: {e})")

    print()
    old_cols = set(old_rows[0].keys())
    new_cols = set(new_rows[0].keys())
    print(f"columns only in player_stats.csv: {sorted(old_cols - new_cols)}")
    print(f"columns only in stats_player_week_2025.csv: {sorted(new_cols - old_cols)}")


if __name__ == "__main__":
    main()
