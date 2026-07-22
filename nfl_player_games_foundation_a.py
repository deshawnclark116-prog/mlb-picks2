#!/usr/bin/env python3
"""
NFL_PLAYER_GAMES_FOUNDATION_A

Data foundation for the NFL pipeline -- the NFL analog of hr_sqlite_foundation_a
for MLB. Builds /data/nfl_model/nfl_model.sqlite with two tables:

    player_games   one row per player per game (weekly stats, joined to schedule)
    games          one row per game (schedule/matchup context)

Source: nflverse (github.com/nflverse/nflverse-data), the free, actively
maintained, statistics-only NFL data project -- the closest NFL analog to
Statcast. Two CSV releases:
    player_stats   weekly player-level box score stats
    schedules      game schedule / matchup / home-away

Self-adapting like hitter_calibration_audit_a.py: reads the actual CSV headers
at runtime and matches known nflverse column-name variants rather than assuming
an exact schema, since column sets have changed across nflverse versions. Any
expected column that can't be found is reported explicitly, never silently
defaulted to zero.

This script is the data-loading step ONLY -- no feature engineering, no models,
no strict-D-1 logic (that belongs in a clean-baseline script downstream, same as
total_bases_clean_baseline_a.py did for MLB). Read-only on the network sources;
writes only nfl_model.sqlite.

Run (Render -- needs outbound internet to github.com; MLB's statsapi calls
already prove Render has general egress, unlike this dev sandbox)
--------------------------------------------------------------------
python -u nfl_player_games_foundation_a.py --season 2024 2025
python -u nfl_player_games_foundation_a.py --season 2024 2025 2>&1 | tee /data/nfl_model/nfl_player_games_foundation_a.log

Run (local test against a small synthetic CSV, no network)
------------------------------------------------------------
python -u nfl_player_games_foundation_a.py --player-stats-csv test.csv --schedules-csv sched.csv --db /tmp/test.sqlite
"""

import argparse
import csv
import gzip
import io
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

RELEASES_API = "https://api.github.com/repos/nflverse/nflverse-data/releases/tags/{tag}"
UA = {"User-Agent": "nfl-foundation/1.0"}
# Default sources are RELEASE TAGS, not guessed filenames -- the actual asset
# name is discovered from the GitHub release metadata at runtime (see
# resolve_release_asset). This avoids hardcoding a filename that nflverse may
# have renamed; only the tag name (which nflreadr itself documents) is assumed.
PLAYER_STATS_SOURCE = "player_stats"
SCHEDULES_SOURCE = "schedules"
DEFAULT_DB = Path("/data/nfl_model/nfl_model.sqlite")

# Known nflverse column-name variants per logical field, in preference order.
# Adding a variant here is the only change needed if nflverse renames a column.
PLAYER_STATS_COLUMNS = {
    "player_id": ["player_id", "gsis_id"],
    "player_name": ["player_display_name", "player_name"],
    "position": ["position"],
    "team": ["recent_team", "team"],
    "opponent": ["opponent_team", "opponent"],
    "season": ["season"],
    "week": ["week"],
    "season_type": ["season_type"],
    "completions": ["completions"],
    "attempts": ["attempts"],
    "passing_yards": ["passing_yards"],
    "passing_tds": ["passing_tds"],
    "interceptions": ["interceptions", "passing_interceptions"],
    "carries": ["carries"],
    "rushing_yards": ["rushing_yards"],
    "rushing_tds": ["rushing_tds"],
    "receptions": ["receptions"],
    "targets": ["targets"],
    "receiving_yards": ["receiving_yards"],
    "receiving_tds": ["receiving_tds"],
}
REQUIRED_FIELDS = ["player_id", "player_name", "team", "season", "week"]

SCHEDULES_COLUMNS = {
    "game_id": ["game_id"],
    "season": ["season"],
    "week": ["week"],
    "season_type": ["game_type", "season_type"],
    "gameday": ["gameday", "game_date"],
    "home_team": ["home_team"],
    "away_team": ["away_team"],
}

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    season_type TEXT,
    game_date TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_games_season_week ON games(season, week);

CREATE TABLE IF NOT EXISTS player_games (
    player_id TEXT NOT NULL,
    player_name TEXT,
    position TEXT,
    team TEXT NOT NULL,
    opponent TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    season_type TEXT,
    game_id TEXT,
    game_date TEXT,
    is_home INTEGER,
    completions INTEGER,
    attempts INTEGER,
    passing_yards INTEGER,
    passing_tds INTEGER,
    interceptions INTEGER,
    carries INTEGER,
    rushing_yards INTEGER,
    rushing_tds INTEGER,
    receptions INTEGER,
    targets INTEGER,
    receiving_yards INTEGER,
    receiving_tds INTEGER,
    PRIMARY KEY (player_id, season, week, season_type)
);
CREATE INDEX IF NOT EXISTS idx_pg_player_season ON player_games(player_id, season, week);
CREATE INDEX IF NOT EXISTS idx_pg_team_week ON player_games(team, season, week);
CREATE INDEX IF NOT EXISTS idx_pg_game ON player_games(game_id);
"""


def _http_get_bytes(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def resolve_release_asset(tag, timeout=30):
    """Query the GitHub release API for `tag` and find its current CSV asset.

    Some nflverse-data release tags (notably 'player_stats') bundle decades of
    split-by-year/split-by-category files alongside the one current,
    comprehensive file -- e.g. 'player_stats.csv' sits next to
    'player_stats_kicking_2017.csv', 'player_stats_season_2021.csv', legacy
    'stats_player_reg_2001.csv', etc. Picking "any .csv" is not safe there.

    Selection order:
      1. an asset named EXACTLY '<tag>.csv' (the current comprehensive file)
      2. exactly '<tag>.csv.gz'
      3. first asset ending '.csv' (fallback for tags with one clean file, e.g.
         'schedules' -> 'games.csv')
      4. first asset ending '.csv.gz'
    """
    url = RELEASES_API.format(tag=tag)
    data = json.loads(_http_get_bytes(url, timeout=timeout).decode("utf-8"))
    assets = data.get("assets", [])
    names = [a["name"] for a in assets]
    print(f"  [{tag}] {len(assets)} release assets found")

    def find(pred):
        return next((a for a in assets if pred(a["name"].lower())), None)

    chosen = (find(lambda n: n == f"{tag}.csv")
              or find(lambda n: n == f"{tag}.csv.gz")
              or find(lambda n: n.endswith(".csv"))
              or find(lambda n: n.endswith(".csv.gz")))
    if not chosen:
        raise SystemExit(f"FAIL: release '{tag}' has no .csv/.csv.gz asset. "
                          f"Sample of assets found: {names[:20]}")
    print(f"  [{tag}] using: {chosen['name']}")
    return chosen["browser_download_url"], chosen["name"].lower().endswith(".gz")


_release_cache = {}


def resolve_per_season_asset(tag, season, timeout=30):
    """Find a release asset for one specific season, e.g.
    'player_stats_2025.csv'. Needed because the comprehensive '<tag>.csv'
    lags: observed 2026-07, player_stats.csv ends at 2024 while the
    completed 2025 season ships only as a per-season asset. The
    '<tag>_<season>.' prefix requirement naturally excludes the
    kicking/def/season-aggregate variants ('player_stats_def_2025.csv',
    'player_stats_season_2021.csv', ...). Returns None if no such asset."""
    if tag not in _release_cache:
        url = RELEASES_API.format(tag=tag)
        _release_cache[tag] = json.loads(_http_get_bytes(url, timeout=timeout).decode("utf-8"))
    assets = _release_cache[tag].get("assets", [])

    def find(pred):
        return next((a for a in assets if pred(a["name"].lower())), None)

    # nflverse has renamed per-season weekly files across generations --
    # try each known naming scheme in order, newest first.
    exact_candidates = [
        f"stats_player_week_{season}.csv", f"stats_player_week_{season}.csv.gz",
        f"{tag}_{season}.csv", f"{tag}_{season}.csv.gz",
        f"stats_player_{season}.csv", f"stats_player_{season}.csv.gz",
    ]
    chosen = None
    for cand in exact_candidates:
        chosen = find(lambda n, c=cand: n == c)
        if chosen:
            break
    if not chosen:
        chosen = find(lambda n: n.startswith(f"{tag}_{season}.")
                      or n.startswith(f"stats_player_week_{season}."))
    if not chosen:
        near = [a["name"] for a in assets if str(season) in a["name"]]
        print(f"  [{tag}] per-season {season}: no known-pattern asset; "
              f"assets containing '{season}': {near[:25]}")
        return None, None
    print(f"  [{tag}] per-season {season} using: {chosen['name']}")
    return chosen["browser_download_url"], chosen["name"].lower().endswith(".gz")


def fetch_text(source, timeout=60):
    """Read a CSV from: a local path (testing), a direct http(s) URL, or a
    nflverse-data release TAG NAME (auto-discovers the actual asset via the
    GitHub release API). Transparently gunzips .gz assets."""
    p = Path(str(source))
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")

    if str(source).startswith("http://") or str(source).startswith("https://"):
        url, is_gz = str(source), str(source).lower().endswith(".gz")
    else:
        url, is_gz = resolve_release_asset(str(source), timeout=timeout)

    raw = _http_get_bytes(url, timeout=timeout)
    if is_gz:
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def resolve_columns(header, spec, label):
    """Map logical field -> actual CSV column name found in header. Reports misses."""
    resolved = {}
    missing = []
    for field, variants in spec.items():
        found = next((v for v in variants if v in header), None)
        if found:
            resolved[field] = found
        else:
            missing.append(field)
    print(f"  [{label}] resolved {len(resolved)}/{len(spec)} fields" +
          (f"; MISSING: {missing}" if missing else ""), flush=True)
    return resolved, missing


def to_int(v):
    try:
        if v is None or v == "":
            return None
        return int(round(float(v)))
    except Exception:
        return None


def load_player_stats(text, seasons):
    reader = csv.DictReader(io.StringIO(text))
    header = reader.fieldnames or []
    resolved, missing = resolve_columns(header, PLAYER_STATS_COLUMNS, "player_stats")
    for req in REQUIRED_FIELDS:
        if req not in resolved:
            raise SystemExit(f"FAIL: required field '{req}' not found in player_stats header: {header}")

    rows = []
    for r in reader:
        season = to_int(r.get(resolved["season"]))
        if seasons and season not in seasons:
            continue
        row = {
            "player_id": r.get(resolved["player_id"]),
            "player_name": r.get(resolved["player_name"]),
            "position": r.get(resolved.get("position", ""), None) if "position" in resolved else None,
            "team": r.get(resolved["team"]),
            "opponent": r.get(resolved.get("opponent", ""), None) if "opponent" in resolved else None,
            "season": season,
            "week": to_int(r.get(resolved["week"])),
            "season_type": r.get(resolved.get("season_type", ""), "REG") if "season_type" in resolved else "REG",
        }
        for stat in ("completions", "attempts", "passing_yards", "passing_tds", "interceptions",
                     "carries", "rushing_yards", "rushing_tds", "receptions", "targets",
                     "receiving_yards", "receiving_tds"):
            col = resolved.get(stat)
            row[stat] = to_int(r.get(col)) if col else None
        if row["player_id"] and row["season"] is not None and row["week"] is not None:
            rows.append(row)
    return rows


def load_schedules(text, seasons):
    reader = csv.DictReader(io.StringIO(text))
    header = reader.fieldnames or []
    resolved, missing = resolve_columns(header, SCHEDULES_COLUMNS, "schedules")
    for req in ("game_id", "season", "week", "home_team", "away_team"):
        if req not in resolved:
            raise SystemExit(f"FAIL: required field '{req}' not found in schedules header: {header}")

    rows = []
    for r in reader:
        season = to_int(r.get(resolved["season"]))
        if seasons and season not in seasons:
            continue
        rows.append({
            "game_id": r.get(resolved["game_id"]),
            "season": season,
            "week": to_int(r.get(resolved["week"])),
            "season_type": r.get(resolved.get("season_type", ""), "REG") if "season_type" in resolved else "REG",
            "game_date": r.get(resolved.get("gameday", ""), None) if "gameday" in resolved else None,
            "home_team": r.get(resolved["home_team"]),
            "away_team": r.get(resolved["away_team"]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, nargs="*", default=None,
                    help="seasons to load (default: all available)")
    ap.add_argument("--player-stats-csv", default=PLAYER_STATS_SOURCE,
                    help="release tag, direct URL, or local file (default: release tag 'player_stats')")
    ap.add_argument("--schedules-csv", default=SCHEDULES_SOURCE,
                    help="release tag, direct URL, or local file (default: release tag 'schedules')")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    args = ap.parse_args()

    seasons = set(args.season) if args.season else None
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("NFL_PLAYER_GAMES_FOUNDATION_A\n=============================", flush=True)
    print(f"seasons: {sorted(seasons) if seasons else 'all'}", flush=True)

    print("\nfetching schedules ...", flush=True)
    sched_text = fetch_text(args.schedules_csv)
    games = load_schedules(sched_text, seasons)
    print(f"  {len(games)} games")

    print("\nfetching player_stats ...", flush=True)
    ps_text = fetch_text(args.player_stats_csv)
    player_rows = load_player_stats(ps_text, seasons)
    print(f"  {len(player_rows)} player-week rows")

    # Per-season fallback for requested seasons the comprehensive file
    # doesn't cover (see resolve_per_season_asset). Only applies when the
    # source is a release tag -- a direct URL or local file is taken as-is.
    src = str(args.player_stats_csv)
    if seasons and not (src.startswith("http") or Path(src).exists()):
        have = {r["season"] for r in player_rows}
        for s in sorted(s for s in seasons if s not in have):
            url, is_gz = resolve_per_season_asset(src, s)
            if url is None:
                print(f"  [player_stats] season {s}: not in comprehensive file and "
                      f"no per-season asset found -- skipping")
                continue
            raw = _http_get_bytes(url)
            if is_gz:
                raw = gzip.decompress(raw)
            extra = load_player_stats(raw.decode("utf-8", errors="replace"), {s})
            print(f"  [player_stats] season {s}: +{len(extra)} rows from per-season asset")
            player_rows.extend(extra)

    # join player rows to games: match on (season, week, team) against
    # home_team or away_team, so we know game_id, game_date, and is_home.
    game_by_key = {}
    for g in games:
        for side, team_field in (("home", "home_team"), ("away", "away_team")):
            key = (g["season"], g["week"], g[team_field])
            game_by_key[key] = (g, side)

    matched = unmatched = 0
    for r in player_rows:
        key = (r["season"], r["week"], r["team"])
        hit = game_by_key.get(key)
        if hit:
            g, side = hit
            r["game_id"] = g["game_id"]
            r["game_date"] = g["game_date"]
            r["is_home"] = 1 if side == "home" else 0
            matched += 1
        else:
            r["game_id"] = None
            r["game_date"] = None
            r["is_home"] = None
            unmatched += 1
    print(f"\nplayer-week rows matched to a game: {matched}   unmatched: {unmatched}")

    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)

    conn.executemany(
        "INSERT OR REPLACE INTO games (game_id, season, week, season_type, game_date, home_team, away_team) "
        "VALUES (:game_id, :season, :week, :season_type, :game_date, :home_team, :away_team)",
        games)

    conn.executemany(
        """INSERT OR REPLACE INTO player_games
           (player_id, player_name, position, team, opponent, season, week, season_type,
            game_id, game_date, is_home, completions, attempts, passing_yards, passing_tds,
            interceptions, carries, rushing_yards, rushing_tds, receptions, targets,
            receiving_yards, receiving_tds)
           VALUES
           (:player_id, :player_name, :position, :team, :opponent, :season, :week, :season_type,
            :game_id, :game_date, :is_home, :completions, :attempts, :passing_yards, :passing_tds,
            :interceptions, :carries, :rushing_yards, :rushing_tds, :receptions, :targets,
            :receiving_yards, :receiving_tds)""",
        player_rows)
    conn.commit()

    n_games = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    n_pg = conn.execute("SELECT COUNT(*) FROM player_games").fetchone()[0]
    by_season = conn.execute(
        "SELECT season, COUNT(*), SUM(CASE WHEN receptions IS NOT NULL THEN 1 ELSE 0 END) "
        "FROM player_games GROUP BY season ORDER BY season").fetchall()
    conn.close()

    print(f"\nDB WRITTEN: {db_path}")
    print(f"  games: {n_games}   player_games: {n_pg}")
    print(f"  {'season':8s}{'rows':>8s}{'has_receptions':>16s}")
    for s, n, r in by_season:
        print(f"  {s:<8}{n:>8}{r:>16}")
    print("\nNo feature engineering here. Next: a strict-D-1 clean baseline for the")
    print("flagship market (receptions), same pattern as total_bases_clean_baseline_a.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
