#!/usr/bin/env python3
"""
TENNIS_PLAYER_MATCHES_FOUNDATION_A

Data foundation for the tennis pipeline -- same role
cfb_player_games_foundation_a.py plays for CFB. Builds tennis_model.sqlite
with two tables:

    player_matches   one row per player per match (aces, double faults,
                      serve points, surface, best_of, round -- joined
                      with the opponent's own perspective row for the
                      same match)
    matches           one row per match (both players, surface, level,
                      date, score, retirement/walkover flag)

Source: originally targeted Jeff Sackmann's tennis_atp / tennis_wta GitHub
repos -- CONFIRMED GONE as of 2026-09-08 (both return a real "Repository
not found" from an unrestricted GitHub Actions runner via curl, git
ls-remote, and the GitHub REST API; the JeffSackmann account itself is
still active but its only remaining public repo is tennis_MatchChartingProject,
a different point-by-point-only dataset, not a drop-in replacement).

Repointed to Tennismylife/TML-Database (github.com/Tennismylife/TML-Database),
confirmed real and cloneable from an unrestricted runner. Its own README
says it was "originally inspired by Jeff Sackmann's tennis_atp repository"
and its CSV header (verified live against its 2026.csv) carries the exact
same column names this script already depended on -- w_ace/w_df/w_svpt/
w_1stIn/w_1stWon/w_2ndWon/w_SvGms/w_bpSaved/w_bpFaced and the mirrored l_*
columns -- so parse_matches_csv()'s column mapping below is unchanged.
The only real differences: one file per season named `{year}.csv` (not
`{tour}_matches_{year}.csv`), and licensed CC BY-NC-SA (NonCommercial) --
same monetization caveat as before.

ATP ONLY for now: TML-Database's README and repo name are ATP-specific: no
WTA equivalent has been found. Requesting --tours wta raises a clear error
below rather than silently fetching nothing -- WTA support is a real, open
gap in this pipeline until a working WTA source is found, not a stealth
scope cut.

Scope for v1: ATP tour-level main-draw and qualifying matches only --
Challenger-level matches are a deliberate non-goal for this first pass,
same spirit as CFB's rushing/receiving-only first pass leaving passing
props for later.

Retirements/walkovers are real data-quality hazards here: a match that
ended early on injury (RET) or never started (W/O) produces truncated,
non-representative counting stats for the player who "won" it and a
near-zero, meaningless line for the player who didn't finish. Detected
via the `score` field (contains RET/W/O/DEF/ABD markers) and excluded
entirely from player_matches -- not silently averaged in.

Run
---
python -u tennis_player_matches_foundation_a.py --seasons 2015 2016 ... 2025
python -u tennis_player_matches_foundation_a.py --seasons 2024 2025 \
    --raw-dir /data/tennis_raw --db tennis_models/tennis_model.sqlite
"""
import argparse
import csv
import sqlite3
import sys
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

ATP_BASE = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"
WTA_BASE = None  # no confirmed WTA source yet -- see module docstring
DEFAULT_RAW_DIR = Path("/data/tennis_raw")
DEFAULT_DB = Path("/data/tennis_model/tennis_model.sqlite")
UA = {"User-Agent": "tennis-foundation/1.0"}

# Markers inside the raw `score` field that indicate the match didn't
# finish on its own merits -- confirmed convention in Sackmann's data per
# public documentation (not live-verified from this session, see caveat
# above). "ABD"/"ABN" = abandoned (weather etc.), same treatment as RET.
INCOMPLETE_MARKERS = ("RET", "W/O", "WEA", "DEF", "ABD", "ABN")

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    tour TEXT NOT NULL,
    match_date TEXT NOT NULL,
    tourney_id TEXT,
    tourney_name TEXT,
    surface TEXT,
    tourney_level TEXT,
    best_of INTEGER,
    round TEXT,
    winner_id TEXT,
    loser_id TEXT,
    score TEXT,
    is_incomplete INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);

CREATE TABLE IF NOT EXISTS player_matches (
    player_id TEXT NOT NULL,
    player_name TEXT,
    opponent_id TEXT,
    opponent_name TEXT,
    tour TEXT NOT NULL,
    match_id TEXT NOT NULL,
    match_date TEXT NOT NULL,
    tourney_id TEXT,
    tourney_name TEXT,
    surface TEXT,
    tourney_level TEXT,
    best_of INTEGER,
    round TEXT,
    is_winner INTEGER,
    aces INTEGER,
    double_faults INTEGER,
    serve_points INTEGER,
    first_serve_in INTEGER,
    first_serve_won INTEGER,
    second_serve_won INTEGER,
    serve_games INTEGER,
    break_points_saved INTEGER,
    break_points_faced INTEGER,
    PRIMARY KEY (player_id, match_id)
);
CREATE INDEX IF NOT EXISTS idx_pm_player_date ON player_matches(player_id, match_date);
CREATE INDEX IF NOT EXISTS idx_pm_match ON player_matches(match_id);
"""


def _fetch(url, dest, timeout=120):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        f.write(r.read())


def _local_or_fetch(raw_dir, tour, season, timeout=120):
    # Local cache filename stays tour-prefixed for readability even
    # though TML-Database's own source filename (below) is just
    # `{season}.csv` -- no tour prefix, since it's ATP-only.
    path = raw_dir / f"{tour}_matches_{season}.csv"
    if path.exists():
        return path
    if tour == "wta":
        raise SystemExit(
            "no confirmed WTA data source -- JeffSackmann/tennis_wta is "
            "gone and no replacement has been found yet (see module "
            "docstring). Run with --tours atp until this is resolved.")
    url = f"{ATP_BASE}/{season}.csv"
    print(f"  fetching {url} ...", flush=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    _fetch(url, path, timeout=timeout)
    return path


def to_int(v):
    try:
        if v is None or v == "" or v == "NA":
            return None
        return int(round(float(v)))
    except Exception:
        return None


def is_incomplete(score):
    if not score:
        return True
    s = score.upper()
    return any(m in s for m in INCOMPLETE_MARKERS)


def parse_matches_csv(path, tour):
    """One raw CSV row (one match, winner_*/loser_* columns) -> one
    `matches` row + two `player_matches` rows (winner's and loser's own
    perspective). Column names per Sackmann's documented schema -- see
    module docstring's caveat about not being live-verified yet."""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    matches_out = []
    player_rows_out = []

    for r in rows:
        tourney_id = r.get("tourney_id")
        match_num = r.get("match_num")
        if not tourney_id or not match_num:
            continue
        match_id = f"{tour}_{tourney_id}_{match_num}"

        raw_date = r.get("tourney_date") or ""
        match_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}" if len(raw_date) == 8 else None
        if not match_date:
            continue

        score = r.get("score") or ""
        incomplete = is_incomplete(score)
        best_of = to_int(r.get("best_of"))
        surface = r.get("surface") or None
        tourney_level = r.get("tourney_level") or None
        tourney_name = r.get("tourney_name") or None
        round_ = r.get("round") or None
        winner_id = r.get("winner_id")
        loser_id = r.get("loser_id")

        matches_out.append({
            "match_id": match_id, "tour": tour, "match_date": match_date,
            "tourney_id": tourney_id, "tourney_name": tourney_name,
            "surface": surface, "tourney_level": tourney_level, "best_of": best_of,
            "round": round_, "winner_id": winner_id, "loser_id": loser_id,
            "score": score, "is_incomplete": 1 if incomplete else 0,
        })

        if incomplete:
            # Excluded from player_matches entirely -- a RET/W/O match's
            # counting stats (for either player) don't represent a real,
            # complete performance. The match itself is still recorded
            # above so schedule/result lookups aren't silently missing it.
            continue

        common = {
            "tour": tour, "match_id": match_id, "match_date": match_date,
            "tourney_id": tourney_id, "tourney_name": tourney_name,
            "surface": surface, "tourney_level": tourney_level,
            "best_of": best_of, "round": round_,
        }
        for side, opp_side, is_winner in (("winner", "loser", 1), ("loser", "winner", 0)):
            pid = r.get(f"{side}_id")
            if not pid:
                continue
            player_rows_out.append({
                **common,
                "player_id": pid, "player_name": r.get(f"{side}_name"),
                "opponent_id": r.get(f"{opp_side}_id"), "opponent_name": r.get(f"{opp_side}_name"),
                "is_winner": is_winner,
                "aces": to_int(r.get(f"{'w' if side == 'winner' else 'l'}_ace")),
                "double_faults": to_int(r.get(f"{'w' if side == 'winner' else 'l'}_df")),
                "serve_points": to_int(r.get(f"{'w' if side == 'winner' else 'l'}_svpt")),
                "first_serve_in": to_int(r.get(f"{'w' if side == 'winner' else 'l'}_1stIn")),
                "first_serve_won": to_int(r.get(f"{'w' if side == 'winner' else 'l'}_1stWon")),
                "second_serve_won": to_int(r.get(f"{'w' if side == 'winner' else 'l'}_2ndWon")),
                "serve_games": to_int(r.get(f"{'w' if side == 'winner' else 'l'}_SvGms")),
                "break_points_saved": to_int(r.get(f"{'w' if side == 'winner' else 'l'}_bpSaved")),
                "break_points_faced": to_int(r.get(f"{'w' if side == 'winner' else 'l'}_bpFaced")),
            })

    return matches_out, player_rows_out


def selftest_schema(raw_dir, tours):
    """Fetches one real recent season and checks the columns this script
    depends on actually exist, before trusting anything downstream --
    the substitute for the live fetch-and-inspect this script's column
    mapping couldn't get at write time. Run this first on a real run."""
    required = ["tourney_id", "tourney_name", "surface", "tourney_level",
                "tourney_date", "match_num", "winner_id", "winner_name",
                "loser_id", "loser_name", "score", "best_of", "round",
                "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
                "w_SvGms", "w_bpSaved", "w_bpFaced",
                "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
                "l_SvGms", "l_bpSaved", "l_bpFaced"]
    ok = True
    for tour in tours:
        path = _local_or_fetch(raw_dir, tour, 2024)
        header = next(csv.reader(open(path, newline="", encoding="utf-8")))
        missing = [c for c in required if c not in header]
        if missing:
            print(f"  {tour}: MISSING COLUMNS {missing}")
            ok = False
        else:
            print(f"  {tour}: all {len(required)} required columns present")
    print("SELFTEST_SCHEMA PASSED" if ok else "SELFTEST_SCHEMA FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--tours", nargs="+", default=["atp"], choices=["atp", "wta"])
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--selftest-schema", action="store_true",
                     help="Fetch one real season and verify required columns exist, then exit.")
    args = ap.parse_args()

    print("TENNIS_PLAYER_MATCHES_FOUNDATION_A\n===================================")

    if args.selftest_schema:
        ok = selftest_schema(args.raw_dir, args.tours)
        return 0 if ok else 1

    args.db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(args.db))
    con.executescript(SCHEMA)

    total_matches = total_player_rows = total_incomplete = 0
    for tour in args.tours:
        for season in args.seasons:
            path = _local_or_fetch(args.raw_dir, tour, season)
            matches, player_rows = parse_matches_csv(path, tour)
            n_incomplete = sum(m["is_incomplete"] for m in matches)
            print(f"  {tour} {season}: {len(matches)} matches "
                  f"({n_incomplete} incomplete/excluded), {len(player_rows)} player-match rows")
            total_matches += len(matches)
            total_player_rows += len(player_rows)
            total_incomplete += n_incomplete

            con.executemany(
                "INSERT OR REPLACE INTO matches VALUES "
                "(:match_id, :tour, :match_date, :tourney_id, :tourney_name, :surface, "
                ":tourney_level, :best_of, :round, :winner_id, :loser_id, :score, :is_incomplete)",
                matches)
            con.executemany(
                "INSERT OR REPLACE INTO player_matches VALUES "
                "(:player_id, :player_name, :opponent_id, :opponent_name, :tour, :match_id, "
                ":match_date, :tourney_id, :tourney_name, :surface, :tourney_level, :best_of, "
                ":round, :is_winner, :aces, :double_faults, :serve_points, :first_serve_in, "
                ":first_serve_won, :second_serve_won, :serve_games, :break_points_saved, "
                ":break_points_faced)",
                player_rows)
            con.commit()

    print(f"\ntotal: {total_matches} matches ({total_incomplete} incomplete/excluded), "
          f"{total_player_rows} player-match rows written to {args.db}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
