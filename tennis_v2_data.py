"""Tennis V2 Phase0 data layer (research only).

Loads the pinned TML ATP / WTA season files for 2015-2024 ONLY (2025 is sealed, 2026 exposed: never opened), verifies sha256 against source_manifest.json
(fail closed), and builds the production tables by importing tennis_player_matches_foundation_a.parse_matches_csv UNCHANGED so the audit sees exactly the rows the
production foundation produces. Raw CSVs are not redistributed by this repository."""
import hashlib
import json
import os
import re
import sqlite3
import urllib.request
from pathlib import Path

import tennis_player_matches_foundation_a as F

REPO = Path(__file__).resolve().parent
OUT = REPO / "tennis_models" / "tennis_outcome_engine_v2"
MANIFEST = json.loads((OUT / "source_manifest.json").read_text())
RAW_DIR = Path(os.environ.get("TENNIS_V2_RAW", "/tmp/tennis_v2_raw"))
SEASON_CAP = 2024
SEASONS = tuple(range(2015, SEASON_CAP + 1))
UA = {"User-Agent": "tennis-v2-research/1.0"}
ROUND_RANK = {"R128": 1.0, "R64": 2.0, "R32": 3.0, "R16": 4.0, "QF": 5.0, "SF": 6.0, "F": 7.0, "RR": 0.0, "BR": 7.5, "3rd/4th": 7.5}
SET_RE = re.compile(r"^(\d+)-(\d+)")


def url_for(tour, season):
    t = MANIFEST[tour]["url_template"]
    return t.format(season=season)


def fetch_verified(tour, season, raw_dir=RAW_DIR):
    if season > SEASON_CAP:
        raise ValueError("season %d is sealed/exposed and must not be read" % season)
    name = "%s_%d.csv" % (tour, season)
    path = Path(raw_dir) / name
    want = MANIFEST["files"][name]["sha256"]
    if not path.exists():
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url_for(tour, season), headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise RuntimeError("source hash mismatch for %s: %s != %s (failing closed)" % (name, got, want))
    return path


def build_db(tour, raw_dir=RAW_DIR):
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(F.SCHEMA)
    for season in SEASONS:
        path = fetch_verified(tour, season, raw_dir)
        matches, prow = F.parse_matches_csv(path, tour)
        con.executemany("INSERT OR REPLACE INTO matches VALUES (:match_id, :tour, :match_date, :tourney_id, :tourney_name, :surface, :tourney_level, :best_of, :round, :minutes, "
                        ":winner_id, :winner_name, :winner_rank, :winner_rank_points, :winner_age, :loser_id, :loser_name, :loser_rank, :loser_rank_points, :loser_age, :score, :is_incomplete)", matches)
        con.executemany("INSERT OR REPLACE INTO player_matches VALUES (:player_id, :player_name, :opponent_id, :opponent_name, :tour, :match_id, :match_date, :tourney_id, :tourney_name, :surface, "
                        ":tourney_level, :best_of, :round, :is_winner, :aces, :double_faults, :serve_points, :first_serve_in, :first_serve_won, :second_serve_won, :serve_games, "
                        ":break_points_saved, :break_points_faced)", prow)
    con.commit()
    assert con.execute("select max(match_date) from matches").fetchone()[0] <= "%d-12-31" % SEASON_CAP
    return con


def match_num(match_id):
    return int(match_id.rsplit("_", 1)[1])


def as_implemented_key(r):
    return (r["match_date"], r["match_id"])


def canonical_key(r):
    return (r["match_date"], r["tourney_id"] or "", ROUND_RANK.get(r["round"], 3.5), match_num(r["match_id"]))


def parse_sets(score):
    out = []
    for token in (score or "").split():
        m = SET_RE.match(token)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


def tiebreak_sets(score):
    return sum(1 for tok in (score or "").split() if "(" in tok)


def leak_flags(rows_in_order):
    """match_id -> True when, at processing time, a participant had ALREADY been processed in a strictly later round of the same tournament (look-ahead)."""
    seen = {}
    out = {}
    for r in rows_in_order:
        rk = ROUND_RANK.get(r["round"], 3.5)
        flag = False
        for pid in (r["winner_id"], r["loser_id"]):
            prev = seen.get((pid, r["tourney_id"]))
            if prev is not None and prev > rk:
                flag = True
        out[r["match_id"]] = flag
        for pid in (r["winner_id"], r["loser_id"]):
            k = (pid, r["tourney_id"])
            seen[k] = max(seen.get(k, -1), rk)
    return out


def all_matches(con):
    return [dict(r) for r in con.execute("SELECT * FROM matches")]
