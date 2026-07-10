#!/usr/bin/env python3
"""
HITS_RAW_STRUCTURAL_AUDIT_A

Purpose
-------
Run the first raw-signal diagnostic against the locked batter-hits incumbent.

NO model changes.
NO feature promotion.
NO stacking.
NO rescue tuning.

The question is only:

    After controlling for the incumbent's own probability,
    do any structurally plausible candidate signals still show
    monotonic, forward-stable residual signal?

Audit families
--------------
CONTACT SKILL
- batter_contact_rate_60d
- batter_whiff_rate_60d
- batter_k_per_pa_60d

BATTED-BALL QUALITY
- batter_hard_hit_rate_60d
- batter_line_drive_rate_60d
- batter_barrel_rate_60d
- batter_avg_ev_60d

OPPOSING PITCHER CONTACT ALLOWED
- opp_pitcher_k_per_pa_60d
- opp_pitcher_contact_allowed_rate_60d
- opp_pitcher_hit_per_pa_60d
- opp_pitcher_hard_hit_allowed_rate_60d

OPPORTUNITY
- season_pa_per_game
- recent15_pa_avg
- recent5_pa_avg

Validation years
----------------
2024 and 2025 only.

Why:
- The locked rolling-forward baseline already produced untouched fold scores for
  2024 and 2025.
- The available Statcast pitch database covers 2024 onward.
- 2026 is burned for incumbent holdout use and is not used here.

Control method
--------------
For each candidate:
1. Apply minimum sample/coverage gate.
2. Split rows into baseline-probability deciles.
3. Inside each baseline decile, rank candidate values into ten bins.
4. Aggregate actual-minus-baseline-prediction residual by candidate bin.
5. Report raw hit rate, baseline prediction, residual, monotonicity,
   D10-D1 residual spread, and year stability.

This is a diagnostic only. No candidate is promoted by this script.

Memory safety
-------------
Designed for a 512 MB Render instance:
- stdlib only in parent analysis
- no pandas
- no sklearn
- no full Statcast load into Python RAM
- SQLite temp work is forced to disk
- uses existing rolling-forward baseline score files
- streams season JSONL line by line

Prerequisites
-------------
Existing files from the completed hits rolling-forward baseline:
    /data/hr_model/hits_rfv_lite_b/scores_2024.json
    /data/hr_model/hits_rfv_lite_b/scores_2025.json

Historical source files:
    /data/season_2024.jsonl or .json
    /data/season_2025.jsonl or .json

Statcast database:
    /data/hr_model/hr_model.sqlite
    table: statcast_pitches

Run
---
python -u hits_raw_structural_audit_a.py 2>&1 | tee /data/hr_model/hits_raw_structural_audit_a.log

Outputs
-------
/data/hr_model/hits_raw_structural_audit_a_results.json
/data/hr_model/hits_raw_structural_audit_a_report.txt
/data/hr_model/hits_raw_structural_audit_a_work/audit.sqlite

Paste back
----------
AUDIT PREFLIGHT
POOLED CONTROLLED SIGNAL RANKING
YEAR STABILITY
NEXT READ
"""

import csv
import glob
import json
import math
import re
import sqlite3
import sys
from collections import deque
from pathlib import Path

DB_PATH = Path("/data/hr_model/hr_model.sqlite")
BASELINE_DIR = Path("/data/hr_model/hits_rfv_lite_b")
WORK_DIR = Path("/data/hr_model/hits_raw_structural_audit_a_work")
WORK_DB = WORK_DIR / "audit.sqlite"
OUT_JSON = Path("/data/hr_model/hits_raw_structural_audit_a_results.json")
OUT_TXT = Path("/data/hr_model/hits_raw_structural_audit_a_report.txt")

YEARS = [2024, 2025]

FEATURE_ALIASES = {
    "player_id": ["player_id", "batter_id", "person_id", "id"],
    "date": ["date", "game_date"],
    "hits": ["hits", "h"],
    "ab": ["at_bats", "atBats", "ab"],
    "pa": ["plate_appearances", "plateAppearances", "pa"],
    "hr": ["home_runs", "homeRuns", "hr"],
    "bb": ["walks", "base_on_balls", "baseOnBalls", "bb"],
    "so": ["strikeouts", "strike_outs", "strikeOuts", "so"],
    "batting_order": ["batting_order", "battingOrder", "lineup_spot", "order"],
    "player_type": ["player_type", "type"],
    "game_id": ["game_id", "game_pk", "gamePk"],
}

# Expected direction is the structural prior only.
# +1 means higher candidate values should produce more positive residual.
# -1 means higher candidate values should produce more negative residual.
CANDIDATES = {
    "batter_contact_rate_60d": {
        "family": "contact_skill",
        "direction": +1,
        "sample_col": "b_swings_60d",
        "min_sample": 100,
    },
    "batter_whiff_rate_60d": {
        "family": "contact_skill",
        "direction": -1,
        "sample_col": "b_swings_60d",
        "min_sample": 100,
    },
    "batter_k_per_pa_60d": {
        "family": "contact_skill",
        "direction": -1,
        "sample_col": "b_pa_60d",
        "min_sample": 50,
    },
    "batter_hard_hit_rate_60d": {
        "family": "batted_ball_quality",
        "direction": +1,
        "sample_col": "b_bbe_60d",
        "min_sample": 30,
    },
    "batter_line_drive_rate_60d": {
        "family": "batted_ball_quality",
        "direction": +1,
        "sample_col": "b_bbe_60d",
        "min_sample": 30,
    },
    "batter_barrel_rate_60d": {
        "family": "batted_ball_quality",
        "direction": +1,
        "sample_col": "b_bbe_60d",
        "min_sample": 30,
    },
    "batter_avg_ev_60d": {
        "family": "batted_ball_quality",
        "direction": +1,
        "sample_col": "b_ev_n_60d",
        "min_sample": 30,
    },
    "opp_pitcher_k_per_pa_60d": {
        "family": "opposing_pitcher",
        "direction": -1,
        "sample_col": "p_pa_60d",
        "min_sample": 75,
    },
    "opp_pitcher_contact_allowed_rate_60d": {
        "family": "opposing_pitcher",
        "direction": +1,
        "sample_col": "p_swings_60d",
        "min_sample": 150,
    },
    "opp_pitcher_hit_per_pa_60d": {
        "family": "opposing_pitcher",
        "direction": +1,
        "sample_col": "p_pa_60d",
        "min_sample": 75,
    },
    "opp_pitcher_hard_hit_allowed_rate_60d": {
        "family": "opposing_pitcher",
        "direction": +1,
        "sample_col": "p_bbe_60d",
        "min_sample": 50,
    },
    "season_pa_per_game": {
        "family": "opportunity",
        "direction": +1,
        "sample_col": None,
        "min_sample": 0,
    },
    "recent15_pa_avg": {
        "family": "opportunity",
        "direction": +1,
        "sample_col": None,
        "min_sample": 0,
    },
    "recent5_pa_avg": {
        "family": "opportunity",
        "direction": +1,
        "sample_col": None,
        "min_sample": 0,
    },
}


def first_value(row, names, default=None):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    for container in ("stats", "hitting", "batting"):
        nested = row.get(container)
        if isinstance(nested, dict):
            for name in names:
                if name in nested and nested[name] is not None:
                    return nested[name]
    return default


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def discover_season_file(year):
    candidates = [
        Path(f"/data/season_{year}.jsonl"),
        Path(f"/data/season_{year}.json"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def iter_json_records(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = ""
        pos = f.tell()
        for line in f:
            if line.strip():
                first = line.lstrip()[0]
                break
        f.seek(pos)

        if first == "[":
            raise RuntimeError(
                f"{path} is one large JSON array. "
                "This audit refuses to json.load() it under the 512 MB memory limit."
            )

        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(
                    f"JSON parse failure in {path} line {line_no}: {e}"
                ) from e


def normalize_batter_row(row, seq):
    ptype = str(
        first_value(row, FEATURE_ALIASES["player_type"], "") or ""
    ).lower()

    if ptype and "batter" not in ptype and "hitter" not in ptype:
        return None

    player_id = first_value(row, FEATURE_ALIASES["player_id"])
    game_date = first_value(row, FEATURE_ALIASES["date"])
    hits = first_value(row, FEATURE_ALIASES["hits"])
    ab = first_value(row, FEATURE_ALIASES["ab"])

    if player_id is None or game_date is None or hits is None or ab is None:
        return None

    bb = to_float(first_value(row, FEATURE_ALIASES["bb"], 0))
    pa_raw = first_value(row, FEATURE_ALIASES["pa"], None)
    pa = to_float(pa_raw, to_float(ab) + bb)

    return (
        str(player_id),
        str(game_date),
        str(first_value(row, FEATURE_ALIASES["game_id"], seq)),
        int(seq),
        to_float(hits),
        to_float(ab),
        pa,
        to_float(first_value(row, FEATURE_ALIASES["hr"], 0)),
        bb,
        to_float(first_value(row, FEATURE_ALIASES["so"], 0)),
        to_float(first_value(row, FEATURE_ALIASES["batting_order"], 0)),
    )


def prepare_work_db():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(WORK_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-20000")
    return conn


def build_audit_rows_for_year(conn, year, season_path, scores_path):
    table = f"audit_rows_{year}"

    existing = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()

    if existing:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{year}: audit rows REUSED | rows={count:,}", flush=True)
        return count

    print(f"{year}: building audit-row alignment...", flush=True)

    spool = f"spool_{year}"
    conn.execute(f"DROP TABLE IF EXISTS {spool}")
    conn.execute(
        f"""
        CREATE TABLE {spool} (
            player_id TEXT,
            game_date TEXT,
            game_id TEXT,
            seq INTEGER,
            hits REAL,
            ab REAL,
            pa REAL,
            hr REAL,
            bb REAL,
            so REAL,
            batting_order REAL
        )
        """
    )

    batch = []
    raw_seen = 0
    normalized = 0

    for seq, row in enumerate(iter_json_records(season_path), 1):
        raw_seen += 1
        norm = normalize_batter_row(row, seq)
        if norm is None:
            continue
        batch.append(norm)
        normalized += 1

        if len(batch) >= 2000:
            conn.executemany(
                f"INSERT INTO {spool} VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                batch,
            )
            conn.commit()
            batch.clear()

    if batch:
        conn.executemany(
            f"INSERT INTO {spool} VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            batch,
        )
        conn.commit()

    conn.execute(
        f"""
        CREATE INDEX idx_{spool}
        ON {spool}(player_id, game_date, game_id, seq)
        """
    )
    conn.commit()

    scored = json.loads(scores_path.read_text(encoding="utf-8"))
    y_binary = scored["y_binary"]
    probs = scored["probs"]

    conn.execute(
        f"""
        CREATE TABLE {table} (
            row_id INTEGER PRIMARY KEY,
            year INTEGER,
            player_id TEXT,
            game_date TEXT,
            game_id TEXT,
            actual_hits REAL,
            actual_hit INTEGER,
            baseline_prob REAL,
            batting_order REAL,
            season_pa_per_game REAL,
            recent15_pa_avg REAL,
            recent5_pa_avg REAL
        )
        """
    )

    cursor = conn.execute(
        f"""
        SELECT
            player_id, game_date, game_id, seq,
            hits, ab, pa, hr, bb, so, batting_order
        FROM {spool}
        ORDER BY player_id, game_date, game_id, seq
        """
    )

    current_player = None
    cum_ab = 0.0
    cum_pa = 0.0
    games = 0
    recent_pa = deque(maxlen=15)

    eligible_idx = 0
    insert_batch = []

    for row in cursor:
        (
            player_id,
            game_date,
            game_id,
            seq,
            hits,
            ab,
            pa,
            hr,
            bb,
            so,
            batting_order,
        ) = row

        if player_id != current_player:
            current_player = player_id
            cum_ab = 0.0
            cum_pa = 0.0
            games = 0
            recent_pa = deque(maxlen=15)

        if cum_ab >= 20.0 and games >= 5:
            if eligible_idx >= len(probs):
                raise RuntimeError(
                    f"{year}: built more eligible rows than score file."
                )

            actual_hit = 1 if float(hits) >= 1.0 else 0
            if actual_hit != int(y_binary[eligible_idx]):
                raise RuntimeError(
                    f"{year}: score alignment failure at row {eligible_idx}. "
                    f"rebuilt actual={actual_hit}, score actual={y_binary[eligible_idx]}"
                )

            last5 = list(recent_pa)[-5:]

            insert_batch.append(
                (
                    eligible_idx,
                    year,
                    str(player_id),
                    str(game_date),
                    str(game_id),
                    float(hits),
                    actual_hit,
                    float(probs[eligible_idx]),
                    float(batting_order),
                    (cum_pa / games) if games > 0 else None,
                    (sum(recent_pa) / len(recent_pa)) if recent_pa else None,
                    (sum(last5) / len(last5)) if last5 else None,
                )
            )

            eligible_idx += 1

            if len(insert_batch) >= 2000:
                conn.executemany(
                    f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    insert_batch,
                )
                conn.commit()
                insert_batch.clear()

        cum_ab += float(ab)
        cum_pa += float(pa)
        games += 1
        recent_pa.append(float(pa))

    if insert_batch:
        conn.executemany(
            f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            insert_batch,
        )
        conn.commit()

    if eligible_idx != len(probs):
        raise RuntimeError(
            f"{year}: score alignment count mismatch. "
            f"rebuilt={eligible_idx:,}, score_file={len(probs):,}"
        )

    conn.execute(
        f"CREATE INDEX idx_{table}_player_date "
        f"ON {table}(player_id, game_date)"
    )
    conn.execute(
        f"CREATE INDEX idx_{table}_game "
        f"ON {table}(game_id, player_id)"
    )
    conn.commit()

    conn.execute(f"DROP TABLE {spool}")
    conn.commit()

    print(
        f"{year}: raw={raw_seen:,} normalized={normalized:,} "
        f"aligned_eligible={eligible_idx:,}",
        flush=True,
    )

    return eligible_idx


def build_statcast_derived_tables(conn):
    exists = conn.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='batter_roll60'
        """
    ).fetchone()

    if exists:
        print("Statcast rolling tables REUSED.", flush=True)
        return

    print("Building Statcast daily aggregates on disk...", flush=True)

    # Attach source read-only.
    conn.execute(
        f"ATTACH DATABASE 'file:{DB_PATH}?mode=ro' AS src"
    )

    stat_min, stat_max, stat_n = conn.execute(
        """
        SELECT MIN(game_date), MAX(game_date), COUNT(*)
        FROM src.statcast_pitches
        WHERE game_date >= '2024-01-01'
          AND game_date < '2026-01-01'
        """
    ).fetchone()

    print(
        f"Statcast source rows 2024-2025={stat_n:,} "
        f"range={stat_min}..{stat_max}",
        flush=True,
    )

    conn.execute("DROP TABLE IF EXISTS batter_daily")
    conn.execute(
        """
        CREATE TABLE batter_daily AS
        SELECT
            CAST(batter_id AS TEXT) AS player_id,
            game_date,
            SUM(
                CASE WHEN description IN (
                    'swinging_strike',
                    'swinging_strike_blocked',
                    'foul',
                    'foul_tip',
                    'hit_into_play',
                    'hit_into_play_no_out',
                    'hit_into_play_score',
                    'missed_bunt',
                    'foul_bunt'
                ) THEN 1 ELSE 0 END
            ) AS swings,
            SUM(COALESCE(is_whiff, 0)) AS whiffs,
            SUM(
                CASE WHEN events IS NOT NULL AND events <> ''
                THEN 1 ELSE 0 END
            ) AS pa,
            SUM(
                CASE WHEN events IN ('strikeout','strikeout_double_play')
                THEN 1 ELSE 0 END
            ) AS strikeouts,
            SUM(COALESCE(is_bbe, 0)) AS bbe,
            SUM(
                CASE WHEN events IN ('single','double','triple','home_run')
                THEN 1 ELSE 0 END
            ) AS hits,
            SUM(
                CASE WHEN COALESCE(is_bbe,0)=1 AND launch_speed >= 95
                THEN 1 ELSE 0 END
            ) AS hard_hit,
            SUM(
                CASE WHEN COALESCE(is_bbe,0)=1 AND bb_type='line_drive'
                THEN 1 ELSE 0 END
            ) AS line_drives,
            SUM(COALESCE(is_barrel, 0)) AS barrels,
            SUM(
                CASE WHEN COALESCE(is_bbe,0)=1 AND launch_speed IS NOT NULL
                THEN launch_speed ELSE 0 END
            ) AS ev_sum,
            SUM(
                CASE WHEN COALESCE(is_bbe,0)=1 AND launch_speed IS NOT NULL
                THEN 1 ELSE 0 END
            ) AS ev_n
        FROM src.statcast_pitches
        WHERE game_date >= '2024-01-01'
          AND game_date < '2026-01-01'
        GROUP BY CAST(batter_id AS TEXT), game_date
        """
    )
    conn.execute(
        "CREATE INDEX idx_batter_daily ON batter_daily(player_id, game_date)"
    )
    conn.commit()

    conn.execute("DROP TABLE IF EXISTS pitcher_daily")
    conn.execute(
        """
        CREATE TABLE pitcher_daily AS
        SELECT
            CAST(pitcher_id AS TEXT) AS pitcher_id,
            game_date,
            SUM(
                CASE WHEN description IN (
                    'swinging_strike',
                    'swinging_strike_blocked',
                    'foul',
                    'foul_tip',
                    'hit_into_play',
                    'hit_into_play_no_out',
                    'hit_into_play_score',
                    'missed_bunt',
                    'foul_bunt'
                ) THEN 1 ELSE 0 END
            ) AS swings,
            SUM(COALESCE(is_whiff, 0)) AS whiffs,
            SUM(
                CASE WHEN events IS NOT NULL AND events <> ''
                THEN 1 ELSE 0 END
            ) AS pa,
            SUM(
                CASE WHEN events IN ('strikeout','strikeout_double_play')
                THEN 1 ELSE 0 END
            ) AS strikeouts,
            SUM(COALESCE(is_bbe, 0)) AS bbe,
            SUM(
                CASE WHEN events IN ('single','double','triple','home_run')
                THEN 1 ELSE 0 END
            ) AS hits,
            SUM(
                CASE WHEN COALESCE(is_bbe,0)=1 AND launch_speed >= 95
                THEN 1 ELSE 0 END
            ) AS hard_hit
        FROM src.statcast_pitches
        WHERE game_date >= '2024-01-01'
          AND game_date < '2026-01-01'
        GROUP BY CAST(pitcher_id AS TEXT), game_date
        """
    )
    conn.execute(
        "CREATE INDEX idx_pitcher_daily ON pitcher_daily(pitcher_id, game_date)"
    )
    conn.commit()

    print("Building first-pitcher batter-game map...", flush=True)

    conn.execute("DROP TABLE IF EXISTS batter_game_pitcher")
    conn.execute(
        """
        CREATE TABLE batter_game_pitcher AS
        SELECT
            CAST(game_pk AS TEXT) AS game_id,
            CAST(batter_id AS TEXT) AS player_id,
            CAST(pitcher_id AS TEXT) AS pitcher_id
        FROM (
            SELECT
                game_pk,
                batter_id,
                pitcher_id,
                ROW_NUMBER() OVER (
                    PARTITION BY game_pk, batter_id
                    ORDER BY
                        COALESCE(at_bat_number, 999999),
                        COALESCE(pitch_number, 999999)
                ) AS rn
            FROM src.statcast_pitches
            WHERE game_date >= '2024-01-01'
              AND game_date < '2026-01-01'
        )
        WHERE rn=1
        """
    )
    conn.execute(
        "CREATE INDEX idx_bgp "
        "ON batter_game_pitcher(game_id, player_id)"
    )
    conn.commit()

    print("Building 60-day lagged batter snapshots...", flush=True)

    conn.execute("DROP TABLE IF EXISTS batter_roll60")
    conn.execute(
        """
        CREATE TABLE batter_roll60 AS
        SELECT
            player_id,
            game_date,
            SUM(swings) OVER w AS swings_60d,
            SUM(whiffs) OVER w AS whiffs_60d,
            SUM(pa) OVER w AS pa_60d,
            SUM(strikeouts) OVER w AS strikeouts_60d,
            SUM(bbe) OVER w AS bbe_60d,
            SUM(hits) OVER w AS hits_60d,
            SUM(hard_hit) OVER w AS hard_hit_60d,
            SUM(line_drives) OVER w AS line_drives_60d,
            SUM(barrels) OVER w AS barrels_60d,
            SUM(ev_sum) OVER w AS ev_sum_60d,
            SUM(ev_n) OVER w AS ev_n_60d
        FROM batter_daily
        WINDOW w AS (
            PARTITION BY player_id
            ORDER BY julianday(game_date)
            RANGE BETWEEN 60 PRECEDING AND 1 PRECEDING
        )
        """
    )
    conn.execute(
        "CREATE INDEX idx_batter_roll60 "
        "ON batter_roll60(player_id, game_date)"
    )
    conn.commit()

    print("Building 60-day lagged pitcher snapshots...", flush=True)

    conn.execute("DROP TABLE IF EXISTS pitcher_roll60")
    conn.execute(
        """
        CREATE TABLE pitcher_roll60 AS
        SELECT
            pitcher_id,
            game_date,
            SUM(swings) OVER w AS swings_60d,
            SUM(whiffs) OVER w AS whiffs_60d,
            SUM(pa) OVER w AS pa_60d,
            SUM(strikeouts) OVER w AS strikeouts_60d,
            SUM(bbe) OVER w AS bbe_60d,
            SUM(hits) OVER w AS hits_60d,
            SUM(hard_hit) OVER w AS hard_hit_60d
        FROM pitcher_daily
        WINDOW w AS (
            PARTITION BY pitcher_id
            ORDER BY julianday(game_date)
            RANGE BETWEEN 60 PRECEDING AND 1 PRECEDING
        )
        """
    )
    conn.execute(
        "CREATE INDEX idx_pitcher_roll60 "
        "ON pitcher_roll60(pitcher_id, game_date)"
    )
    conn.commit()

    conn.execute("DETACH DATABASE src")
    conn.commit()


def build_joined_audit_table(conn):
    exists = conn.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='audit_joined'
        """
    ).fetchone()

    if exists:
        count = conn.execute(
            "SELECT COUNT(*) FROM audit_joined"
        ).fetchone()[0]
        print(f"Joined audit table REUSED | rows={count:,}", flush=True)
        return

    print("Joining incumbent predictions to candidate signals...", flush=True)

    conn.execute("DROP TABLE IF EXISTS audit_all")
    conn.execute(
        """
        CREATE TABLE audit_all AS
        SELECT * FROM audit_rows_2024
        UNION ALL
        SELECT * FROM audit_rows_2025
        """
    )
    conn.execute(
        "CREATE INDEX idx_audit_all_player_date "
        "ON audit_all(player_id, game_date)"
    )
    conn.execute(
        "CREATE INDEX idx_audit_all_game "
        "ON audit_all(game_id, player_id)"
    )
    conn.commit()

    conn.execute("DROP TABLE IF EXISTS audit_joined")
    conn.execute(
        """
        CREATE TABLE audit_joined AS
        SELECT
            a.row_id,
            a.year,
            a.player_id,
            a.game_date,
            a.game_id,
            a.actual_hits,
            a.actual_hit,
            a.baseline_prob,
            a.batting_order,
            a.season_pa_per_game,
            a.recent15_pa_avg,
            a.recent5_pa_avg,

            br.swings_60d AS b_swings_60d,
            br.whiffs_60d AS b_whiffs_60d,
            br.pa_60d AS b_pa_60d,
            br.strikeouts_60d AS b_strikeouts_60d,
            br.bbe_60d AS b_bbe_60d,
            br.hits_60d AS b_hits_60d,
            br.hard_hit_60d AS b_hard_hit_60d,
            br.line_drives_60d AS b_line_drives_60d,
            br.barrels_60d AS b_barrels_60d,
            br.ev_sum_60d AS b_ev_sum_60d,
            br.ev_n_60d AS b_ev_n_60d,

            bgp.pitcher_id AS opp_pitcher_id,

            pr.swings_60d AS p_swings_60d,
            pr.whiffs_60d AS p_whiffs_60d,
            pr.pa_60d AS p_pa_60d,
            pr.strikeouts_60d AS p_strikeouts_60d,
            pr.bbe_60d AS p_bbe_60d,
            pr.hits_60d AS p_hits_60d,
            pr.hard_hit_60d AS p_hard_hit_60d,

            CASE
                WHEN br.swings_60d > 0
                THEN 1.0 - (1.0 * br.whiffs_60d / br.swings_60d)
            END AS batter_contact_rate_60d,

            CASE
                WHEN br.swings_60d > 0
                THEN 1.0 * br.whiffs_60d / br.swings_60d
            END AS batter_whiff_rate_60d,

            CASE
                WHEN br.pa_60d > 0
                THEN 1.0 * br.strikeouts_60d / br.pa_60d
            END AS batter_k_per_pa_60d,

            CASE
                WHEN br.bbe_60d > 0
                THEN 1.0 * br.hard_hit_60d / br.bbe_60d
            END AS batter_hard_hit_rate_60d,

            CASE
                WHEN br.bbe_60d > 0
                THEN 1.0 * br.line_drives_60d / br.bbe_60d
            END AS batter_line_drive_rate_60d,

            CASE
                WHEN br.bbe_60d > 0
                THEN 1.0 * br.barrels_60d / br.bbe_60d
            END AS batter_barrel_rate_60d,

            CASE
                WHEN br.ev_n_60d > 0
                THEN 1.0 * br.ev_sum_60d / br.ev_n_60d
            END AS batter_avg_ev_60d,

            CASE
                WHEN pr.pa_60d > 0
                THEN 1.0 * pr.strikeouts_60d / pr.pa_60d
            END AS opp_pitcher_k_per_pa_60d,

            CASE
                WHEN pr.swings_60d > 0
                THEN 1.0 - (1.0 * pr.whiffs_60d / pr.swings_60d)
            END AS opp_pitcher_contact_allowed_rate_60d,

            CASE
                WHEN pr.pa_60d > 0
                THEN 1.0 * pr.hits_60d / pr.pa_60d
            END AS opp_pitcher_hit_per_pa_60d,

            CASE
                WHEN pr.bbe_60d > 0
                THEN 1.0 * pr.hard_hit_60d / pr.bbe_60d
            END AS opp_pitcher_hard_hit_allowed_rate_60d

        FROM audit_all a
        LEFT JOIN batter_roll60 br
          ON br.player_id = a.player_id
         AND br.game_date = a.game_date
        LEFT JOIN batter_game_pitcher bgp
          ON bgp.game_id = a.game_id
         AND bgp.player_id = a.player_id
        LEFT JOIN pitcher_roll60 pr
          ON pr.pitcher_id = bgp.pitcher_id
         AND pr.game_date = a.game_date
        """
    )
    conn.execute(
        "CREATE INDEX idx_audit_joined_year "
        "ON audit_joined(year)"
    )
    conn.commit()

    count = conn.execute(
        "SELECT COUNT(*) FROM audit_joined"
    ).fetchone()[0]
    print(f"Joined audit rows={count:,}", flush=True)


def spearman_deciles(deciles):
    if len(deciles) < 3:
        return None

    xs = [float(d["decile"]) for d in deciles]
    ys = [float(d["mean_residual"]) for d in deciles]

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(xs, ys)
    )
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if den_x == 0 or den_y == 0:
        return 0.0

    return num / (den_x * den_y)


def assign_quantile_bins(indices, values, n_bins=10):
    ordered = sorted(indices, key=lambda i: values[i])
    bins = {}

    n = len(ordered)
    if n == 0:
        return bins

    for rank, idx in enumerate(ordered):
        b = min(n_bins, int(rank * n_bins / n) + 1)
        bins[idx] = b

    return bins


def controlled_decile_audit(rows, candidate_name, config):
    values = [r[candidate_name] for r in rows]
    probs = [r["baseline_prob"] for r in rows]

    # Baseline-probability deciles.
    all_idx = list(range(len(rows)))
    baseline_bins = assign_quantile_bins(all_idx, probs, 10)

    candidate_bins = {}

    for baseline_decile in range(1, 11):
        idx = [
            i for i in all_idx
            if baseline_bins.get(i) == baseline_decile
        ]
        inner = assign_quantile_bins(idx, values, 10)
        candidate_bins.update(inner)

    groups = {d: [] for d in range(1, 11)}

    for i, row in enumerate(rows):
        d = candidate_bins.get(i)
        if d is not None:
            groups[d].append(row)

    deciles = []

    for d in range(1, 11):
        g = groups[d]
        if not g:
            continue

        n = len(g)
        mean_candidate = sum(r[candidate_name] for r in g) / n
        actual = sum(r["actual_hit"] for r in g) / n
        pred = sum(r["baseline_prob"] for r in g) / n
        residual = actual - pred

        deciles.append(
            {
                "decile": d,
                "rows": n,
                "mean_candidate": mean_candidate,
                "actual_hit_rate": actual,
                "mean_baseline_prob": pred,
                "mean_residual": residual,
            }
        )

    rho = spearman_deciles(deciles)

    if len(deciles) >= 2:
        raw_span = (
            deciles[-1]["mean_residual"]
            - deciles[0]["mean_residual"]
        )
    else:
        raw_span = None

    direction = config["direction"]
    oriented_span = (
        raw_span * direction
        if raw_span is not None
        else None
    )
    oriented_rho = (
        rho * direction
        if rho is not None
        else None
    )

    return {
        "candidate": candidate_name,
        "family": config["family"],
        "expected_direction": direction,
        "rows": len(rows),
        "deciles": deciles,
        "residual_spearman": rho,
        "oriented_residual_spearman": oriented_rho,
        "raw_d10_minus_d1_residual": raw_span,
        "oriented_d10_minus_d1_residual": oriented_span,
    }


def fetch_candidate_rows(conn, candidate_name, config, year=None):
    where = [f"{candidate_name} IS NOT NULL"]
    params = []

    sample_col = config["sample_col"]
    if sample_col:
        where.append(f"{sample_col} >= ?")
        params.append(config["min_sample"])

    if year is not None:
        where.append("year = ?")
        params.append(year)

    sql = f"""
        SELECT
            year,
            actual_hit,
            baseline_prob,
            {candidate_name} AS candidate_value
        FROM audit_joined
        WHERE {' AND '.join(where)}
    """

    rows = []

    for year_v, actual_hit, baseline_prob, candidate_value in conn.execute(
        sql, params
    ):
        rows.append(
            {
                "year": int(year_v),
                "actual_hit": int(actual_hit),
                "baseline_prob": float(baseline_prob),
                candidate_name: float(candidate_value),
            }
        )

    return rows


def run_audits(conn):
    results = {}

    for candidate_name, config in CANDIDATES.items():
        pooled_rows = fetch_candidate_rows(
            conn,
            candidate_name,
            config,
            year=None,
        )

        pooled = (
            controlled_decile_audit(
                pooled_rows,
                candidate_name,
                config,
            )
            if len(pooled_rows) >= 1000
            else {
                "candidate": candidate_name,
                "family": config["family"],
                "rows": len(pooled_rows),
                "insufficient_rows": True,
            }
        )

        by_year = {}

        for year in YEARS:
            year_rows = fetch_candidate_rows(
                conn,
                candidate_name,
                config,
                year=year,
            )

            by_year[str(year)] = (
                controlled_decile_audit(
                    year_rows,
                    candidate_name,
                    config,
                )
                if len(year_rows) >= 500
                else {
                    "candidate": candidate_name,
                    "family": config["family"],
                    "rows": len(year_rows),
                    "insufficient_rows": True,
                }
            )

        year_signs = []

        for year in YEARS:
            yr = by_year[str(year)]
            span = yr.get("oriented_d10_minus_d1_residual")
            if span is not None:
                year_signs.append(span > 0)

        stable_same_direction = (
            len(year_signs) == len(YEARS)
            and all(year_signs)
        )

        results[candidate_name] = {
            "config": config,
            "pooled": pooled,
            "by_year": by_year,
            "stable_same_expected_direction_both_years": stable_same_direction,
        }

    return results


def build_summary(results):
    ranking = []

    for name, result in results.items():
        pooled = result["pooled"]

        if pooled.get("insufficient_rows"):
            continue

        ranking.append(
            {
                "candidate": name,
                "family": pooled["family"],
                "rows": pooled["rows"],
                "oriented_residual_spearman": pooled[
                    "oriented_residual_spearman"
                ],
                "oriented_d10_minus_d1_residual": pooled[
                    "oriented_d10_minus_d1_residual"
                ],
                "stable_both_years": result[
                    "stable_same_expected_direction_both_years"
                ],
                "span_2024": result["by_year"]["2024"].get(
                    "oriented_d10_minus_d1_residual"
                ),
                "span_2025": result["by_year"]["2025"].get(
                    "oriented_d10_minus_d1_residual"
                ),
            }
        )

    ranking.sort(
        key=lambda x: (
            x["stable_both_years"],
            x["oriented_d10_minus_d1_residual"]
            if x["oriented_d10_minus_d1_residual"] is not None
            else -999,
        ),
        reverse=True,
    )

    return ranking


def write_text_report(preflight, results, ranking):
    lines = []

    lines.append("HITS_RAW_STRUCTURAL_AUDIT_A")
    lines.append("=" * 28)
    lines.append("")
    lines.append("AUDIT PREFLIGHT")
    lines.append("---------------")

    for key, value in preflight.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("POOLED CONTROLLED SIGNAL RANKING")
    lines.append("--------------------------------")

    for i, row in enumerate(ranking, 1):
        lines.append(
            f"{i:2d}. {row['candidate']} | "
            f"family={row['family']} "
            f"n={row['rows']:,} "
            f"oriented_span={row['oriented_d10_minus_d1_residual']:+.6f} "
            f"oriented_rho={row['oriented_residual_spearman']:+.4f} "
            f"stable_both_years={row['stable_both_years']}"
        )

    lines.append("")
    lines.append("YEAR STABILITY")
    lines.append("--------------")

    for row in ranking:
        s24 = row["span_2024"]
        s25 = row["span_2025"]

        s24_txt = "NA" if s24 is None else f"{s24:+.6f}"
        s25_txt = "NA" if s25 is None else f"{s25:+.6f}"

        lines.append(
            f"{row['candidate']}: "
            f"2024={s24_txt} "
            f"2025={s25_txt} "
            f"stable={row['stable_both_years']}"
        )

    lines.append("")
    lines.append("NEXT READ")
    lines.append("---------")
    lines.append(
        "This audit does not promote any feature. "
        "A candidate must first show meaningful baseline-controlled residual "
        "signal, expected-direction behavior, adequate coverage, and year stability "
        "before it can be authorized for a single-scalar model gate."
    )

    lines.append("")
    lines.append("DETAILED DECILES")
    lines.append("----------------")

    for name, result in results.items():
        lines.append("")
        lines.append(name)
        lines.append("~" * len(name))

        for scope_name, audit in [
            ("POOLED", result["pooled"]),
            ("2024", result["by_year"]["2024"]),
            ("2025", result["by_year"]["2025"]),
        ]:
            lines.append(f"[{scope_name}]")

            if audit.get("insufficient_rows"):
                lines.append(
                    f"insufficient_rows={audit.get('rows', 0)}"
                )
                continue

            lines.append(
                f"rows={audit['rows']} "
                f"oriented_span={audit['oriented_d10_minus_d1_residual']:+.6f} "
                f"oriented_rho={audit['oriented_residual_spearman']:+.4f}"
            )

            for d in audit["deciles"]:
                lines.append(
                    f"d{d['decile']:02d} "
                    f"n={d['rows']:,} "
                    f"x={d['mean_candidate']:.6f} "
                    f"actual={d['actual_hit_rate']:.6f} "
                    f"pred={d['mean_baseline_prob']:.6f} "
                    f"resid={d['mean_residual']:+.6f}"
                )

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("HITS_RAW_STRUCTURAL_AUDIT_A", flush=True)
    print("===========================", flush=True)

    print("\nAUDIT PREFLIGHT", flush=True)
    print("---------------", flush=True)

    if not DB_PATH.exists():
        raise RuntimeError(f"Missing Statcast DB: {DB_PATH}")

    season_files = {}
    score_files = {}

    for year in YEARS:
        season_path = discover_season_file(year)
        score_path = BASELINE_DIR / f"scores_{year}.json"

        if season_path is None:
            raise RuntimeError(f"Missing season file for {year}")
        if not score_path.exists():
            raise RuntimeError(
                f"Missing rolling-forward score file: {score_path}"
            )

        season_files[year] = season_path
        score_files[year] = score_path

        print(
            f"{year}: season={season_path} "
            f"score={score_path}",
            flush=True,
        )

    print(f"statcast_db={DB_PATH}", flush=True)
    print("2026_use=FORBIDDEN_AS_CLEAN_HOLDOUT", flush=True)
    print("model_changes=NONE", flush=True)

    conn = prepare_work_db()

    audit_counts = {}

    for year in YEARS:
        audit_counts[year] = build_audit_rows_for_year(
            conn,
            year,
            season_files[year],
            score_files[year],
        )

    print("\nSTATCAST FEATURE BUILD", flush=True)
    print("----------------------", flush=True)

    build_statcast_derived_tables(conn)
    build_joined_audit_table(conn)

    print("\nRUNNING BASELINE-CONTROLLED DECILE AUDITS...", flush=True)
    results = run_audits(conn)
    ranking = build_summary(results)

    joined_count = conn.execute(
        "SELECT COUNT(*) FROM audit_joined"
    ).fetchone()[0]

    batter_cov = conn.execute(
        """
        SELECT
            AVG(CASE WHEN b_swings_60d IS NOT NULL THEN 1.0 ELSE 0 END)
        FROM audit_joined
        """
    ).fetchone()[0]

    pitcher_cov = conn.execute(
        """
        SELECT
            AVG(CASE WHEN p_swings_60d IS NOT NULL THEN 1.0 ELSE 0 END)
        FROM audit_joined
        """
    ).fetchone()[0]

    preflight = {
        "years": YEARS,
        "audit_rows_2024": audit_counts[2024],
        "audit_rows_2025": audit_counts[2025],
        "joined_rows": joined_count,
        "batter_statcast_coverage": batter_cov,
        "pitcher_statcast_coverage": pitcher_cov,
        "2026_clean_holdout": "NO_BURNED",
        "model_changes": "NONE",
    }

    payload = {
        "script": "HITS_RAW_STRUCTURAL_AUDIT_A",
        "purpose": "raw_signal_only_no_model_changes",
        "preflight": preflight,
        "candidate_definitions": CANDIDATES,
        "ranking": ranking,
        "results": results,
    }

    OUT_JSON.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    write_text_report(preflight, results, ranking)

    print("\nPOOLED CONTROLLED SIGNAL RANKING", flush=True)
    print("--------------------------------", flush=True)

    for i, row in enumerate(ranking, 1):
        print(
            f"{i:2d}. {row['candidate']} "
            f"family={row['family']} "
            f"n={row['rows']:,} "
            f"span={row['oriented_d10_minus_d1_residual']:+.6f} "
            f"rho={row['oriented_residual_spearman']:+.4f} "
            f"stable_both_years={row['stable_both_years']}",
            flush=True,
        )

    print("\nYEAR STABILITY", flush=True)
    print("--------------", flush=True)

    for row in ranking:
        s24 = row["span_2024"]
        s25 = row["span_2025"]
        s24_txt = "NA" if s24 is None else f"{s24:+.6f}"
        s25_txt = "NA" if s25 is None else f"{s25:+.6f}"

        print(
            f"{row['candidate']}: "
            f"2024={s24_txt} "
            f"2025={s25_txt} "
            f"stable={row['stable_both_years']}",
            flush=True,
        )

    print("\nNEXT READ", flush=True)
    print("---------", flush=True)
    print(
        "No feature is promoted here. "
        "Review only candidates with adequate coverage, meaningful "
        "baseline-controlled residual spread, expected-direction behavior, "
        "and stability across both 2024 and 2025. "
        "Only then may one scalar be authorized for a formal challenger gate.",
        flush=True,
    )

    print("\nOUTPUTS", flush=True)
    print(OUT_JSON, flush=True)
    print(OUT_TXT, flush=True)

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
