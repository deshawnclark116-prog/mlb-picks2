#!/usr/bin/env python3
"""
HITS_PITCHER_CONTACT_2026_FORMAL_GATE_A

ONE-SHOT FORMAL FORWARD GATE
============================

Frozen family representative:
    opp_pitcher_contact_allowed_rate_60d

Why this candidate:
    It was selected BEFORE opening 2026 results because it had the strongest
    overall 2025 INTERNAL-SCREEN combination among the pitcher-contact survivors:
    Brier, logloss, AUC, top-5%, top-10%, official-tier hit rate, and monthly
    stability.

Holdout doctrine
----------------
2026 was already burned for the original opportunity-family work because the
production batter-hits artifact had trained through partial 2026.

But 2026 had NOT been used to evaluate this pitcher-contact candidate family.

Therefore this script reconstructs a CLEAN baseline through 2025 only, freezes
the single pitcher-contact challenger, then opens 2026 exactly once.

This script does NOT:
- use the current production batter_hits artifact for 2026 scoring
- test multiple pitcher candidates on 2026
- tune thresholds after seeing 2026
- stack opportunity features
- rescue a failed candidate

Development data
----------------
Candidate/calibration head training:
    2024 OOF baseline probabilities
    2025 OOF baseline probabilities
    + strictly lagged 60-day opposing-pitcher contact-allowed feature

Formal holdout
--------------
    2026 only

Baseline reconstruction
-----------------------
    Reuse clean rolling-forward booster through 2024.
    Add 2025 training only.
    Score 2026 with a model trained through 2025 and never on 2026.

Candidate architecture
----------------------
Locked baseline probability:
    p0

Calibration-only control:
    logit(p_control) = a + b * logit(p0)

Frozen pitcher challenger:
    logit(p_candidate)
        = a + b * logit(p0)
        + c * z(opp_pitcher_contact_allowed_rate_60d)

Candidate feature:
    prior 60-day opposing-pitcher contact allowed rate
    = 1 - whiffs / swings

Pregame leakage rule:
    same-day pitches are excluded.
    Historical opposing pitcher identity is recovered as the first pitcher faced
    in the completed-game feed; pitcher identity itself was pregame-knowable,
    but this recovery method is explicitly disclosed.

Missing candidate feature:
    fall back to clean baseline probability.

Fixed formal promotion gate
---------------------------
A candidate passes ONLY if ALL are true:

1. 2026 full-board Brier delta vs clean baseline <= -0.0005
2. 2026 full-board logloss does not worsen vs clean baseline
3. 2026 full-board AUC delta vs clean baseline >= -0.0005
4. 2026 full-board top-5% actual hit rate does not decline
5. 2026 full-board top-10% actual hit rate does not decline
6. 2026 official-tier actual hit rate does not decline
7. 2026 official-tier coverage remains >= 90% of clean baseline
8. Candidate beats calibration-only control by Brier <= -0.0001
9. Candidate logloss does not worsen vs calibration-only control
10. Candidate top-5% does not decline vs calibration-only control
11. Candidate top-10% does not decline vs calibration-only control
12. Learned candidate coefficient has the expected positive sign
13. Candidate improves/holds monthly Brier vs baseline in at least half of
    eligible 2026 months with >= 500 rows

This gate is locked before 2026 results are opened.

Possible final verdicts
-----------------------
PASSES:
    OPP_PITCHER_CONTACT_ALLOWED_60D_PASSES_2026_FORMAL_GATE

FAILS:
    OPP_PITCHER_CONTACT_ALLOWED_60D_FAILS_2026_FORMAL_GATE_FREEZE_NO_RESCUE_TUNING

Memory safety
-------------
Designed for 512 MB Render:
- no pandas
- no sklearn
- no full Statcast load in Python
- SQLite temp work on disk
- one XGBoost child process at a time
- existing 2019-2025 baseline feature files reused
- existing clean booster through 2024 reused

Expected prerequisites
----------------------
/data/hr_model/hits_rfv_lite_b/booster_through_2024.json
/data/hr_model/hits_rfv_lite_b/features_2025.libsvm
/data/hr_model/hits_rfv_lite_b/scores_2024.json
/data/hr_model/hits_rfv_lite_b/scores_2025.json
/data/hr_model/hits_raw_structural_audit_a_work/audit.sqlite
/data/hr_model/hr_model.sqlite
/data/season_2026.jsonl  (or .json)

Run
---
python -u hits_pitcher_contact_2026_formal_gate_a.py 2>&1 | tee /data/hr_model/hits_pitcher_contact_2026_formal_gate_a.log

Outputs
-------
/data/hr_model/hits_pitcher_contact_2026_formal_gate_a_results.json
/data/hr_model/hits_pitcher_contact_2026_formal_gate_a_report.txt

Paste back
----------
FORMAL GATE PREFLIGHT
CLEAN 2026 BASELINE RECONSTRUCTION
FROZEN CHALLENGER TRAINING
2026 FORMAL HOLDOUT COMPARISON
MONTHLY STABILITY
STRICT FORMAL GATE
FINAL VERDICT
"""

import argparse
import csv
import json
import math
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths / locked constants
# ---------------------------------------------------------------------------

BASE_DIR = Path("/data/hr_model/hits_rfv_lite_b")
AUDIT_DB = Path("/data/hr_model/hits_raw_structural_audit_a_work/audit.sqlite")
STATCAST_DB = Path("/data/hr_model/hr_model.sqlite")

WORK_DIR = Path("/data/hr_model/hits_pitcher_contact_2026_formal_gate_a_work")
OUT_JSON = Path("/data/hr_model/hits_pitcher_contact_2026_formal_gate_a_results.json")
OUT_TXT = Path("/data/hr_model/hits_pitcher_contact_2026_formal_gate_a_report.txt")

CANDIDATE = "opp_pitcher_contact_allowed_rate_60d"
CANDIDATE_SAMPLE_COL = "p_swings_60d"
CANDIDATE_MIN_SAMPLE = 150
EXPECTED_SIGN = +1

OFFICIAL_MIN = 0.630

BASE_FEATURES = [
    "season_avg",
    "recent15_avg",
    "recent5_avg",
    "hr_rate",
    "bb_rate",
    "so_rate",
    "batting_order",
    "games_played",
]

TRAIN_PARAMS = {
    "objective": "count:poisson",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    "nthread": 2,
}

RIDGE = 1e-3
MAX_ITER = 35
TOL = 1e-8


ALIASES = {
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


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clip(value, lo, hi):
    return max(lo, min(hi, value))


def sigmoid(x):
    x = clip(x, -35.0, 35.0)

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)


def logit(p):
    p = clip(float(p), 1e-8, 1.0 - 1e-8)
    return math.log(p / (1.0 - p))


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
    for path in (
        Path(f"/data/season_{year}.jsonl"),
        Path(f"/data/season_{year}.json"),
    ):
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
                f"{path} is one giant JSON array. "
                "This script refuses to json.load() it under 512 MB."
            )

        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"JSON parse error in {path} line {line_no}: {exc}"
                ) from exc


def normalize_batter_row(row, seq):
    ptype = str(
        first_value(row, ALIASES["player_type"], "") or ""
    ).lower()

    if ptype and "batter" not in ptype and "hitter" not in ptype:
        return None

    player_id = first_value(row, ALIASES["player_id"])
    game_date = first_value(row, ALIASES["date"])
    hits = first_value(row, ALIASES["hits"])
    ab = first_value(row, ALIASES["ab"])

    if player_id is None or game_date is None or hits is None or ab is None:
        return None

    bb = to_float(first_value(row, ALIASES["bb"], 0))
    pa_raw = first_value(row, ALIASES["pa"], None)
    pa = to_float(pa_raw, to_float(ab) + bb)

    return (
        str(player_id),
        str(game_date),
        str(first_value(row, ALIASES["game_id"], seq)),
        int(seq),
        to_float(hits),
        to_float(ab),
        pa,
        to_float(first_value(row, ALIASES["hr"], 0)),
        bb,
        to_float(first_value(row, ALIASES["so"], 0)),
        to_float(first_value(row, ALIASES["batting_order"], 0)),
    )


# ---------------------------------------------------------------------------
# Build exact 2026 baseline feature file
# ---------------------------------------------------------------------------

def build_2026_baseline_features(season_path):
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    libsvm_path = WORK_DIR / "features_2026_clean.libsvm"
    meta_path = WORK_DIR / "features_2026_clean_meta.csv"
    summary_path = WORK_DIR / "features_2026_clean_summary.json"

    if libsvm_path.exists() and meta_path.exists() and summary_path.exists():
        summary = load_json(summary_path)

        print(
            f"2026 clean feature file REUSED | "
            f"eligible_rows={summary['eligible_rows']:,}",
            flush=True,
        )

        return libsvm_path, meta_path, summary

    spool_db = WORK_DIR / "season_2026_spool.sqlite"

    if spool_db.exists():
        spool_db.unlink()

    conn = sqlite3.connect(str(spool_db))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")

    conn.execute(
        """
        CREATE TABLE games (
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

    raw_seen = 0
    normalized = 0
    batch = []

    for seq, row in enumerate(iter_json_records(season_path), 1):
        raw_seen += 1
        norm = normalize_batter_row(row, seq)

        if norm is None:
            continue

        batch.append(norm)
        normalized += 1

        if len(batch) >= 2000:
            conn.executemany(
                "INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                batch,
            )
            conn.commit()
            batch.clear()

    if batch:
        conn.executemany(
            "INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            batch,
        )
        conn.commit()

    conn.execute(
        """
        CREATE INDEX idx_games_2026
        ON games(player_id, game_date, game_id, seq)
        """
    )
    conn.commit()

    cursor = conn.execute(
        """
        SELECT
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
            batting_order
        FROM games
        ORDER BY player_id, game_date, game_id, seq
        """
    )

    current_player = None
    cum_h = 0.0
    cum_ab = 0.0
    cum_pa = 0.0
    cum_hr = 0.0
    cum_bb = 0.0
    cum_so = 0.0
    games_played = 0
    recent_h = deque(maxlen=15)

    eligible = 0

    with open(libsvm_path, "w", encoding="utf-8") as libsvm_f, \
         open(meta_path, "w", encoding="utf-8", newline="") as meta_f:

        writer = csv.writer(meta_f)
        writer.writerow(
            [
                "row_id",
                "player_id",
                "game_date",
                "game_id",
                "actual_hits",
                "actual_hit",
                "batting_order",
            ]
        )

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
                cum_h = 0.0
                cum_ab = 0.0
                cum_pa = 0.0
                cum_hr = 0.0
                cum_bb = 0.0
                cum_so = 0.0
                games_played = 0
                recent_h = deque(maxlen=15)

            if cum_ab >= 20.0 and games_played >= 5:
                recent15 = (
                    sum(recent_h) / len(recent_h)
                    if recent_h
                    else 0.0
                )

                last5 = list(recent_h)[-5:]

                recent5 = (
                    sum(last5) / len(last5)
                    if last5
                    else 0.0
                )

                features = [
                    cum_h / cum_ab if cum_ab > 0 else 0.0,
                    recent15,
                    recent5,
                    cum_hr / cum_pa if cum_pa > 0 else 0.0,
                    cum_bb / cum_pa if cum_pa > 0 else 0.0,
                    cum_so / cum_pa if cum_pa > 0 else 0.0,
                    batting_order,
                    float(games_played),
                ]

                label = float(hits)

                line = [f"{label:.10g}"]

                for feature_idx, value in enumerate(features, 1):
                    line.append(
                        f"{feature_idx}:{float(value):.10g}"
                    )

                libsvm_f.write(" ".join(line) + "\n")

                writer.writerow(
                    [
                        eligible,
                        player_id,
                        game_date,
                        game_id,
                        float(hits),
                        1 if float(hits) >= 1.0 else 0,
                        float(batting_order),
                    ]
                )

                eligible += 1

            cum_h += float(hits)
            cum_ab += float(ab)
            cum_pa += float(pa)
            cum_hr += float(hr)
            cum_bb += float(bb)
            cum_so += float(so)
            games_played += 1
            recent_h.append(float(hits))

    conn.close()

    try:
        spool_db.unlink()
    except Exception:
        pass

    summary = {
        "year": 2026,
        "season_path": str(season_path),
        "raw_seen": raw_seen,
        "normalized_rows": normalized,
        "eligible_rows": eligible,
        "feature_names": BASE_FEATURES,
        "libsvm_path": str(libsvm_path),
        "meta_path": str(meta_path),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(
        f"2026 clean feature build: "
        f"raw={raw_seen:,} "
        f"normalized={normalized:,} "
        f"eligible={eligible:,}",
        flush=True,
    )

    return libsvm_path, meta_path, summary


# ---------------------------------------------------------------------------
# Child-process XGBoost train/score
# ---------------------------------------------------------------------------

def run_child(args, label):
    env = dict(os.environ)

    env["OMP_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    print(f"\n{label}", flush=True)
    print("-" * len(label), flush=True)

    cp = subprocess.run(
        [sys.executable, "-u", str(Path(__file__).resolve())] + args,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if cp.stdout:
        print(
            cp.stdout,
            end="" if cp.stdout.endswith("\n") else "\n",
            flush=True,
        )

    if cp.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {cp.returncode}"
        )


def worker_train(args):
    import xgboost as xgb

    dtrain = xgb.DMatrix(
        f"{args.train_file}?format=libsvm"
    )

    previous = None
    prior_rounds = 0

    if args.prev_model and Path(args.prev_model).exists():
        previous = xgb.Booster()
        previous.load_model(args.prev_model)
        prior_rounds = int(previous.num_boosted_rounds())

    booster = xgb.train(
        TRAIN_PARAMS,
        dtrain,
        num_boost_round=60,
        xgb_model=previous,
    )

    booster.save_model(args.out_model)

    print(f"prior_rounds={prior_rounds}", flush=True)
    print("added_rounds=60", flush=True)
    print(
        f"total_rounds={booster.num_boosted_rounds()}",
        flush=True,
    )
    print(f"saved_model={args.out_model}", flush=True)


def worker_score(args):
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(args.model_path)

    dtest = xgb.DMatrix(
        f"{args.score_file}?format=libsvm"
    )

    expected_hits = booster.predict(dtest)

    dates = []
    actual_hits = []
    actual_hit = []
    player_ids = []
    game_ids = []

    with open(
        args.meta_file,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        for row in csv.DictReader(f):
            dates.append(row["game_date"])
            actual_hits.append(float(row["actual_hits"]))
            actual_hit.append(int(row["actual_hit"]))
            player_ids.append(str(row["player_id"]))
            game_ids.append(str(row["game_id"]))

    if len(expected_hits) != len(actual_hits):
        raise RuntimeError(
            f"Prediction/meta mismatch: "
            f"{len(expected_hits)} vs {len(actual_hits)}"
        )

    probs = [
        min(
            max(
                1.0 - math.exp(-max(float(lam), 1e-8)),
                1e-8,
            ),
            1.0 - 1e-8,
        )
        for lam in expected_hits
    ]

    payload = {
        "test_year": 2026,
        "model_rounds": int(booster.num_boosted_rounds()),
        "dates": dates,
        "player_ids": player_ids,
        "game_ids": game_ids,
        "actual_hits": actual_hits,
        "y_binary": actual_hit,
        "expected_hits": [float(x) for x in expected_hits],
        "probs": probs,
    }

    Path(args.out_score_json).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    print(f"scored_rows={len(actual_hit):,}", flush=True)
    print(
        f"model_rounds={payload['model_rounds']}",
        flush=True,
    )
    print(
        f"saved_scores={args.out_score_json}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Build 2026 pitcher-contact candidate feature
# ---------------------------------------------------------------------------

def build_2026_candidate_features(score_2026):
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    work_db = WORK_DIR / "candidate_2026.sqlite"
    summary_path = WORK_DIR / "candidate_2026_summary.json"

    if work_db.exists() and summary_path.exists():
        summary = load_json(summary_path)

        count = sqlite3.connect(str(work_db)).execute(
            "SELECT COUNT(*) FROM candidate_rows"
        ).fetchone()[0]

        if count == len(score_2026["y_binary"]):
            print(
                f"2026 candidate feature table REUSED | rows={count:,}",
                flush=True,
            )
            return work_db, summary

    if work_db.exists():
        work_db.unlink()

    conn = sqlite3.connect(str(work_db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-16000")

    conn.execute(
        f"ATTACH DATABASE 'file:{STATCAST_DB}?mode=ro' AS src"
    )

    source_info = conn.execute(
        """
        SELECT
            MIN(game_date),
            MAX(game_date),
            COUNT(*)
        FROM src.statcast_pitches
        WHERE game_date >= '2026-01-01'
          AND game_date < '2027-01-01'
        """
    ).fetchone()

    print(
        f"2026 Statcast source: "
        f"rows={source_info[2]:,} "
        f"range={source_info[0]}..{source_info[1]}",
        flush=True,
    )

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
            SUM(COALESCE(is_whiff, 0)) AS whiffs
        FROM src.statcast_pitches
        WHERE game_date >= '2026-01-01'
          AND game_date < '2027-01-01'
        GROUP BY CAST(pitcher_id AS TEXT), game_date
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_pitcher_daily
        ON pitcher_daily(pitcher_id, game_date)
        """
    )

    conn.execute(
        """
        CREATE TABLE pitcher_roll60 AS
        SELECT
            pitcher_id,
            game_date,
            SUM(swings) OVER w AS swings_60d,
            SUM(whiffs) OVER w AS whiffs_60d
        FROM pitcher_daily
        WINDOW w AS (
            PARTITION BY pitcher_id
            ORDER BY julianday(game_date)
            RANGE BETWEEN 60 PRECEDING AND 1 PRECEDING
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_pitcher_roll60
        ON pitcher_roll60(pitcher_id, game_date)
        """
    )

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
            WHERE game_date >= '2026-01-01'
              AND game_date < '2027-01-01'
        )
        WHERE rn = 1
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_batter_game_pitcher
        ON batter_game_pitcher(game_id, player_id)
        """
    )

    conn.execute(
        """
        CREATE TABLE baseline_rows (
            row_id INTEGER PRIMARY KEY,
            player_id TEXT,
            game_id TEXT,
            game_date TEXT,
            actual_hit INTEGER,
            baseline_prob REAL
        )
        """
    )

    rows = []

    for i in range(len(score_2026["y_binary"])):
        rows.append(
            (
                i,
                str(score_2026["player_ids"][i]),
                str(score_2026["game_ids"][i]),
                str(score_2026["dates"][i]),
                int(score_2026["y_binary"][i]),
                float(score_2026["probs"][i]),
            )
        )

        if len(rows) >= 2000:
            conn.executemany(
                "INSERT INTO baseline_rows VALUES (?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
            rows.clear()

    if rows:
        conn.executemany(
            "INSERT INTO baseline_rows VALUES (?,?,?,?,?,?)",
            rows,
        )
        conn.commit()

    conn.execute(
        """
        CREATE TABLE candidate_rows AS
        SELECT
            b.row_id,
            b.player_id,
            b.game_id,
            b.game_date,
            b.actual_hit,
            b.baseline_prob,
            bgp.pitcher_id,
            pr.swings_60d AS p_swings_60d,
            CASE
                WHEN pr.swings_60d > 0
                THEN 1.0 - (1.0 * pr.whiffs_60d / pr.swings_60d)
            END AS opp_pitcher_contact_allowed_rate_60d
        FROM baseline_rows b
        LEFT JOIN batter_game_pitcher bgp
          ON bgp.game_id = b.game_id
         AND bgp.player_id = b.player_id
        LEFT JOIN pitcher_roll60 pr
          ON pr.pitcher_id = bgp.pitcher_id
         AND pr.game_date = b.game_date
        ORDER BY b.row_id
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_candidate_rows_row_id
        ON candidate_rows(row_id)
        """
    )

    conn.execute("DETACH DATABASE src")
    conn.commit()

    total_rows = conn.execute(
        "SELECT COUNT(*) FROM candidate_rows"
    ).fetchone()[0]

    valid_rows = conn.execute(
        """
        SELECT COUNT(*)
        FROM candidate_rows
        WHERE opp_pitcher_contact_allowed_rate_60d IS NOT NULL
          AND p_swings_60d >= ?
        """,
        (CANDIDATE_MIN_SAMPLE,),
    ).fetchone()[0]

    summary = {
        "candidate": CANDIDATE,
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "coverage": (
            valid_rows / total_rows
            if total_rows
            else 0.0
        ),
        "min_sample": CANDIDATE_MIN_SAMPLE,
        "statcast_rows_2026": source_info[2],
        "statcast_min_date": source_info[0],
        "statcast_max_date": source_info[1],
        "opponent_pitcher_recovery": (
            "first pitcher faced in completed-game feed; "
            "pitcher identity is pregame-knowable"
        ),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    conn.close()

    print(
        f"2026 candidate feature coverage: "
        f"{valid_rows:,}/{total_rows:,} "
        f"({summary['coverage']:.4f})",
        flush=True,
    )

    return work_db, summary


# ---------------------------------------------------------------------------
# Logistic correction head
# ---------------------------------------------------------------------------

def solve_linear(matrix, vector):
    n = len(vector)

    a = [
        list(row) + [float(vector[i])]
        for i, row in enumerate(matrix)
    ]

    for col in range(n):
        pivot = max(
            range(col, n),
            key=lambda r: abs(a[r][col]),
        )

        if abs(a[pivot][col]) < 1e-12:
            raise RuntimeError("Singular logistic Hessian.")

        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]

        pivot_value = a[col][col]

        for j in range(col, n + 1):
            a[col][j] /= pivot_value

        for row in range(n):
            if row == col:
                continue

            factor = a[row][col]

            if factor == 0:
                continue

            for j in range(col, n + 1):
                a[row][j] -= factor * a[col][j]

    return [a[i][n] for i in range(n)]


def fit_logistic(features, y, ridge=RIDGE, max_iter=MAX_ITER):
    if not features:
        raise RuntimeError("No training rows.")

    d = len(features[0])
    beta = [0.0] * d

    for iteration in range(max_iter):
        grad = [0.0] * d
        hess = [[0.0] * d for _ in range(d)]

        for x, yi in zip(features, y):
            eta = sum(
                beta[j] * x[j]
                for j in range(d)
            )

            p = sigmoid(eta)
            residual = yi - p
            w = max(p * (1.0 - p), 1e-8)

            for j in range(d):
                grad[j] += x[j] * residual

                for k in range(j, d):
                    hess[j][k] += w * x[j] * x[k]

        for j in range(d):
            for k in range(j):
                hess[j][k] = hess[k][j]

        for j in range(1, d):
            grad[j] -= ridge * beta[j]
            hess[j][j] += ridge

        step = solve_linear(hess, grad)

        for j in range(d):
            beta[j] += step[j]

        if max(abs(s) for s in step) < TOL:
            return {
                "coefficients": beta,
                "iterations": iteration + 1,
                "converged": True,
            }

    return {
        "coefficients": beta,
        "iterations": max_iter,
        "converged": False,
    }


def predict_logistic(features, coefficients):
    return [
        sigmoid(
            sum(
                c * xj
                for c, xj in zip(coefficients, x)
            )
        )
        for x in features
    ]


def load_development_rows():
    if not AUDIT_DB.exists():
        raise RuntimeError(f"Missing audit DB: {AUDIT_DB}")

    score_2024 = load_json(BASE_DIR / "scores_2024.json")
    score_2025 = load_json(BASE_DIR / "scores_2025.json")

    conn = sqlite3.connect(str(AUDIT_DB))

    dev = []

    for year, scores in (
        (2024, score_2024),
        (2025, score_2025),
    ):
        raw = list(
            conn.execute(
                f"""
                SELECT
                    row_id,
                    {CANDIDATE},
                    {CANDIDATE_SAMPLE_COL}
                FROM audit_joined
                WHERE year = ?
                ORDER BY row_id
                """,
                (year,),
            )
        )

        if len(raw) != len(scores["y_binary"]):
            raise RuntimeError(
                f"{year}: audit/score alignment mismatch "
                f"{len(raw)} vs {len(scores['y_binary'])}"
            )

        for expected_idx, (
            row_id,
            candidate_value,
            sample_n,
        ) in enumerate(raw):

            if int(row_id) != expected_idx:
                raise RuntimeError(
                    f"{year}: row_id alignment failure at "
                    f"{expected_idx}, got {row_id}"
                )

            if (
                candidate_value is None
                or sample_n is None
                or float(sample_n) < CANDIDATE_MIN_SAMPLE
            ):
                continue

            dev.append(
                {
                    "year": year,
                    "actual_hit": int(
                        scores["y_binary"][expected_idx]
                    ),
                    "baseline_prob": float(
                        scores["probs"][expected_idx]
                    ),
                    "candidate_value": float(candidate_value),
                }
            )

    conn.close()

    return dev


def train_frozen_heads():
    dev = load_development_rows()

    if len(dev) < 1000:
        raise RuntimeError(
            f"Too few development rows: {len(dev)}"
        )

    values = [
        row["candidate_value"]
        for row in dev
    ]

    mean = sum(values) / len(values)

    variance = sum(
        (value - mean) ** 2
        for value in values
    ) / len(values)

    std = math.sqrt(variance)

    if std <= 1e-12:
        raise RuntimeError("Candidate standard deviation is zero.")

    y = [
        row["actual_hit"]
        for row in dev
    ]

    control_x = []
    candidate_x = []

    for row in dev:
        base_logit = clip(
            logit(row["baseline_prob"]),
            -12.0,
            12.0,
        )

        z = clip(
            (row["candidate_value"] - mean) / std,
            -5.0,
            5.0,
        )

        control_x.append(
            [1.0, base_logit]
        )

        candidate_x.append(
            [1.0, base_logit, z]
        )

    control_fit = fit_logistic(
        control_x,
        y,
    )

    candidate_fit = fit_logistic(
        candidate_x,
        y,
    )

    return {
        "development_rows": len(dev),
        "development_years": [2024, 2025],
        "candidate_mean": mean,
        "candidate_std": std,
        "control_fit": control_fit,
        "candidate_fit": candidate_fit,
        "candidate_coefficient": (
            candidate_fit["coefficients"][-1]
        ),
    }


# ---------------------------------------------------------------------------
# 2026 evaluation
# ---------------------------------------------------------------------------

def auc_rank(y, p):
    n_pos = sum(y)
    n_neg = len(y) - n_pos

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    pairs = sorted(
        enumerate(p),
        key=lambda item: item[1],
    )

    ranks = [0.0] * len(y)

    i = 0

    while i < len(pairs):
        j = i + 1

        while (
            j < len(pairs)
            and pairs[j][1] == pairs[i][1]
        ):
            j += 1

        avg_rank = ((i + 1) + j) / 2.0

        for k in range(i, j):
            ranks[pairs[k][0]] = avg_rank

        i = j

    sum_pos_ranks = sum(
        ranks[i]
        for i, label in enumerate(y)
        if label == 1
    )

    return (
        sum_pos_ranks
        - n_pos * (n_pos + 1) / 2.0
    ) / (n_pos * n_neg)


def metric_block(y, p):
    n = len(y)

    brier = sum(
        (pi - yi) ** 2
        for yi, pi in zip(y, p)
    ) / n

    logloss = sum(
        -(
            yi * math.log(
                clip(pi, 1e-8, 1.0 - 1e-8)
            )
            + (1 - yi)
            * math.log(
                1.0 - clip(pi, 1e-8, 1.0 - 1e-8)
            )
        )
        for yi, pi in zip(y, p)
    ) / n

    order = sorted(
        range(n),
        key=lambda i: p[i],
        reverse=True,
    )

    out = {
        "rows": n,
        "actual_hit_rate": sum(y) / n,
        "mean_probability": sum(p) / n,
        "brier": brier,
        "logloss": logloss,
        "auc": auc_rank(y, p),
    }

    for fraction, name in (
        (0.05, "top5"),
        (0.10, "top10"),
    ):
        k = max(
            1,
            math.ceil(n * fraction),
        )

        idx = order[:k]

        out[f"{name}_rows"] = k
        out[f"{name}_hits"] = sum(
            y[i]
            for i in idx
        )
        out[f"{name}_actual"] = (
            sum(y[i] for i in idx) / k
        )
        out[f"{name}_mean_prob"] = (
            sum(p[i] for i in idx) / k
        )

    official_idx = [
        i
        for i, pi in enumerate(p)
        if pi >= OFFICIAL_MIN
    ]

    out["official_rows"] = len(official_idx)
    out["official_hits"] = sum(
        y[i]
        for i in official_idx
    )

    out["official_hit_rate"] = (
        sum(y[i] for i in official_idx)
        / len(official_idx)
        if official_idx
        else None
    )

    out["official_mean_prob"] = (
        sum(p[i] for i in official_idx)
        / len(official_idx)
        if official_idx
        else None
    )

    return out


def monthly_brier(y, p, dates):
    groups = defaultdict(list)

    for i, date_text in enumerate(dates):
        month = str(date_text)[:7]
        groups[month].append(i)

    out = {}

    for month, idx in sorted(groups.items()):
        out[month] = {
            "rows": len(idx),
            "brier": sum(
                (p[i] - y[i]) ** 2
                for i in idx
            ) / len(idx),
        }

    return out


def evaluate_2026(
    score_2026,
    candidate_db,
    frozen_heads,
):
    conn = sqlite3.connect(str(candidate_db))

    raw = list(
        conn.execute(
            f"""
            SELECT
                row_id,
                p_swings_60d,
                {CANDIDATE}
            FROM candidate_rows
            ORDER BY row_id
            """
        )
    )

    conn.close()

    y = list(score_2026["y_binary"])
    dates = list(score_2026["dates"])
    baseline_probs = list(score_2026["probs"])

    if len(raw) != len(y):
        raise RuntimeError(
            f"2026 candidate/baseline mismatch: "
            f"{len(raw)} vs {len(y)}"
        )

    valid_indices = []
    control_x = []
    candidate_x = []

    mean = frozen_heads["candidate_mean"]
    std = frozen_heads["candidate_std"]

    for expected_idx, (
        row_id,
        sample_n,
        candidate_value,
    ) in enumerate(raw):

        if int(row_id) != expected_idx:
            raise RuntimeError(
                f"2026 row_id alignment failure at "
                f"{expected_idx}, got {row_id}"
            )

        if (
            candidate_value is None
            or sample_n is None
            or float(sample_n) < CANDIDATE_MIN_SAMPLE
        ):
            continue

        base_logit = clip(
            logit(baseline_probs[expected_idx]),
            -12.0,
            12.0,
        )

        z = clip(
            (float(candidate_value) - mean) / std,
            -5.0,
            5.0,
        )

        valid_indices.append(expected_idx)

        control_x.append(
            [1.0, base_logit]
        )

        candidate_x.append(
            [1.0, base_logit, z]
        )

    control_subset = predict_logistic(
        control_x,
        frozen_heads["control_fit"]["coefficients"],
    )

    candidate_subset = predict_logistic(
        candidate_x,
        frozen_heads["candidate_fit"]["coefficients"],
    )

    control_full = list(baseline_probs)
    candidate_full = list(baseline_probs)

    for idx, prob in zip(
        valid_indices,
        control_subset,
    ):
        control_full[idx] = float(prob)

    for idx, prob in zip(
        valid_indices,
        candidate_subset,
    ):
        candidate_full[idx] = float(prob)

    baseline_metrics = metric_block(
        y,
        baseline_probs,
    )

    control_metrics = metric_block(
        y,
        control_full,
    )

    candidate_metrics = metric_block(
        y,
        candidate_full,
    )

    delta_vs_baseline = {
        "brier": (
            candidate_metrics["brier"]
            - baseline_metrics["brier"]
        ),
        "logloss": (
            candidate_metrics["logloss"]
            - baseline_metrics["logloss"]
        ),
        "auc": (
            candidate_metrics["auc"]
            - baseline_metrics["auc"]
        ),
        "top5": (
            candidate_metrics["top5_actual"]
            - baseline_metrics["top5_actual"]
        ),
        "top10": (
            candidate_metrics["top10_actual"]
            - baseline_metrics["top10_actual"]
        ),
        "official_hit_rate": (
            candidate_metrics["official_hit_rate"]
            - baseline_metrics["official_hit_rate"]
        ),
    }

    delta_vs_control = {
        "brier": (
            candidate_metrics["brier"]
            - control_metrics["brier"]
        ),
        "logloss": (
            candidate_metrics["logloss"]
            - control_metrics["logloss"]
        ),
        "auc": (
            candidate_metrics["auc"]
            - control_metrics["auc"]
        ),
        "top5": (
            candidate_metrics["top5_actual"]
            - control_metrics["top5_actual"]
        ),
        "top10": (
            candidate_metrics["top10_actual"]
            - control_metrics["top10_actual"]
        ),
        "official_hit_rate": (
            candidate_metrics["official_hit_rate"]
            - control_metrics["official_hit_rate"]
        ),
    }

    official_coverage_ratio = (
        candidate_metrics["official_rows"]
        / baseline_metrics["official_rows"]
        if baseline_metrics["official_rows"] > 0
        else None
    )

    baseline_monthly = monthly_brier(
        y,
        baseline_probs,
        dates,
    )

    candidate_monthly = monthly_brier(
        y,
        candidate_full,
        dates,
    )

    monthly = {}
    eligible_months = 0
    monthly_nonworse = 0

    for month in sorted(baseline_monthly):
        base = baseline_monthly[month]
        cand = candidate_monthly[month]
        delta = cand["brier"] - base["brier"]

        eligible = base["rows"] >= 500

        if eligible:
            eligible_months += 1

            if delta <= 0.0:
                monthly_nonworse += 1

        monthly[month] = {
            "rows": base["rows"],
            "eligible_for_stability_gate": eligible,
            "baseline_brier": base["brier"],
            "candidate_brier": cand["brier"],
            "delta": delta,
        }

    required_nonworse = (
        math.ceil(eligible_months / 2)
        if eligible_months > 0
        else 0
    )

    coefficient_sign_pass = (
        frozen_heads["candidate_coefficient"]
        * EXPECTED_SIGN
        > 0
    )

    gate = {
        "brier_vs_clean_baseline_pass": (
            delta_vs_baseline["brier"] <= -0.0005
        ),
        "logloss_vs_clean_baseline_pass": (
            delta_vs_baseline["logloss"] <= 0.0
        ),
        "auc_vs_clean_baseline_pass": (
            delta_vs_baseline["auc"] >= -0.0005
        ),
        "top5_vs_clean_baseline_pass": (
            delta_vs_baseline["top5"] >= 0.0
        ),
        "top10_vs_clean_baseline_pass": (
            delta_vs_baseline["top10"] >= 0.0
        ),
        "official_hit_rate_vs_clean_baseline_pass": (
            delta_vs_baseline["official_hit_rate"] >= 0.0
        ),
        "official_coverage_pass": (
            official_coverage_ratio is not None
            and official_coverage_ratio >= 0.90
        ),
        "brier_vs_calibration_control_pass": (
            delta_vs_control["brier"] <= -0.0001
        ),
        "logloss_vs_calibration_control_pass": (
            delta_vs_control["logloss"] <= 0.0
        ),
        "top5_vs_calibration_control_pass": (
            delta_vs_control["top5"] >= 0.0
        ),
        "top10_vs_calibration_control_pass": (
            delta_vs_control["top10"] >= 0.0
        ),
        "expected_coefficient_sign_pass": (
            coefficient_sign_pass
        ),
        "monthly_brier_stability_pass": (
            eligible_months > 0
            and monthly_nonworse >= required_nonworse
        ),
    }

    gate["overall_formal_gate_pass"] = all(
        gate.values()
    )

    verdict = (
        "OPP_PITCHER_CONTACT_ALLOWED_60D_PASSES_2026_FORMAL_GATE"
        if gate["overall_formal_gate_pass"]
        else
        "OPP_PITCHER_CONTACT_ALLOWED_60D_FAILS_2026_FORMAL_GATE_FREEZE_NO_RESCUE_TUNING"
    )

    return {
        "candidate": CANDIDATE,
        "valid_rows": len(valid_indices),
        "total_rows": len(y),
        "coverage": (
            len(valid_indices) / len(y)
            if y
            else 0.0
        ),
        "baseline_metrics": baseline_metrics,
        "calibration_control_metrics": control_metrics,
        "candidate_metrics": candidate_metrics,
        "delta_vs_clean_baseline": delta_vs_baseline,
        "delta_vs_calibration_control": delta_vs_control,
        "official_coverage_ratio": official_coverage_ratio,
        "monthly_2026": monthly,
        "eligible_months": eligible_months,
        "monthly_brier_nonworse": monthly_nonworse,
        "monthly_brier_required_nonworse": required_nonworse,
        "gate": gate,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main_parent():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "HITS_PITCHER_CONTACT_2026_FORMAL_GATE_A",
        flush=True,
    )
    print(
        "======================================",
        flush=True,
    )

    print("\nFORMAL GATE PREFLIGHT", flush=True)
    print("---------------------", flush=True)

    season_2026 = discover_season_file(2026)

    if season_2026 is None:
        raise RuntimeError(
            "Missing /data/season_2026.jsonl or .json"
        )

    required_paths = [
        BASE_DIR / "booster_through_2024.json",
        BASE_DIR / "features_2025.libsvm",
        BASE_DIR / "scores_2024.json",
        BASE_DIR / "scores_2025.json",
        AUDIT_DB,
        STATCAST_DB,
        season_2026,
    ]

    for path in required_paths:
        if not path.exists():
            raise RuntimeError(
                f"Missing required file: {path}"
            )

        print(
            f"exists=True | {path}",
            flush=True,
        )

    print(
        f"frozen_candidate={CANDIDATE}",
        flush=True,
    )
    print(
        "candidate_count_opened_on_2026=1",
        flush=True,
    )
    print(
        "candidate_selected_before_2026_results=True",
        flush=True,
    )
    print(
        "2026_prior_pitcher_family_evaluation=False",
        flush=True,
    )
    print(
        "baseline_current_production_artifact_used=False",
        flush=True,
    )
    print(
        "baseline_reconstruction=train_through_2025_only",
        flush=True,
    )
    print(
        "stacking=False",
        flush=True,
    )
    print(
        "rescue_tuning=FORBIDDEN",
        flush=True,
    )

    print(
        "\nCLEAN 2026 BASELINE RECONSTRUCTION",
        flush=True,
    )
    print(
        "----------------------------------",
        flush=True,
    )

    feature_2026, meta_2026, feature_summary = (
        build_2026_baseline_features(
            season_2026
        )
    )

    clean_model_2025 = (
        WORK_DIR
        / "clean_baseline_booster_through_2025.json"
    )

    run_child(
        [
            "--worker-train",
            "--train-file",
            str(BASE_DIR / "features_2025.libsvm"),
            "--prev-model",
            str(BASE_DIR / "booster_through_2024.json"),
            "--out-model",
            str(clean_model_2025),
        ],
        "TRAIN CLEAN BASELINE THROUGH 2025",
    )

    clean_scores_2026_path = (
        WORK_DIR
        / "clean_baseline_scores_2026.json"
    )

    run_child(
        [
            "--worker-score",
            "--model-path",
            str(clean_model_2025),
            "--score-file",
            str(feature_2026),
            "--meta-file",
            str(meta_2026),
            "--out-score-json",
            str(clean_scores_2026_path),
        ],
        "SCORE CLEAN FORWARD HOLDOUT 2026",
    )

    clean_scores_2026 = load_json(
        clean_scores_2026_path
    )

    print(
        f"clean_2026_rows="
        f"{len(clean_scores_2026['y_binary']):,}",
        flush=True,
    )
    print(
        f"clean_model_rounds="
        f"{clean_scores_2026['model_rounds']}",
        flush=True,
    )

    print(
        "\nFROZEN CHALLENGER TRAINING",
        flush=True,
    )
    print(
        "--------------------------",
        flush=True,
    )

    frozen_heads = train_frozen_heads()

    print(
        f"development_rows="
        f"{frozen_heads['development_rows']:,}",
        flush=True,
    )
    print(
        f"development_years="
        f"{frozen_heads['development_years']}",
        flush=True,
    )
    print(
        f"candidate_mean="
        f"{frozen_heads['candidate_mean']:.8f}",
        flush=True,
    )
    print(
        f"candidate_std="
        f"{frozen_heads['candidate_std']:.8f}",
        flush=True,
    )
    print(
        f"control_coefficients="
        f"{frozen_heads['control_fit']['coefficients']}",
        flush=True,
    )
    print(
        f"candidate_coefficients="
        f"{frozen_heads['candidate_fit']['coefficients']}",
        flush=True,
    )
    print(
        f"candidate_c="
        f"{frozen_heads['candidate_coefficient']:+.8f}",
        flush=True,
    )
    print(
        f"candidate_fit_converged="
        f"{frozen_heads['candidate_fit']['converged']}",
        flush=True,
    )

    print(
        "\nBUILDING 2026 PITCHER-CONTACT FEATURE",
        flush=True,
    )
    print(
        "-------------------------------------",
        flush=True,
    )

    candidate_db, candidate_summary = (
        build_2026_candidate_features(
            clean_scores_2026
        )
    )

    print(
        "\n2026 FORMAL HOLDOUT COMPARISON",
        flush=True,
    )
    print(
        "------------------------------",
        flush=True,
    )

    result = evaluate_2026(
        clean_scores_2026,
        candidate_db,
        frozen_heads,
    )

    b = result["baseline_metrics"]
    c = result["calibration_control_metrics"]
    h = result["candidate_metrics"]
    db = result["delta_vs_clean_baseline"]
    dc = result["delta_vs_calibration_control"]

    print(
        f"CLEAN BASELINE: "
        f"n={b['rows']:,} "
        f"brier={b['brier']:.8f} "
        f"logloss={b['logloss']:.8f} "
        f"auc={b['auc']:.6f} "
        f"top5={b['top5_actual']:.6f} "
        f"top10={b['top10_actual']:.6f} "
        f"official_n={b['official_rows']:,} "
        f"official_hit_rate={b['official_hit_rate']:.6f}",
        flush=True,
    )

    print(
        f"CALIBRATION CONTROL: "
        f"n={c['rows']:,} "
        f"brier={c['brier']:.8f} "
        f"logloss={c['logloss']:.8f} "
        f"auc={c['auc']:.6f} "
        f"top5={c['top5_actual']:.6f} "
        f"top10={c['top10_actual']:.6f} "
        f"official_n={c['official_rows']:,} "
        f"official_hit_rate={c['official_hit_rate']:.6f}",
        flush=True,
    )

    print(
        f"FROZEN CHALLENGER: "
        f"n={h['rows']:,} "
        f"valid_n={result['valid_rows']:,} "
        f"coverage={result['coverage']:.4f} "
        f"brier={h['brier']:.8f} "
        f"logloss={h['logloss']:.8f} "
        f"auc={h['auc']:.6f} "
        f"top5={h['top5_actual']:.6f} "
        f"top10={h['top10_actual']:.6f} "
        f"official_n={h['official_rows']:,} "
        f"official_hit_rate={h['official_hit_rate']:.6f}",
        flush=True,
    )

    print(
        "DELTA VS CLEAN BASELINE: "
        f"brier={db['brier']:+.8f} "
        f"logloss={db['logloss']:+.8f} "
        f"auc={db['auc']:+.6f} "
        f"top5={db['top5']:+.6f} "
        f"top10={db['top10']:+.6f} "
        f"official_hit_rate="
        f"{db['official_hit_rate']:+.6f}",
        flush=True,
    )

    print(
        "DELTA VS CALIBRATION CONTROL: "
        f"brier={dc['brier']:+.8f} "
        f"logloss={dc['logloss']:+.8f} "
        f"auc={dc['auc']:+.6f} "
        f"top5={dc['top5']:+.6f} "
        f"top10={dc['top10']:+.6f} "
        f"official_hit_rate="
        f"{dc['official_hit_rate']:+.6f}",
        flush=True,
    )

    print(
        f"official_coverage_ratio="
        f"{result['official_coverage_ratio']:.6f}",
        flush=True,
    )

    print("\nMONTHLY STABILITY", flush=True)
    print("-----------------", flush=True)

    for month, row in result["monthly_2026"].items():
        print(
            f"{month}: "
            f"n={row['rows']:,} "
            f"eligible={row['eligible_for_stability_gate']} "
            f"baseline={row['baseline_brier']:.8f} "
            f"candidate={row['candidate_brier']:.8f} "
            f"delta={row['delta']:+.8f}",
            flush=True,
        )

    print(
        f"eligible_months={result['eligible_months']}",
        flush=True,
    )
    print(
        f"monthly_brier_nonworse="
        f"{result['monthly_brier_nonworse']}",
        flush=True,
    )
    print(
        f"monthly_brier_required_nonworse="
        f"{result['monthly_brier_required_nonworse']}",
        flush=True,
    )

    print("\nSTRICT FORMAL GATE", flush=True)
    print("------------------", flush=True)

    for key, value in result["gate"].items():
        print(
            f"{key}: {value}",
            flush=True,
        )

    print("\nFINAL VERDICT", flush=True)
    print("-------------", flush=True)
    print(
        f"verdict={result['verdict']}",
        flush=True,
    )

    if result["gate"]["overall_formal_gate_pass"]:
        print(
            "next_step=LOCK_AS_FORMAL_VALIDATED_CHALLENGER; "
            "do not stack with opportunity features yet; "
            "run production-integration audit and live shadow deployment first.",
            flush=True,
        )
    else:
        print(
            "next_step=FREEZE_PITCHER_CONTACT_CANDIDATE; "
            "no rescue tuning on 2026; "
            "do not test the other 2025 survivors on this same holdout.",
            flush=True,
        )

    payload = {
        "script": "HITS_PITCHER_CONTACT_2026_FORMAL_GATE_A",
        "candidate": CANDIDATE,
        "candidate_selected_before_2026_results": True,
        "candidate_count_opened_on_2026": 1,
        "holdout_year": 2026,
        "holdout_status_for_pitcher_contact_family": (
            "FORMAL_FORWARD_HOLDOUT"
        ),
        "baseline_construction": (
            "clean_incremental_model_through_2025_only"
        ),
        "production_artifact_used_for_2026_scoring": False,
        "stacking": False,
        "rescue_tuning": "FORBIDDEN",
        "fixed_gate": {
            "brier_vs_clean_baseline_max": -0.0005,
            "logloss_vs_clean_baseline_max": 0.0,
            "auc_vs_clean_baseline_min": -0.0005,
            "top5_vs_clean_baseline_min": 0.0,
            "top10_vs_clean_baseline_min": 0.0,
            "official_hit_rate_vs_clean_baseline_min": 0.0,
            "official_coverage_min_ratio": 0.90,
            "brier_vs_calibration_control_max": -0.0001,
            "logloss_vs_calibration_control_max": 0.0,
            "top5_vs_calibration_control_min": 0.0,
            "top10_vs_calibration_control_min": 0.0,
            "expected_coefficient_sign_required": True,
            "monthly_brier_stability_rule": (
                "at_least_half_of_eligible_months_with_500plus_rows_nonworse"
            ),
        },
        "clean_feature_summary": feature_summary,
        "frozen_heads": frozen_heads,
        "candidate_feature_summary": candidate_summary,
        "result": result,
    }

    OUT_JSON.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "HITS_PITCHER_CONTACT_2026_FORMAL_GATE_A",
        "=" * 40,
        "",
        f"Frozen candidate: {CANDIDATE}",
        "Candidate opened on 2026: exactly one.",
        "Clean baseline: trained through 2025 only.",
        "No stacking. No rescue tuning.",
        "",
        "2026 FORMAL HOLDOUT COMPARISON",
        "------------------------------",
        json.dumps(
            {
                "clean_baseline": result["baseline_metrics"],
                "calibration_control": (
                    result["calibration_control_metrics"]
                ),
                "frozen_challenger": result["candidate_metrics"],
                "delta_vs_clean_baseline": (
                    result["delta_vs_clean_baseline"]
                ),
                "delta_vs_calibration_control": (
                    result["delta_vs_calibration_control"]
                ),
                "official_coverage_ratio": (
                    result["official_coverage_ratio"]
                ),
            },
            indent=2,
        ),
        "",
        "MONTHLY STABILITY",
        "-----------------",
        json.dumps(
            result["monthly_2026"],
            indent=2,
        ),
        "",
        "STRICT FORMAL GATE",
        "------------------",
        json.dumps(
            result["gate"],
            indent=2,
        ),
        "",
        f"FINAL VERDICT: {result['verdict']}",
    ]

    OUT_TXT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nOUTPUTS", flush=True)
    print(OUT_JSON, flush=True)
    print(OUT_TXT, flush=True)

    return 0


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--worker-train",
        action="store_true",
    )

    parser.add_argument(
        "--worker-score",
        action="store_true",
    )

    parser.add_argument("--train-file")
    parser.add_argument("--prev-model")
    parser.add_argument("--out-model")

    parser.add_argument("--model-path")
    parser.add_argument("--score-file")
    parser.add_argument("--meta-file")
    parser.add_argument("--out-score-json")

    return parser


def main():
    args = build_parser().parse_args()

    if args.worker_train:
        worker_train(args)
        return 0

    if args.worker_score:
        worker_score(args)
        return 0

    return main_parent()


if __name__ == "__main__":
    raise SystemExit(main())
