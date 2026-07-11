#!/usr/bin/env python3
'''
HITS_B1_BLIND_LIVE_LOGGER_A

Blind prospective prediction logger for the frozen batter-hits holdout.

This module is intentionally outcome-blind. It never reads or stores:
- actual hits
- hit/miss result
- Brier
- logloss
- AUC
- calibration
- top-bucket performance
- official-tier performance
- any outcome-conditioned statistic

It logs only pregame inputs, A0/B0/B1 probabilities, immutable hashes, and
operational metadata.

Frozen models
-------------
A0:
    8 base features
    count:poisson
    P(hit >= 1) = 1 - exp(-lambda)

B0:
    same 8 base features
    binary:logistic

B1:
    same 8 base features
    + opp_pitcher_contact_allowed_rate_60d
    binary:logistic

Frozen pitcher-contact definition
---------------------------------
Prior 60-day opposing-pitcher contact allowed rate:

    1 - whiffs / swings

Swing descriptions:
    swinging_strike
    swinging_strike_blocked
    foul
    foul_tip
    hit_into_play
    hit_into_play_no_out
    hit_into_play_score
    missed_bunt
    foul_bunt

Same-day pitches are excluded.

Feature is available only when prior 60-day swings >= 150.
Otherwise feature 9 is missing (NaN), matching development training.

Expected production hook
------------------------
Call log_blind_hits_prediction(...) immediately after:

    base = batter_feature_row(pid)
    if base:
        base["batting_order"] = spot

and before build_batter_prop_picks(...).

CLI
---
Verify immutable lock + model freeze:
    python -u hits_b1_blind_live_logger_a.py --verify

Show outcome-blind operational status:
    python -u hits_b1_blind_live_logger_a.py --status
'''

import argparse
import hashlib
import json
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb


LOCK_JSON = Path("/data/hr_model/hits_b1_prospective_holdout_lock_a.json")
LOCK_SHA = Path("/data/hr_model/hits_b1_prospective_holdout_lock_a.sha256")

MODEL_DIR = Path("/data/hr_model/hits_b1_prospective_models_a")
MANIFEST_JSON = MODEL_DIR / "manifest.json"
MANIFEST_SHA = MODEL_DIR / "manifest.sha256"

STATCAST_DB = Path("/data/hr_model/hr_model.sqlite")

HOLDOUT_DIR = Path("/data/hr_model/hits_b1_prospective_holdout_a")
HOLDOUT_DB = HOLDOUT_DIR / "blind_predictions.sqlite"
LATEST_STATUS_JSON = HOLDOUT_DIR / "latest_operational_status.json"

TRIGGER_ROWS = 10000
MIN_PRIOR_SWINGS = 150

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

CANDIDATE = "opp_pitcher_contact_allowed_rate_60d"
B1_FEATURES = BASE_FEATURES + [CANDIDATE]

SWING_DESCRIPTIONS = (
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
    "missed_bunt",
    "foul_bunt",
)

_INIT_LOCK = threading.Lock()
_STATE = None


def _canonical_json_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _stored_sha(path):
    return path.read_text(encoding="utf-8").strip().split()[0]


def _verify_and_load_state():
    global _STATE

    if _STATE is not None:
        return _STATE

    with _INIT_LOCK:
        if _STATE is not None:
            return _STATE

        required = [
            LOCK_JSON,
            LOCK_SHA,
            MANIFEST_JSON,
            MANIFEST_SHA,
            STATCAST_DB,
        ]

        for path in required:
            if not path.exists():
                raise RuntimeError(f"Missing required frozen artifact: {path}")

        lock_payload = _load_json(LOCK_JSON)
        lock_actual = _sha256_bytes(_canonical_json_bytes(lock_payload))
        lock_expected = _stored_sha(LOCK_SHA)

        if lock_actual != lock_expected:
            raise RuntimeError("Holdout-lock SHA-256 mismatch.")

        manifest = _load_json(MANIFEST_JSON)
        manifest_actual = _sha256_bytes(_canonical_json_bytes(manifest))
        manifest_expected = _stored_sha(MANIFEST_SHA)

        if manifest_actual != manifest_expected:
            raise RuntimeError("Model-manifest SHA-256 mismatch.")

        if manifest.get("holdout_lock_sha256") != lock_actual:
            raise RuntimeError(
                "Model manifest is not bound to the current holdout lock."
            )

        for filename, metadata in manifest.get("artifacts", {}).items():
            path = MODEL_DIR / filename
            if not path.exists():
                raise RuntimeError(f"Missing frozen model artifact: {path}")
            actual = _sha256_file(path)
            expected = metadata.get("sha256")
            if actual != expected:
                raise RuntimeError(
                    f"Frozen artifact hash mismatch: {filename}"
                )

        model_cells = manifest.get("model_cells", {})
        for cell in ("A0", "B0", "B1"):
            if cell not in model_cells:
                raise RuntimeError(f"Manifest missing model cell {cell}")

        expected_columns = {
            "A0": BASE_FEATURES,
            "B0": BASE_FEATURES,
            "B1": B1_FEATURES,
        }

        boosters = {}
        columns = {}

        for cell in ("A0", "B0", "B1"):
            cell_meta = model_cells[cell]
            model_path = MODEL_DIR / cell_meta["model_file"]
            columns_path = MODEL_DIR / cell_meta["columns_file"]
            cols = _load_json(columns_path)

            if cols != expected_columns[cell]:
                raise RuntimeError(
                    f"{cell} frozen columns do not match locked specification."
                )

            booster = xgb.Booster()
            booster.load_model(str(model_path))

            if int(booster.num_boosted_rounds()) != 120:
                raise RuntimeError(
                    f"{cell} boosted rounds changed from frozen 120."
                )

            boosters[cell] = booster
            columns[cell] = cols

        holdout_start = lock_payload.get(
            "holdout_start_game_date_america_new_york"
        )
        if not holdout_start:
            raise RuntimeError("Holdout start date missing from immutable lock.")

        trigger_rows = int(
            lock_payload.get("holdout_trigger", {}).get("minimum_rows", 0)
        )
        if trigger_rows != TRIGGER_ROWS:
            raise RuntimeError(
                f"Holdout trigger mismatch: {trigger_rows} != {TRIGGER_ROWS}"
            )

        _STATE = {
            "lock_payload": lock_payload,
            "lock_sha256": lock_actual,
            "manifest": manifest,
            "manifest_sha256": manifest_actual,
            "holdout_start_game_date": holdout_start,
            "trigger_rows": trigger_rows,
            "boosters": boosters,
            "columns": columns,
        }
        return _STATE


def _connect_holdout_db():
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(HOLDOUT_DB), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS blind_predictions (
            unique_key TEXT PRIMARY KEY,
            game_date TEXT NOT NULL,
            game_id TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT,
            team TEXT,
            opponent TEXT,
            opposing_pitcher_id TEXT,
            lineup_spot INTEGER,

            season_avg REAL NOT NULL,
            recent15_avg REAL NOT NULL,
            recent5_avg REAL NOT NULL,
            hr_rate REAL NOT NULL,
            bb_rate REAL NOT NULL,
            so_rate REAL NOT NULL,
            batting_order REAL NOT NULL,
            games_played REAL NOT NULL,

            a0_lambda REAL NOT NULL,
            a0_probability REAL NOT NULL,
            b0_probability REAL NOT NULL,
            b1_probability REAL NOT NULL,

            pitcher_contact_allowed_rate_60d REAL,
            pitcher_prior_swings_60d INTEGER,
            pitcher_prior_whiffs_60d INTEGER,
            pitcher_feature_available INTEGER NOT NULL,

            holdout_lock_sha256 TEXT NOT NULL,
            model_manifest_sha256 TEXT NOT NULL,
            prediction_timestamp_utc TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS logger_state (
            key TEXT PRIMARY KEY,
            value INTEGER NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS operational_errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT NOT NULL,
            game_date TEXT,
            game_id TEXT,
            player_id TEXT,
            opposing_pitcher_id TEXT,
            error_type TEXT,
            error_message TEXT
        )
        """
    )

    conn.commit()
    return conn


def _increment_state(conn, key, amount=1):
    conn.execute(
        """
        INSERT INTO logger_state(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = value + excluded.value
        """,
        (key, int(amount)),
    )


def _record_operational_error(
    *,
    game_date=None,
    game_id=None,
    player_id=None,
    opposing_pitcher_id=None,
    exc,
):
    try:
        conn = _connect_holdout_db()
        conn.execute(
            """
            INSERT INTO operational_errors (
                timestamp_utc,
                game_date,
                game_id,
                player_id,
                opposing_pitcher_id,
                error_type,
                error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).replace(
                    microsecond=0
                ).isoformat(),
                str(game_date) if game_date is not None else None,
                str(game_id) if game_id is not None else None,
                str(player_id) if player_id is not None else None,
                str(opposing_pitcher_id)
                if opposing_pitcher_id is not None
                else None,
                type(exc).__name__,
                str(exc)[:1000],
            ),
        )
        _increment_state(conn, "error_attempts", 1)
        conn.commit()
        conn.close()
    except Exception:
        pass


def _pitcher_contact_allowed_60d(pitcher_id, game_date):
    '''
    Exact live analogue of the frozen formal-gate feature.

    Window:
        prior 60 calendar days
        game_date - 60 days through game_date - 1 day

    Same-day pitches excluded.
    '''
    if pitcher_id is None or not game_date:
        return {
            "available": False,
            "contact_allowed_rate_60d": None,
            "swings_60d": 0,
            "whiffs_60d": 0,
        }

    placeholders = ",".join("?" for _ in SWING_DESCRIPTIONS)

    sql = f"""
        SELECT
            SUM(
                CASE
                    WHEN description IN ({placeholders})
                    THEN 1
                    ELSE 0
                END
            ) AS swings_60d,
            SUM(COALESCE(is_whiff, 0)) AS whiffs_60d
        FROM statcast_pitches
        WHERE CAST(pitcher_id AS TEXT) = ?
          AND game_date >= date(?, '-60 days')
          AND game_date < ?
    """

    params = [
        *SWING_DESCRIPTIONS,
        str(pitcher_id),
        str(game_date),
        str(game_date),
    ]

    uri = f"file:{STATCAST_DB}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    try:
        row = conn.execute(sql, params).fetchone()
    finally:
        conn.close()

    swings = int(row[0] or 0)
    whiffs = int(row[1] or 0)

    if swings < MIN_PRIOR_SWINGS:
        return {
            "available": False,
            "contact_allowed_rate_60d": None,
            "swings_60d": swings,
            "whiffs_60d": whiffs,
        }

    contact = 1.0 - (whiffs / swings)

    return {
        "available": True,
        "contact_allowed_rate_60d": float(contact),
        "swings_60d": swings,
        "whiffs_60d": whiffs,
    }


def _base_vector(base_features):
    values = []
    for feature in BASE_FEATURES:
        if feature not in base_features:
            raise RuntimeError(f"Missing frozen base feature: {feature}")
        value = base_features[feature]
        if value is None:
            raise RuntimeError(f"Frozen base feature is None: {feature}")
        values.append(float(value))
    return values


def _predict_one(booster, values):
    array = np.array([values], dtype=np.float32)
    pred = booster.predict(xgb.DMatrix(array))
    if len(pred) != 1:
        raise RuntimeError(
            f"Expected one prediction, received {len(pred)}."
        )
    return float(pred[0])


def log_blind_hits_prediction(
    *,
    game_date,
    game_id,
    player_id,
    player_name,
    team,
    opponent,
    opposing_pitcher_id,
    lineup_spot,
    base_features,
):
    '''
    Score and append one blind A0/B0/B1 pregame prediction.

    Returns operational metadata only.
    Never computes outcome-conditioned performance.
    '''
    try:
        state = _verify_and_load_state()
        game_date = str(game_date)

        if game_date < state["holdout_start_game_date"]:
            return {
                "status": "before_holdout_start",
                "game_date": game_date,
                "holdout_start_game_date": state[
                    "holdout_start_game_date"
                ],
            }

        base = dict(base_features)
        base["batting_order"] = float(lineup_spot)
        base_values = _base_vector(base)

        a0_lambda = _predict_one(
            state["boosters"]["A0"],
            base_values,
        )
        a0_probability = 1.0 - math.exp(
            -max(float(a0_lambda), 1e-12)
        )
        a0_probability = min(
            max(a0_probability, 1e-8),
            1.0 - 1e-8,
        )

        b0_probability = _predict_one(
            state["boosters"]["B0"],
            base_values,
        )
        b0_probability = min(
            max(float(b0_probability), 1e-8),
            1.0 - 1e-8,
        )

        candidate = _pitcher_contact_allowed_60d(
            opposing_pitcher_id,
            game_date,
        )

        b1_values = list(base_values)
        if candidate["available"]:
            b1_values.append(
                float(candidate["contact_allowed_rate_60d"])
            )
        else:
            b1_values.append(float("nan"))

        b1_probability = _predict_one(
            state["boosters"]["B1"],
            b1_values,
        )
        b1_probability = min(
            max(float(b1_probability), 1e-8),
            1.0 - 1e-8,
        )

        unique_key = "|".join(
            [str(game_id), str(player_id), game_date]
        )

        timestamp_utc = datetime.now(
            timezone.utc
        ).replace(microsecond=0).isoformat()

        conn = _connect_holdout_db()
        try:
            _increment_state(conn, "total_attempts", 1)

            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO blind_predictions (
                    unique_key,
                    game_date,
                    game_id,
                    player_id,
                    player_name,
                    team,
                    opponent,
                    opposing_pitcher_id,
                    lineup_spot,

                    season_avg,
                    recent15_avg,
                    recent5_avg,
                    hr_rate,
                    bb_rate,
                    so_rate,
                    batting_order,
                    games_played,

                    a0_lambda,
                    a0_probability,
                    b0_probability,
                    b1_probability,

                    pitcher_contact_allowed_rate_60d,
                    pitcher_prior_swings_60d,
                    pitcher_prior_whiffs_60d,
                    pitcher_feature_available,

                    holdout_lock_sha256,
                    model_manifest_sha256,
                    prediction_timestamp_utc
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?
                )
                """,
                (
                    unique_key,
                    game_date,
                    str(game_id),
                    str(player_id),
                    str(player_name or ""),
                    str(team or ""),
                    str(opponent or ""),
                    str(opposing_pitcher_id)
                    if opposing_pitcher_id is not None
                    else None,
                    int(lineup_spot),

                    float(base["season_avg"]),
                    float(base["recent15_avg"]),
                    float(base["recent5_avg"]),
                    float(base["hr_rate"]),
                    float(base["bb_rate"]),
                    float(base["so_rate"]),
                    float(base["batting_order"]),
                    float(base["games_played"]),

                    float(a0_lambda),
                    float(a0_probability),
                    float(b0_probability),
                    float(b1_probability),

                    float(candidate["contact_allowed_rate_60d"])
                    if candidate["available"]
                    else None,
                    int(candidate["swings_60d"]),
                    int(candidate["whiffs_60d"]),
                    1 if candidate["available"] else 0,

                    state["lock_sha256"],
                    state["manifest_sha256"],
                    timestamp_utc,
                ),
            )

            inserted = cursor.rowcount == 1
            if inserted:
                _increment_state(conn, "inserted_rows", 1)
            else:
                _increment_state(conn, "duplicate_attempts", 1)

            conn.commit()
        finally:
            conn.close()

        return {
            "status": "inserted" if inserted else "duplicate_ignored",
            "unique_key": unique_key,
            "game_date": game_date,
            "pitcher_feature_available": candidate["available"],
            "pitcher_prior_swings_60d": candidate["swings_60d"],
        }

    except Exception as exc:
        _record_operational_error(
            game_date=game_date,
            game_id=game_id,
            player_id=player_id,
            opposing_pitcher_id=opposing_pitcher_id,
            exc=exc,
        )
        return {
            "status": "operational_error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def blind_hits_logger_status():
    state = _verify_and_load_state()
    conn = _connect_holdout_db()

    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS rows,
                MIN(game_date) AS min_game_date,
                MAX(game_date) AS max_game_date,
                COUNT(DISTINCT game_date) AS distinct_dates,
                SUM(
                    CASE WHEN pitcher_feature_available = 1
                    THEN 1 ELSE 0 END
                ) AS feature_available_rows
            FROM blind_predictions
            """
        ).fetchone()

        counts = {
            key: int(value)
            for key, value in conn.execute(
                "SELECT key, value FROM logger_state"
            )
        }

        error_rows = conn.execute(
            "SELECT COUNT(*) FROM operational_errors"
        ).fetchone()[0]
    finally:
        conn.close()

    rows = int(row[0] or 0)
    feature_available_rows = int(row[4] or 0)
    total_attempts = int(counts.get("total_attempts", 0))
    duplicate_attempts = int(counts.get("duplicate_attempts", 0))
    error_attempts = int(counts.get("error_attempts", 0))

    status = {
        "status": "FROZEN_PROSPECTIVE_CHALLENGER_BLIND_COLLECTION",
        "outcome_blind": True,
        "outcome_conditioned_metrics_exposed": False,
        "holdout_start_game_date": state["holdout_start_game_date"],
        "trigger_rows": state["trigger_rows"],
        "logged_rows": rows,
        "rows_remaining_to_trigger": max(
            state["trigger_rows"] - rows,
            0,
        ),
        "trigger_progress_pct": round(
            rows / state["trigger_rows"] * 100.0,
            4,
        ),
        "min_game_date": row[1],
        "max_game_date": row[2],
        "distinct_game_dates": int(row[3] or 0),
        "pitcher_feature_available_rows": feature_available_rows,
        "pitcher_feature_coverage": (
            feature_available_rows / rows
            if rows
            else None
        ),
        "total_attempts": total_attempts,
        "duplicate_attempts": duplicate_attempts,
        "duplicate_attempt_rate": (
            duplicate_attempts / total_attempts
            if total_attempts
            else 0.0
        ),
        "error_attempts": error_attempts,
        "operational_error_rows": int(error_rows),
        "operational_error_rate": (
            error_attempts / total_attempts
            if total_attempts
            else 0.0
        ),
        "holdout_lock_sha256": state["lock_sha256"],
        "model_manifest_sha256": state["manifest_sha256"],
        "formal_gate_open_authorized": (
            rows >= state["trigger_rows"]
        ),
        "database": str(HOLDOUT_DB),
    }

    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_STATUS_JSON.write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )

    return status


def verify_blind_logger():
    state = _verify_and_load_state()

    return {
        "verdict": "BLIND_LIVE_LOGGER_VERIFIED",
        "outcome_blind": True,
        "holdout_start_game_date": state["holdout_start_game_date"],
        "trigger_rows": state["trigger_rows"],
        "holdout_lock_sha256": state["lock_sha256"],
        "model_manifest_sha256": state["manifest_sha256"],
        "models": {
            cell: {
                "boosted_rounds": int(
                    state["boosters"][cell].num_boosted_rounds()
                ),
                "columns": state["columns"][cell],
            }
            for cell in ("A0", "B0", "B1")
        },
        "pitcher_contact_definition": {
            "window_days": 60,
            "same_day_excluded": True,
            "minimum_prior_swings": MIN_PRIOR_SWINGS,
            "formula": "1 - whiffs / swings",
            "swing_descriptions": list(SWING_DESCRIPTIONS),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.verify:
        print(json.dumps(verify_blind_logger(), indent=2))
        return 0

    if args.status:
        print(json.dumps(blind_hits_logger_status(), indent=2))
        return 0

    print(
        "Use --verify or --status. "
        "Live predictions are written through the api.py integration hook."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
