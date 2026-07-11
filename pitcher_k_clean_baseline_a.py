#!/usr/bin/env python3
"""
PITCHER_K_CLEAN_BASELINE_A

Leak-safe reconstruction and audit of the current Prop Edge pitcher-strikeout
incumbent.

This script is READ-ONLY with respect to production. It does not:
- modify api.py, ksim.py, lineupk.py, or bvp.py
- change production predictions
- call external APIs
- touch the frozen batter-hits prospective holdout
- tune thresholds
- rescue failed hypotheses
- test any challenger architecture

It builds two complementary incumbent baselines:

LAYER A — BROAD HISTORICAL ARCHITECTURE BASELINE
------------------------------------------------
Reconstructs the current K architecture from local Statcast history with strict
D-1 feature construction.

Primary forward baseline:
    2025

Warm-up / diagnostic year:
    2024

Why 2025 is primary:
    Local pitch-level H2H/BVP history begins in 2024. The live production code
    uses career H2H from MLB Stats API. An honest historical reconstruction
    cannot query today's career totals for old games without leakage.
    Therefore:
      - 2024 is useful as a warm-up diagnostic.
      - 2025 can use 2024 + prior-2025 local H2H history and is the preferred
        clean local-history forward baseline.
    This is disclosed explicitly and never hidden.

Exact incumbent logic preserved:
- pitcher outings with BF >= 12 build the pitcher profile
- minimum 3 qualifying prior outings
- recency weights exp(-0.6 * age)
- final K/BF center = 85% recency-weighted + 15% season
- expected workload = mean BF over last 5 qualifying outings
- volatility pool = last 12 per-outing K/BF rates
- general batter K rate vs pitcher hand, minimum 15 season PA
- fallback current-season overall K rate, minimum 15 PA
- final fallback league average 0.22
- H2H minimum 2 PA
- batter blend = 40% H2H + 60% general
- CURRENT n_data SEMANTICS PRESERVED:
    n_data counts H2H-qualified batters only
- lineup expectation activates only when n_data >= 5
- pitcher/lineup blend = 55% pitcher + 45% lineup
- BVP lineup nudge sample gate = 20 prior H2H PA
- k_nudge = lineup H2H K rate / 0.22, capped [0.85, 1.15],
  rounded to 3 decimals
- k_nudge applied to central blended K/BF and recent start-rate pool
- K simulation distribution:
    BF ~ Normal(expected_bf, 2.5), rounded, clamped [9, 30]
    >=4 start rates -> 70% sampled start rate + 30% central rate
    TTO factors 1.00 / 0.94 / 0.85
    K probability clamped [0.02, 0.60]
- 10,000 simulations
- deterministic stable seed FOR RESEARCH REPRODUCIBILITY ONLY

Research simulation note:
    The historical simulator uses vectorized binomial draws. This is
    distribution-equivalent to production's per-PA Bernoulli loop, but not
    bitwise RNG-stream identical.

Historical lineup recovery note:
    The completed-game Statcast feed is used to recover the first nine unique
    batters faced by the starter. Batter identity is pregame-knowable when the
    confirmed lineup is posted, but the recovery method itself is retrospective
    and is explicitly disclosed.

Starter recovery note:
    The two earliest distinct pitchers to appear in a game are treated as the
    two starting pitchers. This handles normal starts and openers consistently
    with the available local schema.

LAYER B — EXACT RECENT REAL-LINE AUDIT
--------------------------------------
Uses saved live pitcher-K candidate rows and archived real K lines. It never
fabricates historical sportsbook lines.

Actual strikeouts are resolved from:
    1. local Statcast pitcher-game totals when available
    2. /data/record.json fallback for graded production rows

Metrics:
- projection bias / MAE / RMSE / CRPS
- simulated 10th-90th and IQR coverage
- by year and month
- lineup layer on/off
- BVP nudge on/off
- prior-outing count bands
- workload bands
- exact recent real-line Brier / logloss / side hit rate
- OVER / UNDER splits
- official / watchlist splits
- lineup flag / legacy-warning splits
- integer-line push behavior
- exact sample coverage

Known findings are preserved, not changed:
K-A01:
    lineup n_data counts H2H-qualified batters only.

K-A02:
    production ksim is nondeterministic because it uses an unseeded RNG.

K-A03:
    official threshold starts at 0.63 while MEDIUM confidence starts at 0.64.

K-H01:
    possible TTO double-counting.

K-H02:
    possible H2H signal double-counting between lineupk and bvp layers.

Memory safety:
- no pandas
- no sklearn
- SQLite disk work
- streaming game/date processing
- compact dictionaries only
- one historical starter scored at a time
- 512 MB Render target

Run
---
python -u pitcher_k_clean_baseline_a.py 2>&1 | tee /data/hr_model/pitcher_k_clean_baseline_a.log

Outputs
-------
/data/hr_model/pitcher_k_clean_baseline_a_results.json
/data/hr_model/pitcher_k_clean_baseline_a_report.txt
/data/hr_model/pitcher_k_clean_baseline_a_work/baseline.sqlite

Paste back
----------
SOURCE PROVENANCE
LOCAL DATA COVERAGE
HISTORICAL RECONSTRUCTION DISCLOSURES
2024 WARM-UP DIAGNOSTIC
2025 PRIMARY FORWARD BASELINE
ARCHITECTURE SPLITS
EXACT RECENT REAL-LINE AUDIT
BASELINE STATUS
FINAL VERDICT
"""

import hashlib
import json
import math
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# =============================================================================
# Paths / frozen incumbent constants
# =============================================================================

DATA_DIR = Path("/data")
HR_DIR = Path("/data/hr_model")
STATCAST_DB = HR_DIR / "hr_model.sqlite"

PRED_DIR = DATA_DIR / "predictions"
K_LINE_DIR = DATA_DIR / "k_lines"
RECORD_JSON = DATA_DIR / "record.json"

WORK_DIR = HR_DIR / "pitcher_k_clean_baseline_a_work"
WORK_DB = WORK_DIR / "baseline.sqlite"

OUT_JSON = HR_DIR / "pitcher_k_clean_baseline_a_results.json"
OUT_TXT = HR_DIR / "pitcher_k_clean_baseline_a_report.txt"

SOURCE_FILES = [
    Path("api.py"),
    Path("ksim.py"),
    Path("lineupk.py"),
    Path("bvp.py"),
]

HISTORICAL_YEARS = [2024, 2025]
PRIMARY_YEAR = 2025
WARMUP_YEAR = 2024

LEAGUE_AVG_K = 0.22
RECENCY_DECAY = 0.6
SEASON_ANCHOR = 0.15

PITCHER_WEIGHT = 0.55
LINEUP_WEIGHT = 0.45

H2H_WEIGHT = 0.40
GEN_WEIGHT = 0.60
MIN_H2H_PA = 2
MIN_GENERAL_PA = 15

BVP_MIN_SAMPLE_PA = 20
K_NUDGE_MIN = 0.85
K_NUDGE_MAX = 1.15

MIN_QUALIFYING_BF = 12
MIN_PRIOR_QUALIFYING_OUTINGS = 3
RECENT_BF_WINDOW = 5
VOLATILITY_RATE_WINDOW = 12

SIMS = 10000
BF_SD = 2.5
BF_MIN = 9
BF_MAX = 30

TTO_1 = 1.00
TTO_2 = 0.94
TTO_3 = 0.85

KSIM_NO_BET_MIN = 0.59
WATCHLIST_MIN = 0.60
OFFICIAL_MIN = 0.63

SWING_EVENTS_K = {
    "strikeout",
    "strikeout_double_play",
}

KNOWN_FINDINGS = {
    "K-A01": (
        "lineup_k_expectation n_data counts qualifying H2H batters only; "
        "production activates the 45% lineup component only when n_data >= 5."
    ),
    "K-A02": (
        "Production ksim uses unseeded np.random.RandomState(), so identical "
        "inputs can produce different probabilities near decision thresholds."
    ),
    "K-A03": (
        "Official threshold begins at 0.63 while MEDIUM confidence begins at "
        "0.64, so some LOW-confidence predictions can be official."
    ),
    "K-H01": (
        "Open hypothesis: realized full-outing K/BF may already contain TTO "
        "effects before the simulator applies another 1.00/0.94/0.85 decay."
    ),
    "K-H02": (
        "Open hypothesis: same-pitcher H2H K history can enter through both "
        "lineupk's batter blend and bvp.py's lineup-wide multiplicative nudge."
    ),
}


# =============================================================================
# Generic helpers
# =============================================================================

def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def stable_seed(*parts):
    raw = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()[:16]
    return int(digest, 16) % (2 ** 32)


def safe_json_load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=None):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def clip(value, lo, hi):
    return max(lo, min(hi, value))


def norm_name(value):
    if value is None:
        return ""

    s = unicodedata.normalize("NFKD", str(value))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace(".", " ").replace("-", " ")
    s = s.replace("'", " ").replace("’", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    s = "".join(c for c in s if c.isalpha() or c == " ")
    return re.sub(r"\s+", " ", s).strip()


def source_provenance():
    rows = []

    for path in SOURCE_FILES:
        row = {
            "path": str(path),
            "exists": path.exists(),
        }

        if path.exists():
            row.update(
                {
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

        rows.append(row)

    return rows


# =============================================================================
# SQLite build: pitch appearances, pitcher games, starter games
# =============================================================================

def connect_work_db():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(WORK_DB), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-24000")
    conn.execute("PRAGMA busy_timeout=60000")

    return conn


def source_statcast_info():
    if not STATCAST_DB.exists():
        raise RuntimeError(f"Missing Statcast DB: {STATCAST_DB}")

    conn = sqlite3.connect(f"file:{STATCAST_DB}?mode=ro", uri=True)

    try:
        cols = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(statcast_pitches)"
            )
        }

        required = {
            "game_pk",
            "game_date",
            "at_bat_number",
            "pitch_number",
            "batter_id",
            "pitcher_id",
            "p_throws",
            "events",
        }

        missing = sorted(required - cols)

        if missing:
            raise RuntimeError(
                "Statcast DB missing required columns: "
                + ", ".join(missing)
            )

        row = conn.execute(
            """
            SELECT
                COUNT(*),
                MIN(game_date),
                MAX(game_date),
                COUNT(DISTINCT game_pk)
            FROM statcast_pitches
            """
        ).fetchone()

        by_year = []

        for yr in HISTORICAL_YEARS + [2026]:
            r = conn.execute(
                """
                SELECT
                    COUNT(*),
                    MIN(game_date),
                    MAX(game_date),
                    COUNT(DISTINCT game_pk)
                FROM statcast_pitches
                WHERE SUBSTR(game_date, 1, 4) = ?
                """,
                (str(yr),),
            ).fetchone()

            by_year.append(
                {
                    "year": yr,
                    "pitch_rows": int(r[0] or 0),
                    "min_date": r[1],
                    "max_date": r[2],
                    "games": int(r[3] or 0),
                }
            )

        return {
            "path": str(STATCAST_DB),
            "columns": sorted(cols),
            "overall": {
                "pitch_rows": int(row[0] or 0),
                "min_date": row[1],
                "max_date": row[2],
                "games": int(row[3] or 0),
            },
            "by_year": by_year,
        }

    finally:
        conn.close()


def ensure_base_tables(conn):
    """
    Build once and reuse.

    pa:
        one row per completed plate appearance, final-pitch event only.

    pitcher_game:
        one row per pitcher-game.

    starter_games:
        the two earliest distinct pitchers to appear in each game.
    """

    completion = WORK_DIR / "base_tables_complete.json"

    required_tables = {
        "pa",
        "pitcher_game",
        "starter_games",
    }

    existing_tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }

    if completion.exists() and required_tables.issubset(existing_tables):
        summary = safe_json_load(completion)

        if isinstance(summary, dict):
            print("Base tables already complete; reusing.", flush=True)
            return summary

    print("Building local plate-appearance history tables...", flush=True)

    for table in (
        "historical_rows",
        "starter_games",
        "pitcher_game",
        "pa",
    ):
        conn.execute(f"DROP TABLE IF EXISTS {table}")

    conn.execute(f"ATTACH DATABASE 'file:{STATCAST_DB}?mode=ro' AS src")

    # Include 2024 onward so recent 2026 exact-line candidates can resolve
    # local actual Ks when local Statcast coverage exists.
    conn.execute(
        """
        CREATE TABLE pa AS
        WITH ranked AS (
            SELECT
                CAST(game_pk AS TEXT) AS game_id,
                game_date,
                CAST(at_bat_number AS INTEGER) AS at_bat_number,
                CAST(pitch_number AS INTEGER) AS pitch_number,
                CAST(batter_id AS TEXT) AS batter_id,
                CAST(pitcher_id AS TEXT) AS pitcher_id,
                p_throws,
                events,
                ROW_NUMBER() OVER (
                    PARTITION BY game_pk, at_bat_number
                    ORDER BY
                        COALESCE(pitch_number, 0) DESC
                ) AS rn
            FROM src.statcast_pitches
            WHERE game_date >= '2024-01-01'
        )
        SELECT
            game_id,
            game_date,
            at_bat_number,
            batter_id,
            pitcher_id,
            p_throws,
            events,
            CASE
                WHEN events IN ('strikeout', 'strikeout_double_play')
                THEN 1
                ELSE 0
            END AS is_k
        FROM ranked
        WHERE rn = 1
          AND batter_id IS NOT NULL
          AND pitcher_id IS NOT NULL
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_pa_date_game
        ON pa(game_date, game_id, at_bat_number)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_pa_batter_date
        ON pa(batter_id, game_date)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_pa_pitcher_date
        ON pa(pitcher_id, game_date)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_pa_h2h_date
        ON pa(batter_id, pitcher_id, game_date)
        """
    )

    conn.execute(
        """
        CREATE TABLE pitcher_game AS
        SELECT
            game_id,
            game_date,
            pitcher_id,
            MIN(at_bat_number) AS first_ab,
            COUNT(*) AS bf,
            SUM(is_k) AS so,
            MAX(p_throws) AS p_throws
        FROM pa
        GROUP BY game_id, game_date, pitcher_id
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_pitcher_game_pitcher_date
        ON pitcher_game(pitcher_id, game_date)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_pitcher_game_game
        ON pitcher_game(game_id, first_ab)
        """
    )

    conn.execute(
        """
        CREATE TABLE starter_games AS
        WITH ranked AS (
            SELECT
                game_id,
                game_date,
                pitcher_id,
                first_ab,
                bf,
                so,
                p_throws,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id
                    ORDER BY first_ab, pitcher_id
                ) AS starter_rank
            FROM pitcher_game
        )
        SELECT
            game_id,
            game_date,
            pitcher_id,
            first_ab,
            bf,
            so,
            p_throws,
            starter_rank
        FROM ranked
        WHERE starter_rank <= 2
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_starter_games_date
        ON starter_games(game_date, game_id)
        """
    )

    conn.execute("DETACH DATABASE src")
    conn.commit()

    summary = {
        "built_at_utc": now_utc(),
        "pa_rows": conn.execute(
            "SELECT COUNT(*) FROM pa"
        ).fetchone()[0],
        "pitcher_game_rows": conn.execute(
            "SELECT COUNT(*) FROM pitcher_game"
        ).fetchone()[0],
        "starter_game_rows": conn.execute(
            "SELECT COUNT(*) FROM starter_games"
        ).fetchone()[0],
        "min_date": conn.execute(
            "SELECT MIN(game_date) FROM pa"
        ).fetchone()[0],
        "max_date": conn.execute(
            "SELECT MAX(game_date) FROM pa"
        ).fetchone()[0],
    }

    completion.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(
        f"Base tables built: "
        f"PA={summary['pa_rows']:,} "
        f"pitcher_game={summary['pitcher_game_rows']:,} "
        f"starter_game={summary['starter_game_rows']:,} "
        f"range={summary['min_date']}..{summary['max_date']}",
        flush=True,
    )

    return summary


# =============================================================================
# Exact incumbent historical reconstruction
# =============================================================================

def create_historical_rows_table(conn):
    conn.execute("DROP TABLE IF EXISTS historical_rows")

    conn.execute(
        """
        CREATE TABLE historical_rows (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER NOT NULL,
            game_date TEXT NOT NULL,
            game_id TEXT NOT NULL,
            pitcher_id TEXT NOT NULL,
            pitcher_throws TEXT,
            actual_k INTEGER NOT NULL,
            actual_bf INTEGER NOT NULL,

            prior_qualifying_outings INTEGER NOT NULL,
            season_k_per_bf REAL NOT NULL,
            recent_weighted_k_per_bf REAL NOT NULL,
            final_k_per_bf REAL NOT NULL,
            expected_bf REAL NOT NULL,

            lineup_size INTEGER NOT NULL,
            h2h_qualified_batters INTEGER NOT NULL,
            lineup_component_used INTEGER NOT NULL,
            lineup_avg_k_rate REAL,
            lineup_expected_k REAL,

            bvp_sample_pa INTEGER NOT NULL,
            bvp_lineup_k_rate REAL,
            k_nudge REAL NOT NULL,

            blended_projection_pre_nudge REAL NOT NULL,
            blended_k_per_bf_post_nudge REAL NOT NULL,

            sim_mean REAL NOT NULL,
            sim_median REAL NOT NULL,
            sim_iqr REAL NOT NULL,
            sim_q10 REAL NOT NULL,
            sim_q90 REAL NOT NULL,
            sim_crps REAL NOT NULL,
            actual_in_iqr INTEGER NOT NULL,
            actual_in_10_90 INTEGER NOT NULL,

            research_seed INTEGER NOT NULL,
            simulation_mode TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_historical_rows_year_date
        ON historical_rows(year, game_date)
        """
    )

    conn.commit()


def pitcher_profile(prior_outings):
    """
    prior_outings:
        list of (game_date, bf, so), current season only, D-1.
    """
    qualifying = [
        row
        for row in prior_outings
        if row[1] >= MIN_QUALIFYING_BF
    ]

    if len(qualifying) < MIN_PRIOR_QUALIFYING_OUTINGS:
        return None

    sos = [float(row[2]) for row in qualifying]
    bfs = [float(row[1]) for row in qualifying]

    cum_so = sum(sos)
    cum_bf = sum(bfs)

    if cum_bf <= 0:
        return None

    season_kbf = cum_so / cum_bf

    n = len(sos)
    weights = [
        math.exp(-RECENCY_DECAY * (n - 1 - i))
        for i in range(n)
    ]

    numerator = sum(
        w * so
        for w, so in zip(weights, sos)
    )
    denominator = sum(
        w * bf
        for w, bf in zip(weights, bfs)
    )

    recent_kbf = (
        numerator / denominator
        if denominator > 0
        else season_kbf
    )

    final_kbf = (
        (1.0 - SEASON_ANCHOR) * recent_kbf
        + SEASON_ANCHOR * season_kbf
    )

    recent_bf = bfs[-RECENT_BF_WINDOW:]
    expected_bf = sum(recent_bf) / len(recent_bf)

    per_outing_rates = [
        so / bf if bf > 0 else 0.0
        for so, bf in zip(sos, bfs)
    ][-VOLATILITY_RATE_WINDOW:]

    return {
        "qualifying_outings": len(qualifying),
        "season_kbf": season_kbf,
        "recent_kbf": recent_kbf,
        "final_kbf": final_kbf,
        "expected_bf": expected_bf,
        "per_outing_rates": per_outing_rates,
    }


def vectorized_ksim(
    k_per_bf,
    expected_bf,
    start_k_rates,
    *,
    seed,
    actual_k,
    sims=SIMS,
):
    """
    Distribution-equivalent vectorized research version of current ksim logic.

    Production:
        loops PA by PA using rng.rand()

    Research:
        uses binomial draws by TTO segment.

    The probability model is the same; the random-number stream is not bitwise
    identical. A stable seed is used for reproducibility.
    """
    rng = np.random.RandomState(seed)

    if start_k_rates and len(start_k_rates) >= 4:
        pool = np.array(start_k_rates, dtype=np.float64)
        pool = 0.7 * pool + 0.3 * float(k_per_bf)
    else:
        pool = np.array([float(k_per_bf)], dtype=np.float64)

    bf = np.rint(
        rng.normal(float(expected_bf), BF_SD, size=sims)
    ).astype(np.int16)

    bf = np.clip(bf, BF_MIN, BF_MAX)

    pool_idx = rng.randint(0, len(pool), size=sims)
    k_sample = pool[pool_idx]

    n1 = np.minimum(bf, 9)
    n2 = np.minimum(np.maximum(bf - 9, 0), 9)
    n3 = np.maximum(bf - 18, 0)

    p1 = np.clip(k_sample * TTO_1, 0.02, 0.60)
    p2 = np.clip(k_sample * TTO_2, 0.02, 0.60)
    p3 = np.clip(k_sample * TTO_3, 0.02, 0.60)

    results = (
        rng.binomial(n1, p1)
        + rng.binomial(n2, p2)
        + rng.binomial(n3, p3)
    ).astype(np.int16)

    mean_ks = float(np.mean(results))
    q10, q25, q50, q75, q90 = np.percentile(
        results,
        [10, 25, 50, 75, 90],
    )

    sorted_results = np.sort(results.astype(np.float64))
    n = len(sorted_results)
    coeff = 2.0 * np.arange(1, n + 1) - n - 1.0

    mean_abs = float(
        np.mean(np.abs(sorted_results - float(actual_k)))
    )

    half_pairwise = float(
        np.sum(coeff * sorted_results) / (n * n)
    )

    crps = mean_abs - half_pairwise

    return {
        "mean": mean_ks,
        "median": float(q50),
        "iqr": float(q75 - q25),
        "q10": float(q10),
        "q25": float(q25),
        "q75": float(q75),
        "q90": float(q90),
        "crps": float(crps),
        "actual_in_iqr": int(
            float(q25) <= float(actual_k) <= float(q75)
        ),
        "actual_in_10_90": int(
            float(q10) <= float(actual_k) <= float(q90)
        ),
    }


def reconstruct_historical_baseline(conn):
    completion = WORK_DIR / "historical_reconstruction_complete.json"

    if completion.exists():
        summary = safe_json_load(completion)

        if isinstance(summary, dict):
            existing = conn.execute(
                "SELECT COUNT(*) FROM historical_rows"
            ).fetchone()[0] if "historical_rows" in {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            } else 0

            if existing == int(summary.get("rows", -1)):
                print(
                    "Historical reconstruction already complete; reusing.",
                    flush=True,
                )
                return summary

    create_historical_rows_table(conn)

    # D-1 state.
    # pitcher_outings[(year, pitcher_id)] = [(date, bf, so), ...]
    pitcher_outings = defaultdict(list)

    # Current-season general K state.
    # hand_state[(year, batter_id, p_throws)] = [PA, K]
    hand_state = defaultdict(lambda: [0, 0])
    season_state = defaultdict(lambda: [0, 0])

    # Local-history H2H state across 2024 onward.
    # h2h_state[(batter_id, pitcher_id)] = [PA, K]
    h2h_state = defaultdict(lambda: [0, 0])

    historical_rows = 0
    skipped_profile = 0
    skipped_lineup = 0

    yearly_scored = defaultdict(int)
    yearly_starters = defaultdict(int)

    dates = [
        row[0]
        for row in conn.execute(
            """
            SELECT DISTINCT game_date
            FROM pa
            WHERE SUBSTR(game_date, 1, 4) IN ('2024', '2025')
            ORDER BY game_date
            """
        )
    ]

    insert_sql = """
        INSERT INTO historical_rows (
            year,
            game_date,
            game_id,
            pitcher_id,
            pitcher_throws,
            actual_k,
            actual_bf,

            prior_qualifying_outings,
            season_k_per_bf,
            recent_weighted_k_per_bf,
            final_k_per_bf,
            expected_bf,

            lineup_size,
            h2h_qualified_batters,
            lineup_component_used,
            lineup_avg_k_rate,
            lineup_expected_k,

            bvp_sample_pa,
            bvp_lineup_k_rate,
            k_nudge,

            blended_projection_pre_nudge,
            blended_k_per_bf_post_nudge,

            sim_mean,
            sim_median,
            sim_iqr,
            sim_q10,
            sim_q90,
            sim_crps,
            actual_in_iqr,
            actual_in_10_90,

            research_seed,
            simulation_mode
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?
        )
    """

    batch = []

    for date_idx, game_date in enumerate(dates, 1):
        year = int(game_date[:4])

        starters = list(
            conn.execute(
                """
                SELECT
                    game_id,
                    pitcher_id,
                    bf,
                    so,
                    p_throws
                FROM starter_games
                WHERE game_date = ?
                ORDER BY game_id, starter_rank
                """,
                (game_date,),
            )
        )

        for (
            game_id,
            pitcher_id,
            actual_bf,
            actual_k,
            p_throws,
        ) in starters:
            yearly_starters[year] += 1

            profile = pitcher_profile(
                pitcher_outings[(year, str(pitcher_id))]
            )

            if profile is None:
                skipped_profile += 1
                continue

            # Retrospective recovery of confirmed-lineup identity from completed
            # game data. First nine unique batters faced by this starter.
            lineup = []
            seen = set()

            for (batter_id,) in conn.execute(
                """
                SELECT batter_id
                FROM pa
                WHERE game_id = ?
                  AND pitcher_id = ?
                ORDER BY at_bat_number
                """,
                (str(game_id), str(pitcher_id)),
            ):
                batter_id = str(batter_id)

                if batter_id in seen:
                    continue

                seen.add(batter_id)
                lineup.append(batter_id)

                if len(lineup) >= 9:
                    break

            if not lineup:
                skipped_lineup += 1
                continue

            throws = str(p_throws or "R")

            rates = []
            h2h_qualified = 0

            bvp_tot_pa = 0
            bvp_tot_k = 0

            for batter_id in lineup[:9]:
                hand_key = (
                    year,
                    batter_id,
                    throws,
                )
                season_key = (
                    year,
                    batter_id,
                )

                hand_pa, hand_k = hand_state[hand_key]
                season_pa, season_k = season_state[season_key]

                if hand_pa >= MIN_GENERAL_PA:
                    gen_rate = hand_k / hand_pa
                elif season_pa >= MIN_GENERAL_PA:
                    gen_rate = season_k / season_pa
                else:
                    gen_rate = LEAGUE_AVG_K

                h2h_pa, h2h_k = h2h_state[
                    (batter_id, str(pitcher_id))
                ]

                if h2h_pa >= MIN_H2H_PA:
                    h2h_rate = h2h_k / h2h_pa
                    rate = (
                        H2H_WEIGHT * h2h_rate
                        + GEN_WEIGHT * gen_rate
                    )
                    h2h_qualified += 1
                else:
                    rate = gen_rate

                rates.append(rate)

                if h2h_pa > 0:
                    bvp_tot_pa += h2h_pa
                    bvp_tot_k += h2h_k

            lineup_avg_k = (
                sum(rates) / len(rates)
                if rates
                else None
            )

            lineup_expected_k = (
                lineup_avg_k * profile["expected_bf"]
                if lineup_avg_k is not None
                else None
            )

            lineup_component_used = int(
                lineup_expected_k is not None
                and lineup_expected_k > 0
                and h2h_qualified >= 5
            )

            pitcher_projection = (
                profile["final_kbf"]
                * profile["expected_bf"]
            )

            if lineup_component_used:
                blended = (
                    PITCHER_WEIGHT * pitcher_projection
                    + LINEUP_WEIGHT * lineup_expected_k
                )
            else:
                blended = pitcher_projection

            if bvp_tot_pa >= BVP_MIN_SAMPLE_PA:
                bvp_lineup_k_rate = (
                    bvp_tot_k / bvp_tot_pa
                    if bvp_tot_pa > 0
                    else None
                )

                raw_nudge = (
                    bvp_lineup_k_rate / LEAGUE_AVG_K
                    if bvp_lineup_k_rate is not None
                    else 1.0
                )

                k_nudge = round(
                    clip(
                        raw_nudge,
                        K_NUDGE_MIN,
                        K_NUDGE_MAX,
                    ),
                    3,
                )
            else:
                bvp_lineup_k_rate = None
                k_nudge = 1.0

            blended_kbf = (
                (blended / profile["expected_bf"])
                * k_nudge
            )

            start_rates = list(
                profile["per_outing_rates"]
            )

            if start_rates and k_nudge != 1.0:
                start_rates = [
                    rate * k_nudge
                    for rate in start_rates
                ]

            seed = stable_seed(
                "PITCHER_K_CLEAN_BASELINE_A",
                game_date,
                game_id,
                pitcher_id,
                SIMS,
            )

            sim = vectorized_ksim(
                blended_kbf,
                profile["expected_bf"],
                start_rates,
                seed=seed,
                actual_k=int(actual_k),
                sims=SIMS,
            )

            batch.append(
                (
                    year,
                    str(game_date),
                    str(game_id),
                    str(pitcher_id),
                    throws,
                    int(actual_k),
                    int(actual_bf),

                    int(profile["qualifying_outings"]),
                    float(profile["season_kbf"]),
                    float(profile["recent_kbf"]),
                    float(profile["final_kbf"]),
                    float(profile["expected_bf"]),

                    int(len(lineup[:9])),
                    int(h2h_qualified),
                    int(lineup_component_used),
                    (
                        float(lineup_avg_k)
                        if lineup_avg_k is not None
                        else None
                    ),
                    (
                        float(lineup_expected_k)
                        if lineup_expected_k is not None
                        else None
                    ),

                    int(bvp_tot_pa),
                    (
                        float(bvp_lineup_k_rate)
                        if bvp_lineup_k_rate is not None
                        else None
                    ),
                    float(k_nudge),

                    float(blended),
                    float(blended_kbf),

                    float(sim["mean"]),
                    float(sim["median"]),
                    float(sim["iqr"]),
                    float(sim["q10"]),
                    float(sim["q90"]),
                    float(sim["crps"]),
                    int(sim["actual_in_iqr"]),
                    int(sim["actual_in_10_90"]),

                    int(seed),
                    "vectorized_distribution_equivalent_stable_seed_v1",
                )
            )

            historical_rows += 1
            yearly_scored[year] += 1

            if len(batch) >= 250:
                conn.executemany(insert_sql, batch)
                conn.commit()
                batch.clear()

        # D-1 discipline: update all history only AFTER every game on this date
        # has been scored.

        # Update pitcher outing history.
        for (
            pitcher_id,
            bf,
            so,
        ) in conn.execute(
            """
            SELECT
                pitcher_id,
                bf,
                so
            FROM pitcher_game
            WHERE game_date = ?
            """,
            (game_date,),
        ):
            pitcher_outings[
                (year, str(pitcher_id))
            ].append(
                (
                    game_date,
                    int(bf),
                    int(so),
                )
            )

        # Update batter general and H2H history.
        for (
            batter_id,
            pitcher_id,
            throws,
            is_k,
        ) in conn.execute(
            """
            SELECT
                batter_id,
                pitcher_id,
                COALESCE(p_throws, 'R'),
                is_k
            FROM pa
            WHERE game_date = ?
            """,
            (game_date,),
        ):
            batter_id = str(batter_id)
            pitcher_id = str(pitcher_id)
            throws = str(throws or "R")
            is_k = int(is_k or 0)

            hand_state[
                (year, batter_id, throws)
            ][0] += 1
            hand_state[
                (year, batter_id, throws)
            ][1] += is_k

            season_state[
                (year, batter_id)
            ][0] += 1
            season_state[
                (year, batter_id)
            ][1] += is_k

            h2h_state[
                (batter_id, pitcher_id)
            ][0] += 1
            h2h_state[
                (batter_id, pitcher_id)
            ][1] += is_k

        if date_idx % 20 == 0 or date_idx == len(dates):
            print(
                f"Historical progress: "
                f"{date_idx}/{len(dates)} dates | "
                f"scored={historical_rows:,}",
                flush=True,
            )

    if batch:
        conn.executemany(insert_sql, batch)
        conn.commit()

    summary = {
        "rows": historical_rows,
        "yearly_scored": dict(sorted(yearly_scored.items())),
        "yearly_starters_seen": dict(sorted(yearly_starters.items())),
        "skipped_missing_prior_profile": skipped_profile,
        "skipped_missing_lineup": skipped_lineup,
        "simulation_mode": (
            "vectorized_distribution_equivalent_stable_seed_v1"
        ),
        "sims_per_row": SIMS,
        "primary_year": PRIMARY_YEAR,
        "warmup_year": WARMUP_YEAR,
    }

    completion.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


# =============================================================================
# Historical metrics / splits
# =============================================================================

def historical_metric_block(rows):
    rows = list(rows)

    if not rows:
        return {
            "rows": 0,
        }

    n = len(rows)

    actual = [
        float(row["actual_k"])
        for row in rows
    ]
    pred = [
        float(row["sim_mean"])
        for row in rows
    ]
    crps = [
        float(row["sim_crps"])
        for row in rows
    ]

    errors = [
        p - y
        for y, p in zip(actual, pred)
    ]

    abs_errors = [
        abs(e)
        for e in errors
    ]

    sq_errors = [
        e * e
        for e in errors
    ]

    return {
        "rows": n,
        "actual_mean_k": sum(actual) / n,
        "projected_mean_k": sum(pred) / n,
        "projection_bias": sum(errors) / n,
        "mae": sum(abs_errors) / n,
        "rmse": math.sqrt(sum(sq_errors) / n),
        "mean_crps": sum(crps) / n,
        "iqr_coverage": (
            sum(int(row["actual_in_iqr"]) for row in rows)
            / n
        ),
        "q10_q90_coverage": (
            sum(int(row["actual_in_10_90"]) for row in rows)
            / n
        ),
        "mean_expected_bf": (
            sum(float(row["expected_bf"]) for row in rows)
            / n
        ),
        "mean_actual_bf": (
            sum(float(row["actual_bf"]) for row in rows)
            / n
        ),
        "workload_bias_bf": (
            sum(
                float(row["expected_bf"])
                - float(row["actual_bf"])
                for row in rows
            )
            / n
        ),
        "lineup_component_used_rows": sum(
            int(row["lineup_component_used"])
            for row in rows
        ),
        "lineup_component_used_rate": (
            sum(
                int(row["lineup_component_used"])
                for row in rows
            )
            / n
        ),
        "bvp_nudge_active_rows": sum(
            1
            for row in rows
            if abs(float(row["k_nudge"]) - 1.0) > 1e-12
        ),
        "bvp_nudge_active_rate": (
            sum(
                1
                for row in rows
                if abs(float(row["k_nudge"]) - 1.0) > 1e-12
            )
            / n
        ),
    }


def fetch_historical_rows(conn, where="", params=()):
    sql = """
        SELECT
            year,
            game_date,
            game_id,
            pitcher_id,
            pitcher_throws,
            actual_k,
            actual_bf,

            prior_qualifying_outings,
            season_k_per_bf,
            recent_weighted_k_per_bf,
            final_k_per_bf,
            expected_bf,

            lineup_size,
            h2h_qualified_batters,
            lineup_component_used,
            lineup_avg_k_rate,
            lineup_expected_k,

            bvp_sample_pa,
            bvp_lineup_k_rate,
            k_nudge,

            blended_projection_pre_nudge,
            blended_k_per_bf_post_nudge,

            sim_mean,
            sim_median,
            sim_iqr,
            sim_q10,
            sim_q90,
            sim_crps,
            actual_in_iqr,
            actual_in_10_90
        FROM historical_rows
    """

    if where:
        sql += " WHERE " + where

    cols = [
        "year",
        "game_date",
        "game_id",
        "pitcher_id",
        "pitcher_throws",
        "actual_k",
        "actual_bf",

        "prior_qualifying_outings",
        "season_k_per_bf",
        "recent_weighted_k_per_bf",
        "final_k_per_bf",
        "expected_bf",

        "lineup_size",
        "h2h_qualified_batters",
        "lineup_component_used",
        "lineup_avg_k_rate",
        "lineup_expected_k",

        "bvp_sample_pa",
        "bvp_lineup_k_rate",
        "k_nudge",

        "blended_projection_pre_nudge",
        "blended_k_per_bf_post_nudge",

        "sim_mean",
        "sim_median",
        "sim_iqr",
        "sim_q10",
        "sim_q90",
        "sim_crps",
        "actual_in_iqr",
        "actual_in_10_90",
    ]

    return [
        dict(zip(cols, row))
        for row in conn.execute(sql, params)
    ]


def split_metric(rows, key_func):
    grouped = defaultdict(list)

    for row in rows:
        grouped[str(key_func(row))].append(row)

    return {
        key: historical_metric_block(grouped[key])
        for key in sorted(grouped)
    }


def outing_count_bucket(row):
    n = int(row["prior_qualifying_outings"])

    if n <= 4:
        return "3_4"
    if n <= 7:
        return "5_7"
    if n <= 11:
        return "8_11"
    return "12_plus"


def expected_bf_bucket(row):
    x = float(row["expected_bf"])

    if x < 18:
        return "lt_18"
    if x < 21:
        return "18_20.99"
    if x < 24:
        return "21_23.99"
    return "24_plus"


def nudge_bucket(row):
    x = float(row["k_nudge"])

    if x < 0.95:
        return "down_strong"
    if x < 0.9995:
        return "down_mild"
    if x <= 1.0005:
        return "neutral"
    if x <= 1.05:
        return "up_mild"
    return "up_strong"


def historical_summary(conn):
    all_rows = fetch_historical_rows(conn)

    warmup_rows = [
        row for row in all_rows
        if int(row["year"]) == WARMUP_YEAR
    ]

    primary_rows = [
        row for row in all_rows
        if int(row["year"]) == PRIMARY_YEAR
    ]

    primary_splits = {
        "by_month": split_metric(
            primary_rows,
            lambda r: str(r["game_date"])[:7],
        ),
        "by_lineup_component": split_metric(
            primary_rows,
            lambda r: (
                "lineup_used"
                if int(r["lineup_component_used"]) == 1
                else "pitcher_only"
            ),
        ),
        "by_bvp_nudge": split_metric(
            primary_rows,
            nudge_bucket,
        ),
        "by_prior_outing_count": split_metric(
            primary_rows,
            outing_count_bucket,
        ),
        "by_expected_bf": split_metric(
            primary_rows,
            expected_bf_bucket,
        ),
        "by_pitcher_hand": split_metric(
            primary_rows,
            lambda r: r["pitcher_throws"] or "unknown",
        ),
    }

    return {
        "all_2024_2025": historical_metric_block(all_rows),
        "warmup_2024": historical_metric_block(warmup_rows),
        "primary_2025": historical_metric_block(primary_rows),
        "primary_2025_splits": primary_splits,
    }


# =============================================================================
# Exact recent real-line audit
# =============================================================================

def load_record_actual_map():
    payload = safe_json_load(RECORD_JSON)

    by_id = {}
    by_name = {}

    if not isinstance(payload, dict):
        return by_id, by_name

    rows = payload.get("results", [])

    if not isinstance(rows, list):
        return by_id, by_name

    for row in rows:
        if not isinstance(row, dict):
            continue

        if row.get("prop_type") != "pitcher_strikeouts":
            continue

        actual = safe_float(row.get("actual"))

        if actual is None:
            continue

        date = str(row.get("date") or "")
        game_id = str(row.get("game_id") or "")
        player_id = row.get("player_id")
        player_name = norm_name(row.get("player"))

        if date and game_id and player_id is not None:
            by_id[
                (date, game_id, str(player_id))
            ] = float(actual)

        if date and game_id and player_name:
            by_name[
                (date, game_id, player_name)
            ] = float(actual)

    return by_id, by_name


def local_actual_k_map(conn):
    out = {}

    for (
        game_date,
        game_id,
        pitcher_id,
        so,
    ) in conn.execute(
        """
        SELECT
            game_date,
            game_id,
            pitcher_id,
            so
        FROM pitcher_game
        """
    ):
        out[
            (
                str(game_date),
                str(game_id),
                str(pitcher_id),
            )
        ] = float(so)

    return out


def line_from_candidate(row):
    for key in (
        "k_line",
        "pick_line",
    ):
        value = safe_float(row.get(key))

        if value is not None:
            return value

    pick = str(row.get("pick") or "")
    m = re.search(
        r"(?:OVER|UNDER)\s+(-?\d+(?:\.\d+)?)",
        pick,
        re.I,
    )

    if m:
        return safe_float(m.group(1))

    return None


def side_from_candidate(row):
    side = str(row.get("pick_side") or "").upper().strip()

    if side in {"OVER", "UNDER"}:
        return side

    pick = str(row.get("pick") or "").upper().strip()

    if pick.startswith("OVER"):
        return "OVER"

    if pick.startswith("UNDER"):
        return "UNDER"

    ksim = row.get("ksim")

    if isinstance(ksim, dict):
        side = str(ksim.get("side") or "").upper().strip()

        if side in {"OVER", "UNDER"}:
            return side

    return None


def candidate_probability(row):
    p = safe_float(row.get("model_prob"))

    if p is not None:
        return p

    ksim = row.get("ksim")

    if isinstance(ksim, dict):
        return safe_float(ksim.get("side_prob"))

    return None


def exact_line_metric_block(rows):
    rows = list(rows)

    resolved = [
        row
        for row in rows
        if row.get("result") in {"hit", "miss"}
        and row.get("probability") is not None
    ]

    pushes = sum(
        1
        for row in rows
        if row.get("result") == "push"
    )

    if not resolved:
        return {
            "rows": len(rows),
            "resolved_rows": 0,
            "pushes": pushes,
        }

    n = len(resolved)

    hits = sum(
        1
        for row in resolved
        if row["result"] == "hit"
    )

    brier = sum(
        (
            float(row["probability"])
            - (1.0 if row["result"] == "hit" else 0.0)
        ) ** 2
        for row in resolved
    ) / n

    logloss = sum(
        -(
            (1.0 if row["result"] == "hit" else 0.0)
            * math.log(
                clip(float(row["probability"]), 1e-8, 1.0 - 1e-8)
            )
            + (0.0 if row["result"] == "hit" else 1.0)
            * math.log(
                1.0
                - clip(float(row["probability"]), 1e-8, 1.0 - 1e-8)
            )
        )
        for row in resolved
    ) / n

    return {
        "rows": len(rows),
        "resolved_rows": n,
        "pushes": pushes,
        "hits": hits,
        "misses": n - hits,
        "hit_rate": hits / n,
        "brier": brier,
        "logloss": logloss,
        "mean_probability": (
            sum(float(row["probability"]) for row in resolved)
            / n
        ),
        "calibration_gap_hit_rate_minus_mean_prob": (
            hits / n
            - sum(float(row["probability"]) for row in resolved) / n
        ),
    }


def exact_line_split(rows, key_func):
    groups = defaultdict(list)

    for row in rows:
        groups[str(key_func(row))].append(row)

    return {
        key: exact_line_metric_block(groups[key])
        for key in sorted(groups)
    }


def audit_recent_real_lines(conn):
    local_actual = local_actual_k_map(conn)
    record_by_id, record_by_name = load_record_actual_map()

    files = sorted(
        PRED_DIR.glob("pitcher_k_candidates_*.json")
    )

    audited = []
    all_candidate_rows = 0
    line_rows = 0
    actual_resolved = 0

    for path in files:
        payload = safe_json_load(path)

        if not isinstance(payload, list):
            continue

        date_match = re.search(
            r"(\d{4}-\d{2}-\d{2})",
            path.name,
        )

        file_date = (
            date_match.group(1)
            if date_match
            else None
        )

        for row in payload:
            if not isinstance(row, dict):
                continue

            all_candidate_rows += 1

            line = line_from_candidate(row)

            has_line = bool(
                row.get("has_k_line")
                or row.get("has_line")
                or line is not None
            )

            if not has_line or line is None:
                continue

            line_rows += 1

            game_date = str(
                row.get("date")
                or file_date
                or ""
            )
            game_id = str(row.get("game_id") or "")
            pitcher_id = row.get("player_id")
            player_name = norm_name(row.get("player"))

            actual = None
            actual_source = None

            if pitcher_id is not None:
                actual = local_actual.get(
                    (
                        game_date,
                        game_id,
                        str(pitcher_id),
                    )
                )

                if actual is not None:
                    actual_source = "local_statcast"

            if actual is None and pitcher_id is not None:
                actual = record_by_id.get(
                    (
                        game_date,
                        game_id,
                        str(pitcher_id),
                    )
                )

                if actual is not None:
                    actual_source = "record_json_player_id"

            if actual is None and player_name:
                actual = record_by_name.get(
                    (
                        game_date,
                        game_id,
                        player_name,
                    )
                )

                if actual is not None:
                    actual_source = "record_json_player_name"

            side = side_from_candidate(row)
            probability = candidate_probability(row)

            if (
                actual is not None
                and side in {"OVER", "UNDER"}
            ):
                actual_resolved += 1

                if float(actual) == float(line):
                    result = "push"
                elif side == "OVER":
                    result = (
                        "hit"
                        if float(actual) > float(line)
                        else "miss"
                    )
                else:
                    result = (
                        "hit"
                        if float(actual) < float(line)
                        else "miss"
                    )
            else:
                result = "unresolved"

            warnings = row.get("legacy_k_warnings")

            if not isinstance(warnings, list):
                warnings = []

            board_status = str(
                row.get("board_status")
                or row.get("candidate_status")
                or "unknown"
            )

            lineup_flag = bool(
                row.get("k_has_lineup_kr")
                or (
                    isinstance(row.get("bvp_flag"), str)
                    and row.get("bvp_flag").startswith("lineup_kr")
                )
            )

            audited.append(
                {
                    "date": game_date,
                    "game_id": game_id,
                    "pitcher_id": (
                        str(pitcher_id)
                        if pitcher_id is not None
                        else None
                    ),
                    "player": row.get("player"),
                    "line": float(line),
                    "integer_line": (
                        abs(float(line) - round(float(line))) < 1e-12
                    ),
                    "side": side,
                    "probability": probability,
                    "actual_k": actual,
                    "actual_source": actual_source,
                    "result": result,
                    "board_status": board_status,
                    "confidence": row.get("confidence") or "unknown",
                    "lineup_flag": lineup_flag,
                    "legacy_warnings_count": len(warnings),
                    "legacy_warnings": warnings,
                    "k_nudge": safe_float(row.get("k_nudge")),
                    "projection_gap": safe_float(
                        row.get("k_projection_gap")
                    ),
                    "generated_at": row.get("generated_at"),
                    "source_file": str(path),
                }
            )

    resolved_rows = [
        row
        for row in audited
        if row["result"] in {"hit", "miss", "push"}
    ]

    summary = {
        "candidate_files": len(files),
        "all_candidate_rows": all_candidate_rows,
        "rows_with_real_line": line_rows,
        "rows_with_actual_resolved": len(resolved_rows),
        "rows_with_actual_hit_or_miss": sum(
            1
            for row in resolved_rows
            if row["result"] in {"hit", "miss"}
        ),
        "overall": exact_line_metric_block(resolved_rows),
        "by_side": exact_line_split(
            resolved_rows,
            lambda r: r["side"] or "unknown",
        ),
        "by_board_status": exact_line_split(
            resolved_rows,
            lambda r: r["board_status"],
        ),
        "by_confidence": exact_line_split(
            resolved_rows,
            lambda r: r["confidence"],
        ),
        "by_lineup_flag": exact_line_split(
            resolved_rows,
            lambda r: (
                "with_lineup_kr"
                if r["lineup_flag"]
                else "no_lineup_kr"
            ),
        ),
        "by_legacy_warning_presence": exact_line_split(
            resolved_rows,
            lambda r: (
                "with_legacy_warning"
                if r["legacy_warnings_count"] > 0
                else "no_legacy_warning"
            ),
        ),
        "by_integer_vs_half_line": exact_line_split(
            resolved_rows,
            lambda r: (
                "integer_line"
                if r["integer_line"]
                else "non_integer_line"
            ),
        ),
        "actual_source_counts": dict(
            sorted(
                (
                    source,
                    sum(
                        1
                        for row in resolved_rows
                        if row["actual_source"] == source
                    ),
                )
                for source in {
                    row["actual_source"]
                    for row in resolved_rows
                    if row["actual_source"]
                }
            )
        ),
        "sample_date_min": min(
            (
                row["date"]
                for row in resolved_rows
                if row["date"]
            ),
            default=None,
        ),
        "sample_date_max": max(
            (
                row["date"]
                for row in resolved_rows
                if row["date"]
            ),
            default=None,
        ),
        "sample_warning": (
            "This exact sportsbook-line audit uses only archived real lines. "
            "It is recent and must not be represented as a multi-season exact-"
            "line backtest."
        ),
    }

    return summary, audited


# =============================================================================
# Source-code architecture sanity checks
# =============================================================================

def source_sanity_checks():
    checks = {}

    source_text = {}

    for path in SOURCE_FILES:
        if path.exists():
            source_text[path.name] = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        else:
            source_text[path.name] = ""

    api = source_text.get("api.py", "")
    ksim = source_text.get("ksim.py", "")
    lineupk = source_text.get("lineupk.py", "")
    bvp = source_text.get("bvp.py", "")

    checks["api_recency_decay_0_6"] = (
        "RECENCY_DECAY = 0.6" in api
    )
    checks["api_season_anchor_0_15"] = (
        "SEASON_ANCHOR = 0.15" in api
    )
    checks["api_pitcher_weight_0_55"] = (
        "PITCHER_WEIGHT = 0.55" in api
    )
    checks["api_lineup_weight_0_45"] = (
        "LINEUP_WEIGHT = 0.45" in api
    )
    checks["api_official_min_0_63"] = (
        "K_MC_OFFICIAL_MIN_PROB = 0.63" in api
    )
    checks["api_lean_min_0_60"] = (
        "K_MC_LEAN_MIN_PROB = 0.60" in api
    )
    checks["ksim_unseeded_random_state"] = (
        "np.random.RandomState()" in ksim
    )
    checks["ksim_tto_0_94"] = (
        "decay = 0.94" in ksim
    )
    checks["ksim_tto_0_85"] = (
        "decay = 0.85" in ksim
    )
    checks["lineupk_h2h_weight_0_40"] = (
        "H2H_WEIGHT = 0.40" in lineupk
    )
    checks["lineupk_gen_weight_0_60"] = (
        "GEN_WEIGHT = 0.60" in lineupk
    )
    checks["lineupk_min_h2h_pa_2"] = (
        "MIN_H2H_PA = 2" in lineupk
    )
    checks["bvp_nudge_cap_0_85_1_15"] = (
        "max(0.85, min(1.15, raw))" in bvp
    )

    return checks


# =============================================================================
# Report
# =============================================================================

def fmt_metric_block(label, block):
    if not block or int(block.get("rows", 0)) == 0:
        return f"{label}: rows=0"

    return (
        f"{label}: "
        f"n={block['rows']:,} "
        f"actual_mean={block['actual_mean_k']:.4f} "
        f"projected_mean={block['projected_mean_k']:.4f} "
        f"bias={block['projection_bias']:+.4f} "
        f"mae={block['mae']:.4f} "
        f"rmse={block['rmse']:.4f} "
        f"crps={block['mean_crps']:.4f} "
        f"iqr_coverage={block['iqr_coverage']:.4f} "
        f"q10_q90_coverage={block['q10_q90_coverage']:.4f} "
        f"workload_bias_bf={block['workload_bias_bf']:+.4f}"
    )


def fmt_exact_block(label, block):
    if not block:
        return f"{label}: unavailable"

    return (
        f"{label}: "
        f"rows={block.get('rows', 0):,} "
        f"resolved={block.get('resolved_rows', 0):,} "
        f"pushes={block.get('pushes', 0):,} "
        f"hits={block.get('hits', 0):,} "
        f"misses={block.get('misses', 0):,} "
        f"hit_rate={block.get('hit_rate')} "
        f"brier={block.get('brier')} "
        f"logloss={block.get('logloss')} "
        f"mean_prob={block.get('mean_probability')}"
    )


def build_report_text(payload):
    lines = []

    lines.append("PITCHER_K_CLEAN_BASELINE_A")
    lines.append("=" * 26)
    lines.append("")

    lines.append("SOURCE PROVENANCE")
    lines.append("-----------------")

    for row in payload["source_provenance"]:
        lines.append(
            f"{row['path']}: "
            f"exists={row['exists']} "
            f"sha256={row.get('sha256')}"
        )

    lines.append("")
    lines.append("LOCAL DATA COVERAGE")
    lines.append("-------------------")
    lines.append(
        json.dumps(
            payload["local_data_coverage"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("SOURCE ARCHITECTURE SANITY CHECKS")
    lines.append("---------------------------------")

    for key, value in payload[
        "source_architecture_sanity_checks"
    ].items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("HISTORICAL RECONSTRUCTION DISCLOSURES")
    lines.append("-------------------------------------")

    for disclosure in payload[
        "historical_reconstruction_disclosures"
    ]:
        lines.append(f"- {disclosure}")

    lines.append("")
    lines.append("KNOWN ARCHITECTURE FINDINGS")
    lines.append("---------------------------")

    for key, value in payload[
        "known_architecture_findings"
    ].items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("2024 WARM-UP DIAGNOSTIC")
    lines.append("-----------------------")
    lines.append(
        fmt_metric_block(
            "2024",
            payload["historical_metrics"]["warmup_2024"],
        )
    )

    lines.append("")
    lines.append("2025 PRIMARY FORWARD BASELINE")
    lines.append("-----------------------------")
    lines.append(
        fmt_metric_block(
            "2025",
            payload["historical_metrics"]["primary_2025"],
        )
    )

    lines.append("")
    lines.append("ARCHITECTURE SPLITS")
    lines.append("-------------------")
    lines.append(
        json.dumps(
            payload["historical_metrics"][
                "primary_2025_splits"
            ],
            indent=2,
        )
    )

    lines.append("")
    lines.append("EXACT RECENT REAL-LINE AUDIT")
    lines.append("----------------------------")
    exact = payload["exact_recent_real_line_audit"]

    lines.append(
        f"candidate_files={exact['candidate_files']}"
    )
    lines.append(
        f"all_candidate_rows={exact['all_candidate_rows']}"
    )
    lines.append(
        f"rows_with_real_line={exact['rows_with_real_line']}"
    )
    lines.append(
        f"rows_with_actual_resolved="
        f"{exact['rows_with_actual_resolved']}"
    )
    lines.append(
        f"sample_date_range="
        f"{exact['sample_date_min']}..{exact['sample_date_max']}"
    )
    lines.append(
        fmt_exact_block(
            "OVERALL",
            exact["overall"],
        )
    )

    lines.append("")
    lines.append("BY SIDE")
    lines.append(
        json.dumps(
            exact["by_side"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("BY BOARD STATUS")
    lines.append(
        json.dumps(
            exact["by_board_status"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("BY LINEUP FLAG")
    lines.append(
        json.dumps(
            exact["by_lineup_flag"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("BY LEGACY WARNING")
    lines.append(
        json.dumps(
            exact["by_legacy_warning_presence"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("INTEGER VS NON-INTEGER LINE")
    lines.append(
        json.dumps(
            exact["by_integer_vs_half_line"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("BASELINE STATUS")
    lines.append("---------------")
    lines.append(
        json.dumps(
            payload["baseline_status"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("FINAL VERDICT")
    lines.append("-------------")
    lines.append(payload["final_verdict"])

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    HR_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print("PITCHER_K_CLEAN_BASELINE_A", flush=True)
    print("==========================", flush=True)

    print("")
    print("SOURCE PROVENANCE", flush=True)
    print("-----------------", flush=True)

    provenance = source_provenance()

    for row in provenance:
        print(
            f"{row['path']}: "
            f"exists={row['exists']} "
            f"sha256={row.get('sha256')}",
            flush=True,
        )

    if not all(row["exists"] for row in provenance):
        raise RuntimeError(
            "Missing one or more incumbent source files."
        )

    statcast_info = source_statcast_info()

    print("")
    print("LOCAL DATA COVERAGE", flush=True)
    print("-------------------", flush=True)
    print(
        json.dumps(statcast_info, indent=2),
        flush=True,
    )

    sanity = source_sanity_checks()

    print("")
    print("SOURCE ARCHITECTURE SANITY CHECKS", flush=True)
    print("---------------------------------", flush=True)

    for key, value in sanity.items():
        print(f"{key}: {value}", flush=True)

    if not all(sanity.values()):
        failed = [
            key
            for key, value in sanity.items()
            if not value
        ]

        raise RuntimeError(
            "Incumbent source sanity checks failed: "
            + ", ".join(failed)
        )

    conn = connect_work_db()

    try:
        base_tables = ensure_base_tables(conn)

        print("")
        print("HISTORICAL RECONSTRUCTION DISCLOSURES", flush=True)
        print("-------------------------------------", flush=True)

        disclosures = [
            (
                "Primary broad forward baseline is 2025. "
                "2024 is warm-up diagnostic because local H2H history begins "
                "in 2024 and cannot reproduce pre-2024 career BvP without "
                "leaking future/current API totals."
            ),
            (
                "Historical lineups are recovered retrospectively as the first "
                "nine unique batters faced by the starter. Batter identity is "
                "pregame-knowable when confirmed, but the recovery method uses "
                "completed-game local Statcast."
            ),
            (
                "Starting pitchers are recovered as the two earliest distinct "
                "pitchers to appear in each game."
            ),
            (
                "All batter/pitcher profile features are D-1. Same-day history "
                "is excluded from every game scored on that date."
            ),
            (
                "Current n_data semantics are preserved exactly: only H2H-"
                "qualified batters increment n_data; lineup component requires "
                "n_data >= 5."
            ),
            (
                "Historical simulation uses a stable deterministic research "
                "seed and vectorized binomial draws. Distributional logic is "
                "preserved; random stream is not bitwise-identical to the "
                "production per-PA loop."
            ),
            (
                "Exact recent sportsbook audit uses only saved real lines and "
                "is not represented as a multi-season exact-line backtest."
            ),
        ]

        for item in disclosures:
            print(f"- {item}", flush=True)

        print("")
        print("BUILD HISTORICAL INCUMBENT BASELINE", flush=True)
        print("-----------------------------------", flush=True)

        reconstruction = reconstruct_historical_baseline(conn)

        print(
            json.dumps(reconstruction, indent=2),
            flush=True,
        )

        metrics = historical_summary(conn)

        print("")
        print("2024 WARM-UP DIAGNOSTIC", flush=True)
        print("-----------------------", flush=True)
        print(
            fmt_metric_block(
                "2024",
                metrics["warmup_2024"],
            ),
            flush=True,
        )

        print("")
        print("2025 PRIMARY FORWARD BASELINE", flush=True)
        print("-----------------------------", flush=True)
        print(
            fmt_metric_block(
                "2025",
                metrics["primary_2025"],
            ),
            flush=True,
        )

        print("")
        print("ARCHITECTURE SPLITS", flush=True)
        print("-------------------", flush=True)
        print(
            json.dumps(
                metrics["primary_2025_splits"],
                indent=2,
            ),
            flush=True,
        )

        print("")
        print("EXACT RECENT REAL-LINE AUDIT", flush=True)
        print("----------------------------", flush=True)

        exact_summary, exact_rows = audit_recent_real_lines(conn)

        print(
            f"candidate_files={exact_summary['candidate_files']}",
            flush=True,
        )
        print(
            f"all_candidate_rows="
            f"{exact_summary['all_candidate_rows']}",
            flush=True,
        )
        print(
            f"rows_with_real_line="
            f"{exact_summary['rows_with_real_line']}",
            flush=True,
        )
        print(
            f"rows_with_actual_resolved="
            f"{exact_summary['rows_with_actual_resolved']}",
            flush=True,
        )
        print(
            f"sample_date_range="
            f"{exact_summary['sample_date_min']}.."
            f"{exact_summary['sample_date_max']}",
            flush=True,
        )
        print(
            fmt_exact_block(
                "OVERALL",
                exact_summary["overall"],
            ),
            flush=True,
        )

        historical_primary_n = int(
            metrics["primary_2025"].get("rows", 0)
        )
        exact_resolved_n = int(
            exact_summary.get(
                "rows_with_actual_hit_or_miss",
                0,
            )
        )

        baseline_status = {
            "historical_primary_year": PRIMARY_YEAR,
            "historical_primary_rows": historical_primary_n,
            "historical_primary_eligible": (
                historical_primary_n >= 1000
            ),
            "exact_recent_real_line_rows_resolved": (
                exact_resolved_n
            ),
            "exact_recent_real_line_sample_is_recent_only": True,
            "production_changed": False,
            "external_api_calls": False,
            "hits_prospective_holdout_touched": False,
            "challenger_tested": False,
            "threshold_tuning": False,
            "rescue_tuning": False,
        }

        if historical_primary_n >= 1000:
            final_verdict = (
                "PITCHER_K_INCUMBENT_CLEAN_LOCAL_HISTORY_BASELINE_COMPLETE_"
                "READY_FOR_ARCHITECTURE_DIAGNOSIS_NO_CHALLENGER_TESTED"
            )
        else:
            final_verdict = (
                "PITCHER_K_INCUMBENT_BASELINE_BUILT_BUT_PRIMARY_SAMPLE_TOO_"
                "SMALL_FOR_FORMAL_ARCHITECTURE_TESTING"
            )

        payload = {
            "script": "PITCHER_K_CLEAN_BASELINE_A",
            "generated_at_utc": now_utc(),
            "mode": "READ_ONLY_LOCAL_DATA_ONLY",
            "source_provenance": provenance,
            "source_architecture_sanity_checks": sanity,
            "known_architecture_findings": KNOWN_FINDINGS,
            "local_data_coverage": {
                "statcast": statcast_info,
                "base_tables": base_tables,
            },
            "historical_reconstruction_disclosures": disclosures,
            "frozen_incumbent_parameters": {
                "league_avg_k": LEAGUE_AVG_K,
                "recency_decay": RECENCY_DECAY,
                "season_anchor": SEASON_ANCHOR,
                "pitcher_weight": PITCHER_WEIGHT,
                "lineup_weight": LINEUP_WEIGHT,
                "h2h_weight": H2H_WEIGHT,
                "gen_weight": GEN_WEIGHT,
                "min_h2h_pa": MIN_H2H_PA,
                "min_general_pa": MIN_GENERAL_PA,
                "bvp_min_sample_pa": BVP_MIN_SAMPLE_PA,
                "k_nudge_min": K_NUDGE_MIN,
                "k_nudge_max": K_NUDGE_MAX,
                "min_qualifying_bf": MIN_QUALIFYING_BF,
                "min_prior_qualifying_outings": (
                    MIN_PRIOR_QUALIFYING_OUTINGS
                ),
                "recent_bf_window": RECENT_BF_WINDOW,
                "volatility_rate_window": VOLATILITY_RATE_WINDOW,
                "sims": SIMS,
                "bf_sd": BF_SD,
                "bf_min": BF_MIN,
                "bf_max": BF_MAX,
                "tto_factors": [
                    TTO_1,
                    TTO_2,
                    TTO_3,
                ],
                "ksim_no_bet_min": KSIM_NO_BET_MIN,
                "watchlist_min": WATCHLIST_MIN,
                "official_min": OFFICIAL_MIN,
            },
            "reconstruction_summary": reconstruction,
            "historical_metrics": metrics,
            "exact_recent_real_line_audit": exact_summary,
            "baseline_status": baseline_status,
            "final_verdict": final_verdict,
        }

        OUT_JSON.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        report = build_report_text(payload)

        OUT_TXT.write_text(
            report,
            encoding="utf-8",
        )

        print("")
        print("BASELINE STATUS", flush=True)
        print("---------------", flush=True)
        print(
            json.dumps(baseline_status, indent=2),
            flush=True,
        )

        print("")
        print("FINAL VERDICT", flush=True)
        print("-------------", flush=True)
        print(final_verdict, flush=True)

        print("")
        print("OUTPUTS", flush=True)
        print(OUT_JSON, flush=True)
        print(OUT_TXT, flush=True)
        print(WORK_DB, flush=True)

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
