#!/usr/bin/env python3
"""
PITCHER_K_D0_2026_FORMAL_GATE_A

ONE-SHOT FORMAL OUT-OF-TIME HOLDOUT GATE
========================================

Frozen challenger
-----------------
D0_FIXED_LINEUP_ACTIVATION

Comparator
----------
A0_CURRENT_INCUMBENT

Exact isolated change
---------------------
A0:
    Activates the 45% lineup component only when at least 5 opposing batters
    have qualifying H2H strikeout history against this exact pitcher.

D0:
    Activates the same 45% lineup component when at least 5 opposing batters
    have genuinely usable K information from any of:
        - handedness split with >= 15 prior PA
        - current-season overall K rate with >= 15 prior PA
        - H2H strikeout history with >= 2 prior PA

Everything else remains frozen and identical:
    - pitcher profile
    - qualifying outing BF >= 12
    - minimum 3 prior qualifying outings
    - 85% recent / 15% season anchor
    - expected BF = last-5 qualifying outing mean
    - last-12 outing volatility pool
    - 55% pitcher / 45% lineup blend
    - 40% H2H / 60% general batter blend
    - BVP k_nudge
    - TTO decay 1.00 / 0.94 / 0.85
    - 10,000 simulation distribution
    - no decision-threshold changes

Holdout doctrine
----------------
2024:
    history warm-up only

2025:
    development + stability confirmation only
    D0 was selected on 2025 and therefore 2025 is NOT a formal holdout

2026:
    one-shot formal out-of-time holdout for D0

No other K challenger is opened on this holdout.
No BVP rescue.
No TTO rescue.
No stacking.
No threshold tuning.
No post-result rescue tuning.
No production promotion from this script alone.

Source
------
/data/hr_model/pitcher_k_clean_baseline_a_work/baseline.sqlite

The baseline DB already contains strict local D-1 tables built from Statcast:
    pa
    pitcher_game
    starter_games

This script reconstructs 2024 -> 2025 -> 2026 history sequentially and scores
ONLY 2026 starter rows before same-day state updates.

Formal gate is frozen before 2026 metrics are calculated
--------------------------------------------------------
A D0 formal pass requires ALL conditions:

1. Paired-row integrity passes.
2. At least 1,000 eligible 2026 paired starter rows.
3. D0 improves overall CRPS by at least 0.002 absolute.
4. D0 improves overall CRPS by at least 0.25% relative.
5. Date-block bootstrap P(D0 CRPS < A0 CRPS) >= 95%.
6. 2026 MAE does not worsen.
7. 2026 RMSE does not worsen.
8. 2026 absolute projection bias does not worsen.
9. At least 3 eligible months exist, where eligible month >= 200 rows.
10. D0 wins CRPS in at least half of eligible months.
11. Exact intervention subset has at least 300 rows.
12. Intervention subset CRPS improves by at least 0.25% relative.
13. Intervention subset MAE does not worsen.
14. At least 10 eligible predeclared slice buckets exist.
15. At least 60% of eligible slice buckets are CRPS nonworse.
16. No eligible slice worsens CRPS by more than 5% relative to A0.

The exact intervention subset is:
    A0 lineup component OFF
    D0 lineup component ON

Bootstrap
---------
5,000 paired date-block bootstrap replicates.
Dates, not individual rows, are resampled.
Deterministic bootstrap seed is frozen in this script.

Formal pass status
------------------
A pass creates:
    FORMAL_VALIDATED_CHALLENGER_PENDING_IMPLEMENTATION_PARITY_AUDIT

A pass does NOT automatically authorize production promotion.

A fail creates:
    D0_FIXED_LINEUP_ACTIVATION_FAILS_2026_FORMAL_GATE
    _FREEZE_NO_RESCUE_TUNING

One-shot protection
-------------------
If a completed formal result already exists, this script refuses to rerun.
Use:
    python -u pitcher_k_d0_2026_formal_gate_a.py --verify-existing

to verify and print the existing verdict without reopening the holdout.

Memory safety
-------------
- no pandas
- no sklearn
- SQLite on disk
- row-at-a-time simulation
- paired common random numbers
- 512 MB Render target

Run
---
python -u pitcher_k_d0_2026_formal_gate_a.py 2>&1 | tee /data/hr_model/pitcher_k_d0_2026_formal_gate_a.log

Outputs
-------
/data/hr_model/pitcher_k_d0_2026_formal_gate_a_results.json
/data/hr_model/pitcher_k_d0_2026_formal_gate_a_report.txt
/data/hr_model/pitcher_k_d0_2026_formal_gate_a_work/formal_rows.sqlite
/data/hr_model/pitcher_k_d0_2026_formal_gate_a_work/gate_manifest.json
/data/hr_model/pitcher_k_d0_2026_formal_gate_a_work/gate_manifest.sha256

Paste back
----------
FORMAL GATE PREFLIGHT
IMMUTABLE GATE FREEZE
2026 HOLDOUT SOURCE COVERAGE
BUILD CLEAN 2026 PAIRED ROWS
2026 PRIMARY FORMAL COMPARISON
DATE-BLOCK BOOTSTRAP
INTERVENTION SUBSET
MONTHLY STABILITY
SLICE BREADTH
STRICT FORMAL GATE
FINAL FORMAL VERDICT
"""

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# =============================================================================
# Paths
# =============================================================================

HR_DIR = Path("/data/hr_model")

BASELINE_DB = (
    HR_DIR
    / "pitcher_k_clean_baseline_a_work"
    / "baseline.sqlite"
)

ARCH_LAB_RESULTS = (
    HR_DIR
    / "pitcher_k_architecture_lab_v1_results.json"
)

STABILITY_RESULTS = (
    HR_DIR
    / "pitcher_k_d0_stability_confirmation_a_results.json"
)

WORK_DIR = (
    HR_DIR
    / "pitcher_k_d0_2026_formal_gate_a_work"
)

FORMAL_DB = WORK_DIR / "formal_rows.sqlite"

GATE_MANIFEST = WORK_DIR / "gate_manifest.json"
GATE_MANIFEST_SHA = WORK_DIR / "gate_manifest.sha256"

OUT_JSON = (
    HR_DIR
    / "pitcher_k_d0_2026_formal_gate_a_results.json"
)

OUT_TXT = (
    HR_DIR
    / "pitcher_k_d0_2026_formal_gate_a_report.txt"
)


# =============================================================================
# Frozen architecture constants
# =============================================================================

A0_CELL = "A0_CURRENT_INCUMBENT"
D0_CELL = "D0_FIXED_LINEUP_ACTIVATION"

HOLDOUT_YEAR = 2026
HISTORY_START_YEAR = 2024

LEAGUE_AVG_K = 0.22

RECENCY_DECAY = 0.6
SEASON_ANCHOR = 0.15

PITCHER_WEIGHT = 0.55
LINEUP_WEIGHT = 0.45

H2H_WEIGHT = 0.40
GEN_WEIGHT = 0.60

MIN_H2H_PA = 2
MIN_GENERAL_PA = 15
MIN_LINEUP_DATA_BATTERS = 5

BVP_MIN_SAMPLE_PA = 20
K_NUDGE_MIN = 0.85
K_NUDGE_MAX = 1.15

MIN_QUALIFYING_BF = 12
MIN_PRIOR_QUALIFYING_OUTINGS = 3
RECENT_BF_WINDOW = 5
VOLATILITY_RATE_WINDOW = 12

BF_SD = 2.5
BF_MIN = 9
BF_MAX = 30

TTO_FACTORS = (1.00, 0.94, 0.85)

SIMS = 10000


# =============================================================================
# Frozen formal gate
# =============================================================================

MIN_FORMAL_ROWS = 1000

MIN_ABSOLUTE_CRPS_IMPROVEMENT = 0.0020
MIN_RELATIVE_CRPS_IMPROVEMENT_PCT = 0.25

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260711
BOOTSTRAP_MIN_P_IMPROVE = 0.95

ELIGIBLE_MONTH_MIN_ROWS = 200
MIN_ELIGIBLE_MONTHS = 3

MIN_INTERVENTION_ROWS = 300
MIN_INTERVENTION_RELATIVE_CRPS_IMPROVEMENT_PCT = 0.25

ELIGIBLE_SLICE_MIN_ROWS = 200
MIN_ELIGIBLE_SLICE_BUCKETS = 10
MIN_SLICE_NONWORSE_FRACTION = 0.60
MAX_ALLOWED_RELATIVE_CRPS_WORSENING_PCT = 5.0

TIE_EPS = 1e-12


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
    digest = hashlib.sha256(
        raw.encode("utf-8", "ignore")
    ).hexdigest()[:16]

    return int(digest, 16) % (2 ** 32)


def safe_json_load(path):
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8")
        )
    except Exception:
        return None


def clip(value, lo, hi):
    return max(lo, min(hi, value))


def connect_ro(path):
    if not path.exists():
        raise RuntimeError(
            f"Missing required file: {path}"
        )

    conn = sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True,
        timeout=60,
    )

    conn.execute("PRAGMA busy_timeout=60000")

    return conn


def connect_formal():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(FORMAL_DB),
        timeout=60,
    )

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-24000")
    conn.execute("PRAGMA busy_timeout=60000")

    return conn


def check_required_tables(conn):
    required = {
        "pa",
        "pitcher_game",
        "starter_games",
    }

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }

    missing = sorted(required - tables)

    if missing:
        raise RuntimeError(
            "Baseline DB missing required tables: "
            + ", ".join(missing)
        )

    return sorted(tables)


# =============================================================================
# Prior-registry checks
# =============================================================================

def verify_prior_registry():
    arch = safe_json_load(ARCH_LAB_RESULTS)
    stability = safe_json_load(STABILITY_RESULTS)

    if not isinstance(arch, dict):
        raise RuntimeError(
            f"Missing or unreadable architecture result: {ARCH_LAB_RESULTS}"
        )

    if not isinstance(stability, dict):
        raise RuntimeError(
            f"Missing or unreadable stability result: {STABILITY_RESULTS}"
        )

    arch_verdict = (
        arch
        .get("results", {})
        .get("final_verdict")
    )

    stability_verdict = stability.get("final_verdict")

    expected_arch_token = (
        "DEVELOPMENT_SURVIVOR_FOUND_"
        "D0_FIXED_LINEUP_ACTIVATION"
    )

    expected_stability_token = (
        "D0_FIXED_LINEUP_ACTIVATION_BROADLY_STABLE_"
        "DEVELOPMENT_SURVIVOR"
    )

    if (
        not isinstance(arch_verdict, str)
        or expected_arch_token not in arch_verdict
    ):
        raise RuntimeError(
            "Architecture lab does not show D0 as the development survivor."
        )

    if (
        not isinstance(stability_verdict, str)
        or expected_stability_token not in stability_verdict
    ):
        raise RuntimeError(
            "Stability confirmation does not show D0 as broadly stable."
        )

    return {
        "architecture_lab_verdict": arch_verdict,
        "stability_verdict": stability_verdict,
        "architecture_result_sha256": sha256_file(
            ARCH_LAB_RESULTS
        ),
        "stability_result_sha256": sha256_file(
            STABILITY_RESULTS
        ),
    }


# =============================================================================
# Gate manifest
# =============================================================================

def gate_definition():
    return {
        "candidate": D0_CELL,
        "comparator": A0_CELL,
        "holdout_year": HOLDOUT_YEAR,

        "frozen_architecture": {
            "league_avg_k": LEAGUE_AVG_K,
            "recency_decay": RECENCY_DECAY,
            "season_anchor": SEASON_ANCHOR,
            "pitcher_weight": PITCHER_WEIGHT,
            "lineup_weight": LINEUP_WEIGHT,
            "h2h_weight": H2H_WEIGHT,
            "general_weight": GEN_WEIGHT,
            "min_h2h_pa": MIN_H2H_PA,
            "min_general_pa": MIN_GENERAL_PA,
            "min_lineup_data_batters": (
                MIN_LINEUP_DATA_BATTERS
            ),
            "bvp_min_sample_pa": BVP_MIN_SAMPLE_PA,
            "k_nudge_min": K_NUDGE_MIN,
            "k_nudge_max": K_NUDGE_MAX,
            "min_qualifying_bf": MIN_QUALIFYING_BF,
            "min_prior_qualifying_outings": (
                MIN_PRIOR_QUALIFYING_OUTINGS
            ),
            "recent_bf_window": RECENT_BF_WINDOW,
            "volatility_rate_window": (
                VOLATILITY_RATE_WINDOW
            ),
            "bf_sd": BF_SD,
            "bf_min": BF_MIN,
            "bf_max": BF_MAX,
            "tto_factors": list(TTO_FACTORS),
            "sims": SIMS,
        },

        "formal_gate": {
            "min_formal_rows": MIN_FORMAL_ROWS,

            "min_absolute_crps_improvement": (
                MIN_ABSOLUTE_CRPS_IMPROVEMENT
            ),

            "min_relative_crps_improvement_pct": (
                MIN_RELATIVE_CRPS_IMPROVEMENT_PCT
            ),

            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_min_p_improve": (
                BOOTSTRAP_MIN_P_IMPROVE
            ),

            "eligible_month_min_rows": (
                ELIGIBLE_MONTH_MIN_ROWS
            ),

            "min_eligible_months": MIN_ELIGIBLE_MONTHS,

            "min_intervention_rows": MIN_INTERVENTION_ROWS,

            "min_intervention_relative_crps_improvement_pct": (
                MIN_INTERVENTION_RELATIVE_CRPS_IMPROVEMENT_PCT
            ),

            "eligible_slice_min_rows": (
                ELIGIBLE_SLICE_MIN_ROWS
            ),

            "min_eligible_slice_buckets": (
                MIN_ELIGIBLE_SLICE_BUCKETS
            ),

            "min_slice_nonworse_fraction": (
                MIN_SLICE_NONWORSE_FRACTION
            ),

            "max_allowed_relative_crps_worsening_pct": (
                MAX_ALLOWED_RELATIVE_CRPS_WORSENING_PCT
            ),

            "mae_nonworse_required": True,
            "rmse_nonworse_required": True,
            "absolute_bias_nonworse_required": True,
            "intervention_mae_nonworse_required": True,
        },

        "decision_policy": {
            "all_conditions_must_pass": True,
            "rescue_tuning": "FORBIDDEN",
            "stacking": False,
            "threshold_changes": False,
            "alternate_k_challengers_on_same_holdout": (
                "FORBIDDEN"
            ),
            "automatic_production_promotion": False,
            "pass_status": (
                "FORMAL_VALIDATED_CHALLENGER_"
                "PENDING_IMPLEMENTATION_PARITY_AUDIT"
            ),
        },
    }


def freeze_gate_manifest(prior_registry):
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()

    manifest = {
        "script": "PITCHER_K_D0_2026_FORMAL_GATE_A",
        "frozen_at_utc": now_utc(),

        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),

        "baseline_db": str(BASELINE_DB),
        "baseline_db_size_bytes": (
            BASELINE_DB.stat().st_size
        ),

        "prior_registry": prior_registry,

        "holdout_integrity_declaration": {
            "candidate": D0_CELL,
            "candidate_selected_using_2025_development": True,
            "candidate_stability_confirmed_using_2025": True,
            "candidate_2026_formal_results_previously_opened": False,
            "2026_used_for_formal_decision_in_this_script_only": True,
            "one_shot_holdout": True,
        },

        "gate_definition": gate_definition(),
    }

    canonical = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    manifest_sha = hashlib.sha256(canonical).hexdigest()

    if GATE_MANIFEST.exists():
        existing = safe_json_load(GATE_MANIFEST)

        if not isinstance(existing, dict):
            raise RuntimeError(
                f"Existing gate manifest unreadable: {GATE_MANIFEST}"
            )

        existing_canonical = json.dumps(
            existing,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        existing_sha = hashlib.sha256(
            existing_canonical
        ).hexdigest()

        # Timestamp can differ only if we generated a new candidate manifest.
        # Reuse the prior frozen manifest rather than silently replacing it.
        existing_script_sha = existing.get("script_sha256")
        current_script_sha = manifest.get("script_sha256")

        if existing_script_sha != current_script_sha:
            raise RuntimeError(
                "Existing gate manifest was frozen under a different script "
                "hash. Refusing to overwrite or reopen the holdout."
            )

        manifest = existing
        manifest_sha = existing_sha

    else:
        GATE_MANIFEST.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        GATE_MANIFEST_SHA.write_text(
            manifest_sha + "\n",
            encoding="utf-8",
        )

    return manifest, manifest_sha


# =============================================================================
# Pitcher profile
# =============================================================================

def pitcher_profile(prior_outings):
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

    expected_bf = (
        sum(recent_bf) / len(recent_bf)
    )

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


# =============================================================================
# Simulation
# =============================================================================

def vectorized_ksim(
    k_per_bf,
    expected_bf,
    start_k_rates,
    *,
    seed,
    actual_k,
    sims=SIMS,
):
    rng = np.random.RandomState(int(seed))

    if start_k_rates and len(start_k_rates) >= 4:
        pool = np.array(
            start_k_rates,
            dtype=np.float64,
        )

        pool = (
            0.7 * pool
            + 0.3 * float(k_per_bf)
        )

    else:
        pool = np.array(
            [float(k_per_bf)],
            dtype=np.float64,
        )

    bf = np.rint(
        rng.normal(
            float(expected_bf),
            BF_SD,
            size=sims,
        )
    ).astype(np.int16)

    bf = np.clip(
        bf,
        BF_MIN,
        BF_MAX,
    )

    pool_idx = rng.randint(
        0,
        len(pool),
        size=sims,
    )

    k_sample = pool[pool_idx]

    n1 = np.minimum(bf, 9)
    n2 = np.minimum(
        np.maximum(bf - 9, 0),
        9,
    )
    n3 = np.maximum(bf - 18, 0)

    t1, t2, t3 = TTO_FACTORS

    p1 = np.clip(
        k_sample * t1,
        0.02,
        0.60,
    )

    p2 = np.clip(
        k_sample * t2,
        0.02,
        0.60,
    )

    p3 = np.clip(
        k_sample * t3,
        0.02,
        0.60,
    )

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

    sorted_results = np.sort(
        results.astype(np.float64)
    )

    n = len(sorted_results)

    coeff = (
        2.0 * np.arange(1, n + 1)
        - n
        - 1.0
    )

    mean_abs = float(
        np.mean(
            np.abs(
                sorted_results
                - float(actual_k)
            )
        )
    )

    half_pairwise = float(
        np.sum(
            coeff * sorted_results
        )
        / (n * n)
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
            float(q25)
            <= float(actual_k)
            <= float(q75)
        ),

        "actual_in_10_90": int(
            float(q10)
            <= float(actual_k)
            <= float(q90)
        ),
    }


# =============================================================================
# Formal DB
# =============================================================================

def create_formal_table(conn):
    conn.execute(
        "DROP TABLE IF EXISTS formal_rows"
    )

    conn.execute(
        """
        CREATE TABLE formal_rows (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,

            year INTEGER NOT NULL,
            game_date TEXT NOT NULL,
            game_id TEXT NOT NULL,
            pitcher_id TEXT NOT NULL,
            pitcher_hand TEXT,

            actual_k INTEGER NOT NULL,
            actual_bf INTEGER NOT NULL,

            expected_bf REAL NOT NULL,
            prior_qualifying_outings INTEGER NOT NULL,

            h2h_qualified_batters INTEGER NOT NULL,
            usable_data_batters INTEGER NOT NULL,

            a0_lineup_used INTEGER NOT NULL,
            d0_lineup_used INTEGER NOT NULL,

            intervention_status TEXT NOT NULL,

            k_nudge REAL NOT NULL,

            a0_mean REAL NOT NULL,
            a0_median REAL NOT NULL,
            a0_iqr REAL NOT NULL,
            a0_q10 REAL NOT NULL,
            a0_q90 REAL NOT NULL,
            a0_crps REAL NOT NULL,
            a0_actual_in_iqr INTEGER NOT NULL,
            a0_actual_in_10_90 INTEGER NOT NULL,

            d0_mean REAL NOT NULL,
            d0_median REAL NOT NULL,
            d0_iqr REAL NOT NULL,
            d0_q10 REAL NOT NULL,
            d0_q90 REAL NOT NULL,
            d0_crps REAL NOT NULL,
            d0_actual_in_iqr INTEGER NOT NULL,
            d0_actual_in_10_90 INTEGER NOT NULL,

            research_seed INTEGER NOT NULL,

            UNIQUE (
                game_date,
                game_id,
                pitcher_id
            )
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_formal_rows_date
        ON formal_rows(game_date)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_formal_rows_intervention
        ON formal_rows(intervention_status)
        """
    )

    conn.commit()


# =============================================================================
# Source coverage
# =============================================================================

def source_coverage(base_conn):
    rows = {}

    for table, date_col in (
        ("pa", "game_date"),
        ("pitcher_game", "game_date"),
        ("starter_games", "game_date"),
    ):
        r = base_conn.execute(
            f"""
            SELECT
                COUNT(*),
                MIN({date_col}),
                MAX({date_col})
            FROM {table}
            WHERE SUBSTR({date_col}, 1, 4) = ?
            """,
            (str(HOLDOUT_YEAR),),
        ).fetchone()

        rows[table] = {
            "rows_2026": int(r[0] or 0),
            "min_date_2026": r[1],
            "max_date_2026": r[2],
        }

    distinct_dates = base_conn.execute(
        """
        SELECT COUNT(DISTINCT game_date)
        FROM pa
        WHERE SUBSTR(game_date, 1, 4) = ?
        """,
        (str(HOLDOUT_YEAR),),
    ).fetchone()[0]

    distinct_games = base_conn.execute(
        """
        SELECT COUNT(DISTINCT game_id)
        FROM pa
        WHERE SUBSTR(game_date, 1, 4) = ?
        """,
        (str(HOLDOUT_YEAR),),
    ).fetchone()[0]

    rows["distinct_dates_2026"] = int(
        distinct_dates or 0
    )

    rows["distinct_games_2026"] = int(
        distinct_games or 0
    )

    return rows


# =============================================================================
# Build clean 2026 paired rows
# =============================================================================

def build_2026_rows(base_conn, formal_conn):
    create_formal_table(formal_conn)

    pitcher_outings = defaultdict(list)

    hand_state = defaultdict(
        lambda: [0, 0]
    )

    season_state = defaultdict(
        lambda: [0, 0]
    )

    h2h_state = defaultdict(
        lambda: [0, 0]
    )

    dates = [
        row[0]
        for row in base_conn.execute(
            """
            SELECT DISTINCT game_date
            FROM pa
            WHERE game_date >= '2024-01-01'
              AND game_date < '2027-01-01'
            ORDER BY game_date
            """
        )
    ]

    insert_sql = """
        INSERT INTO formal_rows (
            year,
            game_date,
            game_id,
            pitcher_id,
            pitcher_hand,

            actual_k,
            actual_bf,

            expected_bf,
            prior_qualifying_outings,

            h2h_qualified_batters,
            usable_data_batters,

            a0_lineup_used,
            d0_lineup_used,

            intervention_status,

            k_nudge,

            a0_mean,
            a0_median,
            a0_iqr,
            a0_q10,
            a0_q90,
            a0_crps,
            a0_actual_in_iqr,
            a0_actual_in_10_90,

            d0_mean,
            d0_median,
            d0_iqr,
            d0_q10,
            d0_q90,
            d0_crps,
            d0_actual_in_iqr,
            d0_actual_in_10_90,

            research_seed
        )
        VALUES (
            ?, ?, ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?,
            ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?
        )
    """

    batch = []

    stats = {
        "dates_seen_total": len(dates),
        "holdout_dates_seen": 0,
        "holdout_starters_seen": 0,
        "holdout_rows_scored": 0,
        "skipped_missing_prior_profile": 0,
        "skipped_missing_lineup": 0,
        "newly_activated_rows": 0,
        "unchanged_on_rows": 0,
        "unchanged_off_rows": 0,
        "unexpected_current_on_fixed_off_rows": 0,
    }

    for date_idx, game_date in enumerate(
        dates,
        1,
    ):
        year = int(str(game_date)[:4])

        starters = list(
            base_conn.execute(
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

        if year == HOLDOUT_YEAR:
            stats["holdout_dates_seen"] += 1

        for (
            game_id,
            pitcher_id,
            actual_bf,
            actual_k,
            p_throws,
        ) in starters:
            if year == HOLDOUT_YEAR:
                stats["holdout_starters_seen"] += 1

            profile = pitcher_profile(
                pitcher_outings[
                    (year, str(pitcher_id))
                ]
            )

            if profile is None:
                if year == HOLDOUT_YEAR:
                    stats[
                        "skipped_missing_prior_profile"
                    ] += 1

                continue

            lineup = []
            seen = set()

            for (batter_id,) in base_conn.execute(
                """
                SELECT batter_id
                FROM pa
                WHERE game_id = ?
                  AND pitcher_id = ?
                ORDER BY at_bat_number
                """,
                (
                    str(game_id),
                    str(pitcher_id),
                ),
            ):
                batter_id = str(batter_id)

                if batter_id in seen:
                    continue

                seen.add(batter_id)
                lineup.append(batter_id)

                if len(lineup) >= 9:
                    break

            if not lineup:
                if year == HOLDOUT_YEAR:
                    stats[
                        "skipped_missing_lineup"
                    ] += 1

                continue

            # Only 2026 is scored. 2024-2025 are history warm-up.
            if year == HOLDOUT_YEAR:
                throws = str(p_throws or "R")

                rates = []

                h2h_qualified = 0
                usable_data_batters = 0

                bvp_tot_pa = 0
                bvp_tot_k = 0

                for batter_id in lineup[:9]:
                    hand_pa, hand_k = hand_state[
                        (
                            year,
                            batter_id,
                            throws,
                        )
                    ]

                    season_pa, season_k = season_state[
                        (
                            year,
                            batter_id,
                        )
                    ]

                    if hand_pa >= MIN_GENERAL_PA:
                        gen_rate = hand_k / hand_pa
                        general_has_real_data = True

                    elif season_pa >= MIN_GENERAL_PA:
                        gen_rate = season_k / season_pa
                        general_has_real_data = True

                    else:
                        gen_rate = LEAGUE_AVG_K
                        general_has_real_data = False

                    h2h_pa, h2h_k = h2h_state[
                        (
                            batter_id,
                            str(pitcher_id),
                        )
                    ]

                    h2h_has_real_data = (
                        h2h_pa >= MIN_H2H_PA
                    )

                    if h2h_has_real_data:
                        h2h_rate = h2h_k / h2h_pa

                        rate = (
                            H2H_WEIGHT * h2h_rate
                            + GEN_WEIGHT * gen_rate
                        )

                        h2h_qualified += 1

                    else:
                        rate = gen_rate

                    if (
                        general_has_real_data
                        or h2h_has_real_data
                    ):
                        usable_data_batters += 1

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
                    lineup_avg_k
                    * profile["expected_bf"]
                    if lineup_avg_k is not None
                    else None
                )

                a0_lineup_used = int(
                    lineup_expected_k is not None
                    and lineup_expected_k > 0
                    and h2h_qualified
                    >= MIN_LINEUP_DATA_BATTERS
                )

                d0_lineup_used = int(
                    lineup_expected_k is not None
                    and lineup_expected_k > 0
                    and usable_data_batters
                    >= MIN_LINEUP_DATA_BATTERS
                )

                if (
                    a0_lineup_used == 0
                    and d0_lineup_used == 1
                ):
                    intervention_status = (
                        "newly_activated"
                    )

                elif (
                    a0_lineup_used == 1
                    and d0_lineup_used == 1
                ):
                    intervention_status = (
                        "unchanged_on"
                    )

                elif (
                    a0_lineup_used == 0
                    and d0_lineup_used == 0
                ):
                    intervention_status = (
                        "unchanged_off"
                    )

                else:
                    intervention_status = (
                        "current_on_fixed_off_unexpected"
                    )

                stats[
                    f"{intervention_status}_rows"
                ] += 1

                pitcher_projection = (
                    profile["final_kbf"]
                    * profile["expected_bf"]
                )

                if a0_lineup_used:
                    a0_blended = (
                        PITCHER_WEIGHT
                        * pitcher_projection
                        + LINEUP_WEIGHT
                        * lineup_expected_k
                    )

                else:
                    a0_blended = pitcher_projection

                if d0_lineup_used:
                    d0_blended = (
                        PITCHER_WEIGHT
                        * pitcher_projection
                        + LINEUP_WEIGHT
                        * lineup_expected_k
                    )

                else:
                    d0_blended = pitcher_projection

                if bvp_tot_pa >= BVP_MIN_SAMPLE_PA:
                    bvp_lineup_k_rate = (
                        bvp_tot_k / bvp_tot_pa
                        if bvp_tot_pa > 0
                        else None
                    )

                    raw_nudge = (
                        bvp_lineup_k_rate
                        / LEAGUE_AVG_K
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
                    k_nudge = 1.0

                a0_kbf = (
                    (
                        a0_blended
                        / profile["expected_bf"]
                    )
                    * k_nudge
                )

                d0_kbf = (
                    (
                        d0_blended
                        / profile["expected_bf"]
                    )
                    * k_nudge
                )

                a0_start_rates = list(
                    profile["per_outing_rates"]
                )

                d0_start_rates = list(
                    profile["per_outing_rates"]
                )

                if (
                    a0_start_rates
                    and k_nudge != 1.0
                ):
                    a0_start_rates = [
                        rate * k_nudge
                        for rate in a0_start_rates
                    ]

                if (
                    d0_start_rates
                    and k_nudge != 1.0
                ):
                    d0_start_rates = [
                        rate * k_nudge
                        for rate in d0_start_rates
                    ]

                seed = stable_seed(
                    "PITCHER_K_D0_2026_FORMAL_GATE_A",
                    game_date,
                    game_id,
                    pitcher_id,
                    SIMS,
                )

                a0_sim = vectorized_ksim(
                    a0_kbf,
                    profile["expected_bf"],
                    a0_start_rates,
                    seed=seed,
                    actual_k=int(actual_k),
                    sims=SIMS,
                )

                d0_sim = vectorized_ksim(
                    d0_kbf,
                    profile["expected_bf"],
                    d0_start_rates,
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

                        float(profile["expected_bf"]),
                        int(
                            profile[
                                "qualifying_outings"
                            ]
                        ),

                        int(h2h_qualified),
                        int(usable_data_batters),

                        int(a0_lineup_used),
                        int(d0_lineup_used),

                        intervention_status,

                        float(k_nudge),

                        float(a0_sim["mean"]),
                        float(a0_sim["median"]),
                        float(a0_sim["iqr"]),
                        float(a0_sim["q10"]),
                        float(a0_sim["q90"]),
                        float(a0_sim["crps"]),
                        int(a0_sim["actual_in_iqr"]),
                        int(
                            a0_sim[
                                "actual_in_10_90"
                            ]
                        ),

                        float(d0_sim["mean"]),
                        float(d0_sim["median"]),
                        float(d0_sim["iqr"]),
                        float(d0_sim["q10"]),
                        float(d0_sim["q90"]),
                        float(d0_sim["crps"]),
                        int(d0_sim["actual_in_iqr"]),
                        int(
                            d0_sim[
                                "actual_in_10_90"
                            ]
                        ),

                        int(seed),
                    )
                )

                stats["holdout_rows_scored"] += 1

                if len(batch) >= 200:
                    formal_conn.executemany(
                        insert_sql,
                        batch,
                    )

                    formal_conn.commit()
                    batch.clear()

        # Strict D-1:
        # update history only after all games on this date are scored.

        for (
            pitcher_id,
            bf,
            so,
        ) in base_conn.execute(
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
                (
                    year,
                    str(pitcher_id),
                )
            ].append(
                (
                    game_date,
                    int(bf),
                    int(so),
                )
            )

        for (
            batter_id,
            pitcher_id,
            throws,
            is_k,
        ) in base_conn.execute(
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
                (
                    year,
                    batter_id,
                    throws,
                )
            ][0] += 1

            hand_state[
                (
                    year,
                    batter_id,
                    throws,
                )
            ][1] += is_k

            season_state[
                (
                    year,
                    batter_id,
                )
            ][0] += 1

            season_state[
                (
                    year,
                    batter_id,
                )
            ][1] += is_k

            h2h_state[
                (
                    batter_id,
                    pitcher_id,
                )
            ][0] += 1

            h2h_state[
                (
                    batter_id,
                    pitcher_id,
                )
            ][1] += is_k

        if (
            date_idx % 20 == 0
            or date_idx == len(dates)
        ):
            print(
                f"Formal row build progress: "
                f"{date_idx}/{len(dates)} dates | "
                f"2026_scored={stats['holdout_rows_scored']:,}",
                flush=True,
            )

    if batch:
        formal_conn.executemany(
            insert_sql,
            batch,
        )

        formal_conn.commit()

    actual_rows = formal_conn.execute(
        """
        SELECT COUNT(*)
        FROM formal_rows
        """
    ).fetchone()[0]

    stats["formal_db_rows"] = int(
        actual_rows
    )

    stats["row_count_parity"] = (
        int(actual_rows)
        == int(stats["holdout_rows_scored"])
    )

    return stats


# =============================================================================
# Load formal rows
# =============================================================================

def load_formal_rows(conn):
    cols = [
        "year",
        "game_date",
        "game_id",
        "pitcher_id",
        "pitcher_hand",

        "actual_k",
        "actual_bf",

        "expected_bf",
        "prior_qualifying_outings",

        "h2h_qualified_batters",
        "usable_data_batters",

        "a0_lineup_used",
        "d0_lineup_used",

        "intervention_status",

        "k_nudge",

        "a0_mean",
        "a0_median",
        "a0_iqr",
        "a0_q10",
        "a0_q90",
        "a0_crps",
        "a0_actual_in_iqr",
        "a0_actual_in_10_90",

        "d0_mean",
        "d0_median",
        "d0_iqr",
        "d0_q10",
        "d0_q90",
        "d0_crps",
        "d0_actual_in_iqr",
        "d0_actual_in_10_90",

        "research_seed",
    ]

    sql = """
        SELECT
            year,
            game_date,
            game_id,
            pitcher_id,
            pitcher_hand,

            actual_k,
            actual_bf,

            expected_bf,
            prior_qualifying_outings,

            h2h_qualified_batters,
            usable_data_batters,

            a0_lineup_used,
            d0_lineup_used,

            intervention_status,

            k_nudge,

            a0_mean,
            a0_median,
            a0_iqr,
            a0_q10,
            a0_q90,
            a0_crps,
            a0_actual_in_iqr,
            a0_actual_in_10_90,

            d0_mean,
            d0_median,
            d0_iqr,
            d0_q10,
            d0_q90,
            d0_crps,
            d0_actual_in_iqr,
            d0_actual_in_10_90,

            research_seed

        FROM formal_rows
        ORDER BY game_date, game_id, pitcher_id
    """

    rows = []

    for raw in conn.execute(sql):
        row = dict(zip(cols, raw))

        row["actual_k"] = float(
            row["actual_k"]
        )

        row["a0_mean"] = float(
            row["a0_mean"]
        )

        row["d0_mean"] = float(
            row["d0_mean"]
        )

        row["a0_crps"] = float(
            row["a0_crps"]
        )

        row["d0_crps"] = float(
            row["d0_crps"]
        )

        row["a0_error"] = (
            row["a0_mean"]
            - row["actual_k"]
        )

        row["d0_error"] = (
            row["d0_mean"]
            - row["actual_k"]
        )

        row["a0_abs_error"] = abs(
            row["a0_error"]
        )

        row["d0_abs_error"] = abs(
            row["d0_error"]
        )

        row["a0_sq_error"] = (
            row["a0_error"] ** 2
        )

        row["d0_sq_error"] = (
            row["d0_error"] ** 2
        )

        row["crps_delta_d0_minus_a0"] = (
            row["d0_crps"]
            - row["a0_crps"]
        )

        row["projection_shift_d0_minus_a0"] = (
            row["d0_mean"]
            - row["a0_mean"]
        )

        rows.append(row)

    return rows


# =============================================================================
# Integrity
# =============================================================================

def paired_integrity(rows):
    keys = [
        (
            row["game_date"],
            row["game_id"],
            row["pitcher_id"],
        )
        for row in rows
    ]

    duplicate_keys = (
        len(keys)
        - len(set(keys))
    )

    wrong_year_rows = sum(
        1
        for row in rows
        if int(row["year"]) != HOLDOUT_YEAR
    )

    unexpected_transition_rows = sum(
        1
        for row in rows
        if row["intervention_status"]
        == "current_on_fixed_off_unexpected"
    )

    return {
        "paired_rows": len(rows),

        "duplicate_paired_keys": (
            duplicate_keys
        ),

        "wrong_year_rows": (
            wrong_year_rows
        ),

        "unexpected_current_on_fixed_off_rows": (
            unexpected_transition_rows
        ),

        "integrity_pass": (
            duplicate_keys == 0
            and wrong_year_rows == 0
            and unexpected_transition_rows == 0
        ),
    }


# =============================================================================
# Metrics
# =============================================================================

def metric_block(rows):
    rows = list(rows)

    if not rows:
        return {
            "rows": 0,
        }

    n = len(rows)

    actual_mean = (
        sum(
            row["actual_k"]
            for row in rows
        )
        / n
    )

    a0_mean = (
        sum(
            row["a0_mean"]
            for row in rows
        )
        / n
    )

    d0_mean = (
        sum(
            row["d0_mean"]
            for row in rows
        )
        / n
    )

    a0_bias = (
        sum(
            row["a0_error"]
            for row in rows
        )
        / n
    )

    d0_bias = (
        sum(
            row["d0_error"]
            for row in rows
        )
        / n
    )

    a0_mae = (
        sum(
            row["a0_abs_error"]
            for row in rows
        )
        / n
    )

    d0_mae = (
        sum(
            row["d0_abs_error"]
            for row in rows
        )
        / n
    )

    a0_rmse = math.sqrt(
        sum(
            row["a0_sq_error"]
            for row in rows
        )
        / n
    )

    d0_rmse = math.sqrt(
        sum(
            row["d0_sq_error"]
            for row in rows
        )
        / n
    )

    a0_crps = (
        sum(
            row["a0_crps"]
            for row in rows
        )
        / n
    )

    d0_crps = (
        sum(
            row["d0_crps"]
            for row in rows
        )
        / n
    )

    crps_delta = (
        d0_crps - a0_crps
    )

    absolute_crps_improvement = (
        a0_crps - d0_crps
    )

    relative_crps_improvement_pct = (
        (
            a0_crps - d0_crps
        )
        / a0_crps
        * 100.0
        if a0_crps != 0
        else None
    )

    shifts = [
        row[
            "projection_shift_d0_minus_a0"
        ]
        for row in rows
    ]

    crps_wins = sum(
        1
        for row in rows
        if row[
            "crps_delta_d0_minus_a0"
        ] < -TIE_EPS
    )

    crps_losses = sum(
        1
        for row in rows
        if row[
            "crps_delta_d0_minus_a0"
        ] > TIE_EPS
    )

    crps_ties = (
        n - crps_wins - crps_losses
    )

    return {
        "rows": n,

        "actual_mean_k": actual_mean,

        "a0_projected_mean_k": a0_mean,
        "d0_projected_mean_k": d0_mean,

        "a0_projection_bias": a0_bias,
        "d0_projection_bias": d0_bias,

        "absolute_bias_delta_d0_minus_a0": (
            abs(d0_bias)
            - abs(a0_bias)
        ),

        "a0_mae": a0_mae,
        "d0_mae": d0_mae,

        "mae_delta_d0_minus_a0": (
            d0_mae - a0_mae
        ),

        "a0_rmse": a0_rmse,
        "d0_rmse": d0_rmse,

        "rmse_delta_d0_minus_a0": (
            d0_rmse - a0_rmse
        ),

        "a0_mean_crps": a0_crps,
        "d0_mean_crps": d0_crps,

        "crps_delta_d0_minus_a0": (
            crps_delta
        ),

        "absolute_crps_improvement": (
            absolute_crps_improvement
        ),

        "relative_crps_improvement_pct": (
            relative_crps_improvement_pct
        ),

        "a0_iqr_coverage": (
            sum(
                int(
                    row[
                        "a0_actual_in_iqr"
                    ]
                )
                for row in rows
            )
            / n
        ),

        "d0_iqr_coverage": (
            sum(
                int(
                    row[
                        "d0_actual_in_iqr"
                    ]
                )
                for row in rows
            )
            / n
        ),

        "a0_q10_q90_coverage": (
            sum(
                int(
                    row[
                        "a0_actual_in_10_90"
                    ]
                )
                for row in rows
            )
            / n
        ),

        "d0_q10_q90_coverage": (
            sum(
                int(
                    row[
                        "d0_actual_in_10_90"
                    ]
                )
                for row in rows
            )
            / n
        ),

        "crps_row_wins": crps_wins,
        "crps_row_losses": crps_losses,
        "crps_row_ties": crps_ties,

        "crps_row_win_rate_ex_ties": (
            crps_wins
            / (
                crps_wins
                + crps_losses
            )
            if (
                crps_wins
                + crps_losses
            ) > 0
            else None
        ),

        "mean_projection_shift_d0_minus_a0": (
            sum(shifts) / n
        ),

        "median_projection_shift_d0_minus_a0": (
            statistics.median(shifts)
        ),

        "positive_projection_shift_rows": sum(
            1
            for shift in shifts
            if shift > TIE_EPS
        ),

        "negative_projection_shift_rows": sum(
            1
            for shift in shifts
            if shift < -TIE_EPS
        ),

        "zero_projection_shift_rows": sum(
            1
            for shift in shifts
            if abs(shift) <= TIE_EPS
        ),

        "mean_expected_bf": (
            sum(
                float(row["expected_bf"])
                for row in rows
            )
            / n
        ),

        "mean_actual_bf": (
            sum(
                float(row["actual_bf"])
                for row in rows
            )
            / n
        ),
    }


def grouped_metrics(rows, key_func):
    groups = defaultdict(list)

    for row in rows:
        groups[str(key_func(row))].append(
            row
        )

    return {
        key: metric_block(groups[key])
        for key in sorted(groups)
    }


# =============================================================================
# Date-block bootstrap
# =============================================================================

def date_block_bootstrap(rows):
    groups = defaultdict(
        lambda: {
            "sum_delta": 0.0,
            "n": 0,
        }
    )

    for row in rows:
        d = str(row["game_date"])

        groups[d]["sum_delta"] += (
            row["crps_delta_d0_minus_a0"]
        )

        groups[d]["n"] += 1

    dates = sorted(groups)

    if not dates:
        return {
            "date_blocks": 0,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "p_d0_better_than_a0": None,
            "delta_ci_2_5": None,
            "delta_ci_50": None,
            "delta_ci_97_5": None,
        }

    sums = np.array(
        [
            groups[d]["sum_delta"]
            for d in dates
        ],
        dtype=np.float64,
    )

    counts = np.array(
        [
            groups[d]["n"]
            for d in dates
        ],
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    stats = np.empty(
        BOOTSTRAP_REPS,
        dtype=np.float64,
    )

    m = len(dates)

    for i in range(BOOTSTRAP_REPS):
        idx = rng.integers(
            0,
            m,
            size=m,
        )

        denominator = float(
            np.sum(counts[idx])
        )

        if denominator <= 0:
            stats[i] = np.nan
            continue

        stats[i] = float(
            np.sum(sums[idx])
            / denominator
        )

    stats = stats[
        np.isfinite(stats)
    ]

    if len(stats) == 0:
        raise RuntimeError(
            "Bootstrap produced no finite statistics."
        )

    p_improve = float(
        np.mean(stats < 0.0)
    )

    q2_5, q50, q97_5 = np.percentile(
        stats,
        [2.5, 50.0, 97.5],
    )

    observed_delta = (
        sum(
            row[
                "crps_delta_d0_minus_a0"
            ]
            for row in rows
        )
        / len(rows)
    )

    return {
        "date_blocks": len(dates),
        "bootstrap_reps": int(
            len(stats)
        ),
        "bootstrap_seed": (
            BOOTSTRAP_SEED
        ),
        "observed_mean_crps_delta_d0_minus_a0": (
            observed_delta
        ),
        "p_d0_better_than_a0": (
            p_improve
        ),
        "delta_ci_2_5": float(q2_5),
        "delta_ci_50": float(q50),
        "delta_ci_97_5": float(q97_5),
    }


# =============================================================================
# Monthly stability
# =============================================================================

def monthly_stability(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[
            str(row["game_date"])[:7]
        ].append(row)

    months = {}

    eligible_months = 0
    crps_wins = 0

    for month in sorted(groups):
        block = metric_block(
            groups[month]
        )

        eligible = (
            block["rows"]
            >= ELIGIBLE_MONTH_MIN_ROWS
        )

        challenger_win = (
            block[
                "crps_delta_d0_minus_a0"
            ] < 0
        )

        if eligible:
            eligible_months += 1

            if challenger_win:
                crps_wins += 1

        months[month] = {
            **block,
            "eligible": eligible,
            "challenger_win": (
                challenger_win
            ),
        }

    required_wins = (
        math.ceil(
            eligible_months / 2
        )
        if eligible_months > 0
        else 0
    )

    return {
        "eligible_months": (
            eligible_months
        ),
        "minimum_required_eligible_months": (
            MIN_ELIGIBLE_MONTHS
        ),
        "crps_wins": crps_wins,
        "required_wins": required_wins,
        "months": months,
    }


# =============================================================================
# Intervention
# =============================================================================

def intervention_analysis(rows):
    groups = grouped_metrics(
        rows,
        lambda row: row[
            "intervention_status"
        ],
    )

    intervention_rows = [
        row
        for row in rows
        if row["intervention_status"]
        == "newly_activated"
    ]

    unchanged_rows = [
        row
        for row in rows
        if row["intervention_status"]
        != "newly_activated"
    ]

    return {
        "definition": (
            "A0 lineup component OFF and "
            "D0 lineup component ON"
        ),

        "intervention_rows": len(
            intervention_rows
        ),

        "intervention_rate": (
            len(intervention_rows)
            / len(rows)
            if rows
            else 0.0
        ),

        "intervention_metrics": (
            metric_block(
                intervention_rows
            )
        ),

        "unchanged_rows": len(
            unchanged_rows
        ),

        "unchanged_metrics": (
            metric_block(
                unchanged_rows
            )
        ),

        "by_intervention_status": groups,
    }


# =============================================================================
# Slice definitions
# =============================================================================

def season_phase_bucket(row):
    month = int(
        str(row["game_date"])[5:7]
    )

    if month <= 4:
        return "through_april"

    if month <= 6:
        return "may_june"

    return "july_onward"


def pitcher_hand_bucket(row):
    hand = str(
        row["pitcher_hand"]
        or "unknown"
    ).upper()

    if hand in {"L", "R"}:
        return hand

    return "unknown"


def prior_outing_bucket(row):
    n = int(
        row["prior_qualifying_outings"]
    )

    if n <= 4:
        return "3_4"

    if n <= 7:
        return "5_7"

    if n <= 11:
        return "8_11"

    return "12_plus"


def workload_bucket(row):
    bf = float(
        row["expected_bf"]
    )

    if bf < 18:
        return "lt_18"

    if bf < 21:
        return "18_20.99"

    if bf < 24:
        return "21_23.99"

    return "24_plus"


def h2h_qualified_bucket(row):
    n = int(
        row[
            "h2h_qualified_batters"
        ]
    )

    if n == 0:
        return "0"

    if n <= 2:
        return "1_2"

    if n <= 4:
        return "3_4"

    return "5_plus"


def usable_data_bucket(row):
    n = int(
        row[
            "usable_data_batters"
        ]
    )

    if n <= 4:
        return "0_4"

    if n <= 6:
        return "5_6"

    return "7_9"


def a0_lineup_state_bucket(row):
    return (
        "a0_lineup_on"
        if int(
            row["a0_lineup_used"]
        ) == 1
        else "a0_lineup_off"
    )


def intervention_status_bucket(row):
    return str(
        row["intervention_status"]
    )


SLICE_DEFINITIONS = {
    "season_phase": season_phase_bucket,
    "pitcher_hand": pitcher_hand_bucket,
    "prior_qualifying_outings": (
        prior_outing_bucket
    ),
    "expected_bf": workload_bucket,
    "h2h_qualified_batters": (
        h2h_qualified_bucket
    ),
    "usable_data_batters": (
        usable_data_bucket
    ),
    "a0_lineup_state": (
        a0_lineup_state_bucket
    ),
    "intervention_status": (
        intervention_status_bucket
    ),
}


# =============================================================================
# Slice breadth
# =============================================================================

def slice_breadth(rows):
    all_slices = {}

    eligible_buckets = []
    nonworse_buckets = []
    materially_damaged_buckets = []

    for (
        slice_name,
        key_func,
    ) in SLICE_DEFINITIONS.items():
        grouped = grouped_metrics(
            rows,
            key_func,
        )

        all_slices[
            slice_name
        ] = grouped

        for (
            bucket_name,
            block,
        ) in grouped.items():
            eligible = (
                block["rows"]
                >= ELIGIBLE_SLICE_MIN_ROWS
            )

            if not eligible:
                continue

            key = (
                f"{slice_name}|"
                f"{bucket_name}"
            )

            eligible_buckets.append(key)

            if (
                block[
                    "crps_delta_d0_minus_a0"
                ] <= 0
            ):
                nonworse_buckets.append(
                    key
                )

            rel_improvement = block[
                "relative_crps_improvement_pct"
            ]

            relative_worsening_pct = (
                -rel_improvement
                if rel_improvement is not None
                else None
            )

            if (
                relative_worsening_pct
                is not None
                and relative_worsening_pct
                > MAX_ALLOWED_RELATIVE_CRPS_WORSENING_PCT
            ):
                materially_damaged_buckets.append(
                    {
                        "slice_bucket": key,
                        "rows": block["rows"],
                        "relative_crps_worsening_pct": (
                            relative_worsening_pct
                        ),
                        "a0_mean_crps": (
                            block[
                                "a0_mean_crps"
                            ]
                        ),
                        "d0_mean_crps": (
                            block[
                                "d0_mean_crps"
                            ]
                        ),
                    }
                )

    fraction_nonworse = (
        len(nonworse_buckets)
        / len(eligible_buckets)
        if eligible_buckets
        else 0.0
    )

    return {
        "eligible_slice_bucket_count": (
            len(eligible_buckets)
        ),

        "minimum_required_eligible_slice_buckets": (
            MIN_ELIGIBLE_SLICE_BUCKETS
        ),

        "crps_nonworse_bucket_count": (
            len(nonworse_buckets)
        ),

        "crps_nonworse_fraction": (
            fraction_nonworse
        ),

        "minimum_required_nonworse_fraction": (
            MIN_SLICE_NONWORSE_FRACTION
        ),

        "max_allowed_relative_crps_worsening_pct": (
            MAX_ALLOWED_RELATIVE_CRPS_WORSENING_PCT
        ),

        "eligible_buckets": (
            eligible_buckets
        ),

        "crps_nonworse_buckets": (
            nonworse_buckets
        ),

        "materially_damaged_buckets": (
            materially_damaged_buckets
        ),

        "slices": all_slices,
    }


# =============================================================================
# Formal gate
# =============================================================================

def build_formal_gate(
    integrity,
    primary,
    bootstrap,
    intervention,
    monthly,
    breadth,
):
    intervention_block = (
        intervention[
            "intervention_metrics"
        ]
    )

    gate = {
        "paired_integrity_pass": (
            integrity[
                "integrity_pass"
            ]
        ),

        "minimum_formal_rows_pass": (
            primary["rows"]
            >= MIN_FORMAL_ROWS
        ),

        "overall_absolute_crps_improvement_pass": (
            primary[
                "absolute_crps_improvement"
            ]
            >= MIN_ABSOLUTE_CRPS_IMPROVEMENT
        ),

        "overall_relative_crps_improvement_pass": (
            primary[
                "relative_crps_improvement_pct"
            ]
            is not None
            and primary[
                "relative_crps_improvement_pct"
            ]
            >= MIN_RELATIVE_CRPS_IMPROVEMENT_PCT
        ),

        "date_block_bootstrap_probability_pass": (
            bootstrap[
                "p_d0_better_than_a0"
            ]
            is not None
            and bootstrap[
                "p_d0_better_than_a0"
            ]
            >= BOOTSTRAP_MIN_P_IMPROVE
        ),

        "overall_mae_nonworse_pass": (
            primary[
                "mae_delta_d0_minus_a0"
            ] <= 0
        ),

        "overall_rmse_nonworse_pass": (
            primary[
                "rmse_delta_d0_minus_a0"
            ] <= 0
        ),

        "overall_absolute_bias_nonworse_pass": (
            primary[
                "absolute_bias_delta_d0_minus_a0"
            ] <= 0
        ),

        "minimum_eligible_months_pass": (
            monthly[
                "eligible_months"
            ]
            >= MIN_ELIGIBLE_MONTHS
        ),

        "monthly_crps_stability_pass": (
            monthly[
                "eligible_months"
            ]
            >= MIN_ELIGIBLE_MONTHS
            and monthly[
                "crps_wins"
            ]
            >= monthly[
                "required_wins"
            ]
        ),

        "intervention_subset_min_rows_pass": (
            intervention[
                "intervention_rows"
            ]
            >= MIN_INTERVENTION_ROWS
        ),

        "intervention_subset_relative_crps_improvement_pass": (
            intervention_block.get(
                "relative_crps_improvement_pct"
            )
            is not None
            and intervention_block[
                "relative_crps_improvement_pct"
            ]
            >= MIN_INTERVENTION_RELATIVE_CRPS_IMPROVEMENT_PCT
        ),

        "intervention_subset_mae_nonworse_pass": (
            intervention_block.get(
                "mae_delta_d0_minus_a0"
            )
            is not None
            and intervention_block[
                "mae_delta_d0_minus_a0"
            ] <= 0
        ),

        "minimum_eligible_slice_buckets_pass": (
            breadth[
                "eligible_slice_bucket_count"
            ]
            >= MIN_ELIGIBLE_SLICE_BUCKETS
        ),

        "slice_breadth_nonworse_fraction_pass": (
            breadth[
                "eligible_slice_bucket_count"
            ]
            >= MIN_ELIGIBLE_SLICE_BUCKETS
            and breadth[
                "crps_nonworse_fraction"
            ]
            >= MIN_SLICE_NONWORSE_FRACTION
        ),

        "no_materially_damaged_eligible_slice_pass": (
            len(
                breadth[
                    "materially_damaged_buckets"
                ]
            ) == 0
        ),
    }

    gate[
        "overall_formal_gate_pass"
    ] = all(gate.values())

    return gate


# =============================================================================
# Reporting
# =============================================================================

def fmt_block(label, block):
    if (
        not block
        or block.get("rows", 0) == 0
    ):
        return f"{label}: rows=0"

    return (
        f"{label}: "
        f"n={block['rows']:,} "
        f"A0_crps={block['a0_mean_crps']:.8f} "
        f"D0_crps={block['d0_mean_crps']:.8f} "
        f"delta={block['crps_delta_d0_minus_a0']:+.8f} "
        f"abs_improve={block['absolute_crps_improvement']:.8f} "
        f"rel_improve={block['relative_crps_improvement_pct']:+.4f}% "
        f"A0_mae={block['a0_mae']:.6f} "
        f"D0_mae={block['d0_mae']:.6f} "
        f"A0_rmse={block['a0_rmse']:.6f} "
        f"D0_rmse={block['d0_rmse']:.6f} "
        f"A0_bias={block['a0_projection_bias']:+.6f} "
        f"D0_bias={block['d0_projection_bias']:+.6f}"
    )


def build_report(payload):
    lines = []

    lines.append(
        "PITCHER_K_D0_2026_FORMAL_GATE_A"
    )

    lines.append("=" * 35)
    lines.append("")

    lines.append("FORMAL GATE PREFLIGHT")
    lines.append("---------------------")
    lines.append(
        json.dumps(
            payload["preflight"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("IMMUTABLE GATE FREEZE")
    lines.append("---------------------")
    lines.append(
        json.dumps(
            payload["gate_freeze"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("2026 HOLDOUT SOURCE COVERAGE")
    lines.append("----------------------------")
    lines.append(
        json.dumps(
            payload["source_coverage"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("BUILD CLEAN 2026 PAIRED ROWS")
    lines.append("----------------------------")
    lines.append(
        json.dumps(
            payload["build_summary"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("2026 PRIMARY FORMAL COMPARISON")
    lines.append("------------------------------")
    lines.append(
        fmt_block(
            "2026",
            payload["primary_2026"],
        )
    )

    lines.append("")
    lines.append("DATE-BLOCK BOOTSTRAP")
    lines.append("--------------------")
    lines.append(
        json.dumps(
            payload["date_block_bootstrap"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("INTERVENTION SUBSET")
    lines.append("-------------------")
    lines.append(
        json.dumps(
            payload["intervention_analysis"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("MONTHLY STABILITY")
    lines.append("-----------------")
    lines.append(
        json.dumps(
            payload["monthly_stability"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("SLICE BREADTH")
    lines.append("-------------")
    lines.append(
        json.dumps(
            payload["slice_breadth"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("STRICT FORMAL GATE")
    lines.append("------------------")
    lines.append(
        json.dumps(
            payload["formal_gate"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("FINAL FORMAL VERDICT")
    lines.append("--------------------")
    lines.append(
        payload["final_verdict"]
    )

    return "\n".join(lines)


# =============================================================================
# Existing-result verification
# =============================================================================

def verify_existing():
    if not OUT_JSON.exists():
        print(
            "No completed formal result exists.",
            flush=True,
        )
        return 1

    payload = safe_json_load(OUT_JSON)

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Existing result unreadable: {OUT_JSON}"
        )

    print(
        "EXISTING FORMAL RESULT VERIFIED",
        flush=True,
    )

    print(
        f"verdict={payload.get('final_verdict')}",
        flush=True,
    )

    print(
        f"gate_manifest_sha256="
        f"{payload.get('gate_freeze', {}).get('gate_manifest_sha256')}",
        flush=True,
    )

    return 0


# =============================================================================
# Main
# =============================================================================

def main_parent():
    if OUT_JSON.exists():
        raise RuntimeError(
            "Completed formal result already exists. "
            "Refusing to rerun the one-shot 2026 holdout. "
            "Use --verify-existing instead."
        )

    print(
        "PITCHER_K_D0_2026_FORMAL_GATE_A",
        flush=True,
    )

    print(
        "===================================",
        flush=True,
    )

    print("")
    print(
        "FORMAL GATE PREFLIGHT",
        flush=True,
    )

    print(
        "---------------------",
        flush=True,
    )

    prior_registry = verify_prior_registry()

    base_conn = connect_ro(BASELINE_DB)
    formal_conn = connect_formal()

    try:
        tables = check_required_tables(
            base_conn
        )

        preflight = {
            "candidate": D0_CELL,
            "comparator": A0_CELL,

            "holdout_year": HOLDOUT_YEAR,

            "baseline_db": str(
                BASELINE_DB
            ),

            "required_tables_present": True,
            "baseline_tables": tables,

            "prior_registry_verified": True,
            "prior_registry": prior_registry,

            "candidate_count_opened_on_2026": 1,

            "alternate_k_challengers_on_same_holdout": (
                "FORBIDDEN"
            ),

            "development_data": [
                2024,
                2025,
            ],

            "formal_holdout_data": [
                2026,
            ],

            "production_changed": False,
            "external_api_calls": False,
            "stacking": False,
            "threshold_tuning": False,
            "rescue_tuning": False,

            "production_promotion_authorized": False,
        }

        print(
            json.dumps(
                preflight,
                indent=2,
            ),
            flush=True,
        )

        print("")
        print(
            "IMMUTABLE GATE FREEZE",
            flush=True,
        )

        print(
            "---------------------",
            flush=True,
        )

        manifest, manifest_sha = (
            freeze_gate_manifest(
                prior_registry
            )
        )

        gate_freeze = {
            "manifest_path": str(
                GATE_MANIFEST
            ),

            "manifest_sha_path": str(
                GATE_MANIFEST_SHA
            ),

            "gate_manifest_sha256": (
                manifest_sha
            ),

            "script_sha256": (
                manifest[
                    "script_sha256"
                ]
            ),

            "holdout_year": (
                HOLDOUT_YEAR
            ),

            "formal_gate_frozen_before_metrics": True,

            "gate_definition": (
                manifest[
                    "gate_definition"
                ]
            ),
        }

        print(
            json.dumps(
                gate_freeze,
                indent=2,
            ),
            flush=True,
        )

        print("")
        print(
            "2026 HOLDOUT SOURCE COVERAGE",
            flush=True,
        )

        print(
            "----------------------------",
            flush=True,
        )

        coverage = source_coverage(
            base_conn
        )

        print(
            json.dumps(
                coverage,
                indent=2,
            ),
            flush=True,
        )

        starter_rows_2026 = (
            coverage[
                "starter_games"
            ]["rows_2026"]
        )

        if starter_rows_2026 <= 0:
            raise RuntimeError(
                "No 2026 starter rows exist in baseline source tables."
            )

        print("")
        print(
            "BUILD CLEAN 2026 PAIRED ROWS",
            flush=True,
        )

        print(
            "----------------------------",
            flush=True,
        )

        build_summary = build_2026_rows(
            base_conn,
            formal_conn,
        )

        print(
            json.dumps(
                build_summary,
                indent=2,
            ),
            flush=True,
        )

        rows = load_formal_rows(
            formal_conn
        )

        integrity = paired_integrity(
            rows
        )

        print("")
        print(
            "2026 PRIMARY FORMAL COMPARISON",
            flush=True,
        )

        print(
            "------------------------------",
            flush=True,
        )

        primary = metric_block(
            rows
        )

        print(
            fmt_block(
                "2026",
                primary,
            ),
            flush=True,
        )

        print("")
        print(
            "DATE-BLOCK BOOTSTRAP",
            flush=True,
        )

        print(
            "--------------------",
            flush=True,
        )

        bootstrap = date_block_bootstrap(
            rows
        )

        print(
            json.dumps(
                bootstrap,
                indent=2,
            ),
            flush=True,
        )

        intervention = intervention_analysis(
            rows
        )

        print("")
        print(
            "INTERVENTION SUBSET",
            flush=True,
        )

        print(
            "-------------------",
            flush=True,
        )

        print(
            f"definition="
            f"{intervention['definition']}",
            flush=True,
        )

        print(
            f"intervention_rows="
            f"{intervention['intervention_rows']:,}",
            flush=True,
        )

        print(
            f"intervention_rate="
            f"{intervention['intervention_rate']:.6f}",
            flush=True,
        )

        print(
            fmt_block(
                "INTERVENTION",
                intervention[
                    "intervention_metrics"
                ],
            ),
            flush=True,
        )

        print(
            fmt_block(
                "UNCHANGED",
                intervention[
                    "unchanged_metrics"
                ],
            ),
            flush=True,
        )

        monthly = monthly_stability(
            rows
        )

        print("")
        print(
            "MONTHLY STABILITY",
            flush=True,
        )

        print(
            "-----------------",
            flush=True,
        )

        for (
            month,
            block,
        ) in monthly[
            "months"
        ].items():
            print(
                f"{month}: "
                f"n={block['rows']:,} "
                f"eligible={block['eligible']} "
                f"A0_crps={block['a0_mean_crps']:.8f} "
                f"D0_crps={block['d0_mean_crps']:.8f} "
                f"delta={block['crps_delta_d0_minus_a0']:+.8f} "
                f"rel_improve={block['relative_crps_improvement_pct']:+.4f}% "
                f"challenger_win={block['challenger_win']}",
                flush=True,
            )

        print(
            f"eligible_months="
            f"{monthly['eligible_months']}",
            flush=True,
        )

        print(
            f"required_wins="
            f"{monthly['required_wins']}",
            flush=True,
        )

        print(
            f"crps_wins="
            f"{monthly['crps_wins']}",
            flush=True,
        )

        breadth = slice_breadth(
            rows
        )

        print("")
        print(
            "SLICE BREADTH",
            flush=True,
        )

        print(
            "-------------",
            flush=True,
        )

        print(
            f"eligible_slice_bucket_count="
            f"{breadth['eligible_slice_bucket_count']}",
            flush=True,
        )

        print(
            f"crps_nonworse_bucket_count="
            f"{breadth['crps_nonworse_bucket_count']}",
            flush=True,
        )

        print(
            f"crps_nonworse_fraction="
            f"{breadth['crps_nonworse_fraction']:.6f}",
            flush=True,
        )

        print(
            f"materially_damaged_buckets="
            f"{len(breadth['materially_damaged_buckets'])}",
            flush=True,
        )

        for (
            slice_name,
            grouped,
        ) in breadth[
            "slices"
        ].items():
            print(
                "",
                flush=True,
            )

            print(
                f"[{slice_name}]",
                flush=True,
            )

            for (
                bucket,
                block,
            ) in grouped.items():
                eligible = (
                    block["rows"]
                    >= ELIGIBLE_SLICE_MIN_ROWS
                )

                print(
                    f"{bucket}: "
                    f"n={block['rows']:,} "
                    f"eligible={eligible} "
                    f"A0_crps={block['a0_mean_crps']:.8f} "
                    f"D0_crps={block['d0_mean_crps']:.8f} "
                    f"delta={block['crps_delta_d0_minus_a0']:+.8f} "
                    f"rel_improve={block['relative_crps_improvement_pct']:+.4f}%",
                    flush=True,
                )

        formal_gate = build_formal_gate(
            integrity,
            primary,
            bootstrap,
            intervention,
            monthly,
            breadth,
        )

        print("")
        print(
            "STRICT FORMAL GATE",
            flush=True,
        )

        print(
            "------------------",
            flush=True,
        )

        for (
            key,
            value,
        ) in formal_gate.items():
            print(
                f"{key}: {value}",
                flush=True,
            )

        if formal_gate[
            "overall_formal_gate_pass"
        ]:
            final_verdict = (
                "D0_FIXED_LINEUP_ACTIVATION_PASSES_2026_FORMAL_GATE_"
                "FORMAL_VALIDATED_CHALLENGER_"
                "PENDING_IMPLEMENTATION_PARITY_AUDIT_"
                "NO_AUTOMATIC_PRODUCTION_PROMOTION"
            )

            formal_status = (
                "FORMAL_VALIDATED_CHALLENGER_"
                "PENDING_IMPLEMENTATION_PARITY_AUDIT"
            )

        else:
            final_verdict = (
                "D0_FIXED_LINEUP_ACTIVATION_FAILS_2026_FORMAL_GATE_"
                "FREEZE_NO_RESCUE_TUNING"
            )

            formal_status = (
                "FAILED_FORMAL_HOLDOUT_FROZEN"
            )

        print("")
        print(
            "FINAL FORMAL VERDICT",
            flush=True,
        )

        print(
            "--------------------",
            flush=True,
        )

        print(
            final_verdict,
            flush=True,
        )

        payload = {
            "script": (
                "PITCHER_K_D0_2026_FORMAL_GATE_A"
            ),

            "generated_at_utc": (
                now_utc()
            ),

            "candidate": D0_CELL,
            "comparator": A0_CELL,
            "holdout_year": HOLDOUT_YEAR,

            "formal_status": formal_status,

            "preflight": preflight,
            "gate_freeze": gate_freeze,
            "source_coverage": coverage,
            "build_summary": build_summary,
            "paired_integrity": integrity,

            "primary_2026": primary,
            "date_block_bootstrap": bootstrap,
            "intervention_analysis": intervention,
            "monthly_stability": monthly,
            "slice_breadth": breadth,

            "formal_gate": formal_gate,
            "final_verdict": final_verdict,

            "policy": {
                "all_conditions_required": True,
                "production_changed": False,
                "production_promotion_authorized": False,
                "rescue_tuning": "FORBIDDEN",
                "threshold_changes": "FORBIDDEN",
                "stacking": "FORBIDDEN",
                "alternate_k_challenger_on_same_holdout": (
                    "FORBIDDEN"
                ),
                "next_step_if_pass": (
                    "implementation parity audit, then explicit "
                    "production-promotion decision"
                ),
                "next_step_if_fail": (
                    "freeze D0; no rescue tuning on 2026"
                ),
            },
        }

        OUT_JSON.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        report = build_report(
            payload
        )

        OUT_TXT.write_text(
            report,
            encoding="utf-8",
        )

        print("")
        print(
            "OUTPUTS",
            flush=True,
        )

        print(
            OUT_JSON,
            flush=True,
        )

        print(
            OUT_TXT,
            flush=True,
        )

        print(
            FORMAL_DB,
            flush=True,
        )

        print(
            GATE_MANIFEST,
            flush=True,
        )

        print(
            GATE_MANIFEST_SHA,
            flush=True,
        )

    finally:
        base_conn.close()
        formal_conn.close()

    return 0


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--verify-existing",
        action="store_true",
    )

    return parser


def main():
    args = build_parser().parse_args()

    if args.verify_existing:
        return verify_existing()

    return main_parent()


if __name__ == "__main__":
    raise SystemExit(main())
