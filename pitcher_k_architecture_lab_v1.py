#!/usr/bin/env python3
"""
PITCHER_K_ARCHITECTURE_LAB_V1

Development-only architecture lab for the current Prop Edge pitcher-strikeout
incumbent.

Purpose
-------
Use the completed clean local-history baseline to isolate three existing
architecture hypotheses WITHOUT touching production and WITHOUT opening 2026
challenger outcomes.

Development window
------------------
2024: warm-up diagnostic
2025: primary development comparison

No 2026 challenger evaluation is performed.

Cells
-----
A0_CURRENT_INCUMBENT
    Exact incumbent reconstruction already stored in:
    /data/hr_model/pitcher_k_clean_baseline_a_work/baseline.sqlite

B0_NO_BVP_NUDGE
    Same architecture, but k_nudge is fixed at 1.0.
    This isolates K-H02 / the multiplicative same-pitcher BVP layer.

C0_NO_EXTRA_TTO_DECAY
    Same architecture and BVP nudge, but simulator TTO factors are:
        1.00 / 1.00 / 1.00
    instead of:
        1.00 / 0.94 / 0.85
    This isolates K-H01.

D0_FIXED_LINEUP_ACTIVATION
    Same architecture and BVP nudge, but the 45% lineup component activates
    when at least 5 batters have real usable K data:
        - handedness split >= 15 PA, or
        - season K rate >= 15 PA, or
        - H2H >= 2 PA
    instead of the current K-A01 behavior where n_data increments only for
    H2H-qualified batters.

Important discipline
--------------------
- Development only.
- No production change.
- No threshold tuning.
- No rescue tuning.
- No stacking.
- No 2026 challenger outcomes.
- Every challenger changes ONE architecture component only.
- A0 is reused from the already completed clean baseline.
- Common random numbers are used by reusing A0's frozen research seed for each
  paired challenger row, reducing Monte Carlo comparison noise.

Primary development metric
--------------------------
Mean CRPS on 2025.

Secondary protections
---------------------
- MAE
- RMSE
- absolute projection bias
- IQR coverage
- 10th-90th percentile coverage
- monthly CRPS stability

Development-survivor rule
-------------------------
A challenger is marked DEVELOPMENT_SURVIVOR only if ALL are true on 2025:

1. mean CRPS < A0
2. MAE <= A0
3. RMSE <= A0
4. absolute projection bias <= A0
5. CRPS wins in at least half of eligible months
   (eligible month = >= 200 paired rows)

This is NOT a production promotion gate.
A survivor may only become eligible to freeze for a separate future/prospective
formal holdout.

Memory safety
-------------
- no pandas
- no sklearn
- SQLite on disk
- one row / one cell simulation at a time
- 10,000 simulations per challenger row
- 512 MB Render target

Run
---
python -u pitcher_k_architecture_lab_v1.py 2>&1 | tee /data/hr_model/pitcher_k_architecture_lab_v1.log

Outputs
-------
/data/hr_model/pitcher_k_architecture_lab_v1_results.json
/data/hr_model/pitcher_k_architecture_lab_v1_report.txt
/data/hr_model/pitcher_k_architecture_lab_v1_work/lab.sqlite

Paste back
----------
LAB PREFLIGHT
A0 BASELINE REUSE
BUILD PAIRED CHALLENGER CELLS
2025 PRIMARY DEVELOPMENT COMPARISON
MONTHLY STABILITY
DEVELOPMENT SURVIVOR GATE
FINAL DEVELOPMENT VERDICT
"""

import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# =============================================================================
# Paths / constants
# =============================================================================

HR_DIR = Path("/data/hr_model")

BASELINE_DB = (
    HR_DIR
    / "pitcher_k_clean_baseline_a_work"
    / "baseline.sqlite"
)

WORK_DIR = HR_DIR / "pitcher_k_architecture_lab_v1_work"
LAB_DB = WORK_DIR / "lab.sqlite"

OUT_JSON = HR_DIR / "pitcher_k_architecture_lab_v1_results.json"
OUT_TXT = HR_DIR / "pitcher_k_architecture_lab_v1_report.txt"

SIMS = 10000

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

A0_TTO = (1.00, 0.94, 0.85)
NO_TTO_DECAY = (1.00, 1.00, 1.00)

ELIGIBLE_MONTH_MIN_ROWS = 200

CELLS = {
    "A0_CURRENT_INCUMBENT": {
        "description": (
            "Exact already-built clean incumbent baseline."
        ),
    },
    "B0_NO_BVP_NUDGE": {
        "description": (
            "Same incumbent architecture with k_nudge fixed at 1.0."
        ),
    },
    "C0_NO_EXTRA_TTO_DECAY": {
        "description": (
            "Same incumbent architecture with TTO factors 1.00/1.00/1.00."
        ),
    },
    "D0_FIXED_LINEUP_ACTIVATION": {
        "description": (
            "Same incumbent architecture, but lineup activates when >=5 "
            "batters have real usable K data rather than >=5 H2H-qualified "
            "batters."
        ),
    },
}


# =============================================================================
# Helpers
# =============================================================================

def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_json_load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def clip(value, lo, hi):
    return max(lo, min(hi, value))


def connect_baseline():
    if not BASELINE_DB.exists():
        raise RuntimeError(
            f"Missing clean baseline DB: {BASELINE_DB}"
        )

    conn = sqlite3.connect(
        f"file:{BASELINE_DB}?mode=ro",
        uri=True,
        timeout=60,
    )
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def connect_lab():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(LAB_DB), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-24000")
    conn.execute("PRAGMA busy_timeout=60000")

    return conn


def baseline_schema_check(conn):
    required_tables = {
        "pa",
        "pitcher_game",
        "starter_games",
        "historical_rows",
    }

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }

    missing = sorted(required_tables - tables)

    if missing:
        raise RuntimeError(
            "Baseline DB missing required tables: "
            + ", ".join(missing)
        )

    historical_cols = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(historical_rows)"
        )
    }

    required_cols = {
        "year",
        "game_date",
        "game_id",
        "pitcher_id",
        "actual_k",
        "actual_bf",
        "research_seed",
        "sim_mean",
        "sim_iqr",
        "sim_q10",
        "sim_q90",
        "sim_crps",
        "actual_in_iqr",
        "actual_in_10_90",
        "k_nudge",
        "lineup_component_used",
    }

    missing_cols = sorted(required_cols - historical_cols)

    if missing_cols:
        raise RuntimeError(
            "historical_rows missing required columns: "
            + ", ".join(missing_cols)
        )

    return {
        "tables": sorted(tables),
        "historical_columns": sorted(historical_cols),
        "missing_tables": missing,
        "missing_historical_columns": missing_cols,
    }


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
# Simulator
# =============================================================================

def vectorized_ksim(
    k_per_bf,
    expected_bf,
    start_k_rates,
    *,
    seed,
    actual_k,
    tto_factors,
    sims=SIMS,
):
    rng = np.random.RandomState(int(seed))

    if start_k_rates and len(start_k_rates) >= 4:
        pool = np.array(start_k_rates, dtype=np.float64)
        pool = 0.7 * pool + 0.3 * float(k_per_bf)
    else:
        pool = np.array(
            [float(k_per_bf)],
            dtype=np.float64,
        )

    bf = np.rint(
        rng.normal(float(expected_bf), BF_SD, size=sims)
    ).astype(np.int16)

    bf = np.clip(bf, BF_MIN, BF_MAX)

    pool_idx = rng.randint(0, len(pool), size=sims)
    k_sample = pool[pool_idx]

    n1 = np.minimum(bf, 9)
    n2 = np.minimum(np.maximum(bf - 9, 0), 9)
    n3 = np.maximum(bf - 18, 0)

    t1, t2, t3 = tto_factors

    p1 = np.clip(k_sample * t1, 0.02, 0.60)
    p2 = np.clip(k_sample * t2, 0.02, 0.60)
    p3 = np.clip(k_sample * t3, 0.02, 0.60)

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
                sorted_results - float(actual_k)
            )
        )
    )

    half_pairwise = float(
        np.sum(coeff * sorted_results)
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
# Lab DB
# =============================================================================

def create_lab_table(conn):
    conn.execute("DROP TABLE IF EXISTS lab_rows")

    conn.execute(
        """
        CREATE TABLE lab_rows (
            cell TEXT NOT NULL,
            year INTEGER NOT NULL,
            game_date TEXT NOT NULL,
            game_id TEXT NOT NULL,
            pitcher_id TEXT NOT NULL,

            actual_k INTEGER NOT NULL,
            actual_bf INTEGER NOT NULL,

            expected_bf REAL NOT NULL,
            prior_qualifying_outings INTEGER NOT NULL,

            current_h2h_qualified_batters INTEGER NOT NULL,
            fixed_usable_data_batters INTEGER NOT NULL,
            current_lineup_used INTEGER NOT NULL,
            fixed_lineup_used INTEGER NOT NULL,

            current_k_nudge REAL NOT NULL,

            sim_mean REAL NOT NULL,
            sim_median REAL NOT NULL,
            sim_iqr REAL NOT NULL,
            sim_q10 REAL NOT NULL,
            sim_q90 REAL NOT NULL,
            sim_crps REAL NOT NULL,
            actual_in_iqr INTEGER NOT NULL,
            actual_in_10_90 INTEGER NOT NULL,

            research_seed INTEGER NOT NULL,

            PRIMARY KEY (
                cell,
                game_date,
                game_id,
                pitcher_id
            )
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_lab_rows_cell_year_date
        ON lab_rows(cell, year, game_date)
        """
    )

    conn.commit()


def copy_a0_rows(base_conn, lab_conn):
    rows = list(
        base_conn.execute(
            """
            SELECT
                year,
                game_date,
                game_id,
                pitcher_id,
                actual_k,
                actual_bf,
                expected_bf,
                prior_qualifying_outings,
                h2h_qualified_batters,
                lineup_component_used,
                k_nudge,
                sim_mean,
                sim_median,
                sim_iqr,
                sim_q10,
                sim_q90,
                sim_crps,
                actual_in_iqr,
                actual_in_10_90,
                research_seed
            FROM historical_rows
            WHERE year IN (2024, 2025)
            ORDER BY game_date, game_id, pitcher_id
            """
        )
    )

    insert_sql = """
        INSERT INTO lab_rows (
            cell,
            year,
            game_date,
            game_id,
            pitcher_id,

            actual_k,
            actual_bf,

            expected_bf,
            prior_qualifying_outings,

            current_h2h_qualified_batters,
            fixed_usable_data_batters,
            current_lineup_used,
            fixed_lineup_used,

            current_k_nudge,

            sim_mean,
            sim_median,
            sim_iqr,
            sim_q10,
            sim_q90,
            sim_crps,
            actual_in_iqr,
            actual_in_10_90,

            research_seed
        )
        VALUES (
            ?, ?, ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?
        )
    """

    payload = []

    for row in rows:
        (
            year,
            game_date,
            game_id,
            pitcher_id,
            actual_k,
            actual_bf,
            expected_bf,
            prior_outings,
            h2h_qualified,
            current_lineup_used,
            k_nudge,
            sim_mean,
            sim_median,
            sim_iqr,
            sim_q10,
            sim_q90,
            sim_crps,
            actual_in_iqr,
            actual_in_10_90,
            seed,
        ) = row

        # fixed_usable_data_batters and fixed_lineup_used are filled with -1 for
        # A0 because the already-completed baseline did not store that diagnostic.
        payload.append(
            (
                "A0_CURRENT_INCUMBENT",
                int(year),
                str(game_date),
                str(game_id),
                str(pitcher_id),

                int(actual_k),
                int(actual_bf),

                float(expected_bf),
                int(prior_outings),

                int(h2h_qualified),
                -1,
                int(current_lineup_used),
                -1,

                float(k_nudge),

                float(sim_mean),
                float(sim_median),
                float(sim_iqr),
                float(sim_q10),
                float(sim_q90),
                float(sim_crps),
                int(actual_in_iqr),
                int(actual_in_10_90),

                int(seed),
            )
        )

        if len(payload) >= 500:
            lab_conn.executemany(insert_sql, payload)
            lab_conn.commit()
            payload.clear()

    if payload:
        lab_conn.executemany(insert_sql, payload)
        lab_conn.commit()

    return len(rows)


# =============================================================================
# Paired challenger reconstruction
# =============================================================================

def build_challengers(base_conn, lab_conn):
    a0_keys = {}

    for (
        year,
        game_date,
        game_id,
        pitcher_id,
        actual_k,
        actual_bf,
        research_seed,
    ) in base_conn.execute(
        """
        SELECT
            year,
            game_date,
            game_id,
            pitcher_id,
            actual_k,
            actual_bf,
            research_seed
        FROM historical_rows
        WHERE year IN (2024, 2025)
        """
    ):
        a0_keys[
            (
                str(game_date),
                str(game_id),
                str(pitcher_id),
            )
        ] = {
            "year": int(year),
            "actual_k": int(actual_k),
            "actual_bf": int(actual_bf),
            "research_seed": int(research_seed),
        }

    pitcher_outings = defaultdict(list)

    hand_state = defaultdict(lambda: [0, 0])
    season_state = defaultdict(lambda: [0, 0])
    h2h_state = defaultdict(lambda: [0, 0])

    dates = [
        row[0]
        for row in base_conn.execute(
            """
            SELECT DISTINCT game_date
            FROM pa
            WHERE SUBSTR(game_date, 1, 4) IN ('2024', '2025')
            ORDER BY game_date
            """
        )
    ]

    insert_sql = """
        INSERT INTO lab_rows (
            cell,
            year,
            game_date,
            game_id,
            pitcher_id,

            actual_k,
            actual_bf,

            expected_bf,
            prior_qualifying_outings,

            current_h2h_qualified_batters,
            fixed_usable_data_batters,
            current_lineup_used,
            fixed_lineup_used,

            current_k_nudge,

            sim_mean,
            sim_median,
            sim_iqr,
            sim_q10,
            sim_q90,
            sim_crps,
            actual_in_iqr,
            actual_in_10_90,

            research_seed
        )
        VALUES (
            ?, ?, ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?
        )
    """

    batch = []
    challenger_rows = defaultdict(int)
    eligible_keys_seen = set()

    for date_idx, game_date in enumerate(dates, 1):
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

        for (
            game_id,
            pitcher_id,
            actual_bf,
            actual_k,
            p_throws,
        ) in starters:
            key = (
                str(game_date),
                str(game_id),
                str(pitcher_id),
            )

            a0_meta = a0_keys.get(key)

            if a0_meta is None:
                continue

            profile = pitcher_profile(
                pitcher_outings[
                    (year, str(pitcher_id))
                ]
            )

            if profile is None:
                raise RuntimeError(
                    "Paired reconstruction profile mismatch for "
                    f"{key}"
                )

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
                raise RuntimeError(
                    "Paired reconstruction lineup mismatch for "
                    f"{key}"
                )

            throws = str(p_throws or "R")

            rates = []

            h2h_qualified = 0
            usable_data_batters = 0

            bvp_tot_pa = 0
            bvp_tot_k = 0

            for batter_id in lineup[:9]:
                hand_pa, hand_k = hand_state[
                    (year, batter_id, throws)
                ]

                season_pa, season_k = season_state[
                    (year, batter_id)
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
                    (batter_id, str(pitcher_id))
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
                lineup_avg_k * profile["expected_bf"]
                if lineup_avg_k is not None
                else None
            )

            current_lineup_used = int(
                lineup_expected_k is not None
                and lineup_expected_k > 0
                and h2h_qualified
                >= MIN_LINEUP_DATA_BATTERS
            )

            fixed_lineup_used = int(
                lineup_expected_k is not None
                and lineup_expected_k > 0
                and usable_data_batters
                >= MIN_LINEUP_DATA_BATTERS
            )

            pitcher_projection = (
                profile["final_kbf"]
                * profile["expected_bf"]
            )

            if current_lineup_used:
                blended_current = (
                    PITCHER_WEIGHT * pitcher_projection
                    + LINEUP_WEIGHT * lineup_expected_k
                )
            else:
                blended_current = pitcher_projection

            if fixed_lineup_used:
                blended_fixed = (
                    PITCHER_WEIGHT * pitcher_projection
                    + LINEUP_WEIGHT * lineup_expected_k
                )
            else:
                blended_fixed = pitcher_projection

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

                current_k_nudge = round(
                    clip(
                        raw_nudge,
                        K_NUDGE_MIN,
                        K_NUDGE_MAX,
                    ),
                    3,
                )

            else:
                current_k_nudge = 1.0

            base_start_rates = list(
                profile["per_outing_rates"]
            )

            seed = a0_meta["research_seed"]

            # -------------------------------------------------------------
            # B0: remove BVP nudge only.
            # -------------------------------------------------------------
            b0_kbf = (
                blended_current
                / profile["expected_bf"]
            )

            b0_start_rates = list(
                base_start_rates
            )

            b0_sim = vectorized_ksim(
                b0_kbf,
                profile["expected_bf"],
                b0_start_rates,
                seed=seed,
                actual_k=int(actual_k),
                tto_factors=A0_TTO,
                sims=SIMS,
            )

            # -------------------------------------------------------------
            # C0: preserve nudge, remove extra TTO decay only.
            # -------------------------------------------------------------
            c0_kbf = (
                (
                    blended_current
                    / profile["expected_bf"]
                )
                * current_k_nudge
            )

            c0_start_rates = list(
                base_start_rates
            )

            if (
                c0_start_rates
                and current_k_nudge != 1.0
            ):
                c0_start_rates = [
                    rate * current_k_nudge
                    for rate in c0_start_rates
                ]

            c0_sim = vectorized_ksim(
                c0_kbf,
                profile["expected_bf"],
                c0_start_rates,
                seed=seed,
                actual_k=int(actual_k),
                tto_factors=NO_TTO_DECAY,
                sims=SIMS,
            )

            # -------------------------------------------------------------
            # D0: fixed lineup activation only.
            # -------------------------------------------------------------
            d0_kbf = (
                (
                    blended_fixed
                    / profile["expected_bf"]
                )
                * current_k_nudge
            )

            d0_start_rates = list(
                base_start_rates
            )

            if (
                d0_start_rates
                and current_k_nudge != 1.0
            ):
                d0_start_rates = [
                    rate * current_k_nudge
                    for rate in d0_start_rates
                ]

            d0_sim = vectorized_ksim(
                d0_kbf,
                profile["expected_bf"],
                d0_start_rates,
                seed=seed,
                actual_k=int(actual_k),
                tto_factors=A0_TTO,
                sims=SIMS,
            )

            for cell, sim in (
                ("B0_NO_BVP_NUDGE", b0_sim),
                ("C0_NO_EXTRA_TTO_DECAY", c0_sim),
                (
                    "D0_FIXED_LINEUP_ACTIVATION",
                    d0_sim,
                ),
            ):
                batch.append(
                    (
                        cell,
                        year,
                        str(game_date),
                        str(game_id),
                        str(pitcher_id),

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
                        int(current_lineup_used),
                        int(fixed_lineup_used),

                        float(current_k_nudge),

                        float(sim["mean"]),
                        float(sim["median"]),
                        float(sim["iqr"]),
                        float(sim["q10"]),
                        float(sim["q90"]),
                        float(sim["crps"]),
                        int(sim["actual_in_iqr"]),
                        int(sim["actual_in_10_90"]),

                        int(seed),
                    )
                )

                challenger_rows[cell] += 1

            eligible_keys_seen.add(key)

            if len(batch) >= 300:
                lab_conn.executemany(
                    insert_sql,
                    batch,
                )
                lab_conn.commit()
                batch.clear()

        # D-1 update only after all games for date are scored.

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
                (year, str(pitcher_id))
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
                f"Challenger progress: "
                f"{date_idx}/{len(dates)} dates | "
                f"paired_rows={len(eligible_keys_seen):,}",
                flush=True,
            )

    if batch:
        lab_conn.executemany(
            insert_sql,
            batch,
        )
        lab_conn.commit()

    a0_count = len(a0_keys)

    if len(eligible_keys_seen) != a0_count:
        raise RuntimeError(
            "Paired row mismatch: "
            f"challenger_keys={len(eligible_keys_seen):,} "
            f"A0_keys={a0_count:,}"
        )

    for cell in (
        "B0_NO_BVP_NUDGE",
        "C0_NO_EXTRA_TTO_DECAY",
        "D0_FIXED_LINEUP_ACTIVATION",
    ):
        if challenger_rows[cell] != a0_count:
            raise RuntimeError(
                f"{cell} row mismatch: "
                f"{challenger_rows[cell]:,} vs {a0_count:,}"
            )

    return {
        "a0_rows": a0_count,
        "paired_keys": len(eligible_keys_seen),
        "challenger_rows": dict(
            sorted(challenger_rows.items())
        ),
    }


# =============================================================================
# Metrics
# =============================================================================

def fetch_rows(conn, cell, year=None):
    sql = """
        SELECT
            cell,
            year,
            game_date,
            game_id,
            pitcher_id,
            actual_k,
            actual_bf,
            expected_bf,
            prior_qualifying_outings,
            current_h2h_qualified_batters,
            fixed_usable_data_batters,
            current_lineup_used,
            fixed_lineup_used,
            current_k_nudge,
            sim_mean,
            sim_median,
            sim_iqr,
            sim_q10,
            sim_q90,
            sim_crps,
            actual_in_iqr,
            actual_in_10_90
        FROM lab_rows
        WHERE cell = ?
    """

    params = [cell]

    if year is not None:
        sql += " AND year = ?"
        params.append(int(year))

    cols = [
        "cell",
        "year",
        "game_date",
        "game_id",
        "pitcher_id",
        "actual_k",
        "actual_bf",
        "expected_bf",
        "prior_qualifying_outings",
        "current_h2h_qualified_batters",
        "fixed_usable_data_batters",
        "current_lineup_used",
        "fixed_lineup_used",
        "current_k_nudge",
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


def metric_block(rows):
    rows = list(rows)

    if not rows:
        return {"rows": 0}

    n = len(rows)

    errors = [
        float(row["sim_mean"])
        - float(row["actual_k"])
        for row in rows
    ]

    return {
        "rows": n,
        "actual_mean_k": (
            sum(float(row["actual_k"]) for row in rows)
            / n
        ),
        "projected_mean_k": (
            sum(float(row["sim_mean"]) for row in rows)
            / n
        ),
        "projection_bias": sum(errors) / n,
        "absolute_projection_bias": abs(
            sum(errors) / n
        ),
        "mae": (
            sum(abs(error) for error in errors)
            / n
        ),
        "rmse": math.sqrt(
            sum(error * error for error in errors)
            / n
        ),
        "mean_crps": (
            sum(float(row["sim_crps"]) for row in rows)
            / n
        ),
        "iqr_coverage": (
            sum(int(row["actual_in_iqr"]) for row in rows)
            / n
        ),
        "q10_q90_coverage": (
            sum(
                int(row["actual_in_10_90"])
                for row in rows
            )
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
    }


def monthly_metrics(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[str(row["game_date"])[:7]].append(row)

    return {
        month: metric_block(groups[month])
        for month in sorted(groups)
    }


def delta_block(challenger, baseline):
    return {
        "mean_crps": (
            challenger["mean_crps"]
            - baseline["mean_crps"]
        ),
        "mean_crps_relative_pct": (
            (
                challenger["mean_crps"]
                - baseline["mean_crps"]
            )
            / baseline["mean_crps"]
            * 100.0
        ),
        "mae": (
            challenger["mae"]
            - baseline["mae"]
        ),
        "rmse": (
            challenger["rmse"]
            - baseline["rmse"]
        ),
        "absolute_projection_bias": (
            challenger["absolute_projection_bias"]
            - baseline["absolute_projection_bias"]
        ),
        "iqr_coverage": (
            challenger["iqr_coverage"]
            - baseline["iqr_coverage"]
        ),
        "q10_q90_coverage": (
            challenger["q10_q90_coverage"]
            - baseline["q10_q90_coverage"]
        ),
    }


def build_results(lab_conn):
    cell_results = {}

    for cell in CELLS:
        warmup_rows = fetch_rows(
            lab_conn,
            cell,
            WARMUP_YEAR,
        )

        primary_rows = fetch_rows(
            lab_conn,
            cell,
            PRIMARY_YEAR,
        )

        cell_results[cell] = {
            "description": CELLS[cell]["description"],
            "warmup_2024": metric_block(
                warmup_rows
            ),
            "primary_2025": metric_block(
                primary_rows
            ),
            "monthly_2025": monthly_metrics(
                primary_rows
            ),
        }

    a0 = cell_results[
        "A0_CURRENT_INCUMBENT"
    ]["primary_2025"]

    deltas = {}

    for cell in CELLS:
        if cell == "A0_CURRENT_INCUMBENT":
            continue

        deltas[cell] = delta_block(
            cell_results[cell]["primary_2025"],
            a0,
        )

    monthly_stability = {}
    survivor_gate = {}

    a0_monthly = cell_results[
        "A0_CURRENT_INCUMBENT"
    ]["monthly_2025"]

    for cell in CELLS:
        if cell == "A0_CURRENT_INCUMBENT":
            continue

        challenger_monthly = cell_results[
            cell
        ]["monthly_2025"]

        rows = {}
        eligible_months = 0
        crps_wins = 0

        for month in sorted(a0_monthly):
            base = a0_monthly[month]
            chal = challenger_monthly[month]

            eligible = (
                base["rows"]
                >= ELIGIBLE_MONTH_MIN_ROWS
            )

            delta = (
                chal["mean_crps"]
                - base["mean_crps"]
            )

            if eligible:
                eligible_months += 1

                if delta < 0:
                    crps_wins += 1

            rows[month] = {
                "rows": base["rows"],
                "eligible": eligible,
                "a0_crps": base["mean_crps"],
                "challenger_crps": (
                    chal["mean_crps"]
                ),
                "delta": delta,
                "challenger_win": delta < 0,
            }

        required_wins = (
            math.ceil(eligible_months / 2)
            if eligible_months > 0
            else 0
        )

        monthly_stability[cell] = {
            "eligible_months": eligible_months,
            "crps_wins": crps_wins,
            "required_wins": required_wins,
            "months": rows,
        }

        base = a0
        chal = cell_results[cell]["primary_2025"]

        gate = {
            "crps_improves": (
                chal["mean_crps"]
                < base["mean_crps"]
            ),
            "mae_nonworse": (
                chal["mae"]
                <= base["mae"]
            ),
            "rmse_nonworse": (
                chal["rmse"]
                <= base["rmse"]
            ),
            "absolute_bias_nonworse": (
                chal["absolute_projection_bias"]
                <= base["absolute_projection_bias"]
            ),
            "monthly_crps_stability_pass": (
                eligible_months > 0
                and crps_wins >= required_wins
            ),
        }

        gate["development_survivor"] = all(
            gate.values()
        )

        survivor_gate[cell] = gate

    survivors = [
        cell
        for cell, gate in survivor_gate.items()
        if gate["development_survivor"]
    ]

    # Rank only by primary CRPS. This is a development read, not promotion.
    ranked = sorted(
        (
            (
                cell,
                cell_results[cell][
                    "primary_2025"
                ]["mean_crps"],
            )
            for cell in CELLS
        ),
        key=lambda item: item[1],
    )

    if survivors:
        best_survivor = min(
            survivors,
            key=lambda cell: (
                cell_results[cell][
                    "primary_2025"
                ]["mean_crps"]
            ),
        )

        final_verdict = (
            "DEVELOPMENT_SURVIVOR_FOUND_"
            f"{best_survivor}_"
            "ELIGIBLE_FOR_SEPARATE_STABILITY_CONFIRMATION_"
            "NO_PRODUCTION_PROMOTION"
        )
    else:
        best_survivor = None

        final_verdict = (
            "NO_DEVELOPMENT_SURVIVOR_"
            "INCUMBENT_REMAINS_CHAMPION_"
            "NO_RESCUE_TUNING"
        )

    return {
        "cell_results": cell_results,
        "deltas_vs_a0_2025": deltas,
        "monthly_stability": monthly_stability,
        "development_survivor_gate": survivor_gate,
        "development_survivors": survivors,
        "best_survivor": best_survivor,
        "ranked_by_2025_crps": [
            {
                "rank": i + 1,
                "cell": cell,
                "mean_crps": crps,
            }
            for i, (cell, crps) in enumerate(
                ranked
            )
        ],
        "final_verdict": final_verdict,
    }


# =============================================================================
# Report
# =============================================================================

def fmt_metrics(label, block):
    return (
        f"{label}: "
        f"n={block['rows']:,} "
        f"actual={block['actual_mean_k']:.6f} "
        f"projected={block['projected_mean_k']:.6f} "
        f"bias={block['projection_bias']:+.6f} "
        f"mae={block['mae']:.6f} "
        f"rmse={block['rmse']:.6f} "
        f"crps={block['mean_crps']:.6f} "
        f"iqr={block['iqr_coverage']:.6f} "
        f"q10_q90={block['q10_q90_coverage']:.6f}"
    )


def build_report(payload):
    lines = []

    lines.append("PITCHER_K_ARCHITECTURE_LAB_V1")
    lines.append("=" * 29)

    lines.append("")
    lines.append("LAB PREFLIGHT")
    lines.append("-------------")
    lines.append(
        json.dumps(
            payload["preflight"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("A0 BASELINE REUSE")
    lines.append("-----------------")
    lines.append(
        json.dumps(
            payload["a0_reuse"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("BUILD PAIRED CHALLENGER CELLS")
    lines.append("------------------------------")
    lines.append(
        json.dumps(
            payload["paired_build"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("2025 PRIMARY DEVELOPMENT COMPARISON")
    lines.append("-----------------------------------")

    for cell in CELLS:
        lines.append(
            fmt_metrics(
                cell,
                payload["results"][
                    "cell_results"
                ][cell]["primary_2025"],
            )
        )

    lines.append("")
    lines.append("DELTAS VS A0")
    lines.append("------------")
    lines.append(
        json.dumps(
            payload["results"][
                "deltas_vs_a0_2025"
            ],
            indent=2,
        )
    )

    lines.append("")
    lines.append("MONTHLY STABILITY")
    lines.append("-----------------")
    lines.append(
        json.dumps(
            payload["results"][
                "monthly_stability"
            ],
            indent=2,
        )
    )

    lines.append("")
    lines.append("DEVELOPMENT SURVIVOR GATE")
    lines.append("-------------------------")
    lines.append(
        json.dumps(
            payload["results"][
                "development_survivor_gate"
            ],
            indent=2,
        )
    )

    lines.append("")
    lines.append("RANKED BY 2025 CRPS")
    lines.append("-------------------")
    lines.append(
        json.dumps(
            payload["results"][
                "ranked_by_2025_crps"
            ],
            indent=2,
        )
    )

    lines.append("")
    lines.append("FINAL DEVELOPMENT VERDICT")
    lines.append("-------------------------")
    lines.append(
        payload["results"]["final_verdict"]
    )

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    print("PITCHER_K_ARCHITECTURE_LAB_V1", flush=True)
    print("=============================", flush=True)

    print("")
    print("LAB PREFLIGHT", flush=True)
    print("-------------", flush=True)

    base_conn = connect_baseline()
    lab_conn = connect_lab()

    try:
        schema = baseline_schema_check(base_conn)

        a0_2025_rows = base_conn.execute(
            """
            SELECT COUNT(*)
            FROM historical_rows
            WHERE year = 2025
            """
        ).fetchone()[0]

        a0_all_rows = base_conn.execute(
            """
            SELECT COUNT(*)
            FROM historical_rows
            WHERE year IN (2024, 2025)
            """
        ).fetchone()[0]

        preflight = {
            "baseline_db": str(BASELINE_DB),
            "baseline_db_exists": BASELINE_DB.exists(),
            "schema_ok": True,
            "a0_rows_2024_2025": int(a0_all_rows),
            "a0_rows_2025": int(a0_2025_rows),
            "development_years": [
                WARMUP_YEAR,
                PRIMARY_YEAR,
            ],
            "primary_development_year": PRIMARY_YEAR,
            "touch_2026": False,
            "external_api_calls": False,
            "production_changed": False,
            "challenger_cells": [
                cell
                for cell in CELLS
                if cell != "A0_CURRENT_INCUMBENT"
            ],
            "stacking": False,
            "threshold_tuning": False,
            "rescue_tuning": False,
            "common_random_numbers": True,
            "sims_per_challenger_row": SIMS,
        }

        print(
            json.dumps(preflight, indent=2),
            flush=True,
        )

        print("")
        print("A0 BASELINE REUSE", flush=True)
        print("-----------------", flush=True)

        create_lab_table(lab_conn)

        a0_copied = copy_a0_rows(
            base_conn,
            lab_conn,
        )

        a0_reuse = {
            "source": (
                "pitcher_k_clean_baseline_a_work/"
                "baseline.sqlite::historical_rows"
            ),
            "rows_copied": a0_copied,
            "resimulated": False,
            "reason": (
                "Reuse the already-completed clean incumbent exactly."
            ),
        }

        print(
            json.dumps(a0_reuse, indent=2),
            flush=True,
        )

        print("")
        print("BUILD PAIRED CHALLENGER CELLS", flush=True)
        print("------------------------------", flush=True)

        paired_build = build_challengers(
            base_conn,
            lab_conn,
        )

        print(
            json.dumps(paired_build, indent=2),
            flush=True,
        )

        results = build_results(lab_conn)

        print("")
        print("2025 PRIMARY DEVELOPMENT COMPARISON", flush=True)
        print("-----------------------------------", flush=True)

        for cell in CELLS:
            print(
                fmt_metrics(
                    cell,
                    results["cell_results"][
                        cell
                    ]["primary_2025"],
                ),
                flush=True,
            )

        print("")
        print("DELTAS VS A0", flush=True)
        print("------------", flush=True)
        print(
            json.dumps(
                results[
                    "deltas_vs_a0_2025"
                ],
                indent=2,
            ),
            flush=True,
        )

        print("")
        print("MONTHLY STABILITY", flush=True)
        print("-----------------", flush=True)
        print(
            json.dumps(
                results["monthly_stability"],
                indent=2,
            ),
            flush=True,
        )

        print("")
        print("DEVELOPMENT SURVIVOR GATE", flush=True)
        print("-------------------------", flush=True)
        print(
            json.dumps(
                results[
                    "development_survivor_gate"
                ],
                indent=2,
            ),
            flush=True,
        )

        print("")
        print("FINAL DEVELOPMENT VERDICT", flush=True)
        print("-------------------------", flush=True)
        print(
            results["final_verdict"],
            flush=True,
        )

        payload = {
            "script": "PITCHER_K_ARCHITECTURE_LAB_V1",
            "generated_at_utc": now_utc(),
            "mode": "DEVELOPMENT_ONLY_NO_2026",
            "preflight": preflight,
            "baseline_schema": schema,
            "cells": CELLS,
            "development_survivor_rule": {
                "primary_metric": "mean_crps_2025",
                "conditions": [
                    "mean_crps < A0",
                    "mae <= A0",
                    "rmse <= A0",
                    "absolute_projection_bias <= A0",
                    (
                        "CRPS wins in at least half of eligible months; "
                        f"eligible month requires >= "
                        f"{ELIGIBLE_MONTH_MIN_ROWS} rows"
                    ),
                ],
                "production_promotion_authorized": False,
            },
            "a0_reuse": a0_reuse,
            "paired_build": paired_build,
            "results": results,
        }

        OUT_JSON.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        report = build_report(payload)

        OUT_TXT.write_text(
            report,
            encoding="utf-8",
        )

        print("")
        print("OUTPUTS", flush=True)
        print(OUT_JSON, flush=True)
        print(OUT_TXT, flush=True)
        print(LAB_DB, flush=True)

    finally:
        base_conn.close()
        lab_conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
