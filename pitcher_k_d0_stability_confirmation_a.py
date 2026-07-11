#!/usr/bin/env python3
"""
PITCHER_K_D0_STABILITY_CONFIRMATION_A

D0-only breadth and intervention stability confirmation for the pitcher-K model.

Candidate
---------
D0_FIXED_LINEUP_ACTIVATION

Exact isolated change
---------------------
Current A0 behavior:
    Activate the 45% lineup component only when at least 5 opposing batters
    have qualifying H2H strikeout history against this exact pitcher.

D0 behavior:
    Activate the same 45% lineup component when at least 5 opposing batters
    have genuinely usable K information from any of:
        - handedness split with >= 15 prior PA
        - current-season overall K rate with >= 15 prior PA
        - H2H strikeout history with >= 2 prior PA

Everything else remains unchanged:
    - pitcher profile
    - 85% recent / 15% season anchor
    - expected BF model
    - last-12 outing volatility pool
    - 55% pitcher / 45% lineup blend
    - 40% H2H / 60% general batter blend
    - BVP k_nudge
    - TTO decay
    - Monte Carlo distribution
    - thresholds

Purpose
-------
Confirm that D0's 2025 development win is broad and actually comes from the
rows where D0 changes the architecture.

This is NOT an independent formal validation because D0 was selected using
2025 development results. It is a breadth/stability confirmation only.

No 2026 challenger outcomes are opened.
No production change.
No threshold tuning.
No rescue tuning.
No stacking.

Data sources
------------
/data/hr_model/pitcher_k_clean_baseline_a_work/baseline.sqlite
/data/hr_model/pitcher_k_architecture_lab_v1_work/lab.sqlite

Primary confirmation window
---------------------------
2025

Supporting warm-up diagnostic
-----------------------------
2024

The script does NOT resimulate. It reuses the already-frozen A0 and D0 paired
simulation rows from the completed architecture lab.

Primary questions
-----------------
1. Does D0 remain better on overall 2025 CRPS?
2. Are MAE, RMSE, and absolute bias still protected?
3. Is the monthly advantage broad?
4. Does the gain appear in the exact intervention subset:
       A0 lineup OFF -> D0 lineup ON
5. Is the benefit stable across predeclared pitcher/workload/time slices?
6. Is there any materially damaged eligible slice?
7. Is D0 correcting a broad architecture issue or only exploiting a tiny subset?

Breadth confirmation gate
-------------------------
A D0 stability confirmation passes only if ALL hard conditions are true:

1. 2025 overall CRPS improves vs A0.
2. 2025 MAE does not worsen.
3. 2025 RMSE does not worsen.
4. 2025 absolute projection bias does not worsen.
5. D0 wins CRPS in at least half of eligible 2025 months
   (eligible month: >= 200 paired rows).
6. The 2025 intervention subset has at least 200 rows.
7. The 2025 intervention subset improves CRPS vs A0.
8. The 2025 intervention subset MAE does not worsen.
9. Across all predeclared eligible 2025 slice buckets (>= 200 rows),
   at least 60% are CRPS nonworse vs A0.
10. No eligible 2025 slice may worsen CRPS by more than 5% relative to A0.

2024 is supporting/advisory only because local H2H history begins in 2024.

Possible final verdicts
-----------------------
PASS:
    D0_FIXED_LINEUP_ACTIVATION_BROADLY_STABLE_DEVELOPMENT_SURVIVOR
    _ELIGIBLE_TO_FREEZE_FOR_SEPARATE_FUTURE_OR_PROSPECTIVE_HOLDOUT
    _NO_PRODUCTION_PROMOTION

FAIL:
    D0_FIXED_LINEUP_ACTIVATION_FAILS_STABILITY_CONFIRMATION
    _FREEZE_NO_RESCUE_TUNING

Memory safety
-------------
- no pandas
- no sklearn
- no resimulation
- paired SQLite reads
- 512 MB Render target

Run
---
python -u pitcher_k_d0_stability_confirmation_a.py 2>&1 | tee /data/hr_model/pitcher_k_d0_stability_confirmation_a.log

Outputs
-------
/data/hr_model/pitcher_k_d0_stability_confirmation_a_results.json
/data/hr_model/pitcher_k_d0_stability_confirmation_a_report.txt

Paste back
----------
STABILITY PREFLIGHT
PAIRED ROW INTEGRITY
2024 SUPPORTING WARM-UP
2025 PRIMARY CONFIRMATION
INTERVENTION SUBSET
MONTHLY STABILITY
SLICE BREADTH
BREADTH CONFIRMATION GATE
FINAL STABILITY VERDICT
"""

import json
import math
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# =============================================================================
# Paths / locked constants
# =============================================================================

HR_DIR = Path("/data/hr_model")

BASELINE_DB = (
    HR_DIR
    / "pitcher_k_clean_baseline_a_work"
    / "baseline.sqlite"
)

LAB_DB = (
    HR_DIR
    / "pitcher_k_architecture_lab_v1_work"
    / "lab.sqlite"
)

ARCH_LAB_RESULTS = (
    HR_DIR
    / "pitcher_k_architecture_lab_v1_results.json"
)

OUT_JSON = (
    HR_DIR
    / "pitcher_k_d0_stability_confirmation_a_results.json"
)

OUT_TXT = (
    HR_DIR
    / "pitcher_k_d0_stability_confirmation_a_report.txt"
)

A0_CELL = "A0_CURRENT_INCUMBENT"
D0_CELL = "D0_FIXED_LINEUP_ACTIVATION"

PRIMARY_YEAR = 2025
WARMUP_YEAR = 2024

ELIGIBLE_MONTH_MIN_ROWS = 200
ELIGIBLE_SLICE_MIN_ROWS = 200

MIN_SLICE_NONWORSE_FRACTION = 0.60
MAX_ALLOWED_RELATIVE_CRPS_WORSENING_PCT = 5.0

MIN_INTERVENTION_ROWS = 200

TIE_EPS = 1e-12


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


def connect_ro(path):
    if not path.exists():
        raise RuntimeError(f"Missing required file: {path}")

    conn = sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True,
        timeout=60,
    )
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def check_tables(conn, required):
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }

    missing = sorted(set(required) - tables)

    if missing:
        raise RuntimeError(
            "Missing required tables: " + ", ".join(missing)
        )

    return sorted(tables)


# =============================================================================
# Paired data load
# =============================================================================

def load_pitcher_hand_map(base_conn):
    return {
        (
            str(game_date),
            str(game_id),
            str(pitcher_id),
        ): str(pitcher_throws or "unknown")
        for (
            game_date,
            game_id,
            pitcher_id,
            pitcher_throws,
        ) in base_conn.execute(
            """
            SELECT
                game_date,
                game_id,
                pitcher_id,
                pitcher_throws
            FROM historical_rows
            WHERE year IN (?, ?)
            """,
            (WARMUP_YEAR, PRIMARY_YEAR),
        )
    }


def load_paired_rows(base_conn, lab_conn):
    hand_map = load_pitcher_hand_map(base_conn)

    sql = """
        SELECT
            a.year,
            a.game_date,
            a.game_id,
            a.pitcher_id,

            a.actual_k,
            a.actual_bf,
            a.expected_bf,
            a.prior_qualifying_outings,

            d.current_h2h_qualified_batters,
            d.fixed_usable_data_batters,
            d.current_lineup_used,
            d.fixed_lineup_used,
            d.current_k_nudge,

            a.sim_mean,
            a.sim_median,
            a.sim_iqr,
            a.sim_q10,
            a.sim_q90,
            a.sim_crps,
            a.actual_in_iqr,
            a.actual_in_10_90,

            d.sim_mean,
            d.sim_median,
            d.sim_iqr,
            d.sim_q10,
            d.sim_q90,
            d.sim_crps,
            d.actual_in_iqr,
            d.actual_in_10_90

        FROM lab_rows a
        JOIN lab_rows d
          ON d.game_date = a.game_date
         AND d.game_id = a.game_id
         AND d.pitcher_id = a.pitcher_id

        WHERE a.cell = ?
          AND d.cell = ?
          AND a.year IN (?, ?)

        ORDER BY a.game_date, a.game_id, a.pitcher_id
    """

    rows = []

    for raw in lab_conn.execute(
        sql,
        (
            A0_CELL,
            D0_CELL,
            WARMUP_YEAR,
            PRIMARY_YEAR,
        ),
    ):
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
            usable_data_batters,
            current_lineup_used,
            fixed_lineup_used,
            current_k_nudge,

            a0_mean,
            a0_median,
            a0_iqr,
            a0_q10,
            a0_q90,
            a0_crps,
            a0_in_iqr,
            a0_in_10_90,

            d0_mean,
            d0_median,
            d0_iqr,
            d0_q10,
            d0_q90,
            d0_crps,
            d0_in_iqr,
            d0_in_10_90,
        ) = raw

        key = (
            str(game_date),
            str(game_id),
            str(pitcher_id),
        )

        pitcher_hand = hand_map.get(key, "unknown")

        current_lineup_used = int(current_lineup_used)
        fixed_lineup_used = int(fixed_lineup_used)

        if current_lineup_used == 0 and fixed_lineup_used == 1:
            intervention_status = "newly_activated"
        elif current_lineup_used == 1 and fixed_lineup_used == 1:
            intervention_status = "unchanged_on"
        elif current_lineup_used == 0 and fixed_lineup_used == 0:
            intervention_status = "unchanged_off"
        else:
            intervention_status = "current_on_fixed_off_unexpected"

        actual_k = float(actual_k)
        a0_mean = float(a0_mean)
        d0_mean = float(d0_mean)
        a0_crps = float(a0_crps)
        d0_crps = float(d0_crps)

        a0_error = a0_mean - actual_k
        d0_error = d0_mean - actual_k

        row = {
            "year": int(year),
            "game_date": str(game_date),
            "game_id": str(game_id),
            "pitcher_id": str(pitcher_id),
            "pitcher_hand": pitcher_hand,

            "actual_k": actual_k,
            "actual_bf": float(actual_bf),
            "expected_bf": float(expected_bf),
            "prior_qualifying_outings": int(prior_outings),

            "current_h2h_qualified_batters": int(h2h_qualified),
            "fixed_usable_data_batters": int(usable_data_batters),
            "current_lineup_used": current_lineup_used,
            "fixed_lineup_used": fixed_lineup_used,
            "current_k_nudge": float(current_k_nudge),
            "intervention_status": intervention_status,

            "a0_mean": a0_mean,
            "a0_median": float(a0_median),
            "a0_iqr": float(a0_iqr),
            "a0_q10": float(a0_q10),
            "a0_q90": float(a0_q90),
            "a0_crps": a0_crps,
            "a0_in_iqr": int(a0_in_iqr),
            "a0_in_10_90": int(a0_in_10_90),

            "d0_mean": d0_mean,
            "d0_median": float(d0_median),
            "d0_iqr": float(d0_iqr),
            "d0_q10": float(d0_q10),
            "d0_q90": float(d0_q90),
            "d0_crps": d0_crps,
            "d0_in_iqr": int(d0_in_iqr),
            "d0_in_10_90": int(d0_in_10_90),

            "a0_error": a0_error,
            "d0_error": d0_error,
            "a0_abs_error": abs(a0_error),
            "d0_abs_error": abs(d0_error),
            "a0_sq_error": a0_error * a0_error,
            "d0_sq_error": d0_error * d0_error,

            "crps_delta_d0_minus_a0": d0_crps - a0_crps,
            "projection_shift_d0_minus_a0": d0_mean - a0_mean,
            "abs_error_improvement_a0_minus_d0": (
                abs(a0_error) - abs(d0_error)
            ),
        }

        rows.append(row)

    return rows


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
        sum(row["actual_k"] for row in rows) / n
    )

    a0_mean = (
        sum(row["a0_mean"] for row in rows) / n
    )

    d0_mean = (
        sum(row["d0_mean"] for row in rows) / n
    )

    a0_bias = (
        sum(row["a0_error"] for row in rows) / n
    )

    d0_bias = (
        sum(row["d0_error"] for row in rows) / n
    )

    a0_mae = (
        sum(row["a0_abs_error"] for row in rows) / n
    )

    d0_mae = (
        sum(row["d0_abs_error"] for row in rows) / n
    )

    a0_rmse = math.sqrt(
        sum(row["a0_sq_error"] for row in rows) / n
    )

    d0_rmse = math.sqrt(
        sum(row["d0_sq_error"] for row in rows) / n
    )

    a0_crps = (
        sum(row["a0_crps"] for row in rows) / n
    )

    d0_crps = (
        sum(row["d0_crps"] for row in rows) / n
    )

    crps_delta = d0_crps - a0_crps

    relative_crps_delta_pct = (
        crps_delta / a0_crps * 100.0
        if a0_crps != 0
        else None
    )

    shifts = [
        row["projection_shift_d0_minus_a0"]
        for row in rows
    ]

    crps_wins = sum(
        1
        for row in rows
        if row["crps_delta_d0_minus_a0"] < -TIE_EPS
    )

    crps_losses = sum(
        1
        for row in rows
        if row["crps_delta_d0_minus_a0"] > TIE_EPS
    )

    crps_ties = n - crps_wins - crps_losses

    return {
        "rows": n,

        "actual_mean_k": actual_mean,

        "a0_projected_mean_k": a0_mean,
        "d0_projected_mean_k": d0_mean,

        "a0_projection_bias": a0_bias,
        "d0_projection_bias": d0_bias,
        "absolute_bias_delta_d0_minus_a0": (
            abs(d0_bias) - abs(a0_bias)
        ),

        "a0_mae": a0_mae,
        "d0_mae": d0_mae,
        "mae_delta_d0_minus_a0": d0_mae - a0_mae,

        "a0_rmse": a0_rmse,
        "d0_rmse": d0_rmse,
        "rmse_delta_d0_minus_a0": d0_rmse - a0_rmse,

        "a0_mean_crps": a0_crps,
        "d0_mean_crps": d0_crps,
        "crps_delta_d0_minus_a0": crps_delta,
        "relative_crps_delta_pct": relative_crps_delta_pct,

        "a0_iqr_coverage": (
            sum(row["a0_in_iqr"] for row in rows) / n
        ),
        "d0_iqr_coverage": (
            sum(row["d0_in_iqr"] for row in rows) / n
        ),

        "a0_q10_q90_coverage": (
            sum(row["a0_in_10_90"] for row in rows) / n
        ),
        "d0_q10_q90_coverage": (
            sum(row["d0_in_10_90"] for row in rows) / n
        ),

        "crps_row_wins": crps_wins,
        "crps_row_losses": crps_losses,
        "crps_row_ties": crps_ties,
        "crps_row_win_rate_ex_ties": (
            crps_wins / (crps_wins + crps_losses)
            if crps_wins + crps_losses > 0
            else None
        ),

        "mean_projection_shift_d0_minus_a0": (
            sum(shifts) / n
        ),
        "median_projection_shift_d0_minus_a0": (
            statistics.median(shifts)
        ),
        "positive_projection_shift_rows": sum(
            1 for shift in shifts if shift > TIE_EPS
        ),
        "negative_projection_shift_rows": sum(
            1 for shift in shifts if shift < -TIE_EPS
        ),
        "zero_projection_shift_rows": sum(
            1 for shift in shifts if abs(shift) <= TIE_EPS
        ),

        "mean_expected_bf": (
            sum(row["expected_bf"] for row in rows) / n
        ),
        "mean_actual_bf": (
            sum(row["actual_bf"] for row in rows) / n
        ),
    }


def grouped_metrics(rows, key_func):
    groups = defaultdict(list)

    for row in rows:
        groups[str(key_func(row))].append(row)

    return {
        key: metric_block(groups[key])
        for key in sorted(groups)
    }


# =============================================================================
# Slice definitions
# =============================================================================

def season_half_bucket(row):
    month = int(row["game_date"][5:7])

    if month <= 6:
        return "first_half_through_june"

    return "second_half_july_onward"


def pitcher_hand_bucket(row):
    hand = str(row["pitcher_hand"] or "unknown").upper()

    if hand in {"L", "R"}:
        return hand

    return "unknown"


def prior_outing_bucket(row):
    n = int(row["prior_qualifying_outings"])

    if n <= 4:
        return "3_4"

    if n <= 7:
        return "5_7"

    if n <= 11:
        return "8_11"

    return "12_plus"


def workload_bucket(row):
    bf = float(row["expected_bf"])

    if bf < 18:
        return "lt_18"

    if bf < 21:
        return "18_20.99"

    if bf < 24:
        return "21_23.99"

    return "24_plus"


def h2h_qualified_bucket(row):
    n = int(row["current_h2h_qualified_batters"])

    if n == 0:
        return "0"

    if n <= 2:
        return "1_2"

    if n <= 4:
        return "3_4"

    return "5_plus"


def usable_data_bucket(row):
    n = int(row["fixed_usable_data_batters"])

    if n <= 4:
        return "0_4"

    if n <= 6:
        return "5_6"

    return "7_9"


def current_lineup_bucket(row):
    return (
        "a0_lineup_on"
        if int(row["current_lineup_used"]) == 1
        else "a0_lineup_off"
    )


def intervention_bucket(row):
    return row["intervention_status"]


SLICE_DEFINITIONS = {
    "season_half": season_half_bucket,
    "pitcher_hand": pitcher_hand_bucket,
    "prior_qualifying_outings": prior_outing_bucket,
    "expected_bf": workload_bucket,
    "h2h_qualified_batters": h2h_qualified_bucket,
    "usable_data_batters": usable_data_bucket,
    "a0_lineup_state": current_lineup_bucket,
    "intervention_status": intervention_bucket,
}


# =============================================================================
# Confirmation analysis
# =============================================================================

def monthly_stability(primary_rows):
    groups = defaultdict(list)

    for row in primary_rows:
        groups[row["game_date"][:7]].append(row)

    months = {}

    eligible_months = 0
    crps_wins = 0

    for month in sorted(groups):
        block = metric_block(groups[month])

        eligible = (
            block["rows"] >= ELIGIBLE_MONTH_MIN_ROWS
        )

        challenger_win = (
            block["crps_delta_d0_minus_a0"] < 0
        )

        if eligible:
            eligible_months += 1

            if challenger_win:
                crps_wins += 1

        months[month] = {
            **block,
            "eligible": eligible,
            "challenger_win": challenger_win,
        }

    required_wins = (
        math.ceil(eligible_months / 2)
        if eligible_months > 0
        else 0
    )

    return {
        "eligible_months": eligible_months,
        "crps_wins": crps_wins,
        "required_wins": required_wins,
        "pass": (
            eligible_months > 0
            and crps_wins >= required_wins
        ),
        "months": months,
    }


def slice_breadth(primary_rows):
    all_slices = {}

    eligible_buckets = []
    nonworse_buckets = []
    materially_damaged_buckets = []

    for slice_name, key_func in SLICE_DEFINITIONS.items():
        grouped = grouped_metrics(
            primary_rows,
            key_func,
        )

        all_slices[slice_name] = grouped

        for bucket_name, block in grouped.items():
            eligible = (
                block["rows"] >= ELIGIBLE_SLICE_MIN_ROWS
            )

            if not eligible:
                continue

            key = f"{slice_name}|{bucket_name}"

            eligible_buckets.append(key)

            if block["crps_delta_d0_minus_a0"] <= 0:
                nonworse_buckets.append(key)

            rel = block["relative_crps_delta_pct"]

            if (
                rel is not None
                and rel
                > MAX_ALLOWED_RELATIVE_CRPS_WORSENING_PCT
            ):
                materially_damaged_buckets.append(
                    {
                        "slice_bucket": key,
                        "rows": block["rows"],
                        "relative_crps_delta_pct": rel,
                        "a0_mean_crps": block["a0_mean_crps"],
                        "d0_mean_crps": block["d0_mean_crps"],
                    }
                )

    fraction_nonworse = (
        len(nonworse_buckets) / len(eligible_buckets)
        if eligible_buckets
        else 0.0
    )

    return {
        "eligible_slice_bucket_count": len(eligible_buckets),
        "crps_nonworse_bucket_count": len(nonworse_buckets),
        "crps_nonworse_fraction": fraction_nonworse,
        "minimum_required_nonworse_fraction": (
            MIN_SLICE_NONWORSE_FRACTION
        ),
        "no_material_damage_threshold_relative_pct": (
            MAX_ALLOWED_RELATIVE_CRPS_WORSENING_PCT
        ),
        "eligible_buckets": eligible_buckets,
        "crps_nonworse_buckets": nonworse_buckets,
        "materially_damaged_buckets": materially_damaged_buckets,
        "slices": all_slices,
    }


def intervention_analysis(primary_rows):
    groups = grouped_metrics(
        primary_rows,
        intervention_bucket,
    )

    intervention_rows = [
        row
        for row in primary_rows
        if row["intervention_status"] == "newly_activated"
    ]

    unchanged_rows = [
        row
        for row in primary_rows
        if row["intervention_status"] != "newly_activated"
    ]

    intervention_block = metric_block(intervention_rows)
    unchanged_block = metric_block(unchanged_rows)

    total = len(primary_rows)

    return {
        "intervention_definition": (
            "A0 lineup component OFF and D0 lineup component ON"
        ),
        "intervention_rows": len(intervention_rows),
        "intervention_rate": (
            len(intervention_rows) / total
            if total
            else 0.0
        ),
        "intervention_metrics": intervention_block,
        "unchanged_rows": len(unchanged_rows),
        "unchanged_metrics": unchanged_block,
        "by_intervention_status": groups,
    }


def parity_and_integrity(rows):
    keys = [
        (
            row["year"],
            row["game_date"],
            row["game_id"],
            row["pitcher_id"],
        )
        for row in rows
    ]

    duplicates = len(keys) - len(set(keys))

    by_year = defaultdict(int)

    for row in rows:
        by_year[row["year"]] += 1

    impossible_transition_rows = [
        row
        for row in rows
        if row["intervention_status"]
        == "current_on_fixed_off_unexpected"
    ]

    return {
        "paired_rows": len(rows),
        "duplicate_paired_keys": duplicates,
        "rows_by_year": dict(sorted(by_year.items())),
        "unexpected_current_on_fixed_off_rows": (
            len(impossible_transition_rows)
        ),
        "integrity_pass": (
            duplicates == 0
            and len(impossible_transition_rows) == 0
        ),
    }


def build_gate(primary, monthly, intervention, breadth, integrity):
    intervention_block = intervention["intervention_metrics"]

    gate = {
        "paired_integrity_pass": (
            integrity["integrity_pass"]
        ),

        "overall_2025_crps_improves": (
            primary["crps_delta_d0_minus_a0"] < 0
        ),

        "overall_2025_mae_nonworse": (
            primary["mae_delta_d0_minus_a0"] <= 0
        ),

        "overall_2025_rmse_nonworse": (
            primary["rmse_delta_d0_minus_a0"] <= 0
        ),

        "overall_2025_absolute_bias_nonworse": (
            primary[
                "absolute_bias_delta_d0_minus_a0"
            ] <= 0
        ),

        "monthly_crps_stability_pass": (
            monthly["pass"]
        ),

        "intervention_subset_min_rows_pass": (
            intervention["intervention_rows"]
            >= MIN_INTERVENTION_ROWS
        ),

        "intervention_subset_crps_improves": (
            intervention_block.get(
                "crps_delta_d0_minus_a0",
                float("inf"),
            ) < 0
        ),

        "intervention_subset_mae_nonworse": (
            intervention_block.get(
                "mae_delta_d0_minus_a0",
                float("inf"),
            ) <= 0
        ),

        "slice_breadth_nonworse_fraction_pass": (
            breadth["eligible_slice_bucket_count"] > 0
            and breadth["crps_nonworse_fraction"]
            >= MIN_SLICE_NONWORSE_FRACTION
        ),

        "no_materially_damaged_eligible_slice_pass": (
            len(breadth["materially_damaged_buckets"]) == 0
        ),
    }

    gate["overall_stability_confirmation_pass"] = all(
        gate.values()
    )

    return gate


# =============================================================================
# Reporting
# =============================================================================

def fmt_block(label, block):
    if not block or block.get("rows", 0) == 0:
        return f"{label}: rows=0"

    return (
        f"{label}: "
        f"n={block['rows']:,} "
        f"A0_crps={block['a0_mean_crps']:.8f} "
        f"D0_crps={block['d0_mean_crps']:.8f} "
        f"delta={block['crps_delta_d0_minus_a0']:+.8f} "
        f"rel={block['relative_crps_delta_pct']:+.4f}% "
        f"A0_mae={block['a0_mae']:.6f} "
        f"D0_mae={block['d0_mae']:.6f} "
        f"A0_rmse={block['a0_rmse']:.6f} "
        f"D0_rmse={block['d0_rmse']:.6f} "
        f"A0_bias={block['a0_projection_bias']:+.6f} "
        f"D0_bias={block['d0_projection_bias']:+.6f}"
    )


def build_report(payload):
    lines = []

    lines.append("PITCHER_K_D0_STABILITY_CONFIRMATION_A")
    lines.append("=" * 38)
    lines.append("")

    lines.append("STABILITY PREFLIGHT")
    lines.append("-------------------")
    lines.append(
        json.dumps(
            payload["preflight"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("PAIRED ROW INTEGRITY")
    lines.append("--------------------")
    lines.append(
        json.dumps(
            payload["paired_row_integrity"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("2024 SUPPORTING WARM-UP")
    lines.append("-----------------------")
    lines.append(
        fmt_block(
            "2024",
            payload["supporting_2024"],
        )
    )

    lines.append("")
    lines.append("2025 PRIMARY CONFIRMATION")
    lines.append("-------------------------")
    lines.append(
        fmt_block(
            "2025",
            payload["primary_2025"],
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
    lines.append("BREADTH CONFIRMATION GATE")
    lines.append("-------------------------")
    lines.append(
        json.dumps(
            payload["gate"],
            indent=2,
        )
    )

    lines.append("")
    lines.append("FINAL STABILITY VERDICT")
    lines.append("-----------------------")
    lines.append(payload["final_verdict"])

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    print("PITCHER_K_D0_STABILITY_CONFIRMATION_A", flush=True)
    print("======================================", flush=True)

    print("")
    print("STABILITY PREFLIGHT", flush=True)
    print("-------------------", flush=True)

    base_conn = connect_ro(BASELINE_DB)
    lab_conn = connect_ro(LAB_DB)

    try:
        baseline_tables = check_tables(
            base_conn,
            {"historical_rows"},
        )

        lab_tables = check_tables(
            lab_conn,
            {"lab_rows"},
        )

        lab_result = safe_json_load(ARCH_LAB_RESULTS)

        prior_verdict = None

        if isinstance(lab_result, dict):
            prior_verdict = (
                lab_result
                .get("results", {})
                .get("final_verdict")
            )

        a0_count = lab_conn.execute(
            """
            SELECT COUNT(*)
            FROM lab_rows
            WHERE cell = ?
              AND year IN (?, ?)
            """,
            (
                A0_CELL,
                WARMUP_YEAR,
                PRIMARY_YEAR,
            ),
        ).fetchone()[0]

        d0_count = lab_conn.execute(
            """
            SELECT COUNT(*)
            FROM lab_rows
            WHERE cell = ?
              AND year IN (?, ?)
            """,
            (
                D0_CELL,
                WARMUP_YEAR,
                PRIMARY_YEAR,
            ),
        ).fetchone()[0]

        preflight = {
            "candidate": D0_CELL,
            "a0_cell": A0_CELL,
            "baseline_db": str(BASELINE_DB),
            "lab_db": str(LAB_DB),
            "baseline_tables": baseline_tables,
            "lab_tables": lab_tables,
            "prior_architecture_lab_verdict": prior_verdict,
            "a0_rows_2024_2025": int(a0_count),
            "d0_rows_2024_2025": int(d0_count),
            "row_count_parity": int(a0_count) == int(d0_count),
            "primary_confirmation_year": PRIMARY_YEAR,
            "supporting_warmup_year": WARMUP_YEAR,
            "independent_formal_validation": False,
            "purpose": (
                "breadth_and_intervention_stability_confirmation_only"
            ),
            "touch_2026": False,
            "external_api_calls": False,
            "production_changed": False,
            "stacking": False,
            "threshold_tuning": False,
            "rescue_tuning": False,
            "resimulation": False,
        }

        print(
            json.dumps(preflight, indent=2),
            flush=True,
        )

        rows = load_paired_rows(
            base_conn,
            lab_conn,
        )

        integrity = parity_and_integrity(rows)

        print("")
        print("PAIRED ROW INTEGRITY", flush=True)
        print("--------------------", flush=True)
        print(
            json.dumps(integrity, indent=2),
            flush=True,
        )

        warmup_rows = [
            row
            for row in rows
            if row["year"] == WARMUP_YEAR
        ]

        primary_rows = [
            row
            for row in rows
            if row["year"] == PRIMARY_YEAR
        ]

        supporting_2024 = metric_block(warmup_rows)
        primary_2025 = metric_block(primary_rows)

        print("")
        print("2024 SUPPORTING WARM-UP", flush=True)
        print("-----------------------", flush=True)
        print(
            fmt_block(
                "2024",
                supporting_2024,
            ),
            flush=True,
        )

        print("")
        print("2025 PRIMARY CONFIRMATION", flush=True)
        print("-------------------------", flush=True)
        print(
            fmt_block(
                "2025",
                primary_2025,
            ),
            flush=True,
        )

        intervention = intervention_analysis(
            primary_rows
        )

        print("")
        print("INTERVENTION SUBSET", flush=True)
        print("-------------------", flush=True)
        print(
            f"definition={intervention['intervention_definition']}",
            flush=True,
        )
        print(
            f"intervention_rows={intervention['intervention_rows']:,}",
            flush=True,
        )
        print(
            f"intervention_rate={intervention['intervention_rate']:.6f}",
            flush=True,
        )
        print(
            fmt_block(
                "INTERVENTION",
                intervention["intervention_metrics"],
            ),
            flush=True,
        )
        print(
            fmt_block(
                "UNCHANGED",
                intervention["unchanged_metrics"],
            ),
            flush=True,
        )

        monthly = monthly_stability(primary_rows)

        print("")
        print("MONTHLY STABILITY", flush=True)
        print("-----------------", flush=True)

        for month, block in monthly["months"].items():
            print(
                f"{month}: "
                f"n={block['rows']:,} "
                f"eligible={block['eligible']} "
                f"A0_crps={block['a0_mean_crps']:.8f} "
                f"D0_crps={block['d0_mean_crps']:.8f} "
                f"delta={block['crps_delta_d0_minus_a0']:+.8f} "
                f"challenger_win={block['challenger_win']}",
                flush=True,
            )

        print(
            f"eligible_months={monthly['eligible_months']}",
            flush=True,
        )
        print(
            f"crps_wins={monthly['crps_wins']}",
            flush=True,
        )
        print(
            f"required_wins={monthly['required_wins']}",
            flush=True,
        )
        print(
            f"monthly_stability_pass={monthly['pass']}",
            flush=True,
        )

        breadth = slice_breadth(primary_rows)

        print("")
        print("SLICE BREADTH", flush=True)
        print("-------------", flush=True)

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
            f"required_nonworse_fraction="
            f"{breadth['minimum_required_nonworse_fraction']:.6f}",
            flush=True,
        )
        print(
            f"materially_damaged_buckets="
            f"{len(breadth['materially_damaged_buckets'])}",
            flush=True,
        )

        for slice_name, grouped in breadth["slices"].items():
            print("")
            print(f"[{slice_name}]", flush=True)

            for bucket, block in grouped.items():
                eligible = (
                    block["rows"] >= ELIGIBLE_SLICE_MIN_ROWS
                )

                print(
                    f"{bucket}: "
                    f"n={block['rows']:,} "
                    f"eligible={eligible} "
                    f"A0_crps={block['a0_mean_crps']:.8f} "
                    f"D0_crps={block['d0_mean_crps']:.8f} "
                    f"delta={block['crps_delta_d0_minus_a0']:+.8f} "
                    f"rel={block['relative_crps_delta_pct']:+.4f}%",
                    flush=True,
                )

        gate = build_gate(
            primary_2025,
            monthly,
            intervention,
            breadth,
            integrity,
        )

        print("")
        print("BREADTH CONFIRMATION GATE", flush=True)
        print("-------------------------", flush=True)

        for key, value in gate.items():
            print(
                f"{key}: {value}",
                flush=True,
            )

        if gate["overall_stability_confirmation_pass"]:
            final_verdict = (
                "D0_FIXED_LINEUP_ACTIVATION_BROADLY_STABLE_"
                "DEVELOPMENT_SURVIVOR_ELIGIBLE_TO_FREEZE_FOR_"
                "SEPARATE_FUTURE_OR_PROSPECTIVE_HOLDOUT_"
                "NO_PRODUCTION_PROMOTION"
            )
        else:
            final_verdict = (
                "D0_FIXED_LINEUP_ACTIVATION_FAILS_STABILITY_"
                "CONFIRMATION_FREEZE_NO_RESCUE_TUNING"
            )

        print("")
        print("FINAL STABILITY VERDICT", flush=True)
        print("-----------------------", flush=True)
        print(final_verdict, flush=True)

        payload = {
            "script": "PITCHER_K_D0_STABILITY_CONFIRMATION_A",
            "generated_at_utc": now_utc(),
            "candidate": D0_CELL,
            "mode": (
                "DEVELOPMENT_BREADTH_CONFIRMATION_ONLY_"
                "NO_2026_NO_PRODUCTION_CHANGE"
            ),
            "preflight": preflight,
            "paired_row_integrity": integrity,
            "supporting_2024": supporting_2024,
            "primary_2025": primary_2025,
            "intervention_analysis": intervention,
            "monthly_stability": monthly,
            "slice_breadth": breadth,
            "gate_definition": {
                "hard_conditions": [
                    "paired integrity passes",
                    "2025 overall CRPS improves",
                    "2025 MAE does not worsen",
                    "2025 RMSE does not worsen",
                    "2025 absolute projection bias does not worsen",
                    (
                        "CRPS wins in at least half of eligible 2025 months; "
                        f"eligible month >= {ELIGIBLE_MONTH_MIN_ROWS} rows"
                    ),
                    (
                        f"intervention subset >= {MIN_INTERVENTION_ROWS} rows"
                    ),
                    "intervention subset CRPS improves",
                    "intervention subset MAE does not worsen",
                    (
                        f"at least {MIN_SLICE_NONWORSE_FRACTION:.0%} of "
                        "eligible predeclared slice buckets are CRPS nonworse"
                    ),
                    (
                        "no eligible slice worsens CRPS by more than "
                        f"{MAX_ALLOWED_RELATIVE_CRPS_WORSENING_PCT:.1f}%"
                    ),
                ],
                "eligible_slice_min_rows": ELIGIBLE_SLICE_MIN_ROWS,
                "eligible_month_min_rows": ELIGIBLE_MONTH_MIN_ROWS,
                "2024_status": "supporting_advisory_only",
                "formal_validation": False,
                "production_promotion_authorized": False,
            },
            "gate": gate,
            "final_verdict": final_verdict,
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

    finally:
        base_conn.close()
        lab_conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
