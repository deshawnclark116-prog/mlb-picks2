#!/usr/bin/env python3
"""
PITCHER_K_CHAMPION_BASELINE_PREFLIGHT_A

Read-only preflight for reconstructing and forward-validating the exact current
Prop Edge pitcher-strikeout champion.

This script does NOT:
- change production
- retrain anything
- call external APIs
- modify the batter-hits prospective holdout
- inspect batter-hits prospective outcomes
- tune K thresholds or features

It inspects only local source/data availability and freezes provenance for:
- api.py
- ksim.py
- lineupk.py
- bvp.py

It reports whether the local data can support an honest historical K baseline.

Current active K architecture reconstructed from source
-------------------------------------------------------
Pitcher baseline:
- starts with BF >= 12
- minimum 3 qualifying starts
- recency weights exp(-0.6 * age)
- final K/BF center = 85% recency-weighted + 15% season
- workload = average BF over last 5 qualifying starts
- volatility pool = last 12 per-start K/BF rates

Lineup layer:
- general batter K rate vs pitcher handedness, minimum 15 PA
- fallback current-season K rate, minimum 15 PA
- final fallback league average 0.22
- H2H available at 2 PA
- H2H blend = 40% H2H + 60% general
- current n_data semantics count H2H-available batters only
- api.py activates lineup expectation only when n >= 5

Pitcher/lineup blend:
- 55% pitcher
- 45% lineup

BVP lineup nudge:
- pooled career lineup BvP K rate
- sample gate = 20 total PA
- raw nudge = lineup_k_rate / 0.22
- capped to [0.85, 1.15]
- rounded to 3 decimals
- applied to central blended K/BF
- applied to per-start K-rate volatility pool

K simulation:
- 10,000 simulations by default
- unseeded np.random.RandomState()
- BF ~ Normal(expected_bf, 2.5), clamped to [9, 30]
- if >=4 recent start rates:
    sampled pool = 70% actual per-start rate + 30% center
- times-through-order factors:
    1.00 for PA 1-9
    0.94 for PA 10-18
    0.85 for PA 19+
- side chosen by larger raw P(over) or P(under)
- no_bet below 0.59
- production watchlist >= 0.60
- production official >= 0.63
- real FanDuel main K line required

Known architecture findings to preserve
---------------------------------------
K-A01:
    lineup_k_expectation n_data counts qualifying H2H batters, not all batters
    with usable general handedness/season K data.

K-A02:
    production ksim is nondeterministic because no stable seed is supplied.

K-A03:
    LOW confidence (0.59-0.639...) can still become official at >=0.63 because
    confidence and board thresholds are semantically misaligned.

K-H01:
    possible TTO double-counting because realized full-start K/BF rates are
    sampled and then an additional hard-coded TTO decay is applied.

K-H02:
    possible H2H double-counting because same-pitcher batter K history can enter
    both lineupk's 40/60 H2H blend and bvp.py's lineup-wide k_nudge.

Run
---
python -u pitcher_k_champion_baseline_preflight_a.py 2>&1 | tee /data/hr_model/pitcher_k_champion_baseline_preflight_a.log

Outputs
-------
/data/hr_model/pitcher_k_champion_baseline_preflight_a_results.json
/data/hr_model/pitcher_k_champion_baseline_preflight_a_report.txt

Paste back
----------
SOURCE PROVENANCE
STATCAST COVERAGE
SEASON FILE COVERAGE
PREDICTION ARCHIVE COVERAGE
K-CANDIDATE ARCHIVE COVERAGE
K-LINE CACHE COVERAGE
RECORD K COVERAGE
BASELINE FEASIBILITY
FINAL PREFLIGHT VERDICT
"""

import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path


ROOT = Path.cwd()

DATA_DIR = Path("/data")
HR_DIR = Path("/data/hr_model")
STATCAST_DB = HR_DIR / "hr_model.sqlite"

PRED_DIR = DATA_DIR / "predictions"
LOG_DIR = DATA_DIR / "candidate_logs"
K_LINE_DIR = DATA_DIR / "k_lines"
RECORD_JSON = DATA_DIR / "record.json"

OUT_JSON = HR_DIR / "pitcher_k_champion_baseline_preflight_a_results.json"
OUT_TXT = HR_DIR / "pitcher_k_champion_baseline_preflight_a_report.txt"

SOURCE_FILES = [
    Path("api.py"),
    Path("ksim.py"),
    Path("lineupk.py"),
    Path("bvp.py"),
]

ARCHITECTURE_FINDINGS = {
    "K-A01": (
        "lineup_k_expectation n_data counts qualifying H2H batters only; "
        "api.py requires n>=5 before activating the 45% lineup projection."
    ),
    "K-A02": (
        "ksim uses unseeded np.random.RandomState(), so identical inputs can "
        "produce different probabilities and board status near thresholds."
    ),
    "K-A03": (
        "Official threshold is 0.63 while MEDIUM confidence begins at 0.64, "
        "so some LOW-confidence predictions can be official."
    ),
    "K-H01": (
        "Possible TTO double-counting: realized full-start K/BF is sampled, "
        "then additional 1.00/0.94/0.85 TTO decay is applied."
    ),
    "K-H02": (
        "Possible H2H double-counting: same-pitcher batter K history can enter "
        "both lineupk H2H blending and bvp lineup-wide k_nudge."
    ),
}


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def safe_json_load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def source_provenance():
    rows = []

    for path in SOURCE_FILES:
        row = {
            "path": str(path),
            "exists": path.exists(),
        }

        if path.exists():
            stat = path.stat()
            row.update(
                {
                    "size_bytes": stat.st_size,
                    "mtime_epoch": stat.st_mtime,
                    "sha256": sha256_file(path),
                }
            )

        rows.append(row)

    return rows


def statcast_coverage():
    if not STATCAST_DB.exists():
        return {
            "exists": False,
            "path": str(STATCAST_DB),
        }

    conn = sqlite3.connect(f"file:{STATCAST_DB}?mode=ro", uri=True)

    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

        if "statcast_pitches" not in tables:
            return {
                "exists": True,
                "path": str(STATCAST_DB),
                "statcast_pitches_exists": False,
                "tables": sorted(tables),
            }

        cols = [
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(statcast_pitches)"
            )
        ]

        overall = conn.execute(
            """
            SELECT
                COUNT(*),
                MIN(game_date),
                MAX(game_date),
                COUNT(DISTINCT game_pk),
                COUNT(DISTINCT pitcher_id),
                COUNT(DISTINCT batter_id)
            FROM statcast_pitches
            """
        ).fetchone()

        by_year = []

        for row in conn.execute(
            """
            SELECT
                SUBSTR(game_date, 1, 4) AS year,
                COUNT(*) AS pitch_rows,
                COUNT(DISTINCT game_pk) AS games,
                COUNT(DISTINCT pitcher_id) AS pitchers,
                COUNT(DISTINCT batter_id) AS batters,
                MIN(game_date),
                MAX(game_date)
            FROM statcast_pitches
            GROUP BY SUBSTR(game_date, 1, 4)
            ORDER BY year
            """
        ):
            by_year.append(
                {
                    "year": row[0],
                    "pitch_rows": int(row[1]),
                    "games": int(row[2]),
                    "pitchers": int(row[3]),
                    "batters": int(row[4]),
                    "min_date": row[5],
                    "max_date": row[6],
                }
            )

        required_cols = {
            "game_pk",
            "game_date",
            "at_bat_number",
            "pitch_number",
            "batter_id",
            "pitcher_id",
            "p_throws",
            "description",
            "events",
            "is_whiff",
        }

        return {
            "exists": True,
            "path": str(STATCAST_DB),
            "statcast_pitches_exists": True,
            "columns": cols,
            "missing_required_columns": sorted(required_cols - set(cols)),
            "overall": {
                "pitch_rows": int(overall[0]),
                "min_date": overall[1],
                "max_date": overall[2],
                "games": int(overall[3]),
                "pitchers": int(overall[4]),
                "batters": int(overall[5]),
            },
            "by_year": by_year,
        }

    finally:
        conn.close()


def season_file_coverage():
    rows = []

    for path in sorted(DATA_DIR.glob("season_*.json*")):
        if path.suffix not in {".json", ".jsonl"}:
            continue

        year_match = re.search(r"season_(\d{4})", path.name)

        rows.append(
            {
                "path": str(path),
                "year": (
                    int(year_match.group(1))
                    if year_match
                    else None
                ),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return rows


def json_file_date(name):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else None


def prediction_archive_coverage():
    rows = []

    if not PRED_DIR.exists():
        return {
            "exists": False,
            "path": str(PRED_DIR),
            "files": [],
        }

    files = sorted(PRED_DIR.glob("predictions_*.json"))

    total_rows = 0

    for path in files:
        payload = safe_json_load(path)
        count = len(payload) if isinstance(payload, list) else None

        if count is not None:
            total_rows += count

        rows.append(
            {
                "date": json_file_date(path.name),
                "path": str(path),
                "rows": count,
                "size_bytes": path.stat().st_size,
            }
        )

    dates = [
        row["date"]
        for row in rows
        if row["date"]
    ]

    return {
        "exists": True,
        "path": str(PRED_DIR),
        "file_count": len(rows),
        "total_rows": total_rows,
        "min_date": min(dates) if dates else None,
        "max_date": max(dates) if dates else None,
        "files": rows,
    }


def k_candidate_archive_coverage():
    if not PRED_DIR.exists():
        return {
            "exists": False,
            "path": str(PRED_DIR),
            "files": [],
        }

    files = sorted(PRED_DIR.glob("pitcher_k_candidates_*.json"))
    rows = []
    total_rows = 0
    with_line_total = 0

    union_keys = set()

    for path in files:
        payload = safe_json_load(path)

        if not isinstance(payload, list):
            count = None
            with_line = None
            date = json_file_date(path.name)
            keys = []
        else:
            count = len(payload)
            date = json_file_date(path.name)
            keys = sorted(
                {
                    key
                    for row in payload
                    if isinstance(row, dict)
                    for key in row.keys()
                }
            )

            with_line = sum(
                1
                for row in payload
                if isinstance(row, dict)
                and (
                    row.get("has_k_line")
                    or row.get("has_line")
                    or row.get("k_line") is not None
                    or row.get("pick_line") is not None
                )
            )

            total_rows += count
            with_line_total += with_line
            union_keys.update(keys)

        rows.append(
            {
                "date": date,
                "path": str(path),
                "rows": count,
                "with_k_line_rows": with_line,
                "keys": keys,
                "size_bytes": path.stat().st_size,
            }
        )

    dates = [
        row["date"]
        for row in rows
        if row["date"]
    ]

    return {
        "exists": True,
        "path": str(PRED_DIR),
        "file_count": len(rows),
        "total_rows": total_rows,
        "with_k_line_rows": with_line_total,
        "min_date": min(dates) if dates else None,
        "max_date": max(dates) if dates else None,
        "union_keys": sorted(union_keys),
        "files": rows,
    }


def k_line_cache_coverage():
    if not K_LINE_DIR.exists():
        return {
            "exists": False,
            "path": str(K_LINE_DIR),
            "files": [],
        }

    files = sorted(K_LINE_DIR.glob("k_lines_*.json"))
    rows = []
    total_lines = 0
    providers = {}

    for path in files:
        payload = safe_json_load(path)

        if isinstance(payload, dict):
            line_count = int(payload.get("line_count") or 0)
            provider = payload.get("provider")
            date = payload.get("date_et") or json_file_date(path.name)
        else:
            line_count = None
            provider = None
            date = json_file_date(path.name)

        if line_count is not None:
            total_lines += line_count

        if provider:
            providers[str(provider)] = (
                providers.get(str(provider), 0) + 1
            )

        rows.append(
            {
                "date": date,
                "path": str(path),
                "provider": provider,
                "line_count": line_count,
                "size_bytes": path.stat().st_size,
            }
        )

    dates = [
        row["date"]
        for row in rows
        if row["date"]
    ]

    return {
        "exists": True,
        "path": str(K_LINE_DIR),
        "file_count": len(rows),
        "total_lines": total_lines,
        "providers": providers,
        "min_date": min(dates) if dates else None,
        "max_date": max(dates) if dates else None,
        "files": rows,
    }


def record_k_coverage():
    if not RECORD_JSON.exists():
        return {
            "exists": False,
            "path": str(RECORD_JSON),
        }

    payload = safe_json_load(RECORD_JSON)

    if not isinstance(payload, dict):
        return {
            "exists": True,
            "path": str(RECORD_JSON),
            "parse_ok": False,
        }

    results = payload.get("results", [])

    if not isinstance(results, list):
        results = []

    k_rows = [
        row
        for row in results
        if isinstance(row, dict)
        and row.get("prop_type") == "pitcher_strikeouts"
    ]

    dates = sorted(
        {
            str(row.get("date"))
            for row in k_rows
            if row.get("date")
        }
    )

    union_keys = sorted(
        {
            key
            for row in k_rows
            for key in row.keys()
        }
    )

    field_coverage = {}

    important_fields = [
        "date",
        "api_version",
        "player",
        "player_id",
        "game_id",
        "pick",
        "projected",
        "actual",
        "result",
        "model_prob",
        "confidence",
        "bvp_flag",
        "line_source",
        "k_line_locked",
        "k_gate_version",
        "k_projection_gap",
        "k_has_lineup_kr",
        "avg_bf",
        "recent_k_avg",
        "outs_per_start",
        "starts",
    ]

    for field in important_fields:
        field_coverage[field] = sum(
            1
            for row in k_rows
            if row.get(field) is not None
        )

    return {
        "exists": True,
        "path": str(RECORD_JSON),
        "parse_ok": True,
        "k_rows": len(k_rows),
        "min_date": dates[0] if dates else None,
        "max_date": dates[-1] if dates else None,
        "distinct_dates": len(dates),
        "union_keys": union_keys,
        "important_field_nonnull_counts": field_coverage,
    }


def candidate_log_k_coverage():
    if not LOG_DIR.exists():
        return {
            "exists": False,
            "path": str(LOG_DIR),
            "files": [],
        }

    files = sorted(LOG_DIR.glob("candidates_*.json"))
    rows = []
    total_k_rows = 0

    for path in files:
        payload = safe_json_load(path)

        if isinstance(payload, list):
            k_rows = [
                row
                for row in payload
                if isinstance(row, dict)
                and row.get("prop_type") == "pitcher_strikeouts"
            ]
            count = len(k_rows)
        else:
            count = None

        if count is not None:
            total_k_rows += count

        rows.append(
            {
                "date": json_file_date(path.name),
                "path": str(path),
                "k_rows": count,
                "size_bytes": path.stat().st_size,
            }
        )

    dates = [
        row["date"]
        for row in rows
        if row["date"]
    ]

    return {
        "exists": True,
        "path": str(LOG_DIR),
        "file_count": len(rows),
        "total_k_rows": total_k_rows,
        "min_date": min(dates) if dates else None,
        "max_date": max(dates) if dates else None,
        "files": rows,
    }


def baseline_feasibility(report):
    statcast = report["statcast_coverage"]
    seasons = report["season_file_coverage"]
    k_candidates = report["k_candidate_archive_coverage"]
    k_lines = report["k_line_cache_coverage"]
    record_k = report["record_k_coverage"]

    statcast_years = {
        str(row.get("year"))
        for row in statcast.get("by_year", [])
    }

    season_years = {
        str(row.get("year"))
        for row in seasons
        if row.get("year") is not None
    }

    has_pitch_history = (
        statcast.get("statcast_pitches_exists") is True
        and not statcast.get("missing_required_columns")
    )

    has_2024_2025_statcast = {
        "2024",
        "2025",
    }.issubset(statcast_years)

    has_2024_2025_seasons = {
        "2024",
        "2025",
    }.issubset(season_years)

    has_k_candidate_archive = (
        int(k_candidates.get("file_count") or 0) > 0
        and int(k_candidates.get("with_k_line_rows") or 0) > 0
    )

    has_k_line_cache = (
        int(k_lines.get("file_count") or 0) > 0
        and int(k_lines.get("total_lines") or 0) > 0
    )

    has_graded_k_history = (
        int(record_k.get("k_rows") or 0) > 0
    )

    exact_historical_line_archive = (
        has_k_candidate_archive or has_k_line_cache
    )

    if (
        has_pitch_history
        and has_2024_2025_statcast
        and has_2024_2025_seasons
        and exact_historical_line_archive
    ):
        verdict = (
            "READY_FOR_CLEAN_HISTORICAL_K_BASELINE_WITH_AVAILABLE_LINE_ARCHIVE"
        )
    elif (
        has_pitch_history
        and has_2024_2025_statcast
        and has_2024_2025_seasons
    ):
        verdict = (
            "PITCH_AND_GAME_HISTORY_READY_BUT_EXACT_HISTORICAL_K_LINE_ARCHIVE_"
            "MAY_LIMIT_FULL_DECISION_LEVEL_BACKTEST"
        )
    else:
        verdict = (
            "MISSING_REQUIRED_HISTORICAL_INPUTS_FOR_EXACT_K_BASELINE"
        )

    return {
        "has_exact_source_provenance": all(
            row.get("exists")
            for row in report["source_provenance"]
        ),
        "has_required_statcast_pitch_history": has_pitch_history,
        "has_2024_2025_statcast": has_2024_2025_statcast,
        "has_2024_2025_season_files": has_2024_2025_seasons,
        "has_k_candidate_archive_with_lines": has_k_candidate_archive,
        "has_k_line_cache_archive": has_k_line_cache,
        "has_any_exact_historical_k_line_archive": (
            exact_historical_line_archive
        ),
        "has_graded_k_history": has_graded_k_history,
        "historical_lineup_reconstruction_possible_from_statcast": (
            has_pitch_history
        ),
        "historical_pitcher_profile_reconstruction_possible_from_statcast": (
            has_pitch_history
        ),
        "warning": (
            "Historical lineup/general/H2H features must be rebuilt from "
            "time-lagged local history. Do not query today's MLB career/current-"
            "season API endpoints for old folds because that would leak future "
            "information into the historical baseline."
        ),
        "verdict": verdict,
    }


def render_report(report):
    lines = []

    lines.append("PITCHER_K_CHAMPION_BASELINE_PREFLIGHT_A")
    lines.append("=" * 39)
    lines.append("")

    lines.append("SOURCE PROVENANCE")
    lines.append("-----------------")

    for row in report["source_provenance"]:
        lines.append(
            f"{row['path']}: "
            f"exists={row['exists']} "
            f"sha256={row.get('sha256')}"
        )

    lines.append("")
    lines.append("LOCKED ARCHITECTURE FINDINGS")
    lines.append("----------------------------")

    for key, value in report["architecture_findings"].items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("STATCAST COVERAGE")
    lines.append("-----------------")
    lines.append(json.dumps(report["statcast_coverage"], indent=2))

    lines.append("")
    lines.append("SEASON FILE COVERAGE")
    lines.append("--------------------")
    lines.append(json.dumps(report["season_file_coverage"], indent=2))

    lines.append("")
    lines.append("PREDICTION ARCHIVE COVERAGE")
    lines.append("---------------------------")
    lines.append(json.dumps(report["prediction_archive_coverage"], indent=2))

    lines.append("")
    lines.append("K-CANDIDATE ARCHIVE COVERAGE")
    lines.append("----------------------------")
    lines.append(json.dumps(report["k_candidate_archive_coverage"], indent=2))

    lines.append("")
    lines.append("K-LINE CACHE COVERAGE")
    lines.append("---------------------")
    lines.append(json.dumps(report["k_line_cache_coverage"], indent=2))

    lines.append("")
    lines.append("CANDIDATE-LOG K COVERAGE")
    lines.append("------------------------")
    lines.append(json.dumps(report["candidate_log_k_coverage"], indent=2))

    lines.append("")
    lines.append("RECORD K COVERAGE")
    lines.append("-----------------")
    lines.append(json.dumps(report["record_k_coverage"], indent=2))

    lines.append("")
    lines.append("BASELINE FEASIBILITY")
    lines.append("--------------------")
    lines.append(json.dumps(report["baseline_feasibility"], indent=2))

    lines.append("")
    lines.append("FINAL PREFLIGHT VERDICT")
    lines.append("-----------------------")
    lines.append(report["baseline_feasibility"]["verdict"])

    return "\n".join(lines)


def main():
    HR_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "script": "PITCHER_K_CHAMPION_BASELINE_PREFLIGHT_A",
        "mode": "READ_ONLY_LOCAL_DATA_ONLY",
        "production_changed": False,
        "external_api_calls": False,
        "hits_prospective_holdout_touched": False,
        "source_provenance": source_provenance(),
        "architecture_findings": ARCHITECTURE_FINDINGS,
        "statcast_coverage": statcast_coverage(),
        "season_file_coverage": season_file_coverage(),
        "prediction_archive_coverage": prediction_archive_coverage(),
        "k_candidate_archive_coverage": k_candidate_archive_coverage(),
        "k_line_cache_coverage": k_line_cache_coverage(),
        "candidate_log_k_coverage": candidate_log_k_coverage(),
        "record_k_coverage": record_k_coverage(),
    }

    report["baseline_feasibility"] = baseline_feasibility(report)

    OUT_JSON.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    text = render_report(report)

    OUT_TXT.write_text(
        text,
        encoding="utf-8",
    )

    print(text)
    print("")
    print("OUTPUTS")
    print(OUT_JSON)
    print(OUT_TXT)
    print("")
    print(
        f"verdict={report['baseline_feasibility']['verdict']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
