#!/usr/bin/env python3
"""
HITS_TRAINING_PROVENANCE_LITE_B

Memory-safe provenance check for a 512 MB Render instance.

Purpose:
- Do NOT import api.py, train.py, pandas, xgboost, or numpy.
- Do NOT load season files into memory.
- List every /data/season_*.json file.
- Compare file modification times against the current batter_hits model artifact.
- Stream each season file line-by-line only to recover:
    * line count
    * batter-row count where player_type == "batter"
    * min/max date when available
- Determine whether season_2026.json existed before the current model artifact,
  which strongly indicates 2026 was included because train.py iterates every
  sorted /data/season_*.json file.

Run:
python -u hits_training_provenance_lite_b.py 2>&1 | tee /data/hr_model/hits_training_provenance_lite_b.log
"""

import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path

DATA_GLOB = "/data/season_*.json"
MODEL_PATH = Path("/data/models/batter_hits.json")
COLS_PATH = Path("/data/models/batter_hits_columns.json")


def fmt_ts(ts):
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def inspect_stream(path):
    total_lines = 0
    batter_rows = 0
    min_date = None
    max_date = None
    parse_errors = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception:
                parse_errors += 1
                continue

            if row.get("player_type") == "batter":
                batter_rows += 1

            d = row.get("date")
            if d is not None:
                d = str(d)
                if min_date is None or d < min_date:
                    min_date = d
                if max_date is None or d > max_date:
                    max_date = d

    return {
        "total_lines": total_lines,
        "batter_rows": batter_rows,
        "min_date": min_date,
        "max_date": max_date,
        "parse_errors": parse_errors,
    }


def main():
    print("HITS_TRAINING_PROVENANCE_LITE_B")
    print("================================")
    print("mode: STREAMING / MEMORY SAFE")
    print("imports: stdlib only")
    print()

    files = [Path(p) for p in sorted(glob.glob(DATA_GLOB))]

    model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    cols_mtime = COLS_PATH.stat().st_mtime if COLS_PATH.exists() else None

    print("MODEL ARTIFACTS")
    print("---------------")
    print(
        f"{MODEL_PATH} | exists={MODEL_PATH.exists()} | "
        f"mtime_utc={fmt_ts(model_mtime)} | "
        f"size={MODEL_PATH.stat().st_size if MODEL_PATH.exists() else None}"
    )
    print(
        f"{COLS_PATH} | exists={COLS_PATH.exists()} | "
        f"mtime_utc={fmt_ts(cols_mtime)} | "
        f"size={COLS_PATH.stat().st_size if COLS_PATH.exists() else None}"
    )
    print()

    print("SEASON FILES")
    print("------------")
    print(f"count={len(files)}")

    any_2026 = False
    season_2026_before_model = False

    for path in files:
        st = path.stat()
        before_model = (
            model_mtime is not None and st.st_mtime <= model_mtime
        )

        if "2026" in path.name:
            any_2026 = True
            if before_model:
                season_2026_before_model = True

        print()
        print(f"FILE: {path.name}")
        print(f"SIZE: {st.st_size:,}")
        print(f"MTIME_UTC: {fmt_ts(st.st_mtime)}")
        print(f"EXISTED_BY_MODEL_MTIME: {before_model}")

        stats = inspect_stream(path)
        print(f"TOTAL_LINES: {stats['total_lines']:,}")
        print(f"BATTER_ROWS: {stats['batter_rows']:,}")
        print(f"DATE_MIN: {stats['min_date']}")
        print(f"DATE_MAX: {stats['max_date']}")
        print(f"PARSE_ERRORS: {stats['parse_errors']:,}")

    print()
    print("HOLDOUT READ")
    print("------------")

    if not MODEL_PATH.exists():
        verdict = "MODEL_ARTIFACT_MISSING_CANNOT_RESOLVE"
    elif not any_2026:
        verdict = "NO_2026_SEASON_FILE_FOUND_2026_MAY_BE_VALID_HOLDOUT"
    elif season_2026_before_model:
        verdict = (
            "SEASON_2026_EXISTED_BEFORE_MODEL_ARTIFACT_"
            "2026_IS_NOT_A_CLEAN_UNTOUCHED_HOLDOUT_UNDER_CURRENT_TRAINER"
        )
    else:
        verdict = (
            "SEASON_2026_EXISTS_BUT_APPEARS_NEWER_THAN_MODEL_ARTIFACT_"
            "2026_MAY_STILL_BE_A_VALID_HOLDOUT"
        )

    print(f"verdict: {verdict}")
    print()
    print(
        "Interpretation rule: train.py iterates every sorted /data/season_*.json "
        "file. Therefore, if season_2026.json existed before the current model "
        "artifact was written, the current trainer would normally have included it."
    )


if __name__ == "__main__":
    main()
