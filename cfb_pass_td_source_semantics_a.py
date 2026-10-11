#!/usr/bin/env python3
"""Independent source-field semantics falsification for passing TD play events.

The historical passing touchdown simulator infers a touchdown when an input
row has BOTH completion_player_id and touchdown_player_id. That is not
necessarily the structure of cfbfastR's per-play-derived player_stats rows.
Count exact co-occurrence per source vintage, never label a completion as a
touchdown based on an unrelated player's separate row or game-level total.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def scan(path):
    hash_obj = hashlib.sha256()
    with Path(path).open("rb") as raw:
        for b in iter(lambda: raw.read(1024 * 1024), b""):
            hash_obj.update(b)
    counts = {
        "source_rows": 0, "completion_rows": 0,
        "touchdown_identity_rows": 0,
        "completion_with_touchdown_identity": 0,
        "reception_with_touchdown_identity": 0,
        "rush_with_touchdown_identity": 0,
        "pass_interception_rows": 0,
    }
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"completion_player_id", "touchdown_player_id",
                    "reception_player_id", "rush_player_id",
                    "interception_thrown_player_id"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("REQUIRED_REAL_PLAYER_STAT_COLUMNS_MISSING")
        valid = lambda val: val is not None and str(val).strip() not in ("", "NA", "NaN")
        for row in reader:
            counts["source_rows"] += 1
            complete = valid(row["completion_player_id"])
            td = valid(row["touchdown_player_id"])
            if complete:
                counts["completion_rows"] += 1
            if td:
                counts["touchdown_identity_rows"] += 1
            if complete and td:
                counts["completion_with_touchdown_identity"] += 1
            if valid(row["reception_player_id"]) and td:
                counts["reception_with_touchdown_identity"] += 1
            if valid(row["rush_player_id"]) and td:
                counts["rush_with_touchdown_identity"] += 1
            if valid(row["interception_thrown_player_id"]):
                counts["pass_interception_rows"] += 1
    return {
        "source_file_sha256": hash_obj.hexdigest(),
        "size_bytes": Path(path).stat().st_size,
        "counts": counts,
        "existing_completion_plus_td_rule_has_any_positive_evidence":
            counts["completion_with_touchdown_identity"] > 0,
        "positive_overlap_does_not_alone_certify_true_passing_touchdown_semantics": True,
        "production_or_historical_forecasts_modified": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season-source", nargs=2, action="append", required=True,
                   metavar=("SEASON", "CSV_PATH"))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    results = {}
    for season, path in args.season_source:
        if season in results:
            raise ValueError("DUPLICATE_SEASON")
        results[season] = scan(path)
    out = {
        "schema": "CFB_PASSING_TD_SOURCE_FIELD_SEMANTICS_V1",
        "research_only": True,
        "source_years": results,
        "scientific_verdict": "SOURCE_CO_OCCURRENCE_AUDIT_NOT_EVENT_LABEL_MODEL_PROMOTION",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
