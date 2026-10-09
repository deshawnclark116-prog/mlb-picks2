#!/usr/bin/env python3
"""Source-only Phase1I-A feasibility audit, before fitting or performance."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from nfl_v2_phase1h_sources import records, present, local_name

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "nfl_models/nfl_player_outcome_v2/phase1h_source_coverage.json"


def verify(directory, frozen=FROZEN):
    corpus = json.loads(Path(frozen).read_text())
    for year, sources in corpus["seasons"].items():
        for kind, meta in sources.items():
            path = Path(directory) / local_name(kind, int(year))
            if meta["status"] == "NOT_PUBLISHED":
                if path.exists():
                    raise ValueError(f"Unaudited new source: {path.name}")
            elif not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
                raise ValueError(f"Frozen source changed: {path.name}; no silent revised-corpus scoring")
    return corpus


def audit(directory):
    corpus = verify(directory)
    out = {"schema": "nfl-v2-phase1i-source-feasibility-v1", "corpus": "Frozen Phase1H 2023-2026 W1-4 files; no new model-performance inputs",
           "source_sha256": {f"{kind}_{y}": m.get("sha256") for y, ss in corpus["seasons"].items() for kind,m in ss.items()},
           "temporal_contract": "Only completed REG source games in strictly earlier season/week AND earlier game date. Retrospective releases do not prove archived original provider publication times; no historical T24/T90 snapshot-equivalence claim.",
           "seasons": {},
           "route_feasibility": {
               "investigated_provider": "nflverse/nflreadr public data dictionaries and nflverse nextgen_stats release; no proprietary scraping",
               "dictionary_urls": ["https://raw.githubusercontent.com/nflverse/nflreadr/main/data-raw/dictionary_participation.csv",
                                   "https://raw.githubusercontent.com/nflverse/nflreadr/main/data-raw/dictionary_nextgenstats.csv"],
               "ngs_release_url": "https://github.com/nflverse/nflverse-data/releases/tag/nextgen_stats",
               "routes_run": {"status": "BLOCKED_DATA", "reason": "Participation player-on-play lists are not receiver route/run labels; no complete per-player route-opportunity ledger in inspected public corpus."},
               "route_type": {"status": "HISTORICAL_TARGETED_ROUTE_LABEL_ONLY_NOT_USED", "reason": "Participation route is the targeted receiver route, not all routes run; available 2023-2025, provider 2026 absent."},
               "alignment": {"status": "BLOCKED_DATA", "reason": "Formation/personnel/position lists do not identify each receiver's route alignment."},
               "separation": {"status": "BLOCKED_DATA_ROUTE_LEVEL", "reason": "NGS receiving files expose weekly/season aggregate avg_separation and avg_cushion, not per-route/QB-receiver depth/coverage separation; no original cutoff snapshots established. No aggregate proxy used."},
               "defender_assignment": {"status": "BLOCKED_DATA", "reason": "Coverage/man-zone team labels and on-play defender identities are not receiver-defender assignment."},
               "decision": "Continue with PBP depth/completion mechanics; no route/separation inputs or delay."}}
    for season in (2023, 2024, 2025, 2026):
        part = {}
        path = Path(directory) / f"participation_{season}.csv"
        if path.exists():
            for r in records(path):
                part[(r["nflverse_game_id"], str(int(float(r["play_id"]))))] = r
        c, depth = Counter(), Counter()
        for r in records(Path(directory) / f"pbp_{season}.csv.gz"):
            if r.get("season_type") != "REG" or r.get("play_type") != "pass" or r.get("no_play") == "1" or r.get("two_point_attempt") == "1" or not present(r.get("receiver_player_id")):
                continue
            if season == 2026 and int(r["week"]) > 4:
                continue
            c["target_plays"] += 1
            for field in ("passer_player_id", "receiver_player_id", "air_yards", "complete_pass", "pass_location", "pass_length"):
                c[field] += present(r.get(field))
            if present(r.get("air_yards")):
                a = float(r["air_yards"])
                depth["behind_LOS" if a<0 else "0_9" if a<10 else "10_19" if a<20 else "20_plus"] += 1
                if r.get("complete_pass") == "1" and present(r.get("yards_after_catch")):
                    c["completed_targets_with_air_yac"] += 1
                    c["air_plus_yac_reconciles_play_yards"] += abs(a+float(r["yards_after_catch"])-float(r["yards_gained"])) < .01
            p = part.get((r["game_id"], str(int(float(r["play_id"])))), {})
            c["joined_pressure_labels"] += present(p.get("was_pressure"))
            c["joined_coverage_labels"] += p.get("defense_man_zone_type") in {"MAN_COVERAGE", "ZONE_COVERAGE"}
            c["joined_targeted_route_labels"] += present(p.get("route"))
        out["seasons"][str(season)] = {"counts": dict(c), "depth_counts": dict(depth),
            "pbp_depth_chain_status": "AVAILABLE_PRIOR_COMPLETED_GAMES",
            "target_location": "COARSE_LEFT_MIDDLE_RIGHT; NOT_ALIGNMENT",
            "pressure_and_coverage": "BLOCKED_CURRENT_PROVIDER_NOT_PUBLISHED" if season == 2026 else "HISTORICAL_PRIOR_GAMES_ONLY",
            "pressure_blitz_use": "NOT_TESTED_ISOLATE_NEW_DEPTH_CHAIN; no Phase1H corrections reused"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    result = audit(a.data_dir)
    Path(a.out).write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result["seasons"],indent=2))


if __name__ == "__main__":
    main()
