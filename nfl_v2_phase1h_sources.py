#!/usr/bin/env python3
"""Research-only Phase1H source retrieval/coverage; never evaluates a model."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
SEASONS = (2023, 2024, 2025, 2026)
SOURCES = {
    "stats": ("stats_player", "stats_player_week_{y}.csv"),
    "pbp": ("pbp", "play_by_play_{y}.csv.gz"),
    "roster": ("weekly_rosters", "roster_weekly_{y}.csv"),
    "participation": ("pbp_participation", "pbp_participation_{y}.csv"),
    "ftn": ("ftn_charting", "ftn_charting_{y}.csv"),
    "injuries": ("injuries", "injuries_{y}.csv"),
}


def local_name(kind, year):
    return {"stats": f"stats_player_week_{year}.csv", "pbp": f"pbp_{year}.csv.gz",
            "roster": f"roster_weekly_{year}.csv"}.get(kind, f"{kind}_{year}.csv")


def retrieve(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    # Release metadata distinguishes not published from network retrieval failures.
    for kind, (tag, pattern) in SOURCES.items():
        dest = directory / f"release_{kind}.json"
        subprocess.run(["curl", "-fsSL", "--retry", "3", "--max-time", "90",
                        f"https://api.github.com/repos/nflverse/nflverse-data/releases/tags/{tag}",
                        "-o", str(dest)], check=True)
        assets = {a["name"]: a for a in json.loads(dest.read_text())["assets"]}
        for year in SEASONS:
            name = pattern.format(y=year)
            a = assets.get(name)
            manifest[f"{kind}_{year}"] = {
                "url": f"{RELEASE}/{tag}/{name}",
                "status": "PUBLISHED" if a else "NOT_PUBLISHED",
                "local_name": local_name(kind, year),
                "asset_created_at": a["created_at"] if a else None,
                "asset_updated_at": a["updated_at"] if a else None,
                "asset_size": a["size"] if a else None,
            }
            if not a and kind in {"stats", "pbp", "roster"}:
                raise RuntimeError(f"Required source not published: {kind} {year}")

    def fetch(item):
        _, meta = item
        if meta["status"] != "PUBLISHED":
            return
        dest = directory / meta["local_name"]
        if not dest.exists() or dest.stat().st_size != meta["asset_size"]:
            tmp = dest.with_suffix(dest.suffix + ".partial")
            subprocess.run(["curl", "-fsSL", "--retry", "3", "--max-time", "240",
                            meta["url"], "-o", str(tmp)], check=True)
            tmp.replace(dest)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, manifest.items()))
    (directory / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)


def present(value):
    return str(value or "").strip().upper() not in {"", "NA", "NAN", "NONE"}


def audit(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "source_manifest.json").read_text())
    out = {"schema": "nfl-v2-phase1h-coverage-v1", "seasons": {},
           "historical_timing_limit": "Release timestamps describe current retrospective files, not original per-play publication. Features require strictly earlier REG weeks AND earlier game dates; no claim of archived T24/T90 provider snapshots.",
           "personnel_policy": "BLOCKED_DATA: historical injury publication/cutoff timestamps not proven; target-game snaps/absence oracle forbidden.",
           "current_2026_coverage": "Provider participation/man-zone/true pressure unavailable; any scheme/pressure history used in 2026 is explicitly stale prior-season fallback."}
    for year in SEASONS:
        s = {}
        for kind in SOURCES:
            meta = dict(manifest[f"{kind}_{year}"])
            path = directory / meta["local_name"]
            if meta["status"] != "PUBLISHED":
                s[kind] = meta
                continue
            meta["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            n = targets = air = catches = yac = mz = pressure = blitz = timestamped = modified = 0
            games, weeks = set(), set()
            for r in records(path):
                n += 1
                gid = r.get("game_id") or r.get("nflverse_game_id")
                if gid:
                    games.add(gid)
                    if len(gid.split("_")) >= 2:
                        weeks.add(gid.split("_")[1])
                if kind == "pbp" and r.get("season_type") == "REG":
                    if present(r.get("receiver_player_id")) and r.get("two_point_attempt") != "1" and r.get("play_type") == "pass":
                        targets += 1
                        air += present(r.get("air_yards"))
                        if r.get("complete_pass") == "1":
                            catches += 1
                            yac += present(r.get("yards_after_catch"))
                elif kind == "participation":
                    mz += r.get("defense_man_zone_type") in {"MAN_COVERAGE", "ZONE_COVERAGE"}
                    pressure += present(r.get("was_pressure"))
                elif kind == "ftn":
                    blitz += present(r.get("n_blitzers"))
                elif kind == "injuries":
                    timestamped += any(present(r.get(k)) for k in ("published_at", "report_timestamp", "timestamp", "dt"))
                    modified += present(r.get("date_modified"))
            meta.update(rows=n, games=len(games), max_game_week=max(weeks, default=None))
            if kind == "pbp":
                meta.update(regular_targets=targets, air_yards_present=air,
                            air_yards_fraction=air / targets if targets else None,
                            regular_catches=catches, yac_present=yac)
            if kind == "participation":
                meta.update(man_zone_labels=mz, true_pressure_labels=pressure)
            if kind == "ftn":
                meta["blitz_labels"] = blitz
            if kind == "injuries":
                meta.update(publication_timestamp_rows=timestamped, modification_timestamp_rows=modified,
                            timestamp_limit="date_modified exists but does not establish original publication or a cutoff snapshot",
                            feature_status="BLOCKED_DATA")
            s[kind] = meta
        out["seasons"][str(year)] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.fetch:
        retrieve(a.data_dir)
    result = audit(a.data_dir)
    Path(a.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for year, sources in result["seasons"].items():
        print(year, {k: {x: v for x, v in m.items() if x in {"status", "rows", "air_yards_fraction", "man_zone_labels", "true_pressure_labels", "blitz_labels", "publication_timestamp_rows"}} for k, m in sources.items()}, flush=True)


if __name__ == "__main__":
    main()
