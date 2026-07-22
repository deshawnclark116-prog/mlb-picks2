#!/usr/bin/env python3
"""
BACKUP_NFL_TO_REPO_A

Same problem as the MLB model backup (backup_models_to_repo_a.py), for NFL:
every NFL baseline dataset, trained model, calibration, and report JSON only
ever lived on Render's persistent disk (/data/nfl_model) -- the CODE that
produces them was in git (nfl_*.py), the actual artifacts and results were
not. Two consequences: (1) if that disk were ever lost, months of
receptions/rushing-yards pipeline work -- baselines, trained models, the
whole diagnostic trail -- would be gone with no way to recover it, and (2)
right now nobody can even tell whether the last committed step in either
pipeline (nfl_receptions_position_recalibration_a, nfl_rushing_yards_
recalibration_a) actually passed its gate, because those reports were never
pulled off the disk either.

Copies the ENTIRE /data/nfl_model tree (baselines, trained models, workdirs,
calibration.json files, *_report.json files, *.log files) into nfl_models/
in this git checkout, preserving the directory structure, and writes a
manifest (sha256 + size + mtime per file, relative path) so a future backup
run can show exactly what changed. Safe to re-run any time -- read-only on
/data/nfl_model, only writes into <repo>/nfl_models/.

This is one half of the fix; unlike the MLB model backup, there is
currently no auto-restore wired into anything (there's no NFL serving path
in api.py yet to restore into) -- that can be added once/if a market
actually gets deployed. For now this just makes sure the work is not
disk-bound.

Run (Render, from the repo root)
----------------------------------
python -u backup_nfl_to_repo_a.py
git add nfl_models/
git commit -m "Backup NFL pipeline data to repo"
git push origin main
"""

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SRC_DIR = Path("/data/nfl_model")
REPO_DIR = Path(__file__).resolve().parent / "nfl_models"

# Skip __pycache__ and anything that isn't a real artifact.
SKIP_NAMES = {"__pycache__"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not SRC_DIR.exists():
        print(f"ERROR: {SRC_DIR} not found -- run this on Render, not locally.")
        return 1

    REPO_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in SRC_DIR.rglob("*")
                    if p.is_file() and not any(part in SKIP_NAMES for part in p.parts))
    if not files:
        print(f"ERROR: no files found under {SRC_DIR}")
        return 1

    print("BACKUP_NFL_TO_REPO_A\n=====================")
    print(f"source: {SRC_DIR}\ndest:   {REPO_DIR}\n")

    manifest = {}
    total_bytes = 0
    for src in files:
        rel = src.relative_to(SRC_DIR)
        dst = REPO_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        changed = not dst.exists() or sha256_file(src) != sha256_file(dst)
        shutil.copy2(src, dst)
        size = src.stat().st_size
        total_bytes += size
        manifest[str(rel)] = {
            "sha256": sha256_file(dst),
            "size_bytes": size,
            "source_mtime_utc": datetime.fromtimestamp(
                src.stat().st_mtime, tz=timezone.utc).isoformat(),
        }
        print(f"  {'[changed]' if changed else '[same]   '} {rel}  ({size:,} bytes)")

    (REPO_DIR / "manifest.json").write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_files": len(files),
        "total_bytes": total_bytes,
        "files": manifest,
    }, indent=2))

    print(f"\n{len(files)} files, {total_bytes:,} bytes total, copied to {REPO_DIR}")

    # Surface any *_report.json verdicts directly, since that's the actual
    # open question this backup is meant to resolve.
    reports = sorted(REPO_DIR.rglob("*_report*.json"))
    if reports:
        print(f"\nFound {len(reports)} report file(s) -- verdicts:")
        for r in reports:
            try:
                data = json.loads(r.read_text())
                verdict = data.get("verdict", "(no verdict field)")
                print(f"  {r.relative_to(REPO_DIR)}: {verdict}")
            except Exception as e:
                print(f"  {r.relative_to(REPO_DIR)}: could not parse ({e})")

    print("\nNext steps (run these yourself -- not run automatically by this script):")
    print("  git add nfl_models/")
    print('  git commit -m "Backup NFL pipeline data to repo"')
    print("  git push origin main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
