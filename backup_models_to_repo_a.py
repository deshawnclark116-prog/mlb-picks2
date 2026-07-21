#!/usr/bin/env python3
"""
BACKUP_MODELS_TO_REPO_A

Every trained model only ever lived on Render's persistent disk
(/data/models) -- the CODE that produces them was in git, but the actual
gated, validated artifacts themselves were not. If that disk were ever
lost, wiped, or the service recreated, weeks of predictions-first
validation work would be gone with no way to recover it.

Copies every model file from MODEL_DIR into a models/ directory in this
git checkout, and writes a manifest (sha256 + size + mtime per file) so a
future backup run can show exactly what changed. Run this again any time a
model gets retrained/re-gated and promoted, to keep the git backup current
-- it's a plain copy, safe to re-run any time.

This is one half of the fix; the other half (load_models() in api.py
auto-restoring from this exact models/ directory into /data/models on
startup if a file is missing there) is already in place, so a fresh
deploy onto an empty /data self-heals without any manual steps, using
whatever was most recently committed here.

Read-only on /data/models. Writes only into <repo>/models/.

Run (Render, from the repo root)
----------------------------------
python -u backup_models_to_repo_a.py
git add models/
git commit -m "Backup trained models to repo"
git push origin main
"""

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

MODEL_DIR = Path("/data/models")
REPO_MODEL_DIR = Path(__file__).resolve().parent / "models"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not MODEL_DIR.exists():
        print(f"ERROR: {MODEL_DIR} not found -- run this on Render, not locally.")
        return 1

    REPO_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(MODEL_DIR.glob("*.json"))
    if not files:
        print(f"ERROR: no .json files found in {MODEL_DIR}")
        return 1

    print("BACKUP_MODELS_TO_REPO_A\n=======================")
    print(f"source: {MODEL_DIR}\ndest:   {REPO_MODEL_DIR}\n")

    manifest = {}
    total_bytes = 0
    for src in files:
        dst = REPO_MODEL_DIR / src.name
        changed = not dst.exists() or sha256_file(src) != sha256_file(dst)
        shutil.copy2(src, dst)
        size = src.stat().st_size
        total_bytes += size
        manifest[src.name] = {
            "sha256": sha256_file(dst),
            "size_bytes": size,
            "source_mtime_utc": datetime.fromtimestamp(
                src.stat().st_mtime, tz=timezone.utc).isoformat(),
        }
        print(f"  {'[changed]' if changed else '[same]   '} {src.name}  ({size:,} bytes)")

    (REPO_MODEL_DIR / "manifest.json").write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_files": len(files),
        "total_bytes": total_bytes,
        "files": manifest,
    }, indent=2))

    print(f"\n{len(files)} files, {total_bytes:,} bytes total, copied to {REPO_MODEL_DIR}")
    print("\nNext steps (run these yourself -- not run automatically by this script):")
    print("  git add models/")
    print('  git commit -m "Backup trained models to repo"')
    print("  git push origin main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
