"""
NHL_FWD_STATE (Phase 0C) -- durable, append-only forward-capture state: gzip content-addressed raw blobs + jsonl ledgers.

Blob identity = SHA256 of the UNCOMPRESSED raw response; file name blobs/<sha>.gz; every read decompresses, re-hashes and verifies; the same sha with different bytes is a hard error.
Ledgers are append-only jsonl (fsync'd); immutable record kinds never change once written. Only derived summaries (status.json / readiness.json) are overwritten.
Stdlib only. No model or production code.
"""
import gzip
import threading
import uuid
import hashlib
import io
import json
import os
from pathlib import Path


class HardError(RuntimeError):
    pass


def sha256_hex(b):
    return hashlib.sha256(b).hexdigest()


def det_gzip(raw):
    bio = io.BytesIO()
    with gzip.GzipFile(fileobj=bio, mode="wb", mtime=0, compresslevel=9) as g:
        g.write(raw)
    return bio.getvalue()


class BlobStore:
    def __init__(self, root):
        self.dir = Path(root) / "blobs"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, sha):
        return self.dir / f"{sha}.gz"

    def get(self, sha):
        p = self.path(sha)
        if not p.exists():
            raise HardError(f"missing blob {sha}")
        raw = gzip.decompress(p.read_bytes())
        if sha256_hex(raw) != sha:
            raise HardError(f"blob {sha} failed hash verification (stored bytes differ)")
        return raw

    def put(self, raw):
        """Store raw bytes under their own sha. An existing blob is re-verified (never rewritten); a mismatch is a HARD ERROR."""
        if not isinstance(raw, (bytes, bytearray)) or not len(raw):
            raise HardError("empty or non-bytes payload")
        sha = sha256_hex(bytes(raw))
        p = self.path(sha)
        if p.exists():
            self.get(sha)                                         # raises HardError if the stored bytes are not the bytes named by the file
            return {"sha256": sha, "bytes": len(raw), "stored": "existing"}
        tmp = self.dir / f".{sha}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}.tmp"
        with open(tmp, "xb") as fh:
            fh.write(det_gzip(bytes(raw))); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, p)
        return {"sha256": sha, "bytes": len(raw), "stored": "new"}

    def put_as(self, sha, raw):
        """Store under an EXPECTED identity: the bytes must hash to it (same id + different bytes = HardError)."""
        if sha256_hex(bytes(raw)) != sha:
            raise HardError(f"bytes do not hash to {sha}")
        return self.put(raw)

    def verify_all(self):
        bad = []
        for p in sorted(self.dir.glob("*.gz")):
            try:
                self.get(p.stem)
            except Exception as e:                                # noqa
                bad.append((p.name, str(e)))
        return bad

    def size_bytes(self):
        return sum(p.stat().st_size for p in self.dir.glob("*.gz"))


class Ledger:
    """Append-only jsonl. `append_unique(key_fields, row)`: a row whose key already exists is skipped when identical in `immutable` fields, and is a HardError when those differ."""

    def __init__(self, root, name):
        self.path = Path(root) / name
        Path(root).mkdir(parents=True, exist_ok=True)

    def read(self):
        if not self.path.exists():
            return []
        rows = []
        for i, line in enumerate(self.path.read_bytes().split(b"\n")):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception as e:                            # noqa
                    raise HardError(f"{self.path}: unparsable line {i}: {e}")
        return rows

    def append(self, row):
        with open(self.path, "ab") as fh:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"); fh.flush(); os.fsync(fh.fileno())

    def append_unique(self, key_fields, row, immutable=()):
        key = tuple(row[k] for k in key_fields)
        for r in self.read():
            if tuple(r.get(k) for k in key_fields) == key:
                for f in immutable:
                    if r.get(f) != row.get(f):
                        raise HardError(f"{self.path.name}: key {key} already recorded with a different {f} ({r.get(f)!r} vs {row.get(f)!r})")
                return r, False
        self.append(row)
        return row, True


def write_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)
