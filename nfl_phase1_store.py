"""
NFL_PHASE1_STORE  (Phase 1C, shadow research)

Append-only, idempotent, crash-safe record store used by the forecast log and the score log.

  * Records are grouped in immutable BATCH files (one per game for forecasts): written to a temp file, fsynced, then atomically renamed.
    A batch has a header line, record lines, and a footer carrying the record count and the sha256 of every preceding byte.
  * A record has a deterministic `id`. Its identity bytes are the canonical JSON of the record (volatile fields such as generated_at live in the
    batch header, never in the record). Re-appending the same id with identical bytes is a verified no-op; the same id with different bytes is a HARD ERROR.
  * A finalized batch whose footer / hash does not verify raises StoreCorrupt. Temp files (crashed writers) are never read; `recover()` reports them.
  * Nothing is ever overwritten or deleted by this module.
"""
import hashlib
import json
import os
import stat
from pathlib import Path


class HardError(RuntimeError):
    pass


class StoreCorrupt(RuntimeError):
    pass


def canon(rec):
    """Canonical bytes of a record (sorted keys, floats rounded to 6 dp, no volatile fields)."""
    def fix(o):
        if isinstance(o, float):
            return round(o, 6)
        if isinstance(o, dict):
            return {k: fix(v) for k, v in sorted(o.items())}
        if isinstance(o, (list, tuple)):
            return [fix(v) for v in o]
        return o
    return json.dumps(fix(rec), sort_keys=True, separators=(",", ":")).encode()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def make_id(*parts):
    return sha("|".join(str(p) for p in parts).encode())[:32]


class Store:
    def __init__(self, root, kind):
        self.dir = Path(root) / kind
        (self.dir / "batches").mkdir(parents=True, exist_ok=True)

    # ---- reading / verification
    def batch_files(self):
        return sorted((self.dir / "batches").glob("*.jsonl"))

    def temp_files(self):
        return sorted((self.dir / "batches").glob("*.tmp"))

    def read_batch(self, path):
        raw = Path(path).read_bytes()
        lines = raw.split(b"\n")
        if lines and lines[-1] == b"":
            lines = lines[:-1]
        if len(lines) < 2:
            raise StoreCorrupt(f"{path}: too short")
        try:
            footer = json.loads(lines[-1])
            header = json.loads(lines[0])
        except Exception as e:
            raise StoreCorrupt(f"{path}: unparsable header/footer ({e})")
        if not footer.get("_footer"):
            raise StoreCorrupt(f"{path}: missing footer (partially written?)")
        body = b"\n".join(lines[:-1]) + b"\n"
        if sha(body) != footer["sha256"]:
            raise StoreCorrupt(f"{path}: hash mismatch (file modified or truncated)")
        recs = [json.loads(x) for x in lines[1:-1]]
        if len(recs) != footer["n"]:
            raise StoreCorrupt(f"{path}: record count mismatch")
        return header, recs

    def index(self):
        """id -> sha256(canonical bytes) over every finalized batch (all verified)."""
        idx = {}
        for f in self.batch_files():
            _, recs = self.read_batch(f)
            for r in recs:
                h = sha(canon(r))
                if r["id"] in idx and idx[r["id"]] != h:
                    raise HardError(f"conflicting bytes for id {r['id']} across batches")
                idx[r["id"]] = h
        return idx

    def all_records(self):
        out = []
        for f in self.batch_files():
            out.extend(self.read_batch(f)[1])
        return out

    def recover(self):
        """Report crashed writers' temp files (they are ignored, never loaded)."""
        return [str(p) for p in self.temp_files()]

    # ---- writing
    def append_batch(self, name, header, records):
        """Append records. Returns {'written': n, 'verified_duplicates': m}. Raises HardError on a same-id / different-bytes conflict."""
        idx = self.index()
        new, dup = [], 0
        seen, new_ids = {}, set()
        for r in records:
            if "id" not in r:
                raise HardError("record without id")
            h = sha(canon(r))
            if r["id"] in seen and seen[r["id"]] != h:
                raise HardError(f"conflicting duplicate id inside the batch: {r['id']}")
            seen[r["id"]] = h
            if r["id"] in idx:
                if idx[r["id"]] != h:
                    raise HardError(f"forecast id {r['id']} already stored with DIFFERENT bytes (refusing to overwrite)")
                dup += 1
                continue
            if r["id"] not in new_ids:
                new_ids.add(r["id"]); new.append(r)
        if not new:
            return {"written": 0, "verified_duplicates": dup}
        final = self.dir / "batches" / f"{name}.jsonl"
        head = json.dumps({**header, "_batch": name, "n": len(new)}, sort_keys=True).encode() + b"\n"
        body = head + b"".join(canon(r) + b"\n" for r in new)
        foot = json.dumps({"_footer": True, "n": len(new), "sha256": sha(body)}, sort_keys=True).encode() + b"\n"
        tmp = self.dir / "batches" / f"{name}.{os.getpid()}.tmp"
        with open(tmp, "xb") as fh:
            fh.write(body + foot); fh.flush(); os.fsync(fh.fileno())
        os.chmod(tmp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        if final.exists():                                   # a batch with this name exists but did not contain these ids: keep both, never overwrite
            k = 1
            while (self.dir / "batches" / f"{name}.part{k}.jsonl").exists():
                k += 1
            final = self.dir / "batches" / f"{name}.part{k}.jsonl"
        os.rename(tmp, final)                                # atomic on POSIX
        return {"written": len(new), "verified_duplicates": dup}
