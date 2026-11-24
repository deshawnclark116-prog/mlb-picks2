"""Storage-layer helper for the NHL V2 Phase1A-SOG forward ledger (research only; NOT part of the locked engine code).

Why: GitHub rejects any single file over 100 MiB, and the append-only ledger.jsonl has reached that size.
The engine, its lock and the ledger bytes are unchanged. Only how the ledger is stored in git changes:
  ledger.jsonl            working copy, read and appended by nhl_v2_phase1a_sog_forward.py (git-ignored)
  ledger.part-NNNN.jsonl  committed pieces, line-aligned, <= CHUNK bytes each; concatenated in order they equal ledger.jsonl byte for byte

Because the ledger is append-only and pieces are packed greedily by whole lines, an existing piece never changes once a later
piece exists; only the newest piece grows. pack() enforces this and refuses to rewrite history.

  python nhl_v2_ledger_pack.py unpack   rebuild ledger.jsonl from the pieces (do this after a fresh clone, before verify/run)
  python nhl_v2_ledger_pack.py pack     refresh the committed pieces from ledger.jsonl (do this before git add/commit)
  python nhl_v2_ledger_pack.py check    exit non-zero unless the pieces and ledger.jsonl are byte-identical
"""
import hashlib
import sys
from pathlib import Path

FWD = Path(__file__).resolve().parent / "nhl_models" / "nhl_player_outcome_v2" / "phase1a_sog_forward"
LEDGER = FWD / "ledger.jsonl"
CHUNK = 20 * 1024 * 1024
PREFIX, SUFFIX = "ledger.part-", ".jsonl"


def parts():
    return sorted(FWD.glob(PREFIX + "*" + SUFFIX))


def joined():
    return b"".join(p.read_bytes() for p in parts())


def split_lines(data, chunk=CHUNK):
    out, cur, size = [], [], 0
    for line in data.splitlines(keepends=True):
        if cur and size + len(line) > chunk:
            out.append(b"".join(cur)); cur, size = [], 0
        cur.append(line); size += len(line)
    if cur:
        out.append(b"".join(cur))
    return out


def pack():
    data = LEDGER.read_bytes()
    if data and not data.endswith(b"\n"):
        raise SystemExit("ledger.jsonl does not end with a newline; refusing to pack")
    want = split_lines(data)
    have = parts()
    if len(have) > len(want):
        raise SystemExit("more pieces on disk than the ledger needs; refusing (ledger shorter than committed pieces?)")
    for i, chunk in enumerate(want):
        p = FWD / ("%s%04d%s" % (PREFIX, i, SUFFIX))
        if p.exists():
            old = p.read_bytes()
            if old == chunk:
                continue
            if not chunk.startswith(old):
                raise SystemExit("piece %d would be rewritten, not appended to; refusing (append-only violated)" % i)
        p.write_bytes(chunk)
    if joined() != data:
        raise SystemExit("pieces do not rebuild ledger.jsonl exactly; refusing")
    print("packed %d pieces, %d bytes, sha256 %s" % (len(want), len(data), hashlib.sha256(data).hexdigest()))


def unpack():
    data = joined()
    if not data:
        raise SystemExit("no pieces found")
    if LEDGER.exists():
        cur = LEDGER.read_bytes()
        if cur == data:
            print("ledger.jsonl already matches pieces"); return
        if cur.startswith(data):
            print("ledger.jsonl is ahead of the pieces (unpacked rows not yet packed); leaving it"); return
        if not data.startswith(cur):
            raise SystemExit("ledger.jsonl and pieces diverge; refusing to overwrite")
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_bytes(data); tmp.replace(LEDGER)
    print("rebuilt ledger.jsonl: %d bytes, sha256 %s" % (len(data), hashlib.sha256(data).hexdigest()))


def check():
    if not LEDGER.exists() or LEDGER.read_bytes() != joined():
        raise SystemExit("MISMATCH: ledger.jsonl and pieces differ")
    print("ok: pieces rebuild ledger.jsonl exactly")


if __name__ == "__main__":
    cmds = {"pack": pack, "unpack": unpack, "check": check}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        raise SystemExit(__doc__)
    cmds[sys.argv[1]]()
