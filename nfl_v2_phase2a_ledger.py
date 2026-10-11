#!/usr/bin/env python3
"""Append-only, hash-chained JSONL ledger primitives shared by the Phase2A forecast engine, grader and QB-state store (stdlib only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GENESIS = '0' * 64


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def rnd(v, nd=6):
    if isinstance(v, float):
        return round(v, nd) if v == v and abs(v) != float('inf') else None
    if isinstance(v, dict):
        return {k: rnd(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [rnd(x, nd) for x in v]
    return v


def canon(value):
    return json.dumps(rnd(value), sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def read_ledger(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_chain(path):
    """Raises ValueError if any line was altered, removed or reordered; returns (line_count, last_line_sha256)."""
    prev, n = GENESIS, 0
    for i, line in enumerate(read_ledger(path)):
        if line['seq'] != i + 1 or line['prev_line_sha256'] != prev:
            raise ValueError(f'ledger chain broken at line {i + 1}: {path}')
        if line['record_sha256'] != sha_bytes(canon(line['record']).encode()):
            raise ValueError(f'ledger record altered at line {i + 1}: {path}')
        if line['line_sha256'] != sha_bytes((prev + line['record_sha256']).encode()):
            raise ValueError(f'ledger line hash altered at line {i + 1}: {path}')
        prev, n = line['line_sha256'], n + 1
    return n, prev


def append_record(path, record, key_field):
    """Append one record; existing lines are never touched. A record whose key already exists is skipped (returns None)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n, prev = verify_chain(path)
    if any(x['record'].get(key_field) == record[key_field] for x in read_ledger(path)):
        return None
    rec_sha = sha_bytes(canon(record).encode())
    line = {'seq': n + 1, 'prev_line_sha256': prev, 'record_sha256': rec_sha, 'line_sha256': sha_bytes((prev + rec_sha).encode()), 'record': rnd(record)}
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(line, sort_keys=True, separators=(',', ':')) + '\n')
    return line
