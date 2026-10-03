"""
NFL_PHASE1_SNAPSHOTS  (Phase 1 hardening; used by the clean-forward system)

Why this exists
---------------
The nflverse injury files carry no row-level publication timestamps, so historical development can only
evaluate the availability model as a "historical final-weekly-report proxy under the documented nflverse
timing assumption" (nfl_phase1_data.AS_OF_ASSUMPTIONS A2). It cannot prove that a given row was public at T-24h.

For the CLEAN-FORWARD system that gap is closed by construction. At every forecast horizon we:
  1. download the current source file,
  2. hash it (sha256),
  3. record the retrieval timestamp,
  4. keep the bytes as an immutable snapshot (never overwritten, made read-only),
  5. forecast ONLY from that saved snapshot (load_snapshot verifies the hash again),
and repeat independently at T-90m (a new download, a new file). An append-only manifest records every snapshot.

A snapshot proves WHEN WE saw the data (retrieval time), not when the source published it; a snapshot retrieved
at T-24h can never contain information that existed after T-24h, which is the property the forecast needs.

Nothing here reads a sportsbook source.
"""
import hashlib
import json
import os
import stat
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc
ROOT = Path(__file__).resolve().parent / "nfl_forward_snapshots"
HORIZONS = {"T24": timedelta(hours=24), "T90": timedelta(minutes=90)}
RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
SOURCES = {   # source name -> URL template ({season})
    "injuries": RELEASE + "/injuries/injuries_{season}.csv",
    "weekly_rosters": RELEASE + "/weekly_rosters/roster_weekly_{season}.csv",
    "depth_charts": RELEASE + "/depth_charts/depth_charts_{season}.csv",
}


class SnapshotError(RuntimeError):
    pass


def utc_now():
    return datetime.now(UTC)


def sha256_hex(b):
    return hashlib.sha256(b).hexdigest()


def iso(ts):
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_iso(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


def http_fetcher(url, timeout=120):
    def fetch():
        req = urllib.request.Request(url, headers={"User-Agent": "nfl-phase1-snapshots"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    return fetch


def forecast_time(kickoff, horizon):
    if horizon not in HORIZONS:
        raise SnapshotError(f"unknown horizon {horizon}")
    return kickoff - HORIZONS[horizon]


def take_snapshot(source, fetch, kickoff, horizon, root=ROOT, now=None, tag_url=None):
    """Download -> hash -> store immutably -> append to the manifest. Returns the manifest record.

    Raises SnapshotError when (a) the retrieval happens after the forecast time of the requested horizon (the snapshot
    would be unusable and must not be created for that forecast), or (b) the target path already exists (never overwrite).
    """
    root = Path(root)
    f_ts = forecast_time(kickoff, horizon)
    retrieved = (now or utc_now())
    if retrieved > f_ts:
        raise SnapshotError(f"{source}/{horizon}: retrieval {iso(retrieved)} is after the forecast time {iso(f_ts)}")
    data = fetch()
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise SnapshotError(f"{source}/{horizon}: empty or non-bytes payload")
    digest = sha256_hex(data)
    d = root / source
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{horizon}_{retrieved.strftime('%Y%m%dT%H%M%S%fZ')}_{digest[:16]}.snapshot"
    try:
        with open(path, "xb") as fh:                       # 'x' = fail if it exists: no overwrite, ever
            fh.write(data); fh.flush(); os.fsync(fh.fileno())
    except FileExistsError:
        raise SnapshotError(f"snapshot already exists and is immutable: {path}")
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    rec = {"source": source, "horizon": horizon, "kickoff": iso(kickoff), "forecast_ts": iso(f_ts), "retrieval_ts": iso(retrieved),
           "sha256": digest, "bytes": len(data), "path": str(path.relative_to(root)), "url": tag_url}
    with open(root / "manifest.jsonl", "a") as mf:         # append-only
        mf.write(json.dumps(rec, sort_keys=True) + "\n"); mf.flush(); os.fsync(mf.fileno())
    return rec


def read_manifest(root=ROOT):
    p = Path(root) / "manifest.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def load_snapshot(rec, root=ROOT):
    """Read a snapshot back and re-verify its hash. Forecasts use ONLY bytes obtained through this function."""
    path = Path(root) / rec["path"]
    data = path.read_bytes()
    if sha256_hex(data) != rec["sha256"]:
        raise SnapshotError(f"hash mismatch for {path}: snapshot was modified")
    return data


def usable_snapshot(source, horizon, kickoff, root=ROOT):
    """The latest manifest record for (source, horizon, kickoff) retrieved at or before the horizon's forecast time."""
    f_ts = forecast_time(kickoff, horizon)
    cands = [r for r in read_manifest(root) if r["source"] == source and r["horizon"] == horizon and r["kickoff"] == iso(kickoff)
             and parse_iso(r["retrieval_ts"]) <= f_ts]
    if not cands:
        raise SnapshotError(f"no usable {source}/{horizon} snapshot for kickoff {iso(kickoff)}")
    return max(cands, key=lambda r: r["retrieval_ts"])


def forecast_inputs(sources, kickoff, horizon, root=ROOT):
    """Bytes + provenance for every requested source, all from saved, hash-verified snapshots whose retrieval time is
    <= the forecast time. The provenance dict is what the forecast log stores next to each availability forecast."""
    out, prov = {}, {}
    f_ts = forecast_time(kickoff, horizon)
    for s in sources:
        rec = usable_snapshot(s, horizon, kickoff, root)
        if parse_iso(rec["retrieval_ts"]) > f_ts:
            raise SnapshotError("snapshot retrieved after forecast time")
        out[s] = load_snapshot(rec, root)
        prov[s] = {k: rec[k] for k in ("sha256", "retrieval_ts", "forecast_ts", "horizon", "path", "bytes")}
    return out, prov


def verify_all(root=ROOT):
    """Audit: every manifest line parses, the file exists, the hash matches, the file is read-only, retrieval times never
    move backwards in the manifest (append-only), and no path is listed twice."""
    problems, seen, last = [], set(), None
    for i, rec in enumerate(read_manifest(root)):
        p = Path(root) / rec["path"]
        if not p.exists():
            problems.append(f"line {i}: missing file {rec['path']}")
            continue
        if sha256_hex(p.read_bytes()) != rec["sha256"]:
            problems.append(f"line {i}: hash mismatch {rec['path']}")
        if os.access(p, os.W_OK) and os.geteuid() != 0:
            problems.append(f"line {i}: snapshot is writable {rec['path']}")
        if rec["path"] in seen:
            problems.append(f"line {i}: duplicate path {rec['path']}")
        seen.add(rec["path"])
        ts = parse_iso(rec["retrieval_ts"])
        if last is not None and ts < last:
            problems.append(f"line {i}: retrieval time moves backwards")
        last = ts
    return problems


def snapshot_plan(kickoff, seasons=(2026,)):
    """The retrieval schedule for one game: two independent downloads per source (T-24h and T-90m)."""
    plan = []
    for h in ("T24", "T90"):
        for s in ("injuries", "weekly_rosters", "depth_charts"):
            for season in seasons:
                plan.append({"source": s, "horizon": h, "forecast_ts": iso(forecast_time(kickoff, h)),
                             "url": SOURCES[s].format(season=season)})
    return plan
