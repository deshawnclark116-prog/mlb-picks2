"""
NHL_OUTCOME_SNAPSHOT -- forward-only pregame snapshot recorder (Phase 0B prototype). Append-only observations; raw bytes (including the SCHEDULE response) stored by sha256.

Hard rules
  * the raw schedule bytes that establish each game's scheduled start are stored by sha256 and every endpoint observation references that schedule identity (hash + retrieval start / completion)
  * no pregame fetch is INITIATED at or after the scheduled puck drop (hard stop, no grace period)
  * a fetch that starts before puck drop but COMPLETES at or after it is REJECTED_LATE (bytes retained for audit, never pregame evidence)
  * a valid pregame observation requires retrieval_completed_at < scheduled_start_utc; minutes_to_start is computed from the COMPLETION timestamp
  * an early retrieval is never relabelled as a later horizon; a provider failure is recorded explicitly (no substitute source)
Time and HTTP are injectable (tests never wait or touch the network).

  python -u nhl_outcome_snapshot.py watch --root DIR [--horizon-minutes 150] [--every-sec 120] [--max-hours 3]
  python -u nhl_outcome_snapshot.py once  --root DIR --games ID...
"""
import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

API = "https://api-web.nhle.com/v1"
UTC = timezone.utc
HORIZONS_MIN = {"T24H": 1440, "T90": 90, "T30": 30, "T10": 10, "T2": 2}      # candidate labels only; NO horizon is validated (see the forward contract)
ENDPOINTS = ("landing", "right-rail", "boxscore", "play-by-play")
VALID, REJECTED_LATE, PROVIDER_ERROR, REFUSED = "VALID_PREGAME", "REJECTED_LATE", "PROVIDER_ERROR", "REFUSED_NOT_STARTED_BEFORE_PUCK_DROP"


def iso(t):
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_iso(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC) if "." in s else datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def real_http(url):
    req = urllib.request.Request(url, headers={"User-Agent": "nhl-snapshots"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
        hdr = {k: v for k, v in r.headers.items() if k.lower() in ("etag", "last-modified", "cache-control")}
        return raw, hdr, r.status


def nearest_label(minutes_to_start):
    lab = min(HORIZONS_MIN, key=lambda k: abs(HORIZONS_MIN[k] - minutes_to_start))
    return lab if abs(HORIZONS_MIN[lab] - minutes_to_start) <= max(3, 0.1 * HORIZONS_MIN[lab]) else "POLL"


def put_blob(root, raw):
    root = Path(root)
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(raw).hexdigest()
    p = root / "blobs" / sha
    if not p.exists():
        tmp = root / "blobs" / f".{sha}.tmp"
        tmp.write_bytes(raw); os.replace(tmp, p)
    return sha


def verify_blob(root, sha):
    """The stored bytes must hash-roundtrip."""
    return hashlib.sha256((Path(root) / "blobs" / sha).read_bytes()).hexdigest() == sha


def availability(ep, d):
    """Source-specific availability state. The goalie entries are NON-AUTHORITATIVE observations, never a confirmed starter."""
    if ep == "landing":
        m = d.get("matchup") or {}
        gc = m.get("goalieComparison") or {}
        txt = json.dumps(gc).lower()
        return {"has_matchup": bool(m), "has_goalie_comparison": bool(gc), "starter_like_field_present_unvalidated": ('"starter' in txt or "startingGoalie".lower() in txt)}
    if ep == "right-rail":
        gi = d.get("gameInfo") or {}
        return {"scratches_away": len((gi.get("awayTeam") or {}).get("scratches", [])), "scratches_home": len((gi.get("homeTeam") or {}).get("scratches", [])), "n_referees": len(gi.get("referees", []))}
    if ep == "boxscore":
        pb = d.get("playerByGameStats") or {}
        return {"n_skaters": sum(len(pb.get(s, {}).get(q, [])) for s in ("awayTeam", "homeTeam") for q in ("forwards", "defense")), "n_goalies": sum(len(pb.get(s, {}).get("goalies", [])) for s in ("awayTeam", "homeTeam")),
                "boxscore_starter_field_values_observed_unvalidated": [g.get("starter") for s in ("awayTeam", "homeTeam") for g in pb.get(s, {}).get("goalies", [])]}
    if ep == "play-by-play":
        rs = d.get("rosterSpots", [])
        return {"n_plays": len(d.get("plays", [])), "n_rosterSpots": len(rs), "n_rosterSpot_goalies": sum(1 for r in rs if r.get("positionCode") == "G")}
    return {}


def append_rows(root, rows):
    with open(Path(root) / "observations.jsonl", "ab") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True).encode() + b"\n")
        fh.flush(); os.fsync(fh.fileno())


def fetch_schedule(root, http=real_http, now_fn=lambda: datetime.now(UTC), url=f"{API}/schedule/now"):
    """Retrieve the schedule, store the RAW bytes by sha256, and return its identity + {game_id: scheduled start}."""
    started = now_fn()
    raw, hdr, status = http(url)
    completed = now_fn()
    sha = put_blob(root, raw)
    starts = {}
    for day in json.loads(raw)["gameWeek"]:
        for g in day["games"]:
            starts[g["id"]] = datetime.strptime(g["startTimeUTC"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    ident = {"schedule_sha256": sha, "schedule_url": url, "schedule_retrieval_started_at": iso(started), "schedule_retrieval_completed_at": iso(completed), "schedule_bytes": len(raw), "schedule_http_status": status}
    append_rows(root, [{"record_type": "schedule", **ident, "n_games": len(starts)}])
    return ident, starts


def observe(root, game, start, sched, http=real_http, now_fn=lambda: datetime.now(UTC)):
    """One observation per endpoint for `game`. `sched` = the identity returned by fetch_schedule (the schedule bytes that established `start`)."""
    Path(root).mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in ENDPOINTS:
        url = f"{API}/gamecenter/{game}/{ep}"
        base = {"record_type": "observation", "game_id": game, "endpoint": ep, "source_url": url, "scheduled_start_utc": iso(start), **{k: sched[k] for k in ("schedule_sha256", "schedule_retrieval_started_at", "schedule_retrieval_completed_at")}}
        started = now_fn()
        if started >= start:                                                   # hard stop at puck drop: nothing is even initiated
            rows.append({**base, "retrieval_started_at": iso(started), "observation_status": REFUSED, "http_status": None})
            continue
        try:
            raw, hdr, status = http(url)
            completed = now_fn()
            d = json.loads(raw)
        except Exception as e:                                                 # noqa  -- explicit failure, no alternate source
            rows.append({**base, "retrieval_started_at": iso(started), "retrieval_completed_at": iso(now_fn()), "observation_status": PROVIDER_ERROR, "http_status": getattr(e, "code", None), "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        sha = put_blob(root, raw)
        mts = (start - completed).total_seconds() / 60                         # from the COMPLETION timestamp
        valid = completed < start
        rows.append({**base, "retrieval_started_at": iso(started), "retrieval_completed_at": iso(completed), "retrieval_ts": iso(completed), "minutes_to_start": round(mts, 3),
                     "intended_horizon": nearest_label(mts) if valid else "NONE_REJECTED_LATE", "sha256": sha, "bytes": len(raw), "http_status": status,
                     "etag": hdr.get("ETag") or hdr.get("etag"), "last_modified": hdr.get("Last-Modified") or hdr.get("last-modified"), "game_state": d.get("gameState"),
                     "availability": availability(ep, d), "observation_status": VALID if valid else REJECTED_LATE})
    append_rows(root, rows)
    return rows


def watch(root, horizon_minutes, every, max_hours, t24_games=(), http=real_http, now_fn=lambda: datetime.now(UTC), sleep=time.sleep):
    t_end = now_fn() + timedelta(hours=max_hours)
    last = {}
    while now_fn() < t_end:
        sched, starts = fetch_schedule(root, http, now_fn)
        live = False
        for gid, start in sorted(starts.items(), key=lambda x: x[1]):
            mts = (start - now_fn()).total_seconds() / 60
            if mts <= 0:                                                       # hard stop at scheduled puck drop: no grace period
                continue
            near = mts <= horizon_minutes
            gap = every if near else 1800 if gid in t24_games else None
            if gap is None:
                continue
            live = True
            if gid not in last or (now_fn() - last[gid]).total_seconds() >= gap:
                observe(root, gid, start, sched, http, now_fn)
                last[gid] = now_fn()
        if not live:
            break
        sleep(30)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["watch", "once"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--horizon-minutes", type=int, default=150)
    ap.add_argument("--every-sec", type=int, default=120)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--games", nargs="*", type=int, default=[])
    ap.add_argument("--t24-games", nargs="*", type=int, default=[])
    a = ap.parse_args()
    if a.cmd == "once":
        sched, starts = fetch_schedule(a.root)
        for g in a.games:
            observe(a.root, g, starts[g], sched)
    else:
        watch(a.root, a.horizon_minutes, a.every_sec, a.max_hours, set(a.t24_games))
