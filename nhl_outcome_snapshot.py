"""
NHL_OUTCOME_SNAPSHOT -- forward-only pregame snapshot recorder (Phase 0B prototype). Append-only observations; raw bytes stored by sha256.

Every observation records: retrieval timestamp (after the last byte arrived), intended horizon label, game start (from the SAME retrieval's schedule entry), source URL, sha256 of the raw
bytes, and a source-specific availability state. Nothing is relabelled: `intended_horizon` is the nearest scheduled horizon the observation was taken for, `minutes_to_start` is the exact value.

  python -u nhl_outcome_snapshot.py watch --root DIR [--horizon-minutes 150] [--every-sec 120] [--max-hours 3]
  python -u nhl_outcome_snapshot.py once  --root DIR --games ID...
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

API = "https://api-web.nhle.com/v1"
UTC = timezone.utc
HORIZONS_MIN = {"T24H": 1440, "T90": 90, "T30": 30, "T10": 10, "T2": 2}      # label -> minutes before start (candidates; the observed timing decides the recommended set)
ENDPOINTS = ("landing", "right-rail", "boxscore", "play-by-play")


def iso(t):
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def http(url):
    t0 = datetime.now(UTC)
    req = urllib.request.Request(url, headers={"User-Agent": "nhl-snapshots"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
        hdr = {k: v for k, v in r.headers.items() if k.lower() in ("etag", "last-modified", "cache-control")}
    return raw, datetime.now(UTC), hdr


def nearest_label(minutes_to_start):
    lab = min(HORIZONS_MIN, key=lambda k: abs(HORIZONS_MIN[k] - minutes_to_start))
    return lab if abs(HORIZONS_MIN[lab] - minutes_to_start) <= max(3, 0.1 * HORIZONS_MIN[lab]) else "POLL"


def availability(ep, d):
    """Source-specific availability state of the fields we care about."""
    if ep == "landing":
        m = d.get("matchup") or {}
        gc = m.get("goalieComparison") or {}
        return {"has_matchup": bool(m), "has_goalie_comparison": bool(gc), "goalie_starter_field": any("start" in k.lower() for k in json.dumps(gc).lower().split('"') if k.isalpha())}
    if ep == "right-rail":
        gi = d.get("gameInfo") or {}
        return {"scratches_away": len((gi.get("awayTeam") or {}).get("scratches", [])), "scratches_home": len((gi.get("homeTeam") or {}).get("scratches", [])), "n_referees": len(gi.get("referees", []))}
    if ep == "boxscore":
        pb = d.get("playerByGameStats") or {}
        return {"n_skaters": sum(len(pb.get(s, {}).get(q, [])) for s in ("awayTeam", "homeTeam") for q in ("forwards", "defense")), "n_goalies": sum(len(pb.get(s, {}).get("goalies", [])) for s in ("awayTeam", "homeTeam")),
                "starter_flags": [g.get("starter") for s in ("awayTeam", "homeTeam") for g in pb.get(s, {}).get("goalies", [])]}
    if ep == "play-by-play":
        rs = d.get("rosterSpots", [])
        return {"n_plays": len(d.get("plays", [])), "n_rosterSpots": len(rs), "n_rosterSpot_goalies": sum(1 for r in rs if r.get("positionCode") == "G")}
    return {}


def observe(root, game, start, now_fn=lambda: datetime.now(UTC)):
    root = Path(root)
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in ENDPOINTS:
        url = f"{API}/gamecenter/{game}/{ep}"
        try:
            raw, ts, hdr = http(url)
            d = json.loads(raw)
            sha = hashlib.sha256(raw).hexdigest()
            p = root / "blobs" / sha
            if not p.exists():
                p.write_bytes(raw)
            mts = (start - ts).total_seconds() / 60
            rows.append({"game_id": game, "endpoint": ep, "url": url, "retrieval_ts": iso(ts), "game_start": iso(start), "minutes_to_start": round(mts, 2), "intended_horizon": nearest_label(mts),
                         "sha256": sha, "bytes": len(raw), "http_status": 200, "etag": hdr.get("ETag") or hdr.get("etag"), "game_state": d.get("gameState"), "availability": availability(ep, d)})
        except Exception as e:                                     # noqa
            rows.append({"game_id": game, "endpoint": ep, "url": url, "retrieval_ts": iso(now_fn()), "game_start": iso(start), "http_status": getattr(e, "code", None), "error": f"{type(e).__name__}: {e}"[:200]})
    with open(root / "observations.jsonl", "ab") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True).encode() + b"\n")
        fh.flush(); os.fsync(fh.fileno())
    return rows


def schedule():
    raw, ts, _ = http(f"{API}/schedule/now")
    out = {}
    for day in json.loads(raw)["gameWeek"]:
        for g in day["games"]:
            out[g["id"]] = datetime.strptime(g["startTimeUTC"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return out, ts


def watch(root, horizon_minutes, every, max_hours, t24_games=()):
    t_end = datetime.now(UTC) + timedelta(hours=max_hours)
    last = {}
    while datetime.now(UTC) < t_end:
        sch, ts = schedule()
        for gid, start in sorted(sch.items(), key=lambda x: x[1]):
            mts = (start - datetime.now(UTC)).total_seconds() / 60
            if mts < -5:                                           # no pregame observation after the game started
                continue
            near = mts <= horizon_minutes
            gap = every if near else 1800 if gid in t24_games else None
            if gap is None:
                continue
            if gid not in last or (datetime.now(UTC) - last[gid]).total_seconds() >= gap:
                observe(root, gid, start)
                last[gid] = datetime.now(UTC)
        if not any((s - datetime.now(UTC)).total_seconds() / 60 > -5 and ((s - datetime.now(UTC)).total_seconds() / 60 <= horizon_minutes or g in t24_games) for g, s in sch.items()):
            break
        time.sleep(30)


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
        sch, _ = schedule()
        for g in a.games:
            observe(a.root, g, sch[g])
    else:
        watch(a.root, a.horizon_minutes, a.every_sec, a.max_hours, set(a.t24_games))
