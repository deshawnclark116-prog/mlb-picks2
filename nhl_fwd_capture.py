"""
NHL_FWD_CAPTURE (Phase 0C) -- registered capture timing, decision keys, grouped concurrent horizon capture, schedule provenance, postgame truth. Collects hockey information only: NO prediction, NO model.

Registered BEFORE any compliant evidence (see nhl_models/nhl_outcome_engine/phase0c_capture_protocol.json):
  cutoff = scheduled_start_utc - horizon; decision key = (game_id, horizon, scheduled_start_utc)
  capture starts CAPTURE_START_LEAD_SECONDS before the cutoff; a horizon observation qualifies only if it STARTED before the cutoff and COMPLETED in [cutoff - 180 s, cutoff]
  early -> EARLY_NOT_HORIZON_EVIDENCE, after the cutoff -> MISSED_HORIZON_CUTOFF, initiated at/after puck drop -> REFUSED; no backfill, no relabel
Time, sleep, http and the worker map are injectable.
"""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nhl_fwd_state as S
import nhl_outcome_snapshot as SN

UTC = timezone.utc
API = "https://api-web.nhle.com/v1"
PROTOCOL_VERSION = "nhl-forward-capture-protocol-1"
HORIZONS = {"T24H": 1440, "T90": 90, "T30": 30, "T10": 10, "T2": 2}
HORIZON_ORDER = ["T24H", "T90", "T30", "T10", "T2"]
CAPTURE_START_LEAD_SECONDS = 120
WAKE_LEAD_SECONDS = 1200
VALID_WINDOW_SECONDS = 180
MAX_WORKERS = 8
GROUP_TOLERANCE_SECONDS = 60
REQUIRED_ENDPOINTS = ("landing", "right-rail", "boxscore", "play-by-play")
POSTGAME_ENDPOINTS = ("boxscore", "play-by-play", "right-rail")
POSTGAME_DELAY_MINUTES = 150
PREGAME_STATES = ("FUT", "PRE")
FINAL_STATES = ("OFF", "FINAL")
COMPLETE_VALID = "COMPLETE_VALID"
MISSED, EARLY, REFUSED, PROVIDER_ERROR, NOT_PREGAME, VALID = "MISSED_HORIZON_CUTOFF", "EARLY_NOT_HORIZON_EVIDENCE", "REFUSED", "PROVIDER_ERROR", "REJECTED_NOT_PREGAME_STATE", "VALID_HORIZON"
REVISED = "SCHEDULE_REVISED_NEW_KEY"
PROVIDER_FAILURE = "PROVIDER_FAILURE"
STATUS_PRIORITY = [REFUSED, MISSED, PROVIDER_FAILURE, NOT_PREGAME, EARLY]          # worst first; COMPLETE_VALID only when every required endpoint is VALID_HORIZON


def iso(t):
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_iso(s):
    return SN.parse_iso(s)


def cutoff_of(start, horizon):
    return start - timedelta(minutes=HORIZONS[horizon])


def key_of(game_id, horizon, start_iso):
    return f"{game_id}|{horizon}|{start_iso}"


class State:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.blobs = S.BlobStore(self.root)
        self.schedule = S.Ledger(self.root, "schedule_observations.jsonl")
        self.planned = S.Ledger(self.root, "planned_keys.jsonl")
        self.obs = S.Ledger(self.root, "endpoint_observations.jsonl")
        self.results = S.Ledger(self.root, "horizon_results.jsonl")
        self.postgame = S.Ledger(self.root, "postgame_truth.jsonl")
        self.lock = threading.Lock()


# ------------------------------------------------------------------ schedule + planning
def parse_schedule(raw):
    out = {}
    for day in json.loads(raw)["gameWeek"]:
        for g in day["games"]:
            if g.get("gameType") != 2:
                continue
            out[g["id"]] = {"start": datetime.strptime(g["startTimeUTC"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC), "start_iso_raw": g["startTimeUTC"], "game_date": day["date"], "state": g.get("gameState"),
                            "away": g["awayTeam"]["abbrev"], "home": g["homeTeam"]["abbrev"]}
    return out


def observe_schedule(st, http, clock, url=f"{API}/schedule/now"):
    """One schedule retrieval: started / completed timestamps, raw sha. The raw blob is stored by plan_keys when it establishes a new decision key."""
    started = clock()
    raw, hdr, status = http(url)
    completed = clock()
    return {"raw": raw, "schedule_sha256": S.sha256_hex(raw), "schedule_url": url, "schedule_retrieval_started_at": iso(started), "schedule_retrieval_completed_at": iso(completed), "schedule_http_status": status, "parsed": parse_schedule(raw)}


def plan_keys(st, sched, now):
    """Plan one decision key per (game, horizon, scheduled start) for games not yet started. A revised start creates NEW keys and closes the old open keys; nothing old is mutated."""
    parsed = sched["parsed"]
    existing = {r["key"]: r for r in st.planned.read()}
    done = {r["key"] for r in st.results.read()}
    new_rows, revised = [], []
    for gid, g in sorted(parsed.items()):
        if g["state"] not in PREGAME_STATES or g["start"] <= now:
            continue
        start_iso = iso(g["start"])
        for r in existing.values():
            if r["game_id"] == gid and r["scheduled_start_utc"] != start_iso and r["key"] not in done:
                revised.append(r)
        for hz in HORIZON_ORDER:
            k = key_of(gid, hz, start_iso)
            if k in existing:
                continue
            new_rows.append({"key": k, "game_id": gid, "horizon": hz, "scheduled_start_utc": start_iso, "cutoff": iso(cutoff_of(g["start"], hz)), "game_date": g["game_date"], "away": g["away"], "home": g["home"],
                             "schedule_sha256": sched["schedule_sha256"], "schedule_retrieval_started_at": sched["schedule_retrieval_started_at"], "schedule_retrieval_completed_at": sched["schedule_retrieval_completed_at"], "planned_at": iso(now)})
    blob_stored = False
    if new_rows:
        st.blobs.put_as(sched["schedule_sha256"], sched["raw"])               # the exact raw response that established these starts
        blob_stored = True
    st.schedule.append({"schedule_sha256": sched["schedule_sha256"], "url": sched["schedule_url"], "retrieval_started_at": sched["schedule_retrieval_started_at"], "retrieval_completed_at": sched["schedule_retrieval_completed_at"],
                        "http_status": sched["schedule_http_status"], "n_games": len(parsed), "blob_stored": blob_stored, "new_keys": len(new_rows)})
    for r in new_rows:
        st.planned.append_unique(["key"], r, immutable=("scheduled_start_utc", "cutoff"))
        if parse_iso(r["cutoff"]) <= now:                                           # first seen after the cutoff: never backfilled
            st.results.append_unique(["key"], {"key": r["key"], "game_id": r["game_id"], "horizon": r["horizon"], "scheduled_start_utc": r["scheduled_start_utc"], "status": MISSED, "reason": "first_seen_after_cutoff", "finalized_at": iso(now),
                                               "schedule_sha256": r["schedule_sha256"]})
    for r in revised:
        st.results.append_unique(["key"], {"key": r["key"], "game_id": r["game_id"], "horizon": r["horizon"], "scheduled_start_utc": r["scheduled_start_utc"], "status": REVISED, "reason": "scheduled start changed; evidence of this key stays immutable",
                                           "finalized_at": iso(now), "schedule_sha256": r["schedule_sha256"]})
    return new_rows, revised


# ------------------------------------------------------------------ endpoint capture
def classify(started, completed, cutoff, start, game_state):
    if started >= start:
        return REFUSED
    if started >= cutoff or completed > cutoff:
        return MISSED
    if game_state is not None and game_state not in PREGAME_STATES:                  # right-rail carries no gameState
        return NOT_PREGAME
    if completed < cutoff - timedelta(seconds=VALID_WINDOW_SECONDS):
        return EARLY
    return VALID


def capture_endpoint(st, krow, ep, http, clock):
    """One endpoint observation for a decision key; an existing verified observation is reused (restart safety)."""
    key = krow["key"]
    for r in st.obs.read():
        if r["key"] == key and r["endpoint"] == ep:
            if r.get("raw_sha256"):
                st.blobs.get(r["raw_sha256"])                                     # verify the stored bytes on reuse
            return r, False
    start, cutoff = parse_iso(krow["scheduled_start_utc"]), parse_iso(krow["cutoff"])
    url = f"{API}/gamecenter/{krow['game_id']}/{ep}"
    base = {"key": key, "game_id": krow["game_id"], "horizon": krow["horizon"], "endpoint": ep, "source_url": url, "scheduled_start_utc": krow["scheduled_start_utc"], "cutoff": krow["cutoff"],
            "schedule_sha256": krow["schedule_sha256"], "schedule_retrieval_started_at": krow["schedule_retrieval_started_at"], "schedule_retrieval_completed_at": krow["schedule_retrieval_completed_at"], "label": "PREGAME_HORIZON_CAPTURE"}
    started = clock()
    if started >= start:                                                          # hard stop at puck drop: nothing is initiated
        row = {**base, "retrieval_started_at": iso(started), "observation_status": REFUSED, "http_status": None}
    elif started >= cutoff:
        row = {**base, "retrieval_started_at": iso(started), "observation_status": MISSED, "reason": "not_initiated_before_cutoff", "http_status": None}
    else:
        try:
            raw, hdr, status = http(url)
            completed = clock()
            d = json.loads(raw)
        except Exception as e:                                                    # noqa  -- explicit failure, no substitute source
            row = {**base, "retrieval_started_at": iso(started), "retrieval_completed_at": iso(clock()), "observation_status": PROVIDER_ERROR, "http_status": getattr(e, "code", None), "error": f"{type(e).__name__}: {e}"[:200]}
        else:
            info = st.blobs.put(raw)
            status_ = classify(started, completed, cutoff, start, d.get("gameState"))
            row = {**base, "retrieval_started_at": iso(started), "retrieval_completed_at": iso(completed), "retrieval_ts": iso(completed), "seconds_before_cutoff": round((cutoff - completed).total_seconds(), 3),
                   "minutes_to_start": round((start - completed).total_seconds() / 60, 3), "raw_sha256": info["sha256"], "bytes": info["bytes"], "http_status": status, "etag": hdr.get("ETag") or hdr.get("etag"),
                   "game_state": d.get("gameState"), "availability": SN.availability(ep, d), "observation_status": status_}
    with st.lock:
        return st.obs.append_unique(["key", "endpoint"], row, immutable=("raw_sha256",))[0], True


def finalize_key(st, krow, now):
    key = krow["key"]
    if any(r["key"] == key for r in st.results.read()):
        return None
    obs = {r["endpoint"]: r for r in st.obs.read() if r["key"] == key}
    cutoff = parse_iso(krow["cutoff"])
    statuses = {ep: (obs[ep]["observation_status"] if ep in obs else (MISSED if now > cutoff else None)) for ep in REQUIRED_ENDPOINTS}
    if any(v is None for v in statuses.values()):
        return None                                                               # still capturing (cutoff not reached)
    if all(v == VALID for v in statuses.values()):
        gs = COMPLETE_VALID
    else:
        bad = {PROVIDER_ERROR: PROVIDER_FAILURE}
        present = {bad.get(v, v) for v in statuses.values()}
        gs = next((s for s in STATUS_PRIORITY if s in present), MISSED)
    row = {"key": key, "game_id": krow["game_id"], "horizon": krow["horizon"], "scheduled_start_utc": krow["scheduled_start_utc"], "cutoff": krow["cutoff"], "status": gs, "endpoint_statuses": statuses, "finalized_at": iso(now),
           "schedule_sha256": krow["schedule_sha256"], "max_retrieval_completed_at": max((o["retrieval_completed_at"] for o in obs.values() if o.get("retrieval_completed_at")), default=None)}
    return st.results.append_unique(["key"], row, immutable=("status",))[0]


def capture_key(st, krow, http, clock):
    """All required endpoints of one decision key, deterministic order; returns the finalized result (or None while pending)."""
    for ep in REQUIRED_ENDPOINTS:
        capture_endpoint(st, krow, ep, http, clock)
    return finalize_key(st, krow, clock())


def default_map(fn, items, workers=MAX_WORKERS):
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, workers, len(items))) as ex:
        return list(ex.map(fn, items))


def due_groups(st, now, wake_seconds=WAKE_LEAD_SECONDS):
    """Open keys whose cutoff is within the wake window (or already past, to finalize / mark missed), grouped by (near-)identical cutoff. Games with the same puck time share a group."""
    done = {r["key"] for r in st.results.read()}
    open_keys = [r for r in st.planned.read() if r["key"] not in done]
    due = [r for r in open_keys if (parse_iso(r["cutoff"]) - now).total_seconds() <= wake_seconds]
    due.sort(key=lambda r: (r["cutoff"], r["game_id"], r["horizon"]))
    groups = []
    for r in due:
        c = parse_iso(r["cutoff"])
        if groups and (c - parse_iso(groups[-1][0]["cutoff"])).total_seconds() <= GROUP_TOLERANCE_SECONDS:
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


# ------------------------------------------------------------------ postgame truth
def derive_truth(box, pbp):
    sk, gl, sog, team_sog = {}, {}, {}, {}
    for side in ("awayTeam", "homeTeam"):
        t = box[side]["abbrev"]
        pb = box["playerByGameStats"][side]
        sk[t] = sorted(p["playerId"] for pos in ("forwards", "defense") for p in pb[pos])
        gl[t] = sorted(p["playerId"] for p in pb["goalies"])
        for pos in ("forwards", "defense"):
            for p in pb[pos]:
                sog[str(p["playerId"])] = p.get("sog", 0)
        team_sog[t] = box[side].get("sog")
    return {"final_dressed_skater_ids_by_team": sk, "final_goalie_ids_by_team": gl, "final_sog_by_skater": sog, "final_team_sog": team_sog}


def capture_postgame(st, game_id, planned_row, http, clock):
    """Exactly one POSTGAME_TRUTH bundle per game, only when the boxscore is FINAL/OFF; idempotent. Never a pregame feature."""
    if any(r["game_id"] == game_id for r in st.postgame.read()):
        return None
    ends = {}
    raws = {}
    for ep in POSTGAME_ENDPOINTS:
        started = clock()
        try:
            raw, hdr, status = http(f"{API}/gamecenter/{game_id}/{ep}")
            completed = clock()
        except Exception as e:                                                    # noqa
            if ep == "right-rail":
                ends[ep] = {"status": PROVIDER_ERROR, "error": f"{type(e).__name__}: {e}"[:150]}
                continue
            return {"status": "POSTGAME_PENDING_OR_FAILED", "game_id": game_id, "error": f"{ep}: {type(e).__name__}: {e}"[:150]}
        d = json.loads(raw)
        if ep == "boxscore" and d.get("gameState") not in FINAL_STATES:
            return {"status": "POSTGAME_PENDING", "game_id": game_id, "game_state": d.get("gameState")}
        info = st.blobs.put(raw)
        raws[ep] = d
        ends[ep] = {"raw_sha256": info["sha256"], "bytes": info["bytes"], "retrieval_started_at": iso(started), "retrieval_completed_at": iso(completed), "http_status": status, "game_state": d.get("gameState"), "source_url": f"{API}/gamecenter/{game_id}/{ep}"}
    row = {"game_id": game_id, "label": "POSTGAME_TRUTH", "never_a_pregame_feature": True, "game_date": planned_row["game_date"], "scheduled_start_utc": planned_row["scheduled_start_utc"], "endpoints": ends, **derive_truth(raws["boxscore"], raws["play-by-play"])}
    with st.lock:
        return st.postgame.append_unique(["game_id"], row, immutable=("endpoints",))[0]


def postgame_candidates(st, now, limit=60):
    have = {r["game_id"] for r in st.postgame.read()}
    latest = {}
    for r in st.planned.read():                                                   # the latest scheduled start of a game is the one it was played at
        if r["game_id"] not in latest or r["scheduled_start_utc"] > latest[r["game_id"]]["scheduled_start_utc"]:
            latest[r["game_id"]] = r
    out = [r for g, r in sorted(latest.items()) if g not in have and parse_iso(r["scheduled_start_utc"]) + timedelta(minutes=POSTGAME_DELAY_MINUTES) <= now]
    return out[:limit]
