"""
NFL_PHASE1E_SOURCES (Phase 1E, shadow) -- live source registry + real-provider fetch audit.

  python nfl_phase1e_sources.py registry                       -> nfl_models/nfl_player_outcome_phase1e/live_source_registry.json
  python nfl_phase1e_sources.py fetch --cas DIR --out FILE     -> contacts the REAL provider (no mirror, no cached file), stores in the CAS, parses, validates

Nothing here reads a local development copy of the data.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import nfl_phase1_data as P1
import nfl_phase1d_cas as CAS

REPO = Path(__file__).resolve().parent
OUT = REPO / "nfl_models" / "nfl_player_outcome_phase1e"
UTC = timezone.utc
REG_VERSION = "live-source-registry-1"

# expected columns (the frozen loaders read these); a missing column = schema failure
SCHEMA = {
    "games.csv": ["game_id", "season", "game_type", "week", "gameday", "gametime", "away_team", "home_team", "result"],
    "players.csv": ["gsis_id"],
    "injuries_{s}.csv": ["season", "week", "team", "gsis_id", "report_status"],
    "roster_weekly_{s}.csv": ["season", "week", "team", "gsis_id", "position", "status"],
    "stats_player_week_{s}.csv": ["player_id", "season", "week", "team"],
    "snap_counts_{s}.csv": ["game_id", "season", "week", "team"],
    "pbp_{s}.csv.gz": ["game_id", "play_id", "posteam", "defteam", "play_type"],
    "participation_{s}.csv": ["nflverse_game_id", "play_id"],
    "ftn_{s}.csv": ["nflverse_game_id"],
    "depth_charts_{s}.csv": ["dt"],
}
CADENCE = {   # provider update cadence is OBSERVED (Last-Modified of the live objects, see real_provider_fetch_audit.json), not assumed
    "games.csv": "continuously updated by the provider (kickoff revisions, results)",
    "players.csv": "provider-updated roughly daily",
}


def tpl_of(name):
    for t in list(CAS.STATIC_FILES) + list(CAS.SEASON_FILES):
        if "{s}" not in t:
            if name == t:
                return t
        else:
            a, b = t.split("{s}")
            if name.startswith(a) and name.endswith(b) and name[len(a):len(name) - len(b)].isdigit():
                return t
    raise KeyError(name)


def required_at(name):
    """(T24, T90, failure policy) for a logical source. 'optional' sources are recorded as unavailable, never imputed."""
    if name.startswith(CAS.OPTIONAL_LIVE):
        return False, False, "OPTIONAL: absence recorded in the snapshot set (unavailable_optional_sources); downstream treats the data as absent; never substituted"
    return True, True, "REQUIRED: retrieval failure or schema failure -> no snapshot set for the game-horizon -> SAFE_EXPLICIT_FAILURE; no substitution of another provider or a cached copy"


def build_registry():
    srcs = []
    for name in CAS.logical_files():
        t1, t2, pol = required_at(name)
        tpl = tpl_of(name)
        srcs.append({"logical_name": name, "provider": "nflverse (GitHub release assets, nflverse/nflverse-data)", "endpoint": CAS.provider_url(name), "mechanism": "HTTPS GET, follow redirects, whole file",
                     "provider_id": CAS.provider_id(name), "expected_schema_required_columns": SCHEMA[tpl], "parser_version": CAS.PARSER_VERSION, "format": "csv.gz" if name.endswith(".gz") else "csv",
                     "cadence": CADENCE.get(name, "provider refreshes the live season's file on its own schedule; observed Last-Modified recorded in real_provider_fetch_audit.json"),
                     "availability_lag": "unbounded by contract: the provider's bytes at retrieval time are used, their age (retrieval time - Last-Modified) is recorded; completed games enter the files after the game",
                     "required_at_T24": t1, "required_at_T90": t2, "failure_policy": pol,
                     "stored_transform": "sportsbook / unused columns removed before storage (raw sha256 recorded)" if name == "games.csv" else "none (raw bytes stored)"})
    return {"version": REG_VERSION, "n_sources": len(srcs), "seasons": list(P1.SEASONS), "no_silent_substitution": True,
            "provider_lag_policy": {
                "rule": "At a cutoff use the latest bytes the provider serves BEFORE the cutoff and record their age. A later revision is never labelled T24/T90 (retrieval_ts <= cutoff is enforced by take_snapshot_set).",
                "operational_validity_limits": ["games.csv must contain the target game with a kickoff (else SAFE_EXPLICIT_FAILURE)",
                                                "every required source retrieved with a valid schema",
                                                "every game of the live season whose kickoff + 24h <= cutoff must have a stats_player row set (completed-week data present) -- otherwise the provider lags and the forecast is SAFE_EXPLICIT_FAILURE('provider_lag')"],
                "no_invented_data": True},
            "sources": srcs}


def check_schema(name, raw):
    tpl = tpl_of(name)
    body = gzip.decompress(raw) if name.endswith(".gz") else raw
    head = body[:2_000_000].decode("utf-8", "replace")
    header = next(csv.reader(io.StringIO(head)))
    missing = [c for c in SCHEMA[tpl] if c not in header]
    return {"n_columns": len(header), "missing_required_columns": missing}, body


def summarize(name, body):
    """latest season / week / date represented in the payload (parsed with the standard csv reader)."""
    text = body.decode("utf-8", "replace")
    rdr = csv.DictReader(io.StringIO(text, newline=""))
    n, mx_s, mx_w, mx_d, kv = 0, None, None, None, {}
    for r in rdr:
        n += 1
        try:
            s = int(r["season"]) if r.get("season", "").isdigit() else None
        except Exception:
            s = None
        if s is not None:
            w = int(float(r["week"])) if r.get("week") not in (None, "") and re.match(r"^\d+(\.0)?$", r["week"]) else None
            if mx_s is None or (s, w or 0) > (mx_s, mx_w or 0):
                mx_s, mx_w = s, w
        for c in ("gameday", "game_date", "dt", "date_modified", "report_date"):
            v = r.get(c)
            if v and (mx_d is None or v > mx_d):
                mx_d = v
    return {"n_rows": n, "latest_season": mx_s, "latest_week_of_latest_season": mx_w, "latest_date_field": mx_d}


def fetch_all(cas_root, seasons=None, log=print):
    store, ledger = CAS.BlobStore(cas_root), CAS.Ledger(cas_root)
    now0 = datetime.now(UTC)
    results = []
    sched_raw = None
    for name in CAS.logical_files(seasons or P1.SEASONS):
        url = CAS.provider_url(name)
        opt = name.startswith(CAS.OPTIONAL_LIVE)
        t0 = time.time()
        rec = {"logical_name": name, "url": url, "optional": opt}
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nfl-phase1-snapshots"})
            with urllib.request.urlopen(req, timeout=300) as r:
                raw = r.read()
                rec["http_status"] = r.status
                rec["final_url"] = r.geturl()
                rec["last_modified"] = r.headers.get("Last-Modified")
                rec["etag"] = r.headers.get("ETag")
            rec["retrieved_at_utc"] = datetime.now(UTC).isoformat()
            rec["seconds"] = round(time.time() - t0, 1)
            rec["raw_bytes"] = len(raw)
            rec["raw_sha256"] = hashlib.sha256(raw).hexdigest()
            stored = raw
            if name == "games.csv":
                stored, _, _ = CAS.sanitize_schedule(raw)
            info = store.put(stored)
            rec["stored_sha256"], rec["stored_bytes"], rec["cas_blob"] = info["sha256"], info["bytes"], info["blob"]
            rec["cas_roundtrip_hash_verified"] = hashlib.sha256(store.get(info["sha256"])).hexdigest() == info["sha256"]
            sch, body = check_schema(name, store.get(info["sha256"]))
            rec["schema"] = sch
            rec["schema_ok"] = not sch["missing_required_columns"]
            rec["content"] = summarize(name, body)
            if rec.get("last_modified"):
                lm = datetime.strptime(rec["last_modified"], "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=UTC)
                rec["age_hours_at_retrieval"] = round((datetime.fromisoformat(rec["retrieved_at_utc"]) - lm).total_seconds() / 3600, 2)
            rec["status"] = "OK" if rec["schema_ok"] else "SCHEMA_FAIL"
        except urllib.error.HTTPError as e:
            rec["status"] = "UNAVAILABLE_OPTIONAL" if opt and e.code == 404 else "HTTP_ERROR"
            rec["http_status"] = e.code
        except Exception as e:
            rec["status"] = "ERROR"
            rec["error"] = f"{type(e).__name__}: {e}"
        results.append(rec)
        log(f"{name}: {rec['status']} {rec.get('raw_bytes', '')}")
    required_ok = all(r["status"] == "OK" for r in results if not r["optional"])
    return {"now_utc": now0.isoformat(), "uses_local_mirror": False, "uses_cached_dev_file": False, "cas_root": "fresh store created for this audit",
            "required_sources_all_ok": required_ok, "n_sources": len(results), "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["registry", "fetch"])
    ap.add_argument("--cas")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "registry":
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "live_source_registry.json").write_text(json.dumps(build_registry(), indent=1))
    else:
        rep = fetch_all(a.cas)
        Path(a.out).write_text(json.dumps(rep, indent=1))
        print("required_sources_all_ok", rep["required_sources_all_ok"])


if __name__ == "__main__":
    main()
