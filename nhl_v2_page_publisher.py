#!/usr/bin/env python3
"""NHL V2 read-only player-stat projection publisher.

This is the NHL counterpart of nfl_phase1_publisher.py. It NEVER issues a bet,
never runs the engine, and never writes to the frozen research branch.
Reads the complete B2 hash-chain, verifies manifest/blobs, rejects late rows,
quarantines every duplicate game/horizon/player identity, and publishes
original point estimates with *separately* recomputed NB2 threshold probabilities.

The old frozen P1..P5 fields are shifted and MUST NOT be displayed.
The independent interpretation is posthoc, not a new pregame prediction.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = "nhl-v2-verified-projections-view-v1"
HORIZONS = ("T24H", "T90", "T30")
CUTOFF_MINS = {"T24H": 1440, "T90": 90, "T30": 30}
GENESIS = "0" * 64
WINDOW_SECS = 900
ROOT_REL = Path("nhl_models/nhl_player_outcome_v2/phase1a_sog_forward")
MAX_SHOTS = 400


class NHLPublicationError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(b):
    return hashlib.sha256(b).hexdigest()


def utc(value):
    if not isinstance(value, str):
        raise NHLPublicationError("MISSING_TIMESTAMP")
    try:
        d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise NHLPublicationError("INVALID_TIMESTAMP") from e
    if d.tzinfo is None:
        raise NHLPublicationError("NAIVE_TIMESTAMP")
    return d.astimezone(timezone.utc)


def number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise NHLPublicationError("INVALID_" + name.upper())
    x = float(value)
    if x < 0 or (positive and x == 0):
        raise NHLPublicationError("INVALID_" + name.upper())
    return x


def nb2(mu, alpha):
    """Independent NB2 recurrence; no unsafe historical Pk reuse."""
    mu, alpha = number(mu, "expected_sog"), number(alpha, "dispersion", True)
    r = 1 / alpha
    p = r / (r + mu)
    p0 = math.exp(r * math.log(p))
    pmf, cdf, tail, quantiles = p0, 0.0, {}, {}
    boundaries = {0.10: "p10", 0.25: "p25", 0.5: "median", 0.75: "p75", 0.90: "p90"}
    for k in range(MAX_SHOTS + 1):
        if k > 0:
            pmf *= (k - 1 + r) / k * (1 - p)
        cdf += pmf
        if k < 5:
            tail[f"ge{k + 1}"] = max(0.0, min(1.0, 1 - cdf))
        for level, name in boundaries.items():
            if name not in quantiles and cdf >= level - 1e-12:
                quantiles[name] = k
        if len(quantiles) == 5 and k >= 5:
            break
    if len(quantiles) != 5:
        raise NHLPublicationError("NB2_TAIL_DID_NOT_CONVERGE")
    if not all(tail[f"ge{k}"] + 1e-10 >= tail[f"ge{k + 1}"] for k in range(1, 5)):
        raise NHLPublicationError("NB2_TAIL_NOT_MONOTONE")
    return quantiles, {k: round(v, 6) for k, v in tail.items()}


def verify_chain(rows):
    prev = GENESIS
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("seq") != i or row.get("prev_hash") != prev:
            raise NHLPublicationError(f"FROZEN_LEDGER_CHAIN_INVALID:{i}")
        expected = digest(canonical({k: v for k, v in row.items() if k != "row_hash"}).encode())
        if row.get("row_hash") != expected:
            raise NHLPublicationError(f"FROZEN_LEDGER_HASH_INVALID:{i}")
        prev = expected
    return prev


def verify_provenance(row, root, manifest_cache, blob_cache):
    key = row.get("source_manifest_sha256")
    if not isinstance(key, str) or len(key) != 64 or any(x not in "0123456789abcdef" for x in key):
        raise NHLPublicationError("MISSING_SOURCE_MANIFEST_HASH")
    if key not in manifest_cache:
        mf = root / "manifests" / (key + ".json")
        if not mf.is_file():
            raise NHLPublicationError("MISSING_SOURCE_MANIFEST")
        data = json.loads(mf.read_text())
        expected = digest(canonical({k: v for k, v in data.items() if k != "manifest_sha256"}).encode())
        if data.get("manifest_sha256") != key or expected != key:
            raise NHLPublicationError("SOURCE_MANIFEST_HASH_INVALID")
        manifest_cache[key] = data
    manifest = manifest_cache[key]
    count = 0
    cutoff = utc(row["cutoff_at"])
    for source in manifest.get("sources", []):
        if not source.get("predictive"):
            continue
        count += 1
        if source.get("completeness_status") != "COMPLETE" or source.get("http_status") != 200:
            raise NHLPublicationError("INCOMPLETE_PREDICTIVE_SOURCE")
        if utc(source["retrieval_completed_utc"]) > cutoff:
            raise NHLPublicationError("LATE_PREDICTIVE_SOURCE")
        sha = source.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise NHLPublicationError("MISSING_PREDICTIVE_SOURCE_HASH")
        if sha not in blob_cache:
            blob = root / "blobs" / (sha + ".gz")
            if not blob.is_file() or digest(gzip.decompress(blob.read_bytes())) != sha:
                raise NHLPublicationError("UNVERIFIED_PREDICTIVE_SOURCE_BLOB")
            blob_cache.add(sha)
    if not count:
        raise NHLPublicationError("NO_PREDICTIVE_SOURCE")
    return True


def build(rows, root, source_ref):
    """Raises on global chain/source failure, quarantines local identity defects."""
    endhash = verify_chain(rows)
    manifest_cache, blob_cache = {}, set()
    valid, invalid_reasons = [], Counter()
    original_forecasts = [x for x in rows if x.get("record_type") == "FORECAST"]
    for row in original_forecasts:
        if row.get("forecast_horizon") not in HORIZONS:
            invalid_reasons["UNSUPPORTED_HORIZON"] += 1
            continue
        kickoff = utc(row["scheduled_start"])
        cutoff = utc(row["cutoff_at"])
        generated = utc(row["generated_at"])
        if cutoff != kickoff - timedelta(minutes=CUTOFF_MINS[row["forecast_horizon"]]):
            invalid_reasons["INCORRECT_CUTOFF"] += 1
            continue
        if not (cutoff - timedelta(seconds=WINDOW_SECS) <= generated <= cutoff):
            invalid_reasons["LATE_OR_UNTIMED_PREGAME_ROW"] += 1
            continue
        if row.get("schedule_state", "OK") != "OK":
            invalid_reasons["BAD_SCHEDULE_STATE"] += 1
            continue
        verify_provenance(row, root, manifest_cache, blob_cache)
        valid.append(row)

    # Collision quarantine is all-or-nothing (no silently picking a preferred
    # team or one of two different forecast batches).
    identity = defaultdict(list)
    ids = defaultdict(list)
    for row in valid:
        identity[(str(row["game_id"]), row["forecast_horizon"], str(row["player_id"]))].append(row)
        ids[str(row["forecast_id"])].append(row)
    bad_identity = {k for k, group in identity.items() if len(group) != 1}
    bad_ids = {k for k, group in ids.items() if len(group) != 1}
    results = {h: [] for h in HORIZONS}
    for row in valid:
        identity_key = (str(row["game_id"]), row["forecast_horizon"], str(row["player_id"]))
        if identity_key in bad_identity or str(row["forecast_id"]) in bad_ids:
            invalid_reasons["DUPLICATE_PLAYER_OR_FORECAST_ID"] += 1
            continue
        mu = number(row["expected_sog"], "expected_sog")
        alpha = number(row["dispersion"], "dispersion", True)
        q, probs = nb2(mu, alpha)
        receipt = row.get("receipt") or {}
        comparator = row.get("comparators") or {}
        # Original frozen median and original mean are kept EXACTLY as model
        # estimates, while new thresholds and percentiles are visibly posthoc.
        rec = {
            "id": row["forecast_id"], "game_id": str(row["game_id"]),
            "date": row.get("schedule_date"), "kickoff": row["scheduled_start"],
            "cutoff": row["cutoff_at"], "generated_at": row["generated_at"],
            "horizon": row["forecast_horizon"],
            "player_id": row["player_id"],
            "player": receipt.get("player_name") or ("Player #" + str(row["player_id"])),
            "team": row["team"], "opponent": row["opponent"],
            "position": receipt.get("position") or "–",
            "mean": round(mu, 3), "median": round(float(row["median_sog"]), 3),
            "variance": round(number(row["variance"], "variance"), 4),
            "sd": round(math.sqrt(float(row["variance"])), 3),
            "p10": q["p10"], "p25": q["p25"], "p75": q["p75"], "p90": q["p90"],
            "probabilities_posthoc": probs,
            "availability": row.get("availability_state") or "NOT_CAPTURED",
            "availability_verified": row.get("availability_confidence") == "CERTIFIED",
            "meaningful": row.get("meaningful_expected_participant") is True,
            "sog_last10": receipt.get("recent_sog_history_last10_appearances") or [],
            "prior10_simple_mean": comparator.get("simple_prior10_mean"),
            "human_baseline_mean": comparator.get("human_frozen_mean"),
            "opponent_sog_allowed_mean5": receipt.get("opp_sog_allowed_mean5"),
            "recent_toi_seconds": (receipt.get("toi_seconds") or {}).get("recent3"),
            "power_play_toi_seconds": receipt.get("pp_toi_recent3_seconds"),
            "back_to_back": receipt.get("back_to_back"),
            "row_hash": row["row_hash"],
            "source_manifest_sha256": row["source_manifest_sha256"],
            "model_version": row.get("engine_version"),
            "status": "UNVERIFIED_LINEUP_RESEARCH" if row.get("availability_confidence") != "CERTIFIED" else "RESEARCH",
        }
        results[row["forecast_horizon"]].append(rec)
    for h in HORIZONS:
        results[h].sort(key=lambda r: (r["kickoff"], r["game_id"], -r["mean"], r["player_id"]))
    dates = sorted({r["date"] or r["kickoff"][:10] for group in results.values() for r in group})
    default_date = dates[-1] if dates else None
    return {
        "schema": SCHEMA,
        "source_branch": source_ref,
        "source_ledger_rows": len(rows),
        "source_ledger_head_sha256": endhash,
        "original_frozen_forecasts": len(original_forecasts),
        "quarantined_records": sum(invalid_reasons.values()),
        "exclusion_reasons": dict(sorted(invalid_reasons.items())),
        "disclaimer": "NHL V2 B2 shadow research. Frozen original mean/median only; NB2 threshold probabilities and p10/p90 are posthoc corrected interpretations, NOT original pregame published probabilities. Availability is not certified. No sportsbook lines or bets.",
        "horizons": list(HORIZONS),
        "dates": dates,
        "default_date": default_date,
        "default_horizon": next((h for h in HORIZONS if any((r["date"] or r["kickoff"][:10]) == default_date for r in results[h])), None),
        "forecasts": results,
        "published_player_rows": sum(map(len, results.values())),
        "official_betting_picks": 0,
        "model_promoted": False,
        "original_ledger_modified": False,
    }


def publish(source, out, ref):
    root = Path(source) / ROOT_REL
    ledger = root / "ledger.jsonl"
    if not ledger.is_file():
        raise NHLPublicationError("FROZEN_SOURCE_LEDGER_NOT_FOUND")
    rows = [json.loads(s) for s in ledger.read_text(encoding="utf-8").splitlines() if s.strip()]
    data = build(rows, root, ref)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in data.items() if k not in ("forecasts", "dates")}, sort_keys=True))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path, help="read-only research branch checkout")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source-ref", default="codex/nhl-outcome-engine-v2")
    ap.add_argument("--html-template", type=Path,
                    help="Optional read-only NHL page template for an offline research preview")
    ap.add_argument("--standalone-preview", type=Path)
    args = ap.parse_args()
    data = publish(args.source, args.out, args.source_ref)
    if args.standalone_preview:
        if not args.html_template:
            raise NHLPublicationError("STANDALONE_PREVIEW_NEEDS_REAL_PAGE_TEMPLATE")
        template = args.html_template.read_text(encoding="utf-8")
        marker = '<script id="nhl-data-embedded" type="application/json"></script>'
        if template.count(marker) != 1:
            raise NHLPublicationError("PREVIEW_TEMPLATE_NO_EMBED_SLOT")
        # Prevent HTML/script injection from untrusted player-name strings:
        # JSON escapes every '<' but still parses to the original names.
        raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        safe = raw.replace("<", "\\u003c")
        html = template.replace(marker, '<script id="nhl-data-embedded" type="application/json">' + safe + "</script>")
        args.standalone_preview.parent.mkdir(parents=True, exist_ok=True)
        args.standalone_preview.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
