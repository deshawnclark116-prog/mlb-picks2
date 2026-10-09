"""Independent read-only delivery audit: durable NFL shadow receipt -> public player row.

No forecast generation, time overrides for live publishing, network calls, or writes to
historical state. Designed to be run AFTER each read-only publisher invocation.
"""
import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nfl_phase1_publisher as PUB

HORIZONS = ("T24", "T90")
GOOD = ("DONE", "PARTIAL_V2_MISSING")


def parse_utc(value):
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError("timezone required")
    return t.astimezone(timezone.utc)


def state_commit(root):
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def generated_times(state):
    """Read actual immutable batch headers, never infer forecast generation times."""
    result = {}
    for file in (Path(state) / "forecasts" / "batches").glob("*.jsonl"):
        verified = PUB.read_batch_verified(file)
        if verified is None:
            continue
        header, records = verified
        for r in records:
            if r.get("id") and header.get("generated_at"):
                result.setdefault(r["id"], header["generated_at"])
    return result


def audit(state, published, now=None, game_id=None, fail_recent_hours=6):
    root = Path(state)
    now = parse_utc(now) if isinstance(now, str) else (now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        raise ValueError("naive audit clock forbidden")
    now = now.astimezone(timezone.utc)
    doc = json.loads(Path(published).read_text())
    fields = doc.get("row_fields") or []
    if "id" not in fields or "game_id" not in fields:
        raise ValueError("PUBLICATION_SCHEMA_INVALID: row_fields need id and game_id")
    i_id, i_game = fields.index("id"), fields.index("game_id")
    public = {}
    commit_sha = state_commit(root)
    for hz in HORIZONS:
        for r in doc.get("forecasts", {}).get(hz, []):
            if len(r) != len(fields):
                raise ValueError("PUBLICATION_SCHEMA_INVALID: malformed forecast row")
            if not game_id or r[i_game] == game_id:
                public.setdefault((r[i_game], hz), []).append(r[i_id])

    last = PUB.ledger_last_state(root)
    week = (doc.get("status", {}).get("season"), doc.get("status", {}).get("week"))
    valid = (week[0] is not None and week[1] is not None)
    if valid:
        research = PUB.genuine_records(root, last, week)
    else:
        research = {h: {} for h in HORIZONS}
    generations = generated_times(root)
    items = []
    for r in sorted(last.values(), key=lambda x: (x["cutoff"], x["game_id"], x["horizon"])):
        if game_id and r["game_id"] != game_id:
            continue
        if valid and PUB.week_of(r["game_id"]) != week:
            continue
        cutoff = parse_utc(r["cutoff"])
        if cutoff > now:
            continue
        hz, gid = r["horizon"], r["game_id"]
        if hz not in HORIZONS:
            continue
        originals = research[hz].get(gid, [])
        expected = sorted(q["id"] for q in originals)
        published_ids = sorted(public.get((gid, hz), []))
        if r["state"] not in GOOD:
            verdict = "MISSED_CUTOFF" if r["state"] == "MISSED_REAL_CUTOFF" else (
                "FAILED" if r["state"] == "FAILED" else "DUE_BUT_NOT_COMPLETED")
        elif not expected:
            verdict = "DONE_WITHOUT_VERIFIED_RECORDS"
        elif len(published_ids) != len(set(published_ids)):
            verdict = "DUPLICATE_PUBLIC_IDS"
        elif expected != published_ids:
            verdict = "PUBLICATION_MISMATCH"
        else:
            verdict = "DELIVERED"

        retrieved = sorted({q.get("input_snapshots", {}).get("retrieval_ts") for q in originals
                            if q.get("input_snapshots", {}).get("retrieval_ts")})
        generated = sorted({generations[q["id"]] for q in originals if q["id"] in generations})
        item = {
            "game_id": gid, "horizon": hz, "cutoff_at": r["cutoff"], "kickoff_at": r["kickoff"],
            "ledger_state": r["state"], "ledger_recorded_at": r.get("at"),
            "source_retrieved_at": retrieved, "model_generated_at": generated,
            "latest_state_commit_sha": commit_sha, "snapshot_set_id": r.get("snapshot_set_id"),
            "expected_verified_rows": len(expected), "published_rows": len(published_ids),
            "matching_receipt_ids": expected if verdict == "DELIVERED" else [],
            "missing_public_ids": sorted(set(expected) - set(published_ids)),
            "extraneous_public_ids": sorted(set(published_ids) - set(expected)),
            "verdict": verdict, "manual_recovery": bool(r.get("manual_recovery")),
            "recent_gate": now - cutoff <= timedelta(hours=fail_recent_hours),
        }
        items.append(item)

    # A public week with no matching real state must never appear as a success.
    known = {(q["game_id"], q["horizon"]) for q in last.values()}
    unknown_public = sorted((gid, hz) for (gid, hz) in public if (gid, hz) not in known)
    has_failure = any(q["verdict"] != "DELIVERED" and q["recent_gate"] for q in items)
    if unknown_public:
        has_failure = True
    return {
        "schema": "nfl-shadow-delivery-audit-v1",
        "as_of_utc": now.isoformat().replace("+00:00", "Z"),
        "week": list(week), "state_head": commit_sha,
        "n_due": len(items), "n_delivered": sum(q["verdict"] == "DELIVERED" for q in items),
        "n_recent_failures": sum(q["verdict"] != "DELIVERED" and q["recent_gate"] for q in items),
        "unknown_public_games": [{"game_id": gid, "horizon": hz} for gid, hz in unknown_public],
        "status": "FAIL" if has_failure else "PASS" if items else "NOT_DUE",
        "receipts": items,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--published", required=True)
    ap.add_argument("--as-of", help="Test/replay only; production leaves this unset")
    ap.add_argument("--game-id")
    ap.add_argument("--fail-recent-hours", type=float, default=6)
    ap.add_argument("--out", help="Write a new report only; never edit state or published JSON")
    opts = ap.parse_args()
    result = audit(opts.state, opts.published, opts.as_of, opts.game_id, opts.fail_recent_hours)
    if opts.out:
        dest = Path(opts.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "n_due": result["n_due"],
                      "n_delivered": result["n_delivered"],
                      "n_recent_failures": result["n_recent_failures"],
                      "unknown_public_games": result["unknown_public_games"]}, sort_keys=True))
    raise SystemExit(1 if result["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
