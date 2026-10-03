"""
NHL_FWD_OPS -- OPERATIONAL liveness / provenance helpers for the forward collector. Infrastructure only: nothing here is a scientific feature, a horizon, a cutoff, a denominator or an evidence-qualification input,
and nothing here can create, move or relabel evidence. stdlib only.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc
STALE_AFTER_MINUTES = 35          # collector_health.healthy turns False when no invocation has been seen for this long
WATCHDOG_FRESH_MINUTES = 16       # the watchdog does nothing when the collector invoked within this many minutes (primary requests a 15-minute cadence)
OPS_LEDGER = "ops_invocations.jsonl"


def iso(t):
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


def cron_minutes(cron):
    """Minute set of an hourly-pattern cron ('7,22,37,52 * * * *' or '*/15 * * * *'); None when it is not an every-hour pattern."""
    f = (cron or "").split()
    if len(f) != 5 or f[1:] != ["*", "*", "*", "*"]:
        return None
    m = f[0]
    if m.startswith("*/"):
        return list(range(0, 60, int(m[2:])))
    return sorted({int(x) for x in m.split(",")})


def intended_slot(cron, now):
    """Latest cron slot <= now (the GitHub-requested time this invocation answers). GitHub does not expose the scheduled timestamp, so it is derived from the cron expression; None when not derivable."""
    mins = cron_minutes(cron)
    if not mins:
        return None
    base = now.astimezone(UTC).replace(second=0, microsecond=0)
    for back in range(0, 125):
        t = base - timedelta(minutes=back)
        if t.minute in mins:
            return t
    return None


def provenance(env, now):
    """Operational provenance of THIS invocation, taken from workflow-supplied env. Not a scientific feature."""
    cron = env.get("NHL_CRON") or None
    slot = intended_slot(cron, now) if (env.get("NHL_EVENT_NAME") == "schedule" and cron) else None
    return {"workflow_event_name": env.get("NHL_EVENT_NAME") or "local", "run_id": env.get("NHL_RUN_ID") or None, "invoker": env.get("NHL_INVOKER") or "primary",
            "scheduled_cron": cron, "intended_slot_utc": iso(slot) if slot else None, "delay_from_intended_slot_min": round((now - slot).total_seconds() / 60, 2) if slot else None,
            "scheduled_event_timestamp_available": False}


def collector_health(last_invocation_iso, last_event_type, now, previous_invocation_iso=None, stale_after_minutes=STALE_AFTER_MINUTES):
    """Pure function of timestamps. `healthy` = the collector invoked within the stale threshold. `previous_gap_*` expose a historical outage without rewriting any evidence."""
    gap = (now - parse(last_invocation_iso)).total_seconds() / 60 if last_invocation_iso else None
    prev_gap = ((parse(last_invocation_iso) - parse(previous_invocation_iso)).total_seconds() / 60) if (last_invocation_iso and previous_invocation_iso) else None
    return {"last_invocation_utc": last_invocation_iso, "last_event_type": last_event_type, "stale_after_minutes": stale_after_minutes, "healthy": gap is not None and gap <= stale_after_minutes,
            "minutes_since_last_invocation": round(gap, 2) if gap is not None else None, "gap_before_this_invocation_min": round(prev_gap, 2) if prev_gap is not None else None,
            "gap_before_this_invocation_exceeded_stale_threshold": (prev_gap > stale_after_minutes) if prev_gap is not None else None}


def read_ops(root):
    p = Path(root) / OPS_LEDGER
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def record_invocation(root, prov, now):
    """Append-only operational ledger (one line per invocation). Never read by evidence evaluation."""
    p = Path(root) / OPS_LEDGER
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "ab") as fh:
        fh.write(json.dumps({"invoked_at": iso(now), **prov}, sort_keys=True, separators=(",", ":")).encode() + b"\n")
