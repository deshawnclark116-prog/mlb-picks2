"""Shared, read-only The Odds API quota lock across MLB/NFL/Tennis/NHL.

No API calls, secret handling, or inference that a green workflow means
usable credits. The authenticated NHL research diagnostic recorded a real
provider OUT_OF_USAGE_CREDITS response. All sports using the SAME key
must honor it. This is an immediate, repository-backed protection; it is
not a replacement for a central atomic provider-budget service.

Lock automatically expires only after the next UTC calendar month
(allowing a single future refreshed probe; the provider billing reset
may occur on a different date). Manual researched credential refresh can
update/remove the provider status, never silently bypass this guard.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_STATUS_FILE = (Path(__file__).resolve().parent /
                       "docs/nhl_v2_fanduel_source_status.json")


def the_odds_api_quota_locked(*, now=None, status_file=None):
    """Return True only for a verifiable shared provider quota exhaustion.

    Fail open if evidence is missing/corrupt: this guard is not a source
    of invented provider-status claims, and missing telemetry is an
    operational defect that should be reported independently.
    """
    path = Path(status_file) if status_file is not None else DEFAULT_STATUS_FILE
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("provider") != "the_odds_api":
            return False
        if report.get("status") != "PROVIDER_OUT_OF_USAGE_CREDITS_DO_NOT_AUTOMATICALLY_RETRY":
            return False
        if "OUT_OF_USAGE_CREDITS" not in str(report.get("provider_access_blocked", "")):
            return False
        recorded = datetime.fromisoformat(
            report["captured_at_utc"].replace("Z", "+00:00"))
        if recorded.tzinfo is None:
            return False
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("Quota guard requires timezone-aware clock")
    recorded = recorded.astimezone(timezone.utc)
    current = current.astimezone(timezone.utc)
    if current < recorded:
        # Do not let a future-dated status poison present-day production.
        return False
    return (current.year, current.month) == (recorded.year, recorded.month)


def source_status(*, now=None, status_file=None):
    return (
        "QUOTA_EXHAUSTED_SHARED_PROVIDER_NO_NEW_REQUESTS"
        if the_odds_api_quota_locked(now=now,status_file=status_file)
        else "NO_ACTIVE_SHARED_QUOTA_LOCK"
    )
