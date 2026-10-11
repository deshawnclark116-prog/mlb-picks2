"""Serving-only safety boundary for the independent game-day NFL model.

Frozen T24/T90 batches and historical ledgers are outside this module.
A pick without an authenticated game kickoff cannot become a pregame entry.
"""
from datetime import datetime, timezone


def as_utc(value):
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("invalid kickoff timestamp")
    if dt.tzinfo is None:
        raise ValueError("kickoff timestamp has no timezone")
    return dt.astimezone(timezone.utc)


def schedule_manifest(rows):
    """Preserve *all* matchups, even with zero eligible player projections."""
    result, identities = [], set()
    for home, away, kickoff in rows:
        if not home or not away or home == away or not kickoff:
            raise ValueError("incomplete NFL schedule row")
        iso = as_utc(kickoff).isoformat().replace("+00:00", "Z")
        identity = frozenset((home, away))
        if identity in identities:
            raise ValueError("duplicate scheduled matchup")
        identities.add(identity)
        result.append({"home_team": home, "away_team": away, "kickoff_utc": iso})
    return sorted(result, key=lambda r: (r["kickoff_utc"], r["home_team"], r["away_team"]))


def pregame_only(picks, kickoff_by_pair, clock):
    """Drop started/unknown-kickoff rows *before* append-only grading.

    No fake replacement prediction is generated. The failure counts are returned
    so the publisher reports why rows are missing without claiming coverage.
    """
    now = as_utc(clock)
    accepted, reasons = [], {"kickoff_passed": 0, "unresolved_schedule": 0, "invalid_kickoff": 0}
    for original in picks:
        row = dict(original)
        pair = frozenset((row.get("team"), row.get("opponent")))
        kickoff = kickoff_by_pair.get(pair)
        if kickoff is None:
            reasons["unresolved_schedule"] += 1
            continue
        try:
            future = as_utc(kickoff) > now
        except (TypeError, ValueError):
            reasons["invalid_kickoff"] += 1
            continue
        if not future:
            reasons["kickoff_passed"] += 1
            continue
        row["kickoff_utc"] = as_utc(kickoff).isoformat().replace("+00:00", "Z")
        accepted.append(row)
    return accepted, reasons
