#!/usr/bin/env python3
"""NHL performance-FIRST shot-count projection contract.

Read-only adapter for the *separate, NOT PROMOTED* V2 immutable NB2
FORECAST ledger. Correctly derives P(SOG >= K) from (mu, alpha), not the
known historically shifted P1..P5 fields. Never manufactures book prices,
lines, roster confirmations, source times, or pregame forecast receipts.

Research projections are valuable independently of any sportsbook offer.
No betting pick is produced without future separate authorization, source
qualification, proper model validation, and real book/lineup evidence.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

SCHEMA = "NHL_V2_COUNT_PROJECTION_RESEARCH_ONLY_V1"
HORIZONS = frozenset({"T24H", "T90", "T30"})
PRIORITY = {"T24H": 0, "T90": 1, "T30": 2}


class ProjectionIntegrityError(ValueError):
    pass


def utc(value):
    if not isinstance(value, str):
        raise ProjectionIntegrityError("MISSING_UTC_TIME")
    try:
        d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProjectionIntegrityError("INVALID_UTC_TIME") from exc
    if d.tzinfo is None:
        raise ProjectionIntegrityError("NAIVE_TIME_NOT_ALLOWED")
    return d.astimezone(timezone.utc)


def finite_nonnegative(v, field, *, positive=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ProjectionIntegrityError("INVALID_" + field.upper())
    val = float(v)
    if not math.isfinite(val) or val < 0 or (positive and val == 0):
        raise ProjectionIntegrityError("INVALID_" + field.upper())
    return val


def _nb2_pmf(mu, alpha, max_k=120):
    mu = finite_nonnegative(mu, "mu")
    alpha = finite_nonnegative(alpha, "alpha", positive=True)
    if not 0 < alpha <= 20 or mu > 50:
        raise ProjectionIntegrityError("UNSUPPORTED_COUNT_DISTRIBUTION_RANGE")
    r = 1.0 / alpha
    p = r / (r + mu)
    p0 = p ** r
    if not math.isfinite(p0):
        raise ProjectionIntegrityError("NONFINITE_PMF")
    pmf = [p0]
    factor = mu / (r + mu)
    for k in range(max_k):
        pmf.append(pmf[-1] * (k + r) / (k + 1) * factor)
    mass = sum(pmf)
    if mass < 0.999999:  # refuse silently truncating a heavy-tail distribution
        raise ProjectionIntegrityError("NB2_TAIL_MASS_NOT_CAPTURED")
    # Normalize tiny float error only; no ad-hoc calibration or tuning.
    return [x / mass for x in pmf]


def _quantile(pmf, quantile):
    running = 0.0
    for i, p in enumerate(pmf):
        running += p
        if running + 1e-12 >= quantile:
            return i
    raise ProjectionIntegrityError("COUNT_QUANTILE_OUTSIDE_SUPPORT")


def count_projection(expected_sog, dispersion):
    """A genuinely line-independent distribution of integer NHL shot counts."""
    mu = finite_nonnegative(expected_sog, "expected_sog")
    pmf = _nb2_pmf(mu, dispersion)
    cdf = [0.0]
    for p in pmf:
        cdf.append(cdf[-1] + p)
    return {
        "expected_sog": mu, "median_sog": _quantile(pmf, .5),
        "p10_sog": _quantile(pmf, .10), "p90_sog": _quantile(pmf, .90),
        "P1": round(max(0., 1-cdf[1]), 8),
        "P2": round(max(0., 1-cdf[2]), 8),
        "P3": round(max(0., 1-cdf[3]), 8),
        "P4": round(max(0., 1-cdf[4]), 8),
        "P5": round(max(0., 1-cdf[5]), 8),
        "probability_definition": "P(SOG >= K); NB2 CDF evaluated at K-1",
        "distribution_family": "negative_binomial_NB2",
    }


def probability_over_half_line(expected_sog, dispersion, line):
    """For e.g. 1.5/2.5/3.5, OVER means SOG >= floor(line)+1."""
    line = finite_nonnegative(line, "sportsbook_main_line")
    if line > 12.5 or abs((line % 1) - 0.5) > 1e-9:
        raise ProjectionIntegrityError("NOT_HALF_SHOT_MAIN_LINE")
    n = math.floor(line)
    pmf = _nb2_pmf(expected_sog, dispersion)
    over = max(0.0, min(1.0, 1.0 - sum(pmf[:n+1])))
    return {"line": line, "over_probability": round(over, 8),
            "under_probability": round(1-over, 8)}


def _valid_hex_hash(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def research_board(immutable_forecasts, *, as_of_utc):
    """Select one latest real-precutoff model forecast per player/game.

    If the frozen ledger names the same player on both teams for one game,
    quarantine the entire player/game. Preserves source row bytes; does not
    attempt a guessed player-team resolution or invert shifted legacy Pk.
    """
    as_of = utc(as_of_utc)
    groups = defaultdict(list)
    excluded = Counter()
    for original in immutable_forecasts:
        if not isinstance(original, dict) or original.get("record_type") != "FORECAST":
            excluded["NOT_FORECAST"] += 1
            continue
        pid, game = original.get("player_id"), original.get("game_id")
        if (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or
            isinstance(game, bool) or not isinstance(game, int) or game <= 0):
            excluded["INVALID_PLAYER_OR_GAME_ID"] += 1
            continue
        start, issued, cutoff = (utc(original.get(k)) for k in
                                 ("scheduled_start", "generated_at", "cutoff_at"))
        if not issued <= cutoff < start or issued > as_of:
            excluded["INVALID_OR_POST_CUTOFF_RECEIPT"] += 1
            continue
        if as_of >= start:
            excluded["GAME_STARTED_NOT_A_PREMATCH_PROJECTION"] += 1
            continue
        if original.get("forecast_horizon") not in HORIZONS:
            excluded["UNKNOWN_MODEL_HORIZON"] += 1
            continue
        if not _valid_hex_hash(original.get("source_manifest_sha256")):
            excluded["MISSING_SOURCE_VINTAGE_HASH"] += 1
            continue
        groups[(game, pid)].append(original)
    projections = []
    for (game, pid), rows in sorted(groups.items()):
        teams = {r.get("team") for r in rows}
        starts = {r["scheduled_start"] for r in rows}
        if len(teams) != 1 or None in teams or len(starts) != 1:
            excluded["CROSS_TEAM_OR_SCHEDULE_CONFLICT_QUARANTINED"] += len(rows)
            continue
        # Different T24/T90/T30 estimates are independent frozen horizon
        # forecasts; choose the latest eligible horizon to show a single
        # display row, never collapse or average their probabilities.
        keys = [(utc(r["generated_at"]), PRIORITY[r["forecast_horizon"]]) for r in rows]
        latest = max(keys)
        winners = [r for r, k in zip(rows, keys) if k == latest]
        if len(winners) != 1:
            excluded["DUPLICATE_FORECAST_SAME_HORIZON_AND_TIME"] += len(rows)
            continue
        r = winners[0]
        distribution = count_projection(r["expected_sog"], r["dispersion"])
        projections.append({
            "game_id": game, "player_id": pid,
            "player": r.get("receipt", {}).get("player_name"),
            "team": r["team"], "opponent": r.get("opponent"),
            "scheduled_start": r["scheduled_start"],
            "forecast_horizon": r["forecast_horizon"],
            "original_forecast_generated_at": r["generated_at"],
            "source_manifest_sha256": r["source_manifest_sha256"],
            "candidate_status": r.get("candidate_status"),
            "availability_state": r.get("availability_state"),
            "availability_confirmed_for_game": False,
            "model_promoted": False,
            "research_only": True, "book_main_line": None,
            "betting_pick": None,
            "original_legacy_P1_through_P5_used": False,
            **distribution,
        })
        excluded["SECONDARY_HORIZON_NOT_SELECTED_FOR_DISPLAY"] += len(rows)-1
    return {
        "schema": SCHEMA,
        "as_of_utc": as_of.isoformat().replace("+00:00", "Z"),
        "status": "RESEARCH_POINT_PROJECTIONS_ONLY_NOT_OFFICIAL_PICKS",
        "projections": projections,
        "projection_count": len(projections),
        "official_shots_on_goal_picks": 0,
        "excluded": dict(sorted(excluded.items())),
        "historical_immutable_ledger_untouched": True,
        "actual_sportsbook_main_lines_available": False,
        "lineup_and_identity_certified": False,
        "nb2_historical_Pk_offset_corrected_posthoc": True,
    }
