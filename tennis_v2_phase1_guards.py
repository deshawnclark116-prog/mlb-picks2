"""Tennis V2 Phase1A — causal/identity/surface guards.

Research only. No file/network access and no production tennis imports.
The Phase1 protocol is the authority; this module cannot replace audited
historical timestamps. Tournament dates in TML are *start dates*.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Literal


class EvidenceBlocked(ValueError):
    """A fail-closed data or timing condition."""


def _day(value: date | str) -> date:
    if type(value) is date:
        return value
    if not isinstance(value, str) or len(value) != 10:
        raise EvidenceBlocked("BLOCKED_TIMING: expected YYYY-MM-DD tournament start")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise EvidenceBlocked("BLOCKED_TIMING: invalid date") from exc
    if parsed.isoformat() != value:
        raise EvidenceBlocked("BLOCKED_TIMING: non-canonical date")
    return parsed


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise EvidenceBlocked("BLOCKED_TIMING: UTC-aware timestamp required")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class HistoricalResult:
    tour: Literal["atp", "wta"]
    event_key: str  # season-specific event ID, never a normalized name
    match_key: str
    tournament_start: date | str  # TML tourney_date, NOT match played_at
    decided: bool
    completed: bool
    winner_id: str
    loser_id: str


def historical_asof(
    results: Iterable[HistoricalResult],
    *,
    target_tour: Literal["atp", "wta"],
    target_event_key: str,
    target_tournament_start: date | str,
    purpose: Literal["rating", "performance"],
    embargo_days: int = 28,
) -> tuple[HistoricalResult, ...]:
    """History known conservatively before a *tournament*, not before a round.

    Do not relax embargo based on a row's round or lexical match number.
    Any current-tournament match is excluded; returns sorted, stable results.
    """
    if target_tour not in ("atp", "wta") or not target_event_key:
        raise EvidenceBlocked("BLOCKED_DATA: invalid tour/target event")
    if purpose not in ("rating", "performance") or embargo_days != 28:
        raise EvidenceBlocked("BLOCKED_DATA: protocol-locked purpose/embargo")
    cutoff = _day(target_tournament_start) - timedelta(days=embargo_days)
    eligible: list[tuple[date, HistoricalResult]] = []
    seen: set[tuple[str, str]] = set()
    for r in results:
        if r.tour not in ("atp", "wta") or not r.event_key or not r.match_key:
            raise EvidenceBlocked("BLOCKED_DATA: malformed historical result")
        d = _day(r.tournament_start)
        unique_key = (r.tour, r.match_key)
        if unique_key in seen:
            raise EvidenceBlocked("BLOCKED_DATA: duplicate match identity")
        seen.add(unique_key)
        if r.completed and not r.decided:
            raise EvidenceBlocked("BLOCKED_DATA: completed but no decided winner")
        if r.decided and (not r.winner_id or not r.loser_id or r.winner_id == r.loser_id):
            raise EvidenceBlocked("BLOCKED_DATA: undecidable player identity")
        if r.tour != target_tour or r.event_key == target_event_key or d > cutoff:
            continue
        if not r.decided:  # W/O, unplayed, or unresolved result
            continue
        if purpose == "performance" and not r.completed:
            continue  # retirement may inform rating, never serve/game means
        eligible.append((d, r))
    eligible.sort(key=lambda item: (item[0], item[1].tour, item[1].event_key, item[1].match_key))
    return tuple(row for _, row in eligible)


@dataclass(frozen=True)
class FinalizedResult:
    tour: Literal["atp", "wta"]
    event_key: str
    match_key: str
    finalized_at_utc: datetime
    source_retrieved_at_utc: datetime


def forward_asof(
    results: Iterable[FinalizedResult],
    *,
    target_tour: Literal["atp", "wta"],
    target_event_key: str,
    scheduled_start_utc: datetime,
    forecast_created_at_utc: datetime,
) -> tuple[FinalizedResult, ...]:
    """Only final results observed before a genuinely pre-match decision."""
    start = _utc(scheduled_start_utc)
    decision = _utc(forecast_created_at_utc)
    if decision >= start:
        raise EvidenceBlocked("BLOCKED_TIMING: decision is not before scheduled start")
    if target_tour not in ("atp", "wta") or not target_event_key:
        raise EvidenceBlocked("BLOCKED_DATA: invalid target")
    eligible: list[tuple[datetime, FinalizedResult]] = []
    seen: set[tuple[str, str]] = set()
    for r in results:
        if r.tour not in ("atp", "wta") or not r.event_key or not r.match_key:
            raise EvidenceBlocked("BLOCKED_DATA: malformed finalized result")
        unique = (r.tour, r.match_key)
        if unique in seen:
            raise EvidenceBlocked("BLOCKED_DATA: duplicate match identity")
        seen.add(unique)
        final = _utc(r.finalized_at_utc)
        retrieved = _utc(r.source_retrieved_at_utc)
        if final > retrieved:
            raise EvidenceBlocked("BLOCKED_TIMING: result retrieved before finalization")
        if r.tour == target_tour and r.event_key != target_event_key and retrieved <= decision:
            eligible.append((retrieved, r))
    eligible.sort(key=lambda item: (item[0], item[1].tour, item[1].event_key, item[1].match_key))
    return tuple(row for _, row in eligible)


@dataclass(frozen=True)
class IdentityEvidence:
    tour: Literal["atp", "wta"]
    provider_id: str
    tml_player_id: str
    match_class: str
    independent_crosswalk_verified: bool
    candidates: int
    verified_at_utc: datetime


@dataclass(frozen=True)
class SurfaceEvidence:
    tour: Literal["atp", "wta"]
    event_key: str
    season: int
    surface: str
    match_class: str
    provenance_date_utc: datetime
    evidence_season: int  # historical inference must predate current season


@dataclass(frozen=True)
class FormatEvidence:
    best_of: int
    final_set_rule: str


@dataclass(frozen=True)
class Eligibility:
    allowed: bool
    blocked_reasons: tuple[str, ...]


def pre_match_gate(
    *,
    tour: Literal["atp", "wta"],
    event_key: str,
    season: int,
    decision_at_utc: datetime,
    player1: IdentityEvidence,
    player2: IdentityEvidence,
    surface: SurfaceEvidence,
    fmt: FormatEvidence,
) -> Eligibility:
    """Report all blockers; never silently make a hard-court or ID guess."""
    decision = _utc(decision_at_utc)
    problems: list[str] = []
    if tour not in ("atp", "wta") or not event_key or season < 1990:
        problems.append("BLOCKED_DATA")
    for p in (player1, player2):
        if (p.tour != tour or p.match_class not in ("EXACT", "NORMALIZED_EXACT")
                or not p.provider_id or not p.tml_player_id
                or not p.independent_crosswalk_verified or p.candidates != 1):
            problems.append("BLOCKED_IDENTITY")
        if _utc(p.verified_at_utc) > decision:
            problems.append("BLOCKED_TIMING")
    if player1.provider_id == player2.provider_id or player1.tml_player_id == player2.tml_player_id:
        problems.append("BLOCKED_IDENTITY")
    if (surface.tour != tour or surface.event_key != event_key or surface.season != season
            or surface.surface not in ("Hard", "Clay", "Grass", "Carpet")
            or surface.match_class not in ("KNOWN_EXACT", "HISTORICAL_TOURNEY_INFERENCE")):
        problems.append("BLOCKED_SURFACE")
    if surface.match_class == "HISTORICAL_TOURNEY_INFERENCE" and surface.evidence_season >= season:
        problems.append("BLOCKED_SURFACE")
    if surface.evidence_season > season:
        problems.append("BLOCKED_SURFACE")
    if _utc(surface.provenance_date_utc) > decision:
        problems.append("BLOCKED_TIMING")
    if fmt.best_of not in (3, 5) or fmt.final_set_rule not in (
        "TB7_AT_6_ALL_SETS", "TB10_AT_6_FINAL",
        "ADVANTAGE_FINAL_SET", "TB7_AT_12_FINAL",
    ):
        problems.append("BLOCKED_FORMAT")
    return Eligibility(not problems, tuple(sorted(set(problems))))
