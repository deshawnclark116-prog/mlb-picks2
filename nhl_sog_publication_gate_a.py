#!/usr/bin/env python3
"""NHL projection-first publication policy, v1.

The live legacy SOG classifiers answer P(SOG >= 3) for a fixed 2.5
threshold. They DO NOT produce per-skater expected shots, use actual book
main lines, or confirm a player is dressing. No model probability makes an
unverified line a bettable offer. Preserve original log for forward grading;
quarantine *both* fixed-line OVER and UNDER cards from PUBLIC picks.

This is NOT a policy of suppressing UNDERS as a direction. Future official
OVER or UNDER selections must pass the SAME independent count-projection,
actual book-line, as-of and lineup gates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json

SOG_CLASSIFIERS = frozenset({"shots_on_goal", "shots_on_goal_early_season"})
SCHEMA = "NHL_SOG_UNPRICED_CLASSIFIER_PUBLICATION_GATE_V1"
BLOCK = "UNVERIFIED_FIXED_2_5_CLASSIFIER_NOT_A_BETTABLE_PLAYER_PROJECTION"
READY = "COUNT_MODEL_AND_REAL_LINE_VERIFICATION_REQUIRED"


def _side(row: dict) -> str:
    pick = row.get("pick")
    if isinstance(pick, str) and pick.startswith("UNDER "):
        return "UNDER"
    if isinstance(pick, str) and pick.startswith("OVER "):
        return "OVER"
    return "UNKNOWN"


def quarantine_fixed_line_sog(picks: list[dict], *, season: int, game_date: str):
    """Return unaltered non-SOG picks + independent public-safe audit summary.

    No raw unverified legacy row is passed to an official-publication path.
    Duplicate/conflicting player-game models are counted, never chosen via
    confidence and never used to generate fabricated projection averages.
    """
    if not isinstance(picks, list):
        raise TypeError("picks must be a list")
    kept, quarantined = [], []
    market_sides = defaultdict(Counter)
    identities = defaultdict(list)
    for p in picks:
        if not isinstance(p, dict):
            raise ValueError("malformed pick row cannot be silently served")
        if p.get("market") not in SOG_CLASSIFIERS:
            kept.append(p)
            continue
        quarantined.append(p)
        market_sides[p["market"]][_side(p)] += 1
        pid = p.get("player_id")
        team, opponent = p.get("team"), p.get("opponent")
        if pid is None or not team or not opponent or team == opponent:
            # Unknown identity cannot accidentally be promoted or deduped.
            identity = ("UNVERIFIED", len(quarantined))
        else:
            identity = (str(pid), tuple(sorted((str(team), str(opponent)))))
        identities[identity].append(p)
    duplicate_groups = {k: v for k, v in identities.items() if len(v) > 1}
    conflicts = [k for k, rows in duplicate_groups.items()
                 if len({_side(row) for row in rows}) > 1]
    audit = {
        "schema": SCHEMA,
        "season": season, "game_date": game_date,
        "policy": BLOCK,
        "required_for_future_public_sog": READY,
        "removed_total": len(quarantined),
        "removed_by_market_and_side": {
            k: dict(v) for k, v in sorted(market_sides.items())
        },
        "distinct_candidate_player_games": len(identities),
        "duplicate_candidate_groups": len(duplicate_groups),
        "duplicate_candidate_rows": sum(len(v) for v in duplicate_groups.values()),
        "contradictory_player_game_groups": len(conflicts),
        "historical_original_pregame_log_preserved": True,
        "existing_classifier_model_unchanged": True,
        "book_main_lines_verified": False,
        "dressed_lineups_verified": False,
        "new_count_model_promoted": False,
        "official_shots_on_goal_picks": 0,
        "suppressed_overs_as_well_as_unders": True,
        "explanation": (
            "Legacy fixed 2.5-shot classifiers are research-only. "
            "A lower model P(OVER 2.5) is not a verified player's UNDER offer. "
            "Official NHL SOG picks stay disabled until a validated count "
            "distribution, verified actual FanDuel main line, confirmed "
            "player/game identity and lineup, and pregame source timestamps "
            "all pass separate gates. Both directions are treated equally."
        ),
    }
    if any(row.get("market") in SOG_CLASSIFIERS for row in kept):
        raise AssertionError("legacy SOG escape into official board")
    return kept, audit


def fail_closed_official_sog(picks: list[dict]):
    """Reserved hard publication assertion: no unverified SOG pick survives."""
    for p in picks:
        if p.get("market") in SOG_CLASSIFIERS:
            raise RuntimeError("LEGACY_NHL_SOG_UNVERIFIED_MAIN_LINE_PUBLICATION_BLOCKED")
    return True


def write_daily_snapshot_preserving_legacy(path, payload):
    """Preserve exact original date-snapshot bytes before changing policy.

    Historical pre-policy fixed-line cards may be wrong to *publish*, but are
    valid original prediction evidence and must not be erased by a fix.
    Write the original to a stable single-copy backup, fail on conflicting
    preexisting backups, then write the new production-policy snapshot.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "nhl_sog_publication_integrity" not in payload:
        raise RuntimeError("PUBLICATION_GATE_AUDIT_MISSING")
    fail_closed_official_sog(payload.get("picks", []))
    backup = path.with_name(path.stem + "_legacy_fixed_2_5_pre_policy.json")
    if path.exists():
        old = path.read_bytes()
        try:
            previous = json.loads(old)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("CORRUPT_HISTORICAL_NHL_SNAPSHOT_NO_OVERWRITE") from exc
        if "nhl_sog_publication_integrity" not in previous:
            if backup.exists():
                if backup.read_bytes() != old:
                    raise RuntimeError("HISTORICAL_LEGACY_BACKUP_CONFLICT")
            else:
                backup.write_bytes(old)
    path.write_text(json.dumps(payload, indent=2))
    return backup if backup.exists() else None


def split_record_by_publication_scope(results):
    """Keep original result facts intact; segregate legacy unpriced SOG.

    Historical fixed-line hits are still valid *classifier outcomes*, but
    not verified 2.5 sportsbook bet outcomes. Headline/published record
    totals therefore exclude this group. Everything remains available in
    a labeled historical research compartment for forensic comparisons.
    """
    if not isinstance(results, list):
        raise TypeError("results must be a list")
    official, legacy = [], []
    for row in results:
        if not isinstance(row, dict):
            raise ValueError("invalid graded record")
        target = legacy if row.get("market") in SOG_CLASSIFIERS else official
        target.append(row)
    return official, legacy


def result_scope_summary(results):
    total = len(results)
    hits = sum(1 for r in results if r.get("result") == "hit")
    markets = defaultdict(lambda: {"hits":0, "total":0})
    for r in results:
        m = r.get("market")
        if not isinstance(m, str):
            raise ValueError("graded result without market")
        if r.get("result") not in {"hit", "miss"}:
            raise ValueError("graded result without a binary outcome")
        v = markets[m]
        v["total"] += 1
        v["hits"] += (r["result"] == "hit")
    by_market = {
        m: {**v, "hit_rate": round(v["hits"]/v["total"]*100,1)}
        for m,v in sorted(markets.items())
    }
    summary = {"total":total, "hits":hits, "misses":total-hits,
               "hit_rate":round(hits/total*100,1) if total else 0}
    return summary, by_market
