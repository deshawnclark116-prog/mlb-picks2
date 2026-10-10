#!/usr/bin/env python3
"""Publish NFL-style NHL V2 shot-count forecasts as RESEARCH, never as bets.

Read-only and forward-evidence-preserving. The frozen B2 ledger is independently
hash-chain-verified and the model/code lock is verified before emitting anything.
Historical B2 P1..P5 are shifted; this separate view derives *posthoc* proper
NB2 thresholds from original expected_sog and dispersion, without overwriting
or claiming corrected probability labels were part of the original forecast.

Never read docs/nhl_predictions.json (legacy fixed 2.5-shot classifier).
No sportsbook lines, fabricated probabilities, retconning, roster certification,
or aggregation of contradictory cross-team forecasts.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import nbinom

import nhl_v2_phase1c_probabilities as P

SCHEMA = "nhl-v2-sog-nfl-style-shadow-v1"
HORIZONS = ("T24H", "T90", "T30")
BLOCKED = "RESEARCH_ONLY_NOT_VALIDATED_FOR_BETTING"
FIELDS = (
    "game_id", "player_id", "player", "team", "opponent", "position",
    "game_date", "scheduled_start", "forecast_horizon", "cutoff_at",
    "generated_at", "model_version", "mean", "median", "sd",
    "p10", "p25", "p75", "p90", "p0", "p_ge1", "p_ge2", "p_ge3",
    "p_ge4", "p_ge5", "meaningful_candidate", "availability_state",
    "availability_confidence", "recent10_sog", "recent3_toi_seconds",
    "recent3_pp_toi_seconds", "opp_sog_allowed_mean5", "rest_hours",
    "original_row_hash", "forecast_id", "posthoc_probability_semantics",
)


class BoardSafetyError(ValueError):
    pass


def ts(value):
    if not isinstance(value, str):
        raise BoardSafetyError("missing verified timestamp")
    try:
        d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BoardSafetyError("invalid timestamp") from exc
    if d.tzinfo is None:
        raise BoardSafetyError("naive timestamp")
    return d.astimezone(timezone.utc)


def safe_float(raw, name, *, nonnegative=False):
    if isinstance(raw, bool) or not isinstance(raw, (float, int)):
        raise BoardSafetyError(f"invalid {name}")
    v = float(raw)
    if not math.isfinite(v) or (nonnegative and v < 0):
        raise BoardSafetyError(f"invalid {name}")
    return v


def fmt(x, nd=4):
    return round(float(x), nd)


def normalize(row):
    if row.get("record_type") != "FORECAST":
        raise BoardSafetyError("forecast-only normalization")
    if row.get("forecast_horizon") not in HORIZONS:
        raise BoardSafetyError("unrecognized horizon")
    if row.get("availability_used_in_forecast") is not False:
        raise BoardSafetyError("unknown target-game availability input")
    if row.get("availability_confidence") != "NOT_CERTIFIED":
        raise BoardSafetyError("unexpected dressed status; require new qualification")
    if row.get("schedule_state", "OK") != "OK":
        raise BoardSafetyError("schedule revised or cancelled")
    if not row.get("source_manifest_sha256"):
        raise BoardSafetyError("missing source manifest")
    if not row.get("row_hash") or not row.get("forecast_id"):
        raise BoardSafetyError("missing immutable row evidence")
    start = ts(row.get("scheduled_start"))
    cutoff = ts(row.get("cutoff_at"))
    gen = ts(row.get("generated_at"))
    if gen > cutoff or cutoff >= start:
        raise BoardSafetyError("forecast generated late or after start")
    if row.get("player_id") is None or row.get("game_id") is None:
        raise BoardSafetyError("missing unique identity")
    if not row.get("team") or not row.get("opponent") or row["team"] == row["opponent"]:
        raise BoardSafetyError("ambiguous team assignment")

    mu = safe_float(row.get("expected_sog"), "expected_sog", nonnegative=True)
    alpha = safe_float(row.get("dispersion"), "dispersion")
    if alpha <= 0:
        raise BoardSafetyError("non-positive dispersion")
    # Correct the legacy shifted P1..P5 only in a clearly labeled derivative.
    # Unlike current live NHL fixed-line board, this is a count distribution.
    corrected = P.nb2_thresholds(mu, alpha)
    r = 1.0 / alpha
    p = r / (r + mu)
    v = mu + alpha * mu * mu
    q = [float(nbinom.ppf(prob, r, p)) for prob in (.1, .25, .5, .75, .9)]
    if not all(math.isfinite(x) and x >= 0 for x in q):
        raise BoardSafetyError("bad NB2 quantile")
    p0 = float(nbinom.pmf(0, r, p))
    if not 0 <= p0 <= 1:
        raise BoardSafetyError("invalid p0")
    receipt = row.get("receipt") or {}
    hist = receipt.get("recent_sog_history_last10_appearances") or []
    if not isinstance(hist, list) or any(
        isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in hist
    ):
        raise BoardSafetyError("invalid past shot-history receipt")
    toi = receipt.get("toi_seconds") or {}
    version = row.get("engine_version") or row.get("model_version") or "B2 locked"

    values = {
        "game_id": str(row["game_id"]),
        "player_id": str(row["player_id"]),
        "player": receipt.get("player_name") or f"Player {row['player_id']}",
        "team": str(row["team"]), "opponent": str(row["opponent"]),
        "position": receipt.get("position") or "–",
        "game_date": row.get("schedule_date") or start.date().isoformat(),
        "scheduled_start": row["scheduled_start"],
        "forecast_horizon": row["forecast_horizon"],
        "cutoff_at": row["cutoff_at"],
        "generated_at": row["generated_at"],
        "model_version": version,
        "mean": fmt(mu), "median": fmt(q[2]), "sd": fmt(math.sqrt(v)),
        "p10": fmt(q[0]), "p25": fmt(q[1]), "p75": fmt(q[3]), "p90": fmt(q[4]),
        "p0": fmt(p0), "p_ge1": fmt(corrected["P1"]),
        "p_ge2": fmt(corrected["P2"]), "p_ge3": fmt(corrected["P3"]),
        "p_ge4": fmt(corrected["P4"]), "p_ge5": fmt(corrected["P5"]),
        "meaningful_candidate": row.get("meaningful_expected_participant") is True,
        "availability_state": row.get("availability_state") or "NOT_CAPTURED",
        "availability_confidence": "NOT_CERTIFIED",
        "recent10_sog": hist,
        "recent3_toi_seconds": toi.get("recent3"),
        "recent3_pp_toi_seconds": receipt.get("pp_toi_recent3_seconds"),
        "opp_sog_allowed_mean5": receipt.get("opp_sog_allowed_mean5"),
        "rest_hours": receipt.get("rest_hours"),
        "original_row_hash": row["row_hash"], "forecast_id": row["forecast_id"],
        "posthoc_probability_semantics": "CORRECTED_FROM_ORIGINAL_MU_ALPHA_NOT_ORIGINAL_P1_P5",
    }
    return [values[x] for x in FIELDS]


def derive(rows, *, verified_chain=False, verified_lock=False, source_sha=None):
    """Build a display view; caller MUST have verified whole ledger+frozen lock."""
    if not verified_chain or not verified_lock:
        raise BoardSafetyError("original model and ledger verification required")
    seen_row_hashes = set()
    groups = defaultdict(list)
    decisions = Counter()
    for r in rows:
        if r.get("record_type") != "FORECAST":
            decisions[str(r.get("record_type", "UNKNOWN"))] += 1
            continue
        hashval = r.get("row_hash")
        if not hashval or hashval in seen_row_hashes:
            raise BoardSafetyError("duplicate or missing original row hash")
        seen_row_hashes.add(hashval)
        ident = (str(r.get("game_id")), str(r.get("scheduled_start")),
                 r.get("forecast_horizon"), str(r.get("player_id")))
        groups[ident].append(r)

    forecasts = {h: [] for h in HORIZONS}
    games = {}
    stats = Counter()
    for group, candidates in sorted(groups.items()):
        if len(candidates) != 1:
            stats["quarantined_duplicate_identity_groups"] += 1
            stats["quarantined_duplicate_identity_rows"] += len(candidates)
            if len({x.get("team") for x in candidates}) > 1:
                stats["cross_team_identity_conflicts"] += 1
            continue
        row = candidates[0]
        try:
            transformed = normalize(row)
        except (BoardSafetyError, P.ProbabilityIntegrityError, OverflowError, ValueError):
            stats["quarantined_invalid_forecasts"] += 1
            continue
        h = row["forecast_horizon"]
        forecasts[h].append(transformed)
        game_key = (str(row["game_id"]), row["scheduled_start"])
        if game_key not in games:
            games[game_key] = {
                "game_id": str(row["game_id"]),
                "game_date": row.get("schedule_date") or row["scheduled_start"][:10],
                "scheduled_start": row["scheduled_start"],
                "matchup": " vs ".join(sorted((row["team"], row["opponent"]))),
                "horizons": {},
            }
        if h in games[game_key]["horizons"]:
            prev = games[game_key]["horizons"][h]
            if prev["cutoff_at"] != row["cutoff_at"]:
                # Never publish a game with ambiguous cutoff provenance.
                raise BoardSafetyError("conflicting original per-game cutoff")
        else:
            games[game_key]["horizons"][h] = {
                "cutoff_at": row["cutoff_at"],
                "model_version": transformed[FIELDS.index("model_version")],
            }
    for h in HORIZONS:
        forecasts[h].sort(key=lambda r: (r[FIELDS.index("scheduled_start")],
                         r[FIELDS.index("team")], r[FIELDS.index("player")]))
    all_games = sorted(games.values(), key=lambda g: (g["scheduled_start"], g["game_id"]))
    dates = sorted({g["game_date"] for g in all_games})
    return {
        "schema": SCHEMA,
        "model": "NHL V2 Phase1A B2 frozen NB2 count model",
        "data_version": "posthoc_NB2_probabilities_corrected",
        "status": BLOCKED,
        "not_a_betting_board": True,
        "no_sportsbook_input": True,
        "dressed_lineups_verified": False,
        "actual_player_book_lines_verified": False,
        "historical_corrected_probabilities_were_not_published_pregame": True,
        "original_ledger_untouched": True,
        "verified_original_hash_chain": True,
        "verified_frozen_code_and_model_lock": True,
        "source_ledger_sha256": source_sha,
        "excluded": dict(stats),
        "nonforecast_decisions": dict(decisions),
        "row_fields": list(FIELDS),
        "horizons": list(HORIZONS),
        "dates": dates,
        "games": all_games,
        "forecasts": forecasts,
        "total_valid_forecasts": sum(len(v) for v in forecasts.values()),
        "empty_message": "No verified NHL V2 shot-count forecasts are available. Existing fixed-2.5 legacy bets will not be substituted.",
    }


def main():
    ap = argparse.ArgumentParser(description="Read-only, projection-first NHL V2 shadow publisher")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import hashlib
    import nhl_v2_phase1a_sog_forward as F
    F.verify_lock()
    ledger = F.Ledger()
    ledger.verify()
    provenance = F.audit_ledger()
    if provenance:
        raise BoardSafetyError("source provenance audit has failures")
    original_sha = hashlib.sha256(ledger.path.read_bytes()).hexdigest()
    data = derive(ledger.rows(), verified_chain=True, verified_lock=True,
                  source_sha=original_sha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, sort_keys=True, indent=2,
                                    allow_nan=False, ensure_ascii=False) + "\n")
    print(json.dumps({
        "schema": SCHEMA,
        "original_hash": original_sha,
        "valid": data["total_valid_forecasts"],
        "dates": len(data["dates"]),
        "quarantined": data["excluded"],
        "official_picks": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
