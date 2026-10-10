"""NHL Phase1C: versioned, read-only interpretation of frozen B2 probability fields.

Original B2 forecast records are immutable. Historical P1..P5 are mislabeled:
P1=1 and Pk=P(SOG>=k-1) for k=2..5. This adapter independently computes
P(SOG>=1..5) from the frozen (expected_sog, dispersion) NB2 parameters.
It never mutates a forecast, never claims the corrected values were
published at the historical forecast time, and is NOT a model promotion.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import nbinom

SCHEMA = "nhl-b2-probability-interpretation-v1"
HORIZONS = frozenset({"T24H", "T90", "T30"})
THRESHOLDS = (1, 2, 3, 4, 5)
UNSAFE_LEGACY = "LEGACY_B2_SHIFTED_FIELDS_UNSAFE_FOR_THRESHOLD_DISPLAY"


class ProbabilityIntegrityError(ValueError):
    pass


def _finite_nonnegative(value: object, field: str, *, strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProbabilityIntegrityError(f"{field} must be a finite number")
    v = float(value)
    if not math.isfinite(v) or v < 0 or (strictly_positive and v <= 0):
        raise ProbabilityIntegrityError(f"{field} must be finite and {'positive' if strictly_positive else 'nonnegative'}")
    return v


def nb2_thresholds(expected_sog: float, dispersion: float) -> dict[str, float]:
    """P(SOG>=k) is sf(k-1), NOT sf(k), for integer k in 1..5."""
    mu = _finite_nonnegative(expected_sog, "expected_sog")
    alpha = _finite_nonnegative(dispersion, "dispersion", strictly_positive=True)
    r = 1.0 / alpha
    p = r / (r + mu)
    values = {f"P{k}": float(nbinom.sf(k - 1, r, p)) for k in THRESHOLDS}
    for v in values.values():
        if not 0 <= v <= 1 or not math.isfinite(v):
            raise ProbabilityIntegrityError("invalid NB2 survival probability")
    if any(values[f"P{k}"] + 1e-12 < values[f"P{k+1}"] for k in range(1, 5)):
        raise ProbabilityIntegrityError("survival distribution is not monotone")
    return values


def interpretation(row: dict) -> dict:
    """Returns a separate metadata sidecar; does not copy or change legacy probabilities."""
    if row.get("record_type") != "FORECAST":
        raise ProbabilityIntegrityError("only immutable FORECAST records accepted")
    for f in ("id", "game_id", "forecast_horizon", "cutoff_at", "expected_sog", "dispersion"):
        if f not in row:
            raise ProbabilityIntegrityError(f"missing required frozen field {f}")
    if row["forecast_horizon"] not in HORIZONS:
        raise ProbabilityIntegrityError("invalid NHL horizon")
    corrected = nb2_thresholds(row["expected_sog"], row["dispersion"])
    old = {f"P{k}": _finite_nonnegative(row[f"P{k}"], f"P{k}") for k in THRESHOLDS}
    mismatch = [k for k in THRESHOLDS if not math.isclose(old[f"P{k}"], corrected[f"P{k}"], abs_tol=1e-8, rel_tol=0)]
    return {
        "schema": SCHEMA,
        "record_id": row["id"], "game_id": row["game_id"],
        "forecast_horizon": row["forecast_horizon"], "original_cutoff_at": row["cutoff_at"],
        "probability_semantics": "P(SOG>=integer_threshold)",
        "calculation": "NB2_SF_K_MINUS_ONE_FROM_ORIGINAL_MU_ALPHA",
        "corrected_posthoc": corrected,
        "legacy_fields_policy": UNSAFE_LEGACY,
        "legacy_mismatch_thresholds": mismatch,
        "original_record_unchanged": True,
        "historical_pregame_publication_claimed": False,
        "evaluation_only": True,
    }


def audit(rows) -> dict:
    seen, n, mismatch = set(), 0, {f"P{k}": 0 for k in THRESHOLDS}
    for row in rows:
        if row.get("record_type") != "FORECAST":
            continue
        item = interpretation(row)
        if item["record_id"] in seen:
            raise ProbabilityIntegrityError("duplicate immutable forecast ID")
        seen.add(item["record_id"])
        n += 1
        for k in item["legacy_mismatch_thresholds"]:
            mismatch[f"P{k}"] += 1
    return {"schema": SCHEMA, "n_forecasts_checked": n,
            "legacy_shifted_threshold_mismatches": mismatch,
            "legacy_policy": UNSAFE_LEGACY,
            "frozen_engine_or_ledger_modified": False,
            "historical_original_publication_corrected": False,
            "research_promotion": False}


def main():
    ap = argparse.ArgumentParser(description="Read-only audit of frozen NHL forecast ledger")
    ap.add_argument("--output", type=Path, required=True, help="Separate derived report, not forecast ledger")
    opts = ap.parse_args()
    import nhl_v2_phase1a_sog_forward as F
    F.verify_lock()
    ledger = F.Ledger()
    ledger.verify()
    result = audit(ledger.rows())
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
