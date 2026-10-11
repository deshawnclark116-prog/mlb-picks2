#!/usr/bin/env python3
"""NHL V2 frozen first-pregame shot-count accuracy, independent of betting lines.

Read the original hash-chained forecast ledger and official-final skater outcome
ledger. Both are separately verified by the locked original research engine and
strong official boxscore-reconciled grader before this script is invoked.
This file never forecasts, changes original rows, fits parameters or promotes picks.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

SCHEMA = "NHL_V2_FIRST_PREGAME_ACTUAL_SHOT_COUNT_GRADE_V1"
HORIZONS = ("T24H", "T90", "T30")


def utc(s):
    if not isinstance(s, str):
        raise ValueError("timestamp missing")
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("timestamp lacks timezone")
    return d.astimezone(timezone.utc)


def nonnegative_number(x):
    return (isinstance(x, (float, int)) and not isinstance(x, bool)
            and math.isfinite(x) and x >= 0)


def first_pregame_pairs(forecasts, outcomes):
    """Reject ambiguous identities, invalid clocks or unverifiable final games."""
    used_forecasts, identity_count = {}, Counter()
    for row in forecasts:
        if row.get("record_type") != "FORECAST":
            continue
        if row.get("forecast_horizon") not in HORIZONS:
            raise ValueError("unsupported horizon in frozen forecast")
        identity = (str(row.get("game_id")), row.get("forecast_horizon"),
                    row.get("scheduled_start"), str(row.get("player_id")))
        identity_count[identity] += 1
        rh = row.get("row_hash")
        if not rh or rh in used_forecasts:
            raise ValueError("duplicate or absent original forecast row hash")
        used_forecasts[rh] = row
    conflicting = {identity for identity, n in identity_count.items() if n > 1}
    used_outcomes = set()
    accepted = []
    excluded = Counter()
    for g in outcomes:
        if g.get("record_type") != "GRADE":
            if g.get("record_type") == "UNGRADED":
                excluded["SOURCE_OFFICIAL_UNGRADED"] += 1
            continue
        rh = g.get("forecast_row_hash")
        if not isinstance(rh, str) or rh in used_outcomes:
            raise ValueError("repeated or absent grade for original forecast row")
        used_outcomes.add(rh)
        f = used_forecasts.get(rh)
        if not f:
            raise ValueError("official outcome does not link to original forecast row")
        if (f.get("forecast_id") != g.get("forecast_id")
                or str(f.get("game_id")) != str(g.get("game_id"))
                or str(f.get("player_id")) != str(g.get("player_id"))
                or f.get("forecast_horizon") != g.get("forecast_horizon")):
            raise ValueError("original forecast and final outcome identity mismatch")
        identity = (str(f["game_id"]), f["forecast_horizon"],
                    f["scheduled_start"], str(f["player_id"]))
        if identity in conflicting:
            raise ValueError("ambiguous duplicate player/team forecast graded")
        if not (utc(f["generated_at"]) <= utc(f["cutoff_at"])
                < utc(f["scheduled_start"])):
            raise ValueError("post-cutoff or post-puck-drop forecast contamination")
        if g.get("game_state") not in ("OFF", "FINAL"):
            raise ValueError("outcome not official-final")
        sog = g.get("actual_sog")
        if (not isinstance(sog, int) or isinstance(sog, bool) or sog < 0
                or g.get("played") not in (True, False)):
            raise ValueError("unverified actual shots or participation")
        predicted = f.get("expected_sog")
        if not nonnegative_number(predicted):
            raise ValueError("invalid first-pregame expected shot count")
        if g["played"] is False and sog != 0:
            raise ValueError("nonparticipant cannot have official shots")
        accepted.append((f, g))
    excluded["pending_or_never_graded_forecast_rows"] = (
        len(used_forecasts) - len(used_outcomes)
    )
    excluded["quarantined_ambiguous_forecast_rows"] = sum(
        identity_count[k] for k in conflicting
    )
    return accepted, dict(excluded)


def summarize(rows):
    """All shot-count performance is compared on exactly matched player-games."""
    if not rows:
        return {"n": 0, "status": "NO_OFFICIAL_FIRST_PREGAME_MATCHED_OUTCOMES"}
    differences = [float(f["expected_sog"]) - g["actual_sog"] for f, g in rows]
    out = {
        "n": len(rows),
        "games": len({f["game_id"] for f, _ in rows}),
        "played": sum(g["played"] for _, g in rows),
        "predicted_mean_shots": round(mean(float(f["expected_sog"]) for f, _ in rows), 4),
        "actual_mean_shots": round(mean(g["actual_sog"] for _, g in rows), 4),
        "mae_expected_shot_count": round(mean(abs(v) for v in differences), 4),
        "bias_projected_minus_actual": round(mean(differences), 4),
        "within_1_shot": round(mean(abs(v) <= 1 for v in differences), 4),
        "within_2_shots": round(mean(abs(v) <= 2 for v in differences), 4),
        "large_miss_over_3_shots": sum(abs(v) > 3 for v in differences),
    }
    for name, key in (("prior_10_game_average", "simple_prior10_mean"),
                      ("frozen_human_baseline", "human_frozen_mean")):
        selected = [(f, g) for f, g in rows
                    if isinstance(f.get("comparators"), dict)
                    and nonnegative_number(f["comparators"].get(key))]
        if not selected:
            out[name] = {"n": 0, "status": "COMPARATOR_UNAVAILABLE"}
            continue
        baseline_mae = mean(abs(float(f["comparators"][key]) - g["actual_sog"])
                            for f, g in selected)
        model_mae = mean(abs(float(f["expected_sog"]) - g["actual_sog"])
                         for f, g in selected)
        out[name] = {
            "n": len(selected),
            "same_rows_model_mae": round(model_mae, 4),
            "same_rows_comparator_mae": round(baseline_mae, 4),
            "mae_gain_positive_favors_model": round(baseline_mae - model_mae, 4),
        }
    return out


def report(forecasts, outcomes):
    pairs, reasons = first_pregame_pairs(forecasts, outcomes)
    versions = sorted({f.get("engine_version") for f, _ in pairs})
    if any(not v for v in versions):
        raise ValueError("unversioned NHL research forecast")
    sections = {}
    for h in HORIZONS:
        hrows = [(f, g) for f, g in pairs if f["forecast_horizon"] == h]
        sections[h] = {
            "all_candidates_unconditional": summarize(hrows),
            "meaningful_expected_participants": summarize([
                (f, g) for f, g in hrows
                if f.get("meaningful_expected_participant") is True]),
            "meaningful_and_actually_played_diagnostic_only": summarize([
                (f, g) for f, g in hrows
                if f.get("meaningful_expected_participant") is True and g["played"]]),
        }
    return {
        "schema": SCHEMA,
        "status": "FORWARD_OBSERVATIONAL_NO_BETTING_PROMOTION",
        "original_model_versions": versions,
        "multiple_model_versions_must_not_be_pooled": len(versions) > 1,
        "first_pregame_official_grade_rows": len(pairs),
        "excluded": reasons,
        "by_horizon": sections,
        "original_forecasts_mutated": False,
        "official_betting_picks": 0,
        "no_recalibration_or_feature_tuning_performed": True,
        "no_alt_line_or_sportsbook_market_substitution": True,
        "note": (
            "Primary target is actual shot-count MAE on meaningful expected participants; "
            "played-only is diagnostic and can be biased by hindsight participation. "
            "Historical V2 probability threshold fields were repaired only posthoc, "
            "not originally published as pregame probabilities."
        )
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-ledger", required=True, type=Path)
    p.add_argument("--official-grade-ledger", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    forecasts = [json.loads(x) for x in args.forecast_ledger.read_text().splitlines()
                 if x.strip()]
    grades = [json.loads(x) for x in args.official_grade_ledger.read_text().splitlines()
              if x.strip()] if args.official_grade_ledger.exists() else []
    result = report(forecasts, grades)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2,
                                   allow_nan=False) + "\n")
    print(json.dumps({
        "schema": result["schema"],
        "matched_official_count": result["first_pregame_official_grade_rows"],
        "T90_meaningful_mae": result["by_horizon"]["T90"]
                              ["meaningful_expected_participants"]
                              .get("mae_expected_shot_count"),
        "reasons": result["excluded"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
