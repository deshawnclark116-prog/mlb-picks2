#!/usr/bin/env python3
"""CFB research-to-frontend adapter: genuine pregame player-stat projections.

Reads the *original* immutably archived research output, not refreshed
docs/cfb_predictions.json and not a recomputed postgame model. Emits a
read-only display contract; never changes the existing live classifier
and never represents baseline averages as calibrated model predictions.

The origin GitHub Actions run must be successfully completed, named, and
prove its run started before EVERY emitted player's original kickoff.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "cfb-player-stat-projections-research-v1"
BASE_SCHEMA = "CFB_2026_BOX_SCORE_LAST3_PROSPECTIVE_BASELINE_V1"
MC_SCHEMA = "CFB_2026_PER_EVENT_SHADOW_V1"
ORIGIN_WORKFLOW = "CFB 2026 event data qualification (research, no serving)"
MARKETS = {"rushing_yards", "passing_touchdowns", "anytime_touchdowns"}
BINS = {"rushing_yards": "yards", "passing_touchdowns": "touchdowns", "anytime_touchdowns": "touchdowns"}


class PublicationBlocked(ValueError):
    pass


def dt(s):
    if not isinstance(s, str):
        raise PublicationBlocked("MISSING_TIME")
    try:
        result = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError as e:
        raise PublicationBlocked("INVALID_TIME") from e
    if result.tzinfo is None:
        raise PublicationBlocked("NON_UTC_OR_NAIVE_TIME")
    return result.astimezone(timezone.utc)


def numeric(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise PublicationBlocked("BAD_" + name.upper())
    return float(value)


def digest(b):
    return hashlib.sha256(b).hexdigest()


def _origin(meta, expected_id):
    if (meta.get("id") != expected_id or meta.get("name") != ORIGIN_WORKFLOW
        or meta.get("status") != "completed" or meta.get("conclusion") != "success"
        or not isinstance(meta.get("head_sha"), str) or len(meta["head_sha"]) != 40):
        raise PublicationBlocked("UNVERIFIED_ORIGINAL_RESEARCH_RUN")
    created, started, finished = [dt(meta[k]) for k in ("created_at", "run_started_at", "updated_at")]
    if not created <= started <= finished:
        raise PublicationBlocked("INVALID_ORIGIN_RUN_TIMELINE")
    return created, started, finished


def convert(base, shadow, meta, *, original_run_id, base_bytes=None, shadow_bytes=None):
    if base.get("schema") != BASE_SCHEMA or base.get("status") != "FROZEN_PREGAME_NAIVE_POINT_BASELINE_ONLY":
        raise PublicationBlocked("ORIGINAL_BASELINE_SCHEMA_NOT_CERTIFIED")
    if shadow.get("schema") != MC_SCHEMA or shadow.get("status") != "NEW_PREKICKOFF_RESEARCH_ONLY_NOT_PRODUCTION":
        raise PublicationBlocked("ORIGINAL_SHADOW_SCHEMA_NOT_CERTIFIED")
    if (base.get("historical_forecasts_backfilled") != 0 or base.get("production_model_modified") is not False
        or shadow.get("historical_backfills") != 0 or shadow.get("forward_original_ledger_modified") is not False):
        raise PublicationBlocked("RETROACTIVE_OR_PRODUCTION_DATA")
    if not isinstance(base.get("forecasts"), list) or len(base["forecasts"]) != base.get("baseline_rows"):
        raise PublicationBlocked("BASELINE_ROW_COUNT_MISMATCH")
    if not isinstance(shadow.get("shadow"), list) or len(shadow["shadow"]) != shadow.get("shadow_predictions"):
        raise PublicationBlocked("MONTE_CARLO_ROW_COUNT_MISMATCH")
    created, started, finished = _origin(meta, original_run_id)
    base_at, mc_at = dt(base["generated_at_utc"]), dt(shadow["generated_at_utc"])
    if not started <= base_at <= finished or not started <= mc_at <= finished:
        raise PublicationBlocked("ORIGINAL_FORECAST_NOT_WITHIN_ACTIONS_RUN")
    out, bykey = [], {}
    for row in base["forecasts"]:
        key = (str(row["game_id"]), str(row["player_id"]), row["market"])
        if key in bykey:
            raise PublicationBlocked("DUPLICATE_PLAYER_GAME_MARKET")
        if row["market"] not in MARKETS or row.get("season") != 2026 or row.get("week") != 6:
            raise PublicationBlocked("WRONG_MARKET_OR_SEASON")
        kickoff = dt(row["kickoff_utc"])
        if not created < kickoff or not started < kickoff or not base_at < kickoff:
            raise PublicationBlocked("BASELINE_NOT_GENERATED_BEFORE_KICKOFF")
        if dt(row["research_generated_at_utc"]) != base_at or dt(row["board_generated_at_utc"]) > base_at:
            raise PublicationBlocked("ORIGINAL_BASELINE_DATE_DISAGREEMENT")
        vals = row.get("last3_values")
        ids = row.get("last3_game_ids")
        if not isinstance(vals, list) or len(vals) != 3 or not isinstance(ids, list) or len(ids) != 3 or len(set(ids)) != 3:
            raise PublicationBlocked("UNVERIFIED_THREE_GAME_HISTORY")
        vals = [numeric(v, "prior_game_total") for v in vals]
        mean, median = numeric(row["point_mean_last3"], "baseline_mean"), numeric(row["point_median_last3"], "baseline_median")
        if abs(mean - sum(vals)/3) > 0.00006 or abs(median - sorted(vals)[1]) > 0.000001:
            raise PublicationBlocked("RETROACTIVE_BASELINE_REWRITE")
        result = {
            "game_id": key[0], "player_id": key[1], "player": row["player"],
            "market": key[2], "team": row["team"], "opponent": row["opponent"],
            "position": row["position"], "kickoff_utc": row["kickoff_utc"],
            "unit": BINS[key[2]], "baseline": {
                "mean": round(mean, 4), "median": round(median, 4),
                "last_three": vals, "prior_game_ids": ids,
                "kind": "three_game_average_NOT_trained_model",
            },
            "monte_carlo": None, "presentation_status": "PREGAME_RESEARCH_ONLY",
        }
        bykey[key] = result
        out.append(result)
    used_mc = set()
    for row in shadow["shadow"]:
        key = (str(row["game_id"]), str(row["player_id"]), row["market"])
        if key in used_mc or key not in bykey or key[2] != "rushing_yards":
            raise PublicationBlocked("DUPLICATE_OR_UNMATCHED_MONTE_CARLO")
        used_mc.add(key)
        parent = bykey[key]
        if row["team"] != parent["team"] or dt(row["kickoff_utc"]) != dt(parent["kickoff_utc"]):
            raise PublicationBlocked("MONTE_CARLO_IDENTITY_OR_KICKOFF_DIFFERENCE")
        kickoff = dt(row["kickoff_utc"])
        if not started < kickoff or not mc_at < kickoff or dt(row["forecast_generated_at_utc"]) != mc_at:
            raise PublicationBlocked("MONTE_CARLO_NOT_ORIGINALLY_PREGAME")
        n = row["n_simulations"]
        hist = row.get("empirical_outcome_histogram")
        if (type(n) is not int or n < 100 or not isinstance(hist, list) or not hist
            or row.get("empirical_outcome_histogram_schema") != "INTEGER_STAT_TOTAL_AND_FREQUENCY_V1"):
            raise PublicationBlocked("MONTE_CARLO_HISTOGRAM_MISSING")
        total_n, weighted, prev, above = 0, 0, None, 0
        line = numeric(row["original_classifier_line"], "original_fixed_research_threshold")
        for bin_ in hist:
            value, count = bin_.get("stat_total"), bin_.get("count")
            if type(value) is not int or type(count) is not int or count <= 0 or (prev is not None and value <= prev):
                raise PublicationBlocked("BAD_OR_UNSORTED_SIMULATED_OUTCOMES")
            total_n += count
            weighted += value * count
            if value > line:
                above += count
            prev = value
        mean = numeric(row["projected_mean"], "mc_mean")
        median = numeric(row["projected_median"], "mc_median")
        p10, p90 = numeric(row["p10"], "p10"), numeric(row["p90"], "p90")
        if (total_n != n or not p10 <= median <= p90
            or abs(weighted/n - numeric(row["sample_mean_unrounded"], "sample_mean")) > 0.00001
            or abs(weighted/n - mean) > 0.051
            or abs(above/n - numeric(row["p_over_fixed_line"], "research_threshold_prob")) > 0.00051):
            raise PublicationBlocked("SIMULATED_DISTRIBUTION_NOT_CERTIFIED")
        parent["monte_carlo"] = {
            "mean": mean, "median": median, "p10": p10, "p90": p90,
            "n_simulations": n,
            "kind": "uncalibrated_empirical_distribution_research_only",
            "source_sha256": row["source_sha256"],
        }
    out.sort(key=lambda r: (r["kickoff_utc"], r["market"], r["team"], r["player"]))
    return {
        "schema": SCHEMA,
        "original_run_id": original_run_id, "original_workflow": ORIGIN_WORKFLOW,
        "original_actions_sha": meta["head_sha"], "original_run_created_at": meta["created_at"],
        "original_run_started_at": meta["run_started_at"],
        "original_baseline_generated_at": base["generated_at_utc"],
        "original_monte_carlo_generated_at": shadow["generated_at_utc"],
        "original_baseline_sha256": digest(base_bytes) if base_bytes is not None else None,
        "original_monte_carlo_sha256": digest(shadow_bytes) if shadow_bytes is not None else None,
        "forecast_count": len(out), "monte_carlo_count": len(used_mc),
        "by_market": dict(Counter(r["market"] for r in out)),
        "official_new_betting_picks": 0, "is_live_predictions_feed": False,
        "research_only": True, "backfilled_forecasts": 0,
        "reason": "Baseline is raw past-three-game mean; simulated distribution has not earned promotion. No verified sportsbook line or player injury/lineup confirmation.",
        "forecasts": out,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--shadow", type=Path, required=True)
    p.add_argument("--origin-metadata", type=Path, required=True)
    p.add_argument("--origin-run-id", required=True, type=int)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    b, s = a.baseline.read_bytes(), a.shadow.read_bytes()
    result = convert(json.loads(b), json.loads(s),
                     json.loads(a.origin_metadata.read_bytes()),
                     original_run_id=a.origin_run_id, base_bytes=b, shadow_bytes=s)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "forecasts"},sort_keys=True))


if __name__ == "__main__":
    main()
