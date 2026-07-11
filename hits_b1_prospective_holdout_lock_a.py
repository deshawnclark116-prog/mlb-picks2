#!/usr/bin/env python3
"""
HITS_B1_PROSPECTIVE_HOLDOUT_LOCK_A

Creates an immutable prospective-holdout lock for the frozen B1 batter-hits
challenger before any future holdout outcomes are inspected.

Frozen challenger:
    B1 = direct binary XGBoost P(hit >= 1)
         + opp_pitcher_contact_allowed_rate_60d

Frozen base features:
    season_avg
    recent15_avg
    recent5_avg
    hr_rate
    bb_rate
    so_rate
    batting_order
    games_played

Frozen extra feature:
    opp_pitcher_contact_allowed_rate_60d
    prior 60-day opposing-pitcher contact allowed rate
    minimum 150 prior pitcher swings

Frozen XGBoost architecture:
    objective=binary:logistic
    learning_rate=0.05
    max_depth=6
    subsample=0.8
    colsample_bytree=0.8
    min_child_weight=5
    tree_method=hist
    nthread=2
    seed=42
    num_boost_round=120

Holdout trigger:
    10,000 eligible batter-games.

No peeking before trigger:
    Allowed:
      - prediction-row count
      - duplicate-key count
      - pipeline health
      - feature coverage/missingness
      - dates represented
      - operational latency/errors
      - model/spec/hash verification

    Forbidden:
      - hit rate
      - Brier
      - logloss
      - AUC
      - calibration
      - top buckets
      - fixed-coverage performance
      - official-tier performance
      - monthly performance
      - any outcome-conditioned analysis

Formal gate:
    B1 passes only if ALL locked conditions pass.

Run:
    python -u hits_b1_prospective_holdout_lock_a.py

Verify later:
    python -u hits_b1_prospective_holdout_lock_a.py --verify
"""

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


OUT_JSON = Path("/data/hr_model/hits_b1_prospective_holdout_lock_a.json")
OUT_TXT = Path("/data/hr_model/hits_b1_prospective_holdout_lock_a.txt")
OUT_SHA = Path("/data/hr_model/hits_b1_prospective_holdout_lock_a.sha256")

LOCK_VERSION = "hits_b1_prospective_holdout_lock_a_v1"

BASE_FEATURES = [
    "season_avg",
    "recent15_avg",
    "recent5_avg",
    "hr_rate",
    "bb_rate",
    "so_rate",
    "batting_order",
    "games_played",
]

XGB_PARAMS = {
    "objective": "binary:logistic",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    "nthread": 2,
    "seed": 42,
    "num_boost_round": 120,
}


def canonical_json_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_lock_payload():
    freeze_utc = datetime.now(timezone.utc).replace(microsecond=0)
    eastern = freeze_utc.astimezone(ZoneInfo("America/New_York"))
    holdout_start_date = (eastern.date() + timedelta(days=1)).isoformat()

    return {
        "lock_version": LOCK_VERSION,
        "status": "FROZEN_PROSPECTIVE_CHALLENGER",
        "production_promotion_authorized": False,
        "formal_gate_completed": False,
        "freeze_timestamp_utc": freeze_utc.isoformat(),
        "freeze_timestamp_america_new_york": eastern.isoformat(),
        "holdout_start_game_date_america_new_york": holdout_start_date,
        "holdout_trigger": {
            "type": "eligible_batter_games",
            "minimum_rows": 10000,
            "open_once_only": True,
        },
        "challenger": {
            "name": "B1",
            "target": "actual_hit",
            "target_definition": "1 if actual game hits >= 1 else 0",
            "architecture": "direct_binary_xgboost",
            "probability_output": "P(hit >= 1)",
            "base_features": BASE_FEATURES,
            "extra_feature": {
                "name": "opp_pitcher_contact_allowed_rate_60d",
                "definition": (
                    "prior 60-day opposing-pitcher contact allowed rate"
                ),
                "window_days": 60,
                "minimum_prior_pitcher_swings": 150,
                "expected_direction": "positive",
                "frozen": True,
            },
            "xgboost_params": XGB_PARAMS,
            "monte_carlo_used": False,
            "poisson_mapping_used": False,
            "frozen": True,
        },
        "comparators": {
            "A0": {
                "role": "clean_incumbent_architecture",
                "architecture": (
                    "count_poisson_xgboost_then_exact_1_minus_exp_minus_lambda"
                ),
                "fixed_threshold": 0.63,
            },
            "B0": {
                "role": "architecture_isolation_control",
                "architecture": (
                    "direct_binary_xgboost_without_pitcher_contact"
                ),
            },
            "B1": {
                "role": "frozen_prospective_challenger",
                "architecture": (
                    "direct_binary_xgboost_plus_"
                    "opp_pitcher_contact_allowed_rate_60d"
                ),
            },
        },
        "eligibility": {
            "definition": (
                "unique post-freeze batter-game row for which both clean A0 "
                "and frozen B1 can produce a prediction using the locked pipeline"
            ),
            "unique_key_required": True,
            "recommended_unique_key": "game_pk|player_id|game_date",
            "duplicate_rows_disallowed": True,
        },
        "no_peeking": {
            "allowed_before_trigger": [
                "total prediction-row count",
                "duplicate-key count",
                "pipeline success/failure",
                "feature coverage and missingness",
                "dates represented",
                "operational latency and errors",
                "model/spec/hash verification",
            ],
            "forbidden_before_trigger": [
                "hit rate",
                "Brier score",
                "logloss",
                "AUC",
                "calibration",
                "top-5%",
                "top-10%",
                "fixed-coverage performance",
                "official-tier performance",
                "monthly performance",
                "any outcome-conditioned analysis",
            ],
            "outcomes_must_remain_uninspected": True,
        },
        "decision_policy": {
            "fixed_threshold_0_63": {
                "status": "DIAGNOSTIC_ONLY",
                "reason": "0.63 is not architecture-neutral",
            },
            "fixed_coverage": {
                "status": "FORMAL_DECISION_LAYER_COMPARISON",
                "definition": (
                    "N equals A0 count of predictions with P >= 0.63 over the "
                    "complete prospective holdout; each cell is evaluated on "
                    "its own top N predictions"
                ),
            },
        },
        "formal_gate": {
            "all_conditions_required": True,
            "conditions": [
                {
                    "id": 1,
                    "name": "relative_brier_vs_A0",
                    "rule": (
                        "(A0_brier - B1_brier) / A0_brier >= 0.0025"
                    ),
                    "minimum_relative_improvement": 0.0025,
                },
                {
                    "id": 2,
                    "name": "paired_date_block_bootstrap_brier_probability",
                    "rule": (
                        "P(B1 Brier < A0 Brier) >= 0.95 using paired "
                        "calendar-date block bootstrap"
                    ),
                    "minimum_probability": 0.95,
                },
                {
                    "id": 3,
                    "name": "logloss_vs_A0",
                    "rule": "B1_logloss <= A0_logloss",
                },
                {
                    "id": 4,
                    "name": "auc_protection_vs_A0",
                    "rule": "B1_auc >= A0_auc - 0.0005",
                    "maximum_auc_decline": 0.0005,
                },
                {
                    "id": 5,
                    "name": "top5_protection_vs_A0",
                    "rule": (
                        "B1_top5_actual_hit_rate >= A0_top5_actual_hit_rate"
                    ),
                },
                {
                    "id": 6,
                    "name": "top10_protection_vs_A0",
                    "rule": (
                        "B1_top10_actual_hit_rate >= A0_top10_actual_hit_rate"
                    ),
                },
                {
                    "id": 7,
                    "name": "fixed_coverage_hit_rate_vs_A0",
                    "rule": (
                        "B1_fixed_coverage_hit_rate >= A0_fixed_coverage_hit_rate"
                    ),
                },
                {
                    "id": 8,
                    "name": "feature_incremental_brier_vs_B0",
                    "rule": "B1_brier < B0_brier",
                },
                {
                    "id": 9,
                    "name": "feature_incremental_logloss_vs_B0",
                    "rule": "B1_logloss <= B0_logloss",
                },
                {
                    "id": 10,
                    "name": "feature_fixed_coverage_hit_rate_vs_B0",
                    "rule": (
                        "B1_fixed_coverage_hit_rate >= B0_fixed_coverage_hit_rate"
                    ),
                },
                {
                    "id": 11,
                    "name": "monthly_brier_stability_vs_A0",
                    "rule": (
                        "B1 monthly Brier <= A0 monthly Brier in at least half "
                        "of eligible months having >= 500 holdout rows"
                    ),
                    "minimum_rows_per_eligible_month": 500,
                    "minimum_nonworse_fraction": 0.5,
                },
                {
                    "id": 12,
                    "name": "overall_pitcher_feature_coverage",
                    "rule": (
                        "overall pitcher-contact feature coverage >= 0.50"
                    ),
                    "minimum_coverage": 0.50,
                },
                {
                    "id": 13,
                    "name": "monthly_pitcher_feature_coverage_floor",
                    "rule": (
                        "no eligible calendar month may have pitcher-contact "
                        "feature coverage < 0.40"
                    ),
                    "minimum_monthly_coverage": 0.40,
                },
            ],
            "failure_policy": {
                "rescue_tuning_on_same_holdout": False,
                "threshold_changes_on_same_holdout": False,
                "feature_window_changes_on_same_holdout": False,
                "alternate_pitcher_contact_candidate_on_same_holdout": False,
                "verdict_if_any_condition_fails": (
                    "B1_FAILS_PROSPECTIVE_FORMAL_GATE_FREEZE_NO_RESCUE_TUNING"
                ),
            },
            "pass_verdict": (
                "B1_PASSES_PROSPECTIVE_FORMAL_GATE_"
                "FORMAL_VALIDATED_CHALLENGER_PENDING_IMPLEMENTATION_PARITY_AUDIT"
            ),
        },
        "prospective_evaluation": {
            "open_once_only": True,
            "minimum_eligible_rows": 10000,
            "outcomes_hidden_until_trigger": True,
            "bootstrap_block": "calendar_date",
            "fixed_threshold_reporting": "diagnostic_only",
            "fixed_coverage_reporting": "formal_decision_layer",
        },
        "operational_abort_rules": {
            "note": (
                "Operational integrity failures may abort the holdout before "
                "opening outcomes. Performance may never be used for early abort."
            ),
            "abort_and_restart_with_new_freeze_if": [
                "frozen model/spec hash changes",
                "scoring implementation changes materially",
                "target construction changes",
                "required source-data semantics change",
                "duplicate-key rate exceeds 0.001",
                "scoring error rate exceeds 0.01",
            ],
            "performance_based_early_abort_allowed": False,
        },
        "implementation_parity_audit_required_after_formal_pass": True,
        "research_history": {
            "development_years": [2024, 2025],
            "development_status": (
                "B1_BROADLY_STABLE_DEVELOPMENT_WINNER_"
                "ELIGIBLE_TO_FREEZE_FOR_FUTURE_PROSPECTIVE_HOLDOUT"
            ),
            "2026_poisson_family_formal_gate": (
                "opp_pitcher_contact_allowed_rate_60d failed the 2026 "
                "Poisson-architecture formal gate because official-tier hit "
                "rate declined; that failure remains valid and is not rescued."
            ),
            "architecture_lab_status": (
                "B1 was the clear 2024-2025 development winner. "
                "This lock does not reinterpret or reuse 2026 outcomes."
            ),
        },
    }


def verify_existing():
    if not OUT_JSON.exists():
        print(f"VERIFY FAILED: missing lock JSON: {OUT_JSON}")
        return 1

    payload = load_json(OUT_JSON)
    actual = sha256_bytes(canonical_json_bytes(payload))

    expected = None
    if OUT_SHA.exists():
        expected = OUT_SHA.read_text(
            encoding="utf-8"
        ).strip().split()[0]

    print("HITS_B1_PROSPECTIVE_HOLDOUT_LOCK_A VERIFY")
    print("==========================================")
    print(f"lock_json={OUT_JSON}")
    print(f"actual_sha256={actual}")
    print(f"stored_sha256={expected}")

    if expected is None:
        print("verdict=VERIFY_FAILED_NO_STORED_SHA256")
        return 1

    if actual != expected:
        print("verdict=VERIFY_FAILED_HASH_MISMATCH")
        return 1

    print("verdict=LOCK_VERIFIED_IMMUTABLE")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        return verify_existing()

    if OUT_JSON.exists():
        print("EXISTING LOCK FOUND. REFUSING TO OVERWRITE.")
        print(f"lock_json={OUT_JSON}")
        print("Running verification instead...")
        return verify_existing()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    payload = build_lock_payload()
    lock_hash = sha256_bytes(canonical_json_bytes(payload))

    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    OUT_SHA.write_text(
        f"{lock_hash}  {OUT_JSON.name}\n",
        encoding="utf-8",
    )

    lines = [
        "HITS_B1_PROSPECTIVE_HOLDOUT_LOCK_A",
        "=" * 39,
        "",
        f"LOCK VERSION: {payload['lock_version']}",
        f"STATUS: {payload['status']}",
        f"FREEZE UTC: {payload['freeze_timestamp_utc']}",
        (
            "FREEZE AMERICA/NEW_YORK: "
            f"{payload['freeze_timestamp_america_new_york']}"
        ),
        (
            "HOLDOUT START GAME DATE: "
            f"{payload['holdout_start_game_date_america_new_york']}"
        ),
        "TRIGGER: 10,000 eligible batter-games",
        "",
        "FROZEN CHALLENGER",
        "-----------------",
        "B1 = direct binary XGBoost P(hit >= 1)",
        "+ opp_pitcher_contact_allowed_rate_60d",
        "Window: 60 days",
        "Minimum prior pitcher swings: 150",
        "Monte Carlo: not used",
        "Poisson mapping: not used",
        "",
        "BASE FEATURES",
        "-------------",
        *[f"- {feature}" for feature in BASE_FEATURES],
        "",
        "NO-PEEKING RULE",
        "---------------",
        (
            "Before 10,000 eligible rows, no outcome-conditioned performance "
            "analysis is allowed."
        ),
        "",
        "FORMAL DECISION LAYER",
        "---------------------",
        (
            "Fixed threshold P>=0.63 is diagnostic only because it is not "
            "architecture-neutral."
        ),
        (
            "Formal decision comparison uses fixed coverage where N equals "
            "A0's official-board size over the complete prospective holdout."
        ),
        "",
        "FORMAL GATE",
        "-----------",
    ]

    for condition in payload["formal_gate"]["conditions"]:
        lines.append(
            f"{condition['id']}. {condition['name']}: {condition['rule']}"
        )

    lines.extend(
        [
            "",
            "FAILURE POLICY",
            "--------------",
            "Any failed condition = overall fail.",
            "No rescue tuning on the same holdout.",
            "No threshold changes on the same holdout.",
            "No feature-window changes on the same holdout.",
            "No alternate pitcher-contact candidate rescue on the same holdout.",
            "",
            "PASS POLICY",
            "-----------",
            (
                "A formal pass creates a FORMAL_VALIDATED_CHALLENGER pending "
                "implementation parity audit. It does not automatically authorize "
                "production promotion."
            ),
            "",
            "LOCK FINGERPRINT",
            "----------------",
            lock_hash,
        ]
    )

    OUT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("HITS_B1_PROSPECTIVE_HOLDOUT_LOCK_A")
    print("==================================")
    print(f"status={payload['status']}")
    print(f"freeze_timestamp_utc={payload['freeze_timestamp_utc']}")
    print(
        "freeze_timestamp_america_new_york="
        f"{payload['freeze_timestamp_america_new_york']}"
    )
    print(
        "holdout_start_game_date_america_new_york="
        f"{payload['holdout_start_game_date_america_new_york']}"
    )
    print("trigger_eligible_batter_games=10000")
    print("challenger=B1_direct_binary_plus_pitcher_contact")
    print("feature=opp_pitcher_contact_allowed_rate_60d")
    print("window_days=60")
    print("minimum_prior_pitcher_swings=150")
    print("production_promotion_authorized=False")
    print("formal_gate_completed=False")
    print("outcomes_hidden_until_trigger=True")
    print(f"sha256={lock_hash}")
    print("")
    print("OUTPUTS")
    print(OUT_JSON)
    print(OUT_TXT)
    print(OUT_SHA)
    print("")
    print("verdict=PROSPECTIVE_HOLDOUT_LOCK_CREATED_IMMUTABLY")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
