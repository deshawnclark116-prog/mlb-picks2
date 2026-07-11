#!/usr/bin/env python3
"""
HITS_B1_STABILITY_CONFIRMATION_REPORT_A

Purpose
-------
Development-only stability confirmation for the current architecture-lab winner:

    B1 = direct binary batter-hit model
         + frozen opp_pitcher_contact_allowed_rate_60d

This report answers:

1. Is B1's aggregate advantage over B0 stable fold by fold?
2. Is B1's advantage over the old Poisson architecture A0 broad or concentrated?
3. Does B1 remain competitive month by month?
4. Does the frozen pitcher-contact feature improve equal-volume selection?
5. Is board expansion at the fixed 0.63 threshold merely a probability-scale issue?
6. Is feature coverage stable enough to justify freezing B1 for a future prospective holdout?

Important doctrine status
-------------------------
- 2024-2025 DEVELOPMENT DATA ONLY.
- 2026 is not touched.
- This script does not retrain anything.
- This script cannot promote B1 to production.
- No threshold rescue.
- No feature-window tuning.
- No stacking.
- No 2026 reuse.

Inputs
------
/data/hr_model/hits_probability_architecture_lab_v1_results.json

and completed fold files:

/data/hr_model/hits_probability_architecture_lab_v1_work/F1/fold_results.json
...
/data/hr_model/hits_probability_architecture_lab_v1_work/F5/fold_results.json

Outputs
-------
/data/hr_model/hits_b1_stability_confirmation_report_a_results.json
/data/hr_model/hits_b1_stability_confirmation_report_a_report.txt

Run
---
python -u hits_b1_stability_confirmation_report_a.py 2>&1 | tee /data/hr_model/hits_b1_stability_confirmation_report_a.log

Paste back
----------
CONFIRMATION PREFLIGHT
FOLD-BY-FOLD B1 VS B0
FOLD-BY-FOLD B1 VS A0
MONTH-BY-MONTH B1 VS B0
MONTH-BY-MONTH B1 VS A0
EQUAL-VOLUME CONFIRMATION
BOARD-BEHAVIOR CONFIRMATION
FEATURE-COVERAGE STABILITY
DEVELOPMENT STABILITY READ
FINAL DEVELOPMENT STATUS
"""

import json
import math
from collections import defaultdict
from pathlib import Path


LAB_JSON = Path("/data/hr_model/hits_probability_architecture_lab_v1_results.json")
WORK_DIR = Path("/data/hr_model/hits_probability_architecture_lab_v1_work")

OUT_JSON = Path("/data/hr_model/hits_b1_stability_confirmation_report_a_results.json")
OUT_TXT = Path("/data/hr_model/hits_b1_stability_confirmation_report_a_report.txt")

CELLS = ["A0", "A1", "B0", "B1", "C0", "C1"]
FOLDS = ["F1", "F2", "F3", "F4", "F5"]
OFFICIAL_MIN = 0.63


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clip(value, lo, hi):
    return max(lo, min(hi, value))


def auc_rank(y, p):
    n_pos = sum(y)
    n_neg = len(y) - n_pos

    if n_pos == 0 or n_neg == 0:
        return None

    pairs = sorted(enumerate(p), key=lambda item: item[1])
    ranks = [0.0] * len(y)

    i = 0

    while i < len(pairs):
        j = i + 1

        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1

        avg_rank = ((i + 1) + j) / 2.0

        for k in range(i, j):
            ranks[pairs[k][0]] = avg_rank

        i = j

    sum_pos_ranks = sum(
        ranks[i]
        for i, label in enumerate(y)
        if label == 1
    )

    return (
        sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    ) / (n_pos * n_neg)


def metric_block(y, p):
    n = len(y)

    brier = sum(
        (pi - yi) ** 2
        for yi, pi in zip(y, p)
    ) / n

    logloss = sum(
        -(
            yi * math.log(clip(pi, 1e-8, 1.0 - 1e-8))
            + (1 - yi)
            * math.log(1.0 - clip(pi, 1e-8, 1.0 - 1e-8))
        )
        for yi, pi in zip(y, p)
    ) / n

    order = sorted(range(n), key=lambda i: p[i], reverse=True)

    out = {
        "rows": n,
        "actual_hit_rate": sum(y) / n,
        "mean_probability": sum(p) / n,
        "brier": brier,
        "logloss": logloss,
        "auc": auc_rank(y, p),
    }

    for fraction, name in ((0.05, "top5"), (0.10, "top10")):
        k = max(1, math.ceil(n * fraction))
        idx = order[:k]

        out[f"{name}_rows"] = k
        out[f"{name}_hits"] = sum(y[i] for i in idx)
        out[f"{name}_actual"] = sum(y[i] for i in idx) / k
        out[f"{name}_mean_prob"] = sum(p[i] for i in idx) / k

    official_idx = [
        i
        for i, pi in enumerate(p)
        if pi >= OFFICIAL_MIN
    ]

    out["official_rows"] = len(official_idx)
    out["official_hits"] = sum(y[i] for i in official_idx)
    out["official_hit_rate"] = (
        sum(y[i] for i in official_idx) / len(official_idx)
        if official_idx
        else None
    )
    out["official_mean_prob"] = (
        sum(p[i] for i in official_idx) / len(official_idx)
        if official_idx
        else None
    )

    return out


def delta_metrics(challenger, baseline):
    return {
        "brier": challenger["brier"] - baseline["brier"],
        "logloss": challenger["logloss"] - baseline["logloss"],
        "auc": (
            challenger["auc"] - baseline["auc"]
            if (
                challenger["auc"] is not None
                and baseline["auc"] is not None
            )
            else None
        ),
        "top5": challenger["top5_actual"] - baseline["top5_actual"],
        "top10": challenger["top10_actual"] - baseline["top10_actual"],
        "official_hit_rate": (
            challenger["official_hit_rate"] - baseline["official_hit_rate"]
            if (
                challenger["official_hit_rate"] is not None
                and baseline["official_hit_rate"] is not None
            )
            else None
        ),
        "official_rows": challenger["official_rows"] - baseline["official_rows"],
    }


def fixed_coverage_metrics(y, p, n):
    if n <= 0:
        return {
            "rows": 0,
            "hits": 0,
            "hit_rate": None,
            "mean_probability": None,
        }

    order = sorted(
        range(len(y)),
        key=lambda i: p[i],
        reverse=True,
    )

    idx = order[:n]

    return {
        "rows": len(idx),
        "hits": sum(y[i] for i in idx),
        "hit_rate": sum(y[i] for i in idx) / len(idx),
        "mean_probability": sum(p[i] for i in idx) / len(idx),
    }


def format_delta(value, digits=8):
    if value is None:
        return "None"
    return f"{value:+.{digits}f}"


def fold_comparison(fold_result):
    y = fold_result["y"]
    probabilities = fold_result["probabilities"]

    metrics = {
        cell: metric_block(y, probabilities[cell])
        for cell in ("A0", "B0", "B1")
    }

    fixed_n = int(
        fold_result["fixed_coverage_n_from_A0_official"]
    )

    fixed = {
        cell: fixed_coverage_metrics(
            y,
            probabilities[cell],
            fixed_n,
        )
        for cell in ("A0", "B0", "B1")
    }

    return {
        "fold": fold_result["fold"]["name"],
        "eval_start": fold_result["fold"]["eval_start"],
        "eval_end": fold_result["fold"]["eval_end"],
        "rows": len(y),
        "candidate_eval_coverage": fold_result["candidate_eval_coverage"],
        "metrics": metrics,
        "fixed_coverage_n": fixed_n,
        "fixed_coverage": fixed,
        "B1_vs_B0": delta_metrics(metrics["B1"], metrics["B0"]),
        "B1_vs_A0": delta_metrics(metrics["B1"], metrics["A0"]),
        "B1_vs_B0_fixed_coverage_hit_rate": (
            fixed["B1"]["hit_rate"] - fixed["B0"]["hit_rate"]
            if (
                fixed["B1"]["hit_rate"] is not None
                and fixed["B0"]["hit_rate"] is not None
            )
            else None
        ),
        "B1_vs_A0_fixed_coverage_hit_rate": (
            fixed["B1"]["hit_rate"] - fixed["A0"]["hit_rate"]
            if (
                fixed["B1"]["hit_rate"] is not None
                and fixed["A0"]["hit_rate"] is not None
            )
            else None
        ),
    }


def build_monthly_from_folds(fold_results):
    month_rows = defaultdict(
        lambda: {
            "y": [],
            "A0": [],
            "B0": [],
            "B1": [],
        }
    )

    for fold_result in fold_results:
        y = fold_result["y"]
        dates = fold_result["dates"]
        probs = fold_result["probabilities"]

        for i, date_text in enumerate(dates):
            month = str(date_text)[:7]
            month_rows[month]["y"].append(y[i])

            for cell in ("A0", "B0", "B1"):
                month_rows[month][cell].append(
                    probs[cell][i]
                )

    out = {}

    for month, row in sorted(month_rows.items()):
        metrics = {
            cell: metric_block(
                row["y"],
                row[cell],
            )
            for cell in ("A0", "B0", "B1")
        }

        a0_official_n = metrics["A0"]["official_rows"]

        fixed = {
            cell: fixed_coverage_metrics(
                row["y"],
                row[cell],
                a0_official_n,
            )
            for cell in ("A0", "B0", "B1")
        }

        out[month] = {
            "rows": len(row["y"]),
            "metrics": metrics,
            "fixed_coverage_n": a0_official_n,
            "fixed_coverage": fixed,
            "B1_vs_B0": delta_metrics(
                metrics["B1"],
                metrics["B0"],
            ),
            "B1_vs_A0": delta_metrics(
                metrics["B1"],
                metrics["A0"],
            ),
            "B1_vs_B0_fixed_coverage_hit_rate": (
                fixed["B1"]["hit_rate"] - fixed["B0"]["hit_rate"]
                if (
                    fixed["B1"]["hit_rate"] is not None
                    and fixed["B0"]["hit_rate"] is not None
                )
                else None
            ),
            "B1_vs_A0_fixed_coverage_hit_rate": (
                fixed["B1"]["hit_rate"] - fixed["A0"]["hit_rate"]
                if (
                    fixed["B1"]["hit_rate"] is not None
                    and fixed["A0"]["hit_rate"] is not None
                )
                else None
            ),
        }

    return out


def count_wins(rows, comparison_key, metric, lower_is_better=False):
    wins = 0
    ties = 0
    losses = 0

    for row in rows:
        value = row[comparison_key][metric]

        if value is None:
            continue

        if lower_is_better:
            # For deltas, lower than zero = challenger wins.
            if value < -1e-12:
                wins += 1
            elif value > 1e-12:
                losses += 1
            else:
                ties += 1
        else:
            if value > 1e-12:
                wins += 1
            elif value < -1e-12:
                losses += 1
            else:
                ties += 1

    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def stability_summary(fold_comparisons, monthly):
    fold_rows = list(fold_comparisons)
    month_rows = list(monthly.values())

    summary = {}

    for comparison in ("B1_vs_B0", "B1_vs_A0"):
        summary[comparison] = {
            "folds": {
                "brier": count_wins(
                    fold_rows,
                    comparison,
                    "brier",
                    lower_is_better=True,
                ),
                "logloss": count_wins(
                    fold_rows,
                    comparison,
                    "logloss",
                    lower_is_better=True,
                ),
                "auc": count_wins(
                    fold_rows,
                    comparison,
                    "auc",
                    lower_is_better=False,
                ),
                "top5": count_wins(
                    fold_rows,
                    comparison,
                    "top5",
                    lower_is_better=False,
                ),
                "top10": count_wins(
                    fold_rows,
                    comparison,
                    "top10",
                    lower_is_better=False,
                ),
                "official_hit_rate": count_wins(
                    fold_rows,
                    comparison,
                    "official_hit_rate",
                    lower_is_better=False,
                ),
            },
            "months": {
                "brier": count_wins(
                    month_rows,
                    comparison,
                    "brier",
                    lower_is_better=True,
                ),
                "logloss": count_wins(
                    month_rows,
                    comparison,
                    "logloss",
                    lower_is_better=True,
                ),
                "auc": count_wins(
                    month_rows,
                    comparison,
                    "auc",
                    lower_is_better=False,
                ),
                "top5": count_wins(
                    month_rows,
                    comparison,
                    "top5",
                    lower_is_better=False,
                ),
                "top10": count_wins(
                    month_rows,
                    comparison,
                    "top10",
                    lower_is_better=False,
                ),
                "official_hit_rate": count_wins(
                    month_rows,
                    comparison,
                    "official_hit_rate",
                    lower_is_better=False,
                ),
            },
        }

    return summary


def development_stability_read(
    lab,
    fold_comparisons,
    monthly,
    stability,
):
    """
    Retrospective development diagnostic only.

    This is NOT a promotion gate.
    Because aggregate 2024-2025 results were already opened before this script,
    these rules cannot create formal evidence. They only classify breadth of
    already-known development performance.
    """

    fold_count = len(fold_comparisons)
    month_count = len(monthly)

    b1_vs_b0_fold_brier = stability["B1_vs_B0"]["folds"]["brier"]["wins"]
    b1_vs_b0_fold_logloss = stability["B1_vs_B0"]["folds"]["logloss"]["wins"]
    b1_vs_b0_fold_auc = stability["B1_vs_B0"]["folds"]["auc"]["wins"]
    b1_vs_b0_month_brier = stability["B1_vs_B0"]["months"]["brier"]["wins"]
    b1_vs_b0_month_logloss = stability["B1_vs_B0"]["months"]["logloss"]["wins"]

    aggregate_feature_effect = lab[
        "architecture_read"
    ]["feature_effect_within_architecture"]["B"]

    fixed_coverage_delta = aggregate_feature_effect[
        "fixed_coverage_hit_rate_delta"
    ]

    coverage_values = [
        row["candidate_eval_coverage"]
        for row in fold_comparisons
    ]

    coverage_min = min(coverage_values)
    coverage_max = max(coverage_values)

    criteria = {
        "B1_beats_B0_brier_in_at_least_4_of_5_folds": (
            b1_vs_b0_fold_brier >= 4
        ),
        "B1_beats_B0_logloss_in_at_least_4_of_5_folds": (
            b1_vs_b0_fold_logloss >= 4
        ),
        "B1_auc_nonfragile_at_fold_level": (
            b1_vs_b0_fold_auc >= 3
        ),
        "B1_beats_B0_brier_in_at_least_half_of_months": (
            b1_vs_b0_month_brier >= math.ceil(month_count / 2)
        ),
        "B1_beats_B0_logloss_in_at_least_half_of_months": (
            b1_vs_b0_month_logloss >= math.ceil(month_count / 2)
        ),
        "B1_improves_equal_volume_hit_rate_aggregate": (
            fixed_coverage_delta is not None
            and fixed_coverage_delta > 0.0
        ),
        "feature_coverage_not_collapsing_across_folds": (
            coverage_min >= 0.50
        ),
    }

    broadly_stable = all(criteria.values())

    if broadly_stable:
        status = (
            "B1_BROADLY_STABLE_DEVELOPMENT_WINNER_"
            "ELIGIBLE_TO_FREEZE_FOR_FUTURE_PROSPECTIVE_HOLDOUT"
        )
    else:
        status = (
            "B1_AGGREGATE_WINNER_BUT_STABILITY_NOT_BROAD_ENOUGH_"
            "FOR_FUTURE_HOLDOUT_FREEZE"
        )

    return {
        "status": status,
        "criteria": criteria,
        "fold_count": fold_count,
        "month_count": month_count,
        "candidate_coverage_min": coverage_min,
        "candidate_coverage_max": coverage_max,
        "aggregate_B_feature_effect": aggregate_feature_effect,
        "production_promotion_authorized": False,
        "formal_gate": False,
        "retrospective_development_diagnostic_only": True,
    }


def main():
    print("HITS_B1_STABILITY_CONFIRMATION_REPORT_A", flush=True)
    print("======================================", flush=True)

    print("\nCONFIRMATION PREFLIGHT", flush=True)
    print("----------------------", flush=True)
    print("status=DEVELOPMENT_CONFIRMATION_ONLY", flush=True)
    print("years=2024,2025", flush=True)
    print("touch_2026=False", flush=True)
    print("retraining=False", flush=True)
    print("threshold_rescue=FORBIDDEN", flush=True)
    print("feature_window_tuning=FORBIDDEN", flush=True)
    print("stacking=False", flush=True)
    print("production_promotion_authorized=False", flush=True)
    print(
        "candidate=B1_direct_binary_plus_"
        "opp_pitcher_contact_allowed_rate_60d",
        flush=True,
    )
    print("comparison_1=B1_vs_B0", flush=True)
    print("comparison_2=B1_vs_A0", flush=True)

    if not LAB_JSON.exists():
        raise RuntimeError(f"Missing lab results: {LAB_JSON}")

    lab = load_json(LAB_JSON)

    if lab.get("touch_2026") is not False:
        raise RuntimeError(
            "Lab JSON does not explicitly confirm touch_2026=False."
        )

    fold_results = []
    fold_comparisons = []

    for fold_name in FOLDS:
        path = WORK_DIR / fold_name / "fold_results.json"

        if not path.exists():
            raise RuntimeError(f"Missing completed fold: {path}")

        fold_result = load_json(path)
        fold_results.append(fold_result)
        fold_comparisons.append(
            fold_comparison(fold_result)
        )

    monthly = build_monthly_from_folds(fold_results)

    stability = stability_summary(
        fold_comparisons,
        monthly,
    )

    development_read = development_stability_read(
        lab,
        fold_comparisons,
        monthly,
        stability,
    )

    print("\nFOLD-BY-FOLD B1 VS B0", flush=True)
    print("---------------------", flush=True)

    for row in fold_comparisons:
        d = row["B1_vs_B0"]

        print(
            f"{row['fold']}: "
            f"n={row['rows']:,} "
            f"brier={format_delta(d['brier'])} "
            f"logloss={format_delta(d['logloss'])} "
            f"auc={format_delta(d['auc'], 6)} "
            f"top5={format_delta(d['top5'], 6)} "
            f"top10={format_delta(d['top10'], 6)} "
            f"official_hit_rate={format_delta(d['official_hit_rate'], 6)} "
            f"fixed_coverage_hit_rate="
            f"{format_delta(row['B1_vs_B0_fixed_coverage_hit_rate'], 6)}",
            flush=True,
        )

    print("\nFOLD-BY-FOLD B1 VS A0", flush=True)
    print("---------------------", flush=True)

    for row in fold_comparisons:
        d = row["B1_vs_A0"]

        print(
            f"{row['fold']}: "
            f"n={row['rows']:,} "
            f"brier={format_delta(d['brier'])} "
            f"logloss={format_delta(d['logloss'])} "
            f"auc={format_delta(d['auc'], 6)} "
            f"top5={format_delta(d['top5'], 6)} "
            f"top10={format_delta(d['top10'], 6)} "
            f"official_hit_rate={format_delta(d['official_hit_rate'], 6)} "
            f"fixed_coverage_hit_rate="
            f"{format_delta(row['B1_vs_A0_fixed_coverage_hit_rate'], 6)}",
            flush=True,
        )

    print("\nMONTH-BY-MONTH B1 VS B0", flush=True)
    print("------------------------", flush=True)

    for month, row in monthly.items():
        d = row["B1_vs_B0"]

        print(
            f"{month}: "
            f"n={row['rows']:,} "
            f"brier={format_delta(d['brier'])} "
            f"logloss={format_delta(d['logloss'])} "
            f"auc={format_delta(d['auc'], 6)} "
            f"top5={format_delta(d['top5'], 6)} "
            f"top10={format_delta(d['top10'], 6)} "
            f"official_hit_rate={format_delta(d['official_hit_rate'], 6)} "
            f"fixed_coverage_hit_rate="
            f"{format_delta(row['B1_vs_B0_fixed_coverage_hit_rate'], 6)}",
            flush=True,
        )

    print("\nMONTH-BY-MONTH B1 VS A0", flush=True)
    print("------------------------", flush=True)

    for month, row in monthly.items():
        d = row["B1_vs_A0"]

        print(
            f"{month}: "
            f"n={row['rows']:,} "
            f"brier={format_delta(d['brier'])} "
            f"logloss={format_delta(d['logloss'])} "
            f"auc={format_delta(d['auc'], 6)} "
            f"top5={format_delta(d['top5'], 6)} "
            f"top10={format_delta(d['top10'], 6)} "
            f"official_hit_rate={format_delta(d['official_hit_rate'], 6)} "
            f"fixed_coverage_hit_rate="
            f"{format_delta(row['B1_vs_A0_fixed_coverage_hit_rate'], 6)}",
            flush=True,
        )

    print("\nEQUAL-VOLUME CONFIRMATION", flush=True)
    print("-------------------------", flush=True)

    aggregate_fixed = lab["fixed_coverage"]

    for cell in ("A0", "B0", "B1"):
        row = aggregate_fixed[cell]

        print(
            f"{cell}: "
            f"rows={row['rows']:,} "
            f"hits={row['hits']:,} "
            f"hit_rate={row['hit_rate']:.6f} "
            f"mean_probability={row['mean_probability']:.6f}",
            flush=True,
        )

    print(
        "B1_minus_B0_fixed_coverage_hit_rate="
        f"{aggregate_fixed['B1']['hit_rate'] - aggregate_fixed['B0']['hit_rate']:+.6f}",
        flush=True,
    )

    print(
        "B1_minus_A0_fixed_coverage_hit_rate="
        f"{aggregate_fixed['B1']['hit_rate'] - aggregate_fixed['A0']['hit_rate']:+.6f}",
        flush=True,
    )

    print("\nBOARD-BEHAVIOR CONFIRMATION", flush=True)
    print("---------------------------", flush=True)

    metrics = lab["metrics"]

    for cell in ("A0", "B0", "B1"):
        row = metrics[cell]

        print(
            f"{cell}: "
            f"official_n={row['official_rows']:,} "
            f"official_hit_rate={row['official_hit_rate']:.6f} "
            f"mean_official_probability={row['official_mean_prob']:.6f} "
            f"inflation_vs_A0="
            f"{row['official_board_size_inflation_vs_A0']:+.4f}",
            flush=True,
        )

    print(
        "read=0.63_IS_NOT_ARCHITECTURE_NEUTRAL; "
        "equal-volume comparison is required for architecture selection.",
        flush=True,
    )

    print("\nFEATURE-COVERAGE STABILITY", flush=True)
    print("--------------------------", flush=True)

    for row in fold_comparisons:
        print(
            f"{row['fold']}: "
            f"eval_range={row['eval_start']}..{row['eval_end']} "
            f"candidate_eval_coverage={row['candidate_eval_coverage']:.4f}",
            flush=True,
        )

    print(
        f"coverage_min={development_read['candidate_coverage_min']:.4f}",
        flush=True,
    )
    print(
        f"coverage_max={development_read['candidate_coverage_max']:.4f}",
        flush=True,
    )

    print("\nDEVELOPMENT STABILITY READ", flush=True)
    print("--------------------------", flush=True)

    for comparison, section in stability.items():
        print(f"\n[{comparison}]", flush=True)

        for level, metrics_section in section.items():
            print(f"{level}:", flush=True)

            for metric, record in metrics_section.items():
                print(
                    f"  {metric}: "
                    f"wins={record['wins']} "
                    f"ties={record['ties']} "
                    f"losses={record['losses']}",
                    flush=True,
                )

    print("\nRetrospective breadth criteria:", flush=True)

    for key, value in development_read["criteria"].items():
        print(f"{key}: {value}", flush=True)

    print(
        f"development_status={development_read['status']}",
        flush=True,
    )

    print("\nFINAL DEVELOPMENT STATUS", flush=True)
    print("------------------------", flush=True)
    print(
        f"status={development_read['status']}",
        flush=True,
    )
    print("production_promotion_authorized=False", flush=True)
    print("formal_gate=False", flush=True)
    print("2026_touched=False", flush=True)
    print(
        "next_step="
        "IF_BROADLY_STABLE_FREEZE_B1_ARCHITECTURE_FEATURE_LINEAGE_"
        "AND_PREDECLARE_PROSPECTIVE_HOLDOUT_START_DATE_TRIGGER_AND_GATE; "
        "OTHERWISE_DO_NOT_FREEZE_B1.",
        flush=True,
    )

    payload = {
        "script": "HITS_B1_STABILITY_CONFIRMATION_REPORT_A",
        "status": "DEVELOPMENT_CONFIRMATION_ONLY",
        "years": [2024, 2025],
        "touch_2026": False,
        "retraining": False,
        "candidate": (
            "B1_direct_binary_plus_"
            "opp_pitcher_contact_allowed_rate_60d"
        ),
        "threshold_rescue": "FORBIDDEN",
        "feature_window_tuning": "FORBIDDEN",
        "stacking": False,
        "production_promotion_authorized": False,
        "fold_comparisons": fold_comparisons,
        "monthly_comparisons": monthly,
        "stability_summary": stability,
        "development_stability_read": development_read,
        "aggregate_equal_volume": {
            cell: aggregate_fixed[cell]
            for cell in ("A0", "B0", "B1")
        },
        "aggregate_board_behavior": {
            cell: {
                "official_rows": metrics[cell]["official_rows"],
                "official_hit_rate": metrics[cell]["official_hit_rate"],
                "official_mean_prob": metrics[cell]["official_mean_prob"],
                "official_board_size_inflation_vs_A0": (
                    metrics[cell]["official_board_size_inflation_vs_A0"]
                ),
            }
            for cell in ("A0", "B0", "B1")
        },
    }

    OUT_JSON.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "HITS_B1_STABILITY_CONFIRMATION_REPORT_A",
        "=" * 40,
        "",
        "2024-2025 DEVELOPMENT CONFIRMATION ONLY.",
        "2026 was not touched.",
        "No retraining.",
        "No production promotion authorized.",
        "",
        "DEVELOPMENT STABILITY READ",
        "--------------------------",
        json.dumps(
            development_read,
            indent=2,
        ),
        "",
        "STABILITY SUMMARY",
        "-----------------",
        json.dumps(
            stability,
            indent=2,
        ),
        "",
        "FINAL DEVELOPMENT STATUS",
        "------------------------",
        development_read["status"],
    ]

    OUT_TXT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nOUTPUTS", flush=True)
    print(OUT_JSON, flush=True)
    print(OUT_TXT, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
