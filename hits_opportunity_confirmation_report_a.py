#!/usr/bin/env python3
"""
HITS_OPPORTUNITY_CONFIRMATION_REPORT_A

Purpose
-------
Direct confirmation audit for the two batter-hits opportunity scalars that
already passed the full promotion gate independently:

    1. season_pa_per_game
    2. recent15_pa_avg

This script does NOT retrain anything.
This script does NOT tune anything.
This script does NOT stack the candidates.
This script does NOT use 2026 as a clean holdout.

It only compares already-generated untouched rolling-forward predictions from
2022-2025 and answers:

    Which opportunity representation is more robust for production promotion?

Metrics
-------
- Brier
- Logloss
- AUC
- Top 5%
- Top 10%
- Official-tier hit rate
- Official-tier coverage
- Calibration deciles
- Expected calibration error (ECE)
- Maximum calibration gap
- Monthly stability
- Fold-by-fold stability
- Official-pick overlap / Jaccard
- Candidate-unique official picks
- Head-to-head disagreement cases
- Top-5% overlap and unique hit rates

Selection doctrine
------------------
No arbitrary point score.

A candidate can be named CLEAR CONFIRMATION WINNER only if one of these
predeclared conditions is met:

A) Probability-quality dominance:
   - Brier better by at least 0.00010
   - logloss no worse
   - AUC no worse by more than 0.0005
   - top 5%, top 10%, and official hit rate do not decline versus rival

OR

B) Decision-layer dominance:
   - top 5% better by at least 0.005
   - top 10% better by at least 0.003
   - official hit rate better by at least 0.003
   - Brier no worse by more than 0.00020
   - logloss no worse by more than 0.00050
   - AUC no worse by more than 0.001

Otherwise:
    NO_CLEAR_DOMINANCE

That outcome is valid and does not authorize stacking or rescue tuning.

Run
---
python -u hits_opportunity_confirmation_report_a.py 2>&1 | tee /data/hr_model/hits_opportunity_confirmation_report_a.log

Outputs
-------
/data/hr_model/hits_opportunity_confirmation_report_a_results.json
/data/hr_model/hits_opportunity_confirmation_report_a_report.txt

Paste back
----------
CONFIRMATION PREFLIGHT
DIRECT AGGREGATE COMPARISON
MONTHLY STABILITY
OFFICIAL PICK OVERLAP
TOP-5 OVERLAP
SELECTION READ
FINAL CONFIRMATION VERDICT
"""

import json
import math
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path("/data/hr_model/hits_rfv_lite_b")
BATCH_DIR = Path("/data/hr_model/hits_multi_scalar_batch_gate_b_work")
BATCH_RESULTS = Path("/data/hr_model/hits_multi_scalar_batch_gate_b_results.json")

OUT_JSON = Path("/data/hr_model/hits_opportunity_confirmation_report_a_results.json")
OUT_TXT = Path("/data/hr_model/hits_opportunity_confirmation_report_a_report.txt")

CANDIDATES = ["season_pa_per_game", "recent15_pa_avg"]
YEARS = [2022, 2023, 2024, 2025]
OFFICIAL_MIN = 0.630


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def auc_rank(y, p):
    n_pos = sum(y)
    n_neg = len(y) - n_pos

    if n_pos == 0 or n_neg == 0:
        return float("nan")

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
            yi * math.log(min(max(pi, 1e-8), 1.0 - 1e-8))
            + (1 - yi)
            * math.log(1.0 - min(max(pi, 1e-8), 1.0 - 1e-8))
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

    official_idx = [i for i, pi in enumerate(p) if pi >= OFFICIAL_MIN]

    out["official_rows"] = len(official_idx)
    out["official_hits"] = sum(y[i] for i in official_idx)
    out["official_hit_rate"] = (
        sum(y[i] for i in official_idx) / len(official_idx)
        if official_idx else None
    )
    out["official_mean_prob"] = (
        sum(p[i] for i in official_idx) / len(official_idx)
        if official_idx else None
    )

    return out


def calibration_deciles(y, p):
    order = sorted(range(len(y)), key=lambda i: p[i])
    groups = []

    n = len(order)

    for decile in range(10):
        start = decile * n // 10
        end = (decile + 1) * n // 10
        idx = order[start:end]

        if not idx:
            continue

        actual = sum(y[i] for i in idx) / len(idx)
        pred = sum(p[i] for i in idx) / len(idx)
        gap = actual - pred

        groups.append(
            {
                "decile": decile + 1,
                "rows": len(idx),
                "actual_hit_rate": actual,
                "mean_probability": pred,
                "gap_actual_minus_pred": gap,
            }
        )

    ece = sum(
        (d["rows"] / len(y)) * abs(d["gap_actual_minus_pred"])
        for d in groups
    )

    max_gap = max(
        abs(d["gap_actual_minus_pred"])
        for d in groups
    )

    return {
        "deciles": groups,
        "ece": ece,
        "max_abs_gap": max_gap,
    }


def hit_rate_for_indices(y, idx):
    idx = list(idx)

    if not idx:
        return {
            "rows": 0,
            "hits": 0,
            "hit_rate": None,
        }

    hits = sum(y[i] for i in idx)

    return {
        "rows": len(idx),
        "hits": hits,
        "hit_rate": hits / len(idx),
    }


def compare_sets(y, set_a, set_b):
    inter = set_a & set_b
    union = set_a | set_b
    only_a = set_a - set_b
    only_b = set_b - set_a

    return {
        "a_rows": len(set_a),
        "b_rows": len(set_b),
        "intersection_rows": len(inter),
        "union_rows": len(union),
        "jaccard": len(inter) / len(union) if union else None,
        "intersection": hit_rate_for_indices(y, inter),
        "only_a": hit_rate_for_indices(y, only_a),
        "only_b": hit_rate_for_indices(y, only_b),
    }


def month_key(date_text):
    return str(date_text)[:7]


def build_monthly(y, dates, candidate_probs):
    months = sorted(set(month_key(d) for d in dates))
    out = {}

    for month in months:
        idx = [
            i
            for i, d in enumerate(dates)
            if month_key(d) == month
        ]

        if not idx:
            continue

        month_y = [y[i] for i in idx]
        month_result = {
            "rows": len(idx),
            "actual_hit_rate": sum(month_y) / len(month_y),
            "candidates": {},
        }

        for candidate, probs in candidate_probs.items():
            month_p = [probs[i] for i in idx]
            m = metric_block(month_y, month_p)

            month_result["candidates"][candidate] = {
                "brier": m["brier"],
                "logloss": m["logloss"],
                "auc": m["auc"],
                "top5_actual": m["top5_actual"],
                "top10_actual": m["top10_actual"],
                "official_rows": m["official_rows"],
                "official_hit_rate": m["official_hit_rate"],
            }

        out[month] = month_result

    return out


def count_month_wins(monthly, metric, lower_is_better=False):
    counts = {candidate: 0 for candidate in CANDIDATES}
    ties = 0

    for month, row in monthly.items():
        vals = {
            c: row["candidates"][c][metric]
            for c in CANDIDATES
        }

        a = vals[CANDIDATES[0]]
        b = vals[CANDIDATES[1]]

        if a is None or b is None:
            continue

        if abs(a - b) < 1e-12:
            ties += 1
            continue

        if lower_is_better:
            winner = CANDIDATES[0] if a < b else CANDIDATES[1]
        else:
            winner = CANDIDATES[0] if a > b else CANDIDATES[1]

        counts[winner] += 1

    return {
        "wins": counts,
        "ties": ties,
    }


def selection_read(metrics):
    a = CANDIDATES[0]
    b = CANDIDATES[1]

    def direct(candidate, rival):
        c = metrics[candidate]
        r = metrics[rival]

        deltas = {
            "brier": c["brier"] - r["brier"],
            "logloss": c["logloss"] - r["logloss"],
            "auc": c["auc"] - r["auc"],
            "top5": c["top5_actual"] - r["top5_actual"],
            "top10": c["top10_actual"] - r["top10_actual"],
            "official_hit_rate": (
                c["official_hit_rate"] - r["official_hit_rate"]
            ),
        }

        probability_quality_dominance = (
            deltas["brier"] <= -0.00010
            and deltas["logloss"] <= 0.0
            and deltas["auc"] >= -0.0005
            and deltas["top5"] >= 0.0
            and deltas["top10"] >= 0.0
            and deltas["official_hit_rate"] >= 0.0
        )

        decision_layer_dominance = (
            deltas["top5"] >= 0.005
            and deltas["top10"] >= 0.003
            and deltas["official_hit_rate"] >= 0.003
            and deltas["brier"] <= 0.00020
            and deltas["logloss"] <= 0.00050
            and deltas["auc"] >= -0.001
        )

        return {
            "candidate": candidate,
            "rival": rival,
            "deltas": deltas,
            "probability_quality_dominance": probability_quality_dominance,
            "decision_layer_dominance": decision_layer_dominance,
            "clear_winner": (
                probability_quality_dominance
                or decision_layer_dominance
            ),
        }

    a_read = direct(a, b)
    b_read = direct(b, a)

    winners = [
        row["candidate"]
        for row in (a_read, b_read)
        if row["clear_winner"]
    ]

    if len(winners) == 1:
        verdict = f"CLEAR_CONFIRMATION_WINNER_{winners[0].upper()}"
    else:
        verdict = "NO_CLEAR_DOMINANCE"

    return {
        "candidate_reads": {
            a: a_read,
            b: b_read,
        },
        "verdict": verdict,
    }


def main():
    print("HITS_OPPORTUNITY_CONFIRMATION_REPORT_A", flush=True)
    print("======================================", flush=True)

    print("\nCONFIRMATION PREFLIGHT", flush=True)
    print("----------------------", flush=True)

    if not BATCH_RESULTS.exists():
        raise RuntimeError(
            f"Missing batch results: {BATCH_RESULTS}"
        )

    batch = load_json(BATCH_RESULTS)

    for candidate in CANDIDATES:
        gate = batch["results"][candidate]["gate"]

        if not gate["overall_pass"]:
            raise RuntimeError(
                f"{candidate} did not pass original promotion gate."
            )

        print(
            f"{candidate}: original_gate_pass=True",
            flush=True,
        )

    all_y = []
    all_dates = []
    pooled_probs = {candidate: [] for candidate in CANDIDATES}
    fold_results = {}

    global_offset = 0

    for year in YEARS:
        baseline_path = BASE_DIR / f"scores_{year}.json"
        baseline = load_json(baseline_path)

        y = baseline["y_binary"]
        dates = baseline["dates"]

        all_y.extend(y)
        all_dates.extend(dates)

        fold_results[str(year)] = {
            "rows": len(y),
            "candidates": {},
        }

        for candidate in CANDIDATES:
            score_path = BATCH_DIR / f"{candidate}_scores_{year}.json"

            if not score_path.exists():
                raise RuntimeError(
                    f"Missing candidate scores: {score_path}"
                )

            scored = load_json(score_path)

            if scored["y_binary"] != y:
                raise RuntimeError(
                    f"{candidate} {year}: actual-label alignment failed."
                )

            probs = scored["probs"]
            pooled_probs[candidate].extend(probs)

            fold_results[str(year)]["candidates"][candidate] = (
                metric_block(y, probs)
            )

        global_offset += len(y)

        print(
            f"{year}: rows={len(y):,} alignment=PASS",
            flush=True,
        )

    aggregate_metrics = {
        candidate: metric_block(all_y, pooled_probs[candidate])
        for candidate in CANDIDATES
    }

    calibration = {
        candidate: calibration_deciles(
            all_y,
            pooled_probs[candidate],
        )
        for candidate in CANDIDATES
    }

    monthly = build_monthly(
        all_y,
        all_dates,
        pooled_probs,
    )

    monthly_win_summary = {
        "brier": count_month_wins(
            monthly,
            "brier",
            lower_is_better=True,
        ),
        "logloss": count_month_wins(
            monthly,
            "logloss",
            lower_is_better=True,
        ),
        "auc": count_month_wins(
            monthly,
            "auc",
            lower_is_better=False,
        ),
        "top5_actual": count_month_wins(
            monthly,
            "top5_actual",
            lower_is_better=False,
        ),
        "top10_actual": count_month_wins(
            monthly,
            "top10_actual",
            lower_is_better=False,
        ),
        "official_hit_rate": count_month_wins(
            monthly,
            "official_hit_rate",
            lower_is_better=False,
        ),
    }

    a = CANDIDATES[0]
    b = CANDIDATES[1]

    official_sets = {
        candidate: {
            i
            for i, p in enumerate(pooled_probs[candidate])
            if p >= OFFICIAL_MIN
        }
        for candidate in CANDIDATES
    }

    official_overlap = compare_sets(
        all_y,
        official_sets[a],
        official_sets[b],
    )

    top5_sets = {}
    for candidate in CANDIDATES:
        probs = pooled_probs[candidate]
        k = max(1, math.ceil(len(probs) * 0.05))

        top5_sets[candidate] = set(
            sorted(
                range(len(probs)),
                key=lambda i: probs[i],
                reverse=True,
            )[:k]
        )

    top5_overlap = compare_sets(
        all_y,
        top5_sets[a],
        top5_sets[b],
    )

    selection = selection_read(aggregate_metrics)

    print("\nDIRECT AGGREGATE COMPARISON", flush=True)
    print("---------------------------", flush=True)

    for candidate in CANDIDATES:
        m = aggregate_metrics[candidate]
        c = calibration[candidate]

        print(
            f"{candidate}: "
            f"n={m['rows']:,} "
            f"brier={m['brier']:.8f} "
            f"logloss={m['logloss']:.8f} "
            f"auc={m['auc']:.6f} "
            f"top5={m['top5_actual']:.6f} "
            f"top10={m['top10_actual']:.6f} "
            f"official_n={m['official_rows']:,} "
            f"official_hit_rate={m['official_hit_rate']:.6f} "
            f"ece={c['ece']:.6f} "
            f"max_cal_gap={c['max_abs_gap']:.6f}",
            flush=True,
        )

    print("\nMONTHLY STABILITY", flush=True)
    print("-----------------", flush=True)

    for metric, row in monthly_win_summary.items():
        print(
            f"{metric}: "
            f"{a}_wins={row['wins'][a]} "
            f"{b}_wins={row['wins'][b]} "
            f"ties={row['ties']}",
            flush=True,
        )

    print("\nFOLD-BY-FOLD STABILITY", flush=True)
    print("----------------------", flush=True)

    for year in YEARS:
        row = fold_results[str(year)]["candidates"]

        print(
            f"{year}: "
            f"{a} brier={row[a]['brier']:.8f} "
            f"top5={row[a]['top5_actual']:.4f} "
            f"official={row[a]['official_hit_rate']:.4f} | "
            f"{b} brier={row[b]['brier']:.8f} "
            f"top5={row[b]['top5_actual']:.4f} "
            f"official={row[b]['official_hit_rate']:.4f}",
            flush=True,
        )

    print("\nOFFICIAL PICK OVERLAP", flush=True)
    print("---------------------", flush=True)

    print(
        f"{a}_official_rows={official_overlap['a_rows']:,}",
        flush=True,
    )
    print(
        f"{b}_official_rows={official_overlap['b_rows']:,}",
        flush=True,
    )
    print(
        f"intersection_rows={official_overlap['intersection_rows']:,}",
        flush=True,
    )
    print(
        f"jaccard={official_overlap['jaccard']:.6f}",
        flush=True,
    )
    print(
        f"intersection_hit_rate="
        f"{official_overlap['intersection']['hit_rate']}",
        flush=True,
    )
    print(
        f"{a}_only_rows={official_overlap['only_a']['rows']:,} "
        f"hit_rate={official_overlap['only_a']['hit_rate']}",
        flush=True,
    )
    print(
        f"{b}_only_rows={official_overlap['only_b']['rows']:,} "
        f"hit_rate={official_overlap['only_b']['hit_rate']}",
        flush=True,
    )

    print("\nTOP-5 OVERLAP", flush=True)
    print("-------------", flush=True)

    print(
        f"intersection_rows={top5_overlap['intersection_rows']:,}",
        flush=True,
    )
    print(
        f"jaccard={top5_overlap['jaccard']:.6f}",
        flush=True,
    )
    print(
        f"intersection_hit_rate={top5_overlap['intersection']['hit_rate']}",
        flush=True,
    )
    print(
        f"{a}_only_rows={top5_overlap['only_a']['rows']:,} "
        f"hit_rate={top5_overlap['only_a']['hit_rate']}",
        flush=True,
    )
    print(
        f"{b}_only_rows={top5_overlap['only_b']['rows']:,} "
        f"hit_rate={top5_overlap['only_b']['hit_rate']}",
        flush=True,
    )

    print("\nSELECTION READ", flush=True)
    print("--------------", flush=True)

    for candidate in CANDIDATES:
        read = selection["candidate_reads"][candidate]
        d = read["deltas"]

        print(
            f"{candidate}: "
            f"vs={read['rival']} "
            f"brier_delta={d['brier']:+.8f} "
            f"logloss_delta={d['logloss']:+.8f} "
            f"auc_delta={d['auc']:+.6f} "
            f"top5_delta={d['top5']:+.6f} "
            f"top10_delta={d['top10']:+.6f} "
            f"official_hit_rate_delta={d['official_hit_rate']:+.6f} "
            f"probability_quality_dominance="
            f"{read['probability_quality_dominance']} "
            f"decision_layer_dominance="
            f"{read['decision_layer_dominance']}",
            flush=True,
        )

    print("\nFINAL CONFIRMATION VERDICT", flush=True)
    print("--------------------------", flush=True)

    print(f"verdict={selection['verdict']}", flush=True)

    if selection["verdict"].startswith("CLEAR_CONFIRMATION_WINNER_"):
        winner = selection["verdict"].replace(
            "CLEAR_CONFIRMATION_WINNER_",
            "",
        ).lower()

        print(
            f"next_step=LOCK_{winner.upper()}_AS_HITS_OPPORTUNITY_PROMOTION_"
            "CANDIDATE_THEN_RUN_FINAL_PRODUCTION_INTEGRATION_AUDIT",
            flush=True,
        )
    else:
        print(
            "next_step=NO_STACKING_AND_NO_RESCUE_TUNING; "
            "retain both as independently passing candidates and decide whether "
            "to prefer the Brier-leading representation or require genuinely "
            "new forward data for separation.",
            flush=True,
        )

    payload = {
        "script": "HITS_OPPORTUNITY_CONFIRMATION_REPORT_A",
        "candidates": CANDIDATES,
        "years": YEARS,
        "2026_clean_holdout": "NO_BURNED",
        "retraining": "NONE",
        "stacking": "NONE",
        "rescue_tuning": "FORBIDDEN",
        "aggregate_metrics": aggregate_metrics,
        "calibration": calibration,
        "monthly": monthly,
        "monthly_win_summary": monthly_win_summary,
        "fold_results": fold_results,
        "official_overlap": official_overlap,
        "top5_overlap": top5_overlap,
        "selection": selection,
    }

    OUT_JSON.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    lines = [
        "HITS_OPPORTUNITY_CONFIRMATION_REPORT_A",
        "=" * 38,
        "",
        "No retraining. No stacking. No rescue tuning.",
        "2026 remains burned as a clean historical holdout.",
        "",
        "DIRECT AGGREGATE COMPARISON",
        "---------------------------",
    ]

    for candidate in CANDIDATES:
        lines.append("")
        lines.append(candidate)
        lines.append("~" * len(candidate))
        lines.append(
            json.dumps(
                {
                    "metrics": aggregate_metrics[candidate],
                    "calibration": {
                        "ece": calibration[candidate]["ece"],
                        "max_abs_gap": calibration[candidate]["max_abs_gap"],
                    },
                },
                indent=2,
            )
        )

    lines.extend(
        [
            "",
            "MONTHLY WIN SUMMARY",
            "-------------------",
            json.dumps(monthly_win_summary, indent=2),
            "",
            "OFFICIAL PICK OVERLAP",
            "---------------------",
            json.dumps(official_overlap, indent=2),
            "",
            "TOP-5 OVERLAP",
            "-------------",
            json.dumps(top5_overlap, indent=2),
            "",
            "SELECTION READ",
            "--------------",
            json.dumps(selection, indent=2),
            "",
            f"FINAL VERDICT: {selection['verdict']}",
        ]
    )

    OUT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\nOUTPUTS", flush=True)
    print(OUT_JSON, flush=True)
    print(OUT_TXT, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
