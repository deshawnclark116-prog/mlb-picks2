#!/usr/bin/env python3
"""
HITS_PITCHER_CONTACT_INTERNAL_SCREEN_A

Doctrine-safe batch screen for the next structurally distinct batter-hits family:
opposing-pitcher contact vulnerability.

IMPORTANT STATUS
----------------
This is NOT a promotion gate and does NOT create a new clean holdout.

Why:
- The raw structural audit already inspected both 2024 and 2025.
- Therefore 2025 has already influenced candidate selection.
- Calling 2025 "untouched" now would violate the doctrine.

What this script does:
- Uses 2024 only to fit one lightweight residual/logit correction head per candidate.
- Evaluates each head on 2025.
- Tests all candidates independently in one run.
- No stacking.
- No rescue tuning.
- No 2026 clean-holdout claim.
- A surviving candidate is only:
    SURVIVES_INTERNAL_SCREEN_PENDING_NEW_FORWARD_DATA
  It is NOT promoted to production.

Candidates
----------
1. opp_pitcher_k_per_pa_60d
2. opp_pitcher_contact_allowed_rate_60d
3. opp_pitcher_hit_per_pa_60d
4. opp_pitcher_hard_hit_allowed_rate_60d

Architecture
------------
Locked baseline probability p0 is frozen.

For valid candidate rows only:

Calibration control:
    logit(p_control) = a + b * logit(p0)

Candidate head:
    logit(p_candidate) = a + b * logit(p0) + c * z(candidate)

The candidate must beat:
1. the untouched locked baseline on 2025, and
2. the calibration-only control on the exact same rows/full-board fallback logic.

Missing candidate rows fall back to the locked baseline probability.

Data
----
Baseline rolling-forward scores:
    /data/hr_model/hits_rfv_lite_b/scores_2024.json
    /data/hr_model/hits_rfv_lite_b/scores_2025.json

Candidate features:
    /data/hr_model/hits_raw_structural_audit_a_work/audit.sqlite
    table: audit_joined

Fixed internal screen
---------------------
A candidate survives only if ALL are true:

1. 2025 full-board Brier delta vs locked baseline <= -0.0005
2. 2025 full-board logloss does not worsen vs locked baseline
3. 2025 full-board AUC delta vs locked baseline >= -0.0005
4. 2025 full-board top-5% actual hit rate does not decline
5. 2025 full-board top-10% actual hit rate does not decline
6. 2025 official-tier actual hit rate does not decline
7. 2025 official-tier coverage remains >= 90% of baseline
8. Candidate beats calibration-only control by Brier <= -0.0001
9. Candidate logloss does not worsen vs calibration-only control
10. Candidate top-5% does not decline vs calibration-only control
11. Candidate top-10% does not decline vs calibration-only control
12. Learned candidate coefficient has the expected structural sign
13. Candidate improves/holds monthly Brier vs baseline in at least 4 months

Even a full pass means:
    SURVIVES_INTERNAL_SCREEN_PENDING_NEW_FORWARD_DATA

It does NOT mean:
    PROMOTED

Memory safety
-------------
- no pandas
- no sklearn
- no XGBoost retraining
- SQLite reads only
- small Python lists
- suitable for the 512 MB Render instance

Run
---
python -u hits_pitcher_contact_internal_screen_a.py 2>&1 | tee /data/hr_model/hits_pitcher_contact_internal_screen_a.log

Outputs
-------
/data/hr_model/hits_pitcher_contact_internal_screen_a_results.json
/data/hr_model/hits_pitcher_contact_internal_screen_a_report.txt

Paste back
----------
SCREEN PREFLIGHT
2024 TRAINED HEADS
2025 FULL-BOARD COMPARISON
MONTHLY STABILITY
STRICT INTERNAL SCREEN
FINAL SCREEN VERDICTS
"""

import json
import math
import sqlite3
from pathlib import Path

BASE_DIR = Path("/data/hr_model/hits_rfv_lite_b")
AUDIT_DB = Path("/data/hr_model/hits_raw_structural_audit_a_work/audit.sqlite")

OUT_JSON = Path("/data/hr_model/hits_pitcher_contact_internal_screen_a_results.json")
OUT_TXT = Path("/data/hr_model/hits_pitcher_contact_internal_screen_a_report.txt")

OFFICIAL_MIN = 0.630

CANDIDATES = {
    "opp_pitcher_k_per_pa_60d": {
        "sample_col": "p_pa_60d",
        "min_sample": 75,
        "expected_sign": -1,
    },
    "opp_pitcher_contact_allowed_rate_60d": {
        "sample_col": "p_swings_60d",
        "min_sample": 150,
        "expected_sign": +1,
    },
    "opp_pitcher_hit_per_pa_60d": {
        "sample_col": "p_pa_60d",
        "min_sample": 75,
        "expected_sign": +1,
    },
    "opp_pitcher_hard_hit_allowed_rate_60d": {
        "sample_col": "p_bbe_60d",
        "min_sample": 50,
        "expected_sign": +1,
    },
}

RIDGE = 1e-3
MAX_ITER = 35
TOL = 1e-8


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clip(value, lo, hi):
    return max(lo, min(hi, value))


def sigmoid(x):
    x = clip(x, -35.0, 35.0)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p):
    p = clip(float(p), 1e-8, 1.0 - 1e-8)
    return math.log(p / (1.0 - p))


def solve_linear(matrix, vector):
    n = len(vector)
    a = [list(row) + [float(vector[i])] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))

        if abs(a[pivot][col]) < 1e-12:
            raise RuntimeError("Singular logistic Hessian.")

        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]

        pivot_value = a[col][col]

        for j in range(col, n + 1):
            a[col][j] /= pivot_value

        for row in range(n):
            if row == col:
                continue

            factor = a[row][col]

            if factor == 0:
                continue

            for j in range(col, n + 1):
                a[row][j] -= factor * a[col][j]

    return [a[i][n] for i in range(n)]


def fit_logistic(features, y, ridge=RIDGE, max_iter=MAX_ITER):
    if not features:
        raise RuntimeError("No training rows.")

    d = len(features[0])
    beta = [0.0] * d

    for iteration in range(max_iter):
        grad = [0.0] * d
        hess = [[0.0] * d for _ in range(d)]

        for x, yi in zip(features, y):
            eta = sum(beta[j] * x[j] for j in range(d))
            p = sigmoid(eta)
            residual = yi - p
            w = max(p * (1.0 - p), 1e-8)

            for j in range(d):
                grad[j] += x[j] * residual

                for k in range(j, d):
                    hess[j][k] += w * x[j] * x[k]

        for j in range(d):
            for k in range(j):
                hess[j][k] = hess[k][j]

        # Do not penalize intercept.
        for j in range(1, d):
            grad[j] -= ridge * beta[j]
            hess[j][j] += ridge

        step = solve_linear(hess, grad)

        for j in range(d):
            beta[j] += step[j]

        max_step = max(abs(s) for s in step)

        if max_step < TOL:
            return {
                "coefficients": beta,
                "iterations": iteration + 1,
                "converged": True,
            }

    return {
        "coefficients": beta,
        "iterations": max_iter,
        "converged": False,
    }


def predict_logistic(features, coefficients):
    return [
        sigmoid(sum(c * xj for c, xj in zip(coefficients, x)))
        for x in features
    ]


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
            yi * math.log(clip(pi, 1e-8, 1.0 - 1e-8))
            + (1 - yi) * math.log(
                1.0 - clip(pi, 1e-8, 1.0 - 1e-8)
            )
        )
        for yi, pi in zip(y, p)
    ) / n

    order = sorted(
        range(n),
        key=lambda i: p[i],
        reverse=True,
    )

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
        i for i, pi in enumerate(p)
        if pi >= OFFICIAL_MIN
    ]

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


def month_key(date_text):
    return str(date_text)[:7]


def monthly_brier(y, p, dates):
    groups = {}

    for i, date_text in enumerate(dates):
        key = month_key(date_text)
        groups.setdefault(key, []).append(i)

    out = {}

    for month, idx in sorted(groups.items()):
        out[month] = {
            "rows": len(idx),
            "brier": sum(
                (p[i] - y[i]) ** 2
                for i in idx
            ) / len(idx),
        }

    return out


def table_columns(conn, table):
    return {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})")
    }


def load_candidate_rows(conn, year, candidate, config, score_payload):
    columns = table_columns(conn, "audit_joined")

    needed = {
        "year",
        "row_id",
        candidate,
        config["sample_col"],
    }

    missing = needed - columns

    if missing:
        raise RuntimeError(
            f"Missing audit columns for {candidate}: {sorted(missing)}"
        )

    query = f"""
        SELECT
            row_id,
            {candidate},
            {config['sample_col']}
        FROM audit_joined
        WHERE year = ?
        ORDER BY row_id
    """

    raw = list(conn.execute(query, (year,)))

    y = score_payload["y_binary"]
    probs = score_payload["probs"]
    dates = score_payload["dates"]

    if len(raw) != len(y):
        raise RuntimeError(
            f"{candidate} {year}: audit/score row mismatch "
            f"{len(raw)} vs {len(y)}"
        )

    values = [None] * len(y)
    valid = [False] * len(y)

    for expected_idx, (row_id, value, sample_n) in enumerate(raw):
        if int(row_id) != expected_idx:
            raise RuntimeError(
                f"{candidate} {year}: row_id alignment failed "
                f"at {expected_idx}, got {row_id}"
            )

        if (
            value is not None
            and sample_n is not None
            and float(sample_n) >= float(config["min_sample"])
        ):
            values[expected_idx] = float(value)
            valid[expected_idx] = True

    return {
        "y": y,
        "baseline_probs": probs,
        "dates": dates,
        "values": values,
        "valid": valid,
    }


def standardize_from_train(train_values):
    valid_values = [v for v in train_values if v is not None]

    if len(valid_values) < 100:
        raise RuntimeError("Too few valid training values.")

    mean = sum(valid_values) / len(valid_values)
    variance = sum(
        (v - mean) ** 2
        for v in valid_values
    ) / len(valid_values)

    std = math.sqrt(variance)

    if std <= 1e-12:
        raise RuntimeError("Candidate standard deviation is zero.")

    return mean, std


def build_design(data, mean, std, include_candidate):
    features = []
    y = []
    indices = []

    for i, is_valid in enumerate(data["valid"]):
        if not is_valid:
            continue

        base_logit = clip(
            logit(data["baseline_probs"][i]),
            -12.0,
            12.0,
        )

        row = [1.0, base_logit]

        if include_candidate:
            z = clip(
                (data["values"][i] - mean) / std,
                -5.0,
                5.0,
            )
            row.append(z)

        features.append(row)
        y.append(int(data["y"][i]))
        indices.append(i)

    return features, y, indices


def full_board_with_fallback(
    baseline_probs,
    valid_indices,
    fitted_probs,
):
    out = list(baseline_probs)

    for idx, prob in zip(valid_indices, fitted_probs):
        out[idx] = float(prob)

    return out


def evaluate_candidate(conn, candidate, config, score_2024, score_2025):
    train = load_candidate_rows(
        conn,
        2024,
        candidate,
        config,
        score_2024,
    )

    test = load_candidate_rows(
        conn,
        2025,
        candidate,
        config,
        score_2025,
    )

    mean, std = standardize_from_train(
        [
            train["values"][i] if train["valid"][i] else None
            for i in range(len(train["values"]))
        ]
    )

    control_x_train, y_train, train_idx = build_design(
        train,
        mean,
        std,
        include_candidate=False,
    )

    candidate_x_train, y_train_2, train_idx_2 = build_design(
        train,
        mean,
        std,
        include_candidate=True,
    )

    if y_train != y_train_2 or train_idx != train_idx_2:
        raise RuntimeError(
            f"{candidate}: training design alignment failure."
        )

    control_fit = fit_logistic(
        control_x_train,
        y_train,
    )

    candidate_fit = fit_logistic(
        candidate_x_train,
        y_train,
    )

    control_x_test, y_test, test_idx = build_design(
        test,
        mean,
        std,
        include_candidate=False,
    )

    candidate_x_test, y_test_2, test_idx_2 = build_design(
        test,
        mean,
        std,
        include_candidate=True,
    )

    if y_test != y_test_2 or test_idx != test_idx_2:
        raise RuntimeError(
            f"{candidate}: test design alignment failure."
        )

    control_subset_probs = predict_logistic(
        control_x_test,
        control_fit["coefficients"],
    )

    candidate_subset_probs = predict_logistic(
        candidate_x_test,
        candidate_fit["coefficients"],
    )

    baseline_full = list(test["baseline_probs"])

    control_full = full_board_with_fallback(
        baseline_full,
        test_idx,
        control_subset_probs,
    )

    candidate_full = full_board_with_fallback(
        baseline_full,
        test_idx,
        candidate_subset_probs,
    )

    y_full = list(test["y"])
    dates = list(test["dates"])

    baseline_metrics = metric_block(
        y_full,
        baseline_full,
    )

    control_metrics = metric_block(
        y_full,
        control_full,
    )

    candidate_metrics = metric_block(
        y_full,
        candidate_full,
    )

    delta_vs_baseline = {
        "brier": (
            candidate_metrics["brier"]
            - baseline_metrics["brier"]
        ),
        "logloss": (
            candidate_metrics["logloss"]
            - baseline_metrics["logloss"]
        ),
        "auc": (
            candidate_metrics["auc"]
            - baseline_metrics["auc"]
        ),
        "top5": (
            candidate_metrics["top5_actual"]
            - baseline_metrics["top5_actual"]
        ),
        "top10": (
            candidate_metrics["top10_actual"]
            - baseline_metrics["top10_actual"]
        ),
        "official_hit_rate": (
            candidate_metrics["official_hit_rate"]
            - baseline_metrics["official_hit_rate"]
        ),
    }

    delta_vs_control = {
        "brier": (
            candidate_metrics["brier"]
            - control_metrics["brier"]
        ),
        "logloss": (
            candidate_metrics["logloss"]
            - control_metrics["logloss"]
        ),
        "auc": (
            candidate_metrics["auc"]
            - control_metrics["auc"]
        ),
        "top5": (
            candidate_metrics["top5_actual"]
            - control_metrics["top5_actual"]
        ),
        "top10": (
            candidate_metrics["top10_actual"]
            - control_metrics["top10_actual"]
        ),
        "official_hit_rate": (
            candidate_metrics["official_hit_rate"]
            - control_metrics["official_hit_rate"]
        ),
    }

    official_coverage_ratio = (
        candidate_metrics["official_rows"]
        / baseline_metrics["official_rows"]
        if baseline_metrics["official_rows"] > 0
        else None
    )

    baseline_monthly = monthly_brier(
        y_full,
        baseline_full,
        dates,
    )

    candidate_monthly = monthly_brier(
        y_full,
        candidate_full,
        dates,
    )

    monthly_rows = {}
    monthly_nonworse = 0

    for month in sorted(baseline_monthly):
        base = baseline_monthly[month]
        chal = candidate_monthly[month]

        delta = chal["brier"] - base["brier"]

        if delta <= 0:
            monthly_nonworse += 1

        monthly_rows[month] = {
            "rows": base["rows"],
            "baseline_brier": base["brier"],
            "candidate_brier": chal["brier"],
            "delta": delta,
        }

    candidate_coefficient = candidate_fit["coefficients"][-1]

    sign_pass = (
        candidate_coefficient * config["expected_sign"] > 0
    )

    gate = {
        "brier_vs_baseline_pass": (
            delta_vs_baseline["brier"] <= -0.0005
        ),
        "logloss_vs_baseline_pass": (
            delta_vs_baseline["logloss"] <= 0.0
        ),
        "auc_vs_baseline_pass": (
            delta_vs_baseline["auc"] >= -0.0005
        ),
        "top5_vs_baseline_pass": (
            delta_vs_baseline["top5"] >= 0.0
        ),
        "top10_vs_baseline_pass": (
            delta_vs_baseline["top10"] >= 0.0
        ),
        "official_hit_rate_vs_baseline_pass": (
            delta_vs_baseline["official_hit_rate"] >= 0.0
        ),
        "official_coverage_pass": (
            official_coverage_ratio is not None
            and official_coverage_ratio >= 0.90
        ),
        "brier_vs_calibration_control_pass": (
            delta_vs_control["brier"] <= -0.0001
        ),
        "logloss_vs_calibration_control_pass": (
            delta_vs_control["logloss"] <= 0.0
        ),
        "top5_vs_calibration_control_pass": (
            delta_vs_control["top5"] >= 0.0
        ),
        "top10_vs_calibration_control_pass": (
            delta_vs_control["top10"] >= 0.0
        ),
        "expected_coefficient_sign_pass": sign_pass,
        "monthly_brier_pass": monthly_nonworse >= 4,
    }

    gate["overall_internal_screen_pass"] = all(gate.values())

    verdict = (
        "SURVIVES_INTERNAL_SCREEN_PENDING_NEW_FORWARD_DATA"
        if gate["overall_internal_screen_pass"]
        else "FAILS_INTERNAL_SCREEN_FREEZE_NO_RESCUE_TUNING"
    )

    return {
        "candidate": candidate,
        "config": config,
        "train_2024": {
            "total_rows": len(train["y"]),
            "valid_rows": sum(train["valid"]),
            "coverage": (
                sum(train["valid"]) / len(train["valid"])
                if train["valid"] else 0.0
            ),
            "candidate_mean": mean,
            "candidate_std": std,
        },
        "test_2025": {
            "total_rows": len(test["y"]),
            "valid_rows": sum(test["valid"]),
            "coverage": (
                sum(test["valid"]) / len(test["valid"])
                if test["valid"] else 0.0
            ),
        },
        "calibration_control_fit": control_fit,
        "candidate_fit": candidate_fit,
        "candidate_coefficient": candidate_coefficient,
        "baseline_metrics": baseline_metrics,
        "calibration_control_metrics": control_metrics,
        "candidate_metrics": candidate_metrics,
        "delta_vs_baseline": delta_vs_baseline,
        "delta_vs_calibration_control": delta_vs_control,
        "official_coverage_ratio": official_coverage_ratio,
        "monthly_2025": monthly_rows,
        "monthly_brier_nonworse": monthly_nonworse,
        "gate": gate,
        "verdict": verdict,
    }


def main():
    print("HITS_PITCHER_CONTACT_INTERNAL_SCREEN_A", flush=True)
    print("=====================================", flush=True)

    print("\nSCREEN PREFLIGHT", flush=True)
    print("----------------", flush=True)

    print("status=INTERNAL_SCREEN_ONLY", flush=True)
    print("production_promotion_authorized=False", flush=True)
    print("reason=2025_ALREADY_SEEN_IN_RAW_AUDIT", flush=True)
    print("train_year=2024", flush=True)
    print("evaluation_year=2025", flush=True)
    print("2026_clean_holdout=False_burned", flush=True)
    print("stacking=False", flush=True)
    print("rescue_tuning=FORBIDDEN", flush=True)
    print(f"candidates={list(CANDIDATES)}", flush=True)

    if not AUDIT_DB.exists():
        raise RuntimeError(f"Missing audit DB: {AUDIT_DB}")

    score_2024_path = BASE_DIR / "scores_2024.json"
    score_2025_path = BASE_DIR / "scores_2025.json"

    if not score_2024_path.exists():
        raise RuntimeError(f"Missing baseline scores: {score_2024_path}")

    if not score_2025_path.exists():
        raise RuntimeError(f"Missing baseline scores: {score_2025_path}")

    score_2024 = load_json(score_2024_path)
    score_2025 = load_json(score_2025_path)

    print(
        f"2024_baseline_rows={len(score_2024['y_binary']):,}",
        flush=True,
    )
    print(
        f"2025_baseline_rows={len(score_2025['y_binary']):,}",
        flush=True,
    )

    conn = sqlite3.connect(str(AUDIT_DB))
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-16000")

    results = {}

    for candidate, config in CANDIDATES.items():
        print(
            f"\n===== RUNNING INDEPENDENT SCREEN: {candidate} =====",
            flush=True,
        )

        results[candidate] = evaluate_candidate(
            conn,
            candidate,
            config,
            score_2024,
            score_2025,
        )

    conn.close()

    print("\n2024 TRAINED HEADS", flush=True)
    print("------------------", flush=True)

    for candidate, result in results.items():
        control_coef = result["calibration_control_fit"]["coefficients"]
        candidate_coef = result["candidate_fit"]["coefficients"]

        print(
            f"{candidate}: "
            f"train_valid={result['train_2024']['valid_rows']:,} "
            f"coverage={result['train_2024']['coverage']:.4f} "
            f"control_coef={control_coef} "
            f"candidate_coef={candidate_coef} "
            f"candidate_c={result['candidate_coefficient']:+.6f} "
            f"converged={result['candidate_fit']['converged']}",
            flush=True,
        )

    print("\n2025 FULL-BOARD COMPARISON", flush=True)
    print("--------------------------", flush=True)

    for candidate, result in results.items():
        d = result["delta_vs_baseline"]
        dc = result["delta_vs_calibration_control"]

        print(
            f"{candidate}: "
            f"valid_coverage={result['test_2025']['coverage']:.4f} "
            f"vs_baseline_brier={d['brier']:+.8f} "
            f"logloss={d['logloss']:+.8f} "
            f"auc={d['auc']:+.6f} "
            f"top5={d['top5']:+.6f} "
            f"top10={d['top10']:+.6f} "
            f"official_hit_rate={d['official_hit_rate']:+.6f} "
            f"official_coverage_ratio={result['official_coverage_ratio']:.4f} "
            f"vs_control_brier={dc['brier']:+.8f} "
            f"vs_control_logloss={dc['logloss']:+.8f}",
            flush=True,
        )

    print("\nMONTHLY STABILITY", flush=True)
    print("-----------------", flush=True)

    for candidate, result in results.items():
        print(
            f"\n[{candidate}] "
            f"monthly_brier_nonworse={result['monthly_brier_nonworse']}",
            flush=True,
        )

        for month, row in result["monthly_2025"].items():
            print(
                f"{month}: "
                f"n={row['rows']:,} "
                f"baseline={row['baseline_brier']:.8f} "
                f"candidate={row['candidate_brier']:.8f} "
                f"delta={row['delta']:+.8f}",
                flush=True,
            )

    print("\nSTRICT INTERNAL SCREEN", flush=True)
    print("----------------------", flush=True)

    for candidate, result in results.items():
        print(f"\n[{candidate}]", flush=True)

        for key, value in result["gate"].items():
            print(f"{key}: {value}", flush=True)

        print(f"verdict={result['verdict']}", flush=True)

    print("\nFINAL SCREEN VERDICTS", flush=True)
    print("---------------------", flush=True)

    survivors = []

    for candidate, result in results.items():
        print(
            f"{candidate}: {result['verdict']}",
            flush=True,
        )

        if result["gate"]["overall_internal_screen_pass"]:
            survivors.append(candidate)

    if survivors:
        print(
            "next_step=DO_NOT_PROMOTE_YET; "
            "survivors require genuinely new forward data before production promotion.",
            flush=True,
        )
    else:
        print(
            "next_step=FREEZE_THIS_PITCHER_CONTACT_SCREEN; "
            "no rescue tuning on 2024/2025.",
            flush=True,
        )

    payload = {
        "script": "HITS_PITCHER_CONTACT_INTERNAL_SCREEN_A",
        "status": "INTERNAL_SCREEN_ONLY",
        "production_promotion_authorized": False,
        "reason": "2025_ALREADY_SEEN_IN_RAW_AUDIT",
        "train_year": 2024,
        "evaluation_year": 2025,
        "2026_clean_holdout": "NO_BURNED",
        "stacking": False,
        "rescue_tuning": "FORBIDDEN",
        "fixed_internal_screen": {
            "brier_vs_baseline_max": -0.0005,
            "logloss_vs_baseline_max": 0.0,
            "auc_vs_baseline_min": -0.0005,
            "top5_vs_baseline_min": 0.0,
            "top10_vs_baseline_min": 0.0,
            "official_hit_rate_vs_baseline_min": 0.0,
            "official_coverage_min_ratio": 0.90,
            "brier_vs_calibration_control_max": -0.0001,
            "logloss_vs_calibration_control_max": 0.0,
            "top5_vs_calibration_control_min": 0.0,
            "top10_vs_calibration_control_min": 0.0,
            "expected_coefficient_sign_required": True,
            "monthly_brier_nonworse_minimum": 4,
        },
        "results": results,
        "survivors": survivors,
    }

    OUT_JSON.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    lines = [
        "HITS_PITCHER_CONTACT_INTERNAL_SCREEN_A",
        "=" * 37,
        "",
        "INTERNAL SCREEN ONLY.",
        "2025 was already seen in raw candidate selection.",
        "No production promotion can be earned here.",
        "No stacking. No rescue tuning.",
        "",
        "FINAL SCREEN VERDICTS",
        "---------------------",
    ]

    for candidate, result in results.items():
        lines.extend(
            [
                "",
                candidate,
                "~" * len(candidate),
                f"verdict={result['verdict']}",
                f"candidate_coefficient={result['candidate_coefficient']:+.8f}",
                f"delta_vs_baseline={json.dumps(result['delta_vs_baseline'], sort_keys=True)}",
                f"delta_vs_calibration_control={json.dumps(result['delta_vs_calibration_control'], sort_keys=True)}",
                f"official_coverage_ratio={result['official_coverage_ratio']}",
                f"monthly_brier_nonworse={result['monthly_brier_nonworse']}",
                json.dumps(result["gate"], indent=2),
            ]
        )

    lines.extend(
        [
            "",
            f"SURVIVORS={survivors}",
            "",
            "Survival means pending genuinely new forward data.",
            "It does not mean promoted.",
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
