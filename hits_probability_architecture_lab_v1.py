#!/usr/bin/env python3
"""
HITS_PROBABILITY_ARCHITECTURE_LAB_V1

Focused 3x2 architecture benchmark for batter hits.

DEVELOPMENT STATUS
------------------
2024-2025 DEVELOPMENT DATA ONLY.
This lab cannot earn formal production promotion.

It does NOT touch 2026.

Frozen feature lineage:
    opp_pitcher_contact_allowed_rate_60d
    prior 60-day opposing-pitcher contact-allowed rate
    minimum 150 prior swings
    no feature-window tuning
    no threshold rescue
    no 2026 access

SIX CELLS
---------
A0  Count model -> exact Poisson analytic P(hit>=1) = 1 - exp(-lambda)
A1  Count model + frozen pitcher-contact feature -> exact Poisson analytic P

B0  Direct binary model -> P(hit>=1)
B1  Direct binary model + frozen pitcher-contact feature

C0  Count model lambda -> training-only logistic mapping to P(hit>=1)
C1  Count model + frozen pitcher-contact feature lambda
    -> training-only logistic mapping to P(hit>=1)

IMPORTANT
---------
The 250,000-simulation Monte Carlo layer is intentionally NOT used.
For an Over 0.5 hit target under a Poisson assumption:
    P(hit >= 1) = 1 - exp(-lambda)
exactly.

TIME-ORDERED OUTER FOLDS
------------------------
F1: train all eligible rows before 2024-07-01
    evaluate 2024-07-01 through 2024-08-31

F2: train all eligible rows before 2024-09-01
    evaluate 2024-09-01 through 2024-09-30

F3: train all eligible rows before 2025-04-01
    evaluate 2025-04-01 through 2025-05-31

F4: train all eligible rows before 2025-06-01
    evaluate 2025-06-01 through 2025-07-31

F5: train all eligible rows before 2025-08-01
    evaluate 2025-08-01 through 2025-09-30

No evaluation row is used to fit the model that scores it.

TRAINING-ONLY CALIBRATION FOR C0/C1
-----------------------------------
Within each outer fold:
1. Split the available training dates chronologically.
2. Use the final 25% of training dates as an inner calibration tail,
   bounded to 21-45 dates while leaving at least 30 core dates.
3. Train an inner count model on the earlier core only.
4. Generate truly out-of-sample lambdas on the calibration tail.
5. Fit:
       logit(P(hit>=1)) = a + b * lambda
   using only the calibration tail.
6. Train the final count model on ALL outer-training rows.
7. Apply the frozen a,b mapping to that fold's outer evaluation lambdas.

Thus the logistic mapping never fits and scores the same observations.

FAIR COMPARISONS
----------------
1. Fixed threshold:
       P >= 0.63

2. Fixed coverage:
       For each outer fold, N equals A0's number of P>=0.63 picks.
       Every cell is evaluated on its own top N picks for that fold.
       This compares equal opportunity volume.

METRICS
-------
Primary:
- Brier
- Logloss
- Calibration intercept
- Calibration slope
- ECE (10 equal-frequency bins)

Protected:
- AUC
- Top 5%
- Top 10%
- Official-tier hit rate at P>=0.63
- Fixed-coverage hit rate

Diagnostic:
- Zero-hit calibration by lambda band
- Probability sharpness / quantiles
- Official-board size inflation vs A0
- Monthly stability
- Feature coverage

MODEL SETTINGS
--------------
Same XGBoost structural settings across cells:
    learning_rate = 0.05
    max_depth = 6
    subsample = 0.8
    colsample_bytree = 0.8
    min_child_weight = 5
    tree_method = hist
    nthread = 2
    num_boost_round = 120

A cells use objective=count:poisson.
B cells use objective=binary:logistic.
C cells reuse count-model lambdas and apply the training-only logistic map.

MEMORY SAFETY
-------------
Designed for a 512 MB Render instance:
- no pandas
- no sklearn
- one XGBoost child process at a time
- disk-backed fold files
- SQLite read-only feature source
- completed fold predictions are cached/reused

EXPECTED INPUTS
---------------
/data/hr_model/hits_rfv_lite_b/features_2024.libsvm
/data/hr_model/hits_rfv_lite_b/features_2024_meta.csv
/data/hr_model/hits_rfv_lite_b/features_2025.libsvm
/data/hr_model/hits_rfv_lite_b/features_2025_meta.csv
/data/hr_model/hits_raw_structural_audit_a_work/audit.sqlite

RUN
---
python -u hits_probability_architecture_lab_v1.py 2>&1 | tee /data/hr_model/hits_probability_architecture_lab_v1.log

OUTPUTS
-------
/data/hr_model/hits_probability_architecture_lab_v1_results.json
/data/hr_model/hits_probability_architecture_lab_v1_report.txt

PASTE BACK
----------
LAB PREFLIGHT
FOLD SUMMARY
SIX-CELL AGGREGATE
FIXED THRESHOLD COMPARISON
FIXED COVERAGE COMPARISON
ZERO-HIT POISSON AUDIT
MONTHLY STABILITY SUMMARY
ARCHITECTURE READ
FINAL LAB STATUS
"""

import argparse
import csv
import json
import math
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


# =============================================================================
# Locked paths / constants
# =============================================================================

BASE_DIR = Path("/data/hr_model/hits_rfv_lite_b")
AUDIT_DB = Path("/data/hr_model/hits_raw_structural_audit_a_work/audit.sqlite")
WORK_DIR = Path("/data/hr_model/hits_probability_architecture_lab_v1_work")

OUT_JSON = Path("/data/hr_model/hits_probability_architecture_lab_v1_results.json")
OUT_TXT = Path("/data/hr_model/hits_probability_architecture_lab_v1_report.txt")

CANDIDATE = "opp_pitcher_contact_allowed_rate_60d"
SAMPLE_COL = "p_swings_60d"
MIN_SAMPLE = 150

OFFICIAL_MIN = 0.630
NUM_BOOST_ROUND = 120

CELLS = ["A0", "A1", "B0", "B1", "C0", "C1"]

OUTER_FOLDS = [
    {
        "name": "F1",
        "train_before": "2024-07-01",
        "eval_start": "2024-07-01",
        "eval_end": "2024-09-01",
    },
    {
        "name": "F2",
        "train_before": "2024-09-01",
        "eval_start": "2024-09-01",
        "eval_end": "2024-10-01",
    },
    {
        "name": "F3",
        "train_before": "2025-04-01",
        "eval_start": "2025-04-01",
        "eval_end": "2025-06-01",
    },
    {
        "name": "F4",
        "train_before": "2025-06-01",
        "eval_start": "2025-06-01",
        "eval_end": "2025-08-01",
    },
    {
        "name": "F5",
        "train_before": "2025-08-01",
        "eval_start": "2025-08-01",
        "eval_end": "2025-10-01",
    },
]

COUNT_PARAMS = {
    "objective": "count:poisson",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    "nthread": 2,
    "seed": 42,
}

BINARY_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    "nthread": 2,
    "seed": 42,
}

ZERO_HIT_BANDS = [
    (0.00, 0.25, "0.00-0.25"),
    (0.25, 0.50, "0.25-0.50"),
    (0.50, 0.75, "0.50-0.75"),
    (0.75, 1.00, "0.75-1.00"),
    (1.00, 1.25, "1.00-1.25"),
    (1.25, 1.50, "1.25-1.50"),
    (1.50, 2.00, "1.50-2.00"),
    (2.00, float("inf"), "2.00+"),
]


# =============================================================================
# Generic helpers
# =============================================================================

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clip(value, lo, hi):
    return max(lo, min(hi, value))


def sigmoid(x):
    x = clip(float(x), -35.0, 35.0)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p):
    p = clip(float(p), 1e-8, 1.0 - 1e-8)
    return math.log(p / (1.0 - p))


def poisson_hit_prob(lam):
    return clip(1.0 - math.exp(-max(float(lam), 1e-12)), 1e-8, 1.0 - 1e-8)


def parse_libsvm_label(line):
    return float(line.split(None, 1)[0])


def relabel_libsvm_line(line, new_label):
    parts = line.rstrip("\n").split(None, 1)
    if len(parts) == 1:
        return f"{new_label:.10g}\n"
    return f"{new_label:.10g} {parts[1]}\n"


def append_feature(line, feature_index, value):
    base = line.rstrip("\n")
    if value is None:
        return base + "\n"
    return base + f" {feature_index}:{float(value):.10g}\n"


def quantile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    pos = (len(sorted_values) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return float(sorted_values[lo])

    weight = pos - lo
    return (
        float(sorted_values[lo]) * (1.0 - weight)
        + float(sorted_values[hi]) * weight
    )


# =============================================================================
# Simple logistic solver
# =============================================================================

def solve_linear(matrix, vector):
    n = len(vector)
    a = [list(row) + [float(vector[i])] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))

        if abs(a[pivot][col]) < 1e-12:
            raise RuntimeError("Singular Hessian in logistic solve.")

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


def fit_logistic(features, y, ridge=1e-3, max_iter=50, tol=1e-8):
    if not features:
        raise RuntimeError("No rows for logistic fit.")

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

        # No ridge penalty on intercept.
        for j in range(1, d):
            grad[j] -= ridge * beta[j]
            hess[j][j] += ridge

        step = solve_linear(hess, grad)

        for j in range(d):
            beta[j] += step[j]

        if max(abs(s) for s in step) < tol:
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


def logistic_predict(features, coefficients):
    return [
        sigmoid(sum(c * xj for c, xj in zip(coefficients, x)))
        for x in features
    ]


# =============================================================================
# Source loading / alignment
# =============================================================================

def audit_columns(conn):
    return {
        row[1]
        for row in conn.execute("PRAGMA table_info(audit_joined)")
    }


def load_year_rows(year):
    feature_path = BASE_DIR / f"features_{year}.libsvm"
    meta_path = BASE_DIR / f"features_{year}_meta.csv"

    if not feature_path.exists():
        raise RuntimeError(f"Missing {feature_path}")

    if not meta_path.exists():
        raise RuntimeError(f"Missing {meta_path}")

    with open(feature_path, "r", encoding="utf-8") as f:
        feature_lines = f.readlines()

    meta_rows = []

    with open(meta_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            meta_rows.append(row)

    if len(feature_lines) != len(meta_rows):
        raise RuntimeError(
            f"{year}: feature/meta mismatch "
            f"{len(feature_lines)} vs {len(meta_rows)}"
        )

    conn = sqlite3.connect(str(AUDIT_DB))
    cols = audit_columns(conn)

    needed = {"year", "row_id", CANDIDATE, SAMPLE_COL}
    missing = needed - cols

    if missing:
        conn.close()
        raise RuntimeError(
            f"Audit DB missing columns: {sorted(missing)}"
        )

    audit_rows = list(
        conn.execute(
            f"""
            SELECT row_id, {CANDIDATE}, {SAMPLE_COL}
            FROM audit_joined
            WHERE year = ?
            ORDER BY row_id
            """,
            (year,),
        )
    )
    conn.close()

    if len(audit_rows) != len(feature_lines):
        raise RuntimeError(
            f"{year}: audit/feature mismatch "
            f"{len(audit_rows)} vs {len(feature_lines)}"
        )

    rows = []
    valid_feature = 0

    for i, (line, meta, audit) in enumerate(
        zip(feature_lines, meta_rows, audit_rows)
    ):
        row_id, candidate_value, sample_n = audit

        if int(row_id) != i:
            raise RuntimeError(
                f"{year}: row_id alignment failed at {i}, got {row_id}"
            )

        game_date = (
            meta.get("game_date")
            or meta.get("date")
            or ""
        )

        actual_hits = meta.get("actual_hits")

        if actual_hits is None or actual_hits == "":
            actual_hits = parse_libsvm_label(line)

        actual_hits = float(actual_hits)
        actual_hit = 1 if actual_hits >= 1.0 else 0

        frozen_value = None

        if (
            candidate_value is not None
            and sample_n is not None
            and float(sample_n) >= MIN_SAMPLE
        ):
            frozen_value = float(candidate_value)
            valid_feature += 1

        rows.append(
            {
                "year": year,
                "row_id": i,
                "game_date": str(game_date),
                "actual_hits": actual_hits,
                "actual_hit": actual_hit,
                "base_line": line,
                "candidate_value": frozen_value,
            }
        )

    return rows, {
        "year": year,
        "rows": len(rows),
        "candidate_valid_rows": valid_feature,
        "candidate_coverage": (
            valid_feature / len(rows) if rows else 0.0
        ),
    }


def load_all_rows():
    rows_2024, summary_2024 = load_year_rows(2024)
    rows_2025, summary_2025 = load_year_rows(2025)

    all_rows = rows_2024 + rows_2025

    all_rows.sort(
        key=lambda r: (
            r["game_date"],
            r["year"],
            r["row_id"],
        )
    )

    for global_id, row in enumerate(all_rows):
        row["global_id"] = global_id

    return all_rows, {
        "2024": summary_2024,
        "2025": summary_2025,
    }


# =============================================================================
# Fold construction
# =============================================================================

def rows_before(rows, date_cutoff):
    return [row for row in rows if row["game_date"] < date_cutoff]


def rows_between(rows, start_date, end_date):
    return [
        row
        for row in rows
        if start_date <= row["game_date"] < end_date
    ]


def inner_calibration_split(train_rows):
    dates = sorted({row["game_date"] for row in train_rows})

    if len(dates) < 40:
        raise RuntimeError(
            f"Need at least 40 training dates, got {len(dates)}"
        )

    n_cal = max(21, int(round(0.25 * len(dates))))
    n_cal = min(45, n_cal)

    if len(dates) - n_cal < 30:
        n_cal = len(dates) - 30

    if n_cal < 7:
        raise RuntimeError(
            f"Inner calibration tail too small: {n_cal} dates"
        )

    calibration_dates = set(dates[-n_cal:])

    core = [
        row
        for row in train_rows
        if row["game_date"] not in calibration_dates
    ]

    calibration = [
        row
        for row in train_rows
        if row["game_date"] in calibration_dates
    ]

    return core, calibration, {
        "train_dates": len(dates),
        "core_dates": len(dates) - n_cal,
        "calibration_dates": n_cal,
        "calibration_start": min(calibration_dates),
        "calibration_end": max(calibration_dates),
    }


# =============================================================================
# Fold files
# =============================================================================

def write_libsvm(rows, out_path, binary_label, include_candidate):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            label = (
                float(row["actual_hit"])
                if binary_label
                else float(row["actual_hits"])
            )

            line = relabel_libsvm_line(row["base_line"], label)

            if include_candidate:
                line = append_feature(
                    line,
                    9,
                    row["candidate_value"],
                )

            f.write(line)

    return out_path


def prepare_fold_files(fold, train_rows, eval_rows, core_rows, cal_rows):
    fold_dir = WORK_DIR / fold["name"]
    fold_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    datasets = {
        "train": train_rows,
        "eval": eval_rows,
        "core": core_rows,
        "cal": cal_rows,
    }

    for dataset_name, dataset_rows in datasets.items():
        for info_name, include_candidate in (
            ("base", False),
            ("feature", True),
        ):
            for target_name, binary_label in (
                ("count", False),
                ("binary", True),
            ):
                key = f"{dataset_name}_{info_name}_{target_name}"
                path = fold_dir / f"{key}.libsvm"

                if not path.exists():
                    write_libsvm(
                        dataset_rows,
                        path,
                        binary_label=binary_label,
                        include_candidate=include_candidate,
                    )

                paths[key] = path

    return paths


# =============================================================================
# XGBoost child workers
# =============================================================================

def run_child(args, label):
    env = dict(os.environ)

    env["OMP_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    print(f"\n{label}", flush=True)
    print("-" * len(label), flush=True)

    cp = subprocess.run(
        [sys.executable, "-u", str(Path(__file__).resolve())] + args,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if cp.stdout:
        print(
            cp.stdout,
            end="" if cp.stdout.endswith("\n") else "\n",
            flush=True,
        )

    if cp.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {cp.returncode}"
        )


def worker_train(args):
    import xgboost as xgb

    params = COUNT_PARAMS if args.objective == "count" else BINARY_PARAMS

    dtrain = xgb.DMatrix(
        f"{args.train_file}?format=libsvm"
    )

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
    )

    booster.save_model(args.out_model)

    print(f"objective={args.objective}", flush=True)
    print(f"rows={dtrain.num_row():,}", flush=True)
    print(f"rounds={booster.num_boosted_rounds()}", flush=True)
    print(f"saved_model={args.out_model}", flush=True)


def worker_score(args):
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(args.model_path)

    dtest = xgb.DMatrix(
        f"{args.score_file}?format=libsvm"
    )

    pred = booster.predict(dtest)

    payload = {
        "rows": int(len(pred)),
        "prediction": [float(x) for x in pred],
    }

    Path(args.out_json).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    print(f"scored_rows={len(pred):,}", flush=True)
    print(f"saved_scores={args.out_json}", flush=True)


def train_or_reuse(train_file, objective, model_path, label):
    if model_path.exists():
        print(f"{label}: model REUSED | {model_path}", flush=True)
        return

    run_child(
        [
            "--worker-train",
            "--objective", objective,
            "--train-file", str(train_file),
            "--out-model", str(model_path),
        ],
        label,
    )


def score_or_reuse(model_path, score_file, out_json, label):
    if out_json.exists():
        print(f"{label}: scores REUSED | {out_json}", flush=True)
        return load_json(out_json)["prediction"]

    run_child(
        [
            "--worker-score",
            "--model-path", str(model_path),
            "--score-file", str(score_file),
            "--out-json", str(out_json),
        ],
        label,
    )

    return load_json(out_json)["prediction"]


# =============================================================================
# Metrics
# =============================================================================

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


def calibration_intercept_slope(y, p):
    x = [[1.0, clip(logit(pi), -12.0, 12.0)] for pi in p]

    try:
        fit = fit_logistic(
            x,
            y,
            ridge=1e-6,
            max_iter=75,
            tol=1e-9,
        )

        return {
            "intercept": fit["coefficients"][0],
            "slope": fit["coefficients"][1],
            "converged": fit["converged"],
        }
    except Exception as exc:
        return {
            "intercept": None,
            "slope": None,
            "converged": False,
            "error": str(exc),
        }


def ece_equal_frequency(y, p, bins=10):
    order = sorted(range(len(y)), key=lambda i: p[i])
    rows = []
    total = len(y)

    for b in range(bins):
        start = b * total // bins
        end = (b + 1) * total // bins
        idx = order[start:end]

        if not idx:
            continue

        actual = sum(y[i] for i in idx) / len(idx)
        mean_p = sum(p[i] for i in idx) / len(idx)

        rows.append(
            {
                "bin": b + 1,
                "rows": len(idx),
                "actual_hit_rate": actual,
                "mean_probability": mean_p,
                "gap_actual_minus_pred": actual - mean_p,
            }
        )

    ece = sum(
        (row["rows"] / total)
        * abs(row["gap_actual_minus_pred"])
        for row in rows
    )

    return {
        "ece": ece,
        "bins": rows,
    }


def probability_sharpness(p):
    values = sorted(float(x) for x in p)
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n

    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "p05": quantile(values, 0.05),
        "p25": quantile(values, 0.25),
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p95": quantile(values, 0.95),
        "fraction_ge_0.50": sum(x >= 0.50 for x in values) / n,
        "fraction_ge_0.63": sum(x >= 0.63 for x in values) / n,
        "fraction_ge_0.70": sum(x >= 0.70 for x in values) / n,
        "fraction_ge_0.80": sum(x >= 0.80 for x in values) / n,
    }


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

    out["calibration"] = calibration_intercept_slope(y, p)
    out["ece"] = ece_equal_frequency(y, p, bins=10)
    out["sharpness"] = probability_sharpness(p)

    return out


def zero_hit_audit(y, lambdas):
    rows = []

    for lo, hi, label in ZERO_HIT_BANDS:
        idx = [
            i
            for i, lam in enumerate(lambdas)
            if float(lam) >= lo and float(lam) < hi
        ]

        if not idx:
            rows.append(
                {
                    "band": label,
                    "rows": 0,
                    "mean_lambda": None,
                    "observed_zero_rate": None,
                    "poisson_implied_zero_rate": None,
                    "gap_observed_minus_poisson": None,
                }
            )
            continue

        observed_zero = sum(1 - y[i] for i in idx) / len(idx)

        poisson_zero = sum(
            math.exp(-max(float(lambdas[i]), 0.0))
            for i in idx
        ) / len(idx)

        rows.append(
            {
                "band": label,
                "rows": len(idx),
                "mean_lambda": sum(lambdas[i] for i in idx) / len(idx),
                "observed_zero_rate": observed_zero,
                "poisson_implied_zero_rate": poisson_zero,
                "gap_observed_minus_poisson": observed_zero - poisson_zero,
            }
        )

    return rows


# =============================================================================
# One fold
# =============================================================================

def execute_fold(all_rows, fold):
    fold_name = fold["name"]
    fold_dir = WORK_DIR / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)

    final_json = fold_dir / "fold_results.json"

    if final_json.exists():
        print(f"{fold_name}: completed fold REUSED", flush=True)
        return load_json(final_json)

    train_rows = rows_before(all_rows, fold["train_before"])
    eval_rows = rows_between(
        all_rows,
        fold["eval_start"],
        fold["eval_end"],
    )

    if not train_rows:
        raise RuntimeError(f"{fold_name}: no training rows")

    if not eval_rows:
        raise RuntimeError(f"{fold_name}: no evaluation rows")

    core_rows, cal_rows, inner_info = inner_calibration_split(train_rows)

    print(
        f"{fold_name}: train={len(train_rows):,} "
        f"core={len(core_rows):,} "
        f"cal={len(cal_rows):,} "
        f"eval={len(eval_rows):,} "
        f"eval_range={fold['eval_start']}..{fold['eval_end']}",
        flush=True,
    )

    paths = prepare_fold_files(
        fold,
        train_rows,
        eval_rows,
        core_rows,
        cal_rows,
    )

    # -------------------------------------------------------------------------
    # Final outer models
    # -------------------------------------------------------------------------

    model_a0 = fold_dir / "A0_count_base.json"
    model_a1 = fold_dir / "A1_count_feature.json"
    model_b0 = fold_dir / "B0_binary_base.json"
    model_b1 = fold_dir / "B1_binary_feature.json"

    train_or_reuse(
        paths["train_base_count"],
        "count",
        model_a0,
        f"{fold_name} TRAIN A0/A-C0 COUNT BASE",
    )

    train_or_reuse(
        paths["train_feature_count"],
        "count",
        model_a1,
        f"{fold_name} TRAIN A1/A-C1 COUNT FEATURE",
    )

    train_or_reuse(
        paths["train_base_binary"],
        "binary",
        model_b0,
        f"{fold_name} TRAIN B0 DIRECT BINARY BASE",
    )

    train_or_reuse(
        paths["train_feature_binary"],
        "binary",
        model_b1,
        f"{fold_name} TRAIN B1 DIRECT BINARY FEATURE",
    )

    # -------------------------------------------------------------------------
    # Outer evaluation predictions
    # -------------------------------------------------------------------------

    lambda_a0 = score_or_reuse(
        model_a0,
        paths["eval_base_count"],
        fold_dir / "eval_lambda_A0.json",
        f"{fold_name} SCORE A0 OUTER EVAL",
    )

    lambda_a1 = score_or_reuse(
        model_a1,
        paths["eval_feature_count"],
        fold_dir / "eval_lambda_A1.json",
        f"{fold_name} SCORE A1 OUTER EVAL",
    )

    prob_b0 = score_or_reuse(
        model_b0,
        paths["eval_base_binary"],
        fold_dir / "eval_prob_B0.json",
        f"{fold_name} SCORE B0 OUTER EVAL",
    )

    prob_b1 = score_or_reuse(
        model_b1,
        paths["eval_feature_binary"],
        fold_dir / "eval_prob_B1.json",
        f"{fold_name} SCORE B1 OUTER EVAL",
    )

    prob_a0 = [poisson_hit_prob(lam) for lam in lambda_a0]
    prob_a1 = [poisson_hit_prob(lam) for lam in lambda_a1]

    prob_b0 = [
        clip(float(p), 1e-8, 1.0 - 1e-8)
        for p in prob_b0
    ]

    prob_b1 = [
        clip(float(p), 1e-8, 1.0 - 1e-8)
        for p in prob_b1
    ]

    # -------------------------------------------------------------------------
    # Inner training-only calibration for C0/C1
    # -------------------------------------------------------------------------

    core_model_c0 = fold_dir / "C0_inner_count_base.json"
    core_model_c1 = fold_dir / "C1_inner_count_feature.json"

    train_or_reuse(
        paths["core_base_count"],
        "count",
        core_model_c0,
        f"{fold_name} TRAIN C0 INNER CORE COUNT BASE",
    )

    train_or_reuse(
        paths["core_feature_count"],
        "count",
        core_model_c1,
        f"{fold_name} TRAIN C1 INNER CORE COUNT FEATURE",
    )

    cal_lambda_c0 = score_or_reuse(
        core_model_c0,
        paths["cal_base_count"],
        fold_dir / "cal_lambda_C0.json",
        f"{fold_name} SCORE C0 INNER CALIBRATION TAIL",
    )

    cal_lambda_c1 = score_or_reuse(
        core_model_c1,
        paths["cal_feature_count"],
        fold_dir / "cal_lambda_C1.json",
        f"{fold_name} SCORE C1 INNER CALIBRATION TAIL",
    )

    cal_y = [row["actual_hit"] for row in cal_rows]

    cal_fit_c0 = fit_logistic(
        [[1.0, max(float(lam), 0.0)] for lam in cal_lambda_c0],
        cal_y,
        ridge=1e-3,
        max_iter=50,
    )

    cal_fit_c1 = fit_logistic(
        [[1.0, max(float(lam), 0.0)] for lam in cal_lambda_c1],
        cal_y,
        ridge=1e-3,
        max_iter=50,
    )

    prob_c0 = logistic_predict(
        [[1.0, max(float(lam), 0.0)] for lam in lambda_a0],
        cal_fit_c0["coefficients"],
    )

    prob_c1 = logistic_predict(
        [[1.0, max(float(lam), 0.0)] for lam in lambda_a1],
        cal_fit_c1["coefficients"],
    )

    y_eval = [row["actual_hit"] for row in eval_rows]
    dates_eval = [row["game_date"] for row in eval_rows]
    ids_eval = [row["global_id"] for row in eval_rows]

    probabilities = {
        "A0": prob_a0,
        "A1": prob_a1,
        "B0": prob_b0,
        "B1": prob_b1,
        "C0": prob_c0,
        "C1": prob_c1,
    }

    # Fixed coverage N = A0 official board size for THIS fold.
    fixed_n = sum(p >= OFFICIAL_MIN for p in prob_a0)

    fixed_coverage = {}

    for cell in CELLS:
        order = sorted(
            range(len(y_eval)),
            key=lambda i: probabilities[cell][i],
            reverse=True,
        )

        idx = order[:fixed_n] if fixed_n > 0 else []

        fixed_coverage[cell] = {
            "rows": len(idx),
            "hits": sum(y_eval[i] for i in idx),
            "hit_rate": (
                sum(y_eval[i] for i in idx) / len(idx)
                if idx
                else None
            ),
            "mean_probability": (
                sum(probabilities[cell][i] for i in idx) / len(idx)
                if idx
                else None
            ),
        }

    fold_result = {
        "fold": fold,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "inner_calibration": inner_info,
        "candidate_train_coverage": (
            sum(row["candidate_value"] is not None for row in train_rows)
            / len(train_rows)
        ),
        "candidate_eval_coverage": (
            sum(row["candidate_value"] is not None for row in eval_rows)
            / len(eval_rows)
        ),
        "calibrator_C0": cal_fit_c0,
        "calibrator_C1": cal_fit_c1,
        "fixed_coverage_n_from_A0_official": fixed_n,
        "fixed_coverage": fixed_coverage,
        "y": y_eval,
        "dates": dates_eval,
        "global_ids": ids_eval,
        "lambdas": {
            "A0": [float(x) for x in lambda_a0],
            "A1": [float(x) for x in lambda_a1],
        },
        "probabilities": probabilities,
    }

    final_json.write_text(
        json.dumps(fold_result),
        encoding="utf-8",
    )

    return fold_result


# =============================================================================
# Aggregate reporting
# =============================================================================

def aggregate_folds(fold_results):
    y = []
    dates = []
    global_ids = []

    probabilities = {
        cell: []
        for cell in CELLS
    }

    lambdas = {
        "A0": [],
        "A1": [],
    }

    fixed_coverage_totals = {
        cell: {
            "rows": 0,
            "hits": 0,
            "weighted_probability_sum": 0.0,
        }
        for cell in CELLS
    }

    fold_summaries = []

    for fold_result in fold_results:
        y.extend(fold_result["y"])
        dates.extend(fold_result["dates"])
        global_ids.extend(fold_result["global_ids"])

        for cell in CELLS:
            probabilities[cell].extend(
                fold_result["probabilities"][cell]
            )

            fc = fold_result["fixed_coverage"][cell]
            fixed_coverage_totals[cell]["rows"] += fc["rows"]
            fixed_coverage_totals[cell]["hits"] += fc["hits"]

            if fc["rows"] > 0 and fc["mean_probability"] is not None:
                fixed_coverage_totals[cell][
                    "weighted_probability_sum"
                ] += fc["mean_probability"] * fc["rows"]

        lambdas["A0"].extend(fold_result["lambdas"]["A0"])
        lambdas["A1"].extend(fold_result["lambdas"]["A1"])

        fold_summaries.append(
            {
                "fold": fold_result["fold"]["name"],
                "train_rows": fold_result["train_rows"],
                "eval_rows": fold_result["eval_rows"],
                "eval_start": fold_result["fold"]["eval_start"],
                "eval_end": fold_result["fold"]["eval_end"],
                "candidate_train_coverage": (
                    fold_result["candidate_train_coverage"]
                ),
                "candidate_eval_coverage": (
                    fold_result["candidate_eval_coverage"]
                ),
                "fixed_coverage_n_from_A0_official": (
                    fold_result["fixed_coverage_n_from_A0_official"]
                ),
                "calibrator_C0_coefficients": (
                    fold_result["calibrator_C0"]["coefficients"]
                ),
                "calibrator_C1_coefficients": (
                    fold_result["calibrator_C1"]["coefficients"]
                ),
            }
        )

    metrics = {
        cell: metric_block(y, probabilities[cell])
        for cell in CELLS
    }

    fixed_coverage = {}

    for cell in CELLS:
        row = fixed_coverage_totals[cell]

        fixed_coverage[cell] = {
            "rows": row["rows"],
            "hits": row["hits"],
            "hit_rate": (
                row["hits"] / row["rows"]
                if row["rows"]
                else None
            ),
            "mean_probability": (
                row["weighted_probability_sum"] / row["rows"]
                if row["rows"]
                else None
            ),
        }

    a0_official_rows = metrics["A0"]["official_rows"]

    for cell in CELLS:
        metrics[cell]["official_board_size_inflation_vs_A0"] = (
            metrics[cell]["official_rows"] / a0_official_rows - 1.0
            if a0_official_rows > 0
            else None
        )

    zero_hit = {
        "A0": zero_hit_audit(y, lambdas["A0"]),
        "A1": zero_hit_audit(y, lambdas["A1"]),
    }

    monthly = monthly_metrics(
        y,
        dates,
        probabilities,
    )

    return {
        "y": y,
        "dates": dates,
        "global_ids": global_ids,
        "probabilities": probabilities,
        "lambdas": lambdas,
        "metrics": metrics,
        "fixed_coverage": fixed_coverage,
        "zero_hit_audit": zero_hit,
        "monthly": monthly,
        "fold_summaries": fold_summaries,
    }


def monthly_metrics(y, dates, probabilities):
    groups = defaultdict(list)

    for i, date_text in enumerate(dates):
        groups[str(date_text)[:7]].append(i)

    out = {}

    for month, idx in sorted(groups.items()):
        row = {
            "rows": len(idx),
            "actual_hit_rate": sum(y[i] for i in idx) / len(idx),
            "cells": {},
        }

        for cell in CELLS:
            month_y = [y[i] for i in idx]
            month_p = [probabilities[cell][i] for i in idx]

            m = metric_block(month_y, month_p)

            row["cells"][cell] = {
                "brier": m["brier"],
                "logloss": m["logloss"],
                "auc": m["auc"],
                "top5_actual": m["top5_actual"],
                "top10_actual": m["top10_actual"],
                "official_rows": m["official_rows"],
                "official_hit_rate": m["official_hit_rate"],
                "mean_probability": m["mean_probability"],
            }

        out[month] = row

    return out


def architecture_read(metrics, fixed_coverage):
    """
    Diagnostic read only. This lab cannot promote anything.

    We report:
    - best cell for each primary metric
    - best architecture base cell (0) and feature cell (1)
    - feature deltas within each architecture
    - whether any architecture shows primary-metric dominance with protected
      metrics preserved versus its paired baseline.
    """

    best = {
        "brier": min(CELLS, key=lambda c: metrics[c]["brier"]),
        "logloss": min(CELLS, key=lambda c: metrics[c]["logloss"]),
        "ece": min(CELLS, key=lambda c: metrics[c]["ece"]["ece"]),
        "auc": max(CELLS, key=lambda c: metrics[c]["auc"]),
        "top5": max(CELLS, key=lambda c: metrics[c]["top5_actual"]),
        "top10": max(CELLS, key=lambda c: metrics[c]["top10_actual"]),
        "official_hit_rate": max(
            CELLS,
            key=lambda c: (
                metrics[c]["official_hit_rate"]
                if metrics[c]["official_hit_rate"] is not None
                else -1.0
            ),
        ),
        "fixed_coverage_hit_rate": max(
            CELLS,
            key=lambda c: (
                fixed_coverage[c]["hit_rate"]
                if fixed_coverage[c]["hit_rate"] is not None
                else -1.0
            ),
        ),
    }

    pairings = {
        "A": ("A0", "A1"),
        "B": ("B0", "B1"),
        "C": ("C0", "C1"),
    }

    feature_effect = {}

    for architecture, (base_cell, feature_cell) in pairings.items():
        base = metrics[base_cell]
        feat = metrics[feature_cell]

        feature_effect[architecture] = {
            "brier_delta": feat["brier"] - base["brier"],
            "logloss_delta": feat["logloss"] - base["logloss"],
            "ece_delta": feat["ece"]["ece"] - base["ece"]["ece"],
            "auc_delta": feat["auc"] - base["auc"],
            "top5_delta": feat["top5_actual"] - base["top5_actual"],
            "top10_delta": feat["top10_actual"] - base["top10_actual"],
            "official_hit_rate_delta": (
                feat["official_hit_rate"] - base["official_hit_rate"]
                if (
                    feat["official_hit_rate"] is not None
                    and base["official_hit_rate"] is not None
                )
                else None
            ),
            "official_rows_delta": (
                feat["official_rows"] - base["official_rows"]
            ),
            "fixed_coverage_hit_rate_delta": (
                fixed_coverage[feature_cell]["hit_rate"]
                - fixed_coverage[base_cell]["hit_rate"]
                if (
                    fixed_coverage[feature_cell]["hit_rate"] is not None
                    and fixed_coverage[base_cell]["hit_rate"] is not None
                )
                else None
            ),
        }

    # Cell-level diagnostic dominance only; no production promotion.
    diagnostic_survivors = {}

    for architecture, (base_cell, feature_cell) in pairings.items():
        d = feature_effect[architecture]

        diagnostic_survivors[architecture] = {
            "feature_improves_brier": d["brier_delta"] < 0.0,
            "feature_improves_logloss": d["logloss_delta"] < 0.0,
            "feature_improves_or_holds_auc": d["auc_delta"] >= -0.0005,
            "feature_protects_top5": d["top5_delta"] >= 0.0,
            "feature_protects_top10": d["top10_delta"] >= 0.0,
            "feature_protects_fixed_coverage": (
                d["fixed_coverage_hit_rate_delta"] is not None
                and d["fixed_coverage_hit_rate_delta"] >= 0.0
            ),
            "feature_protects_official_hit_rate": (
                d["official_hit_rate_delta"] is not None
                and d["official_hit_rate_delta"] >= 0.0
            ),
        }

        diagnostic_survivors[architecture]["all_protected"] = all(
            diagnostic_survivors[architecture].values()
        )

    return {
        "best_cell_by_metric": best,
        "feature_effect_within_architecture": feature_effect,
        "diagnostic_feature_survival": diagnostic_survivors,
        "production_promotion_authorized": False,
        "status": "DEVELOPMENT_LAB_ONLY",
    }


# =============================================================================
# Parent
# =============================================================================

def main_parent():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print("HITS_PROBABILITY_ARCHITECTURE_LAB_V1", flush=True)
    print("====================================", flush=True)

    print("\nLAB PREFLIGHT", flush=True)
    print("-------------", flush=True)
    print("status=DEVELOPMENT_LAB_ONLY", flush=True)
    print("years=2024,2025", flush=True)
    print("touch_2026=False", flush=True)
    print(f"frozen_candidate={CANDIDATE}", flush=True)
    print(f"frozen_window_days=60", flush=True)
    print(f"frozen_min_prior_swings={MIN_SAMPLE}", flush=True)
    print("threshold_rescue=FORBIDDEN", flush=True)
    print("feature_window_tuning=FORBIDDEN", flush=True)
    print("monte_carlo_used=False", flush=True)
    print("poisson_mapping=exact_1_minus_exp_minus_lambda", flush=True)
    print("calibration_mapping=logit_p_equals_a_plus_b_times_lambda", flush=True)
    print(f"num_boost_round={NUM_BOOST_ROUND}", flush=True)
    print("fixed_threshold=0.63", flush=True)
    print("fixed_coverage_reference=A0_official_count_per_fold", flush=True)
    print("production_promotion_authorized=False", flush=True)

    for required in [
        BASE_DIR / "features_2024.libsvm",
        BASE_DIR / "features_2024_meta.csv",
        BASE_DIR / "features_2025.libsvm",
        BASE_DIR / "features_2025_meta.csv",
        AUDIT_DB,
    ]:
        if not required.exists():
            raise RuntimeError(f"Missing required input: {required}")
        print(f"exists=True | {required}", flush=True)

    all_rows, source_summary = load_all_rows()

    print(
        f"loaded_rows={len(all_rows):,} "
        f"2024={source_summary['2024']['rows']:,} "
        f"2025={source_summary['2025']['rows']:,}",
        flush=True,
    )

    print(
        f"candidate_coverage_2024="
        f"{source_summary['2024']['candidate_coverage']:.4f}",
        flush=True,
    )

    print(
        f"candidate_coverage_2025="
        f"{source_summary['2025']['candidate_coverage']:.4f}",
        flush=True,
    )

    fold_results = []

    print("\nFOLD EXECUTION", flush=True)
    print("--------------", flush=True)

    for fold in OUTER_FOLDS:
        fold_results.append(
            execute_fold(all_rows, fold)
        )

    aggregate = aggregate_folds(fold_results)

    print("\nFOLD SUMMARY", flush=True)
    print("------------", flush=True)

    for row in aggregate["fold_summaries"]:
        print(
            f"{row['fold']}: "
            f"train={row['train_rows']:,} "
            f"eval={row['eval_rows']:,} "
            f"range={row['eval_start']}..{row['eval_end']} "
            f"candidate_train_coverage={row['candidate_train_coverage']:.4f} "
            f"candidate_eval_coverage={row['candidate_eval_coverage']:.4f} "
            f"A0_official_N={row['fixed_coverage_n_from_A0_official']:,}",
            flush=True,
        )

    print("\nSIX-CELL AGGREGATE", flush=True)
    print("------------------", flush=True)

    for cell in CELLS:
        m = aggregate["metrics"][cell]

        print(
            f"{cell}: "
            f"n={m['rows']:,} "
            f"brier={m['brier']:.8f} "
            f"logloss={m['logloss']:.8f} "
            f"auc={m['auc']:.6f} "
            f"cal_intercept={m['calibration']['intercept']} "
            f"cal_slope={m['calibration']['slope']} "
            f"ece={m['ece']['ece']:.6f} "
            f"top5={m['top5_actual']:.6f} "
            f"top10={m['top10_actual']:.6f} "
            f"official_n={m['official_rows']:,} "
            f"official_hit_rate={m['official_hit_rate']} "
            f"official_inflation_vs_A0="
            f"{m['official_board_size_inflation_vs_A0']:+.4f}",
            flush=True,
        )

    print("\nFIXED THRESHOLD COMPARISON", flush=True)
    print("--------------------------", flush=True)

    for cell in CELLS:
        m = aggregate["metrics"][cell]

        print(
            f"{cell}: "
            f"threshold=0.63 "
            f"rows={m['official_rows']:,} "
            f"hits={m['official_hits']:,} "
            f"hit_rate={m['official_hit_rate']} "
            f"mean_prob={m['official_mean_prob']} "
            f"inflation_vs_A0="
            f"{m['official_board_size_inflation_vs_A0']:+.4f}",
            flush=True,
        )

    print("\nFIXED COVERAGE COMPARISON", flush=True)
    print("-------------------------", flush=True)

    for cell in CELLS:
        fc = aggregate["fixed_coverage"][cell]

        print(
            f"{cell}: "
            f"rows={fc['rows']:,} "
            f"hits={fc['hits']:,} "
            f"hit_rate={fc['hit_rate']} "
            f"mean_prob={fc['mean_probability']}",
            flush=True,
        )

    print("\nZERO-HIT POISSON AUDIT", flush=True)
    print("----------------------", flush=True)

    for cell in ("A0", "A1"):
        print(f"\n[{cell}]", flush=True)

        for row in aggregate["zero_hit_audit"][cell]:
            print(
                f"{row['band']}: "
                f"n={row['rows']:,} "
                f"mean_lambda={row['mean_lambda']} "
                f"observed_zero={row['observed_zero_rate']} "
                f"poisson_zero={row['poisson_implied_zero_rate']} "
                f"gap={row['gap_observed_minus_poisson']}",
                flush=True,
            )

    print("\nMONTHLY STABILITY SUMMARY", flush=True)
    print("-------------------------", flush=True)

    for month, row in aggregate["monthly"].items():
        parts = []

        for cell in CELLS:
            m = row["cells"][cell]
            parts.append(
                f"{cell}:brier={m['brier']:.6f},"
                f"official={m['official_hit_rate']}"
            )

        print(
            f"{month} n={row['rows']:,} "
            + " | ".join(parts),
            flush=True,
        )

    architecture = architecture_read(
        aggregate["metrics"],
        aggregate["fixed_coverage"],
    )

    print("\nARCHITECTURE READ", flush=True)
    print("-----------------", flush=True)

    print(
        "best_cell_by_metric="
        + json.dumps(
            architecture["best_cell_by_metric"],
            sort_keys=True,
        ),
        flush=True,
    )

    for arch, row in architecture[
        "feature_effect_within_architecture"
    ].items():
        print(
            f"{arch} feature_delta: "
            f"brier={row['brier_delta']:+.8f} "
            f"logloss={row['logloss_delta']:+.8f} "
            f"ece={row['ece_delta']:+.8f} "
            f"auc={row['auc_delta']:+.6f} "
            f"top5={row['top5_delta']:+.6f} "
            f"top10={row['top10_delta']:+.6f} "
            f"official_hit_rate={row['official_hit_rate_delta']} "
            f"official_rows_delta={row['official_rows_delta']:+d} "
            f"fixed_coverage_hit_rate="
            f"{row['fixed_coverage_hit_rate_delta']}",
            flush=True,
        )

    for arch, row in architecture[
        "diagnostic_feature_survival"
    ].items():
        print(
            f"{arch} diagnostic_feature_survival="
            f"{row['all_protected']} "
            f"details={json.dumps(row, sort_keys=True)}",
            flush=True,
        )

    print("\nFINAL LAB STATUS", flush=True)
    print("----------------", flush=True)
    print("status=DEVELOPMENT_LAB_COMPLETE", flush=True)
    print("production_promotion_authorized=False", flush=True)
    print("2026_touched=False", flush=True)
    print(
        "next_step=REVIEW_ARCHITECTURE_RESULTS_AND_FREEZE_ONE_ARCHITECTURE_"
        "OR_NO_CLEAR_WINNER_BEFORE_DECLARING_FUTURE_HOLDOUT_FREEZE_DATE",
        flush=True,
    )

    payload = {
        "script": "HITS_PROBABILITY_ARCHITECTURE_LAB_V1",
        "status": "DEVELOPMENT_LAB_ONLY",
        "years": [2024, 2025],
        "touch_2026": False,
        "frozen_candidate": CANDIDATE,
        "frozen_window_days": 60,
        "frozen_min_prior_swings": MIN_SAMPLE,
        "threshold_rescue": "FORBIDDEN",
        "feature_window_tuning": "FORBIDDEN",
        "monte_carlo_used": False,
        "poisson_mapping": "1-exp(-lambda)",
        "calibration_mapping": "logit(P)=a+b*lambda",
        "num_boost_round": NUM_BOOST_ROUND,
        "official_threshold": OFFICIAL_MIN,
        "fixed_coverage_reference": "A0 official count per fold",
        "outer_folds": OUTER_FOLDS,
        "source_summary": source_summary,
        "fold_summaries": aggregate["fold_summaries"],
        "metrics": aggregate["metrics"],
        "fixed_coverage": aggregate["fixed_coverage"],
        "zero_hit_audit": aggregate["zero_hit_audit"],
        "monthly": aggregate["monthly"],
        "architecture_read": architecture,
        "production_promotion_authorized": False,
    }

    OUT_JSON.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "HITS_PROBABILITY_ARCHITECTURE_LAB_V1",
        "=" * 36,
        "",
        "2024-2025 DEVELOPMENT DATA ONLY.",
        "2026 was not touched.",
        "No production promotion is authorized by this lab.",
        "",
        "SIX-CELL AGGREGATE",
        "------------------",
    ]

    for cell in CELLS:
        report_lines.extend(
            [
                "",
                cell,
                "~" * len(cell),
                json.dumps(
                    aggregate["metrics"][cell],
                    indent=2,
                ),
                "",
                "fixed_coverage:",
                json.dumps(
                    aggregate["fixed_coverage"][cell],
                    indent=2,
                ),
            ]
        )

    report_lines.extend(
        [
            "",
            "ZERO-HIT POISSON AUDIT",
            "----------------------",
            json.dumps(
                aggregate["zero_hit_audit"],
                indent=2,
            ),
            "",
            "ARCHITECTURE READ",
            "-----------------",
            json.dumps(
                architecture,
                indent=2,
            ),
            "",
            "FINAL LAB STATUS",
            "----------------",
            "DEVELOPMENT_LAB_COMPLETE",
            "PRODUCTION_PROMOTION_AUTHORIZED=False",
            "2026_TOUCHED=False",
        ]
    )

    OUT_TXT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nOUTPUTS", flush=True)
    print(OUT_JSON, flush=True)
    print(OUT_TXT, flush=True)

    return 0


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--worker-train",
        action="store_true",
    )

    parser.add_argument(
        "--worker-score",
        action="store_true",
    )

    parser.add_argument(
        "--objective",
        choices=["count", "binary"],
    )

    parser.add_argument("--train-file")
    parser.add_argument("--out-model")
    parser.add_argument("--model-path")
    parser.add_argument("--score-file")
    parser.add_argument("--out-json")

    return parser


def main():
    args = build_parser().parse_args()

    if args.worker_train:
        worker_train(args)
        return 0

    if args.worker_score:
        worker_score(args)
        return 0

    return main_parent()


if __name__ == "__main__":
    raise SystemExit(main())
