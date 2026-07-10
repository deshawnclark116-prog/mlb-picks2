#!/usr/bin/env python3
"""
HITS_RECENT5_PA_SINGLE_SCALAR_GATE_A

Formal doctrine-compliant challenger gate for the strongest raw audit signal:
    recent5_pa_avg

No stacking. No hyperparameter tuning. No 2026 rescue tuning.

Baseline:
    Locked 8-feature batter-hits incumbent.

Challenger:
    Exact same model + exactly one scalar:
        recent5_pa_avg

Rolling-forward folds:
    Train 2019-2021 -> Test 2022
    Train 2019-2022 -> Test 2023
    Train 2019-2023 -> Test 2024
    Train 2019-2024 -> Test 2025

2026:
    Burned by incumbent training. Not used as a clean holdout.

Fixed promotion gate, locked before seeing challenger results:
    1. Aggregate Brier delta <= -0.0005
    2. Aggregate logloss does not worsen
    3. Aggregate AUC delta >= -0.0005
    4. Aggregate top-5% actual hit rate does not decline
    5. Aggregate top-10% actual hit rate does not decline
    6. Official-tier hit rate at p >= 0.630 does not decline
    7. Official-tier coverage remains >= 90% of baseline coverage
    8. At least 3 of 4 forward folds improve or hold Brier
    9. No fold suffers > 1.0 percentage-point top-5% collapse
   10. No fold suffers > 1.0 percentage-point top-10% collapse

Memory safety:
    - no pandas
    - no sklearn
    - streams season JSONL
    - SQLite temp work on disk
    - fresh child process for every XGBoost train/score step
    - reuses completed baseline scores from hits_rfv_lite_b

Run:
python -u hits_recent5_pa_single_scalar_gate_a.py 2>&1 | tee /data/hr_model/hits_recent5_pa_single_scalar_gate_a.log

Outputs:
    /data/hr_model/hits_recent5_pa_single_scalar_gate_a_results.json
    /data/hr_model/hits_recent5_pa_single_scalar_gate_a_report.txt
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
from collections import deque
from pathlib import Path

BASE_DIR = Path("/data/hr_model/hits_rfv_lite_b")
WORK_DIR = Path("/data/hr_model/hits_recent5_pa_gate_a_work")
OUT_JSON = Path("/data/hr_model/hits_recent5_pa_single_scalar_gate_a_results.json")
OUT_TXT = Path("/data/hr_model/hits_recent5_pa_single_scalar_gate_a_report.txt")

OFFICIAL_MIN = 0.630
WATCHLIST_MIN = 0.606

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
CHALLENGER_FEATURES = BASE_FEATURES + ["recent5_pa_avg"]

TRAIN_PARAMS = {
    "objective": "count:poisson",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    "nthread": 2,
}

ALIASES = {
    "player_id": ["player_id", "batter_id", "person_id", "id"],
    "date": ["date", "game_date"],
    "hits": ["hits", "h"],
    "ab": ["at_bats", "atBats", "ab"],
    "pa": ["plate_appearances", "plateAppearances", "pa"],
    "bb": ["walks", "base_on_balls", "baseOnBalls", "bb"],
    "player_type": ["player_type", "type"],
    "game_id": ["game_id", "game_pk", "gamePk"],
}


def first_value(row, names, default=None):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]

    for container_name in ("stats", "hitting", "batting"):
        nested = row.get(container_name)
        if isinstance(nested, dict):
            for name in names:
                if name in nested and nested[name] is not None:
                    return nested[name]

    return default


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def discover_season_file(year):
    for path in [
        Path(f"/data/season_{year}.jsonl"),
        Path(f"/data/season_{year}.json"),
    ]:
        if path.exists():
            return path
    return None


def iter_json_records(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = ""
        pos = f.tell()

        for line in f:
            if line.strip():
                first = line.lstrip()[0]
                break

        f.seek(pos)

        if first == "[":
            raise RuntimeError(
                f"{path} is a giant JSON array. "
                "This script refuses to load it under a 512 MB memory limit."
            )

        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"JSON parse error in {path} line {line_no}: {exc}"
                ) from exc


def normalize_row(row, seq):
    ptype = str(first_value(row, ALIASES["player_type"], "") or "").lower()

    if ptype and "batter" not in ptype and "hitter" not in ptype:
        return None

    player_id = first_value(row, ALIASES["player_id"])
    game_date = first_value(row, ALIASES["date"])
    hits = first_value(row, ALIASES["hits"])
    ab = first_value(row, ALIASES["ab"])

    if player_id is None or game_date is None or hits is None or ab is None:
        return None

    bb = to_float(first_value(row, ALIASES["bb"], 0))
    pa_raw = first_value(row, ALIASES["pa"], None)
    pa = to_float(pa_raw, to_float(ab) + bb)

    return (
        str(player_id),
        str(game_date),
        str(first_value(row, ALIASES["game_id"], seq)),
        int(seq),
        to_float(hits),
        to_float(ab),
        pa,
    )


def count_lines(path):
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            count += 1
    return count


def build_recent5_sidecar(year, season_path):
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    sidecar = WORK_DIR / f"recent5_pa_{year}.txt"
    summary = WORK_DIR / f"recent5_pa_{year}_summary.json"
    baseline_libsvm = BASE_DIR / f"features_{year}.libsvm"

    if sidecar.exists() and summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        print(
            f"{year}: sidecar REUSED | eligible_rows={data['eligible_rows']:,}",
            flush=True,
        )
        return data

    if not baseline_libsvm.exists():
        raise RuntimeError(f"Missing baseline feature file: {baseline_libsvm}")

    db_path = WORK_DIR / f"spool_{year}.sqlite"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute(
        """
        CREATE TABLE games (
            player_id TEXT,
            game_date TEXT,
            game_id TEXT,
            seq INTEGER,
            hits REAL,
            ab REAL,
            pa REAL
        )
        """
    )

    raw_seen = 0
    normalized = 0
    batch = []

    for seq, row in enumerate(iter_json_records(season_path), 1):
        raw_seen += 1
        norm = normalize_row(row, seq)
        if norm is None:
            continue

        batch.append(norm)
        normalized += 1

        if len(batch) >= 2000:
            conn.executemany(
                "INSERT INTO games VALUES (?,?,?,?,?,?,?)",
                batch,
            )
            batch.clear()

    if batch:
        conn.executemany(
            "INSERT INTO games VALUES (?,?,?,?,?,?,?)",
            batch,
        )

    conn.commit()
    conn.execute(
        """
        CREATE INDEX idx_games
        ON games(player_id, game_date, game_id, seq)
        """
    )
    conn.commit()

    cursor = conn.execute(
        """
        SELECT player_id, game_date, game_id, seq, hits, ab, pa
        FROM games
        ORDER BY player_id, game_date, game_id, seq
        """
    )

    current_player = None
    cum_ab = 0.0
    games = 0
    recent_pa = deque(maxlen=5)
    eligible = 0

    with open(sidecar, "w", encoding="utf-8") as fout:
        for player_id, game_date, game_id, seq, hits, ab, pa in cursor:
            if player_id != current_player:
                current_player = player_id
                cum_ab = 0.0
                games = 0
                recent_pa = deque(maxlen=5)

            if cum_ab >= 20.0 and games >= 5:
                value = sum(recent_pa) / len(recent_pa) if recent_pa else 0.0
                fout.write(f"{value:.10g}\n")
                eligible += 1

            cum_ab += float(ab)
            games += 1
            recent_pa.append(float(pa))

    conn.close()

    try:
        db_path.unlink()
    except Exception:
        pass

    baseline_rows = count_lines(baseline_libsvm)

    if eligible != baseline_rows:
        raise RuntimeError(
            f"{year}: eligibility alignment mismatch. "
            f"recent5_sidecar={eligible:,}, baseline_libsvm={baseline_rows:,}"
        )

    data = {
        "year": year,
        "raw_seen": raw_seen,
        "normalized_rows": normalized,
        "eligible_rows": eligible,
        "baseline_rows": baseline_rows,
        "season_path": str(season_path),
        "sidecar": str(sidecar),
    }

    summary.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(
        f"{year}: raw={raw_seen:,} normalized={normalized:,} "
        f"eligible={eligible:,} alignment=PASS",
        flush=True,
    )

    return data


def build_challenger_libsvm(year):
    baseline_path = BASE_DIR / f"features_{year}.libsvm"
    sidecar_path = WORK_DIR / f"recent5_pa_{year}.txt"
    challenger_path = WORK_DIR / f"features_{year}_plus_recent5_pa.libsvm"

    if challenger_path.exists():
        print(f"{year}: challenger feature file REUSED", flush=True)
        return challenger_path

    rows = 0

    with open(baseline_path, "r", encoding="utf-8") as base_f, \
         open(sidecar_path, "r", encoding="utf-8") as side_f, \
         open(challenger_path, "w", encoding="utf-8") as out_f:

        while True:
            base_line = base_f.readline()
            side_line = side_f.readline()

            if not base_line and not side_line:
                break

            if not base_line or not side_line:
                raise RuntimeError(
                    f"{year}: baseline/sidecar length mismatch during merge."
                )

            value = float(side_line.strip())
            out_f.write(base_line.rstrip("\n") + f" 9:{value:.10g}\n")
            rows += 1

    print(f"{year}: challenger rows={rows:,}", flush=True)
    return challenger_path


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
        raise RuntimeError(f"{label} failed with exit code {cp.returncode}")


def worker_train(args):
    import xgboost as xgb

    train_uri = f"{args.train_file}?format=libsvm"
    dtrain = xgb.DMatrix(train_uri)

    previous = None
    prior_rounds = 0

    if args.prev_model and Path(args.prev_model).exists():
        previous = xgb.Booster()
        previous.load_model(args.prev_model)
        prior_rounds = int(previous.num_boosted_rounds())

    booster = xgb.train(
        TRAIN_PARAMS,
        dtrain,
        num_boost_round=60,
        xgb_model=previous,
    )

    booster.save_model(args.out_model)

    print(f"prior_rounds={prior_rounds}", flush=True)
    print("added_rounds=60", flush=True)
    print(f"total_rounds={booster.num_boosted_rounds()}", flush=True)
    print(f"saved_model={args.out_model}", flush=True)


def worker_score(args):
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(args.model_path)

    score_uri = f"{args.score_file}?format=libsvm"
    dtest = xgb.DMatrix(score_uri)
    pred = booster.predict(dtest)

    meta_path = BASE_DIR / f"features_{args.test_year}_meta.csv"

    actual_hits = []
    dates = []

    with open(meta_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            dates.append(row["game_date"])
            actual_hits.append(float(row["actual_hits"]))

    if len(pred) != len(actual_hits):
        raise RuntimeError(
            f"Prediction/meta mismatch: {len(pred)} vs {len(actual_hits)}"
        )

    probs = [
        min(
            max(1.0 - math.exp(-max(float(lam), 1e-8)), 1e-8),
            1.0 - 1e-8,
        )
        for lam in pred
    ]

    y_binary = [1 if h >= 1.0 else 0 for h in actual_hits]

    payload = {
        "test_year": int(args.test_year),
        "model_rounds": int(booster.num_boosted_rounds()),
        "dates": dates,
        "y_binary": y_binary,
        "probs": probs,
    }

    Path(args.out_score_json).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    print(f"scored_rows={len(y_binary):,}", flush=True)
    print(f"model_rounds={payload['model_rounds']}", flush=True)
    print(f"saved_scores={args.out_score_json}", flush=True)


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

    return out


def threshold_block(y, p):
    def one(indices):
        if not indices:
            return {
                "rows": 0,
                "hits": 0,
                "actual_hit_rate": None,
                "mean_probability": None,
            }

        return {
            "rows": len(indices),
            "hits": sum(y[i] for i in indices),
            "actual_hit_rate": sum(y[i] for i in indices) / len(indices),
            "mean_probability": sum(p[i] for i in indices) / len(indices),
        }

    return {
        "official_ge_0.630": one(
            [i for i, pi in enumerate(p) if pi >= OFFICIAL_MIN]
        ),
        "watchlist_0.606_to_0.630": one(
            [
                i for i, pi in enumerate(p)
                if WATCHLIST_MIN <= pi < OFFICIAL_MIN
            ]
        ),
        "all_ge_0.606": one(
            [i for i, pi in enumerate(p) if pi >= WATCHLIST_MIN]
        ),
    }


def load_score(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def print_metric(label, m):
    print(
        f"{label}: "
        f"n={m['rows']:,} "
        f"brier={m['brier']:.8f} "
        f"logloss={m['logloss']:.8f} "
        f"auc={m['auc']:.6f} "
        f"top5={m['top5_actual']:.4f} "
        f"top10={m['top10_actual']:.4f}",
        flush=True,
    )


def main_parent():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print("HITS_RECENT5_PA_SINGLE_SCALAR_GATE_A", flush=True)
    print("====================================", flush=True)

    print("\nGATE PREFLIGHT", flush=True)
    print("--------------", flush=True)
    print("candidate=recent5_pa_avg", flush=True)
    print("features_added=1", flush=True)
    print("stacking=FORBIDDEN", flush=True)
    print("2026_use=FORBIDDEN_AS_CLEAN_HOLDOUT", flush=True)
    print("hyperparameter_tuning=NONE", flush=True)

    season_files = {}

    for year in range(2019, 2026):
        season_path = discover_season_file(year)

        if season_path is None:
            raise RuntimeError(f"Missing season file for {year}")

        baseline_feature = BASE_DIR / f"features_{year}.libsvm"
        if not baseline_feature.exists():
            raise RuntimeError(
                f"Missing completed baseline feature file: {baseline_feature}"
            )

        season_files[year] = season_path

        print(
            f"{year}: season={season_path} "
            f"baseline_features={baseline_feature}",
            flush=True,
        )

    print("\nBUILDING SINGLE-SCALAR FEATURE FILES", flush=True)
    print("------------------------------------", flush=True)

    feature_summaries = {}

    for year in range(2019, 2026):
        summary = build_recent5_sidecar(year, season_files[year])
        build_challenger_libsvm(year)
        feature_summaries[str(year)] = summary

    print("\nROLLING FORWARD CHALLENGER TRAINING", flush=True)
    print("-----------------------------------", flush=True)

    current_model = None
    fold_results = []

    pooled_base_y = []
    pooled_base_p = []
    pooled_chal_y = []
    pooled_chal_p = []

    for train_year in range(2019, 2025):
        model_path = WORK_DIR / f"challenger_through_{train_year}.json"
        train_file = (
            WORK_DIR
            / f"features_{train_year}_plus_recent5_pa.libsvm"
        )

        args = [
            "--worker-train",
            "--train-file", str(train_file),
            "--out-model", str(model_path),
        ]

        if current_model is not None:
            args += ["--prev-model", str(current_model)]

        run_child(
            args,
            f"TRAIN CHALLENGER THROUGH {train_year}",
        )

        current_model = model_path
        test_year = train_year + 1

        if test_year < 2022 or test_year > 2025:
            continue

        challenger_score_path = (
            WORK_DIR / f"challenger_scores_{test_year}.json"
        )

        run_child(
            [
                "--worker-score",
                "--model-path", str(current_model),
                "--score-file", str(
                    WORK_DIR
                    / f"features_{test_year}_plus_recent5_pa.libsvm"
                ),
                "--test-year", str(test_year),
                "--out-score-json", str(challenger_score_path),
            ],
            f"SCORE CHALLENGER FORWARD HOLDOUT {test_year}",
        )

        baseline_score_path = BASE_DIR / f"scores_{test_year}.json"

        if not baseline_score_path.exists():
            raise RuntimeError(
                f"Missing baseline score file: {baseline_score_path}"
            )

        baseline = load_score(baseline_score_path)
        challenger = load_score(challenger_score_path)

        if baseline["y_binary"] != challenger["y_binary"]:
            raise RuntimeError(
                f"{test_year}: baseline/challenger actual-label alignment failed."
            )

        y = baseline["y_binary"]
        base_p = baseline["probs"]
        chal_p = challenger["probs"]

        base_m = metric_block(y, base_p)
        chal_m = metric_block(y, chal_p)

        base_t = threshold_block(y, base_p)
        chal_t = threshold_block(y, chal_p)

        deltas = {
            "brier": chal_m["brier"] - base_m["brier"],
            "logloss": chal_m["logloss"] - base_m["logloss"],
            "auc": chal_m["auc"] - base_m["auc"],
            "top5_actual": (
                chal_m["top5_actual"] - base_m["top5_actual"]
            ),
            "top10_actual": (
                chal_m["top10_actual"] - base_m["top10_actual"]
            ),
        }

        fold_results.append(
            {
                "train_start": 2019,
                "train_end": train_year,
                "test_year": test_year,
                "baseline": base_m,
                "challenger": chal_m,
                "baseline_thresholds": base_t,
                "challenger_thresholds": chal_t,
                "deltas": deltas,
            }
        )

        pooled_base_y.extend(y)
        pooled_base_p.extend(base_p)
        pooled_chal_y.extend(y)
        pooled_chal_p.extend(chal_p)

    baseline_agg = metric_block(pooled_base_y, pooled_base_p)
    challenger_agg = metric_block(pooled_chal_y, pooled_chal_p)

    baseline_thr = threshold_block(pooled_base_y, pooled_base_p)
    challenger_thr = threshold_block(pooled_chal_y, pooled_chal_p)

    aggregate_deltas = {
        "brier": challenger_agg["brier"] - baseline_agg["brier"],
        "logloss": challenger_agg["logloss"] - baseline_agg["logloss"],
        "auc": challenger_agg["auc"] - baseline_agg["auc"],
        "top5_actual": (
            challenger_agg["top5_actual"]
            - baseline_agg["top5_actual"]
        ),
        "top10_actual": (
            challenger_agg["top10_actual"]
            - baseline_agg["top10_actual"]
        ),
    }

    base_official = baseline_thr["official_ge_0.630"]
    chal_official = challenger_thr["official_ge_0.630"]

    official_hit_rate_delta = (
        chal_official["actual_hit_rate"]
        - base_official["actual_hit_rate"]
    )

    official_coverage_ratio = (
        chal_official["rows"] / base_official["rows"]
        if base_official["rows"] > 0
        else None
    )

    brier_folds_nonworse = sum(
        1
        for fold in fold_results
        if fold["deltas"]["brier"] <= 0
    )

    worst_fold_top5_delta = min(
        fold["deltas"]["top5_actual"]
        for fold in fold_results
    )

    worst_fold_top10_delta = min(
        fold["deltas"]["top10_actual"]
        for fold in fold_results
    )

    gate = {
        "brier_pass": aggregate_deltas["brier"] <= -0.0005,
        "logloss_pass": aggregate_deltas["logloss"] <= 0.0,
        "auc_pass": aggregate_deltas["auc"] >= -0.0005,
        "top5_pass": aggregate_deltas["top5_actual"] >= 0.0,
        "top10_pass": aggregate_deltas["top10_actual"] >= 0.0,
        "official_hit_rate_pass": official_hit_rate_delta >= 0.0,
        "official_coverage_pass": (
            official_coverage_ratio is not None
            and official_coverage_ratio >= 0.90
        ),
        "fold_brier_pass": brier_folds_nonworse >= 3,
        "no_top5_collapse_pass": worst_fold_top5_delta >= -0.01,
        "no_top10_collapse_pass": worst_fold_top10_delta >= -0.01,
    }

    gate["overall_pass"] = all(gate.values())

    print("\nROLLING FORWARD COMPARISON", flush=True)
    print("--------------------------", flush=True)

    for fold in fold_results:
        year = fold["test_year"]
        print_metric(
            f"{year} BASELINE",
            fold["baseline"],
        )
        print_metric(
            f"{year} CHALLENGER",
            fold["challenger"],
        )
        d = fold["deltas"]
        print(
            f"{year} DELTA: "
            f"brier={d['brier']:+.8f} "
            f"logloss={d['logloss']:+.8f} "
            f"auc={d['auc']:+.6f} "
            f"top5={d['top5_actual']:+.4f} "
            f"top10={d['top10_actual']:+.4f}",
            flush=True,
        )

    print("\nAGGREGATE BASELINE VS CHALLENGER", flush=True)
    print("--------------------------------", flush=True)

    print_metric("BASELINE", baseline_agg)
    print_metric("CHALLENGER", challenger_agg)

    print(
        "DELTA: "
        f"brier={aggregate_deltas['brier']:+.8f} "
        f"logloss={aggregate_deltas['logloss']:+.8f} "
        f"auc={aggregate_deltas['auc']:+.6f} "
        f"top5={aggregate_deltas['top5_actual']:+.4f} "
        f"top10={aggregate_deltas['top10_actual']:+.4f}",
        flush=True,
    )

    print("\nOFFICIAL-TIER COMPARISON", flush=True)
    print("------------------------", flush=True)

    print(
        f"BASELINE official: "
        f"n={base_official['rows']:,} "
        f"hit_rate={base_official['actual_hit_rate']:.4f} "
        f"mean_prob={base_official['mean_probability']:.4f}",
        flush=True,
    )

    print(
        f"CHALLENGER official: "
        f"n={chal_official['rows']:,} "
        f"hit_rate={chal_official['actual_hit_rate']:.4f} "
        f"mean_prob={chal_official['mean_probability']:.4f}",
        flush=True,
    )

    print(
        f"official_hit_rate_delta={official_hit_rate_delta:+.6f}",
        flush=True,
    )

    print(
        f"official_coverage_ratio={official_coverage_ratio:.4f}",
        flush=True,
    )

    print("\nSTRICT GATE READ", flush=True)
    print("----------------", flush=True)

    print(
        f"aggregate_brier_delta={aggregate_deltas['brier']:+.8f}",
        flush=True,
    )
    print(
        f"aggregate_logloss_delta={aggregate_deltas['logloss']:+.8f}",
        flush=True,
    )
    print(
        f"aggregate_auc_delta={aggregate_deltas['auc']:+.6f}",
        flush=True,
    )
    print(
        f"aggregate_top5_delta={aggregate_deltas['top5_actual']:+.6f}",
        flush=True,
    )
    print(
        f"aggregate_top10_delta={aggregate_deltas['top10_actual']:+.6f}",
        flush=True,
    )
    print(
        f"brier_folds_nonworse={brier_folds_nonworse}/4",
        flush=True,
    )
    print(
        f"worst_fold_top5_delta={worst_fold_top5_delta:+.6f}",
        flush=True,
    )
    print(
        f"worst_fold_top10_delta={worst_fold_top10_delta:+.6f}",
        flush=True,
    )

    for key, value in gate.items():
        print(f"{key}: {value}", flush=True)

    verdict = (
        "RECENT5_PA_SINGLE_SCALAR_PASSES_PROMOTION_GATE"
        if gate["overall_pass"]
        else "RECENT5_PA_SINGLE_SCALAR_FAILS_PROMOTION_GATE"
    )

    print(f"verdict: {verdict}", flush=True)

    if gate["overall_pass"]:
        next_step = (
            "Lock recent5_pa_avg as the only promoted hits challenger scalar. "
            "Then run one confirmation report before any production integration."
        )
    else:
        next_step = (
            "Freeze recent5_pa_avg. Do not rescue-tune it on these holdouts. "
            "The next candidate may be tested only as a separate preauthorized branch."
        )

    print(f"next_step: {next_step}", flush=True)

    payload = {
        "script": "HITS_RECENT5_PA_SINGLE_SCALAR_GATE_A",
        "candidate": "recent5_pa_avg",
        "baseline_features": BASE_FEATURES,
        "challenger_features": CHALLENGER_FEATURES,
        "train_params": TRAIN_PARAMS,
        "fold_results": fold_results,
        "aggregate_baseline": baseline_agg,
        "aggregate_challenger": challenger_agg,
        "aggregate_deltas": aggregate_deltas,
        "baseline_thresholds": baseline_thr,
        "challenger_thresholds": challenger_thr,
        "official_hit_rate_delta": official_hit_rate_delta,
        "official_coverage_ratio": official_coverage_ratio,
        "brier_folds_nonworse": brier_folds_nonworse,
        "worst_fold_top5_delta": worst_fold_top5_delta,
        "worst_fold_top10_delta": worst_fold_top10_delta,
        "fixed_gate": {
            "brier_delta_required": -0.0005,
            "logloss_max_delta": 0.0,
            "auc_min_delta": -0.0005,
            "top5_min_delta": 0.0,
            "top10_min_delta": 0.0,
            "official_hit_rate_min_delta": 0.0,
            "official_coverage_min_ratio": 0.90,
            "minimum_nonworse_brier_folds": 3,
            "worst_allowed_fold_top5_delta": -0.01,
            "worst_allowed_fold_top10_delta": -0.01,
        },
        "gate": gate,
        "verdict": verdict,
        "next_step": next_step,
        "2026_clean_holdout": "NO_BURNED",
        "rescue_tuning": "FORBIDDEN",
    }

    OUT_JSON.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "HITS_RECENT5_PA_SINGLE_SCALAR_GATE_A",
        "=" * 36,
        "",
        "Candidate: recent5_pa_avg",
        "No stacking. No 2026 rescue tuning.",
        "",
        "AGGREGATE BASELINE VS CHALLENGER",
        "--------------------------------",
        json.dumps(
            {
                "baseline": baseline_agg,
                "challenger": challenger_agg,
                "deltas": aggregate_deltas,
            },
            indent=2,
        ),
        "",
        "OFFICIAL TIER",
        "-------------",
        json.dumps(
            {
                "baseline": base_official,
                "challenger": chal_official,
                "hit_rate_delta": official_hit_rate_delta,
                "coverage_ratio": official_coverage_ratio,
            },
            indent=2,
        ),
        "",
        "STRICT GATE",
        "-----------",
        json.dumps(gate, indent=2),
        "",
        f"VERDICT: {verdict}",
        f"NEXT STEP: {next_step}",
    ]

    OUT_TXT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nOUTPUTS", flush=True)
    print(OUT_JSON, flush=True)
    print(OUT_TXT, flush=True)

    return 0


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--worker-train", action="store_true")
    parser.add_argument("--worker-score", action="store_true")

    parser.add_argument("--train-file")
    parser.add_argument("--prev-model")
    parser.add_argument("--out-model")
    parser.add_argument("--model-path")
    parser.add_argument("--score-file")
    parser.add_argument("--test-year", type=int)
    parser.add_argument("--out-score-json")

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
