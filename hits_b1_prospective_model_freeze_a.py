#!/usr/bin/env python3
"""
HITS_B1_PROSPECTIVE_MODEL_FREEZE_A

Purpose
-------
Train, freeze, hash, and manifest the exact three model artifacts needed for the
blind prospective batter-hits holdout:

    A0 = 8-feature count-Poisson comparator
         probability later computed as exact 1 - exp(-lambda)

    B0 = 8-feature direct-binary architecture control

    B1 = 8-feature direct-binary challenger
         + frozen opp_pitcher_contact_allowed_rate_60d

This script is designed for the existing 512 MB Render environment.

Doctrine / integrity
--------------------
- Uses 2024-2025 DEVELOPMENT DATA ONLY.
- Does not touch 2026 rows.
- Does not inspect any post-freeze outcomes.
- Does not tune thresholds.
- Does not change the 60-day pitcher-contact window.
- Does not change the 150-prior-swing minimum.
- Does not stack opportunity features.
- Uses exactly the architecture frozen in the prospective holdout lock.
- Trains one XGBoost child process at a time.
- Refuses to overwrite a completed immutable model freeze.
- Hashes every final artifact.
- Binds the model manifest to the immutable holdout-lock SHA-256.

Expected inputs
---------------
/data/hr_model/hits_b1_prospective_holdout_lock_a.json
/data/hr_model/hits_b1_prospective_holdout_lock_a.sha256

/data/hr_model/hits_rfv_lite_b/features_2024.libsvm
/data/hr_model/hits_rfv_lite_b/features_2025.libsvm

/data/hr_model/hits_raw_structural_audit_a_work/audit.sqlite

Frozen audit columns
--------------------
candidate:
    opp_pitcher_contact_allowed_rate_60d

sample-size gate:
    p_swings_60d >= 150

Frozen XGBoost parameters
-------------------------
learning_rate      = 0.05
max_depth          = 6
subsample          = 0.8
colsample_bytree   = 0.8
min_child_weight   = 5
tree_method        = hist
nthread            = 2
seed               = 42
num_boost_round    = 120

Final immutable output directory
--------------------------------
/data/hr_model/hits_b1_prospective_models_a/

Final artifacts
---------------
A0_count_poisson.json
A0_columns.json

B0_direct_binary.json
B0_columns.json

B1_direct_binary_pitcher_contact.json
B1_columns.json

model_spec.json
manifest.json
manifest.sha256

Run
---
python -u hits_b1_prospective_model_freeze_a.py 2>&1 | tee /data/hr_model/hits_b1_prospective_model_freeze_a.log

Verify later
------------
python -u hits_b1_prospective_model_freeze_a.py --verify
"""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# =============================================================================
# Paths / frozen constants
# =============================================================================

LOCK_JSON = Path("/data/hr_model/hits_b1_prospective_holdout_lock_a.json")
LOCK_SHA = Path("/data/hr_model/hits_b1_prospective_holdout_lock_a.sha256")

BASE_DIR = Path("/data/hr_model/hits_rfv_lite_b")
AUDIT_DB = Path("/data/hr_model/hits_raw_structural_audit_a_work/audit.sqlite")

WORK_DIR = Path("/data/hr_model/hits_b1_prospective_models_a_work")
FINAL_DIR = Path("/data/hr_model/hits_b1_prospective_models_a")

FINAL_MANIFEST = FINAL_DIR / "manifest.json"
FINAL_MANIFEST_SHA = FINAL_DIR / "manifest.sha256"

YEARS = [2024, 2025]

CANDIDATE = "opp_pitcher_contact_allowed_rate_60d"
SAMPLE_COL = "p_swings_60d"
MIN_SAMPLE = 150

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

B1_FEATURES = BASE_FEATURES + [CANDIDATE]

NUM_BOOST_ROUND = 120

COMMON_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    "nthread": 2,
    "seed": 42,
}

MODEL_SPECS = {
    "A0": {
        "filename": "A0_count_poisson.json",
        "columns_filename": "A0_columns.json",
        "objective": "count:poisson",
        "target": "actual_hits_count",
        "features": BASE_FEATURES,
        "probability_mapping": "1-exp(-lambda)",
        "train_file": "train_A0_count.libsvm",
    },
    "B0": {
        "filename": "B0_direct_binary.json",
        "columns_filename": "B0_columns.json",
        "objective": "binary:logistic",
        "target": "1_if_actual_hits_ge_1_else_0",
        "features": BASE_FEATURES,
        "probability_mapping": "direct_binary_probability",
        "train_file": "train_B0_binary.libsvm",
    },
    "B1": {
        "filename": "B1_direct_binary_pitcher_contact.json",
        "columns_filename": "B1_columns.json",
        "objective": "binary:logistic",
        "target": "1_if_actual_hits_ge_1_else_0",
        "features": B1_FEATURES,
        "probability_mapping": "direct_binary_probability",
        "train_file": "train_B1_binary_pitcher_contact.libsvm",
    },
}


# =============================================================================
# Hash / JSON helpers
# =============================================================================

def canonical_json_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# =============================================================================
# Holdout lock verification
# =============================================================================

def verify_holdout_lock():
    if not LOCK_JSON.exists():
        raise RuntimeError(f"Missing holdout lock JSON: {LOCK_JSON}")

    if not LOCK_SHA.exists():
        raise RuntimeError(f"Missing holdout lock SHA file: {LOCK_SHA}")

    payload = load_json(LOCK_JSON)
    actual = sha256_bytes(canonical_json_bytes(payload))
    expected = LOCK_SHA.read_text(
        encoding="utf-8"
    ).strip().split()[0]

    if actual != expected:
        raise RuntimeError(
            "Holdout lock hash mismatch. Refusing to train prospective models."
        )

    challenger = payload.get("challenger", {})
    extra = challenger.get("extra_feature", {})
    params = challenger.get("xgboost_params", {})

    expected_checks = {
        "status": payload.get("status") == "FROZEN_PROSPECTIVE_CHALLENGER",
        "target": challenger.get("target") == "actual_hit",
        "architecture": challenger.get("architecture") == "direct_binary_xgboost",
        "base_features": challenger.get("base_features") == BASE_FEATURES,
        "extra_feature_name": extra.get("name") == CANDIDATE,
        "window_days": int(extra.get("window_days", -1)) == 60,
        "minimum_prior_pitcher_swings": (
            int(extra.get("minimum_prior_pitcher_swings", -1)) == MIN_SAMPLE
        ),
        "objective": params.get("objective") == "binary:logistic",
        "learning_rate": float(params.get("learning_rate", -1)) == 0.05,
        "max_depth": int(params.get("max_depth", -1)) == 6,
        "subsample": float(params.get("subsample", -1)) == 0.8,
        "colsample_bytree": float(params.get("colsample_bytree", -1)) == 0.8,
        "min_child_weight": int(params.get("min_child_weight", -1)) == 5,
        "tree_method": params.get("tree_method") == "hist",
        "seed": int(params.get("seed", -1)) == 42,
        "num_boost_round": int(params.get("num_boost_round", -1)) == 120,
    }

    failed = [
        key
        for key, passed in expected_checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "Frozen holdout specification mismatch for: "
            + ", ".join(failed)
        )

    return {
        "verified": True,
        "sha256": actual,
        "freeze_timestamp_utc": payload.get("freeze_timestamp_utc"),
        "holdout_start_game_date": payload.get(
            "holdout_start_game_date_america_new_york"
        ),
        "trigger_rows": payload.get(
            "holdout_trigger", {}
        ).get("minimum_rows"),
        "spec_checks": expected_checks,
    }


# =============================================================================
# Source validation / exact training-file construction
# =============================================================================

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


def audit_columns(conn):
    return {
        row[1]
        for row in conn.execute("PRAGMA table_info(audit_joined)")
    }


def validate_sources():
    required = [
        LOCK_JSON,
        LOCK_SHA,
        AUDIT_DB,
    ]

    for year in YEARS:
        required.append(BASE_DIR / f"features_{year}.libsvm")

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required inputs:\n" + "\n".join(missing)
        )

    conn = sqlite3.connect(str(AUDIT_DB))
    cols = audit_columns(conn)
    conn.close()

    needed = {"year", "row_id", CANDIDATE, SAMPLE_COL}
    missing_cols = needed - cols

    if missing_cols:
        raise RuntimeError(
            "Audit DB is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )


def build_training_files():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    path_a0 = WORK_DIR / MODEL_SPECS["A0"]["train_file"]
    path_b0 = WORK_DIR / MODEL_SPECS["B0"]["train_file"]
    path_b1 = WORK_DIR / MODEL_SPECS["B1"]["train_file"]

    completion = WORK_DIR / "training_files_complete.json"

    if (
        completion.exists()
        and path_a0.exists()
        and path_b0.exists()
        and path_b1.exists()
    ):
        metadata = load_json(completion)

        # Verify staged files still match recorded hashes.
        current_hashes = {
            "A0": sha256_file(path_a0),
            "B0": sha256_file(path_b0),
            "B1": sha256_file(path_b1),
        }

        if current_hashes == metadata.get("training_file_sha256"):
            print("Training files already complete and hash-verified; reusing.")
            return metadata

        raise RuntimeError(
            "Existing staged training files do not match their recorded hashes."
        )

    # Remove partial staged files before rebuilding.
    for path in (path_a0, path_b0, path_b1, completion):
        if path.exists():
            path.unlink()

    total_rows = 0
    total_positive_binary = 0
    candidate_valid_rows = 0
    per_year = {}

    conn = sqlite3.connect(str(AUDIT_DB))

    try:
        with (
            open(path_a0, "w", encoding="utf-8") as fout_a0,
            open(path_b0, "w", encoding="utf-8") as fout_b0,
            open(path_b1, "w", encoding="utf-8") as fout_b1,
        ):
            for year in YEARS:
                feature_path = BASE_DIR / f"features_{year}.libsvm"

                cursor = conn.execute(
                    f"""
                    SELECT row_id, {CANDIDATE}, {SAMPLE_COL}
                    FROM audit_joined
                    WHERE year = ?
                    ORDER BY row_id
                    """,
                    (year,),
                )

                year_rows = 0
                year_positive = 0
                year_candidate_valid = 0

                with open(
                    feature_path,
                    "r",
                    encoding="utf-8",
                ) as fin:
                    for row_id, (line, audit_row) in enumerate(
                        zip(fin, cursor)
                    ):
                        audit_row_id, candidate_value, sample_n = audit_row

                        if int(audit_row_id) != row_id:
                            raise RuntimeError(
                                f"{year}: audit row alignment failed at "
                                f"feature row {row_id}; audit row_id={audit_row_id}"
                            )

                        actual_hits = parse_libsvm_label(line)
                        binary_target = 1.0 if actual_hits >= 1.0 else 0.0

                        frozen_candidate_value = None

                        if (
                            candidate_value is not None
                            and sample_n is not None
                            and float(sample_n) >= MIN_SAMPLE
                        ):
                            frozen_candidate_value = float(candidate_value)
                            candidate_valid_rows += 1
                            year_candidate_valid += 1

                        # A0: count target, base 8 features.
                        fout_a0.write(
                            relabel_libsvm_line(
                                line,
                                actual_hits,
                            )
                        )

                        # B0: binary target, base 8 features.
                        fout_b0.write(
                            relabel_libsvm_line(
                                line,
                                binary_target,
                            )
                        )

                        # B1: binary target, base 8 + feature index 9 when valid.
                        binary_line = relabel_libsvm_line(
                            line,
                            binary_target,
                        )

                        fout_b1.write(
                            append_feature(
                                binary_line,
                                9,
                                frozen_candidate_value,
                            )
                        )

                        total_rows += 1
                        year_rows += 1

                        if binary_target == 1.0:
                            total_positive_binary += 1
                            year_positive += 1

                # Make sure audit query had no extra rows.
                extra_audit_row = cursor.fetchone()

                if extra_audit_row is not None:
                    raise RuntimeError(
                        f"{year}: audit table contains more rows than "
                        f"features file."
                    )

                per_year[str(year)] = {
                    "rows": year_rows,
                    "binary_positive_rows": year_positive,
                    "binary_positive_rate": (
                        year_positive / year_rows
                        if year_rows
                        else None
                    ),
                    "candidate_valid_rows": year_candidate_valid,
                    "candidate_coverage": (
                        year_candidate_valid / year_rows
                        if year_rows
                        else None
                    ),
                    "source_feature_file": str(feature_path),
                    "source_feature_file_size": feature_path.stat().st_size,
                    "source_feature_file_sha256": sha256_file(feature_path),
                }

                print(
                    f"{year}: rows={year_rows:,} "
                    f"binary_positive_rate="
                    f"{per_year[str(year)]['binary_positive_rate']:.6f} "
                    f"candidate_coverage="
                    f"{per_year[str(year)]['candidate_coverage']:.6f}",
                    flush=True,
                )
    finally:
        conn.close()

    training_hashes = {
        "A0": sha256_file(path_a0),
        "B0": sha256_file(path_b0),
        "B1": sha256_file(path_b1),
    }

    metadata = {
        "years": YEARS,
        "total_rows": total_rows,
        "binary_positive_rows": total_positive_binary,
        "binary_positive_rate": (
            total_positive_binary / total_rows
            if total_rows
            else None
        ),
        "candidate_valid_rows": candidate_valid_rows,
        "candidate_coverage": (
            candidate_valid_rows / total_rows
            if total_rows
            else None
        ),
        "per_year": per_year,
        "training_files": {
            "A0": str(path_a0),
            "B0": str(path_b0),
            "B1": str(path_b1),
        },
        "training_file_sha256": training_hashes,
    }

    completion.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata


# =============================================================================
# XGBoost child-process training
# =============================================================================

def run_child(args, label):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    print("")
    print(label)
    print("-" * len(label))

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

    params = dict(COMMON_PARAMS)
    params["objective"] = args.objective

    if args.objective == "binary:logistic":
        params["eval_metric"] = "logloss"

    dtrain = xgb.DMatrix(
        f"{args.train_file}?format=libsvm"
    )

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
    )

    booster.save_model(args.out_model)

    summary = {
        "objective": args.objective,
        "rows": int(dtrain.num_row()),
        "columns": int(dtrain.num_col()),
        "boosted_rounds": int(booster.num_boosted_rounds()),
        "model_path": args.out_model,
    }

    print(json.dumps(summary, indent=2), flush=True)


def train_models(training_metadata):
    model_outputs = {}

    for cell in ("A0", "B0", "B1"):
        spec = MODEL_SPECS[cell]
        model_path = WORK_DIR / spec["filename"]
        train_path = WORK_DIR / spec["train_file"]

        if not train_path.exists():
            raise RuntimeError(
                f"Missing staged training file for {cell}: {train_path}"
            )

        expected_train_hash = training_metadata[
            "training_file_sha256"
        ][cell]

        current_train_hash = sha256_file(train_path)

        if current_train_hash != expected_train_hash:
            raise RuntimeError(
                f"{cell}: staged training file hash mismatch."
            )

        if model_path.exists():
            print(
                f"{cell}: existing work model found; verifying basic load.",
                flush=True,
            )

            # Use a tiny child verification path to avoid keeping xgboost loaded
            # in the parent process.
            verify_json = WORK_DIR / f"{cell}_work_model_verify.json"

            run_child(
                [
                    "--worker-verify-model",
                    "--model-path", str(model_path),
                    "--out-json", str(verify_json),
                ],
                f"{cell} VERIFY EXISTING WORK MODEL",
            )

            verification = load_json(verify_json)

            if verification.get("boosted_rounds") != NUM_BOOST_ROUND:
                raise RuntimeError(
                    f"{cell}: existing work model has wrong boosted-round count."
                )
        else:
            run_child(
                [
                    "--worker-train",
                    "--objective", spec["objective"],
                    "--train-file", str(train_path),
                    "--out-model", str(model_path),
                ],
                f"{cell} TRAIN {spec['objective']}",
            )

        model_outputs[cell] = {
            "work_model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "model_size_bytes": model_path.stat().st_size,
        }

    return model_outputs


def worker_verify_model(args):
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(args.model_path)

    payload = {
        "model_path": args.model_path,
        "boosted_rounds": int(booster.num_boosted_rounds()),
        "feature_names": booster.feature_names,
    }

    Path(args.out_json).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2), flush=True)


# =============================================================================
# Final immutable freeze
# =============================================================================

def model_spec_payload(lock_info, training_metadata):
    return {
        "freeze_version": "hits_b1_prospective_model_freeze_a_v1",
        "created_at_utc": datetime.now(
            timezone.utc
        ).replace(microsecond=0).isoformat(),
        "development_years_only": YEARS,
        "touch_2026": False,
        "post_freeze_outcomes_inspected": False,
        "holdout_lock_sha256": lock_info["sha256"],
        "holdout_freeze_timestamp_utc": (
            lock_info["freeze_timestamp_utc"]
        ),
        "holdout_start_game_date": (
            lock_info["holdout_start_game_date"]
        ),
        "holdout_trigger_rows": lock_info["trigger_rows"],
        "frozen_candidate": CANDIDATE,
        "frozen_window_days": 60,
        "frozen_minimum_prior_pitcher_swings": MIN_SAMPLE,
        "base_features": BASE_FEATURES,
        "b1_features": B1_FEATURES,
        "common_xgboost_params": COMMON_PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "models": MODEL_SPECS,
        "training_data_summary": training_metadata,
        "rescue_tuning": False,
        "threshold_tuning": False,
        "stacking": False,
        "production_promotion_authorized": False,
        "status": "FROZEN_PROSPECTIVE_MODEL_ARTIFACTS",
    }


def finalize_immutable_freeze(
    lock_info,
    training_metadata,
    model_outputs,
):
    if FINAL_MANIFEST.exists():
        raise RuntimeError(
            "Final immutable manifest already exists. Use --verify."
        )

    if FINAL_DIR.exists() and any(FINAL_DIR.iterdir()):
        raise RuntimeError(
            f"Final directory exists and is non-empty without a completed "
            f"manifest: {FINAL_DIR}. Refusing to overwrite."
        )

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    spec_payload = model_spec_payload(
        lock_info,
        training_metadata,
    )

    spec_path = FINAL_DIR / "model_spec.json"
    spec_path.write_text(
        json.dumps(spec_payload, indent=2),
        encoding="utf-8",
    )

    artifacts = {}

    for cell in ("A0", "B0", "B1"):
        spec = MODEL_SPECS[cell]

        source_model = Path(
            model_outputs[cell]["work_model_path"]
        )
        final_model = FINAL_DIR / spec["filename"]
        final_columns = FINAL_DIR / spec["columns_filename"]

        shutil.copy2(source_model, final_model)

        final_columns.write_text(
            json.dumps(spec["features"], indent=2),
            encoding="utf-8",
        )

        artifacts[spec["filename"]] = {
            "sha256": sha256_file(final_model),
            "size_bytes": final_model.stat().st_size,
            "cell": cell,
            "type": "xgboost_model",
        }

        artifacts[spec["columns_filename"]] = {
            "sha256": sha256_file(final_columns),
            "size_bytes": final_columns.stat().st_size,
            "cell": cell,
            "type": "feature_columns",
        }

    artifacts["model_spec.json"] = {
        "sha256": sha256_file(spec_path),
        "size_bytes": spec_path.stat().st_size,
        "type": "frozen_model_spec",
    }

    manifest = {
        "manifest_version": "hits_b1_prospective_model_manifest_a_v1",
        "status": "IMMUTABLE_PROSPECTIVE_MODEL_FREEZE_COMPLETE",
        "created_at_utc": datetime.now(
            timezone.utc
        ).replace(microsecond=0).isoformat(),
        "holdout_lock_sha256": lock_info["sha256"],
        "holdout_start_game_date": (
            lock_info["holdout_start_game_date"]
        ),
        "development_years": YEARS,
        "touch_2026": False,
        "post_freeze_outcomes_inspected": False,
        "artifacts": artifacts,
        "model_cells": {
            cell: {
                "model_file": MODEL_SPECS[cell]["filename"],
                "columns_file": MODEL_SPECS[cell]["columns_filename"],
                "objective": MODEL_SPECS[cell]["objective"],
                "target": MODEL_SPECS[cell]["target"],
                "features": MODEL_SPECS[cell]["features"],
                "probability_mapping": MODEL_SPECS[cell][
                    "probability_mapping"
                ],
                "num_boost_round": NUM_BOOST_ROUND,
            }
            for cell in ("A0", "B0", "B1")
        },
        "production_promotion_authorized": False,
    }

    manifest_hash = sha256_bytes(
        canonical_json_bytes(manifest)
    )

    FINAL_MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    FINAL_MANIFEST_SHA.write_text(
        f"{manifest_hash}  {FINAL_MANIFEST.name}\n",
        encoding="utf-8",
    )

    return manifest, manifest_hash


# =============================================================================
# Verification
# =============================================================================

def verify_final_freeze():
    if not FINAL_MANIFEST.exists():
        print(f"VERIFY FAILED: missing {FINAL_MANIFEST}")
        return 1

    if not FINAL_MANIFEST_SHA.exists():
        print(f"VERIFY FAILED: missing {FINAL_MANIFEST_SHA}")
        return 1

    manifest = load_json(FINAL_MANIFEST)
    actual_manifest_hash = sha256_bytes(
        canonical_json_bytes(manifest)
    )
    stored_manifest_hash = FINAL_MANIFEST_SHA.read_text(
        encoding="utf-8"
    ).strip().split()[0]

    print("HITS_B1_PROSPECTIVE_MODEL_FREEZE_A VERIFY")
    print("==========================================")
    print(f"manifest={FINAL_MANIFEST}")
    print(f"actual_manifest_sha256={actual_manifest_hash}")
    print(f"stored_manifest_sha256={stored_manifest_hash}")

    if actual_manifest_hash != stored_manifest_hash:
        print("verdict=VERIFY_FAILED_MANIFEST_HASH_MISMATCH")
        return 1

    lock_info = verify_holdout_lock()

    if manifest.get("holdout_lock_sha256") != lock_info["sha256"]:
        print("verdict=VERIFY_FAILED_HOLDOUT_LOCK_BINDING_MISMATCH")
        return 1

    all_ok = True

    for filename, metadata in manifest.get("artifacts", {}).items():
        path = FINAL_DIR / filename

        if not path.exists():
            print(f"{filename}: MISSING")
            all_ok = False
            continue

        actual = sha256_file(path)
        expected = metadata.get("sha256")
        ok = actual == expected
        all_ok = all_ok and ok

        print(
            f"{filename}: "
            f"ok={ok} "
            f"sha256={actual}"
        )

    if not all_ok:
        print("verdict=VERIFY_FAILED_ARTIFACT_MISMATCH")
        return 1

    print("verdict=PROSPECTIVE_MODEL_FREEZE_VERIFIED_IMMUTABLE")
    return 0


# =============================================================================
# Parent
# =============================================================================

def main_parent():
    print("HITS_B1_PROSPECTIVE_MODEL_FREEZE_A")
    print("==================================")

    print("")
    print("PREFLIGHT")
    print("---------")

    validate_sources()
    lock_info = verify_holdout_lock()

    print(f"holdout_lock_verified={lock_info['verified']}")
    print(f"holdout_lock_sha256={lock_info['sha256']}")
    print(
        f"holdout_start_game_date="
        f"{lock_info['holdout_start_game_date']}"
    )
    print(f"development_years={YEARS}")
    print("touch_2026=False")
    print("post_freeze_outcomes_inspected=False")
    print(f"candidate={CANDIDATE}")
    print("window_days=60")
    print(f"minimum_prior_pitcher_swings={MIN_SAMPLE}")
    print(f"num_boost_round={NUM_BOOST_ROUND}")
    print("threshold_rescue=FORBIDDEN")
    print("feature_window_tuning=FORBIDDEN")
    print("stacking=False")
    print("production_promotion_authorized=False")

    if FINAL_MANIFEST.exists():
        print("")
        print("Completed immutable freeze already exists.")
        print("Running verification instead.")
        return verify_final_freeze()

    print("")
    print("BUILD TRAINING FILES")
    print("--------------------")

    training_metadata = build_training_files()

    print(
        f"total_rows={training_metadata['total_rows']:,}"
    )
    print(
        f"binary_positive_rate="
        f"{training_metadata['binary_positive_rate']:.6f}"
    )
    print(
        f"candidate_coverage="
        f"{training_metadata['candidate_coverage']:.6f}"
    )

    print("")
    print("TRAIN THREE FROZEN CELLS")
    print("------------------------")

    model_outputs = train_models(training_metadata)

    print("")
    print("FINALIZE IMMUTABLE FREEZE")
    print("-------------------------")

    manifest, manifest_hash = finalize_immutable_freeze(
        lock_info,
        training_metadata,
        model_outputs,
    )

    print(
        f"manifest_sha256={manifest_hash}"
    )

    for cell in ("A0", "B0", "B1"):
        model_file = MODEL_SPECS[cell]["filename"]
        artifact = manifest["artifacts"][model_file]

        print(
            f"{cell}: "
            f"model={model_file} "
            f"sha256={artifact['sha256']} "
            f"size={artifact['size_bytes']:,}"
        )

    print("")
    print("FINAL STATUS")
    print("------------")
    print(
        "status=IMMUTABLE_PROSPECTIVE_MODEL_FREEZE_COMPLETE"
    )
    print(
        f"holdout_start_game_date="
        f"{lock_info['holdout_start_game_date']}"
    )
    print("touch_2026=False")
    print("post_freeze_outcomes_inspected=False")
    print("production_promotion_authorized=False")
    print(
        "next_step=BUILD_BLIND_LIVE_A0_B0_B1_LOGGER_BOUND_TO_"
        "HOLDOUT_LOCK_SHA_AND_MODEL_MANIFEST_SHA"
    )

    print("")
    print("OUTPUT DIRECTORY")
    print(FINAL_DIR)

    return 0


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--verify",
        action="store_true",
    )

    parser.add_argument(
        "--worker-train",
        action="store_true",
    )

    parser.add_argument(
        "--worker-verify-model",
        action="store_true",
    )

    parser.add_argument(
        "--objective",
        choices=["count:poisson", "binary:logistic"],
    )

    parser.add_argument("--train-file")
    parser.add_argument("--out-model")
    parser.add_argument("--model-path")
    parser.add_argument("--out-json")

    return parser


def main():
    args = build_parser().parse_args()

    if args.worker_train:
        worker_train(args)
        return 0

    if args.worker_verify_model:
        worker_verify_model(args)
        return 0

    if args.verify:
        return verify_final_freeze()

    return main_parent()


if __name__ == "__main__":
    raise SystemExit(main())
