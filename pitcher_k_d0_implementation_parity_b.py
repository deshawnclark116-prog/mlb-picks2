#!/usr/bin/env python3
"""
PITCHER_K_D0_IMPLEMENTATION_PARITY_AUDIT_B

READ-ONLY IMPLEMENTATION PARITY AUDIT
=====================================

Purpose
-------
Audit whether the live production pitcher-K path in:

    api.py
    lineupk.py
    bvp.py
    ksim.py

matches the formally validated challenger:

    D0_FIXED_LINEUP_ACTIVATION

This script does NOT:
- change production code
- retrain anything
- resimulate historical outcomes
- touch the batter-hits prospective holdout
- call external APIs
- tune thresholds
- rescue any failed branch
- promote D0 automatically

Expected current outcome
------------------------
The current production code is expected to FAIL parity because D0 has not yet
been implemented. That is correct and safe. The audit should identify the exact
blocking mismatches before any production patch is written.

Validated D0 definition
-----------------------
A0 incumbent:
    Activate the 45% lineup component only when at least 5 opposing batters have
    qualifying H2H strikeout history against this exact pitcher.

D0 challenger:
    Activate the same 45% lineup component when at least 5 opposing batters have
    genuinely usable K information from any of:
        - handedness split with >= 15 prior PA
        - current-season overall K rate with >= 15 prior PA
        - H2H K history with >= 2 prior PA

Everything else must stay unchanged:
    - qualifying outing BF >= 12
    - minimum 3 prior qualifying outings
    - 85% recent / 15% season pitcher anchor
    - expected BF = last-5 qualifying outings
    - last-12 outing K-rate volatility pool
    - 55% pitcher / 45% lineup blend
    - 40% H2H / 60% general batter blend
    - BVP K nudge
    - TTO decay 1.00 / 0.94 / 0.85
    - 10,000 simulations
    - no threshold changes

Formal validation prerequisite
------------------------------
The audit requires the completed formal result:

    /data/hr_model/pitcher_k_d0_2026_formal_gate_a_results.json

and verifies that:
    - D0 passed the formal 2026 gate
    - overall_formal_gate_pass == True
    - production promotion was not automatically authorized

Parity classes
--------------
PASS:
    Exact or implementation-equivalent match.

BLOCKER:
    Must be resolved before D0 can be promoted.

ADVISORY:
    Does not automatically invalidate the architecture, but must be explicitly
    resolved or accepted before promotion because research and live production
    are not perfectly identical.

Known high-value audit targets
------------------------------
1. D0 usable-data counting semantics.
2. API activation gate.
3. Pitcher-profile constants and windows.
4. Lineup blend weights.
5. H2H/general blend weights and sample thresholds.
6. BVP K-nudge sample threshold and cap.
7. Monte Carlo workload, TTO, volatility, and simulation count.
8. Strict D-1 / same-day exclusion parity.
9. H2H history horizon parity.
10. BVP history horizon parity.
11. Confirmed-lineup live source vs historical first-nine recovery.
12. Deterministic vs nondeterministic Monte Carlo execution.

Run
---
python -u pitcher_k_d0_implementation_parity_b.py 2>&1 | tee /data/hr_model/pitcher_k_d0_implementation_parity_b.log

Outputs
-------
/data/hr_model/pitcher_k_d0_implementation_parity_b_results.json
/data/hr_model/pitcher_k_d0_implementation_parity_b_report.txt

Paste back
----------
PARITY PREFLIGHT
FORMAL VALIDATION LOCK
CORE ARCHITECTURE PARITY
D0 ACTIVATION PARITY
HISTORY AND TIMING PARITY
MONTE CARLO PARITY
BLOCKERS
ADVISORIES
FINAL PARITY VERDICT
"""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone


# =============================================================================
# Paths
# =============================================================================

HR_DIR = Path("/data/hr_model")

FORMAL_RESULT = (
    HR_DIR
    / "pitcher_k_d0_2026_formal_gate_a_results.json"
)

FORMAL_MANIFEST = (
    HR_DIR
    / "pitcher_k_d0_2026_formal_gate_a_work"
    / "gate_manifest.json"
)

FORMAL_MANIFEST_SHA = (
    HR_DIR
    / "pitcher_k_d0_2026_formal_gate_a_work"
    / "gate_manifest.sha256"
)

OUT_JSON = (
    HR_DIR
    / "pitcher_k_d0_implementation_parity_b_results.json"
)

OUT_TXT = (
    HR_DIR
    / "pitcher_k_d0_implementation_parity_b_report.txt"
)


# =============================================================================
# Expected validated D0 spec
# =============================================================================

EXPECTED = {
    "candidate": "D0_FIXED_LINEUP_ACTIVATION",
    "comparator": "A0_CURRENT_INCUMBENT",

    "min_qualifying_bf": 12,
    "min_prior_qualifying_outings": 3,

    "recency_decay": 0.6,
    "season_anchor": 0.15,

    "pitcher_weight": 0.55,
    "lineup_weight": 0.45,

    "h2h_weight": 0.40,
    "general_weight": 0.60,

    "min_h2h_pa": 2,
    "min_general_pa": 15,
    "min_lineup_data_batters": 5,

    "bvp_min_sample_pa": 20,
    "k_nudge_min": 0.85,
    "k_nudge_max": 1.15,

    "recent_bf_window": 5,
    "volatility_rate_window": 12,

    "bf_sd": 2.5,
    "bf_min": 9,
    "bf_max": 30,

    "tto_factors": [1.00, 0.94, 0.85],
    "sims": 10000,

    "d0_usable_data_rule": (
        "count batter as usable when handedness split >=15 prior PA "
        "OR current-season overall K rate >=15 prior PA "
        "OR H2H K history >=2 prior PA"
    ),

    "activation_rule": (
        "activate lineup component when usable_data_batters >= 5"
    ),

    "strict_same_day_exclusion": True,

    "research_h2h_history_start": "2024-01-01",
    "research_bvp_history_start": "2024-01-01",

    "research_lineup_recovery": (
        "first nine distinct batters faced in completed-game feed"
    ),

    "live_lineup_source_expected": (
        "confirmed pregame lineup when available"
    ),
}


# =============================================================================
# Generic helpers
# =============================================================================

def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def norm_code(text):
    return re.sub(r"\s+", "", text or "")


def close_num(a, b, tol=1e-12):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def status_row(
    *,
    check_id,
    section,
    description,
    expected,
    observed,
    status,
    evidence,
    remediation=None,
):
    allowed_statuses = {
        "PASS",
        "BLOCKER",
        "DERIVED_BLOCKER",
        "ADVISORY",
        "INTENTIONAL_DIFFERENCE",
    }

    if status not in allowed_statuses:
        raise ValueError(f"Bad status: {status}")

    return {
        "check_id": check_id,
        "section": section,
        "description": description,
        "expected": expected,
        "observed": observed,
        "status": status,
        "evidence": evidence,
        "remediation": remediation,
    }


# =============================================================================
# Source inspector
# =============================================================================

class SourceInspector:
    def __init__(self, path):
        self.path = Path(path)

        if not self.path.exists():
            raise RuntimeError(f"Missing source file: {self.path}")

        self.text = self.path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        self.tree = ast.parse(
            self.text,
            filename=str(self.path),
        )

        self.lines = self.text.splitlines()

        self.assignments = self._collect_assignments()
        self.functions = self._collect_functions()

    def _collect_assignments(self):
        out = {}

        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    continue

                target = node.targets[0]

                if not isinstance(target, ast.Name):
                    continue

                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    continue

                out[target.id] = value

            elif isinstance(node, ast.AnnAssign):
                if not isinstance(node.target, ast.Name):
                    continue

                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    continue

                out[node.target.id] = value

        return out

    def _collect_functions(self):
        out = {}

        for node in self.tree.body:
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                out[node.name] = node

        return out

    def function_source(self, name):
        node = self.functions.get(name)

        if node is None:
            return ""

        if not hasattr(node, "end_lineno"):
            return ""

        return "\n".join(
            self.lines[
                node.lineno - 1:
                node.end_lineno
            ]
        )

    def has_text(self, text):
        return norm_code(text) in norm_code(self.text)

    def assignment(self, name, default=None):
        return self.assignments.get(name, default)


def _ast_is_pa_ge_15(test):
    """
    Robustly recognize: pa >= 15
    """
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "pa"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.GtE)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and float(test.comparators[0].value) >= 15.0
    )


def _ast_returns_rate_true(nodes):
    """
    Detect a return tuple whose second element is literal True.
    """
    for node in nodes:
        for child in ast.walk(node):
            if not isinstance(child, ast.Return):
                continue

            value = child.value

            if not (
                isinstance(value, ast.Tuple)
                and len(value.elts) >= 2
            ):
                continue

            flag = value.elts[1]

            if (
                isinstance(flag, ast.Constant)
                and flag.value is True
            ):
                return True

    return False


def analyze_general_k_real_data_flags(inspector):
    """
    Confirm that general_k_rate_vs_hand has two independent >=15 PA gates
    that return a real-data flag True:
      1) handedness split
      2) current-season fallback
    """
    node = inspector.functions.get(
        "general_k_rate_vs_hand"
    )

    if node is None:
        return {
            "function_found": False,
            "pa_ge_15_true_return_count": 0,
            "pass": False,
        }

    count = 0

    for child in ast.walk(node):
        if not isinstance(child, ast.If):
            continue

        if not _ast_is_pa_ge_15(child.test):
            continue

        if _ast_returns_rate_true(child.body):
            count += 1

    return {
        "function_found": True,
        "pa_ge_15_true_return_count": count,
        "pass": count >= 2,
    }


def analyze_blended_batter_availability(inspector):
    """
    Inspect whether blended_batter_k_rate preserves general-data availability
    instead of discarding it and returning False whenever H2H is absent.

    This is the exact A0 -> D0 semantic boundary.
    """
    node = inspector.functions.get(
        "blended_batter_k_rate"
    )

    if node is None:
        return {
            "function_found": False,
            "general_flag_target": None,
            "general_flag_discarded": None,
            "fallback_returns_false": None,
            "fallback_returns_general_flag": None,
            "h2h_true_return_detected": None,
            "d0_semantics_present": False,
        }

    general_flag_target = None
    fallback_returns_false = False
    fallback_returns_general_flag = False
    h2h_true_return_detected = False

    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            call = child.value

            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "general_k_rate_vs_hand"
            ):
                continue

            if len(child.targets) != 1:
                continue

            target = child.targets[0]

            if (
                isinstance(target, ast.Tuple)
                and len(target.elts) >= 2
            ):
                second = target.elts[1]

                if isinstance(second, ast.Name):
                    general_flag_target = second.id

        if isinstance(child, ast.Return):
            value = child.value

            if not (
                isinstance(value, ast.Tuple)
                and len(value.elts) >= 2
            ):
                continue

            flag = value.elts[1]

            if (
                isinstance(flag, ast.Constant)
                and flag.value is False
            ):
                fallback_returns_false = True

            if (
                isinstance(flag, ast.Constant)
                and flag.value is True
            ):
                h2h_true_return_detected = True

            if (
                general_flag_target
                and isinstance(flag, ast.Name)
                and flag.id == general_flag_target
            ):
                fallback_returns_general_flag = True

    general_flag_discarded = (
        general_flag_target == "_"
    )

    d0_semantics_present = (
        not general_flag_discarded
        and general_flag_target not in (None, "")
        and fallback_returns_general_flag
        and h2h_true_return_detected
        and not fallback_returns_false
    )

    return {
        "function_found": True,
        "general_flag_target": general_flag_target,
        "general_flag_discarded": general_flag_discarded,
        "fallback_returns_false": fallback_returns_false,
        "fallback_returns_general_flag": fallback_returns_general_flag,
        "h2h_true_return_detected": h2h_true_return_detected,
        "d0_semantics_present": d0_semantics_present,
    }


# =============================================================================
# Formal validation lock
# =============================================================================

def verify_formal_validation():
    if not FORMAL_RESULT.exists():
        raise RuntimeError(
            f"Missing formal validation result: {FORMAL_RESULT}"
        )

    payload = load_json(FORMAL_RESULT)

    verdict = payload.get("final_verdict")
    gate = payload.get("formal_gate") or {}
    policy = payload.get("policy") or {}

    passed = (
        isinstance(verdict, str)
        and "D0_FIXED_LINEUP_ACTIVATION_PASSES_2026_FORMAL_GATE"
        in verdict
        and gate.get("overall_formal_gate_pass") is True
    )

    if not passed:
        raise RuntimeError(
            "Formal result does not prove D0 passed the 2026 formal gate."
        )

    manifest_sha_recorded = (
        payload
        .get("gate_freeze", {})
        .get("gate_manifest_sha256")
    )

    manifest_sha_actual = None
    manifest_sha_file = None

    if FORMAL_MANIFEST.exists():
        manifest = load_json(FORMAL_MANIFEST)

        canonical = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        manifest_sha_actual = hashlib.sha256(
            canonical
        ).hexdigest()

    if FORMAL_MANIFEST_SHA.exists():
        manifest_sha_file = (
            FORMAL_MANIFEST_SHA
            .read_text(encoding="utf-8")
            .strip()
        )

    manifest_match = all(
        value
        and value == manifest_sha_recorded
        for value in (
            manifest_sha_actual,
            manifest_sha_file,
        )
    )

    return {
        "formal_verdict": verdict,
        "overall_formal_gate_pass": True,
        "formal_status": payload.get("formal_status"),

        "production_promotion_authorized": (
            policy.get(
                "production_promotion_authorized",
                False,
            )
        ),

        "formal_result_sha256": sha256_file(
            FORMAL_RESULT
        ),

        "gate_manifest_sha256_recorded": (
            manifest_sha_recorded
        ),

        "gate_manifest_sha256_actual": (
            manifest_sha_actual
        ),

        "gate_manifest_sha256_file": (
            manifest_sha_file
        ),

        "gate_manifest_integrity_pass": (
            manifest_match
        ),
    }


# =============================================================================
# Source discovery
# =============================================================================

def discover_source_file(name, root=None):
    candidates = []

    if root:
        candidates.append(
            Path(root) / name
        )

    candidates.extend(
        [
            Path.cwd() / name,
            Path("/opt/render/project/src") / name,
            Path("/opt/render/project/src/src") / name,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise RuntimeError(
        f"Could not find {name}. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


# =============================================================================
# Core architecture checks
# =============================================================================

def audit_core_architecture(api, lineupk, bvp, ksim):
    rows = []

    # --- api.py constants ---

    for (
        check_id,
        variable,
        expected_value,
        description,
    ) in [
        (
            "CORE-01",
            "RECENCY_DECAY",
            EXPECTED["recency_decay"],
            "Pitcher recent-form exponential decay.",
        ),
        (
            "CORE-02",
            "SEASON_ANCHOR",
            EXPECTED["season_anchor"],
            "Season anchor weight.",
        ),
        (
            "CORE-03",
            "PITCHER_WEIGHT",
            EXPECTED["pitcher_weight"],
            "Pitcher-side blend weight.",
        ),
        (
            "CORE-04",
            "LINEUP_WEIGHT",
            EXPECTED["lineup_weight"],
            "Lineup-side blend weight.",
        ),
    ]:
        observed = api.assignment(variable)

        rows.append(
            status_row(
                check_id=check_id,
                section="CORE ARCHITECTURE PARITY",
                description=description,
                expected=expected_value,
                observed=observed,
                status=(
                    "PASS"
                    if close_num(
                        observed,
                        expected_value,
                    )
                    else "BLOCKER"
                ),
                evidence=f"api.py assignment {variable}",
                remediation=(
                    None
                    if close_num(
                        observed,
                        expected_value,
                    )
                    else (
                        f"Restore {variable} to "
                        f"{expected_value}."
                    )
                ),
            )
        )

    # --- pitcher_feature_row semantics ---

    pitcher_src = api.function_source(
        "pitcher_feature_row"
    )

    pitcher_norm = norm_code(pitcher_src)

    checks = [
        (
            "CORE-05",
            "Qualifying outing requires BF >= 12.",
            "ifbf>=12:",
            EXPECTED["min_qualifying_bf"],
            "BF >= 12",
        ),
        (
            "CORE-06",
            "Expected BF uses last five qualifying outings.",
            'sum(bfs[-5:])/len(bfs[-5:])',
            EXPECTED["recent_bf_window"],
            "last 5 qualifying outings",
        ),
        (
            "CORE-07",
            "Volatility pool uses last 12 qualifying outing K rates.",
            '"per_start_krate":per_start_krate[-12:]',
            EXPECTED["volatility_rate_window"],
            "last 12 qualifying outing K rates",
        ),
        (
            "CORE-08",
            "Minimum prior qualifying outings is 3.",
            "ifn_starts<3:",
            EXPECTED["min_prior_qualifying_outings"],
            "minimum 3 qualifying outings",
        ),
    ]

    for (
        check_id,
        description,
        needle,
        expected_value,
        observed_text,
    ) in checks:
        ok = norm_code(needle) in pitcher_norm

        rows.append(
            status_row(
                check_id=check_id,
                section="CORE ARCHITECTURE PARITY",
                description=description,
                expected=expected_value,
                observed=observed_text if ok else "not detected",
                status="PASS" if ok else "BLOCKER",
                evidence=(
                    "api.py::pitcher_feature_row static source check"
                ),
                remediation=(
                    None
                    if ok
                    else (
                        "Inspect pitcher_feature_row and restore "
                        "the validated window/threshold."
                    )
                ),
            )
        )

    # Exact recent/season blend formula.
    blend_ok = (
        "(1-SEASON_ANCHOR)*rec_kbf+SEASON_ANCHOR*season_kbf"
        in pitcher_norm
    )

    rows.append(
        status_row(
            check_id="CORE-09",
            section="CORE ARCHITECTURE PARITY",
            description=(
                "Pitcher K/BF uses 85% recent + 15% season anchor."
            ),
            expected=(
                "(1 - SEASON_ANCHOR) * recent_kbf "
                "+ SEASON_ANCHOR * season_kbf"
            ),
            observed=(
                "formula detected"
                if blend_ok
                else "formula not detected"
            ),
            status="PASS" if blend_ok else "BLOCKER",
            evidence="api.py::pitcher_feature_row",
            remediation=(
                None
                if blend_ok
                else (
                    "Restore the validated recent/season "
                    "anchor formula."
                )
            ),
        )
    )

    # --- lineupk constants ---

    for (
        check_id,
        variable,
        expected_value,
        description,
    ) in [
        (
            "CORE-10",
            "H2H_WEIGHT",
            EXPECTED["h2h_weight"],
            "H2H batter K-rate weight.",
        ),
        (
            "CORE-11",
            "GEN_WEIGHT",
            EXPECTED["general_weight"],
            "General batter K-rate weight.",
        ),
        (
            "CORE-12",
            "MIN_H2H_PA",
            EXPECTED["min_h2h_pa"],
            "Minimum H2H PA threshold.",
        ),
    ]:
        observed = lineupk.assignment(variable)

        rows.append(
            status_row(
                check_id=check_id,
                section="CORE ARCHITECTURE PARITY",
                description=description,
                expected=expected_value,
                observed=observed,
                status=(
                    "PASS"
                    if close_num(
                        observed,
                        expected_value,
                    )
                    else "BLOCKER"
                ),
                evidence=f"lineupk.py assignment {variable}",
                remediation=(
                    None
                    if close_num(
                        observed,
                        expected_value,
                    )
                    else (
                        f"Restore {variable} to "
                        f"{expected_value}."
                    )
                ),
            )
        )

    general_src = lineupk.function_source(
        "general_k_rate_vs_hand"
    )

    general_norm = norm_code(general_src)

    hand_min_ok = "ifpa>=15:" in general_norm

    # There should be two >=15 checks:
    # one for handedness split, one for season fallback.
    min15_count = general_norm.count(
        "ifpa>=15:"
    )

    rows.append(
        status_row(
            check_id="CORE-13",
            section="CORE ARCHITECTURE PARITY",
            description=(
                "General batter K data requires >=15 PA for "
                "handedness split, then >=15 PA for season fallback."
            ),
            expected=(
                "two separate >=15 PA gates"
            ),
            observed=(
                f"{min15_count} detected"
            ),
            status=(
                "PASS"
                if hand_min_ok
                and min15_count >= 2
                else "BLOCKER"
            ),
            evidence="lineupk.py::general_k_rate_vs_hand",
            remediation=(
                None
                if hand_min_ok
                and min15_count >= 2
                else (
                    "Restore >=15 PA handedness split gate "
                    "and >=15 PA season fallback gate."
                )
            ),
        )
    )

    # --- bvp semantics ---

    lineup_bvp_src = bvp.function_source(
        "lineup_vs_pitcher"
    )

    lineup_bvp_norm = norm_code(
        lineup_bvp_src
    )

    nudge_cap_ok = (
        "max(0.85,min(1.15,raw))"
        in lineup_bvp_norm
    )

    rows.append(
        status_row(
            check_id="CORE-14",
            section="CORE ARCHITECTURE PARITY",
            description=(
                "BVP lineup K nudge remains capped to 0.85..1.15."
            ),
            expected="0.85 <= k_nudge <= 1.15",
            observed=(
                "0.85..1.15 cap detected"
                if nudge_cap_ok
                else "validated cap not detected"
            ),
            status="PASS" if nudge_cap_ok else "BLOCKER",
            evidence="bvp.py::lineup_vs_pitcher",
            remediation=(
                None
                if nudge_cap_ok
                else (
                    "Restore validated BVP nudge cap "
                    "0.85..1.15."
                )
            ),
        )
    )

    api_run_src = api.function_source(
        "run_predictions"
    )

    api_run_norm = norm_code(
        api_run_src
    )

    bvp_sample_ok = (
        'ifagg["sample_pa"]>=20:'
        in api_run_norm
    )

    rows.append(
        status_row(
            check_id="CORE-15",
            section="CORE ARCHITECTURE PARITY",
            description=(
                "BVP K nudge activates only at >=20 combined H2H PA."
            ),
            expected=EXPECTED["bvp_min_sample_pa"],
            observed=(
                20
                if bvp_sample_ok
                else "not detected"
            ),
            status="PASS" if bvp_sample_ok else "BLOCKER",
            evidence="api.py::run_predictions",
            remediation=(
                None
                if bvp_sample_ok
                else (
                    "Restore BVP lineup sample gate to >=20 PA."
                )
            ),
        )
    )

    return rows


# =============================================================================
# D0 activation checks
# =============================================================================

def audit_d0_activation(api, lineupk):
    rows = []

    lineup_src = lineupk.function_source(
        "lineup_k_expectation"
    )

    run_src = api.function_source(
        "run_predictions"
    )

    lineup_norm = norm_code(lineup_src).lower()
    run_norm = norm_code(run_src).lower()

    # ------------------------------------------------------------------
    # D0-01
    # Corrected detector:
    # The v1 audit falsely marked this as a blocker because its static
    # string detector was case-sensitive for Python's literal True.
    # Use AST semantics instead.
    # ------------------------------------------------------------------
    general_analysis = analyze_general_k_real_data_flags(
        lineupk
    )

    general_real_data_supported = (
        general_analysis["pass"]
    )

    rows.append(
        status_row(
            check_id="D0-01",
            section="D0 ACTIVATION PARITY",
            description=(
                "General handedness/season K helper can identify "
                "real usable general data."
            ),
            expected=(
                "real-data flag True for handedness >=15 PA or "
                "season >=15 PA"
            ),
            observed=general_analysis,
            status=(
                "PASS"
                if general_real_data_supported
                else "BLOCKER"
            ),
            evidence=(
                "AST inspection of "
                "lineupk.py::general_k_rate_vs_hand"
            ),
            remediation=(
                None
                if general_real_data_supported
                else (
                    "Return a real-data availability flag True for "
                    "both handedness >=15 PA and season fallback >=15 PA."
                )
            ),
        )
    )

    # ------------------------------------------------------------------
    # D0-02
    # Exact A0 -> D0 semantic boundary.
    # ------------------------------------------------------------------
    blend_analysis = analyze_blended_batter_availability(
        lineupk
    )

    if blend_analysis["d0_semantics_present"]:
        d0_status = "PASS"
        d0_observed = (
            "general-data availability is preserved and used "
            "when H2H is absent"
        )

    elif (
        blend_analysis["general_flag_discarded"]
        and blend_analysis["fallback_returns_false"]
    ):
        d0_status = "BLOCKER"
        d0_observed = (
            "general_k_rate_vs_hand availability flag is discarded "
            "and fallback returns False when H2H is absent"
        )

    else:
        d0_status = "BLOCKER"
        d0_observed = (
            "validated D0 usable-data semantics not detected"
        )

    rows.append(
        status_row(
            check_id="D0-02",
            section="D0 ACTIVATION PARITY",
            description=(
                "Each batter is counted as usable when ANY validated "
                "general or H2H K source is available."
            ),
            expected=EXPECTED["d0_usable_data_rule"],
            observed={
                "summary": d0_observed,
                "analysis": blend_analysis,
            },
            status=d0_status,
            evidence=(
                "AST inspection of "
                "lineupk.py::blended_batter_k_rate"
            ),
            remediation=(
                None
                if d0_status == "PASS"
                else (
                    "Patch blended_batter_k_rate so its availability "
                    "flag is True when handedness split >=15 PA, "
                    "season K >=15 PA, or H2H >=2 PA. "
                    "Do not change the 40/60 rate blend itself."
                )
            ),
        )
    )

    # ------------------------------------------------------------------
    # D0-03
    # ------------------------------------------------------------------
    n_data_increment_ok = (
        "ifused:n_data+=1"
        in lineup_norm
    )

    rows.append(
        status_row(
            check_id="D0-03",
            section="D0 ACTIVATION PARITY",
            description=(
                "lineup_k_expectation increments n_data from the "
                "per-batter availability flag."
            ),
            expected="if used: n_data += 1",
            observed=(
                "detected"
                if n_data_increment_ok
                else "not detected"
            ),
            status=(
                "PASS"
                if n_data_increment_ok
                else "BLOCKER"
            ),
            evidence="lineupk.py::lineup_k_expectation",
            remediation=(
                None
                if n_data_increment_ok
                else (
                    "Restore per-batter n_data increment semantics."
                )
            ),
        )
    )

    # ------------------------------------------------------------------
    # D0-04
    # ------------------------------------------------------------------
    api_gate_ok = (
        "ifekandn>=5:"
        in run_norm
    )

    rows.append(
        status_row(
            check_id="D0-04",
            section="D0 ACTIVATION PARITY",
            description=(
                "API activates the lineup component at n_data >= 5."
            ),
            expected=EXPECTED["activation_rule"],
            observed=(
                "if ek and n >= 5"
                if api_gate_ok
                else "validated activation gate not detected"
            ),
            status=(
                "PASS"
                if api_gate_ok
                else "BLOCKER"
            ),
            evidence="api.py::run_predictions",
            remediation=(
                None
                if api_gate_ok
                else (
                    "Restore activation threshold to n_data >= 5."
                )
            ),
        )
    )

    # ------------------------------------------------------------------
    # D0-05
    # This is a derived end-to-end status when D0-03 and D0-04 are already
    # correct but D0-02 is still wrong. Do not double-count it as an
    # independent architecture defect.
    # ------------------------------------------------------------------
    end_to_end_ok = (
        d0_status == "PASS"
        and n_data_increment_ok
        and api_gate_ok
    )

    if end_to_end_ok:
        end_status = "PASS"

    elif (
        d0_status != "PASS"
        and n_data_increment_ok
        and api_gate_ok
    ):
        end_status = "DERIVED_BLOCKER"

    else:
        end_status = "BLOCKER"

    rows.append(
        status_row(
            check_id="D0-05",
            section="D0 ACTIVATION PARITY",
            description=(
                "End-to-end production activation count matches "
                "validated D0 semantics."
            ),
            expected=(
                "usable-data count from general/H2H sources "
                "drives >=5 lineup activation gate"
            ),
            observed=(
                "full D0 activation parity"
                if end_to_end_ok
                else (
                    "derived failure from D0-02 only"
                    if end_status == "DERIVED_BLOCKER"
                    else "not yet in parity"
                )
            ),
            status=end_status,
            evidence=(
                "combined lineupk.py + api.py static parity"
            ),
            remediation=(
                None
                if end_to_end_ok
                else (
                    "Resolve D0-02 first; keep D0-03 and D0-04 "
                    "unchanged unless independently failing."
                )
            ),
        )
    )

    return rows


# =============================================================================
# History and timing checks
# =============================================================================

def audit_history_and_timing(api, lineupk, bvp):
    rows = []

    # Strict D-1 / same-day exclusion.
    pitcher_src = api.function_source(
        "pitcher_feature_row"
    )

    lineup_general_src = lineupk.function_source(
        "general_k_rate_vs_hand"
    )

    same_day_filter_detected = any(
        token in norm_code(pitcher_src + "\n" + lineup_general_src)
        for token in (
            "beforetoday_et()",
            "date<today_et()",
            "exclude_same_day",
            "strict_d1",
            "game_date<",
        )
    )

    rows.append(
        status_row(
            check_id="TIME-01",
            section="HISTORY AND TIMING PARITY",
            description=(
                "Live pitcher and batter history enforce strict D-1 "
                "same-day exclusion."
            ),
            expected=True,
            observed=(
                "explicit same-day exclusion detected"
                if same_day_filter_detected
                else (
                    "not explicitly enforced; season/gameLog endpoints "
                    "may include earlier same-day games"
                )
            ),
            status=(
                "PASS"
                if same_day_filter_detected
                else "BLOCKER"
            ),
            evidence=(
                "api.py::pitcher_feature_row + "
                "lineupk.py::general_k_rate_vs_hand"
            ),
            remediation=(
                None
                if same_day_filter_detected
                else (
                    "Add strict pregame as-of-date handling so earlier "
                    "same-day games cannot enter pitcher or batter history. "
                    "This matters especially for doubleheaders or reruns "
                    "after an earlier game becomes final."
                )
            ),
        )
    )

    # H2H horizon parity.
    h2h_src = lineupk.function_source(
        "head_to_head_k_rate"
    )

    h2h_norm = norm_code(h2h_src)

    uses_vsplayer_without_start_date = (
        'stats="vsplayer"'
        in h2h_norm.lower()
        and "startdate" not in h2h_norm.lower()
        and "2024" not in h2h_norm
    )

    rows.append(
        status_row(
            check_id="HIST-01",
            section="HISTORY AND TIMING PARITY",
            description=(
                "Live H2H K history horizon matches the validated "
                "2024-forward research horizon."
            ),
            expected=(
                "H2H history starts 2024-01-01"
            ),
            observed=(
                "career vsPlayer history with no 2024 cutoff detected"
                if uses_vsplayer_without_start_date
                else "explicit bounded horizon detected"
            ),
            status=(
                "ADVISORY"
                if uses_vsplayer_without_start_date
                else "PASS"
            ),
            evidence="lineupk.py::head_to_head_k_rate",
            remediation=(
                None
                if not uses_vsplayer_without_start_date
                else (
                    "Before promotion, choose and lock one policy: "
                    "(A) reproduce the validated 2024-forward H2H horizon "
                    "in production, or "
                    "(B) run a dedicated parity shadow showing career H2H "
                    "does not materially alter D0 decisions. "
                    "Do not silently assume career H2H is equivalent."
                )
            ),
        )
    )

    # BVP horizon parity.
    bvp_src = bvp.function_source(
        "batter_vs_pitcher"
    )

    bvp_norm = norm_code(bvp_src).lower()

    bvp_unbounded = (
        'stats="vsplayer"'
        in bvp_norm
        and "startdate" not in bvp_norm
        and "2024" not in bvp_norm
    )

    rows.append(
        status_row(
            check_id="HIST-02",
            section="HISTORY AND TIMING PARITY",
            description=(
                "Live BVP K-nudge history horizon matches the "
                "validated 2024-forward research horizon."
            ),
            expected=(
                "BVP history starts 2024-01-01"
            ),
            observed=(
                "career vsPlayer history with no 2024 cutoff detected"
                if bvp_unbounded
                else "explicit bounded horizon detected"
            ),
            status=(
                "ADVISORY"
                if bvp_unbounded
                else "PASS"
            ),
            evidence="bvp.py::batter_vs_pitcher",
            remediation=(
                None
                if not bvp_unbounded
                else (
                    "Before promotion, reproduce the validated horizon "
                    "or run a dedicated shadow parity comparison. "
                    "Career BVP cannot be assumed identical to 2024-forward BVP."
                )
            ),
        )
    )

    # Lineup source parity.
    lineup_fetch_src = api.function_source(
        "get_confirmed_lineup"
    )

    lineup_fetch_norm = norm_code(
        lineup_fetch_src
    )

    confirmed_lineup_detected = (
        'get(side,{}).get("battingorder",[])'
        in lineup_fetch_norm.lower()
    )

    rows.append(
        status_row(
            check_id="HIST-03",
            section="HISTORY AND TIMING PARITY",
            description=(
                "Live production uses confirmed pregame lineup rather than "
                "historical first-nine recovery."
            ),
            expected=(
                "confirmed pregame lineup for live predictions"
            ),
            observed=(
                "confirmed battingOrder source detected"
                if confirmed_lineup_detected
                else "confirmed lineup source not detected"
            ),
            status=(
                "PASS"
                if confirmed_lineup_detected
                else "BLOCKER"
            ),
            evidence="api.py::get_confirmed_lineup",
            remediation=(
                None
                if confirmed_lineup_detected
                else (
                    "Use confirmed pregame lineup before D0 promotion."
                )
            ),
        )
    )

    # Explicitly disclose research-vs-live source difference.
    rows.append(
        status_row(
            check_id="HIST-04",
            section="HISTORY AND TIMING PARITY",
            description=(
                "Research holdout lineup recovery source is not literally "
                "identical to live production lineup source."
            ),
            expected=(
                "difference explicitly acknowledged and live source is "
                "pregame-knowable"
            ),
            observed=(
                "research used first nine distinct batters faced; "
                "live code uses confirmed battingOrder"
            ),
            status="INTENTIONAL_DIFFERENCE",
            evidence=(
                "formal gate methodology vs api.py::get_confirmed_lineup"
            ),
            remediation=(
                "Keep confirmed pregame lineup in production. "
                "Treat this as an implementation-source difference, not a reason "
                "to replace confirmed lineups with postgame reconstruction. "
                "Include it in the live shadow audit."
            ),
        )
    )

    return rows


# =============================================================================
# Monte Carlo checks
# =============================================================================

def audit_monte_carlo(ksim):
    rows = []

    sim_src = ksim.function_source(
        "simulate"
    )

    start_src = ksim.function_source(
        "_simulate_start"
    )

    sim_norm = norm_code(sim_src)
    start_norm = norm_code(start_src)

    # sims default
    sims_default_ok = (
        "sims=10000"
        in sim_norm
    )

    rows.append(
        status_row(
            check_id="MC-01",
            section="MONTE CARLO PARITY",
            description=(
                "Production K simulation count is 10,000."
            ),
            expected=EXPECTED["sims"],
            observed=(
                10000
                if sims_default_ok
                else "not detected"
            ),
            status=(
                "PASS"
                if sims_default_ok
                else "BLOCKER"
            ),
            evidence="ksim.py::simulate signature",
            remediation=(
                None
                if sims_default_ok
                else (
                    "Restore K simulation count to 10,000."
                )
            ),
        )
    )

    # BF workload
    bf_sd_ok = (
        "rng.normal(expected_bf,2.5)"
        in start_norm
    )

    bf_clip_ok = (
        "max(9,min(30,bf))"
        in start_norm
    )

    rows.append(
        status_row(
            check_id="MC-02",
            section="MONTE CARLO PARITY",
            description=(
                "Workload simulation uses Normal(expected_bf, 2.5) "
                "clamped to 9..30 BF."
            ),
            expected={
                "sd": EXPECTED["bf_sd"],
                "min": EXPECTED["bf_min"],
                "max": EXPECTED["bf_max"],
            },
            observed=(
                {
                    "sd": 2.5,
                    "min": 9,
                    "max": 30,
                }
                if bf_sd_ok
                and bf_clip_ok
                else "not fully detected"
            ),
            status=(
                "PASS"
                if bf_sd_ok
                and bf_clip_ok
                else "BLOCKER"
            ),
            evidence="ksim.py::_simulate_start",
            remediation=(
                None
                if bf_sd_ok
                and bf_clip_ok
                else (
                    "Restore validated workload simulation semantics."
                )
            ),
        )
    )

    # TTO
    tto_ok = all(
        token in start_norm
        for token in (
            "decay=1.0",
            "decay=0.94",
            "decay=0.85",
        )
    )

    rows.append(
        status_row(
            check_id="MC-03",
            section="MONTE CARLO PARITY",
            description=(
                "TTO decay remains 1.00 / 0.94 / 0.85."
            ),
            expected=EXPECTED["tto_factors"],
            observed=(
                [1.00, 0.94, 0.85]
                if tto_ok
                else "not fully detected"
            ),
            status="PASS" if tto_ok else "BLOCKER",
            evidence="ksim.py::_simulate_start",
            remediation=(
                None
                if tto_ok
                else (
                    "Restore validated TTO factors."
                )
            ),
        )
    )

    # Volatility pool
    pool_ok = (
        "0.7*pool+0.3*k_per_bf"
        in sim_norm
    )

    min_four_ok = (
        "len(start_k_rates)>=4"
        in sim_norm
    )

    rows.append(
        status_row(
            check_id="MC-04",
            section="MONTE CARLO PARITY",
            description=(
                "Volatility pool uses per-start rates when >=4 are "
                "available, shrunk 70% start / 30% center."
            ),
            expected=(
                ">=4 start rates; pool = 0.7*start + 0.3*k_per_bf"
            ),
            observed=(
                "validated volatility pool detected"
                if pool_ok
                and min_four_ok
                else "not fully detected"
            ),
            status=(
                "PASS"
                if pool_ok
                and min_four_ok
                else "BLOCKER"
            ),
            evidence="ksim.py::simulate",
            remediation=(
                None
                if pool_ok
                and min_four_ok
                else (
                    "Restore validated volatility-pool semantics."
                )
            ),
        )
    )

    # RNG determinism.
    nondeterministic_rng = (
        "np.random.randomstate()"
        in sim_norm.lower()
        and "seed" not in sim_norm.lower()
    )

    rows.append(
        status_row(
            check_id="MC-05",
            section="MONTE CARLO PARITY",
            description=(
                "Production MC execution determinism matches formal-gate "
                "deterministic paired simulation."
            ),
            expected=(
                "deterministic seedable execution for exact reproducibility"
            ),
            observed=(
                "RandomState() without explicit seed"
                if nondeterministic_rng
                else "seedable/deterministic path detected"
            ),
            status=(
                "ADVISORY"
                if nondeterministic_rng
                else "PASS"
            ),
            evidence="ksim.py::simulate",
            remediation=(
                None
                if not nondeterministic_rng
                else (
                    "Before production promotion, decide whether exact "
                    "reproducibility is required. Recommended: add optional seed "
                    "support without changing default distribution semantics, "
                    "then use a stable seed for logged live predictions."
                )
            ),
        )
    )

    # Formal gate used vectorized binomial by TTO segment; production uses
    # per-PA Bernoulli. Distribution-equivalent when p is constant within segment.
    per_pa_bernoulli = (
        "ifrng.rand()<p_k:"
        in start_norm
    )

    rows.append(
        status_row(
            check_id="MC-06",
            section="MONTE CARLO PARITY",
            description=(
                "Production per-PA Bernoulli simulation is distribution-equivalent "
                "to formal-gate segment-wise binomial simulation."
            ),
            expected=(
                "same Bernoulli/binomial K distribution under frozen TTO p values"
            ),
            observed=(
                "per-PA Bernoulli implementation detected"
                if per_pa_bernoulli
                else "expected implementation not detected"
            ),
            status=(
                "PASS"
                if per_pa_bernoulli
                else "BLOCKER"
            ),
            evidence=(
                "ksim.py::_simulate_start vs formal-gate methodology"
            ),
            remediation=(
                None
                if per_pa_bernoulli
                else (
                    "Inspect simulation path for distribution drift."
                )
            ),
        )
    )

    return rows


# =============================================================================
# Final verdict and report
# =============================================================================

def summarize_checks(checks):
    passes = [
        row
        for row in checks
        if row["status"] == "PASS"
    ]

    blockers = [
        row
        for row in checks
        if row["status"] == "BLOCKER"
    ]

    derived_blockers = [
        row
        for row in checks
        if row["status"] == "DERIVED_BLOCKER"
    ]

    advisories = [
        row
        for row in checks
        if row["status"] == "ADVISORY"
    ]

    intentional_differences = [
        row
        for row in checks
        if row["status"] == "INTENTIONAL_DIFFERENCE"
    ]

    promotion_blocking_count = (
        len(blockers)
        + len(derived_blockers)
    )

    return {
        "total_checks": len(checks),

        "pass_count": len(passes),
        "blocker_count": len(blockers),
        "derived_blocker_count": len(derived_blockers),
        "promotion_blocking_count": promotion_blocking_count,
        "advisory_count": len(advisories),
        "intentional_difference_count": len(intentional_differences),

        "passes": passes,
        "blockers": blockers,
        "derived_blockers": derived_blockers,
        "advisories": advisories,
        "intentional_differences": intentional_differences,
    }


def build_report(payload):
    lines = []

    lines.append(
        "PITCHER_K_D0_IMPLEMENTATION_PARITY_AUDIT_B"
    )

    lines.append("=" * 43)
    lines.append("")

    for section in (
        "PARITY PREFLIGHT",
        "FORMAL VALIDATION LOCK",
        "CORE ARCHITECTURE PARITY",
        "D0 ACTIVATION PARITY",
        "HISTORY AND TIMING PARITY",
        "MONTE CARLO PARITY",
        "BLOCKERS",
        "DERIVED BLOCKERS",
        "ADVISORIES",
        "INTENTIONAL DIFFERENCES",
        "FINAL PARITY VERDICT",
    ):
        lines.append(section)
        lines.append("-" * len(section))

        if section == "PARITY PREFLIGHT":
            lines.append(
                json.dumps(
                    payload["preflight"],
                    indent=2,
                )
            )

        elif section == "FORMAL VALIDATION LOCK":
            lines.append(
                json.dumps(
                    payload["formal_validation_lock"],
                    indent=2,
                )
            )

        elif section in {
            "CORE ARCHITECTURE PARITY",
            "D0 ACTIVATION PARITY",
            "HISTORY AND TIMING PARITY",
            "MONTE CARLO PARITY",
        }:
            rows = [
                row
                for row in payload["checks"]
                if row["section"] == section
            ]

            lines.append(
                json.dumps(
                    rows,
                    indent=2,
                )
            )

        elif section == "BLOCKERS":
            lines.append(
                json.dumps(
                    payload["summary"]["blockers"],
                    indent=2,
                )
            )

        elif section == "DERIVED BLOCKERS":
            lines.append(
                json.dumps(
                    payload["summary"]["derived_blockers"],
                    indent=2,
                )
            )

        elif section == "ADVISORIES":
            lines.append(
                json.dumps(
                    payload["summary"]["advisories"],
                    indent=2,
                )
            )

        elif section == "INTENTIONAL DIFFERENCES":
            lines.append(
                json.dumps(
                    payload["summary"]["intentional_differences"],
                    indent=2,
                )
            )

        elif section == "FINAL PARITY VERDICT":
            lines.append(
                payload["final_verdict"]
            )

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        help=(
            "Optional repo source root containing "
            "api.py, lineupk.py, bvp.py, ksim.py"
        ),
    )

    args = parser.parse_args()

    print(
        "PITCHER_K_D0_IMPLEMENTATION_PARITY_AUDIT_B",
        flush=True,
    )

    print(
        "===========================================",
        flush=True,
    )

    print("")
    print("PARITY PREFLIGHT", flush=True)
    print("----------------", flush=True)

    api_path = discover_source_file(
        "api.py",
        args.root,
    )

    lineupk_path = discover_source_file(
        "lineupk.py",
        args.root,
    )

    bvp_path = discover_source_file(
        "bvp.py",
        args.root,
    )

    ksim_path = discover_source_file(
        "ksim.py",
        args.root,
    )

    preflight = {
        "candidate": EXPECTED["candidate"],
        "comparator": EXPECTED["comparator"],

        "source_files": {
            "api.py": str(api_path),
            "lineupk.py": str(lineupk_path),
            "bvp.py": str(bvp_path),
            "ksim.py": str(ksim_path),
        },

        "source_hashes": {
            "api.py": sha256_file(api_path),
            "lineupk.py": sha256_file(lineupk_path),
            "bvp.py": sha256_file(bvp_path),
            "ksim.py": sha256_file(ksim_path),
        },

        "formal_result": str(FORMAL_RESULT),
        "external_api_calls": False,
        "production_changed": False,
        "retraining": False,
        "resimulation": False,
        "threshold_tuning": False,
        "rescue_tuning": False,
        "stacking": False,
        "hits_prospective_holdout_touched": False,

        "audit_revision": "B",

        "audit_corrections": {
            "D0-01": (
                "Replaced fragile case-sensitive string detector with "
                "AST semantic inspection. The prior v1 D0-01 blocker was "
                "a detector false positive for the supplied lineupk.py."
            ),

            "D0-05": (
                "Classified as DERIVED_BLOCKER when D0-02 alone causes "
                "the end-to-end activation mismatch."
            ),

            "HIST-04": (
                "Classified as INTENTIONAL_DIFFERENCE because confirmed "
                "pregame battingOrder is the correct live source and should "
                "not be degraded to postgame first-nine recovery."
            ),
        },
    }

    print(
        json.dumps(
            preflight,
            indent=2,
        ),
        flush=True,
    )

    print("")
    print("FORMAL VALIDATION LOCK", flush=True)
    print("----------------------", flush=True)

    formal_lock = verify_formal_validation()

    print(
        json.dumps(
            formal_lock,
            indent=2,
        ),
        flush=True,
    )

    api = SourceInspector(api_path)
    lineupk = SourceInspector(lineupk_path)
    bvp = SourceInspector(bvp_path)
    ksim = SourceInspector(ksim_path)

    checks = []

    checks.extend(
        audit_core_architecture(
            api,
            lineupk,
            bvp,
            ksim,
        )
    )

    checks.extend(
        audit_d0_activation(
            api,
            lineupk,
        )
    )

    checks.extend(
        audit_history_and_timing(
            api,
            lineupk,
            bvp,
        )
    )

    checks.extend(
        audit_monte_carlo(
            ksim,
        )
    )

    for section in (
        "CORE ARCHITECTURE PARITY",
        "D0 ACTIVATION PARITY",
        "HISTORY AND TIMING PARITY",
        "MONTE CARLO PARITY",
    ):
        print("")
        print(section, flush=True)
        print("-" * len(section), flush=True)

        for row in checks:
            if row["section"] != section:
                continue

            print(
                f"{row['check_id']}: "
                f"{row['status']} | "
                f"{row['description']}",
                flush=True,
            )

            print(
                f"  expected={row['expected']}",
                flush=True,
            )

            print(
                f"  observed={row['observed']}",
                flush=True,
            )

            if row.get("remediation"):
                print(
                    f"  remediation={row['remediation']}",
                    flush=True,
                )

    summary = summarize_checks(checks)

    print("")
    print("BLOCKERS", flush=True)
    print("--------", flush=True)

    if summary["blockers"]:
        for row in summary["blockers"]:
            print(
                f"{row['check_id']}: "
                f"{row['description']}",
                flush=True,
            )

            print(
                f"  observed={row['observed']}",
                flush=True,
            )

            print(
                f"  remediation={row['remediation']}",
                flush=True,
            )
    else:
        print("none", flush=True)

    print("")
    print("DERIVED BLOCKERS", flush=True)
    print("----------------", flush=True)

    if summary["derived_blockers"]:
        for row in summary["derived_blockers"]:
            print(
                f"{row['check_id']}: "
                f"{row['description']}",
                flush=True,
            )

            print(
                f"  observed={row['observed']}",
                flush=True,
            )

            print(
                f"  remediation={row['remediation']}",
                flush=True,
            )
    else:
        print("none", flush=True)

    print("")
    print("ADVISORIES", flush=True)
    print("----------", flush=True)

    if summary["advisories"]:
        for row in summary["advisories"]:
            print(
                f"{row['check_id']}: "
                f"{row['description']}",
                flush=True,
            )

            print(
                f"  observed={row['observed']}",
                flush=True,
            )

            print(
                f"  remediation={row['remediation']}",
                flush=True,
            )
    else:
        print("none", flush=True)

    print("")
    print("INTENTIONAL DIFFERENCES", flush=True)
    print("-----------------------", flush=True)

    if summary["intentional_differences"]:
        for row in summary["intentional_differences"]:
            print(
                f"{row['check_id']}: "
                f"{row['description']}",
                flush=True,
            )

            print(
                f"  observed={row['observed']}",
                flush=True,
            )

            print(
                f"  remediation={row['remediation']}",
                flush=True,
            )
    else:
        print("none", flush=True)

    if (
        summary["promotion_blocking_count"] == 0
        and formal_lock[
            "gate_manifest_integrity_pass"
        ]
    ):
        final_verdict = (
            "D0_IMPLEMENTATION_PARITY_AUDIT_B_PASS_"
            "NO_BLOCKERS_READY_FOR_ADVISORY_RESOLUTION_AND_"
            "EXPLICIT_PRODUCTION_PROMOTION_DECISION_"
            "NO_AUTOMATIC_PROMOTION"
        )
    else:
        final_verdict = (
            "D0_IMPLEMENTATION_PARITY_AUDIT_B_FAIL_"
            "TRUE_BLOCKERS_OR_DERIVED_BLOCKERS_REMAIN_"
            "PATCH_REQUIRED_NO_PRODUCTION_PROMOTION"
        )

    print("")
    print("FINAL PARITY VERDICT", flush=True)
    print("--------------------", flush=True)
    print(final_verdict, flush=True)

    payload = {
        "script": (
            "PITCHER_K_D0_IMPLEMENTATION_PARITY_AUDIT_B"
        ),

        "generated_at_utc": now_utc(),

        "candidate": EXPECTED["candidate"],
        "comparator": EXPECTED["comparator"],

        "preflight": preflight,
        "formal_validation_lock": formal_lock,

        "validated_spec": EXPECTED,

        "checks": checks,
        "summary": summary,

        "final_verdict": final_verdict,

        "policy": {
            "production_changed": False,
            "automatic_promotion": False,

            "if_blockers_exist": (
                "patch only validated true blockers; derived blockers clear "
                "automatically when their parent blocker is resolved. "
                "Then rerun audit and perform live shadow parity check before "
                "explicit promotion decision."
            ),

            "if_only_advisories_remain": (
                "resolve or explicitly accept each advisory before "
                "production promotion"
            ),

            "rescue_tuning": "FORBIDDEN",
            "threshold_changes": "FORBIDDEN",
            "stacking": "FORBIDDEN",
        },
    }

    OUT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    OUT_TXT.write_text(
        build_report(payload),
        encoding="utf-8",
    )

    print("")
    print("OUTPUTS", flush=True)
    print(OUT_JSON, flush=True)
    print(OUT_TXT, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
