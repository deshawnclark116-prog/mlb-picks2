"""
Research invariant tests for nfl_player_outcome_error_budget_a.py.

pytest is not installed in this environment, so this file runs standalone
(and is also pytest-compatible: every check is a test_* function).

  python -u tests/test_nfl_player_outcome_invariants.py --data-dir /tmp/nfl_data
  python -u tests/test_nfl_player_outcome_invariants.py --unit-only

Unit checks use synthetic data (Shapley math, coalition fallback, CRPS).
Data checks run the full diagnostic and fail loudly on any invariant failure.
"""
import argparse
import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nfl_player_outcome_error_budget_a as eb  # noqa: E402


def test_shapley_additive_game_equals_solo_values():
    comps = ["a", "b", "c"]
    solo = {"a": 3.0, "b": 1.0, "c": 0.5}
    v = {m: sum(solo[c] for c, b in zip(comps, m) if b) for m in itertools.product([0, 1], repeat=3)}
    phi = eb.shapley(comps, v)
    for c in comps:
        assert abs(phi[c] - solo[c]) < 1e-12, (c, phi[c])


def test_shapley_efficiency_with_interaction():
    comps = ["a", "b"]
    v = {(0, 0): 0.0, (1, 0): 1.0, (0, 1): 1.0, (1, 1): 5.0}     # 3.0 of pure interaction
    phi = eb.shapley(comps, v)
    assert abs(sum(phi.values()) - 5.0) < 1e-12
    assert abs(phi["a"] - 2.5) < 1e-12 and abs(phi["b"] - 2.5) < 1e-12


def test_shapley_is_order_independent():
    comps = ["x", "y", "z", "w"]
    rng = np.random.default_rng(0)
    v = {m: float(rng.normal()) if any(m) else 0.0 for m in itertools.product([0, 1], repeat=4)}
    phi = eb.shapley(comps, v)
    # permutation average computed directly
    direct = {c: 0.0 for c in comps}
    perms = list(itertools.permutations(range(4)))
    for p in perms:
        m = [0, 0, 0, 0]
        for i in p:
            before = tuple(m); m[i] = 1
            direct[comps[i]] += (v[tuple(m)] - v[before]) / len(perms)
    for c in comps:
        assert abs(phi[c] - direct[c]) < 1e-9


def test_chain_exact_reconstruction_and_undefined_fallback():
    # yards = T * S * (E + X); a player with 0 carries has undefined E/X
    T = np.array([30.0, 25.0]); car = np.array([10.0, 0.0]); base = np.array([35.0, 0.0]); expl = np.array([22.0, 0.0])
    act = {"T": T, "S": car / T, "E": np.array([3.5, np.nan]), "X": np.array([2.2, np.nan])}
    prd = {"T": np.array([28.0, 27.0]), "S": np.array([0.3, 0.1]), "E": np.array([4.0, 4.0]), "X": np.array([0.5, 0.5])}
    y = base + expl
    out = eb.chain_budget("t", ["T", "S", "E", "X"], prd, act, lambda v: v["T"] * v["S"] * (v["E"] + v["X"]), y)
    assert out["exact_reconstruction_max_abs_err"] < 1e-9
    assert len(out["coalitions"]) == 16
    sh = out["shapley"]["mae"]
    assert abs(sum(sh["phi"].values()) - sh["total_value"]) < 1e-9


def test_crps_point_mass_equals_absolute_error():
    pts = np.array([[5.0], [10.0]]); w = np.ones((2, 1)); y = np.array([7.0, 4.0])
    assert abs(eb.crps_discrete(pts, w, y) - np.mean([2.0, 6.0])) < 1e-12


def test_market_features_absent_from_core():
    for cols in (eb.TEAM_CORE, eb.SHARE_CORE, eb.RATE_CORE, eb.VOL_CORE, eb.CORE, eb.AVAIL_COLS):
        assert not set(cols) & set(eb.MARKET_FEATURES)
        assert not any(c.startswith("ORC_") for c in cols)


def test_windows_disjoint():
    for s in range(2022, 2027):
        for w in range(1, 19):
            assert sum([eb.TRAIN(s, w), eb.VALID(s, w), s in eb.EVAL_SEASONS]) <= 1, (s, w)


def run_data_checks(data_dir):
    report, R = eb.run(data_dir, perturb=True)
    failed = [r for r in R if not r["pass"]]
    for r in R:
        print(("PASS " if r["pass"] else "FAIL ") + r["invariant"] + (f"  [{r['detail']}]" if r["detail"] else ""))
    print(f"\n{len(R) - len(failed)}/{len(R)} data invariants pass")
    eb.assert_invariants(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("NFL_DATA_DIR"))
    ap.add_argument("--unit-only", action="store_true")
    args = ap.parse_args()
    units = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in units:
        t()
        print(f"PASS unit {t.__name__}")
    print(f"{len(units)}/{len(units)} unit checks pass\n")
    if not args.unit_only:
        if not args.data_dir:
            raise SystemExit("--data-dir (or NFL_DATA_DIR) required for data invariants")
        run_data_checks(args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
