#!/usr/bin/env python3
"""
TENNIS_TOTAL_GAMES_CHAMPION_GATE_A

Champion-gate validation for a tennis total_games market (total games
played by BOTH players combined in a match -- a "total" market, unlike
total_aces/double_faults which were single-player props).

Originally tried the CORRECTED grading methodology from the aces gate
(one graded observation per match, using only the swept line nearest
the model's own predicted mean) -- but for a continuous, large-scale
target like total games (~20-25 typical, spread ~3-4 games), that
approach is structurally broken: the gap between the nearest half-
integer line and the model's own mean is capped at 0.5 games by
construction, which relative to a 3-4 game spread can never generate
a confident (>=0.70) over/under probability -- confirmed empirically
(literal n=0 HIGH-confidence observations in both arms, not merely a
small or disappointing number). That's a flaw in the evaluation
framework, discovered BEFORE looking at any pass/fail result, not a
post-hoc excuse -- so this gate uses direct regression metrics (MAE,
paired-error bootstrap) instead of a binary hit-rate at all, which is
the natural fit for a continuous target anyway and sidesteps both the
aces gate's pseudo-replication problem and this nearest-line problem.

Ground truth: total games parsed directly from `matches.score` (e.g.
"6-4 7-6(4)" -> (6+4) + (7+6) = 23 games; tiebreak parenthetical is
stripped, not counted as games). Incomplete matches (RET/W/O/etc,
already flagged by tennis_player_matches_foundation_a.py) are excluded
-- a match that didn't finish has a truncated, non-representative game
count, same reasoning as excluding them from player_matches.

Feature model: for each player, recency-weighted history of
(games_per_set, sets_played) from their own past matches, blended with
a surface-specific subset the same way the aces gate blended surface
ace-rate. A match's predicted total = the AVERAGE of both players' own
recency-weighted (games_per_set * expected_sets) -- both players
contribute symmetrically to how long a match runs, unlike a single-
player prop. Naive baseline drops the surface blend (plain overall
recency-weighted average only), same "does the extra complexity earn
its keep" comparison used throughout this session.

PRE-REGISTERED PASS BAR (regression-metric shape, decided before running):
  1. Model HOLDOUT MAE improves on naive HOLDOUT MAE by >= 5% (relative)
  2. Model confidently beats naive on HOLDOUT: paired bootstrap over
     per-match (|naive_error| - |model_error|) is confidently positive
     (P(mean paired diff <= 0) < 0.10, i.e. P(model better) >= 0.90)
  3. Model HOLDOUT MAE not confidently worse than model VAL MAE
     (paired-arms bootstrap, P(holdout worse) < 0.90)
  4. n >= 200 in both VAL and HOLDOUT

Split by date: DEV 2015-2019 (unused), VAL 2020-2022, HOLDOUT 2023-2024.
"""
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tennis_model.sqlite")
REPORT_PATH = Path("tennis_total_games_gate_report.json")

RECENCY_DECAY = 0.6
MIN_PRIOR_MATCHES = 10
MIN_SURFACE_MATCHES_FOR_BLEND = 4
LINE_SWEEP = [x + 0.5 for x in range(10, 46)]  # 10.5 .. 45.5
SIMS = 8000
SEED = 20260908

VAL_START, VAL_END = "2020-01-01", "2022-12-31"
HOLDOUT_START, HOLDOUT_END = "2023-01-01", "2024-12-31"

HIGH = 0.70

SET_RE = re.compile(r"^(\d+)-(\d+)")


def parse_total_games(score):
    """'6-4 7-6(4)' -> 23. Returns (total_games, n_sets) or (None, None)
    if the score can't be parsed (unexpected format, empty, etc)."""
    if not score:
        return None, None
    total = 0
    n_sets = 0
    for token in score.split():
        m = SET_RE.match(token)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        total += a + b
        n_sets += 1
    if n_sets == 0:
        return None, None
    return total, n_sets


def load_rows(con):
    cur = con.execute(
        "SELECT match_id, match_date, surface, best_of, winner_id, loser_id, score "
        "FROM matches WHERE is_incomplete = 0 AND surface IS NOT NULL "
        "ORDER BY match_date, match_id")
    return cur.fetchall()


def recency_weighted_mean(values, decay=RECENCY_DECAY):
    if not values:
        return None, 0
    num = den = 0.0
    w = 1.0
    for v in reversed(values):
        num += w * v
        den += w
        w *= decay
    return (num / den if den > 0 else None), len(values)


def main():
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = load_rows(con)
    con.close()
    print(f"loaded {len(rows)} complete matches with known surface from {DB_PATH}")

    # Per-player chronological history: (games_per_set, n_sets, surface, best_of)
    player_hist = defaultdict(list)

    observations = []
    n_unparseable = 0
    for match_id, match_date, surface, best_of, winner_id, loser_id, score in rows:
        total_games, n_sets = parse_total_games(score)
        if total_games is None:
            n_unparseable += 1
            continue
        games_per_set = total_games / n_sets

        feats = {}
        eligible = True
        for pid in (winner_id, loser_id):
            hist = player_hist[pid]
            if len(hist) < MIN_PRIOR_MATCHES:
                eligible = False
                continue
            overall_gps = [g for g, _, _, _ in hist]
            overall_sets = [s for _, s, _, _ in hist]
            surf_gps = [g for g, _, sf, _ in hist if sf == surface]
            surf_sets = [s for _, s, sf, _ in hist if sf == surface]
            bo_sets = [s for _, s, _, bo in hist if bo == best_of]

            own_gps, own_n = recency_weighted_mean(overall_gps)
            own_sets_mean, _ = recency_weighted_mean(overall_sets)
            surf_gps_mean, surf_n = recency_weighted_mean(surf_gps)
            surf_sets_mean, _ = recency_weighted_mean(surf_sets)
            bo_sets_mean, bo_n = recency_weighted_mean(bo_sets)

            if surf_n >= MIN_SURFACE_MATCHES_FOR_BLEND and surf_gps_mean is not None:
                blend_w = min(surf_n / (surf_n + 10), 0.7)
                combined_gps = blend_w * surf_gps_mean + (1 - blend_w) * own_gps
            else:
                combined_gps = own_gps

            expected_sets = bo_sets_mean if bo_n >= MIN_SURFACE_MATCHES_FOR_BLEND else own_sets_mean

            feats[pid] = {
                "naive_gps": own_gps, "naive_sets": own_sets_mean,
                "combined_gps": combined_gps, "expected_sets": expected_sets,
            }

        if not eligible:
            for pid in (winner_id, loser_id):
                total_g_this, sets_this = total_games, n_sets
                player_hist[pid].append((games_per_set, sets_this, surface, best_of))
            continue

        fw, fl = feats[winner_id], feats[loser_id]
        predicted_mean = (
            (fw["combined_gps"] * fw["expected_sets"]) +
            (fl["combined_gps"] * fl["expected_sets"])
        ) / 2.0
        naive_mean = (
            (fw["naive_gps"] * fw["naive_sets"]) +
            (fl["naive_gps"] * fl["naive_sets"])
        ) / 2.0

        observations.append({
            "match_date": match_date,
            "predicted_mean": predicted_mean,
            "naive_mean": naive_mean,
            "actual_total_games": total_games,
        })

        for pid in (winner_id, loser_id):
            player_hist[pid].append((games_per_set, n_sets, surface, best_of))

    print(f"skipped {n_unparseable} matches with unparseable score")
    print(f"built {len(observations)} point-in-time observations")

    val_obs = [o for o in observations if VAL_START <= o["match_date"] <= VAL_END]
    hold_obs = [o for o in observations if HOLDOUT_START <= o["match_date"] <= HOLDOUT_END]
    print(f"VAL: {len(val_obs)} obs, HOLDOUT: {len(hold_obs)} obs")

    def errors(obs_list, mean_key):
        model_err = np.array([abs(o[mean_key] - o["actual_total_games"]) for o in obs_list])
        return model_err

    def mae(err):
        return float(np.mean(err)) if len(err) else None

    def bootstrap_p_paired_worse(err_a, err_b, seed=SEED, b=5000):
        """P(mean(err_a) >= mean(err_b)) via paired bootstrap over the
        SAME observations (both arms see the same matches) -- legit,
        non-pseudo-replicated since each match contributes exactly one
        paired (err_a, err_b) draw per resample."""
        if len(err_a) == 0 or len(err_b) == 0 or len(err_a) != len(err_b):
            return None
        r = np.random.default_rng(seed)
        n = len(err_a)
        worse = 0
        for _ in range(b):
            idx = r.integers(0, n, size=n)
            if err_a[idx].mean() >= err_b[idx].mean():
                worse += 1
        return worse / b

    def bootstrap_p_worse_unpaired(err_a, err_b, seed=SEED, b=5000):
        if len(err_a) == 0 or len(err_b) == 0:
            return None
        r = np.random.default_rng(seed)
        worse = 0
        for _ in range(b):
            ra = r.choice(err_a, size=len(err_a), replace=True).mean()
            rb = r.choice(err_b, size=len(err_b), replace=True).mean()
            if ra >= rb:  # arm A's error is >= arm B's -- A is "worse"
                worse += 1
        return worse / b

    model_val_err = errors(val_obs, "predicted_mean")
    model_hold_err = errors(hold_obs, "predicted_mean")
    naive_val_err = errors(val_obs, "naive_mean")
    naive_hold_err = errors(hold_obs, "naive_mean")

    model_val_mae = mae(model_val_err)
    model_hold_mae = mae(model_hold_err)
    naive_val_mae = mae(naive_val_err)
    naive_hold_mae = mae(naive_hold_err)

    mae_improvement_pct = (
        (naive_hold_mae - model_hold_mae) / naive_hold_mae
        if naive_hold_mae else None
    )
    p_model_worse_than_naive_holdout = bootstrap_p_paired_worse(model_hold_err, naive_hold_err)
    p_holdout_worse_than_val = bootstrap_p_worse_unpaired(model_hold_err, model_val_err)

    check1 = mae_improvement_pct is not None and mae_improvement_pct >= 0.05
    check2 = p_model_worse_than_naive_holdout is not None and p_model_worse_than_naive_holdout < 0.10
    check3 = p_holdout_worse_than_val is not None and p_holdout_worse_than_val < 0.90
    check4 = len(val_obs) >= 200 and len(hold_obs) >= 200

    passed = check1 and check2 and check3 and check4

    report = {
        "grading_methodology": "regression_mae_paired_bootstrap",
        "n_total_observations": len(observations),
        "n_val_observations": len(val_obs),
        "n_holdout_observations": len(hold_obs),
        "model_val_mae": model_val_mae,
        "model_holdout_mae": model_hold_mae,
        "naive_val_mae": naive_val_mae,
        "naive_holdout_mae": naive_hold_mae,
        "mae_improvement_pct_vs_naive_holdout": mae_improvement_pct,
        "p_model_error_worse_than_naive_holdout": p_model_worse_than_naive_holdout,
        "p_holdout_error_worse_than_val": p_holdout_worse_than_val,
        "checks": {
            "1_mae_improves_on_naive_by_ge_5pct": check1,
            "2_model_confidently_beats_naive_holdout": check2,
            "3_holdout_not_confidently_worse_than_val": check3,
            "4_n_ge_200_both_windows": check4,
        },
        "passed": passed,
        "verdict": "TENNIS_TOTAL_GAMES_GATE_PASSED" if passed else "TENNIS_TOTAL_GAMES_GATE_FAILED",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
