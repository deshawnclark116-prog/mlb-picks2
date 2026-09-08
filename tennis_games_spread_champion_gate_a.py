#!/usr/bin/env python3
"""
TENNIS_GAMES_SPREAD_CHAMPION_GATE_A

Champion-gate validation for a tennis games-handicap (spread) market --
confirmed real and live on The Odds API (e.g. "Sabalenka -3.5" / "Noskova
+3.5" games for a match), unlike set_betting which passed its backtest
but has no real market to grade against.

Target: game MARGIN for a match, from an arbitrary but outcome-
independent "P1" perspective (lexicographically smaller player_id, same
anti-leakage trick as the moneyline/set-betting gates) -- margin_p1 =
(games P1 won across the whole match) - (games P2 won). Ground truth
parsed directly from `matches.score`, using the same "winner's games
listed first in each set token" convention as the other gates; P1's
games are attributed by checking whether P1 is the actual winner_id.

Feature model: for each player, recency-weighted average GAME MARGIN
per set played (their own games won minus opponent's games won, per
set, averaged over their past matches), surface-blended the same way
the total_games/aces gates blended surface-specific history. Predicted
margin = (P1's blended margin-per-set - P2's blended margin-per-set) *
expected number of sets in this match (blended average of both
players' own history, conditioned on best_of -- same expected-sets
signal already built for total_games). Naive baseline drops the
surface blend (plain recency-weighted margin-per-set only).

Regression evaluation (MAE + paired bootstrap), same shape and same
pre-registered bar as tennis_total_games_champion_gate_a.py -- chosen
for the same reason: a continuous target, and the nearest-line
classification trick proved structurally unable to produce graded
observations for continuous large-scale targets there.

PRE-REGISTERED PASS BAR (decided before running):
  1. Model HOLDOUT MAE improves on naive HOLDOUT MAE by >= 5% (relative)
  2. Model confidently beats naive on HOLDOUT (paired bootstrap,
     P(model not better) < 0.10)
  3. Model HOLDOUT MAE not confidently worse than model VAL MAE
     (unpaired bootstrap, P(holdout worse) < 0.90)
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
REPORT_PATH = Path("tennis_games_spread_gate_report.json")

RECENCY_DECAY = 0.6
MIN_PRIOR_MATCHES = 10
MIN_SURFACE_MATCHES_FOR_BLEND = 4
SEED = 20260908

VAL_START, VAL_END = "2020-01-01", "2022-12-31"
HOLDOUT_START, HOLDOUT_END = "2023-01-01", "2024-12-31"

SET_RE = re.compile(r"^(\d+)-(\d+)")


def parse_sets(score):
    """Returns list of (winner_games, loser_games) per set, per the
    winner-listed-first convention, or None if unparseable."""
    sets = []
    for token in (score or "").split():
        m = SET_RE.match(token)
        if not m:
            continue
        sets.append((int(m.group(1)), int(m.group(2))))
    return sets if sets else None


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

    # Per-player chronological history: (margin_per_set, n_sets, surface, best_of)
    player_hist = defaultdict(list)

    observations = []
    n_unparseable = 0
    for match_id, match_date, surface, best_of, winner_id, loser_id, score in rows:
        sets = parse_sets(score)
        if sets is None:
            n_unparseable += 1
            continue
        n_sets = len(sets)
        winner_games = sum(w for w, l in sets)
        loser_games = sum(l for w, l in sets)
        winner_margin_per_set = (winner_games - loser_games) / n_sets

        p1, p2 = sorted([winner_id, loser_id])
        p1_is_winner = p1 == winner_id
        p1_total_games = winner_games if p1_is_winner else loser_games
        p2_total_games = loser_games if p1_is_winner else winner_games
        actual_margin_p1 = p1_total_games - p2_total_games

        feats = {}
        eligible = True
        for pid, is_winner_side in ((p1, p1_is_winner), (p2, not p1_is_winner)):
            hist = player_hist[pid]
            if len(hist) < MIN_PRIOR_MATCHES:
                eligible = False
                continue
            overall_margins = [m for m, _, _, _ in hist]
            overall_sets = [s for _, s, _, _ in hist]
            surf_margins = [m for m, _, sf, _ in hist if sf == surface]
            surf_n_count = sum(1 for _, _, sf, _ in hist if sf == surface)
            bo_sets = [s for _, s, _, bo in hist if bo == best_of]

            own_margin, _ = recency_weighted_mean(overall_margins)
            own_sets_mean, _ = recency_weighted_mean(overall_sets)
            surf_margin_mean, surf_n = recency_weighted_mean(surf_margins)
            bo_sets_mean, bo_n = recency_weighted_mean(bo_sets)

            if surf_n >= MIN_SURFACE_MATCHES_FOR_BLEND and surf_margin_mean is not None:
                blend_w = min(surf_n / (surf_n + 10), 0.7)
                combined_margin = blend_w * surf_margin_mean + (1 - blend_w) * own_margin
            else:
                combined_margin = own_margin

            expected_sets = bo_sets_mean if bo_n >= MIN_SURFACE_MATCHES_FOR_BLEND else own_sets_mean

            feats[pid] = {
                "naive_margin": own_margin, "naive_sets": own_sets_mean,
                "combined_margin": combined_margin, "expected_sets": expected_sets,
            }

        if eligible:
            f1, f2 = feats[p1], feats[p2]
            expected_sets_blend = (f1["expected_sets"] + f2["expected_sets"]) / 2.0
            predicted_margin = (f1["combined_margin"] - f2["combined_margin"]) * expected_sets_blend
            naive_expected_sets_blend = (f1["naive_sets"] + f2["naive_sets"]) / 2.0
            naive_margin = (f1["naive_margin"] - f2["naive_margin"]) * naive_expected_sets_blend

            observations.append({
                "match_date": match_date,
                "predicted_margin": predicted_margin,
                "naive_margin_pred": naive_margin,
                "actual_margin": actual_margin_p1,
            })

        for pid, is_winner_side in ((p1, p1_is_winner), (p2, not p1_is_winner)):
            side_games = winner_games if is_winner_side else loser_games
            opp_games = loser_games if is_winner_side else winner_games
            margin_per_set = (side_games - opp_games) / n_sets
            player_hist[pid].append((margin_per_set, n_sets, surface, best_of))

    print(f"skipped {n_unparseable} matches with unparseable score")
    print(f"built {len(observations)} point-in-time observations")

    val_obs = [o for o in observations if VAL_START <= o["match_date"] <= VAL_END]
    hold_obs = [o for o in observations if HOLDOUT_START <= o["match_date"] <= HOLDOUT_END]
    print(f"VAL: {len(val_obs)} obs, HOLDOUT: {len(hold_obs)} obs")

    def errors(obs_list, key):
        return np.array([abs(o[key] - o["actual_margin"]) for o in obs_list])

    def mae(err):
        return float(np.mean(err)) if len(err) else None

    def bootstrap_p_paired_worse(err_a, err_b, seed=SEED, b=5000):
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
            if ra >= rb:
                worse += 1
        return worse / b

    model_val_err = errors(val_obs, "predicted_margin")
    model_hold_err = errors(hold_obs, "predicted_margin")
    naive_val_err = errors(val_obs, "naive_margin_pred")
    naive_hold_err = errors(hold_obs, "naive_margin_pred")

    model_val_mae = mae(model_val_err)
    model_hold_mae = mae(model_hold_err)
    naive_val_mae = mae(naive_val_err)
    naive_hold_mae = mae(naive_hold_err)

    mae_improvement_pct = (
        (naive_hold_mae - model_hold_mae) / naive_hold_mae if naive_hold_mae else None)
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
        "verdict": "TENNIS_GAMES_SPREAD_GATE_PASSED" if passed else "TENNIS_GAMES_SPREAD_GATE_FAILED",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
