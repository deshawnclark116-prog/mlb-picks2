#!/usr/bin/env python3
"""
TENNIS_SET_BETTING_CHAMPION_GATE_A

Champion-gate validation for a tennis set-betting / correct-score market
(predict the exact set score, e.g. 2-0, 2-1 for best-of-3; 3-0, 3-1, 3-2
for best-of-5 -- 4 or 6 possible outcomes per match). Strictly harder
than moneyline: getting the winner wrong also means getting the set
score wrong, so this inherits moneyline's ceiling and adds more ways to
be wrong on top of it. Building and testing honestly rather than
assuming that in advance.

Reuses the exact same Elo machinery as tennis_moneyline_champion_gate_a.py
(overall + surface-blended rating, tennis-standard K-factor decay,
lexicographic p1/p2 assignment to avoid winner-label leakage). The new
step: convert the Elo-implied per-MATCH win probability p into a per-SET
win probability q by inverting the standard best-of-N race formula
(assumes sets are i.i.d. given q -- a standard simplifying assumption in
the tennis-analytics literature; real tennis has some serve/game
momentum dependence this ignores), then computes the full distribution
over every possible set-score outcome (both sides) via the race-to-N
binomial formula, and predicts the single most likely outcome (argmax
over all 2N categories, not conditioned on knowing who wins).

Naive baseline: always predict the single most common outcome category
observed in DEV+VAL history for that best_of (no player-specific signal
at all) -- multi-class analog of the other gates' naive arms.

PRE-REGISTERED PASS BAR:
  1. VAL accuracy     >= naive VAL accuracy + 0.05 (absolute)
  2. HOLDOUT accuracy >= naive HOLDOUT accuracy + 0.05 (absolute)
  3. Model confidently beats naive on HOLDOUT (paired bootstrap over
     per-match (naive_correct - model_correct), P(model not better) < 0.10)
  4. HOLDOUT accuracy not confidently worse than VAL (bootstrap P < 0.90)
  5. n >= 200 in both VAL and HOLDOUT

Split by date: DEV 2015-2019 (Elo warmup + naive-baseline category
learned here, not evaluated), VAL 2020-2022, HOLDOUT 2023-2024.
"""
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from math import comb

DB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tennis_model.sqlite")
REPORT_PATH = Path("tennis_set_betting_gate_report.json")

MIN_PRIOR_MATCHES = 5
MIN_SURFACE_MATCHES_FOR_BLEND = 4
INITIAL_ELO = 1500.0
SEED = 20260908

DEV_END = "2019-12-31"
VAL_START, VAL_END = "2020-01-01", "2022-12-31"
HOLDOUT_START, HOLDOUT_END = "2023-01-01", "2024-12-31"

SET_RE = re.compile(r"^(\d+)-(\d+)")


def parse_set_counts(score):
    """Returns (sets_won_by_side_listed_first_each_token, sets_won_by_other)
    -- TML-Database/Sackmann convention lists the match WINNER's games
    first in every set token, so this returns (winner_sets, loser_sets)."""
    w_sets = l_sets = 0
    for token in (score or "").split():
        m = SET_RE.match(token)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            w_sets += 1
        elif b > a:
            l_sets += 1
    if w_sets == 0 and l_sets == 0:
        return None, None
    return w_sets, l_sets


def k_factor(n_matches):
    return 250.0 / ((n_matches + 5) ** 0.4)


def elo_expected(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def match_win_prob_from_set_prob(q, races_to):
    """P(win the race-to-N series) given per-set win prob q."""
    total = 0.0
    for k in range(races_to, 2 * races_to):
        total += comb(k - 1, races_to - 1) * (q ** races_to) * ((1 - q) ** (k - races_to))
    return total


def invert_to_set_prob(p, races_to, tol=1e-9):
    """Solve for q in [0.5, 1) such that match_win_prob_from_set_prob(q)==p,
    for p >= 0.5 (mirror for p<0.5). Monotonic in q -- bisection is exact
    enough and simple.

    p == 0.5 exactly must short-circuit to q=0.5 rather than recurse --
    real bug caught on WTA data (two equally-rated players, e.g. two
    still-at-INITIAL_ELO players facing off, produce combined_elo(p1) ==
    combined_elo(p2) bit-for-bit): the p<=0.5 branch calls itself with
    1-p, which is ALSO exactly 0.5, recursing forever until Python's
    stack limit kills it. Never surfaced on ATP data, evidently by luck
    of the float arithmetic never landing on an exact tie there."""
    if p == 0.5:
        return 0.5
    if p < 0.5:
        return 1 - invert_to_set_prob(1 - p, races_to, tol)
    lo, hi = 0.5, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if match_win_prob_from_set_prob(mid, races_to) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def outcome_distribution(q, races_to):
    """Full distribution over (winner_sets, loser_sets) outcomes for
    BOTH possible winners, keyed by ('P1', w, l) / ('P2', w, l)."""
    dist = {}
    for k in range(races_to, 2 * races_to):
        loser_sets = k - races_to
        prob_p1_wins_this_way = comb(k - 1, races_to - 1) * (q ** races_to) * ((1 - q) ** loser_sets)
        prob_p2_wins_this_way = comb(k - 1, races_to - 1) * ((1 - q) ** races_to) * (q ** loser_sets)
        dist[("P1", races_to, loser_sets)] = prob_p1_wins_this_way
        dist[("P2", races_to, loser_sets)] = prob_p2_wins_this_way
    return dist


def load_rows(con):
    cur = con.execute(
        "SELECT match_id, match_date, surface, best_of, winner_id, loser_id, score "
        "FROM matches WHERE is_incomplete = 0 AND surface IS NOT NULL "
        "AND best_of IN (3, 5) ORDER BY match_date, match_id")
    return cur.fetchall()


def main():
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = load_rows(con)
    con.close()
    print(f"loaded {len(rows)} complete matches (best_of 3 or 5) from {DB_PATH}")

    overall_elo, overall_n = {}, {}
    surface_elo, surface_n = {}, {}

    observations = []
    dev_category_counts = Counter()
    n_unparseable = 0

    for match_id, match_date, surface, best_of, winner_id, loser_id, score in rows:
        w_sets, l_sets = parse_set_counts(score)
        if w_sets is None:
            n_unparseable += 1
            continue
        races_to = best_of // 2 + 1

        p1, p2 = sorted([winner_id, loser_id])
        p1_is_winner = 1 if p1 == winner_id else 0
        actual_category = ("P1" if p1_is_winner else "P2", w_sets, l_sets)

        e1o, e2o = overall_elo.get(p1, INITIAL_ELO), overall_elo.get(p2, INITIAL_ELO)
        n1o, n2o = overall_n.get(p1, 0), overall_n.get(p2, 0)
        e1s, e2s = surface_elo.get((p1, surface), INITIAL_ELO), surface_elo.get((p2, surface), INITIAL_ELO)
        n1s, n2s = surface_n.get((p1, surface), 0), surface_n.get((p2, surface), 0)

        if match_date <= DEV_END:
            dev_category_counts[(best_of, actual_category)] += 1
        elif n1o >= MIN_PRIOR_MATCHES and n2o >= MIN_PRIOR_MATCHES:
            min_surf_n = min(n1s, n2s)
            if min_surf_n >= MIN_SURFACE_MATCHES_FOR_BLEND:
                blend_w = min(min_surf_n / (min_surf_n + 15), 0.6)
                c1 = blend_w * e1s + (1 - blend_w) * e1o
                c2 = blend_w * e2s + (1 - blend_w) * e2o
            else:
                c1, c2 = e1o, e2o

            p = elo_expected(c1, c2)
            q = invert_to_set_prob(p, races_to)
            dist = outcome_distribution(q, races_to)
            predicted_category = max(dist, key=dist.get)

            observations.append({
                "match_date": match_date,
                "best_of": best_of,
                "predicted_category": predicted_category,
                "actual_category": actual_category,
                "model_correct": 1 if predicted_category == actual_category else 0,
            })

        k1, k2 = k_factor(n1o), k_factor(n2o)
        e1n = e1o + k1 * (p1_is_winner - elo_expected(e1o, e2o))
        e2n = e2o + k2 * ((1 - p1_is_winner) - elo_expected(e2o, e1o))
        overall_elo[p1], overall_elo[p2] = e1n, e2n
        overall_n[p1], overall_n[p2] = n1o + 1, n2o + 1

        ks1, ks2 = k_factor(n1s), k_factor(n2s)
        es1n = e1s + ks1 * (p1_is_winner - elo_expected(e1s, e2s))
        es2n = e2s + ks2 * ((1 - p1_is_winner) - elo_expected(e2s, e1s))
        surface_elo[(p1, surface)], surface_elo[(p2, surface)] = es1n, es2n
        surface_n[(p1, surface)], surface_n[(p2, surface)] = n1s + 1, n2s + 1

    print(f"skipped {n_unparseable} matches with unparseable score")
    print(f"built {len(observations)} point-in-time observations")

    # Naive baseline: modal category per best_of, learned from DEV only.
    modal_category = {}
    for bo in (3, 5):
        cats = {cat: n for (b, cat), n in dev_category_counts.items() if b == bo}
        if cats:
            modal_category[bo] = max(cats, key=cats.get)
    print(f"naive modal categories (from DEV 2015-2019): {modal_category}")

    for o in observations:
        naive_cat = modal_category.get(o["best_of"])
        o["naive_correct"] = 1 if naive_cat == o["actual_category"] else 0

    val_obs = [o for o in observations if VAL_START <= o["match_date"] <= VAL_END]
    hold_obs = [o for o in observations if HOLDOUT_START <= o["match_date"] <= HOLDOUT_END]
    print(f"VAL: {len(val_obs)} obs, HOLDOUT: {len(hold_obs)} obs")

    def rate(obs_list, key):
        vals = [o[key] for o in obs_list]
        return (sum(vals) / len(vals)) if vals else None

    val_model_acc = rate(val_obs, "model_correct")
    hold_model_acc = rate(hold_obs, "model_correct")
    hold_naive_acc = rate(hold_obs, "naive_correct")

    def bootstrap_p_paired_model_not_better(obs_list, seed=SEED, b=5000):
        if not obs_list:
            return None
        r = np.random.default_rng(seed)
        model = np.array([o["model_correct"] for o in obs_list])
        naive = np.array([o["naive_correct"] for o in obs_list])
        n = len(obs_list)
        not_better = 0
        for _ in range(b):
            idx = r.integers(0, n, size=n)
            if model[idx].mean() <= naive[idx].mean():
                not_better += 1
        return not_better / b

    def bootstrap_p_worse_unpaired(hits_a, hits_b, seed=SEED, b=5000):
        if not hits_a or not hits_b:
            return None
        r = np.random.default_rng(seed)
        a, bb = np.array(hits_a), np.array(hits_b)
        worse = 0
        for _ in range(b):
            ra = r.choice(a, size=len(a), replace=True).mean()
            rb = r.choice(bb, size=len(bb), replace=True).mean()
            if ra < rb:
                worse += 1
        return worse / b

    p_model_not_better_holdout = bootstrap_p_paired_model_not_better(hold_obs)
    p_hold_worse_than_val = bootstrap_p_worse_unpaired(
        [o["model_correct"] for o in hold_obs], [o["model_correct"] for o in val_obs])

    val_naive_acc = rate(val_obs, "naive_correct")
    check1 = val_model_acc is not None and val_naive_acc is not None and (val_model_acc - val_naive_acc) >= 0.05
    check2 = hold_model_acc is not None and hold_naive_acc is not None and (hold_model_acc - hold_naive_acc) >= 0.05
    check3 = p_model_not_better_holdout is not None and p_model_not_better_holdout < 0.10
    check4 = p_hold_worse_than_val is not None and p_hold_worse_than_val < 0.90
    check5 = len(val_obs) >= 200 and len(hold_obs) >= 200

    passed = check1 and check2 and check3 and check4 and check5

    report = {
        "modal_category_by_best_of": {str(k): list(v) for k, v in modal_category.items()},
        "n_total_observations": len(observations),
        "n_val_observations": len(val_obs),
        "n_holdout_observations": len(hold_obs),
        "val_model_accuracy": val_model_acc,
        "val_naive_accuracy": val_naive_acc,
        "holdout_model_accuracy": hold_model_acc,
        "holdout_naive_accuracy": hold_naive_acc,
        "p_model_not_better_than_naive_holdout": p_model_not_better_holdout,
        "p_holdout_worse_than_val": p_hold_worse_than_val,
        "checks": {
            "1_val_model_beats_naive_by_ge_005": check1,
            "2_holdout_model_beats_naive_by_ge_005": check2,
            "3_model_confidently_beats_naive_holdout": check3,
            "4_holdout_not_confidently_worse_than_val": check4,
            "5_n_ge_200_both_windows": check5,
        },
        "passed": passed,
        "verdict": "TENNIS_SET_BETTING_GATE_PASSED" if passed else "TENNIS_SET_BETTING_GATE_FAILED",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
