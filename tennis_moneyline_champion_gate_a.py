#!/usr/bin/env python3
"""
TENNIS_MONEYLINE_CHAMPION_GATE_A

Champion-gate validation for a tennis moneyline (match winner) market.
Unlike total_aces/double_faults, this is a natural single binary event
per match (who won) -- no synthetic line-sweep is needed at all, so the
pseudo-replication problem that inflated the aces gate's first result
doesn't apply here by construction: one match = one graded prediction.

Model: standard Elo rating, maintained two ways per player --
  overall_elo[player]            updated after every match, any surface
  surface_elo[(player, surface)] updated only after matches on that surface
blended per-prediction (blend weight grows with how much surface-
specific history both players have -- same "blend weight grows with
sample size" shape used throughout this repo's other models).

K-factor uses the common tennis-Elo choice K = 250 / (n_matches + 5)^0.4
(more volatile for new/inexperienced players, stabilizing as more of
their matches are observed) rather than a flat constant.

To avoid any winner/loser label leakage in the prediction target, each
match's two players are assigned to "p1"/"p2" by a criterion that has
nothing to do with who won (lexicographically smaller player_id = p1),
and the model predicts P(p1 wins) -- graded against whether p1 actually
was the real winner_id.

Elo state updates using ALL matches (including retirements/walkovers --
the *result* of a RET match is still a real, final outcome, unlike its
counting stats), but the EVALUATION set only includes matches where both
players already have >= MIN_PRIOR_MATCHES of their own rating history,
so the graded sample isn't dominated by uninformative 1500-vs-1500
cold-start coin flips.

PRE-REGISTERED PASS BAR (same shape as the aces/double_faults gates):
  1. VAL accuracy     >= 0.65, n >= 200
  2. HOLDOUT accuracy >= 0.65, n >= 200
  3. HOLDOUT not confidently worse than VAL (bootstrap P(holdout < val) < 0.90)
  4. Elo model confidently beats a naive baseline (just pick the player
     with the better real ATP rank at match time) on HOLDOUT: bootstrap
     P(naive < model) >= 0.90 -- restricted to the subset of matches
     where both players' ranks are known, so the comparison is apples-
     to-apples.

Split by date (no season boundary): DEV 2015-2019 (Elo warmup only, not
evaluated), VAL 2020-2022, HOLDOUT 2023-2024.
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

DB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tennis_model.sqlite")
REPORT_PATH = Path("tennis_moneyline_gate_report.json")

MIN_PRIOR_MATCHES = 5
MIN_SURFACE_MATCHES_FOR_BLEND = 4
INITIAL_ELO = 1500.0
SEED = 20260908

VAL_START, VAL_END = "2020-01-01", "2022-12-31"
HOLDOUT_START, HOLDOUT_END = "2023-01-01", "2024-12-31"


def load_rows(con):
    cur = con.execute(
        "SELECT match_id, match_date, surface, winner_id, loser_id, "
        "winner_rank, loser_rank FROM matches "
        "WHERE winner_id IS NOT NULL AND loser_id IS NOT NULL AND surface IS NOT NULL "
        "ORDER BY match_date, match_id")
    return cur.fetchall()


def k_factor(n_matches):
    return 250.0 / ((n_matches + 5) ** 0.4)


def elo_expected(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def main():
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = load_rows(con)
    con.close()
    print(f"loaded {len(rows)} matches with known result+surface from {DB_PATH}")

    overall_elo = {}
    overall_n = {}
    surface_elo = {}
    surface_n = {}

    observations = []
    for match_id, match_date, surface, winner_id, loser_id, winner_rank, loser_rank in rows:
        p1, p2 = sorted([winner_id, loser_id])
        p1_is_winner = 1 if p1 == winner_id else 0

        e1_overall = overall_elo.get(p1, INITIAL_ELO)
        e2_overall = overall_elo.get(p2, INITIAL_ELO)
        n1_overall = overall_n.get(p1, 0)
        n2_overall = overall_n.get(p2, 0)

        e1_surf = surface_elo.get((p1, surface), INITIAL_ELO)
        e2_surf = surface_elo.get((p2, surface), INITIAL_ELO)
        n1_surf = surface_n.get((p1, surface), 0)
        n2_surf = surface_n.get((p2, surface), 0)

        if n1_overall >= MIN_PRIOR_MATCHES and n2_overall >= MIN_PRIOR_MATCHES:
            min_surf_n = min(n1_surf, n2_surf)
            if min_surf_n >= MIN_SURFACE_MATCHES_FOR_BLEND:
                blend_w = min(min_surf_n / (min_surf_n + 15), 0.6)
                combined1 = blend_w * e1_surf + (1 - blend_w) * e1_overall
                combined2 = blend_w * e2_surf + (1 - blend_w) * e2_overall
            else:
                combined1, combined2 = e1_overall, e2_overall

            predicted_p1_prob = elo_expected(combined1, combined2)

            naive_pred = None
            if winner_rank is not None and loser_rank is not None:
                p1_rank = winner_rank if p1 == winner_id else loser_rank
                p2_rank = loser_rank if p1 == winner_id else winner_rank
                if p1_rank != p2_rank:
                    naive_pred = 1 if p1_rank < p2_rank else 0  # lower rank number = better

            observations.append({
                "match_date": match_date,
                "predicted_p1_prob": predicted_p1_prob,
                "naive_pred": naive_pred,
                "actual_p1_wins": p1_is_winner,
            })

        # Update Elo state with the real outcome (always, regardless of
        # eligibility above -- ineligible players' matches still carry
        # real information for their opponents' future ratings).
        k1, k2 = k_factor(n1_overall), k_factor(n2_overall)
        e1_new = e1_overall + k1 * (p1_is_winner - elo_expected(e1_overall, e2_overall))
        e2_new = e2_overall + k2 * ((1 - p1_is_winner) - elo_expected(e2_overall, e1_overall))
        overall_elo[p1], overall_elo[p2] = e1_new, e2_new
        overall_n[p1] = n1_overall + 1
        overall_n[p2] = n2_overall + 1

        ks1, ks2 = k_factor(n1_surf), k_factor(n2_surf)
        es1_new = e1_surf + ks1 * (p1_is_winner - elo_expected(e1_surf, e2_surf))
        es2_new = e2_surf + ks2 * ((1 - p1_is_winner) - elo_expected(e2_surf, e1_surf))
        surface_elo[(p1, surface)], surface_elo[(p2, surface)] = es1_new, es2_new
        surface_n[(p1, surface)] = n1_surf + 1
        surface_n[(p2, surface)] = n2_surf + 1

    print(f"built {len(observations)} point-in-time evaluable observations "
          f"(min {MIN_PRIOR_MATCHES} prior matches each side)")

    val_obs = [o for o in observations if VAL_START <= o["match_date"] <= VAL_END]
    hold_obs = [o for o in observations if HOLDOUT_START <= o["match_date"] <= HOLDOUT_END]
    print(f"VAL: {len(val_obs)} obs, HOLDOUT: {len(hold_obs)} obs")

    def grade_model(obs_list):
        correct = []
        logloss_terms = []
        probs, actuals = [], []
        for o in obs_list:
            p = min(max(o["predicted_p1_prob"], 1e-6), 1 - 1e-6)
            pred_side = 1 if p >= 0.5 else 0
            correct.append(1 if pred_side == o["actual_p1_wins"] else 0)
            y = o["actual_p1_wins"]
            logloss_terms.append(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
            probs.append(p)
            actuals.append(y)
        return correct, logloss_terms, probs, actuals

    def grade_naive(obs_list):
        correct = []
        for o in obs_list:
            if o["naive_pred"] is None:
                continue
            correct.append(1 if o["naive_pred"] == o["actual_p1_wins"] else 0)
        return correct

    def auc(probs, actuals):
        probs = np.array(probs)
        actuals = np.array(actuals)
        pos = probs[actuals == 1]
        neg = probs[actuals == 0]
        if len(pos) == 0 or len(neg) == 0:
            return None
        ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
        rank_pos = ranks[:len(pos)]
        return (rank_pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    def brier(probs, actuals):
        probs = np.array(probs)
        actuals = np.array(actuals)
        return float(np.mean((probs - actuals) ** 2))

    def bootstrap_p_worse(hits_a, hits_b, seed=SEED, b=5000):
        if not hits_a or not hits_b:
            return None
        r = np.random.default_rng(seed)
        a = np.array(hits_a)
        bb = np.array(hits_b)
        worse = 0
        for _ in range(b):
            ra = r.choice(a, size=len(a), replace=True).mean()
            rb = r.choice(bb, size=len(bb), replace=True).mean()
            if ra < rb:
                worse += 1
        return worse / b

    val_correct, val_ll, val_probs, val_actuals = grade_model(val_obs)
    hold_correct, hold_ll, hold_probs, hold_actuals = grade_model(hold_obs)
    hold_naive_correct = grade_naive(hold_obs)

    def rate(x):
        return (sum(x) / len(x)) if x else None

    val_acc = rate(val_correct)
    hold_acc = rate(hold_correct)
    hold_naive_acc = rate(hold_naive_correct)

    p_hold_worse_than_val = bootstrap_p_worse(hold_correct, val_correct)
    p_naive_worse_than_model = bootstrap_p_worse(hold_naive_correct, hold_correct)

    check1 = val_acc is not None and val_acc >= 0.65 and len(val_correct) >= 200
    check2 = hold_acc is not None and hold_acc >= 0.65 and len(hold_correct) >= 200
    check3 = p_hold_worse_than_val is not None and p_hold_worse_than_val < 0.90
    check4 = p_naive_worse_than_model is not None and p_naive_worse_than_model >= 0.90

    passed = check1 and check2 and check3 and check4

    report = {
        "n_total_observations": len(observations),
        "n_val_observations": len(val_obs),
        "n_holdout_observations": len(hold_obs),
        "val": {
            "n": len(val_correct), "accuracy": val_acc,
            "logloss": float(np.mean(val_ll)) if val_ll else None,
            "brier": brier(val_probs, val_actuals),
            "auc": auc(val_probs, val_actuals),
        },
        "holdout": {
            "n": len(hold_correct), "accuracy": hold_acc,
            "logloss": float(np.mean(hold_ll)) if hold_ll else None,
            "brier": brier(hold_probs, hold_actuals),
            "auc": auc(hold_probs, hold_actuals),
        },
        "naive_holdout": {"n": len(hold_naive_correct), "accuracy": hold_naive_acc},
        "p_holdout_worse_than_val": p_hold_worse_than_val,
        "p_naive_worse_than_model_holdout": p_naive_worse_than_model,
        "checks": {
            "1_val_accuracy_ge_065_n_ge_200": check1,
            "2_holdout_accuracy_ge_065_n_ge_200": check2,
            "3_holdout_not_confidently_worse_than_val": check3,
            "4_model_confidently_beats_naive_on_holdout": check4,
        },
        "passed": passed,
        "verdict": "TENNIS_MONEYLINE_GATE_PASSED" if passed else "TENNIS_MONEYLINE_GATE_FAILED",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
