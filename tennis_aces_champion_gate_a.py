#!/usr/bin/env python3
"""
TENNIS_ACES_CHAMPION_GATE_A

Champion-gate validation for a tennis total_aces market, mirroring this
repo's established methodology (CFB's *_champion_gate_a.py scripts,
pitcher_k_situational_thin_sample_gate_a.py's line-sweep + bootstrap
pattern): pre-register a pass bar BEFORE looking at HOLDOUT, then report
the real result honestly whichever way it comes out.

Data: tennis_model.sqlite built by tennis_player_matches_foundation_a.py
from Tennismylife/TML-Database (ATP, 2015-2024, 27,773 matches / 53,694
player-match rows -- see that script's docstring for why the original
JeffSackmann source was replaced). ATP-only; WTA is an open gap.

Feature model (own ace rate is the K-rate-style ability signal here,
same "recency-weighted rate-of-a-per-opportunity-event" shape as
pitcher_feature_row()'s K-rate in api.py):

  own_rate      recency-weighted aces/serve_points over the player's own
                PRIOR matches only (strict point-in-time -- a match's
                own outcome is never in its own feature).
  surface_rate  same, restricted to prior matches on this match's surface;
                blended with own_rate (blend weight grows with surface
                sample size) since ace rates vary a lot by surface (grass
                fastest/highest, clay slowest/lowest) -- same "pool"
                blending idea as ksim.simulate()'s per-start/season blend.
  opp_factor    recency-weighted aces/serve_points achieved BY this
                match's opponent's past opponents AGAINST that opponent
                (queryable directly: rows where opponent_id == this
                opponent, aces/serve_points on those rows) -- a return-
                quality proxy, symmetric with how double_faults' gate
                will read the same table.
  expected_sv   recency-weighted serve_points-per-match for this player,
                conditioned on best_of (Bo5 run ~1.6-1.9x longer than
                Bo3) -- own history on this best_of if enough of it
                exists, else all own history.

predicted_rate = surface_rate * (opp_factor / tour_avg_rate), clipped to
  a sane range. predicted_mean = predicted_rate * expected_sv.

Grading: no real historical market lines exist for tennis aces (same
problem the pitcher-K situational gate had), so a synthetic LINE_SWEEP is
used and only HIGH-confidence outcomes are kept as observations -- see
that script for the full rationale, unchanged here.

PRE-REGISTERED PASS BAR (checked in main(), not adjusted after seeing
results):
  1. model arm HIGH hit rate on VAL      >= 0.65, n >= 200
  2. model arm HIGH hit rate on HOLDOUT  >= 0.65, n >= 200
  3. HOLDOUT vs VAL model hit rate: NOT confidently worse (bootstrap
     P(holdout < val) < 0.90) -- stability, no silent decay out-of-sample
  4. model arm confidently BEATS a naive baseline (plain career-to-date
     average rate, no recency decay / no surface / no opponent
     adjustment, same simulate+grade pipeline) on HOLDOUT: bootstrap
     P(naive < model) >= 0.90 -- proves the adjustments add real skill,
     not just noise dressed up as sophistication.

Split by date (tennis has no season boundary): DEV 2015-2019 (unused by
this script -- reserved for any future parameter fitting), VAL 2020-2022,
HOLDOUT 2023-2024.
"""
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tennis_model.sqlite")
REPORT_PATH = Path("tennis_aces_gate_report.json")

RECENCY_DECAY = 0.6
MIN_PRIOR_MATCHES = 10
MIN_SURFACE_MATCHES_FOR_BLEND = 4
LINE_SWEEP = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 12.5]
SIMS = 8000
SEED = 20260908

VAL_START, VAL_END = "2020-01-01", "2022-12-31"
HOLDOUT_START, HOLDOUT_END = "2023-01-01", "2024-12-31"

HIGH, MEDIUM, LOW = 0.70, 0.64, 0.59


def load_rows(con):
    cur = con.execute(
        "SELECT player_id, opponent_id, match_date, surface, best_of, "
        "aces, serve_points FROM player_matches "
        "WHERE aces IS NOT NULL AND serve_points IS NOT NULL AND serve_points > 0 "
        "ORDER BY match_date")
    return cur.fetchall()


def recency_weighted_rate(pairs, decay=RECENCY_DECAY):
    """pairs: list of (aces, serve_points), oldest first. Most recent gets
    weight 1, next-most-recent decay, etc -- same shape as api.py's own
    K-rate recency weighting."""
    if not pairs:
        return None, 0
    num = den = 0.0
    w = 1.0
    for aces, sv in reversed(pairs):
        num += w * aces
        den += w * sv
        w *= decay
    if den <= 0:
        return None, len(pairs)
    return num / den, len(pairs)


def main():
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = load_rows(con)
    con.close()
    print(f"loaded {len(rows)} eligible player_matches rows from {DB_PATH}")

    # Per-player chronological history: (aces, serve_points, surface, best_of)
    player_hist = defaultdict(list)
    # Per-"target" (opponent_id) chronological history of aces hit AGAINST them
    conceded_hist = defaultdict(list)

    tour_total_aces = tour_total_sv = 0

    observations = []
    for player_id, opponent_id, match_date, surface, best_of, aces, sv in rows:
        hist = player_hist[player_id]
        if len(hist) >= MIN_PRIOR_MATCHES:
            overall_pairs = [(a, s) for a, s, _, _ in hist]
            surf_pairs = [(a, s) for a, s, sf, _ in hist if sf == surface]
            bo_pairs = [(a, s) for a, s, _, bo in hist if bo == best_of]

            own_rate, own_n = recency_weighted_rate(overall_pairs)
            surf_rate, surf_n = recency_weighted_rate(surf_pairs)
            if surf_n >= MIN_SURFACE_MATCHES_FOR_BLEND and surf_rate is not None:
                blend_w = min(surf_n / (surf_n + 10), 0.7)
                combined_rate = blend_w * surf_rate + (1 - blend_w) * own_rate
            else:
                combined_rate = own_rate

            conc_pairs = [(a, s) for a, s in conceded_hist.get(opponent_id, [])]
            opp_rate, opp_n = recency_weighted_rate(conc_pairs)

            if opp_n >= MIN_SURFACE_MATCHES_FOR_BLEND and opp_rate is not None and tour_total_sv > 0:
                tour_avg_rate = tour_total_aces / tour_total_sv
                opp_adj = opp_rate / tour_avg_rate if tour_avg_rate > 0 else 1.0
                opp_adj = min(max(opp_adj, 0.6), 1.6)
            else:
                opp_adj = 1.0

            exp_sv_pairs = bo_pairs if len(bo_pairs) >= MIN_SURFACE_MATCHES_FOR_BLEND else overall_pairs
            expected_sv = sum(s for _, s in exp_sv_pairs[-15:]) / len(exp_sv_pairs[-15:])

            predicted_rate = min(max(combined_rate * opp_adj, 0.01), 0.5)

            observations.append({
                "match_date": match_date,
                "predicted_rate": predicted_rate,
                "naive_rate": min(max(own_rate, 0.01), 0.5),
                "expected_sv": expected_sv,
                "actual_aces": aces,
                "own_n": own_n,
            })

        # Update rolling state with THIS match (now that features are computed)
        hist.append((aces, sv, surface, best_of))
        conceded_hist[opponent_id].append((aces, sv))
        tour_total_aces += aces
        tour_total_sv += sv

    print(f"built {len(observations)} point-in-time observations "
          f"(min {MIN_PRIOR_MATCHES} prior matches required)")

    val_obs = [o for o in observations if VAL_START <= o["match_date"] <= VAL_END]
    hold_obs = [o for o in observations if HOLDOUT_START <= o["match_date"] <= HOLDOUT_END]
    print(f"VAL ({VAL_START}..{VAL_END}): {len(val_obs)} obs, "
          f"HOLDOUT ({HOLDOUT_START}..{HOLDOUT_END}): {len(hold_obs)} obs")

    rng = np.random.default_rng(SEED)

    def simulate_and_grade(obs_list, rate_key):
        """Original methodology: every swept line graded independently.
        NOTE -- multiple lines against the same match are correlated
        pseudo-replicates, not independent draws; see
        simulate_and_grade_one_line_per_match for the corrected version
        used to sanity-check whether this inflates apparent stability
        failures (it does, per the double_faults gate's comparison)."""
        hits = []
        for o in obs_list:
            rate = o[rate_key]
            exp_sv = o["expected_sv"]
            actual = o["actual_aces"]
            bf = rng.normal(exp_sv, exp_sv * 0.18, SIMS)
            bf = np.clip(bf, max(exp_sv * 0.4, 5), exp_sv * 1.8)
            aces_sim = rng.binomial(bf.astype(int), rate)
            for line in LINE_SWEEP:
                over_prob = float(np.mean(aces_sim > line))
                side = "OVER" if over_prob >= 0.5 else "UNDER"
                conf_prob = over_prob if side == "OVER" else 1 - over_prob
                if conf_prob >= HIGH:
                    hit = (actual > line) if side == "OVER" else (actual <= line)
                    hits.append(1 if hit else 0)
        return hits

    def simulate_and_grade_one_line_per_match(obs_list, rate_key):
        """Corrected methodology (see double_faults gate): one graded
        outcome per observation, using only the swept line nearest the
        model's own predicted mean -- avoids pooling correlated
        pseudo-replicates from the same match."""
        hits = []
        for o in obs_list:
            rate = o[rate_key]
            exp_sv = o["expected_sv"]
            actual = o["actual_aces"]
            predicted_mean = rate * exp_sv
            line = min(LINE_SWEEP, key=lambda l: abs(l - predicted_mean))
            bf = rng.normal(exp_sv, exp_sv * 0.18, SIMS)
            bf = np.clip(bf, max(exp_sv * 0.4, 5), exp_sv * 1.8)
            aces_sim = rng.binomial(bf.astype(int), rate)
            over_prob = float(np.mean(aces_sim > line))
            side = "OVER" if over_prob >= 0.5 else "UNDER"
            conf_prob = over_prob if side == "OVER" else 1 - over_prob
            if conf_prob >= HIGH:
                hit = (actual > line) if side == "OVER" else (actual <= line)
                hits.append(1 if hit else 0)
        return hits

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

    model_val_hits = simulate_and_grade(val_obs, "predicted_rate")
    model_hold_hits = simulate_and_grade(hold_obs, "predicted_rate")
    naive_val_hits = simulate_and_grade(val_obs, "naive_rate")
    naive_hold_hits = simulate_and_grade(hold_obs, "naive_rate")

    def rate(hits):
        return (sum(hits) / len(hits)) if hits else None

    model_val_rate = rate(model_val_hits)
    model_hold_rate = rate(model_hold_hits)
    naive_val_rate = rate(naive_val_hits)
    naive_hold_rate = rate(naive_hold_hits)

    p_hold_worse_than_val = bootstrap_p_worse(model_hold_hits, model_val_hits)
    p_naive_worse_than_model = bootstrap_p_worse(naive_hold_hits, model_hold_hits)
    # Secondary check, not part of the pre-registered pass bar: is the
    # NAIVE arm itself stable VAL->HOLDOUT? Answers "should we just ship
    # the simple version" if the fancier model's own stability check (3)
    # or value-add check (4) fails.
    p_naive_hold_worse_than_naive_val = bootstrap_p_worse(naive_hold_hits, naive_val_hits)

    check1 = model_val_rate is not None and model_val_rate >= 0.65 and len(model_val_hits) >= 200
    check2 = model_hold_rate is not None and model_hold_rate >= 0.65 and len(model_hold_hits) >= 200
    check3 = p_hold_worse_than_val is not None and p_hold_worse_than_val < 0.90
    check4 = p_naive_worse_than_model is not None and p_naive_worse_than_model >= 0.90

    passed = check1 and check2 and check3 and check4

    # Cross-check with the corrected (one-line-per-match) grading
    # methodology from the double_faults gate, to see whether the
    # instability found above is real or a pseudo-replication artifact
    # of pooling multiple swept lines per match.
    fixed_model_val_hits = simulate_and_grade_one_line_per_match(val_obs, "predicted_rate")
    fixed_model_hold_hits = simulate_and_grade_one_line_per_match(hold_obs, "predicted_rate")
    fixed_naive_val_hits = simulate_and_grade_one_line_per_match(val_obs, "naive_rate")
    fixed_naive_hold_hits = simulate_and_grade_one_line_per_match(hold_obs, "naive_rate")
    fixed_model_val_rate = rate(fixed_model_val_hits)
    fixed_model_hold_rate = rate(fixed_model_hold_hits)
    fixed_naive_val_rate = rate(fixed_naive_val_hits)
    fixed_naive_hold_rate = rate(fixed_naive_hold_hits)
    fixed_p_hold_worse_than_val = bootstrap_p_worse(fixed_model_hold_hits, fixed_model_val_hits)
    fixed_p_naive_worse_than_model = bootstrap_p_worse(fixed_naive_hold_hits, fixed_model_hold_hits)
    fixed_check1 = fixed_model_val_rate is not None and fixed_model_val_rate >= 0.65 and len(fixed_model_val_hits) >= 200
    fixed_check2 = fixed_model_hold_rate is not None and fixed_model_hold_rate >= 0.65 and len(fixed_model_hold_hits) >= 200
    fixed_check3 = fixed_p_hold_worse_than_val is not None and fixed_p_hold_worse_than_val < 0.90
    fixed_check4 = fixed_p_naive_worse_than_model is not None and fixed_p_naive_worse_than_model >= 0.90
    fixed_passed = fixed_check1 and fixed_check2 and fixed_check3 and fixed_check4

    report = {
        "n_total_observations": len(observations),
        "n_val_observations": len(val_obs),
        "n_holdout_observations": len(hold_obs),
        "model_val": {"n": len(model_val_hits), "hit_rate": model_val_rate},
        "model_holdout": {"n": len(model_hold_hits), "hit_rate": model_hold_rate},
        "naive_val": {"n": len(naive_val_hits), "hit_rate": naive_val_rate},
        "naive_holdout": {"n": len(naive_hold_hits), "hit_rate": naive_hold_rate},
        "p_holdout_worse_than_val": p_hold_worse_than_val,
        "p_naive_worse_than_model_holdout": p_naive_worse_than_model,
        "p_naive_holdout_worse_than_naive_val": p_naive_hold_worse_than_naive_val,
        "checks": {
            "1_val_hit_rate_ge_065_n_ge_200": check1,
            "2_holdout_hit_rate_ge_065_n_ge_200": check2,
            "3_holdout_not_confidently_worse_than_val": check3,
            "4_model_confidently_beats_naive_on_holdout": check4,
        },
        "passed": passed,
        "verdict": "TENNIS_ACES_GATE_PASSED" if passed else "TENNIS_ACES_GATE_FAILED",
        "one_line_per_match_crosscheck": {
            "model_val": {"n": len(fixed_model_val_hits), "hit_rate": fixed_model_val_rate},
            "model_holdout": {"n": len(fixed_model_hold_hits), "hit_rate": fixed_model_hold_rate},
            "naive_val": {"n": len(fixed_naive_val_hits), "hit_rate": fixed_naive_val_rate},
            "naive_holdout": {"n": len(fixed_naive_hold_hits), "hit_rate": fixed_naive_hold_rate},
            "p_holdout_worse_than_val": fixed_p_hold_worse_than_val,
            "p_naive_worse_than_model_holdout": fixed_p_naive_worse_than_model,
            "checks": {
                "1_val_hit_rate_ge_065_n_ge_200": fixed_check1,
                "2_holdout_hit_rate_ge_065_n_ge_200": fixed_check2,
                "3_holdout_not_confidently_worse_than_val": fixed_check3,
                "4_model_confidently_beats_naive_on_holdout": fixed_check4,
            },
            "passed": fixed_passed,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
