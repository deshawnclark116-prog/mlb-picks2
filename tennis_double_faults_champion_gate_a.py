#!/usr/bin/env python3
"""
TENNIS_DOUBLE_FAULTS_CHAMPION_GATE_A

Same data/machinery as tennis_aces_champion_gate_a.py (recency-weighted
own rate, surface blend, opponent-conceded-rate adjustment, best_of-aware
expected serve_points) applied to double_faults instead of aces.

METHODOLOGY FIX vs the aces gate: that script graded every line in
LINE_SWEEP independently per observation and pooled all HIGH-confidence
line/observation pairs into one bootstrap sample. Multiple swept lines
against the SAME underlying match are not independent draws -- pooling
them inflates the bootstrap's apparent statistical power (n counts
correlated pseudo-replicates as if they were independent), which is the
likely reason a real but small (~1pp) VAL->HOLDOUT gap bootstrapped to
p=1.0 ("confidently worse") in the aces gate. Here, each observation
contributes AT MOST ONE graded outcome: the single swept line closest to
the model's own predicted mean (the line an actual sportsbook would post
closest to), only kept if that one line clears the HIGH-confidence bar.
This is a fairer test of whether the aces gate's instability finding was
about aces specifically or about that shared grading flaw -- if
double_faults ALSO fails check 3 under this corrected, de-duplicated
methodology, the problem is real and market-specific (or a genuine era
effect); if it now PASSES cleanly, the aces gate's failure was likely a
statistical-power artifact worth revisiting with this same fix.

Same pre-registered pass bar as the aces gate (1: VAL hit rate >= 0.65,
n>=200; 2: HOLDOUT hit rate >= 0.65, n>=200; 3: HOLDOUT not confidently
worse than VAL; 4: model confidently beats naive baseline on HOLDOUT),
same VAL/HOLDOUT date split, decided before running.
"""
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tennis_model.sqlite")
REPORT_PATH = Path("tennis_double_faults_gate_report.json")

RECENCY_DECAY = 0.6
MIN_PRIOR_MATCHES = 10
MIN_SURFACE_MATCHES_FOR_BLEND = 4
LINE_SWEEP = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
SIMS = 8000
SEED = 20260908

VAL_START, VAL_END = "2020-01-01", "2022-12-31"
HOLDOUT_START, HOLDOUT_END = "2023-01-01", "2024-12-31"

HIGH = 0.70


def load_rows(con):
    cur = con.execute(
        "SELECT player_id, opponent_id, match_date, surface, best_of, "
        "double_faults, serve_points FROM player_matches "
        "WHERE double_faults IS NOT NULL AND serve_points IS NOT NULL AND serve_points > 0 "
        "ORDER BY match_date")
    return cur.fetchall()


def recency_weighted_rate(pairs, decay=RECENCY_DECAY):
    if not pairs:
        return None, 0
    num = den = 0.0
    w = 1.0
    for df, sv in reversed(pairs):
        num += w * df
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

    player_hist = defaultdict(list)
    conceded_hist = defaultdict(list)
    tour_total_df = tour_total_sv = 0

    observations = []
    for player_id, opponent_id, match_date, surface, best_of, df, sv in rows:
        hist = player_hist[player_id]
        if len(hist) >= MIN_PRIOR_MATCHES:
            overall_pairs = [(d, s) for d, s, _, _ in hist]
            surf_pairs = [(d, s) for d, s, sf, _ in hist if sf == surface]
            bo_pairs = [(d, s) for d, s, _, bo in hist if bo == best_of]

            own_rate, own_n = recency_weighted_rate(overall_pairs)
            surf_rate, surf_n = recency_weighted_rate(surf_pairs)
            if surf_n >= MIN_SURFACE_MATCHES_FOR_BLEND and surf_rate is not None:
                blend_w = min(surf_n / (surf_n + 10), 0.7)
                combined_rate = blend_w * surf_rate + (1 - blend_w) * own_rate
            else:
                combined_rate = own_rate

            conc_pairs = list(conceded_hist.get(opponent_id, []))
            opp_rate, opp_n = recency_weighted_rate(conc_pairs)
            if opp_n >= MIN_SURFACE_MATCHES_FOR_BLEND and opp_rate is not None and tour_total_sv > 0:
                tour_avg_rate = tour_total_df / tour_total_sv
                opp_adj = opp_rate / tour_avg_rate if tour_avg_rate > 0 else 1.0
                opp_adj = min(max(opp_adj, 0.6), 1.6)
            else:
                opp_adj = 1.0

            exp_sv_pairs = bo_pairs if len(bo_pairs) >= MIN_SURFACE_MATCHES_FOR_BLEND else overall_pairs
            expected_sv = sum(s for _, s in exp_sv_pairs[-15:]) / len(exp_sv_pairs[-15:])

            predicted_rate = min(max(combined_rate * opp_adj, 0.005), 0.25)

            observations.append({
                "match_date": match_date,
                "predicted_rate": predicted_rate,
                "naive_rate": min(max(own_rate, 0.005), 0.25),
                "expected_sv": expected_sv,
                "actual_df": df,
            })

        hist.append((df, sv, surface, best_of))
        conceded_hist[opponent_id].append((df, sv))
        tour_total_df += df
        tour_total_sv += sv

    print(f"built {len(observations)} point-in-time observations")

    val_obs = [o for o in observations if VAL_START <= o["match_date"] <= VAL_END]
    hold_obs = [o for o in observations if HOLDOUT_START <= o["match_date"] <= HOLDOUT_END]
    print(f"VAL: {len(val_obs)} obs, HOLDOUT: {len(hold_obs)} obs")

    rng = np.random.default_rng(SEED)

    def simulate_and_grade_one_line_per_match(obs_list, rate_key):
        """Fix vs the aces gate: grade only the single swept line closest
        to the predicted mean per observation, not every line -- avoids
        pooling correlated pseudo-replicates from the same match."""
        hits = []
        for o in obs_list:
            rate = o[rate_key]
            exp_sv = o["expected_sv"]
            actual = o["actual_df"]
            predicted_mean = rate * exp_sv
            line = min(LINE_SWEEP, key=lambda l: abs(l - predicted_mean))

            bf = rng.normal(exp_sv, exp_sv * 0.18, SIMS)
            bf = np.clip(bf, max(exp_sv * 0.4, 5), exp_sv * 1.8)
            df_sim = rng.binomial(bf.astype(int), rate)
            over_prob = float(np.mean(df_sim > line))
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

    model_val_hits = simulate_and_grade_one_line_per_match(val_obs, "predicted_rate")
    model_hold_hits = simulate_and_grade_one_line_per_match(hold_obs, "predicted_rate")
    naive_val_hits = simulate_and_grade_one_line_per_match(val_obs, "naive_rate")
    naive_hold_hits = simulate_and_grade_one_line_per_match(hold_obs, "naive_rate")

    def rate(hits):
        return (sum(hits) / len(hits)) if hits else None

    model_val_rate = rate(model_val_hits)
    model_hold_rate = rate(model_hold_hits)
    naive_val_rate = rate(naive_val_hits)
    naive_hold_rate = rate(naive_hold_hits)

    p_hold_worse_than_val = bootstrap_p_worse(model_hold_hits, model_val_hits)
    p_naive_worse_than_model = bootstrap_p_worse(naive_hold_hits, model_hold_hits)
    p_naive_hold_worse_than_naive_val = bootstrap_p_worse(naive_hold_hits, naive_val_hits)

    check1 = model_val_rate is not None and model_val_rate >= 0.65 and len(model_val_hits) >= 200
    check2 = model_hold_rate is not None and model_hold_rate >= 0.65 and len(model_hold_hits) >= 200
    check3 = p_hold_worse_than_val is not None and p_hold_worse_than_val < 0.90
    check4 = p_naive_worse_than_model is not None and p_naive_worse_than_model >= 0.90

    passed = check1 and check2 and check3 and check4

    report = {
        "grading_methodology": "one_line_per_match_nearest_to_predicted_mean",
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
        "verdict": "TENNIS_DOUBLE_FAULTS_GATE_PASSED" if passed else "TENNIS_DOUBLE_FAULTS_GATE_FAILED",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
