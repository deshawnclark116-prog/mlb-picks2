#!/usr/bin/env python3
"""
NFL_RECV_SIM_GATE_A

Validates the SAME nfl_sim.simulate() function (no new math -- the
two-stage bootstrap is identical whether the "event" is a carry or a
target) against the untouched 2024 WR receiving_yards holdout. Same
population/eligibility as the shipped classifier (nfl_receiving_yards_
champion_walkforward_gate_a.py: WR, >=3 prior games, recent3 targets >=
5) -- workload here is TARGETS, not receptions (unlike CFB's recv_sim
gate, which uses receptions -- see nfl_pbp_foundation_a.py's docstring
for why targets is the right call for NFL specifically: it matches this
market's own already-shipped classifier feature, and an incompletion is
a real 0-yard outcome for that target, not a missing one).

Pre-registered pass (written before this script has ever been run,
same bar as nfl_rush_sim_gate_a.py):
  1. challenger mean CRPS < baseline mean CRPS (point-mass at recent3
     avg receiving yards)
  2. bootstrap P(challenger CRPS < baseline CRPS) >= 0.90
  3. challenger implied AUC >= 0.55

Run
---
python -u nfl_recv_sim_gate_a.py
"""
import argparse
import sqlite3
import sys

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import nfl_sim as sim

MODEL_DB_DEFAULT = "nfl_models/nfl_model.sqlite"
CARRY_DB_DEFAULT = "nfl_models/nfl_carry_log.sqlite"

MIN_PRIOR_GAMES_FOR_RATE = 3
MIN_RECENT_TARGETS_PER_GAME = 5
LINE = 49.5
HOLDOUT_SEASON = 2024
RECENT_GAMES_WINDOW = 8
SIMS_PER_ROW = 4000


def crps_from_samples(samples, actual):
    s = np.sort(np.asarray(samples, dtype=np.float64))
    n = len(s)
    coeff = 2.0 * np.arange(1, n + 1) - n - 1.0
    mean_abs = float(np.mean(np.abs(s - float(actual))))
    half_pairwise = float(np.sum(coeff * s) / (n * n))
    return mean_abs - half_pairwise


def auc(scores, labels):
    labels = np.asarray(labels)
    pos = labels.sum(); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    s = np.asarray(scores)[order]; i = 0; n = len(s)
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def build_asof_rows(model_con, carry_con, season):
    rows = model_con.execute("""
        SELECT player_id, player_name, team, opponent, week, game_id, targets, receiving_yards
        FROM player_games WHERE position='WR' AND season=? AND season_type='REG'
        ORDER BY player_id, week, game_date
    """, (season,)).fetchall()

    target_rows = carry_con.execute("""
        SELECT player_id, week, game_id, yards FROM recv_targets WHERE season=?
        ORDER BY player_id, week, game_id, target_index
    """, (season,)).fetchall()
    by_player_game = {}
    for pid, wk, gid, yards in target_rows:
        by_player_game.setdefault((pid, wk, gid), []).append(yards)

    out = []
    cur_pid = None
    hist = []
    for pid, pname, team, opp, week, gid, targets, rec_yards in rows:
        if pid != cur_pid:
            cur_pid = pid
            hist = []
        recent = hist[-RECENT_GAMES_WINDOW:]
        n_prior = len(hist)
        t3 = [t for (_, t, _) in hist[-3:]]
        recent_target_rate = sum(t3) / len(t3) if t3 else 0.0
        if n_prior >= MIN_PRIOR_GAMES_FOR_RATE and recent_target_rate >= MIN_RECENT_TARGETS_PER_GAME:
            counts = [t for (_, t, _) in recent]
            pool = [y for (_, _, ys) in recent for y in ys]
            if counts and pool:
                out.append({
                    "player_id": pid, "player_name": pname, "week": week,
                    "counts": counts, "pool": pool,
                    "actual": rec_yards if rec_yards is not None else 0,
                    "over_line": 1 if (rec_yards or 0) >= (LINE + 0.5) else 0,
                    "recent3_avg_rec_yards": (sum(sum(ys) for (_, _, ys) in hist[-3:])
                                               / len(hist[-3:])),
                })
        this_game_yards = by_player_game.get((pid, week, gid), [])
        hist.append((week, targets if targets is not None else 0, this_game_yards))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-db", default=MODEL_DB_DEFAULT)
    ap.add_argument("--carry-db", default=CARRY_DB_DEFAULT)
    args = ap.parse_args()

    print("NFL_RECV_SIM_GATE_A\n====================")
    model_con = sqlite3.connect(f"file:{args.model_db}?mode=ro", uri=True)
    carry_con = sqlite3.connect(f"file:{args.carry_db}?mode=ro", uri=True)

    rows = build_asof_rows(model_con, carry_con, HOLDOUT_SEASON)
    print(f"holdout {HOLDOUT_SEASON}: {len(rows)} eligible WR rows")
    if not rows:
        print("No eligible rows -- cannot gate.")
        return 1

    rng = np.random.RandomState(20260901)
    challenger_crps = np.empty(len(rows))
    baseline_crps = np.empty(len(rows))
    prob_over = np.empty(len(rows))
    over_line = np.empty(len(rows))

    for i, row in enumerate(rows):
        result = sim.simulate(row["counts"], row["pool"], LINE, sims=SIMS_PER_ROW, rng=rng)
        challenger_crps[i] = crps_from_samples(result["samples"], row["actual"])
        baseline_crps[i] = abs(row["actual"] - row["recent3_avg_rec_yards"])
        prob_over[i] = result["prob_over"]
        over_line[i] = row["over_line"]

    mean_challenger = float(np.mean(challenger_crps))
    mean_baseline = float(np.mean(baseline_crps))
    print(f"\nmean CRPS -- challenger (simulator): {mean_challenger:.3f}")
    print(f"mean CRPS -- baseline (point-mass at recent3 avg): {mean_baseline:.3f}")

    diffs = baseline_crps - challenger_crps
    n_boot = 5000
    n = len(rows)
    boot_better = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if np.mean(diffs[idx]) > 0:
            boot_better += 1
    p_challenger_better = boot_better / n_boot
    print(f"bootstrap P(challenger CRPS < baseline CRPS): {p_challenger_better:.3f}")

    challenger_auc = auc(prob_over, over_line)
    print(f"challenger implied AUC (prob_over vs actual over_line): {challenger_auc:.4f}")

    pass_crps = mean_challenger < mean_baseline
    pass_boot = p_challenger_better >= 0.90
    pass_auc = challenger_auc >= 0.55

    print("\n--- Pre-registered gate ---")
    print(f"1. challenger mean CRPS < baseline mean CRPS: {'PASS' if pass_crps else 'FAIL'}")
    print(f"2. bootstrap P(challenger better) >= 0.90: {'PASS' if pass_boot else 'FAIL'}")
    print(f"3. challenger implied AUC >= 0.55: {'PASS' if pass_auc else 'FAIL'}")

    overall = pass_crps and pass_boot and pass_auc
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
