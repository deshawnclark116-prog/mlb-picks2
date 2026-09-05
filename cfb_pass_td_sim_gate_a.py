#!/usr/bin/env python3
"""
CFB_PASS_TD_SIM_GATE_A

passing_touchdowns is a low-count binary-per-attempt stat, not a
continuous yardage stat -- but cfb_rush_sim.simulate()'s bootstrap
doesn't actually assume continuity anywhere: it draws N per-event
outcomes with replacement from a pooled set of the player's own real
recent per-event outcomes and sums them. A touchdown-or-not flag (0/1)
per attempt is just another real per-event outcome, so this reuses the
SAME function again, unchanged: counts = pass attempts per recent game,
pool = pooled 0/1 touchdown flags from those same recent attempts,
line = 1.5. No separate ksim-style parametric-rate module needed --
tested empirically rather than assumed, per this repo's predictions-
first validation discipline.

Same population as the shipped classifier (cfb_passing_touchdowns_
champion_gate_a.py: QB, all-division, no Power4 scoping, >=3 prior
games, recent3 attempts >= 15).

Pre-registered pass (written before this script has ever been run):
  1. challenger mean CRPS < baseline mean CRPS (point-mass at recent3
     avg passing TDs)
  2. bootstrap P(challenger CRPS < baseline CRPS) >= 0.90
  3. challenger implied AUC >= 0.55

Run
---
python -u cfb_pass_td_sim_gate_a.py
"""
import sqlite3
import sys

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import cfb_rush_sim as sim

MODEL_DB = "cfb_models/cfb_model.sqlite"
CARRY_DB = "cfb_models/cfb_carry_log.sqlite"

MIN_PRIOR_GAMES_FOR_RATE = 3
MIN_RECENT_ATTEMPTS_PER_GAME = 15
LINE = 1.5
HOLDOUT_SEASON = 2025
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
        SELECT player_id, player_name, team, opponent, week, game_id, pass_attempts, passing_touchdowns
        FROM player_games WHERE position='QB' AND season=?
        ORDER BY player_id, week, game_date
    """, (season,)).fetchall()

    attempt_rows = carry_con.execute("""
        SELECT player_id, week, game_id, is_touchdown FROM pass_attempts_log WHERE season=?
        ORDER BY player_id, week, game_id, attempt_index
    """, (season,)).fetchall()
    by_player_game = {}
    for pid, wk, gid, is_td in attempt_rows:
        by_player_game.setdefault((pid, wk, gid), []).append(is_td)

    out = []
    cur_pid = None
    hist = []
    for pid, pname, team, opp, week, gid, attempts, pass_td in rows:
        if pid != cur_pid:
            cur_pid = pid
            hist = []
        recent = hist[-RECENT_GAMES_WINDOW:]
        n_prior = len(hist)
        c3 = [c for (_, c, _) in hist[-3:]]
        recent_rate = sum(c3) / len(c3) if c3 else 0.0
        if n_prior >= MIN_PRIOR_GAMES_FOR_RATE and recent_rate >= MIN_RECENT_ATTEMPTS_PER_GAME:
            counts = [c for (_, c, _) in recent]
            pool = [f for (_, _, fs) in recent for f in fs]
            if counts and pool:
                out.append({
                    "player_id": pid, "player_name": pname, "week": week,
                    "counts": counts, "pool": pool,
                    "actual": pass_td if pass_td is not None else 0,
                    "over_line": 1 if (pass_td or 0) >= (LINE + 0.5) else 0,
                    "recent3_avg_pass_td": (sum(sum(fs) for (_, _, fs) in hist[-3:])
                                              / len(hist[-3:])),
                })
        this_game_flags = by_player_game.get((pid, week, gid), [])
        hist.append((week, attempts if attempts is not None else 0, this_game_flags))
    return out


def main():
    print("CFB_PASS_TD_SIM_GATE_A\n=======================")
    model_con = sqlite3.connect(f"file:{MODEL_DB}?mode=ro", uri=True)
    carry_con = sqlite3.connect(f"file:{CARRY_DB}?mode=ro", uri=True)

    rows = build_asof_rows(model_con, carry_con, HOLDOUT_SEASON)
    print(f"holdout {HOLDOUT_SEASON}: {len(rows)} eligible QB rows (all-division)")
    if not rows:
        print("No eligible rows -- cannot gate.")
        return 1

    rng = np.random.RandomState(20250901)
    challenger_crps = np.empty(len(rows))
    baseline_crps = np.empty(len(rows))
    prob_over = np.empty(len(rows))
    over_line = np.empty(len(rows))

    for i, row in enumerate(rows):
        result = sim.simulate(row["counts"], row["pool"], LINE, sims=SIMS_PER_ROW, rng=rng)
        challenger_crps[i] = crps_from_samples(result["samples"], row["actual"])
        baseline_crps[i] = abs(row["actual"] - row["recent3_avg_pass_td"])
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
