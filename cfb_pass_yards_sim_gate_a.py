#!/usr/bin/env python3
"""
CFB_PASS_YARDS_SIM_GATE_A

Same cfb_rush_sim.simulate() bootstrap, this time on real per-attempt
passing yardage: counts = pass attempts in a recent real game, pool =
pooled per-attempt yards from those games (completions carry their real
yards; incompletions and interceptions correctly contribute 0, matching
the shipped classifier's own passing_yards aggregation logic). Same
population/eligibility as the shipped classifier
(cfb_passing_yards_champion_gate_c.py: QB, Power4-vs-Power4 only, >=3
prior games, recent3 attempts >= 15), so directly comparable to its
validated holdout AUC of 0.6609.

Pre-registered pass (written before this script has ever been run):
  1. challenger mean CRPS < baseline mean CRPS (point-mass at recent3
     avg passing yards)
  2. bootstrap P(challenger CRPS < baseline CRPS) >= 0.90
  3. challenger implied AUC >= 0.55

Run
---
python -u cfb_pass_yards_sim_gate_a.py
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
LINE = 214.5
HOLDOUT_SEASON = 2025
RECENT_GAMES_WINDOW = 8
SIMS_PER_ROW = 4000
POWER4 = {"Big Ten", "ACC", "SEC", "Big 12"}


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


def power4_game_ids(model_con):
    rows = model_con.execute(
        "SELECT game_id FROM games WHERE home_conference IN ({0}) AND away_conference IN ({0})"
        .format(",".join("?" for _ in POWER4)), tuple(POWER4) * 2).fetchall()
    return {r[0] for r in rows}


def build_asof_rows(model_con, carry_con, season):
    p4_games = power4_game_ids(model_con)
    rows = model_con.execute("""
        SELECT player_id, player_name, team, opponent, week, game_id, pass_attempts, passing_yards
        FROM player_games WHERE position='QB' AND season=?
        ORDER BY player_id, week, game_date
    """, (season,)).fetchall()

    attempt_rows = carry_con.execute("""
        SELECT player_id, week, game_id, yards FROM pass_attempts_log WHERE season=?
        ORDER BY player_id, week, game_id, attempt_index
    """, (season,)).fetchall()
    by_player_game = {}
    for pid, wk, gid, yards in attempt_rows:
        by_player_game.setdefault((pid, wk, gid), []).append(yards)

    out = []
    cur_pid = None
    hist = []
    for pid, pname, team, opp, week, gid, attempts, pass_yards in rows:
        if pid != cur_pid:
            cur_pid = pid
            hist = []
        if gid in p4_games:
            recent = hist[-RECENT_GAMES_WINDOW:]
            n_prior = len(hist)
            c3 = [c for (_, c, _) in hist[-3:]]
            recent_rate = sum(c3) / len(c3) if c3 else 0.0
            if n_prior >= MIN_PRIOR_GAMES_FOR_RATE and recent_rate >= MIN_RECENT_ATTEMPTS_PER_GAME:
                counts = [c for (_, c, _) in recent]
                pool = [y for (_, _, ys) in recent for y in ys]
                if counts and pool:
                    out.append({
                        "player_id": pid, "player_name": pname, "week": week,
                        "counts": counts, "pool": pool,
                        "actual": pass_yards if pass_yards is not None else 0,
                        "over_line": 1 if (pass_yards or 0) >= (LINE + 0.5) else 0,
                        "recent3_avg_pass_yards": (sum(sum(ys) for (_, _, ys) in hist[-3:])
                                                     / len(hist[-3:])),
                    })
        this_game_yards = by_player_game.get((pid, week, gid), [])
        hist.append((week, attempts if attempts is not None else 0, this_game_yards))
    return out


def main():
    print("CFB_PASS_YARDS_SIM_GATE_A\n==========================")
    model_con = sqlite3.connect(f"file:{MODEL_DB}?mode=ro", uri=True)
    carry_con = sqlite3.connect(f"file:{CARRY_DB}?mode=ro", uri=True)

    rows = build_asof_rows(model_con, carry_con, HOLDOUT_SEASON)
    print(f"holdout {HOLDOUT_SEASON}: {len(rows)} eligible QB rows (Power4-vs-Power4 only)")
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
        baseline_crps[i] = abs(row["actual"] - row["recent3_avg_pass_yards"])
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
    print("(shipped classifier's validated holdout AUC for comparison: 0.6609)")

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
