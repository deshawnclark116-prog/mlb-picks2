#!/usr/bin/env python3
"""
NFL_RUSH_SIM_GATE_A

Validates nfl_sim.py against the untouched 2024 holdout, same
population/eligibility as the shipped classifier (nfl_rushing_yards_
champion_gate_a.py: RB, >=3 prior games, recent3 carries >= 12) so the
two are directly comparable. Mirrors cfb_rush_sim_gate_a.py exactly --
same CRPS-based champion-vs-challenger framing already used throughout
this repo for continuous targets:

  baseline    a degenerate point-mass "distribution" at the player's
              own recent3_avg_rush_yards (no uncertainty at all -- CRPS
              of a point mass reduces to plain absolute error). Does
              actually simulating a distribution beat just guessing the
              recent average?
  challenger  nfl_sim.py's real Monte Carlo simulation, built on real
              per-carry yardage from nflverse's play-by-play (see
              nfl_pbp_foundation_a.py) -- NOT the same model as the
              nfl_per_player_line_gate_a.py regression test (that one
              tried to beat the naive average on point-prediction MAE
              and failed; this one asks a different, more useful
              question -- does a full simulated distribution, not just
              a single number, beat a bare point guess).

Also reports the simulator's implied AUC (using prob_over as the score)
against the actual over-line outcome, for direct comparison with the
already-shipped classifier's validated holdout AUC.

Pre-registered pass (written before this script has ever been run,
identical bar to CFB's):
  1. challenger mean CRPS < baseline mean CRPS
  2. bootstrap P(challenger CRPS < baseline CRPS) >= 0.90
  3. challenger implied AUC >= 0.55

Run
---
python -u nfl_rush_sim_gate_a.py
"""
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import argparse
import sqlite3

import nfl_sim as sim

MODEL_DB_DEFAULT = "nfl_models/nfl_model.sqlite"
CARRY_DB_DEFAULT = "nfl_models/nfl_carry_log.sqlite"

MIN_PRIOR_GAMES_FOR_RATE = 3
MIN_RECENT_CARRIES_PER_GAME = 12
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
    """Strictly-prior-games discipline (season-scoped, matching
    CFB's exact convention): for a row at (player, week), the carry
    pools only ever include games with a strictly earlier week in the
    same season."""
    rows = model_con.execute("""
        SELECT player_id, player_name, team, opponent, week, game_id, carries, rushing_yards
        FROM player_games WHERE position='RB' AND season=? AND season_type='REG'
        ORDER BY player_id, week, game_date
    """, (season,)).fetchall()

    carry_rows = carry_con.execute("""
        SELECT player_id, week, game_id, yards FROM rush_carries WHERE season=?
        ORDER BY player_id, week, game_id, carry_index
    """, (season,)).fetchall()
    by_player_game = {}
    for pid, wk, gid, yards in carry_rows:
        by_player_game.setdefault((pid, wk, gid), []).append(yards)

    out = []
    cur_pid = None
    hist = []
    for pid, pname, team, opp, week, gid, carries, rush_yards in rows:
        if pid != cur_pid:
            cur_pid = pid
            hist = []
        recent = hist[-RECENT_GAMES_WINDOW:]
        n_prior = len(hist)
        c3 = [c for (_, c, _) in hist[-3:]]
        recent_carry_rate = sum(c3) / len(c3) if c3 else 0.0
        if n_prior >= MIN_PRIOR_GAMES_FOR_RATE and recent_carry_rate >= MIN_RECENT_CARRIES_PER_GAME:
            counts = [c for (_, c, _) in recent]
            pool = [y for (_, _, ys) in recent for y in ys]
            if counts and pool:
                out.append({
                    "player_id": pid, "player_name": pname, "week": week,
                    "counts": counts, "pool": pool,
                    "actual": rush_yards if rush_yards is not None else 0,
                    "over_line": 1 if (rush_yards or 0) >= (LINE + 0.5) else 0,
                    "recent3_avg_rush_yards": (sum(sum(ys) for (_, _, ys) in hist[-3:])
                                                / len(hist[-3:])),
                })
        this_game_yards = by_player_game.get((pid, week, gid), [])
        hist.append((week, carries if carries is not None else 0, this_game_yards))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-db", default=MODEL_DB_DEFAULT)
    ap.add_argument("--carry-db", default=CARRY_DB_DEFAULT)
    args = ap.parse_args()

    print("NFL_RUSH_SIM_GATE_A\n====================")
    model_con = sqlite3.connect(f"file:{args.model_db}?mode=ro", uri=True)
    carry_con = sqlite3.connect(f"file:{args.carry_db}?mode=ro", uri=True)

    rows = build_asof_rows(model_con, carry_con, HOLDOUT_SEASON)
    print(f"holdout {HOLDOUT_SEASON}: {len(rows)} eligible RB rows")
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
        baseline_crps[i] = abs(row["actual"] - row["recent3_avg_rush_yards"])
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
