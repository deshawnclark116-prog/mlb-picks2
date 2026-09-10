#!/usr/bin/env python3
"""
NFL_PRIOR_SEASON_SIM_GATE_A

The rush/recv Monte Carlo simulators (nfl_rush_sim_gate_a.py /
nfl_recv_sim_gate_a.py, both PASSED) need a player's own recent REAL
games to build their two-stage bootstrap pools -- which weeks 1-3 don't
have yet for the CURRENT season, same gap nfl_prior_season_early_gate_a.py
found for the classifier. This tests the same fix, applied to the
simulator instead: use the player's LAST real season's per-play carry/
target pools (nfl_pbp_foundation_a.py) as the bootstrap source for
weeks 1-3 of the NEXT season, exactly the real, immediate-impact case
this repo's own board needs right now (week 1, 2026) -- there is no
"current-season recent games" yet, only last season's.

Real, direct advantage over the classifier's own prior-season arm: the
classifier had to match players via ESPN's live roster + normalized-name
join (no shared id with nflverse). The simulator's pools are ALREADY in
nflverse's own gsis_id namespace (same as the weekly stats that define
this test's population), so this is a plain id join -- no roster fetch,
no name-matching risk.

Population: current season weeks 1-3, RB (rushing_yards) / WR
(receiving_yards), no current-season game requirement (matches
nfl_prior_season_early_gate_a.py's population exactly). Split: DEV =
weeks 1-3 of 2023+2024 (priors 2022+2023), HOLDOUT = weeks 1-3 of 2025
(prior 2024) -- but this session's carry-log ingestion only covers
2022-2024 real PBP, so this run instead reports on HOLDOUT = weeks 1-3
of 2024 (prior 2023), the same holdout year nfl_rush_sim_gate_a.py /
nfl_recv_sim_gate_a.py already use, disclosed here rather than silently
using a period nothing else in this gate touches for the SAME market
(the in-season sim's own dev/holdout).

Pre-registered pass (identical bar to the other two sim gates):
  1. challenger mean CRPS < baseline mean CRPS (point-mass at the
     player's own prior-season per-game average)
  2. bootstrap P(challenger CRPS < baseline CRPS) >= 0.90
  3. challenger implied AUC >= 0.55

Run
---
python -u nfl_prior_season_sim_gate_a.py
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

MAX_WEEK = 3
LINE = 49.5
HOLDOUT_SEASON = 2024
SIMS_PER_ROW = 4000

MARKETS = {
    "rushing_yards": {"position": "RB", "stat": "rushing_yards", "event_table": "rush_carries",
                       "event_idx": "carry_index", "game_stat_col": "carries"},
    "receiving_yards": {"position": "WR", "stat": "receiving_yards", "event_table": "recv_targets",
                         "event_idx": "target_index", "game_stat_col": "targets"},
}


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


def load_prior_season_pools(carry_con, season, cfg):
    """player_id -> (game_counts:list[int], event_pool:list[float]) built
    from the WHOLE prior real season (every week)."""
    rows = carry_con.execute(f"""
        SELECT player_id, week, game_id, yards FROM {cfg['event_table']} WHERE season=?
        ORDER BY player_id, week, game_id, {cfg['event_idx']}
    """, (season,)).fetchall()
    by_player_game = {}
    for pid, wk, gid, yards in rows:
        by_player_game.setdefault(pid, {}).setdefault((wk, gid), []).append(yards)

    out = {}
    for pid, games in by_player_game.items():
        counts = [len(v) for v in games.values()]
        pool = [y for v in games.values() for y in v]
        out[pid] = (counts, pool)
    return out


def build_rows(model_con, prior_pools, season, cfg):
    rows = model_con.execute(f"""
        SELECT player_id, player_name, week, {cfg['stat']}
        FROM player_games WHERE position=? AND season=? AND week<=? AND season_type='REG'
        ORDER BY player_id, week
    """, (cfg["position"], season, MAX_WEEK)).fetchall()

    out = []
    for pid, pname, week, actual in rows:
        pools = prior_pools.get(pid)
        if not pools or not pools[0] or not pools[1]:
            continue
        counts, pool = pools
        actual = actual if actual is not None else 0
        # naive baseline: player's own prior-season mean per-event yards
        # times their own prior-season mean event count per game.
        naive_avg = (sum(pool) / len(pool)) * (sum(counts) / len(counts))
        out.append({
            "player_id": pid, "player_name": pname, "week": week,
            "counts": counts, "pool": pool, "actual": actual,
            "over_line": 1 if actual >= (LINE + 0.5) else 0,
            "naive_avg": naive_avg,
        })
    return out


def run_market(mkt_key, cfg, model_con, carry_con):
    print(f"\n{'='*70}\n{mkt_key} (weeks 1-{MAX_WEEK} only, prior-season pools)\n{'='*70}")
    prior_season = HOLDOUT_SEASON - 1
    prior_pools = load_prior_season_pools(carry_con, prior_season, cfg)
    print(f"  prior season {prior_season}: {len(prior_pools)} players with real per-event history")

    rows = build_rows(model_con, prior_pools, HOLDOUT_SEASON, cfg)
    matched = len(rows)
    print(f"  holdout {HOLDOUT_SEASON} weeks 1-{MAX_WEEK}: {matched} rows matched to prior-season pools")
    if not rows:
        return {"passed": False, "reason": "no matched rows"}

    rng = np.random.RandomState(20260901)
    challenger_crps = np.empty(len(rows))
    baseline_crps = np.empty(len(rows))
    prob_over = np.empty(len(rows))
    over_line = np.empty(len(rows))

    for i, row in enumerate(rows):
        result = sim.simulate(row["counts"], row["pool"], LINE, sims=SIMS_PER_ROW, rng=rng)
        challenger_crps[i] = crps_from_samples(result["samples"], row["actual"])
        baseline_crps[i] = abs(row["actual"] - row["naive_avg"])
        prob_over[i] = result["prob_over"]
        over_line[i] = row["over_line"]

    mean_challenger = float(np.mean(challenger_crps))
    mean_baseline = float(np.mean(baseline_crps))
    print(f"  mean CRPS -- challenger (simulator): {mean_challenger:.3f}")
    print(f"  mean CRPS -- baseline (point-mass at prior-season avg): {mean_baseline:.3f}")

    diffs = baseline_crps - challenger_crps
    n_boot = 5000
    n = len(rows)
    boot_better = sum(1 for _ in range(n_boot) if np.mean(diffs[rng.randint(0, n, size=n)]) > 0)
    p_challenger_better = boot_better / n_boot
    print(f"  bootstrap P(challenger CRPS < baseline CRPS): {p_challenger_better:.3f}")

    challenger_auc = auc(prob_over, over_line)
    print(f"  challenger implied AUC: {challenger_auc:.4f}")

    pass_crps = mean_challenger < mean_baseline
    pass_boot = p_challenger_better >= 0.90
    pass_auc = challenger_auc >= 0.55
    passed = pass_crps and pass_boot and pass_auc
    print(f"  GATE: CRPS better -> {pass_crps}  bootstrap>=0.90 -> {pass_boot}  AUC>=0.55 -> {pass_auc}")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    return {"n": matched, "mean_challenger_crps": round(mean_challenger, 3),
            "mean_baseline_crps": round(mean_baseline, 3),
            "p_challenger_better": round(p_challenger_better, 4),
            "implied_auc": round(challenger_auc, 4), "passed": passed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-db", default=MODEL_DB_DEFAULT)
    ap.add_argument("--carry-db", default=CARRY_DB_DEFAULT)
    args = ap.parse_args()

    print("NFL_PRIOR_SEASON_SIM_GATE_A\n===========================")
    model_con = sqlite3.connect(f"file:{args.model_db}?mode=ro", uri=True)
    carry_con = sqlite3.connect(f"file:{args.carry_db}?mode=ro", uri=True)

    results = {}
    for mkt_key, cfg in MARKETS.items():
        results[mkt_key] = run_market(mkt_key, cfg, model_con, carry_con)

    print(f"\n\n{'#'*70}\nSUMMARY\n{'#'*70}")
    for mkt, r in results.items():
        print(f"  {mkt:18s} {r}")

    any_pass = any(r.get("passed") for r in results.values())
    return 0 if any_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
