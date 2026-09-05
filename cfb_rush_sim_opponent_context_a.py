#!/usr/bin/env python3
"""
CFB_RUSH_SIM_OPPONENT_CONTEXT_A

Tests a real gap in cfb_rush_sim.py: it draws a simulated game's carry
COUNT uniformly from the player's own recent real games, blind to who
that game was against. Unlike MLB (where a starting pitcher's batters-
faced is mostly about his own pitch count -- opponent-lineup context was
tested for pitcher_walks and found no proven improvement, see a154e51/
a06165e), a CFB running back's carry volume plausibly depends a lot on
(a) how good the opponent's run defense is and (b) game script (a big
lead means more carries late; a big deficit means the run game gets
abandoned) -- both already used as real, validated features in the
shipped classifier (opp_rush_yards_allowed_per_game, projected_margin).
This is a testable hypothesis, not an assumption either way.

Method
------
1. Fit carries_this_game ~ recent3_avg_carries + opp_rush_yards_allowed_
   per_game + projected_margin via plain OLS on DEV+VAL seasons only
   (2022-2024) -- the 2025 holdout is never touched during fitting, same
   discipline as every other model in this repo.
2. If the fit shows a real, non-trivial opponent/game-script effect
   (checked via bootstrap CI on the DEV+VAL fit, not just a raw
   coefficient), use it to scale each bootstrap-drawn carry count for
   the CURRENT matchup: multiply by predicted_carries_this_matchup /
   recent3_avg_carries. The per-carry yardage pool is untouched --
   only the volume draw is context-adjusted.
3. Re-gate on the SAME untouched 2025 holdout used for the original
   simulator, but this time the baseline arm is the ALREADY-VALIDATED
   plain simulator (cfb_rush_sim.py) -- a stricter bar than beating a
   flat average, since the plain simulator already cleared that one.

Pre-registered pass (written before this script has ever been run):
  1. opponent-conditioned mean CRPS < plain-simulator mean CRPS
  2. bootstrap P(context-conditioned better) >= 0.90
  3. context-conditioned implied AUC >= plain simulator's AUC (0.6101)
     minus 0.02 (a small AUC dip is acceptable if CRPS clearly improves,
     since CRPS is the primary target here -- shape/spread, not ranking)

Run
---
python -u cfb_rush_sim_opponent_context_a.py
"""
import sqlite3
import sys
from collections import deque

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
MIN_RECENT_CARRIES_PER_GAME = 12
LINE = 69.5
DEV_VAL_SEASONS = (2022, 2023, 2024)
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


def build_team_margin_asof(conn):
    games = conn.execute(
        "SELECT game_id, season, week, home_team, away_team, home_points, away_points "
        "FROM games ORDER BY season, week").fetchall()
    by_season_week = {}
    for g in games:
        by_season_week.setdefault((g[1], g[2]), []).append(g)
    team_state = {}
    margin_asof = {}
    for (season, week) in sorted(by_season_week):
        for g in by_season_week[(season, week)]:
            _, _, _, home, away, hp, ap = g
            for team in (home, away):
                st = team_state.get((season, team), [0, 0, 0])
                margin_asof[(season, team, week)] = (st[0] - st[1]) / st[2] if st[2] > 0 else None
        for g in by_season_week[(season, week)]:
            _, _, _, home, away, hp, ap = g
            hp = hp if hp is not None else 0
            ap = ap if ap is not None else 0
            hst = team_state.setdefault((season, home), [0, 0, 0])
            hst[0] += hp; hst[1] += ap; hst[2] += 1
            ast = team_state.setdefault((season, away), [0, 0, 0])
            ast[0] += ap; ast[1] += hp; ast[2] += 1
    return margin_asof


def build_opp_rush_asof(conn):
    """As-of (player_id, season, week) -> that game's OPPONENT's season-
    to-date rushing yards allowed per game. Identical logic to the
    shipped classifier's feature (cfb_rushing_yards_clean_baseline_b.py)."""
    rows = conn.execute("""
        SELECT player_id, opponent, season, week, rushing_yards
        FROM player_games WHERE position='RB'
        ORDER BY season, week
    """).fetchall()
    by_season_week = {}
    for r in rows:
        by_season_week.setdefault((r[2], r[3]), []).append(r)
    opp_state = {}
    opp_asof = {}
    for (season, week) in sorted(by_season_week):
        wk_rows = by_season_week[(season, week)]
        for pid, opp, s, w, ry in wk_rows:
            st = opp_state.get((s, opp))
            opp_asof[(pid, s, w)] = (st[0] / st[1]) if st and st[1] > 0 else None
        for pid, opp, s, w, ry in wk_rows:
            ry = ry if ry is not None else 0
            st = opp_state.setdefault((s, opp), [0, 0])
            st[0] += ry
            st[1] += 1
    return opp_asof


def build_rows(model_con, carry_con, seasons):
    margin_asof = build_team_margin_asof(model_con)
    opp_asof = build_opp_rush_asof(model_con)

    carry_rows = carry_con.execute(
        "SELECT player_id, week, game_id, yards, season FROM rush_carries "
        "WHERE season IN ({0}) ORDER BY player_id, week, game_id, carry_index"
        .format(",".join("?" for _ in seasons)), seasons).fetchall()
    by_player_game = {}
    for pid, wk, gid, yards, season in carry_rows:
        by_player_game.setdefault((pid, season, wk, gid), []).append(yards)

    placeholders = ",".join("?" for _ in seasons)
    rows = model_con.execute(f"""
        SELECT player_id, player_name, team, opponent, season, week, game_id, carries, rushing_yards
        FROM player_games WHERE position='RB' AND season IN ({placeholders})
        ORDER BY player_id, season, week, game_date
    """, seasons).fetchall()

    out = []
    cur_key = None
    hist = []
    for pid, pname, team, opp, season, week, gid, carries, rush_yards in rows:
        key = (pid, season)
        if key != cur_key:
            cur_key = key
            hist = []
        recent = hist[-RECENT_GAMES_WINDOW:]
        n_prior = len(hist)
        c3 = [c for (_, c, _) in hist[-3:]]
        recent_carry_rate = sum(c3) / len(c3) if c3 else 0.0
        if n_prior >= MIN_PRIOR_GAMES_FOR_RATE and recent_carry_rate >= MIN_RECENT_CARRIES_PER_GAME:
            counts = [c for (_, c, _) in recent]
            pool = [y for (_, _, ys) in recent for y in ys]
            team_margin = margin_asof.get((season, team, week))
            opp_margin = margin_asof.get((season, opp, week))
            proj_margin = (team_margin - opp_margin) if (team_margin is not None and opp_margin is not None) else None
            opp_rush_allowed = opp_asof.get((pid, season, week))
            if counts and pool and proj_margin is not None and opp_rush_allowed is not None:
                out.append({
                    "player_id": pid, "player_name": pname, "season": season, "week": week,
                    "counts": counts, "pool": pool,
                    "recent3_avg_carries": recent_carry_rate,
                    "opp_rush_allowed": opp_rush_allowed,
                    "projected_margin": proj_margin,
                    "actual_carries": carries if carries is not None else 0,
                    "actual": rush_yards if rush_yards is not None else 0,
                    "over_line": 1 if (rush_yards or 0) >= (LINE + 0.5) else 0,
                })
        this_game_yards = by_player_game.get((pid, season, week, gid), [])
        hist.append((week, carries if carries is not None else 0, this_game_yards))
    return out


def fit_ols(rows):
    """carries_this_game ~ recent3_avg_carries + opp_rush_allowed + projected_margin"""
    X = np.array([[1.0, r["recent3_avg_carries"], r["opp_rush_allowed"], r["projected_margin"]]
                  for r in rows])
    y = np.array([r["actual_carries"] for r in rows], dtype=np.float64)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef  # [intercept, b_recent3, b_opp, b_margin]


def bootstrap_coef_ci(rows, n_boot=2000, rng=None):
    if rng is None:
        rng = np.random.RandomState(20250901)
    n = len(rows)
    boots = np.empty((n_boot, 4))
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        sample = [rows[j] for j in idx]
        boots[i] = fit_ols(sample)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return lo, hi


def main():
    print("CFB_RUSH_SIM_OPPONENT_CONTEXT_A\n================================")
    model_con = sqlite3.connect(f"file:{MODEL_DB}?mode=ro", uri=True)
    carry_con = sqlite3.connect(f"file:{CARRY_DB}?mode=ro", uri=True)

    dev_val_rows = build_rows(model_con, carry_con, DEV_VAL_SEASONS)
    print(f"DEV+VAL {DEV_VAL_SEASONS}: {len(dev_val_rows)} eligible RB rows for fitting")

    coef = fit_ols(dev_val_rows)
    lo, hi = bootstrap_coef_ci(dev_val_rows)
    names = ["intercept", "recent3_avg_carries", "opp_rush_allowed", "projected_margin"]
    print("\nOLS fit (carries_this_game ~ ...), DEV+VAL only:")
    for name, c, l, h in zip(names, coef, lo, hi):
        sig = "significant" if (l > 0) != (h < 0) and not (l <= 0 <= h) else "NOT significant (CI crosses 0)"
        print(f"  {name:24s} coef={c:+.4f}  95% CI=[{l:+.4f}, {h:+.4f}]  {sig}")

    b_opp = coef[2]
    b_margin = coef[3]
    ci_opp_excludes_0 = not (lo[2] <= 0 <= hi[2])
    ci_margin_excludes_0 = not (lo[3] <= 0 <= hi[3])
    print(f"\nopp_rush_allowed effect real (CI excludes 0): {ci_opp_excludes_0}")
    print(f"projected_margin effect real (CI excludes 0): {ci_margin_excludes_0}")

    if not (ci_opp_excludes_0 or ci_margin_excludes_0):
        print("\nNeither opponent context nor game script shows a statistically real "
              "effect on carry volume beyond the player's own recent rate -- "
              "stopping here, nothing to gate. Honest negative result.")
        return 0

    holdout_rows = build_rows(model_con, carry_con, (HOLDOUT_SEASON,))
    print(f"\nholdout {HOLDOUT_SEASON}: {len(holdout_rows)} eligible RB rows")

    rng = np.random.RandomState(20250901)
    plain_crps = np.empty(len(holdout_rows))
    context_crps = np.empty(len(holdout_rows))
    context_prob_over = np.empty(len(holdout_rows))
    over_line = np.empty(len(holdout_rows))

    for i, row in enumerate(holdout_rows):
        plain_result = sim.simulate(row["counts"], row["pool"], LINE, sims=SIMS_PER_ROW, rng=rng)
        plain_crps[i] = crps_from_samples(plain_result["samples"], row["actual"])

        predicted_carries = (coef[0] + coef[1] * row["recent3_avg_carries"]
                              + coef[2] * row["opp_rush_allowed"] + coef[3] * row["projected_margin"])
        ratio = predicted_carries / row["recent3_avg_carries"] if row["recent3_avg_carries"] > 0 else 1.0
        ratio = max(0.4, min(2.0, ratio))  # sane bounds, no runaway extrapolation
        adjusted_counts = [max(0, int(round(c * ratio))) for c in row["counts"]]

        context_result = sim.simulate(adjusted_counts, row["pool"], LINE, sims=SIMS_PER_ROW, rng=rng)
        context_crps[i] = crps_from_samples(context_result["samples"], row["actual"])
        context_prob_over[i] = context_result["prob_over"]
        over_line[i] = row["over_line"]

    mean_plain = float(np.mean(plain_crps))
    mean_context = float(np.mean(context_crps))
    print(f"\nmean CRPS -- plain simulator (already shipped): {mean_plain:.3f}")
    print(f"mean CRPS -- opponent/game-script-conditioned:   {mean_context:.3f}")

    diffs = plain_crps - context_crps
    n_boot = 5000
    n = len(holdout_rows)
    boot_better = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if np.mean(diffs[idx]) > 0:
            boot_better += 1
    p_context_better = boot_better / n_boot
    print(f"bootstrap P(context-conditioned CRPS < plain CRPS): {p_context_better:.3f}")

    context_auc = auc(context_prob_over, over_line)
    print(f"context-conditioned implied AUC: {context_auc:.4f} (plain simulator: 0.6101)")

    pass_crps = mean_context < mean_plain
    pass_boot = p_context_better >= 0.90
    pass_auc = context_auc >= (0.6101 - 0.02)

    print("\n--- Pre-registered gate (vs. the already-shipped plain simulator) ---")
    print(f"1. context CRPS < plain CRPS: {'PASS' if pass_crps else 'FAIL'}")
    print(f"2. bootstrap P(context better) >= 0.90: {'PASS' if pass_boot else 'FAIL'}")
    print(f"3. context AUC >= 0.5901: {'PASS' if pass_auc else 'FAIL'}")

    overall = pass_crps and pass_boot and pass_auc
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
