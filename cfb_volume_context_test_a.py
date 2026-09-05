#!/usr/bin/env python3
"""
CFB_VOLUME_CONTEXT_TEST_A

Same test as cfb_rush_sim_opponent_context_a.py (does the simulator's
blind carry-count draw miss a real opponent/game-script effect?),
generalized across the other 3 markets: receiving_yards, passing_yards,
passing_touchdowns. One shared method, run once per market with that
market's own eligibility/opponent-stat/event-pool config -- not because
the answer is assumed to be the same, but because the TEST is identical
and each market gets its own independent, honestly-reported verdict.

Method (identical to the rushing pilot)
----------------------------------------
1. Fit this_game_volume ~ recent3_avg_volume + opp_allowed_per_game +
   projected_margin via OLS on DEV+VAL seasons only (2022-2024) --
   holdout (2025) never touched during fitting.
2. Bootstrap CI on each coefficient -- only treat an effect as real if
   its 95% CI excludes 0.
3. If opp_allowed shows a real effect, scale each bootstrap-drawn event
   count for the CURRENT matchup by predicted_volume / recent3_avg_volume
   (event-outcome pool itself untouched) and re-gate CRPS/bootstrap/AUC
   against the ALREADY-VALIDATED plain simulator for that market on the
   untouched 2025 holdout.
4. If neither effect is real, or the context-adjusted version doesn't
   beat the plain simulator, report that honestly -- no forcing.

Run
---
python -u cfb_volume_context_test_a.py
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
DEV_VAL_SEASONS = (2022, 2023, 2024)
HOLDOUT_SEASON = 2025
RECENT_GAMES_WINDOW = 8
SIMS_PER_ROW = 4000
POWER4 = {"Big Ten", "ACC", "SEC", "Big 12"}
MIN_PRIOR_GAMES_FOR_RATE = 3

MARKETS = {
    "receiving_yards": {
        "position": "WR", "volume_field": "receptions", "outcome_field": "receiving_yards",
        "min_recent_rate": 5, "line": 59.5, "power4_only": True,
        "event_table": "recv_catches", "event_value_col": "yards",
        "plain_auc_reference": 0.6050,
    },
    "passing_yards": {
        "position": "QB", "volume_field": "pass_attempts", "outcome_field": "passing_yards",
        "min_recent_rate": 15, "line": 214.5, "power4_only": True,
        "event_table": "pass_attempts_log", "event_value_col": "yards",
        "plain_auc_reference": 0.6521,
    },
    "passing_touchdowns": {
        "position": "QB", "volume_field": "pass_attempts", "outcome_field": "passing_touchdowns",
        "min_recent_rate": 15, "line": 1.5, "power4_only": False,
        "event_table": "pass_attempts_log", "event_value_col": "is_touchdown",
        "plain_auc_reference": 0.6518,
    },
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


def build_opp_asof(conn, position, outcome_field):
    rows = conn.execute(f"""
        SELECT player_id, opponent, season, week, {outcome_field}
        FROM player_games WHERE position=?
        ORDER BY season, week
    """, (position,)).fetchall()
    by_season_week = {}
    for r in rows:
        by_season_week.setdefault((r[2], r[3]), []).append(r)
    opp_state = {}
    opp_asof = {}
    for (season, week) in sorted(by_season_week):
        wk_rows = by_season_week[(season, week)]
        for pid, opp, s, w, val in wk_rows:
            st = opp_state.get((s, opp))
            opp_asof[(pid, s, w)] = (st[0] / st[1]) if st and st[1] > 0 else None
        for pid, opp, s, w, val in wk_rows:
            val = val if val is not None else 0
            st = opp_state.setdefault((s, opp), [0, 0])
            st[0] += val
            st[1] += 1
    return opp_asof


def power4_game_ids(model_con):
    rows = model_con.execute(
        "SELECT game_id FROM games WHERE home_conference IN ({0}) AND away_conference IN ({0})"
        .format(",".join("?" for _ in POWER4)), tuple(POWER4) * 2).fetchall()
    return {r[0] for r in rows}


def build_rows(model_con, carry_con, seasons, cfg):
    margin_asof = build_team_margin_asof(model_con)
    opp_asof = build_opp_asof(model_con, cfg["position"], cfg["outcome_field"])
    p4_games = power4_game_ids(model_con) if cfg["power4_only"] else None

    value_col = cfg["event_value_col"]
    placeholders = ",".join("?" for _ in seasons)
    event_rows = carry_con.execute(f"""
        SELECT player_id, week, game_id, {value_col}, season FROM {cfg['event_table']}
        WHERE season IN ({placeholders})
        ORDER BY player_id, week, game_id
    """, seasons).fetchall()
    by_player_game = {}
    for pid, wk, gid, val, season in event_rows:
        by_player_game.setdefault((pid, season, wk, gid), []).append(val)

    rows = model_con.execute(f"""
        SELECT player_id, player_name, team, opponent, season, week, game_id,
               {cfg['volume_field']}, {cfg['outcome_field']}
        FROM player_games WHERE position=? AND season IN ({placeholders})
        ORDER BY player_id, season, week, game_date
    """, (cfg["position"], *seasons)).fetchall()

    out = []
    cur_key = None
    hist = []
    for pid, pname, team, opp, season, week, gid, volume, outcome in rows:
        key = (pid, season)
        if key != cur_key:
            cur_key = key
            hist = []
        eligible_game = (p4_games is None) or (gid in p4_games)
        if eligible_game:
            recent = hist[-RECENT_GAMES_WINDOW:]
            n_prior = len(hist)
            c3 = [c for (_, c, _) in hist[-3:]]
            recent_rate = sum(c3) / len(c3) if c3 else 0.0
            if n_prior >= MIN_PRIOR_GAMES_FOR_RATE and recent_rate >= cfg["min_recent_rate"]:
                counts = [c for (_, c, _) in recent]
                pool = [v for (_, _, vs) in recent for v in vs]
                team_margin = margin_asof.get((season, team, week))
                opp_margin = margin_asof.get((season, opp, week))
                proj_margin = (team_margin - opp_margin) if (team_margin is not None and opp_margin is not None) else None
                opp_allowed = opp_asof.get((pid, season, week))
                if counts and pool and proj_margin is not None and opp_allowed is not None:
                    out.append({
                        "player_id": pid, "player_name": pname, "season": season, "week": week,
                        "counts": counts, "pool": pool,
                        "recent3_avg_volume": recent_rate,
                        "opp_allowed": opp_allowed,
                        "projected_margin": proj_margin,
                        "actual_volume": volume if volume is not None else 0,
                        "actual": outcome if outcome is not None else 0,
                        "over_line": 1 if (outcome or 0) >= (cfg["line"] + 0.5) else 0,
                    })
        this_game_vals = by_player_game.get((pid, season, week, gid), [])
        hist.append((week, volume if volume is not None else 0, this_game_vals))
    return out


def fit_ols(rows):
    X = np.array([[1.0, r["recent3_avg_volume"], r["opp_allowed"], r["projected_margin"]]
                  for r in rows])
    y = np.array([r["actual_volume"] for r in rows], dtype=np.float64)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def bootstrap_coef_ci(rows, n_boot=2000, rng=None):
    if rng is None:
        rng = np.random.RandomState(20250901)
    n = len(rows)
    boots = np.empty((n_boot, 4))
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots[i] = fit_ols([rows[j] for j in idx])
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return lo, hi


def run_market(market_name, cfg, model_con, carry_con):
    print(f"\n=== {market_name} ===")
    dev_val_rows = build_rows(model_con, carry_con, DEV_VAL_SEASONS, cfg)
    print(f"DEV+VAL {DEV_VAL_SEASONS}: {len(dev_val_rows)} eligible rows for fitting")
    if len(dev_val_rows) < 50:
        print("Too few rows to fit reliably -- skipping.")
        return

    coef = fit_ols(dev_val_rows)
    lo, hi = bootstrap_coef_ci(dev_val_rows)
    names = ["intercept", "recent3_avg_volume", "opp_allowed", "projected_margin"]
    for name, c, l, h in zip(names, coef, lo, hi):
        real = not (l <= 0 <= h)
        print(f"  {name:24s} coef={c:+.4f}  95% CI=[{l:+.4f}, {h:+.4f}]  "
              f"{'REAL EFFECT' if real else 'not significant (CI crosses 0)'}")

    ci_opp_real = not (lo[2] <= 0 <= hi[2])
    ci_margin_real = not (lo[3] <= 0 <= hi[3])
    if not (ci_opp_real or ci_margin_real):
        print("Neither opponent context nor game script shows a real effect on volume "
              "beyond the player's own recent rate -- honest negative result, nothing to gate.")
        return

    holdout_rows = build_rows(model_con, carry_con, (HOLDOUT_SEASON,), cfg)
    print(f"holdout {HOLDOUT_SEASON}: {len(holdout_rows)} eligible rows")
    if not holdout_rows:
        print("No eligible holdout rows -- cannot gate.")
        return

    rng = np.random.RandomState(20250901)
    plain_crps = np.empty(len(holdout_rows))
    context_crps = np.empty(len(holdout_rows))
    context_prob_over = np.empty(len(holdout_rows))
    over_line = np.empty(len(holdout_rows))

    for i, row in enumerate(holdout_rows):
        plain_result = sim.simulate(row["counts"], row["pool"], cfg["line"], sims=SIMS_PER_ROW, rng=rng)
        plain_crps[i] = crps_from_samples(plain_result["samples"], row["actual"])

        predicted_volume = (coef[0] + coef[1] * row["recent3_avg_volume"]
                             + coef[2] * row["opp_allowed"] + coef[3] * row["projected_margin"])
        ratio = predicted_volume / row["recent3_avg_volume"] if row["recent3_avg_volume"] > 0 else 1.0
        ratio = max(0.4, min(2.0, ratio))
        adjusted_counts = [max(0, int(round(c * ratio))) for c in row["counts"]]

        context_result = sim.simulate(adjusted_counts, row["pool"], cfg["line"], sims=SIMS_PER_ROW, rng=rng)
        context_crps[i] = crps_from_samples(context_result["samples"], row["actual"])
        context_prob_over[i] = context_result["prob_over"]
        over_line[i] = row["over_line"]

    mean_plain = float(np.mean(plain_crps))
    mean_context = float(np.mean(context_crps))
    print(f"mean CRPS -- plain simulator (shipped): {mean_plain:.3f}")
    print(f"mean CRPS -- context-conditioned:       {mean_context:.3f}")

    diffs = plain_crps - context_crps
    n_boot = 5000
    n = len(holdout_rows)
    boot_better = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if np.mean(diffs[idx]) > 0:
            boot_better += 1
    p_context_better = boot_better / n_boot
    print(f"bootstrap P(context-conditioned better): {p_context_better:.3f}")

    context_auc = auc(context_prob_over, over_line)
    ref_auc = cfg["plain_auc_reference"]
    print(f"context-conditioned implied AUC: {context_auc:.4f} (plain simulator: {ref_auc:.4f})")

    pass_crps = mean_context < mean_plain
    pass_boot = p_context_better >= 0.90
    pass_auc = context_auc >= (ref_auc - 0.02)

    print(f"gate: CRPS improves={pass_crps}  bootstrap>=0.90={pass_boot}  AUC ok={pass_auc}")
    overall = pass_crps and pass_boot and pass_auc
    print(f"VERDICT for {market_name}: {'PASS' if overall else 'FAIL'}")


def main():
    print("CFB_VOLUME_CONTEXT_TEST_A\n==========================")
    model_con = sqlite3.connect(f"file:{MODEL_DB}?mode=ro", uri=True)
    carry_con = sqlite3.connect(f"file:{CARRY_DB}?mode=ro", uri=True)
    for market_name, cfg in MARKETS.items():
        run_market(market_name, cfg, model_con, carry_con)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
