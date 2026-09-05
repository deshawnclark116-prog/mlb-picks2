#!/usr/bin/env python3
"""
CFB_QB_DLINE_PRESSURE_TEST_A

Tests a specific, more targeted version of the opponent-context question
than cfb_volume_context_test_a.py's passing_yards/passing_touchdowns
result: "opp_rush/pass_yards_allowed" is a blended team-defense stat that
never isolates the pass rush specifically. Sacks are a much more direct
signal of the opposing D-LINE's pressure -- and they're not in
player_games or pass_attempts_log at all right now (sacks aren't counted
as pass attempts in the official stat convention this pipeline already
matches, verified earlier against Dillon Gabriel's real stat line).

Extracts sacks_conceded_by_team_per_game directly from the raw play-level
CSV (sack_taken_player_id present, "team" column = the sacked/offense
side -- confirmed directly against real rows, e.g. Alabama's TJ Finley/
Jalen Milroe both sacked in game 401628319), builds an as-of "opponent's
defense sacks recorded per game" feature (that opponent's OWN pass-rush
production, looking up the games they played on defense), and tests it
two separate ways against DEV+VAL (2022-2024), holdout untouched:

  A. Does it move ATTEMPT VOLUME (does more pressure mean fewer/more
     drop-backs)?
  B. Does it move PER-ATTEMPT EFFICIENCY -- yards per attempt, and TD
     rate per attempt (does more pressure mean he gains less per throw,
     separate from how many times he throws)?

B is the mechanism "it's about the dline" most plausibly points at --
untested so far, since every previous test in this repo only ever
adjusted the simulated COUNT, never the per-event outcome POOL itself.

Run
---
python -u cfb_qb_dline_pressure_test_a.py
"""
import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

MODEL_DB = "cfb_models/cfb_model.sqlite"
RAW_DIR = Path("/data/cfb_raw")
SEASONS = (2022, 2023, 2024, 2025)
DEV_VAL_SEASONS = (2022, 2023, 2024)
HOLDOUT_SEASON = 2025
MIN_PRIOR_GAMES_FOR_RATE = 3
MIN_RECENT_ATTEMPTS_PER_GAME = 15


def to_int(v):
    try:
        return int(round(float(v)))
    except Exception:
        return None


def load_games_filter(schedule_path):
    games = {}
    with open(schedule_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("home_division") or "").lower() != "fbs":
                continue
            if (r.get("away_division") or "").lower() != "fbs":
                continue
            if r.get("season_type") != "regular":
                continue
            gid = r.get("game_id")
            week = to_int(r.get("week"))
            if not gid or week is None:
                continue
            games[gid] = week
    return games


def extract_sacks_conceded(season):
    """(game_id, offense_team) -> sacks conceded count, for one season."""
    sched_path = RAW_DIR / f"schedules_{season}.csv"
    ps_path = RAW_DIR / f"player_stats_{season}.csv"
    games = load_games_filter(sched_path)
    conceded = {}
    with open(ps_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gid = r.get("game_id")
            if gid not in games:
                continue
            stp = r.get("sack_taken_player_id")
            if stp and stp != "NA":
                team = r.get("team")
                key = (gid, team)
                conceded[key] = conceded.get(key, 0) + 1
    return conceded, games


def build_sack_asof(model_con, conceded_by_season):
    """As-of (team, season, week) -> that team's OWN defense's season-to-
    date sacks recorded per game (i.e. their pass-rush production, looked
    up by an opposing QB via player_games.opponent)."""
    game_rows = model_con.execute(
        "SELECT game_id, season, week, home_team, away_team FROM games ORDER BY season, week"
    ).fetchall()
    by_season_week = {}
    for gid, season, week, home, away in game_rows:
        by_season_week.setdefault((season, week), []).append((gid, home, away))

    def_state = {}   # (season, team) -> [sack_sum, n_games]
    def_asof = {}    # (team, season, week) -> sacks/game as of before that week

    for (season, week) in sorted(by_season_week):
        conceded = conceded_by_season.get(season, {})
        for gid, home, away in by_season_week[(season, week)]:
            for team in (home, away):
                st = def_state.get((season, team))
                def_asof[(team, season, week)] = (st[0] / st[1]) if st and st[1] > 0 else None
        for gid, home, away in by_season_week[(season, week)]:
            # home's defense sacked away's QB; away's defense sacked home's QB
            home_sacks_recorded = conceded.get((gid, away), 0)
            away_sacks_recorded = conceded.get((gid, home), 0)
            hst = def_state.setdefault((season, home), [0, 0])
            hst[0] += home_sacks_recorded; hst[1] += 1
            ast = def_state.setdefault((season, away), [0, 0])
            ast[0] += away_sacks_recorded; ast[1] += 1
    return def_asof


def fit_ols(X_rows, y_vals):
    X = np.array(X_rows)
    y = np.array(y_vals, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def bootstrap_ci(X_rows, y_vals, n_boot=2000, rng=None):
    if rng is None:
        rng = np.random.RandomState(20250901)
    n = len(X_rows)
    k = len(X_rows[0])
    boots = np.empty((n_boot, k))
    X_rows = np.array(X_rows)
    y_vals = np.array(y_vals, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        coef, *_ = np.linalg.lstsq(X_rows[idx], y_vals[idx], rcond=None)
        boots[i] = coef
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return lo, hi


def report_fit(name, names, coef, lo, hi):
    print(f"\n{name}")
    for nm, c, l, h in zip(names, coef, lo, hi):
        real = not (l <= 0 <= h)
        print(f"  {nm:24s} coef={c:+.5f}  95% CI=[{l:+.5f}, {h:+.5f}]  "
              f"{'REAL EFFECT' if real else 'not significant'}")


def main():
    print("CFB_QB_DLINE_PRESSURE_TEST_A\n=============================")
    model_con = sqlite3.connect(f"file:{MODEL_DB}?mode=ro", uri=True)

    conceded_by_season = {}
    for season in SEASONS:
        conceded, _ = extract_sacks_conceded(season)
        conceded_by_season[season] = conceded
        print(f"season {season}: {sum(conceded.values())} sacks conceded across "
              f"{len(conceded)} (game,team) rows")

    sack_asof = build_sack_asof(model_con, conceded_by_season)

    rows = model_con.execute("""
        SELECT player_id, player_name, team, opponent, season, week, game_id,
               pass_attempts, passing_yards, passing_touchdowns
        FROM player_games WHERE position='QB' AND season IN (2022,2023,2024,2025)
        ORDER BY player_id, season, week, game_date
    """).fetchall()

    out = []
    cur_key = None
    hist = []  # (week, attempts, yards, td)
    for pid, pname, team, opp, season, week, gid, attempts, yards, td in rows:
        key = (pid, season)
        if key != cur_key:
            cur_key = key
            hist = []
        n_prior = len(hist)
        a3 = [a for (_, a, _, _) in hist[-3:]]
        recent_rate = sum(a3) / len(a3) if a3 else 0.0
        if n_prior >= MIN_PRIOR_GAMES_FOR_RATE and recent_rate >= MIN_RECENT_ATTEMPTS_PER_GAME:
            opp_sacks = sack_asof.get((opp, season, week))
            if opp_sacks is not None and attempts and attempts > 0:
                y3 = [y for (_, a, y, _) in hist[-3:] if a]
                a3_full = [a for (_, a, _, _) in hist[-3:]]
                recent3_ypa = (sum(y3) / sum(a3_full)) if a3_full and sum(a3_full) > 0 else 0.0
                td3 = [t for (_, a, _, t) in hist[-3:]]
                recent3_td_rate = (sum(td3) / sum(a3_full)) if a3_full and sum(a3_full) > 0 else 0.0
                out.append({
                    "season": season, "week": week,
                    "recent3_avg_attempts": recent_rate,
                    "recent3_ypa": recent3_ypa,
                    "recent3_td_rate": recent3_td_rate,
                    "opp_sacks_per_game": opp_sacks,
                    "actual_attempts": attempts,
                    "actual_ypa": (yards / attempts) if yards is not None else 0.0,
                    "actual_td_rate": ((td or 0) / attempts),
                })
        hist.append((week, attempts if attempts is not None else 0,
                      yards if yards is not None else 0, td if td is not None else 0))

    dev_val = [r for r in out if r["season"] in DEV_VAL_SEASONS]
    print(f"\nDEV+VAL rows: {len(dev_val)}")

    # Model A: attempt volume ~ recent3_avg_attempts + opp_sacks_per_game
    Xa = [[1.0, r["recent3_avg_attempts"], r["opp_sacks_per_game"]] for r in dev_val]
    ya = [r["actual_attempts"] for r in dev_val]
    coef_a = fit_ols(Xa, ya)
    lo_a, hi_a = bootstrap_ci(Xa, ya)
    report_fit("Model A: pass attempts ~ recent3_avg_attempts + opp_sacks_per_game",
               ["intercept", "recent3_avg_attempts", "opp_sacks_per_game"], coef_a, lo_a, hi_a)

    # Model B: yards per attempt ~ recent3_ypa + opp_sacks_per_game
    Xb = [[1.0, r["recent3_ypa"], r["opp_sacks_per_game"]] for r in dev_val]
    yb = [r["actual_ypa"] for r in dev_val]
    coef_b = fit_ols(Xb, yb)
    lo_b, hi_b = bootstrap_ci(Xb, yb)
    report_fit("Model B: yards per attempt ~ recent3_ypa + opp_sacks_per_game",
               ["intercept", "recent3_ypa", "opp_sacks_per_game"], coef_b, lo_b, hi_b)

    # Model C: TD rate per attempt ~ recent3_td_rate + opp_sacks_per_game
    Xc = [[1.0, r["recent3_td_rate"], r["opp_sacks_per_game"]] for r in dev_val]
    yc = [r["actual_td_rate"] for r in dev_val]
    coef_c = fit_ols(Xc, yc)
    lo_c, hi_c = bootstrap_ci(Xc, yc)
    report_fit("Model C: TD rate per attempt ~ recent3_td_rate + opp_sacks_per_game",
               ["intercept", "recent3_td_rate", "opp_sacks_per_game"], coef_c, lo_c, hi_c)

    print("\n--- Summary ---")
    print(f"opp_sacks_per_game -> attempt volume:   "
          f"{'REAL' if not (lo_a[2] <= 0 <= hi_a[2]) else 'no real effect'}")
    print(f"opp_sacks_per_game -> yards per attempt: "
          f"{'REAL' if not (lo_b[2] <= 0 <= hi_b[2]) else 'no real effect'}")
    print(f"opp_sacks_per_game -> TD rate per attempt: "
          f"{'REAL' if not (lo_c[2] <= 0 <= hi_c[2]) else 'no real effect'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
