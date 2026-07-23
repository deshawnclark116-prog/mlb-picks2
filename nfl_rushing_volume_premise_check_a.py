#!/usr/bin/env python3
"""
NFL_RUSHING_VOLUME_PREMISE_CHECK_A

Before building the volume/efficiency architecture for rushing_yards,
verify its two load-bearing assumptions against real data rather than
assume them:

  1. Team-game rushing volume (total team carries) is meaningfully
     predicted by game script (spread_line, total_line) -- the whole
     reason for decoupling volume from a black-box model in the first
     place.
  2. An RB's SHARE of team carries is more stable game-to-game than
     their RAW carry count -- the reason to predict share x team-volume
     rather than raw carries directly.

If either premise is weak, the architecture needs rethinking before any
more engineering goes into it. Read-only, no model training, no writes.

Run
---
python -u nfl_rushing_volume_premise_check_a.py
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

REPO = Path(__file__).resolve().parent
DB_DEFAULT = REPO / "nfl_models" / "nfl_model.sqlite"


def pearson(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    con = sqlite3.connect(f"file:{DB_DEFAULT}?mode=ro", uri=True)
    seasons = sorted(r[0] for r in con.execute("SELECT DISTINCT season FROM player_games"))
    print(f"player_games seasons in db: {seasons}")

    # -------- Verify spread_line's sign convention against moneyline first --------
    # Moneyline is unambiguous (negative = favorite) -- use it to empirically
    # determine which side spread_line is signed from, rather than assume a
    # convention from memory (which nearly got this backwards: an early look
    # at the OLS coefficient direction didn't match "favorites run more").
    print("================ Verifying spread_line sign convention (via moneyline) ================")
    conv_rows = con.execute("""
        SELECT game_id, home_team, away_team, spread_line, home_moneyline, away_moneyline
        FROM games
        WHERE spread_line IS NOT NULL AND home_moneyline IS NOT NULL
              AND away_moneyline IS NOT NULL AND home_moneyline != away_moneyline
        LIMIT 500
    """).fetchall()
    home_favored_when_spread_neg = 0
    home_favored_when_spread_pos = 0
    for (gid, home, away, spread, hml, aml) in conv_rows:
        home_favored = hml < aml
        if spread < 0:
            home_favored_when_spread_neg += int(home_favored)
        elif spread > 0:
            home_favored_when_spread_pos += int(home_favored)
    n_neg = sum(1 for r in conv_rows if r[3] < 0)
    n_pos = sum(1 for r in conv_rows if r[3] > 0)
    print(f"  when spread_line < 0 (n={n_neg}): home favored {home_favored_when_spread_neg}/{n_neg} of the time")
    print(f"  when spread_line > 0 (n={n_pos}): home favored {home_favored_when_spread_pos}/{n_pos} of the time")
    # If home is favored almost always when spread<0, spread_line is HOME-signed
    # (negative=home favored). If home is favored almost always when spread>0,
    # spread_line is AWAY-signed (negative=away favored, so positive=home favored).
    home_signed = (home_favored_when_spread_neg / max(n_neg, 1)) > (home_favored_when_spread_pos / max(n_pos, 1))
    print(f"  CONCLUSION: spread_line is {'HOME' if home_signed else 'AWAY'}-signed "
          f"(negative = {'home' if home_signed else 'away'} favored)")

    # -------- Premise 1: team-game rush volume vs game script --------
    print("\n================ PREMISE 1: team rush volume vs game script ================")
    team_carries = con.execute("""
        SELECT pg.game_id, pg.team, pg.season, pg.week,
               SUM(pg.carries) as team_carries,
               g.home_team, g.away_team, g.spread_line, g.total_line
        FROM player_games pg
        JOIN games g ON pg.game_id = g.game_id
        WHERE pg.season_type = 'REG' AND pg.carries IS NOT NULL
              AND g.spread_line IS NOT NULL AND g.total_line IS NOT NULL
        GROUP BY pg.game_id, pg.team
    """).fetchall()
    print(f"team-game rows with carries + spread/total: {len(team_carries)}")

    own_spread, totals, carries = [], [], []
    for (gid, team, season, week, tc, home, away, spread, total) in team_carries:
        # "own spread": negative = this team favored, regardless of home/away,
        # using the empirically-determined sign convention above (NOT assumed).
        if home_signed:
            own = spread if team == home else -spread
        else:
            own = -spread if team == home else spread
        own_spread.append(own)
        totals.append(total)
        carries.append(tc)

    r_spread = pearson(own_spread, carries)
    r_total = pearson(totals, carries)
    print(f"corr(team_carries, own_spread [negative=favored]): {r_spread:+.4f}")
    print(f"corr(team_carries, total_line): {r_total:+.4f}")

    # Simple 2-var linear regression via least squares, report R^2.
    X = np.column_stack([np.ones(len(carries)), own_spread, totals])
    yv = np.asarray(carries, dtype=float)
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"OLS team_carries ~ own_spread + total_line:  R^2={r2:.4f}  "
          f"coef(spread)={coef[1]:+.4f}  coef(total)={coef[2]:+.4f}")
    print(f"(negative coef(spread) expected: more negative spread = bigger "
          f"favorite = MORE carries, since spread is negative-favored)")
    print(f"(positive coef(total) expected-ish: higher total = more plays overall)")

    # -------- Premise 2: carry share stability vs raw-carries stability --------
    print("\n================ PREMISE 2: share stability vs raw-count stability ================")
    rb_rows = con.execute("""
        SELECT pg.player_id, pg.player_name, pg.season, pg.week, pg.game_id, pg.team, pg.carries
        FROM player_games pg
        WHERE pg.position = 'RB' AND pg.season_type = 'REG' AND pg.carries IS NOT NULL
        ORDER BY pg.player_id, pg.season, pg.week
    """).fetchall()
    team_totals = {(gid, team): tc for (gid, team, season, week, tc, home, away, spread, total)
                   in team_carries}

    by_player = {}
    for (pid, pname, season, week, gid, team, carries_n) in rb_rows:
        tt = team_totals.get((gid, team))
        if tt is None or tt == 0:
            continue
        share = carries_n / tt
        by_player.setdefault(pid, {"name": pname, "rows": []})
        by_player[pid]["rows"].append((season, week, carries_n, share))

    # For players with >= 8 games, compute coefficient of variation (std/mean)
    # for raw carries vs for share -- lower CV = more stable/predictable.
    cv_raw, cv_share = [], []
    for pid, d in by_player.items():
        rows = d["rows"]
        if len(rows) < 8:
            continue
        raw_vals = np.array([r[2] for r in rows], dtype=float)
        share_vals = np.array([r[3] for r in rows], dtype=float)
        if raw_vals.mean() > 0:
            cv_raw.append(float(raw_vals.std() / raw_vals.mean()))
        if share_vals.mean() > 0:
            cv_share.append(float(share_vals.std() / share_vals.mean()))

    print(f"RBs with >= 8 games: {len(cv_raw)}")
    print(f"mean CV(raw carries):  {np.mean(cv_raw):.4f}  median {np.median(cv_raw):.4f}")
    print(f"mean CV(carry share):  {np.mean(cv_share):.4f}  median {np.median(cv_share):.4f}")
    print(f"(lower CV = more stable/predictable; share should be LOWER if the "
          f"premise holds -- team-level noise no longer pollutes the player signal)")

    # Week-over-week autocorrelation (lag-1) for raw vs share, pooled across
    # players with enough games -- a more direct "is next week predictable
    # from this week" check than overall CV.
    def lag1_autocorr(rows, idx):
        rows_sorted = sorted(rows, key=lambda r: (r[0], r[1]))
        vals = [r[idx] for r in rows_sorted]
        if len(vals) < 4:
            return None
        x = np.array(vals[:-1]); y = np.array(vals[1:])
        return pearson(x, y)

    ac_raw, ac_share = [], []
    for pid, d in by_player.items():
        if len(d["rows"]) < 8:
            continue
        a = lag1_autocorr(d["rows"], 2)
        b = lag1_autocorr(d["rows"], 3)
        if a is not None and not np.isnan(a):
            ac_raw.append(a)
        if b is not None and not np.isnan(b):
            ac_share.append(b)
    print(f"\nmean lag-1 autocorrelation, raw carries:  {np.mean(ac_raw):+.4f}")
    print(f"mean lag-1 autocorrelation, carry share:  {np.mean(ac_share):+.4f}")
    print(f"(higher = this week predicts next week better; share should be "
          f"HIGHER if the premise holds)")

    con.close()


if __name__ == "__main__":
    raise SystemExit(main())
