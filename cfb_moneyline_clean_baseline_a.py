#!/usr/bin/env python3
"""
CFB_MONEYLINE_CLEAN_BASELINE_A

The REAL "who wins the game" market -- unlike every other market in this
file's family (a player's own stat line vs a line), this one is team-level:
predict which of the two real teams in a real, already-scheduled FBS game
wins outright. CFB is a much better fit for this than MLB turned out to be
(MLB's moneyline was built, tried, and retired 2026-08-25 at 163/302, 54.0%
lifetime -- never cleared a bar worth keeping live): CFB has enormous real
team-quality separation (blowouts are common, unlike MLB's day-to-day
pitcher-driven variance), and this repo already has 8 real seasons
(2018-2026) of real final scores in cfb_model.sqlite's games table
(home_points/away_points, confirmed present and populated) to train on --
far more signal-per-game than any player prop here works with.

Each real game produces TWO rows (one per team's own perspective) rather
than one home-anchored row: (this team's own asof stats, the opponent's
asof stats, is_home, is_neutral_site) -> did THIS team win. This is
symmetric by construction -- there's no "home team" bias baked into what
the model learns, and the same trained classifier scores either side of
any future matchup by just swapping which team's features go first.

Team state (win rate, net scoring margin, points for/against per game) is
tracked per-season, asof each week, the same running-state pattern already
proven for CFB's anytime_touchdowns margin features -- reset each season
(no cross-season carryover), strictly prior-week discipline throughout.

MIN_PRIOR_GAMES=3 for BOTH teams before a game produces a row: same
minimum-sample floor as every other market in this repo, for the same
reason (a 0-2 game sample can't support a real probability claim) --
means the first ~2-3 weeks of each season won't have moneyline rows,
same known early-season gap anytime_touchdowns has before its own
prior-season variant exists.

Run
---
python -u cfb_moneyline_clean_baseline_a.py --source cfb_models/cfb_model.sqlite
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

SOURCE_DEFAULT = "cfb_models/cfb_model.sqlite"
WORKDIR_DEFAULT = "cfb_models/cfb_moneyline_clean_baseline_a_work"
MIN_PRIOR_GAMES = 3
DEV_SEASONS = (2018, 2019, 2020, 2021, 2022)
VAL_SEASON = 2023
HOLDOUT_SEASON = 2024

MODEL_COLUMNS = [
    "team_net_margin", "team_win_rate", "team_avg_points_for", "team_avg_points_against",
    "opp_net_margin", "opp_win_rate", "opp_avg_points_for", "opp_avg_points_against",
    "projected_margin", "is_home", "is_neutral_site",
    "team_games_played", "opp_games_played",
]


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_team_state_asof(conn):
    """(season, team, week) -> state dict, computed strictly BEFORE that
    week using only real prior games this season. Reset per season, same
    convention as cfb_anytime_touchdowns_clean_baseline_a.py's
    build_team_margin_asof()."""
    games = conn.execute(
        "SELECT game_id, season, week, home_team, away_team, home_points, away_points, neutral_site "
        "FROM games WHERE home_points IS NOT NULL AND away_points IS NOT NULL "
        "ORDER BY season, week").fetchall()
    by_season_week = {}
    for g in games:
        by_season_week.setdefault((g[1], g[2]), []).append(g)

    # team_state[(season, team)] = [wins, points_for, points_against, games]
    team_state = {}
    state_asof = {}
    for (season, week) in sorted(by_season_week):
        for g in by_season_week[(season, week)]:
            _, _, _, home, away, hp, ap, neutral = g
            for team in (home, away):
                st = team_state.get((season, team), [0, 0, 0, 0])
                n = st[3]
                state_asof[(season, team, week)] = {
                    "games_played": n,
                    "win_rate": (st[0] / n) if n > 0 else None,
                    "net_margin": ((st[1] - st[2]) / n) if n > 0 else None,
                    "avg_points_for": (st[1] / n) if n > 0 else None,
                    "avg_points_against": (st[2] / n) if n > 0 else None,
                }
        for g in by_season_week[(season, week)]:
            _, _, _, home, away, hp, ap, neutral = g
            hst = team_state.setdefault((season, home), [0, 0, 0, 0])
            hst[0] += 1 if hp > ap else 0
            hst[1] += hp; hst[2] += ap; hst[3] += 1
            ast = team_state.setdefault((season, away), [0, 0, 0, 0])
            ast[0] += 1 if ap > hp else 0
            ast[1] += ap; ast[2] += hp; ast[3] += 1
    return state_asof


def build_rows(conn):
    state_asof = build_team_state_asof(conn)

    games = conn.execute(
        "SELECT game_id, season, week, home_team, away_team, home_points, away_points, neutral_site "
        "FROM games WHERE home_points IS NOT NULL AND away_points IS NOT NULL "
        "ORDER BY season, week").fetchall()

    out = []
    for gid, season, week, home, away, hp, ap, neutral in games:
        home_st = state_asof.get((season, home, week))
        away_st = state_asof.get((season, away, week))
        if not home_st or not away_st:
            continue
        if home_st["games_played"] < MIN_PRIOR_GAMES or away_st["games_played"] < MIN_PRIOR_GAMES:
            continue
        home_won = 1 if hp > ap else 0
        is_neutral = 1.0 if neutral else 0.0

        def feat_row(team, opp, own_st, opp_st, is_home, won):
            proj_margin = (own_st["net_margin"] - opp_st["net_margin"])
            return {
                "game_id": gid, "season": season, "week": week,
                "team": team, "opponent": opp,
                "team_net_margin": own_st["net_margin"],
                "team_win_rate": own_st["win_rate"],
                "team_avg_points_for": own_st["avg_points_for"],
                "team_avg_points_against": own_st["avg_points_against"],
                "opp_net_margin": opp_st["net_margin"],
                "opp_win_rate": opp_st["win_rate"],
                "opp_avg_points_for": opp_st["avg_points_for"],
                "opp_avg_points_against": opp_st["avg_points_against"],
                "projected_margin": proj_margin,
                "is_home": 1.0 if is_home else 0.0,
                "is_neutral_site": is_neutral,
                "team_games_played": own_st["games_played"],
                "opp_games_played": opp_st["games_played"],
                "team_won": won,
            }

        out.append(feat_row(home, away, home_st, away_st, True, home_won))
        out.append(feat_row(away, home, away_st, home_st, False, 1 - home_won))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE_DEFAULT)
    ap.add_argument("--workdir", default=WORKDIR_DEFAULT)
    args = ap.parse_args()

    src = Path(args.source)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    base_db = work / "baseline.sqlite"

    print("CFB_MONEYLINE_CLEAN_BASELINE_A\n===============================")
    print(f"source={src}\nworkdir={work}")

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    rows = build_rows(conn)
    conn.close()
    print(f"\ntotal eligible team-perspective rows: {len(rows)} ({len(rows)//2} games)")

    if base_db.exists():
        base_db.unlink()
    out = sqlite3.connect(str(base_db))
    cols_sql = ", ".join(f"{c} REAL" for c in MODEL_COLUMNS)
    out.execute(f"""CREATE TABLE cfb_moneyline_baseline (
        game_id TEXT, team TEXT, opponent TEXT,
        season INTEGER, week INTEGER, {cols_sql},
        team_won INTEGER
    )""")
    insert_cols = (["game_id", "team", "opponent", "season", "week"]
                   + MODEL_COLUMNS + ["team_won"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO cfb_moneyline_baseline ({', '.join(insert_cols)}) VALUES ({placeholder})"

    by_season = {}
    batch = []
    for r in rows:
        batch.append(tuple(r[c] for c in insert_cols))
        st = by_season.setdefault(r["season"], {"rows": 0, "won": 0})
        st["rows"] += 1
        st["won"] += r["team_won"]
        if len(batch) >= 5000:
            out.executemany(ins, batch); batch = []
    if batch:
        out.executemany(ins, batch)
    out.commit()
    out.close()

    manifest = {
        "script": "CFB_MONEYLINE_CLEAN_BASELINE_A",
        "generated_at_utc": now_utc(),
        "source_db": str(src),
        "source_db_sha256": sha256_file(src),
        "baseline_db": str(base_db),
        "model_columns": MODEL_COLUMNS,
        "eligibility": {"min_prior_games_each_team": MIN_PRIOR_GAMES},
        "target": "team_won = 1 if this team's perspective row's team scored more points",
        "dev_seasons": list(DEV_SEASONS), "val_season": VAL_SEASON,
        "holdout_season": HOLDOUT_SEASON,
        "total_rows": len(rows), "by_season": by_season,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'season':8s}{'rows':>8s}{'win_rate':>10s}")
    for s in sorted(by_season):
        st = by_season[s]
        rate = st["won"] / st["rows"] if st["rows"] else 0
        print(f"{s:<8}{st['rows']:>8}{rate:>10.4f}")

    print(f"\nbaseline db: {base_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
