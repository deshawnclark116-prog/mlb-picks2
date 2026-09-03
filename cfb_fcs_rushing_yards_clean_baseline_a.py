#!/usr/bin/env python3
"""
CFB_FCS_RUSHING_YARDS_CLEAN_BASELINE_A

FCS analog of cfb_rushing_yards_clean_baseline_b.py -- same population
definition, same eligibility rule, same feature set (including the
team-strength margin features that turned out to matter for FBS, reused
here directly rather than rediscovered, since the underlying reason --
big talent-gap blowouts distorting garbage-time volume -- applies at
least as much in FCS, which spans everything from playoff-contending
programs to teams that lose by 50).

Real, disclosed data-quality constraint found before writing a line of
model code (see cfb_espn_fcs_historical_foundation_a.py's backfill
results): ESPN's FCS box-score coverage is historically thin -- 2022 has
essentially no usable data (23 QB / 49 RB / 30 WR rows for the WHOLE
season), 2023 is still thin, and only 2024-2025 have real volume. DEV and
HOLDOUT are chosen on that basis, decided here before any model has been
run (not a post-hoc peek): DEV_SEASONS=(2024,) only, HOLDOUT_SEASON=2025
(untouched). 2022/2023 are excluded entirely -- too little signal to
usefully train on, and including them wouldn't be free (it would dilute
real 2024 signal with mostly-missing rows).

LINE is NOT copied from the FBS market (69.5) -- FCS rushing volume/
production is a real, different distribution and must be derived fresh,
same closest-to-50%-over-rate procedure as the original FBS derivation,
computed on DEV (2024) only.

Run
---
python -u cfb_fcs_rushing_yards_clean_baseline_a.py
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

SOURCE_DEFAULT = "cfb_models/cfb_fcs_model.sqlite"
WORKDIR_DEFAULT = "/data/cfb_model/cfb_fcs_rushing_yards_clean_baseline_a_work"
MIN_PRIOR_GAMES_FOR_RATE = 3
MIN_RECENT_CARRIES_PER_GAME = 8  # loosened from FBS's 12 -- FCS box-score
                                  # rows are sparser (see docstring), and a
                                  # feature back's real per-game carries
                                  # shouldn't itself differ much by division
THRESHOLD_CANDIDATES = [29.5, 39.5, 49.5, 59.5, 69.5, 79.5]
DEV_SEASONS = (2024,)
HOLDOUT_SEASON = 2025

MODEL_COLUMNS = [
    "season_avg_rush_yards", "recent3_avg_rush_yards", "recent5_avg_rush_yards",
    "season_avg_carries", "recent3_avg_carries", "yards_per_carry",
    "opp_rush_yards_allowed_per_game", "is_home", "games_played",
    "team_net_margin", "opp_net_margin", "projected_margin",
]


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def build_rows(conn):
    margin_asof = build_team_margin_asof(conn)

    rows = conn.execute("""
        SELECT player_id, player_name, team, opponent, season, week,
               is_home, carries, rushing_yards
        FROM player_games
        WHERE position = 'RB'
        ORDER BY player_id, season, week, game_date
    """).fetchall()

    by_season_week = {}
    for r in rows:
        by_season_week.setdefault((r[4], r[5]), []).append(r)

    opp_state = {}
    opp_asof = {}
    for (season, week) in sorted(by_season_week):
        wk_rows = by_season_week[(season, week)]
        for r in wk_rows:
            pid, opp = r[0], r[3]
            key = (pid, season, week)
            st = opp_state.get((season, opp))
            opp_asof[key] = (st[0] / st[1]) if st and st[1] > 0 else None
        for r in wk_rows:
            opp = r[3]
            ry = r[8] if r[8] is not None else 0
            st = opp_state.setdefault((season, opp), [0, 0])
            st[0] += ry
            st[1] += 1

    out = []
    cur_key = None
    group = []

    def flush(group):
        cum_ry = cum_carries = 0
        n_prior = 0
        ry_hist = deque(maxlen=15)
        carries_hist = deque(maxlen=15)
        for r in group:
            pid, pname, team, opp, season, week, is_home, carries, rush_yards = r
            c3 = list(carries_hist)[-3:]
            recent_carry_rate = sum(c3) / len(c3) if c3 else 0.0
            if (n_prior >= MIN_PRIOR_GAMES_FOR_RATE
                    and recent_carry_rate >= MIN_RECENT_CARRIES_PER_GAME):
                r3 = list(ry_hist)[-3:]
                r5 = list(ry_hist)[-5:]
                team_margin = margin_asof.get((season, team, week))
                opp_margin = margin_asof.get((season, opp, week))
                proj_margin = (team_margin - opp_margin) if (team_margin is not None and opp_margin is not None) else None
                feat = {
                    "season_avg_rush_yards": cum_ry / n_prior,
                    "recent3_avg_rush_yards": sum(r3) / len(r3) if r3 else 0.0,
                    "recent5_avg_rush_yards": sum(r5) / len(r5) if r5 else 0.0,
                    "season_avg_carries": cum_carries / n_prior,
                    "recent3_avg_carries": sum(c3) / len(c3) if c3 else 0.0,
                    "yards_per_carry": (cum_ry / cum_carries) if cum_carries > 0 else 0.0,
                    "opp_rush_yards_allowed_per_game": opp_asof.get((pid, season, week)),
                    "is_home": 1.0 if is_home else 0.0,
                    "games_played": n_prior,
                    "team_net_margin": team_margin,
                    "opp_net_margin": opp_margin,
                    "projected_margin": proj_margin,
                }
                actual_ry = rush_yards if rush_yards is not None else 0
                out.append({
                    "player_id": pid, "player_name": pname, "team": team,
                    "opponent": opp, "season": season, "week": week,
                    **feat, "actual_rushing_yards": actual_ry,
                })
            cum_ry += rush_yards if rush_yards is not None else 0
            cum_carries += carries if carries is not None else 0
            ry_hist.append(rush_yards if rush_yards is not None else 0)
            carries_hist.append(carries if carries is not None else 0)
            n_prior += 1
        return

    for r in rows:
        key = (r[0], r[4])
        if key != cur_key:
            if group:
                flush(group)
            group = []
            cur_key = key
        group.append(r)
    if group:
        flush(group)
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

    print("CFB_FCS_RUSHING_YARDS_CLEAN_BASELINE_A\n=======================================")
    print(f"source={src}\nworkdir={work}")

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    rows = build_rows(conn)
    conn.close()
    print(f"\ntotal eligible RB rows: {len(rows)}")
    by_season_n = {}
    for r in rows:
        by_season_n[r["season"]] = by_season_n.get(r["season"], 0) + 1
    print(f"by season: {by_season_n}")

    dev_ry = [r["actual_rushing_yards"] for r in rows if r["season"] in DEV_SEASONS]
    print(f"\nTHRESHOLD DIAGNOSTIC (dev {DEV_SEASONS} only, n={len(dev_ry)})")
    print(f"  {'threshold':>10s}{'over_rate':>12s}")
    best_t, best_dist = None, 1.0
    for t in THRESHOLD_CANDIDATES:
        rate = sum(1 for y in dev_ry if y >= t + 0.5) / len(dev_ry) if dev_ry else 0
        print(f"  {'>' + str(t):>10s}{rate:>12.3f}")
        dist = abs(rate - 0.5)
        if dist < best_dist:
            best_dist, best_t = dist, t
    LINE = best_t
    print(f"  closest-to-50% threshold: over {LINE}")

    if base_db.exists():
        base_db.unlink()
    out = sqlite3.connect(str(base_db))
    cols_sql = ", ".join(f"{c} REAL" for c in MODEL_COLUMNS)
    out.execute(f"""CREATE TABLE cfb_fcs_rushing_yards_baseline (
        player_id TEXT, player_name TEXT, team TEXT, opponent TEXT,
        season INTEGER, week INTEGER, {cols_sql},
        actual_rushing_yards INTEGER, over_line INTEGER
    )""")
    insert_cols = (["player_id", "player_name", "team", "opponent", "season", "week"]
                   + MODEL_COLUMNS + ["actual_rushing_yards", "over_line"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO cfb_fcs_rushing_yards_baseline ({', '.join(insert_cols)}) VALUES ({placeholder})"

    by_season = {}
    batch = []
    for r in rows:
        r["over_line"] = 1 if r["actual_rushing_yards"] >= (LINE + 0.5) else 0
        batch.append(tuple(r[c] for c in insert_cols))
        st = by_season.setdefault(r["season"], {"rows": 0, "over": 0})
        st["rows"] += 1
        st["over"] += r["over_line"]
        if len(batch) >= 5000:
            out.executemany(ins, batch); batch = []
    if batch:
        out.executemany(ins, batch)
    out.commit()
    out.close()

    manifest = {
        "script": "CFB_FCS_RUSHING_YARDS_CLEAN_BASELINE_A",
        "generated_at_utc": now_utc(),
        "source_db": str(src),
        "source_db_sha256": sha256_file(src),
        "baseline_db": str(base_db),
        "model_columns": MODEL_COLUMNS,
        "eligibility": {"position": "RB",
                        "min_prior_games_for_rate": MIN_PRIOR_GAMES_FOR_RATE,
                        "min_recent_carries_per_game": MIN_RECENT_CARRIES_PER_GAME},
        "target": f"over_line = actual_rushing_yards >= {LINE + 0.5} (line {LINE})",
        "line": LINE,
        "dev_seasons": list(DEV_SEASONS), "holdout_season": HOLDOUT_SEASON,
        "total_rows": len(rows), "by_season": by_season,
        "excluded_seasons": [2022, 2023],
        "excluded_reason": "too little box-score coverage to usefully train/validate on -- see module docstring",
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'season':8s}{'rows':>8s}{'over_' + str(LINE):>10s}")
    for s in sorted(by_season):
        st = by_season[s]
        rate = st["over"] / st["rows"] if st["rows"] else 0
        print(f"{s:<8}{st['rows']:>8}{rate:>10.4f}")

    print(f"\nbaseline db: {base_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
