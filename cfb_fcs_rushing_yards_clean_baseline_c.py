#!/usr/bin/env python3
"""
CFB_FCS_RUSHING_YARDS_CLEAN_BASELINE_C

Four techniques (Platt, regularization reusing the FBS market's own
fix, pooling a bigger holdout, isotonic) all failed to fix rushing_
yards' calibration gap, and the SAME modest-but-real gap showed up on
all three other FCS markets too -- pointing at something structural
about the FCS population itself, not a per-market model issue. FBS
receiving_yards/passing_yards hit an analogous wall and it took
Power4-vs-Power4 scoping (restricting to a more competitively
homogeneous population) to actually fix it, not another model-side
correction.

FCS has no equivalent "Power 4" label, but it does have a real,
uncontroversial structural fact to scope by instead of inventing one:
the Ivy League and Pioneer League are the only two FCS conferences that
structurally never award athletic scholarships (official conference
policy, not derived from any performance data) -- a real competitive-
tier distinction that exists independent of this season's or any
season's outcomes, decided on that basis alone, not by looking at which
split would fix the calibration numbers. Scoping to games where BOTH
teams are from a scholarship-awarding conference removes 453 of 2095
FCS-vs-FCS games (21.6%), leaving 1610 -- a similar-scale population cut
to what Power4 scoping did on the FBS side.

Same DEV=2022-2023/VAL=2024/HOLDOUT=2025 structure, same eligibility,
same feature set as clean_baseline_a. LINE re-derived fresh on this
scoped population (not assumed to match clean_baseline_a's).

Run
---
python -u cfb_fcs_rushing_yards_clean_baseline_c.py
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
WORKDIR_DEFAULT = "/data/cfb_model/cfb_fcs_rushing_yards_clean_baseline_c_work"
MIN_PRIOR_GAMES_FOR_RATE = 3
MIN_RECENT_CARRIES_PER_GAME = 12
THRESHOLD_CANDIDATES = [29.5, 39.5, 49.5, 59.5, 69.5, 79.5]
DEV_SEASONS = (2022, 2023)
VAL_SEASON = 2024
HOLDOUT_SEASON = 2025
NON_SCHOLARSHIP = {"Ivy", "Pioneer"}


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scholarship_game_ids(conn):
    rows = conn.execute(
        "SELECT game_id FROM games WHERE "
        "(home_conference IS NULL OR home_conference NOT IN (?, ?)) AND "
        "(away_conference IS NULL OR away_conference NOT IN (?, ?))",
        tuple(NON_SCHOLARSHIP) * 2).fetchall()
    return {r[0] for r in rows}


MODEL_COLUMNS = [
    "season_avg_rush_yards", "recent3_avg_rush_yards", "recent5_avg_rush_yards",
    "season_avg_carries", "recent3_avg_carries", "yards_per_carry",
    "opp_rush_yards_allowed_per_game", "is_home", "games_played",
    "team_net_margin", "opp_net_margin", "projected_margin",
]


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
    sch_games = scholarship_game_ids(conn)

    rows = conn.execute("""
        SELECT player_id, player_name, team, opponent, season, week,
               is_home, carries, rushing_yards, game_id
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
            pid, pname, team, opp, season, week, is_home, carries, rush_yards, gid = r
            c3 = list(carries_hist)[-3:]
            recent_carry_rate = sum(c3) / len(c3) if c3 else 0.0
            if (n_prior >= MIN_PRIOR_GAMES_FOR_RATE
                    and recent_carry_rate >= MIN_RECENT_CARRIES_PER_GAME
                    and gid in sch_games):
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

    print("CFB_FCS_RUSHING_YARDS_CLEAN_BASELINE_C\n=======================================")
    print(f"source={src}\nworkdir={work}\nscholarship-conference-only scoping (excludes {NON_SCHOLARSHIP})")

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
        "script": "CFB_FCS_RUSHING_YARDS_CLEAN_BASELINE_C",
        "generated_at_utc": now_utc(),
        "source_db": str(src), "source_db_sha256": sha256_file(src),
        "baseline_db": str(base_db), "model_columns": MODEL_COLUMNS,
        "scoping": "scholarship-conference-only (excludes Ivy, Pioneer)",
        "eligibility": {"position": "RB",
                        "min_prior_games_for_rate": MIN_PRIOR_GAMES_FOR_RATE,
                        "min_recent_carries_per_game": MIN_RECENT_CARRIES_PER_GAME},
        "target": f"over_line = actual_rushing_yards >= {LINE + 0.5} (line {LINE})",
        "line": LINE, "dev_seasons": list(DEV_SEASONS), "val_season": VAL_SEASON,
        "holdout_season": HOLDOUT_SEASON, "total_rows": len(rows), "by_season": by_season,
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
