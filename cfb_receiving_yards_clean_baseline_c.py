#!/usr/bin/env python3
"""
CFB_RECEIVING_YARDS_CLEAN_BASELINE_C

gate_b (margin features + regularization) improved receiving_yards but
still failed decisively (AUC 0.5615, still loses to constant on logloss/
Brier). Different angle: scope the population to Power 4-vs-Power 4 games
only (Big Ten/ACC/SEC/Big 12, confirmed as the real conference labels in
this data). Theory: WR usage/target distribution is probably the noisiest
in Group of 5 / mismatch games specifically -- backups mopping up garbage
time, or a Power 4 program's passing game shut down entirely by a bad
mismatch -- exactly the kind of non-stationary population that would
suppress AUC. Real Power 4 vs Power 4 games are the closest thing to a
"competitive, meaningful" population, and likely the closest to what a
real WR receiving-yards market would actually be offered on anyway (props
are rarely posted on a 45-point mismatch).

Same features as clean_baseline_b (receptions-based volume + team-margin
context). Everything else identical.

Run
---
python -u cfb_receiving_yards_clean_baseline_c.py
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

SOURCE_DEFAULT = "/data/cfb_model/cfb_model.sqlite"
WORKDIR_DEFAULT = "/data/cfb_model/cfb_receiving_yards_clean_baseline_c_work"
MIN_PRIOR_GAMES_FOR_RATE = 3
MIN_RECENT_RECEPTIONS_PER_GAME = 5
LINE = 59.5
DEV_SEASONS = (2022, 2023)
HOLDOUT_SEASON = 2025
POWER4 = {"Big Ten", "ACC", "SEC", "Big 12"}

MODEL_COLUMNS = [
    "season_avg_rec_yards", "recent3_avg_rec_yards", "recent5_avg_rec_yards",
    "season_avg_receptions", "recent3_avg_receptions", "yards_per_reception",
    "opp_rec_yards_allowed_per_game", "is_home", "games_played",
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


def power4_game_ids(conn):
    rows = conn.execute(
        "SELECT game_id FROM games WHERE home_conference IN ({0}) AND away_conference IN ({0})"
        .format(",".join("?" for _ in POWER4)), tuple(POWER4) * 2).fetchall()
    return {r[0] for r in rows}


def build_rows(conn):
    margin_asof = build_team_margin_asof(conn)
    p4_games = power4_game_ids(conn)

    rows = conn.execute("""
        SELECT player_id, player_name, team, opponent, season, week,
               is_home, receptions, receiving_yards, game_id
        FROM player_games
        WHERE position = 'WR'
        ORDER BY player_id, season, week
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
        cum_ry = cum_rec = 0
        n_prior = 0
        ry_hist = deque(maxlen=15)
        rec_hist = deque(maxlen=15)
        for r in group:
            pid, pname, team, opp, season, week, is_home, receptions, rec_yards, gid = r
            c3 = list(rec_hist)[-3:]
            recent_rec_rate = sum(c3) / len(c3) if c3 else 0.0
            if (n_prior >= MIN_PRIOR_GAMES_FOR_RATE
                    and recent_rec_rate >= MIN_RECENT_RECEPTIONS_PER_GAME
                    and gid in p4_games):
                r3 = list(ry_hist)[-3:]
                r5 = list(ry_hist)[-5:]
                team_margin = margin_asof.get((season, team, week))
                opp_margin = margin_asof.get((season, opp, week))
                proj_margin = (team_margin - opp_margin) if (team_margin is not None and opp_margin is not None) else None
                feat = {
                    "season_avg_rec_yards": cum_ry / n_prior,
                    "recent3_avg_rec_yards": sum(r3) / len(r3) if r3 else 0.0,
                    "recent5_avg_rec_yards": sum(r5) / len(r5) if r5 else 0.0,
                    "season_avg_receptions": cum_rec / n_prior,
                    "recent3_avg_receptions": sum(c3) / len(c3) if c3 else 0.0,
                    "yards_per_reception": (cum_ry / cum_rec) if cum_rec > 0 else 0.0,
                    "opp_rec_yards_allowed_per_game": opp_asof.get((pid, season, week)),
                    "is_home": 1.0 if is_home else 0.0,
                    "games_played": n_prior,
                    "team_net_margin": team_margin,
                    "opp_net_margin": opp_margin,
                    "projected_margin": proj_margin,
                }
                actual_ry = rec_yards if rec_yards is not None else 0
                out.append({
                    "player_id": pid, "player_name": pname, "team": team,
                    "opponent": opp, "season": season, "week": week,
                    **feat, "actual_receiving_yards": actual_ry,
                })
            cum_ry += rec_yards if rec_yards is not None else 0
            cum_rec += receptions if receptions is not None else 0
            ry_hist.append(rec_yards if rec_yards is not None else 0)
            rec_hist.append(receptions if receptions is not None else 0)
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

    print("CFB_RECEIVING_YARDS_CLEAN_BASELINE_C\n=====================================")
    print(f"source={src}\nworkdir={work}\nline={LINE}\nPower4 conferences: {sorted(POWER4)}")

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    rows = build_rows(conn)
    conn.close()
    print(f"\ntotal eligible WR rows (Power4-vs-Power4 only): {len(rows)}")

    if base_db.exists():
        base_db.unlink()
    out = sqlite3.connect(str(base_db))
    cols_sql = ", ".join(f"{c} REAL" for c in MODEL_COLUMNS)
    out.execute(f"""CREATE TABLE cfb_receiving_yards_baseline (
        player_id TEXT, player_name TEXT, team TEXT, opponent TEXT,
        season INTEGER, week INTEGER, {cols_sql},
        actual_receiving_yards INTEGER, over_line INTEGER
    )""")
    insert_cols = (["player_id", "player_name", "team", "opponent", "season", "week"]
                   + MODEL_COLUMNS + ["actual_receiving_yards", "over_line"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO cfb_receiving_yards_baseline ({', '.join(insert_cols)}) VALUES ({placeholder})"

    by_season = {}
    batch = []
    for r in rows:
        r["over_line"] = 1 if r["actual_receiving_yards"] >= (LINE + 0.5) else 0
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
        "script": "CFB_RECEIVING_YARDS_CLEAN_BASELINE_C",
        "generated_at_utc": now_utc(),
        "source_db": str(src),
        "source_db_sha256": sha256_file(src),
        "baseline_db": str(base_db),
        "model_columns": MODEL_COLUMNS,
        "eligibility": {"position": "WR", "power4_only": sorted(POWER4),
                        "min_prior_games_for_rate": MIN_PRIOR_GAMES_FOR_RATE,
                        "min_recent_receptions_per_game": MIN_RECENT_RECEPTIONS_PER_GAME},
        "target": f"over_line = actual_receiving_yards >= {LINE + 0.5} (line {LINE})",
        "line": LINE,
        "dev_seasons": list(DEV_SEASONS), "holdout_season": HOLDOUT_SEASON,
        "total_rows": len(rows), "by_season": by_season,
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
