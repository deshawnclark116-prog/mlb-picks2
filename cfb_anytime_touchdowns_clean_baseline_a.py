#!/usr/bin/env python3
"""
CFB_ANYTIME_TOUCHDOWNS_CLEAN_BASELINE_A

The REAL "anytime TD" prop, unlike rushing_touchdowns (which only counts
rushing scores): a real sportsbook anytime-TD prop pays out on a
touchdown scored ANY way -- a RB catching a TD pass or a WR taking an
end-around for a score both count. rushing_touchdowns/receiving_touchdowns
being scored SEPARATELY was too narrow; this combines both into one
target for one combined RB+WR population.

Eligibility reuses each position's own already-validated volume floor
rather than inventing a new combined-touches number: RB with recent3
carries >= 12 (rushing_yards' own bar) OR WR with recent3 receptions >=5
(receiving_yards' own bar) -- a player qualifies via whichever real
role they actually have, not an arbitrary blended threshold.

LINE=0.5: diagnostic-derived from real 2018-2024 data. Among eligible
RB+WR games, combined-TD distribution was 0:4949, 1:2773, 2:1041, 3:289,
4:66, 5:12, 6:3, 8:1 -- over 0.5 (>=1 TD) splits 45.8%/54.2%, about as
balanced a line as this stat gets.

Same 2025-exclusion as every other TD market here (cfbfastR's
touchdown_player_id field is measurably incomplete for 2025 specifically):
DEV=2018-2022, VAL=2023, HOLDOUT=2024.

Run
---
python -u cfb_anytime_touchdowns_clean_baseline_a.py
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
WORKDIR_DEFAULT = "/data/cfb_model/cfb_anytime_touchdowns_clean_baseline_a_work"
MIN_PRIOR_GAMES_FOR_RATE = 3
MIN_RECENT_CARRIES_PER_GAME = 12
MIN_RECENT_RECEPTIONS_PER_GAME = 5
LINE = 0.5
DEV_SEASONS = (2018, 2019, 2020, 2021, 2022)
VAL_SEASON = 2023
HOLDOUT_SEASON = 2024

MODEL_COLUMNS = [
    "season_avg_total_td", "recent3_avg_total_td", "recent5_avg_total_td",
    "season_avg_touches", "recent3_avg_touches", "td_per_touch",
    "opp_total_td_allowed_per_game", "is_home", "games_played",
    "team_net_margin", "opp_net_margin", "projected_margin", "is_wr",
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
        SELECT player_id, player_name, position, team, opponent, season, week,
               is_home, carries, receptions, rushing_touchdowns, receiving_touchdowns
        FROM player_games
        WHERE position IN ('RB', 'WR')
        ORDER BY player_id, season, week, game_date
    """).fetchall()

    by_season_week = {}
    for r in rows:
        by_season_week.setdefault((r[5], r[6]), []).append(r)

    opp_state = {}
    opp_asof = {}
    for (season, week) in sorted(by_season_week):
        wk_rows = by_season_week[(season, week)]
        for r in wk_rows:
            pid, opp = r[0], r[4]
            key = (pid, season, week)
            st = opp_state.get((season, opp))
            opp_asof[key] = (st[0] / st[1]) if st and st[1] > 0 else None
        for r in wk_rows:
            opp = r[4]
            total_td = (r[10] or 0) + (r[11] or 0)
            st = opp_state.setdefault((season, opp), [0, 0])
            st[0] += total_td
            st[1] += 1

    out = []
    cur_key = None
    group = []

    def flush(group):
        cum_td = cum_touch = 0
        n_prior = 0
        td_hist = deque(maxlen=15)
        touch_hist = deque(maxlen=15)
        car_hist = deque(maxlen=15)
        rec_hist = deque(maxlen=15)
        for r in group:
            pid, pname, pos, team, opp, season, week, is_home, carries, receptions, rtd, rectd = r
            c3 = list(car_hist)[-3:]
            rc3 = list(rec_hist)[-3:]
            recent_car_rate = sum(c3) / len(c3) if c3 else 0.0
            recent_rec_rate = sum(rc3) / len(rc3) if rc3 else 0.0
            eligible = (n_prior >= MIN_PRIOR_GAMES_FOR_RATE
                        and ((pos == "RB" and recent_car_rate >= MIN_RECENT_CARRIES_PER_GAME)
                             or (pos == "WR" and recent_rec_rate >= MIN_RECENT_RECEPTIONS_PER_GAME)))
            if eligible:
                r3 = list(td_hist)[-3:]
                r5 = list(td_hist)[-5:]
                t3 = list(touch_hist)[-3:]
                team_margin = margin_asof.get((season, team, week))
                opp_margin = margin_asof.get((season, opp, week))
                proj_margin = (team_margin - opp_margin) if (team_margin is not None and opp_margin is not None) else None
                feat = {
                    "season_avg_total_td": cum_td / n_prior,
                    "recent3_avg_total_td": sum(r3) / len(r3) if r3 else 0.0,
                    "recent5_avg_total_td": sum(r5) / len(r5) if r5 else 0.0,
                    "season_avg_touches": cum_touch / n_prior,
                    "recent3_avg_touches": sum(t3) / len(t3) if t3 else 0.0,
                    "td_per_touch": (cum_td / cum_touch) if cum_touch > 0 else 0.0,
                    "opp_total_td_allowed_per_game": opp_asof.get((pid, season, week)),
                    "is_home": 1.0 if is_home else 0.0,
                    "games_played": n_prior,
                    "team_net_margin": team_margin,
                    "opp_net_margin": opp_margin,
                    "projected_margin": proj_margin,
                    "is_wr": 1.0 if pos == "WR" else 0.0,
                }
                actual_td = (rtd or 0) + (rectd or 0)
                out.append({
                    "player_id": pid, "player_name": pname, "team": team,
                    "opponent": opp, "season": season, "week": week,
                    **feat, "actual_total_touchdowns": actual_td,
                })
            this_td = (rtd or 0) + (rectd or 0)
            this_touch = (carries or 0) + (receptions or 0)
            cum_td += this_td
            cum_touch += this_touch
            td_hist.append(this_td)
            touch_hist.append(this_touch)
            car_hist.append(carries if carries is not None else 0)
            rec_hist.append(receptions if receptions is not None else 0)
            n_prior += 1
        return

    for r in rows:
        key = (r[0], r[5])
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

    print("CFB_ANYTIME_TOUCHDOWNS_CLEAN_BASELINE_A\n========================================")
    print(f"source={src}\nworkdir={work}\nline={LINE}")

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    rows = build_rows(conn)
    conn.close()
    print(f"\ntotal eligible RB+WR rows: {len(rows)}")

    if base_db.exists():
        base_db.unlink()
    out = sqlite3.connect(str(base_db))
    cols_sql = ", ".join(f"{c} REAL" for c in MODEL_COLUMNS)
    out.execute(f"""CREATE TABLE cfb_anytime_touchdowns_baseline (
        player_id TEXT, player_name TEXT, team TEXT, opponent TEXT,
        season INTEGER, week INTEGER, {cols_sql},
        actual_total_touchdowns INTEGER, over_line INTEGER
    )""")
    insert_cols = (["player_id", "player_name", "team", "opponent", "season", "week"]
                   + MODEL_COLUMNS + ["actual_total_touchdowns", "over_line"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO cfb_anytime_touchdowns_baseline ({', '.join(insert_cols)}) VALUES ({placeholder})"

    by_season = {}
    batch = []
    for r in rows:
        r["over_line"] = 1 if r["actual_total_touchdowns"] >= (LINE + 0.5) else 0
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
        "script": "CFB_ANYTIME_TOUCHDOWNS_CLEAN_BASELINE_A",
        "generated_at_utc": now_utc(),
        "source_db": str(src),
        "source_db_sha256": sha256_file(src),
        "baseline_db": str(base_db),
        "model_columns": MODEL_COLUMNS,
        "eligibility": {"position": "RB or WR",
                        "min_prior_games_for_rate": MIN_PRIOR_GAMES_FOR_RATE,
                        "min_recent_carries_per_game_rb": MIN_RECENT_CARRIES_PER_GAME,
                        "min_recent_receptions_per_game_wr": MIN_RECENT_RECEPTIONS_PER_GAME},
        "target": f"over_line = actual_total_touchdowns (rush+rec) >= {LINE + 0.5} (line {LINE})",
        "line": LINE,
        "dev_seasons": list(DEV_SEASONS), "val_season": VAL_SEASON,
        "holdout_season": HOLDOUT_SEASON,
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
