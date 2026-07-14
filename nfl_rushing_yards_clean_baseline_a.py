#!/usr/bin/env python3
"""
NFL_RUSHING_YARDS_CLEAN_BASELINE_A

Second NFL market, after receptions (parked -- calibration issue concentrated
in TE, likely from mixing three positions with different reception profiles
into one shared model). Applying that lesson directly here: rushing yards is
scoped to RB ONLY. RBs are the dominant, structurally homogeneous rushing
population (QB scrambles and WR jet sweeps are rare, differently-shaped, and
excluded on purpose -- a disciplined scoping choice, not an oversight).

Builds a STRICT D-1 dataset from nfl_model.sqlite::player_games. Every feature
for a given player-game uses only that player's (and that opponent's) games
with a strictly earlier (season, week) in the same season.

Threshold selection is folded in here (not a separate round-trip script this
time): computes the actual_rushing_yards distribution on 2023 (DEV ONLY, never
touches the 2024 holdout) across real prop-line candidates and reports the
threshold closest to a genuine 50/50 split.

Eligible population: position == RB, >= 3 prior games, AND a rolling recent
(last 3 games) carries-per-game rate >= 8 (ensures a real, CURRENT rushing
role -- committee/backup backs with a handful of lifetime touches are
excluded even if they've accumulated carries historically).

Features
--------
  season_avg_rush_yards, recent3_avg_rush_yards, recent5_avg_rush_yards
  season_avg_carries, recent3_avg_carries
  yards_per_carry        (season rushing_yards / season carries, efficiency)
  opp_rush_yards_allowed_per_game   (as-of: mean rushing yards allowed by this
                                      opponent across ALL eligible RBs who have
                                      faced them so far this season)
  is_home, games_played

Target
------
  over_line = 1 if actual_rushing_yards >= LINE + 0.5 else 0
  LINE is chosen from the diagnostic below (nearest 50/50 on 2023 dev).

Split: 2023 = development, 2024 = one-shot out-of-time holdout.

Read-only on nfl_model.sqlite. Writes only its own baseline.sqlite + manifest.

Run (Render)
------------
python -u nfl_rushing_yards_clean_baseline_a.py 2>&1 | tee /data/nfl_model/nfl_rushing_yards_clean_baseline_a.log
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

SOURCE_DEFAULT = "/data/nfl_model/nfl_model.sqlite"
WORKDIR_DEFAULT = "/data/nfl_model/nfl_rushing_yards_clean_baseline_a_work"
# v2: MIN_PRIOR_CARRIES=5 (lifetime cumulative) was too permissive -- it let in
# committee/third-down backs and backups with a handful of touches, dragging
# the whole distribution down and making the threshold diagnostic pick an
# unrealistically low line (29.5). Real rushing-yards props are only offered
# on backs with genuine, CURRENT volume. Eligibility now requires a rolling
# recent carries-per-game rate, not a lifetime minimum.
MIN_PRIOR_GAMES_FOR_RATE = 3     # games of history before the rate check applies
MIN_RECENT_CARRIES_PER_GAME = 8  # recent3 carries/game -- a real committee-lead/
                                  # bell-cow role, not a committee/passing-down back
THRESHOLD_CANDIDATES = [29.5, 39.5, 49.5, 59.5, 69.5, 79.5, 89.5]

MODEL_COLUMNS = [
    "season_avg_rush_yards", "recent3_avg_rush_yards", "recent5_avg_rush_yards",
    "season_avg_carries", "recent3_avg_carries", "yards_per_carry",
    "opp_rush_yards_allowed_per_game", "is_home", "games_played",
]


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_rows(conn):
    rows = conn.execute("""
        SELECT player_id, player_name, team, opponent, season, week,
               is_home, carries, rushing_yards
        FROM player_games
        WHERE position = 'RB'
        ORDER BY player_id, season, week
    """).fetchall()

    # opponent as-of context: mean rushing yards allowed per game, computed
    # week-by-week across ALL eligible RBs who've faced that opponent so far
    # this season (strictly earlier weeks only).
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

    print("NFL_RUSHING_YARDS_CLEAN_BASELINE_A\n===================================")
    print(f"source={src}\nworkdir={work}")

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    rows = build_rows(conn)
    conn.close()
    print(f"\ntotal eligible RB rows: {len(rows)}")

    # --- threshold diagnostic on 2023 (dev) only ---
    dev_ry = [r["actual_rushing_yards"] for r in rows if r["season"] == 2023]
    print(f"\nTHRESHOLD DIAGNOSTIC (2023 dev only, n={len(dev_ry)})")
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
    out.execute(f"""CREATE TABLE nfl_rushing_yards_baseline (
        player_id TEXT, player_name TEXT, team TEXT, opponent TEXT,
        season INTEGER, week INTEGER, {cols_sql},
        actual_rushing_yards INTEGER, over_line INTEGER
    )""")
    insert_cols = (["player_id", "player_name", "team", "opponent", "season", "week"]
                   + MODEL_COLUMNS + ["actual_rushing_yards", "over_line"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO nfl_rushing_yards_baseline ({', '.join(insert_cols)}) VALUES ({placeholder})"

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
        "script": "NFL_RUSHING_YARDS_CLEAN_BASELINE_A",
        "generated_at_utc": now_utc(),
        "source_db": str(src),
        "source_db_sha256": sha256_file(src),
        "baseline_db": str(base_db),
        "model_columns": MODEL_COLUMNS,
        "eligibility": {"position": "RB",
                        "min_prior_games_for_rate": MIN_PRIOR_GAMES_FOR_RATE,
                        "min_recent_carries_per_game": MIN_RECENT_CARRIES_PER_GAME},
        "strict_d1": "features use only games with strictly earlier (season,week); "
                      "opponent context uses only weeks strictly earlier in that season",
        "target": f"over_line = actual_rushing_yards >= {LINE + 0.5} (line {LINE})",
        "line": LINE,
        "dev_season": 2023, "holdout_season": 2024,
        "total_rows": len(rows), "by_season": by_season,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'season':8s}{'rows':>8s}{'over_' + str(LINE):>10s}")
    for s in sorted(by_season):
        st = by_season[s]
        rate = st["over"] / st["rows"] if st["rows"] else 0
        print(f"{s:<8}{st['rows']:>8}{rate:>10.4f}")

    print(f"\nbaseline db: {base_db}")
    print("Read-only on nfl_model.sqlite. No production state changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
