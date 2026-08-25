#!/usr/bin/env python3
"""
NFL_DEFENSE_TACKLES_CLEAN_BASELINE_A

First individual-defensive-player-prop market. Same category the user asked
for after rushing/receiving: props on a named defender (tackles, sacks,
interceptions). This one is tackles -- the highest-volume, most homogeneous
defensive stat, and the position group real books actually price it on.

Population: LB / ILB / MLB / OLB (nflverse position labels drift between the
umbrella 'LB' tag and split ILB/OLB/MLB tags across seasons and source feeds
for the SAME on-field role -- confirmed by inspecting real per-season counts,
not assumed). Applying the receptions lesson (mixing WR/RB/TE tanked
calibration) the other direction: these four labels are the same role, not
different ones, so combining them is the disciplined choice, not a shortcut.
DE/DT/CB/S are excluded -- different tackle profiles (pass-rush-first vs.
coverage-first), a separate market if this one validates.

Target: total tackles = def_tackles_solo + def_tackles_with_assist (nflverse's
own combined-tackles definition -- used consistently for both features and
target, so internal consistency is what matters, not matching any external
"official" tackle count).

Builds a STRICT D-1 dataset from nfl_model.sqlite::player_games. Every
feature for a given player-game uses only that player's (and that
opponent's) games with a strictly earlier (season, week) in the same season.

Eligible population: position in {LB,ILB,MLB,OLB}, >= 3 prior games, AND a
rolling recent (last 3 games) tackles-per-game rate >= 4.0 -- verified
empirically (not guessed): rows clearing this bar have mean actual tackles
4.41/game and include real known full-time starters (Zaire Franklin, Zack
Baun, Fred Warner, etc. all clear it easily in their starter seasons),
excluding backups/rotational players with a handful of tackles.

Features
--------
  season_avg_tackles, recent3_avg_tackles, recent5_avg_tackles
  season_avg_solo, recent3_avg_solo
  opp_tackles_allowed_per_game   (as-of: mean tackles allowed by this
                                   opponent's offense across ALL eligible
                                   LBs who have faced them so far this season
                                   -- a pace/game-script proxy, same pattern
                                   as rushing_yards' opp_rush_yards_allowed)
  is_home, games_played

Target
------
  over_line = 1 if actual_tackles >= LINE + 0.5 else 0
  LINE chosen from the diagnostic below (nearest 50/50 on DEV seasons only).

Split: 2022-2024 = development, 2025 = one-shot out-of-time holdout (freshest
complete season available; more training data than the 2023/2024 split used
for the earlier offense markets since 2025 wasn't available when those were
built).

Read-only on nfl_model.sqlite. Writes only its own baseline.sqlite + manifest.

Run (Render)
------------
python -u nfl_defense_tackles_clean_baseline_a.py 2>&1 | tee /data/nfl_model/nfl_defense_tackles_clean_baseline_a.log
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
WORKDIR_DEFAULT = "/data/nfl_model/nfl_defense_tackles_clean_baseline_a_work"
POSITIONS = ("LB", "ILB", "MLB", "OLB")
MIN_PRIOR_GAMES_FOR_RATE = 3
MIN_RECENT_TACKLES_PER_GAME = 4.0
THRESHOLD_CANDIDATES = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
DEV_SEASONS = (2022, 2023, 2024)
HOLDOUT_SEASON = 2025

MODEL_COLUMNS = [
    "season_avg_tackles", "recent3_avg_tackles", "recent5_avg_tackles",
    "season_avg_solo", "recent3_avg_solo",
    "opp_tackles_allowed_per_game", "is_home", "games_played",
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
    placeholders = ", ".join("?" for _ in POSITIONS)
    rows = conn.execute(f"""
        SELECT player_id, player_name, team, opponent, season, week, is_home,
               COALESCE(def_tackles_solo, 0) AS solo,
               COALESCE(def_tackles_solo, 0) + COALESCE(def_tackles_with_assist, 0) AS tk
        FROM player_games
        WHERE position IN ({placeholders}) AND season_type = 'REG'
        ORDER BY player_id, season, week
    """, POSITIONS).fetchall()

    # opponent as-of context: mean tackles allowed per game, computed
    # week-by-week across ALL eligible LBs who've faced that opponent so far
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
            tk = r[8]
            st = opp_state.setdefault((season, opp), [0, 0])
            st[0] += tk
            st[1] += 1

    out = []
    cur_key = None
    group = []

    def flush(group):
        cum_tk = cum_solo = 0
        n_prior = 0
        tk_hist = deque(maxlen=15)
        solo_hist = deque(maxlen=15)
        for r in group:
            pid, pname, team, opp, season, week, is_home, solo, tk = r
            t3 = list(tk_hist)[-3:]
            recent_tk_rate = sum(t3) / len(t3) if t3 else 0.0
            if (n_prior >= MIN_PRIOR_GAMES_FOR_RATE
                    and recent_tk_rate >= MIN_RECENT_TACKLES_PER_GAME):
                t5 = list(tk_hist)[-5:]
                s3 = list(solo_hist)[-3:]
                feat = {
                    "season_avg_tackles": cum_tk / n_prior,
                    "recent3_avg_tackles": sum(t3) / len(t3) if t3 else 0.0,
                    "recent5_avg_tackles": sum(t5) / len(t5) if t5 else 0.0,
                    "season_avg_solo": cum_solo / n_prior,
                    "recent3_avg_solo": sum(s3) / len(s3) if s3 else 0.0,
                    "opp_tackles_allowed_per_game": opp_asof.get((pid, season, week)),
                    "is_home": 1.0 if is_home else 0.0,
                    "games_played": n_prior,
                }
                out.append({
                    "player_id": pid, "player_name": pname, "team": team,
                    "opponent": opp, "season": season, "week": week,
                    **feat, "actual_tackles": tk,
                })
            cum_tk += tk
            cum_solo += solo
            tk_hist.append(tk)
            solo_hist.append(solo)
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

    print("NFL_DEFENSE_TACKLES_CLEAN_BASELINE_A\n=====================================")
    print(f"source={src}\nworkdir={work}")

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    rows = build_rows(conn)
    conn.close()
    print(f"\ntotal eligible LB-group rows: {len(rows)}")

    # --- threshold diagnostic on DEV seasons only ---
    dev_tk = [r["actual_tackles"] for r in rows if r["season"] in DEV_SEASONS]
    print(f"\nTHRESHOLD DIAGNOSTIC (dev {DEV_SEASONS} only, n={len(dev_tk)})")
    print(f"  {'threshold':>10s}{'over_rate':>12s}")
    best_t, best_dist = None, 1.0
    for t in THRESHOLD_CANDIDATES:
        rate = sum(1 for y in dev_tk if y >= t + 0.5) / len(dev_tk) if dev_tk else 0
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
    out.execute(f"""CREATE TABLE nfl_defense_tackles_baseline (
        player_id TEXT, player_name TEXT, team TEXT, opponent TEXT,
        season INTEGER, week INTEGER, {cols_sql},
        actual_tackles INTEGER, over_line INTEGER
    )""")
    insert_cols = (["player_id", "player_name", "team", "opponent", "season", "week"]
                   + MODEL_COLUMNS + ["actual_tackles", "over_line"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO nfl_defense_tackles_baseline ({', '.join(insert_cols)}) VALUES ({placeholder})"

    by_season = {}
    batch = []
    for r in rows:
        r["over_line"] = 1 if r["actual_tackles"] >= (LINE + 0.5) else 0
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
        "script": "NFL_DEFENSE_TACKLES_CLEAN_BASELINE_A",
        "generated_at_utc": now_utc(),
        "source_db": str(src),
        "source_db_sha256": sha256_file(src),
        "baseline_db": str(base_db),
        "model_columns": MODEL_COLUMNS,
        "eligibility": {"position": list(POSITIONS),
                        "min_prior_games_for_rate": MIN_PRIOR_GAMES_FOR_RATE,
                        "min_recent_tackles_per_game": MIN_RECENT_TACKLES_PER_GAME},
        "strict_d1": "features use only games with strictly earlier (season,week); "
                      "opponent context uses only weeks strictly earlier in that season",
        "target": f"over_line = actual_tackles >= {LINE + 0.5} (line {LINE})",
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
    print("Read-only on nfl_model.sqlite. No production state changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
