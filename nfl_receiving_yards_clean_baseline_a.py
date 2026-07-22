#!/usr/bin/env python3
"""
NFL_RECEIVING_YARDS_CLEAN_BASELINE_A

Third NFL market. Built with every scoping lesson the first two markets
paid for, applied from day one:

  WR ONLY. Rushing succeeded by scoping to the structurally homogeneous
  position (RB) from the start; receptions failed by mixing RB/TE/WR into
  one model, and tonight's power-adjusted slice analysis located its
  decisive miscalibration specifically in the RB slice (p=0.0002) -- the
  positions really do behave differently. TE receiving props are real but
  TE usage is structurally distinct (blocking snaps, red-zone roles); a
  TE market can be its own scoped model later if WR clears.

  Eligibility = CURRENT role, not lifetime accumulation (rushing's v1->v3
  lesson): >= 3 prior games this season AND rolling recent (last 3 games)
  targets-per-game >= 5 -- the target share a book would actually post a
  receiving-yards line on, not a depth-chart WR4 with garbage-time catches.

Builds a STRICT D-1 dataset from nfl_model.sqlite::player_games. Every
feature for a player-game uses only that player's (and that opponent's)
games with a strictly earlier (season, week) in the same season.

Threshold selection folded in (rushing's pattern): the actual receiving
yards distribution on 2023 DEV ONLY picks the line closest to 50/50.

Features
--------
  season_avg_rec_yards, recent3_avg_rec_yards, recent5_avg_rec_yards
  season_avg_targets, recent3_avg_targets
  yards_per_target       (season receiving_yards / season targets)
  catch_rate             (season receptions / season targets)
  opp_rec_yards_allowed_per_game  (as-of: mean receiving yards allowed by
                                   this opponent across eligible WRs faced
                                   so far this season, earlier weeks only)
  is_home, games_played

Target
------
  over_line = 1 if actual_receiving_yards >= LINE + 0.5 else 0

Split: 2023 = development, 2024 = one-shot out-of-time holdout.

Read-only on nfl_model.sqlite. Writes only its own baseline.sqlite + manifest.

Run
---
python -u nfl_receiving_yards_clean_baseline_a.py \
    --source nfl_models/nfl_model.sqlite \
    --workdir nfl_models/nfl_receiving_yards_clean_baseline_a_work
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
WORKDIR_DEFAULT = "/data/nfl_model/nfl_receiving_yards_clean_baseline_a_work"

MIN_PRIOR_GAMES_FOR_RATE = 3
MIN_RECENT_TARGETS_PER_GAME = 5
THRESHOLD_CANDIDATES = [29.5, 39.5, 49.5, 59.5, 69.5, 79.5]

MODEL_COLUMNS = [
    "season_avg_rec_yards", "recent3_avg_rec_yards", "recent5_avg_rec_yards",
    "season_avg_targets", "recent3_avg_targets", "yards_per_target",
    "catch_rate", "opp_rec_yards_allowed_per_game", "is_home", "games_played",
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
               is_home, targets, receptions, receiving_yards
        FROM player_games
        WHERE position = 'WR'
        ORDER BY player_id, season, week
    """).fetchall()

    # opponent as-of context: mean receiving yards allowed per game across
    # all WRs who've faced that opponent in strictly earlier weeks this season
    by_season_week = {}
    for r in rows:
        by_season_week.setdefault((r[4], r[5]), []).append(r)

    opp_state = {}
    opp_asof = {}
    for (season, week) in sorted(by_season_week):
        wk_rows = by_season_week[(season, week)]
        for r in wk_rows:
            pid, opp = r[0], r[3]
            st = opp_state.get((season, opp))
            opp_asof[(pid, season, week)] = (st[0] / st[1]) if st and st[1] > 0 else None
        for r in wk_rows:
            opp = r[3]
            ry = r[9] if r[9] is not None else 0
            st = opp_state.setdefault((season, opp), [0, 0])
            st[0] += ry
            st[1] += 1

    out = []
    cur_key = None
    group = []

    def flush(group):
        cum_ry = cum_targets = cum_rec = 0
        n_prior = 0
        ry_hist = deque(maxlen=15)
        tgt_hist = deque(maxlen=15)
        for r in group:
            pid, pname, team, opp, season, week, is_home, targets, receptions, rec_yards = r
            t3 = list(tgt_hist)[-3:]
            recent_target_rate = sum(t3) / len(t3) if t3 else 0.0
            if (n_prior >= MIN_PRIOR_GAMES_FOR_RATE
                    and recent_target_rate >= MIN_RECENT_TARGETS_PER_GAME):
                r3 = list(ry_hist)[-3:]
                r5 = list(ry_hist)[-5:]
                feat = {
                    "season_avg_rec_yards": cum_ry / n_prior,
                    "recent3_avg_rec_yards": sum(r3) / len(r3) if r3 else 0.0,
                    "recent5_avg_rec_yards": sum(r5) / len(r5) if r5 else 0.0,
                    "season_avg_targets": cum_targets / n_prior,
                    "recent3_avg_targets": recent_target_rate,
                    "yards_per_target": (cum_ry / cum_targets) if cum_targets > 0 else 0.0,
                    "catch_rate": (cum_rec / cum_targets) if cum_targets > 0 else 0.0,
                    "opp_rec_yards_allowed_per_game": opp_asof.get((pid, season, week)),
                    "is_home": 1.0 if is_home else 0.0,
                    "games_played": n_prior,
                }
                actual = rec_yards if rec_yards is not None else 0
                out.append({
                    "player_id": pid, "player_name": pname, "team": team,
                    "opponent": opp, "season": season, "week": week,
                    **feat, "actual_receiving_yards": actual,
                })
            cum_ry += rec_yards if rec_yards is not None else 0
            cum_targets += targets if targets is not None else 0
            cum_rec += receptions if receptions is not None else 0
            ry_hist.append(rec_yards if rec_yards is not None else 0)
            tgt_hist.append(targets if targets is not None else 0)
            n_prior += 1

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

    print("NFL_RECEIVING_YARDS_CLEAN_BASELINE_A\n====================================")
    print(f"source={src}\nworkdir={work}")

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    rows = build_rows(conn)
    conn.close()
    print(f"\ntotal eligible WR rows: {len(rows)}")

    dev_ry = [r["actual_receiving_yards"] for r in rows if r["season"] == 2023]
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
    out.execute(f"""CREATE TABLE nfl_receiving_yards_baseline (
        player_id TEXT, player_name TEXT, team TEXT, opponent TEXT,
        season INTEGER, week INTEGER, {cols_sql},
        actual_receiving_yards INTEGER, over_line INTEGER
    )""")
    insert_cols = (["player_id", "player_name", "team", "opponent", "season", "week"]
                   + MODEL_COLUMNS + ["actual_receiving_yards", "over_line"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = (f"INSERT INTO nfl_receiving_yards_baseline ({', '.join(insert_cols)}) "
           f"VALUES ({placeholder})")

    by_season = {}
    batch = []
    for r in rows:
        r["over_line"] = 1 if r["actual_receiving_yards"] >= (LINE + 0.5) else 0
        batch.append(tuple(r[c] for c in insert_cols))
        st = by_season.setdefault(r["season"], {"rows": 0, "over": 0})
        st["rows"] += 1
        st["over"] += r["over_line"]
    out.executemany(ins, batch)
    out.commit()
    out.close()

    manifest = {
        "script": "NFL_RECEIVING_YARDS_CLEAN_BASELINE_A",
        "generated_at_utc": now_utc(),
        "source_db": str(src),
        "source_db_sha256": sha256_file(src),
        "baseline_db": str(base_db),
        "model_columns": MODEL_COLUMNS,
        "eligibility": {"position": "WR",
                        "min_prior_games_for_rate": MIN_PRIOR_GAMES_FOR_RATE,
                        "min_recent_targets_per_game": MIN_RECENT_TARGETS_PER_GAME},
        "strict_d1": "features use only games with strictly earlier (season,week); "
                      "opponent context uses only weeks strictly earlier in that season",
        "target": f"over_line = actual_receiving_yards >= {LINE + 0.5} (line {LINE})",
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
