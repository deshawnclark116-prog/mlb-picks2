#!/usr/bin/env python3
"""
NFL_DEFENSE_INTERCEPTIONS_CLEAN_BASELINE_A

Third individual-defensive-player-prop market. Checked feasibility via a
real-data diagnostic before building (same discipline as sacks): even
among a workload-filtered "ball-hawk" population (recent8 INT rate >= 0.1,
n=3893), the over-rate is only ~9-11% and STABLE across seasons (no drift
like tackles had) -- but that base rate is thin, closer to the RBI/Runs
territory (rare-event target that narrowly failed its gate) than to sacks'
~33%. Building and testing honestly anyway rather than presuming failure --
the whole point of the gate is finding out, not guessing.

Population: CB/S/SAF/FS/DB/LB/MLB/ILB/OLB (secondary + LBs who drop into
coverage -- INTs are a coverage event, not a position-label event; same
recent-role-not-label principle used for sacks).

Target: over_line = 1 if actual_interceptions >= 1 (recorded a pick) -- the
real "anytime interception" prop format, not an arbitrary line search.

Builds a STRICT D-1 dataset from nfl_model.sqlite::player_games. Every
feature uses only that player's (and that opponent's) games with a strictly
earlier (season, week) in the same season.

Eligible population: >= 8 prior games AND a rolling recent8
interceptions-per-game rate >= 0.1 (empirically verified: n=3893, over-rate
9.5%, stable 2022-2025). Longer lookback than sacks (8 vs 5) because INTs
are rarer -- a shorter window is nearly all zeros and uninformative.

Features
--------
  season_avg_int, recent8_avg_int
  season_avg_pass_defended, recent8_avg_pass_defended  (pass-breakup rate --
    a broader "in coverage, near the ball" leading indicator than INTs
    alone, since INTs themselves are too sparse to trust as their own
    short-window feature)
  opp_int_allowed_per_game   (as-of, same pattern as other markets)
  is_home, games_played

Split: 2022-2024 = development, 2025 = one-shot out-of-time holdout.

Read-only on nfl_model.sqlite. Writes only its own baseline.sqlite + manifest.

Run (Render)
------------
python -u nfl_defense_interceptions_clean_baseline_a.py 2>&1 | tee /data/nfl_model/nfl_defense_interceptions_clean_baseline_a.log
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
WORKDIR_DEFAULT = "/data/nfl_model/nfl_defense_interceptions_clean_baseline_a_work"
POSITIONS = ("CB", "S", "SAF", "FS", "DB", "LB", "MLB", "ILB", "OLB")
MIN_PRIOR_GAMES_FOR_RATE = 8
MIN_RECENT_INT_RATE = 0.1
LOOKBACK = 8
LINE = 0.0  # over_line = actual_interceptions >= LINE + 1 (recorded a pick)
DEV_SEASONS = (2022, 2023, 2024)
HOLDOUT_SEASON = 2025

MODEL_COLUMNS = [
    "season_avg_int", "recent8_avg_int",
    "season_avg_pass_defended", "recent8_avg_pass_defended",
    "opp_int_allowed_per_game", "is_home", "games_played",
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
               COALESCE(def_interceptions, 0) AS it,
               COALESCE(def_pass_defended, 0) AS pdf
        FROM player_games
        WHERE position IN ({placeholders}) AND season_type = 'REG'
        ORDER BY player_id, season, week
    """, POSITIONS).fetchall()

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
            it = r[7]
            st = opp_state.setdefault((season, opp), [0, 0])
            st[0] += it
            st[1] += 1

    out = []
    cur_key = None
    group = []

    def flush(group):
        cum_it = cum_pdf = 0.0
        n_prior = 0
        it_hist = deque(maxlen=LOOKBACK)
        pdf_hist = deque(maxlen=LOOKBACK)
        for r in group:
            pid, pname, team, opp, season, week, is_home, it, pdf = r
            r8 = list(it_hist)[-LOOKBACK:]
            recent_int_rate = sum(r8) / len(r8) if r8 else 0.0
            if (n_prior >= MIN_PRIOR_GAMES_FOR_RATE
                    and recent_int_rate >= MIN_RECENT_INT_RATE):
                p8 = list(pdf_hist)[-LOOKBACK:]
                feat = {
                    "season_avg_int": cum_it / n_prior,
                    "recent8_avg_int": recent_int_rate,
                    "season_avg_pass_defended": cum_pdf / n_prior,
                    "recent8_avg_pass_defended": sum(p8) / len(p8) if p8 else 0.0,
                    "opp_int_allowed_per_game": opp_asof.get((pid, season, week)),
                    "is_home": 1.0 if is_home else 0.0,
                    "games_played": n_prior,
                }
                out.append({
                    "player_id": pid, "player_name": pname, "team": team,
                    "opponent": opp, "season": season, "week": week,
                    **feat, "actual_interceptions": it,
                })
            cum_it += it
            cum_pdf += pdf
            it_hist.append(it)
            pdf_hist.append(pdf)
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

    print("NFL_DEFENSE_INTERCEPTIONS_CLEAN_BASELINE_A\n===========================================")
    print(f"source={src}\nworkdir={work}")

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    rows = build_rows(conn)
    conn.close()
    print(f"\ntotal eligible rows: {len(rows)}")

    if base_db.exists():
        base_db.unlink()
    out = sqlite3.connect(str(base_db))
    cols_sql = ", ".join(f"{c} REAL" for c in MODEL_COLUMNS)
    out.execute(f"""CREATE TABLE nfl_defense_interceptions_baseline (
        player_id TEXT, player_name TEXT, team TEXT, opponent TEXT,
        season INTEGER, week INTEGER, {cols_sql},
        actual_interceptions INTEGER, over_line INTEGER
    )""")
    insert_cols = (["player_id", "player_name", "team", "opponent", "season", "week"]
                   + MODEL_COLUMNS + ["actual_interceptions", "over_line"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO nfl_defense_interceptions_baseline ({', '.join(insert_cols)}) VALUES ({placeholder})"

    by_season = {}
    batch = []
    for r in rows:
        r["over_line"] = 1 if r["actual_interceptions"] >= (LINE + 1) else 0
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
        "script": "NFL_DEFENSE_INTERCEPTIONS_CLEAN_BASELINE_A",
        "generated_at_utc": now_utc(),
        "source_db": str(src),
        "source_db_sha256": sha256_file(src),
        "baseline_db": str(base_db),
        "model_columns": MODEL_COLUMNS,
        "eligibility": {"position": list(POSITIONS),
                        "min_prior_games_for_rate": MIN_PRIOR_GAMES_FOR_RATE,
                        "min_recent_int_rate": MIN_RECENT_INT_RATE, "lookback": LOOKBACK},
        "strict_d1": "features use only games with strictly earlier (season,week); "
                      "opponent context uses only weeks strictly earlier in that season",
        "target": f"over_line = actual_interceptions >= {LINE + 1} (recorded a pick)",
        "line": LINE,
        "dev_seasons": list(DEV_SEASONS), "holdout_season": HOLDOUT_SEASON,
        "total_rows": len(rows), "by_season": by_season,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'season':8s}{'rows':>8s}{'over_1':>10s}")
    for s in sorted(by_season):
        st = by_season[s]
        rate = st["over"] / st["rows"] if st["rows"] else 0
        print(f"{s:<8}{st['rows']:>8}{rate:>10.4f}")

    print(f"\nbaseline db: {base_db}")
    print("Read-only on nfl_model.sqlite. No production state changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
