#!/usr/bin/env python3
"""
NFL_RECEPTIONS_CLEAN_BASELINE_A

Rung 2 of the NFL pipeline (mirrors total_bases_clean_baseline_a.py for MLB).

Builds a STRICT D-1 dataset for the NFL receptions market at TWO lines -- 1.5
and 2.5 -- from nfl_model.sqlite::player_games, mirroring how MLB markets carry
multiple thresholds (pitcher_strikeouts, batter_hits: t in (1, 2)). Every
feature for a given player-game uses only that player's (and that opponent's)
games with a strictly earlier (season, week) in the same season. No leakage.
Features don't depend on the line, so both targets are computed once from the
same strict-D-1 walk -- no duplicated feature engineering.

v2: the line was originally set to 0.5 (at least one catch) by analogy with
MLB's "at least one hit," but that turned out to be a ~90% near-lock for WR/TE
-- not a genuine market. nfl_receptions_threshold_diagnostic_a.py measured the
real distribution on 2023 (dev) only: over 2.5 is the threshold closest to a
genuine 50/50 split (overall rate 43.2%); over 1.5 (overall ~64%) is real too
and reflects the more common lower-tier reception prop. Both ship as separate
markets, each gated and validated independently.

Eligible population: pass-catching positions (WR, TE, RB) with >= 3 prior games
of recorded stats in the season.

Features
--------
  season_avg_receptions, recent3_avg_receptions, recent5_avg_receptions
  season_avg_targets, recent3_avg_targets
  catch_rate            (season receptions / season targets, i.e. efficiency)
  opp_receptions_allowed_per_game   (as-of: mean receptions allowed by this
                                      opponent across ALL pass-catchers who
                                      have faced them so far this season)
  is_home, is_wr, is_te, is_rb, games_played

Targets
-------
  over_1_5 = 1 if that game's receptions >= 2 else 0
  over_2_5 = 1 if that game's receptions >= 3 else 0

Split: 2023 = development, 2024 = one-shot out-of-time holdout (2025 is not yet
published in the source data at the time this was built -- see the foundation
script's output). Mirrors the MLB dev/holdout doctrine exactly.

Read-only on nfl_model.sqlite. Writes only its own baseline.sqlite + manifest.

Run (Render)
------------
python -u nfl_receptions_clean_baseline_a.py 2>&1 | tee /data/nfl_model/nfl_receptions_clean_baseline_a.log
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
WORKDIR_DEFAULT = "/data/nfl_model/nfl_receptions_clean_baseline_a_work"
ELIGIBLE_POSITIONS = {"WR", "TE", "RB"}
MIN_PRIOR_GAMES = 3
# v2: corrected from a single 0.5 line (a ~90% near-lock for WR/TE -- not a
# real market) after nfl_receptions_threshold_diagnostic_a.py measured the real
# distribution on 2023 (dev) data. Both lines ship as separate markets.
LINES = [1.5, 2.5]

MODEL_COLUMNS = [
    "season_avg_receptions", "recent3_avg_receptions", "recent5_avg_receptions",
    "season_avg_targets", "recent3_avg_targets", "catch_rate",
    "opp_receptions_allowed_per_game", "is_home", "is_wr", "is_te", "is_rb",
    "games_played",
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
        SELECT player_id, player_name, position, team, opponent, season, week,
               is_home, targets, receptions
        FROM player_games
        WHERE position IN ('WR','TE','RB')
        ORDER BY player_id, season, week
    """).fetchall()

    # --- opponent as-of context: mean receptions allowed per game, computed
    # week-by-week across ALL eligible pass-catchers who have faced that
    # opponent so far this season (strictly earlier weeks only). ---
    by_season_week = {}
    for r in rows:
        by_season_week.setdefault((r[5], r[6]), []).append(r)

    opp_state = {}  # (season, opponent) -> [receptions_allowed_sum, games_count]
    opp_asof = {}   # (player_id, season, week) -> rate or None

    for (season, week) in sorted(by_season_week):
        wk_rows = by_season_week[(season, week)]
        for r in wk_rows:
            pid, opp = r[0], r[4]
            key = (pid, season, week)
            st = opp_state.get((season, opp))
            opp_asof[key] = (st[0] / st[1]) if st and st[1] > 0 else None
        for r in wk_rows:
            opp = r[4]
            recs = r[9] if r[9] is not None else 0
            st = opp_state.setdefault((season, opp), [0, 0])
            st[0] += recs
            st[1] += 1

    # --- per-player strict-D-1 walk ---
    out = []
    cur_key = None
    group = []

    def flush(group):
        cum_rec = cum_tgt = 0
        n_prior = 0
        rec_hist = deque(maxlen=15)
        tgt_hist = deque(maxlen=15)
        for r in group:
            pid, pname, pos, team, opp, season, week, is_home, targets, receptions = r
            if n_prior >= MIN_PRIOR_GAMES:
                r3 = list(rec_hist)[-3:]
                r5 = list(rec_hist)[-5:]
                t3 = list(tgt_hist)[-3:]
                season_avg_rec = cum_rec / n_prior
                season_avg_tgt = cum_tgt / n_prior
                catch_rate = (cum_rec / cum_tgt) if cum_tgt > 0 else 0.0
                feat = {
                    "season_avg_receptions": season_avg_rec,
                    "recent3_avg_receptions": sum(r3) / len(r3) if r3 else 0.0,
                    "recent5_avg_receptions": sum(r5) / len(r5) if r5 else 0.0,
                    "season_avg_targets": season_avg_tgt,
                    "recent3_avg_targets": sum(t3) / len(t3) if t3 else 0.0,
                    "catch_rate": catch_rate,
                    "opp_receptions_allowed_per_game": opp_asof.get((pid, season, week)),
                    "is_home": 1.0 if is_home else 0.0,
                    "is_wr": 1.0 if pos == "WR" else 0.0,
                    "is_te": 1.0 if pos == "TE" else 0.0,
                    "is_rb": 1.0 if pos == "RB" else 0.0,
                    "games_played": n_prior,
                }
                actual_receptions = receptions if receptions is not None else 0
                out.append({
                    "player_id": pid, "player_name": pname, "position": pos,
                    "team": team, "opponent": opp, "season": season, "week": week,
                    **feat,
                    "actual_receptions": actual_receptions,
                    "over_1_5": 1 if actual_receptions >= 2 else 0,
                    "over_2_5": 1 if actual_receptions >= 3 else 0,
                })
            tgt = targets if targets is not None else 0
            rec = receptions if receptions is not None else 0
            cum_rec += rec
            cum_tgt += tgt
            rec_hist.append(rec)
            tgt_hist.append(tgt)
            n_prior += 1
        return

    for r in rows:
        key = (r[0], r[5])  # (player_id, season)
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

    print("NFL_RECEPTIONS_CLEAN_BASELINE_A\n================================")
    print(f"source={src}\nworkdir={work}")

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    rows = build_rows(conn)
    conn.close()

    if base_db.exists():
        base_db.unlink()
    out = sqlite3.connect(str(base_db))
    cols_sql = ", ".join(f"{c} REAL" for c in MODEL_COLUMNS)
    out.execute(f"""CREATE TABLE nfl_receptions_baseline (
        player_id TEXT, player_name TEXT, position TEXT, team TEXT, opponent TEXT,
        season INTEGER, week INTEGER, {cols_sql},
        actual_receptions INTEGER, over_1_5 INTEGER, over_2_5 INTEGER
    )""")
    insert_cols = (["player_id", "player_name", "position", "team", "opponent", "season", "week"]
                   + MODEL_COLUMNS + ["actual_receptions", "over_1_5", "over_2_5"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO nfl_receptions_baseline ({', '.join(insert_cols)}) VALUES ({placeholder})"

    by_season = {}
    batch = []
    for r in rows:
        batch.append(tuple(r[c] for c in insert_cols))
        st = by_season.setdefault(r["season"], {"rows": 0, "over_1_5": 0, "over_2_5": 0, "by_pos": {}})
        st["rows"] += 1
        st["over_1_5"] += r["over_1_5"]
        st["over_2_5"] += r["over_2_5"]
        p = st["by_pos"].setdefault(r["position"], {"rows": 0, "over_1_5": 0, "over_2_5": 0})
        p["rows"] += 1
        p["over_1_5"] += r["over_1_5"]
        p["over_2_5"] += r["over_2_5"]
        if len(batch) >= 5000:
            out.executemany(ins, batch); batch = []
    if batch:
        out.executemany(ins, batch)
    out.commit()
    out.close()

    manifest = {
        "script": "NFL_RECEPTIONS_CLEAN_BASELINE_A",
        "generated_at_utc": now_utc(),
        "source_db": str(src),
        "source_db_sha256": sha256_file(src),
        "baseline_db": str(base_db),
        "model_columns": MODEL_COLUMNS,
        "eligibility": {"positions": sorted(ELIGIBLE_POSITIONS), "min_prior_games": MIN_PRIOR_GAMES},
        "strict_d1": "features use only games with strictly earlier (season,week); "
                      "opponent context uses only weeks strictly earlier in that season",
        "targets": {"over_1_5": "actual_receptions >= 2", "over_2_5": "actual_receptions >= 3"},
        "lines": LINES,
        "dev_season": 2023, "holdout_season": 2024,
        "total_rows": len(rows), "by_season": by_season,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\ntotal eligible rows: {len(rows)}")
    print(f"{'season':8s}{'rows':>8s}{'over_1.5':>10s}{'over_2.5':>10s}")
    for s in sorted(by_season):
        st = by_season[s]
        r15 = st["over_1_5"] / st["rows"] if st["rows"] else 0
        r25 = st["over_2_5"] / st["rows"] if st["rows"] else 0
        print(f"{s:<8}{st['rows']:>8}{r15:>10.4f}{r25:>10.4f}")
        for pos in sorted(st["by_pos"]):
            p = st["by_pos"][pos]
            p15 = p["over_1_5"] / p["rows"] if p["rows"] else 0
            p25 = p["over_2_5"] / p["rows"] if p["rows"] else 0
            print(f"    {pos:6s}{p['rows']:>8}{p15:>10.4f}{p25:>10.4f}")

    print(f"\nbaseline db: {base_db}")
    print("Read-only on nfl_model.sqlite. No production state changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
