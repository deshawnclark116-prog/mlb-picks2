#!/usr/bin/env python3
"""
TOTAL_BASES_CLEAN_BASELINE_A

Rung 2 of the total-bases promotion pipeline (mirrors pitcher_k_clean_baseline_a).

Purpose
-------
Build a STRICT D-1 total-bases baseline dataset from hr_model.sqlite::batter_games,
reproducing the champion's 11 model features honestly (no same-day leak), plus the
realized target. This is the clean, leak-free dataset every later rung (formal
gate, parity) scores against.

The champion (api.py::batter_feature_row + _batter_feat_for) builds features from
season-to-date gameLog with NO strict D-1 cutoff -- earlier same-day games can leak
in. This builder fixes that: for each batter-game, features use ONLY games with a
strictly earlier calendar date in the same season.

Model feature columns reproduced (order from /data/models/batter_total_bases_columns.json):
    season_avg, tb_per_pa, rbi_per_pa, runs_per_pa, hr_rate, bb_rate, so_rate,
    recent5_target, recent15_target, batting_order, games_played

Target:
    actual_tb  = hits + doubles + 2*triples + 3*home_runs   (that game)
    over_1_5   = 1 if actual_tb >= 2 else 0

Eligibility gate (matches api.py::batter_feature_row): prior at_bats >= 20 AND
prior games >= 5. batting_order = that game's lineup_spot (COALESCE 9), matching
production's base["batting_order"] = spot.

This script is READ-ONLY on hr_model.sqlite. It writes ONLY to its own work dir.
It changes no production code, models, or predictions.

Run (Render)
------------
python -u total_bases_clean_baseline_a.py 2>&1 | tee /data/hr_model/total_bases_clean_baseline_a.log

Run (local test against a synthetic db)
---------------------------------------
python -u total_bases_clean_baseline_a.py --source /path/to/test.sqlite --workdir /tmp/tb_work

Output
------
<workdir>/baseline.sqlite            table tb_baseline
<workdir>/manifest.json              freeze manifest (source hash, criteria, counts)
<report json/txt next to log>
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

MODEL_COLUMNS = [
    "season_avg", "tb_per_pa", "rbi_per_pa", "runs_per_pa", "hr_rate",
    "bb_rate", "so_rate", "recent5_target", "recent15_target",
    "batting_order", "games_played",
]

MIN_PRIOR_AB = 20
MIN_PRIOR_GAMES = 5

DEFAULT_SOURCE_CANDIDATES = [
    "/data/hr_model/hr_model.sqlite",
]
DEFAULT_WORKDIR = "/data/hr_model/total_bases_clean_baseline_a_work"


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_source(explicit):
    if explicit:
        return Path(explicit)
    for c in DEFAULT_SOURCE_CANDIDATES:
        if Path(c).exists():
            return Path(c)
    hits = sorted(Path("/data").rglob("hr_model.sqlite")) if Path("/data").exists() else []
    if hits:
        return hits[0]
    raise SystemExit("Could not find hr_model.sqlite; pass --source PATH")


def _num(x):
    return x if isinstance(x, (int, float)) and x is not None else 0


def build_rows(conn):
    """Yield one dict per eligible batter-game, strict-D-1 features + target."""
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(batter_games)")
    cols = {r[1] for r in cur.fetchall()}
    required = {"game_id", "game_date", "batter_id", "lineup_spot", "plate_appearances",
                "at_bats", "hits", "doubles", "triples", "home_runs", "rbi",
                "walks", "strikeouts"}
    missing = required - cols
    if missing:
        raise SystemExit(f"batter_games missing columns: {sorted(missing)}")

    # Pull all batter-seasons, ordered so we can walk each forward by date.
    cur.execute("""
        SELECT batter_id, substr(game_date,1,4) AS season, game_date, game_id,
               lineup_spot, plate_appearances, at_bats, hits, doubles, triples,
               home_runs, rbi, walks, strikeouts
        FROM batter_games
        WHERE at_bats IS NOT NULL
        ORDER BY batter_id, season, game_date, game_id
    """)

    cur_key = None
    games = []

    def flush(group):
        yield_rows = []
        # running aggregates through games with strictly-earlier DATE
        cum_pa = cum_ab = cum_h = cum_hr = cum_bb = cum_so = cum_rbi = cum_runs = cum_tb = 0
        recent_tb = deque(maxlen=15)
        n_prior = 0
        i = 0
        for idx, g in enumerate(group):
            gdate = g["game_date"]
            # advance history to include only strictly-earlier dates
            while i < len(group) and group[i]["game_date"] < gdate:
                h = group[i]
                tb = h["hits"] + h["doubles"] + 2 * h["triples"] + 3 * h["home_runs"]
                cum_pa += _num(h["pa"]); cum_ab += _num(h["ab"]); cum_h += _num(h["hits"])
                cum_hr += _num(h["home_runs"]); cum_bb += _num(h["walks"])
                cum_so += _num(h["strikeouts"]); cum_rbi += _num(h["rbi"])
                cum_runs += _num(h.get("runs", 0)); cum_tb += tb
                recent_tb.append(tb)
                n_prior += 1
                i += 1

            if n_prior < MIN_PRIOR_GAMES or cum_ab < MIN_PRIOR_AB:
                continue

            r5 = list(recent_tb)[-5:]
            r15 = list(recent_tb)
            feat = {
                "season_avg": cum_h / cum_ab if cum_ab else 0.0,
                "tb_per_pa": cum_tb / cum_pa if cum_pa else 0.0,
                "rbi_per_pa": cum_rbi / cum_pa if cum_pa else 0.0,
                "runs_per_pa": cum_runs / cum_pa if cum_pa else 0.0,
                "hr_rate": cum_hr / cum_pa if cum_pa else 0.0,
                "bb_rate": cum_bb / cum_pa if cum_pa else 0.0,
                "so_rate": cum_so / cum_pa if cum_pa else 0.0,
                "recent5_target": sum(r5) / len(r5) if r5 else 0.0,
                "recent15_target": sum(r15) / len(r15) if r15 else 0.0,
                "batting_order": g["lineup_spot"] if g["lineup_spot"] is not None else 9,
                "games_played": n_prior,
            }
            actual_tb = g["hits"] + g["doubles"] + 2 * g["triples"] + 3 * g["home_runs"]
            spot = g["lineup_spot"]
            yield_rows.append({
                **feat,
                "batter_id": g["batter_id"],
                "season": g["season"],
                "game_date": gdate,
                "game_id": g["game_id"],
                "lineup_spot": spot,
                "is_starter": 1 if (spot is not None and 1 <= spot <= 9) else 0,
                "actual_tb": actual_tb,
                "over_1_5": 1 if actual_tb >= 2 else 0,
            })
        return yield_rows

    def norm(row):
        return {
            "batter_id": row[0], "season": row[1], "game_date": row[2], "game_id": row[3],
            "lineup_spot": row[4], "pa": row[5], "ab": row[6], "hits": row[7],
            "doubles": row[8], "triples": row[9], "home_runs": row[10], "rbi": row[11],
            "walks": row[12], "strikeouts": row[13], "runs": 0,
        }

    for row in cur:
        key = (row[0], row[1])
        if key != cur_key:
            if games:
                for out in flush(games):
                    yield out
            games = []
            cur_key = key
        games.append(norm(row))
    if games:
        for out in flush(games):
            yield out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="path to hr_model.sqlite")
    ap.add_argument("--workdir", default=DEFAULT_WORKDIR)
    args = ap.parse_args()

    src = discover_source(args.source)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    base_db = workdir / "baseline.sqlite"

    print("TOTAL_BASES_CLEAN_BASELINE_A", flush=True)
    print("============================", flush=True)
    print(f"source={src}", flush=True)
    print(f"workdir={workdir}", flush=True)

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)

    if base_db.exists():
        base_db.unlink()
    out = sqlite3.connect(str(base_db))
    out.execute("PRAGMA journal_mode=WAL")
    cols_sql = ", ".join(f"{c} REAL" for c in MODEL_COLUMNS)
    out.execute(f"""
        CREATE TABLE tb_baseline (
            batter_id INTEGER, season TEXT, game_date TEXT, game_id TEXT,
            lineup_spot INTEGER, is_starter INTEGER,
            {cols_sql},
            actual_tb INTEGER, over_1_5 INTEGER
        )
    """)

    insert_cols = (["batter_id", "season", "game_date", "game_id", "lineup_spot", "is_starter"]
                   + MODEL_COLUMNS + ["actual_tb", "over_1_5"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO tb_baseline ({', '.join(insert_cols)}) VALUES ({placeholder})"

    by_season = {}
    n = 0
    batch = []
    for r in build_rows(conn):
        batch.append(tuple(r[c] for c in insert_cols))
        s = r["season"]
        st = by_season.setdefault(s, {"rows": 0, "over": 0, "starters": 0, "starter_over": 0})
        st["rows"] += 1
        st["over"] += r["over_1_5"]
        if r["is_starter"]:
            st["starters"] += 1
            st["starter_over"] += r["over_1_5"]
        n += 1
        if len(batch) >= 5000:
            out.executemany(ins, batch); batch = []
    if batch:
        out.executemany(ins, batch)
    out.commit()
    conn.close()

    manifest = {
        "script": "TOTAL_BASES_CLEAN_BASELINE_A",
        "generated_at_utc": now_utc(),
        "source_db": str(src),
        "source_db_sha256": sha256_file(src),
        "baseline_db": str(base_db),
        "model_columns": MODEL_COLUMNS,
        "eligibility": {"min_prior_ab": MIN_PRIOR_AB, "min_prior_games": MIN_PRIOR_GAMES},
        "strict_d1": "features use only games with strictly earlier calendar date (same season)",
        "target": "actual_tb = hits+doubles+2*triples+3*home_runs; over_1_5 = actual_tb>=2",
        "total_rows": n,
        "by_season": by_season,
    }
    (workdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    out_dir = Path("/data/hr_model") if Path("/data/hr_model").exists() else workdir
    (out_dir / "total_bases_clean_baseline_a_report.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nBASELINE BUILT")
    print("--------------")
    print(f"total eligible rows: {n}")
    print(f"{'season':8s}{'rows':>9s}{'over1.5':>10s}{'starters':>10s}{'startOver':>11s}")
    for s in sorted(by_season):
        st = by_season[s]
        orate = st["over"] / st["rows"] if st["rows"] else 0
        srate = st["starter_over"] / st["starters"] if st["starters"] else 0
        print(f"{s:8s}{st['rows']:9d}{orate:10.4f}{st['starters']:10d}{srate:11.4f}")
    print(f"\nbaseline db: {base_db}")
    print("READ-ONLY on hr_model.sqlite. No production state changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
