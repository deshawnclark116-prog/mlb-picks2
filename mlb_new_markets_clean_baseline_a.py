#!/usr/bin/env python3
"""
MLB_NEW_MARKETS_CLEAN_BASELINE_A

First real validation pass on 7 new Courtside markets picked as "likely
tractable" (same count-stat-over-known-opportunity shape as the markets
already live: batter_hits, batter_total_bases, pitcher_strikeouts):

  batter:  walks, doubles, singles, strikeouts (batting Ks, not pitcher Ks)
  pitcher: walks, outs recorded, hits allowed

Data source: backfill.py's season_{year}.jsonl (real MLB box scores via
the Stats API -- the same raw source train.py/train_extra.py used
originally). No new data pipeline needed; every field here (bb, 2b, so,
h_allowed, bb_allowed, outs) is already extracted by
backfill.extract_player_lines. Singles is derived (h - 2b - 3b - hr).

Strict D-1: features for a given game use only that player's games with a
strictly earlier calendar date in the same season -- same discipline as
total_bases_clean_baseline_a.py, which this script's feature shape mirrors
directly (season rate, recent5, recent15, exposure, games_played).

Split: 2024 = dev, 2025 = untouched holdout (both real, complete seasons).

Threshold: picked per market from the dev-only distribution, nearest to a
50/50 split -- same method as the NFL preseason baseline.

Eligibility:
  batter:  cumulative PA >= 20 AND games >= 5 (matches total_bases)
  pitcher: cumulative BF >= 60 AND starts >= 5 (proportionally scaled from
           the K model's per-outing BF>=12 floor to a season-cumulative one)

Read-only on season_{year}.jsonl. Writes only its own workdir.

Run:
    python -u mlb_new_markets_clean_baseline_a.py --data-dir /data \
        --workdir /data/mlb_new_markets_clean_baseline_a_work
"""
import argparse
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

DEV_SEASON = 2024
HOLDOUT_SEASON = 2025

BATTER_MARKETS = {
    "batter_walks": {
        "stat_field": "bb", "threshold_candidates": [0.5, 1.5],
    },
    "batter_doubles": {
        "stat_field": "2b", "threshold_candidates": [0.5, 1.5],
    },
    "batter_singles": {
        "stat_field": "singles", "threshold_candidates": [0.5, 1.5],
    },
    "batter_strikeouts": {
        "stat_field": "so", "threshold_candidates": [0.5, 1.5],
    },
}
PITCHER_MARKETS = {
    "pitcher_walks": {
        "stat_field": "bb_allowed", "threshold_candidates": [1.5, 2.5],
    },
    "pitcher_outs": {
        "stat_field": "outs", "threshold_candidates": [14.5, 16.5, 18.5],
    },
    "pitcher_hits_allowed": {
        "stat_field": "h_allowed", "threshold_candidates": [4.5, 5.5, 6.5],
    },
}

BATTER_MIN_PA = 20
BATTER_MIN_GAMES = 5
PITCHER_MIN_BF = 60
PITCHER_MIN_STARTS = 5


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_rows(data_dir, season, row_type):
    path = Path(data_dir) / f"season_{season}.jsonl"
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") == row_type:
                out.append(r)
    return out


def _n(v):
    return v if isinstance(v, (int, float)) and v is not None else 0


def build_batter_rows(rows, stat_field):
    by_player = {}
    for r in rows:
        by_player.setdefault(r["player_id"], []).append(r)

    out = []
    for pid, games in by_player.items():
        games.sort(key=lambda r: r["date"])
        cum_pa = cum_stat = 0
        recent = deque(maxlen=15)
        n_prior = 0
        for g in games:
            pa = _n(g.get("pa"))
            if stat_field == "singles":
                stat = _n(g.get("h")) - _n(g.get("2b")) - _n(g.get("3b")) - _n(g.get("hr"))
            else:
                stat = _n(g.get(stat_field))

            if n_prior >= BATTER_MIN_GAMES and cum_pa >= BATTER_MIN_PA:
                r5 = list(recent)[-5:]
                r15 = list(recent)
                feat = {
                    "season_rate": cum_stat / cum_pa if cum_pa else 0.0,
                    "recent5_avg": sum(r5) / len(r5) if r5 else 0.0,
                    "recent15_avg": sum(r15) / len(r15) if r15 else 0.0,
                    "season_avg_pa": cum_pa / n_prior,
                    "batting_order": g.get("batting_order") or 9,
                    "games_played": n_prior,
                }
                out.append({
                    "player_id": pid, "name": g.get("name"), "team": g.get("team"),
                    "opponent": g.get("opponent"), "game_pk": g.get("game_pk"),
                    "date": g["date"], **feat, "actual": stat,
                })
            cum_pa += pa
            cum_stat += stat
            recent.append(stat)
            n_prior += 1
    return out


def build_pitcher_rows(rows, stat_field):
    by_player = {}
    for r in rows:
        by_player.setdefault(r["player_id"], []).append(r)

    out = []
    for pid, games in by_player.items():
        games.sort(key=lambda r: r["date"])
        cum_bf = cum_stat = 0
        recent = deque(maxlen=15)
        n_prior = 0
        for g in games:
            bf = _n(g.get("bf"))
            stat = _n(g.get(stat_field))

            if n_prior >= PITCHER_MIN_STARTS and cum_bf >= PITCHER_MIN_BF:
                r5 = list(recent)[-5:]
                r15 = list(recent)
                feat = {
                    "season_rate": cum_stat / cum_bf if cum_bf else 0.0,
                    "recent5_avg": sum(r5) / len(r5) if r5 else 0.0,
                    "recent15_avg": sum(r15) / len(r15) if r15 else 0.0,
                    "season_avg_bf": cum_bf / n_prior,
                    "games_played": n_prior,
                }
                out.append({
                    "player_id": pid, "name": g.get("name"), "team": g.get("team"),
                    "opponent": g.get("opponent"), "game_pk": g.get("game_pk"),
                    "date": g["date"], **feat, "actual": stat,
                })
            cum_bf += bf
            cum_stat += stat
            recent.append(stat)
            n_prior += 1
    return out


FEATURE_COLS_BATTER = ["season_rate", "recent5_avg", "recent15_avg",
                        "season_avg_pa", "batting_order", "games_played"]
FEATURE_COLS_PITCHER = ["season_rate", "recent5_avg", "recent15_avg",
                         "season_avg_bf", "games_played"]


def run_market(mkt_key, cfg, dev_rows, hol_rows, feature_cols, workdir):
    print(f"\n{'='*70}\n{mkt_key}\n{'='*70}")
    print(f"dev {DEV_SEASON} rows: {len(dev_rows)}   holdout {HOLDOUT_SEASON} rows: {len(hol_rows)}")
    if len(dev_rows) < 200 or len(hol_rows) < 100:
        print("  TOO FEW ROWS to validate honestly -- skipping market.")
        return {"skipped": True, "n_dev": len(dev_rows), "n_holdout": len(hol_rows)}

    dev_actual = [r["actual"] for r in dev_rows]
    best_t, best_dist = None, 1.0
    print(f"\nTHRESHOLD DIAGNOSTIC (dev only, n={len(dev_actual)})")
    for t in cfg["threshold_candidates"]:
        rate = sum(1 for y in dev_actual if y >= t + 0.5) / len(dev_actual)
        dist = abs(rate - 0.5)
        if dist < best_dist:
            best_dist, best_t = dist, t
        print(f"  >{t:>6.1f}   over_rate={rate:.3f}")
    LINE = best_t
    print(f"  closest-to-50% threshold: over {LINE}")

    base_db = workdir / f"{mkt_key}_baseline.sqlite"
    if base_db.exists():
        base_db.unlink()
    out = sqlite3.connect(str(base_db))
    cols_sql = ", ".join(f"{c} REAL" for c in feature_cols)
    out.execute(f"""CREATE TABLE {mkt_key}_baseline (
        player_id INTEGER, name TEXT, team TEXT, opponent TEXT, game_pk TEXT,
        season INTEGER, date TEXT, {cols_sql}, actual INTEGER, over_line INTEGER
    )""")
    insert_cols = (["player_id", "name", "team", "opponent", "game_pk", "season", "date"]
                   + feature_cols + ["actual", "over_line"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO {mkt_key}_baseline ({', '.join(insert_cols)}) VALUES ({placeholder})"

    by_season_stat = {DEV_SEASON: {"rows": 0, "over": 0}, HOLDOUT_SEASON: {"rows": 0, "over": 0}}
    batch = []
    for season, rows in ((DEV_SEASON, dev_rows), (HOLDOUT_SEASON, hol_rows)):
        for r in rows:
            r["over_line"] = 1 if r["actual"] >= (LINE + 0.5) else 0
            r["season"] = season
            batch.append(tuple(r[c] for c in insert_cols))
            by_season_stat[season]["rows"] += 1
            by_season_stat[season]["over"] += r["over_line"]
    out.executemany(ins, batch)
    out.commit()
    out.close()

    for s in (DEV_SEASON, HOLDOUT_SEASON):
        st = by_season_stat[s]
        rate = st["over"] / st["rows"] if st["rows"] else 0
        print(f"  {s}: {st['rows']} rows, over_{LINE} rate={rate:.4f}")

    return {
        "skipped": False, "line": LINE, "feature_cols": feature_cols,
        "n_dev": len(dev_rows), "n_holdout": len(hol_rows),
        "baseline_db": str(base_db), "by_season": by_season_stat,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--workdir", default="/data/mlb_new_markets_clean_baseline_a_work")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    print("MLB_NEW_MARKETS_CLEAN_BASELINE_A\n=================================")

    dev_batters = load_rows(args.data_dir, DEV_SEASON, "batter")
    hol_batters = load_rows(args.data_dir, HOLDOUT_SEASON, "batter")
    dev_pitchers = load_rows(args.data_dir, DEV_SEASON, "pitcher")
    hol_pitchers = load_rows(args.data_dir, HOLDOUT_SEASON, "pitcher")
    print(f"raw rows: dev batters={len(dev_batters)} pitchers={len(dev_pitchers)}  "
          f"holdout batters={len(hol_batters)} pitchers={len(hol_pitchers)}")

    manifest = {"script": "MLB_NEW_MARKETS_CLEAN_BASELINE_A",
                "generated_at_utc": now_utc(), "dev_season": DEV_SEASON,
                "holdout_season": HOLDOUT_SEASON, "markets": {}}

    for mkt_key, cfg in BATTER_MARKETS.items():
        dev_rows = build_batter_rows(dev_batters, cfg["stat_field"])
        hol_rows = build_batter_rows(hol_batters, cfg["stat_field"])
        manifest["markets"][mkt_key] = run_market(
            mkt_key, cfg, dev_rows, hol_rows, FEATURE_COLS_BATTER, workdir)

    for mkt_key, cfg in PITCHER_MARKETS.items():
        dev_rows = build_pitcher_rows(dev_pitchers, cfg["stat_field"])
        hol_rows = build_pitcher_rows(hol_pitchers, cfg["stat_field"])
        manifest["markets"][mkt_key] = run_market(
            mkt_key, cfg, dev_rows, hol_rows, FEATURE_COLS_PITCHER, workdir)

    (workdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest: {workdir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
