#!/usr/bin/env python3
"""
NFL_PRESEASON_CLEAN_BASELINE_A

First real modeling pass on the preseason data pulled by
nfl_preseason_data_foundation_a.py + position-enriched by
nfl_preseason_position_enrich_a.py. Both rushing_yards and receiving_yards
in one script (unlike the regular-season pair of separate scripts) since
each market's eligible population here is small enough that the usual
per-market file split would mostly be duplicated boilerplate.

Why this can't just reuse the regular-season design
-----------------------------------------------------
The regular-season baseline requires 3+ prior games *within the same
season* plus a recent-rate floor tuned to real workhorse volume (12+
carries/game). Verified live against this data: only ~35% of
player-seasons ever reach 3 games in a single preseason (max 4 games
exist: HOF + 3 preseason weeks), and preseason workloads are far lower
across the board (RB carries/game: median 5, p95 12 -- what the regular
season used as ITS floor). Copying that design would leave the board
empty almost every week.

Adaptations, each verified against the real data before being chosen:
  - Eligibility is CROSS-SEASON: a player's prior games regardless of
    which preseason they came from, ordered strictly by game_date. Still
    strict D-1 (only games before the one being scored), just not
    reset by season boundary.
  - Recent-rate floors recalibrated from the real distribution instead of
    reused: RB carries/game >= 4 (WR targets/game >= 2), both roughly
    real-population medians, not lifted from a differently-scaled market.
  - Position comes from a live-current-roster join (ESPN box scores don't
    carry it) -- coverage is necessarily incomplete for players no longer
    in the league (25.8% matched in 2021 vs 99.6% in 2026). Unmatched
    rows have position=NULL and are simply excluded, not mislabeled.

Split: dev = seasons 2021-2024, holdout = 2025 (untouched, most complete
labeled season). 2026 is reserved as the live serving target, not part
of validation.

Read-only on the source db. Writes only its own baseline.sqlite + manifest.

Run:
    python -u nfl_preseason_clean_baseline_a.py \
        --source nfl_models/nfl_preseason.sqlite \
        --workdir nfl_models/nfl_preseason_clean_baseline_a_work
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

SOURCE_DEFAULT = "nfl_models/nfl_preseason.sqlite"
WORKDIR_DEFAULT = "nfl_models/nfl_preseason_clean_baseline_a_work"

DEV_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUT_SEASON = 2025
LIVE_SEASON = 2026  # reserved, not scored here

MARKETS = {
    "preseason_rushing_yards": {
        "position": "RB",
        "stat_field": "rushing_yards",
        "rate_field": "carries",
        "min_recent_rate": 4,       # real median carries/game for RB rows (see module docstring)
        "threshold_candidates": [9.5, 14.5, 19.5, 24.5, 29.5, 34.5, 39.5],
        "columns": ["career_avg_yards", "recent3_avg_yards", "recent5_avg_yards",
                    "career_avg_rate", "recent3_avg_rate", "yards_per_unit",
                    "is_home", "games_played"],
    },
    "preseason_receiving_yards": {
        "position": "WR",
        "stat_field": "receiving_yards",
        "rate_field": "targets",
        "min_recent_rate": 2,       # real median targets/game for WR rows
        "threshold_candidates": [7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5],
        "columns": ["career_avg_yards", "recent3_avg_yards", "recent5_avg_yards",
                    "career_avg_rate", "recent3_avg_rate", "yards_per_unit",
                    "is_home", "games_played"],
    },
}
MIN_PRIOR_GAMES = 3


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_rows(conn, mkt_key, cfg):
    stat_field, rate_field = cfg["stat_field"], cfg["rate_field"]
    rows = conn.execute(f"""
        SELECT pg.athlete_id, pg.player_name, pg.team, pg.opponent, pg.is_home,
               pg.season, pg.week, g.game_date, pg.{rate_field}, pg.{stat_field}
        FROM player_games pg
        JOIN games g ON g.game_id = pg.game_id
        WHERE pg.position = ?
        ORDER BY pg.athlete_id, g.game_date
    """, (cfg["position"],)).fetchall()

    out = []
    cur_id, group = None, []

    def flush(group):
        cum_stat = cum_rate = 0
        n_prior = 0
        stat_hist, rate_hist = deque(maxlen=15), deque(maxlen=15)
        for r in group:
            aid, pname, team, opp, is_home, season, week, gdate, rate, stat = r
            r3 = list(rate_hist)[-3:]
            recent_rate = sum(r3) / len(r3) if r3 else 0.0
            if n_prior >= MIN_PRIOR_GAMES and recent_rate >= cfg["min_recent_rate"]:
                s3, s5 = list(stat_hist)[-3:], list(stat_hist)[-5:]
                feat = {
                    "career_avg_yards": cum_stat / n_prior,
                    "recent3_avg_yards": sum(s3) / len(s3) if s3 else 0.0,
                    "recent5_avg_yards": sum(s5) / len(s5) if s5 else 0.0,
                    "career_avg_rate": cum_rate / n_prior,
                    "recent3_avg_rate": sum(r3) / len(r3) if r3 else 0.0,
                    "yards_per_unit": (cum_stat / cum_rate) if cum_rate > 0 else 0.0,
                    "is_home": 1.0 if is_home else 0.0,
                    "games_played": n_prior,
                }
                actual = stat if stat is not None else 0
                out.append({
                    "athlete_id": aid, "player_name": pname, "team": team,
                    "opponent": opp, "season": season, "week": week,
                    "game_date": gdate, **feat, "actual": actual,
                })
            cum_stat += stat if stat is not None else 0
            cum_rate += rate if rate is not None else 0
            stat_hist.append(stat if stat is not None else 0)
            rate_hist.append(rate if rate is not None else 0)
            n_prior += 1

    for r in rows:
        aid = r[0]
        if aid != cur_id:
            if group:
                flush(group)
            group, cur_id = [], aid
        group.append(r)
    if group:
        flush(group)
    return out


def run_market(conn, mkt_key, cfg, workdir):
    print(f"\n{'='*70}\n{mkt_key}\n{'='*70}")
    rows = build_rows(conn, mkt_key, cfg)
    print(f"total eligible rows (all seasons incl. live): {len(rows)}")

    dev_rows = [r for r in rows if r["season"] in DEV_SEASONS]
    holdout_rows = [r for r in rows if r["season"] == HOLDOUT_SEASON]
    print(f"dev ({DEV_SEASONS}): {len(dev_rows)}   holdout ({HOLDOUT_SEASON}): {len(holdout_rows)}")

    dev_actual = [r["actual"] for r in dev_rows]
    if not dev_actual:
        print("  NO DEV ROWS -- cannot pick a threshold. Skipping market.")
        return None

    print(f"\nTHRESHOLD DIAGNOSTIC (dev only, n={len(dev_actual)})")
    best_t, best_dist = None, 1.0
    for t in cfg["threshold_candidates"]:
        rate = sum(1 for y in dev_actual if y >= t + 0.5) / len(dev_actual)
        dist = abs(rate - 0.5)
        marker = ""
        if dist < best_dist:
            best_dist, best_t = dist, t
        print(f"  >{t:>6.1f}   over_rate={rate:.3f}")
    LINE = best_t
    print(f"  closest-to-50% threshold: over {LINE}")

    base_db = workdir / f"{mkt_key}_baseline.sqlite"
    if base_db.exists():
        base_db.unlink()
    out = sqlite3.connect(str(base_db))
    cols_sql = ", ".join(f"{c} REAL" for c in cfg["columns"])
    out.execute(f"""CREATE TABLE {mkt_key}_baseline (
        athlete_id TEXT, player_name TEXT, team TEXT, opponent TEXT,
        season INTEGER, week INTEGER, game_date TEXT, {cols_sql},
        actual INTEGER, over_line INTEGER
    )""")
    insert_cols = (["athlete_id", "player_name", "team", "opponent", "season", "week", "game_date"]
                   + cfg["columns"] + ["actual", "over_line"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO {mkt_key}_baseline ({', '.join(insert_cols)}) VALUES ({placeholder})"

    by_season = {}
    batch = []
    for r in rows:
        r["over_line"] = 1 if r["actual"] >= (LINE + 0.5) else 0
        batch.append(tuple(r[c] for c in insert_cols))
        st = by_season.setdefault(r["season"], {"rows": 0, "over": 0})
        st["rows"] += 1
        st["over"] += r["over_line"]
    out.executemany(ins, batch)
    out.commit()
    out.close()

    print(f"\n{'season':8s}{'rows':>8s}{'over_' + str(LINE):>12s}")
    for s in sorted(by_season):
        st = by_season[s]
        rate = st["over"] / st["rows"] if st["rows"] else 0
        print(f"{s:<8}{st['rows']:>8}{rate:>12.4f}")

    return {
        "line": LINE, "columns": cfg["columns"],
        "eligibility": {"position": cfg["position"], "min_prior_games": MIN_PRIOR_GAMES,
                        "rate_field": cfg["rate_field"], "min_recent_rate": cfg["min_recent_rate"]},
        "dev_seasons": DEV_SEASONS, "holdout_season": HOLDOUT_SEASON,
        "n_dev": len(dev_rows), "n_holdout": len(holdout_rows),
        "baseline_db": str(base_db), "by_season": by_season,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE_DEFAULT)
    ap.add_argument("--workdir", default=WORKDIR_DEFAULT)
    args = ap.parse_args()

    src = Path(args.source)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print("NFL_PRESEASON_CLEAN_BASELINE_A\n===============================")
    print(f"source={src}\nworkdir={workdir}")

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    manifest = {
        "script": "NFL_PRESEASON_CLEAN_BASELINE_A",
        "generated_at_utc": now_utc(),
        "source_db": str(src),
        "source_db_sha256": sha256_file(src),
        "cross_season_eligibility": True,
        "markets": {},
    }
    for mkt_key, cfg in MARKETS.items():
        result = run_market(conn, mkt_key, cfg, workdir)
        if result:
            manifest["markets"][mkt_key] = result
    conn.close()

    (workdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest: {workdir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
