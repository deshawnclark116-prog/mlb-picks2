#!/usr/bin/env python3
"""
NFL_DEFENSE_SACKS_CLEAN_BASELINE_A

Second individual-defensive-player-prop market (tackles failed its gate --
real discrimination but uncorrectable season-over-season calibration drift,
see nfl_defense_tackles_champion_gate_{a,b,c}.py). Sacks is a different
signal shape: checked first via a real-data diagnostic (not assumed) that
the over-rate is stable across 2022-2025 (0.31-0.36, no monotonic drift)
before investing in a full build -- the tackles market's drift wasn't
caught until after the full pipeline was built, so check the cheap thing
first this time.

Population: NOT scoped by position label. Real pass-rushers who record
sacks get inconsistently labeled DE/OLB/LB/MLB/DT/NT/DL across seasons and
source feeds (confirmed: e.g. 1508 of 4342+ sack-eligible player-games this
excerpt carry the generic 'LB' tag, not 'DE' -- many are blitzing edge
players in a 3-4 front). Eligibility is defined by RECENT PASS-RUSH ROLE
instead: a rolling recent5 sacks-per-game rate, which selects genuine
current pass-rushers regardless of how their position happens to be
labeled that season -- same "recent behavior over stale label" principle
already used for rushing_yards' carries filter and tackles' tackle-rate
filter.

Target: over_line = 1 if actual_sacks >= 0.5 (recorded a half-sack or more
that game) -- this IS the real "anytime sack" prop format books offer, not
an arbitrary threshold-diagnostic pick, so no threshold search needed here.

Builds a STRICT D-1 dataset from nfl_model.sqlite::player_games. Every
feature for a given player-game uses only that player's (and that
opponent's) games with a strictly earlier (season, week) in the same season.

Eligible population: DE/DT/OLB/LB/MLB/ILB/NT/DL positions, >= 5 prior games,
AND a rolling recent5 sacks-per-game rate >= 0.3 (verified empirically:
n=4342, sack>=0.5 rate 33.3% -- close to a real 50/50-ish binary split for
a rare event, and stable year over year).

Features
--------
  season_avg_sacks, recent3_avg_sacks, recent5_avg_sacks
  season_avg_qb_hits, recent3_avg_qb_hits   (pressure-rate leading indicator)
  opp_sacks_allowed_per_game   (as-of: mean sacks allowed by this opponent's
                                 offense across ALL eligible pass-rushers who
                                 have faced them so far this season)
  is_home, games_played

Split: 2022-2024 = development, 2025 = one-shot out-of-time holdout (same
convention as the tackles market, for consistency).

Read-only on nfl_model.sqlite. Writes only its own baseline.sqlite + manifest.

Run (Render)
------------
python -u nfl_defense_sacks_clean_baseline_a.py 2>&1 | tee /data/nfl_model/nfl_defense_sacks_clean_baseline_a.log
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
WORKDIR_DEFAULT = "/data/nfl_model/nfl_defense_sacks_clean_baseline_a_work"
POSITIONS = ("DE", "DT", "OLB", "LB", "MLB", "ILB", "NT", "DL")
MIN_PRIOR_GAMES_FOR_RATE = 5
MIN_RECENT_SACK_RATE = 0.3
LINE = 0.0  # over_line = actual_sacks >= LINE + 0.5, i.e. "recorded a sack"
DEV_SEASONS = (2022, 2023, 2024)
HOLDOUT_SEASON = 2025

MODEL_COLUMNS = [
    "season_avg_sacks", "recent3_avg_sacks", "recent5_avg_sacks",
    "season_avg_qb_hits", "recent3_avg_qb_hits",
    "opp_sacks_allowed_per_game", "is_home", "games_played",
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
               COALESCE(def_sacks, 0) AS sk,
               COALESCE(def_qb_hits, 0) AS qh
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
            sk = r[7]
            st = opp_state.setdefault((season, opp), [0, 0])
            st[0] += sk
            st[1] += 1

    out = []
    cur_key = None
    group = []

    def flush(group):
        cum_sk = cum_qh = 0.0
        n_prior = 0
        sk_hist = deque(maxlen=15)
        qh_hist = deque(maxlen=15)
        for r in group:
            pid, pname, team, opp, season, week, is_home, sk, qh = r
            s5 = list(sk_hist)[-5:]
            recent_sack_rate = sum(s5) / len(s5) if s5 else 0.0
            if (n_prior >= MIN_PRIOR_GAMES_FOR_RATE
                    and recent_sack_rate >= MIN_RECENT_SACK_RATE):
                s3 = list(sk_hist)[-3:]
                q3 = list(qh_hist)[-3:]
                feat = {
                    "season_avg_sacks": cum_sk / n_prior,
                    "recent3_avg_sacks": sum(s3) / len(s3) if s3 else 0.0,
                    "recent5_avg_sacks": recent_sack_rate,
                    "season_avg_qb_hits": cum_qh / n_prior,
                    "recent3_avg_qb_hits": sum(q3) / len(q3) if q3 else 0.0,
                    "opp_sacks_allowed_per_game": opp_asof.get((pid, season, week)),
                    "is_home": 1.0 if is_home else 0.0,
                    "games_played": n_prior,
                }
                out.append({
                    "player_id": pid, "player_name": pname, "team": team,
                    "opponent": opp, "season": season, "week": week,
                    **feat, "actual_sacks": sk,
                })
            cum_sk += sk
            cum_qh += qh
            sk_hist.append(sk)
            qh_hist.append(qh)
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

    print("NFL_DEFENSE_SACKS_CLEAN_BASELINE_A\n===================================")
    print(f"source={src}\nworkdir={work}")

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    rows = build_rows(conn)
    conn.close()
    print(f"\ntotal eligible pass-rusher rows: {len(rows)}")

    if base_db.exists():
        base_db.unlink()
    out = sqlite3.connect(str(base_db))
    cols_sql = ", ".join(f"{c} REAL" for c in MODEL_COLUMNS)
    out.execute(f"""CREATE TABLE nfl_defense_sacks_baseline (
        player_id TEXT, player_name TEXT, team TEXT, opponent TEXT,
        season INTEGER, week INTEGER, {cols_sql},
        actual_sacks REAL, over_line INTEGER
    )""")
    insert_cols = (["player_id", "player_name", "team", "opponent", "season", "week"]
                   + MODEL_COLUMNS + ["actual_sacks", "over_line"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO nfl_defense_sacks_baseline ({', '.join(insert_cols)}) VALUES ({placeholder})"

    by_season = {}
    batch = []
    for r in rows:
        r["over_line"] = 1 if r["actual_sacks"] >= (LINE + 0.5) else 0
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
        "script": "NFL_DEFENSE_SACKS_CLEAN_BASELINE_A",
        "generated_at_utc": now_utc(),
        "source_db": str(src),
        "source_db_sha256": sha256_file(src),
        "baseline_db": str(base_db),
        "model_columns": MODEL_COLUMNS,
        "eligibility": {"position": list(POSITIONS),
                        "min_prior_games_for_rate": MIN_PRIOR_GAMES_FOR_RATE,
                        "min_recent_sack_rate": MIN_RECENT_SACK_RATE},
        "strict_d1": "features use only games with strictly earlier (season,week); "
                      "opponent context uses only weeks strictly earlier in that season",
        "target": f"over_line = actual_sacks >= {LINE + 0.5} (recorded a sack)",
        "line": LINE,
        "dev_seasons": list(DEV_SEASONS), "holdout_season": HOLDOUT_SEASON,
        "total_rows": len(rows), "by_season": by_season,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'season':8s}{'rows':>8s}{'over_0.5':>10s}")
    for s in sorted(by_season):
        st = by_season[s]
        rate = st["over"] / st["rows"] if st["rows"] else 0
        print(f"{s:<8}{st['rows']:>8}{rate:>10.4f}")

    print(f"\nbaseline db: {base_db}")
    print("Read-only on nfl_model.sqlite. No production state changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
