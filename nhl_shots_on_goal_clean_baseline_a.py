#!/usr/bin/env python3
"""
NHL_SHOTS_ON_GOAL_CLEAN_BASELINE_A

Real two-sided OVER/UNDER market (unlike points' one-sided anytime
design): every real sportsbook offers both sides of a skater's shots-on-
goal total, so a genuine UNDER pick is real and bettable here, not
fabricated. SHOTS_LINE=2.5 is a real, commonly-offered book line for a
top-six forward (round-number choice, same convention as CFB/NFL's own
fixed lines like 69.5 rushing yards -- not fit to this data's own
median, to avoid look-ahead bias in the LINE itself).

Same structure as nhl_points_clean_baseline_a.py otherwise: strict D-1
asof season/recent-N averages, opponent-shots-allowed tracker, team
context features pulled from nhl_moneyline's own team-state tracker,
same MIN_RECENT_TOI_SECONDS eligibility floor.

Run
---
python -u nhl_shots_on_goal_clean_baseline_a.py --source nhl_models/nhl_model.sqlite
"""
import argparse
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nhl_moneyline_clean_baseline_a import build_team_state_asof
from nhl_points_clean_baseline_a import build_opponent_allowed_tracker

SOURCE_DEFAULT = "nhl_models/nhl_model.sqlite"
WORKDIR_DEFAULT = "nhl_models/nhl_shots_on_goal_clean_baseline_a_work"
MIN_PRIOR_GAMES = 5
MIN_RECENT_TOI_SECONDS = 480
SHOTS_LINE = 2.5
DEV_SEASONS = (2018, 2019, 2020, 2021, 2022)
VAL_SEASON = 2023
HOLDOUT_SEASON = 2024

MODEL_COLUMNS = [
    "season_avg_shots", "recent3_avg_shots", "recent5_avg_shots",
    "season_avg_toi", "recent3_avg_toi", "shots_per_60",
    "opp_shots_allowed_per_game", "is_home", "games_played",
    "team_net_margin", "opp_net_margin", "projected_margin",
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
    team_state_asof = build_team_state_asof(conn)
    opp_allowed_asof = build_opponent_allowed_tracker(conn, "shots")

    rows = conn.execute("""
        SELECT player_id, player_name, team, opponent, season, week, game_id,
               game_date, is_home, shots, toi_seconds
        FROM skater_games sg
        JOIN games g ON sg.game_id = g.game_id
        ORDER BY sg.season, sg.game_date, sg.game_id
    """).fetchall()

    hist = defaultdict(list)
    out = []
    for pid, pname, team, opp, season, week, gid, gdate, is_home, shots, toi in rows:
        h = hist[(season, pid)]
        n = len(h)
        if n >= MIN_PRIOR_GAMES:
            recent3 = h[-3:]
            recent5 = h[-5:]
            recent3_toi = sum(x[1] for x in recent3) / len(recent3)
            if recent3_toi >= MIN_RECENT_TOI_SECONDS:
                season_shots = [x[0] for x in h]
                season_toi = [x[1] for x in h]
                total_toi_hr = sum(season_toi) / 3600.0
                team_st = team_state_asof.get((season, team, week))
                opp_st = team_state_asof.get((season, opp, week))
                team_margin = team_st["net_margin"] if team_st else None
                opp_margin = opp_st["net_margin"] if opp_st else None
                proj_margin = (team_margin - opp_margin) if (team_margin is not None and opp_margin is not None) else None
                out.append({
                    "player_id": pid, "player_name": pname, "team": team, "opponent": opp,
                    "season": season, "week": week, "game_id": gid,
                    "season_avg_shots": sum(season_shots) / n,
                    "recent3_avg_shots": sum(x[0] for x in recent3) / len(recent3),
                    "recent5_avg_shots": sum(x[0] for x in recent5) / len(recent5),
                    "season_avg_toi": sum(season_toi) / n,
                    "recent3_avg_toi": recent3_toi,
                    "shots_per_60": (sum(season_shots) / total_toi_hr) if total_toi_hr > 0 else 0.0,
                    "opp_shots_allowed_per_game": opp_allowed_asof.get((season, opp, gid)),
                    "is_home": 1.0 if is_home else 0.0,
                    "games_played": n,
                    "team_net_margin": team_margin, "opp_net_margin": opp_margin,
                    "projected_margin": proj_margin,
                    "actual_shots": shots,
                    "over_line": 1 if shots >= SHOTS_LINE + 0.5 else 0,
                })
        h.append((shots, toi))
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

    print("NHL_SHOTS_ON_GOAL_CLEAN_BASELINE_A\n===================================")
    print(f"source={src}\nworkdir={work}")

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    rows = build_rows(conn)
    conn.close()
    print(f"\ntotal eligible rows: {len(rows)}")

    if base_db.exists():
        base_db.unlink()
    out = sqlite3.connect(str(base_db))
    cols_sql = ", ".join(f"{c} REAL" for c in MODEL_COLUMNS)
    out.execute(f"""CREATE TABLE nhl_shots_on_goal_baseline (
        player_id INTEGER, player_name TEXT, team TEXT, opponent TEXT,
        season INTEGER, week INTEGER, game_id INTEGER, {cols_sql},
        actual_shots INTEGER, over_line INTEGER
    )""")
    insert_cols = (["player_id", "player_name", "team", "opponent", "season", "week", "game_id"]
                   + MODEL_COLUMNS + ["actual_shots", "over_line"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO nhl_shots_on_goal_baseline ({', '.join(insert_cols)}) VALUES ({placeholder})"

    by_season = {}
    batch = []
    for r in rows:
        batch.append(tuple(r[c] for c in insert_cols))
        st = by_season.setdefault(r["season"], {"rows": 0, "hit": 0})
        st["rows"] += 1
        st["hit"] += r["over_line"]
        if len(batch) >= 5000:
            out.executemany(ins, batch); batch = []
    if batch:
        out.executemany(ins, batch)
    out.commit()
    out.close()

    manifest = {
        "script": "NHL_SHOTS_ON_GOAL_CLEAN_BASELINE_A", "generated_at_utc": now_utc(),
        "source_db": str(src), "source_db_sha256": sha256_file(src),
        "baseline_db": str(base_db), "model_columns": MODEL_COLUMNS,
        "eligibility": {"min_prior_games": MIN_PRIOR_GAMES,
                         "min_recent3_toi_seconds": MIN_RECENT_TOI_SECONDS},
        "target": f"over_line = 1 if actual_shots >= {SHOTS_LINE + 0.5}",
        "dev_seasons": list(DEV_SEASONS), "val_season": VAL_SEASON,
        "holdout_season": HOLDOUT_SEASON, "total_rows": len(rows), "by_season": by_season,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'season':8s}{'rows':>8s}{'over_rate':>12s}")
    for s in sorted(by_season):
        st = by_season[s]
        rate = st["hit"] / st["rows"] if st["rows"] else 0
        print(f"{s:<8}{st['rows']:>8}{rate:>12.4f}")

    print(f"\nbaseline db: {base_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
