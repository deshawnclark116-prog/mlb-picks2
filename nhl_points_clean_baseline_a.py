#!/usr/bin/env python3
"""
NHL_POINTS_CLEAN_BASELINE_A

NHL's first player-level market: real "anytime point" (>=1 point in the
game) -- a genuine one-sided real book market (Anytime Point Scorer),
same shape as CFB/NFL's anytime_touchdowns (no fabricated "no point"
side ever shown, see nhl_serving_builder_a.py). Population: every real
skater-game row (any position -- C/L/R/D; real books don't restrict this
prop by position).

Strict D-1 asof features throughout: season/recent-N averages and the
opponent-allowed rate are all computed from STRICTLY EARLIER games only,
same discipline as every other market in this repo. team_net_margin/
opp_net_margin/projected_margin are pulled from nhl_moneyline_clean_
baseline_a.py's own team-state tracker (same real games table, same
per-week asof convention) -- team offensive/defensive quality plausibly
matters for a skater's point chances just as it does for who wins.

Eligibility: recent3 time-on-ice/game >= MIN_RECENT_TOI_SECONDS -- a
4th-liner or black-ace who barely plays can't be trusted to reflect a
"real role" the way a full prior season's low-TOI average might imply;
this recency floor is the same shape as CFB/NFL's own recent-rate
eligibility rules (e.g. recent3 carries/game >= 12 for RBs).

Run
---
python -u nhl_points_clean_baseline_a.py --source nhl_models/nhl_model.sqlite
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

SOURCE_DEFAULT = "nhl_models/nhl_model.sqlite"
WORKDIR_DEFAULT = "nhl_models/nhl_points_clean_baseline_a_work"
MIN_PRIOR_GAMES = 5
MIN_RECENT_TOI_SECONDS = 480  # recent3 avg >= 8:00/game -- excludes black-aces/scratches-in-disguise
POINTS_LINE = 0.5  # anytime point: >=1 point
DEV_SEASONS = (2018, 2019, 2020, 2021, 2022)
VAL_SEASON = 2023
HOLDOUT_SEASON = 2024

MODEL_COLUMNS = [
    "season_avg_points", "recent3_avg_points", "recent5_avg_points",
    "season_avg_toi", "recent3_avg_toi", "points_per_60",
    "opp_points_allowed_per_game", "is_home", "games_played",
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


def build_team_allowed_asof(games_by_team_game, seasons_of_game):
    """team -> {week-independent, strictly chronological asof by game_date}:
    real per-game "points allowed" tracker -- for each team, walks its own
    games in date order and records, BEFORE each game, the running average
    of how many combined points the OPPOSING team's skaters scored against
    them in each prior game. games_by_team_game: (season, team) ->
    sorted list of (game_date, opponent_total_stat)."""
    allowed_asof = {}  # (season, team, game_id) -> running avg allowed so far
    for (season, team), games in games_by_team_game.items():
        total = 0.0
        n = 0
        for game_id, game_date, opp_total in games:
            allowed_asof[(season, team, game_id)] = (total / n) if n > 0 else None
            total += opp_total
            n += 1
    return allowed_asof


def load_team_game_totals(conn, stat_col):
    """(game_id, team) -> sum(stat_col) across that team's skaters in that
    game -- used both as "this team's own output" (not used directly here)
    and, cross-referenced by opponent, as "stat allowed by the OTHER team
    in that same game"."""
    rows = conn.execute(f"""
        SELECT game_id, team, opponent, season, game_date, SUM({stat_col})
        FROM skater_games GROUP BY game_id, team
    """).fetchall()
    totals = {}
    meta = {}
    for gid, team, opp, season, gdate, total in rows:
        totals[(gid, team)] = total or 0
        meta[(gid, team)] = (opp, season, gdate)
    return totals, meta


def build_opponent_allowed_tracker(conn, stat_col):
    totals, meta = load_team_game_totals(conn, stat_col)
    by_team = defaultdict(list)
    for (gid, team), total in totals.items():
        opp, season, gdate = meta[(gid, team)]
        opp_total = totals.get((gid, opp), 0)
        by_team[(season, team)].append((gdate, gid, opp_total))
    for k in by_team:
        by_team[k].sort()
    games_by_team_game = {k: [(gid, gdate, opp_total) for (gdate, gid, opp_total) in v]
                           for k, v in by_team.items()}
    return build_team_allowed_asof(games_by_team_game, None)


def build_rows(conn):
    team_state_asof = build_team_state_asof(conn)
    opp_allowed_asof = build_opponent_allowed_tracker(conn, "points")

    rows = conn.execute("""
        SELECT player_id, player_name, team, opponent, season, week, game_id,
               game_date, is_home, points, toi_seconds
        FROM skater_games sg
        JOIN games g ON sg.game_id = g.game_id
        ORDER BY sg.season, sg.game_date, sg.game_id
    """).fetchall()

    hist = defaultdict(list)  # (season, player_id) -> [(points, toi), ...]
    out = []
    for pid, pname, team, opp, season, week, gid, gdate, is_home, points, toi in rows:
        h = hist[(season, pid)]
        n = len(h)
        if n >= MIN_PRIOR_GAMES:
            recent3 = h[-3:]
            recent5 = h[-5:]
            recent3_toi = sum(x[1] for x in recent3) / len(recent3)
            if recent3_toi >= MIN_RECENT_TOI_SECONDS:
                season_pts = [x[0] for x in h]
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
                    "season_avg_points": sum(season_pts) / n,
                    "recent3_avg_points": sum(x[0] for x in recent3) / len(recent3),
                    "recent5_avg_points": sum(x[0] for x in recent5) / len(recent5),
                    "season_avg_toi": sum(season_toi) / n,
                    "recent3_avg_toi": recent3_toi,
                    "points_per_60": (sum(season_pts) / total_toi_hr) if total_toi_hr > 0 else 0.0,
                    "opp_points_allowed_per_game": opp_allowed_asof.get((season, opp, gid)),
                    "is_home": 1.0 if is_home else 0.0,
                    "games_played": n,
                    "team_net_margin": team_margin, "opp_net_margin": opp_margin,
                    "projected_margin": proj_margin,
                    "actual_points": points,
                    "over_line": 1 if points >= POINTS_LINE + 0.5 else 0,
                })
        h.append((points, toi))
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

    print("NHL_POINTS_CLEAN_BASELINE_A\n============================")
    print(f"source={src}\nworkdir={work}")

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    rows = build_rows(conn)
    conn.close()
    print(f"\ntotal eligible rows: {len(rows)}")

    if base_db.exists():
        base_db.unlink()
    out = sqlite3.connect(str(base_db))
    cols_sql = ", ".join(f"{c} REAL" for c in MODEL_COLUMNS)
    out.execute(f"""CREATE TABLE nhl_points_baseline (
        player_id INTEGER, player_name TEXT, team TEXT, opponent TEXT,
        season INTEGER, week INTEGER, game_id INTEGER, {cols_sql},
        actual_points INTEGER, over_line INTEGER
    )""")
    insert_cols = (["player_id", "player_name", "team", "opponent", "season", "week", "game_id"]
                   + MODEL_COLUMNS + ["actual_points", "over_line"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO nhl_points_baseline ({', '.join(insert_cols)}) VALUES ({placeholder})"

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
        "script": "NHL_POINTS_CLEAN_BASELINE_A", "generated_at_utc": now_utc(),
        "source_db": str(src), "source_db_sha256": sha256_file(src),
        "baseline_db": str(base_db), "model_columns": MODEL_COLUMNS,
        "eligibility": {"min_prior_games": MIN_PRIOR_GAMES,
                         "min_recent3_toi_seconds": MIN_RECENT_TOI_SECONDS},
        "target": f"over_line = 1 if actual_points >= {POINTS_LINE + 0.5}",
        "dev_seasons": list(DEV_SEASONS), "val_season": VAL_SEASON,
        "holdout_season": HOLDOUT_SEASON, "total_rows": len(rows), "by_season": by_season,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'season':8s}{'rows':>8s}{'anytime_pt_rate':>16s}")
    for s in sorted(by_season):
        st = by_season[s]
        rate = st["hit"] / st["rows"] if st["rows"] else 0
        print(f"{s:<8}{st['rows']:>8}{rate:>16.4f}")

    print(f"\nbaseline db: {base_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
