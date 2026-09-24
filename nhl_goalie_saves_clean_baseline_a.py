#!/usr/bin/env python3
"""
NHL_GOALIE_SAVES_CLEAN_BASELINE_A

Real two-sided OVER/UNDER goalie saves market. SAVES_LINE=24.5 is a
round-number real book line (not fit to this data's own median, same
reasoning as the shots-on-goal line). Population: goalie_games, filtered
to appearances with real meaningful ice time (toi_seconds >=
MIN_TOI_SECONDS) -- a token mop-up relief appearance isn't the same
population as a real start, and this table doesn't carry a separate
"gamesStarted" flag the way the source API's raw row does, so TOI is the
real, direct proxy used instead.

Same asof discipline as every other market here: season/recent-N save
averages and workload (shots-against) computed from strictly earlier
games only. Unlike points/shots (where the SKATER's own team's offense/
defense matters), a goalie's workload is driven by the OPPONENT's own
shot generation, so "opp_shots_for_per_game" (how many shots this
opponent generates per game, tracked the mirror-image way the points/
shots pipelines track "allowed") is the more relevant context feature
than team_net_margin -- both are included since team quality plausibly
still correlates with shot suppression/generation.

Run
---
python -u nhl_goalie_saves_clean_baseline_a.py --source nhl_models/nhl_model.sqlite
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
from nhl_points_clean_baseline_a import build_team_allowed_asof

SOURCE_DEFAULT = "nhl_models/nhl_model.sqlite"
WORKDIR_DEFAULT = "nhl_models/nhl_goalie_saves_clean_baseline_a_work"
MIN_PRIOR_GAMES = 5
MIN_TOI_SECONDS = 1800  # >= 30 min of real ice time -- excludes token relief appearances
SAVES_LINE = 24.5
DEV_SEASONS = (2018, 2019, 2020, 2021, 2022)
VAL_SEASON = 2023
HOLDOUT_SEASON = 2024

MODEL_COLUMNS = [
    "season_avg_saves", "recent3_avg_saves", "recent5_avg_saves",
    "season_avg_shots_against", "recent3_avg_shots_against", "save_pct",
    "opp_shots_for_per_game", "is_home", "games_played",
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


def build_opponent_shots_for_tracker(conn):
    """Mirror image of nhl_points_clean_baseline_a's opponent-ALLOWED
    tracker: here we want how many shots THIS opponent generates per
    game (their own offensive shot volume), keyed by (season, opp_team,
    game_id) -- i.e. how many shots the goalie's actual opponent tends
    to throw at a goalie, asof strictly prior games. Reuses skater_games'
    per-team-per-game shot totals (same table the shots-on-goal pipeline
    aggregates), just tracked as "own production" instead of "allowed"."""
    rows = conn.execute("""
        SELECT game_id, team, season, game_date, SUM(shots)
        FROM skater_games GROUP BY game_id, team
    """).fetchall()
    by_team = defaultdict(list)
    for gid, team, season, gdate, total in rows:
        by_team[(season, team)].append((gdate, gid, total or 0))
    for k in by_team:
        by_team[k].sort()
    games_by_team_game = {k: [(gid, gdate, total) for (gdate, gid, total) in v]
                           for k, v in by_team.items()}
    return build_team_allowed_asof(games_by_team_game, None)


def build_rows(conn):
    team_state_asof = build_team_state_asof(conn)
    opp_shots_for_asof = build_opponent_shots_for_tracker(conn)

    rows = conn.execute("""
        SELECT gg.player_id, gg.player_name, gg.team, gg.opponent, gg.season, g.week, gg.game_id,
               gg.game_date, gg.is_home, gg.saves, gg.shots_against, gg.toi_seconds
        FROM goalie_games gg
        JOIN games g ON gg.game_id = g.game_id
        WHERE gg.toi_seconds >= ?
        ORDER BY gg.season, gg.game_date, gg.game_id
    """, (MIN_TOI_SECONDS,)).fetchall()

    hist = defaultdict(list)  # (season, player_id) -> [(saves, shots_against), ...]
    out = []
    for pid, pname, team, opp, season, week, gid, gdate, is_home, saves, sa, toi in rows:
        h = hist[(season, pid)]
        n = len(h)
        if n >= MIN_PRIOR_GAMES:
            recent3 = h[-3:]
            recent5 = h[-5:]
            season_saves = [x[0] for x in h]
            season_sa = [x[1] for x in h]
            total_saves = sum(season_saves)
            total_sa = sum(season_sa)
            team_st = team_state_asof.get((season, team, week))
            opp_st = team_state_asof.get((season, opp, week))
            team_margin = team_st["net_margin"] if team_st else None
            opp_margin = opp_st["net_margin"] if opp_st else None
            proj_margin = (team_margin - opp_margin) if (team_margin is not None and opp_margin is not None) else None
            out.append({
                "player_id": pid, "player_name": pname, "team": team, "opponent": opp,
                "season": season, "week": week, "game_id": gid,
                "season_avg_saves": total_saves / n,
                "recent3_avg_saves": sum(x[0] for x in recent3) / len(recent3),
                "recent5_avg_saves": sum(x[0] for x in recent5) / len(recent5),
                "season_avg_shots_against": total_sa / n,
                "recent3_avg_shots_against": sum(x[1] for x in recent3) / len(recent3),
                "save_pct": (total_saves / total_sa) if total_sa > 0 else None,
                "opp_shots_for_per_game": opp_shots_for_asof.get((season, opp, gid)),
                "is_home": 1.0 if is_home else 0.0,
                "games_played": n,
                "team_net_margin": team_margin, "opp_net_margin": opp_margin,
                "projected_margin": proj_margin,
                "actual_saves": saves,
                "over_line": 1 if saves >= SAVES_LINE + 0.5 else 0,
            })
        h.append((saves, sa))
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

    print("NHL_GOALIE_SAVES_CLEAN_BASELINE_A\n==================================")
    print(f"source={src}\nworkdir={work}")

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    rows = build_rows(conn)
    conn.close()
    print(f"\ntotal eligible rows: {len(rows)}")

    if base_db.exists():
        base_db.unlink()
    out = sqlite3.connect(str(base_db))
    cols_sql = ", ".join(f"{c} REAL" for c in MODEL_COLUMNS)
    out.execute(f"""CREATE TABLE nhl_goalie_saves_baseline (
        player_id INTEGER, player_name TEXT, team TEXT, opponent TEXT,
        season INTEGER, week INTEGER, game_id INTEGER, {cols_sql},
        actual_saves INTEGER, over_line INTEGER
    )""")
    insert_cols = (["player_id", "player_name", "team", "opponent", "season", "week", "game_id"]
                   + MODEL_COLUMNS + ["actual_saves", "over_line"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO nhl_goalie_saves_baseline ({', '.join(insert_cols)}) VALUES ({placeholder})"

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
        "script": "NHL_GOALIE_SAVES_CLEAN_BASELINE_A", "generated_at_utc": now_utc(),
        "source_db": str(src), "source_db_sha256": sha256_file(src),
        "baseline_db": str(base_db), "model_columns": MODEL_COLUMNS,
        "eligibility": {"min_prior_games": MIN_PRIOR_GAMES, "min_toi_seconds": MIN_TOI_SECONDS},
        "target": f"over_line = 1 if actual_saves >= {SAVES_LINE + 0.5}",
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
