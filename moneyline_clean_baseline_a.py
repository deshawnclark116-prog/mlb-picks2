#!/usr/bin/env python3
"""
MONEYLINE_CLEAN_BASELINE_A

Moneyline has never been trained or validated -- gamelines.py is a fixed,
untrained formula (Pythagorean expectation + log5 + a hardcoded home-field
edge), and record_intelligence tags it "active_probationary", not
"active_mature" like batter_hits/pitcher_strikeouts. Applying the same
predictions-first discipline used for every other market: build a strict
D-1 dataset, train a real classifier, gate it honestly on an untouched
holdout, and let it prove it beats BOTH a flat constant AND the existing
Pythagorean/log5 heuristic (the heuristic is a legitimate, non-trivial
baseline -- sabermetrically well-established -- so it's the real bar to
clear, not a strawman).

Builds a STRICT D-1 dataset directly from the MLB Stats API schedule
endpoint (one call per season covers the whole regular season). Every
feature for a given game uses only that team's (and the opponent's) games
with a strictly earlier date this season -- no leakage, no same-day info.

Eligible population: gameType == 'R' (regular season only -- postseason has
different leverage/rest dynamics and too small a sample; spring training is
not predictive of real form). >= 10 prior team-games this season (a real
season-to-date sample, not a 2-game reaction).

Features (per team, both home and away sides included)
--------------------------------------------------------
  {side}_win_pct           season-to-date win rate
  {side}_run_diff_pg       season-to-date run differential per game
  {side}_pythag_win_pct    season-to-date Pythagorean expectation
                            (rs^1.83 / (rs^1.83 + ra^1.83)) -- the same
                            core ingredient the existing heuristic uses, so
                            the model can learn how (and how much) to trust
                            it rather than applying it blindly
  {side}_recent10_win_pct  rolling last-10-games win rate
  {side}_recent10_run_diff rolling last-10-games run differential per game
  {side}_win_streak        current streak (positive=win streak, negative=
                            losing streak)
  {side}_rest_days         days since that team's last game (capped at 5)
  is_home                  (redundant with the side split, kept for clarity)

Target
------
  home_win = 1 if the home team won else 0

Split: 2024 season = development, 2025 season = one-shot out-of-time
holdout (both untouched -- 2026, the current season, is never touched here
since it's live production data).

Read-only against the MLB Stats API. Writes only its own baseline.sqlite +
manifest.

Run (Render or locally -- statsapi.mlb.com is reachable from both)
--------------------------------------------------------------------
python -u moneyline_clean_baseline_a.py 2>&1 | tee /data/nfl_model/moneyline_clean_baseline_a.log
"""

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

MLB = "https://statsapi.mlb.com/api/v1"
WORKDIR_DEFAULT = "moneyline_clean_baseline_a_work"
DEV_SEASON = 2024
HOLDOUT_SEASON = 2025
MIN_PRIOR_GAMES = 10
PYTH_EXP = 1.83
RECENT_N = 10
MAX_REST_DAYS_CAP = 5.0

MODEL_COLUMNS = [
    "home_win_pct", "home_run_diff_pg", "home_pythag_win_pct",
    "home_recent10_win_pct", "home_recent10_run_diff", "home_win_streak", "home_rest_days",
    "away_win_pct", "away_run_diff_pg", "away_pythag_win_pct",
    "away_recent10_win_pct", "away_recent10_run_diff", "away_win_streak", "away_rest_days",
]

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "prop-edge-moneyline-baseline/1.0"


def _get(url, **params):
    for attempt in range(3):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  retry {url}: {e}")
            time.sleep(1.5 * (attempt + 1))
    return {}


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def fetch_season_games(season):
    """One call per season: full regular-season schedule with final scores."""
    start = f"{season}-03-01"
    end = f"{season}-11-15"
    data = _get(f"{MLB}/schedule", sportId=1, startDate=start, endDate=end, gameType="R")
    out = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            status = g.get("status", {}).get("abstractGameState", "")
            if status != "Final":
                continue
            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            if home.get("score") is None or away.get("score") is None:
                continue
            out.append({
                "game_id": str(g.get("gamePk")),
                "date": g.get("officialDate") or (g.get("gameDate") or "")[:10],
                "home_team": home.get("team", {}).get("name"),
                "away_team": away.get("team", {}).get("name"),
                "home_score": int(home.get("score")),
                "away_score": int(away.get("score")),
                "home_win": 1 if home.get("isWinner") else 0,
            })
    out.sort(key=lambda r: (r["date"], r["game_id"]))
    return out


def pythag(rs, ra):
    if rs <= 0 and ra <= 0:
        return 0.5
    num = rs ** PYTH_EXP
    den = rs ** PYTH_EXP + ra ** PYTH_EXP
    return num / den if den else 0.5


def build_rows(season):
    games = fetch_season_games(season)
    print(f"  season {season}: {len(games)} final regular-season games")

    state = {}  # team -> dict(rs, ra, n, wins, recent=deque, streak, last_date)

    def get_state(team):
        if team not in state:
            state[team] = {"rs": 0.0, "ra": 0.0, "n": 0, "wins": 0,
                            "recent": deque(maxlen=RECENT_N), "streak": 0, "last_date": None}
        return state[team]

    def feat_for(team, game_date):
        st = get_state(team)
        n = st["n"]
        # n can be 0 (team's first game of the season) -- the caller filters
        # these rows out via MIN_PRIOR_GAMES before they're ever written, but
        # must not crash computing them in the first place.
        win_pct = (st["wins"] / n) if n else 0.5
        run_diff_pg = ((st["rs"] - st["ra"]) / n) if n else 0.0
        pyth = pythag(st["rs"], st["ra"])
        recent = list(st["recent"])
        r_win_pct = sum(1 for w, _ in recent if w) / len(recent) if recent else win_pct
        r_run_diff = sum(rd for _, rd in recent) / len(recent) if recent else run_diff_pg
        rest = None
        if st["last_date"] is not None:
            d0 = date.fromisoformat(st["last_date"])
            d1 = date.fromisoformat(game_date)
            rest = min(float((d1 - d0).days), MAX_REST_DAYS_CAP)
        return {
            "win_pct": win_pct, "run_diff_pg": run_diff_pg, "pythag_win_pct": pyth,
            "recent10_win_pct": r_win_pct, "recent10_run_diff": r_run_diff,
            "win_streak": float(st["streak"]), "rest_days": rest if rest is not None else MAX_REST_DAYS_CAP,
        }, n

    def update_state(team, game_date, runs_for, runs_against, won):
        st = get_state(team)
        st["rs"] += runs_for
        st["ra"] += runs_against
        st["n"] += 1
        st["wins"] += 1 if won else 0
        st["recent"].append((won, runs_for - runs_against))
        if won:
            st["streak"] = st["streak"] + 1 if st["streak"] > 0 else 1
        else:
            st["streak"] = st["streak"] - 1 if st["streak"] < 0 else -1
        st["last_date"] = game_date

    rows = []
    for g in games:
        h, a = g["home_team"], g["away_team"]
        hf, hn = feat_for(h, g["date"])
        af, an = feat_for(a, g["date"])
        if hn >= MIN_PRIOR_GAMES and an >= MIN_PRIOR_GAMES:
            row = {
                "game_id": g["game_id"], "date": g["date"], "season": season,
                "home_team": h, "away_team": a,
                "home_win_pct": hf["win_pct"], "home_run_diff_pg": hf["run_diff_pg"],
                "home_pythag_win_pct": hf["pythag_win_pct"],
                "home_recent10_win_pct": hf["recent10_win_pct"],
                "home_recent10_run_diff": hf["recent10_run_diff"],
                "home_win_streak": hf["win_streak"], "home_rest_days": hf["rest_days"],
                "away_win_pct": af["win_pct"], "away_run_diff_pg": af["run_diff_pg"],
                "away_pythag_win_pct": af["pythag_win_pct"],
                "away_recent10_win_pct": af["recent10_win_pct"],
                "away_recent10_run_diff": af["recent10_run_diff"],
                "away_win_streak": af["win_streak"], "away_rest_days": af["rest_days"],
                "home_win": g["home_win"],
                # kept for reference/audit, not fed to the model
                "existing_heuristic_home_pythag": hf["pythag_win_pct"],
                "existing_heuristic_away_pythag": af["pythag_win_pct"],
            }
            rows.append(row)
        update_state(h, g["date"], g["home_score"], g["away_score"], bool(g["home_win"]))
        update_state(a, g["date"], g["away_score"], g["home_score"], not bool(g["home_win"]))

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=WORKDIR_DEFAULT)
    args = ap.parse_args()

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    base_db = work / "baseline.sqlite"

    print("MONEYLINE_CLEAN_BASELINE_A\n==========================")
    print(f"workdir={work}")

    dev_rows = build_rows(DEV_SEASON)
    hol_rows = build_rows(HOLDOUT_SEASON)
    all_rows = dev_rows + hol_rows
    print(f"\ntotal eligible rows: {len(all_rows)}  (dev {DEV_SEASON}: {len(dev_rows)}, holdout {HOLDOUT_SEASON}: {len(hol_rows)})")

    dev_rate = sum(r["home_win"] for r in dev_rows) / len(dev_rows) if dev_rows else 0
    hol_rate = sum(r["home_win"] for r in hol_rows) / len(hol_rows) if hol_rows else 0
    print(f"home_win rate -- dev: {dev_rate:.4f}  holdout: {hol_rate:.4f}")

    if base_db.exists():
        base_db.unlink()
    out = sqlite3.connect(str(base_db))
    cols_sql = ", ".join(f"{c} REAL" for c in MODEL_COLUMNS)
    out.execute(f"""CREATE TABLE moneyline_baseline (
        game_id TEXT, date TEXT, season INTEGER, home_team TEXT, away_team TEXT,
        {cols_sql},
        existing_heuristic_home_pythag REAL, existing_heuristic_away_pythag REAL,
        home_win INTEGER
    )""")
    insert_cols = (["game_id", "date", "season", "home_team", "away_team"]
                   + MODEL_COLUMNS
                   + ["existing_heuristic_home_pythag", "existing_heuristic_away_pythag", "home_win"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO moneyline_baseline ({', '.join(insert_cols)}) VALUES ({placeholder})"
    out.executemany(ins, [tuple(r[c] for c in insert_cols) for r in all_rows])
    out.commit()
    out.close()

    manifest = {
        "script": "MONEYLINE_CLEAN_BASELINE_A",
        "generated_at_utc": now_utc(),
        "baseline_db": str(base_db),
        "model_columns": MODEL_COLUMNS,
        "eligibility": {"game_type": "R (regular season only)", "min_prior_games": MIN_PRIOR_GAMES},
        "strict_d1": "features use only games with a strictly earlier date this season",
        "target": "home_win = 1 if home team won",
        "dev_season": DEV_SEASON, "holdout_season": HOLDOUT_SEASON,
        "total_rows": len(all_rows),
        "dev_rows": len(dev_rows), "holdout_rows": len(hol_rows),
        "dev_home_win_rate": dev_rate, "holdout_home_win_rate": hol_rate,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nbaseline db: {base_db}")
    print("Read-only against the MLB Stats API. No production state changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
