#!/usr/bin/env python3
"""
MONEYLINE_CLEAN_BASELINE_B

Challenger B: isolates ONE change from clean_baseline_a -- adds starting
pitcher quality. The "a" model (team-season-aggregate features only)
FAILED its pre-registered gate on AUC (0.5408 vs the heuristic's 0.5611)
despite beating it on calibration (ECE 0.0133 vs 0.0621). Root-cause
hypothesis: real moneylines are priced mostly off the two probable
starters, not team-wide season averages -- and "a" completely omitted
starting pitcher quality. This is a principled, single-variable follow-up,
not blind tuning.

Everything else is identical to clean_baseline_a (same team features, same
eligibility, same 2024 dev / 2025 holdout split, same strict D-1
discipline) -- only the new pitcher features are added, so any AUC change
can be attributed to this one isolated cause.

New features (per side, as-of strictly-prior starts only)
------------------------------------------------------------
  {side}_pitcher_season_era       season-to-date ERA
  {side}_pitcher_season_k9        season-to-date K/9
  {side}_pitcher_season_bb9       season-to-date BB/9
  {side}_pitcher_recent3_era      rolling last-3-starts ERA
  {side}_pitcher_starts           season starts so far (sample-size context)

Games without a resolvable probable-starter game log for BOTH sides are
dropped (can't build the feature -- excluded, not imputed with a guess).

Read-only against the MLB Stats API. Writes only its own baseline.sqlite +
manifest.

Run (Render or locally)
------------------------
python -u moneyline_clean_baseline_b.py 2>&1 | tee moneyline_clean_baseline_b.log
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
WORKDIR_DEFAULT = "moneyline_clean_baseline_b_work"
DEV_SEASON = 2024
HOLDOUT_SEASON = 2025
MIN_PRIOR_GAMES = 10
MIN_PRIOR_STARTS = 3          # pitcher needs a real sample too
PYTH_EXP = 1.83
RECENT_N = 10
RECENT_PITCHER_N = 3
MAX_REST_DAYS_CAP = 5.0

TEAM_MODEL_COLUMNS = [
    "win_pct", "run_diff_pg", "pythag_win_pct",
    "recent10_win_pct", "recent10_run_diff", "win_streak", "rest_days",
]
PITCHER_MODEL_COLUMNS = [
    "pitcher_season_era", "pitcher_season_k9", "pitcher_season_bb9",
    "pitcher_recent3_era", "pitcher_starts",
]
MODEL_COLUMNS = ([f"home_{c}" for c in TEAM_MODEL_COLUMNS] + [f"away_{c}" for c in TEAM_MODEL_COLUMNS]
                 + [f"home_{c}" for c in PITCHER_MODEL_COLUMNS] + [f"away_{c}" for c in PITCHER_MODEL_COLUMNS])

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "prop-edge-moneyline-baseline-b/1.0"


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


def fetch_season_games_with_starters(season):
    start = f"{season}-03-01"
    end = f"{season}-11-15"
    data = _get(f"{MLB}/schedule", sportId=1, startDate=start, endDate=end,
                gameType="R", hydrate="probablePitcher")
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
            hp = (home.get("probablePitcher") or {}).get("id")
            ap = (away.get("probablePitcher") or {}).get("id")
            if not hp or not ap:
                continue
            out.append({
                "game_id": str(g.get("gamePk")),
                "date": g.get("officialDate") or (g.get("gameDate") or "")[:10],
                "home_team": home.get("team", {}).get("name"),
                "away_team": away.get("team", {}).get("name"),
                "home_score": int(home.get("score")),
                "away_score": int(away.get("score")),
                "home_win": 1 if home.get("isWinner") else 0,
                "home_pitcher": hp,
                "away_pitcher": ap,
            })
    out.sort(key=lambda r: (r["date"], r["game_id"]))
    return out


def fetch_pitcher_gamelog(pid, season):
    data = _get(f"{MLB}/people/{pid}/stats", stats="gameLog", season=season, group="pitching")
    splits = []
    try:
        splits = data.get("stats", [{}])[0].get("splits", [])
    except Exception:
        splits = []
    starts = []
    for sp in splits:
        st = sp.get("stat", {})
        if not st.get("gamesStarted"):
            continue
        d = sp.get("date")
        if not d:
            continue
        ip_str = st.get("inningsPitched", "0.0")
        try:
            whole, frac = ip_str.split(".") if "." in ip_str else (ip_str, "0")
            outs = int(whole) * 3 + int(frac)
        except Exception:
            outs = 0
        starts.append({
            "date": d,
            "er": float(st.get("earnedRuns", 0) or 0),
            "k": float(st.get("strikeOuts", 0) or 0),
            "bb": float(st.get("baseOnBalls", 0) or 0),
            "outs": outs,
        })
    starts.sort(key=lambda s: s["date"])
    return starts


def pythag(rs, ra):
    if rs <= 0 and ra <= 0:
        return 0.5
    num = rs ** PYTH_EXP
    den = rs ** PYTH_EXP + ra ** PYTH_EXP
    return num / den if den else 0.5


def build_rows(season):
    games = fetch_season_games_with_starters(season)
    print(f"  season {season}: {len(games)} final regular-season games with both probable starters")

    pitcher_ids = set()
    for g in games:
        pitcher_ids.add(g["home_pitcher"])
        pitcher_ids.add(g["away_pitcher"])
    print(f"  fetching game logs for {len(pitcher_ids)} distinct pitchers ...", flush=True)
    gamelogs = {}
    for i, pid in enumerate(sorted(pitcher_ids)):
        gamelogs[pid] = fetch_pitcher_gamelog(pid, season)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(pitcher_ids)} pitchers fetched")
    print(f"  done fetching game logs")

    def pitcher_feat_as_of(pid, game_date):
        starts = [s for s in gamelogs.get(pid, []) if s["date"] < game_date]
        n = len(starts)
        if n < MIN_PRIOR_STARTS:
            return None
        tot_er = sum(s["er"] for s in starts)
        tot_k = sum(s["k"] for s in starts)
        tot_bb = sum(s["bb"] for s in starts)
        tot_outs = sum(s["outs"] for s in starts)
        ip = tot_outs / 3.0
        era = (tot_er * 9 / ip) if ip > 0 else 5.5
        k9 = (tot_k * 9 / ip) if ip > 0 else 7.0
        bb9 = (tot_bb * 9 / ip) if ip > 0 else 3.5
        recent = starts[-RECENT_PITCHER_N:]
        r_er = sum(s["er"] for s in recent)
        r_outs = sum(s["outs"] for s in recent)
        r_ip = r_outs / 3.0
        recent_era = (r_er * 9 / r_ip) if r_ip > 0 else era
        return {
            "pitcher_season_era": era, "pitcher_season_k9": k9, "pitcher_season_bb9": bb9,
            "pitcher_recent3_era": recent_era, "pitcher_starts": float(n),
        }

    state = {}

    def get_state(team):
        if team not in state:
            state[team] = {"rs": 0.0, "ra": 0.0, "n": 0, "wins": 0,
                            "recent": deque(maxlen=RECENT_N), "streak": 0, "last_date": None}
        return state[team]

    def team_feat_for(team, game_date):
        st = get_state(team)
        n = st["n"]
        win_pct = (st["wins"] / n) if n else 0.5
        run_diff_pg = ((st["rs"] - st["ra"]) / n) if n else 0.0
        pyth = pythag(st["rs"], st["ra"])
        recent = list(st["recent"])
        r_win_pct = sum(1 for w, _ in recent if w) / len(recent) if recent else win_pct
        r_run_diff = sum(rd for _, rd in recent) / len(recent) if recent else run_diff_pg
        rest = MAX_REST_DAYS_CAP
        if st["last_date"] is not None:
            d0 = date.fromisoformat(st["last_date"])
            d1 = date.fromisoformat(game_date)
            rest = min(float((d1 - d0).days), MAX_REST_DAYS_CAP)
        return {
            "win_pct": win_pct, "run_diff_pg": run_diff_pg, "pythag_win_pct": pyth,
            "recent10_win_pct": r_win_pct, "recent10_run_diff": r_run_diff,
            "win_streak": float(st["streak"]), "rest_days": rest,
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
        hf, hn = team_feat_for(h, g["date"])
        af, an = team_feat_for(a, g["date"])
        hpf = pitcher_feat_as_of(g["home_pitcher"], g["date"])
        apf = pitcher_feat_as_of(g["away_pitcher"], g["date"])
        if hn >= MIN_PRIOR_GAMES and an >= MIN_PRIOR_GAMES and hpf and apf:
            row = {"game_id": g["game_id"], "date": g["date"], "season": season,
                   "home_team": h, "away_team": a}
            for c in TEAM_MODEL_COLUMNS:
                row[f"home_{c}"] = hf[c]
                row[f"away_{c}"] = af[c]
            for c in PITCHER_MODEL_COLUMNS:
                row[f"home_{c}"] = hpf[c]
                row[f"away_{c}"] = apf[c]
            row["existing_heuristic_home_pythag"] = hf["pythag_win_pct"]
            row["existing_heuristic_away_pythag"] = af["pythag_win_pct"]
            row["home_win"] = g["home_win"]
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

    print("MONEYLINE_CLEAN_BASELINE_B\n==========================")
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
    out.execute(f"""CREATE TABLE moneyline_baseline_b (
        game_id TEXT, date TEXT, season INTEGER, home_team TEXT, away_team TEXT,
        {cols_sql},
        existing_heuristic_home_pythag REAL, existing_heuristic_away_pythag REAL,
        home_win INTEGER
    )""")
    insert_cols = (["game_id", "date", "season", "home_team", "away_team"]
                   + MODEL_COLUMNS
                   + ["existing_heuristic_home_pythag", "existing_heuristic_away_pythag", "home_win"])
    placeholder = ", ".join("?" for _ in insert_cols)
    ins = f"INSERT INTO moneyline_baseline_b ({', '.join(insert_cols)}) VALUES ({placeholder})"
    out.executemany(ins, [tuple(r[c] for c in insert_cols) for r in all_rows])
    out.commit()
    out.close()

    manifest = {
        "script": "MONEYLINE_CLEAN_BASELINE_B",
        "generated_at_utc": now_utc(),
        "baseline_db": str(base_db),
        "model_columns": MODEL_COLUMNS,
        "eligibility": {"game_type": "R (regular season only)", "min_prior_games": MIN_PRIOR_GAMES,
                        "min_prior_pitcher_starts": MIN_PRIOR_STARTS},
        "strict_d1": "features use only games/starts with a strictly earlier date this season",
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
