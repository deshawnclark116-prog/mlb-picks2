"""
CFB_PHASE1_DATA -- frozen research tables + the point-in-time candidate universe for the CFB Outcome Engine v1 (RESEARCH / SHADOW ONLY).

Frozen tables (cfb_models/cfb_outcome_engine/phase1_data/, deterministic gzip jsonl + sha256 manifest), seasons 2018-2025, NO 2026 row:
  team_games_{season}   one row per (game, FBS-or-FCS team) built from the cfbfastR-data play-attributed rows (rush / pass-attempt / sack / red-zone plays) joined to the schedule (final points, division, neutral site)
  player_games_{season} FBS-vs-FBS player lines taken from the legacy frozen DB cfb_models/cfb_model.sqlite (read-only: it embeds the completion / reception attribution fix); FBS-vs-FCS player lines do not exist there
AS-OF CONTRACT: CFB has week granularity, so a source game is usable for a target (season, week) iff its (season, week) < the target's (the whole target week is excluded). NO TARGET-GAME PARTICIPATION ORACLE: candidates are decided from
strictly earlier weeks only; target rows are read afterwards for labels.
  python cfb_phase1_data.py freeze --raw DIR [--db cfb_models/cfb_model.sqlite]
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

import cfb_phase1_common as C

REPO = Path(__file__).resolve().parent
OUT = REPO / "cfb_models" / "cfb_outcome_engine"
DATA = OUT / "phase1_data"
SEASONS = list(range(2018, 2026))
SKILL = ("QB", "RB", "FB", "WR", "TE")
RECENCY_TEAM_GAMES = 12                      # candidate: the player's LAST appearance (any team) was for this team within the team's last 12 completed games
PLAYER_FIELDS = ["player_id", "player_name", "position", "team", "opponent", "season", "week", "game_id", "game_date", "is_home", "carries", "rushing_yards", "rushing_touchdowns", "receptions", "receiving_yards", "receiving_touchdowns",
                 "pass_attempts", "completions", "passing_yards", "passing_touchdowns", "passing_interceptions"]
csv.field_size_limit(10 ** 8)


def det_gzip(text):
    bio = io.BytesIO()
    with gzip.GzipFile(fileobj=bio, mode="wb", mtime=0, compresslevel=9) as g:
        g.write(text.encode())
    return bio.getvalue()


def nz(v):
    return v not in ("", "NA", None)


# ------------------------------------------------------------------ build (raw -> frozen)
def read_schedule(raw, season):
    out = {}
    for r in csv.DictReader(open(Path(raw) / f"cfb_schedules_{season}.csv", newline="", encoding="utf-8")):
        if r["season_type"] != "regular" or r["completed"] != "TRUE" or not (nz(r["home_points"]) and nz(r["away_points"])):
            continue
        if "fbs" not in (r["home_division"], r["away_division"]):
            continue
        out[r["game_id"]] = r
    return out


def team_games_from_plays(raw, season, sched):
    acc = defaultdict(lambda: {"rush": set(), "pass": set(), "sack": set(), "rz_rush": set(), "rz_pass": set(), "rush_yds": 0, "pass_comp_yds": 0, "completions": 0, "ints": 0})
    with open(Path(raw) / f"player_stats_{season}.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["game_id"] not in sched:
                continue
            a = acc[(r["game_id"], r["team"])]; p = r["play_id"]
            rz = nz(r["yards_to_goal"]) and float(r["yards_to_goal"]) <= 20
            if nz(r["rush_player_id"]):
                if p not in a["rush"]:
                    a["rush_yds"] += int(float(r["rush_yds"])) if nz(r["rush_yds"]) else 0
                a["rush"].add(p)
                if rz:
                    a["rz_rush"].add(p)
            comp, inc, itc = nz(r["completion_player_id"]), nz(r["incompletion_player_id"]), nz(r["interception_thrown_player_id"])
            if comp or inc or itc:
                a["pass"].add(p)
                if rz:
                    a["rz_pass"].add(p)
                if comp:
                    a["completions"] += 1
                    a["pass_comp_yds"] += int(float(r["completion_yds"])) if nz(r["completion_yds"]) else 0
                a["ints"] += itc
            if nz(r["sack_taken_player_id"]):
                a["sack"].add(p)
    rows = []
    for gid, g in sched.items():
        for side in ("home", "away"):
            team = g[f"{side}_team"]; opp = g["away_team" if side == "home" else "home_team"]
            a = acc.get((gid, team))
            rows.append({"game_id": gid, "season": int(g["season"]), "week": int(g["week"]), "game_date": g["start_date"][:10], "team": team, "opponent": opp, "is_home": int(side == "home"), "neutral": int(g["neutral_site"] == "TRUE"),
                         "team_div": g[f"{side}_division"], "opp_div": g["away_division" if side == "home" else "home_division"], "team_conf": g[f"{side}_conference"], "opp_conf": g["away_conference" if side == "home" else "home_conference"],
                         "team_points": int(float(g[f"{side}_points"])), "opp_points": int(float(g["away_points" if side == "home" else "home_points"])), "has_play_rows": int(a is not None),
                         "rush_plays": len(a["rush"]) if a else None, "pass_att_plays": len(a["pass"]) if a else None, "sack_plays": len(a["sack"]) if a else None,
                         "plays": (len(a["rush"]) + len(a["pass"]) + len(a["sack"])) if a else None, "rz_rush_plays": len(a["rz_rush"]) if a else None, "rz_pass_plays": len(a["rz_pass"]) if a else None,
                         "rush_yards": a["rush_yds"] if a else None, "completions": a["completions"] if a else None, "pass_comp_yards": a["pass_comp_yds"] if a else None, "interceptions": a["ints"] if a else None})
    rows.sort(key=lambda r: (r["season"], r["week"], r["game_id"], -r["is_home"]))
    return rows


def player_games_from_db(db, season):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cur = con.execute(f"SELECT {','.join(PLAYER_FIELDS)} FROM player_games WHERE season = ? ORDER BY week, game_id, player_id", (season,))
    rows = [dict(zip(PLAYER_FIELDS, r)) for r in cur]
    con.close()
    for r in rows:
        C.assert_research_allowed(r["season"], r["week"], "fit")
    return rows


def write_jsonl(path, rows):
    b = det_gzip("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows))
    Path(path).write_bytes(b)
    return {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "rows": len(rows)}


def freeze(raw, db, out=DATA):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    files = {}
    for s in SEASONS:
        sched = read_schedule(raw, s)
        C.assert_research_allowed(s, 1, "fit")
        files[f"team_games_{s}.jsonl.gz"] = write_jsonl(out / f"team_games_{s}.jsonl.gz", team_games_from_plays(raw, s, sched))
        files[f"player_games_{s}.jsonl.gz"] = write_jsonl(out / f"player_games_{s}.jsonl.gz", player_games_from_db(db, s))
    man = {"dataset": "cfb-outcome-engine-research-tables-v1", "seasons": SEASONS, "contains_2026": False, "files": files,
           "sources": {"team_games": "cfbfastR-data player_stats (play-attributed rows) + schedules", "player_games": "cfb_models/cfb_model.sqlite player_games (read-only legacy frozen DB, 2018-2025 only)"}}
    man["manifest_content_sha256"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    return man


# ------------------------------------------------------------------ load (frozen -> research rows) with cutoff enforcement
def read_jsonl(path):
    return [json.loads(l) for l in gzip.decompress(Path(path).read_bytes()).decode().splitlines() if l]


def verify_manifest(data_dir=DATA):
    man = json.loads((Path(data_dir) / "manifest.json").read_text())
    for name, meta in man["files"].items():
        got = hashlib.sha256((Path(data_dir) / name).read_bytes()).hexdigest()
        if got != meta["sha256"]:
            raise RuntimeError(f"{name}: sha256 mismatch (frozen table altered)")
    return man


def load_frozen(data_dir=DATA, seasons=SEASONS, verify=True):  # dev code passes seasons <= 2024 (2025 is reserved)
    if verify:
        verify_manifest(data_dir)
    tg, pg = [], []
    for s in seasons:
        tg += C.filter_research_rows(read_jsonl(Path(data_dir) / f"team_games_{s}.jsonl.gz"), "fit")
        pg += C.filter_research_rows(read_jsonl(Path(data_dir) / f"player_games_{s}.jsonl.gz"), "fit")
    return tg, pg


# ------------------------------------------------------------------ point-in-time candidate universe
def pos_class(code):
    return code if code in SKILL else None


def build_candidates(team_games, player_games, target_seasons=C.TARGET_SEASONS, missing_out=None):
    """For every FBS-vs-FBS target team-game, the candidate skill players decided ONLY from strictly earlier weeks, then the actual target-game lines are read for labels.
    Candidate rule (documented, deterministic): the player's most recent PRIOR appearance (any team, any earlier week) was for this team AND occurred within this team's last RECENCY_TEAM_GAMES completed games AND the player's position at
    that appearance is a skill position (QB / RB / FB / WR / TE). Non-participants stay candidates with all outcomes 0.
    Returns (rows, coverage): rows is a list of dicts {season, week, game_id, team, opponent, player_id, position, ...labels}."""
    games_by_week = defaultdict(list)                                        # (season, week) -> team-game rows
    for r in team_games:
        games_by_week[(r["season"], r["week"])].append(r)
    pg_by_game_team = defaultdict(list)
    for r in player_games:
        pg_by_game_team[(r["game_id"], r["team"])].append(r)
    team_ordinal = defaultdict(int)                                          # team -> completed team games so far (all seasons, FBS or FCS opponent)
    last_seen = {}                                                           # player -> (team, team_ordinal_at_appearance, position, (season, week))
    seen_teams = defaultdict(set); seen_team_season = {}                     # newcomer audit state: teams a player has appeared for; (player, team) -> last season
    out, cov = [], {s: {"team_games": 0, "candidate_rows": 0, "actual_producers": 0, "producers_in_candidates": 0, "newcomer_producers": 0, "participants_in_candidates": 0} for s in target_seasons}
    for sw in sorted(games_by_week):
        batch = sorted(games_by_week[sw], key=lambda r: (r["game_id"], r["team"]))
        season, week = sw
        # ---- candidates for every team-game of this week from state as of the END of the previous week (no same-week information)
        for tg in batch:
            if season not in target_seasons or not (tg["team_div"] == "fbs" and tg["opp_div"] == "fbs"):
                continue
            ordn = team_ordinal[tg["team"]]
            cands = [(p, v) for p, v in last_seen.items() if v[0] == tg["team"] and ordn - v[1] <= RECENCY_TEAM_GAMES and pos_class(v[2])]
            actual = {r["player_id"]: r for r in pg_by_game_team.get((tg["game_id"], tg["team"]), [])}
            cset = {p for p, _ in cands}
            c = cov[season]; c["team_games"] += 1; c["candidate_rows"] += len(cands)
            prod = [p for p, r in actual.items() if (r["carries"] or 0) + (r["receptions"] or 0) + (r["pass_attempts"] or 0) > 0 and pos_class(r["position"])]
            if missing_out is not None:
                for p in prod:
                    if p not in cset:
                        a = actual[p]
                        missing_out.append({"season": season, "week": week, "game_id": tg["game_id"], "team": tg["team"], "player_id": p, "position": a["position"], "prior_teams": sorted(seen_teams[p]), "appeared_before_anywhere": bool(seen_teams[p]), "appeared_for_team_before": tg["team"] in seen_teams[p],
                                            "appeared_for_team_last_season": seen_team_season.get((p, tg["team"])) == season - 1, "last_seen_team": (last_seen.get(p) or (None,))[0],
                                            "carries": a["carries"] or 0, "receptions": a["receptions"] or 0, "pass_att": a["pass_attempts"] or 0, "rush_yds": a["rushing_yards"] or 0, "rec_yds": a["receiving_yards"] or 0, "pass_yds": a["passing_yards"] or 0,
                                            "rush_td": a["rushing_touchdowns"] or 0, "rec_td": a["receiving_touchdowns"] or 0, "pass_td": a["passing_touchdowns"] or 0})
            c["actual_producers"] += len(prod); c["producers_in_candidates"] += sum(p in cset for p in prod); c["newcomer_producers"] += sum(p not in cset for p in prod)
            for p, v in sorted(cands, key=lambda x: x[0]):
                a = actual.get(p)
                c["participants_in_candidates"] += int(a is not None and (a["carries"] or 0) + (a["receptions"] or 0) + (a["pass_attempts"] or 0) > 0)
                out.append({"season": season, "week": week, "game_id": tg["game_id"], "team": tg["team"], "opponent": tg["opponent"], "player_id": p, "position": v[2], "last_team_ordinal": v[1], "team_ordinal": ordn,
                            "y_carries": (a or {}).get("carries") or 0, "y_rush_yards": (a or {}).get("rushing_yards") or 0, "y_receptions": (a or {}).get("receptions") or 0, "y_rec_yards": (a or {}).get("receiving_yards") or 0,
                            "y_pass_att": (a or {}).get("pass_attempts") or 0, "y_completions": (a or {}).get("completions") or 0, "y_pass_yards": (a or {}).get("passing_yards") or 0,
                            "y_pass_td": (a or {}).get("passing_touchdowns") or 0, "y_rush_td": (a or {}).get("rushing_touchdowns") or 0, "y_rec_td": (a or {}).get("receiving_touchdowns") or 0, "y_int": (a or {}).get("passing_interceptions") or 0})
        # ---- only now fold this week's games into the state
        for tg in batch:
            team_ordinal[tg["team"]] += 1
        for tg in batch:
            for r in pg_by_game_team.get((tg["game_id"], tg["team"]), []):
                last_seen[r["player_id"]] = (tg["team"], team_ordinal[tg["team"]], r["position"], sw)
                seen_teams[r["player_id"]].add(tg["team"]); seen_team_season[(r["player_id"], tg["team"])] = season
    return out, cov


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["freeze"]); ap.add_argument("--raw", required=True); ap.add_argument("--db", default=str(REPO / "cfb_models" / "cfb_model.sqlite")); a = ap.parse_args()
    man = freeze(a.raw, a.db)
    print(json.dumps({k: v["rows"] for k, v in man["files"].items()}, indent=0), man["manifest_content_sha256"])


if __name__ == "__main__":
    main()
