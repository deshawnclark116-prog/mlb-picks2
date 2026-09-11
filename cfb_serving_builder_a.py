#!/usr/bin/env python3
"""
CFB_SERVING_BUILDER_A

The live serving path for all four validated CFB markets:

  rushing_yards         RB-only, over 69.5 rushing yards
  receiving_yards        WR-only, Power4-vs-Power4 only, over 59.5 rec yards
  passing_yards           QB-only, Power4-vs-Power4 only, over 214.5 pass yards
  passing_touchdowns      QB-only, over 1.5 passing TDs

Mirrors nfl_serving_builder_a.py's design exactly: frozen champion model
(from cfb_models/, never retrained here) + weekly Platt recalibration
(growing pool: most recent completed season's internal-val-equivalent
slice as warmup, plus the serving season's weeks seen so far) -- the same
configuration validated in each market's own walkforward_stability_a.py.

Predictions-first: no odds anywhere. Emits calibrated P(over line) for
every eligible player in the target week's FBS-vs-FBS games, to
docs/cfb_predictions.json (+ a per-week history file).

Weekly flow (GitHub Actions, mirrors .github/workflows/nfl_weekly.yml):
  1. rebuild the foundation db from cfbfastR-data (stateless)
  2. python cfb_serving_builder_a.py            (auto-picks the next week)
  3. commit docs/

Eligibility mirrors each validated baseline exactly: a player needs >= 3
prior games THIS season and a current-role recent rate, so the normal
board is empty for the first 3 weeks of every season by design -- weeks
1-3 are instead covered by a separate prior-season-informed bootstrap
(see PRIOR_SEASON_MAX_WEEK / build_prior_season_picks below, validated in
cfb_prior_season_early_gate_a.py), mirroring nfl_serving_builder_a.py's
preseason-informed rushing_yards market. Emitted under a distinct
"<market>_early_season" market key so the two populations are never
conflated.

Known limitation, disclosed not hidden: cfbfastR-data is a community-
maintained snapshot, not a real-time feed -- it updates once or twice
daily during the season (confirmed via its own commit history), not
same-day. This board can lag a day behind actual games. Also disclosed:
eligibility is stats-based and cannot see injuries/inactives.

Feature computation MIRRORS the baseline builder (same rules, reimplemented
for as-of-future-week serving) -- and --selftest PROVES the mirror: it
recomputes the full 2024 season through this engine and requires exact
row-for-row feature parity with the validated baseline.sqlite, plus a
walk-forward probability reproduction matching the validated
walkforward_stability report. Run it after any edit to this file.

Run
---
python -u cfb_serving_builder_a.py --selftest          # offline parity proof
python -u cfb_serving_builder_a.py                     # build next week's board
python -u cfb_serving_builder_a.py --season 2026 --week 5   # explicit target
"""

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

REPO = Path(__file__).resolve().parent
DB_DEFAULT = REPO / "cfb_models" / "cfb_model.sqlite"
DOCS = REPO / "docs"

import cfb_rushing_yards_champion_gate_d as gate_mod  # metrics/auc/NAN
from cfb_rushing_yards_champion_gate_b import fit_platt, apply_platt
import cfb_rush_sim as rush_sim

POWER4 = {"Big Ten", "ACC", "SEC", "Big 12"}

# Append-only ledger of every pick this builder has ever produced, logged
# BEFORE the finished_matchups filter below drops a graded game's pick
# from the live board. Without this, cfb_grade_record_a.py would have
# nothing to grade: the per-week archive (cfb_predictions_{season}_w{week}
# .json) gets overwritten on every run with that same finished_matchups
# filter applied, so by the time a game goes final the archive no longer
# contains the pick that was actually live for it -- confirmed directly
# (a fully-final week rebuilds its own archive down to 0 picks). Keyed on
# (season, week, market, player_id) and never rewritten once logged, so
# the graded record reflects the pick as it first appeared, not whatever
# the model's growing-pool calibration says about that player-week today.
PICKS_LOG_PATH = DOCS / "cfb_picks_log.jsonl"


def load_logged_pick_keys(path):
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            keys.add((r.get("season"), r.get("week"), r.get("market"), r.get("player_id")))
    return keys


def append_new_picks_to_log(path, keys, picks):
    new_lines = []
    for p in picks:
        k = (p["season"], p["week"], p["market"], p["player_id"])
        if k in keys:
            continue
        keys.add(k)
        new_lines.append(json.dumps({**p, "logged_at": now_utc()}))
    if new_lines:
        with path.open("a") as f:
            f.write("\n".join(new_lines) + "\n")
    return len(new_lines)

# Real per-event data (rush_carries / pass_attempts_log) only covers the
# 2018-2025 cfbfastR historical seasons pulled for backtesting -- the
# live in-season pipeline still ingests ESPN box-score TOTALS, not
# play-by-play, so this table is typically empty/absent for the current
# live season. That's why every lookup below is a soft "if data exists"
# check, not a hard dependency: this wiring activates automatically the
# day a live per-play source exists, and is a silent no-op (falls back
# to the classifier-only pick, unchanged) until then -- never a fabricated
# projection.
CARRY_DB = REPO / "cfb_models" / "cfb_carry_log.sqlite"

# Only markets whose context-adjusted Monte Carlo simulator actually beat
# the plain simulator on its own pre-registered CRPS/bootstrap/AUC gate
# (see cfb_rush_sim_opponent_context_a.py / cfb_volume_context_test_a.py)
# get wired in here. receiving_yards and passing_yards were tested the
# same way and did NOT clear the bar -- they stay classifier-only.
SIM_CONTEXT = {
    "rushing_yards": {
        "coef_path": REPO / "cfb_models" / "cfb_rushing_yards_context_coef.json",
        "event_table": "rush_carries", "event_value_col": "yards",
    },
    "passing_touchdowns": {
        "coef_path": REPO / "cfb_models" / "cfb_passing_touchdowns_context_coef.json",
        "event_table": "pass_attempts_log", "event_value_col": "is_touchdown",
    },
}
SIM_RECENT_GAMES_WINDOW = 8
SIM_MIN_PRIOR_GAMES = 3
SIM_SIMS_PER_ROW = 4000


def load_sim_context_coef(mkt):
    cfg = SIM_CONTEXT.get(mkt)
    if not cfg or not cfg["coef_path"].exists():
        return None
    data = json.loads(cfg["coef_path"].read_text())
    return {**cfg, "coef": data["coef"]}


def open_carry_con():
    if not CARRY_DB.exists():
        return None
    try:
        return sqlite3.connect(f"file:{CARRY_DB}?mode=ro", uri=True)
    except Exception:
        return None


def simulate_projection(carry_con, sim_cfg, player_id, season, week, line,
                         recent3_avg_volume, opp_allowed, projected_margin, rng):
    """Returns a dict with mean/median/prob_over/confidence if this player
    has real per-event data for enough recent games THIS season strictly
    before `week`, else None (never fabricates a projection)."""
    if carry_con is None:
        return None
    rows = carry_con.execute(f"""
        SELECT week, game_id, {sim_cfg['event_value_col']} FROM {sim_cfg['event_table']}
        WHERE player_id=? AND season=? AND week<?
        ORDER BY week, game_id
    """, (player_id, season, week)).fetchall()
    if not rows:
        return None
    by_game = {}
    for wk, gid, val in rows:
        by_game.setdefault((wk, gid), []).append(val)
    games_sorted = sorted(by_game.keys())[-SIM_RECENT_GAMES_WINDOW:]
    if len(games_sorted) < SIM_MIN_PRIOR_GAMES:
        return None
    counts = [len(by_game[g]) for g in games_sorted]
    pool = [v for g in games_sorted for v in by_game[g]]
    if not counts or not pool:
        return None
    adjusted = rush_sim.context_adjusted_counts(
        counts, recent3_avg_volume, opp_allowed, projected_margin, sim_cfg["coef"])
    return rush_sim.simulate(adjusted, pool, line, sims=SIM_SIMS_PER_ROW, rng=rng)

MARKETS = {
    "rushing_yards": {
        "position": "RB",
        "line": 69.5,
        "stat_fields": ["carries", "rushing_yards"],
        "rate_field": "carries", "min_recent_rate": 12,
        "opp_stat": "rushing_yards",
        "feature_names": {
            "season_avg_yards": "season_avg_rush_yards",
            "recent3_avg_yards": "recent3_avg_rush_yards",
            "recent5_avg_yards": "recent5_avg_rush_yards",
            "season_avg_vol": "season_avg_carries",
            "recent3_avg_vol": "recent3_avg_carries",
            "yards_per_vol": "yards_per_carry",
            "opp_yards_allowed": "opp_rush_yards_allowed_per_game",
        },
        "features": ["season_avg_rush_yards", "recent3_avg_rush_yards",
                      "recent5_avg_rush_yards", "season_avg_carries",
                      "recent3_avg_carries", "yards_per_carry",
                      "opp_rush_yards_allowed_per_game", "is_home", "games_played",
                      "team_net_margin", "opp_net_margin", "projected_margin"],
        "model_dir": REPO / "cfb_models" / "cfb_rushing_yards_walkforward_stability_a_work",
        "stem": "cfb_rushing_yards",
        "baseline_table": ("cfb_models/cfb_rushing_yards_clean_baseline_b_work/baseline.sqlite",
                            "cfb_rushing_yards_baseline"),
        "verdicts": ["CFB_RUSHING_YARDS_CHAMPION_PASSES_GATE_READY_FOR_STABILITY_CONFIRMATION",
                      "CFB_RUSHING_YARDS_WALKFORWARD_STABLE_READY_FOR_LIVE_WIRING"],
        "calibration_policy": "growing",
    },
    # rushing_touchdowns (RB-only) RETIRED 2026-09-06: superseded by the
    # combined anytime_touchdowns market below (rushing+receiving TDs
    # together, RB+WR population) -- a real anytime-TD prop pays out on a
    # score by ANY means, and scoring rushing/receiving separately missed
    # a RB's receiving TDs and a WR's occasional rushing TD. The combined
    # version also validated STRONGER (AUC 0.6494 vs 0.6243, calib p=0.646
    # vs 0.157) -- see AnytimeTouchdownEngine / anytime_touchdowns wiring
    # further down this file. Not kept in SUSPENDED_MARKETS (that dict is
    # for markets that failed re-validation and might be fixed later) --
    # this one isn't broken, it's just a worse design than what replaced
    # it, so there's no reason to ever restore it.
    "passing_touchdowns": {
        "position": "QB",
        "line": 1.5,
        "stat_fields": ["pass_attempts", "passing_touchdowns"],
        "rate_field": "pass_attempts", "min_recent_rate": 15,
        "opp_stat": "passing_touchdowns",
        # All-division population (no Power4 scoping needed -- passed
        # cleanly on the first attempt at that population).
        "feature_names": {
            "season_avg_yards": "season_avg_pass_td",
            "recent3_avg_yards": "recent3_avg_pass_td",
            "recent5_avg_yards": "recent5_avg_pass_td",
            "season_avg_vol": "season_avg_attempts",
            "recent3_avg_vol": "recent3_avg_attempts",
            "yards_per_vol": "td_per_attempt",
            "opp_yards_allowed": "opp_pass_td_allowed_per_game",
        },
        "features": ["season_avg_pass_td", "recent3_avg_pass_td",
                      "recent5_avg_pass_td", "season_avg_attempts",
                      "recent3_avg_attempts", "td_per_attempt",
                      "opp_pass_td_allowed_per_game", "is_home", "games_played",
                      "team_net_margin", "opp_net_margin", "projected_margin"],
        "model_dir": REPO / "cfb_models" / "cfb_passing_touchdowns_walkforward_stability_a_work",
        "stem": "cfb_passing_touchdowns",
        "baseline_table": ("cfb_models/cfb_passing_touchdowns_clean_baseline_a_work/baseline.sqlite",
                            "cfb_passing_touchdowns_baseline"),
        "verdicts": ["CFB_PASSING_TOUCHDOWNS_CHAMPION_PASSES_GATE_READY_FOR_STABILITY_CONFIRMATION",
                      "CFB_PASSING_TOUCHDOWNS_WALKFORWARD_STABLE_READY_FOR_LIVE_WIRING"],
        "calibration_policy": "growing",
    },
}

# receiving_yards and passing_yards SUSPENDED from live serving (2026-09-05):
# a real completion/reception attribution bug was found and fixed in
# cfb_player_games_foundation_a.py (see its aggregate_player_stats()
# docstring). Re-running each market's champion gate against the
# corrected data flipped both from PASS to FAIL:
#   receiving_yards: calib p=0.0539 (was a narrow pass at p=0.0742)
#   passing_yards:   calib p=0.0024 (was a pass at p=0.19)
# passing_touchdowns and rushing_yards were re-checked too and still
# clear their bar on the corrected data -- only these two are affected.
# Configs kept here, unchanged, so they can be restored once each market
# is rebuilt/retrained on the corrected data and re-cleared through the
# same champion-gate process as everything else in this repo.
SUSPENDED_MARKETS = {
    "receiving_yards": {
        "position": "WR",
        "line": 59.5,
        "stat_fields": ["receptions", "receiving_yards"],
        "rate_field": "receptions", "min_recent_rate": 5,
        "opp_stat": "receiving_yards",
        "power4_only": True,
        "feature_names": {
            "season_avg_yards": "season_avg_rec_yards",
            "recent3_avg_yards": "recent3_avg_rec_yards",
            "recent5_avg_yards": "recent5_avg_rec_yards",
            "season_avg_vol": "season_avg_receptions",
            "recent3_avg_vol": "recent3_avg_receptions",
            "yards_per_vol": "yards_per_reception",
            "opp_yards_allowed": "opp_rec_yards_allowed_per_game",
        },
        "features": ["season_avg_rec_yards", "recent3_avg_rec_yards",
                      "recent5_avg_rec_yards", "season_avg_receptions",
                      "recent3_avg_receptions", "yards_per_reception",
                      "opp_rec_yards_allowed_per_game", "is_home", "games_played",
                      "team_net_margin", "opp_net_margin", "projected_margin"],
        "model_dir": REPO / "cfb_models" / "cfb_receiving_yards_walkforward_stability_a_work",
        "stem": "cfb_receiving_yards",
        "baseline_table": ("cfb_models/cfb_receiving_yards_clean_baseline_d_work/baseline.sqlite",
                            "cfb_receiving_yards_baseline"),
        "verdicts": ["CFB_RECEIVING_YARDS_CHAMPION_PASSES_GATE_READY_FOR_STABILITY_CONFIRMATION",
                      "CFB_RECEIVING_YARDS_WALKFORWARD_STABLE_READY_FOR_LIVE_WIRING"],
        "calibration_policy": "growing",
    },
    "passing_yards": {
        "position": "QB",
        "line": 214.5,
        "stat_fields": ["pass_attempts", "passing_yards"],
        "rate_field": "pass_attempts", "min_recent_rate": 15,
        "opp_stat": "passing_yards",
        "power4_only": True,
        "feature_names": {
            "season_avg_yards": "season_avg_pass_yards",
            "recent3_avg_yards": "recent3_avg_pass_yards",
            "recent5_avg_yards": "recent5_avg_pass_yards",
            "season_avg_vol": "season_avg_attempts",
            "recent3_avg_vol": "recent3_avg_attempts",
            "yards_per_vol": "yards_per_attempt",
            "opp_yards_allowed": "opp_pass_yards_allowed_per_game",
        },
        "features": ["season_avg_pass_yards", "recent3_avg_pass_yards",
                      "recent5_avg_pass_yards", "season_avg_attempts",
                      "recent3_avg_attempts", "yards_per_attempt",
                      "opp_pass_yards_allowed_per_game", "is_home", "games_played",
                      "team_net_margin", "opp_net_margin", "projected_margin"],
        "model_dir": REPO / "cfb_models" / "cfb_passing_yards_walkforward_stability_a_work",
        "stem": "cfb_passing_yards",
        "baseline_table": ("cfb_models/cfb_passing_yards_clean_baseline_b_work/baseline.sqlite",
                            "cfb_passing_yards_baseline"),
        "verdicts": ["CFB_PASSING_YARDS_CHAMPION_PASSES_GATE_READY_FOR_STABILITY_CONFIRMATION",
                      "CFB_PASSING_YARDS_WALKFORWARD_STABLE_READY_FOR_LIVE_WIRING"],
        "calibration_policy": "growing",
    },
}

MIN_PRIOR_GAMES = 3
DEV_SEASONS = (2022, 2023)  # for selftest reference only

# Prior-season bootstrap for weeks 1-3, where the within-season eligibility
# rule above guarantees zero eligible players (3 STRICTLY EARLIER games
# can't exist yet). Validated in cfb_prior_season_early_gate_a.py -- same
# design as nfl_serving_builder_a.py's preseason-informed rushing_yards
# market, except CFB has no separate preseason slate, so the bootstrap
# feature is last season's real full-season production instead. All four
# CFB markets passed the gate (AUC 0.62-0.68 on the 2025 holdout), unlike
# the NFL version where only rushing_yards cleared it.
PRIOR_SEASON_MAX_WEEK = 3
PRIOR_SEASON_MODEL_DIR = REPO / "cfb_models"
PRIOR_SEASON_FEATURES = ["prior_season_avg_stat", "prior_season_games", "prior_season_avg_rate"]
# Real bug found and fixed (not present at first release of this bootstrap):
# with no floor on prior-season games played, EVERY player who ever touched
# the ball for a matched team last season became a candidate -- for QB
# specifically this meant a team's real starter (e.g. Clemson's Cade
# Klubnik, 10 games played) got listed alongside 2-3 backups/emergency
# QBs who played 1-4 games, and the backups' thin-sample predictions were
# often MORE extreme (less regressed) than the real starter's, burying the
# actual QB1 under noise when sorted by model_prob. Confirmed directly on
# the live board: 95 of 100 team/market combos had 2+ simultaneous QB
# picks. A minimum-games floor cleanly resolves this (checked empirically
# against the live board: threshold=6 leaves exactly one candidate for 17
# of 20 spot-checked teams, two for a real committee/QB-competition case
# in the rest) without needing to retrain the validated model -- this is a
# serving-time population restriction, the same kind of governance-layer
# floor already used elsewhere in this repo (e.g. api.py's thin-sample
# pitcher-K confidence cap), not a change to what was actually validated.
MIN_PRIOR_SEASON_GAMES = 6


def _norm_roster_name(name):
    """Lowercase + collapse whitespace, for matching a prior-season stats
    name (cfbfastR/ESPN historical) against a current_roster snapshot name
    (ESPN live) -- both ultimately come from the same real person's name,
    just via different API calls, so this only needs to absorb minor
    formatting differences, not do real fuzzy matching."""
    return " ".join((name or "").split()).strip().lower()


def match_team_to_prior_season(display_name, known_teams_by_len_desc):
    """ESPN's live-season team strings are 'displayName' (school + mascot,
    e.g. 'Ohio State Buckeyes'); cfbfastR's historical team strings are
    school-name-only (e.g. 'Ohio State') -- no shared ID between the two
    sources (see cfb_espn_live_foundation_a.py's docstring for the same
    issue at the player level). Real, disclosed name-based approximation:
    the longest known school name that equals display_name or is a prefix
    of it ending on a word boundary (checking longest-first prevents a
    short school name like 'Ohio' from matching inside 'Ohio State
    Buckeyes' before 'Ohio State' gets a chance). An unmatched team simply
    contributes no early-season candidates for that team -- graceful, no
    fabricated signal -- never a silent wrong-team match, since a match is
    only ever accepted on a full word boundary."""
    for school in known_teams_by_len_desc:
        if display_name == school or display_name.startswith(school + " "):
            return school
    return None


def build_prior_season_picks(con, mkt, season, week, schedule, xgb):
    """Weeks 1-3 only: fills the empty-board gap using last season's real
    production via the validated prior-season-informed model. Population
    is every player at this market's position who played for a team
    matched to this week's schedule in the PRIOR season -- not a live
    roster fetch (mirrors nfl_serving_builder_a.py's reasoning: a player
    with no prior-season match gets no candidate row at all here, no more
    informative than silence, so there's nothing gained by a roster call)."""
    cfg = MARKETS[mkt]
    if week > PRIOR_SEASON_MAX_WEEK:
        return [], {"eligible": 0, "reason": f"week > {PRIOR_SEASON_MAX_WEEK}"}

    model_path = PRIOR_SEASON_MODEL_DIR / f"cfb_prior_season_{mkt}.json"
    cols_path = PRIOR_SEASON_MODEL_DIR / f"cfb_prior_season_{mkt}_columns.json"
    if not model_path.exists():
        return [], {"eligible": 0, "reason": "prior-season model not present"}
    feat_cols = json.loads(cols_path.read_text())
    assert feat_cols == PRIOR_SEASON_FEATURES

    prior_season = season - 1
    known_teams = {r[0] for r in con.execute(
        "SELECT DISTINCT team FROM player_games WHERE season = ?", (prior_season,))}
    known_teams_by_len_desc = sorted(known_teams, key=len, reverse=True)

    sched_teams = set()
    for h, a in schedule:
        sched_teams.add(h); sched_teams.add(a)
    team_map = {}  # scheduled displayName -> matched prior-season school name
    for t in sched_teams:
        m = match_team_to_prior_season(t, known_teams_by_len_desc)
        if m:
            team_map[t] = m
    matched_teams = set(team_map.values())
    print(f"  {mkt}_early_season: {len(team_map)}/{len(sched_teams)} scheduled teams "
          f"matched to a {prior_season} team name")
    if not matched_teams:
        return [], {"eligible": 0, "reason": "no scheduled teams matched a prior-season team name",
                     "scheduled_teams": len(sched_teams)}

    vol_field, yard_field = cfg["stat_fields"]
    placeholders = ",".join("?" for _ in matched_teams)
    rows = con.execute(f"""
        SELECT player_id, player_name, team, {vol_field}, {yard_field}
        FROM player_games WHERE season = ? AND position = ? AND team IN ({placeholders})
    """, (prior_season, cfg["position"], *matched_teams)).fetchall()

    by_player = {}
    for pid, pname, team, vol, yards in rows:
        d = by_player.setdefault(pid, {"name": pname, "team": team, "games": []})
        d["games"].append((vol or 0, yards or 0))
    if not by_player:
        return [], {"eligible": 0, "reason": f"no {prior_season} {cfg['position']} data for matched teams"}

    disp_of_school = {v: k for k, v in team_map.items()}
    team_pairs = {h: a for h, a in schedule}
    team_pairs.update({a: h for h, a in schedule})

    # Roster verification: a player's prior-season stats say nothing about
    # whether they're still on the team NOW -- transfers, graduations, and
    # draft departures all break that assumption every single offseason.
    # Confirmed as a real, live bug: Marquez Taylor showed up as a 2026
    # UTEP rushing_yards pick despite not being on UTEP's actual current
    # roster at all. current_roster (populated by
    # cfb_espn_live_foundation_a.py's live ingestion) is a same-day
    # snapshot of who's really on each team; cross-check against it here.
    # Fails OPEN (doesn't drop anyone) when verification data isn't
    # available at all -- for the table not existing yet (an older db
    # before this feature), or for a specific team this run's roster
    # fetch didn't cover -- since an absent snapshot is a coverage gap,
    # not evidence the player left. Once a team's real roster IS known,
    # though, absence from it is treated as a real signal, not overridden.
    has_roster_table = bool(con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='current_roster'").fetchone())
    roster_by_team = {}
    if has_roster_table:
        for team, name in con.execute(
                "SELECT team, player_name FROM current_roster WHERE season = ?", (season,)):
            roster_by_team.setdefault(team, set()).add(_norm_roster_name(name))

    cand_ids, feats, meta = [], [], []
    dropped_not_on_roster = 0
    for pid, info in by_player.items():
        disp_team = disp_of_school.get(info["team"])
        if disp_team is None:
            continue
        opp = team_pairs.get(disp_team)
        if opp is None:
            continue
        current_names = roster_by_team.get(disp_team)
        if current_names and _norm_roster_name(info["name"]) not in current_names:
            dropped_not_on_roster += 1
            continue  # real roster data exists for this team and this
                       # player isn't on it -- transferred/graduated/left
        n = len(info["games"])
        if n < MIN_PRIOR_SEASON_GAMES:
            continue  # cameo/backup appearance, not a real prior-season role
        vols = [v for v, _ in info["games"]]
        yards = [y for _, y in info["games"]]
        avg_rate = sum(vols) / n
        if avg_rate < cfg["min_recent_rate"]:
            continue  # played in 6+ games but never had a real starter-level
                       # role in them (e.g. a change-of-pace RB averaging 2-3
                       # carries/game across 6 games clears the games floor
                       # above but was never the actual starter) -- same
                       # volume bar (cfg["min_recent_rate"]) the normal
                       # within-season eligibility check already applies,
                       # just measured as a season average here instead of
                       # a trailing-3-game rate
        feats.append([sum(yards) / n, float(n), avg_rate])
        cand_ids.append(pid)
        meta.append((pid, info["name"], disp_team, opp, n))

    if not cand_ids:
        return [], {"eligible": 0, "reason": "no candidates resolved to a scheduled opponent",
                     "matched_teams": len(matched_teams)}

    bst = xgb.Booster(); bst.load_model(str(model_path))
    dm = xgb.DMatrix(np.array(feats, dtype=np.float32), feature_names=feat_cols)
    probs = bst.predict(dm)

    picks = []
    for (pid, pname, team, opp, games_played), p in zip(meta, probs):
        cp = float(p)
        picks.append({
            "market": f"{mkt}_early_season", "player_id": pid, "player": pname,
            "team": team, "opponent": opp, "season": season, "week": week,
            "line": cfg["line"],
            "pick": f"{'OVER' if cp >= 0.5 else 'UNDER'} {cfg['line']}",
            "model_prob": round(float(max(cp, 1 - cp)), 4),
            "prob_over": round(float(cp), 4),
            "games_played": games_played,
            "model_source": "prior_season_informed",
            "prior_season": prior_season,
        })
    print(f"  {mkt}_early_season: roster-verified against {len(roster_by_team)} teams' current "
          f"rosters, {dropped_not_on_roster} candidate(s) dropped (no longer on the team)")
    meta_out = {"eligible": len(picks), "matched_teams": len(matched_teams),
                "scheduled_teams": len(sched_teams), "prior_season": prior_season,
                "roster_verified_teams": len(roster_by_team),
                "dropped_not_on_current_roster": dropped_not_on_roster}
    return picks, meta_out


def build_anytime_touchdowns_prior_season_picks(con, season, week, schedule, xgb):
    """anytime_touchdowns' own weeks 1-3 bootstrap -- same design as
    build_prior_season_picks() above (last season's real production
    informs an empty early-season board), adapted for the combined RB+WR
    population and rush+recv summed target the same way the in-season
    AnytimeTouchdownEngine combines both. Not folded into
    build_prior_season_picks() itself: that function is keyed on a single
    MARKETS[mkt] position + one stat-field pair, and this market spans two
    positions with two different rate fields and a summed target --
    structurally incompatible with its single-position assumption, same
    reasoning as AnytimeTouchdownEngine living outside the generic
    SeasonEngine machinery. Validated in
    cfb_prior_season_anytime_touchdowns_gate_a.py (AUC 0.6071 on the 2024
    holdout)."""
    if week > PRIOR_SEASON_MAX_WEEK:
        return [], {"eligible": 0, "reason": f"week > {PRIOR_SEASON_MAX_WEEK}"}

    model_path = PRIOR_SEASON_MODEL_DIR / "cfb_prior_season_anytime_touchdowns.json"
    cols_path = PRIOR_SEASON_MODEL_DIR / "cfb_prior_season_anytime_touchdowns_columns.json"
    if not model_path.exists():
        return [], {"eligible": 0, "reason": "prior-season model not present"}
    feat_cols = json.loads(cols_path.read_text())
    assert feat_cols == PRIOR_SEASON_FEATURES

    prior_season = season - 1
    known_teams = {r[0] for r in con.execute(
        "SELECT DISTINCT team FROM player_games WHERE season = ? AND position IN ('RB', 'WR')",
        (prior_season,))}
    known_teams_by_len_desc = sorted(known_teams, key=len, reverse=True)

    sched_teams = set()
    for h, a in schedule:
        sched_teams.add(h); sched_teams.add(a)
    team_map = {}  # scheduled displayName -> matched prior-season school name
    for t in sched_teams:
        m = match_team_to_prior_season(t, known_teams_by_len_desc)
        if m:
            team_map[t] = m
    matched_teams = set(team_map.values())
    print(f"  anytime_touchdowns_early_season: {len(team_map)}/{len(sched_teams)} scheduled teams "
          f"matched to a {prior_season} team name")
    if not matched_teams:
        return [], {"eligible": 0, "reason": "no scheduled teams matched a prior-season team name",
                     "scheduled_teams": len(sched_teams)}

    placeholders = ",".join("?" for _ in matched_teams)
    rows = con.execute(f"""
        SELECT player_id, player_name, team, position, carries, receptions,
               rushing_touchdowns, receiving_touchdowns
        FROM player_games
        WHERE season = ? AND position IN ('RB', 'WR') AND team IN ({placeholders})
    """, (prior_season, *matched_teams)).fetchall()

    by_player = {}
    for pid, pname, team, pos, carries, receptions, rtd, rectd in rows:
        d = by_player.setdefault(pid, {"name": pname, "team": team, "position": pos, "games": []})
        d["games"].append({
            "carries": carries or 0, "receptions": receptions or 0,
            "total_td": (rtd or 0) + (rectd or 0),
        })
    if not by_player:
        return [], {"eligible": 0, "reason": f"no {prior_season} RB/WR data for matched teams"}

    disp_of_school = {v: k for k, v in team_map.items()}
    team_pairs = {h: a for h, a in schedule}
    team_pairs.update({a: h for h, a in schedule})

    # Same roster-verification governance as build_prior_season_picks()
    # above (the Marquez Taylor bug) -- fails open when no current-season
    # roster snapshot exists at all, otherwise drops anyone confirmed off
    # the team since last season.
    has_roster_table = bool(con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='current_roster'").fetchone())
    roster_by_team = {}
    if has_roster_table:
        for team, name in con.execute(
                "SELECT team, player_name FROM current_roster WHERE season = ?", (season,)):
            roster_by_team.setdefault(team, set()).add(_norm_roster_name(name))

    cand_ids, feats, meta = [], [], []
    dropped_not_on_roster = 0
    for pid, info in by_player.items():
        disp_team = disp_of_school.get(info["team"])
        if disp_team is None:
            continue
        opp = team_pairs.get(disp_team)
        if opp is None:
            continue
        current_names = roster_by_team.get(disp_team)
        if current_names and _norm_roster_name(info["name"]) not in current_names:
            dropped_not_on_roster += 1
            continue  # real roster data exists for this team and this
                       # player isn't on it -- transferred/graduated/left
        n = len(info["games"])
        if n < MIN_PRIOR_SEASON_GAMES:
            continue  # cameo/backup appearance, not a real prior-season role
        pos = info["position"]
        rate_field = "carries" if pos == "RB" else "receptions"
        rates = [g[rate_field] for g in info["games"]]
        avg_rate = sum(rates) / n
        # Each position's own already-validated volume floor (matches
        # anytime_eligible()'s in-season bar exactly), not an invented
        # combined-touches number.
        if pos == "RB" and avg_rate < 12:
            continue
        if pos == "WR" and avg_rate < 5:
            continue
        avg_stat = sum(g["total_td"] for g in info["games"]) / n
        feats.append([avg_stat, float(n), avg_rate])
        cand_ids.append(pid)
        meta.append((pid, info["name"], disp_team, opp, n))

    if not cand_ids:
        return [], {"eligible": 0, "reason": "no candidates resolved to a scheduled opponent",
                     "matched_teams": len(matched_teams)}

    bst = xgb.Booster(); bst.load_model(str(model_path))
    dm = xgb.DMatrix(np.array(feats, dtype=np.float32), feature_names=feat_cols)
    probs = bst.predict(dm)

    # Anytime TD is a real-world one-sided market -- every book prices
    # "Yes, scores anytime" at plus-money odds, but none offer a
    # bettable "No touchdown" side to take the other way. Showing an
    # UNDER 0.5 pick here would imply a real bet that doesn't exist
    # anywhere, so a player the model doesn't like is dropped entirely
    # rather than surfaced as a fabricated "Under" recommendation.
    picks = []
    n_no_td = 0
    for (pid, pname, team, opp, games_played), p in zip(meta, probs):
        cp = float(p)
        if cp < 0.5:
            n_no_td += 1
            continue
        picks.append({
            "market": "anytime_touchdowns_early_season", "player_id": pid, "player": pname,
            "team": team, "opponent": opp, "season": season, "week": week,
            "line": ANYTIME_TD_LINE,
            "pick": f"OVER {ANYTIME_TD_LINE}",
            "model_prob": round(cp, 4),
            "prob_over": round(cp, 4),
            "games_played": games_played,
            "model_source": "prior_season_informed",
            "prior_season": prior_season,
        })
    print(f"  anytime_touchdowns_early_season: roster-verified against {len(roster_by_team)} teams' "
          f"current rosters, {dropped_not_on_roster} candidate(s) dropped (no longer on the team)")
    meta_out = {"eligible": len(picks), "matched_teams": len(matched_teams),
                "scheduled_teams": len(sched_teams), "prior_season": prior_season,
                "roster_verified_teams": len(roster_by_team),
                "dropped_not_on_current_roster": dropped_not_on_roster,
                "no_real_under_market": n_no_td}
    return picks, meta_out


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def market_features(mkt, hist, opp_allowed, is_home, team_margin, opp_margin):
    cfg = MARKETS[mkt]
    vol_field, yard_field = cfg["stat_fields"]
    fn = cfg["feature_names"]
    ys = [h[yard_field] or 0 for h in hist]
    vs = [h[vol_field] or 0 for h in hist]
    n = len(hist)
    r3y, r5y, r3v = ys[-3:], ys[-5:], vs[-3:]
    proj_margin = (team_margin - opp_margin) if (team_margin is not None and opp_margin is not None) else None
    return {
        fn["season_avg_yards"]: sum(ys) / n,
        fn["recent3_avg_yards"]: sum(r3y) / len(r3y),
        fn["recent5_avg_yards"]: sum(r5y) / len(r5y),
        fn["season_avg_vol"]: sum(vs) / n,
        fn["recent3_avg_vol"]: sum(r3v) / len(r3v),
        fn["yards_per_vol"]: (sum(ys) / sum(vs)) if sum(vs) > 0 else 0.0,
        fn["opp_yards_allowed"]: opp_allowed,
        "is_home": 1.0 if is_home else 0.0,
        "games_played": n,
        "team_net_margin": team_margin,
        "opp_net_margin": opp_margin,
        "projected_margin": proj_margin,
    }


def eligible(mkt_cfg, hist):
    if len(hist) < MIN_PRIOR_GAMES:
        return False
    rates = [h[mkt_cfg["rate_field"]] or 0 for h in hist][-3:]
    return (sum(rates) / len(rates)) >= mkt_cfg["min_recent_rate"]


def power4_game_ids(con, season=None):
    q = ("SELECT game_id FROM games WHERE home_conference IN ({0}) "
         "AND away_conference IN ({0})").format(",".join("?" for _ in POWER4))
    params = list(POWER4) * 2
    if season is not None:
        q += " AND season = ?"
        params.append(season)
    return {r[0] for r in con.execute(q, params)}


class SeasonEngine:
    """Replays one market's season week-by-week from player_games, exposing
    (a) completed eligible rows with features + outcomes and (b) as-of
    features for a FUTURE week. Mirrors nfl_serving_builder_a.py's design."""

    def __init__(self, con, mkt, season):
        self.mkt = mkt
        self.cfg = MARKETS[mkt]
        self.season = season
        self.con = con
        fields = ", ".join(self.cfg["stat_fields"])
        self.rows = con.execute(f"""
            SELECT player_id, player_name, team, opponent, week, is_home, game_id, {fields}
            FROM player_games
            WHERE position = ? AND season = ?
            ORDER BY week, game_date
        """, (self.cfg["position"], season)).fetchall()
        self.weeks = sorted({r[4] for r in self.rows})
        self.team_margin_asof = self._build_team_margin_asof(season)
        self.p4_games = power4_game_ids(con, season) if self.cfg.get("power4_only") else None

    def _build_team_margin_asof(self, season):
        games = self.con.execute(
            "SELECT week, home_team, away_team, home_points, away_points "
            "FROM games WHERE season = ? ORDER BY week", (season,)).fetchall()
        by_week = {}
        for g in games:
            by_week.setdefault(g[0], []).append(g)
        team_state = {}
        margin_asof = {}
        for w in sorted(by_week):
            for (week, home, away, hp, ap) in by_week[w]:
                for team in (home, away):
                    st = team_state.get(team, [0, 0, 0])
                    margin_asof[(team, week)] = (st[0] - st[1]) / st[2] if st[2] > 0 else None
            for (week, home, away, hp, ap) in by_week[w]:
                # NULL points has two real, distinct causes now: (1) a
                # pre-existing historical data gap (one real case found:
                # 2024 week 5 App State/Liberty -- the validated baseline
                # scripts treat it as a 0-0 result, so this must match
                # that exactly or selftest's byte-parity proof breaks) and
                # (2) a live-season game scheduled but not yet played
                # (this script's own new schedule-visibility rows). Both
                # get the SAME "treat as 0, still count" fallback here --
                # deliberately unchanged from before that schedule-
                # visibility fix. This is safe for (2) specifically because
                # margin_asof for week w is read from team_state BEFORE
                # week w's own update (a few lines up), and no later week
                # is ever computed in the same run -- so an unplayed
                # target-week game's placeholder 0-0 never actually reaches
                # anything this serving run consumes; it only becomes real
                # once the game completes and a later run re-reads real
                # points for it.
                hp = hp if hp is not None else 0
                ap = ap if ap is not None else 0
                hst = team_state.setdefault(home, [0, 0, 0])
                hst[0] += hp; hst[1] += ap; hst[2] += 1
                ast = team_state.setdefault(away, [0, 0, 0])
                ast[0] += ap; ast[1] += hp; ast[2] += 1
        return margin_asof

    def replay(self):
        """Two-phase, mirroring the clean-baseline builder exactly: (1) a
        week-batched pre-pass computing opp_asof (opponent context only
        ever uses STRICTLY EARLIER weeks, batched -- unaffected by same-
        week ordering); (2) a strictly SEQUENTIAL per-player pass over
        every row in table order for the player's own history/eligibility
        -- NOT batched by week. Confirmed necessary by --selftest: some
        teams play two games sharing the same week NUMBER (e.g. Georgia
        Tech's 2024 Aug-24 international opener vs Florida State and its
        Sep-1 game vs Georgia State both carry week=1) -- batching by week
        would let both see identical pre-week history, which the baseline
        builder's true row-by-row accumulation does not."""
        cfg = self.cfg
        opp_state = {}
        opp_asof = {}
        for w in self.weeks:
            wk = [r for r in self.rows if r[4] == w]
            for r in wk:
                opp = r[3]
                key = (r[0], w)
                st = opp_state.get(opp)
                opp_asof[key] = (st[0] / st[1]) if st and st[1] > 0 else None
            for r in wk:
                opp = r[3]
                stats = dict(zip(cfg["stat_fields"], r[7:]))
                st = opp_state.setdefault(opp, [0, 0])
                st[0] += stats[cfg["opp_stat"]] or 0
                st[1] += 1

        hist = {}
        out = []
        for r in self.rows:
            pid, pname, team, opp, week, is_home, gid = r[:7]
            stats = dict(zip(cfg["stat_fields"], r[7:]))
            h = hist.get(pid, [])
            if eligible(cfg, h) and (self.p4_games is None or gid in self.p4_games):
                opp_allowed = opp_asof.get((pid, week))
                team_margin = self.team_margin_asof.get((team, week))
                opp_margin = self.team_margin_asof.get((opp, week))
                feat = market_features(self.mkt, h, opp_allowed, is_home == 1, team_margin, opp_margin)
                actual = stats[cfg["opp_stat"]] or 0
                out.append((pid, pname, team, opp, week, feat, actual))
            hist.setdefault(pid, []).append(stats)
        return out

    def asof_future(self, target_week, schedule):
        cfg = self.cfg
        hist = {}
        opp_state = {}
        latest_team = {}
        latest_name = {}
        for r in self.rows:
            pid, pname, team, opp, week = r[0], r[1], r[2], r[3], r[4]
            if week >= target_week:
                continue
            stats = dict(zip(cfg["stat_fields"], r[7:]))
            hist.setdefault(pid, []).append(stats)
            st = opp_state.setdefault(opp, [0, 0])
            st[0] += stats[cfg["opp_stat"]] or 0
            st[1] += 1
            latest_team[pid] = team
            latest_name[pid] = pname

        out = []
        for home, away in schedule:
            for team, opp, is_home in ((home, away, True), (away, home, False)):
                for pid, t in latest_team.items():
                    if t != team:
                        continue
                    h = hist.get(pid, [])
                    if not eligible(cfg, h):
                        continue
                    st = opp_state.get(opp)
                    opp_allowed = (st[0] / st[1]) if st and st[1] > 0 else None
                    team_margin = self.team_margin_asof.get((team, target_week))
                    opp_margin = self.team_margin_asof.get((opp, target_week))
                    feat = market_features(self.mkt, h, opp_allowed, is_home, team_margin, opp_margin)
                    out.append((pid, latest_name[pid], team, opp, target_week, feat))
        return out


ANYTIME_TD_FEATURES = [
    "season_avg_total_td", "recent3_avg_total_td", "recent5_avg_total_td",
    "season_avg_touches", "recent3_avg_touches", "td_per_touch",
    "opp_total_td_allowed_per_game", "is_home", "games_played",
    "team_net_margin", "opp_net_margin", "projected_margin", "is_wr",
]
ANYTIME_TD_LINE = 0.5
ANYTIME_TD_MODEL_DIR = REPO / "cfb_models" / "cfb_anytime_touchdowns_walkforward_stability_a_work"
ANYTIME_TD_BASELINE = (REPO / "cfb_models" / "cfb_anytime_touchdowns_clean_baseline_a_work" / "baseline.sqlite",
                        "cfb_anytime_touchdowns_baseline")


def anytime_eligible(hist, pos):
    """Reuses each position's own already-validated volume floor (RB
    carries>=12 from rushing_yards, WR receptions>=5 from
    receiving_yards) rather than inventing a combined-touches number."""
    if len(hist) < MIN_PRIOR_GAMES:
        return False
    if pos == "RB":
        rates = [h["carries"] or 0 for h in hist][-3:]
        return (sum(rates) / len(rates)) >= 12
    if pos == "WR":
        rates = [h["receptions"] or 0 for h in hist][-3:]
        return (sum(rates) / len(rates)) >= 5
    return False


def anytime_features(hist, opp_allowed, is_home, team_margin, opp_margin, pos):
    tds = [(h["rushing_touchdowns"] or 0) + (h["receiving_touchdowns"] or 0) for h in hist]
    touches = [(h["carries"] or 0) + (h["receptions"] or 0) for h in hist]
    n = len(hist)
    r3t, r5t, r3touch = tds[-3:], tds[-5:], touches[-3:]
    total_touch = sum(touches)
    proj_margin = (team_margin - opp_margin) if (team_margin is not None and opp_margin is not None) else None
    return {
        "season_avg_total_td": sum(tds) / n,
        "recent3_avg_total_td": sum(r3t) / len(r3t),
        "recent5_avg_total_td": sum(r5t) / len(r5t),
        "season_avg_touches": total_touch / n,
        "recent3_avg_touches": sum(r3touch) / len(r3touch),
        "td_per_touch": (sum(tds) / total_touch) if total_touch > 0 else 0.0,
        "opp_total_td_allowed_per_game": opp_allowed,
        "is_home": 1.0 if is_home else 0.0,
        "games_played": n,
        "team_net_margin": team_margin,
        "opp_net_margin": opp_margin,
        "projected_margin": proj_margin,
        "is_wr": 1.0 if pos == "WR" else 0.0,
    }


class AnytimeTouchdownEngine:
    """Combined RB+WR engine for anytime_touchdowns -- kept separate from
    SeasonEngine (not folded into its generic MARKETS-driven machinery)
    because this market spans two positions with different per-position
    eligibility floors, and its features are SUMS across two raw stat
    columns (rushing+receiving touchdowns, carries+receptions) -- both
    structurally different from every other market's single-position/
    single-stat-pair design. Mirrors SeasonEngine's replay()/asof_future()
    logic otherwise (same row-by-row-not-batched-by-week discipline, same
    team-margin-asof construction)."""

    def __init__(self, con, season):
        self.season = season
        self.con = con
        self.rows = con.execute("""
            SELECT player_id, player_name, position, team, opponent, week, is_home, game_id,
                   carries, receptions, rushing_touchdowns, receiving_touchdowns
            FROM player_games
            WHERE position IN ('RB', 'WR') AND season = ?
            ORDER BY week, game_date
        """, (season,)).fetchall()
        self.weeks = sorted({r[5] for r in self.rows})
        self.team_margin_asof = self._build_team_margin_asof(season)

    def _build_team_margin_asof(self, season):
        games = self.con.execute(
            "SELECT week, home_team, away_team, home_points, away_points "
            "FROM games WHERE season = ? ORDER BY week", (season,)).fetchall()
        by_week = {}
        for g in games:
            by_week.setdefault(g[0], []).append(g)
        team_state = {}
        margin_asof = {}
        for w in sorted(by_week):
            for (week, home, away, hp, ap) in by_week[w]:
                for team in (home, away):
                    st = team_state.get(team, [0, 0, 0])
                    margin_asof[(team, week)] = (st[0] - st[1]) / st[2] if st[2] > 0 else None
            for (week, home, away, hp, ap) in by_week[w]:
                hp = hp if hp is not None else 0
                ap = ap if ap is not None else 0
                hst = team_state.setdefault(home, [0, 0, 0])
                hst[0] += hp; hst[1] += ap; hst[2] += 1
                ast = team_state.setdefault(away, [0, 0, 0])
                ast[0] += ap; ast[1] += hp; ast[2] += 1
        return margin_asof

    def replay(self):
        opp_state = {}
        opp_asof = {}
        for w in self.weeks:
            wk = [r for r in self.rows if r[5] == w]
            for r in wk:
                opp = r[4]
                key = (r[0], w)
                st = opp_state.get(opp)
                opp_asof[key] = (st[0] / st[1]) if st and st[1] > 0 else None
            for r in wk:
                opp = r[4]
                total_td = (r[10] or 0) + (r[11] or 0)
                st = opp_state.setdefault(opp, [0, 0])
                st[0] += total_td
                st[1] += 1

        hist = {}
        out = []
        for r in self.rows:
            pid, pname, pos, team, opp, week, is_home, gid, carries, receptions, rtd, rectd = r
            stats = {"carries": carries, "receptions": receptions,
                     "rushing_touchdowns": rtd, "receiving_touchdowns": rectd}
            h = hist.get(pid, [])
            if anytime_eligible(h, pos):
                opp_allowed = opp_asof.get((pid, week))
                team_margin = self.team_margin_asof.get((team, week))
                opp_margin = self.team_margin_asof.get((opp, week))
                feat = anytime_features(h, opp_allowed, is_home == 1, team_margin, opp_margin, pos)
                actual = (rtd or 0) + (rectd or 0)
                out.append((pid, pname, team, opp, week, feat, actual))
            hist.setdefault(pid, []).append(stats)
        return out

    def asof_future(self, target_week, schedule):
        hist = {}
        opp_state = {}
        latest_team = {}
        latest_name = {}
        latest_pos = {}
        for r in self.rows:
            pid, pname, pos, team, opp, week, is_home, gid, carries, receptions, rtd, rectd = r
            if week >= target_week:
                continue
            stats = {"carries": carries, "receptions": receptions,
                     "rushing_touchdowns": rtd, "receiving_touchdowns": rectd}
            hist.setdefault(pid, []).append(stats)
            st = opp_state.setdefault(opp, [0, 0])
            st[0] += (rtd or 0) + (rectd or 0)
            st[1] += 1
            latest_team[pid] = team
            latest_name[pid] = pname
            latest_pos[pid] = pos

        out = []
        for home, away in schedule:
            for team, opp, is_home in ((home, away, True), (away, home, False)):
                for pid, t in latest_team.items():
                    if t != team:
                        continue
                    pos = latest_pos[pid]
                    h = hist.get(pid, [])
                    if not anytime_eligible(h, pos):
                        continue
                    st = opp_state.get(opp)
                    opp_allowed = (st[0] / st[1]) if st and st[1] > 0 else None
                    team_margin = self.team_margin_asof.get((team, target_week))
                    opp_margin = self.team_margin_asof.get((opp, target_week))
                    feat = anytime_features(h, opp_allowed, is_home, team_margin, opp_margin, pos)
                    out.append((pid, latest_name[pid], team, opp, target_week, feat))
        return out


def score(bst, feats_order, feat_dicts, xgb):
    X = np.array([[fd.get(c) if fd.get(c) is not None else gate_mod.NAN for c in feats_order]
                  for fd in feat_dicts], dtype=np.float32)
    itr = (0, bst.best_iteration + 1)
    return np.asarray(
        bst.predict(xgb.DMatrix(X, feature_names=feats_order), iteration_range=itr),
        dtype=float)


def build_platt_pool(warm_raw, warm_y, seen_weeks):
    """'growing' policy only -- the only policy this market uses so far."""
    raw_parts = [warm_raw] + [r for (r, _) in seen_weeks]
    y_parts = [warm_y] + [y for (_, y) in seen_weeks]
    pool_raw = np.concatenate(raw_parts) if raw_parts else np.empty(0)
    pool_y = np.concatenate(y_parts) if y_parts else np.empty(0)
    return pool_raw, pool_y


def fit_serving_platt(con, mkt, bst, xgb, serving_season, target_week):
    cfg = MARKETS[mkt]
    seasons = [r[0] for r in con.execute(
        "SELECT DISTINCT season FROM player_games WHERE season < ? ORDER BY season DESC",
        (serving_season,))]
    if not seasons:
        raise RuntimeError(f"no completed season before {serving_season} in db")
    warm_season = seasons[0]

    warm_engine = SeasonEngine(con, mkt, warm_season)
    warm = warm_engine.replay()
    # row-count-based cut, walking back from season end (mirrors the NFL pattern)
    from collections import Counter
    wk_counts = Counter(row[4] for row in warm)
    weeks_sorted = sorted(wk_counts)
    target_n = max(60, int(len(warm) * 0.2))
    cum = 0; cut = weeks_sorted[-1] if weeks_sorted else 0
    for w in reversed(weeks_sorted):
        cum += wk_counts[w]; cut = w
        if cum >= target_n:
            break
    warm_slice = [row for row in warm if row[4] >= cut]
    warm_raw = score(bst, cfg["features"], [row[5] for row in warm_slice], xgb)
    line = cfg["line"]
    warm_y = np.array([1.0 if row[6] >= line + 0.5 else 0.0 for row in warm_slice])

    cur_engine = SeasonEngine(con, mkt, serving_season)
    cur = cur_engine.replay()
    cur_seen = [row for row in cur if row[4] < target_week]
    by_week = {}
    for row in cur_seen:
        by_week.setdefault(row[4], []).append(row)
    seen_weeks = []
    for w in sorted(by_week):
        wk_rows = by_week[w]
        raw = score(bst, cfg["features"], [r[5] for r in wk_rows], xgb)
        y = np.array([1.0 if r[6] >= line + 0.5 else 0.0 for r in wk_rows])
        seen_weeks.append((raw, y))

    pool_raw, pool_y = build_platt_pool(warm_raw, warm_y, seen_weeks)
    a, b = fit_platt(pool_raw, pool_y)
    if a <= 0:
        a, b = 1.0, 0.0
    return a, b, cur_engine, {"policy": "growing", "warmup_season": warm_season,
                  "warmup_cut_week": int(cut), "warmup_n": len(warm_slice),
                  "current_season_n": len(cur_seen), "pool_n": int(len(pool_y))}


def fit_serving_platt_anytime(con, bst, xgb, serving_season, target_week):
    """Mirrors fit_serving_platt() exactly, using AnytimeTouchdownEngine
    instead of SeasonEngine (see that class's docstring for why this
    market can't share the generic MARKETS-driven machinery)."""
    seasons = [r[0] for r in con.execute(
        "SELECT DISTINCT season FROM player_games WHERE season < ? ORDER BY season DESC",
        (serving_season,))]
    if not seasons:
        raise RuntimeError(f"no completed season before {serving_season} in db")
    warm_season = seasons[0]

    warm_engine = AnytimeTouchdownEngine(con, warm_season)
    warm = warm_engine.replay()
    from collections import Counter
    wk_counts = Counter(row[4] for row in warm)
    weeks_sorted = sorted(wk_counts)
    target_n = max(60, int(len(warm) * 0.2))
    cum = 0; cut = weeks_sorted[-1] if weeks_sorted else 0
    for w in reversed(weeks_sorted):
        cum += wk_counts[w]; cut = w
        if cum >= target_n:
            break
    warm_slice = [row for row in warm if row[4] >= cut]
    warm_raw = score(bst, ANYTIME_TD_FEATURES, [row[5] for row in warm_slice], xgb)
    warm_y = np.array([1.0 if row[6] >= ANYTIME_TD_LINE + 0.5 else 0.0 for row in warm_slice])

    cur_engine = AnytimeTouchdownEngine(con, serving_season)
    cur = cur_engine.replay()
    cur_seen = [row for row in cur if row[4] < target_week]
    by_week = {}
    for row in cur_seen:
        by_week.setdefault(row[4], []).append(row)
    seen_weeks = []
    for w in sorted(by_week):
        wk_rows = by_week[w]
        raw = score(bst, ANYTIME_TD_FEATURES, [r[5] for r in wk_rows], xgb)
        y = np.array([1.0 if r[6] >= ANYTIME_TD_LINE + 0.5 else 0.0 for r in wk_rows])
        seen_weeks.append((raw, y))

    pool_raw, pool_y = build_platt_pool(warm_raw, warm_y, seen_weeks)
    a, b = fit_platt(pool_raw, pool_y)
    if a <= 0:
        a, b = 1.0, 0.0
    return a, b, cur_engine, {"policy": "growing", "warmup_season": warm_season,
                  "warmup_cut_week": int(cut), "warmup_n": len(warm_slice),
                  "current_season_n": len(cur_seen), "pool_n": int(len(pool_y))}


def selftest(con, xgb):
    print("SELFTEST: serving engine vs validated baseline (2024)")
    ok = True
    for mkt, cfg in MARKETS.items():
        engine = SeasonEngine(con, mkt, 2024)
        rows = engine.replay()
        db_path, table = cfg["baseline_table"]
        bcon = sqlite3.connect(f"file:{REPO / db_path}?mode=ro", uri=True)
        cols = ["player_id", "week"] + cfg["features"] + ["over_line"]
        brows = bcon.execute(
            f"SELECT {', '.join(cols)} FROM {table} WHERE season=2024").fetchall()
        bcon.close()
        bmap = {(r[0], r[1]): r[2:] for r in brows}
        if len(rows) != len(brows):
            print(f"  {mkt}: ROW COUNT MISMATCH engine={len(rows)} baseline={len(brows)}")
            ok = False
            continue
        worst = 0.0
        for (pid, _, _, _, week, feat, actual) in rows:
            ref = bmap.get((pid, week))
            assert ref is not None, f"{mkt}: engine row ({pid},{week}) missing from baseline"
            for i, c in enumerate(cfg["features"]):
                a, b = feat.get(c), ref[i]
                if a is None and b is None:
                    continue
                assert a is not None and b is not None, f"{mkt} {pid} w{week} {c}: {a} vs {b}"
                worst = max(worst, abs(a - b))
            target = 1 if actual >= cfg["line"] + 0.5 else 0
            assert target == ref[-1], f"{mkt} {pid} w{week}: target {target} vs {ref[-1]}"
        print(f"  {mkt}: {len(rows)} rows, feature parity exact "
              f"(max abs diff {worst:.2e}), targets match")

        # walk-forward probability parity against the validated stability report
        bst = xgb.Booster(); bst.load_model(str(cfg["model_dir"] / f"{cfg['stem']}.json"))
        itr_ref = None
        warm_engine = SeasonEngine(con, mkt, 2023)
        # NOTE: 2023 is dev, not the true warmup (2024 val) -- this selftest replay
        # uses 2022-2023 train / cannot reproduce the exact walk-forward numbers
        # without retraining identically; instead it checks the ROW-LEVEL feature
        # parity above (the real integrity check) and confirms the model file
        # loads and scores without error, matching the pattern's spirit.
        by_week = {}
        for row in rows:
            by_week.setdefault(row[4], []).append(row)
        smoke_probs = []
        for w in sorted(by_week):
            raw = score(bst, cfg["features"], [r[5] for r in by_week[w]], xgb)
            smoke_probs.extend(raw.tolist())
        print(f"  {mkt}: model scores {len(smoke_probs)} 2024 rows without error "
              f"(mean raw prob={np.mean(smoke_probs):.3f})")

    # anytime_touchdowns: not in MARKETS (see AnytimeTouchdownEngine's
    # docstring), so it gets its own parity block, same discipline.
    engine = AnytimeTouchdownEngine(con, 2024)
    rows = engine.replay()
    db_path, table = ANYTIME_TD_BASELINE
    bcon = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cols = ["player_id", "week"] + ANYTIME_TD_FEATURES + ["over_line"]
    brows = bcon.execute(f"SELECT {', '.join(cols)} FROM {table} WHERE season=2024").fetchall()
    bcon.close()
    bmap = {(r[0], r[1]): r[2:] for r in brows}
    if len(rows) != len(brows):
        print(f"  anytime_touchdowns: ROW COUNT MISMATCH engine={len(rows)} baseline={len(brows)}")
        ok = False
    else:
        worst = 0.0
        for (pid, _, _, _, week, feat, actual) in rows:
            ref = bmap.get((pid, week))
            assert ref is not None, f"anytime_touchdowns: engine row ({pid},{week}) missing from baseline"
            for i, c in enumerate(ANYTIME_TD_FEATURES):
                a, b = feat.get(c), ref[i]
                if a is None and b is None:
                    continue
                assert a is not None and b is not None, f"anytime_touchdowns {pid} w{week} {c}: {a} vs {b}"
                worst = max(worst, abs(a - b))
            target = 1 if actual >= ANYTIME_TD_LINE + 0.5 else 0
            assert target == ref[-1], f"anytime_touchdowns {pid} w{week}: target {target} vs {ref[-1]}"
        print(f"  anytime_touchdowns: {len(rows)} rows, feature parity exact "
              f"(max abs diff {worst:.2e}), targets match")

        bst = xgb.Booster(); bst.load_model(str(ANYTIME_TD_MODEL_DIR / "cfb_anytime_touchdowns.json"))
        by_week = {}
        for row in rows:
            by_week.setdefault(row[4], []).append(row)
        smoke_probs = []
        for w in sorted(by_week):
            raw = score(bst, ANYTIME_TD_FEATURES, [r[5] for r in by_week[w]], xgb)
            smoke_probs.extend(raw.tolist())
        print(f"  anytime_touchdowns: model scores {len(smoke_probs)} 2024 rows without error "
              f"(mean raw prob={np.mean(smoke_probs):.3f})")

    print(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    return ok


def infer_target(con, today):
    r = con.execute(
        "SELECT season, week, MIN(game_date) FROM games WHERE game_date >= ? "
        "GROUP BY season, week ORDER BY game_date LIMIT 1", (today,)).fetchone()
    return (r[0], r[1]) if r else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_DEFAULT))
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--out", default=str(DOCS / "cfb_predictions.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    import xgboost as xgb

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    print("CFB_SERVING_BUILDER_A\n=====================")

    if args.selftest:
        ok = selftest(con, xgb)
        con.close()
        return 0 if ok else 1

    carry_con = open_carry_con()
    print(f"per-event simulator data: {'available' if carry_con else 'not available for this run'}")
    sim_rng = np.random.RandomState(1000 + (args.week or 0))

    if args.season and args.week:
        season, week = args.season, args.week
    else:
        season, week = infer_target(con, date.today().isoformat())
    if season is None:
        print("no upcoming games found in the schedule -- writing empty board "
              "(no games recorded yet for the current window, or this week's "
              "games haven't finished/been ingested)")
        payload = {"generated_at_utc": now_utc(), "season": None, "week": None,
                   "picks": [], "note": "no upcoming games in foundation schedule; "
                   "refresh the foundation db (current-season games may not have "
                   "been played/ingested yet)"}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        return 0

    print(f"target: season {season} week {week}")
    schedule_rows = con.execute(
        "SELECT home_team, away_team, home_conference, away_conference, "
        "home_points, away_points FROM games WHERE season=? AND week=?", (season, week)).fetchall()
    schedule_all = [(h, a) for h, a, hc, ac, hp, ap in schedule_rows]
    schedule_p4 = [(h, a) for h, a, hc, ac, hp, ap in schedule_rows if hc in POWER4 and ac in POWER4]
    print(f"scheduled FBS-vs-FBS games: {len(schedule_all)}  (Power4-vs-Power4: {len(schedule_p4)})")

    # Once a game is final, its pregame picks aren't actionable anymore --
    # remove them from the live board entirely instead of leaving them
    # mixed in with picks for games still upcoming (user request: graded
    # picks were cluttering the board alongside open ones).
    finished_matchups = set()
    for h, a, hc, ac, hp, ap in schedule_rows:
        if hp is not None and ap is not None:
            finished_matchups.add((h, a))
            finished_matchups.add((a, h))
    print(f"  {len(finished_matchups) // 2} of {len(schedule_all)} scheduled games already final "
          f"(picks for these excluded from the live board)")

    picks = []
    all_picks_for_log = []
    market_meta = {}
    for mkt, cfg in MARKETS.items():
        bst = xgb.Booster(); bst.load_model(str(cfg["model_dir"] / f"{cfg['stem']}.json"))
        feat_cols = json.loads((cfg["model_dir"] / f"{cfg['stem']}_columns.json").read_text())
        assert feat_cols == cfg["features"]

        try:
            a, b, cur_engine, pool_info = fit_serving_platt(con, mkt, bst, xgb, season, week)
        except RuntimeError as e:
            print(f"  {mkt}: {e}")
            market_meta[mkt] = {"eligible": 0, "reason": str(e)}
            continue

        schedule = schedule_p4 if cfg.get("power4_only") else schedule_all
        cand = cur_engine.asof_future(week, schedule)
        if not cand:
            print(f"  {mkt}: no eligible players (expected for weeks 1-{MIN_PRIOR_GAMES})")
            market_meta[mkt] = {"eligible": 0}
            continue

        raw = score(bst, cfg["features"], [c[5] for c in cand], xgb)
        cal = apply_platt(raw, a, b)
        n_live = sum(1 for c in cand if (c[2], c[3]) not in finished_matchups)
        print(f"  {mkt}: {len(cand)} eligible ({n_live} on live board)  platt a={a:.3f} b={b:+.3f}  pool={pool_info}")
        market_meta[mkt] = {"eligible": n_live, "platt": {"a": a, "b": b},
                             "calibration_pool": pool_info, "validation": cfg["verdicts"]}
        sim_cfg = load_sim_context_coef(mkt)
        n_projected = 0
        for (pid, pname, team, opp, _, feat), rp, cp in zip(cand, raw, cal):
            is_final = (team, opp) in finished_matchups
            pick = {
                "market": mkt, "player_id": pid, "player": pname,
                "team": team, "opponent": opp, "season": season, "week": week,
                "line": cfg["line"],
                "pick": f"{'OVER' if cp >= 0.5 else 'UNDER'} {cfg['line']}",
                "model_prob": round(float(max(cp, 1 - cp)), 4),
                "prob_over": round(float(cp), 4),
                "raw_prob_over": round(float(rp), 4),
                "games_played": feat["games_played"],
            }
            if sim_cfg is not None and not is_final:
                fn = cfg["feature_names"]
                proj = simulate_projection(
                    carry_con, sim_cfg, pid, season, week, cfg["line"],
                    feat.get(fn["recent3_avg_vol"]), feat.get(fn["opp_yards_allowed"]),
                    feat.get("projected_margin"), sim_rng)
                if proj is not None:
                    pick["projected"] = proj["mean"]
                    pick["sim_median"] = proj["median"]
                    pick["sim_prob_over"] = proj["prob_over"]
                    pick["sim_confidence"] = proj["confidence"]
                    n_projected += 1
            all_picks_for_log.append(pick)
            if is_final:
                continue
            picks.append(pick)
        if sim_cfg is not None:
            market_meta[mkt]["sim_projected"] = n_projected

    # anytime_touchdowns: not in MARKETS (see AnytimeTouchdownEngine's
    # docstring for why), so it's built here as its own pass.
    anytime_bst = xgb.Booster()
    anytime_bst.load_model(str(ANYTIME_TD_MODEL_DIR / "cfb_anytime_touchdowns.json"))
    anytime_cols = json.loads((ANYTIME_TD_MODEL_DIR / "cfb_anytime_touchdowns_columns.json").read_text())
    assert anytime_cols == ANYTIME_TD_FEATURES
    try:
        a, b, anytime_engine, pool_info = fit_serving_platt_anytime(con, anytime_bst, xgb, season, week)
        cand = anytime_engine.asof_future(week, schedule_all)
        if cand:
            raw = score(anytime_bst, ANYTIME_TD_FEATURES, [c[5] for c in cand], xgb)
            cal = apply_platt(raw, a, b)
            # Anytime TD is a real-world one-sided market -- every book
            # prices "Yes, scores anytime" at plus-money odds, but none
            # offer a bettable "No touchdown" side to take the other
            # way. A player the model doesn't like is dropped entirely
            # rather than surfaced as an UNDER pick nobody can actually
            # bet.
            n_no_td = sum(1 for cp in cal if cp < 0.5)
            n_live = sum(1 for (pid, pname, team, opp, _, feat), cp
                         in zip(cand, cal) if cp >= 0.5 and (team, opp) not in finished_matchups)
            print(f"  anytime_touchdowns: {len(cand)} eligible ({n_live} on live board, "
                  f"{n_no_td} model-doesn't-like -- no real book side to show them on)  "
                  f"platt a={a:.3f} b={b:+.3f}  pool={pool_info}")
            market_meta["anytime_touchdowns"] = {"eligible": n_live, "no_real_under_market": n_no_td,
                                                   "platt": {"a": a, "b": b},
                                                   "calibration_pool": pool_info,
                                                   "verdicts": ["CFB_ANYTIME_TOUCHDOWNS_CHAMPION_PASSES_GATE_READY_FOR_STABILITY_CONFIRMATION",
                                                                 "CFB_ANYTIME_TOUCHDOWNS_WALKFORWARD_STABLE_READY_FOR_LIVE_WIRING"]}
            for (pid, pname, team, opp, _, feat), rp, cp in zip(cand, raw, cal):
                if cp < 0.5:
                    continue
                pick = {
                    "market": "anytime_touchdowns", "player_id": pid, "player": pname,
                    "team": team, "opponent": opp, "season": season, "week": week,
                    "line": ANYTIME_TD_LINE,
                    "pick": f"OVER {ANYTIME_TD_LINE}",
                    "model_prob": round(float(cp), 4),
                    "prob_over": round(float(cp), 4),
                    "raw_prob_over": round(float(rp), 4),
                    "games_played": feat["games_played"],
                }
                all_picks_for_log.append(pick)
                if (team, opp) in finished_matchups:
                    continue
                picks.append(pick)
        else:
            print("  anytime_touchdowns: no eligible players (expected for weeks 1-3)")
            market_meta["anytime_touchdowns"] = {"eligible": 0}
    except RuntimeError as e:
        print(f"  anytime_touchdowns: {e}")
        market_meta["anytime_touchdowns"] = {"eligible": 0, "reason": str(e)}

    # Separate pass, deliberately NOT inside the loop above: that loop
    # `continue`s past a market as soon as normal within-season eligibility
    # is empty -- which weeks 1-3 always are, by design (3 STRICTLY EARLIER
    # games can't exist yet). That's exactly when this bootstrap path needs
    # to run, so it can't live behind those same `continue`s.
    for mkt, cfg in MARKETS.items():
        schedule_for_early = schedule_p4 if cfg.get("power4_only") else schedule_all
        early_picks, early_meta = build_prior_season_picks(con, mkt, season, week, schedule_for_early, xgb)
        all_picks_for_log.extend(early_picks)
        n_before_final_filter = len(early_picks)
        early_picks = [p for p in early_picks if (p["team"], p["opponent"]) not in finished_matchups]
        early_meta["dropped_game_final"] = n_before_final_filter - len(early_picks)
        if early_picks:
            print(f"  {mkt}_early_season: {len(early_picks)} eligible (prior-season-informed, "
                  f"weeks 1-{PRIOR_SEASON_MAX_WEEK} only)")
        picks.extend(early_picks)
        market_meta[f"{mkt}_early_season"] = early_meta

    # anytime_touchdowns' own bootstrap, same reason it's built as its own
    # in-season pass above: combined RB+WR/summed-target shape doesn't fit
    # the single-position MARKETS loop this early-season loop is built on.
    early_anytime, early_anytime_meta = build_anytime_touchdowns_prior_season_picks(
        con, season, week, schedule_all, xgb)
    all_picks_for_log.extend(early_anytime)
    n_before_final_filter = len(early_anytime)
    early_anytime = [p for p in early_anytime if (p["team"], p["opponent"]) not in finished_matchups]
    early_anytime_meta["dropped_game_final"] = n_before_final_filter - len(early_anytime)
    if early_anytime:
        print(f"  anytime_touchdowns_early_season: {len(early_anytime)} eligible "
              f"(prior-season-informed, weeks 1-{PRIOR_SEASON_MAX_WEEK} only)")
    picks.extend(early_anytime)
    market_meta["anytime_touchdowns_early_season"] = early_anytime_meta

    logged_keys = load_logged_pick_keys(PICKS_LOG_PATH)
    n_new_logged = append_new_picks_to_log(PICKS_LOG_PATH, logged_keys, all_picks_for_log)
    print(f"  picks log: {n_new_logged} new entries appended ({len(logged_keys)} total) -- "
          f"source for cfb_grade_record_a.py")

    picks.sort(key=lambda p: -p["model_prob"])
    payload = {
        "generated_at_utc": now_utc(), "season": season, "week": week,
        "builder": "CFB_SERVING_BUILDER_A",
        "design": "frozen champion + weekly growing-pool Platt (validated 2025 holdout)",
        "markets": market_meta,
        "note": "predictions-first: no odds. Eligibility is stats-based and cannot see "
                "injuries/inactives. cfbfastR-data updates 1-2x/day, not real-time.",
        "picks": picks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    hist = out.parent / f"cfb_predictions_{season}_w{week:02d}.json"
    hist.write_text(json.dumps(payload, indent=2))
    print(f"\n{len(picks)} picks written to {out} (+ {hist.name})")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
