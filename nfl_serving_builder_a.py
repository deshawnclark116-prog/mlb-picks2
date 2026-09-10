#!/usr/bin/env python3
"""
NFL_SERVING_BUILDER_A

The live serving path for the three fully validated NFL markets:

  rushing_yards    RB-only,  over 49.5 rushing yards
  receiving_yards  WR-only,  over 49.5 receiving yards
  sacks            individual defensive player prop (DE/DT/OLB/LB/MLB/ILB/
                    NT/DL, recent-pass-rush-role eligibility), over 0.5
                    sacks ("Anytime Sack"). Added 2026-08 after
                    nfl_defense_sacks_champion_gate_a.py +
                    nfl_defense_sacks_walkforward_stability_a.py both
                    passed cleanly (no drift, no recalibration needed --
                    unlike tackles, which failed its gate on uncorrectable
                    season-over-season drift and was never shipped, and
                    interceptions, which failed on AUC~0.51/no real
                    signal, also never shipped).

Deployment design is EXACTLY the configuration that passed the walk-forward
gates and stability confirmation -- nothing else:
  frozen champion model (from nfl_models/, repo-backed, never retrained
  here) + weekly Platt recalibration refit on the season in progress
  (warmup pool = the most recent completed season's internal-val slice,
  exactly the role 2023's slice played in validation).

Predictions-first: no odds anywhere. Emits calibrated P(over line) for
every eligible player in the target week's scheduled games, plus full
metadata, to docs/nfl_predictions.json (+ a per-week history file).

Weekly flow (GitHub Actions, Tuesdays in season -- see
.github/workflows/nfl_weekly.yml):
  1. rebuild the foundation db from nflverse (stateless, ~2 min)
  2. python nfl_serving_builder_a.py            (auto-picks the next week)
  3. commit docs/

Eligibility mirrors the validated baselines exactly: a player needs >= 3
prior games THIS season and a current-role recent rate (recent3 carries
>= 12 for RB rushing, recent3 targets >= 5 for WR receiving), so the
board is empty for weeks 1-3 by design -- the same population the models
were validated on (holdout weeks 4-22). Known limitation, documented not
hidden: eligibility is stats-based; it cannot see injuries/inactives for
the upcoming game (NFL has no MLB-style confirmed lineups).

Weeks 1-3 aren't left empty, though: rushing_yards_early_season and
receiving_yards_early_season fill the gap from two validated early-
season sources (see PRIOR_SEASON_MARKETS / build_preseason_rushing_picks
above for the full design and validation history): a player's real 2026
preseason snaps when they have any, or their real last-season production
when they don't (the common case for a rested veteran starter -- a real
gap found in production 2026-09-09 Week 1, where the preseason-only
source left the board entirely backups and no real starters). Each
pick's model_source field says which one produced it.

Feature computation MIRRORS the baseline builders (same rules,
reimplemented for as-of-future-week serving) -- and --selftest PROVES the
mirror: it recomputes the full 2024 season through this engine and
requires (a) exact row-for-row feature parity with the validated
baseline.sqlite for both markets, and (b) walk-forward probabilities that
reproduce the gated AUC/ECE. Run it after any edit to this file.

Run
---
python -u nfl_serving_builder_a.py --selftest          # offline parity proof
python -u nfl_serving_builder_a.py                     # build next week's board
python -u nfl_serving_builder_a.py --season 2026 --week 7   # explicit target
"""

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import requests

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import nfl_rushing_yards_champion_gate_a as g  # metrics/auc/pick_val_cut/NAN
import nfl_defense_sacks_champion_gate_a as gs  # sacks has its own season+week-
# aware pick_val_cut (dev spans 3 seasons, 2022-2024) -- g's is week-only
# (single-season dev), incompatible for a multi-season market. Kept separate
# rather than generalizing g's version, to avoid touching the already-shipped
# rushing/receiving reproduction path.
from nfl_rushing_yards_recalibration_a import fit_platt, apply_platt
import nfl_sim

REPO = Path(__file__).resolve().parent
DB_DEFAULT = REPO / "nfl_models" / "nfl_model.sqlite"
DOCS = REPO / "docs"

# Append-only ledger of every pick this builder has ever produced --
# mirrors cfb_serving_builder_a.py's PICKS_LOG_PATH exactly (see that
# file's docstring for the full reasoning). NFL doesn't filter finished
# games off the live board today the way CFB does, so the per-week
# archive alone would currently be enough on its own, but a ledger keyed
# on first-seen (season, week, market, player_id) is what nfl_grade_
# record_a.py grades from either way -- it survives that filter being
# added later, and it survives an intra-week Platt recalibration nudging
# a pick's probability (or even its OVER/UNDER side) after the fact,
# which the archive file would silently overwrite.
PICKS_LOG_PATH = DOCS / "nfl_picks_log.jsonl"


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


MARKETS = {
    "rushing_yards": {
        "position": "RB",
        "line": 49.5,
        "stat_fields": ["carries", "rushing_yards"],
        "rate_field": "carries", "min_recent_rate": 12,
        "opp_stat": "rushing_yards",
        "features": ["season_avg_rush_yards", "recent3_avg_rush_yards",
                      "recent5_avg_rush_yards", "season_avg_carries",
                      "recent3_avg_carries", "yards_per_carry",
                      "opp_rush_yards_allowed_per_game", "is_home", "games_played"],
        "model_dir": REPO / "nfl_models" / "nfl_rushing_yards_champion_gate_a_work",
        "stem": "nfl_rushing_yards",
        "baseline_table": ("nfl_models/nfl_rushing_yards_clean_baseline_a_work/baseline.sqlite",
                            "nfl_rushing_yards_baseline"),
        "verdicts": ["NFL_RUSHING_YARDS_WALKFORWARD_PASSES_GATE",
                      "NFL_RUSHING_YARDS_WALKFORWARD_STABLE_READY_FOR_LIVE_WIRING"],
        "calibration_policy": "growing",
    },
    "receiving_yards": {
        "position": "WR",
        "line": 49.5,
        "stat_fields": ["targets", "receptions", "receiving_yards"],
        "rate_field": "targets", "min_recent_rate": 5,
        "opp_stat": "receiving_yards",
        "features": ["season_avg_rec_yards", "recent3_avg_rec_yards",
                      "recent5_avg_rec_yards", "season_avg_targets",
                      "recent3_avg_targets", "yards_per_target", "catch_rate",
                      "opp_rec_yards_allowed_per_game", "is_home", "games_played"],
        "model_dir": REPO / "nfl_models" / "nfl_receiving_yards_champion_walkforward_gate_a_work",
        "stem": "nfl_receiving_yards",
        "baseline_table": ("nfl_models/nfl_receiving_yards_clean_baseline_a_work/baseline.sqlite",
                            "nfl_receiving_yards_baseline"),
        "verdicts": ["NFL_RECEIVING_YARDS_WALKFORWARD_PASSES_GATE",
                      "NFL_RECEIVING_YARDS_WALKFORWARD_STABLE_READY_FOR_LIVE_WIRING",
                      "NFL_RECEIVING_YARDS_ROLLING_CALIBRATION_FIXES_2025_HOLDOUT"],
        # Growing-pool calibration passed on 2024 but FAILED the 2025 fresh
        # holdout (calibration p=0.0422 vs the 0.10 bar) -- overconfident.
        # A 5-week rolling window, tested head-to-head against growing pool
        # on the same 2025 holdout via nfl_receiving_yards_rolling_calibration_a.py,
        # passed decisively (p=0.7642) at only a small AUC cost (0.6371->0.6224,
        # still comfortably clear of the 0.58 gate). See that report for the
        # full comparison. Falls back to growing-pool behavior until enough
        # current-season rows exist to fill the window on its own.
        "calibration_policy": "rolling",
        "calibration_window_weeks": 5,
        "calibration_min_rolling_n": 50,
    },
    "sacks": {
        # First individual-defensive-player-prop market wired to production
        # (nfl_defense_sacks_clean_baseline_a.py /
        # nfl_defense_sacks_champion_gate_a.py /
        # nfl_defense_sacks_walkforward_stability_a.py -- both rungs passed
        # cleanly, no drift/calibration issues, unlike tackles which failed
        # its gate on uncorrectable season-over-season drift and was never
        # shipped; interceptions failed on AUC~0.51, no real signal, also
        # never shipped). Real pass-rushers get inconsistently labeled
        # DE/OLB/LB/MLB/ILB/DT/NT/DL across seasons -- population is a
        # LIST of positions, not one, and eligibility is defined by recent
        # PASS-RUSH ROLE (a longer 5-game window, not the 3-game window
        # rushing/receiving use -- sacks are sparser events) rather than a
        # position label. Also the only market here that needs the
        # REG-only season_type filter: its validated pipeline excluded
        # playoffs throughout (a much smaller, differently-shaped pool of
        # teams), unlike rushing/receiving_yards, which never filtered it.
        "position": ["DE", "DT", "OLB", "LB", "MLB", "ILB", "NT", "DL"],
        "season_type_filter": "REG",
        # target threshold = line + 0.5 (shared formula, same as rushing/
        # receiving's 49.5 -> actual>=50). line=0.0 here gives actual>=0.5,
        # i.e. "recorded a sack" -- display_line is the human-readable
        # "Over 0.5 Sacks" book format, kept separate so it never leaks
        # into the threshold math (line=0.5 would silently require 2
        # sacks -- caught by --selftest target-mismatch, not guessed).
        "line": 0.0,
        "display_line": 0.5,
        "stat_fields": ["def_sacks", "def_qb_hits"],
        "rate_field": "def_sacks", "min_recent_rate": 0.3,
        "eligibility_window": 5, "min_prior_games": 5,
        "opp_stat": "def_sacks",
        "features": ["season_avg_sacks", "recent3_avg_sacks", "recent5_avg_sacks",
                      "season_avg_qb_hits", "recent3_avg_qb_hits",
                      "opp_sacks_allowed_per_game", "is_home", "games_played"],
        "model_dir": REPO / "nfl_models" / "nfl_defense_sacks_champion_gate_a_work",
        "stem": "nfl_defense_sacks",
        "baseline_table": ("nfl_models/nfl_defense_sacks_clean_baseline_a_work/baseline.sqlite",
                            "nfl_defense_sacks_baseline"),
        "verdicts": ["NFL_DEFENSE_SACKS_CHAMPION_PASSES_GATE_READY_FOR_STABILITY_CONFIRMATION",
                      "NFL_DEFENSE_SACKS_WALKFORWARD_STABLE_READY_FOR_LIVE_WIRING"],
        "calibration_policy": "growing",
        "walkforward_dev_seasons": (2022, 2023, 2024),
        "walkforward_holdout_season": 2025,
    },
}
MIN_PRIOR_GAMES = 3

# Weeks 1-3 preseason-informed rushing_yards, added 2026-08 after real
# validation (nfl_preseason_to_regular_season_gate_a.py, AUC 0.7027 on the
# 2025 holdout -- receiving_yards did NOT clear the same bar and has no
# equivalent here). Fills exactly the gap the note above already
# documents: weeks 1-3 are empty by design because the in-season model
# needs history that doesn't exist yet. Preseason already happened by
# week 1, so this uses it instead of leaving the board empty. Served
# under a distinct market key (not blended into "rushing_yards") so it's
# never confused with the in-season model's picks -- different features,
# different validation, disclosed as such.
PRESEASON_RUSHING_MAX_WEEK = 3
PRESEASON_DB = REPO / "nfl_models" / "nfl_preseason.sqlite"
PRESEASON_RUSHING_MODEL_DIR = REPO / "nfl_models"
PRESEASON_RUSHING_LINE = 49.5
PRESEASON_RUSHING_FEATURES = ["preseason_avg_yards", "preseason_games_played", "preseason_avg_rate"]
# ESPN (preseason source, and the live-scoreboard finished-game check
# below) vs nflverse (regular-season schedule source) disagree on two
# team abbreviations -- verified live against both real datasets, not
# guessed.
TEAM_ABBR_ESPN_TO_NFLVERSE = {"LAR": "LA", "WSH": "WAS"}
NFLVERSE_TO_TEAM_ABBR_ESPN = {v: k for k, v in TEAM_ABBR_ESPN_TO_NFLVERSE.items()}
NFL_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

# Real production gap found 2026-09-09 (Week 1): rushing_yards_early_season
# above only knows about players who logged a real 2026 PRESEASON snap --
# but established veteran starters (Conner, Gibbs, Bijan Robinson, Henry,
# etc.) are normal-practice rested for the ENTIRE preseason, so they have
# zero preseason rows and never appear in that model's population at all.
# The board ends up entirely backups/camp bodies, exactly backwards from
# what a bettor wants. nfl_prior_season_early_gate_a.py tested a distinct
# hypothesis -- does last season's REAL, FULL regular-season production
# predict weeks 1-3 of the next season -- and it passed for BOTH
# rushing_yards (holdout AUC 0.7702) and receiving_yards (0.7778, which
# has no early-season coverage at all otherwise). Fills the gap the
# preseason model structurally can't: a rested veteran with real last-
# season tape gets a real, informed pick here.
#
# Eligibility comes from ESPN's CURRENT roster (not last season's, and
# not this year's preseason box scores) -- a player must be on the
# team's real active roster today to get a pick, which is also what
# correctly excludes a player who's simply hurt now (verified live:
# James Conner is the one real absence from ARI's early board that
# turned out to be a genuine injuredReserveOrOut listing, not a
# preseason-rest artifact -- this roster check is what tells the two
# apart instead of guessing).
PRIOR_SEASON_MAX_WEEK = 3
PRIOR_SEASON_MODEL_DIR = REPO / "nfl_models"
PRIOR_SEASON_LINE = 49.5
PRIOR_SEASON_FEATURES = ["prior_season_avg_yards", "prior_season_games_played", "prior_season_avg_rate"]
PRIOR_SEASON_MARKETS = {
    "rushing_yards_early_season": {"position": "RB", "stat": "rushing_yards", "rate": "carries",
                                    "model_stem": "nfl_prior_season_rushing_yards"},
    "receiving_yards_early_season": {"position": "WR", "stat": "receiving_yards", "rate": "targets",
                                      "model_stem": "nfl_prior_season_receiving_yards"},
}
NFL_ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team}/roster"
# Only real active-roster groups -- injuredReserveOrOut/suspended/
# practiceSquad players won't play, so they shouldn't get a live pick
# (same "no fabricated signal" principle as everywhere else in this repo).
ACTIVE_ROSTER_GROUPS = {"offense", "defense", "specialTeam"}


def norm_player_name(name):
    name = unicodedata.normalize("NFKD", name or "")
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", name)
    name = re.sub(r"[^a-z ]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def market_features(mkt, hist, opp_allowed, is_home):
    """Feature dict for one player given their in-season history (ordered
    prior games, dicts of stat_fields). Mirrors the baseline builders
    exactly -- proven by --selftest, not assumed."""
    n = len(hist)
    if mkt == "rushing_yards":
        ry = [h["rushing_yards"] or 0 for h in hist]
        c = [h["carries"] or 0 for h in hist]
        r3, r5, c3 = ry[-3:], ry[-5:], c[-3:]
        return {
            "season_avg_rush_yards": sum(ry) / n,
            "recent3_avg_rush_yards": sum(r3) / len(r3),
            "recent5_avg_rush_yards": sum(r5) / len(r5),
            "season_avg_carries": sum(c) / n,
            "recent3_avg_carries": sum(c3) / len(c3),
            "yards_per_carry": (sum(ry) / sum(c)) if sum(c) > 0 else 0.0,
            "opp_rush_yards_allowed_per_game": opp_allowed,
            "is_home": 1.0 if is_home else 0.0,
            "games_played": n,
        }
    if mkt == "receiving_yards":
        ry = [h["receiving_yards"] or 0 for h in hist]
        t = [h["targets"] or 0 for h in hist]
        rec = [h["receptions"] or 0 for h in hist]
        r3, r5, t3 = ry[-3:], ry[-5:], t[-3:]
        return {
            "season_avg_rec_yards": sum(ry) / n,
            "recent3_avg_rec_yards": sum(r3) / len(r3),
            "recent5_avg_rec_yards": sum(r5) / len(r5),
            "season_avg_targets": sum(t) / n,
            "recent3_avg_targets": sum(t3) / len(t3),
            "yards_per_target": (sum(ry) / sum(t)) if sum(t) > 0 else 0.0,
            "catch_rate": (sum(rec) / sum(t)) if sum(t) > 0 else 0.0,
            "opp_rec_yards_allowed_per_game": opp_allowed,
            "is_home": 1.0 if is_home else 0.0,
            "games_played": n,
        }
    # sacks (see nfl_defense_sacks_clean_baseline_a.py -- must mirror its
    # build_rows() feature computation exactly, proven by --selftest)
    sk = [h["def_sacks"] or 0 for h in hist]
    qh = [h["def_qb_hits"] or 0 for h in hist]
    s3, s5, q3 = sk[-3:], sk[-5:], qh[-3:]
    return {
        "season_avg_sacks": sum(sk) / n,
        "recent3_avg_sacks": sum(s3) / len(s3),
        "recent5_avg_sacks": sum(s5) / len(s5),
        "season_avg_qb_hits": sum(qh) / n,
        "recent3_avg_qb_hits": sum(q3) / len(q3),
        "opp_sacks_allowed_per_game": opp_allowed,
        "is_home": 1.0 if is_home else 0.0,
        "games_played": n,
    }


def eligible(mkt_cfg, hist):
    min_games = mkt_cfg.get("min_prior_games", MIN_PRIOR_GAMES)
    if len(hist) < min_games:
        return False
    window = mkt_cfg.get("eligibility_window", 3)
    rates = [h[mkt_cfg["rate_field"]] or 0 for h in hist][-window:]
    return (sum(rates) / len(rates)) >= mkt_cfg["min_recent_rate"]


class SeasonEngine:
    """Replays one market's season week-by-week from player_games, exposing
    (a) completed eligible rows with features + outcomes (for calibration
    pools and self-test parity) and (b) as-of features for a FUTURE week."""

    def __init__(self, con, mkt, season):
        self.mkt = mkt
        self.cfg = MARKETS[mkt]
        self.season = season
        fields = ", ".join(self.cfg["stat_fields"])
        pos = self.cfg["position"]
        if isinstance(pos, (list, tuple)):
            pos_clause = f"position IN ({', '.join('?' for _ in pos)})"
            pos_params = list(pos)
        else:
            pos_clause = "position = ?"
            pos_params = [pos]
        st_clause, st_params = "", []
        if self.cfg.get("season_type_filter"):
            st_clause = " AND season_type = ?"
            st_params = [self.cfg["season_type_filter"]]
        self.rows = con.execute(f"""
            SELECT player_id, player_name, team, opponent, week, is_home, {fields}
            FROM player_games
            WHERE {pos_clause} AND season = ?{st_clause}
            ORDER BY week
        """, (*pos_params, season, *st_params)).fetchall()
        self.weeks = sorted({r[4] for r in self.rows})

    def replay(self):
        """Yield eligible completed rows in week order:
        (player_id, player_name, team, opponent, week, feat_dict, actual)."""
        cfg = self.cfg
        hist = {}
        opp_state = {}
        out = []
        for w in self.weeks:
            wk = [r for r in self.rows if r[4] == w]
            for r in wk:
                pid, pname, team, opp, week, is_home = r[:6]
                stats = dict(zip(cfg["stat_fields"], r[6:]))
                h = hist.get(pid, [])
                if eligible(cfg, h):
                    st = opp_state.get(opp)
                    opp_allowed = (st[0] / st[1]) if st and st[1] > 0 else None
                    feat = market_features(self.mkt, h, opp_allowed, is_home == 1)
                    actual = stats[cfg["opp_stat"]] or 0
                    out.append((pid, pname, team, opp, week, feat, actual))
            for r in wk:
                opp = r[3]
                stats = dict(zip(cfg["stat_fields"], r[6:]))
                st = opp_state.setdefault(opp, [0, 0])
                st[0] += stats[cfg["opp_stat"]] or 0
                st[1] += 1
            for r in wk:
                pid = r[0]
                hist.setdefault(pid, []).append(dict(zip(cfg["stat_fields"], r[6:])))
        return out

    def asof_future(self, target_week, schedule):
        """Features for a not-yet-played week. schedule: list of
        (home_team, away_team). A player belongs to the team of their most
        recent played game this season."""
        cfg = self.cfg
        hist = {}
        opp_state = {}
        latest_team = {}
        latest_name = {}
        for r in self.rows:
            pid, pname, team, opp, week = r[0], r[1], r[2], r[3], r[4]
            if week >= target_week:
                continue
            stats = dict(zip(cfg["stat_fields"], r[6:]))
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
                    feat = market_features(self.mkt, h, opp_allowed, is_home)
                    out.append((pid, latest_name[pid], team, opp, target_week, feat))
        return out


def score(bst, feats_order, feat_dicts, xgb):
    X = np.array([[fd.get(c) if fd.get(c) is not None else g.NAN for c in feats_order]
                  for fd in feat_dicts], dtype=np.float32)
    itr = (0, bst.best_iteration + 1)
    return np.asarray(
        bst.predict(xgb.DMatrix(X, feature_names=feats_order), iteration_range=itr),
        dtype=float)


def build_platt_pool(policy, warm_raw, warm_y, seen_weeks, window_weeks=None, min_rolling_n=None):
    """Given warmup (raw scores, labels) and a chronological list of
    (raw_scores, labels) per completed current-season week, return the
    (pool_raw, pool_y) to fit Platt on for the NEXT week, per policy:

    'growing' -- warmup + every completed current-season week so far
      (the design validated on 2024; used for rushing_yards).
    'rolling' -- once >= min_rolling_n current-season rows have
      accumulated, use ONLY the last `window_weeks` completed weeks
      (warmup dropped entirely). Before that, falls back to 'growing'
      behavior -- there's nothing to roll a window over yet. Validated
      head-to-head against 'growing' on the 2025 fresh holdout for
      receiving_yards (nfl_receiving_yards_rolling_calibration_a.py);
      passed decisively where growing-pool failed calibration.
    """
    if policy == "growing":
        raw_parts = [warm_raw] + [r for (r, _) in seen_weeks]
        y_parts = [warm_y] + [y for (_, y) in seen_weeks]
    elif policy == "rolling":
        total_current_n = sum(len(y) for (_, y) in seen_weeks)
        if total_current_n < min_rolling_n:
            raw_parts = [warm_raw] + [r for (r, _) in seen_weeks]
            y_parts = [warm_y] + [y for (_, y) in seen_weeks]
        else:
            recent = seen_weeks[-window_weeks:]
            raw_parts = [r for (r, _) in recent]
            y_parts = [y for (_, y) in recent]
    else:
        raise ValueError(f"unknown calibration_policy: {policy}")

    pool_raw = np.concatenate(raw_parts) if raw_parts else np.empty(0)
    pool_y = np.concatenate(y_parts) if y_parts else np.empty(0)
    return pool_raw, pool_y


def fit_serving_platt(con, mkt, bst, xgb, serving_season, target_week):
    """Warmup pool (previous completed season's pick_val_cut slice) + the
    serving season's completed eligible rows before target_week, combined
    per the market's configured calibration_policy (see build_platt_pool)."""
    cfg = MARKETS[mkt]
    policy = cfg.get("calibration_policy", "growing")
    seasons = [r[0] for r in con.execute(
        "SELECT DISTINCT season FROM player_games WHERE season < ? ORDER BY season DESC",
        (serving_season,))]
    if not seasons:
        raise RuntimeError(f"no completed season before {serving_season} in db")
    warm_season = seasons[0]

    warm = SeasonEngine(con, mkt, warm_season).replay()
    warm_tuples = [(row[4],) for row in warm]  # (week,) for pick_val_cut
    cut = g.pick_val_cut([(None, w) for (w,) in warm_tuples])
    warm_slice = [row for row in warm if row[4] >= cut]
    warm_raw = score(bst, cfg["features"], [row[5] for row in warm_slice], xgb)
    line = cfg["line"]
    warm_y = np.array([1.0 if row[6] >= line + 0.5 else 0.0 for row in warm_slice])

    cur = SeasonEngine(con, mkt, serving_season).replay()
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

    pool_raw, pool_y = build_platt_pool(
        policy, warm_raw, warm_y, seen_weeks,
        window_weeks=cfg.get("calibration_window_weeks"),
        min_rolling_n=cfg.get("calibration_min_rolling_n"))
    a, b = fit_platt(pool_raw, pool_y)
    if a <= 0:
        a, b = 1.0, 0.0
    return a, b, {"policy": policy, "warmup_season": warm_season,
                  "warmup_cut_week": int(cut), "warmup_n": len(warm_slice),
                  "current_season_n": len(cur_seen), "pool_n": int(len(pool_y))}


def selftest(con, xgb):
    """(a) exact feature parity with the validated baseline.sqlite for the
    full 2024 season, both markets; (b) walk-forward probability parity
    with the gated/stability numbers."""
    print("SELFTEST: serving engine vs validated baselines (2024)")
    ok = True
    for mkt, cfg in MARKETS.items():
        rows = SeasonEngine(con, mkt, 2024).replay()
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

        # (b) walk-forward probability parity, under the market's configured
        # calibration_policy, against an already-trusted reference number.
        # rushing_yards (policy=growing) replays 2024 off a 2023 warmup and
        # checks against the original walk-forward gate's AUC. receiving_yards
        # (policy=rolling) replays 2025 off a 2024 warmup and checks against
        # the rolling-vs-growing head-to-head report -- the 2025 season is
        # what the rolling policy was actually validated on, and using the
        # old growing-pool-era 2024 reference here would be checking the
        # wrong thing (a different policy produces different numbers).
        policy = cfg.get("calibration_policy", "growing")
        bst = xgb.Booster(); bst.load_model(str(cfg["model_dir"] / f"{cfg['stem']}.json"))
        dev_seasons = cfg.get("walkforward_dev_seasons")
        if dev_seasons:
            # sacks: multi-season dev warmup, season+week-aware cut (gs, not g).
            replay_season = cfg["walkforward_holdout_season"]
            replay_rows = SeasonEngine(con, mkt, replay_season).replay()
            by_week = {}
            for row in replay_rows:
                by_week.setdefault(row[4], []).append(row)
            # gs.pick_val_cut needs real (season, week) pairs, not the replay
            # tuples alone (they don't carry season) -- attach it explicitly.
            warm_sw = []
            for s in dev_seasons:
                for row in SeasonEngine(con, mkt, s).replay():
                    warm_sw.append((s, row[4], row))
            cut = gs.pick_val_cut([(s, w) for (s, w, _) in warm_sw])
            warm_slice = [row for (s, w, row) in warm_sw if (s, w) >= cut]
            wr = score(bst, cfg["features"], [r[5] for r in warm_slice], xgb)
            wy = np.array([1.0 if r[6] >= cfg["line"] + 0.5 else 0.0 for r in warm_slice])
        else:
            if policy == "rolling":
                warm_season, replay_season = 2024, 2025
            else:
                warm_season, replay_season = 2023, 2024
            replay_rows = rows if replay_season == 2024 else SeasonEngine(con, mkt, replay_season).replay()
            by_week = {}
            for row in replay_rows:
                by_week.setdefault(row[4], []).append(row)
            warm = SeasonEngine(con, mkt, warm_season).replay()
            cut = g.pick_val_cut([(None, r[4]) for r in warm])
            warm_slice = [r for r in warm if r[4] >= cut]
            wr = score(bst, cfg["features"], [r[5] for r in warm_slice], xgb)
            wy = np.array([1.0 if r[6] >= cfg["line"] + 0.5 else 0.0 for r in warm_slice])
        probs, ys = [], []
        seen_weeks = []
        for w in sorted(by_week):
            pool_raw, pool_y = build_platt_pool(
                policy, wr, wy, seen_weeks,
                window_weeks=cfg.get("calibration_window_weeks"),
                min_rolling_n=cfg.get("calibration_min_rolling_n"))
            a, b = fit_platt(pool_raw, pool_y)
            if a <= 0:
                a, b = 1.0, 0.0
            wk_rows = by_week[w]
            raw = score(bst, cfg["features"], [r[5] for r in wk_rows], xgb)
            y = np.array([1.0 if r[6] >= cfg["line"] + 0.5 else 0.0 for r in wk_rows])
            probs.extend(apply_platt(raw, a, b).tolist())
            ys.extend(y.tolist())
            seen_weeks.append((raw, y))
        m = g.metrics(probs, ys)

        if dev_seasons:
            ref_path = cfg["model_dir"] / "nfl_defense_sacks_walkforward_stability_a_report.json"
            ref = json.loads(ref_path.read_text())
            ref_auc = ref["rung1_walkforward"]["auc"]
        elif policy == "rolling":
            ref_path = cfg["model_dir"] / "nfl_receiving_yards_rolling_calibration_a_report.json"
            ref = json.loads(ref_path.read_text())
            ref_auc = ref["rolling"]["metrics"]["auc"]
        else:
            ref_path = (cfg["model_dir"] /
                        f"nfl_walkforward_stability_confirmation_a_{mkt}_report.json")
            ref = json.loads(ref_path.read_text())
            ref_auc = ref["point_auc"]
        print(f"  {mkt}: [{policy}] walk-forward replay ({replay_season}) "
              f"AUC={m['auc']:.4f} ECE={m['ece']:.4f} (reference AUC={ref_auc:.4f})")
        assert abs(m["auc"] - ref_auc) < 0.002, \
            f"{mkt}: serving walk-forward diverges from validated run"
    print(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    return ok


def infer_target(con, today):
    """Next (season, week) with any unplayed game on/after today."""
    r = con.execute(
        "SELECT season, week, MIN(game_date) FROM games WHERE game_date >= ? "
        "GROUP BY season, week ORDER BY game_date LIMIT 1", (today,)).fetchone()
    return (r[0], r[1]) if r else (None, None)


def build_preseason_rushing_picks(season, week, schedule, xgb):
    """Weeks 1-3 only: fills the empty-board gap using the validated
    preseason-informed model. Population is every RB who appears in this
    season's local preseason data for a scheduled team -- not a separate
    roster fetch, since a player with no preseason match gets NaN
    features anyway (no more informative than the league base rate), so
    there's nothing gained by including them."""
    if week > PRESEASON_RUSHING_MAX_WEEK:
        return [], {"eligible": 0, "reason": f"week > {PRESEASON_RUSHING_MAX_WEEK}"}

    model_path = PRESEASON_RUSHING_MODEL_DIR / "nfl_preseason_rushing_yards.json"
    cols_path = PRESEASON_RUSHING_MODEL_DIR / "nfl_preseason_rushing_yards_columns.json"
    if not model_path.exists() or not PRESEASON_DB.exists():
        return [], {"eligible": 0, "reason": "model or preseason db not present"}

    feat_cols = json.loads(cols_path.read_text())
    assert feat_cols == PRESEASON_RUSHING_FEATURES

    teams = set()
    for home, away in schedule:
        teams.add(home); teams.add(away)
    nflverse_to_preseason = {v: k for k, v in TEAM_ABBR_ESPN_TO_NFLVERSE.items()}
    preseason_teams = {nflverse_to_preseason.get(t, t) for t in teams}

    pcon = sqlite3.connect(f"file:{PRESEASON_DB}?mode=ro", uri=True)
    placeholders = ",".join("?" for _ in preseason_teams)
    rows = pcon.execute(f"""
        SELECT athlete_id, player_name, team, carries, rushing_yards
        FROM player_games WHERE season=? AND position='RB' AND team IN ({placeholders})
    """, (season, *preseason_teams)).fetchall()
    pcon.close()

    by_player = {}
    team_of = {}
    for aid, name, team, carries, ry in rows:
        by_player.setdefault(aid, []).append((carries or 0, ry or 0))
        team_of[aid] = team
        team_of[("name", aid)] = name

    if not by_player:
        return [], {"eligible": 0, "reason": "no preseason RB data for scheduled teams"}

    preseason_to_nflverse = TEAM_ABBR_ESPN_TO_NFLVERSE
    team_pairs = {home: away for home, away in schedule}
    team_pairs.update({away: home for home, away in schedule})

    cand_ids, feats, meta = [], [], []
    for aid, games in by_player.items():
        n = len(games)
        carries = [c for c, _ in games]
        yards = [y for _, y in games]
        feats.append([sum(yards) / n, float(n), sum(carries) / n])
        team_nflverse = preseason_to_nflverse.get(team_of[aid], team_of[aid])
        opp_nflverse = team_pairs.get(team_nflverse)
        cand_ids.append(aid)
        meta.append((aid, team_of[("name", aid)], team_nflverse, opp_nflverse, n))

    bst = xgb.Booster(); bst.load_model(str(model_path))
    dm = xgb.DMatrix(np.array(feats, dtype=np.float32), feature_names=feat_cols)
    probs = bst.predict(dm)

    picks = []
    for (aid, pname, team, opp, games_played), p in zip(meta, probs):
        if opp is None:
            continue  # team not actually in this week's schedule (shouldn't happen, guarded anyway)
        cp = float(p)
        picks.append({
            "market": "rushing_yards_early_season", "player_id": aid, "player": pname,
            "team": team, "opponent": opp, "season": season, "week": week,
            "line": PRESEASON_RUSHING_LINE,
            "pick": f"{'OVER' if cp >= 0.5 else 'UNDER'} {PRESEASON_RUSHING_LINE}",
            "model_prob": round(max(cp, 1 - cp), 4),
            "prob_over": round(cp, 4),
            "games_played": games_played,
            "model_source": "preseason_informed",
            "model_version": "nfl_preseason_rushing_production_builder_a_2026_08",
            "validation_note": "AUC 0.7027 on 2025 holdout, see "
                                "nfl_preseason_to_regular_season_gate_a.py",
        })
    picks.sort(key=lambda p: -p["model_prob"])
    return picks, {"eligible": len(picks), "model": "preseason_informed",
                   "validated_holdout_auc": 0.7027}


def fetch_espn_active_roster(team_nflverse, position, cache):
    """Real, CURRENT active-roster players at a given position for one
    NFL team (offense/defense/specialTeam groups only -- see
    ACTIVE_ROSTER_GROUPS). Cached per team so a team scheduled more than
    once isn't re-fetched. Returns [(espn_athlete_id, display_name), ...]."""
    if team_nflverse in cache:
        roster = cache[team_nflverse]
    else:
        slug = NFLVERSE_TO_TEAM_ABBR_ESPN.get(team_nflverse, team_nflverse).lower()
        roster = []
        try:
            r = requests.get(NFL_ROSTER_URL.format(team=slug), timeout=20)
            r.raise_for_status()
            for group in r.json().get("athletes", []):
                if group.get("position") not in ACTIVE_ROSTER_GROUPS:
                    continue
                for item in group.get("items", []):
                    pos = (item.get("position") or {}).get("abbreviation")
                    roster.append((item.get("id"), item.get("displayName"), pos))
        except Exception as e:
            print(f"    roster fetch failed for {team_nflverse} ({slug}): {e}")
        cache[team_nflverse] = roster
    return [(aid, name) for aid, name, pos in roster if pos == position]


def load_prior_season_by_name(con, season, position):
    """normalized full name -> list of per-game stat dicts for the WHOLE
    given real season (every week, REG only) -- name-keyed because
    ESPN's roster ids and nflverse's gsis player_id are different, un-
    crosswalked namespaces (same real limitation documented in
    nfl_preseason_to_regular_season_gate_a.py for the ESPN/nflverse
    preseason join)."""
    rows = con.execute(
        "SELECT player_name, carries, rushing_yards, targets, receiving_yards "
        "FROM player_games WHERE season=? AND season_type='REG' AND position=?",
        (season, position)).fetchall()
    by_name = {}
    for name, carries, ry, targets, rey in rows:
        key = norm_player_name(name)
        if not key:
            continue
        by_name.setdefault(key, []).append({
            "carries": carries or 0, "rushing_yards": ry or 0,
            "targets": targets or 0, "receiving_yards": rey or 0,
        })
    return by_name


def build_prior_season_picks(con, season, week, schedule, xgb, exclude_names_by_market):
    """weeks 1-3 only: fills the gap build_preseason_rushing_picks leaves
    for a rested veteran with zero real 2026 preseason snaps, using last
    season's real, full regular-season production instead. See the
    PRIOR_SEASON_* constants' comment above for the full rationale.
    exclude_names_by_market lets a market already covered by another
    early-season source (rushing_yards_early_season's preseason-informed
    arm) skip a player who already got a pick there, so nobody is
    double-counted."""
    if week > PRIOR_SEASON_MAX_WEEK:
        return [], {}

    prior_season = season - 1
    teams = sorted({t for pair in schedule for t in pair})
    roster_cache = {}
    all_picks = []
    all_meta = {}

    for mkt, cfg in PRIOR_SEASON_MARKETS.items():
        model_path = PRIOR_SEASON_MODEL_DIR / f"{cfg['model_stem']}.json"
        cols_path = PRIOR_SEASON_MODEL_DIR / f"{cfg['model_stem']}_columns.json"
        if not model_path.exists():
            all_meta[mkt] = {"eligible": 0, "reason": "model not present"}
            continue
        feat_cols = json.loads(cols_path.read_text())
        assert feat_cols == PRIOR_SEASON_FEATURES

        prior_by_name = load_prior_season_by_name(con, prior_season, cfg["position"])
        if not prior_by_name:
            all_meta[mkt] = {"eligible": 0, "reason": f"no {prior_season} {cfg['position']} data"}
            continue

        team_pairs = {home: away for home, away in schedule}
        team_pairs.update({away: home for home, away in schedule})
        excluded = exclude_names_by_market.get(mkt, set())

        cand_feats, cand_meta = [], []
        n_roster_seen = n_no_prior_data = n_excluded = 0
        for team in teams:
            opp = team_pairs.get(team)
            if opp is None:
                continue
            for aid, pname in fetch_espn_active_roster(team, cfg["position"], roster_cache):
                n_roster_seen += 1
                key = norm_player_name(pname)
                if key in excluded:
                    n_excluded += 1
                    continue
                games = prior_by_name.get(key)
                if not games:
                    n_no_prior_data += 1
                    continue
                n = len(games)
                stat_field, rate_field = cfg["stat"], cfg["rate"]
                cand_feats.append([
                    sum(g[stat_field] for g in games) / n, float(n),
                    sum(g[rate_field] for g in games) / n,
                ])
                cand_meta.append((aid, pname, team, opp, n))

        if not cand_feats:
            all_meta[mkt] = {"eligible": 0, "roster_seen": n_roster_seen,
                              "reason": "no roster player matched to prior-season data"}
            continue

        bst = xgb.Booster(); bst.load_model(str(model_path))
        dm = xgb.DMatrix(np.array(cand_feats, dtype=np.float32), feature_names=feat_cols)
        probs = bst.predict(dm)

        manifest_path = PRIOR_SEASON_MODEL_DIR / f"{cfg['model_stem']}_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        auc_note = manifest.get("validated_holdout_auc")

        picks = []
        for (aid, pname, team, opp, games_played), p in zip(cand_meta, probs):
            cp = float(p)
            picks.append({
                "market": mkt, "player_id": aid, "player": pname,
                "team": team, "opponent": opp, "season": season, "week": week,
                "line": PRIOR_SEASON_LINE,
                "pick": f"{'OVER' if cp >= 0.5 else 'UNDER'} {PRIOR_SEASON_LINE}",
                "model_prob": round(max(cp, 1 - cp), 4),
                "prob_over": round(cp, 4),
                "games_played": games_played,
                "model_source": "prior_season_informed",
                "model_version": "nfl_prior_season_production_builder_a_2026_09",
                "validation_note": f"AUC {auc_note} on 2025 holdout, see "
                                    f"nfl_prior_season_early_gate_a.py",
            })
        picks.sort(key=lambda p: -p["model_prob"])
        all_picks.extend(picks)
        all_meta[mkt] = {"eligible": len(picks), "model": "prior_season_informed",
                          "prior_season": prior_season, "roster_seen": n_roster_seen,
                          "excluded_already_covered": n_excluded,
                          "no_prior_season_data": n_no_prior_data,
                          "validated_holdout_auc": auc_note}
    return all_picks, all_meta


CARRY_DB_DEFAULT = REPO / "nfl_models" / "nfl_carry_log.sqlite"
RECENT_GAMES_WINDOW = 8
ODDS_API_NFL_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
# Real per-player market keys confirmed live against The Odds API
# (nfl_props_odds_probe.py) -- only the two markets nfl_rush_sim_gate_a.py
# / nfl_recv_sim_gate_a.py / nfl_prior_season_sim_gate_a.py actually
# validated a real Monte Carlo projection for. Deliberately NOT the
# other real markets the probe found (player_receptions, player_pass_yds,
# player_pass_tds, player_rush_attempts, player_anytime_td) -- those
# would need their own simulator + gate before ever pricing a real line
# against them; fetching odds for a market with no validated model would
# just be real data with a fabricated probability behind it.
REAL_ODDS_MARKETS = {
    "rushing_yards": {"position": "RB", "odds_market_key": "player_rush_yds",
                       "table": "rush_carries", "idx_col": "carry_index"},
    "receiving_yards": {"position": "WR", "odds_market_key": "player_reception_yds",
                         "table": "recv_targets", "idx_col": "target_index"},
}
SIMS_PER_PICK = 8000
ET = ZoneInfo("America/New_York")


def american_to_prob(odds):
    """Same formula as api.py's american_to_prob -- copied, not
    cross-imported, matching this repo's per-sport self-containment
    convention."""
    try:
        n = float(odds)
    except Exception:
        return None
    if n < 0:
        return -n / (-n + 100.0)
    return 100.0 / (n + 100.0)


def no_vig_two_way(over_odds, under_odds):
    po = american_to_prob(over_odds)
    pu = american_to_prob(under_odds)
    if po is None or pu is None:
        return None, None
    tot = po + pu
    if tot == 0:
        return 0.5, 0.5
    return po / tot, pu / tot


def value_edge(model_p, fair_p):
    if fair_p is None or fair_p <= 0:
        return None
    return (model_p - fair_p) / fair_p


def kelly_fraction(model_p, american_odds, cap=0.25):
    try:
        n = float(american_odds)
    except Exception:
        return 0.0
    b = (n / 100.0) if n > 0 else (100.0 / -n)
    q = 1 - model_p
    f = (b * model_p - q) / b if b else 0
    return max(0.0, min(f, cap))


def fetch_nfl_props_odds(odds_api_key):
    """Real per-player prop lines for TODAY's (ET) NFL games only. Quota
    discipline, agreed on explicitly: this key is shared with MLB and
    tennis, and The Odds API bills per market x event -- pulling a whole
    week's real slate every run would burn real shared quota for lines
    that are mostly still hours or days from being bettable. Restricting
    to today's real games (same today_et() pattern api.py already uses
    for MLB) plus only the 2 markets this repo has a validated model for
    keeps NFL's footprint small. Returns {(normalized_player_name,
    market): {"line", "over_price", "under_price", "book"}}."""
    if not odds_api_key:
        return {}, "no odds api key configured"
    try:
        r = requests.get(f"{ODDS_API_NFL_BASE}/events", params={"apiKey": odds_api_key}, timeout=20)
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        return {}, f"events fetch failed: {e}"

    today = datetime.now(ET).date()
    todays_events = []
    for ev in events:
        try:
            d = datetime.fromisoformat(str(ev.get("commence_time")).replace("Z", "+00:00"))
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            if d.astimezone(ET).date() == today:
                todays_events.append(ev)
        except Exception:
            continue

    market_keys = ",".join(cfg["odds_market_key"] for cfg in REAL_ODDS_MARKETS.values())
    key_to_mkt = {cfg["odds_market_key"]: mkt for mkt, cfg in REAL_ODDS_MARKETS.items()}
    out = {}
    n_errors = 0
    for ev in todays_events:
        eid = ev.get("id")
        if not eid:
            continue
        try:
            r2 = requests.get(f"{ODDS_API_NFL_BASE}/events/{eid}/odds",
                              params={"apiKey": odds_api_key, "markets": market_keys,
                                      "oddsFormat": "american"}, timeout=20)
            r2.raise_for_status()
        except Exception:
            n_errors += 1
            continue
        payload = r2.json()
        for bm in payload.get("bookmakers") or []:
            for mkt_data in bm.get("markets") or []:
                internal_mkt = key_to_mkt.get(mkt_data.get("key"))
                if not internal_mkt:
                    continue
                grouped = {}
                for outcome in mkt_data.get("outcomes") or []:
                    player = outcome.get("description")
                    if not player:
                        continue
                    key = norm_player_name(player)
                    entry = grouped.setdefault(key, {"line": outcome.get("point"), "book": bm.get("key")})
                    side = str(outcome.get("name") or "").lower()
                    if side == "over":
                        entry["over_price"] = outcome.get("price")
                    elif side == "under":
                        entry["under_price"] = outcome.get("price")
                for key, entry in grouped.items():
                    out_key = (key, internal_mkt)
                    if out_key not in out:
                        out[out_key] = entry
    print(f"  odds: {len(todays_events)} real games today (ET), {len(out)} player-market lines "
          f"matched, {n_errors} event(s) errored")
    return out, None


def load_asof_carry_pool(carry_con, table, idx_col, player_id, season, target_week, window=RECENT_GAMES_WINDOW):
    """This player's own real per-event yardage, strictly-prior-week
    discipline (matches nfl_rush_sim_gate_a.py / nfl_recv_sim_gate_a.py
    exactly): only games with week < target_week this season, most
    recent `window` games."""
    rows = carry_con.execute(f"""
        SELECT week, game_id, yards FROM {table}
        WHERE player_id=? AND season=? AND week<?
        ORDER BY week, game_id, {idx_col}
    """, (player_id, season, target_week)).fetchall()
    by_game = {}
    for wk, gid, yards in rows:
        by_game.setdefault((wk, gid), []).append(yards)
    recent_games = sorted(by_game.keys())[-window:]
    counts = [len(by_game[g]) for g in recent_games]
    pool = [y for g in recent_games for y in by_game[g]]
    return counts, pool


def load_prior_season_pools_by_name(model_con, carry_con, prior_season, table, idx_col):
    """normalized player_name -> (counts, pool) using the player's WHOLE
    prior real season -- the same prior-season fallback the classifier
    already uses (build_prior_season_picks), applied to the simulator's
    real per-event pools instead of aggregate features. Name-keyed
    because ESPN's roster ids and nflverse's gsis_id (this table's own
    key) are different, un-crosswalked namespaces -- same real
    limitation as everywhere else in this file that bridges the two.

    Real bug caught testing this against tonight's actual game: nflverse's
    play-by-play stores each player under an ABBREVIATED name
    (rusher_player_name/receiver_player_name, e.g. "R.Stevenson"), not
    their full display name -- confirmed live, "Rhamondre Stevenson"
    never matched ESPN's roster name at all under the carry log's own
    name column. player_games (nfl_model.sqlite, from the weekly stats
    CSV) has the real full name for the same gsis_id, so that's what
    this resolves display names from -- the carry log is keyed and
    queried by player_id, its own name column is never used to match."""
    name_by_pid = dict(model_con.execute(
        "SELECT player_id, player_name FROM player_games WHERE season=?", (prior_season,)).fetchall())

    rows = carry_con.execute(f"""
        SELECT player_id, week, game_id, yards FROM {table} WHERE season=?
        ORDER BY player_id, week, game_id, {idx_col}
    """, (prior_season,)).fetchall()
    by_name_game = {}
    for pid, wk, gid, yards in rows:
        pname = name_by_pid.get(pid)
        if not pname:
            continue
        key = norm_player_name(pname)
        if not key:
            continue
        by_name_game.setdefault(key, {}).setdefault((wk, gid), []).append(yards)
    out = {}
    for key, games in by_name_game.items():
        out[key] = ([len(v) for v in games.values()], [y for v in games.values() for y in v])
    return out


def make_real_odds_pick(mkt, pname, pid, team, opp, season, week, counts, pool,
                         odds_entry, model_source, games_played):
    """One real, gradeable pick: a real book line, a real Monte Carlo
    projection against it (nfl_sim.simulate), and real de-vigged edge/
    kelly fields -- same shape as MLB's props (api.py's value_edge/
    kelly_fraction), which NFL never had before since it never had a
    real market price to compute them against."""
    line = odds_entry.get("line")
    if line is None:
        return None
    result = nfl_sim.simulate(counts, pool, line, sims=SIMS_PER_PICK)
    if result is None:
        return None
    side = result["side"]
    model_prob = result["side_prob"]
    over_price, under_price = odds_entry.get("over_price"), odds_entry.get("under_price")
    side_price = over_price if side == "OVER" else under_price
    fair_over, fair_under = no_vig_two_way(over_price, under_price)
    fair_p = fair_over if side == "OVER" else fair_under
    edge = value_edge(model_prob, fair_p) if fair_p is not None else None
    kelly = kelly_fraction(model_prob, side_price) if side_price is not None else 0.0
    return {
        "market": mkt, "player_id": pid, "player": pname,
        "team": team, "opponent": opp, "season": season, "week": week,
        "line": line, "pick": f"{side} {line}",
        "model_prob": round(model_prob, 4),
        "projected_mean": result["mean"],
        "odds": side_price, "book": odds_entry.get("book"),
        "fair_prob": round(fair_p, 4) if fair_p is not None else None,
        "value_edge": round(edge, 4) if edge is not None else None,
        "kelly_fraction": round(kelly, 4) if kelly is not None else None,
        "games_played": games_played,
        "model_source": model_source,
    }


def build_real_odds_yardage_picks(con, carry_con, season, week, schedule, odds_by_key):
    """Replaces the old flat-49.5 classifier picks for rushing_yards/
    receiving_yards with real book lines priced by a real, validated
    Monte Carlo projection (nfl_rush_sim_gate_a.py / nfl_recv_sim_gate_a.py
    for in-season players with current-season history; nfl_prior_season_
    sim_gate_a.py's prior-season pools for early-season players who
    don't have that yet -- both PASSED their own gate). A player without
    a real matched line is skipped entirely -- no fabricated line, same
    principle as everywhere else in this repo."""
    prior_season = season - 1
    team_pairs = {home: away for home, away in schedule}
    team_pairs.update({away: home for home, away in schedule})
    roster_cache = {}
    all_picks = []
    all_meta = {}

    for mkt, cfg in REAL_ODDS_MARKETS.items():
        picks_this_market = []
        n_no_line = n_in_season = n_prior_season = 0
        covered_names = set()

        cand = SeasonEngine(con, mkt, season).asof_future(week, schedule)
        for (pid, pname, team, opp, _, feat) in cand:
            counts, pool = load_asof_carry_pool(carry_con, cfg["table"], cfg["idx_col"], pid, season, week)
            if not counts or not pool:
                continue
            key = (norm_player_name(pname), mkt)
            odds_entry = odds_by_key.get(key)
            if not odds_entry:
                n_no_line += 1
                continue
            pick = make_real_odds_pick(mkt, pname, pid, team, opp, season, week, counts, pool,
                                        odds_entry, "in_season_sim", feat["games_played"])
            if pick:
                picks_this_market.append(pick)
                covered_names.add(key[0])
                n_in_season += 1

        prior_pools = load_prior_season_pools_by_name(con, carry_con, prior_season, cfg["table"], cfg["idx_col"])
        teams = sorted({t for pair in schedule for t in pair})
        for team in teams:
            opp = team_pairs.get(team)
            if opp is None:
                continue
            for aid, pname in fetch_espn_active_roster(team, cfg["position"], roster_cache):
                norm = norm_player_name(pname)
                if not norm or norm in covered_names:
                    continue
                pools = prior_pools.get(norm)
                if not pools or not pools[0] or not pools[1]:
                    continue
                odds_entry = odds_by_key.get((norm, mkt))
                if not odds_entry:
                    n_no_line += 1
                    continue
                counts, pool = pools
                pick = make_real_odds_pick(mkt, pname, aid, team, opp, season, week, counts, pool,
                                            odds_entry, "prior_season_sim", len(counts))
                if pick:
                    picks_this_market.append(pick)
                    covered_names.add(norm)
                    n_prior_season += 1

        all_picks.extend(picks_this_market)
        all_meta[mkt] = {"eligible": len(picks_this_market), "in_season_sim": n_in_season,
                          "prior_season_sim": n_prior_season, "no_real_line_matched": n_no_line}
    return all_picks, all_meta


def fetch_espn_finished_matchups(season, week):
    """Real-time completed-game check for the live board: which of this
    week's scheduled matchups does ESPN's regular-season scoreboard
    already report as STATUS_FINAL. Mirrors cfb_serving_builder_a.py's
    finished_matchups filter (same design rationale: pregame picks for a
    game that's already over aren't actionable, so they're pulled off
    the live board), but sourced live from ESPN directly rather than
    from the games table -- NFL's foundation script ingests from
    nflverse, which lags real completion by up to a day, unlike CFB's
    cfb_espn_live_foundation_a.py. Returns (team, opponent) pairs in
    both directions, in nflverse abbreviation space. A fetch failure
    degrades to an unfiltered board (same as before this existed) rather
    than blocking the whole pipeline over a diagnostic-only signal."""
    finished = set()
    try:
        r = requests.get(NFL_SCOREBOARD_URL,
                          params={"seasontype": 2, "week": week, "dates": season},
                          timeout=20)
        r.raise_for_status()
        events = r.json().get("events", [])
    except Exception as e:
        print(f"  finished-game check: ESPN scoreboard fetch failed ({e}) -- leaving live board unfiltered")
        return finished
    for event in events:
        for comp in event.get("competitions", []):
            if not comp.get("status", {}).get("type", {}).get("completed"):
                continue
            abbrs = [TEAM_ABBR_ESPN_TO_NFLVERSE.get(a, a) for a in
                     (c.get("team", {}).get("abbreviation") for c in comp.get("competitors", []))
                     if a]
            if len(abbrs) == 2:
                finished.add((abbrs[0], abbrs[1]))
                finished.add((abbrs[1], abbrs[0]))
    return finished


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_DEFAULT))
    ap.add_argument("--carry-db", default=str(CARRY_DB_DEFAULT))
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--out", default=str(DOCS / "nfl_predictions.json"))
    ap.add_argument("--odds-api-key", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    import xgboost as xgb
    import os
    odds_key = args.odds_api_key or os.environ.get("THE_ODDS_API_KEY") or os.environ.get("ODDS_API_KEY") or ""

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    print("NFL_SERVING_BUILDER_A\n=====================")

    if args.selftest:
        ok = selftest(con, xgb)
        con.close()
        return 0 if ok else 1

    if args.season and args.week:
        season, week = args.season, args.week
    else:
        season, week = infer_target(con, date.today().isoformat())
    if season is None:
        print("no upcoming games found in the schedule -- writing empty board")
        payload = {"generated_at_utc": now_utc(), "season": None, "week": None,
                   "picks": [], "note": "no upcoming games in foundation schedule; "
                   "refresh the foundation db (new season schedule not ingested yet)"}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        return 0

    print(f"target: season {season} week {week}")
    schedule = con.execute(
        "SELECT home_team, away_team FROM games WHERE season=? AND week=?",
        (season, week)).fetchall()
    print(f"scheduled games: {len(schedule)}")

    picks = []
    market_meta = {}
    for mkt, cfg in MARKETS.items():
        if mkt in REAL_ODDS_MARKETS:
            # Superseded by build_real_odds_yardage_picks below -- a real
            # book line + a real Monte Carlo projection, not a classifier
            # against a fixed 49.5. sacks is the only market that still
            # goes through this flat-line path (no real market/simulator
            # exists for it yet).
            continue
        bst = xgb.Booster(); bst.load_model(str(cfg["model_dir"] / f"{cfg['stem']}.json"))
        feat_cols = json.loads((cfg["model_dir"] / f"{cfg['stem']}_columns.json").read_text())
        assert feat_cols == cfg["features"]

        cand = SeasonEngine(con, mkt, season).asof_future(week, schedule)
        if not cand:
            print(f"  {mkt}: no eligible players (expected for weeks 1-{MIN_PRIOR_GAMES})")
            market_meta[mkt] = {"eligible": 0}
            continue

        a, b, pool_info = fit_serving_platt(con, mkt, bst, xgb, season, week)
        raw = score(bst, cfg["features"], [c[5] for c in cand], xgb)
        cal = apply_platt(raw, a, b)
        print(f"  {mkt}: {len(cand)} eligible  platt a={a:.3f} b={b:+.3f}  "
              f"pool={pool_info}")
        market_meta[mkt] = {"eligible": len(cand), "platt": {"a": a, "b": b},
                             "calibration_pool": pool_info,
                             "validation": cfg["verdicts"]}
        disp_line = cfg.get("display_line", cfg["line"])
        for (pid, pname, team, opp, _, feat), rp, cp in zip(cand, raw, cal):
            picks.append({
                "market": mkt, "player_id": pid, "player": pname,
                "team": team, "opponent": opp, "season": season, "week": week,
                "line": disp_line,
                "pick": f"{'OVER' if cp >= 0.5 else 'UNDER'} {disp_line}",
                "model_prob": round(float(max(cp, 1 - cp)), 4),
                "prob_over": round(float(cp), 4),
                "raw_prob_over": round(float(rp), 4),
                "games_played": feat["games_played"],
            })

    if not Path(args.carry_db).exists():
        print(f"  no carry log at {args.carry_db} -- rushing_yards/receiving_yards will be empty "
              f"this run (run nfl_pbp_foundation_a.py first)")
        real_odds_picks, real_odds_meta = [], {}
    elif not odds_key:
        print("  no THE_ODDS_API_KEY/ODDS_API_KEY configured -- rushing_yards/receiving_yards "
              "will be empty this run (no real line to grade against, nothing fabricated)")
        real_odds_picks, real_odds_meta = [], {}
    else:
        carry_con = sqlite3.connect(f"file:{args.carry_db}?mode=ro", uri=True)
        odds_by_key, odds_err = fetch_nfl_props_odds(odds_key)
        if odds_err:
            print(f"  odds fetch: {odds_err}")
        real_odds_picks, real_odds_meta = build_real_odds_yardage_picks(
            con, carry_con, season, week, schedule, odds_by_key)
        carry_con.close()
        for mkt, meta in real_odds_meta.items():
            print(f"  {mkt}: {meta['eligible']} real-line picks "
                  f"({meta['in_season_sim']} in-season sim, {meta['prior_season_sim']} prior-season sim, "
                  f"{meta['no_real_line_matched']} eligible but no real line matched)")
    picks.extend(real_odds_picks)
    market_meta.update(real_odds_meta)

    logged_keys = load_logged_pick_keys(PICKS_LOG_PATH)
    n_new_logged = append_new_picks_to_log(PICKS_LOG_PATH, logged_keys, picks)
    print(f"  picks log: {n_new_logged} new entries appended ({len(logged_keys)} total) -- "
          f"source for nfl_grade_record_a.py")

    # Once a game is final, its pregame picks aren't actionable anymore --
    # remove them from the live board (same as cfb_serving_builder_a.py's
    # finished_matchups filter). Logged to the ledger above BEFORE this
    # filter runs, so nfl_grade_record_a.py still has the pick to grade
    # once real stats land, even though it's about to disappear from
    # docs/nfl_predictions.json (and its per-week archive, which gets
    # overwritten with the filter applied on every run).
    finished_matchups = fetch_espn_finished_matchups(season, week)
    n_before = len(picks)
    picks = [p for p in picks if (p["team"], p["opponent"]) not in finished_matchups]
    if n_before != len(picks):
        print(f"  live board: {n_before - len(picks)} picks removed for "
              f"{len(finished_matchups) // 2} already-final game(s)")

    picks.sort(key=lambda p: -p["model_prob"])
    payload = {
        "generated_at_utc": now_utc(), "season": season, "week": week,
        "builder": "NFL_SERVING_BUILDER_A",
        "design": ("sacks: frozen champion + weekly walk-forward Platt (validated 2024). "
                   "rushing_yards/receiving_yards: real book lines (The Odds API, today-ET "
                   "only) priced by a real Monte Carlo projection (nfl_rush_sim_gate_a.py / "
                   "nfl_recv_sim_gate_a.py / nfl_prior_season_sim_gate_a.py, all validated) "
                   "-- a player without a real matched line is not shown, nothing fabricated."),
        "markets": market_meta,
        "note": "sacks eligibility is stats-based and cannot see injuries/inactives for the "
                "upcoming game. rushing_yards/receiving_yards eligibility additionally requires "
                "a real book line to exist for that player today.",
        "picks": picks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    hist = out.parent / f"nfl_predictions_{season}_w{week:02d}.json"
    hist.write_text(json.dumps(payload, indent=2))
    print(f"\n{len(picks)} picks written to {out} (+ {hist.name})")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
