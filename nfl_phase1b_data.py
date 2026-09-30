"""
NFL_PHASE1B_DATA  (Phase 1B, shadow research)

Per-opportunity outcome tallies + AS-OF features for every Phase 1A candidate player-game.

Unit of analysis: (team-game, candidate). All pregame features are as-of the GAME (not the play), so every carry / target /
attempt of a player in a game shares one feature record. That is exactly the interface the Phase 1A opportunity samples
need in the development assembly, and it keeps training and serving identical.

Reuse (no duplicated parser):
  * nfl_context_v4.PlayData  -> opponent defensive scheme profile (box / stacked-box rate, blitz, pressure, man/zone,
    Cover 0/1/2/3/4/6, explosive-run / explosive-pass rates allowed, sack rate, red-zone TD rates allowed) and the
    player-vs-scheme weekly splits (RB vs stacked/light boxes, receiver vs man/zone). Its as-of rule is week-level
    ('games strictly before this week'), which is conservative relative to Phase 1A's kickoff-level cutoff.
  * nfl_phase1_data.Data / units -> candidates, injury report (assumption A2), defenders and their T-24h statuses.
New here: per-PLAY outcome tallies (yards, air yards, catch, YAC, sack, INT, TD; RZ flag) per (season, week, player), decayed
as-of accumulators (player, position, league, team offense, team defense allowed), OL availability (injury-report based).

Never used: realized same-game participation, same-game box / personnel / coverage, sportsbook columns.
2026: participation (coverage, pressure, man/zone) is unpublished, so those rates come from prior-season accumulators
that simply stop growing (stale but as-of legal); FTN box / blitz continue through 2026.
"""
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np

import nfl_phase1_data as P1

GAMMAS = np.array([1.0, 0.985, 0.95])          # weekly recency decay candidates (tuned on validation per component)
NG = len(GAMMAS)
GRID = np.arange(-10, 100)                       # integer support for yards / air yards / YAC (110 points)
G = len(GRID)
RZ = 20


def week_index(s, w):
    return (s - 2022) * 22 + w


def gclip(y):
    return int(min(max(y, -10), 99)) + 10


# ------------------------------------------------------------------ bin systems (indices into GRID)
def make_bins(lumps_lo, singles, coarse, hi=99):
    """Bins as (lo, hi) inclusive integer intervals: one lower lump, singles, coarse upper bins."""
    bins = [lumps_lo] + [(v, v) for v in singles] + list(coarse)
    return bins


RUSH_BINS = make_bins((-10, -4), range(-3, 30), [(30, 34), (35, 39), (40, 49), (50, 59), (60, 79), (80, 99)])      # 40 bins
AIR_BINS = make_bins((-10, -6), range(-5, 26), [(26, 29), (30, 34), (35, 39), (40, 49), (50, 99)])                  # 37 bins
YAC_BINS = make_bins((-10, -1), range(0, 31), [(31, 35), (36, 40), (41, 50), (51, 60), (61, 99)])                    # 37 bins


def bin_matrix(bins):
    """M [G, K] (indicator of grid point in bin) and E [K, G] (uniform expansion of a bin over its grid points)."""
    K = len(bins)
    M = np.zeros((G, K)); E = np.zeros((K, G))
    for k, (lo, hi) in enumerate(bins):
        idx = np.arange(lo, hi + 1) + 10
        M[idx, k] = 1.0
        E[k, idx] = 1.0 / len(idx)
    assert M.sum() == G and abs(E.sum() - K) < 1e-9
    return M, E


M_R, E_R = bin_matrix(RUSH_BINS)
M_A, E_A = bin_matrix(AIR_BINS)
M_Y, E_Y = bin_matrix(YAC_BINS)
K_R, K_A, K_Y = len(RUSH_BINS), len(AIR_BINS), len(YAC_BINS)


def air_bucket(a):
    return 0 if a < 0 else 1 if a <= 4 else 2 if a <= 14 else 3


# ------------------------------------------------------------------ decayed accumulators
class Store:
    """key -> [NG, dim] decayed sums, as of a week index (lazy decay)."""

    def __init__(self, dim):
        self.dim = dim
        self.d = {}

    def get(self, key, wi):
        e = self.d.get(key)
        if e is None:
            return np.zeros((NG, self.dim))
        vec, last = e
        return vec * (GAMMAS ** (wi - last))[:, None] if wi > last else vec

    def add(self, key, delta, wi):
        cur = self.get(key, wi)
        self.d[key] = [cur + delta[None, :], wi]


def fnum(v):
    try:
        x = float(v)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ per-play tallies
class PlayTallies:
    """Per (season, week, gid) and per (season, week, team) tallies from play-by-play (same filters as Phase 1A)."""

    def __init__(self, data_dir, seasons):
        d = Path(data_dir)
        self.rush = defaultdict(lambda: np.zeros(G))                 # (s,w,gid) exact yard histogram
        self.rush_ev = defaultdict(lambda: np.zeros(6))              # rz_n, rz_td, out_n, out_td, outside_dir_n, inside_dir_n
        self.rush_team = defaultdict(lambda: np.zeros(8))            # (s,w,team) offense; and ('D', ...) defense allowed
        self.tgt_air = defaultdict(lambda: np.zeros(G))              # exact air-yard histogram (all targets, air defined)
        self.tgt_cb = defaultdict(lambda: np.zeros((4, 2)))          # by air bucket: [targets, catches]
        self.tgt_yac = defaultdict(lambda: np.zeros((4, G)))         # exact YAC histogram of catches by air bucket
        self.tgt_ev = defaultdict(lambda: np.zeros(4))               # rz_n, rz_td, out_n, out_td (targets)
        self.tgt_n_total = defaultdict(float)                        # all targets incl. air undefined (reconciliation)
        self.pass_ = defaultdict(lambda: np.zeros(6))                # dropbacks, sacks, attempts, comps, ints, tds
        self.pass_air = defaultdict(lambda: np.zeros(G))
        self.pass_cb = defaultdict(lambda: np.zeros((4, 2)))
        self.pass_yac = defaultdict(lambda: np.zeros((4, G)))
        self.recv_team = defaultdict(lambda: np.zeros(5))            # (s,w,team): tgt, catches, air_sum, yac_sum, catches (air defined)
        self.pass_team = defaultdict(lambda: np.zeros(8))            # (s,w,team) passing side (offense) and 'D' allowed
        self.play_rows = 0
        for s in seasons:
            with gzip.open(d / f"pbp_{s}.csv.gz", "rt", newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r["season_type"] != "REG" or not r["posteam"] or r["play_type"] not in ("pass", "run"):
                        continue
                    if r.get("two_point_attempt") == "1" or r.get("qb_kneel") == "1" or r.get("qb_spike") == "1":
                        continue
                    self.play_rows += 1
                    w = int(r["week"]); off, de = r["posteam"], r["defteam"]
                    yl = fnum(r["yardline_100"]); rz = yl is not None and yl <= RZ
                    rid, qid, pid = r["rusher_player_id"], r["receiver_player_id"], r["passer_player_id"]
                    if r["rush_attempt"] == "1" and rid:
                        y = fnum(r["rushing_yards"]) or 0.0
                        self.rush[(s, w, rid)][gclip(y)] += 1
                        ev = self.rush_ev[(s, w, rid)]
                        td = r["rush_touchdown"] == "1"
                        ev[0 if rz else 2] += 1; ev[1 if rz else 3] += td
                        loc = r.get("run_location") or ""
                        if loc in ("left", "right"):
                            ev[4] += 1                              # outside-tackle-ish (left/right); 'middle' = inside
                        elif loc == "middle":
                            ev[5] += 1
                        for key in ((s, w, off), ("D", s, w, de)):
                            t = self.rush_team[key]
                            t[0] += 1; t[1] += y; t[2] += y < 0; t[3] += y >= 15; t[4] += y >= 20; t[5] += y >= 25
                            t[6] += y if y < 20 else 0.0; t[7] += y < 20
                    is_pass = r["pass_attempt"] == "1"
                    if r["qb_dropback"] == "1" and pid:
                        pt = self.pass_[(s, w, pid)]
                        pt[0] += 1
                        sk = r["sack"] == "1"
                        pt[1] += sk
                        for key in ((s, w, off), ("D", s, w, de)):
                            tt = self.pass_team[key]; tt[0] += 1; tt[1] += sk
                        if is_pass and not sk:
                            pt[2] += 1; pt[3] += r["complete_pass"] == "1"; pt[4] += r["interception"] == "1"; pt[5] += r["pass_touchdown"] == "1"
                            for key in ((s, w, off), ("D", s, w, de)):
                                tt = self.pass_team[key]; tt[2] += 1; tt[3] += r["complete_pass"] == "1"; tt[4] += r["interception"] == "1"; tt[5] += r["pass_touchdown"] == "1"
                    elif is_pass and pid and r["sack"] != "1":
                        # attempts without a recorded dropback flag (rare): still count the attempt
                        pt = self.pass_[(s, w, pid)]
                        pt[0] += 1; pt[2] += 1; pt[3] += r["complete_pass"] == "1"; pt[4] += r["interception"] == "1"; pt[5] += r["pass_touchdown"] == "1"
                    if is_pass and r["sack"] != "1":
                        ay = fnum(r["air_yards"])
                        comp = r["complete_pass"] == "1"
                        ry = (fnum(r["receiving_yards"]) or 0.0) if comp else 0.0
                        if qid:
                            self.tgt_n_total[(s, w, qid)] += 1
                            tev = self.tgt_ev[(s, w, qid)]
                            tdp = r["pass_touchdown"] == "1"
                            tev[0 if rz else 2] += 1; tev[1 if rz else 3] += tdp
                        if ay is not None:
                            b = air_bucket(ay)
                            if pid:
                                self.pass_air[(s, w, pid)][gclip(ay)] += 1
                                self.pass_cb[(s, w, pid)][b, 0] += 1; self.pass_cb[(s, w, pid)][b, 1] += comp
                                if comp:
                                    self.pass_yac[(s, w, pid)][b, gclip(ry - ay)] += 1
                            if qid:
                                self.tgt_air[(s, w, qid)][gclip(ay)] += 1
                                self.tgt_cb[(s, w, qid)][b, 0] += 1; self.tgt_cb[(s, w, qid)][b, 1] += comp
                                if comp:
                                    self.tgt_yac[(s, w, qid)][b, gclip(ry - ay)] += 1
                            for key in ((s, w, off), ("D", s, w, de)):
                                rt = self.recv_team[key]
                                rt[0] += 1; rt[1] += comp
                                if comp:
                                    rt[2] += ay; rt[3] += ry - ay; rt[4] += 1
        self.players_by_week = defaultdict(set)
        for store in (self.rush, self.tgt_air, self.pass_):
            for (s, w, gid) in store:
                self.players_by_week[(s, w)].add(gid)
