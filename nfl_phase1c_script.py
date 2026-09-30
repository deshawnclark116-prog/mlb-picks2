"""
NFL_PHASE1C_SCRIPT  (Phase 1C, shadow research)  -- candidate S1: state-aware game script

Static candidate S0: each team's rush attempts / dropbacks are independent Phase 1A negative-binomial draws for the whole game.
State-aware candidate S1: the game is cut into K equal blocks; in every block both teams run their share of the game's plays, the pass/rush split
in the block depends on the SIMULATED score margin and elapsed time through a transition rule learned on prior games only (TRAIN + VALID,
through 2024 wk18), and the margin is advanced by simulated points (team scoring rate as-of x opponent, TD/FG split learned on prior games).

No sportsbook spread / total is used and no actual game-script information from the target game (final score, score by quarter) enters.
The script output is (R, D) per draw per team (same meaning as S0); the player-level event simulation is unchanged.
"""
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import zlib

K_BLOCKS = 8
SCRAMBLE_PER_DROPBACK = 0.05      # league scrambles per dropback (approx; constants.json scramble share x QB carry share)


def fit_transition(data_dir, last=(2024, 18)):
    """Logistic pass-vs-designed-rush transition on prior plays: features lead (posteam margin /7, clipped +-3), progress, interactions."""
    X, y = [], []
    for s in (2022, 2023, 2024):
        with gzip.open(Path(data_dir) / f"pbp_{s}.csv.gz", "rt", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["season_type"] != "REG" or not r["posteam"] or r["play_type"] not in ("pass", "run"):
                    continue
                if (s, int(r["week"])) > last or r.get("two_point_attempt") == "1" or r.get("qb_kneel") == "1" or r.get("qb_spike") == "1":
                    continue
                if r["score_differential"] in ("", "NA") or r["game_seconds_remaining"] in ("", "NA"):
                    continue
                lead = max(-21.0, min(21.0, float(r["score_differential"]))) / 7.0
                prog = 1.0 - float(r["game_seconds_remaining"]) / 3600.0
                is_pass = r["qb_dropback"] == "1"
                is_designed_rush = r["rush_attempt"] == "1" and r["qb_dropback"] != "1"
                if not (is_pass or is_designed_rush):
                    continue
                X.append([lead, lead * prog, lead * prog * prog]); y.append(1.0 if is_pass else 0.0)
    X = np.array(X); y = np.array(y)
    Z = np.column_stack([np.ones(len(X)), X])

    def f(b):
        z = Z @ b; pr = 1 / (1 + np.exp(-z))
        return (-np.sum(y * np.log(np.clip(pr, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - pr, 1e-9, 1))) + 0.5 * np.sum(b[1:] ** 2)) / len(y), (Z.T @ (pr - y) + np.r_[0, b[1:]]) / len(y)
    coef = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
    eta = Z @ coef
    center = float((eta - coef[0]).mean())               # mean lead effect over the league state distribution (removed so pregame pass rates are preserved)
    return {"coef": [float(c) for c in coef], "center_lead_effect": center, "n_plays": int(len(y)), "pass_rate": float(y.mean())}


def team_points_asof(data_dir):
    """games.csv scores -> function(s, w, team) = (as-of points for, points against per game), decayed and shrunk, strictly earlier weeks."""
    rows = []
    for r in csv.DictReader(open(Path(data_dir) / "games.csv", newline="", encoding="utf-8")):
        if r["game_type"] != "REG" or r["result"] in ("", "NA"):
            continue
        rows.append((int(r["season"]), int(r["week"]), r["home_team"], r["away_team"], float(r["home_score"]), float(r["away_score"])))
    rows.sort()
    hist = defaultdict(list)
    lg_sum = {}
    for s, w, h, a, hs, as_ in rows:
        hist[h].append((s, w, hs, as_)); hist[a].append((s, w, as_, hs))

    def f(s, w, team, gamma=0.9, prior=4.0, league=22.5):
        pf = pa = wt = 0.0
        for (ss, ww, x, y_) in reversed(hist[team]):
            if (ss, ww) >= (s, w):
                continue
            k = gamma ** ((s - ss) * 18 + (w - ww))
            pf += x * k; pa += y_ * k; wt += k
        return (pf + prior * league) / (wt + prior), (pa + prior * league) / (wt + prior)
    return f


def td_points_share(data_dir, last=(2024, 18)):
    """Share of team points that come from touchdowns (7 pts) vs field goals (3 pts): from prior plays."""
    td = fg = 0.0
    for s in (2022, 2023, 2024):
        with gzip.open(Path(data_dir) / f"pbp_{s}.csv.gz", "rt", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["season_type"] != "REG" or (s, int(r["week"])) > last:
                    continue
                if r.get("touchdown") == "1" and r.get("td_team") == r.get("posteam"):
                    td += 7
                if r.get("field_goal_result") == "made":
                    fg += 3
    return td / (td + fg)


class ScriptModel:
    def __init__(self, trans, points_fn, td_share, C):
        self.tr, self.pts, self.td_share, self.C = trans, points_fn, td_share, C

    def make(self, pack, D):
        tr = self.tr; coef = np.array(tr["coef"]); cen = tr["center_lead_effect"]
        C = self.C

        def script_fn(s, w, A_, B_, N, seed):
            gA, gB = pack["games"].get((s, w, A_)), pack["games"].get((s, w, B_))
            if gA is None or gB is None:
                return None
            rng = np.random.default_rng(zlib.crc32(repr((seed, s, w, A_, "script")).encode()))
            info = {}
            for tm, g in ((A_, gA), (B_, gB)):
                mu_r, k_r = g["types"]["carry"]["mu"], g["types"]["carry"]["k"]
                mu_d, k_d = g["types"]["qb_att"]["mu"], g["types"]["qb_att"]["k"]
                z = SCRAMBLE_PER_DROPBACK
                mu_plays = max(mu_r + mu_d - z * mu_d, 20.0)
                f_pass = min(max(mu_d / max(mu_r - z * mu_d + mu_d, 1.0), 0.2), 0.9)
                info[tm] = (mu_plays, k_r, f_pass)
            # expected points per team (as-of offence x opponent defence), per play
            pfA, paA = self.pts(s, w, A_); pfB, paB = self.pts(s, w, B_)
            ptsA = max(22.5 * (pfA / 22.5) * (paB / 22.5), 6.0); ptsB = max(22.5 * (pfB / 22.5) * (paA / 22.5), 6.0)
            out = {}
            plays = {tm: rng.negative_binomial(info[tm][1], info[tm][1] / (info[tm][1] + info[tm][0]), size=N) for tm in (A_, B_)}
            lead = np.zeros(N)                                         # margin from A's perspective
            Dtot = {tm: np.zeros(N, np.int64) for tm in (A_, B_)}; Rtot = {tm: np.zeros(N, np.int64) for tm in (A_, B_)}
            for k in range(K_BLOCKS):
                prog = (k + 0.5) / K_BLOCKS
                for tm, sign, pts in ((A_, 1.0, ptsA), (B_, -1.0, ptsB)):
                    l = np.clip(sign * lead, -21, 21) / 7.0
                    delta = coef[1] * l + coef[2] * l * prog + coef[3] * l * prog * prog - cen
                    f = info[tm][2]
                    p = 1.0 / (1.0 + np.exp(-(np.log(f / (1 - f)) + delta)))
                    nk = np.round(plays[tm] / K_BLOCKS).astype(np.int64)
                    d_k = rng.binomial(nk, p)
                    Dtot[tm] += d_k; Rtot[tm] += nk - d_k
                    lam = pts / K_BLOCKS
                    td_k = rng.poisson(lam * self.td_share / 7.0); fg_k = rng.poisson(lam * (1 - self.td_share) / 3.0)
                    lead += sign * (7.0 * td_k + 3.0 * fg_k)
            for tm in (A_, B_):
                z = rng.binomial(Dtot[tm], SCRAMBLE_PER_DROPBACK)
                out[tm] = {"R": (Rtot[tm] + z).astype(np.int64), "D": Dtot[tm].astype(np.int64)}
            return out
        return script_fn
