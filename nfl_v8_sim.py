#!/usr/bin/env python3
"""NFL V8: whole-game Monte Carlo (the NFL counterpart of the NBA V5 simulation). Research model - not serving.

One simulated game per draw, for one team's offence:
  1. game script: final margin ~ N(spread, 13) and total points ~ N(total line, 9.5): a trailing team throws more, a leading team runs more;
  2. plays: team pace (own and opponent, shrunk to the league) scaled by the total-points draw; pass rate = team base + learned
     slope x margin; targets per dropback learned from play-by-play;
  3. who plays: each player draws availability from P(plays | reported status) learned from earlier weeks (unlisted: league rate);
  4. touches: carries and targets are divided among the AVAILABLE players by recent shares (decayed over the last games) with
     game-to-game share noise, so a missing teammate's touches go to the ones who actually take them;
  5. outcomes: each carry / target draws a real result (yards, catch) from the player's own recent plays mixed with the position's league
     plays (more weight on his own as his sample grows), scaled by the opponent's yards-per-play allowed (shrunk);
  6. one shared game links every player on the team through the script, plays and touches.

Everything is built from games strictly before the one being projected (walk-forward).  No sportsbook price is used.
"""
from __future__ import annotations
import gzip
import math
from collections import defaultdict

import numpy as np
import pandas as pd

POS_RUSH = ("RB", "FB", "QB", "WR", "TE")
POS_REC = ("WR", "TE", "RB", "FB")
HALF_LIFE = 3.0
SHARE_GAMES = 8
OWN_K_CARRY = 40.0
OWN_K_TARGET = 60.0
PLAYER_POOL_MAX = 250
OPP_SHRINK = 0.5
SHARE_CONC = 14.0           # Dirichlet concentration of game-to-game share noise
MARGIN_SD = 13.2
TOTAL_SD = 9.5


def keyof(season, week):
    return season * 100 + week


class Plays:
    """Play-by-play tables, indexed so every lookup can be restricted to earlier weeks."""

    def __init__(self, pbp_paths, pos_of):
        cols = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "yards_gained", "complete_pass",
                "pass_attempt", "sack", "qb_kneel", "qb_spike", "two_point_attempt", "rusher_player_id", "receiver_player_id",
                "posteam_type", "result", "total", "spread_line"]
        frames = [pd.read_csv(p, usecols=lambda c: c in cols, low_memory=False) for p in pbp_paths]
        d = pd.concat(frames, ignore_index=True)
        d = d[(d.season_type == "REG") & d.posteam.notna()]
        d = d[(d.two_point_attempt.fillna(0) == 0) & (d.qb_kneel.fillna(0) == 0) & (d.qb_spike.fillna(0) == 0)]
        d["key"] = d.season * 100 + d.week
        self.pos_of = pos_of
        run = d[(d.play_type == "run") & d.rusher_player_id.notna()]
        tgt = d[(d.pass_attempt == 1) & d.receiver_player_id.notna()]
        # per-player event arrays sorted by key
        self.carry = self._group(run, "rusher_player_id", ["yards_gained"])
        self.targets = self._group(tgt, "receiver_player_id", ["yards_gained", "complete_pass"])
        # league pools by position group (keyed for time restriction)
        self.run_all = run.assign(pos=run.rusher_player_id.map(pos_of))[["key", "pos", "yards_gained"]].dropna()
        self.tgt_all = tgt.assign(pos=tgt.receiver_player_id.map(pos_of))[["key", "pos", "yards_gained", "complete_pass"]].dropna()
        # team-game aggregates
        pl = d[(d.play_type.isin(["run", "pass"]))]
        pl = pl.assign(is_pass=(pl.play_type == "pass").astype(int))
        g = pl.groupby(["game_id", "season", "week", "key", "posteam", "defteam"]).agg(
            plays=("play_type", "size"), passes=("is_pass", "sum"), yds=("yards_gained", "sum"),
            sacks=("sack", "sum")).reset_index()
        runs = run.groupby(["game_id", "posteam"]).agg(rush_yds=("yards_gained", "sum"), rush_att=("yards_gained", "size")).reset_index()
        pas = tgt.groupby(["game_id", "posteam"]).agg(pass_yds=("yards_gained", "sum"), n_tgt=("yards_gained", "size")).reset_index()
        g = g.merge(runs, on=["game_id", "posteam"], how="left").merge(pas, on=["game_id", "posteam"], how="left").fillna(0)
        res = d.groupby("game_id").agg(result=("result", "first"), posteam_type=("posteam_type", "first")).reset_index()
        g = g.merge(d.groupby(["game_id", "posteam"]).posteam_type.first().reset_index(), on=["game_id", "posteam"], how="left", suffixes=("", "_x"))
        gm = d.groupby("game_id").result.first()
        g["margin"] = [(gm.get(gid, 0.0) or 0.0) * (1 if t == "home" else -1) for gid, t in zip(g.game_id, g.posteam_type)]
        g["pass_rate"] = g.passes / g.plays.clip(lower=1)
        g["tgt_per_db"] = g.n_tgt / (g.passes).clip(lower=1)
        self.tg = g.sort_values("key").reset_index(drop=True)
        self.tg_by_team = {t: x.reset_index(drop=True) for t, x in self.tg.groupby("posteam")}
        # team-game: who got the carries/targets (for share windows)
        self.rush_by_game = run.groupby(["posteam", "game_id", "key", "rusher_player_id"]).size().reset_index(name="n")
        self.tgt_by_game = tgt.groupby(["posteam", "game_id", "key", "receiver_player_id"]).size().reset_index(name="n")
        self.defense = self.tg.groupby(["defteam"])

    @staticmethod
    def _group(df, idcol, cols):
        out = {}
        df = df.sort_values("key")
        for pid, x in df.groupby(idcol):
            out[pid] = (x["key"].to_numpy(), x[cols].to_numpy(dtype=float))
        return out


def pool_before(store, pid, key, n_max=PLAYER_POOL_MAX):
    e = store.get(pid)
    if e is None:
        return np.zeros((0, 1))
    keys, arr = e
    i = np.searchsorted(keys, key, side="left")
    return arr[max(0, i - n_max):i]


def league_pool(df, key, groups, cols, last_keys=300):
    sub = df[(df.key < key) & df.pos.isin(groups)]
    sub = sub[sub.key >= key - 300]            # roughly the last three seasons
    return sub[cols].to_numpy(dtype=float)


def decayed_shares(by_game, team, key, pos_filter, pos_of, n_games=SHARE_GAMES):
    sub = by_game[(by_game.posteam == team) & (by_game.key < key)]
    games = sorted(sub.key.unique())[-n_games:]
    if not games:
        return {}
    sub = sub[sub.key.isin(games)]
    w = {k: 0.5 ** ((len(games) - 1 - i) / HALF_LIFE) for i, k in enumerate(games)}
    tot = defaultdict(float)
    wsum = 0.0
    per_game_total = sub.groupby("key").n.sum()
    for k in games:
        wsum += w[k]
    for r in sub.itertuples():
        if pos_of.get(r.rusher_player_id if hasattr(r, "rusher_player_id") else r.receiver_player_id) in pos_filter:
            pid = r.rusher_player_id if hasattr(r, "rusher_player_id") else r.receiver_player_id
            tot[pid] += w[r.key] * r.n / max(per_game_total[r.key], 1)
    s = sum(tot.values())
    return {p: v / s for p, v in tot.items()} if s > 0 else {}


def team_rate(tg_team, key, col, n=8, prior=None, k=4.0):
    x = tg_team[tg_team.key < key].tail(n)
    if prior is None:
        prior = float(tg_team[col].mean()) if len(tg_team) else 0.0
    return (float(x[col].sum()) + k * prior) / (len(x) + k)


class State:
    """Everything the simulator needs about one team for one game, from earlier games only."""

    def __init__(self, plays, league):
        self.p = plays
        self.league = league


def pass_slope(plays, key):
    x = plays.tg[plays.tg.key < key]
    x = x[x.key >= key - 300]
    if len(x) < 200:
        return 0.004
    m = x.margin.to_numpy()
    y = x.pass_rate.to_numpy()
    b = np.cov(m, y)[0, 1] / max(np.var(m), 1e-9)
    return float(b)


def build_team_input(plays, team, opp, season, week, spread, total, status_of, p_by_status, pos_of, league_cache):
    key = keyof(season, week)
    tgt = plays.tg_by_team.get(team)
    og = plays.tg_by_team.get(opp)
    if tgt is None or og is None or len(tgt[tgt.key < key]) < 3:
        return None
    lc = league_cache.get(key)
    if lc is None:
        prior = plays.tg[(plays.tg.key < key) & (plays.tg.key >= key - 300)]
        lc = {"plays": float(prior.plays.mean()), "pass_rate": float(prior.pass_rate.mean()), "tpd": float(prior.tgt_per_db.mean()),
              "ypc": float(plays.run_all[(plays.run_all.key < key) & (plays.run_all.key >= key - 300)].yards_gained.mean()),
              "ypt": float(plays.tgt_all[(plays.tgt_all.key < key) & (plays.tgt_all.key >= key - 300)].yards_gained.mean()),
              "slope": pass_slope(plays, key)}
        league_cache[key] = lc
    plays_own = team_rate(tgt, key, "plays", prior=lc["plays"])
    plays_opp = team_rate(og.assign(plays=og.plays), key, "plays", prior=lc["plays"])
    pr_base = team_rate(tgt, key, "pass_rate", prior=lc["pass_rate"])
    margin_avg = team_rate(tgt, key, "margin", prior=0.0)
    pr_base = pr_base - lc["slope"] * margin_avg
    tpd = team_rate(tgt, key, "tgt_per_db", prior=lc["tpd"], k=10)
    # opponent defence: yards per play allowed relative to league (shrunk)
    dg = plays.tg[(plays.tg.defteam == opp) & (plays.tg.key < key)].tail(8)
    def mult(num, den, lg):
        if len(dg) == 0 or dg[den].sum() <= 0:
            return 1.0
        v = dg[num].sum() / dg[den].sum()
        raw = v / lg
        w = len(dg) / (len(dg) + 6.0)
        return float(1.0 + OPP_SHRINK * w * (raw - 1.0))
    opp_run = mult("rush_yds", "rush_att", lc["ypc"])
    opp_pass = mult("pass_yds", "n_tgt", lc["ypt"])
    rs = decayed_shares(plays.rush_by_game.rename(columns={"rusher_player_id": "rusher_player_id"}), team, key, POS_RUSH, pos_of)
    ts = decayed_shares(plays.tgt_by_game, team, key, POS_REC, pos_of)
    players = sorted(set(rs) | set(ts))
    players = [p for p in players if rs.get(p, 0) >= 0.03 or ts.get(p, 0) >= 0.03]
    avail = []
    for p in players:
        st = status_of.get(p, "UNLISTED")
        avail.append(p_by_status.get(st, p_by_status["UNLISTED"]))
    return {"team": team, "opp": opp, "key": key, "spread": spread, "total": total, "plays": 0.6 * plays_own + 0.4 * plays_opp,
            "pr_base": pr_base, "tpd": tpd, "slope": lc["slope"], "opp_run": opp_run, "opp_pass": opp_pass,
            "league": lc, "players": players, "rs": np.array([rs.get(p, 0.0) for p in players]),
            "ts": np.array([ts.get(p, 0.0) for p in players]), "p_play": np.array(avail)}


def _draw_counts(rng, total, weights, conc):
    """Split `total` (S,) among available players: Dirichlet share noise then sequential binomials. weights (S,J) already zero for absent."""
    S, J = weights.shape
    g = rng.gamma(np.maximum(weights * conc, 1e-9), 1.0)
    g = np.where(weights > 0, g, 0.0)
    p = g / np.maximum(g.sum(axis=1, keepdims=True), 1e-12)
    counts = np.zeros((S, J), dtype=np.int32)
    rem = total.astype(np.int64).copy()
    rem_p = np.ones(S)
    for j in range(J):
        q = np.clip(p[:, j] / np.maximum(rem_p, 1e-12), 0, 1)
        c = rng.binomial(rem, q)
        counts[:, j] = c
        rem -= c
        rem_p -= p[:, j]
    return counts


def _sum_draws(rng, n, pool_own, pool_lg, k_own, scale, ncols):
    """Sum of n[s] draws per sim from the mixed pool. Returns (S, ncols) sums."""
    S = len(n)
    mx = int(n.max()) if S else 0
    if mx == 0:
        return np.zeros((S, ncols))
    own_n = len(pool_own)
    w_own = own_n / (own_n + k_own) if own_n else 0.0
    use_own = rng.random((S, mx)) < w_own
    idx_o = rng.integers(0, max(own_n, 1), (S, mx))
    idx_l = rng.integers(0, len(pool_lg), (S, mx))
    mask = np.arange(mx)[None, :] < n[:, None]
    out = np.zeros((S, ncols))
    for c in range(ncols):
        vo = pool_own[idx_o, c] if own_n else np.zeros((S, mx))
        vl = pool_lg[idx_l, c]
        v = np.where(use_own, vo, vl)
        if c == 0:
            v = v * scale
        out[:, c] = (v * mask).sum(axis=1)
    return out


def simulate_team(plays, ti, S, seed, pool_cache, pos_of):
    rng = np.random.default_rng(seed)
    key, lc = ti["key"], ti["league"]
    J = len(ti["players"])
    margin = rng.normal(ti["spread"], MARGIN_SD, S)
    tot = rng.normal(ti["total"], TOTAL_SD, S) if ti["total"] else np.full(S, 45.0)
    base_total = ti["total"] or 45.0
    n_plays = np.clip(np.round(ti["plays"] * (1 + 0.40 * (tot - base_total) / base_total) + rng.normal(0, 4.0, S)), 35, 95).astype(int)
    pr = np.clip(ti["pr_base"] + ti["slope"] * margin, 0.30, 0.78)
    dropbacks = rng.binomial(n_plays, pr)
    n_targets = rng.binomial(dropbacks, np.clip(ti["tpd"], 0.70, 0.98))
    n_rush = n_plays - dropbacks
    play = rng.random((S, J)) < ti["p_play"][None, :]
    out = {}
    rw = np.where(play, ti["rs"][None, :], 0.0)
    tw = np.where(play, ti["ts"][None, :], 0.0)
    rw = np.where(rw.sum(1, keepdims=True) > 0, rw, 1e-6)
    tw = np.where(tw.sum(1, keepdims=True) > 0, tw, 1e-6)
    rush_n = _draw_counts(rng, n_rush, rw, SHARE_CONC)
    tgt_n = _draw_counts(rng, n_targets, tw, SHARE_CONC)
    res = {}
    for j, pid in enumerate(ti["players"]):
        pos = pos_of.get(pid, "WR")
        row = {"plays": play[:, j]}
        if ti["rs"][j] >= 0.03:
            pc = pool_cache.setdefault(("c", pid, key), pool_before(plays.carry, pid, key))
            lgk = ("lc", pos, key)
            lg = pool_cache.get(lgk)
            if lg is None:
                lg = league_pool(plays.run_all, key, ("RB", "FB") if pos in ("RB", "FB") else (pos,), ["yards_gained"])
                if len(lg) < 50:
                    lg = league_pool(plays.run_all, key, POS_RUSH, ["yards_gained"])
                pool_cache[lgk] = lg
            s = _sum_draws(rng, rush_n[:, j], pc, lg, OWN_K_CARRY, ti["opp_run"], 1)
            row["rush_yards"] = s[:, 0]
            row["carries"] = rush_n[:, j]
        if ti["ts"][j] >= 0.03:
            pt = pool_cache.setdefault(("t", pid, key), pool_before(plays.targets, pid, key))
            lgk = ("lt", pos, key)
            lg = pool_cache.get(lgk)
            if lg is None:
                lg = league_pool(plays.tgt_all, key, (pos,), ["yards_gained", "complete_pass"])
                if len(lg) < 50:
                    lg = league_pool(plays.tgt_all, key, POS_REC, ["yards_gained", "complete_pass"])
                pool_cache[lgk] = lg
            s = _sum_draws(rng, tgt_n[:, j], pt, lg, OWN_K_TARGET, ti["opp_pass"], 2)
            row["rec_yards"] = s[:, 0]
            row["receptions"] = s[:, 1]
            row["targets"] = tgt_n[:, j]
        res[pid] = row
    return res


def summarize(samples, played):
    """Distribution of a stat among sims where the player plays (conditional on playing)."""
    x = samples[played] if played.any() else samples
    mean = float(x.mean())
    k = int(math.floor(mean)) + (1 if mean - math.floor(mean) >= 0.6 - 1e-10 else 0)
    return {"mean": mean, "median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
            "whole": k, "reach": float((x >= k).mean()), "p_play": float(played.mean())}
