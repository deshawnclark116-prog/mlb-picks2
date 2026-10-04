"""
CFB_PHASE1_INTEGRATED -- end-to-end evaluation of the coherent core engine: fit on a training window, simulate every validation team-game (both offences), score every player / team outcome distribution (draw-based CRPS, calibration, coverage),
accounting identities, emergent team points vs the direct points component, win probability. Used for burned development (D1-D4) and, unchanged, for the ONE 2025 integrated confirmation from the frozen bundle.
"""
import math
from collections import defaultdict

import numpy as np

import cfb_phase1_common as C
import cfb_phase1_forecast as FC
import cfb_phase1_sim as SIM
import cfb_phase1_team_environment as TE
import cfb_phase1b_components as K

OUTCOMES = {"carries": ("y_carries", "carries"), "rush_yards": ("y_rush_yards", "rush_yards"), "rush_td": ("y_rush_td", "rush_td"), "receptions": ("y_receptions", "receptions"), "rec_yards": ("y_rec_yards", "rec_yards"), "rec_td": ("y_rec_td", "rec_td"),
            "pass_att": ("y_pass_att", "pass_att"), "completions": ("y_completions", "completions"), "pass_yards": ("y_pass_yards", "pass_yards"), "pass_td": ("y_pass_td", "pass_td"), "int": ("y_int", "int")}
POS_FILTER = {"pass_att": lambda r: r["POS_QB"] == 1, "completions": lambda r: r["POS_QB"] == 1, "pass_yards": lambda r: r["POS_QB"] == 1, "pass_td": lambda r: r["POS_QB"] == 1, "int": lambda r: r["POS_QB"] == 1,
              "receptions": lambda r: r["POS_QB"] != 1, "rec_yards": lambda r: r["POS_QB"] != 1, "rec_td": lambda r: r["POS_QB"] != 1}


def crps_draws(x, y):
    """CRPS of the empirical distribution of draws x [M] at y: E|X-y| - 0.5 E|X-X'| (exact for the empirical measure)."""
    xs = np.sort(x).astype(float); M = len(xs)
    e1 = np.abs(xs - y).mean()
    e2 = (2.0 / M ** 2) * np.sum((2 * np.arange(1, M + 1) - M - 1) * xs)
    return e1 - 0.5 * e2


def summarize_draws(x, y, v):
    xs = np.sort(x)
    M = len(xs)
    below = (xs < y).mean(); eq = (xs == y).mean()
    return {"crps": crps_draws(x, y), "mean": float(xs.mean()), "q10": float(xs[int(0.1 * M)]), "q90": float(xs[int(0.9 * M) - 1]), "q05": float(xs[int(0.05 * M)]), "q95": float(xs[int(0.95 * M) - 1]), "pit": float(below + v * eq)}


def simulate_season(engine, rows_by_key, team_by_key, keys, M, seed0=0, collect_players=True):
    """-> per team-game results: draws summaries for player outcomes, team outcomes, identity checks."""
    out = {"players": defaultdict(list), "teams": [], "identity_fail": defaultdict(int), "n_games": 0, "points_draws": {}}
    for gi, k in enumerate(keys):
        cand = rows_by_key.get(k, []); trow = team_by_key[k]
        spec = engine.team_spec(trow, cand)
        o = SIM.simulate_team(spec, M, seed=seed0 + gi)
        for nm, ok in SIM.check_identities(o).items():
            if not ok:
                out["identity_fail"][nm] += 1
        out["n_games"] += 1
        out["points_draws"][k] = o["points"]
        out["teams"].append({"key": k, "season": trow["season"], "week": trow["week"], "y_plays": trow["y_plays"], "y_rush": trow["y_rush"], "y_pass": trow["y_pass"], "y_points": trow["y_points"], "draws": {"plays": o["N"], "rush": o["R"], "att": o["A"], "points": o["points"]}})
        if collect_players:
            for i, r in enumerate(cand):
                for nm, (ycol, dkey) in OUTCOMES.items():
                    pf = POS_FILTER.get(nm)
                    if pf and not pf(r):
                        continue
                    d = o["players"][dkey].get(i)
                    if d is None:
                        d = np.zeros(M, np.int64)                                        # the candidate holds no draw of this outcome in the simulation (e.g. not an allocated receiver) -> exactly zero
                    out["players"][nm].append((r, d))
    return out


# ------------------------------------------------------------------ simple (Phase 1 B0-style) player baselines for the end-to-end comparison
BASE_STAT = {"carries": "P_CARRIES_L20", "rush_yards": "P_RUSH_YDS_L20", "rush_td": "P_RUSH_TD_L20", "receptions": "P_REC_L20", "rec_yards": "P_REC_YDS_L20", "rec_td": "P_REC_TD_L20", "pass_att": "P_ATT_L20", "completions": "P_COMP_L20",
             "pass_yards": "P_PASS_YDS_L20", "pass_td": "P_PASS_TD_L20", "int": "P_INT_L20"}
KMAX = {"carries": 80, "rush_yards": 700, "rush_td": 8, "receptions": 40, "rec_yards": 600, "rec_td": 8, "pass_att": 90, "completions": 70, "pass_yards": 800, "pass_td": 10, "int": 10}
SHRINK_APPS = 3.0


class NaiveBaseline:
    """mean_i = shrunk per-appearance rate (player's last-20 all-team window shrunk to the training position rate at 3 appearances) x participation (P_APPS_L5 shrunk to the training position rate); NB2 with training-only dispersion."""

    def __init__(self, outcome):
        self.o = outcome

    def _pos(self, rows):
        return np.array([K.pos_group(r) for r in rows])

    def fit(self, rows):
        o, st = self.o, BASE_STAT[self.o]
        pf = POS_FILTER.get(o, lambda r: True)
        rows = [r for r in rows if pf(r)]
        g = self._pos(rows); y = np.array([r[OUTCOMES[o][0]] for r in rows], float)
        part = np.array([1.0 if (r["y_part"] if "y_part" in r else (r["y_carries"] + r["y_receptions"] + r["y_pass_att"] > 0)) else 0.0 for r in rows])
        stat = np.array([r[st] for r in rows], float); apps = np.array([r["P_N_APPS_L20"] for r in rows], float)
        self.rate = {k: float(stat[g == k].sum() / max(apps[g == k].sum(), 1)) for k in set(g.tolist())}
        self.pi = {k: float(part[g == k].mean()) for k in set(g.tolist())}
        self.all_rate = float(stat.sum() / max(apps.sum(), 1)); self.all_pi = float(part.mean())
        mu = self._mean(rows)
        self.alpha = TE.fit_alpha(y, np.maximum(mu, 1e-4))["alpha"]
        return self

    def _mean(self, rows):
        st = BASE_STAT[self.o]; g = self._pos(rows)
        stat = np.array([r[st] for r in rows], float); apps = np.array([r["P_N_APPS_L20"] for r in rows], float)
        rate = (stat + SHRINK_APPS * np.array([self.rate.get(k, self.all_rate) for k in g])) / (apps + SHRINK_APPS)
        a5 = np.array([r["P_APPS_L5"] for r in rows], float)
        part = (a5 + 2.0 * np.array([self.pi.get(k, self.all_pi) for k in g])) / 7.0
        return rate * part

    def crps_rows(self, rows, y):
        mu = np.maximum(self._mean(rows), 1e-4)
        pm, _ = C.nb2_pmf_matrix(mu, self.alpha, KMAX[self.o])
        y = np.minimum(np.asarray(y).astype(int), KMAX[self.o])
        return C.crps_rows(pm, y), C.pmf_mean(pm), pm


def _ks(u):
    return C.ks_uniform(np.asarray(u))


def evaluate_integrated(engine, train_rows, val_rows, val_team, M, seed0, model_id="core", int_valid=None, log=print):
    """Simulate every validation team-game and score everything end to end. int_valid: predicate on a player row for the interception outcome (source-defective seasons excluded)."""
    rows_by_key = defaultdict(list)
    for r in val_rows:
        rows_by_key[(r["game_id"], r["team"])].append(r)
    team_by_key = {(t["game_id"], t["team"]): t for t in val_team}
    keys = sorted(team_by_key)
    sim = simulate_season(engine, rows_by_key, team_by_key, keys, M, seed0)
    res = {"n_team_games": sim["n_games"], "identity_failures": dict(sim["identity_fail"]), "outcomes": {}, "teams": {}}
    for nm, items in sim["players"].items():
        if nm == "int" and int_valid is not None:
            items = [(r, d) for r, d in items if int_valid(r)]
        if not items:
            res["outcomes"][nm] = {"n": 0, "note": "no valid rows (source-defective seasons excluded)"}; continue
        rows = [r for r, _ in items]; y = np.array([r[OUTCOMES[nm][0]] for r in rows], float)
        v = C.pit_v(f"{model_id}_{nm}", [f'{r["game_id"]}|{r["team"]}|{r["player_id"]}' for r in rows])
        sm = [summarize_draws(d, yy, vv) for (r, d), yy, vv in zip(items, y, v)]
        crps = np.array([s["crps"] for s in sm]); mean = np.array([s["mean"] for s in sm])
        base = NaiveBaseline(nm).fit(train_rows); bcr, bmean, _ = base.crps_rows(rows, y)
        cov80 = np.mean([(yy >= s["q10"]) & (yy <= s["q90"]) for yy, s in zip(y, sm)]); cov90 = np.mean([(yy >= s["q05"]) & (yy <= s["q95"]) for yy, s in zip(y, sm)])
        pos = np.array([K.pos_group(r) for r in rows])
        slices = {n: {"rows": int((pos == g).sum()), "engine": float(crps[pos == g].mean()), "baseline": float(bcr[pos == g].mean()), "rel_worse": float(crps[pos == g].mean() / bcr[pos == g].mean() - 1)} for n, g in (("QB", 0), ("RB", 1), ("WR", 2), ("TE", 3)) if (pos == g).sum() >= 200}
        res["outcomes"][nm] = {"n": int(len(y)), "engine_crps": float(crps.mean()), "baseline_crps": float(bcr.mean()), "rel_gain_vs_baseline": float(1 - crps.mean() / bcr.mean()), "engine_rel_mean_bias": float((mean.mean() - y.mean()) / max(y.mean(), 1e-9)),
                               "baseline_rel_mean_bias": float((bmean.mean() - y.mean()) / max(y.mean(), 1e-9)), "coverage_80": float(cov80), "coverage_90": float(cov90), "pit_ks": _ks([s["pit"] for s in sm]), "mean_obs": float(y.mean()), "slices": slices}
    # ---- team outcomes
    teams = sim["teams"]
    trow = [team_by_key[t["key"]] for t in teams]
    tab = TE.table_arrays(trow, TE.TEAM_FEATURES_V2)
    train_team = engine._train_team
    for nm, ykey, dkey, bcol, K_ in (("team_plays", "y_plays", "plays", "T_PLAYS_B0", 150), ("team_rush_attempts", "y_rush", "rush", "T_RUSH_B0", 150), ("team_pass_attempts", "y_pass", "att", "T_PASS_B0", 150), ("team_points", "y_points", "points", "T_PTS_B0", 130)):
        y = np.array([t[ykey] for t in teams], float)
        cr = np.array([crps_draws(t["draws"][dkey], yy) for t, yy in zip(teams, y)])
        mean = np.array([t["draws"][dkey].mean() for t in teams])
        b0 = TE.B0Count(bcol).fit(TE.table_arrays(train_team, TE.TEAM_FEATURES_V2), np.array([r[ykey] for r in train_team], float))
        pm, _ = C.nb2_pmf_matrix(b0.mean(tab), b0.alpha(), K_)
        bcr = C.crps_rows(pm, np.minimum(y.astype(int), K_))
        out = {"n": len(y), "engine_crps": float(cr.mean()), "baseline_B0_crps": float(bcr.mean()), "rel_gain_vs_B0": float(1 - cr.mean() / bcr.mean()), "engine_rel_mean_bias": float((mean.mean() - y.mean()) / y.mean())}
        if nm == "team_points":
            c1 = TE.C1Count(TE.TEAM_FEATURES_V2).fit(TE.table_arrays(train_team, TE.TEAM_FEATURES_V2), np.array([r["y_points"] for r in train_team], float))
            pm1, _ = C.nb2_pmf_matrix(c1.mean(tab), c1.alpha(), K_)
            out["direct_C1_crps"] = float(C.crps_rows(pm1, np.minimum(y.astype(int), K_)).mean()); out["rel_gain_vs_direct_C1"] = float(1 - cr.mean() / out["direct_C1_crps"])
            out["direct_C1_rel_mean_bias"] = float((C.pmf_mean(pm1).mean() - y.mean()) / y.mean())
            vv = C.pit_v(f"{model_id}_points", [f"{t['key'][0]}|{t['key'][1]}" for t in teams]); pits = [np.mean(t["draws"]["points"] < yy) + vv[i] * np.mean(t["draws"]["points"] == yy) for i, (t, yy) in enumerate(zip(teams, y))]; out["pit_ks"] = _ks(pits)
        res["teams"][nm] = out
    # ---- win probability from the two simulated offences of each game
    byg = defaultdict(dict)
    for t in teams:
        gid = t["key"][0]; byg[gid][t["key"]] = t
    p_win, y_win, p_elo = [], [], []
    for gid, d in byg.items():
        if len(d) != 2:
            continue
        a, b = list(d.values())
        h, w = (a, b) if team_by_key[a["key"]]["is_home_row"] == 1 else (b, a)
        if h is w:
            continue
        pts_h, pts_a = h["draws"]["points"], w["draws"]["points"]
        pw = float(np.mean(pts_h > pts_a) + 0.5 * np.mean(pts_h == pts_a))
        hr = team_by_key[h["key"]]
        p_win.append(pw); y_win.append(float(h["y_points"] > w["y_points"])); p_elo.append(1.0 / (1.0 + 10 ** (-hr["RATING_GAP"] / 400.0)))
    if p_win:
        pw, yw, pe = np.clip(p_win, 1e-4, 1 - 1e-4), np.array(y_win), np.clip(p_elo, 1e-4, 1 - 1e-4)
        ll = lambda p: float(-(yw * np.log(p) + (1 - yw) * np.log(1 - p)).mean())
        order = np.argsort(pw); ece = float(sum(len(b) / len(pw) * abs(pw[b].mean() - yw[b].mean()) for b in np.array_split(order, 10)))
        res["teams"]["win_probability"] = {"n_games": len(yw), "engine_logloss": ll(pw), "internal_elo_baseline_logloss": ll(pe), "rel_gain_vs_elo": float(1 - ll(pw) / ll(pe)), "engine_brier": float(np.mean((pw - yw) ** 2)), "elo_brier": float(np.mean((pe - yw) ** 2)), "ece": ece, "mean_pred": float(pw.mean()), "base_rate": float(yw.mean())}
    return res
