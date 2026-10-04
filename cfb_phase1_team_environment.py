"""
CFB_PHASE1_TEAM_ENVIRONMENT -- as-of team / opponent / strength / schedule state and the team-volume components (T1 rush attempts, T2 pass attempts) of the CFB Outcome Engine v1 (RESEARCH / SHADOW ONLY).
Every feature for a target (season, week) uses team-games with (season, week) STRICTLY earlier (whole target week excluded). Internal rating only: no sportsbook spread / total, no provider Elo.
Candidates compared per protocol (cfb_models/cfb_outcome_engine/phase1_protocol.json): B0 = NB2 with mean = the shrunk blend, C1 = NB2 Poisson-GLM on the shared team state.
"""
import math
from collections import defaultdict
from datetime import date

import numpy as np
from scipy import optimize, special
from sklearn.linear_model import PoissonRegressor

import cfb_phase1_common as C

BLEND_K = 3.0
INIT_FBS, INIT_FCS = 1500.0, 1250.0
ELO_K, ELO_HFA, ELO_REGRESS = 20.0, 65.0, 0.33
TEAM_FEATURES = ["T_N_SEASON", "T_RUSH_B0", "T_PASS_B0", "T_PLAYS_B0", "T_PASS_RATE_B0", "T_PTS_B0", "T_RZ_B0", "T_RUSH_L3", "T_PASS_L3", "O_RUSH_ALLOWED_B0", "O_PASS_ALLOWED_B0", "O_PTS_ALLOWED_B0", "O_PLAYS_B0",
                 "T_RATING", "O_RATING", "RATING_GAP", "ABS_RATING_GAP", "T_RATING_N", "O_RATING_N", "O_IS_FCS", "IS_HOME", "IS_NEUTRAL", "REST_DAYS", "SEASON_WEEK"]
STATS = ("rush", "pass", "plays", "pts", "rz")
NB_BOUNDS = (1e-4, 5.0)
POISSON_ALPHA = 0.01


def _d(s):
    y, m, dd = (int(x) for x in s.split("-"))
    return date(y, m, dd)


class TeamState:
    """Chronological accumulator. `features(row)` is valid for rows of the NEXT unprocessed week; `update(rows)` folds a completed week in."""

    def __init__(self):
        self.off = defaultdict(lambda: defaultdict(lambda: {"n": 0, **{k: 0.0 for k in STATS}}))      # team -> season -> sums of the team's own offence
        self.dfn = defaultdict(lambda: defaultdict(lambda: {"n": 0, **{k: 0.0 for k in STATS}}))      # team -> season -> sums allowed (opponent offence)
        self.recent = defaultdict(list)                                                                 # team -> [(rush, pass)] chronological
        self.last_date = {}
        self.rating = {}
        self.rating_n = defaultdict(int)
        self.league = defaultdict(lambda: {"n": 0, **{k: 0.0 for k in STATS}})                          # season -> FBS offence league sums
        self.season_seen = set()

    # ---- helpers
    def _prior_mean(self, tbl, team, season, stat):
        p = tbl[team].get(season - 1)
        if p and p["n"] > 0:
            return p[stat] / p["n"]
        lg = self.league.get(season - 1)
        if lg and lg["n"] > 0:
            return lg[stat] / lg["n"]
        lg = self.league.get(season)
        if lg and lg["n"] > 0:
            return lg[stat] / lg["n"]
        return math.nan

    def blend(self, tbl, team, season, stat):
        cur = tbl[team].get(season)
        n = cur["n"] if cur else 0
        s = cur[stat] if cur else 0.0
        pm = self._prior_mean(tbl, team, season, stat)
        if n == 0 and math.isnan(pm):
            return math.nan
        if math.isnan(pm):
            return s / n
        return (s + BLEND_K * pm) / (n + BLEND_K)

    def _rate(self, team, div):
        if team not in self.rating:
            self.rating[team] = INIT_FCS if div == "fcs" else INIT_FBS
        return self.rating[team]

    def start_season(self, season):
        if season in self.season_seen:
            return
        self.season_seen.add(season)
        for t in self.rating:
            self.rating[t] = (1 - ELO_REGRESS) * self.rating[t] + ELO_REGRESS * INIT_FBS

    # ---- features for a target team-game (as of before its week)
    def features(self, r):
        t, o, s = r["team"], r["opponent"], r["season"]
        self.start_season(s)
        rt, ro = self._rate(t, r["team_div"]), self._rate(o, r["opp_div"])
        gap = rt - ro + (0.0 if r["neutral"] else (ELO_HFA if r["is_home"] else -ELO_HFA))
        rec = self.recent[t][-3:]
        cur = self.off[t].get(s)
        ld = self.last_date.get(t)
        rest = min((_d(r["game_date"]) - ld).days, 21) if ld else math.nan
        f = {"T_N_SEASON": float(cur["n"]) if cur else 0.0, "T_RUSH_B0": self.blend(self.off, t, s, "rush"), "T_PASS_B0": self.blend(self.off, t, s, "pass"), "T_PLAYS_B0": self.blend(self.off, t, s, "plays"),
             "T_PTS_B0": self.blend(self.off, t, s, "pts"), "T_RZ_B0": self.blend(self.off, t, s, "rz"), "T_RUSH_L3": float(np.mean([x[0] for x in rec])) if rec else math.nan, "T_PASS_L3": float(np.mean([x[1] for x in rec])) if rec else math.nan,
             "O_RUSH_ALLOWED_B0": self.blend(self.dfn, o, s, "rush"), "O_PASS_ALLOWED_B0": self.blend(self.dfn, o, s, "pass"), "O_PTS_ALLOWED_B0": self.blend(self.dfn, o, s, "pts"), "O_PLAYS_B0": self.blend(self.off, o, s, "plays"),
             "T_RATING": rt, "O_RATING": ro, "RATING_GAP": gap, "ABS_RATING_GAP": abs(gap), "T_RATING_N": float(self.rating_n[t]), "O_RATING_N": float(self.rating_n[o]), "O_IS_FCS": float(r["opp_div"] == "fcs"),
             "IS_HOME": float(r["is_home"] and not r["neutral"]), "IS_NEUTRAL": float(r["neutral"]), "REST_DAYS": rest if not math.isnan(rest) else 14.0, "SEASON_WEEK": float(r["week"])}
        pr, pp = f["T_RUSH_B0"], f["T_PASS_B0"]
        f["T_PASS_RATE_B0"] = pp / (pp + pr) if not (math.isnan(pr) or math.isnan(pp)) and pp + pr > 0 else math.nan
        return f

    # ---- fold a completed week in
    def update(self, rows):
        by_game = defaultdict(list)
        for r in rows:
            by_game[r["game_id"]].append(r)
        for gid, g in by_game.items():
            for r in g:
                if C.coverage_valid(r):
                    vals = {"rush": r["rush_plays"], "pass": r["pass_att_plays"], "plays": r["plays"], "pts": r["team_points"], "rz": (r["rz_rush_plays"] or 0) + (r["rz_pass_plays"] or 0)}
                else:
                    vals = None
                s = r["season"]
                if vals:
                    a = self.off[r["team"]][s]; a["n"] += 1
                    d = self.dfn[r["opponent"]][s]; d["n"] += 1
                    for k in STATS:
                        a[k] += vals[k]; d[k] += vals[k]
                    self.recent[r["team"]].append((r["rush_plays"], r["pass_att_plays"]))
                    if r["team_div"] == "fbs":
                        lg = self.league[s]; lg["n"] += 1
                        for k in STATS:
                            lg[k] += vals[k]
                self.last_date[r["team"]] = _d(r["game_date"])
            if len(g) == 2:                                                              # Elo update from the final score
                h = next((x for x in g if x["is_home"]), g[0]); a_ = next(x for x in g if x is not h)
                rh, ra = self._rate(h["team"], h["team_div"]), self._rate(a_["team"], a_["team_div"])
                hfa = 0.0 if h["neutral"] else ELO_HFA
                e_h = 1.0 / (1.0 + 10 ** (-(rh - ra + hfa) / 400.0))
                m = h["team_points"] - a_["team_points"]
                s_h = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
                win_gap = (rh - ra + hfa) if m > 0 else (ra - rh - hfa)
                mult = math.log(abs(m) + 1.0) * (2.2 / (win_gap * 0.001 + 2.2)) if m != 0 else 1.0
                delta = ELO_K * mult * (s_h - e_h)
                self.rating[h["team"]] = rh + delta; self.rating[a_["team"]] = ra - delta
                self.rating_n[h["team"]] += 1; self.rating_n[a_["team"]] += 1


def build_team_table(team_games, target_seasons=C.TARGET_SEASONS):
    """One row per FBS-offence team-game with play rows in the target seasons: as-of features + labels (labels attached from the same row AFTER its features are computed; the row's own week is not in the state)."""
    weeks = defaultdict(list)
    for r in team_games:
        weeks[(r["season"], r["week"])].append(r)
    st = TeamState(); out = []
    for sw in sorted(weeks):
        batch = sorted(weeks[sw], key=lambda r: (r["game_id"], -r["is_home"]))
        for r in batch:
            if r["season"] in target_seasons and r["team_div"] == "fbs" and C.coverage_valid(r):
                f = st.features(r)
                out.append({"season": r["season"], "week": r["week"], "game_id": r["game_id"], "team": r["team"], "opponent": r["opponent"], "key": f'{r["game_id"]}|{r["team"]}',
                            "y_rush": r["rush_plays"], "y_pass": r["pass_att_plays"], "y_plays": r["plays"], **f})
        st.update(batch)
    return out


def table_arrays(rows, names=TEAM_FEATURES):
    return {n: np.array([r[n] for r in rows], dtype=float) for n in names}


# ------------------------------------------------------------------ NB2 components
def nb2_loglik(alpha, y, mu):
    r = 1.0 / alpha
    return float(np.sum(special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))))


def fit_alpha(y, mu):
    y = np.asarray(y, float); mu = np.maximum(np.asarray(mu, float), 1e-9)
    lo, hi = math.log(NB_BOUNDS[0]), math.log(NB_BOUNDS[1])
    res = optimize.minimize_scalar(lambda t: -nb2_loglik(math.exp(t), y, mu), bounds=(lo, hi), method="bounded", options={"xatol": 1e-9})
    return {"alpha": math.exp(res.x), "at_lower_bound": bool(res.x - lo < 1e-3), "at_upper_bound": bool(hi - res.x < 1e-3)}


class Prep:
    """Training-only median imputation + missing indicators + standardisation."""

    def __init__(self, names):
        self.names = list(names)

    def fit(self, tab):
        self.med, self.mean, self.sd = {}, {}, {}
        for n in self.names:
            x = tab[n]; fin = np.isfinite(x)
            med = float(np.median(x[fin])) if fin.any() else 0.0
            xi = np.where(fin, x, med)
            self.med[n], self.mean[n] = med, float(xi.mean()); sd = float(xi.std()); self.sd[n] = sd if sd > 0 else 1.0
        return self

    def transform(self, tab):
        cols = [(np.where(np.isfinite(tab[n]), tab[n], self.med[n]) - self.mean[n]) / self.sd[n] for n in self.names]
        cols += [(~np.isfinite(tab[n])).astype(float) for n in self.names]
        return np.column_stack(cols)


class B0Count:
    def __init__(self, mean_name):
        self.mean_name = mean_name

    def fit(self, tab, y):
        mu = np.where(np.isfinite(tab[self.mean_name]), tab[self.mean_name], np.nanmean(tab[self.mean_name]))
        self.fill = float(np.nanmean(tab[self.mean_name]))
        self.nb = fit_alpha(y, mu)
        return self

    def mean(self, tab):
        return np.where(np.isfinite(tab[self.mean_name]), tab[self.mean_name], self.fill)

    def alpha(self):
        return self.nb["alpha"]


class C1Count:
    def __init__(self, names=TEAM_FEATURES):
        self.names = list(names)

    def fit(self, tab, y):
        self.prep = Prep(self.names).fit(tab)
        X = self.prep.transform(tab)
        self.m = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=2000, tol=1e-8).fit(X, y)
        self.nb = fit_alpha(y, self.m.predict(X))
        self.converged = bool(self.m.n_iter_ < 2000)
        return self

    def mean(self, tab):
        return self.m.predict(self.prep.transform(tab))

    def alpha(self):
        return self.nb["alpha"]


# ------------------------------------------------------------------ shared state v2 (Phase 1B amendment 1): efficiency / event / scoring team state
V2_STATS = ("comp", "ryds", "cyds", "ints", "sacks", "rzp")
TEAM_FEATURES_V2 = TEAM_FEATURES + ["T_COMP_PCT_B0", "T_YPC_B0", "T_YPCOMP_B0", "T_INT_RATE_B0", "T_SACK_RATE_B0", "T_TD_B0", "T_RZ_RATE_B0", "O_COMP_PCT_ALLOWED_B0", "O_YPC_ALLOWED_B0", "O_YPCOMP_ALLOWED_B0", "O_INT_FORCED_B0", "O_SACK_FORCED_B0", "O_TD_ALLOWED_B0",
                                    "O_SACK_RATE_B0_DROPBACKS"]


class TeamStateV2(TeamState):
    """TeamState + completion / yardage / interception / sack / touchdown accumulators (all as-of, same B0 blend). tdmap: (game_id, team) -> offensive TDs (rush + receiving) where player lines exist."""

    def __init__(self, tdmap=None):
        super().__init__()
        self.tdmap = tdmap or {}
        self.x_off = defaultdict(lambda: defaultdict(lambda: {"n": 0, "ntd": 0, "td": 0.0, **{k: 0.0 for k in V2_STATS}, "rush": 0.0, "pass": 0.0}))
        self.x_dfn = defaultdict(lambda: defaultdict(lambda: {"n": 0, "ntd": 0, "td": 0.0, **{k: 0.0 for k in V2_STATS}, "rush": 0.0, "pass": 0.0}))

    def _xblend(self, tbl, team, season, stat, nkey="n"):
        cur = tbl[team].get(season); prv = tbl[team].get(season - 1)
        n = cur[nkey] if cur else 0; s = cur[stat] if cur else 0.0
        if prv and prv[nkey] > 0:
            pm = prv[stat] / prv[nkey]
        else:
            pm = self._league_x(season, stat, nkey)
        if n == 0 and math.isnan(pm):
            return math.nan
        return (s + BLEND_K * pm) / (n + BLEND_K) if not math.isnan(pm) else s / n

    def _league_x(self, season, stat, nkey):
        lg = getattr(self, "_xl", {}).get(season - 1) or getattr(self, "_xl", {}).get(season)
        if lg and lg[nkey] > 0:
            return lg[stat] / lg[nkey]
        return math.nan

    def features(self, r):
        f = super().features(r)
        t, o, s = r["team"], r["opponent"], r["season"]
        b = lambda tbl, team, k, nk="n": self._xblend(tbl, team, s, k, nk)
        def ratio(a, d):
            return a / d if not (math.isnan(a) or math.isnan(d)) and d > 0 else math.nan
        t_pass, t_comp, t_rush = b(self.x_off, t, "pass"), b(self.x_off, t, "comp"), b(self.x_off, t, "rush")
        t_sacks = b(self.x_off, t, "sacks")
        f.update({"T_COMP_PCT_B0": ratio(t_comp, t_pass), "T_YPC_B0": ratio(b(self.x_off, t, "ryds"), t_rush), "T_YPCOMP_B0": ratio(b(self.x_off, t, "cyds"), t_comp), "T_INT_RATE_B0": ratio(b(self.x_off, t, "ints"), t_pass),
                  "T_SACK_RATE_B0": ratio(t_sacks, (t_pass + t_sacks) if not (math.isnan(t_pass) or math.isnan(t_sacks)) else math.nan), "T_TD_B0": b(self.x_off, t, "td", "ntd"), "T_RZ_RATE_B0": ratio(b(self.x_off, t, "rzp"), b(self.off, t, "plays") if False else self.blend(self.off, t, s, "plays")),
                  "O_COMP_PCT_ALLOWED_B0": ratio(b(self.x_dfn, o, "comp"), b(self.x_dfn, o, "pass")), "O_YPC_ALLOWED_B0": ratio(b(self.x_dfn, o, "ryds"), b(self.x_dfn, o, "rush")), "O_YPCOMP_ALLOWED_B0": ratio(b(self.x_dfn, o, "cyds"), b(self.x_dfn, o, "comp")),
                  "O_INT_FORCED_B0": ratio(b(self.x_dfn, o, "ints"), b(self.x_dfn, o, "pass")), "O_SACK_FORCED_B0": b(self.x_dfn, o, "sacks"), "O_TD_ALLOWED_B0": b(self.x_dfn, o, "td", "ntd"),
                  "O_SACK_RATE_B0_DROPBACKS": ratio(b(self.x_dfn, o, "sacks"), (b(self.x_dfn, o, "pass") + b(self.x_dfn, o, "sacks")) if not math.isnan(b(self.x_dfn, o, "pass")) else math.nan)})
        return f

    def update(self, rows):
        super().update(rows)
        if not hasattr(self, "_xl"):
            self._xl = defaultdict(lambda: {"n": 0, "ntd": 0, "td": 0.0, **{k: 0.0 for k in V2_STATS}, "rush": 0.0, "pass": 0.0})
        for r in rows:
            if not C.coverage_valid(r):
                continue
            s = r["season"]
            vals = {"comp": r["completions"], "ryds": r["rush_yards"], "cyds": r["pass_comp_yards"], "ints": r["interceptions"], "sacks": r["sack_plays"], "rzp": (r["rz_rush_plays"] or 0) + (r["rz_pass_plays"] or 0), "rush": r["rush_plays"], "pass": r["pass_att_plays"]}
            td = self.tdmap.get((r["game_id"], r["team"]))
            for tbl, team in ((self.x_off, r["team"]), (self.x_dfn, r["opponent"])):
                a = tbl[team][s]; a["n"] += 1
                for k, v in vals.items():
                    a[k] += v
                if td is not None:
                    a["ntd"] += 1; a["td"] += td
            if r["team_div"] == "fbs":
                lg = self._xl[s]; lg["n"] += 1
                for k, v in vals.items():
                    lg[k] += v
                if td is not None:
                    lg["ntd"] += 1; lg["td"] += td


def build_team_table_v2(team_games, player_games, target_seasons=C.TARGET_SEASONS, extra_labels=True):
    """Team-game rows with TEAM_FEATURES_V2 (as-of) + labels for the V2 volume / scoring components."""
    tdmap = defaultdict(int); seen = set()
    for r in player_games:
        k = (r["game_id"], r["team"]); seen.add(k)
        tdmap[k] += (r["rushing_touchdowns"] or 0) + (r["receiving_touchdowns"] or 0)
    tdmap = {k: v for k, v in tdmap.items() if k in seen}
    weeks = defaultdict(list)
    for r in team_games:
        weeks[(r["season"], r["week"])].append(r)
    st = TeamStateV2(tdmap); out = []
    for sw in sorted(weeks):
        batch = sorted(weeks[sw], key=lambda r: (r["game_id"], -r["is_home"]))
        for r in batch:
            if r["season"] in target_seasons and r["team_div"] == "fbs" and C.coverage_valid(r):
                f = st.features(r)
                td = tdmap.get((r["game_id"], r["team"]))
                out.append({"season": r["season"], "week": r["week"], "game_id": r["game_id"], "team": r["team"], "opponent": r["opponent"], "key": f'{r["game_id"]}|{r["team"]}', "y_rush": r["rush_plays"], "y_pass": r["pass_att_plays"], "y_sacks": r["sack_plays"], "y_plays": r["plays"],
                            "y_comp": r["completions"], "y_points": r["team_points"], "y_opp_points": r["opp_points"], "y_td": td if td is not None else math.nan, "y_ints": r["interceptions"], "y_rush_yards": r["rush_yards"], "y_comp_yards": r["pass_comp_yards"], "is_home_row": r["is_home"],
                            "opp_div": r["opp_div"], **f})
        st.update(batch)
    return out
