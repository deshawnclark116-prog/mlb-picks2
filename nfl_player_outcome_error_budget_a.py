"""
NFL_PLAYER_OUTCOME_ERROR_BUDGET_A

Research-only diagnostic. Makes no production predictions, writes no
model artifacts, changes no serving/API behaviour.

Question: WHERE DOES OUR PLAYER-OUTCOME FORECAST ERROR COME FROM?

For each stat family it builds an as-of decomposition of the outcome into
the mechanism that generates it, forecasts each component from pre-game
information only, and then walks an ORACLE LADDER: replace one predicted
component at a time with its ACTUAL value and measure how much error
disappears.

  rushing yards    team carries x carry share x yards/carry (+ explosive tail)
  receiving yards  team targets x target share (routes x targets/route)
                   x catch -> receptions x (air yards + YAC)
  receptions       team targets x target share x catch rate
  passing yards    team plays x dropback rate x QB attempt share
                   x completion rate x yards/completion (air + YAC)
  anytime TD       team scoring / red-zone opportunity x player red-zone
                   opportunity x conversion (binary; logloss/Brier/AUC)

Two oracle flavours are reported at each rung:
  plug-in   the decomposed product with the actual component substituted
  learned   a model re-fit with the actual component as an input (what a
            model could do if that component were known pre-game)

Incumbent = the live NFL v2 architecture (nfl_yardage_v2.FEATURES,
reg:absoluteerror XGBoost), RE-FIT here on 2023 + 2024 wk<=12 (early
stopping on 2024 wk>12) exactly like the research ablations. The shipped
v2 artifact was trained on 2023-2025, so 2025 is in-sample for it; that
in-sample score is reported separately as a leakage measurement.

Evaluation population: players who actually played (a CONDITIONAL
production forecast -- "assuming he is active"). The availability layer
(P(active | injury-report status)) is measured separately.

Temporal protocol (all features as-of, strictly before the game):
  train  2023 + 2024 weeks 1-12
  valid  2024 weeks 13-18 (early stopping, interval calibration)
  eval   2025 (previously examined many times -- NOT pristine) and
         2026 weeks 1-3 (also examined). See README for the clean forward
         window recommendation.

Run
---
  python -u nfl_player_outcome_error_budget_a.py --data-dir /tmp/nfl_data
Needs nflverse files in --data-dir (downloaded if missing): games.csv,
stats_player_week_{y}.csv, snap_counts_{y}.csv, pbp_{y}.csv.gz,
participation_{y}.csv (2022-2025 only), injuries_{y}.csv, players.csv.
"""
import argparse
import csv
import gzip
import json
import math
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

import nfl_yardage_v2 as v2
import nfl_yardage_v3 as v3

REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / "nfl_models" / "nfl_player_outcome_error_budget_a"
SEASONS = [2022, 2023, 2024, 2025, 2026]
REL = v3.RELEASE
MARKET_FEATURES = ("team_spread", "implied_team_total")   # sportsbook-derived
EXPLOSIVE_RUN = 20       # yards; tail-process threshold
RZ = 20                  # yardline_100 <= 20
GL = 5                   # yardline_100 <= 5
XGB = {"max_depth": 5, "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.8,
       "min_child_weight": 15, "reg_lambda": 5.0, "seed": 20260929}


# ------------------------------------------------------------------ utils
def f(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def wsum(xs):
    return sum(x for x in xs if x is not None)


def download(data_dir):
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    want = [("games.csv", f"{REL}/schedules/games.csv"), ("players.csv", f"{REL}/players/players.csv")]
    for y in SEASONS:
        want += [(f"stats_player_week_{y}.csv", f"{REL}/stats_player/stats_player_week_{y}.csv"),
                 (f"snap_counts_{y}.csv", f"{REL}/snap_counts/snap_counts_{y}.csv"),
                 (f"pbp_{y}.csv.gz", f"{REL}/pbp/play_by_play_{y}.csv.gz"),
                 (f"injuries_{y}.csv", f"{REL}/injuries/injuries_{y}.csv")]
        if y <= 2025:
            want.append((f"participation_{y}.csv", f"{REL}/pbp_participation/pbp_participation_{y}.csv"))
    for name, url in want:
        p = data_dir / name
        if p.exists():
            continue
        print(f"  downloading {name}")
        req = urllib.request.Request(url, headers={"User-Agent": "nfl-error-budget"})
        with urllib.request.urlopen(req, timeout=300) as r:
            p.write_bytes(r.read())


# ------------------------------------------------------------------ data
class Football:
    """Game-level player stats + play-by-play derived components."""

    def __init__(self, data_dir):
        d = Path(data_dir)
        self.games = {}                # (s, w, team) -> ctx
        self.team_games = defaultdict(list)
        self.coach = {}                # (s, team) -> head coach (first game)
        for r in csv.DictReader(open(d / "games.csv", newline="", encoding="utf-8")):
            if r["game_type"] != "REG" or int(r["season"]) not in SEASONS:
                continue
            s, w = int(r["season"]), int(r["week"])
            sp, tot, res = f(r["spread_line"]), f(r["total_line"]), f(r["result"])
            for team, opp, home in ((r["home_team"], r["away_team"], True), (r["away_team"], r["home_team"], False)):
                self.games[(s, w, team)] = {
                    "opp": opp, "is_home": home, "spread": (sp if home else -sp) if sp is not None else None,
                    "total": tot, "margin": (res if home else -res) if res is not None else None,
                    "game_id": r["game_id"]}
                self.team_games[team].append((s, w))
                c = r["home_coach"] if home else r["away_coach"]
                self.coach.setdefault((s, team), c)
        for t in self.team_games:
            self.team_games[t].sort()

        self.pl = defaultdict(dict)    # (s, w, team) -> pid -> row
        for s in SEASONS:
            p = d / f"stats_player_week_{s}.csv"
            for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
                if r.get("season_type", "REG") != "REG":
                    continue
                g = lambda k: f(r.get(k)) or 0.0
                self.pl[(s, int(r["week"]), r["team"])][r["player_id"]] = {
                    "name": r["player_display_name"], "pos": r["position"],
                    "car": g("carries"), "rush_yds": g("rushing_yards"), "rtd": g("rushing_tds"),
                    "tgt": g("targets"), "rec": g("receptions"), "rec_yds": g("receiving_yards"),
                    "rectd": g("receiving_tds"), "att": g("attempts"), "cmp": g("completions"),
                    "pass_yds": g("passing_yards"), "ptd": g("passing_tds"), "int": g("passing_interceptions"),
                    "sk": g("sacks_suffered"),
                }
        self.snap = defaultdict(dict)  # (s, w, team) -> normname -> offense_pct
        for s in SEASONS:
            for r in csv.DictReader(open(d / f"snap_counts_{s}.csv", newline="", encoding="utf-8")):
                if r.get("game_type", "REG") != "REG":
                    continue
                if (f(r["offense_snaps"]) or 0) > 0:
                    self.snap[(s, int(r["week"]), r["team"])][v3.norm_name(r["player"])] = f(r["offense_pct"])

        # play-by-play components
        self.tg = defaultdict(lambda: defaultdict(float))    # (s, w, team) team offense
        self.pp = defaultdict(lambda: defaultdict(float))    # (s, w, pid) player
        dropbacks = set()                                    # (game_id, play_id) of dropbacks
        dropback_team = {}
        for s in SEASONS:
            with gzip.open(d / f"pbp_{s}.csv.gz", "rt", newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r["season_type"] != "REG" or not r["posteam"]:
                        continue
                    w = int(r["week"]); t = r["posteam"]; k = (s, w, t)
                    if r.get("two_point_attempt") == "1":
                        continue
                    yl = f(r["yardline_100"])
                    db = r["qb_dropback"] == "1" and r.get("qb_spike") != "1"
                    ru = r["rush_attempt"] == "1" and r.get("qb_kneel") != "1"
                    if db or ru:
                        self.tg[k]["plays"] += 1
                        if db:
                            self.tg[k]["dropbacks"] += 1
                            dropbacks.add((r["game_id"], r["play_id"])); dropback_team[(r["game_id"], r["play_id"])] = (s, w, t)
                        if ru:
                            self.tg[k]["runs"] += 1
                        if yl is not None and yl <= RZ:
                            self.tg[k]["rz_plays"] += 1
                    if r["touchdown"] == "1" and r.get("td_team") == t and (r["pass_touchdown"] == "1" or r["rush_touchdown"] == "1"):
                        self.tg[k]["off_td"] += 1
                    if r["pass_attempt"] == "1" and r["sack"] != "1":
                        self.tg[k]["pass_att"] += 1
                    rid, qid = r["rusher_player_id"], r["receiver_player_id"]
                    if ru and rid:
                        ry = f(r["rushing_yards"]) or 0.0
                        pk = (s, w, rid)
                        if ry >= EXPLOSIVE_RUN:
                            self.pp[pk]["expl_car"] += 1; self.pp[pk]["expl_yds"] += ry
                        else:
                            self.pp[pk]["base_yds"] += ry
                        if yl is not None and yl <= RZ:
                            self.pp[pk]["rz_car"] += 1
                        if yl is not None and yl <= GL:
                            self.pp[pk]["gl_car"] += 1
                    if r["pass_attempt"] == "1" and r["sack"] != "1" and qid:
                        pk = (s, w, qid)
                        ay = f(r["air_yards"])
                        if yl is not None and yl <= RZ:
                            self.pp[pk]["rz_tgt"] += 1
                        if ay is not None and yl is not None and ay >= yl:
                            self.pp[pk]["ez_tgt"] += 1
                        if ay is not None:
                            self.pp[pk]["tgt_air"] += ay
                        if r["complete_pass"] == "1":
                            self.pp[pk]["cmp_air"] += ay or 0.0
                            self.pp[pk]["yac"] += f(r["yards_after_catch"]) or 0.0
                    pid = r["passer_player_id"]
                    if r["pass_attempt"] == "1" and r["sack"] != "1" and pid and r["complete_pass"] == "1":
                        self.pp[(s, w, pid)]["qb_cmp_air"] += f(r["air_yards"]) or 0.0
        # route proxy: on field for a dropback (participation, 2022-2025)
        self.has_routes = set()
        for s in SEASONS:
            p = d / f"participation_{s}.csv"
            if not p.exists():
                continue
            self.has_routes.add(s)
            for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
                key = (r["nflverse_game_id"], r["play_id"])
                if key not in dropbacks:
                    continue
                ss, w, t = dropback_team[key]
                for pid in (r["offense_players"] or "").split(";"):
                    if pid:
                        self.pp[(ss, w, pid)]["routes"] += 1
        # players (draft / rookie)
        self.draft = {}
        for r in csv.DictReader(open(d / "players.csv", newline="", encoding="utf-8")):
            self.draft[r["gsis_id"]] = {"rookie": f(r.get("rookie_season")), "round": f(r.get("draft_round"))}
        # injury reports (final pre-game status)
        self.inj = {}
        for s in SEASONS:
            for r in csv.DictReader(open(d / f"injuries_{s}.csv", newline="", encoding="utf-8")):
                if r.get("game_type", "REG") == "REG":
                    self.inj[(s, int(r["week"]), r["gsis_id"])] = (r["report_status"] or "", r["practice_status"] or "", r["team"])

    def played(self, key):
        names = set(self.snap.get(key, {}))
        for r in self.pl.get(key, {}).values():
            if r["car"] + r["tgt"] + r["att"] > 0:
                names.add(v3.norm_name(r["name"]))
        return names


# ------------------------------------------------------------------ as-of replay
FAMILIES = {
    "rushing_yards": {"pos": ("RB", "FB"), "vol": "car", "min": 5.0},
    "receiving_yards": {"pos": ("WR", "TE", "RB"), "vol": "tgt", "min": 3.0},
    "receptions": {"pos": ("WR", "TE", "RB"), "vol": "tgt", "min": 3.0},
    "passing_yards": {"pos": ("QB",), "vol": "att", "min": 15.0},
    "anytime_td": {"pos": ("RB", "WR", "TE"), "vol": "touch", "min": 4.0},
}


def build_rows(fb):
    """One pass over all weeks. For each eligible player-game emits:
    as-of features, actual components, actual outcome."""
    ph = defaultdict(list)       # pid -> game dicts
    th = defaultdict(list)       # team -> team-game dicts (offense)
    dh = defaultdict(list)       # team -> what offenses did vs this defense
    rows = {fam: [] for fam in FAMILIES}
    weeks = sorted({(k[0], k[1]) for k in fb.pl})
    for (s, w) in weeks:
        for team in list(fb.team_games):
            key = (s, w, team)
            pl = fb.pl.get(key); ctx = fb.games.get(key)
            if not pl or not ctx:
                continue
            played = fb.played(key)
            T = team_feats(th[team], dh[ctx["opp"]], s, ctx)
            prev_team_game = [g for g in fb.team_games[team] if g < (s, w)]
            qb_prev = None
            if th[team]:
                qb_prev = th[team][-1].get("qb")
            for pid, r in pl.items():
                if v3.norm_name(r["name"]) not in played:
                    continue
                h = ph[pid]
                if len(h) < 3:
                    continue
                pos = h[-1]["pos"]
                P = player_feats(h, s, fb, pid)
                A = actuals(fb, key, pid, r, team)
                seg = {"pos": pos, "team_change": 1.0 if h[-1]["team"] != team else 0.0,
                       "rookie": 1.0 if (fb.draft.get(pid, {}).get("rookie") == s) else 0.0,
                       "draft_round": fb.draft.get(pid, {}).get("round"),
                       "margin": ctx["margin"], "week": w,
                       "coach_change": 1.0 if fb.coach.get((s, team)) and fb.coach.get((s - 1, team))
                       and fb.coach[(s, team)] != fb.coach[(s - 1, team)] else 0.0,
                       "qb_change": 1.0 if qb_prev and A["team_qb"] and A["team_qb"] != qb_prev else 0.0,
                       "inj_status": fb.inj.get((s, w, pid), ("", "", ""))[0],
                       "gap_games": sum(1 for g in prev_team_game if (h[-1]["s"], h[-1]["w"]) < g)}
                for fam, c in FAMILIES.items():
                    if pos not in c["pos"]:
                        continue
                    vol = [(g["car"] + g["tgt"]) if c["vol"] == "touch" else g[c["vol"]] for g in h[-3:]]
                    if sum(vol) / 3 < c["min"]:
                        continue
                    rows[fam].append({"key": {"pid": pid, "name": r["name"], "team": team, "s": s, "w": w},
                                      "F": {**T, **P}, "A": A, "seg": seg})
        # absorb this week
        for team in list(fb.team_games):
            key = (s, w, team)
            pl = fb.pl.get(key); ctx = fb.games.get(key)
            if not pl or not ctx:
                continue
            played = fb.played(key)
            tgv = fb.tg.get(key, {})
            tcar = sum(r["car"] for r in pl.values()); ttgt = sum(r["tgt"] for r in pl.values())
            qb = max(pl, key=lambda p: pl[p]["att"]) if any(r["att"] > 0 for r in pl.values()) else None
            tr = {"s": s, "w": w, "car": tcar, "tgt": ttgt, "rush_yds": sum(r["rush_yds"] for r in pl.values()),
                  "rec_yds": sum(r["rec_yds"] for r in pl.values()), "plays": tgv.get("plays", 0.0),
                  "dropbacks": tgv.get("dropbacks", 0.0), "rz_plays": tgv.get("rz_plays", 0.0),
                  "off_td": tgv.get("off_td", 0.0), "pass_att": sum(r["att"] for r in pl.values()), "qb": qb}
            th[team].append(tr)
            dh[ctx["opp"]].append(tr)
            snaps = fb.snap.get(key, {})
            for pid, r in pl.items():
                pp = fb.pp.get((s, w, pid), {})
                ph[pid].append({
                    "s": s, "w": w, "team": team, "pos": r["pos"], **{k: r[k] for k in
                    ("car", "rush_yds", "tgt", "rec", "rec_yds", "att", "cmp", "pass_yds", "rtd", "rectd", "sk")},
                    "played": v3.norm_name(r["name"]) in played,
                    "car_sh": r["car"] / tcar if tcar else None, "tgt_sh": r["tgt"] / ttgt if ttgt else None,
                    "att_sh": r["att"] / tgv["dropbacks"] if tgv.get("dropbacks") else None,
                    "snap": snaps.get(v3.norm_name(r["name"])),
                    "routes": pp.get("routes") if s in fb.has_routes else None,
                    "team_db": tgv.get("dropbacks"),
                    "expl_car": pp.get("expl_car", 0.0), "expl_yds": pp.get("expl_yds", 0.0),
                    "yac": pp.get("yac", 0.0), "cmp_air": pp.get("cmp_air", 0.0), "tgt_air": pp.get("tgt_air", 0.0),
                    "qb_cmp_air": pp.get("qb_cmp_air", 0.0),
                    "rz_car": pp.get("rz_car", 0.0), "gl_car": pp.get("gl_car", 0.0),
                    "rz_tgt": pp.get("rz_tgt", 0.0), "ez_tgt": pp.get("ez_tgt", 0.0),
                    "td": r["rtd"] + r["rectd"],
                })
    return rows


def team_feats(th, dh, s, ctx):
    cur = [g for g in th if g["s"] == s]; pri = [g for g in th if g["s"] == s - 1]
    l3 = th[-3:]
    dcur = [g for g in dh if g["s"] == s]; dpri = [g for g in dh if g["s"] == s - 1]

    def blend(key, cur_, pri_):
        c = mean([g[key] for g in cur_]); p = mean([g[key] for g in pri_])
        if c is None:
            return p
        if p is None:
            return c
        n = len(cur_); return (c * n + p * 3) / (n + 3)
    out = {"is_home": 1.0 if ctx["is_home"] else 0.0, "team_spread": ctx["spread"],
           "implied_team_total": (ctx["total"] / 2 + ctx["spread"] / 2) if ctx["spread"] is not None and ctx["total"] is not None else None,
           "team_n_cur": float(len(cur))}
    for k in ("car", "tgt", "plays", "dropbacks", "rz_plays", "off_td", "pass_att"):
        out[f"t_{k}_blend"] = blend(k, cur, pri)
        out[f"t_{k}_l3"] = mean([g[k] for g in l3])
        out[f"o_{k}_allowed"] = blend(k, dcur, dpri)
    out["t_pass_rate_blend"] = (out["t_dropbacks_blend"] / out["t_plays_blend"]) if out["t_plays_blend"] else None
    out["o_pass_rate_allowed"] = (out["o_dropbacks_allowed"] / out["o_plays_allowed"]) if out["o_plays_allowed"] else None
    out["o_rush_ypc_allowed"] = (wsum([g["rush_yds"] for g in dcur + dpri]) / wsum([g["car"] for g in dcur + dpri])) if wsum([g["car"] for g in dcur + dpri]) else None
    out["o_rec_ypt_allowed"] = (wsum([g["rec_yds"] for g in dcur + dpri]) / wsum([g["tgt"] for g in dcur + dpri])) if wsum([g["tgt"] for g in dcur + dpri]) else None
    return out


POS_PRIOR = {"ypc": (4.3, 60.0), "ypt": (7.5, 40.0), "catch": (0.65, 30.0), "ypr": (11.0, 25.0),
             "yac_pr": (5.0, 25.0), "cmp_pct": (0.64, 150.0), "ypcmp": (10.8, 100.0), "expl_rate": (4.0, 150.0),
             "tprr": (0.18, 60.0), "air_pcmp": (5.8, 100.0)}


def shrink(num, den, prior):
    m, k = POS_PRIOR[prior]
    return (num + m * k) / (den + k)


def player_feats(h, s, fb, pid):
    l3, l8, l16 = h[-3:], h[-8:], h[-16:]
    cur = [g for g in h if g["s"] == s]; pri = [g for g in h if g["s"] == s - 1]
    pl3 = [g for g in l3 if g["played"]]
    F = {"cur_n": float(len(cur)), "prior_n": float(len(pri)),
         "returning": 1.0 if not h[-1]["played"] else 0.0,
         "rookie_year": 1.0 if fb.draft.get(pid, {}).get("rookie") == s else 0.0,
         "draft_round": fb.draft.get(pid, {}).get("round"),
         "snap_l3": mean([g["snap"] for g in l3])}
    for k in ("car", "tgt", "rec", "att", "cmp", "rush_yds", "rec_yds", "pass_yds", "td", "rz_car", "gl_car", "rz_tgt", "ez_tgt"):
        F[f"{k}_l3"] = mean([g[k] for g in l3]); F[f"{k}_cur"] = mean([g[k] for g in cur]); F[f"{k}_pri"] = mean([g[k] for g in pri])
    for k in ("car_sh", "tgt_sh", "att_sh"):
        F[f"{k}_l3"] = mean([g[k] for g in pl3]); F[f"{k}_l8"] = mean([g[k] for g in l8 if g["played"]])
        F[f"{k}_cur"] = mean([g[k] for g in cur if g["played"]])
        xs = [g[k] for g in h[-5:] if g["played"] and g[k] is not None]
        F[f"{k}_sd5"] = float(np.std(xs)) if len(xs) >= 3 else None
    F["car_sh_trend"] = (F["car_sh_l3"] - F["car_sh_l8"]) if F["car_sh_l3"] is not None and F["car_sh_l8"] is not None else None
    F["tgt_sh_trend"] = (F["tgt_sh_l3"] - F["tgt_sh_l8"]) if F["tgt_sh_l3"] is not None and F["tgt_sh_l8"] is not None else None
    # efficiency, shrunk (last 16 games)
    F["ypc"] = shrink(wsum([g["rush_yds"] for g in l16]), wsum([g["car"] for g in l16]), "ypc")
    F["ypt"] = shrink(wsum([g["rec_yds"] for g in l16]), wsum([g["tgt"] for g in l16]), "ypt")
    F["catch"] = shrink(wsum([g["rec"] for g in l16]), wsum([g["tgt"] for g in l16]), "catch")
    F["ypr"] = shrink(wsum([g["rec_yds"] for g in l16]), wsum([g["rec"] for g in l16]), "ypr")
    F["yac_pr"] = shrink(wsum([g["yac"] for g in l16]), wsum([g["rec"] for g in l16]), "yac_pr")
    F["adot"] = (wsum([g["tgt_air"] for g in l16]) / wsum([g["tgt"] for g in l16])) if wsum([g["tgt"] for g in l16]) else None
    F["cmp_pct"] = shrink(wsum([g["cmp"] for g in l16]), wsum([g["att"] for g in l16]), "cmp_pct")
    F["ypcmp"] = shrink(wsum([g["pass_yds"] for g in l16]), wsum([g["cmp"] for g in l16]), "ypcmp")
    F["air_pcmp"] = shrink(wsum([g["qb_cmp_air"] for g in l16]), wsum([g["cmp"] for g in l16]), "air_pcmp")
    # explosive-run tail: explosive yards per 100 carries
    F["expl_rate"] = shrink(100 * wsum([g["expl_yds"] for g in l16]), wsum([g["car"] for g in l16]), "expl_rate") / 100
    F["base_ypc"] = F["ypc"] - F["expl_rate"]
    rt = [g for g in l16 if g["routes"]]
    F["routes_l3"] = mean([g["routes"] for g in l3 if g["routes"] is not None])
    F["route_rate_l3"] = mean([g["routes"] / g["team_db"] for g in l3 if g["routes"] is not None and g["team_db"]])
    F["tprr"] = shrink(wsum([g["tgt"] for g in rt]), wsum([g["routes"] for g in rt]), "tprr") if rt else None
    return F


def actuals(fb, key, pid, r, team):
    s, w, _ = key
    tg = fb.tg.get(key, {}); pp = fb.pp.get((s, w, pid), {})
    pl = fb.pl[key]
    tcar = sum(x["car"] for x in pl.values()); ttgt = sum(x["tgt"] for x in pl.values())
    qb = max(pl, key=lambda p: pl[p]["att"]) if any(x["att"] > 0 for x in pl.values()) else None
    return {"rush_yds": r["rush_yds"], "car": r["car"], "team_car": tcar,
            "rec_yds": r["rec_yds"], "tgt": r["tgt"], "rec": r["rec"], "team_tgt": ttgt,
            "pass_yds": r["pass_yds"], "att": r["att"], "cmp": r["cmp"],
            "team_plays": tg.get("plays", 0.0), "team_db": tg.get("dropbacks", 0.0),
            "routes": pp.get("routes") if s in fb.has_routes else None,
            "cmp_air": pp.get("cmp_air", 0.0), "yac": pp.get("yac", 0.0), "qb_cmp_air": pp.get("qb_cmp_air", 0.0),
            "base_yds": pp.get("base_yds", 0.0), "expl_yds": pp.get("expl_yds", 0.0),
            "td": r["rtd"] + r["rectd"], "team_off_td": tg.get("off_td", 0.0), "team_rz_plays": tg.get("rz_plays", 0.0),
            "rz_car": pp.get("rz_car", 0.0), "gl_car": pp.get("gl_car", 0.0),
            "rz_tgt": pp.get("rz_tgt", 0.0), "ez_tgt": pp.get("ez_tgt", 0.0), "team_qb": qb}


# ------------------------------------------------------------------ metrics
def cont_metrics(pred, act):
    e = pred - act; a = np.abs(e)
    out = {"n": int(len(a)), "mae": round(float(a.mean()), 2), "median_ae": round(float(np.median(a)), 2),
           "rmse": round(float(np.sqrt((e ** 2).mean())), 2), "bias": round(float(e.mean()), 2)}
    for t in (5, 10, 15, 20, 25, 30):
        out[f"within_{t}"] = round(float((a <= t).mean()), 3)
    out["residual_pct"] = {f"p{q}": round(float(np.percentile(act - pred, q)), 1) for q in (5, 25, 50, 75, 95)}
    return out


def count_metrics(pred, act):
    e = pred - act; a = np.abs(e)
    return {"n": int(len(a)), "mae": round(float(a.mean()), 3), "rmse": round(float(np.sqrt((e ** 2).mean())), 3),
            "bias": round(float(e.mean()), 3), "exact_rounded": round(float((np.round(pred) == act).mean()), 3),
            "within_1": round(float((a <= 1).mean()), 3), "within_2": round(float((a <= 2).mean()), 3)}


def binary_metrics(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    order = np.argsort(p); r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    npos = y.sum(); nneg = len(y) - npos
    auc = float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else None
    bins = np.minimum((p * 10).astype(int), 9)
    ece = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()))
    return {"n": int(len(y)), "base_rate": round(float(y.mean()), 4), "logloss": round(ll, 4),
            "brier": round(float(((p - y) ** 2).mean()), 4), "auc": round(auc, 4) if auc else None, "ece": round(ece, 4)}


QGRID = np.linspace(0.05, 0.95, 19)


def dist_metrics(pred, act, val_pred, val_act):
    """Interval / CRPS from validation-period residual quantiles (additive,
    scaled by sqrt of the projection so bigger projections get wider bands)."""
    sc_v = np.sqrt(np.maximum(val_pred, 1.0)); sc = np.sqrt(np.maximum(pred, 1.0))
    z = (val_act - val_pred) / sc_v
    zq = np.quantile(z, QGRID)
    Q = np.maximum(pred[:, None] + zq[None, :] * sc[:, None], 0.0)
    pin = np.mean([np.mean(np.maximum(q * (act - Q[:, i]), (q - 1) * (act - Q[:, i]))) for i, q in enumerate(QGRID)])
    lo80, hi80 = Q[:, 1], Q[:, 17]; lo50, hi50 = Q[:, 4], Q[:, 14]
    return {"crps_approx": round(float(2 * pin), 2),
            "cov80": round(float(((act >= lo80) & (act <= hi80)).mean()), 3), "width80": round(float((hi80 - lo80).mean()), 1),
            "cov50": round(float(((act >= lo50) & (act <= hi50)).mean()), 3), "width50": round(float((hi50 - lo50).mean()), 1)}


def boot_p(e_new, e_ref, seed=1):
    d = e_ref - e_new
    rng = np.random.default_rng(seed)
    bt = d[rng.integers(0, len(d), (2000, len(d)))].mean(axis=1)
    return round(float((bt <= 0).mean()), 4)


# ------------------------------------------------------------------ modelling helpers
def M(rows, cols):
    return np.array([[r["F"].get(c) if r["F"].get(c) is not None else np.nan for c in cols] for r in rows], dtype=np.float32)


def fit(tr, va, cols, ytr, yva, obj="reg:absoluteerror"):
    import xgboost as xgb
    cols = list(dict.fromkeys(cols))
    b = xgb.train({**XGB, "objective": obj}, xgb.DMatrix(M(tr, cols), label=ytr, feature_names=cols), 3000,
                  evals=[(xgb.DMatrix(M(va, cols), label=yva, feature_names=cols), "va")],
                  early_stopping_rounds=80, verbose_eval=False)
    return b


def pred(b, rows, cols):
    import xgboost as xgb
    cols = list(dict.fromkeys(cols))
    return b.predict(xgb.DMatrix(M(rows, cols), feature_names=cols), iteration_range=(0, b.best_iteration + 1))


def split(rows):
    tr = [r for r in rows if r["key"]["s"] == 2023 or (r["key"]["s"] == 2024 and r["key"]["w"] <= 12)]
    va = [r for r in rows if r["key"]["s"] == 2024 and r["key"]["w"] > 12]
    ev = {s: [r for r in rows if r["key"]["s"] == s] for s in (2025, 2026)}
    return tr, va, ev


TEAM_COLS = ["is_home", "team_n_cur", "team_spread", "implied_team_total", "t_car_blend", "t_car_l3", "o_car_allowed",
             "t_tgt_blend", "t_tgt_l3", "o_tgt_allowed", "t_plays_blend", "t_plays_l3", "o_plays_allowed",
             "t_dropbacks_blend", "t_dropbacks_l3", "o_dropbacks_allowed", "t_pass_rate_blend", "o_pass_rate_allowed",
             "t_rz_plays_blend", "o_rz_plays_allowed", "t_off_td_blend", "o_off_td_allowed", "t_pass_att_blend"]
SHARE_COLS = ["cur_n", "prior_n", "returning", "rookie_year", "draft_round", "snap_l3",
              "car_sh_l3", "car_sh_l8", "car_sh_cur", "car_sh_sd5", "car_sh_trend",
              "tgt_sh_l3", "tgt_sh_l8", "tgt_sh_cur", "tgt_sh_sd5", "tgt_sh_trend",
              "att_sh_l3", "att_sh_l8", "routes_l3", "route_rate_l3", "tprr", "team_spread", "is_home"]


def v2_feature_rows(fam_rows, fam, fb_v3):
    """Attach nfl_yardage_v2 features (the incumbent recipe) computed by the
    same v3 replayer the research ablations used."""
    cfg = {"rushing_yards": ("carries", "rushing_yards", ("RB", "FB"), 5.0),
           "receiving_yards": ("targets", "receiving_yards", ("WR", "TE", "RB"), 3.0),
           "receptions": ("targets", "receptions", ("WR", "TE", "RB"), 3.0),
           "passing_yards": ("attempts", "passing_yards", ("QB",), 15.0)}[fam]
    v3.MARKETS[f"_eb_{fam}"] = {"vol": cfg[0], "yds": cfg[1], "positions": cfg[2], "min_last3_vol": cfg[3]}
    rp = v3.Replayer(fb_v3, f"_eb_{fam}")
    got = {(k["pid"], k["s"], k["w"]): f_ for k, f_, _ in rp.replay()}
    del v3.MARKETS[f"_eb_{fam}"]
    out = []
    for r in fam_rows:
        k = (r["key"]["pid"], r["key"]["s"], r["key"]["w"])
        if k in got:
            r["F"].update({f"v2_{c}": got[k].get(c) for c in v2.FEATURES})
            out.append(r)
    return out


# ------------------------------------------------------------------ ladders
def ladder_rushing(rows):
    tr, va, ev = split(rows)
    A = lambda rr, k: np.array([r["A"][k] for r in rr], dtype=float)
    res = {}
    inc_cols = [f"v2_{c}" for c in v2.FEATURES]
    nomkt = [c for c in inc_cols if c[3:] not in MARKET_FEATURES]
    b_inc = fit(tr, va, inc_cols, A(tr, "rush_yds"), A(va, "rush_yds"))
    b_nomkt = fit(tr, va, nomkt, A(tr, "rush_yds"), A(va, "rush_yds"))
    # components
    b_T = fit(tr, va, TEAM_COLS, A(tr, "team_car"), A(va, "team_car"), "reg:squarederror")
    sh = lambda rr: A(rr, "car") / np.maximum(A(rr, "team_car"), 1)
    b_S = fit(tr, va, SHARE_COLS, sh(tr), sh(va), "reg:squarederror")
    # learned oracles
    for r in rows:
        r["F"]["ORC_team_car"] = r["A"]["team_car"]; r["F"]["ORC_car"] = r["A"]["car"]
    b_oT = fit(tr, va, inc_cols + TEAM_COLS + SHARE_COLS + ["ORC_team_car"], A(tr, "rush_yds"), A(va, "rush_yds"))
    b_oC = fit(tr, va, inc_cols + TEAM_COLS + SHARE_COLS + ["ORC_car"], A(tr, "rush_yds"), A(va, "rush_yds"))
    vp_inc = pred(b_inc, va, inc_cols)
    for s, er in ev.items():
        y = A(er, "rush_yds")
        T_hat = pred(b_T, er, TEAM_COLS); S_hat = np.clip(pred(b_S, er, SHARE_COLS), 0, 1)
        E_hat = np.array([r["F"]["ypc"] for r in er])
        base_hat = np.array([r["F"]["base_ypc"] for r in er]); xr = np.array([r["F"]["expl_rate"] for r in er])
        T = A(er, "team_car"); C = A(er, "car")
        inc = pred(b_inc, er, inc_cols)
        L = {
            "0_incumbent_v2_refit": inc,
            "0b_incumbent_without_market_inputs": pred(b_nomkt, er, nomkt),
            "1_decomposed_all_predicted": T_hat * S_hat * E_hat,
            "2_oracle_team_carries (plug-in)": T * S_hat * E_hat,
            "2_oracle_team_carries (learned)": pred(b_oT, er, inc_cols + TEAM_COLS + SHARE_COLS + ["ORC_team_car"]),
            "3_oracle_carries (plug-in)": C * E_hat,
            "3_oracle_carries (learned)": pred(b_oC, er, inc_cols + TEAM_COLS + SHARE_COLS + ["ORC_car"]),
            "4_oracle_carries+non-explosive yards, tail predicted": A(er, "base_yds") + C * xr,
            "4b_oracle_carries+explosive yards, base predicted": C * base_hat + A(er, "expl_yds"),
            "5_oracle_ypc (exact)": y,
        }
        res[str(s)] = {k: cont_metrics(v, y) for k, v in L.items()}
        res[str(s)]["_p_decomposed_vs_incumbent"] = boot_p(np.abs(L["1_decomposed_all_predicted"] - y), np.abs(inc - y))
        res[str(s)]["_distribution_incumbent"] = dist_metrics(inc, y, vp_inc, A(va, "rush_yds"))
        res[str(s)]["_component_errors"] = {
            "team_carries": count_metrics(T_hat, T), "carry_share_mae": round(float(np.abs(S_hat - C / np.maximum(T, 1)).mean()), 4),
            "carries_pred_mae": round(float(np.abs(T_hat * S_hat - C).mean()), 2),
            "ypc_pred_mae_on_carries>=5": round(float(np.abs(E_hat - y / np.maximum(C, 1))[C >= 5].mean()), 2)}
        if s == 2025:
            res["_eval_rows_2025"] = [(r, float(p)) for r, p in zip(er, inc)]
    return res


def ladder_receiving(rows, fb):
    tr, va, ev = split(rows)
    A = lambda rr, k: np.array([r["A"][k] if r["A"][k] is not None else np.nan for r in rr], dtype=float)
    inc_cols = [f"v2_{c}" for c in v2.FEATURES]
    nomkt = [c for c in inc_cols if c[3:] not in MARKET_FEATURES]
    b_inc = fit(tr, va, inc_cols, A(tr, "rec_yds"), A(va, "rec_yds"))
    b_nomkt = fit(tr, va, nomkt, A(tr, "rec_yds"), A(va, "rec_yds"))
    b_T = fit(tr, va, TEAM_COLS, A(tr, "team_tgt"), A(va, "team_tgt"), "reg:squarederror")
    sh = lambda rr: A(rr, "tgt") / np.maximum(A(rr, "team_tgt"), 1)
    b_S = fit(tr, va, SHARE_COLS, sh(tr), sh(va), "reg:squarederror")
    for r in rows:
        r["F"]["ORC_team_tgt"] = r["A"]["team_tgt"]; r["F"]["ORC_tgt"] = r["A"]["tgt"]; r["F"]["ORC_rec"] = r["A"]["rec"]
        r["F"]["ORC_routes"] = r["A"]["routes"]
    X = inc_cols + TEAM_COLS + SHARE_COLS + ["ypt", "catch", "ypr", "yac_pr", "adot"]
    b_o = {k: fit(tr, va, X + [k], A(tr, "rec_yds"), A(va, "rec_yds")) for k in ("ORC_team_tgt", "ORC_routes", "ORC_tgt", "ORC_rec")}
    vp_inc = pred(b_inc, va, inc_cols)
    res = {}
    for s, er in ev.items():
        y = A(er, "rec_yds")
        T_hat = pred(b_T, er, TEAM_COLS); S_hat = np.clip(pred(b_S, er, SHARE_COLS), 0, 1)
        ypt = np.array([r["F"]["ypt"] for r in er]); ypr = np.array([r["F"]["ypr"] for r in er])
        yacr = np.array([r["F"]["yac_pr"] for r in er])
        tprr = np.array([r["F"]["tprr"] if r["F"]["tprr"] is not None else np.nan for r in er])
        T = A(er, "team_tgt"); TG = A(er, "tgt"); RC = A(er, "rec"); RT = A(er, "routes")
        inc = pred(b_inc, er, inc_cols)
        L = {
            "0_incumbent_v2_refit": inc,
            "0b_incumbent_without_market_inputs": pred(b_nomkt, er, nomkt),
            "1_decomposed_all_predicted": T_hat * S_hat * ypt,
            "2_oracle_team_targets (plug-in)": T * S_hat * ypt,
            "2_oracle_team_targets (learned)": pred(b_o["ORC_team_tgt"], er, X + ["ORC_team_tgt"]),
            "4_oracle_targets (plug-in)": TG * ypt,
            "4_oracle_targets (learned)": pred(b_o["ORC_tgt"], er, X + ["ORC_tgt"]),
            "5_oracle_receptions (plug-in)": RC * ypr,
            "5_oracle_receptions (learned)": pred(b_o["ORC_rec"], er, X + ["ORC_rec"]),
            "6_oracle_receptions+completed_air_yards, YAC predicted": A(er, "cmp_air") + RC * yacr,
            "7_oracle_yards (exact)": y,
        }
        if s in fb.has_routes:
            ok = ~np.isnan(tprr) & ~np.isnan(RT)
            L["3_oracle_routes (plug-in)"] = np.where(ok, np.nan_to_num(RT) * np.nan_to_num(tprr), T_hat * S_hat) * ypt
            L["3_oracle_routes (learned)"] = pred(b_o["ORC_routes"], er, X + ["ORC_routes"])
        L = dict(sorted(L.items()))
        res[str(s)] = {k: cont_metrics(v, y) for k, v in L.items()}
        res[str(s)]["_p_decomposed_vs_incumbent"] = boot_p(np.abs(L["1_decomposed_all_predicted"] - y), np.abs(inc - y))
        res[str(s)]["_distribution_incumbent"] = dist_metrics(inc, y, vp_inc, A(va, "rec_yds"))
        res[str(s)]["_component_errors"] = {
            "team_targets": count_metrics(T_hat, T), "target_share_mae": round(float(np.abs(S_hat - TG / np.maximum(T, 1)).mean()), 4),
            "targets_pred": count_metrics(T_hat * S_hat, TG)}
        if s == 2025:
            res["_eval_rows_2025"] = [(r, float(p)) for r, p in zip(er, inc)]
    return res


def ladder_receptions(rows, fb):
    tr, va, ev = split(rows)
    A = lambda rr, k: np.array([r["A"][k] if r["A"][k] is not None else np.nan for r in rr], dtype=float)
    inc_cols = [f"v2_{c}" for c in v2.FEATURES]
    b_inc = fit(tr, va, inc_cols, A(tr, "rec"), A(va, "rec"), "reg:squarederror")
    b_T = fit(tr, va, TEAM_COLS, A(tr, "team_tgt"), A(va, "team_tgt"), "reg:squarederror")
    sh = lambda rr: A(rr, "tgt") / np.maximum(A(rr, "team_tgt"), 1)
    b_S = fit(tr, va, SHARE_COLS, sh(tr), sh(va), "reg:squarederror")
    res = {}
    for s, er in ev.items():
        y = A(er, "rec")
        T_hat = pred(b_T, er, TEAM_COLS); S_hat = np.clip(pred(b_S, er, SHARE_COLS), 0, 1)
        c = np.array([r["F"]["catch"] for r in er])
        tprr = np.array([r["F"]["tprr"] if r["F"]["tprr"] is not None else np.nan for r in er])
        T = A(er, "team_tgt"); TG = A(er, "tgt"); RT = A(er, "routes")
        L = {"0_incumbent_recipe_direct (v2 features, receptions target)": pred(b_inc, er, inc_cols),
             "1_decomposed_all_predicted": T_hat * S_hat * c,
             "2_oracle_team_targets": T * S_hat * c,
             "4_oracle_targets": TG * c,
             "5_oracle_catches (exact)": y}
        if s in fb.has_routes:
            ok = ~np.isnan(tprr) & ~np.isnan(RT)
            L["3_oracle_routes"] = np.where(ok, np.nan_to_num(RT) * np.nan_to_num(tprr), T_hat * S_hat) * c
        res[str(s)] = {k: count_metrics(v, y) for k, v in sorted(L.items())}
    return res


def ladder_passing(rows):
    tr, va, ev = split(rows)
    A = lambda rr, k: np.array([r["A"][k] for r in rr], dtype=float)
    inc_cols = [f"v2_{c}" for c in v2.FEATURES]
    b_inc = fit(tr, va, inc_cols, A(tr, "pass_yds"), A(va, "pass_yds"))
    b_P = fit(tr, va, TEAM_COLS, A(tr, "team_plays"), A(va, "team_plays"), "reg:squarederror")
    pr_ = lambda rr: A(rr, "team_db") / np.maximum(A(rr, "team_plays"), 1)
    b_R = fit(tr, va, TEAM_COLS, pr_(tr), pr_(va), "reg:squarederror")
    for r in rows:
        r["F"]["ORC_att"] = r["A"]["att"]; r["F"]["ORC_cmp"] = r["A"]["cmp"]
    X = inc_cols + TEAM_COLS + ["att_sh_l3", "att_sh_l8", "cmp_pct", "ypcmp", "air_pcmp"]
    b_oA = fit(tr, va, X + ["ORC_att"], A(tr, "pass_yds"), A(va, "pass_yds"))
    b_oC = fit(tr, va, X + ["ORC_cmp"], A(tr, "pass_yds"), A(va, "pass_yds"))
    res = {}
    for s, er in ev.items():
        y = A(er, "pass_yds")
        P_hat = pred(b_P, er, TEAM_COLS); R_hat = np.clip(pred(b_R, er, TEAM_COLS), 0, 1)
        ash = np.array([min(1.0, r["F"]["att_sh_l8"] or 0.9) for r in er])
        cp = np.array([r["F"]["cmp_pct"] for r in er]); ypc = np.array([r["F"]["ypcmp"] for r in er])
        yac_c = np.array([r["F"]["ypcmp"] - r["F"]["air_pcmp"] for r in er])
        P = A(er, "team_plays"); DB = A(er, "team_db"); AT = A(er, "att"); CM = A(er, "cmp")
        inc = pred(b_inc, er, inc_cols)
        L = {"0_incumbent_recipe_direct (v2 features, passing target)": inc,
             "1_decomposed_all_predicted": P_hat * R_hat * ash * cp * ypc,
             "2_oracle_team_plays": P * R_hat * ash * cp * ypc,
             "3_oracle_dropbacks (plays x pass rate)": DB * ash * cp * ypc,
             "4_oracle_attempts (plug-in)": AT * cp * ypc,
             "4_oracle_attempts (learned)": pred(b_oA, er, X + ["ORC_att"]),
             "5_oracle_completions (plug-in)": CM * ypc,
             "5_oracle_completions (learned)": pred(b_oC, er, X + ["ORC_cmp"]),
             "6_oracle_completions+air_yards, YAC predicted": A(er, "qb_cmp_air") + CM * yac_c,
             "7_exact": y}
        res[str(s)] = {k: cont_metrics(v, y) for k, v in L.items()}
        res[str(s)]["_p_decomposed_vs_incumbent"] = boot_p(np.abs(L["1_decomposed_all_predicted"] - y), np.abs(inc - y))
        res[str(s)]["_component_errors"] = {"team_plays": count_metrics(P_hat, P),
                                            "pass_rate_mae": round(float(np.abs(R_hat - DB / np.maximum(P, 1)).mean()), 4),
                                            "attempts_pred": count_metrics(P_hat * R_hat * ash, AT)}
    return res


def ladder_td(rows):
    tr, va, ev = split(rows)
    y = lambda rr: np.array([1.0 if r["A"]["td"] >= 1 else 0.0 for r in rr])
    base = [c for c in SHARE_COLS + TEAM_COLS if c not in ("team_spread", "is_home")] + ["team_spread", "is_home"] + \
           ["car_l3", "tgt_l3", "td_l3", "td_cur", "td_pri", "rz_car_l3", "gl_car_l3", "rz_tgt_l3", "ez_tgt_l3", "rush_yds_l3", "rec_yds_l3"]
    base = list(dict.fromkeys(base))
    for r in rows:
        a = r["A"]
        r["F"].update({"ORC_team_off_td": a["team_off_td"], "ORC_team_rz_plays": a["team_rz_plays"],
                       "ORC_touches": a["car"] + a["tgt"], "ORC_rz_car": a["rz_car"], "ORC_gl_car": a["gl_car"],
                       "ORC_rz_tgt": a["rz_tgt"], "ORC_ez_tgt": a["ez_tgt"]})
    rungs = [("1_incumbent_style (history + team + usage)", []),
             ("2_oracle_team_offensive_TDs", ["ORC_team_off_td"]),
             ("3_oracle_team_red_zone_plays", ["ORC_team_rz_plays"]),
             ("4_oracle_player_touches", ["ORC_touches"]),
             ("5_oracle_player_red_zone+goal_line+end_zone_opps", ["ORC_rz_car", "ORC_gl_car", "ORC_rz_tgt", "ORC_ez_tgt"]),
             ("6_oracle_team_TDs + player_rz_opps", ["ORC_team_off_td", "ORC_rz_car", "ORC_gl_car", "ORC_rz_tgt", "ORC_ez_tgt"])]
    res = {}
    fits = {name: fit(tr, va, base + add, y(tr), y(va), "binary:logistic") for name, add in rungs}
    for s, er in ev.items():
        yy = y(er)
        res[str(s)] = {"0_base_rate": binary_metrics(np.full(len(yy), y(tr).mean()), yy)}
        for name, add in rungs:
            res[str(s)][name] = binary_metrics(pred(fits[name], er, base + add), yy)
    return res


# ------------------------------------------------------------------ segmentation
def segments(eval_rows, stat_key, fam):
    """Incumbent error by regime (2025)."""
    out = {}
    if not eval_rows:
        return out
    vol = {"rushing_yards": "car", "receiving_yards": "tgt"}[fam]
    shk = {"rushing_yards": "car_sh", "receiving_yards": "tgt_sh"}[fam]
    recs = []
    for r, p in eval_rows:
        F, sg = r["F"], r["seg"]
        a = r["A"][stat_key]
        recs.append({"err": p - a, "abs": abs(p - a), "pid": r["key"]["pid"], "w": r["key"]["w"], **{
            "position": sg["pos"],
            "usage_tier": F.get(f"{vol}_l3"),
            "rookie": "rookie" if sg["rookie"] else "veteran",
            "draft": ("undrafted" if sg["draft_round"] is None else "rd1-2" if sg["draft_round"] <= 2 else "rd3+"),
            "role_volatility": F.get(f"{shk}_sd5"),
            "role_trend": F.get(f"{shk}_trend"),
            "returning_from_absence": "returning" if F.get("returning") else "continuous",
            "team_change": "changed team" if sg["team_change"] else "same team",
            "qb_change": "different starting QB" if sg["qb_change"] else "same QB",
            "head_coach_change": "new HC this season" if sg["coach_change"] else "same HC",
            "snap_share": F.get("snap_l3"),
            "game_script_postgame": sg["margin"],
            "opp_strength_ypc_allowed" if fam == "rushing_yards" else "opp_strength_ypt_allowed":
                F.get("o_rush_ypc_allowed") if fam == "rushing_yards" else F.get("o_rec_ypt_allowed"),
            "week_bucket": "wk1-4" if sg["week"] <= 4 else "wk5-9" if sg["week"] <= 9 else "wk10-14" if sg["week"] <= 14 else "wk15-18",
            "injury_report": sg["inj_status"] or "not listed",
        }})
    numeric = {"usage_tier", "role_volatility", "role_trend", "snap_share", "game_script_postgame",
               "opp_strength_ypc_allowed", "opp_strength_ypt_allowed"}
    for fld in [k for k in recs[0] if k not in ("err", "abs", "pid", "w")]:
        vals = [x[fld] for x in recs]
        if fld in numeric:
            xs = np.array([v for v in vals if v is not None], dtype=float)
            if len(xs) < 30:
                continue
            q1, q2 = np.quantile(xs, [1 / 3, 2 / 3])
            lab = lambda v: "missing" if v is None else f"low (<{q1:.2f})" if v < q1 else f"high (>={q2:.2f})" if v >= q2 else "mid"
        else:
            lab = lambda v: str(v)
        g = defaultdict(list)
        for x in recs:
            g[lab(x[fld])].append(x)
        out[fld] = {k: {"n": len(v), "mae": round(float(np.mean([x["abs"] for x in v])), 2),
                        "bias": round(float(np.mean([x["err"] for x in v])), 2)}
                    for k, v in sorted(g.items()) if len(v) >= 15}
    # repeated player residuals: split-half persistence
    by = defaultdict(lambda: ([], []))
    for x in recs:
        by[x["pid"]][x["w"] % 2].append(x["err"])
    pairs = [(np.mean(a), np.mean(b)) for a, b in by.values() if len(a) >= 3 and len(b) >= 3]
    if len(pairs) >= 20:
        a, b = np.array(pairs).T
        out["player_residual_persistence"] = {"players": len(pairs), "odd_vs_even_week_corr": round(float(np.corrcoef(a, b)[0, 1]), 3),
                                              "mean_abs_player_bias": round(float(np.mean(np.abs((a + b) / 2))), 2)}
        # cross-fitted: correct each half by the player's mean residual in the other half
        corr_abs, raw_abs = [], []
        for x in recs:
            o = by[x["pid"]][1 - x["w"] % 2]
            if len(o) >= 3 and len(by[x["pid"]][x["w"] % 2]) >= 3:
                raw_abs.append(x["abs"]); corr_abs.append(abs(x["err"] - np.mean(o)))
        out["player_residual_persistence"]["mae_if_player_bias_removed_crossfit"] = {
            "n": len(raw_abs), "raw": round(float(np.mean(raw_abs)), 2), "corrected": round(float(np.mean(corr_abs)), 2)}
    return out


def availability(fb):
    """P(plays | final injury-report status) for skill players with a real role."""
    out = {}
    tallies = defaultdict(lambda: [0, 0]); snapdrop = defaultdict(list)
    # role players: had >=30% snaps in any of their previous 3 team games
    by_week = defaultdict(dict)
    for (s, w, t), m in fb.snap.items():
        for nn, pct in m.items():
            by_week[(s, w)][(t, nn)] = pct
    name_of = {}
    for key, pl in fb.pl.items():
        for pid, r in pl.items():
            name_of[pid] = (v3.norm_name(r["name"]), r["pos"])
    for (s, w, pid), (status, prac, team) in fb.inj.items():
        if s not in (2023, 2024, 2025) or pid not in name_of:
            continue
        nn, pos = name_of[pid]
        if pos not in ("QB", "RB", "WR", "TE"):
            continue
        prev = [by_week.get((s, ww), {}).get((team, nn)) for ww in range(max(1, w - 3), w)]
        prev = [p for p in prev if p is not None]
        if not prev or max(prev) < 0.3:
            continue
        pct = by_week.get((s, w), {}).get((team, nn))
        st = status or "listed, no game status"
        tallies[st][0] += 1; tallies[st][1] += 1 if pct else 0
        if pct:
            snapdrop[st].append(pct - np.mean(prev))
    for st, (n, p) in tallies.items():
        out[st] = {"n": n, "p_played": round(p / n, 3),
                   "avg_snap_change_if_played": round(float(np.mean(snapdrop[st])), 3) if snapdrop[st] else None}
    return out


def realized_vs_pregame_absence(fb):
    """How often a regular's absence was NOT announced as Out/Doubtful on the
    final report -- the information gap between features built from realized
    participation (research v3/v4/v5 training) and what is knowable pre-game."""
    n_abs = n_announced = 0
    for (s, w, t), m in fb.snap.items():
        if s not in (2024, 2025):
            continue
        prev_keys = [(s, ww, t) for ww in range(max(1, w - 3), w)]
        regs = set()
        for k in prev_keys:
            regs |= {nn for nn, pct in fb.snap.get(k, {}).items() if (pct or 0) >= 0.5}
        if not regs or (s, w, t) not in fb.pl:
            continue
        absent = regs - set(m)
        # map names to injury statuses this week
        inj_names = {}
        for pid, r in ((pid, r) for k in prev_keys for pid, r in fb.pl.get(k, {}).items()):
            st = fb.inj.get((s, w, pid))
            if st:
                inj_names[v3.norm_name(r["name"])] = st[0]
        for nn in absent:
            n_abs += 1
            if inj_names.get(nn) in ("Out", "Doubtful"):
                n_announced += 1
    return {"regular_absences": n_abs, "announced_out_or_doubtful": n_announced,
            "share_unannounced": round(1 - n_announced / n_abs, 3) if n_abs else None,
            "note": "unannounced = IR/PUP/suspension/trade/benching/healthy scratch or late change; "
                    "IR players are not on the weekly report, so part of this is knowable from rosters"}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    print("NFL_PLAYER_OUTCOME_ERROR_BUDGET_A\n=================================")
    download(args.data_dir)
    fb = Football(args.data_dir)
    print("data loaded; building as-of rows ...", flush=True)
    rows = build_rows(fb)
    print({k: len(v) for k, v in rows.items()}, flush=True)
    fb_v3 = v3.Data(args.data_dir, SEASONS)
    # attach receptions/passing columns the v3 loader doesn't carry
    for key, pl in fb.pl.items():
        for pid, r in pl.items():
            row = fb_v3.stats.get(key, {}).get(pid)
            if row is not None:
                row["receptions"] = r["rec"]; row["passing_yards"] = r["pass_yds"]
    report = {"protocol": {"train": "2023 + 2024 wk1-12", "valid": "2024 wk13-18", "eval": ["2025", "2026 wk1-3"],
                           "population": "players who played (conditional production forecast)",
                           "incumbent": "nfl_yardage_v2 feature recipe, reg:absoluteerror XGBoost, re-fit on the train split"}}
    for fam in ("rushing_yards", "receiving_yards", "receptions", "passing_yards"):
        rows[fam] = v2_feature_rows(rows[fam], fam, fb_v3)
    print("rushing ...", flush=True)
    rush = ladder_rushing(rows["rushing_yards"])
    print("receiving ...", flush=True)
    recv = ladder_receiving(rows["receiving_yards"], fb)
    print("receptions / passing / TD ...", flush=True)
    report["rushing_yards"] = {k: v for k, v in rush.items() if not k.startswith("_eval")}
    report["receiving_yards"] = {k: v for k, v in recv.items() if not k.startswith("_eval")}
    report["receptions"] = ladder_receptions(rows["receptions"], fb)
    report["passing_yards"] = ladder_passing(rows["passing_yards"])
    report["anytime_td"] = ladder_td(rows["anytime_td"])
    report["segments_2025"] = {"rushing_yards": segments(rush.get("_eval_rows_2025"), "rush_yds", "rushing_yards"),
                               "receiving_yards": segments(recv.get("_eval_rows_2025"), "rec_yds", "receiving_yards")}
    report["availability_layer"] = {"p_played_given_final_status_2023_2025": availability(fb),
                                    "realized_vs_pregame_absence_2024_2025": realized_vs_pregame_absence(fb)}
    # shipped v2 artifact on its own training season (leakage measurement)
    try:
        import xgboost as xgb
        lk = {}
        for fam, key in (("rushing_yards", "rush_yds"), ("receiving_yards", "rec_yds")):
            b = xgb.Booster(); b.load_model(str(v2.MODEL_DIR / f"nfl_{fam}_v2.json"))
            er = [r for r in rows[fam] if r["key"]["s"] == 2025]
            X = np.array([[r["F"].get(f"v2_{c}") if r["F"].get(f"v2_{c}") is not None else np.nan for c in v2.FEATURES] for r in er], dtype=np.float32)
            p = b.predict(xgb.DMatrix(X, feature_names=v2.FEATURES))
            lk[fam] = cont_metrics(p, np.array([r["A"][key] for r in er], dtype=float))
        report["leakage_shipped_v2_artifact_on_2025_in_sample"] = lk
    except Exception as e:
        report["leakage_shipped_v2_artifact_on_2025_in_sample"] = {"error": str(e)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print_summary(report)
    return 0


def print_summary(rep):
    for fam in ("rushing_yards", "receiving_yards", "passing_yards"):
        print(f"\n== {fam}")
        for s in ("2025", "2026"):
            for k, m in rep[fam][s].items():
                if not k.startswith("_"):
                    print(f"  [{s}] {k:62s} MAE {m['mae']:6.2f}  medAE {m['median_ae']:6.2f}  bias {m['bias']:+6.2f}  "
                          f"w10 {m['within_10']:.0%} w20 {m['within_20']:.0%}  n={m['n']}")
    print("\n== receptions")
    for s in ("2025", "2026"):
        for k, m in rep["receptions"][s].items():
            print(f"  [{s}] {k:62s} MAE {m['mae']:5.3f}  within1 {m['within_1']:.0%}  n={m['n']}")
    print("\n== anytime TD")
    for s in ("2025", "2026"):
        for k, m in rep["anytime_td"][s].items():
            print(f"  [{s}] {k:55s} logloss {m['logloss']:.4f}  brier {m['brier']:.4f}  auc {m['auc']}  ece {m['ece']}")


if __name__ == "__main__":
    raise SystemExit(main())
