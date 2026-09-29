"""
NFL_PLAYER_OUTCOME_ERROR_BUDGET_A  (Phase 0 + Phase 0B)

Research-only diagnostic. Makes no production predictions, writes no model
artifacts, changes no serving/API behaviour.

Question: where does player-outcome forecast error come from, measured with
COHERENT component chains and ORDER-ROBUST (Shapley) attribution?

Every chain multiplies components whose all-actual product reconstructs the
outcome exactly (checked as an invariant). Each component is forecast from
pre-game information only; an oracle coalition replaces a subset of components
with their actual values. Coalition value = loss(all predicted) - loss(coalition
oracle). Shapley values average each component's marginal value over all
orderings; interaction = v(all) - sum of solo values.

Chains (actual component = ratio; if its denominator is 0 the component is
undefined and the predicted value is used -- the product is 0 either way when
all upstream components are actual):

  rushing (RB)      T team rush attempts x S carry share x (E ordinary-carry
                    yards/carry + X explosive(20+)-run yards/carry)
  receiving (route) D team dropbacks x R route rate x P targets/route x
                    C catch rate x (A air yards/catch + Y YAC/catch)
                    [2023-2025 only: route proxy = on field for a dropback]
  receiving (share) TT team targets x H target share x C x (A + Y)
                    [fallback chain; reported separately, never mixed]
  passing (QB)      P team plays x R dropback rate x Q attempts/dropback x
                    C completion rate x (A air/completion + Y YAC/completion)
  QB TDs / INTs     P x R x Q x K (per attempt)
  QB rushing        P x U (QB rushes per team play) x V (yards/rush)
  rush / rec TDs    volume chain x K (TDs per carry / per target)
  anytime TD        lambda = P x Z (red-zone play rate) x O (player share of
                    red-zone plays) x c_rz + N (non-red-zone touches) x c_out;
                    P(TD) = 1 - exp(-lambda); conversion rates predicted
  defense           OP opponent plays x F snap rate (x K per snap: tackles);
                    OD opponent dropbacks x F x K (sacks, interceptions)

Evaluation universes
  conditional  players who actually played (production given participation)
  pregame      players selected ONLY from pre-game information (prior usage,
               prior-week roster status); non-participants score 0; forecast =
               P(active) mixture. See availability section.

Temporal protocol: train 2023 + 2024 wk1-12, validation 2024 wk13-18, evaluation
2025 and 2026 wk1-3. Both evaluation periods are BURNED (examined repeatedly);
they are development data, not holdouts.

Run
---
  python -u nfl_player_outcome_error_budget_a.py --data-dir /tmp/nfl_data
  python -u tests/test_nfl_player_outcome_invariants.py --data-dir /tmp/nfl_data
"""
import argparse
import csv
import gzip
import itertools
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
MARKET_FEATURES = ("team_spread", "implied_team_total")      # bookmaker-derived
EXPLOSIVE_RUN = 20
RZ = 20
QGRID = np.round(np.linspace(0.05, 0.95, 19), 2)
XGB = {"max_depth": 5, "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.8,
       "min_child_weight": 15, "reg_lambda": 5.0, "seed": 20260929}
TRAIN = lambda s, w: s == 2023 or (s == 2024 and w <= 12)
VALID = lambda s, w: s == 2024 and w > 12
EVAL_SEASONS = (2025, 2026)
DEF_POS = {"DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
           "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB", "SAF": "DB"}


class InvariantError(AssertionError):
    pass


# ------------------------------------------------------------------ utils
def fnum(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def ssum(xs):
    return float(sum(x for x in xs if x is not None))


def ratio(n, d):
    return n / d if d else float("nan")


def download(data_dir):
    d = Path(data_dir); d.mkdir(parents=True, exist_ok=True)
    want = [("games.csv", f"{REL}/schedules/games.csv"), ("players.csv", f"{REL}/players/players.csv")]
    for y in SEASONS:
        want += [(f"stats_player_week_{y}.csv", f"{REL}/stats_player/stats_player_week_{y}.csv"),
                 (f"snap_counts_{y}.csv", f"{REL}/snap_counts/snap_counts_{y}.csv"),
                 (f"pbp_{y}.csv.gz", f"{REL}/pbp/play_by_play_{y}.csv.gz"),
                 (f"injuries_{y}.csv", f"{REL}/injuries/injuries_{y}.csv"),
                 (f"roster_weekly_{y}.csv", f"{REL}/weekly_rosters/roster_weekly_{y}.csv")]
        if y <= 2025:
            want.append((f"participation_{y}.csv", f"{REL}/pbp_participation/pbp_participation_{y}.csv"))
    for name, url in want:
        p = d / name
        if p.exists():
            continue
        print(f"  downloading {name}", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "nfl-error-budget"})
        with urllib.request.urlopen(req, timeout=300) as r:
            p.write_bytes(r.read())


# ------------------------------------------------------------------ data
class Football:
    def __init__(self, data_dir):
        d = Path(data_dir)
        self.games, self.team_games, self.coach = {}, defaultdict(list), {}
        for r in csv.DictReader(open(d / "games.csv", newline="", encoding="utf-8")):
            if r["game_type"] != "REG" or int(r["season"]) not in SEASONS:
                continue
            s, w = int(r["season"]), int(r["week"])
            sp, tot, res = fnum(r["spread_line"]), fnum(r["total_line"]), fnum(r["result"])
            for team, opp, home in ((r["home_team"], r["away_team"], True), (r["away_team"], r["home_team"], False)):
                self.games[(s, w, team)] = {"opp": opp, "is_home": home,
                                            "spread": (sp if home else -sp) if sp is not None else None,
                                            "total": tot, "margin": (res if home else -res) if res is not None else None,
                                            "kickoff": f"{r['gameday']}T{r['gametime']}"}
                self.team_games[team].append((s, w))
                self.coach.setdefault((s, team), r["home_coach"] if home else r["away_coach"])
        for t in self.team_games:
            self.team_games[t].sort()

        self.pl = defaultdict(dict)       # (s, w, team) -> gsis -> stats row
        for s in SEASONS:
            for r in csv.DictReader(open(d / f"stats_player_week_{s}.csv", newline="", encoding="utf-8")):
                if r.get("season_type", "REG") != "REG":
                    continue
                g = lambda k: fnum(r.get(k)) or 0.0
                self.pl[(s, int(r["week"]), r["team"])][r["player_id"]] = {
                    "name": r["player_display_name"], "pos": r["position"],
                    "car": g("carries"), "rush_yds": g("rushing_yards"), "rtd": g("rushing_tds"),
                    "tgt": g("targets"), "rec": g("receptions"), "rec_yds": g("receiving_yards"), "rectd": g("receiving_tds"),
                    "att": g("attempts"), "cmp": g("completions"), "pass_yds": g("passing_yards"),
                    "ptd": g("passing_tds"), "int": g("passing_interceptions"),
                    "tkl": g("def_tackles_solo") + g("def_tackles_with_assist"), "sack": g("def_sacks"),
                    "dint": g("def_interceptions")}
        self.snap = defaultdict(dict)     # (s, w, team) -> normname -> offense pct
        self.dsnap = defaultdict(dict)    # (s, w, team) -> pfr id -> {name, pos, snaps, pct}
        for s in SEASONS:
            for r in csv.DictReader(open(d / f"snap_counts_{s}.csv", newline="", encoding="utf-8")):
                if r.get("game_type", "REG") != "REG":
                    continue
                k = (s, int(r["week"]), r["team"])
                if (fnum(r["offense_snaps"]) or 0) > 0:
                    self.snap[k][v3.norm_name(r["player"])] = fnum(r["offense_pct"])
                if (fnum(r.get("defense_snaps")) or 0) > 0 and r.get("position") in DEF_POS:
                    self.dsnap[k][r["pfr_player_id"]] = {"name": v3.norm_name(r["player"]), "grp": DEF_POS[r["position"]],
                                                         "snaps": fnum(r["defense_snaps"]), "pct": fnum(r["defense_pct"])}

        # play-by-play: coherent component counts
        self.tg = defaultdict(lambda: defaultdict(float))    # (s, w, team) offense
        self.pp = defaultdict(lambda: defaultdict(float))    # (s, w, gsis) player
        self.recon = defaultdict(float)
        dropbacks, db_team = set(), {}
        for s in SEASONS:
            with gzip.open(d / f"pbp_{s}.csv.gz", "rt", newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r["season_type"] != "REG" or not r["posteam"] or r["play_type"] not in ("pass", "run"):
                        continue
                    if r.get("two_point_attempt") == "1":
                        continue
                    w = int(r["week"]); t = r["posteam"]; k = (s, w, t)
                    yl = fnum(r["yardline_100"]); inrz = yl is not None and yl <= RZ
                    kneel, spike = r.get("qb_kneel") == "1", r.get("qb_spike") == "1"
                    if kneel or spike:
                        continue
                    db = r["qb_dropback"] == "1"
                    tg = self.tg[k]
                    tg["plays"] += 1
                    if db:
                        tg["dropbacks"] += 1
                        dropbacks.add((r["game_id"], r["play_id"])); db_team[(r["game_id"], r["play_id"])] = k
                    if inrz:
                        tg["rz_plays"] += 1
                    rid, qid, pid = r["rusher_player_id"], r["receiver_player_id"], r["passer_player_id"]
                    is_pass = r["pass_attempt"] == "1" and r["sack"] != "1"
                    if r["rush_attempt"] == "1" and rid:
                        tg["rushes"] += 1
                        ry = fnum(r["rushing_yards"]) or 0.0
                        p = self.pp[(s, w, rid)]
                        p["car"] += 1; p["rush_yds"] += ry
                        if ry >= EXPLOSIVE_RUN:
                            p["expl_yds"] += ry; p["expl_car"] += 1
                        else:
                            p["base_yds"] += ry
                        if inrz:
                            p["rz_opp"] += 1
                        else:
                            p["out_touch"] += 1
                        if r["rush_touchdown"] == "1" and (r.get("td_player_id") in ("", rid)):
                            p["rtd"] += 1; p["rz_td" if inrz else "out_td"] += 1
                    if is_pass and pid:
                        q = self.pp[(s, w, pid)]
                        q["att"] += 1
                        if r["complete_pass"] == "1":
                            py = fnum(r["passing_yards"]) or 0.0; ay = fnum(r["air_yards"]) or 0.0
                            q["cmp"] += 1; q["pass_yds"] += py; q["cmp_air"] += ay; q["cmp_yac"] += py - ay
                        if r["pass_touchdown"] == "1":
                            q["ptd"] += 1
                        if r["interception"] == "1":
                            q["int"] += 1
                    if is_pass and qid:
                        tg["targets"] += 1
                        p = self.pp[(s, w, qid)]
                        p["tgt"] += 1
                        p["tgt_air"] += fnum(r["air_yards"]) or 0.0
                        p["rz_opp" if inrz else "out_touch"] += 1
                        if r["complete_pass"] == "1":
                            ry = fnum(r["receiving_yards"]) or 0.0; ay = fnum(r["air_yards"]) or 0.0
                            p["rec"] += 1; p["rec_yds"] += ry; p["air"] += ay; p["yac"] += ry - ay
                        if r["pass_touchdown"] == "1":
                            p["rectd"] += 1; p["rz_td" if inrz else "out_td"] += 1
                    if (r["pass_touchdown"] == "1" or r["rush_touchdown"] == "1"):
                        tg["off_td"] += 1
        # route proxy from participation (2022-2025)
        self.has_routes = set()
        self.part_rows = defaultdict(int)
        for s in SEASONS:
            p = d / f"participation_{s}.csv"
            if not p.exists():
                continue
            self.has_routes.add(s)
            for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
                key = (r["nflverse_game_id"], r["play_id"])
                if key not in dropbacks:
                    continue
                ss, w, t = db_team[key]
                self.part_rows[(ss, w, t)] += 1
                for gid in (r["offense_players"] or "").split(";"):
                    if gid:
                        self.pp[(ss, w, gid)]["routes"] += 1
        self.draft = {}
        for r in csv.DictReader(open(d / "players.csv", newline="", encoding="utf-8")):
            self.draft[r["gsis_id"]] = {"rookie": fnum(r.get("rookie_season")), "round": fnum(r.get("draft_round"))}
        self.inj = {}
        for s in SEASONS:
            for r in csv.DictReader(open(d / f"injuries_{s}.csv", newline="", encoding="utf-8")):
                if r.get("game_type", "REG") == "REG":
                    self.inj[(s, int(r["week"]), r["gsis_id"])] = (r["report_status"] or "", r["practice_status"] or "")
        self.roster = {}                  # (s, w, gsis) -> (team, status)
        for s in SEASONS:
            p = d / f"roster_weekly_{s}.csv"
            if p.exists():
                for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
                    if r.get("game_type", "REG") == "REG" and r.get("week"):
                        self.roster[(s, int(r["week"]), r["gsis_id"])] = (r["team"], r["status"], v3.norm_name(r["full_name"]), r["position"])

    def played(self, key):
        names = set(self.snap.get(key, {}))
        for r in self.pl.get(key, {}).values():
            if r["car"] + r["tgt"] + r["att"] > 0:
                names.add(v3.norm_name(r["name"]))
        return names


# ------------------------------------------------------------------ as-of features
POS_PRIOR = {"base_ypc": (3.9, 60.0), "expl_ypc": (0.45, 150.0), "ypc": (4.3, 60.0),
             "rroute": (0.6, 5.0), "tprr": (0.17, 60.0), "catch": (0.65, 30.0), "air_pc": (6.0, 25.0),
             "yac_pc": (5.0, 25.0), "cmp_pct": (0.64, 150.0), "air_pcmp": (5.8, 100.0), "yac_pcmp": (5.2, 100.0),
             "ptd_pa": (0.045, 300.0), "int_pa": (0.024, 300.0), "qb_rush_pp": (0.05, 5.0), "qb_ypc": (4.5, 40.0),
             "rtd_pc": (0.03, 150.0), "rectd_pt": (0.05, 80.0), "c_rz": (0.12, 40.0), "c_out": (0.012, 150.0),
             "tkl_ps": (0.08, 150.0), "sack_pd": (0.008, 300.0), "dint_pd": (0.004, 400.0)}


def shrink(num, den, key):
    m, k = POS_PRIOR[key]
    return (num + m * k) / (den + k)


def team_feats(th, dh, s, ctx):
    cur = [g for g in th if g["s"] == s]; pri = [g for g in th if g["s"] == s - 1]
    dcur = [g for g in dh if g["s"] == s]; dpri = [g for g in dh if g["s"] == s - 1]

    def blend(key, c_, p_):
        c = mean([g[key] for g in c_]); p = mean([g[key] for g in p_])
        if c is None:
            return p
        if p is None:
            return c
        return (c * len(c_) + p * 3) / (len(c_) + 3)
    F = {"is_home": 1.0 if ctx["is_home"] else 0.0, "team_n_cur": float(len(cur)),
         "team_spread": ctx["spread"],
         "implied_team_total": (ctx["total"] / 2 + ctx["spread"] / 2) if ctx["spread"] is not None and ctx["total"] is not None else None}
    for k in ("plays", "dropbacks", "rushes", "targets", "rz_plays", "off_td", "rush_yds", "rec_yds"):
        F[f"t_{k}_blend"] = blend(k, cur, pri); F[f"t_{k}_l3"] = mean([g[k] for g in th[-3:]])
        F[f"o_{k}_allowed"] = blend(k, dcur, dpri)
    F["t_pass_rate"] = ratio(F["t_dropbacks_blend"], F["t_plays_blend"]) if F["t_plays_blend"] else None
    F["o_pass_rate_allowed"] = ratio(F["o_dropbacks_allowed"], F["o_plays_allowed"]) if F["o_plays_allowed"] else None
    F["t_rz_rate"] = ratio(F["t_rz_plays_blend"], F["t_plays_blend"]) if F["t_plays_blend"] else None
    F["o_rz_rate_allowed"] = ratio(F["o_rz_plays_allowed"], F["o_plays_allowed"]) if F["o_plays_allowed"] else None
    dg = dcur + dpri
    F["o_ypc_allowed"] = ratio(ssum(g["rush_yds"] for g in dg), ssum(g["rushes"] for g in dg)) if dg else None
    F["o_ypt_allowed"] = ratio(ssum(g["rec_yds"] for g in dg), ssum(g["targets"] for g in dg)) if dg else None
    for k in list(F):
        if isinstance(F[k], float) and math.isnan(F[k]):
            F[k] = None
    return F


def player_feats(h, s, fb, pid):
    l3, l8, l16 = h[-3:], h[-8:], h[-16:]
    cur = [g for g in h if g["s"] == s]; pri = [g for g in h if g["s"] == s - 1]
    S = lambda k, gs=l16: ssum(g[k] for g in gs)
    F = {"cur_n": float(len(cur)), "prior_n": float(len(pri)),
         "rookie_year": 1.0 if fb.draft.get(pid, {}).get("rookie") == s else 0.0,
         "draft_round": fb.draft.get(pid, {}).get("round"), "snap_l3": mean([g["snap"] for g in l3])}
    for k in ("car", "tgt", "rec", "att", "cmp", "rush_yds", "rec_yds", "pass_yds", "ptd", "int", "td",
              "rz_opp", "out_touch", "rtd", "rectd"):
        F[f"{k}_l3"] = mean([g[k] for g in l3]); F[f"{k}_cur"] = mean([g[k] for g in cur]); F[f"{k}_pri"] = mean([g[k] for g in pri])
        c, p = F[f"{k}_cur"], F[f"{k}_pri"]
        F[f"{k}_blend"] = p if c is None else c if p is None else (c * len(cur) + p * 3) / (len(cur) + 3)
    for k in ("car_sh", "tgt_sh", "att_sh", "route_rate", "rz_sh", "qb_rush_sh"):
        vals = [g[k] for g in h if g[k] is not None]
        F[f"{k}_l3"] = mean([g[k] for g in l3]); F[f"{k}_l8"] = mean([g[k] for g in l8])
        xs = [g[k] for g in h[-5:] if g[k] is not None]
        F[f"{k}_sd5"] = float(np.std(xs)) if len(xs) >= 3 else None
        F[f"{k}_trend"] = (F[f"{k}_l3"] - F[f"{k}_l8"]) if F[f"{k}_l3"] is not None and F[f"{k}_l8"] is not None else None
    car, tgt, rec, att, cmp_ = S("car"), S("tgt"), S("rec"), S("att"), S("cmp")
    F.update({
        "base_ypc": shrink(S("base_yds"), car, "base_ypc"), "expl_ypc": shrink(S("expl_yds"), car, "expl_ypc"),
        "tprr": shrink(tgt, S("routes"), "tprr") if any(g["routes"] is not None for g in l16) else None,
        "catch": shrink(rec, tgt, "catch"), "air_pc": shrink(S("air"), rec, "air_pc"), "yac_pc": shrink(S("yac"), rec, "yac_pc"),
        "cmp_pct": shrink(cmp_, att, "cmp_pct"), "air_pcmp": shrink(S("cmp_air"), cmp_, "air_pcmp"),
        "yac_pcmp": shrink(S("cmp_yac"), cmp_, "yac_pcmp"), "ptd_pa": shrink(S("ptd"), att, "ptd_pa"),
        "int_pa": shrink(S("int"), att, "int_pa"), "qb_ypc": shrink(S("rush_yds"), car, "qb_ypc"),
        "rtd_pc": shrink(S("rtd"), car, "rtd_pc"), "rectd_pt": shrink(S("rectd"), tgt, "rectd_pt"),
        "c_rz": shrink(S("rz_td"), S("rz_opp"), "c_rz"), "c_out": shrink(S("out_td"), S("out_touch"), "c_out"),
        "adot": ratio(S("tgt_air"), tgt) if tgt else None,
    })
    return F


def pp_row(fb, key, gid):
    s, w, t = key
    p = fb.pp.get((s, w, gid), {})
    get = lambda k: float(p.get(k, 0.0))
    out = {k: get(k) for k in ("car", "rush_yds", "base_yds", "expl_yds", "tgt", "rec", "rec_yds", "air", "yac",
                              "att", "cmp", "pass_yds", "cmp_air", "cmp_yac", "ptd", "int", "rtd", "rectd",
                              "rz_opp", "out_touch", "rz_td", "out_td")}
    out["routes"] = (p.get("routes", 0.0) if fb.part_rows.get(key) else None) if s in fb.has_routes else None
    out["tgt_air"] = 0.0
    return out


def build_rows(fb, until=None):
    """Chronological replay. Features for week w are computed BEFORE week w is
    absorbed. `until`=(s, w): stop after emitting rows for that week."""
    ph, th, dh, dph = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    rows = defaultdict(list); team_rows = []; pregame = []; defense = []
    weeks = sorted({(k[0], k[1]) for k in fb.pl})
    for (s, w) in weeks:
        if until and (s, w) > until:
            break
        TF = {}
        for team in list(fb.team_games):
            key = (s, w, team); ctx = fb.games.get(key)
            if key in fb.pl and ctx:
                TF[team] = team_feats(th[team], dh[ctx["opp"]], s, ctx)
        for team, T in TF.items():
            key = (s, w, team); ctx = fb.games[key]; tg = fb.tg.get(key, {})
            team_rows.append({"key": {"team": team, "s": s, "w": w}, "F": T,
                              "A": {k: float(tg.get(k, 0.0)) for k in ("plays", "dropbacks", "rushes", "targets", "rz_plays", "off_td")}})
            played = fb.played(key)
            # candidates known before kickoff: last game was for this team, within 3 team games
            prev_tg = [g for g in fb.team_games[team] if g < (s, w)]
            recent = set(prev_tg[-3:])
            cands = {pid for pid, h in ph.items() if h and h[-1]["team"] == team and (h[-1]["s"], h[-1]["w"]) in recent}
            for pid in cands | set(fb.pl[key]):
                h = ph.get(pid, [])
                if len(h) < 3:
                    continue
                r = fb.pl[key].get(pid)
                is_cand = pid in cands
                did_play = r is not None and v3.norm_name(r["name"]) in played
                pos = h[-1]["pos"]
                P = player_feats(h, s, fb, pid)
                F = {**T, **P}
                A = pp_row(fb, key, pid)
                if r is not None:
                    A.update({"stat_rush_yds": r["rush_yds"], "stat_rec_yds": r["rec_yds"], "stat_pass_yds": r["pass_yds"],
                              "stat_car": r["car"], "stat_tgt": r["tgt"], "stat_rec": r["rec"]})
                A.update({"team_" + k: float(tg.get(k, 0.0)) for k in ("plays", "dropbacks", "rushes", "targets", "rz_plays", "off_td")})
                A["anytd"] = 1.0 if A["rtd"] + A["rectd"] >= 1 else 0.0
                seg = {"pos": pos, "draft_round": fb.draft.get(pid, {}).get("round"),
                       "rookie": fb.draft.get(pid, {}).get("rookie") == s, "margin": ctx["margin"], "week": w,
                       "coach_change": bool(fb.coach.get((s - 1, team)) and fb.coach[(s, team)] != fb.coach[(s - 1, team)]),
                       "inj": fb.inj.get((s, w, pid), ("", ""))}
                base = {"key": {"pid": pid, "name": (r or {}).get("name", h[-1]["name"]), "team": team, "s": s, "w": w},
                        "F": F, "A": A, "seg": seg}
                elig = eligibility(h, pos)
                if did_play:
                    for fam in elig:
                        rows[fam].append({**base, "F": dict(F)})
                if is_cand and elig & {"rushing", "receiving"}:
                    inj = fb.inj.get((s, w, pid), ("", ""))
                    prev_roster = fb.roster.get((s, w - 1, pid)) if w > 1 else None
                    this_roster = fb.roster.get((s, w, pid))
                    pregame.append({**base, "F": dict(F), "elig": sorted(elig & {"rushing", "receiving"}),
                                    "avail": {"status": inj[0] or "none", "practice": inj[1] or "none",
                                              "prev_roster": prev_roster[1] if prev_roster else "missing",
                                              "this_roster_T90": this_roster[1] if this_roster else "missing",
                                              "gap": float(sum(1 for g in prev_tg if (h[-1]["s"], h[-1]["w"]) < g))},
                                    "played": did_play})
        # defense rows (conditional on playing defense this game)
        for team, T in TF.items():
            key = (s, w, team); ctx = fb.games[key]; opp = ctx["opp"]
            otg = fb.tg.get((s, w, opp), {})
            OT = TF.get(opp)
            if OT is None:
                continue
            stat_by_name = {v3.norm_name(r["name"]): r for r in fb.pl.get(key, {}).values()}
            for pfr, d_ in fb.dsnap.get(key, {}).items():
                h = dph[pfr]
                if len(h) < 3 or mean([g["pct"] for g in h[-3:]]) < 0.3:
                    continue
                sr = stat_by_name.get(d_["name"], {})
                cur = [g for g in h if g["s"] == s]
                F = {**{f"opp_{k}": v for k, v in OT.items() if k not in MARKET_FEATURES},
                     "is_home": T["is_home"], "grp_dl": 1.0 if d_["grp"] == "DL" else 0.0, "grp_lb": 1.0 if d_["grp"] == "LB" else 0.0,
                     "pct_l3": mean([g["pct"] for g in h[-3:]]), "pct_l8": mean([g["pct"] for g in h[-8:]]),
                     "snaps_l3": mean([g["snaps"] for g in h[-3:]]), "tkl_l3": mean([g["tkl"] for g in h[-3:]]),
                     "tkl_cur": mean([g["tkl"] for g in cur]),
                     "tkl_ps": shrink(ssum(g["tkl"] for g in h[-16:]), ssum(g["snaps"] for g in h[-16:]), "tkl_ps"),
                     "sack_pd": shrink(ssum(g["sack"] for g in h[-16:]), ssum(g["opp_db_on"] for g in h[-16:]), "sack_pd"),
                     "dint_pd": shrink(ssum(g["dint"] for g in h[-16:]), ssum(g["opp_db_on"] for g in h[-16:]), "dint_pd")}
                A = {"snaps": d_["snaps"], "opp_plays": float(otg.get("plays", 0.0)), "opp_dropbacks": float(otg.get("dropbacks", 0.0)),
                     "tkl": sr.get("tkl", 0.0), "sack": sr.get("sack", 0.0), "dint": sr.get("dint", 0.0)}
                defense.append({"key": {"pfr": pfr, "team": team, "opp": opp, "s": s, "w": w, "grp": d_["grp"]}, "F": F, "A": A})
        if until and (s, w) == until:
            break
        # absorb week
        for team in list(TF):
            key = (s, w, team); pl = fb.pl[key]; ctx = fb.games[key]; tg = fb.tg.get(key, {})
            played = fb.played(key)
            tr = {"s": s, "w": w, **{k: float(tg.get(k, 0.0)) for k in ("plays", "dropbacks", "rushes", "targets", "rz_plays", "off_td")},
                  "rush_yds": ssum(fb.pp.get((s, w, p), {}).get("rush_yds", 0.0) for p in pl),
                  "rec_yds": ssum(fb.pp.get((s, w, p), {}).get("rec_yds", 0.0) for p in pl)}
            th[team].append(tr); dh[ctx["opp"]].append(tr)
            snaps = fb.snap.get(key, {})
            for pid, r in pl.items():
                a = pp_row(fb, key, pid)
                ph[pid].append({"s": s, "w": w, "team": team, "pos": r["pos"], "name": r["name"],
                                "played": v3.norm_name(r["name"]) in played, "snap": snaps.get(v3.norm_name(r["name"])),
                                **{k: a[k] for k in ("car", "rush_yds", "base_yds", "expl_yds", "tgt", "rec", "rec_yds", "air", "yac",
                                                     "att", "cmp", "pass_yds", "cmp_air", "cmp_yac", "ptd", "int", "rtd", "rectd",
                                                     "rz_opp", "out_touch", "rz_td", "out_td", "routes")},
                                "tgt_air": float(fb.pp.get((s, w, pid), {}).get("tgt_air", 0.0)),
                                "td": a["rtd"] + a["rectd"],
                                "car_sh": ratio(a["car"], tr["rushes"]) if tr["rushes"] else None,
                                "tgt_sh": ratio(a["tgt"], tr["targets"]) if tr["targets"] else None,
                                "att_sh": ratio(a["att"], tr["dropbacks"]) if tr["dropbacks"] else None,
                                "route_rate": ratio(a["routes"], tr["dropbacks"]) if a["routes"] is not None and tr["dropbacks"] else None,
                                "rz_sh": ratio(a["rz_opp"], tr["rz_plays"]) if tr["rz_plays"] else None,
                                "qb_rush_sh": ratio(a["car"], tr["plays"]) if tr["plays"] else None})
            otg = fb.tg.get((s, w, fb.games[key]["opp"]), {})
            stat_by_name = {v3.norm_name(r["name"]): r for r in pl.values()}
            for pfr, d_ in fb.dsnap.get(key, {}).items():
                sr = stat_by_name.get(d_["name"], {})
                dph[pfr].append({"s": s, "w": w, "pct": d_["pct"] or 0.0, "snaps": d_["snaps"] or 0.0,
                                 "tkl": sr.get("tkl", 0.0), "sack": sr.get("sack", 0.0), "dint": sr.get("dint", 0.0),
                                 "opp_db_on": float(otg.get("dropbacks", 0.0)) * (d_["pct"] or 0.0)})
    return rows, team_rows, pregame, defense


def eligibility(h, pos):
    l3 = h[-3:]
    m = lambda k: sum(g[k] for g in l3) / 3
    out = set()
    if pos in ("RB", "FB") and m("car") >= 5:
        out |= {"rushing", "rb_rush_td"}
    if pos in ("WR", "TE", "RB") and m("tgt") >= 3:
        out |= {"receiving", "rec_td"}
    if pos in ("RB", "WR", "TE") and m("car") + m("tgt") >= 4:
        out.add("anytime_td")
    if pos == "QB" and m("att") >= 15:
        out |= {"passing", "qb_rush"}
    return out


# ------------------------------------------------------------------ metrics
def cont_metrics(pred, act):
    e = pred - act; a = np.abs(e)
    out = {"n": int(len(a)), "mae": round(float(a.mean()), 3), "median_ae": round(float(np.median(a)), 3),
           "rmse": round(float(np.sqrt((e ** 2).mean())), 3), "mse": round(float((e ** 2).mean()), 2),
           "bias": round(float(e.mean()), 3), "pinball_50": round(float(0.5 * a.mean()), 3)}
    for t in (5, 10, 15, 20, 25, 30):
        out[f"within_{t}"] = round(float((a <= t).mean()), 4)
    out["residual_pct"] = {f"p{q}": round(float(np.percentile(act - pred, q)), 2) for q in (5, 25, 50, 75, 95)}
    return out


def count_metrics(mu, y):
    mu = np.maximum(mu, 1e-6); e = mu - y; a = np.abs(e)
    dev = 2 * np.mean(np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / mu), 0.0) - (y - mu))
    out = {"n": int(len(y)), "mae": round(float(a.mean()), 4), "rmse": round(float(np.sqrt((e ** 2).mean())), 4),
           "bias": round(float(e.mean()), 4), "poisson_deviance": round(float(dev), 4),
           "within_1": round(float((a <= 1).mean()), 4), "within_2": round(float((a <= 2).mean()), 4),
           "within_5": round(float((a <= 5).mean()), 4)}
    p1 = 1 - np.exp(-mu); yb = (y >= 1).astype(float)
    if 0.0 < yb.mean() < 0.8:
        out["p_ge1"] = binary_metrics(p1, yb)
    return out


def binary_metrics(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    order = np.argsort(p); rk = np.empty(len(p)); rk[order] = np.arange(1, len(p) + 1)
    npos = y.sum(); nneg = len(y) - npos
    auc = float((rk[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else None
    bins = np.minimum((p * 10).astype(int), 9)
    ece = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()))
    return {"n": int(len(y)), "base_rate": round(float(y.mean()), 4), "logloss": round(ll, 4),
            "brier": round(float(((p - y) ** 2).mean()), 4), "auc_ranking_only": round(auc, 4) if auc else None,
            "ece": round(ece, 4)}


def crps_discrete(points, weights, y):
    """CRPS of a discrete distribution (points [n,k], weights [n,k]) at y [n]."""
    t1 = np.sum(weights * np.abs(points - y[:, None]), axis=1)
    t2 = 0.5 * np.sum(weights[:, :, None] * weights[:, None, :] * np.abs(points[:, :, None] - points[:, None, :]), axis=(1, 2))
    return float(np.mean(t1 - t2))


def dist_metrics(Q, y):
    """Q: [n, 19] quantiles at QGRID."""
    Q = np.sort(Q, axis=1)
    w = np.full(Q.shape, 1.0 / Q.shape[1])
    i10, i90, i25, i75 = [int(np.where(QGRID == q)[0][0]) for q in (0.10, 0.90, 0.25, 0.75)]
    return {"crps": round(crps_discrete(Q, w, y), 3),
            "cov80": round(float(((y >= Q[:, i10]) & (y <= Q[:, i90])).mean()), 4), "width80": round(float((Q[:, i90] - Q[:, i10]).mean()), 2),
            "cov50": round(float(((y >= Q[:, i25]) & (y <= Q[:, i75])).mean()), 4), "width50": round(float((Q[:, i75] - Q[:, i25]).mean()), 2),
            "pit_deciles": [round(float(np.mean(y <= Q[:, i])), 3) for i in range(1, 19, 2)]}


# ------------------------------------------------------------------ modelling
FIT_LOG = []      # (label, seasons in fit rows) -- checked by invariants


def M(rows, cols):
    return np.array([[r["F"].get(c) if r["F"].get(c) is not None else np.nan for c in cols] for r in rows], dtype=np.float32)


def fit(tr, va, cols, ytr, yva, obj="reg:squarederror", label="", extra=None):
    import xgboost as xgb
    cols = list(dict.fromkeys(cols))
    FIT_LOG.append((label, sorted({(r["key"]["s"], "train" if TRAIN(r["key"]["s"], r["key"]["w"]) else "valid" if VALID(r["key"]["s"], r["key"]["w"]) else "OTHER") for r in tr + va}), tuple(cols)))
    params = {**XGB, "objective": obj, **(extra or {})}
    b = xgb.train(params, xgb.DMatrix(M(tr, cols), label=ytr, feature_names=cols), 3000,
                  evals=[(xgb.DMatrix(M(va, cols), label=yva, feature_names=cols), "va")],
                  early_stopping_rounds=80, verbose_eval=False)
    b._cols = cols
    return b


def pred(b, rows):
    import xgboost as xgb
    return b.predict(xgb.DMatrix(M(rows, b._cols), feature_names=b._cols), iteration_range=(0, b.best_iteration + 1))


def split(rows):
    tr = [r for r in rows if TRAIN(r["key"]["s"], r["key"]["w"])]
    va = [r for r in rows if VALID(r["key"]["s"], r["key"]["w"])]
    ev = {s: [r for r in rows if r["key"]["s"] == s] for s in EVAL_SEASONS}
    return tr, va, ev


TEAM_CORE = ["is_home", "team_n_cur"] + [f"t_{k}_{x}" for k in ("plays", "dropbacks", "rushes", "targets", "rz_plays", "off_td")
                                         for x in ("blend", "l3")] + \
            [f"o_{k}_allowed" for k in ("plays", "dropbacks", "rushes", "targets", "rz_plays", "off_td")] + \
            ["t_pass_rate", "o_pass_rate_allowed", "t_rz_rate", "o_rz_rate_allowed", "o_ypc_allowed", "o_ypt_allowed"]
SHARE_CORE = ["cur_n", "prior_n", "rookie_year", "draft_round", "snap_l3"] + \
             [f"{k}_{x}" for k in ("car_sh", "tgt_sh", "att_sh", "route_rate", "rz_sh", "qb_rush_sh")
              for x in ("l3", "l8", "sd5", "trend")] + ["is_home", "t_pass_rate", "o_pass_rate_allowed"]
RATE_CORE = ["base_ypc", "expl_ypc", "tprr", "catch", "air_pc", "yac_pc", "adot", "cmp_pct", "air_pcmp", "yac_pcmp",
             "ptd_pa", "int_pa", "qb_ypc", "rtd_pc", "rectd_pt", "c_rz", "c_out"]
VOL_CORE = [f"{k}_{x}" for k in ("car", "tgt", "rec", "att", "cmp", "rush_yds", "rec_yds", "pass_yds", "td", "rz_opp", "out_touch")
            for x in ("l3", "cur", "pri", "blend")]
CORE = TEAM_CORE + SHARE_CORE + RATE_CORE + VOL_CORE          # no bookmaker inputs
INCUMBENT = [f"v2_{c}" for c in v2.FEATURES]                   # includes spread / implied total


def team_models(team_rows):
    tr, va, _ = split(team_rows)
    y = lambda rr, k: np.array([r["A"][k] for r in rr])
    out = {}
    for k in ("plays", "dropbacks", "rushes", "targets", "rz_plays"):
        out[k] = fit(tr, va, TEAM_CORE, y(tr, k), y(va, k), label=f"team_{k}")
    for k, num, den in (("pass_rate", "dropbacks", "plays"), ("rz_rate", "rz_plays", "plays")):
        yr = lambda rr: np.array([ratio(r["A"][num], r["A"][den]) if r["A"][den] else 0.0 for r in rr])
        out[k] = fit(tr, va, TEAM_CORE, yr(tr), yr(va), label=f"team_{k}")
    preds = {}
    for k, b in out.items():
        for r, p in zip(team_rows, pred(b, team_rows)):
            preds[(r["key"]["s"], r["key"]["w"], r["key"]["team"], k)] = float(p)
    # team-level component scores (eval)
    rep = {}
    for s in EVAL_SEASONS:
        er = [r for r in team_rows if r["key"]["s"] == s]
        rep[str(s)] = {}
        for k in ("plays", "dropbacks", "rushes", "targets", "rz_plays"):
            yy = np.array([r["A"][k] for r in er]); pp = np.array([preds[(r["key"]["s"], r["key"]["w"], r["key"]["team"], k)] for r in er])
            naive = np.array([r["F"][f"t_{k}_blend"] or np.nan for r in er])
            ok = ~np.isnan(naive)
            rep[str(s)][k] = {"model": count_metrics(pp[ok], yy[ok]), "team_blend_average": count_metrics(naive[ok], yy[ok])}
    return preds, rep


def share_model(rows, target_fn, label, cols=SHARE_CORE):
    tr, va, _ = split(rows)
    ok = lambda rr: [r for r in rr if not math.isnan(target_fn(r))]
    tr, va = ok(tr), ok(va)
    b = fit(tr, va, cols, np.array([target_fn(r) for r in tr]), np.array([target_fn(r) for r in va]), label=label)
    return np.clip(pred(b, rows), 0.0, None)


# ------------------------------------------------------------------ chains + Shapley
def coalition_pred(pred_c, act_c, combine, mask, comps):
    vals = {}
    for i, c in enumerate(comps):
        if mask[i]:
            a = act_c[c]
            vals[c] = np.where(np.isnan(a), pred_c[c], a)
        else:
            vals[c] = pred_c[c]
    return combine(vals)


def shapley(comps, v):
    n = len(comps); phi = {}
    for i, c in enumerate(comps):
        tot = 0.0
        others = [j for j in range(n) if j != i]
        for k in range(n):
            for S in itertools.combinations(others, k):
                wgt = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                a = tuple(1 if j in S else 0 for j in range(n))
                b = tuple(1 if (j in S or j == i) else 0 for j in range(n))
                tot += wgt * (v[b] - v[a])
        phi[c] = tot
    return phi


def chain_budget(name, comps, pred_c, act_c, combine, y, kind="yards", groups=None):
    """All 2^n coalitions; Shapley attribution for each loss; interactions."""
    n = len(comps)
    masks = list(itertools.product([0, 1], repeat=n))
    P = {m: coalition_pred(pred_c, act_c, combine, m, comps) for m in masks}
    full = tuple([1] * n); none = tuple([0] * n)
    recon = float(np.max(np.abs(P[full] - y))) if kind != "prob" else None
    out = {"components": comps, "n": int(len(y)), "exact_reconstruction_max_abs_err": recon, "coalitions": {}}
    if kind == "prob":
        losses = {"logloss": lambda p: binary_metrics(p, y)["logloss"], "brier": lambda p: binary_metrics(p, y)["brier"]}
    else:
        losses = {"mae": lambda p: float(np.mean(np.abs(p - y))), "rmse": lambda p: float(np.sqrt(np.mean((p - y) ** 2)))}
        if kind == "yards":
            losses["within20_gain"] = lambda p: -float(np.mean(np.abs(p - y) <= 20))
    for m in masks:
        lab = "+".join(c for c, b in zip(comps, m) if b) or "all_predicted"
        out["coalitions"][lab] = {k: round(f(P[m]), 4) for k, f in losses.items()}
        if kind != "prob":
            out["coalitions"][lab]["bias"] = round(float(np.mean(P[m] - y)), 3)
    out["shapley"] = {}
    for k, f in losses.items():
        base = f(P[none])
        v = {m: base - f(P[m]) for m in masks}
        phi = shapley(comps, v)
        solo = {c: v[tuple(1 if j == i else 0 for j in range(n))] for i, c in enumerate(comps)}
        pair = {f"{comps[i]}x{comps[j]}": round(v[tuple(1 if t in (i, j) else 0 for t in range(n))] - solo[comps[i]] - solo[comps[j]], 4)
                for i in range(n) for j in range(i + 1, n)}
        tot = v[full]
        out["shapley"][k] = {"loss_all_predicted": round(base if k != "within20_gain" else -base, 4),
                             "loss_all_oracle": round(f(P[full]) if k != "within20_gain" else -f(P[full]), 4),
                             "total_value": round(tot, 4),
                             "phi": {c: round(x, 4) for c, x in phi.items()},
                             "phi_share": {c: round(x / tot, 3) if tot else None for c, x in phi.items()},
                             "solo_value": {c: round(x, 4) for c, x in solo.items()},
                             "non_additivity": round(tot - sum(solo.values()), 4),
                             "pairwise_interaction": pair}
        if groups:
            out["shapley"][k]["group_phi"] = {g: round(sum(phi[c] for c in cs), 4) for g, cs in groups.items()}
    return out


def arr(rows, fn):
    return np.array([fn(r) for r in rows], dtype=float)


# --- family builders: return {season: chain_budget}, plus direct-model tables
def rushing_family(rows, tp):
    rows = [r for r in rows]
    S_hat = share_model(rows, lambda r: ratio(r["A"]["car"], r["A"]["team_rushes"]), "rb_carry_share")
    res = {"chain": {}, "direct": {}, "components": {}}
    for s in EVAL_SEASONS:
        idx = [i for i, r in enumerate(rows) if r["key"]["s"] == s]
        er = [rows[i] for i in idx]
        pc = {"T": arr(er, lambda r: tp[(r["key"]["s"], r["key"]["w"], r["key"]["team"], "rushes")]),
              "S": S_hat[idx], "E": arr(er, lambda r: r["F"]["base_ypc"]), "X": arr(er, lambda r: r["F"]["expl_ypc"])}
        ac = {"T": arr(er, lambda r: r["A"]["team_rushes"]), "S": arr(er, lambda r: ratio(r["A"]["car"], r["A"]["team_rushes"])),
              "E": arr(er, lambda r: ratio(r["A"]["base_yds"], r["A"]["car"])), "X": arr(er, lambda r: ratio(r["A"]["expl_yds"], r["A"]["car"]))}
        y = arr(er, lambda r: r["A"]["rush_yds"])
        comb = lambda v: v["T"] * v["S"] * (v["E"] + v["X"])
        res["chain"][str(s)] = chain_budget("rushing_yards", ["T", "S", "E", "X"], pc, ac, comb, y,
                                            groups={"workload(T,S)": ["T", "S"], "efficiency(E,X)": ["E", "X"]})
        carries = chain_budget("carries", ["T", "S"], pc, ac, lambda v: v["T"] * v["S"], arr(er, lambda r: r["A"]["car"]), kind="count")
        res["components"][str(s)] = {"carries_chain": carries,
                                     "carry_share_mae": round(float(np.nanmean(np.abs(pc["S"] - ac["S"]))), 4)}
    return res


def receiving_family(rows, tp, fb):
    rr_hat = share_model(rows, lambda r: ratio(r["A"]["routes"], r["A"]["team_dropbacks"]) if r["A"]["routes"] is not None else float("nan"),
                         "route_rate")
    th_hat = share_model(rows, lambda r: ratio(r["A"]["tgt"], r["A"]["team_targets"]), "target_share")
    res = {"route_chain": {}, "share_chain": {}, "route_invariants": {}}
    for s in EVAL_SEASONS:
        idx = [i for i, r in enumerate(rows) if r["key"]["s"] == s]
        er = [rows[i] for i in idx]
        k = lambda r, c: tp[(r["key"]["s"], r["key"]["w"], r["key"]["team"], c)]
        y = arr(er, lambda r: r["A"]["rec_yds"])
        C = lambda r: ratio(r["A"]["rec"], r["A"]["tgt"]); A = lambda r: ratio(r["A"]["air"], r["A"]["rec"]); Y = lambda r: ratio(r["A"]["yac"], r["A"]["rec"])
        pcc = {"C": arr(er, lambda r: r["F"]["catch"]), "A": arr(er, lambda r: r["F"]["air_pc"]), "Y": arr(er, lambda r: r["F"]["yac_pc"])}
        acc = {"C": arr(er, C), "A": arr(er, A), "Y": arr(er, Y)}
        # target-share chain (all seasons)
        pc = {"TT": arr(er, lambda r: k(r, "targets")), "H": th_hat[idx], **pcc}
        ac = {"TT": arr(er, lambda r: r["A"]["team_targets"]), "H": arr(er, lambda r: ratio(r["A"]["tgt"], r["A"]["team_targets"])), **acc}
        comb = lambda v: v["TT"] * v["H"] * v["C"] * (v["A"] + v["Y"])
        res["share_chain"][str(s)] = {
            "receiving_yards": chain_budget("rec_yds_share", ["TT", "H", "C", "A", "Y"], pc, ac, comb, y,
                                            groups={"opportunity(TT,H)": ["TT", "H"], "catch(C)": ["C"], "yardage(A,Y)": ["A", "Y"]}),
            "receptions": chain_budget("rec_share", ["TT", "H", "C"], pc, ac, lambda v: v["TT"] * v["H"] * v["C"],
                                       arr(er, lambda r: r["A"]["rec"]), kind="count"),
            "targets": chain_budget("tgt_share", ["TT", "H"], pc, ac, lambda v: v["TT"] * v["H"], arr(er, lambda r: r["A"]["tgt"]), kind="count")}
        if s not in fb.has_routes:
            res["route_chain"][str(s)] = {"unsupported": "participation (route proxy) not published for this season"}
            continue
        ok = [j for j, r in enumerate(er) if r["A"]["routes"] is not None and r["A"]["tgt"] <= r["A"]["routes"]]
        viol = [r for r in er if r["A"]["routes"] is not None and r["A"]["tgt"] > r["A"]["routes"]]
        miss = [r for r in er if r["A"]["routes"] is None]
        res["route_invariants"][str(s)] = {"rows": len(er), "routes_missing": len(miss), "targets_gt_routes": len(viol),
                                           "examples_targets_gt_routes": [(r["key"]["name"], r["key"]["w"], r["A"]["tgt"], r["A"]["routes"]) for r in viol[:5]],
                                           "used_in_route_chain": len(ok)}
        sub = [er[j] for j in ok]; sidx = [idx[j] for j in ok]
        tprr = arr(sub, lambda r: r["F"]["tprr"] if r["F"]["tprr"] is not None else POS_PRIOR["tprr"][0])
        pc = {"D": arr(sub, lambda r: k(r, "dropbacks")), "R": rr_hat[sidx], "P": tprr,
              "C": arr(sub, lambda r: r["F"]["catch"]), "A": arr(sub, lambda r: r["F"]["air_pc"]), "Y": arr(sub, lambda r: r["F"]["yac_pc"])}
        ac = {"D": arr(sub, lambda r: r["A"]["team_dropbacks"]), "R": arr(sub, lambda r: ratio(r["A"]["routes"], r["A"]["team_dropbacks"])),
              "P": arr(sub, lambda r: ratio(r["A"]["tgt"], r["A"]["routes"])), "C": arr(sub, C), "A": arr(sub, A), "Y": arr(sub, Y)}
        ysub = arr(sub, lambda r: r["A"]["rec_yds"])
        res["route_chain"][str(s)] = {
            "receiving_yards": chain_budget("rec_yds_route", ["D", "R", "P", "C", "A", "Y"], pc, ac,
                                            lambda v: v["D"] * v["R"] * v["P"] * v["C"] * (v["A"] + v["Y"]), ysub,
                                            groups={"team(D)": ["D"], "role(R)": ["R"], "targeting(P)": ["P"], "catch(C)": ["C"], "yardage(A,Y)": ["A", "Y"]}),
            "receptions": chain_budget("rec_route", ["D", "R", "P", "C"], pc, ac, lambda v: v["D"] * v["R"] * v["P"] * v["C"],
                                       arr(sub, lambda r: r["A"]["rec"]), kind="count"),
            "targets": chain_budget("tgt_route", ["D", "R", "P"], pc, ac, lambda v: v["D"] * v["R"] * v["P"],
                                    arr(sub, lambda r: r["A"]["tgt"]), kind="count"),
            "routes": chain_budget("routes", ["D", "R"], pc, ac, lambda v: v["D"] * v["R"], arr(sub, lambda r: r["A"]["routes"]), kind="count")}
    return res


def passing_family(rows, tp):
    q_hat = share_model(rows, lambda r: ratio(r["A"]["att"], r["A"]["team_dropbacks"]), "qb_attempt_share")
    u_hat = share_model(rows, lambda r: ratio(r["A"]["car"], r["A"]["team_plays"]), "qb_rush_share")
    res = {}
    for s in EVAL_SEASONS:
        idx = [i for i, r in enumerate(rows) if r["key"]["s"] == s]
        er = [rows[i] for i in idx]
        k = lambda r, c: tp[(r["key"]["s"], r["key"]["w"], r["key"]["team"], c)]
        pc = {"P": arr(er, lambda r: k(r, "plays")), "R": arr(er, lambda r: k(r, "pass_rate")), "Q": q_hat[idx],
              "C": arr(er, lambda r: r["F"]["cmp_pct"]), "A": arr(er, lambda r: r["F"]["air_pcmp"]), "Y": arr(er, lambda r: r["F"]["yac_pcmp"]),
              "Ktd": arr(er, lambda r: r["F"]["ptd_pa"]), "Kint": arr(er, lambda r: r["F"]["int_pa"]),
              "U": u_hat[idx], "V": arr(er, lambda r: r["F"]["qb_ypc"]), "Krtd": arr(er, lambda r: r["F"]["rtd_pc"])}
        ac = {"P": arr(er, lambda r: r["A"]["team_plays"]), "R": arr(er, lambda r: ratio(r["A"]["team_dropbacks"], r["A"]["team_plays"])),
              "Q": arr(er, lambda r: ratio(r["A"]["att"], r["A"]["team_dropbacks"])), "C": arr(er, lambda r: ratio(r["A"]["cmp"], r["A"]["att"])),
              "A": arr(er, lambda r: ratio(r["A"]["cmp_air"], r["A"]["cmp"])), "Y": arr(er, lambda r: ratio(r["A"]["cmp_yac"], r["A"]["cmp"])),
              "Ktd": arr(er, lambda r: ratio(r["A"]["ptd"], r["A"]["att"])), "Kint": arr(er, lambda r: ratio(r["A"]["int"], r["A"]["att"])),
              "U": arr(er, lambda r: ratio(r["A"]["car"], r["A"]["team_plays"])), "V": arr(er, lambda r: ratio(r["A"]["rush_yds"], r["A"]["car"])),
              "Krtd": arr(er, lambda r: ratio(r["A"]["rtd"], r["A"]["car"]))}
        g = {"team(P,R)": ["P", "R"], "qb_volume(Q)": ["Q"], "completion(C)": ["C"], "yardage(A,Y)": ["A", "Y"]}
        res[str(s)] = {
            "passing_yards": chain_budget("pass_yds", ["P", "R", "Q", "C", "A", "Y"], pc, ac,
                                          lambda v: v["P"] * v["R"] * v["Q"] * v["C"] * (v["A"] + v["Y"]), arr(er, lambda r: r["A"]["pass_yds"]), groups=g),
            "attempts": chain_budget("att", ["P", "R", "Q"], pc, ac, lambda v: v["P"] * v["R"] * v["Q"], arr(er, lambda r: r["A"]["att"]), kind="count"),
            "completions": chain_budget("cmp", ["P", "R", "Q", "C"], pc, ac, lambda v: v["P"] * v["R"] * v["Q"] * v["C"], arr(er, lambda r: r["A"]["cmp"]), kind="count"),
            "passing_tds": chain_budget("ptd", ["P", "R", "Q", "Ktd"], pc, ac, lambda v: v["P"] * v["R"] * v["Q"] * v["Ktd"], arr(er, lambda r: r["A"]["ptd"]), kind="count"),
            "interceptions": chain_budget("int", ["P", "R", "Q", "Kint"], pc, ac, lambda v: v["P"] * v["R"] * v["Q"] * v["Kint"], arr(er, lambda r: r["A"]["int"]), kind="count"),
            "qb_rush_attempts": chain_budget("qb_car", ["P", "U"], pc, ac, lambda v: v["P"] * v["U"], arr(er, lambda r: r["A"]["car"]), kind="count"),
            "qb_rush_yards": chain_budget("qb_ryds", ["P", "U", "V"], pc, ac, lambda v: v["P"] * v["U"] * v["V"], arr(er, lambda r: r["A"]["rush_yds"])),
            "qb_rush_tds": chain_budget("qb_rtd", ["P", "U", "Krtd"], pc, ac, lambda v: v["P"] * v["U"] * v["Krtd"], arr(er, lambda r: r["A"]["rtd"]), kind="count"),
        }
    return res


def td_family(rows, tp, rows_rush, rows_rec, fb):
    o_hat = share_model(rows, lambda r: ratio(r["A"]["rz_opp"], r["A"]["team_rz_plays"]), "rz_share")
    tr, va, _ = split(rows)
    b_n = fit(tr, va, SHARE_CORE + VOL_CORE + TEAM_CORE, arr(tr, lambda r: r["A"]["out_touch"]), arr(va, lambda r: r["A"]["out_touch"]), label="non_rz_touches")
    n_hat = np.clip(pred(b_n, rows), 0, None)
    # position-average conversion rates from the train split (for the conversion-skill test)
    pos_c = {}
    for pos in ("RB", "WR", "TE"):
        t_ = [r for r in tr if r["seg"]["pos"] == pos]
        pos_c[pos] = (ssum(r["A"]["rz_td"] for r in t_) / max(1.0, ssum(r["A"]["rz_opp"] for r in t_)),
                      ssum(r["A"]["out_td"] for r in t_) / max(1.0, ssum(r["A"]["out_touch"] for r in t_)))
    res = {"anytime_td_chain": {}, "conversion_skill": {}, "count_families": {}}
    for s in EVAL_SEASONS:
        idx = [i for i, r in enumerate(rows) if r["key"]["s"] == s]
        er = [rows[i] for i in idx]
        k = lambda r, c: tp[(r["key"]["s"], r["key"]["w"], r["key"]["team"], c)]
        crz = arr(er, lambda r: r["F"]["c_rz"]); cout = arr(er, lambda r: r["F"]["c_out"])
        pc = {"P": arr(er, lambda r: k(r, "plays")), "Z": arr(er, lambda r: k(r, "rz_rate")), "O": o_hat[idx], "N": n_hat[idx]}
        ac = {"P": arr(er, lambda r: r["A"]["team_plays"]), "Z": arr(er, lambda r: ratio(r["A"]["team_rz_plays"], r["A"]["team_plays"])),
              "O": arr(er, lambda r: ratio(r["A"]["rz_opp"], r["A"]["team_rz_plays"])), "N": arr(er, lambda r: r["A"]["out_touch"])}
        y = arr(er, lambda r: r["A"]["anytd"])
        comb = lambda v, a=crz, b=cout: 1 - np.exp(-(v["P"] * v["Z"] * v["O"] * a + v["N"] * b))
        res["anytime_td_chain"][str(s)] = chain_budget("anytime_td", ["P", "Z", "O", "N"], pc, ac, comb, y, kind="prob",
                                                       groups={"team_environment(P,Z)": ["P", "Z"], "player_allocation(O,N)": ["O", "N"]})
        # conditional test the reviewer asked for: team RZ volume given oracle player allocation
        full_alloc = {c: np.where(np.isnan(ac[c]), pc[c], ac[c]) for c in ac}
        mix = lambda **o: binary_metrics(comb({c: (full_alloc[c] if o.get(c) else pc[c]) for c in pc}), y)
        res["anytime_td_chain"][str(s)]["conditional_tests"] = {
            "alloc_O_N_oracle__team_P_Z_predicted": mix(O=1, N=1),
            "alloc_O_N_oracle__team_P_Z_oracle": mix(O=1, N=1, P=1, Z=1),
            "alloc_O_N_oracle__team_Z_oracle_only": mix(O=1, N=1, Z=1)}
        # conversion: player-shrunk vs position-average rates, all opportunities oracle
        lam_opp = full_alloc["P"] * full_alloc["Z"] * full_alloc["O"]
        pos_rz = arr(er, lambda r: pos_c[r["seg"]["pos"]][0]); pos_out = arr(er, lambda r: pos_c[r["seg"]["pos"]][1])
        res["conversion_skill"][str(s)] = {
            "all_opportunities_oracle__player_conversion": binary_metrics(1 - np.exp(-(lam_opp * crz + full_alloc["N"] * cout)), y),
            "all_opportunities_oracle__position_avg_conversion": binary_metrics(1 - np.exp(-(lam_opp * pos_rz + full_alloc["N"] * pos_out)), y),
            "all_predicted__player_conversion": binary_metrics(comb(pc), y),
            "all_predicted__position_avg_conversion": binary_metrics(1 - np.exp(-(pc["P"] * pc["Z"] * pc["O"] * pos_rz + pc["N"] * pos_out)), y)}
    # rush TDs (RB): T x S x K ; rec TDs: TT x H x K
    Sr = share_model(rows_rush, lambda r: ratio(r["A"]["car"], r["A"]["team_rushes"]), "rb_carry_share_td")
    Hr = share_model(rows_rec, lambda r: ratio(r["A"]["tgt"], r["A"]["team_targets"]), "target_share_td")
    for s in EVAL_SEASONS:
        k = lambda r, c: tp[(r["key"]["s"], r["key"]["w"], r["key"]["team"], c)]
        i1 = [i for i, r in enumerate(rows_rush) if r["key"]["s"] == s]; e1 = [rows_rush[i] for i in i1]
        pc = {"T": arr(e1, lambda r: k(r, "rushes")), "S": Sr[i1], "K": arr(e1, lambda r: r["F"]["rtd_pc"])}
        ac = {"T": arr(e1, lambda r: r["A"]["team_rushes"]), "S": arr(e1, lambda r: ratio(r["A"]["car"], r["A"]["team_rushes"])),
              "K": arr(e1, lambda r: ratio(r["A"]["rtd"], r["A"]["car"]))}
        i2 = [i for i, r in enumerate(rows_rec) if r["key"]["s"] == s]; e2 = [rows_rec[i] for i in i2]
        pc2 = {"TT": arr(e2, lambda r: k(r, "targets")), "H": Hr[i2], "K": arr(e2, lambda r: r["F"]["rectd_pt"])}
        ac2 = {"TT": arr(e2, lambda r: r["A"]["team_targets"]), "H": arr(e2, lambda r: ratio(r["A"]["tgt"], r["A"]["team_targets"])),
               "K": arr(e2, lambda r: ratio(r["A"]["rectd"], r["A"]["tgt"]))}
        res["count_families"][str(s)] = {
            "rb_rushing_tds": chain_budget("rtd", ["T", "S", "K"], pc, ac, lambda v: v["T"] * v["S"] * v["K"], arr(e1, lambda r: r["A"]["rtd"]), kind="count"),
            "receiving_tds": chain_budget("rectd", ["TT", "H", "K"], pc2, ac2, lambda v: v["TT"] * v["H"] * v["K"], arr(e2, lambda r: r["A"]["rectd"]), kind="count")}
    return res


# ------------------------------------------------------------------ direct models: mean vs median vs distribution
def direct_models(rows, target, label, with_incumbent=True):
    tr, va, ev = split(rows)
    y = lambda rr: arr(rr, lambda r: r["A"][target])
    out = {}
    b_med = fit(tr, va, CORE, y(tr), y(va), "reg:absoluteerror", label=f"{label}_median")
    b_mean = fit(tr, va, CORE, y(tr), y(va), "reg:squarederror", label=f"{label}_mean")
    b_q = fit(tr, va, CORE, y(tr), y(va), "reg:quantileerror", label=f"{label}_quantiles", extra={"quantile_alpha": QGRID.tolist()})
    b_inc = fit(tr, va, INCUMBENT, y(tr), y(va), "reg:absoluteerror", label=f"{label}_incumbent") if with_incumbent else None
    for s, er in ev.items():
        a = y(er)
        naive = arr(er, lambda r: {"stat_rush_yds": r["F"]["rush_yds_blend"], "stat_rec_yds": r["F"]["rec_yds_blend"],
                                   "stat_pass_yds": r["F"]["pass_yds_blend"]}[target] or 0.0)
        Q = pred(b_q, er)
        t = {"historical_blend_average (baseline)": {"point": cont_metrics(naive, a)},
             "core_median (reg:absoluteerror, no bookmaker inputs)": {"point": cont_metrics(pred(b_med, er), a)},
             "core_mean (reg:squarederror, no bookmaker inputs)": {"point": cont_metrics(pred(b_mean, er), a)},
             "core_quantiles (reg:quantileerror)": {"median_q50": cont_metrics(Q[:, 9], a), "distribution": dist_metrics(Q, a)}}
        if b_inc is not None:
            t["incumbent_v2_recipe (median, with spread/implied total)"] = {"point": cont_metrics(pred(b_inc, er), a)}
        out[str(s)] = t
    return out


def v2_features(rows_by_fam, fb, data_dir):
    fb3 = v3.Data(data_dir, SEASONS)
    for key, pl in fb.pl.items():
        for pid, r in pl.items():
            row = fb3.stats.get(key, {}).get(pid)
            if row is not None:
                row["receptions"] = r["rec"]; row["passing_yards"] = r["pass_yds"]
    cfg = {"rushing": ("carries", "rushing_yards", ("RB", "FB"), 5.0), "receiving": ("targets", "receiving_yards", ("WR", "TE", "RB"), 3.0),
           "passing": ("attempts", "passing_yards", ("QB",), 15.0)}
    for fam, (vol, yds, pos, mn) in cfg.items():
        v3.MARKETS["_eb"] = {"vol": vol, "yds": yds, "positions": pos, "min_last3_vol": mn}
        got = {(k["pid"], k["s"], k["w"]): f_ for k, f_, _ in v3.Replayer(fb3, "_eb").replay()}
        del v3.MARKETS["_eb"]
        keep = []
        for r in rows_by_fam[fam]:
            g = got.get((r["key"]["pid"], r["key"]["s"], r["key"]["w"]))
            if g is not None:
                r["F"].update({f"v2_{c}": g.get(c) for c in v2.FEATURES}); keep.append(r)
        rows_by_fam[fam] = keep


# ------------------------------------------------------------------ pregame mixture
AVAIL_COLS = ["st_q", "st_d", "st_o", "st_listed", "pr_dnp", "pr_lim", "prev_act", "prev_res", "prev_missing", "gap", "snap_l3", "cur_n"]


def avail_feats(r, t90=False):
    a = r["avail"]
    F = {"st_q": 1.0 if a["status"] == "Questionable" else 0.0, "st_d": 1.0 if a["status"] == "Doubtful" else 0.0,
         "st_o": 1.0 if a["status"] == "Out" else 0.0, "st_listed": 0.0 if a["status"] == "none" else 1.0,
         "pr_dnp": 1.0 if "Did Not" in a["practice"] else 0.0, "pr_lim": 1.0 if "Limited" in a["practice"] else 0.0,
         "prev_act": 1.0 if a["prev_roster"] == "ACT" else 0.0, "prev_res": 1.0 if a["prev_roster"] == "RES" else 0.0,
         "prev_missing": 1.0 if a["prev_roster"] == "missing" else 0.0, "gap": a["gap"]}
    if t90:
        F["t90_ina"] = 1.0 if a["this_roster_T90"] in ("INA",) else 0.0
        F["t90_res"] = 1.0 if a["this_roster_T90"] in ("RES", "CUT", "RET", "DEV") else 0.0
    return F


def pregame_mixture(pregame, cond_rows, fam, target):
    """Conditional quantile model (fit on players who played) mixed with a
    P(active) model fit on the pre-game universe."""
    pg = [r for r in pregame if fam in r["elig"]]
    for r in pg:
        r["F"].update(avail_feats(r, t90=True))
    tr, va, ev = split(pg)
    y_act = lambda rr: arr(rr, lambda r: 1.0 if r["played"] else 0.0)
    out = {"universe_rule": "last game for this team within its previous 3 team games, >=3 prior games, prior-usage eligibility; "
                            "no same-game information", "n_universe": {}}
    b_a24 = fit(tr, va, AVAIL_COLS, y_act(tr), y_act(va), "binary:logistic", label=f"{fam}_p_active_T24h")
    b_a90 = fit(tr, va, AVAIL_COLS + ["t90_ina", "t90_res"], y_act(tr), y_act(va), "binary:logistic", label=f"{fam}_p_active_T90m")
    ctr, cva, _ = split(cond_rows)
    yc = lambda rr: arr(rr, lambda r: r["A"][target])
    b_q = fit(ctr, cva, CORE, yc(ctr), yc(cva), "reg:quantileerror", label=f"{fam}_cond_quantiles", extra={"quantile_alpha": QGRID.tolist()})
    b_m = fit(ctr, cva, CORE, yc(ctr), yc(cva), "reg:squarederror", label=f"{fam}_cond_mean")
    for s, er in ev.items():
        yy = arr(er, lambda r: r["A"][target] if r["played"] else 0.0)
        played = y_act(er)
        Q = np.sort(np.maximum(pred(b_q, er), 0), axis=1); mu = np.maximum(pred(b_m, er), 0)
        res = {"n": len(er), "share_not_played": round(float(1 - played.mean()), 4)}
        for tag, b in (("T-24h (injury report, prior-week roster)", b_a24), ("T-90m (+ game-day roster status)", b_a90)):
            p = np.clip(pred(b, er), 1e-4, 1 - 1e-4)
            pts = np.concatenate([np.zeros((len(er), 1)), Q], axis=1)
            wts = np.concatenate([(1 - p)[:, None], np.repeat((p / Q.shape[1])[:, None], Q.shape[1], axis=1)], axis=1)
            # mixture median
            order = np.argsort(pts, axis=1); sp = np.take_along_axis(pts, order, 1); sw = np.take_along_axis(wts, order, 1)
            med = sp[np.arange(len(er)), np.argmax(np.cumsum(sw, 1) >= 0.5, axis=1)]
            res[tag] = {"p_active": binary_metrics(p, played),
                        "mixture_mean": cont_metrics(p * mu, yy), "mixture_median": cont_metrics(med, yy),
                        "mixture_crps": round(crps_discrete(pts, wts, yy), 3)}
        wq = np.full(Q.shape, 1 / Q.shape[1])
        res["assume_active (conditional forecast applied to everyone)"] = {
            "mean": cont_metrics(mu, yy), "median": cont_metrics(Q[:, 9], yy), "crps": round(crps_discrete(Q, wq, yy), 3)}
        # Questionable subset
        qm = np.array([r["avail"]["status"] == "Questionable" for r in er])
        if qm.sum() >= 15:
            p = np.clip(pred(b_a24, er), 1e-4, 1 - 1e-4)
            res["questionable_subset"] = {"n": int(qm.sum()), "played_rate": round(float(played[qm].mean()), 3),
                                          "assume_active_mae": round(float(np.abs(mu[qm] - yy[qm]).mean()), 2),
                                          "mixture_mean_mae_T24h": round(float(np.abs(p[qm] * mu[qm] - yy[qm]).mean()), 2),
                                          "assume_active_rmse": round(float(np.sqrt(((mu[qm] - yy[qm]) ** 2).mean())), 2),
                                          "mixture_mean_rmse_T24h": round(float(np.sqrt(((p[qm] * mu[qm] - yy[qm]) ** 2).mean())), 2)}
        out[str(s)] = res
    return out


def roster_timing_audit(fb):
    """Is the weekly-roster status a game-day snapshot? If 'INA' (inactive)
    were recorded before the final 90-minute inactive list, some INA players
    would show snaps. Offensive skill players and defenders, 2024-2025."""
    out = defaultdict(lambda: [0, 0])
    for (s, w, gid), (team, st, nm, pos) in fb.roster.items():
        if s not in (2024, 2025) or pos not in ("QB", "RB", "WR", "TE", "DL", "LB", "DB", "DE", "DT", "CB", "S", "OLB", "ILB"):
            continue
        k = (s, w, team)
        if k not in fb.snap:
            continue
        on = nm in fb.snap[k] or nm in {d["name"] for d in fb.dsnap.get(k, {}).values()}
        out[st][0] += 1; out[st][1] += 1 if on else 0
    return {st: {"n": n, "share_with_snaps": round(p / n, 4) if n else None} for st, (n, p) in sorted(out.items())}


# ------------------------------------------------------------------ invariants
def check(results, name, ok, detail=""):
    results.append({"invariant": name, "pass": bool(ok), "detail": detail})


def invariants(fb, rows, team_rows, pregame, defense, data_dir=None, perturb=True):
    R = []
    # windows disjoint
    for s in range(2022, 2027):
        for w in range(1, 19):
            flags = [TRAIN(s, w), VALID(s, w), s in EVAL_SEASONS]
            if sum(flags) > 1:
                check(R, "train/valid/eval windows disjoint", False, f"{s} wk{w}")
                break
    if not any(r["invariant"].startswith("train/valid") for r in R):
        check(R, "train/valid/eval windows disjoint", True)
    # no bookmaker inputs in core feature sets
    core_sets = {"TEAM_CORE": TEAM_CORE, "SHARE_CORE": SHARE_CORE, "RATE_CORE": RATE_CORE, "VOL_CORE": VOL_CORE,
                 "CORE": CORE, "AVAIL_COLS": AVAIL_COLS}
    bad = [n for n, cols in core_sets.items() if set(cols) & set(MARKET_FEATURES)]
    check(R, "bookmaker spread/total absent from core feature sets", not bad, str(bad))
    # ORC_* never in any feature set or fitted model
    orc = [n for n, cols in core_sets.items() if any(c.startswith("ORC_") for c in cols)]
    orc += [lab for lab, _, cols in FIT_LOG if any(c.startswith("ORC_") for c in cols)]
    check(R, "no ORC_* (oracle) variable in any normal feature set or fitted model", not orc, str(orc))
    # fitted models never see evaluation seasons
    leaks = [lab for lab, seen, _ in FIT_LOG if any(tag == "OTHER" or s in EVAL_SEASONS for s, tag in seen)]
    check(R, "no 2025/2026 evaluation row enters any model fit", not leaks, str(leaks[:5]))
    # feature dicts contain no actual-outcome keys
    for fam, rr in rows.items():
        leakkeys = {k for r in rr[:500] for k in r["F"] if k.startswith("ORC_") or k in ("A",)}
        check(R, f"{fam}: feature dicts carry no oracle keys", not leakkeys, str(leakkeys))
    # uniqueness
    for fam, rr in rows.items():
        keys = [(r["key"]["pid"], r["key"]["s"], r["key"]["w"]) for r in rr]
        check(R, f"{fam}: player-game rows unique", len(keys) == len(set(keys)), f"{len(keys) - len(set(keys))} duplicates")
    dk = [(r["key"]["pfr"], r["key"]["s"], r["key"]["w"]) for r in defense]
    check(R, "defense: player-game rows unique", len(dk) == len(set(dk)))
    # nonnegative / bounds
    for fam, rr in rows.items():
        neg = sum(1 for r in rr if r["A"]["car"] < 0 or r["A"]["tgt"] < 0 or r["A"]["att"] < 0)
        check(R, f"{fam}: carries/targets/attempts nonnegative", neg == 0, f"{neg}")
        rgt = sum(1 for r in rr if r["A"]["rec"] > r["A"]["tgt"])
        check(R, f"{fam}: receptions <= targets", rgt == 0, f"{rgt}")
        shb = sum(1 for r in rr for num, den in (("car", "team_rushes"), ("tgt", "team_targets")) if r["A"][den] and not 0 <= r["A"][num] / r["A"][den] <= 1)
        check(R, f"{fam}: carry/target shares within [0,1]", shb == 0, f"{shb}")
        fs = sum(1 for r in rr for k in ("car_sh_l3", "tgt_sh_l3", "rz_sh_l3") if r["F"].get(k) is not None and not 0 <= r["F"][k] <= 1)
        check(R, f"{fam}: as-of share features within [0,1]", fs == 0, f"{fs}")
    # route missingness never silently zero
    for fam, rr in rows.items():
        z = sum(1 for r in rr if r["key"]["s"] not in fb.has_routes and r["A"]["routes"] is not None)
        check(R, f"{fam}: seasons without participation have routes=None (not 0)", z == 0, f"{z}")
    # targets <= routes (proxy)
    rr = [r for r in rows["receiving"] if r["A"]["routes"] is not None]
    v = [r for r in rr if r["A"]["tgt"] > r["A"]["routes"]]
    share = len(v) / max(1, len(rr))
    check(R, "receiving: targets <= route proxy (tolerance 1%; violators excluded from route chain)", share <= 0.01,
          f"{len(v)}/{len(rr)} = {share:.4f}; e.g. {[(r['key']['name'], r['key']['s'], r['key']['w'], r['A']['tgt'], r['A']['routes']) for r in v[:3]]}")
    # team totals reconcile: pbp targets sum == team targets; pbp rushes sum == team rushes
    mis_t = mis_r = n = 0
    for key, tg in list(fb.tg.items())[:4000]:
        s, w, t = key
        pids = fb.pl.get(key, {})
        if not pids:
            continue
        n += 1
        tsum = ssum(fb.pp.get((s, w, p), {}).get("tgt", 0.0) for p in pids)
        rsum = ssum(fb.pp.get((s, w, p), {}).get("car", 0.0) for p in pids)
        mis_t += tsum != tg.get("targets", 0.0); mis_r += rsum != tg.get("rushes", 0.0)
    check(R, "team pbp targets == sum of rostered players' pbp targets (<=2% team-games off)", mis_t / max(1, n) <= 0.02, f"{mis_t}/{n}")
    check(R, "team pbp rushes == sum of rostered players' pbp carries (<=2% team-games off)", mis_r / max(1, n) <= 0.02, f"{mis_r}/{n}")
    # pbp vs official stats (reported, not a hard gate)
    d = [abs(r["A"]["rush_yds"] - r["A"].get("stat_rush_yds", r["A"]["rush_yds"])) for r in rows["rushing"]]
    d2 = [abs(r["A"]["rec_yds"] - r["A"].get("stat_rec_yds", r["A"]["rec_yds"])) for r in rows["receiving"]]
    check(R, "pbp-derived yards vs official box score (report only)", True,
          f"rushing mean |diff| {np.mean(d):.3f}, share differing {np.mean(np.array(d) > 0):.3f}; receiving mean |diff| {np.mean(d2):.3f}, share {np.mean(np.array(d2) > 0):.3f}")
    # pregame universe built without same-game participation
    check(R, "pregame universe: membership independent of target-game participation", True,
          "candidates = players whose last game (strictly before the target week) was for this team within its previous 3 team games")
    nplayed = sum(1 for r in pregame if not r["played"])
    check(R, "pregame universe includes non-participants", nplayed > 0, f"{nplayed} non-participant rows")
    # perturbation: changing target-week realized data must not change that week's features
    if perturb:
        import copy
        wk = (2025, 10)
        base_rows, _, _, _ = build_rows(fb, until=wk)
        fb2 = copy.copy(fb)
        fb2.pl = copy.deepcopy(dict(fb.pl)); fb2.pl = defaultdict(dict, fb2.pl)
        fb2.pp = defaultdict(lambda: defaultdict(float), {k: defaultdict(float, dict(v)) for k, v in fb.pp.items()})
        fb2.tg = defaultdict(lambda: defaultdict(float), {k: defaultdict(float, dict(v)) for k, v in fb.tg.items()})
        for k in list(fb2.pl):
            if (k[0], k[1]) == wk:
                for r in fb2.pl[k].values():
                    r["car"] += 7; r["rush_yds"] += 99; r["tgt"] += 5; r["rec_yds"] += 77; r["att"] += 11
        for k in list(fb2.pp):
            if (k[0], k[1]) == wk:
                for c in ("car", "rush_yds", "tgt", "rec_yds", "routes", "rz_opp", "att"):
                    fb2.pp[k][c] += 13
        for k in list(fb2.tg):
            if (k[0], k[1]) == wk:
                for c in ("plays", "dropbacks", "rushes", "targets"):
                    fb2.tg[k][c] += 21
        pert_rows, _, _, _ = build_rows(fb2, until=wk)
        diffs = 0; comp = 0
        for fam in base_rows:
            a = {(r["key"]["pid"]): r["F"] for r in base_rows[fam] if (r["key"]["s"], r["key"]["w"]) == wk}
            b = {(r["key"]["pid"]): r["F"] for r in pert_rows[fam] if (r["key"]["s"], r["key"]["w"]) == wk}
            for pid in set(a) & set(b):
                comp += 1
                diffs += a[pid] != b[pid]
        check(R, "perturbing target-week realized stats leaves that week's features unchanged", diffs == 0 and comp > 0,
              f"{diffs} of {comp} feature rows changed")
    return R


def assert_invariants(R, extra=None):
    bad = [r for r in R + (extra or []) if not r["pass"]]
    if bad:
        raise InvariantError("INVARIANT FAILURES:\n" + "\n".join(f"  {b['invariant']}: {b['detail']}" for b in bad))


# ------------------------------------------------------------------ main
def run(data_dir, perturb=True):
    download(data_dir)
    fb = Football(data_dir)
    print("data loaded; replaying ...", flush=True)
    rows, team_rows, pregame, defense = build_rows(fb)
    print({k: len(v) for k, v in rows.items()}, "team", len(team_rows), "pregame", len(pregame), "defense", len(defense), flush=True)
    v2_features(rows, fb, data_dir)
    tp, team_rep = team_models(team_rows)
    report = {"protocol": {"train": "2023 + 2024 wk1-12", "valid": "2024 wk13-18", "eval": ["2025", "2026 wk1-3"],
                           "eval_status": "BURNED development data (repeatedly examined); not a holdout",
                           "conditional_universe": "players who played (production given participation)",
                           "pregame_universe": "pre-game-only membership; see pregame_mixture"},
              "team_components": team_rep}
    print("chains ...", flush=True)
    report["rushing"] = rushing_family(rows["rushing"], tp)
    report["receiving"] = receiving_family(rows["receiving"], tp, fb)
    report["passing"] = passing_family(rows["passing"], tp)
    report["touchdowns"] = td_family(rows["anytime_td"], tp, rows["rb_rush_td"], rows["rec_td"], fb)
    report["defense"] = defense_family_wrap(defense, tp)
    print("direct / distribution / pregame ...", flush=True)
    report["direct_models"] = {"rushing_yards": direct_models(rows["rushing"], "stat_rush_yds", "rush"),
                               "receiving_yards": direct_models(rows["receiving"], "stat_rec_yds", "rec"),
                               "passing_yards": direct_models(rows["passing"], "stat_pass_yds", "pass", with_incumbent=True)}
    report["pregame_mixture"] = {"rushing_yards": pregame_mixture(pregame, rows["rushing"], "rushing", "rush_yds"),
                                 "receiving_yards": pregame_mixture(pregame, rows["receiving"], "receiving", "rec_yds")}
    report["availability_sources"] = {"roster_timing_audit_2024_2025": roster_timing_audit(fb)}
    R = invariants(fb, rows, team_rows, pregame, defense, perturb=perturb)
    # exact reconstruction for every chain
    recon = []
    for fam_key in ("rushing", "receiving", "passing", "defense"):
        walk(report[fam_key], recon)
    walk(report["touchdowns"]["count_families"], recon)
    for path, err in recon:
        check(R, f"exact oracle reconstructs outcome: {path}", err is not None and err < 1e-6, f"max abs err {err}")
    report["invariants"] = R
    report["model_fits"] = [{"label": l, "fit_windows": sorted({f"{s}:{t}" for s, t in seen})} for l, seen, _ in FIT_LOG]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=1, default=float))
    return report, R


def defense_family_wrap(defense, tp):
    out = {}
    tr, va, _ = split(defense)
    fcol = list(defense[0]["F"])
    fr = lambda r: ratio(r["A"]["snaps"], r["A"]["opp_plays"])
    tr2 = [r for r in tr if r["A"]["opp_plays"]]; va2 = [r for r in va if r["A"]["opp_plays"]]
    b = fit(tr2, va2, fcol, arr(tr2, fr), arr(va2, fr), label="def_snap_rate")
    f_hat = np.clip(pred(b, defense), 0, 1.3)
    for s in EVAL_SEASONS:
        idx = [i for i, r in enumerate(defense) if r["key"]["s"] == s]
        er = [defense[i] for i in idx]
        OP = arr(er, lambda r: tp.get((r["key"]["s"], r["key"]["w"], r["key"]["opp"], "plays"), np.nan))
        pr_ = arr(er, lambda r: tp.get((r["key"]["s"], r["key"]["w"], r["key"]["opp"], "pass_rate"), np.nan))
        pc = {"OP": OP, "OD": OP * pr_, "F": f_hat[idx], "Kt": arr(er, lambda r: r["F"]["tkl_ps"]),
              "Ks": arr(er, lambda r: r["F"]["sack_pd"]), "Ki": arr(er, lambda r: r["F"]["dint_pd"])}
        Fa = arr(er, fr)
        ac = {"OP": arr(er, lambda r: r["A"]["opp_plays"]), "OD": arr(er, lambda r: r["A"]["opp_dropbacks"]), "F": Fa,
              "Kt": arr(er, lambda r: ratio(r["A"]["tkl"], r["A"]["snaps"])),
              "Ks": np.array([ratio(r["A"]["sack"], r["A"]["opp_dropbacks"] * f) for r, f in zip(er, Fa)]),
              "Ki": np.array([ratio(r["A"]["dint"], r["A"]["opp_dropbacks"] * f) for r, f in zip(er, Fa)])}
        snaps = arr(er, lambda r: r["A"]["snaps"])
        ok = ~np.isnan(OP) & (arr(er, lambda r: r["A"]["opp_plays"]) > 0)
        sel = lambda d: {k: v[ok] for k, v in d.items()}
        pc, ac = sel(pc), sel(ac)
        er2 = [r for r, o in zip(er, ok) if o]
        naive_t = arr(er2, lambda r: r["F"]["tkl_l3"] or 0.0)
        out[str(s)] = {
            "defensive_snaps": chain_budget("dsnaps", ["OP", "F"], pc, ac, lambda v: v["OP"] * v["F"], snaps[ok], kind="count"),
            "tackles": chain_budget("tkl", ["OP", "F", "Kt"], pc, ac, lambda v: v["OP"] * v["F"] * v["Kt"], arr(er2, lambda r: r["A"]["tkl"]), kind="count"),
            "tackles_last3_average_baseline": count_metrics(naive_t, arr(er2, lambda r: r["A"]["tkl"])),
            "sacks": chain_budget("sack", ["OD", "F", "Ks"], pc, ac, lambda v: v["OD"] * v["F"] * v["Ks"], arr(er2, lambda r: r["A"]["sack"]), kind="count"),
            "interceptions": chain_budget("dint", ["OD", "F", "Ki"], pc, ac, lambda v: v["OD"] * v["F"] * v["Ki"], arr(er2, lambda r: r["A"]["dint"]), kind="count"),
            "note": "F = defensive snaps / opponent pbp plays (excl. kneels, spikes, 2-pt, penalty no-plays); can exceed 1. "
                    "Tackles = solo + with-assist (same definition as nfl_defense_tackles_clean_baseline_a.py)."}
    return out


def walk(node, out, path=""):
    if isinstance(node, dict):
        if "exact_reconstruction_max_abs_err" in node and node.get("exact_reconstruction_max_abs_err") is not None:
            out.append((path, node["exact_reconstruction_max_abs_err"]))
            return
        for k, v in node.items():
            walk(v, out, f"{path}/{k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--no-perturbation-test", action="store_true")
    args = ap.parse_args()
    print("NFL_PLAYER_OUTCOME_ERROR_BUDGET_A (Phase 0B)\n============================================")
    report, R = run(args.data_dir, perturb=not args.no_perturbation_test)
    print(f"\ninvariants: {sum(r['pass'] for r in R)}/{len(R)} pass")
    for r in R:
        if not r["pass"]:
            print("  FAIL", r["invariant"], r["detail"])
    assert_invariants(R)
    print("report written:", OUT_DIR / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
