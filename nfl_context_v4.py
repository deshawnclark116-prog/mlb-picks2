#!/usr/bin/env python3
"""
NFL_CONTEXT_V4

Rushing yards, receiving yards and anytime-TD models built on what the
defense does, not just what the player did.

Defense profile (per defense, as of before each game; this season to
date, with last season blended in at 4 games' weight early on):
  by position   rushing yards/TDs allowed to RBs; receiving yards and TDs
                allowed to WRs, TEs and RBs separately
  efficiency    EPA per rush / per dropback allowed, explosive runs (10+)
                and completions (20+) allowed, sack and pressure rate
  red zone      TDs allowed per game inside the 20, share of them rushing
  scheme        man vs zone rate, Cover 0/1/2/3/4/6 shares (NFL tracking
                participation data -- published through 2025 only, so 2026
                uses each defense's 2025 tendencies), stacked-box rate and
                average box, blitz rate (FTN charting, available for 2026)
  injuries      defensive regulars out, by DL / LB / DB (real snaps in
                training, ESPN status when serving)

Offense opportunity (play-by-play):
  red-zone and goal-line (inside 5) carries, red-zone and end-zone
  targets, expected TDs (league TD rate by field position summed over the
  player's actual touches), team red-zone trips and TD rate, team
  goal-line run rate, and how often the QB takes the goal-line carries.

Player vs scheme:
  receiver yards per target vs man and vs zone, matched to this
  opponent's man rate; back yards per carry vs stacked (8+) and light
  boxes, matched to this opponent's stacked-box rate.

Plus everything v3 already had (usage share, teammates out, QB out,
Vegas spread/implied total, pace, weather, history).

Ship rule: on 2025 (never trained on) each market must significantly
beat the naive average AND the same model without the v4 inputs; on the
live 2026 games it must not be significantly worse than naive.

Run
---
python -u nfl_context_v4.py --data-dir /tmp/nflcsv
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import nfl_yardage_v3 as v3

REPO = Path(__file__).resolve().parent
MODEL_DIR = REPO / "nfl_models" / "nfl_context_v4_work"
PRIOR_WEIGHT_GAMES = 4.0
RB_POS, WR_POS, TE_POS = ("RB", "FB"), ("WR",), ("TE",)
DL = {"DE", "DT", "NT", "DL"}; LB = {"LB", "ILB", "OLB", "MLB"}; DB = {"CB", "S", "SS", "FS", "DB"}
COVERS = {"COVER_0": "c0", "COVER_1": "c1", "COVER_2": "c2", "2_MAN": "c2m", "COVER_3": "c3",
          "COVER_4": "c4", "COVER_6": "c6"}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def yl_bucket(yl):
    for b, hi in ((0, 2), (1, 5), (2, 10), (3, 20), (4, 40)):
        if yl <= hi:
            return b
    return 5


class PlayData:
    """Game-level defense tallies, per-player opportunity, per-team red-zone
    tendencies, and player-vs-scheme splits, all from play-by-play."""

    def __init__(self, data_dir, seasons, positions):
        data_dir = Path(data_dir)
        self.pos = positions               # (season, pid) -> position
        part = {}
        for s in seasons:
            p = data_dir / f"participation_{s}.csv"
            if p.exists():
                for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
                    part[(r["nflverse_game_id"], r["play_id"])] = (
                        r["defense_man_zone_type"], r["defense_coverage_type"], r["was_pressure"])
        ftn = {}
        for s in seasons:
            p = data_dir / f"ftn_{s}.csv"
            if p.exists():
                for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
                    ftn[(r["nflverse_game_id"], r["nflverse_play_id"])] = (
                        _f(r["n_defense_box"]), _f(r["n_blitzers"]))
        self.dgame = defaultdict(lambda: defaultdict(float))   # (s, w, defteam)
        self.poff = defaultdict(lambda: defaultdict(float))    # (s, w, pid)
        self.toff = defaultdict(lambda: defaultdict(float))    # (s, w, team)
        self.scheme = defaultdict(lambda: defaultdict(float))  # (s, w, pid): per-scheme yards/attempts
        plays = []
        for s in seasons:
            p = data_dir / f"pbp_{s}.csv.gz"
            if not p.exists():
                continue
            with gzip.open(p, "rt", newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r["season_type"] != "REG" or r["play_type"] not in ("run", "pass"):
                        continue
                    if r["two_point_attempt"] == "1":
                        continue
                    plays.append((s, r))
        # expected-TD table from the earliest season only (never an eval season)
        base = min(seasons)
        num = defaultdict(float); den = defaultdict(float)
        for s, r in plays:
            if s != base:
                continue
            yl = _f(r["yardline_100"])
            if yl is None:
                continue
            if r["play_type"] == "run" and r["rusher_player_id"]:
                k = ("run", yl_bucket(yl)); den[k] += 1; num[k] += r["rush_touchdown"] == "1"
            elif r["receiver_player_id"]:
                k = ("tgt", yl_bucket(yl)); den[k] += 1; num[k] += r["pass_touchdown"] == "1"
        self.xtd = {k: num[k] / den[k] for k in den}

        for s, r in plays:
            w = int(r["week"]); off, de = r["posteam"], r["defteam"]
            yl = _f(r["yardline_100"]) or 50.0
            yds = _f(r["yards_gained"]) or 0.0
            epa = _f(r["epa"]) or 0.0
            d = self.dgame[(s, w, de)]; t = self.toff[(s, w, off)]
            gid, pid = r["game_id"], r["play_id"]
            mz, cov, pressure = part.get((gid, pid), ("", "", ""))
            box, blitz = ftn.get((gid, pid), (None, None))
            rz, gl = yl <= 20, yl <= 5
            if rz:
                t["rz_plays"] += 1; d["rz_plays"] += 1
            if r["play_type"] == "run":
                rid = r["rusher_player_id"]; rpos = self.pos.get((s, rid), "")
                rtd = r["rush_touchdown"] == "1"
                d["rush_att"] += 1; d["rush_yds"] += yds; d["rush_epa"] += epa; d["rush_td"] += rtd
                d["rush_expl"] += yds >= 10
                if rpos in RB_POS:
                    d["rb_rush_yds"] += yds; d["rb_rush_td"] += rtd
                if box is not None:
                    d["box_n"] += 1; d["box_sum"] += box; d["stacked"] += box >= 8
                if rz:
                    d["rz_td"] += rtd; d["rz_rush_td"] += rtd; t["rz_td"] += rtd
                if gl:
                    t["gl_att"] += 1; t["gl_run"] += 1
                    if rpos == "QB":
                        t["gl_qb_run"] += 1
                if rid:
                    po = self.poff[(s, w, rid)]
                    po["xtd"] += self.xtd.get(("run", yl_bucket(yl)), 0.0)
                    po["rz_car"] += rz; po["gl_car"] += gl
                    if box is not None:
                        sc = self.scheme[(s, w, rid)]
                        key = "stacked" if box >= 8 else "light"
                        sc[f"{key}_n"] += 1; sc[f"{key}_y"] += yds
            else:
                d["dropbacks"] += 1; d["pass_epa"] += epa; d["sacks"] += r["sack"] == "1"
                if pressure in ("1", "TRUE", "True", "true"):
                    d["pressure"] += 1
                if pressure != "":
                    d["pressure_n"] += 1
                if blitz is not None:
                    d["blitz_n"] += 1; d["blitz"] += blitz > 0
                if mz in ("MAN_COVERAGE", "ZONE_COVERAGE"):
                    d["mz_n"] += 1; d["man"] += mz == "MAN_COVERAGE"
                if cov:
                    d["cov_n"] += 1
                    if cov in COVERS:
                        d[COVERS[cov]] += 1
                ptd = r["pass_touchdown"] == "1"
                if rz:
                    d["rz_td"] += ptd; t["rz_td"] += ptd
                if gl:
                    t["gl_att"] += 1
                rec = r["receiver_player_id"]
                if rec:
                    rpos = self.pos.get((s, rec), "")
                    grp = "wr" if rpos in WR_POS else "te" if rpos in TE_POS else "rb" if rpos in RB_POS else None
                    ryds = yds if r["complete_pass"] == "1" else 0.0
                    if grp:
                        d[f"{grp}_tgt"] += 1; d[f"{grp}_rec_yds"] += ryds; d[f"{grp}_rec_td"] += ptd
                    d["pass_expl"] += r["complete_pass"] == "1" and yds >= 20
                    po = self.poff[(s, w, rec)]
                    po["xtd"] += self.xtd.get(("tgt", yl_bucket(yl)), 0.0)
                    po["rz_tgt"] += rz
                    ay = _f(r["air_yards"])
                    po["ez_tgt"] += ay is not None and ay >= yl
                    if mz in ("MAN_COVERAGE", "ZONE_COVERAGE"):
                        sc = self.scheme[(s, w, rec)]
                        key = "man" if mz == "MAN_COVERAGE" else "zone"
                        sc[f"{key}_n"] += 1; sc[f"{key}_y"] += ryds
        self.games_of = defaultdict(list)   # (s, defteam) -> weeks played
        for (s, w, de) in self.dgame:
            self.games_of[(s, de)].append(w)
        for k in self.games_of:
            self.games_of[k].sort()
        self._def_cache = {}

    def def_profile(self, s, w, team):
        key = (s, w, team)
        if key in self._def_cache:
            return self._def_cache[key]
        cur = defaultdict(float); n_cur = 0
        for wk in self.games_of.get((s, team), []):
            if wk >= w:
                break
            for k, v in self.dgame[(s, wk, team)].items():
                cur[k] += v
            n_cur += 1
        prv = defaultdict(float); n_prv = 0
        for wk in self.games_of.get((s - 1, team), []):
            for k, v in self.dgame[(s - 1, wk, team)].items():
                prv[k] += v
            n_prv += 1
        wt = PRIOR_WEIGHT_GAMES / n_prv if n_prv else 0.0
        tot = defaultdict(float)
        for k in set(cur) | set(prv):
            tot[k] = cur[k] + wt * prv[k]
        g = n_cur + (PRIOR_WEIGHT_GAMES if n_prv else 0.0)
        # coverage labels don't exist for 2026 yet -- use last season's alone
        cov = prv if cur["mz_n"] == 0 else tot
        rate = lambda a, b, src=tot: (src[a] / src[b]) if src[b] else None
        pg = lambda a: (tot[a] / g) if g else None
        out = {
            "def_games": n_cur,
            "def_rush_ypg": pg("rush_yds"), "def_rb_rush_ypg": pg("rb_rush_yds"),
            "def_rush_ypc": rate("rush_yds", "rush_att"), "def_rush_td_pg": pg("rush_td"),
            "def_rb_rush_td_pg": pg("rb_rush_td"), "def_rush_epa": rate("rush_epa", "rush_att"),
            "def_rush_expl_rate": rate("rush_expl", "rush_att"),
            "def_wr_rec_ypg": pg("wr_rec_yds"), "def_te_rec_ypg": pg("te_rec_yds"),
            "def_rb_rec_ypg": pg("rb_rec_yds"), "def_wr_td_pg": pg("wr_rec_td"),
            "def_te_td_pg": pg("te_rec_td"), "def_rb_rec_td_pg": pg("rb_rec_td"),
            "def_pass_epa": rate("pass_epa", "dropbacks"), "def_pass_expl_rate": rate("pass_expl", "dropbacks"),
            "def_sack_rate": rate("sacks", "dropbacks"), "def_pressure_rate": rate("pressure", "pressure_n"),
            "def_rz_td_pg": pg("rz_td"), "def_rz_rush_td_share": rate("rz_rush_td", "rz_td"),
            "def_man_rate": rate("man", "mz_n", cov), "def_c0": rate("c0", "cov_n", cov),
            "def_c1": rate("c1", "cov_n", cov), "def_c2": rate("c2", "cov_n", cov),
            "def_c3": rate("c3", "cov_n", cov), "def_c4": rate("c4", "cov_n", cov),
            "def_c6": rate("c6", "cov_n", cov),
            "def_stacked_rate": rate("stacked", "box_n"), "def_avg_box": rate("box_sum", "box_n"),
            "def_blitz_rate": rate("blitz", "blitz_n"),
        }
        self._def_cache[key] = out
        return out


DEF_FEATURES = ["def_games", "def_rush_ypg", "def_rb_rush_ypg", "def_rush_ypc", "def_rush_td_pg",
                "def_rb_rush_td_pg", "def_rush_epa", "def_rush_expl_rate", "def_wr_rec_ypg",
                "def_te_rec_ypg", "def_rb_rec_ypg", "def_wr_td_pg", "def_te_td_pg", "def_rb_rec_td_pg",
                "def_pass_epa", "def_pass_expl_rate", "def_sack_rate", "def_pressure_rate",
                "def_rz_td_pg", "def_rz_rush_td_share", "def_man_rate", "def_c0", "def_c1", "def_c2",
                "def_c3", "def_c4", "def_c6", "def_stacked_rate", "def_avg_box", "def_blitz_rate",
                "def_dl_out", "def_lb_out", "def_db_out"]
OPP_FEATURES = ["rz_car_last3", "gl_car_last3", "rz_tgt_last3", "ez_tgt_last3", "xtd_last3", "xtd_season",
                "td_last3", "td_season", "td_prior", "gl_share_last3", "rz_tgt_share_last3",
                "team_rz_plays_pg", "team_rz_td_rate", "team_gl_run_rate", "team_qb_gl_share",
                "scheme_ypt_edge", "scheme_ypc_edge", "is_rb", "is_te"]
V4_FEATURES = v3.FEATURES + DEF_FEATURES + OPP_FEATURES

V4_MARKETS = {
    "rushing_yards": {"vol": "carries", "yds": "rushing_yards", "positions": RB_POS, "min_last3_vol": 5.0},
    "receiving_yards": {"vol": "targets", "yds": "receiving_yards", "positions": WR_POS + TE_POS + RB_POS,
                         "min_last3_vol": 3.0},
    "anytime_td": {"vol": "touches", "yds": "tds", "positions": RB_POS + WR_POS + TE_POS, "min_last3_vol": 4.0},
}


class V4Replayer(v3.Replayer):
    def __init__(self, data, plays, mkt):
        self.d = data; self.p = plays; self.mkt = mkt; self.cfg = V4_MARKETS[mkt]
        self.hist = defaultdict(list); self.opp = {}; self.team_vol = {}
        self.cfg_v3 = {"vol": self.cfg["vol"], "yds": self.cfg["yds"]}

    def _row_val(self, r, field):
        if field == "touches":
            return r["carries"] + r["targets"]
        if field == "tds":
            return r["rtd"] + r["rectd"]
        return r[field]

    def absorb_week(self, s, w):
        for team in list(self.d.team_games):
            key = (s, w, team)
            rows = self.d.stats.get(key)
            if not rows:
                continue
            ctx = self.d.games.get(key)
            tvol = sum(self._row_val(r, self.cfg["vol"]) for r in rows.values())
            snaps = self.d.snaps.get(key, {})
            tgl = sum(self.p.poff[(s, w, pid)]["gl_car"] for pid in rows)
            trz = sum(self.p.poff[(s, w, pid)]["rz_tgt"] for pid in rows)
            for pid, r in rows.items():
                po = self.p.poff[(s, w, pid)]; sc = self.p.scheme[(s, w, pid)]
                self.hist[pid].append({
                    "s": s, "w": w, "vol": self._row_val(r, self.cfg["vol"]),
                    "yds": self._row_val(r, self.cfg["yds"]), "pos": r["pos"], "name": r["name"], "team": team,
                    "share": (self._row_val(r, self.cfg["vol"]) / tvol) if tvol else None,
                    "snap": snaps.get(v3.norm_name(r["name"])), "tshare": r["target_share"],
                    "ashare": r["air_yards_share"], "wopr": r["wopr"], "td": r["rtd"] + r["rectd"],
                    "rz_car": po["rz_car"], "gl_car": po["gl_car"], "rz_tgt": po["rz_tgt"],
                    "ez_tgt": po["ez_tgt"], "xtd": po["xtd"],
                    "gl_share": po["gl_car"] / tgl if tgl else None,
                    "rz_tgt_share": po["rz_tgt"] / trz if trz else None,
                    "sc": dict(sc), "rec_yds": r["receiving_yards"], "tgts": r["targets"],
                    "rush_yds": r["rushing_yards"], "car": r["carries"],
                })
            if ctx:
                st = self.opp.setdefault((s, ctx["opp"]), [0.0, 0])
                st[0] += sum(self._row_val(r, self.cfg["yds"]) for r in rows.values()); st[1] += 1

    def def_out(self, s, w, opp, absent_fn):
        """Defensive regulars out, by group, from the defense's last 3 games."""
        prev = [g for g in self.d.team_games[opp] if g < (s, w)][-3:]
        cnt = defaultdict(int); pos = {}
        for g in prev:
            for nn, (pct, p) in self.d.def_snaps.get((g[0], g[1], opp), {}).items():
                if pct >= 0.5:
                    cnt[nn] += 1; pos[nn] = p
        out = {"def_dl_out": 0.0, "def_lb_out": 0.0, "def_db_out": 0.0}
        for nn, c in cnt.items():
            if c >= min(2, len(prev)) and absent_fn(nn):
                p = pos[nn]
                k = "def_dl_out" if p in DL else "def_lb_out" if p in LB else "def_db_out" if p in DB else None
                if k:
                    out[k] += 1
        return out

    def team_rz(self, s, w, team):
        tot = defaultdict(float); n = 0
        for g in self.d.team_games[team]:
            if g >= (s, w) or g[0] < s - 1:
                continue
            wt = 1.0 if g[0] == s else 0.25
            for k, v in self.p.toff[(g[0], g[1], team)].items():
                tot[k] += wt * v
            n += wt
        return {
            "team_rz_plays_pg": tot["rz_plays"] / n if n else None,
            "team_rz_td_rate": tot["rz_td"] / tot["rz_plays"] if tot["rz_plays"] else None,
            "team_gl_run_rate": tot["gl_run"] / tot["gl_att"] if tot["gl_att"] else None,
            "team_qb_gl_share": tot["gl_qb_run"] / tot["gl_run"] if tot["gl_run"] else None,
        }

    def v4_features(self, pid, s, w, team, ctx, tctx, dout):
        f = self.player_features(pid, s, w, team, ctx, tctx)
        ph = self.hist[pid]; last3 = ph[-3:]
        cur = [g for g in ph if g["s"] == s]; prior = [g for g in ph if g["s"] == s - 1]
        m = v3._mean
        f.update(self.p.def_profile(s, w, ctx["opp"]))
        f.update(dout)
        f.update(self.team_rz(s, w, team))
        f.update({
            "rz_car_last3": m([g["rz_car"] for g in last3]), "gl_car_last3": m([g["gl_car"] for g in last3]),
            "rz_tgt_last3": m([g["rz_tgt"] for g in last3]), "ez_tgt_last3": m([g["ez_tgt"] for g in last3]),
            "xtd_last3": m([g["xtd"] for g in last3]), "xtd_season": m([g["xtd"] for g in cur]),
            "td_last3": m([g["td"] for g in last3]), "td_season": m([g["td"] for g in cur]),
            "td_prior": m([g["td"] for g in prior]),
            "gl_share_last3": m([g["gl_share"] for g in last3]),
            "rz_tgt_share_last3": m([g["rz_tgt_share"] for g in last3]),
            "is_rb": 1.0 if ph[-1]["pos"] in RB_POS else 0.0, "is_te": 1.0 if ph[-1]["pos"] in TE_POS else 0.0,
        })
        # player vs scheme: career-to-date splits shrunk toward his overall rate
        agg = defaultdict(float)
        for g in ph:
            for k, v in g["sc"].items():
                agg[k] += v
        tg = sum(g["tgts"] for g in ph); ry = sum(g["rec_yds"] for g in ph)
        car = sum(g["car"] for g in ph); rsy = sum(g["rush_yds"] for g in ph)
        man_rate = f.get("def_man_rate")
        if tg >= 10 and man_rate is not None:
            base = ry / tg; k = 15.0
            ypt_m = (agg["man_y"] + k * base) / (agg["man_n"] + k)
            ypt_z = (agg["zone_y"] + k * base) / (agg["zone_n"] + k)
            f["scheme_ypt_edge"] = man_rate * ypt_m + (1 - man_rate) * ypt_z - base
        else:
            f["scheme_ypt_edge"] = None
        st = f.get("def_stacked_rate")
        if car >= 20 and st is not None:
            base = rsy / car; k = 20.0
            ypc_s = (agg["stacked_y"] + k * base) / (agg["stacked_n"] + k)
            ypc_l = (agg["light_y"] + k * base) / (agg["light_n"] + k)
            f["scheme_ypc_edge"] = st * ypc_s + (1 - st) * ypc_l - base
        else:
            f["scheme_ypc_edge"] = None
        return f

    def replay(self, until=None):
        out = []
        weeks = sorted({(k[0], k[1]) for k in self.d.stats})
        for (s, w) in weeks:
            if until and (s, w) >= until:
                break
            for team in list(self.d.team_games):
                key = (s, w, team)
                rows = self.d.stats.get(key); ctx = self.d.games.get(key)
                if not rows or not ctx:
                    continue
                played = self.d.played_names(key)
                tctx = self.d.team_context(team, s, w, lambda nn: nn not in played)
                opp_played = set(self.d.def_snaps.get((s, w, ctx["opp"]), {}))
                dout = self.def_out(s, w, ctx["opp"], lambda nn: nn not in opp_played)
                for pid, r in rows.items():
                    if not self.eligible(pid) or v3.norm_name(r["name"]) not in played:
                        continue
                    f = self.v4_features(pid, s, w, team, ctx, tctx, dout)
                    out.append(({"pid": pid, "name": r["name"], "team": team, "s": s, "w": w},
                                f, self._row_val(r, self.cfg["yds"])))
            self.absorb_week(s, w)
        return out


def mat(rows, cols):
    return np.array([[r[1].get(c) if r[1].get(c) is not None else np.nan for c in cols] for r in rows],
                    dtype=np.float32)


def load(data_dir):
    seasons = [2022, 2023, 2024, 2025, 2026]
    data = v3.Data(data_dir, seasons)
    positions = {}
    for (s, w, t), rows in data.stats.items():
        for pid, r in rows.items():
            positions[(s, pid)] = r["pos"]
    plays = PlayData(data_dir, seasons, positions)
    return data, plays


def evaluate(data_dir):
    import xgboost as xgb
    import baseline_audit_a as audit
    data, plays = load(data_dir)
    print("data loaded", flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    for mkt in V4_MARKETS:
        rows = V4Replayer(data, plays, mkt).replay()
        tr = [r for r in rows if r[0]["s"] == 2023 or (r[0]["s"] == 2024 and r[0]["w"] <= 12)]
        va = [r for r in rows if r[0]["s"] == 2024 and r[0]["w"] > 12]
        ev = {s: [r for r in rows if r[0]["s"] == s] for s in (2025, 2026)}
        is_td = mkt == "anytime_td"
        y = (lambda rr: np.array([1.0 if r[2] >= 1 else 0.0 for r in rr])) if is_td else \
            (lambda rr: np.array([r[2] for r in rr], dtype=float))
        obj = {"objective": "binary:logistic"} if is_td else {"objective": "reg:absoluteerror"}
        params = {**v3.PARAMS, **obj}
        fits = {}
        for name, cols in (("v3 inputs", v3.FEATURES), ("v4 inputs", V4_FEATURES)):
            b = xgb.train(params, xgb.DMatrix(mat(tr, cols), label=y(tr), feature_names=cols), 3000,
                          evals=[(xgb.DMatrix(mat(va, cols), label=y(va), feature_names=cols), "va")],
                          early_stopping_rounds=80, verbose_eval=False)
            fits[name] = (b, cols)
        print(f"\n{mkt}: train {len(tr)}  val {len(va)}")
        rep = {}
        for s, er in ev.items():
            pr = {n: b.predict(xgb.DMatrix(mat(er, c), feature_names=c), iteration_range=(0, b.best_iteration + 1))
                  for n, (b, c) in fits.items()}
            if is_td:
                prv = {n: b.predict(xgb.DMatrix(mat(va, c), feature_names=c), iteration_range=(0, b.best_iteration + 1))
                       for n, (b, c) in fits.items()}
                naive = lambda rr: [next((v for v in (r[1]["td_season"] if r[1]["cur_n"] >= 3 else None,
                                                      r[1]["td_prior"], r[1]["td_last3"]) if v is not None), 0.0)
                                    for r in rr]
                r4 = audit.compare(f"anytime TD v4 [{s}]", prv["v4 inputs"], naive(va), y(va), pr["v4 inputs"], naive(er), y(er))
                r3 = audit.compare(f"anytime TD v3-inputs [{s}]", prv["v3 inputs"], naive(va), y(va), pr["v3 inputs"], naive(er), y(er))
                rep[str(s)] = {"v4": r4, "v3_inputs": r3}
            else:
                a = y(er)
                nb = np.array([v3.naive_preds(r[1])[1] for r in er]); nl = np.array([v3.naive_preds(r[1])[0] for r in er])
                best = nb if np.abs(nb - a).mean() <= np.abs(nl - a).mean() else nl
                m4 = np.abs(pr["v4 inputs"] - a); m3 = np.abs(pr["v3 inputs"] - a); mn = np.abs(best - a)
                rng = np.random.default_rng(1)
                def p_not_better(x, ref):
                    d = ref - x; bt = d[rng.integers(0, len(d), (2000, len(d)))].mean(axis=1); return float((bt <= 0).mean())
                rep[str(s)] = {"n": len(er), "v4_mae": round(float(m4.mean()), 2), "v3_inputs_mae": round(float(m3.mean()), 2),
                               "naive_mae": round(float(mn.mean()), 2), "p_v4_not_better_than_naive": p_not_better(m4, mn),
                               "p_v4_not_better_than_v3": p_not_better(m4, m3)}
                print(f"  [{s}] n={len(er):5d}  MAE v4 {m4.mean():5.2f}  v3-inputs {m3.mean():5.2f}  naive {mn.mean():5.2f}  "
                      f"p(v4 !> naive)={rep[str(s)]['p_v4_not_better_than_naive']:.3f}  "
                      f"p(v4 !> v3)={rep[str(s)]['p_v4_not_better_than_v3']:.3f}")
        imp = fits["v4 inputs"][0].get_score(importance_type="gain")
        rep["top_features"] = sorted(imp, key=imp.get, reverse=True)[:15]
        print("  top v4 features:", rep["top_features"])
        fits["v4 inputs"][0].save_model(str(MODEL_DIR / f"nfl_{mkt}_v4_candidate.json"))
        report[mkt] = rep
    (MODEL_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


# --------------------------------------------------------------- serving
def ensure_play_data(data_dir, seasons, live):
    """pbp / FTN charting / tracking participation for `seasons`; the live
    season is always re-downloaded (new games land weekly)."""
    import urllib.request
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    rel = v3.RELEASE
    for y in seasons:
        for name, url in ((f"pbp_{y}.csv.gz", f"{rel}/pbp/play_by_play_{y}.csv.gz"),
                          (f"ftn_{y}.csv", f"{rel}/ftn_charting/ftn_charting_{y}.csv"),
                          (f"participation_{y}.csv", f"{rel}/pbp_participation/pbp_participation_{y}.csv")):
            pth = data_dir / name
            if pth.exists() and y != live:
                continue
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "nfl-context-v4"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    pth.write_bytes(r.read())
            except Exception as e:      # participation isn't published for the live season
                print(f"    v4: {name} not available ({e})")


class TdProjector:
    """Live anytime-TD probabilities from the v4 model (defense profile,
    red-zone/goal-line usage, expected TDs, teammates and defenders out).
    Calibrated on 2025, which the model never trained on."""

    def __init__(self, data_dir, season, week):
        import xgboost as xgb
        self.prob, self.detail, self.rows = {}, {}, []
        path = MODEL_DIR / "nfl_anytime_td_v4_candidate.json"
        if not path.exists():
            return
        seasons = [season - 1, season]
        v3.ensure_data(data_dir, seasons, refresh=(season,))
        ensure_play_data(data_dir, seasons, season)
        data, plays = load_seasons(data_dir, seasons)
        bst = xgb.Booster(); bst.load_model(str(path))
        pred = lambda F: bst.predict(xgb.DMatrix(mat([(None, f) for f in F], V4_FEATURES),
                                                  feature_names=V4_FEATURES))
        rp = V4Replayer(data, plays, "anytime_td")
        hist_rows = rp.replay(until=(season, week))
        cal = [r for r in hist_rows if r[0]["s"] == season - 1 and r[0]["w"] >= 4]
        raw = np.clip(pred([r[1] for r in cal]), 1e-4, 1 - 1e-4)
        y = np.array([1.0 if r[2] >= 1 else 0.0 for r in cal])
        self.a, self.b = platt(np.log(raw / (1 - raw)), y)
        sched = {t: c for (s, w, t), c in data.games.items() if s == season and w == week}
        out_now = v3.espn_unavailable(sched)
        keys, feats, meta = [], [], []
        for pid, ph in rp.hist.items():
            if not ph or not rp.eligible(pid):
                continue
            team = ph[-1]["team"]
            nn = v3.norm_name(ph[-1]["name"])
            if team not in sched or nn in out_now.get(team, set()):
                continue
            ctx = sched[team]
            tctx = data.team_context(team, season, week, lambda x, t=team: x in out_now.get(t, set()))
            dout = rp.def_out(season, week, ctx["opp"], lambda x, o=ctx["opp"]: x in out_now.get(o, set()))
            keys.append((nn, team)); feats.append(rp.v4_features(pid, season, week, team, ctx, tctx, dout))
            meta.append({"player_id": pid, "player": ph[-1]["name"], "team": team, "opponent": ctx["opp"],
                         "games_played": len(ph), "position": ph[-1]["pos"]})
        if not feats:
            return
        raw = np.clip(pred(feats), 1e-4, 1 - 1e-4)
        p = 1 / (1 + np.exp(-(self.a * np.log(raw / (1 - raw)) + self.b)))
        for k, pr, f, mt in zip(keys, p, feats, meta):
            self.rows.append({**mt, "prob": float(pr)})
            self.prob[k] = float(pr)
            self.detail[k] = {kk: f.get(kk) for kk in ("xtd_season", "rz_car_last3", "gl_car_last3",
                                                       "rz_tgt_last3", "ez_tgt_last3", "def_rz_td_pg",
                                                       "implied_team_total", "vacated_tgt", "vacated_car")}
        print(f"    v4 TD: {len(self.prob)} players scored, platt a={self.a:.3f} b={self.b:.3f} (n_cal={len(cal)})")

    def get(self, player_name, team):
        return self.prob.get((v3.norm_name(player_name), team))


def platt(z, y, iters=200):
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * z + b)))
        g = np.array([((p - y) * z).sum(), (p - y).sum()])
        w = p * (1 - p)
        H = np.array([[(w * z * z).sum(), (w * z).sum()], [(w * z).sum(), w.sum()]]) + 1e-6 * np.eye(2)
        a, b = np.array([a, b]) - np.linalg.solve(H, g)
    return float(a), float(b)


def load_seasons(data_dir, seasons):
    data = v3.Data(data_dir, seasons)
    positions = {}
    for (s, w, t), rows in data.stats.items():
        for pid, r in rows.items():
            positions[(s, pid)] = r["pos"]
    return data, PlayData(data_dir, seasons, positions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    print("NFL_CONTEXT_V4\n==============")
    evaluate(args.data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
