#!/usr/bin/env python3
"""
NFL_YARDAGE_V2X - rushing / receiving yardage projection (serving + training).

What changed from nfl_yardage_v2.py (all measured walk-forward on 2025 week 6 onward and 2026, same games, see research/nfl_model_tests/):
  1. trained on 2016 onward instead of 2023 onward (about 4x the examples)       -> receiving 22.43 -> 22.29, rushing 24.79 -> 24.50 yds
  2. six extra inputs known before kickoff:
       own_status / own_practice  the player's own reported injury status and practice participation this week
       tm_out_vol / tm_quest_vol  recent volume of teammates (same position group, played in the team's last game) who are listed
                                  Out/Doubtful, and Questionable, this week
       snap3 / snap1              the player's offensive snap share over his last 3 games / last game
Live gain over the v2 model on the same games: rushing 24.99 -> 24.51, receiving 22.60 -> 22.33 yards average miss.

No sportsbook line is used anywhere.  The same feature code is used to build training rows and live rows (replay3 / Projector3), so
training and serving cannot drift apart.  If the injury or snap source cannot be fetched, the builder falls back to nfl_yardage_v2.

Train:  python -u nfl_yardage_v2x.py --db <extended sqlite> --data-dir <folder with injuries_YYYY.csv, snap_counts_YYYY.csv>
"""
import argparse
import csv
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

import nfl_yardage_v2 as v2

REPO = Path(__file__).resolve().parent
MODEL_DIR = REPO / "nfl_models" / "nfl_yardage_v2x_work"
MARKETS = v2.MARKETS
EXTRA = ["own_status", "own_practice", "tm_out_vol", "tm_quest_vol", "snap3", "snap1"]
FEATURES = v2.FEATURES + EXTRA
TRAIN_START_SEASON = 2016
ROUNDS = 140
STAT = {"Out": 3, "Doubtful": 2, "Questionable": 1}
PRAC = {"Did Not Participate In Practice": 2, "Limited Participation in Practice": 1, "Full Participation in Practice": 0}
RELEASE = "https://github.com/nflverse/nflverse-data/releases/download/%s/%s_%d.csv"
norm_name = v2.norm_name


# ------------------------------------------------------------------ pregame sources (injury report, snap counts)
class Sources:
    """Injury/practice status by (gsis_id, season, week) and offensive snap share by week, from the free nflverse releases."""

    def __init__(self):
        self.inj = {}
        self.snap = defaultdict(list)
        self.years_inj, self.years_snap = set(), set()

    def add_injuries(self, rows, year):
        n = 0
        for r in rows:
            if r.get("game_type") != "REG" or not r.get("gsis_id"):
                continue
            k = (r["gsis_id"], int(r["season"]), int(r["week"]))
            st = STAT.get(r.get("report_status") or "", 0)
            pr = PRAC.get(r.get("practice_status") or "", -1)
            o = self.inj.get(k, (0, -1))
            self.inj[k] = (max(o[0], st), max(o[1], pr))
            n += 1
        if n:
            self.years_inj.add(year)

    def add_snaps(self, rows, year):
        n = 0
        for r in rows:
            if r.get("game_type") != "REG":
                continue
            self.snap[(int(r["season"]), int(r["week"]))].append((norm_name(r["player"]), float(r.get("offense_pct") or 0)))
            n += 1
        if n:
            self.years_snap.add(year)

    @classmethod
    def from_dir(cls, data_dir, years):
        s = cls()
        for y in years:
            p = Path(data_dir) / f"injuries_{y}.csv"
            if p.exists():
                with open(p, newline="") as f:
                    s.add_injuries(csv.DictReader(f), y)
            p = Path(data_dir) / f"snap_counts_{y}.csv"
            if p.exists():
                with open(p, newline="") as f:
                    s.add_snaps(csv.DictReader(f), y)
        return s

    @classmethod
    def fetch(cls, years, cache_dir, timeout=90):
        """Download the releases (cached on disk); a year that cannot be fetched is skipped. The caller checks `usable_for`."""
        import requests
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        for y in years:
            for tag in ("injuries", "snap_counts"):
                dest = cache / f"{tag}_{y}.csv"
                try:
                    r = requests.get(RELEASE % (tag, tag, y), timeout=timeout)
                    if r.status_code == 200 and r.content[:4] != b"Not ":
                        dest.write_bytes(r.content)
                except Exception as e:
                    print(f"  nflverse {tag} {y} not fetched ({e}); using cached copy if any")
        return cls.from_dir(cache, years)

    def usable_for(self, season):
        """Current-season injury report present and snap history present (this or last season)."""
        return season in self.years_inj and bool({season, season - 1} & self.years_snap)


# ------------------------------------------------------------------ features (one code path for training and serving)
def extras(src, cfg, pid, name, season, week, regs, snap_hist):
    o = src.inj.get((pid, season, week), (0, -1))
    tmo = tmq = 0.0
    for p, (avg, pos) in regs.items():
        if p == pid or pos not in cfg["positions"]:
            continue
        st = src.inj.get((p, season, week), (0, -1))[0]
        if st >= 2:
            tmo += avg
        elif st == 1:
            tmq += avg
    sh = snap_hist.get(norm_name(name), [])
    return {"own_status": o[0], "own_practice": o[1], "tm_out_vol": tmo, "tm_quest_vol": tmq,
            "snap3": v2._mean(sh[-3:]), "snap1": sh[-1] if sh else None}


def replay3(con, mkt, src, until=None):
    """Completed player-games -> (meta, features, actual) with the extra inputs; also returns the end state for featuring a future week."""
    h = v2.History(con, mkt)
    cfg = h.cfg
    hist = defaultdict(list)
    opp_state = {}
    snap_hist = defaultdict(list)
    team_regs = defaultdict(dict)
    out = []
    by_week = defaultdict(list)
    for r in h.rows:
        by_week[(r[5], r[6])].append(r)
    for key in sorted(by_week):
        if until and key >= until:
            break
        wk = by_week[key]
        for (pid, name, pos, team, opp, season, week, gid, is_home, vol, yds) in wk:
            ph = hist[pid]
            if pos in cfg["positions"] and len(ph) >= 3 and v2._mean([g["vol"] for g in ph[-3:]]) >= cfg["min_last3_vol"]:
                f = h.features(ph, season, week, team, opp, is_home, opp_state)
                f.update(extras(src, cfg, pid, name, season, week, team_regs[team], snap_hist))
                out.append(({"player_id": pid, "name": name, "team": team, "opp": opp, "season": season, "week": week}, f, float(yds)))
        for (pid, name, pos, team, opp, season, week, gid, is_home, vol, yds) in wk:
            hist[pid].append({"season": season, "week": week, "vol": vol, "yds": yds, "team_vol": h.team_vol[(gid, team)],
                              "team": team, "name": name, "pos": pos})
            st = opp_state.setdefault((season, opp), [0.0, 0])
            st[0] += yds
        for (season, opp, gid) in {(r[5], r[4], r[7]) for r in wk}:
            opp_state[(season, opp)][1] += 1
        for nm, pct in src.snap.get(key, []):
            snap_hist[nm].append(pct)
        fresh = {t: {} for t in {r[3] for r in wk}}
        for (pid, name, pos, team, opp, season, week, gid, is_home, vol, yds) in wk:
            if vol <= 0:
                continue
            a = v2._mean([g["vol"] for g in hist[pid][-3:]])
            if a is not None and a >= cfg["min_last3_vol"]:
                fresh[team][pid] = (a, pos)
        for t, d in fresh.items():
            team_regs[t] = d
    return out, hist, opp_state, snap_hist, team_regs, h


def to_matrix(rows):
    return np.array([[r[1][c] if r[1][c] is not None else np.nan for c in FEATURES] for r in rows], dtype=np.float32)


# ------------------------------------------------------------------ training
def train(db, data_dir, out_dir=MODEL_DIR, start=TRAIN_START_SEASON, write=True):
    import xgboost as xgb
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    src = Sources.from_dir(data_dir, range(2009, 2027))
    params = dict(v2.PARAMS)
    params["nthread"] = 4
    report = {"features": FEATURES, "train_start_season": start, "rounds": ROUNDS, "markets": {}}
    wf = {}
    for mkt in MARKETS:
        rows = replay3(con, mkt, src)[0]
        keys = sorted({(r[0]["season"], r[0]["week"]) for r in rows})
        test = [k for k in keys if (k[0] == 2025 and k[1] >= 6) or k[0] >= 2026]
        act, pred, naive, base = [], [], [], []
        for k in test:
            tr = [r for r in rows if r[0]["season"] >= start and (r[0]["season"], r[0]["week"]) < k]
            te = [r for r in rows if (r[0]["season"], r[0]["week"]) == k]
            if len(tr) < 300 or not te:
                continue
            b = xgb.train(params, xgb.DMatrix(to_matrix(tr), label=[r[2] for r in tr], feature_names=FEATURES), num_boost_round=ROUNDS)
            pred += list(b.predict(xgb.DMatrix(to_matrix(te), feature_names=FEATURES)))
            act += [r[2] for r in te]
            naive += [v2.naive_preds(r[1])[1] for r in te]
        act, pred, naive = np.array(act), np.array(pred), np.array(naive)
        ratios = act / np.maximum(pred, 1.0)
        rng = np.random.default_rng(1)
        d = np.abs(naive - act) - np.abs(pred - act)
        boot = d[rng.integers(0, len(d), (2000, len(d)))].mean(axis=1)
        r_ = np.sort(ratios)
        q = lambda x: float(r_[int(x * (len(r_) - 1))])
        report["markets"][mkt] = {
            "n_test": int(len(act)), "model_mae": round(float(np.abs(pred - act).mean()), 3),
            "naive_blend_mae": round(float(np.abs(naive - act).mean()), 3),
            "improvement_vs_naive_pct": round(float((np.abs(naive - act).mean() - np.abs(pred - act).mean()) / np.abs(naive - act).mean() * 100), 2),
            "p_not_better_than_naive": round(float((boot <= 0).mean()), 4),
            "actual_over_projection_quantiles": {k: round(q(v), 3) for k, v in (("p10", .10), ("p15", .15), ("p25", .25), ("p50", .50), ("p75", .75), ("p90", .90))},
        }
        wf[mkt] = (act, pred)
        full = [r for r in rows if r[0]["season"] >= start]
        final = xgb.train(params, xgb.DMatrix(to_matrix(full), label=[r[2] for r in full], feature_names=FEATURES), num_boost_round=ROUNDS)
        report["markets"][mkt]["n_train_final"] = len(full)
        if write:
            final.save_model(str(out_dir / f"nfl_{mkt}_v2x.json"))
        print(mkt, json.dumps(report["markets"][mkt]))
    report["passed"] = all(m["improvement_vs_naive_pct"] > 0 and m["p_not_better_than_naive"] < 0.05 for m in report["markets"].values())
    if write:
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        import nfl_yardage_floor_table as ft
        doc = {"schema": "NFL_YARDAGE_SAFE_NUMBER_TABLE_V1", "target": ft.TARGET, "sportsbook_line_used": False,
               "method": "walk-forward out-of-sample predictions of the live v2x yardage model; each week uses only earlier weeks",
               "markets": {m: {"n": int(len(a)), "bands": ft.table_for(a, p)} for m, (a, p) in wf.items()}}
        (REPO / "nfl_models" / "nfl_yardage_floor_table.json").write_text(json.dumps(doc, indent=1))
    return report


# ------------------------------------------------------------------ serving
class Projector3:
    """Serving-side: v2x projections for one upcoming week, keyed by (normalized player name, team). Same API as v2.Projector."""

    def __init__(self, con, season, week, src, model_dir=MODEL_DIR):
        import xgboost as xgb
        model_dir = Path(model_dir)
        self.ok = False
        self.proj = {}
        self.blocked = {}
        self.quant = {}
        rp = model_dir / "report.json"
        if not rp.exists() or not src.usable_for(season):
            return
        rep = json.loads(rp.read_text())
        if rep.get("features") != FEATURES or not rep.get("passed"):
            return
        self.quant = {m: rep["markets"][m]["actual_over_projection_quantiles"] for m in MARKETS}
        sched = {}
        for home, away, spread, total in con.execute("SELECT home_team, away_team, spread_line, total_line FROM games WHERE season=? AND week=?", (season, week)):
            sched[home] = (away, True)
            sched[away] = (home, False)
        for mkt in MARKETS:
            _, hist, opp_state, snap_hist, team_regs, h = replay3(con, mkt, src, until=(season, week))
            bst = xgb.Booster()
            bst.load_model(str(model_dir / f"nfl_{mkt}_v2x.json"))
            keys, feats = [], []
            for pid, ph in hist.items():
                team = ph[-1]["team"]
                if team not in sched:
                    continue
                if not v2.serving_opportunity_eligible(ph, mkt):
                    if len(ph) >= 3 and ph[-1]["pos"] in MARKETS[mkt]["positions"]:
                        vol = v2._mean([g["vol"] for g in ph[-3:]])
                        self.blocked[(mkt, norm_name(ph[-1]["name"]), team)] = {
                            "reason": "RECENT_OPPORTUNITY_BELOW_TRAINING_SUPPORT",
                            "last3_opportunities_per_game": float(vol) if vol is not None else None,
                            "training_min_per_game": MARKETS[mkt]["min_last3_vol"]}
                    continue
                opp, is_home = sched[team]
                f = h.features(ph, season, week, team, opp, is_home, opp_state)
                f.update(extras(src, h.cfg, pid, ph[-1]["name"], season, week, team_regs[team], snap_hist))
                feats.append(f)
                keys.append((norm_name(ph[-1]["name"]), team))
            if not feats:
                continue
            X = np.array([[f[c] if f[c] is not None else np.nan for c in FEATURES] for f in feats], dtype=np.float32)
            for k, p in zip(keys, bst.predict(xgb.DMatrix(X, feature_names=FEATURES))):
                self.proj[(mkt,) + k] = max(0.0, float(p))
        self.ok = bool(self.proj)

    def get(self, mkt, player_name, team):
        return self.proj.get((mkt, norm_name(player_name), team))

    def out_of_support(self, mkt, player_name, team):
        return self.blocked.get((mkt, norm_name(player_name), team))


def load_projector(con, season, week, cache_dir=None):
    """The v2x projector with fresh nflverse injury/snap data, or None if anything needed is unavailable (caller falls back to v2)."""
    try:
        cache = cache_dir or os.environ.get("NFL_DATA_DIR", "/tmp/nfl_data")
        src = Sources.fetch([season - 1, season], cache)
        p = Projector3(con, season, week, src)
        return p if p.ok else None
    except Exception as e:
        print(f"  v2x yardage projector unavailable ({e})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="sqlite with player_games/games back to the training start (extended history)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    print("NFL_YARDAGE_V2X\n==============")
    rep = train(a.db, a.data_dir, write=not a.no_write)
    print("PASSED" if rep["passed"] else "FAILED the ship rule")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
