#!/usr/bin/env python3
"""
NFL_YARDAGE_V3

Rushing/receiving yardage projections that use the information a real
handicapper checks, not just a player's own averages:

  who's playing    vacated target/carry share of teammates who are out,
                   starting QB out, player returning from missed games
  role             snap share, target share, air-yards share, WOPR
                   (last 3 games) -- role changes show up here weeks
                   before season averages move
  game context     Vegas spread + implied team total, team pace (plays per
                   game), wind / temperature / dome
  matchup          opponent yards allowed per game to this stat
  history          season-to-date, last-3, prior-season yards/volume,
                   per-touch efficiency

Sources (all free, nflverse releases): weekly player stats (incl. target
share / air yards / WOPR), PFR snap counts, schedules (lines + weather).
Training marks a teammate "out" when he'd played 2 of his team's last 3
games and took zero offensive snaps this game; serving uses ESPN's live
injury status (Out / Doubtful / IR / Suspension / PUP) for the same thing.

Ship rule (train() refuses to write models otherwise), per market:
  2025 (thousands of games): MAE must significantly beat the best naive
        average AND beat v2
  2026 (live season so far, small n): must not be significantly worse
        than the best naive average

Run
---
python -u nfl_yardage_v3.py --data-dir /tmp/nflcsv       # train + gate
"""
import argparse
import csv
import json
import re
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
MODEL_DIR = REPO / "nfl_models" / "nfl_yardage_v3_work"
RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
UNAVAILABLE = {"out", "doubtful", "injured reserve", "suspension", "suspended", "pup",
               "physically unable to perform", "reserve/covid-19", "non-football injury",
               "commissioner exempt list"}

MARKETS = {
    "rushing_yards": {"vol": "carries", "yds": "rushing_yards", "positions": ("RB", "FB"),
                       "min_last3_vol": 5.0},
    "receiving_yards": {"vol": "targets", "yds": "receiving_yards", "positions": ("WR", "TE", "RB"),
                         "min_last3_vol": 3.0},
}
FEATURES = [
    "cur_n", "cur_avg_yds", "cur_avg_vol", "prior_n", "prior_avg_yds", "prior_avg_vol",
    "last3_avg_yds", "last3_avg_vol", "last3_share", "eff_16", "blend_yds",
    "snap_pct_last3", "tgt_share_last3", "air_share_last3", "wopr_last3",
    "vacated_tgt", "vacated_car", "qb_out", "returning", "team_plays_last3",
    "team_spread", "implied_team_total", "opp_allowed_pg", "is_home",
    "wind", "temp", "dome",
]
PARAMS = {"objective": "reg:absoluteerror", "max_depth": 5, "eta": 0.03, "subsample": 0.8,
          "colsample_bytree": 0.8, "min_child_weight": 15, "reg_lambda": 5.0, "seed": 20260927}


def norm_name(n):
    n = unicodedata.normalize("NFKD", n or "")
    n = "".join(c for c in n if not unicodedata.combining(c)).lower()
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", n)
    n = re.sub(r"[^a-z ]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def ensure_data(data_dir, seasons, refresh=()):
    """Downloads any missing season files; always re-downloads seasons in
    `refresh` (the live season) and the schedule (lines move all week)."""
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    want = [("games.csv", f"{RELEASE}/schedules/games.csv", True)]
    for y in seasons:
        live = y in refresh
        want += [(f"stats_player_week_{y}.csv", f"{RELEASE}/stats_player/stats_player_week_{y}.csv", live),
                 (f"snap_counts_{y}.csv", f"{RELEASE}/snap_counts/snap_counts_{y}.csv", live)]
    for name, url, force in want:
        p = data_dir / name
        if p.exists() and not force:
            continue
        req = urllib.request.Request(url, headers={"User-Agent": "nfl-yardage-v3"})
        with urllib.request.urlopen(req, timeout=120) as r:
            p.write_bytes(r.read())


class Data:
    def __init__(self, data_dir, seasons):
        data_dir = Path(data_dir)
        self.seasons = seasons
        self.games = {}          # (season, week, team) -> context
        self.team_games = defaultdict(list)
        for r in csv.DictReader(open(data_dir / "games.csv", newline="", encoding="utf-8")):
            if r["game_type"] != "REG" or int(r["season"]) not in seasons:
                continue
            s, w = int(r["season"]), int(r["week"])
            spread, total = _f(r["spread_line"]), _f(r["total_line"])
            dome = 1.0 if (r.get("roof") or "") in ("dome", "closed") else 0.0
            wx = {"wind": _f(r.get("wind")), "temp": _f(r.get("temp")), "dome": dome}
            played = r["result"] not in ("", "NA")
            for team, opp, home in ((r["home_team"], r["away_team"], True), (r["away_team"], r["home_team"], False)):
                ts = spread if home or spread is None else -spread
                self.games[(s, w, team)] = {"opp": opp, "is_home": home, "spread": ts, "total": total,
                                            "played": played, **wx}
                self.team_games[team].append((s, w))
        for t in self.team_games:
            self.team_games[t].sort()

        self.stats = defaultdict(dict)   # (s, w, team) -> pid -> row
        for s in seasons:
            p = data_dir / f"stats_player_week_{s}.csv"
            if not p.exists():
                continue
            for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
                if r.get("season_type", "REG") != "REG":
                    continue
                self.stats[(s, int(r["week"]), r["team"])][r["player_id"]] = {
                    "pid": r["player_id"], "name": r["player_display_name"], "pos": r["position"],
                    "carries": _f(r["carries"]) or 0.0, "rushing_yards": _f(r["rushing_yards"]) or 0.0,
                    "targets": _f(r["targets"]) or 0.0, "receiving_yards": _f(r["receiving_yards"]) or 0.0,
                    "attempts": _f(r["attempts"]) or 0.0, "target_share": _f(r["target_share"]),
                    "air_yards_share": _f(r["air_yards_share"]), "wopr": _f(r["wopr"]),
                }
        self.snaps = defaultdict(dict)   # (s, w, team) -> normname -> offense_pct
        for s in seasons:
            p = data_dir / f"snap_counts_{s}.csv"
            if not p.exists():
                continue
            for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
                if r.get("game_type", "REG") != "REG":
                    continue
                snaps = _f(r["offense_snaps"]) or 0.0
                if snaps > 0:
                    self.snaps[(s, int(r["week"]), r["team"])][norm_name(r["player"])] = _f(r["offense_pct"])

    def played_names(self, key):
        names = set(self.snaps.get(key, {}))
        for row in self.stats.get(key, {}).values():
            if row["carries"] + row["targets"] + row["attempts"] > 0:
                names.add(norm_name(row["name"]))
        return names

    def team_context(self, team, s, w, absent_fn):
        """Vacated usage / QB-out / pace for team's game at (s, w), from the
        team's previous 3 games. absent_fn(normname) -> bool decides who's
        out this game (real snaps in training, ESPN status in serving)."""
        prev = [g for g in self.team_games[team] if g < (s, w) and (g[0], g[1], team) in self.stats][-3:]
        if not prev:
            return {"vacated_tgt": None, "vacated_car": None, "qb_out": None, "team_plays_last3": None}
        appear = defaultdict(int); tgt = defaultdict(float); car = defaultdict(float); att = defaultdict(float)
        names = {}
        team_tgt = team_car = team_plays = 0.0
        for g in prev:
            rows = self.stats[(g[0], g[1], team)]
            played = self.played_names((g[0], g[1], team))
            gt = sum(r["targets"] for r in rows.values()); gc = sum(r["carries"] for r in rows.values())
            team_tgt += gt; team_car += gc
            team_plays += gc + sum(r["attempts"] for r in rows.values())
            for pid, r in rows.items():
                nn = norm_name(r["name"]); names[pid] = nn
                if nn in played:
                    appear[pid] += 1
                tgt[pid] += r["targets"]; car[pid] += r["carries"]; att[pid] += r["attempts"]
        regulars = [pid for pid, n in appear.items() if n >= min(2, len(prev))]
        vac_t = sum(tgt[p] for p in regulars if absent_fn(names[p])) / team_tgt if team_tgt else 0.0
        vac_c = sum(car[p] for p in regulars if absent_fn(names[p])) / team_car if team_car else 0.0
        qb = max(att, key=att.get) if att and max(att.values()) > 0 else None
        return {"vacated_tgt": vac_t, "vacated_car": vac_c,
                "qb_out": 1.0 if qb is not None and absent_fn(names[qb]) else 0.0,
                "team_plays_last3": team_plays / len(prev)}


def naive_preds(f):
    last = next((v for v in (f["prior_avg_yds"], f["cur_avg_yds"], f["last3_avg_yds"]) if v is not None), 0.0)
    w = min(1.0, (f["cur_n"] or 0) / 8)
    cur = f["cur_avg_yds"] if f["cur_avg_yds"] is not None else last
    return last, w * cur + (1 - w) * last


class Replayer:
    def __init__(self, data, mkt):
        self.d = data; self.mkt = mkt; self.cfg = MARKETS[mkt]
        self.hist = defaultdict(list)       # pid -> list of game dicts
        self.opp = {}                       # (season, opp) -> [yds allowed, games]
        self.team_vol = {}

    def player_features(self, pid, s, w, team, ctx, tctx):
        ph = self.hist[pid]; c = self.cfg
        cur = [g for g in ph if g["s"] == s]; prior = [g for g in ph if g["s"] == s - 1]
        last3, last16 = ph[-3:], ph[-16:]
        vol16 = sum(g["vol"] for g in last16)
        tg = self.d.team_games[team]
        gap = 0
        if ph:
            last_played = (ph[-1]["s"], ph[-1]["w"])
            gap = sum(1 for g in tg if last_played < g < (s, w))
        cur_o = self.opp.get((s, ctx["opp"])); prv_o = self.opp.get((s - 1, ctx["opp"]))
        opp_n = cur_o[1] if cur_o else 0
        if cur_o and cur_o[1] >= 3:
            opp_pg = cur_o[0] / cur_o[1]
        elif prv_o and prv_o[1]:
            n_c = cur_o[1] if cur_o else 0
            opp_pg = ((cur_o[0] if cur_o else 0) + prv_o[0] / prv_o[1] * 3) / (n_c + 3)
        else:
            opp_pg = None
        f = {
            "cur_n": len(cur), "cur_avg_yds": _mean([g["yds"] for g in cur]),
            "cur_avg_vol": _mean([g["vol"] for g in cur]),
            "prior_n": len(prior), "prior_avg_yds": _mean([g["yds"] for g in prior]),
            "prior_avg_vol": _mean([g["vol"] for g in prior]),
            "last3_avg_yds": _mean([g["yds"] for g in last3]), "last3_avg_vol": _mean([g["vol"] for g in last3]),
            "last3_share": _mean([g["share"] for g in last3]),
            "eff_16": (sum(g["yds"] for g in last16) / vol16) if vol16 > 0 else None,
            "snap_pct_last3": _mean([g["snap"] for g in last3]),
            "tgt_share_last3": _mean([g["tshare"] for g in last3]),
            "air_share_last3": _mean([g["ashare"] for g in last3]),
            "wopr_last3": _mean([g["wopr"] for g in last3]),
            "returning": 1.0 if gap >= 1 else 0.0,
            "team_spread": ctx["spread"],
            "implied_team_total": (ctx["total"] / 2 + ctx["spread"] / 2)
                                  if ctx["spread"] is not None and ctx["total"] is not None else None,
            "opp_allowed_pg": opp_pg, "opp_allowed_n": opp_n, "is_home": 1.0 if ctx["is_home"] else 0.0,
            "wind": ctx["wind"], "temp": ctx["temp"], "dome": ctx["dome"],
            **tctx,
        }
        f["blend_yds"] = naive_preds(f)[1]
        return f

    def eligible(self, pid):
        ph = self.hist[pid]
        return (len(ph) >= 3 and ph[-1]["pos"] in self.cfg["positions"]
                and _mean([g["vol"] for g in ph[-3:]]) >= self.cfg["min_last3_vol"])

    def absorb_week(self, s, w):
        c = self.cfg
        for team in list(self.d.team_games):
            key = (s, w, team)
            rows = self.d.stats.get(key)
            if not rows:
                continue
            ctx = self.d.games.get(key)
            tvol = sum(r[c["vol"]] for r in rows.values())
            snaps = self.d.snaps.get(key, {})
            for pid, r in rows.items():
                self.hist[pid].append({
                    "s": s, "w": w, "vol": r[c["vol"]], "yds": r[c["yds"]], "pos": r["pos"],
                    "name": r["name"], "team": team, "share": (r[c["vol"]] / tvol) if tvol else None,
                    "snap": snaps.get(norm_name(r["name"])), "tshare": r["target_share"],
                    "ashare": r["air_yards_share"], "wopr": r["wopr"],
                })
            if ctx:
                st = self.opp.setdefault((s, ctx["opp"]), [0.0, 0])
                st[0] += sum(r[c["yds"]] for r in rows.values()); st[1] += 1

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
                for pid, r in rows.items():
                    if not self.eligible(pid) or norm_name(r["name"]) not in played:
                        continue
                    f = self.player_features(pid, s, w, team, ctx, tctx)
                    out.append(({"pid": pid, "name": r["name"], "team": team, "s": s, "w": w},
                                f, r[self.cfg["yds"]]))
            self.absorb_week(s, w)
        return out


def to_matrix(rows):
    return np.array([[r[1].get(c) if r[1].get(c) is not None else np.nan for c in FEATURES] for r in rows],
                    dtype=np.float32)


def v2_mae(rows, mkt):
    """v2's error on the same rows, for the 'must beat v2' rule."""
    import xgboost as xgb
    import nfl_yardage_v2 as v2
    path = v2.MODEL_DIR / f"nfl_{mkt}_v2.json"
    if not path.exists():
        return None
    bst = xgb.Booster(); bst.load_model(str(path))
    X = np.array([[r[1].get(c) if r[1].get(c) is not None else np.nan for c in v2.FEATURES] for r in rows],
                 dtype=np.float32)
    return bst.predict(xgb.DMatrix(X, feature_names=v2.FEATURES))


def train(data_dir):
    import xgboost as xgb
    data = Data(data_dir, [2022, 2023, 2024, 2025, 2026])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report, finals, all_ok = {}, {}, True
    for mkt in MARKETS:
        rows = Replayer(data, mkt).replay()
        tr = [r for r in rows if r[0]["s"] == 2023 or (r[0]["s"] == 2024 and r[0]["w"] <= 12)]
        va = [r for r in rows if r[0]["s"] == 2024 and r[0]["w"] > 12]
        dtr = xgb.DMatrix(to_matrix(tr), label=[r[2] for r in tr], feature_names=FEATURES)
        dva = xgb.DMatrix(to_matrix(va), label=[r[2] for r in va], feature_names=FEATURES)
        bst = xgb.train(PARAMS, dtr, 3000, evals=[(dva, "val")], early_stopping_rounds=80, verbose_eval=False)
        n_rounds = bst.best_iteration + 1
        rep = {"n_train": len(tr), "n_val": len(va), "rounds": n_rounds}
        ratios, preds_all, acts_all = [], [], []
        for season in (2025, 2026):
            er = [r for r in rows if r[0]["s"] == season]
            pred = bst.predict(xgb.DMatrix(to_matrix(er), feature_names=FEATURES), iteration_range=(0, n_rounds))
            act = np.array([r[2] for r in er], dtype=float)
            nl = np.array([naive_preds(r[1])[0] for r in er]); nb = np.array([naive_preds(r[1])[1] for r in er])
            best_naive = nb if np.abs(nb - act).mean() <= np.abs(nl - act).mean() else nl
            mae_m = float(np.abs(pred - act).mean())
            mae_n = float(np.abs(best_naive - act).mean())
            d = np.abs(best_naive - act) - np.abs(pred - act)
            rng = np.random.default_rng(1)
            boot = d[rng.integers(0, len(d), (2000, len(d)))].mean(axis=1)
            p_not_better, p_worse = float((boot <= 0).mean()), float((boot >= 0).mean())
            v2p = v2_mae(er, mkt)
            mae_v2 = float(np.abs(v2p - act).mean()) if v2p is not None else None
            if season == 2025:
                passed = mae_m < mae_n and p_not_better < 0.05 and (mae_v2 is None or mae_m < mae_v2)
            else:
                passed = p_worse >= 0.05
            all_ok &= passed
            rep[str(season)] = {"n": len(er), "model_mae": round(mae_m, 2), "best_naive_mae": round(mae_n, 2),
                                "v2_mae": round(mae_v2, 2) if mae_v2 else None,
                                "improvement_vs_naive_pct": round((mae_n - mae_m) / mae_n * 100, 1),
                                "p_not_better": round(p_not_better, 4), "passed": passed}
            print(f"  {mkt:16s} {season}: n={len(er):5d}  MAE v3 {mae_m:5.1f}  v2 {mae_v2 or float('nan'):5.1f}  "
                  f"best naive {mae_n:5.1f}  ({(mae_n - mae_m) / mae_n * 100:+.1f}%)  p={p_not_better:.3f}  "
                  f"{'PASS' if passed else 'FAIL'}")
            ratios += list(act / np.maximum(pred, 1.0)); preds_all += list(pred); acts_all += list(act)
        # Real floors by projection size, for alt-line decisions.
        preds_all, acts_all = np.array(preds_all), np.array(acts_all)
        rep["hit_rates_by_projection"] = {}
        for lo, hi in ((40, 60), (60, 75), (75, 90), (90, 250)):
            m = (preds_all >= lo) & (preds_all < hi)
            if m.sum() >= 20:
                rep["hit_rates_by_projection"][f"{lo}-{hi}"] = {
                    "n": int(m.sum()),
                    **{f"{t}+": round(float((acts_all[m] >= t).mean()), 3) for t in (25, 40, 50, 60, 70, 80, 90, 100)}}
        r_ = np.sort(np.array(ratios)); q = lambda x: float(r_[int(x * (len(r_) - 1))])
        rep["actual_over_projection_quantiles"] = {k: round(q(v), 3) for k, v in
                                                   (("p10", .1), ("p25", .25), ("p50", .5), ("p75", .75), ("p90", .9))}
        # importance, so it's visible what the model actually leans on
        imp = bst.get_score(importance_type="gain")
        rep["top_features"] = sorted(imp, key=imp.get, reverse=True)[:10]
        full = [r for r in rows if 2023 <= r[0]["s"] <= 2025]
        finals[mkt] = xgb.train(PARAMS, xgb.DMatrix(to_matrix(full), label=[r[2] for r in full],
                                                     feature_names=FEATURES), n_rounds)
        report[mkt] = rep
    out = {"features": FEATURES, "markets": report, "passed": bool(all_ok)}
    if all_ok:
        for mkt, b in finals.items():
            b.save_model(str(MODEL_DIR / f"nfl_{mkt}_v3.json"))
        (MODEL_DIR / "report.json").write_text(json.dumps(out, indent=2))
        print("\nPASSED -- v3 models written")
    else:
        (MODEL_DIR / "report_failed.json").write_text(json.dumps(out, indent=2))
        print("\nFAILED the ship rule -- no model written")
    return out


# --------------------------------------------------------------- serving
ESPN_ROSTER = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{slug}/roster"
NFLVERSE_TO_ESPN = {"LA": "lar", "WAS": "wsh"}


def espn_unavailable(teams):
    """{team: set(normalized names)} of players ESPN lists as not playing."""
    import requests
    out = {}
    for t in teams:
        names = set()
        try:
            r = requests.get(ESPN_ROSTER.format(slug=NFLVERSE_TO_ESPN.get(t, t).lower()), timeout=20)
            r.raise_for_status()
            for grp in r.json().get("athletes", []):
                for it in grp.get("items", []):
                    st = {(i.get("status") or "").strip().lower() for i in it.get("injuries", [])}
                    if st & UNAVAILABLE or grp.get("position") in ("injuredReserveOrOut", "suspended"):
                        names.add(norm_name(it.get("displayName")))
        except Exception as e:
            print(f"    v3: injury fetch failed for {t}: {e}")
        out[t] = names
    return out


class Projector:
    def __init__(self, data_dir, season, week):
        import xgboost as xgb
        self.ok = (MODEL_DIR / "report.json").exists()
        self.proj, self.detail = {}, {}
        if not self.ok:
            return
        rep = json.loads((MODEL_DIR / "report.json").read_text())
        self.report = rep["markets"]
        data = Data(data_dir, [season - 1, season])
        sched = {t: c for (s, w, t), c in data.games.items() if s == season and w == week}
        out_now = espn_unavailable(sched)
        for mkt in MARKETS:
            rp = Replayer(data, mkt)
            rp.replay(until=(season, week))
            bst = xgb.Booster(); bst.load_model(str(MODEL_DIR / f"nfl_{mkt}_v3.json"))
            tctx = {t: data.team_context(t, season, week, lambda nn, t=t: nn in out_now.get(t, set()))
                    for t in sched}
            keys, feats = [], []
            for pid, ph in rp.hist.items():
                if not ph or not rp.eligible(pid):
                    continue
                team = ph[-1]["team"]
                if team not in sched or norm_name(ph[-1]["name"]) in out_now.get(team, set()):
                    continue
                f = rp.player_features(pid, season, week, team, sched[team], tctx[team])
                keys.append((norm_name(ph[-1]["name"]), team)); feats.append(f)
            if not feats:
                continue
            X = np.array([[f.get(c) if f.get(c) is not None else np.nan for c in FEATURES] for f in feats],
                         dtype=np.float32)
            for k, p, f in zip(keys, bst.predict(xgb.DMatrix(X, feature_names=FEATURES)), feats):
                self.proj[(mkt,) + k] = max(0.0, float(p))
                self.detail[(mkt,) + k] = {"vacated_tgt": f["vacated_tgt"], "vacated_car": f["vacated_car"],
                                           "qb_out": f["qb_out"], "snap_pct_last3": f["snap_pct_last3"],
                                           "tgt_share_last3": f["tgt_share_last3"]}

    def get(self, mkt, player_name, team):
        return self.proj.get((mkt, norm_name(player_name), team))

    def hit_rates(self, mkt, proj):
        for band, rates in self.report[mkt]["hit_rates_by_projection"].items():
            lo, hi = map(float, band.split("-"))
            if lo <= proj < hi:
                return rates
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    print("NFL_YARDAGE_V3\n==============")
    if args.download:
        ensure_data(args.data_dir, [2022, 2023, 2024, 2025, 2026], refresh=(2026,))
    train(args.data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
