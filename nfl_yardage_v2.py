#!/usr/bin/env python3
"""
NFL_YARDAGE_V2

Context-aware rushing/receiving yardage projection. Built because
baseline_audit_a.py showed the existing simulator (which only reshuffles a
player's own past carries/targets) misses by 26.7 yds vs 27.2 for plain
"last season's average" -- no real edge.

Adds what the simulator never looked at:
  - Vegas spread and implied team total (game script: favorites run late,
    underdogs throw; high totals mean more plays and yards)
  - opponent yards allowed per game this season (prior season early on)
  - the player's share of team carries/targets over his last 3 games
    (role changes show up here weeks before season averages move)
  - recent-3 / season-to-date / prior-season yards, volume and efficiency

Ship rule: MAE must significantly beat BOTH naive baselines (last-season
average and season-blended average) on 2025, and must not be significantly
worse than them on the live 2026 season (too few games yet to prove a small
edge there). If it fails, train() refuses to write a model and serving
keeps the old simulator.

Run
---
python -u nfl_yardage_v2.py --db /tmp/nfl_audit.sqlite    # train + gate
"""
import argparse
import json
import re
import sqlite3
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
MODEL_DIR = REPO / "nfl_models" / "nfl_yardage_v2_work"

MARKETS = {
    "rushing_yards": {"vol": "carries", "yds": "rushing_yards", "positions": ("RB", "FB"),
                       "min_last3_vol": 5.0},
    "receiving_yards": {"vol": "targets", "yds": "receiving_yards", "positions": ("WR", "TE", "RB"),
                         "min_last3_vol": 3.0},
}
FEATURES = [
    "cur_n", "cur_avg_yds", "cur_avg_vol", "prior_n", "prior_avg_yds", "prior_avg_vol",
    "last3_avg_yds", "last3_avg_vol", "last3_share", "eff_16",
    "team_spread", "implied_team_total", "opp_allowed_pg", "opp_allowed_n", "is_home",
    "blend_yds",
]
PARAMS = {"objective": "reg:absoluteerror", "max_depth": 4, "eta": 0.04, "subsample": 0.8,
          "colsample_bytree": 0.8, "min_child_weight": 10, "reg_lambda": 5.0, "seed": 20260927}


def norm_name(n):
    n = unicodedata.normalize("NFKD", n or "")
    n = "".join(c for c in n if not unicodedata.combining(c)).lower()
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", n)
    n = re.sub(r"[^a-z ]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


class History:
    """Replays player_games in (season, week) order so every feature only
    ever uses games strictly before the one being predicted."""

    def __init__(self, con, mkt):
        cfg = MARKETS[mkt]
        self.cfg = cfg
        self.rows = con.execute(f"""
            SELECT player_id, player_name, position, team, opponent, season, week, game_id, is_home,
                   COALESCE({cfg['vol']}, 0), COALESCE({cfg['yds']}, 0)
            FROM player_games WHERE season_type = 'REG' ORDER BY season, week
        """).fetchall()
        self.lines = {}
        for gid, season, week, home, away, spread, total in con.execute(
                "SELECT game_id, season, week, home_team, away_team, spread_line, total_line FROM games"):
            self.lines[(season, week, home)] = (spread, total, True)
            self.lines[(season, week, away)] = (-spread if spread is not None else None, total, False)
        self.team_vol = defaultdict(float)
        for r in self.rows:
            self.team_vol[(r[7], r[3])] += r[9]

    def features(self, ph, season, week, team, opp, is_home, opp_state):
        f = self._features(ph, season, week, team, opp, is_home, opp_state)
        f["blend_yds"] = naive_preds(f)[1]
        return f

    def _features(self, ph, season, week, team, opp, is_home, opp_state):
        cur = [g for g in ph if g["season"] == season]
        prior = [g for g in ph if g["season"] == season - 1]
        last3 = ph[-3:]
        last16 = ph[-16:]
        vol16 = sum(g["vol"] for g in last16)
        share = [g["vol"] / g["team_vol"] for g in last3 if g["team_vol"] > 0]
        spread, total, _ = self.lines.get((season, week, team), (None, None, None))
        cur_opp = opp_state.get((season, opp))
        prv_opp = opp_state.get((season - 1, opp))
        if cur_opp and cur_opp[1] >= 3:
            opp_pg, opp_n = cur_opp[0] / cur_opp[1], cur_opp[1]
        elif prv_opp and prv_opp[1] > 0:
            n_c = cur_opp[1] if cur_opp else 0
            tot = (cur_opp[0] if cur_opp else 0) + prv_opp[0] / prv_opp[1] * 3
            opp_pg, opp_n = tot / (n_c + 3), n_c
        else:
            opp_pg, opp_n = None, 0
        return {
            "cur_n": len(cur), "cur_avg_yds": _mean([g["yds"] for g in cur]),
            "cur_avg_vol": _mean([g["vol"] for g in cur]),
            "prior_n": len(prior), "prior_avg_yds": _mean([g["yds"] for g in prior]),
            "prior_avg_vol": _mean([g["vol"] for g in prior]),
            "last3_avg_yds": _mean([g["yds"] for g in last3]),
            "last3_avg_vol": _mean([g["vol"] for g in last3]),
            "last3_share": _mean(share),
            "eff_16": (sum(g["yds"] for g in last16) / vol16) if vol16 > 0 else None,
            "team_spread": spread,
            "implied_team_total": (total / 2 + spread / 2) if (spread is not None and total is not None) else None,
            "opp_allowed_pg": opp_pg, "opp_allowed_n": opp_n,
            "is_home": 1.0 if is_home else 0.0,
        } | {"blend_yds": None}

    def replay(self, until=None):
        """Completed player-games -> (meta, features, actual). Also returns
        the end state (player histories, opponent-allowed tallies) so the
        same pass can feature a future week."""
        cfg = self.cfg
        hist = defaultdict(list)
        opp_state = {}
        out = []
        by_week = defaultdict(list)
        for r in self.rows:
            by_week[(r[5], r[6])].append(r)
        for key in sorted(by_week):
            if until and key >= until:
                break
            wk = by_week[key]
            for (pid, name, pos, team, opp, season, week, gid, is_home, vol, yds) in wk:
                ph = hist[pid]
                if pos in cfg["positions"] and len(ph) >= 3:
                    last3_vol = _mean([g["vol"] for g in ph[-3:]])
                    if last3_vol >= cfg["min_last3_vol"]:
                        feat = self.features(ph, season, week, team, opp, is_home, opp_state)
                        out.append(({"player_id": pid, "name": name, "team": team, "opp": opp,
                                     "season": season, "week": week}, feat, float(yds)))
            for (pid, name, pos, team, opp, season, week, gid, is_home, vol, yds) in wk:
                hist[pid].append({"season": season, "week": week, "vol": vol, "yds": yds,
                                  "team_vol": self.team_vol[(gid, team)], "team": team, "name": name,
                                  "pos": pos})
                st = opp_state.setdefault((season, opp), [0.0, 0])
                st[0] += yds
            games_seen = {(r[5], r[4], r[7]) for r in wk}
            for (season, opp, gid) in games_seen:
                opp_state[(season, opp)][1] += 1
        return out, hist, opp_state


def naive_preds(feat):
    last = next(v for v in (feat["prior_avg_yds"], feat["cur_avg_yds"], feat["last3_avg_yds"])
                if v is not None)
    w = min(1.0, (feat["cur_n"] or 0) / 8)
    cur = feat["cur_avg_yds"] if feat["cur_avg_yds"] is not None else last
    blend = w * cur + (1 - w) * (last if last is not None else cur)
    return last, blend


def to_matrix(rows):
    return np.array([[r[1][c] if r[1][c] is not None else np.nan for c in FEATURES] for r in rows],
                    dtype=np.float32)


def train(db):
    import xgboost as xgb
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    all_ok = True
    for mkt in MARKETS:
        rows, _, _ = History(con, mkt).replay()
        tr = [r for r in rows if r[0]["season"] == 2023
              or (r[0]["season"] == 2024 and r[0]["week"] <= 12)]
        va = [r for r in rows if r[0]["season"] == 2024 and r[0]["week"] > 12]
        ev = {2025: [r for r in rows if r[0]["season"] == 2025],
              2026: [r for r in rows if r[0]["season"] == 2026]}
        dtr = xgb.DMatrix(to_matrix(tr), label=[r[2] for r in tr], feature_names=FEATURES)
        dva = xgb.DMatrix(to_matrix(va), label=[r[2] for r in va], feature_names=FEATURES)
        bst = xgb.train(PARAMS, dtr, num_boost_round=2000, evals=[(dva, "val")],
                        early_stopping_rounds=60, verbose_eval=False)
        n_rounds = bst.best_iteration + 1
        mkt_rep = {"n_train": len(tr), "n_val": len(va), "rounds": n_rounds}
        ratios = []
        for season, er in ev.items():
            pred = bst.predict(xgb.DMatrix(to_matrix(er), feature_names=FEATURES),
                               iteration_range=(0, n_rounds))
            act = np.array([r[2] for r in er])
            nl = np.array([naive_preds(r[1])[0] for r in er], dtype=float)
            nb = np.array([naive_preds(r[1])[1] for r in er], dtype=float)
            ok = ~np.isnan(nl) & ~np.isnan(nb)
            mae_m = float(np.abs(pred[ok] - act[ok]).mean())
            mae_l = float(np.abs(nl[ok] - act[ok]).mean())
            mae_b = float(np.abs(nb[ok] - act[ok]).mean())
            rng = np.random.default_rng(1)
            d = (np.abs(nb[ok] - act[ok]) - np.abs(pred[ok] - act[ok]))
            boot = d[rng.integers(0, len(d), (2000, len(d)))].mean(axis=1)
            p = float((boot <= 0).mean())
            p_worse = float((boot >= 0).mean())
            # 2025 (n in the thousands): must significantly beat the best naive.
            # 2026 (live season, only weeks 2+ have 3 prior games -- n~100-300,
            # too small to prove a ~3% edge): must not be significantly WORSE.
            passed = ((mae_m < min(mae_l, mae_b) and p < 0.05) if season == 2025
                      else p_worse >= 0.05)
            all_ok &= passed
            mkt_rep[str(season)] = {"n": int(ok.sum()), "model_mae": round(mae_m, 2),
                                    "naive_last_season_mae": round(mae_l, 2),
                                    "naive_blend_mae": round(mae_b, 2),
                                    "improvement_vs_best_naive_pct": round((min(mae_l, mae_b) - mae_m) / min(mae_l, mae_b) * 100, 1),
                                    "p_not_better_than_blend": round(p, 4), "p_worse_than_blend": round(p_worse, 4), "passed": passed}
            ratios += list(act[ok] / np.maximum(pred[ok], 1.0))
            print(f"  {mkt:16s} {season}: n={ok.sum():5d}  MAE model {mae_m:5.1f}  "
                  f"last-season {mae_l:5.1f}  blend {mae_b:5.1f}  p={p:.3f}  {'PASS' if passed else 'FAIL'}")
        r = np.sort(np.array(ratios))
        q = lambda x: float(r[int(x * (len(r) - 1))])
        mkt_rep["actual_over_projection_quantiles"] = {k: round(q(v), 3) for k, v in
                                                       (("p10", .10), ("p15", .15), ("p25", .25),
                                                        ("p50", .50), ("p75", .75), ("p90", .90))}
        # Final model: refit on every season with the early-stopped round count.
        full = [r for r in rows if 2023 <= r[0]["season"] <= 2025]
        dfull = xgb.DMatrix(to_matrix(full), label=[r[2] for r in full], feature_names=FEATURES)
        final = xgb.train(PARAMS, dfull, num_boost_round=n_rounds)
        mkt_rep["_final"] = final
        report[mkt] = mkt_rep
    con.close()
    out = {"features": FEATURES, "markets": {m: {k: v for k, v in r.items() if k != "_final"}
                                             for m, r in report.items()}, "passed": bool(all_ok)}
    if all_ok:
        for mkt, r in report.items():
            r["_final"].save_model(str(MODEL_DIR / f"nfl_{mkt}_v2.json"))
        (MODEL_DIR / "report.json").write_text(json.dumps(out, indent=2))
        print("\nPASSED on 2025 and 2026 for both markets -- models written")
    else:
        (MODEL_DIR / "report_failed.json").write_text(json.dumps(out, indent=2))
        print("\nFAILED the ship rule -- no model written")
    return out


class Projector:
    """Serving-side: v2 projections for one upcoming week, keyed by
    (normalized player name, team)."""

    def __init__(self, con, season, week):
        import xgboost as xgb
        self.ok = (MODEL_DIR / "report.json").exists()
        self.proj = {}
        if not self.ok:
            return
        rep = json.loads((MODEL_DIR / "report.json").read_text())
        self.quant = {m: rep["markets"][m]["actual_over_projection_quantiles"] for m in MARKETS}
        sched = {}
        for home, away, spread, total in con.execute(
                "SELECT home_team, away_team, spread_line, total_line FROM games WHERE season=? AND week=?",
                (season, week)):
            sched[home] = (away, True)
            sched[away] = (home, False)
        for mkt in MARKETS:
            h = History(con, mkt)
            _, hist, opp_state = h.replay(until=(season, week))
            bst = xgb.Booster(); bst.load_model(str(MODEL_DIR / f"nfl_{mkt}_v2.json"))
            keys, feats = [], []
            for pid, ph in hist.items():
                if len(ph) < 3 or ph[-1]["pos"] not in MARKETS[mkt]["positions"]:
                    continue
                team = ph[-1]["team"]
                if team not in sched:
                    continue
                opp, is_home = sched[team]
                feats.append(h.features(ph, season, week, team, opp, is_home, opp_state))
                keys.append((norm_name(ph[-1]["name"]), team))
            if not feats:
                continue
            X = np.array([[f[c] if f[c] is not None else np.nan for c in FEATURES] for f in feats],
                         dtype=np.float32)
            preds = bst.predict(xgb.DMatrix(X, feature_names=FEATURES))
            for k, p in zip(keys, preds):
                self.proj[(mkt,) + k] = max(0.0, float(p))

    def get(self, mkt, player_name, team):
        return self.proj.get((mkt, norm_name(player_name), team))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    args = ap.parse_args()
    print("NFL_YARDAGE_V2\n==============")
    train(args.db)


if __name__ == "__main__":
    raise SystemExit(main())
