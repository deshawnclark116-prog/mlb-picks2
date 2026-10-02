"""
NFL_YARDAGE_V5

Predicts the number: rushing yards and receiving yards per player per
game, scored on how close the number lands to what he actually does
(average miss, and share of players within 10 / 20 yards).

What's new over v2/v3: the workload read knows who was on the field.

  depl_last3      how much of the team's usual workload (carries or
                  targets) was missing through injury/rest in each of the
                  player's last 3 games -- a backup's big games with the
                  starter out are recognised as such
  depl_now        same for this week (who is out now)
  returning_share usual workload of teammates who missed some of the
                  player's last 3 games and are back this week (Tank
                  Bigsby 2026-09-28: 3 games with Barkley out, Barkley back)
  share_full8     the player's share in his last 8 games with the team at
                  full strength, and how many such games (n_full8)
  typ_share       his usual share (last 6 games played)
  exp_vol_full    share_full8 x the team's usual volume per game

Then a size correction fitted on late 2024 (never used for training trees)
so featured players aren't pulled toward the middle: the tree's number is
mapped to the median real outcome of players given that number.

Ship rule: on 2025 (held out) average miss must be lower than the live
v2 model's on the same player-games, significantly (paired bootstrap); on
live 2026 games not significantly worse.

Run
---
  python nfl_yardage_v5.py --data-dir /tmp/nfl_data
"""
import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

import nfl_yardage_v3 as v3

REPO = Path(__file__).resolve().parent
MODEL_DIR = REPO / "nfl_models" / "nfl_yardage_v5_work"
REG_MIN = 0.08          # usual share that makes a teammate a regular
REG_WINDOW = 6          # a regular must have played within this many team games

NEW_FEATURES = ["depl_last3", "depl_now", "returning_share", "share_full8", "n_full8",
                "typ_share", "team_vol_pg", "exp_vol_full", "share_vs_full"]
FEATURES = v3.FEATURES + NEW_FEATURES
PARAMS = {**v3.PARAMS}


class TeamUsage:
    """Per team, per game: everyone's share of the market's volume, who
    played, and each player's usual share going into that game."""

    def __init__(self, data, vol):
        self.d = data
        self.games = {}        # team -> list of (s, w)
        self.share = {}        # (s, w, team) -> pid -> share
        self.played = {}       # (s, w, team) -> set(normname)
        self.typ = {}          # (s, w, team) -> pid -> usual share before this game
        self.last_idx = {}     # (s, w, team) -> pid -> index of last game played before this one
        self.names = {}        # pid -> normname
        self.team_vol = {}     # (s, w, team) -> team volume that game
        for team, tg in data.team_games.items():
            tg = [g for g in tg if (g[0], g[1], team) in data.stats]
            self.games[team] = tg
            recent = defaultdict(lambda: deque(maxlen=6))
            last = {}
            for i, (s, w) in enumerate(tg):
                key = (s, w, team)
                rows = data.stats[key]
                tvol = sum(r[vol] for r in rows.values())
                self.team_vol[key] = tvol
                played = data.played_names(key)
                self.played[key] = played
                self.typ[key] = {q: sum(v) / len(v) for q, v in recent.items() if v}
                self.last_idx[key] = dict(last)
                sh = {}
                for pid, r in rows.items():
                    nn = v3.norm_name(r["name"]); self.names[pid] = nn
                    sh[pid] = r[vol] / tvol if tvol else 0.0
                    if nn in played:
                        recent[pid].append(sh[pid]); last[pid] = i
                self.share[key] = sh
            self.idx = getattr(self, "idx", {})
            for i, g in enumerate(tg):
                self.idx[(g[0], g[1], team)] = i

    def regulars(self, team, key_before, idx_now):
        """{pid: usual share} of regulars as of the game `key_before`."""
        typ = self.typ.get(key_before, {}); last = self.last_idx.get(key_before, {})
        return {q: t for q, t in typ.items() if t >= REG_MIN and idx_now - last.get(q, -99) <= REG_WINDOW}

    def depleted(self, team, key, exclude):
        """Usual share of regulars who did not play in game `key`."""
        i = self.idx[key]
        pl = self.played[key]
        return sum(t for q, t in self.regulars(team, key, i).items()
                   if q != exclude and self.names.get(q) not in pl)

    def state_now(self, team, s, w, pid, last3_keys, playing_fn):
        """depl_now / returning_share for the upcoming game at (s, w)."""
        tg = self.games[team]
        prev = [g for g in tg if g < (s, w)]
        if not prev:
            return None, None
        # usual shares going into (s, w): roll forward from the last game
        lk = (prev[-1][0], prev[-1][1], team)
        typ = dict(self.typ.get(lk, {})); last = dict(self.last_idx.get(lk, {}))
        i_last = self.idx[lk]
        rec = defaultdict(list)
        for g in prev[-6:]:
            k = (g[0], g[1], team)
            for q, sh in self.share[k].items():
                if self.names.get(q) in self.played[k]:
                    rec[q].append(sh); last[q] = self.idx[k]
        for q, v in rec.items():
            typ[q] = sum(v[-6:]) / len(v[-6:])
        i_now = i_last + 1
        regs = {q: t for q, t in typ.items() if t >= REG_MIN and i_now - last.get(q, -99) <= REG_WINDOW and q != pid}
        depl_now = sum(t for q, t in regs.items() if not playing_fn(self.names.get(q), q))
        ret = 0.0
        for q, t in regs.items():
            if not playing_fn(self.names.get(q), q) or not last3_keys:
                continue
            missed = sum(1 for k in last3_keys if self.names.get(q) not in self.played[k])
            ret += t * missed / len(last3_keys)
        return depl_now, ret


class Replayer(v3.Replayer):
    def __init__(self, data, mkt, usage):
        super().__init__(data, mkt)
        self.u = usage

    def absorb_week(self, s, w):
        n_before = {pid: len(h) for pid, h in self.hist.items()}
        super().absorb_week(s, w)
        for pid, h in self.hist.items():
            if len(h) > n_before.get(pid, 0):
                g = h[-1]
                key = (g["s"], g["w"], g["team"])
                g["depl"] = self.u.depleted(g["team"], key, pid) if key in self.u.idx else None

    def v5_features(self, pid, s, w, team, ctx, tctx, playing_fn):
        f = self.player_features(pid, s, w, team, ctx, tctx)
        ph = self.hist[pid]
        last3 = ph[-3:]
        m = v3._mean
        f["depl_last3"] = m([g.get("depl") for g in last3])
        # the team's games during his last 3 (same team only)
        tg = [g for g in self.u.games[team] if g < (s, w)]
        l3_keys = [(g[0], g[1], team) for g in tg[-3:]]
        dn, ret = self.u.state_now(team, s, w, pid, l3_keys, playing_fn)
        f["depl_now"] = dn; f["returning_share"] = ret
        full = [g for g in ph[-8:] if g.get("depl") is not None and g["depl"] < REG_MIN and g["team"] == team]
        f["share_full8"] = m([g["share"] for g in full]); f["n_full8"] = float(len(full))
        f["typ_share"] = m([g["share"] for g in ph[-6:]])
        tv = [self.u.team_vol[(g[0], g[1], team)] for g in tg[-6:]]
        f["team_vol_pg"] = m(tv)
        f["exp_vol_full"] = (f["share_full8"] * f["team_vol_pg"]) if f["share_full8"] is not None and f["team_vol_pg"] else None
        f["share_vs_full"] = (f["last3_share"] - f["share_full8"]) if f["last3_share"] is not None and f["share_full8"] is not None else None
        return f

    def replay_v5(self, until=None):
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
                pf = lambda nn, q: nn in played
                for pid, r in rows.items():
                    if not self.eligible(pid) or v3.norm_name(r["name"]) not in played:
                        continue
                    f = self.v5_features(pid, s, w, team, ctx, tctx, pf)
                    out.append(({"pid": pid, "name": r["name"], "team": team, "s": s, "w": w},
                                f, r[self.cfg["yds"]]))
            self.absorb_week(s, w)
        return out


def mat(rows, cols):
    return np.array([[r[1].get(c) if r[1].get(c) is not None else np.nan for c in cols] for r in rows],
                    dtype=np.float32)


def fit_size_map(pred, act, n_bins=12):
    """Monotone map: tree number -> median real outcome at that number."""
    qs = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
    xs, ys = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        msk = (pred >= lo) & (pred <= hi)
        if msk.sum() >= 15:
            xs.append(float(np.median(pred[msk]))); ys.append(float(np.median(act[msk])))
    ys = np.maximum.accumulate(np.array(ys))
    return [float(x) for x in xs], [float(y) for y in ys]


def apply_size_map(pred, sm):
    xs, ys = sm
    out = np.interp(pred, xs, ys)
    # beyond the fitted range keep the tree's slope instead of flattening
    hi = pred > xs[-1]
    out[hi] = ys[-1] + (pred[hi] - xs[-1])
    return np.maximum(out, 0.0)


def closeness(pred, act):
    e = np.abs(pred - act)
    return {"n": int(len(e)), "avg_miss": round(float(e.mean()), 2),
            "within_10": round(float((e <= 10).mean()), 3), "within_20": round(float((e <= 20).mean()), 3),
            "p80_miss": round(float(np.percentile(e, 80)), 1)}


def boot_p(e_new, e_ref, seed=1):
    d = e_ref - e_new
    rng = np.random.default_rng(seed)
    bt = d[rng.integers(0, len(d), (2000, len(d)))].mean(axis=1)
    return float((bt <= 0).mean()), float((bt >= 0).mean())


def train(data_dir):
    import xgboost as xgb
    data = v3.Data(data_dir, [2022, 2023, 2024, 2025, 2026])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report, finals, maps, ok = {}, {}, {}, True
    for mkt in v3.MARKETS:
        usage = TeamUsage(data, v3.MARKETS[mkt]["vol"])
        rows = Replayer(data, mkt, usage).replay_v5()
        tr = [r for r in rows if r[0]["s"] == 2023 or (r[0]["s"] == 2024 and r[0]["w"] <= 12)]
        va = [r for r in rows if r[0]["s"] == 2024 and r[0]["w"] > 12]
        y = lambda rr: np.array([r[2] for r in rr], dtype=float)
        dtr = xgb.DMatrix(mat(tr, FEATURES), label=y(tr), feature_names=FEATURES)
        dva = xgb.DMatrix(mat(va, FEATURES), label=y(va), feature_names=FEATURES)
        bst = xgb.train(PARAMS, dtr, 3000, evals=[(dva, "val")], early_stopping_rounds=80, verbose_eval=False)
        nr = bst.best_iteration + 1
        pv = bst.predict(dva, iteration_range=(0, nr))
        sm = fit_size_map(pv, y(va))
        rep = {"n_train": len(tr), "n_val": len(va), "rounds": nr, "size_map": sm}
        print(f"\n{mkt}: train {len(tr)}  val {len(va)}  rounds {nr}")
        for season in (2025, 2026):
            er = [r for r in rows if r[0]["s"] == season]
            a = y(er)
            raw = bst.predict(xgb.DMatrix(mat(er, FEATURES), feature_names=FEATURES), iteration_range=(0, nr))
            adj = apply_size_map(raw, sm)
            v2p = v3.v2_mae(er, mkt)
            naive = np.array([v3.naive_preds(r[1])[1] for r in er])
            res = {"v5_raw": closeness(raw, a), "v5": closeness(adj, a), "v2_live": closeness(v2p, a),
                   "naive": closeness(naive, a)}
            best = min(("v5_raw", raw), ("v5", adj), key=lambda t: np.abs(t[1] - a).mean())
            p_nb, p_w = boot_p(np.abs(adj - a), np.abs(v2p - a))
            res["p_v5_not_better_than_v2"] = p_nb; res["p_v5_worse_than_v2"] = p_w
            # featured players: the ones the old model under-called
            big = v2p >= 60
            if big.sum() >= 20:
                res["featured_60plus"] = {"n": int(big.sum()), "actual_median": float(np.median(a[big])),
                                          "v2_median_proj": float(np.median(v2p[big])),
                                          "v5_median_proj": float(np.median(adj[big]))}
            passed = (p_nb < 0.05) if season == 2025 else (p_w >= 0.05)
            ok &= passed
            res["passed"] = passed
            rep[str(season)] = res
            print(f"  [{season}] n={len(er)}  avg miss  v5 {res['v5']['avg_miss']:5.2f}  raw {res['v5_raw']['avg_miss']:5.2f}"
                  f"  v2 {res['v2_live']['avg_miss']:5.2f}  naive {res['naive']['avg_miss']:5.2f}   "
                  f"within20 v5 {res['v5']['within_20']:.0%} v2 {res['v2_live']['within_20']:.0%}   "
                  f"p(!better)={p_nb:.3f}  {'PASS' if passed else 'FAIL'}")
            if "featured_60plus" in res:
                fz = res["featured_60plus"]
                print(f"      featured (v2>=60): actual median {fz['actual_median']:.0f}  v2 {fz['v2_median_proj']:.0f}  v5 {fz['v5_median_proj']:.0f}")
        imp = bst.get_score(importance_type="gain")
        rep["top_features"] = sorted(imp, key=imp.get, reverse=True)[:12]
        print("  top:", rep["top_features"])
        full = [r for r in rows if 2023 <= r[0]["s"] <= 2025]
        finals[mkt] = xgb.train(PARAMS, xgb.DMatrix(mat(full, FEATURES), label=y(full), feature_names=FEATURES), nr)
        maps[mkt] = sm
        report[mkt] = rep
    out = {"features": FEATURES, "markets": report, "passed": bool(ok)}
    name = "report.json" if ok else "report_failed.json"
    if ok:
        for mkt, b in finals.items():
            b.save_model(str(MODEL_DIR / f"nfl_{mkt}_v5.json"))
    (MODEL_DIR / name).write_text(json.dumps(out, indent=2, default=float))
    print("\nPASSED -- v5 written" if ok else "\nFAILED ship rule -- nothing written")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    print("NFL_YARDAGE_V5\n==============")
    train(args.data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
