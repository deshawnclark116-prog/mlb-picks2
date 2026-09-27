#!/usr/bin/env python3
"""
CFB_CONTEXT_V2

Context the CFB models never had, built only from real box scores already
in cfb_model.sqlite (no betting-line history exists for free):

  who's playing   vacated carry / reception share of regulars missing from
                  this game's box score, and whether the team's primary
                  passer changed
  role            player's share of team carries / receptions / pass
                  attempts over the team's last 3 games
  pace            team plays (carries + pass attempts) per game, last 3
  team strength   opponent-adjusted power rating (margin-based, blowouts
                  capped, prior season carried over at half weight) for
                  both teams, as of before the game

Every feature uses only games strictly before the one being predicted.
Retrains the four in-season CFB markets on (old features + context) and
scores them with baseline_audit_a.compare() against the same naive
baseline and the currently served model, on the 2025 season.

Run
---
python -u cfb_context_v2.py --db cfb_models/cfb_model.sqlite
"""
import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
MODEL_DIR = REPO / "cfb_models" / "cfb_context_v2_work"
MARGIN_CAP = 28.0
CONTEXT_FEATURES = ["carry_share_last3", "rec_share_last3", "pass_share_last3", "vacated_car",
                    "vacated_rec", "qb_change", "team_plays_last3", "team_rating", "opp_rating",
                    "rating_diff"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def build_ratings(con):
    """(season, team, week) -> rating before that week's games."""
    games = con.execute("""SELECT season, week, home_team, away_team, home_points, away_points, neutral_site
                           FROM games WHERE home_points IS NOT NULL ORDER BY season, week""").fetchall()
    by_season = defaultdict(list)
    for g in games:
        by_season[g[0]].append(g)
    asof = {}
    final = {}
    for season in sorted(by_season):
        prior = {t: 0.5 * r for t, r in final.get(season - 1, {}).items()}
        played = []
        weeks = sorted({g[1] for g in by_season[season]})
        rating = dict(prior)
        for w in weeks:
            for t in {x for g in by_season[season] for x in (g[2], g[3])}:
                asof[(season, t, w)] = rating.get(t, prior.get(t, 0.0))
            played += [g for g in by_season[season] if g[1] == w]
            # iterate: rating = mean(margin + opp rating), shrunk toward prior with 2 pseudo-games
            teams = {x for g in played for x in (g[2], g[3])}
            r = {t: prior.get(t, 0.0) for t in teams}
            for _ in range(30):
                acc = defaultdict(float); n = defaultdict(int)
                for (_, _, h, a, hp, ap, neu) in played:
                    hfa = 0.0 if neu else 2.5
                    m = max(-MARGIN_CAP, min(MARGIN_CAP, hp - ap - hfa))
                    acc[h] += m + r[a]; n[h] += 1
                    acc[a] += -m + r[h]; n[a] += 1
                r = {t: (acc[t] + 2 * prior.get(t, 0.0)) / (n[t] + 2) for t in teams}
                mu = sum(r.values()) / len(r)
                r = {t: v - mu for t, v in r.items()}
            rating = {**prior, **r}
        final[season] = rating
    build_ratings.latest = final
    return asof


def build_context(con):
    """(season, player_id, week) -> context dict for that player's game."""
    rows = con.execute("""SELECT season, week, game_id, game_date, team, player_id, position,
                                 COALESCE(carries,0), COALESCE(receptions,0), COALESCE(pass_attempts,0)
                          FROM player_games ORDER BY season, game_date, game_id""").fetchall()
    games = defaultdict(dict)            # (season, team, game_id) -> {pid: (car, rec, att)}
    order = defaultdict(list)            # (season, team) -> [game_id in date order]
    week_of = {}
    for season, week, gid, gdate, team, pid, pos, car, rec, att in rows:
        k = (season, team, gid)
        if k not in games:
            order[(season, team)].append(gid)
            week_of[(season, gid)] = week
        games[k][pid] = (car, rec, att)
    ctx = {}
    for (season, team), gids in order.items():
        for i, gid in enumerate(gids):
            cur = games[(season, team, gid)]
            prev = gids[max(0, i - 3):i]
            if not prev:
                for pid in cur:
                    ctx[(season, pid, week_of[(season, gid)])] = {}
                continue
            appear = defaultdict(int); car = defaultdict(float); rec = defaultdict(float); att = defaultdict(float)
            tc = tr = ta = 0.0
            for pg in prev:
                g = games[(season, team, pg)]
                for pid, (c, r, a) in g.items():
                    if c + r + a > 0:
                        appear[pid] += 1
                    car[pid] += c; rec[pid] += r; att[pid] += a
                tc += sum(v[0] for v in g.values()); tr += sum(v[1] for v in g.values())
                ta += sum(v[2] for v in g.values())
            regulars = [p for p, n in appear.items() if n >= min(2, len(prev))]
            absent = [p for p in regulars if p not in cur or sum(cur[p]) == 0]
            qb = max(att, key=att.get) if att and max(att.values()) > 0 else None
            base = {
                "vacated_car": sum(car[p] for p in absent) / tc if tc else 0.0,
                "vacated_rec": sum(rec[p] for p in absent) / tr if tr else 0.0,
                "qb_change": 1.0 if qb is not None and qb in absent else 0.0,
                "team_plays_last3": (tc + ta) / len(prev),
            }
            for pid in cur:
                ctx[(season, pid, week_of[(season, gid)])] = {
                    **base,
                    "carry_share_last3": car[pid] / tc if tc else None,
                    "rec_share_last3": rec[pid] / tr if tr else None,
                    "pass_share_last3": att[pid] / ta if ta else None,
                }
    return ctx


def add_context(feat, season, pid, week, team, opp, ctx, ratings):
    f = dict(feat)
    c = ctx.get((season, pid, week), {})
    for k in ("carry_share_last3", "rec_share_last3", "pass_share_last3", "vacated_car",
              "vacated_rec", "qb_change", "team_plays_last3"):
        f[k] = c.get(k)
    tr, orr = ratings.get((season, team, week)), ratings.get((season, opp, week))
    f["team_rating"], f["opp_rating"] = tr, orr
    f["rating_diff"] = (tr - orr) if tr is not None and orr is not None else None
    return f


def mat(feats, cols):
    return np.array([[f.get(c) if f.get(c) is not None else np.nan for c in cols] for f in feats], dtype=np.float32)


def poisson_at_least(lam, k):
    lam = np.maximum(lam, 1e-6)
    cdf = sum(np.exp(-lam) * lam ** i / math.factorial(i) for i in range(k))
    return 1 - cdf


def run(db, eval_seasons=(2025,)):
    import xgboost as xgb
    import cfb_serving_builder_a as sb
    import baseline_audit_a as audit
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    print("building context + ratings ...", flush=True)
    ctx = build_context(con)
    ratings = build_ratings(con)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    seasons = range(2019, max(eval_seasons) + 1)
    results = []
    common = {"max_depth": 4, "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 10, "reg_lambda": 5.0, "seed": 20260927}

    def season_rows(kind, mkt=None):
        out = {}
        for s in seasons:
            if kind == "player":
                eng = sb.SeasonEngine(con, mkt, s) if mkt in sb.MARKETS else sb.AnytimeTouchdownEngine(con, s)
                out[s] = [(add_context(r[5], s, r[0], r[4], r[2], r[3], ctx, ratings), r[6], r[5]) for r in eng.replay()]
            else:
                out[s] = [(add_context(r[3], s, None, r[2], r[0], r[1], ctx, ratings), r[4], r[3])
                          for r in sb.MoneylineEngine(con, s).replay()]
        return out

    specs = [
        ("rushing_yards", "player", sb.MARKETS["rushing_yards"]["features"], "season_avg_rush_yards",
         {"objective": "reg:absoluteerror"}, lambda pred: 1 / (1 + np.exp(-(pred - 69.5) / 20)),
         lambda a: (a > 69.5).astype(float),
         sb.MARKETS["rushing_yards"]["model_dir"] / "cfb_rushing_yards.json"),
        ("passing_touchdowns", "player", sb.MARKETS["passing_touchdowns"]["features"], "season_avg_pass_td",
         {"objective": "count:poisson"}, lambda lam: poisson_at_least(lam, 2),
         lambda a: (a >= 2).astype(float),
         sb.MARKETS["passing_touchdowns"]["model_dir"] / "cfb_passing_touchdowns.json"),
        ("anytime_touchdowns", "player", sb.ANYTIME_TD_FEATURES, "season_avg_total_td",
         {"objective": "count:poisson"}, lambda lam: poisson_at_least(lam, 1),
         lambda a: (a >= 1).astype(float),
         sb.ANYTIME_TD_MODEL_DIR / "cfb_anytime_touchdowns.json"),
        ("moneyline", "team", sb.MONEYLINE_FEATURES, None,
         {"objective": "binary:logistic"}, lambda p: p, lambda a: a.astype(float),
         sb.MONEYLINE_MODEL_DIR / "cfb_moneyline.json"),
    ]
    for mkt, kind, old_cols, naive_col, obj, to_prob, to_y, old_path in specs:
        rows = season_rows(kind, mkt)
        cols = list(old_cols) + [c for c in CONTEXT_FEATURES if c not in old_cols]
        tr = [r for s in range(2019, 2024) for r in rows[s]]
        va = rows[2024]
        label = lambda rr: np.array([r[1] for r in rr], dtype=float)
        dtr = xgb.DMatrix(mat([r[0] for r in tr], cols), label=label(tr), feature_names=cols)
        dva = xgb.DMatrix(mat([r[0] for r in va], cols), label=label(va), feature_names=cols)
        bst = xgb.train({**common, **obj}, dtr, 3000, evals=[(dva, "val")], early_stopping_rounds=80,
                        verbose_eval=False)
        it = (0, bst.best_iteration + 1)
        pred = lambda rr: to_prob(bst.predict(xgb.DMatrix(mat([r[0] for r in rr], cols), feature_names=cols),
                                              iteration_range=it))
        old = xgb.Booster(); old.load_model(str(old_path))
        old_pred = lambda rr: old.predict(xgb.DMatrix(mat([r[2] for r in rr], old_cols), feature_names=old_cols),
                                          iteration_range=(0, old.best_iteration + 1))
        if naive_col:
            naive = lambda rr: [r[2][naive_col] for r in rr]
        else:
            naive = lambda rr: [r[2]["projected_margin"] + (0 if r[2]["is_neutral_site"] else
                                                          (2.5 if r[2]["is_home"] else -2.5)) for r in rr]
        for s in eval_seasons:
            ev = rows[s]
            y_cal, y_ev = to_y(label(va)), to_y(label(ev))
            print(f"\n{mkt} [{s}]")
            r_new = audit.compare(f"cfb {mkt} NEW (context)", pred(va), naive(va), y_cal, pred(ev), naive(ev), y_ev)
            r_old = audit.compare(f"cfb {mkt} served (old)", old_pred(va), naive(va), y_cal, old_pred(ev), naive(ev), y_ev)
            gain_vs_old = r_old["model"]["logloss"] - r_new["model"]["logloss"]
            print(f"  new vs served: logloss {r_new['model']['logloss']:.4f} vs {r_old['model']['logloss']:.4f} "
                  f"(gain {gain_vs_old:+.4f}), AUC {r_new['model']['auc']:.3f} vs {r_old['model']['auc']:.3f}")
            results.append({"market": mkt, "season": s, "new": r_new, "served": r_old,
                            "logloss_gain_vs_served": round(gain_vs_old, 5)})
        imp = bst.get_score(importance_type="gain")
        print("  top features:", sorted(imp, key=imp.get, reverse=True)[:8])
        bst.save_model(str(MODEL_DIR / f"cfb_{mkt}_context_candidate.json"))
        (MODEL_DIR / f"cfb_{mkt}_context_candidate_columns.json").write_text(json.dumps(cols))
    (MODEL_DIR / "report.json").write_text(json.dumps(results, indent=2, default=str))
    return results


SPEC_KIND = {"rushing_yards": ("reg:absoluteerror", "yards", 69.5),
             "passing_touchdowns": ("count:poisson", "atleast", 2),
             "anytime_touchdowns": ("count:poisson", "atleast", 1),
             "moneyline": ("binary:logistic", "prob", None)}


def to_prob(mkt, raw):
    _, kind, k = SPEC_KIND[mkt]
    raw = np.asarray(raw, dtype=float)
    if kind == "yards":
        return 1 / (1 + np.exp(-(raw - k) / 20))
    if kind == "atleast":
        return poisson_at_least(raw, k)
    return raw


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def finalize(db, calib_season=2025):
    """Write production context models: trained 2019-2024 (early-stopped on
    nothing newer than training), probabilities Platt-calibrated on the
    real, corrected calib_season results."""
    import baseline_audit_a as audit
    import xgboost as xgb
    import cfb_serving_builder_a as sb
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    ctx, ratings = build_context(con), build_ratings(con)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    meta = {}
    for mkt, (obj, _, _) in SPEC_KIND.items():
        cand = MODEL_DIR / f"cfb_{mkt}_context_candidate.json"
        cols = json.loads((MODEL_DIR / f"cfb_{mkt}_context_candidate_columns.json").read_text())
        bst = xgb.Booster(); bst.load_model(str(cand))
        if mkt == "moneyline":
            rows = [(add_context(r[3], calib_season, None, r[2], r[0], r[1], ctx, ratings), r[4])
                    for r in sb.MoneylineEngine(con, calib_season).replay()]
            y = np.array([r[1] for r in rows], dtype=float)
        else:
            eng = sb.SeasonEngine(con, mkt, calib_season) if mkt in sb.MARKETS else sb.AnytimeTouchdownEngine(con, calib_season)
            rows = [(add_context(r[5], calib_season, r[0], r[4], r[2], r[3], ctx, ratings), r[6]) for r in eng.replay()]
            a = np.array([r[1] for r in rows], dtype=float)
            _, kind, k = SPEC_KIND[mkt]
            y = (a > k).astype(float) if kind == "yards" else (a >= k).astype(float)
        raw = bst.predict(xgb.DMatrix(mat([r[0] for r in rows], cols), feature_names=cols),
                          iteration_range=(0, bst.best_iteration + 1))
        x = _logit(to_prob(mkt, raw))
        # 1-D Platt on the logit, returned as plain (a, b) for serving
        a_, b_ = 1.0, 0.0
        for _ in range(100):
            pp = 1 / (1 + np.exp(-(a_ * x + b_)))
            w = pp * (1 - pp) + 1e-9
            g = np.array([np.sum((pp - y) * x), np.sum(pp - y)])
            H = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]])
            st = np.linalg.solve(H + 1e-6 * np.eye(2), g); a_, b_ = a_ - st[0], b_ - st[1]
            if np.abs(st).max() < 1e-9:
                break
        bst.save_model(str(MODEL_DIR / f"cfb_{mkt}_context.json"))
        (MODEL_DIR / f"cfb_{mkt}_context_columns.json").write_text(json.dumps(cols))
        meta[mkt] = {"platt_a": float(a_), "platt_b": float(b_), "calibration_season": calib_season,
                     "n_calibration": len(rows), "best_iteration": int(bst.best_iteration)}
        print(f"  {mkt}: calibrated on {len(rows)} real {calib_season} rows  a={a_:.3f} b={b_:+.3f}")
    (MODEL_DIR / "production.json").write_text(json.dumps(meta, indent=2))
    con.close()


class ServingContext:
    """Context features for an upcoming week + production scoring."""

    def __init__(self, con, season, week):
        import xgboost as xgb
        self.ok = (MODEL_DIR / "production.json").exists()
        if not self.ok:
            return
        self.meta = json.loads((MODEL_DIR / "production.json").read_text())
        self.xgb = xgb
        self.models, self.cols = {}, {}
        for mkt in self.meta:
            b = xgb.Booster(); b.load_model(str(MODEL_DIR / f"cfb_{mkt}_context.json"))
            self.models[mkt] = b
            self.cols[mkt] = json.loads((MODEL_DIR / f"cfb_{mkt}_context_columns.json").read_text())
        build_ratings(con)
        self.ratings = build_ratings.latest.get(season, {})
        rows = con.execute("""SELECT game_id, game_date, week, team, player_id,
                                     COALESCE(carries,0), COALESCE(receptions,0), COALESCE(pass_attempts,0)
                              FROM player_games WHERE season=? AND week<? ORDER BY game_date, game_id""",
                           (season, week)).fetchall()
        games = defaultdict(dict); order = defaultdict(list)
        for gid, gdate, wk, team, pid, c, r, a in rows:
            if gid not in games[team]:
                order[team].append(gid)
            games[team].setdefault(gid, {})[pid] = (c, r, a)
        self.player_ctx, self.team_ctx = {}, {}
        for team, gids in order.items():
            prev = gids[-3:]
            tc = sum(v[0] for g in prev for v in games[team][g].values())
            tr = sum(v[1] for g in prev for v in games[team][g].values())
            ta = sum(v[2] for g in prev for v in games[team][g].values())
            self.team_ctx[team] = {"team_plays_last3": (tc + ta) / len(prev),
                                   "vacated_car": 0.0, "vacated_rec": 0.0, "qb_change": 0.0}
            per = defaultdict(lambda: [0.0, 0.0, 0.0])
            for g in prev:
                for pid, (c, r, a) in games[team][g].items():
                    per[pid][0] += c; per[pid][1] += r; per[pid][2] += a
            for pid, (c, r, a) in per.items():
                self.player_ctx[pid] = {"carry_share_last3": c / tc if tc else None,
                                        "rec_share_last3": r / tr if tr else None,
                                        "pass_share_last3": a / ta if ta else None}

    def features(self, feat, pid, team, opp):
        f = dict(feat)
        f.update(self.team_ctx.get(team, {}))
        f.update(self.player_ctx.get(pid, {}) if pid is not None else {})
        tr, orr = self.ratings.get(team, 0.0), self.ratings.get(opp, 0.0)
        f.update({"team_rating": tr, "opp_rating": orr, "rating_diff": tr - orr})
        return f

    def prob(self, mkt, feats):
        cols = self.cols[mkt]; b = self.models[mkt]; m = self.meta[mkt]
        raw = b.predict(self.xgb.DMatrix(mat(feats, cols), feature_names=cols),
                        iteration_range=(0, m["best_iteration"] + 1))
        return 1 / (1 + np.exp(-(m["platt_a"] * _logit(to_prob(mkt, raw)) + m["platt_b"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "cfb_models" / "cfb_model.sqlite"))
    ap.add_argument("--eval-seasons", type=int, nargs="+", default=[2025])
    ap.add_argument("--finalize", action="store_true",
                    help="after evaluating, write production models calibrated on 2025")
    args = ap.parse_args()
    print("CFB_CONTEXT_V2\n==============")
    run(args.db, tuple(args.eval_seasons))
    if args.finalize:
        finalize(args.db)


if __name__ == "__main__":
    raise SystemExit(main())
