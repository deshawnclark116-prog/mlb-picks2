#!/usr/bin/env python3
"""
HITTER_MARKET_CONTEXT_CHALLENGER_B

Generalized context challenger for the count hitter markets that the calibration
audit exposed as miscalibrated: batter_total_bases (over 1.5) and batter_rbi
(over 0.5). Predictions-first: no odds.

The champions are regression -> Poisson models, and the audit showed them badly
miscalibrated (TB ECE ~0.065, anti-predictive at the top). This tests the same
fix that made batter_hits honest:
  - reframe as a direct binary:logistic classifier of P(over the line), and
  - add the zero-train/serve-skew context features.

Three arms on the untouched 2026 holdout (2025 = train w/ internal val):
  champion        : deployed regression model (batter_<market>.json) -> prob_over
  base_logistic   : champion's base features, binary:logistic (isolates the reframe)
  challenger      : base + zero-skew context, binary:logistic (adds sharpness)

Base features (champion's, strict D-1, now incl. runs_per_pa after backfill):
  season_avg, tb_per_pa, rbi_per_pa, runs_per_pa, hr_rate, bb_rate, so_rate,
  recent5_target, recent15_target, batting_order, games_played
Zero-skew context: platoon_advantage, pitcher_is_R, is_home, expected_pa_v1,
  recent_xbh_avg

Reports AUC, log-loss, Brier, ECE, reliability per arm; the headline is whether
ECE collapses vs the champion. Read-only. No production change.

Run (Render)
------------
python -u hitter_market_context_challenger_b.py --market tb  2>&1 | tee /data/hr_model/hitter_market_context_challenger_b_tb.log
python -u hitter_market_context_challenger_b.py --market rbi 2>&1 | tee /data/hr_model/hitter_market_context_challenger_b_rbi.log
"""

import argparse
import json
import math
import sqlite3
import sys
from collections import deque
from pathlib import Path

import numpy as np

import hits_feature_discovery_b as fd  # platoon_advantage, metrics, PARAMS, NAN, fnum

SOURCE = "/data/hr_model/hr_model.sqlite"
MODEL_DIR = Path("/data/models")

MARKETS = {
    "tb":  {"model": "batter_total_bases", "line": 1.5, "target": "tb"},
    "rbi": {"model": "batter_rbi",          "line": 0.5, "target": "rbi"},
}
BASE = ["season_avg", "tb_per_pa", "rbi_per_pa", "runs_per_pa", "hr_rate",
        "bb_rate", "so_rate", "recent5_target", "recent15_target",
        "batting_order", "games_played"]
CONTEXT = ["platoon_advantage", "pitcher_is_R", "is_home", "expected_pa_v1", "recent_xbh_avg"]
MIN_PRIOR_AB = 20
MIN_PRIOR_GAMES = 5


def poisson_cdf(k, lam):
    if lam <= 0:
        return 1.0
    s, term = 0.0, math.exp(-lam)
    for i in range(k + 1):
        if i:
            term *= lam / i
        s += term
    return min(s, 1.0)


def prob_over(expected, line):
    return 1 - poisson_cdf(int(math.floor(line)), max(expected, 1e-6))


def build(conn, target_key, line):
    cols = ["game_id", "game_date", "batter_id", "lineup_spot", "batter_hand",
            "pitcher_hand", "opposing_pitcher_id", "side", "plate_appearances",
            "at_bats", "hits", "doubles", "triples", "home_runs", "rbi", "runs",
            "walks", "strikeouts"]
    rows = conn.execute(
        f"SELECT {','.join(cols)} FROM batter_games WHERE at_bats IS NOT NULL "
        f"ORDER BY batter_id, game_date, game_id").fetchall()
    ix = {c: i for i, c in enumerate(cols)}

    def N(r, k):
        v = r[ix[k]]
        return v if v is not None else 0

    def tgt(r):
        if target_key == "tb":
            return N(r, "hits") + N(r, "doubles") + 2 * N(r, "triples") + 3 * N(r, "home_runs")
        return N(r, target_key)  # rbi

    # expected_pa lookup (spot|side)
    lk = {}
    for sp, sd, avg in conn.execute(
            """SELECT lineup_spot, side, AVG(plate_appearances) FROM batter_games
               WHERE lineup_spot BETWEEN 1 AND 9 AND side IN ('home','away')
               AND plate_appearances IS NOT NULL GROUP BY lineup_spot, side"""):
        lk[f"{int(sp)}|{sd}"] = round(float(avg), 4)

    thr = math.floor(line) + 1
    feat = {}
    cur = None
    group = []

    def flush(group):
        cum = dict(pa=0, ab=0, h=0, hr=0, bb=0, so=0, tb=0, rbi=0, runs=0)
        rec_t = deque(maxlen=15); rec_xbh = deque(maxlen=15)
        n_prior = 0; i = 0
        for g in group:
            gd = g[ix["game_date"]]
            while i < len(group) and group[i][ix["game_date"]] < gd:
                h = group[i]
                tb = N(h, "hits") + N(h, "doubles") + 2 * N(h, "triples") + 3 * N(h, "home_runs")
                cum["pa"] += N(h, "plate_appearances"); cum["ab"] += N(h, "at_bats")
                cum["h"] += N(h, "hits"); cum["hr"] += N(h, "home_runs")
                cum["bb"] += N(h, "walks"); cum["so"] += N(h, "strikeouts")
                cum["tb"] += tb; cum["rbi"] += N(h, "rbi"); cum["runs"] += N(h, "runs")
                rec_t.append(tgt(h))
                rec_xbh.append(N(h, "doubles") + N(h, "triples") + N(h, "home_runs"))
                n_prior += 1; i += 1
            if n_prior < MIN_PRIOR_GAMES or cum["ab"] < MIN_PRIOR_AB:
                continue
            pa = cum["pa"] or 1
            r5 = list(rec_t)[-5:]; r15 = list(rec_t)
            side = g[ix["side"]]
            spot = g[ix["lineup_spot"]]
            f = {
                "season_avg": cum["h"] / cum["ab"] if cum["ab"] else 0.0,
                "tb_per_pa": cum["tb"] / pa, "rbi_per_pa": cum["rbi"] / pa,
                "runs_per_pa": cum["runs"] / pa, "hr_rate": cum["hr"] / pa,
                "bb_rate": cum["bb"] / pa, "so_rate": cum["so"] / pa,
                "recent5_target": sum(r5) / len(r5) if r5 else 0.0,
                "recent15_target": sum(r15) / len(r15) if r15 else 0.0,
                "batting_order": spot if spot is not None else 9,
                "games_played": n_prior,
                "platoon_advantage": fd.platoon_advantage(g[ix["batter_hand"]], g[ix["pitcher_hand"]]),
                "pitcher_is_R": 1.0 if g[ix["pitcher_hand"]] == "R" else (0.0 if g[ix["pitcher_hand"]] in ("L", "S") else fd.NAN),
                "is_home": 1.0 if side == "home" else (0.0 if side == "away" else fd.NAN),
                "expected_pa_v1": lk.get(f"{int(spot)}|{side}", fd.NAN) if spot is not None else fd.NAN,
                "recent_xbh_avg": sum(rec_xbh) / len(rec_xbh) if rec_xbh else 0.0,
            }
            feat[(g[ix["game_id"]], g[ix["batter_id"]])] = {
                "season": gd[:4], "game_date": gd, "f": f,
                "y": 1 if tgt(g) >= thr else 0}
        return

    for r in rows:
        if r[ix["batter_id"]] != cur:
            if group:
                flush(group)
            group = []; cur = r[ix["batter_id"]]
        group.append(r)
    if group:
        flush(group)

    data = {"2025": [], "2026": []}
    for rec in feat.values():
        if rec["season"] in data:
            data[rec["season"]].append(rec)
    return data, lk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=list(MARKETS), required=True)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--produce", action="store_true",
                    help="also train the challenger on ALL data and export deployable artifacts")
    ap.add_argument("--workdir", default="/data/hr_model/hitter_market_context_challenger_b_work")
    args = ap.parse_args()
    import xgboost as xgb
    m = MARKETS[args.market]
    line, tkey = m["line"], m["target"]

    con = sqlite3.connect(f"file:{args.source}?mode=ro", uri=True)
    print(f"HITTER_MARKET_CONTEXT_CHALLENGER_B  [{args.market}]  (over {line})", flush=True)
    print("=" * 60, flush=True)
    data, lk = build(con, tkey, line)
    dev, hol = data["2025"], data["2026"]
    print(f"  2025={len(dev)}  2026={len(hol)}  expected_pa lookup={len(lk)}", flush=True)

    dates = sorted({r["game_date"] for r in dev})
    cut = dates[int(len(dates) * 0.8)]
    tr = [r for r in dev if r["game_date"] < cut]
    va = [r for r in dev if r["game_date"] >= cut]

    def mat(rows, feats):
        X = np.array([[r["f"].get(k, fd.NAN) for k in feats] for r in rows], dtype=np.float32)
        y = np.array([r["y"] for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=feats)

    def train_score(feats):
        b = xgb.train(fd.PARAMS, mat(tr, feats), num_boost_round=800,
                      evals=[(mat(va, feats), "val")], early_stopping_rounds=40,
                      verbose_eval=False)
        p = b.predict(mat(hol, feats), iteration_range=(0, b.best_iteration + 1))
        return fd.metrics(list(map(float, p)), [r["y"] for r in hol])

    arms = {}
    # champion reference: deployed regression model -> prob_over
    cmp_mp = MODEL_DIR / f"{m['model']}.json"
    cmp_cp = MODEL_DIR / f"{m['model']}_columns.json"
    if cmp_mp.exists() and cmp_cp.exists():
        ccols = json.loads(cmp_cp.read_text())
        cb = xgb.Booster(); cb.load_model(str(cmp_mp))
        X = np.array([[r["f"].get(c, fd.NAN) for c in ccols] for r in hol], dtype=np.float32)
        proj = cb.predict(xgb.DMatrix(X, feature_names=ccols))
        arms["champion(reg->poisson)"] = fd.metrics([prob_over(float(p), line) for p in proj],
                                                    [r["y"] for r in hol])
    print("\ntraining base_logistic ...", flush=True)
    arms["base_logistic"] = train_score(BASE)
    print("training challenger (base+context) ...", flush=True)
    arms["challenger"] = train_score(BASE + CONTEXT)

    print(f"\n============ 2026 HOLDOUT [{args.market}] ============")
    print(f"  {'arm':24s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    for name, r in arms.items():
        print(f"  {name:24s} {r['auc']:.4f}  {r['log_loss']:.5f}  {r['brier']:.5f}  {r['ece']:.4f}")

    ch = arms["challenger"]
    print("\nchallenger reliability (pred -> actual):")
    for b in ch["reliability"]:
        print(f"   {b['bin']}  n={b['n']:>6}  pred={b['pred']:.3f}  actual={b['actual']:.3f}")

    champ = arms.get("champion(reg->poisson)")
    if champ:
        print(f"\ncalibration fix: champion ECE {champ['ece']:.4f} -> challenger ECE {ch['ece']:.4f} "
              f"({'IMPROVED' if ch['ece'] < champ['ece'] else 'not improved'})")
        print(f"discrimination: champion AUC {champ['auc']:.4f} -> challenger AUC {ch['auc']:.4f}")

    produced = None
    if args.produce:
        # train the challenger (base+context) on ALL data with a frozen spec,
        # exported as <model>_context.json for deployment (same discipline as hits).
        allrows = dev + hol
        adates = sorted({r["game_date"] for r in allrows})
        acut = adates[int(len(adates) * 0.9)]
        atr = [r for r in allrows if r["game_date"] < acut]
        ava = [r for r in allrows if r["game_date"] >= acut]
        feats = BASE + CONTEXT
        final = xgb.train(fd.PARAMS, mat(atr, feats), num_boost_round=800,
                          evals=[(mat(ava, feats), "val")], early_stopping_rounds=40,
                          verbose_eval=False)
        work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
        stem = f"{m['model']}_context"
        final.save_model(str(work / f"{stem}.json"))
        (work / f"{stem}_columns.json").write_text(json.dumps(feats))
        produced = {"model": str(work / f"{stem}.json"),
                    "columns": str(work / f"{stem}_columns.json"),
                    "trained_rows": len(allrows), "best_iteration": final.best_iteration,
                    "line": line}
        print(f"\nPRODUCED (all-data challenger, not yet deployed to /data/models):")
        print(f"  {produced['model']}")
        print(f"  {produced['columns']}")
        print(f"  (expected_pa_lookup.json already deployed with hits; reused here)")

    report = {"script": "HITTER_MARKET_CONTEXT_CHALLENGER_B", "market": args.market,
              "line": line, "arms": arms, "produced": produced}
    con.close()
    out = Path("/data/hr_model") if Path("/data/hr_model").exists() else Path.cwd()
    (out / f"hitter_market_context_challenger_b_{args.market}_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport written. Read-only. No production model or code changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
