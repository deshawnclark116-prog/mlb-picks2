#!/usr/bin/env python3
"""
HR_CONTEXT_CHALLENGER_A

Builds a real, calibrated batter_home_runs model to replace the hardcoded
~0.358/0.342 tier constant (which the reality check showed is 2-4x too high).
Predictions-first: no odds.

Unlike the other hitter markets, HR is park- and matchup-driven, so the feature
set is HR-specific and includes park factor and the opposing pitcher's
HR-allowed rate. Target: P(home_runs >= 1), line 0.5.

BASE (power, strict D-1 from batter_games)
    hr_rate, tb_per_pa, season_slg, iso, recent5_hr, recent15_hr, recent_xbh_avg,
    so_rate, bb_rate, batting_order, games_played
CONTEXT (HR-relevant, strict D-1)
    platoon_advantage, pitcher_is_R, is_home, expected_pa_v1,
    park_factor_by_batter_hand, opp_pitcher_hr_per_pa, opp_pitcher_hr_pa_seen

Arms on the untouched 2026 holdout (2025 = train w/ internal val):
  base_logistic, challenger (base + context). The current production HR "model"
  is a constant, so its calibration is reported as the fixed-0.35 reference.

Reports base rate, AUC, log-loss, Brier, ECE, reliability. Headline: does a real
model produce HONEST HR probabilities (ECE low, ranking real) vs the ~0.35 lie.

Read-only. No production change.

Run (Render)
------------
python -u hr_context_challenger_a.py 2>&1 | tee /data/hr_model/hr_context_challenger_a.log
"""

import json
import math
import sqlite3
import sys
from collections import deque
from pathlib import Path

import numpy as np

import hits_feature_discovery_b as fd  # platoon_advantage, metrics, PARAMS, NAN

SOURCE = "/data/hr_model/hr_model.sqlite"
MODEL_CONSTANT_ELITE = 0.358
MIN_PRIOR_AB = 40
MIN_PRIOR_GAMES = 10

BASE = ["hr_rate", "tb_per_pa", "season_slg", "iso", "recent5_hr", "recent15_hr",
        "recent_xbh_avg", "so_rate", "bb_rate", "batting_order", "games_played"]
CONTEXT = ["platoon_advantage", "pitcher_is_R", "is_home", "expected_pa_v1",
           "park_factor_by_batter_hand", "opp_pitcher_hr_per_pa", "opp_pitcher_hr_pa_seen"]


def fnum(v):
    return float(v) if v is not None else fd.NAN


def build(conn):
    cols = ["game_id", "game_date", "batter_id", "lineup_spot", "batter_hand",
            "pitcher_hand", "opposing_pitcher_id", "side", "park_factor_by_batter_hand",
            "plate_appearances", "at_bats", "hits", "doubles", "triples", "home_runs",
            "walks", "strikeouts"]
    rows = conn.execute(
        f"SELECT {','.join(cols)} FROM batter_games WHERE at_bats IS NOT NULL "
        f"ORDER BY batter_id, game_date, game_id").fetchall()
    ix = {c: i for i, c in enumerate(cols)}

    def N(r, k):
        v = r[ix[k]]
        return v if v is not None else 0

    # expected_pa lookup
    lk = {}
    for sp, sd, avg in conn.execute(
            """SELECT lineup_spot, side, AVG(plate_appearances) FROM batter_games
               WHERE lineup_spot BETWEEN 1 AND 9 AND side IN ('home','away')
               AND plate_appearances IS NOT NULL GROUP BY lineup_spot, side"""):
        lk[f"{int(sp)}|{sd}"] = round(float(avg), 4)

    feat = {}
    cur = None
    group = []

    def flush(group):
        cum = dict(pa=0, ab=0, h=0, hr=0, bb=0, so=0, tb=0)
        rec_hr = deque(maxlen=15)
        rec_xbh = deque(maxlen=15)
        n_prior = 0
        i = 0
        for g in group:
            gd = g[ix["game_date"]]
            while i < len(group) and group[i][ix["game_date"]] < gd:
                h = group[i]
                tb = N(h, "hits") + N(h, "doubles") + 2 * N(h, "triples") + 3 * N(h, "home_runs")
                cum["pa"] += N(h, "plate_appearances"); cum["ab"] += N(h, "at_bats")
                cum["h"] += N(h, "hits"); cum["hr"] += N(h, "home_runs")
                cum["bb"] += N(h, "walks"); cum["so"] += N(h, "strikeouts"); cum["tb"] += tb
                rec_hr.append(N(h, "home_runs"))
                rec_xbh.append(N(h, "doubles") + N(h, "triples") + N(h, "home_runs"))
                n_prior += 1
                i += 1
            if n_prior < MIN_PRIOR_GAMES or cum["ab"] < MIN_PRIOR_AB:
                continue
            pa = cum["pa"] or 1
            ab = cum["ab"] or 1
            slg = cum["tb"] / ab
            avg = cum["h"] / ab
            r5 = list(rec_hr)[-5:]; r15 = list(rec_hr)
            side = g[ix["side"]]
            spot = g[ix["lineup_spot"]]
            f = {
                "hr_rate": cum["hr"] / pa, "tb_per_pa": cum["tb"] / pa,
                "season_slg": slg, "iso": slg - avg,
                "recent5_hr": sum(r5) / len(r5) if r5 else 0.0,
                "recent15_hr": sum(r15) / len(r15) if r15 else 0.0,
                "recent_xbh_avg": sum(rec_xbh) / len(rec_xbh) if rec_xbh else 0.0,
                "so_rate": cum["so"] / pa, "bb_rate": cum["bb"] / pa,
                "batting_order": spot if spot is not None else 9,
                "games_played": n_prior,
                "platoon_advantage": fd.platoon_advantage(g[ix["batter_hand"]], g[ix["pitcher_hand"]]),
                "pitcher_is_R": 1.0 if g[ix["pitcher_hand"]] == "R" else (0.0 if g[ix["pitcher_hand"]] in ("L", "S") else fd.NAN),
                "is_home": 1.0 if side == "home" else (0.0 if side == "away" else fd.NAN),
                "expected_pa_v1": lk.get(f"{int(spot)}|{side}", fd.NAN) if spot is not None else fd.NAN,
                "park_factor_by_batter_hand": fnum(g[ix["park_factor_by_batter_hand"]]),
            }
            feat[(g[ix["game_id"]], g[ix["batter_id"]])] = {
                "season": gd[:4], "game_date": gd, "opp_pid": g[ix["opposing_pitcher_id"]],
                "f": f, "y": 1 if N(g, "home_runs") >= 1 else 0}
        return

    for r in rows:
        if r[ix["batter_id"]] != cur:
            if group:
                flush(group)
            group = []; cur = r[ix["batter_id"]]
        group.append(r)
    if group:
        flush(group)

    # opposing-pitcher HR-allowed rate, strict D-1 (by date)
    by_date = {}
    for r in rows:
        by_date.setdefault(r[ix["game_date"]], []).append(r)
    pit = {}
    for d in sorted(by_date):
        for r in by_date[d]:
            k = (r[ix["game_id"]], r[ix["batter_id"]])
            if k not in feat:
                continue
            hp = pit.get(r[ix["opposing_pitcher_id"]])
            if hp and hp[1] > 0:
                feat[k]["f"]["opp_pitcher_hr_per_pa"] = hp[0] / hp[1]
                feat[k]["f"]["opp_pitcher_hr_pa_seen"] = float(hp[1])
            else:
                feat[k]["f"]["opp_pitcher_hr_per_pa"] = fd.NAN
                feat[k]["f"]["opp_pitcher_hr_pa_seen"] = 0.0
        for r in by_date[d]:
            pid = r[ix["opposing_pitcher_id"]]
            if pid is None:
                continue
            hp = pit.setdefault(pid, [0, 0])
            hp[0] += N(r, "home_runs"); hp[1] += N(r, "plate_appearances")

    data = {"2025": [], "2026": []}
    for rec in feat.values():
        if rec["season"] in data:
            data[rec["season"]].append(rec)
    return data


# production feature set: power base + cheap context that helped (platoon,
# pitcher hand, home, expected PA). park_factor and opp_pitcher_hr are dropped --
# they did not rank and would need live plumbing. All PROD features compute live.
PROD_CONTEXT = ["platoon_advantage", "pitcher_is_R", "is_home", "expected_pa_v1"]
PROD_FEATURES = BASE + PROD_CONTEXT


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--produce", action="store_true")
    ap.add_argument("--workdir", default="/data/hr_model/hr_context_challenger_a_work")
    args = ap.parse_args()
    import xgboost as xgb
    con = sqlite3.connect(f"file:{SOURCE}?mode=ro", uri=True)
    print("HR_CONTEXT_CHALLENGER_A  (over 0.5)\n" + "=" * 40)
    data = build(con)
    con.close()
    dev, hol = data["2025"], data["2026"]
    print(f"  2025={len(dev)}  2026={len(hol)}")
    base_rate = sum(r["y"] for r in hol) / len(hol)
    print(f"  2026 base rate P(HR>=1) = {base_rate:.4f}   (model constant states {MODEL_CONSTANT_ELITE})")

    dates = sorted({r["game_date"] for r in dev})
    cut = dates[int(len(dates) * 0.8)]
    tr = [r for r in dev if r["game_date"] < cut]
    va = [r for r in dev if r["game_date"] >= cut]

    def mat(rowset, feats):
        X = np.array([[r["f"].get(k, fd.NAN) for k in feats] for r in rowset], dtype=np.float32)
        y = np.array([r["y"] for r in rowset], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=feats)

    def run(feats):
        b = xgb.train(fd.PARAMS, mat(tr, feats), num_boost_round=800,
                      evals=[(mat(va, feats), "val")], early_stopping_rounds=40,
                      verbose_eval=False)
        p = b.predict(mat(hol, feats), iteration_range=(0, b.best_iteration + 1))
        return fd.metrics(list(map(float, p)), [r["y"] for r in hol]), b

    print("\ntraining base_logistic ...", flush=True)
    base, _ = run(BASE)
    print("training challenger (base+context) ...", flush=True)
    chal, bst = run(BASE + CONTEXT)
    print("training PROD (base + cheap context, ship candidate) ...", flush=True)
    prod, _ = run(PROD_FEATURES)
    print(f"  PROD holdout: AUC={prod['auc']} logloss={prod['log_loss']} Brier={prod['brier']} ECE={prod['ece']}")

    # constant-model reference ECE: it emits ~0.35 for every pick
    const_ece = abs(MODEL_CONSTANT_ELITE - base_rate)

    print("\n============ 2026 HOLDOUT [batter_home_runs] ============")
    print(f"  {'arm':22s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    print(f"  {'constant ~0.358':22s} {'n/a':>7s} {'n/a':>9s} {'n/a':>8s} {const_ece:7.4f}  (fixed-prob vs base rate)")
    for name, r in (("base_logistic", base), ("challenger", chal)):
        print(f"  {name:22s} {r['auc']:.4f}  {r['log_loss']:.5f}  {r['brier']:.5f}  {r['ece']:.4f}")

    print("\nchallenger reliability (pred -> actual):")
    for b in chal["reliability"]:
        print(f"   {b['bin']}  n={b['n']:>6}  pred={b['pred']:.3f}  actual={b['actual']:.3f}")

    imp = bst.get_score(importance_type="gain")
    print("\nchallenger feature importance (gain, top 12):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1])[:12]:
        print(f"   {k:28s} {v:9.2f}")

    print(f"\nREAD: constant-model ECE ~{const_ece:.3f} (says {MODEL_CONSTANT_ELITE}, reality {base_rate:.3f}).")
    print(f"  If the challenger's ECE is ~0.02 with AUC clearly >0.5, HR gets HONEST,")
    print(f"  batter-specific probabilities -- calibration is the win, not profitability.")
    produced = None
    if args.produce:
        allrows = dev + hol
        ad = sorted({r["game_date"] for r in allrows})
        ac = ad[int(len(ad) * 0.9)]
        atr = [r for r in allrows if r["game_date"] < ac]
        ava = [r for r in allrows if r["game_date"] >= ac]
        final = xgb.train(fd.PARAMS, mat(atr, PROD_FEATURES), num_boost_round=800,
                          evals=[(mat(ava, PROD_FEATURES), "val")], early_stopping_rounds=40,
                          verbose_eval=False)
        work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
        final.save_model(str(work / "batter_home_runs_context.json"))
        (work / "batter_home_runs_context_columns.json").write_text(json.dumps(PROD_FEATURES))
        produced = {"model": str(work / "batter_home_runs_context.json"), "features": PROD_FEATURES}
        print(f"\nPRODUCED (not yet deployed to /data/models):")
        print(f"  {produced['model']}")
        print(f"  reuses expected_pa_lookup.json already deployed with hits")

    report = {"script": "HR_CONTEXT_CHALLENGER_A", "base_rate": base_rate,
              "constant_ece": const_ece, "base": base, "challenger": chal,
              "prod": prod, "prod_features": PROD_FEATURES, "importance": imp, "produced": produced}
    out = Path("/data/hr_model") if Path("/data/hr_model").exists() else Path.cwd()
    (out / "hr_context_challenger_a_report.json").write_text(json.dumps(report, indent=2))
    print("\nread-only on hr_model.sqlite. no production change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
