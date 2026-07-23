#!/usr/bin/env python3
"""
HR_FEATURE_DISCOVERY_PARKWX_A

Tests whether park factor + weather improve the batter_home_runs_context
model, using the exact same pre-registered-gate discipline as
hits_feature_discovery_b.py -- the earlier hits version of this question
came back a clean FAIL (weather barely used, gain below the bar). HR is a
priori a better candidate: ball flight (and therefore whether a fly ball
clears the fence) is far more sensitive to wind/temperature/park dimensions
than whether a ground ball finds a hole, which is what likely sank the
hits version. Predictions-first: no odds.

park_factor_by_batter_hand was already informally tried in
hr_context_challenger_a.py ("did not rank" -- a comment, not a gated
test) but weather (temp_f, wind_speed_mph, wind_toward_pull_field) has
NEVER been tested for HR at all. This gives both a real, pre-registered
verdict instead of an informal one.

Arms on the untouched 2026 holdout (2025 = train w/ internal val):
  base       = current production batter_home_runs_context features (15)
  base+park  = base + park_factor_by_batter_hand
  full       = base + park_factor_by_batter_hand + temp_f + wind_speed_mph
               + wind_toward_pull_field

Same eligibility/target/feature-computation as hr_context_challenger_a.py
(MIN_PRIOR_AB=40, MIN_PRIOR_GAMES=10, target P(home_runs>=1), strict D-1).

Pre-registered gates (log-loss down >=0.001 AND AUC up >=0.005, calibration
not worse by >0.01 ECE), same bar as the hits version so the two are
directly comparable:
  park_gain : base+park vs base
  wx_gain   : full      vs base+park

Read-only on hr_model.sqlite. Writes only a report to work dir. Touches no
production model or code. No auto-promotion.

Run (Render)
------------
python -u hr_feature_discovery_parkwx_a.py 2>&1 | tee /data/hr_model/hr_feature_discovery_parkwx_a.log
"""
import json
import sqlite3
import sys
from collections import deque
from pathlib import Path

import numpy as np

import hits_feature_discovery_b as fd  # platoon_advantage, metrics, PARAMS, NAN

SOURCE = "/data/hr_model/hr_model.sqlite"
MIN_PRIOR_AB = 40
MIN_PRIOR_GAMES = 10

BASE = ["hr_rate", "tb_per_pa", "season_slg", "iso", "recent5_hr", "recent15_hr",
        "recent_xbh_avg", "so_rate", "bb_rate", "batting_order", "games_played"]
PROD_CONTEXT = ["platoon_advantage", "pitcher_is_R", "is_home", "expected_pa_v1"]
PROD_FEATURES = BASE + PROD_CONTEXT  # matches models/batter_home_runs_context_columns.json
PARK = ["park_factor_by_batter_hand"]
WX = ["temp_f", "wind_speed_mph", "wind_toward_pull_field"]


def fnum(v):
    return float(v) if v is not None else fd.NAN


def build(conn):
    cols = ["game_id", "game_date", "batter_id", "lineup_spot", "batter_hand",
            "pitcher_hand", "opposing_pitcher_id", "side", "park_factor_by_batter_hand",
            "temp_f", "wind_speed_mph", "wind_toward_pull_field",
            "plate_appearances", "at_bats", "hits", "doubles", "triples", "home_runs",
            "walks", "strikeouts"]
    rows = conn.execute(
        f"SELECT {','.join(cols)} FROM batter_games WHERE at_bats IS NOT NULL "
        f"ORDER BY batter_id, game_date, game_id").fetchall()
    ix = {c: i for i, c in enumerate(cols)}

    def N(r, k):
        v = r[ix[k]]
        return v if v is not None else 0

    lk = {}
    for sp, sd, avg in conn.execute(
            """SELECT lineup_spot, side, AVG(plate_appearances) FROM batter_games
               WHERE lineup_spot BETWEEN 1 AND 9 AND side IN ('home','away')
               AND plate_appearances IS NOT NULL GROUP BY lineup_spot, side"""):
        lk[f"{int(sp)}|{sd}"] = round(float(avg), 4)

    feat = {}

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
                "temp_f": fnum(g[ix["temp_f"]]),
                "wind_speed_mph": fnum(g[ix["wind_speed_mph"]]),
                "wind_toward_pull_field": fnum(g[ix["wind_toward_pull_field"]]),
            }
            feat[(g[ix["game_id"]], g[ix["batter_id"]])] = {
                "season": gd[:4], "game_date": gd, "f": f,
                "y": 1 if N(g, "home_runs") >= 1 else 0}
        return

    cur = None
    group = []
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
    return data


def main():
    import xgboost as xgb
    con = sqlite3.connect(f"file:{SOURCE}?mode=ro", uri=True)
    print("HR_FEATURE_DISCOVERY_PARKWX_A\n" + "=" * 30)
    print("building dataset ...", flush=True)
    data = build(con)
    con.close()
    dev, hol = data["2025"], data["2026"]
    print(f"2025={len(dev)} 2026={len(hol)}")

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

    arms = {"base": PROD_FEATURES, "base+park": PROD_FEATURES + PARK,
            "full (park+wx)": PROD_FEATURES + PARK + WX}
    res = {}
    models = {}
    for name, feats in arms.items():
        print(f"training {name} ({len(feats)} feats) ...", flush=True)
        res[name], models[name] = run(feats)

    print("\n2026 HOLDOUT")
    print(f"  {'arm':16s} {'AUC':>7s} {'logloss':>9s} {'Brier':>8s} {'ECE':>7s}")
    for name in arms:
        r = res[name]
        print(f"  {name:16s} {r['auc']:.4f}  {r['log_loss']:.5f}  {r['brier']:.5f}  {r['ece']:.4f}")

    imp = models["full (park+wx)"].get_score(importance_type="gain")
    print("\nFEATURE IMPORTANCE (full model, gain-weighted, top 15):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1])[:15]:
        print(f"   {k:28s} {v:10.2f}")

    def gate(better, worse, label):
        d_ll = worse["log_loss"] - better["log_loss"]
        d_auc = better["auc"] - worse["auc"]
        d_ece = better["ece"] - worse["ece"]
        passed = d_ll >= 0.001 and d_auc >= 0.005 and d_ece <= 0.01
        print(f"\n  {label}: dlogloss={d_ll:+.5f} dAUC={d_auc:+.4f} dECE={d_ece:+.4f} -> {'PASS' if passed else 'no'}")
        return {"d_log_loss": d_ll, "d_auc": d_auc, "d_ece": d_ece, "passed": passed}

    print("\n================ PRE-REGISTERED GATES ================")
    park_gain = gate(res["base+park"], res["base"], "park_gain (base+park vs base)")
    wx_gain = gate(res["full (park+wx)"], res["base+park"], "wx_gain   (full vs base+park)")

    print("\nchallenger (full) reliability:")
    for b in res["full (park+wx)"]["reliability"]:
        print(f"   {b['bin']}  n={b['n']:>6}  pred={b['pred']:.3f}  actual={b['actual']:.3f}")

    report = {"script": "HR_FEATURE_DISCOVERY_PARKWX_A", "holdout": "2026",
              "arms": res, "feature_importance_gain": imp,
              "gates": {"park_gain": park_gain, "wx_gain": wx_gain}}
    out_dir = Path("/data/hr_model") if Path("/data/hr_model").exists() else Path.cwd()
    (out_dir / "hr_feature_discovery_parkwx_a_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {out_dir / 'hr_feature_discovery_parkwx_a_report.json'}")
    print("Read-only. No production model or code changed. No auto-promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
