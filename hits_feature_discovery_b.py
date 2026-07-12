#!/usr/bin/env python3
"""
HITS_FEATURE_DISCOVERY_B

Find the sharpest HONEST batter_hits model before we pay to productionize
anything. Predictions-first: no odds.

Separates features by how hard they are to compute LIVE, so the result tells us
not just "can we get sharper" but "which features are worth the live-plumbing
cost":

  BASE (8)            current champion features
  EASY  (live-cheap)  platoon_advantage, pitcher_is_R, is_home, expected_pa_v1,
                      opp_pitcher_h_per_pa, opp_pitcher_pa_seen,
                      opp_pitcher_k_per_pa, recent_xbh_avg
  PARKWX (live-hard)  park_factor_by_batter_hand, hitterish_park,
                      pitcherish_park, temp_f, wind_speed_mph,
                      wind_toward_pull_field

Arms scored on the untouched 2026 holdout (2025 = train w/ internal val):
  base
  base+easy
  base+easy+parkwx (full)

Reports for each arm: AUC, log-loss, Brier, ECE. Plus gain-based feature
importance for the full model, and a LEAK CHECK on expected_pa_v1 (exact-match
and correlation vs realized PA -- if it just echoes actual PA it is dropped).

Pre-registered gates (log-loss down >=0.001 AND AUC up >=0.005, calibration not
worse by >0.01 ECE):
  easy_gain   : base+easy   vs base        (are cheap features enough?)
  parkwx_gain : full        vs base+easy   (does weather add on top?)

Read-only on hr_model.sqlite. Writes only a report + arm models to work dir.
Touches no production model or code. No auto-promotion.

Run (Render)
------------
python -u hits_feature_discovery_b.py 2>&1 | tee /data/hr_model/hits_feature_discovery_b.log
"""

import argparse
import json
import math
import sqlite3
import sys
from collections import deque
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

SOURCE = "/data/hr_model/hr_model.sqlite"
WORKDIR = Path("/data/hr_model/hits_feature_discovery_b_work")
MIN_PRIOR_AB = 20
MIN_PRIOR_GAMES = 5
NAN = float("nan")

BASE = ["season_avg", "recent15_avg", "recent5_avg", "hr_rate", "bb_rate",
        "so_rate", "batting_order", "games_played"]
EASY = ["platoon_advantage", "pitcher_is_R", "is_home", "expected_pa_v1",
        "opp_pitcher_h_per_pa", "opp_pitcher_pa_seen", "opp_pitcher_k_per_pa",
        "recent_xbh_avg"]
PARKWX = ["park_factor_by_batter_hand", "hitterish_park", "pitcherish_park",
          "temp_f", "wind_speed_mph", "wind_toward_pull_field"]

PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss",
          "max_depth": 4, "eta": 0.05, "subsample": 0.8,
          "colsample_bytree": 0.8, "min_child_weight": 5, "seed": 13}


def platoon_advantage(bh, ph):
    if bh in (None, "") or ph in (None, ""):
        return NAN
    if bh == "S":
        return 1.0
    if (bh == "L" and ph == "R") or (bh == "R" and ph == "L"):
        return 1.0
    return 0.0


def fnum(v):
    return float(v) if v is not None else NAN


def auc(scores, labels):
    import numpy as np
    labels = np.asarray(labels)
    pos = labels.sum(); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    s = np.asarray(scores)[order]; i = 0; n = len(s)
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def metrics(probs, labels):
    import numpy as np
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(labels, dtype=float)
    n = len(y)
    brier = float(np.mean((p - y) ** 2))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    ece = 0.0; rel = []
    for b in range(10):
        m = (p >= b / 10) & (p < (b + 1) / 10) if b < 9 else (p >= 0.9)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        mp = float(p[m].mean()); ar = float(y[m].mean())
        ece += abs(mp - ar) * cnt / n
        rel.append({"bin": f"{b/10:.1f}-{(b+1)/10:.1f}", "n": cnt,
                    "pred": round(mp, 4), "actual": round(ar, 4)})
    return {"n": n, "base_rate": round(float(y.mean()), 4), "auc": round(auc(probs, labels), 4),
            "log_loss": round(ll, 5), "brier": round(brier, 5), "ece": round(ece, 4),
            "reliability": rel}


def build(conn):
    cols = ["game_id", "game_date", "batter_id", "lineup_spot", "batter_hand",
            "pitcher_hand", "opposing_pitcher_id", "side", "expected_pa_v1",
            "park_factor_by_batter_hand", "hitterish_park", "pitcherish_park",
            "temp_f", "wind_speed_mph", "wind_toward_pull_field",
            "plate_appearances", "at_bats", "hits", "doubles", "triples",
            "home_runs", "walks", "strikeouts"]
    rows = conn.execute(
        f"SELECT {','.join(cols)} FROM batter_games WHERE at_bats IS NOT NULL "
        f"ORDER BY batter_id, game_date, game_id").fetchall()
    ix = {c: i for i, c in enumerate(cols)}

    def N(r, k):
        v = r[ix[k]]; return v if v is not None else 0

    feat = {}
    cur = None; group = []

    def flush(group):
        cum = dict(pa=0, ab=0, h=0, hr=0, bb=0, so=0)
        rec_h = deque(maxlen=15); rec_xbh = deque(maxlen=15)
        n_prior = 0; i = 0
        for g in group:
            gd = g[ix["game_date"]]
            while i < len(group) and group[i][ix["game_date"]] < gd:
                h = group[i]
                cum["pa"] += N(h, "plate_appearances"); cum["ab"] += N(h, "at_bats")
                cum["h"] += N(h, "hits"); cum["hr"] += N(h, "home_runs")
                cum["bb"] += N(h, "walks"); cum["so"] += N(h, "strikeouts")
                rec_h.append(N(h, "hits"))
                rec_xbh.append(N(h, "doubles") + N(h, "triples") + N(h, "home_runs"))
                n_prior += 1; i += 1
            if n_prior < MIN_PRIOR_GAMES or cum["ab"] < MIN_PRIOR_AB:
                continue
            pa = cum["pa"] or 1
            r5 = list(rec_h)[-5:]; r15 = list(rec_h)
            side = g[ix["side"]]
            f = {
                "season_avg": cum["h"] / cum["ab"] if cum["ab"] else 0.0,
                "recent5_avg": sum(r5) / len(r5) if r5 else 0.0,
                "recent15_avg": sum(r15) / len(r15) if r15 else 0.0,
                "hr_rate": cum["hr"] / pa, "bb_rate": cum["bb"] / pa, "so_rate": cum["so"] / pa,
                "batting_order": g[ix["lineup_spot"]] if g[ix["lineup_spot"]] is not None else 9,
                "games_played": n_prior,
                "recent_xbh_avg": sum(rec_xbh) / len(rec_xbh) if rec_xbh else 0.0,
                "platoon_advantage": platoon_advantage(g[ix["batter_hand"]], g[ix["pitcher_hand"]]),
                "pitcher_is_R": 1.0 if g[ix["pitcher_hand"]] == "R" else (0.0 if g[ix["pitcher_hand"]] in ("L", "S") else NAN),
                "is_home": 1.0 if side == "home" else (0.0 if side == "away" else NAN),
                "expected_pa_v1": fnum(g[ix["expected_pa_v1"]]),
                "park_factor_by_batter_hand": fnum(g[ix["park_factor_by_batter_hand"]]),
                "hitterish_park": fnum(g[ix["hitterish_park"]]),
                "pitcherish_park": fnum(g[ix["pitcherish_park"]]),
                "temp_f": fnum(g[ix["temp_f"]]), "wind_speed_mph": fnum(g[ix["wind_speed_mph"]]),
                "wind_toward_pull_field": fnum(g[ix["wind_toward_pull_field"]]),
            }
            feat[(g[ix["game_id"]], g[ix["batter_id"]])] = {
                "season": gd[:4], "game_date": gd, "f": f,
                "y": 1 if N(g, "hits") >= 1 else 0,
                "actual_pa": N(g, "plate_appearances")}
        return

    for r in rows:
        if r[ix["batter_id"]] != cur:
            if group:
                flush(group)
            group = []; cur = r[ix["batter_id"]]
        group.append(r)
    if group:
        flush(group)

    # opposing pitcher as-of hits & K allowed
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
                feat[k]["f"]["opp_pitcher_h_per_pa"] = hp[0] / hp[1]
                feat[k]["f"]["opp_pitcher_pa_seen"] = float(hp[1])
                feat[k]["f"]["opp_pitcher_k_per_pa"] = hp[2] / hp[1]
            else:
                feat[k]["f"]["opp_pitcher_h_per_pa"] = NAN
                feat[k]["f"]["opp_pitcher_pa_seen"] = 0.0
                feat[k]["f"]["opp_pitcher_k_per_pa"] = NAN
        for r in by_date[d]:
            pid = r[ix["opposing_pitcher_id"]]
            if pid is None:
                continue
            hp = pit.setdefault(pid, [0, 0, 0])
            hp[0] += N(r, "hits"); hp[1] += N(r, "plate_appearances"); hp[2] += N(r, "strikeouts")

    data = {"2025": [], "2026": []}
    for rec in feat.values():
        if rec["season"] in data:
            data[rec["season"]].append(rec)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--workdir", default=str(WORKDIR))
    args = ap.parse_args()
    import numpy as np, xgboost as xgb
    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(f"file:{args.source}?mode=ro", uri=True)
    print("HITS_FEATURE_DISCOVERY_B\n========================", flush=True)
    print("building dataset ...", flush=True)
    data = build(con); con.close()
    dev, hol = data["2025"], data["2026"]
    print(f"  2025={len(dev)}  2026={len(hol)}", flush=True)

    # leak check on expected_pa_v1 (must be pregame, not realized PA)
    ep = np.array([r["f"]["expected_pa_v1"] for r in hol], dtype=float)
    ap_ = np.array([r["actual_pa"] for r in hol], dtype=float)
    ok = ~np.isnan(ep)
    if ok.sum() > 0:
        exact = float(np.mean(np.round(ep[ok]) == ap_[ok]))
        corr = float(np.corrcoef(ep[ok], ap_[ok])[0, 1])
        print(f"\nLEAK CHECK expected_pa_v1: coverage={ok.mean():.3f} exact_match_vs_actual_PA={exact:.3f} corr={corr:.3f}")
        leaking = exact > 0.95
        if leaking:
            print("  !! expected_pa_v1 appears to echo realized PA -> DROPPING it as a leak")
            EASY_USE = [f for f in EASY if f != "expected_pa_v1"]
        else:
            print("  expected_pa_v1 looks like a genuine pregame estimate -> keep")
            EASY_USE = list(EASY)
    else:
        EASY_USE = [f for f in EASY if f != "expected_pa_v1"]
        print("\nLEAK CHECK expected_pa_v1: no coverage -> excluded")

    dates = sorted({r["game_date"] for r in dev})
    cut = dates[int(len(dates) * 0.8)]
    tr = [r for r in dev if r["game_date"] < cut]
    va = [r for r in dev if r["game_date"] >= cut]

    def mat(rows, feats):
        X = np.array([[r["f"].get(k, NAN) for k in feats] for r in rows], dtype=np.float32)
        y = np.array([r["y"] for r in rows], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=feats)

    def run(feats, name):
        bst = xgb.train(PARAMS, mat(tr, feats), num_boost_round=800,
                        evals=[(mat(va, feats), "val")], early_stopping_rounds=40,
                        verbose_eval=False)
        best = bst.best_iteration + 1
        probs = bst.predict(mat(hol, feats), iteration_range=(0, best))
        bst.save_model(str(work / f"{name}.json"))
        return metrics(list(map(float, probs)), [r["y"] for r in hol]), bst

    arms = {
        "base": BASE,
        "base+easy": BASE + EASY_USE,
        "full": BASE + EASY_USE + PARKWX,
    }
    res = {}; models = {}
    for name, feats in arms.items():
        print(f"\ntraining {name} ({len(feats)} feats) ...", flush=True)
        res[name], models[name] = run(feats, name)

    print("\n================ 2026 HOLDOUT ================")
    print(f"  {'arm':12s} {'AUC':>7s} {'logloss':>9s} {'Brier':>9s} {'ECE':>7s}")
    for name in arms:
        r = res[name]
        print(f"  {name:12s} {r['auc']:.4f}  {r['log_loss']:.5f}  {r['brier']:.5f}  {r['ece']:.4f}")

    # feature importance (gain) for full model
    imp = models["full"].get_score(importance_type="gain")
    print("\nFEATURE IMPORTANCE (full model, gain-weighted, top 15):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1])[:15]:
        print(f"   {k:28s} {v:10.2f}")

    def gate(better, worse, label):
        d_ll = worse["log_loss"] - better["log_loss"]
        d_auc = better["auc"] - worse["auc"]
        d_ece = better["ece"] - worse["ece"]
        passed = d_ll >= 0.001 and d_auc >= 0.005 and d_ece <= 0.01
        print(f"\n  {label}: dloglos={d_ll:+.5f} dAUC={d_auc:+.4f} dECE={d_ece:+.4f} -> {'PASS' if passed else 'no'}")
        return {"d_log_loss": d_ll, "d_auc": d_auc, "d_ece": d_ece, "passed": passed}

    print("\n================ PRE-REGISTERED GATES ================")
    easy_gain = gate(res["base+easy"], res["base"], "easy_gain  (base+easy vs base)")
    parkwx_gain = gate(res["full"], res["base+easy"], "parkwx_gain (full vs base+easy)")

    print("\nchallenger (full) reliability:")
    for b in res["full"]["reliability"]:
        print(f"   {b['bin']}  n={b['n']:>6}  pred={b['pred']:.3f}  actual={b['actual']:.3f}")

    report = {"script": "HITS_FEATURE_DISCOVERY_B", "holdout": "2026",
              "arms": res, "feature_importance_gain": imp,
              "easy_features_used": EASY_USE,
              "gates": {"easy_gain": easy_gain, "parkwx_gain": parkwx_gain}}
    out_dir = Path("/data/hr_model") if Path("/data/hr_model").exists() else work
    (out_dir / "hits_feature_discovery_b_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {out_dir/'hits_feature_discovery_b_report.json'}")
    print("Read-only. No production model or code changed. No auto-promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
