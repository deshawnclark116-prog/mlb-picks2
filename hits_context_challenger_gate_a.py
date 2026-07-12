#!/usr/bin/env python3
"""
HITS_CONTEXT_CHALLENGER_GATE_A

Predictions-first challenger for batter_hits (over 0.5). Tests one hypothesis:
does adding matchup/park/platoon/weather context -- all already sitting in
batter_games and currently ignored by the model -- make the hits prediction
SHARPER (better discrimination) while staying HONEST (calibrated), on the
untouched 2026 holdout?

No odds. No money. The gate is pure prediction quality: log-loss, AUC, Brier,
and calibration (ECE).

Clean isolation
---------------
Three arms, all scored on the SAME strict-D-1 2026 holdout:
  production_champion : the deployed batter_hits.json (regression -> Poisson prob)
  retrained_base      : same 8 base features, retrained binary:logistic on 2025
                        (the control -- isolates 'context features' as the only change)
  challenger          : base 8 + context features, binary:logistic on 2025

The gate compares challenger vs retrained_base (the controlled comparison).
production_champion is reported as the incumbent reference.

Base features (champion's 8, strict D-1 from batter_games)
    season_avg, recent15_avg, recent5_avg, hr_rate, bb_rate, so_rate,
    batting_order, games_played
Context features (strict D-1, previously unused)
    park_factor_by_batter_hand, hitterish_park, pitcherish_park,
    temp_f, wind_speed_mph, wind_toward_pull_field,
    platoon_advantage (derived from batter/pitcher hand),
    opp_pitcher_h_per_pa (opposing pitcher hits-allowed rate, as-of),
    opp_pitcher_pa_seen  (sample size for the above)

Holdout doctrine: 2025 = development/train (with an internal date-based
validation slice for early stopping); 2026 = one-shot holdout, never trained on.

Read-only on hr_model.sqlite. Writes only its own work dir. Does NOT touch
/data/models or any production model. Passing does NOT auto-promote.

Run (Render)
------------
python -u hits_context_challenger_gate_a.py 2>&1 | tee /data/hr_model/hits_context_challenger_gate_a.log
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
MODEL_DIR = Path("/data/models")
WORKDIR = Path("/data/hr_model/hits_context_challenger_gate_a_work")
LINE = 0.5
MIN_PRIOR_AB = 20
MIN_PRIOR_GAMES = 5

BASE_FEATURES = ["season_avg", "recent15_avg", "recent5_avg", "hr_rate",
                 "bb_rate", "so_rate", "batting_order", "games_played"]
CONTEXT_FEATURES = ["park_factor_by_batter_hand", "hitterish_park", "pitcherish_park",
                    "temp_f", "wind_speed_mph", "wind_toward_pull_field",
                    "platoon_advantage", "opp_pitcher_h_per_pa", "opp_pitcher_pa_seen"]

NAN = float("nan")


# ---------------- metrics (predictions-first, no odds) ----------------
def auc(scores, labels):
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    pos = sum(1 for l in labels if l == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return None
    sp = sum(ranks[i] for i in range(n) if labels[i] == 1)
    return (sp - pos * (pos + 1) / 2.0) / (pos * neg)


def metrics(probs, labels):
    n = len(probs)
    base = sum(labels) / n
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / n
    eps = 1e-12
    ll = -sum(y * math.log(min(max(p, eps), 1 - eps)) +
              (1 - y) * math.log(min(max(1 - p, eps), 1 - eps))
              for p, y in zip(probs, labels)) / n
    bins = [[0, 0.0, 0] for _ in range(10)]
    for p, y in zip(probs, labels):
        b = min(9, int(p * 10))
        bins[b][0] += 1; bins[b][1] += p; bins[b][2] += y
    ece = 0.0; rel = []
    for bi, (cnt, sp, sy) in enumerate(bins):
        if cnt == 0:
            continue
        mp = sp / cnt; ar = sy / cnt
        ece += abs(mp - ar) * cnt / n
        rel.append({"bin": f"{bi/10:.1f}-{(bi+1)/10:.1f}", "n": cnt,
                    "mean_pred": round(mp, 4), "actual_rate": round(ar, 4)})
    return {"n": n, "base_rate": round(base, 4), "auc": round(auc(probs, labels), 4),
            "brier": round(brier, 5), "log_loss": round(ll, 5),
            "ece": round(ece, 4), "reliability": rel}


def poisson_cdf(k, lam):
    if lam <= 0:
        return 1.0
    s, term = 0.0, math.exp(-lam)
    for i in range(k + 1):
        if i:
            term *= lam / i
        s += term
    return min(s, 1.0)


def prob_over(expected, line=LINE):
    return 1 - poisson_cdf(int(math.floor(line)), max(expected, 1e-6))


def platoon_advantage(bhand, phand):
    if bhand in (None, "") or phand in (None, ""):
        return NAN
    if bhand == "S":
        return 1.0
    if (bhand == "L" and phand == "R") or (bhand == "R" and phand == "L"):
        return 1.0
    return 0.0


def fnum(v):
    return float(v) if v is not None else NAN


def build_dataset(conn):
    cols = ["game_id", "game_date", "batter_id", "lineup_spot", "batter_hand",
            "pitcher_hand", "opposing_pitcher_id", "park_factor_by_batter_hand",
            "hitterish_park", "pitcherish_park", "temp_f", "wind_speed_mph",
            "wind_toward_pull_field", "plate_appearances", "at_bats", "hits",
            "doubles", "triples", "home_runs", "walks", "strikeouts"]
    rows = conn.execute(
        f"SELECT {', '.join(cols)} FROM batter_games WHERE at_bats IS NOT NULL "
        f"ORDER BY batter_id, game_date, game_id").fetchall()
    ix = {c: i for i, c in enumerate(cols)}

    def N(r, k):
        v = r[ix[k]]
        return v if v is not None else 0

    # ---- Pass A: per-batter strict-D-1 base features + static context ----
    feat = {}  # (game_id,batter_id) -> dict
    cur_key = None
    group = []

    def flush(group):
        cum = dict(pa=0, ab=0, h=0, hr=0, bb=0, so=0)
        rec_h = deque(maxlen=15)
        n_prior = 0
        i = 0
        for g in group:
            gdate = g[ix["game_date"]]
            while i < len(group) and group[i][ix["game_date"]] < gdate:
                h = group[i]
                cum["pa"] += N(h, "plate_appearances"); cum["ab"] += N(h, "at_bats")
                cum["h"] += N(h, "hits"); cum["hr"] += N(h, "home_runs")
                cum["bb"] += N(h, "walks"); cum["so"] += N(h, "strikeouts")
                rec_h.append(N(h, "hits"))
                n_prior += 1
                i += 1
            if n_prior < MIN_PRIOR_GAMES or cum["ab"] < MIN_PRIOR_AB:
                continue
            pa = cum["pa"] or 1
            r5 = list(rec_h)[-5:]; r15 = list(rec_h)
            f = {
                "season_avg": cum["h"] / cum["ab"] if cum["ab"] else 0.0,
                "recent5_avg": sum(r5) / len(r5) if r5 else 0.0,
                "recent15_avg": sum(r15) / len(r15) if r15 else 0.0,
                "hr_rate": cum["hr"] / pa, "bb_rate": cum["bb"] / pa,
                "so_rate": cum["so"] / pa,
                "batting_order": g[ix["lineup_spot"]] if g[ix["lineup_spot"]] is not None else 9,
                "games_played": n_prior,
                # static context (known pregame)
                "park_factor_by_batter_hand": fnum(g[ix["park_factor_by_batter_hand"]]),
                "hitterish_park": fnum(g[ix["hitterish_park"]]),
                "pitcherish_park": fnum(g[ix["pitcherish_park"]]),
                "temp_f": fnum(g[ix["temp_f"]]),
                "wind_speed_mph": fnum(g[ix["wind_speed_mph"]]),
                "wind_toward_pull_field": fnum(g[ix["wind_toward_pull_field"]]),
                "platoon_advantage": platoon_advantage(g[ix["batter_hand"]], g[ix["pitcher_hand"]]),
            }
            label = 1 if N(g, "hits") >= 1 else 0
            feat[(g[ix["game_id"]], g[ix["batter_id"]])] = {
                "season": gdate[:4], "game_date": gdate,
                "opp_pid": g[ix["opposing_pitcher_id"]], "f": f, "y": label}
        return

    for r in rows:
        key = r[ix["batter_id"]]
        if key != cur_key:
            if group:
                flush(group)
            group = []
            cur_key = key
        group.append(r)
    if group:
        flush(group)

    # ---- Pass B: opposing-pitcher hits-allowed rate, strict D-1 (by date) ----
    by_date = {}
    for r in rows:
        by_date.setdefault(r[ix["game_date"]], []).append(r)
    pit = {}  # pid -> [hits_allowed, pa_faced]
    for d in sorted(by_date):
        # assign as-of (dates < d) first
        for r in by_date[d]:
            k = (r[ix["game_id"]], r[ix["batter_id"]])
            if k not in feat:
                continue
            pid = r[ix["opposing_pitcher_id"]]
            hp = pit.get(pid)
            if hp and hp[1] > 0:
                feat[k]["f"]["opp_pitcher_h_per_pa"] = hp[0] / hp[1]
                feat[k]["f"]["opp_pitcher_pa_seen"] = float(hp[1])
            else:
                feat[k]["f"]["opp_pitcher_h_per_pa"] = NAN
                feat[k]["f"]["opp_pitcher_pa_seen"] = 0.0
        # then fold date d into pitcher totals
        for r in by_date[d]:
            pid = r[ix["opposing_pitcher_id"]]
            if pid is None:
                continue
            hp = pit.setdefault(pid, [0, 0])
            hp[0] += N(r, "hits"); hp[1] += N(r, "plate_appearances")

    data = {"2025": [], "2026": []}
    for rec in feat.values():
        s = rec["season"]
        if s in data:
            data[s].append(rec)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--workdir", default=str(WORKDIR))
    args = ap.parse_args()

    import numpy as np
    import xgboost as xgb

    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{args.source}?mode=ro", uri=True)
    print("HITS_CONTEXT_CHALLENGER_GATE_A", flush=True)
    print("==============================", flush=True)
    print("building strict-D-1 dataset (batter + opposing-pitcher as-of) ...", flush=True)
    data = build_dataset(con)
    print(f"  2025 rows: {len(data['2025'])}   2026 rows: {len(data['2026'])}", flush=True)

    dev = data["2025"]; hol: list = data["2026"]
    # internal validation split of 2025 by date (last 20% dates) for early stopping
    dates = sorted({r["game_date"] for r in dev})
    cut = dates[int(len(dates) * 0.8)] if len(dates) > 5 else dates[-1]
    tr = [r for r in dev if r["game_date"] < cut]
    va = [r for r in dev if r["game_date"] >= cut]
    print(f"  train {len(tr)}  val {len(va)}  holdout {len(hol)}  (val cut {cut})", flush=True)

    def matrix(rowset, feats):
        X = np.array([[r["f"].get(k, NAN) for k in feats] for r in rowset], dtype=np.float32)
        y = np.array([r["y"] for r in rowset], dtype=np.float32)
        return xgb.DMatrix(X, label=y, feature_names=feats)

    params = {"objective": "binary:logistic", "eval_metric": "logloss",
              "max_depth": 4, "eta": 0.05, "subsample": 0.8,
              "colsample_bytree": 0.8, "min_child_weight": 5, "seed": 13}

    def train_score(feats, name):
        bst = xgb.train(params, matrix(tr, feats), num_boost_round=800,
                        evals=[(matrix(va, feats), "val")],
                        early_stopping_rounds=40, verbose_eval=False)
        best = bst.best_iteration + 1
        probs = bst.predict(matrix(hol, feats), iteration_range=(0, best))
        labels = [r["y"] for r in hol]
        res = metrics(list(map(float, probs)), labels)
        res["best_iteration"] = best
        bst.save_model(str(work / f"{name}.json"))
        return res

    print("\ntraining retrained_base (8 base features) ...", flush=True)
    base = train_score(BASE_FEATURES, "retrained_base")
    print("training challenger (base + context) ...", flush=True)
    chal = train_score(BASE_FEATURES + CONTEXT_FEATURES, "challenger")

    # production champion (existing regression model) scored on same holdout
    prod = None
    mp = Path(args.model_dir) / "batter_hits.json"
    cp = Path(args.model_dir) / "batter_hits_columns.json"
    if mp.exists() and cp.exists():
        pcols = json.loads(cp.read_text())
        pb = xgb.Booster(); pb.load_model(str(mp))
        X = np.array([[r["f"].get(k, NAN) for k in pcols] for r in hol], dtype=np.float32)
        preds = pb.predict(xgb.DMatrix(X, feature_names=pcols))
        probs = [prob_over(float(p)) for p in preds]
        prod = metrics(probs, [r["y"] for r in hol])

    def show(name, r):
        if r is None:
            print(f"  {name:20s} (unavailable)"); return
        print(f"  {name:20s} AUC={r['auc']}  logloss={r['log_loss']}  "
              f"Brier={r['brier']}  ECE={r['ece']}  base={r['base_rate']}  n={r['n']}")

    print("\n============ 2026 HOLDOUT (prediction quality) ============")
    show("production_champion", prod)
    show("retrained_base", base)
    show("challenger", chal)

    print("\nchallenger reliability (pred -> actual):")
    for b in chal["reliability"]:
        print(f"   {b['bin']}  n={b['n']:>6}  pred={b['mean_pred']:.3f}  actual={b['actual_rate']:.3f}")

    # pre-registered gate: challenger vs retrained_base on holdout
    d_ll = base["log_loss"] - chal["log_loss"]       # want > 0 (lower loss)
    d_auc = chal["auc"] - base["auc"]                # want >= 0.005
    d_ece = chal["ece"] - base["ece"]                # want <= +0.01 (not worse)
    d_brier = base["brier"] - chal["brier"]          # want >= 0
    passed = (d_ll >= 0.001 and d_auc >= 0.005 and d_ece <= 0.01 and d_brier >= 0.0)
    verdict = ("CHALLENGER_IMPROVES_HITS_PREDICTION_QUALITY_ON_2026_HOLDOUT"
               if passed else
               "NO_MATERIAL_IMPROVEMENT_CONTEXT_FEATURES_DO_NOT_CLEAR_GATE")

    print("\n============ PRE-REGISTERED GATE (challenger vs retrained_base) ============")
    print(f"  delta log_loss (base-chal, want>=+0.001): {d_ll:+.5f}")
    print(f"  delta AUC      (chal-base, want>=+0.005): {d_auc:+.4f}")
    print(f"  delta Brier    (base-chal, want>=0):      {d_brier:+.5f}")
    print(f"  delta ECE      (chal-base, want<=+0.010): {d_ece:+.4f}")
    print(f"  VERDICT: {verdict}")

    report = {"script": "HITS_CONTEXT_CHALLENGER_GATE_A", "line": LINE,
              "holdout": "2026", "arms": {"production_champion": prod,
              "retrained_base": base, "challenger": chal},
              "gate": {"delta_log_loss": d_ll, "delta_auc": d_auc,
                       "delta_brier": d_brier, "delta_ece": d_ece,
                       "passed": passed, "verdict": verdict},
              "base_features": BASE_FEATURES, "context_features": CONTEXT_FEATURES}
    out_dir = Path("/data/hr_model") if Path("/data/hr_model").exists() else work
    (out_dir / "hits_context_challenger_gate_a_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    con.close()
    print(f"\nreport: {out_dir/'hits_context_challenger_gate_a_report.json'}")
    print("Read-only on hr_model.sqlite. No production model or code changed. No auto-promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
