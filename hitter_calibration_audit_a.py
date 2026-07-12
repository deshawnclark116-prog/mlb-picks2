#!/usr/bin/env python3
"""
HITTER_CALIBRATION_AUDIT_A

Predictions-first correctness audit. For every hitter count model, score the
CURRENT champion on strict-D-1 (leak-free) features rebuilt from
hr_model.sqlite::batter_games and report how well its stated probabilities
match reality. NO odds, NO money -- only whether the predictions are correct
and honest about their own confidence.

Self-adapting
-------------
Reads the real batter_games schema and each model's real *_columns.json at
runtime. For each market it reconstructs the champion feature-for-feature; if a
required feature cannot be derived faithfully from batter_games (e.g. runs_per_pa
when batter_games has no runs column), the market is SKIPPED with the reason
rather than scored on fabricated inputs.

Markets audited (count markets derivable from batter_games)
-----------------------------------------------------------
batter_hits (line 0.5), batter_total_bases (1.5), batter_rbi (0.5),
batter_home_runs (0.5)

Metrics per season (2025 development, 2026 HOLDOUT)
--------------------------------------------------
n, base_rate, mean_proj vs mean_actual, projection MAE,
AUC (tie-corrected), Brier score, log-loss, ECE (expected calibration error),
and a 10-bin reliability table (mean predicted prob vs actual rate).

Read-only. Changes no production code, models, or predictions.

Run (Render)
------------
python -u hitter_calibration_audit_a.py 2>&1 | tee /data/hr_model/hitter_calibration_audit_a.log
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

MODEL_DIR = Path("/data/models")
DEFAULT_SOURCE = "/data/hr_model/hr_model.sqlite"

# market -> (model file stem, standard line, target key)
MARKETS = {
    "batter_hits": ("batter_hits", 0.5, "hits"),
    "batter_total_bases": ("batter_total_bases", 1.5, "tb"),
    "batter_rbi": ("batter_rbi", 0.5, "rbi"),
    "batter_home_runs": ("batter_home_runs", 0.5, "hr"),
}

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


def metrics(probs, labels, projs, acts):
    n = len(probs)
    base = sum(labels) / n
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / n
    eps = 1e-12
    ll = -sum(y * math.log(min(max(p, eps), 1 - eps)) +
              (1 - y) * math.log(min(max(1 - p, eps), 1 - eps))
              for p, y in zip(probs, labels)) / n
    # 10-bin reliability + ECE
    bins = [[0, 0.0, 0] for _ in range(10)]  # [n, sum_pred, sum_label]
    for p, y in zip(probs, labels):
        b = min(9, int(p * 10))
        bins[b][0] += 1
        bins[b][1] += p
        bins[b][2] += y
    ece = 0.0
    rel = []
    for bi, (cnt, sp, sy) in enumerate(bins):
        if cnt == 0:
            continue
        mp = sp / cnt
        ar = sy / cnt
        ece += abs(mp - ar) * cnt / n
        rel.append({"bin": f"{bi/10:.1f}-{(bi+1)/10:.1f}", "n": cnt,
                    "mean_pred": round(mp, 4), "actual_rate": round(ar, 4)})
    return {
        "n": n, "base_rate": round(base, 4),
        "mean_proj": round(sum(projs) / n, 3), "mean_actual": round(sum(acts) / n, 3),
        "proj_mae": round(sum(abs(p - a) for p, a in zip(projs, acts)) / n, 3),
        "auc": round(auc(probs, labels), 4) if 0 < base < 1 else None,
        "brier": round(brier, 4), "log_loss": round(ll, 4),
        "ece": round(ece, 4), "reliability": rel,
    }


def build_feature_rows(conn, has_runs):
    """Yield per eligible batter-game: all derivable features + all targets."""
    cur = conn.cursor()
    cols_sel = ["batter_id", "substr(game_date,1,4) AS season", "game_date", "game_id",
                "lineup_spot", "plate_appearances", "at_bats", "hits", "doubles",
                "triples", "home_runs", "rbi", "walks", "strikeouts"]
    if has_runs:
        cols_sel.append("runs")
    cur.execute(f"""SELECT {', '.join(cols_sel)} FROM batter_games
                    WHERE at_bats IS NOT NULL
                    ORDER BY batter_id, season, game_date, game_id""")
    names = [d[0].split(" AS ")[-1] if " AS " in d[0] else d[0].split(".")[-1]
             for d in [(c,) for c in cols_sel]]
    # map column order
    idx = {c: i for i, c in enumerate(
        ["batter_id", "season", "game_date", "game_id", "lineup_spot",
         "plate_appearances", "at_bats", "hits", "doubles", "triples",
         "home_runs", "rbi", "walks", "strikeouts"] + (["runs"] if has_runs else []))}

    def N(row, k):
        v = row[idx[k]]
        return v if v is not None else 0

    cur_key = None
    group = []

    def flush(group):
        out = []
        cum = dict(pa=0, ab=0, h=0, hr=0, bb=0, so=0, rbi=0, tb=0, runs=0)
        rec = {k: deque(maxlen=15) for k in ("hits", "tb", "rbi", "hr")}
        n_prior = 0
        i = 0
        for g in group:
            gdate = g[idx["game_date"]]
            while i < len(group) and group[i][idx["game_date"]] < gdate:
                h = group[i]
                tb = N(h, "hits") + N(h, "doubles") + 2 * N(h, "triples") + 3 * N(h, "home_runs")
                cum["pa"] += N(h, "plate_appearances"); cum["ab"] += N(h, "at_bats")
                cum["h"] += N(h, "hits"); cum["hr"] += N(h, "home_runs")
                cum["bb"] += N(h, "walks"); cum["so"] += N(h, "strikeouts")
                cum["rbi"] += N(h, "rbi"); cum["tb"] += tb
                if has_runs:
                    cum["runs"] += N(h, "runs")
                rec["hits"].append(N(h, "hits")); rec["tb"].append(tb)
                rec["rbi"].append(N(h, "rbi")); rec["hr"].append(N(h, "home_runs"))
                n_prior += 1
                i += 1
            if n_prior < MIN_PRIOR_GAMES or cum["ab"] < MIN_PRIOR_AB:
                continue
            pa = cum["pa"] or 1
            def rmean(key, w):
                lst = list(rec[key])[-w:]
                return sum(lst) / len(lst) if lst else 0.0
            feats = {
                "season_avg": cum["h"] / cum["ab"] if cum["ab"] else 0.0,
                "recent5_avg": rmean("hits", 5), "recent15_avg": rmean("hits", 15),
                "hr_rate": cum["hr"] / pa, "bb_rate": cum["bb"] / pa,
                "so_rate": cum["so"] / pa, "tb_per_pa": cum["tb"] / pa,
                "rbi_per_pa": cum["rbi"] / pa,
                "runs_per_pa": (cum["runs"] / pa) if has_runs else None,
                "batting_order": g[idx["lineup_spot"]] if g[idx["lineup_spot"]] is not None else 9,
                "games_played": n_prior,
                # per-market recent target means:
                "_recent5": {k: rmean(k, 5) for k in ("hits", "tb", "rbi", "hr")},
                "_recent15": {k: rmean(k, 15) for k in ("hits", "tb", "rbi", "hr")},
            }
            actual = {
                "hits": N(g, "hits"),
                "tb": N(g, "hits") + N(g, "doubles") + 2 * N(g, "triples") + 3 * N(g, "home_runs"),
                "rbi": N(g, "rbi"), "hr": N(g, "home_runs"),
            }
            out.append({"season": g[idx["season"]], "feats": feats, "actual": actual})
        return out

    for row in cur:
        key = (row[idx["batter_id"]], row[idx["season"]])
        if key != cur_key:
            if group:
                for o in flush(group):
                    yield o
            group = []
            cur_key = key
        group.append(row)
    if group:
        for o in flush(group):
            yield o


def feature_vector(feats, model_cols, target_key):
    """Build the model input vector; return (vector, missing_feature or None)."""
    vec = []
    for c in model_cols:
        if c == "recent5_target":
            vec.append(feats["_recent5"][target_key])
        elif c == "recent15_target":
            vec.append(feats["_recent15"][target_key])
        elif c in feats and c not in ("_recent5", "_recent15"):
            v = feats[c]
            if v is None:
                return None, c
            vec.append(v)
        else:
            return None, c
    return vec, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    args = ap.parse_args()

    import numpy as np
    import xgboost as xgb

    mdir = Path(args.model_dir)
    con = sqlite3.connect(f"file:{args.source}?mode=ro", uri=True)
    bg_cols = [r[1] for r in con.execute("PRAGMA table_info(batter_games)")]
    has_runs = "runs" in bg_cols
    print("HITTER_CALIBRATION_AUDIT_A")
    print("==========================")
    print(f"source={args.source}")
    print(f"batter_games columns ({len(bg_cols)}): {bg_cols}")
    print(f"runs column present: {has_runs}")

    # load models + columns
    loaded = {}
    print("\nMODEL FEATURE COLUMNS")
    print("---------------------")
    for market, (stem, line, tkey) in MARKETS.items():
        mp = mdir / f"{stem}.json"
        cp = mdir / f"{stem}_columns.json"
        if not mp.exists() or not cp.exists():
            print(f"  {market}: model/columns file missing -> skip")
            continue
        cols = json.loads(cp.read_text())
        b = xgb.Booster(); b.load_model(str(mp))
        loaded[market] = {"booster": b, "cols": cols, "line": line, "tkey": tkey}
        need_runs = "runs_per_pa" in cols
        flag = "  (needs runs_per_pa)" if need_runs else ""
        print(f"  {market}: {cols}{flag}")
        if need_runs and not has_runs:
            print(f"     -> WILL SKIP SCORING: requires runs_per_pa but batter_games has no runs column")

    # build clean rows once
    print("\nbuilding strict-D-1 feature rows ...", flush=True)
    rows = list(build_feature_rows(con, has_runs))
    con.close()
    print(f"eligible rows: {len(rows)}")

    report = {"script": "HITTER_CALIBRATION_AUDIT_A", "batter_games_columns": bg_cols,
              "has_runs": has_runs, "markets": {}}

    for market, m in loaded.items():
        cols, line, tkey = m["cols"], m["line"], m["tkey"]
        # split by season, build vectors, skip market if any required feature missing
        season_data = {}
        skip_reason = None
        vecs = {}
        for r in rows:
            vec, missing = feature_vector(r["feats"], cols, tkey)
            if missing is not None:
                skip_reason = f"cannot derive feature '{missing}' from batter_games"
                break
            season_data.setdefault(r["season"], []).append(r)
            vecs.setdefault(r["season"], []).append(vec)
        if skip_reason:
            print(f"\n######## {market}: SKIPPED ({skip_reason}) ########")
            report["markets"][market] = {"skipped": skip_reason}
            continue

        report["markets"][market] = {"line": line, "by_season": {}}
        thresh_over = math.floor(line) + 1
        print(f"\n######## {market}  (over {line}) ########")
        for season in sorted(season_data):
            data = season_data[season]
            X = np.array(vecs[season], dtype=np.float32)
            preds = m["booster"].predict(xgb.DMatrix(X, feature_names=cols))
            probs, labels, projs, acts = [], [], [], []
            for r, proj in zip(data, preds):
                a = r["actual"][tkey]
                probs.append(prob_over(float(proj), line))
                labels.append(1 if a >= thresh_over else 0)
                projs.append(float(proj)); acts.append(a)
            res = metrics(probs, labels, projs, acts)
            report["markets"][market]["by_season"][season] = res
            tag = "HOLDOUT" if season == "2026" else "development"
            print(f"  -- {season} ({tag}) --  n={res['n']} base={res['base_rate']} "
                  f"AUC={res['auc']} Brier={res['brier']} logloss={res['log_loss']} ECE={res['ece']}")
            print(f"     proj mean {res['mean_proj']} vs actual {res['mean_actual']} (MAE {res['proj_mae']})")
            print(f"     reliability (pred -> actual):")
            for b in res["reliability"]:
                print(f"        {b['bin']}  n={b['n']:>6}  pred={b['mean_pred']:.3f}  actual={b['actual_rate']:.3f}")

    out_dir = Path("/data/hr_model") if Path("/data/hr_model").exists() else Path.cwd()
    (out_dir / "hitter_calibration_audit_a_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {out_dir/'hitter_calibration_audit_a_report.json'}")
    print("Read-only. No production state changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
