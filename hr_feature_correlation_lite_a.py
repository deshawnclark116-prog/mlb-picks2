#!/usr/bin/env python3
"""
HR_FEATURE_CORRELATION_LITE_A

Memory-safe replacement for hr_feature_correlation_a.py.

Reads one feature column at a time from SQLite. No full pandas dataframe.

Run:
    python hr_feature_correlation_lite_a.py --min-n 1000 --corr-threshold .85

Fastest:
    python hr_feature_correlation_lite_a.py --min-n 1000 --skip-pairs
"""

import argparse
import csv
import math
import os
import sqlite3
from pathlib import Path


def default_db_path():
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    return p / "hr_model.sqlite" if p.parent.exists() else Path("./hr_model/hr_model.sqlite")


def table_cols(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def is_feature_col(c):
    if c in {"game_id", "batter_id", "opposing_pitcher_id"}:
        return False
    return any(s in c for s in ("_7d", "_15d", "_30d", "_60d"))


def family_for(c):
    if c.startswith(("batter_barrel", "batter_hard_hit", "batter_avg_ev", "batter_max_ev")):
        return "impact_quality"
    if c.startswith(("batter_fly_ball", "batter_la_20_35", "batter_pull_air", "batter_hr_per_air")):
        return "air_shape"
    if c.startswith(("pitcher_barrel", "pitcher_hard_hit", "pitcher_avg_ev", "pitcher_max_ev")):
        return "pitcher_impact_allowed"
    if c.startswith(("pitcher_fly_ball", "pitcher_la_20_35", "pitcher_pull_air", "pitcher_hrfb")):
        return "pitcher_air_damage_allowed"
    if c in {"temp_f", "weather_temp_f", "wind_speed_mph", "weather_wind_mph", "wind_toward_pull_field", "hitterish_park", "pitcherish_park"}:
        return "environment"
    if c in {"lineup_spot", "expected_pa_v1", "plate_appearances"}:
        return "opportunity"
    return "other"


def fnum(x):
    if x is None:
        return None
    try:
        if isinstance(x, str):
            lx = x.strip().lower()
            if lx == "":
                return None
            if lx == "true":
                return 1.0
            if lx == "false":
                return 0.0
        return float(x)
    except Exception:
        return None


def rankdata(vals):
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and vals[order[j]] == vals[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    return pearson(rankdata(xs), rankdata(ys))


def build_feature_sources(conn):
    sources = {}
    bg_cols = set(table_cols(conn, "batter_games"))
    for c in [
        "lineup_spot", "expected_pa_v1", "plate_appearances",
        "temp_f", "weather_temp_f", "wind_speed_mph", "weather_wind_mph",
        "wind_toward_pull_field", "hitterish_park", "pitcherish_park"
    ]:
        if c in bg_cols:
            sources[c] = f"bg.{c}"

    for c in table_cols(conn, "batter_game_features"):
        if is_feature_col(c):
            sources[c] = f"bf.{c}"

    for c in table_cols(conn, "pitcher_game_features"):
        if is_feature_col(c):
            sources[c] = f"pf.{c}"

    return sources


def load_feature_target(conn, expr):
    sql = f"""
    SELECT {expr} AS x, bg.actual_hr AS y
    FROM batter_games bg
    LEFT JOIN batter_game_features bf
      ON bg.game_id = bf.game_id AND bg.batter_id = bf.batter_id
    LEFT JOIN pitcher_game_features pf
      ON bg.game_id = pf.game_id AND bg.batter_id = pf.batter_id
    WHERE bg.actual_hr IS NOT NULL
    """
    xs, ys = [], []
    nonzero = 0
    for x, y in conn.execute(sql):
        xv = fnum(x)
        yv = fnum(y)
        if xv is None or yv is None:
            continue
        xs.append(xv)
        ys.append(yv)
        if xv != 0:
            nonzero += 1
    return xs, ys, len(xs), nonzero


def load_pair(conn, expr_a, expr_b):
    sql = f"""
    SELECT {expr_a} AS a, {expr_b} AS b
    FROM batter_games bg
    LEFT JOIN batter_game_features bf
      ON bg.game_id = bf.game_id AND bg.batter_id = bf.batter_id
    LEFT JOIN pitcher_game_features pf
      ON bg.game_id = pf.game_id AND bg.batter_id = pf.batter_id
    """
    xs, ys = [], []
    for a, b in conn.execute(sql):
        av = fnum(a)
        bv = fnum(b)
        if av is None or bv is None:
            continue
        xs.append(av)
        ys.append(bv)
    return xs, ys, len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--min-n", type=int, default=1000)
    ap.add_argument("--corr-threshold", type=float, default=0.85)
    ap.add_argument("--pair-top", type=int, default=8)
    ap.add_argument("--skip-pairs", action="store_true")
    args = ap.parse_args()

    db = Path(args.db)
    conn = sqlite3.connect(str(db))
    sources = build_feature_sources(conn)

    total_rows = conn.execute("SELECT COUNT(*) FROM batter_games").fetchone()[0]
    hr_hits = conn.execute("SELECT SUM(actual_hr) FROM batter_games").fetchone()[0]

    print("HR_FEATURE_CORRELATION_LITE_A")
    print("=============================")
    print(f"db: {db}")
    print(f"rows: {total_rows}")
    print(f"hr_hits: {hr_hits}")
    print(f"candidate_features: {len(sources)}")
    print(f"min_n: {args.min_n}")

    results = []
    for feature, expr in sorted(sources.items()):
        xs, ys, n, nonzero = load_feature_target(conn, expr)
        if n < args.min_n:
            continue
        corr = spearman(xs, ys)
        if corr is None:
            continue
        results.append({
            "feature": feature,
            "family": family_for(feature),
            "n": n,
            "nonzero": nonzero,
            "spearman_to_hr": corr,
            "abs_corr": abs(corr),
            "expr": expr,
        })

    by_family = {}
    for r in results:
        by_family.setdefault(r["family"], []).append(r)

    print("\nTOP FEATURES BY FAMILY")
    print("----------------------")
    champions = []
    for fam in sorted(by_family):
        sub = sorted(by_family[fam], key=lambda r: -r["abs_corr"])
        print(f"\n[{fam}]")
        for r in sub[:12]:
            print(f"{r['feature']:48s} n={r['n']:6d} nonzero={r['nonzero']:6d} spearman={r['spearman_to_hr']:+.5f}")
        if sub:
            champions.append(sub[0])

    print("\nHIGH CORRELATION PAIRS WITHIN FAMILY")
    print("------------------------------------")
    if args.skip_pairs:
        print("Skipped by --skip-pairs")
    else:
        found = False
        for fam in sorted(by_family):
            sub = sorted(by_family[fam], key=lambda r: -r["abs_corr"])[:args.pair_top]
            pairs = []
            for i, a in enumerate(sub):
                for b in sub[i + 1:]:
                    xs, ys, n = load_pair(conn, a["expr"], b["expr"])
                    if n < args.min_n:
                        continue
                    corr = spearman(xs, ys)
                    if corr is not None and abs(corr) >= args.corr_threshold:
                        pairs.append((abs(corr), corr, a["feature"], b["feature"], n))
            if pairs:
                found = True
                print(f"\n[{fam}]")
                for _, corr, a, b, n in sorted(pairs, reverse=True)[:20]:
                    print(f"{corr:+.3f} n={n:6d}  {a}  <->  {b}")
        if not found:
            print("No high-correlation pairs found at this threshold/top setting.")

    print("\nSUGGESTED REDUCED LOGISTIC FEATURE SET")
    print("--------------------------------------")
    for r in champions:
        print(r["feature"])

    out = db.parent / "hr_feature_correlation_lite_a.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["feature", "family", "n", "nonzero", "spearman_to_hr", "abs_corr"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(results, key=lambda x: -x["abs_corr"]):
            w.writerow({k: r[k] for k in fieldnames})

    print(f"\nCSV_WRITTEN: {out}")
    print("DONE")
    conn.close()


if __name__ == "__main__":
    main()
