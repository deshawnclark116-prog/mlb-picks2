#!/usr/bin/env python3
"""
HR_FEATURE_CORRELATION_A

Runs feature-family correlation audit on the SQLite HR model dataset.

Outputs:
- availability
- Spearman correlation to actual_hr
- top feature per family
- high within-family correlation pairs
- suggested reduced logistic feature set

Run:
  python hr_feature_correlation_a.py

Conservative:
  python hr_feature_correlation_a.py --min-n 500 --corr-threshold .85
"""
import argparse, os, sqlite3
from pathlib import Path

def default_db_path():
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    return p / "hr_model.sqlite" if p.parent.exists() else Path("./hr_model/hr_model.sqlite")

def require_pandas():
    try:
        import pandas as pd
        return pd
    except Exception as e:
        raise SystemExit(f"Missing pandas. Install with: pip install pandas numpy\n{type(e).__name__}: {e}")

def table_cols(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]

def col_exists(conn, table, col):
    return any(r[1] == col for r in conn.execute(f"PRAGMA table_info({table})").fetchall())

def load_dataset(conn, pd):
    bf = [c for c in table_cols(conn, "batter_game_features") if c not in {"game_id", "batter_id"}]
    pf = [c for c in table_cols(conn, "pitcher_game_features") if c not in {"game_id", "batter_id", "opposing_pitcher_id"}]
    base = [
        "bg.game_date", "bg.game_id", "bg.batter_id", "bg.batter_name",
        "bg.lineup_spot", "bg.side", "bg.batter_hand", "bg.pitcher_hand",
        "bg.platoon_bucket", "bg.park_bucket", "bg.hitterish_park", "bg.pitcherish_park",
        "bg.temp_f", "bg.wind_speed_mph", "bg.wind_toward_pull_field", "bg.actual_hr"
    ]
    if col_exists(conn, "batter_games", "expected_pa_v1"):
        base.insert(-1, "bg.expected_pa_v1")
    select = base + [f"bf.{c}" for c in bf] + [f"pf.{c}" for c in pf]
    sql = f"""
    SELECT {", ".join(select)}
    FROM batter_games bg
    LEFT JOIN batter_game_features bf ON bg.game_id=bf.game_id AND bg.batter_id=bf.batter_id
    LEFT JOIN pitcher_game_features pf ON bg.game_id=pf.game_id AND bg.batter_id=pf.batter_id
    """
    return pd.read_sql_query(sql, conn)

def family_for(c):
    if c.startswith(("batter_barrel","batter_hard_hit","batter_avg_ev","batter_max_ev")):
        return "impact_quality"
    if c.startswith(("batter_fly_ball","batter_la_20_35","batter_pull_air")):
        return "air_shape"
    if c.startswith(("pitcher_barrel","pitcher_hard_hit","pitcher_avg_ev","pitcher_max_ev")):
        return "pitcher_impact_allowed"
    if c.startswith(("pitcher_fly_ball","pitcher_la_20_35","pitcher_pull_air","pitcher_hrfb")):
        return "pitcher_air_damage_allowed"
    if c in {"temp_f","wind_speed_mph","wind_toward_pull_field","hitterish_park","pitcherish_park"}:
        return "environment"
    if c in {"lineup_spot","expected_pa_v1"}:
        return "opportunity"
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--corr-threshold", type=float, default=0.85)
    args = ap.parse_args()

    pd = require_pandas()
    conn = sqlite3.connect(args.db)
    df = load_dataset(conn, pd)
    conn.close()

    df["actual_hr"] = pd.to_numeric(df["actual_hr"], errors="coerce")
    feature_cols = []
    for c in df.columns:
        if c in {"actual_hr","game_id","batter_id"}:
            continue
        if any(x in c for x in ["_7d","_15d","_30d","_60d"]) or c in {
            "temp_f","wind_speed_mph","wind_toward_pull_field",
            "hitterish_park","pitcherish_park","lineup_spot","expected_pa_v1"
        }:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                feature_cols.append(c)

    print("HR_FEATURE_CORRELATION_A")
    print("========================")
    print(f"db: {args.db}")
    print(f"rows: {len(df)}")
    print(f"hr_hits: {int(df['actual_hr'].sum())}")
    print(f"feature_cols: {len(feature_cols)}")

    rows = []
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        n = int(s.notna().sum())
        if n < args.min_n:
            continue
        corr = s.corr(df["actual_hr"], method="spearman")
        nonzero = int((s.fillna(0) != 0).sum())
        rows.append({
            "feature": c, "family": family_for(c), "n": n, "nonzero": nonzero,
            "spearman_to_hr": corr, "abs_corr": abs(corr) if corr == corr else 0
        })

    res = pd.DataFrame(rows)
    if res.empty:
        print("No features met min-n. Lower --min-n or build more rows.")
        return
    res = res.sort_values(["family","abs_corr"], ascending=[True, False])

    print("\nTOP FEATURES BY FAMILY")
    print("----------------------")
    champions = []
    for fam, sub in res.groupby("family"):
        print(f"\n[{fam}]")
        top = sub.sort_values("abs_corr", ascending=False).head(12)
        for _, r in top.iterrows():
            print(f"{r['feature']:46s} n={int(r['n']):5d} nonzero={int(r['nonzero']):5d} spearman={r['spearman_to_hr']:+.4f}")
        if len(top):
            champions.append(top.iloc[0]["feature"])

    print("\nHIGH CORRELATION PAIRS WITHIN FAMILY")
    print("------------------------------------")
    for fam, sub in res.groupby("family"):
        cols = list(sub["feature"])
        if len(cols) < 2:
            continue
        mat = df[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman").abs()
        pairs = []
        for i, a in enumerate(cols):
            for b in cols[i+1:]:
                val = mat.loc[a, b]
                if val == val and val >= args.corr_threshold:
                    pairs.append((val, a, b))
        if pairs:
            print(f"\n[{fam}]")
            for val, a, b in sorted(pairs, reverse=True)[:20]:
                print(f"{val:.3f}  {a}  <->  {b}")

    print("\nSUGGESTED REDUCED LOGISTIC FEATURE SET")
    print("--------------------------------------")
    for c in champions:
        print(c)

    out = Path(args.db).parent / "hr_feature_correlation_a.csv"
    res.to_csv(out, index=False)
    print(f"\nCSV_WRITTEN: {out}")
    print("DONE")

if __name__ == "__main__":
    main()
