#!/usr/bin/env python3
"""
HR SQLite Dataset Analysis B

Analyzes the leak-safe SQLite HR model dataset, including pull-air features
from hr_sqlite_feature_builder_b.py.

Run:
    python hr_sqlite_dataset_analysis_b.py --min-n 20 --min-bbe 5
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def default_db_path() -> Path:
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    if p.parent.exists():
        return p / "hr_model.sqlite"
    return Path("./hr_model/hr_model.sqlite")


def fnum(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "" or x == "None":
            return default
        return float(x)
    except Exception:
        return default


def inum(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "" or x == "None":
            return default
        return int(float(x))
    except Exception:
        return default


def table_cols(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def load_rows(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    bf_cols = [c for c in table_cols(conn, "batter_game_features") if c not in {"game_id", "batter_id"}]
    pf_cols = [c for c in table_cols(conn, "pitcher_game_features") if c not in {"game_id", "batter_id", "opposing_pitcher_id"}]

    select_parts = [
        "bg.game_date AS game_date",
        "bg.game_id AS game_id",
        "bg.batter_id AS batter_id",
        "bg.batter_name AS batter_name",
        "bg.lineup_spot AS lineup_spot",
        "bg.batter_hand AS batter_hand",
        "bg.pitcher_hand AS pitcher_hand",
        "bg.platoon_bucket AS platoon_bucket",
        "bg.park_bucket AS park_bucket",
        "bg.hitterish_park AS hitterish_park",
        "bg.pitcherish_park AS pitcherish_park",
        "bg.temp_f AS temp_f",
        "bg.wind_speed_mph AS wind_speed_mph",
        "bg.weather_wind_bucket AS weather_wind_bucket",
        "bg.wind_toward_pull_field AS wind_toward_pull_field",
        "bg.opposing_pitcher_id AS opposing_pitcher_id",
        "bg.opposing_pitcher_name AS opposing_pitcher_name",
        "bg.actual_hr AS actual_hr",
    ]
    select_parts += [f"bf.{c} AS {c}" for c in bf_cols]
    select_parts += [f"pf.{c} AS {c}" for c in pf_cols]

    sql = f"""
    SELECT {", ".join(select_parts)}
    FROM batter_games bg
    LEFT JOIN batter_game_features bf
      ON bg.game_id = bf.game_id
     AND bg.batter_id = bf.batter_id
    LEFT JOIN pitcher_game_features pf
      ON bg.game_id = pf.game_id
     AND bg.batter_id = pf.batter_id
    ORDER BY bg.game_date, bg.game_id, bg.lineup_spot
    """
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(sql).fetchall()]


def rate(rows: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    n = len(rows)
    h = sum(1 for r in rows if inum(r.get("actual_hr")) == 1)
    return h, n, h / n if n else 0.0


def fmt(label: str, rows: List[Dict[str, Any]], baseline: float) -> str:
    h, n, rr = rate(rows)
    lift = rr / baseline if baseline else None
    lift_s = f"{lift:.2f}x" if lift is not None else "n/a"
    return f"{rr:6.2%} n={n:4d} hit={h:3d} lift={lift_s:>7s}  {label}"


def avg(rows: List[Dict[str, Any]], field: str) -> Optional[float]:
    vals = [fnum(r.get(field)) for r in rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def has_feature(r: Dict[str, Any], field: str) -> bool:
    return fnum(r.get(field)) is not None


def ge(field: str, th: float):
    return lambda r: (fnum(r.get(field), -999) or -999) >= th


def eq(field: str, val: Any):
    return lambda r: str(r.get(field)) == str(val)


def min_sample(field: str, th: int):
    return lambda r: (fnum(r.get(field), 0) or 0) >= th


def apply(rows: List[Dict[str, Any]], funcs) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        try:
            if all(fn(r) for fn in funcs):
                out.append(r)
        except Exception:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--min-bbe", type=int, default=5)
    ap.add_argument("--top", type=int, default=80)
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    conn = sqlite3.connect(db)
    rows = load_rows(conn)

    h, n, baseline = rate(rows)
    print("HR SQLITE DATASET ANALYSIS B")
    print("============================")
    print(f"db: {db}")
    print(f"rows: {n}")
    print(f"hr_hits: {h}")
    print(f"baseline_hr_rate: {baseline:.3%}")
    print(f"min_n: {args.min_n}")
    print(f"min_bbe: {args.min_bbe}")

    dates = sorted(set(str(r.get("game_date")) for r in rows))
    print(f"date_count: {len(dates)}")
    if dates:
        print(f"date_range: {dates[0]} to {dates[-1]}")

    feature_fields = []
    for r in rows[:1]:
        feature_fields = [k for k in r.keys() if ("_7d" in k or "_15d" in k or "_30d" in k or "_60d" in k)]
    print("\nFEATURE AVAILABILITY")
    print("--------------------")
    for field in sorted(feature_fields):
        have = sum(1 for r in rows if has_feature(r, field))
        nonzero = sum(1 for r in rows if (fnum(r.get(field), 0) or 0) != 0)
        print(f"{field:46s} have={have:4d}/{n:<4d} nonzero={nonzero:4d}/{n:<4d}")

    hit_rows = [r for r in rows if inum(r.get("actual_hr")) == 1]
    miss_rows = [r for r in rows if inum(r.get("actual_hr")) == 0]

    print("\nHIT VS MISS AVERAGES")
    print("--------------------")
    key_avg_fields = []
    for w in ("7d", "15d", "30d", "60d"):
        key_avg_fields += [
            f"batter_barrel_rate_{w}",
            f"batter_hard_hit_rate_{w}",
            f"batter_la_20_35_rate_{w}",
            f"batter_fly_ball_rate_{w}",
            f"batter_pull_air_rate_{w}",
            f"batter_avg_ev_{w}",
            f"batter_max_ev_{w}",
            f"pitcher_barrel_allowed_rate_{w}",
            f"pitcher_fly_ball_rate_{w}",
            f"pitcher_pull_air_allowed_rate_{w}",
            f"pitcher_hrfb_rate_{w}",
        ]
    for field in key_avg_fields:
        if field not in feature_fields:
            continue
        ah = avg(hit_rows, field)
        am = avg(miss_rows, field)
        delta = None if ah is None or am is None else ah - am
        print(f"{field:46s} hit_avg={ah} miss_avg={am} delta={delta}")

    print("\nSINGLE FEATURE RULE SEARCH")
    print("--------------------------")
    rules = []

    thresholds = {}
    for w in ("7d", "15d", "30d", "60d"):
        thresholds.update({
            f"batter_barrel_rate_{w}": [0.05, 0.08, 0.10, 0.12, 0.15],
            f"batter_hard_hit_rate_{w}": [0.35, 0.40, 0.45, 0.50],
            f"batter_la_20_35_rate_{w}": [0.20, 0.25, 0.30, 0.35],
            f"batter_fly_ball_rate_{w}": [0.25, 0.30, 0.35, 0.40, 0.45],
            f"batter_pull_air_rate_{w}": [0.20, 0.25, 0.30, 0.35, 0.40],
            f"batter_avg_ev_{w}": [88, 90, 92, 94],
            f"batter_max_ev_{w}": [102, 105, 108, 110],
            f"pitcher_barrel_allowed_rate_{w}": [0.05, 0.08, 0.10, 0.12],
            f"pitcher_fly_ball_rate_{w}": [0.25, 0.30, 0.35, 0.40],
            f"pitcher_pull_air_allowed_rate_{w}": [0.20, 0.25, 0.30, 0.35, 0.40],
            f"pitcher_hrfb_rate_{w}": [0.08, 0.10, 0.12, 0.15],
        })

    for field, ths in thresholds.items():
        if field not in feature_fields:
            continue
        if field.startswith("batter_"):
            sample_field = "batter_bbe_" + field.split("_")[-1]
        else:
            sample_field = "pitcher_bbe_allowed_" + field.split("_")[-1]
        for th in ths:
            funcs = [ge(field, th)]
            if sample_field in feature_fields:
                funcs.append(min_sample(sample_field, args.min_bbe))
            filt = apply(rows, funcs)
            if len(filt) >= args.min_n:
                rules.append((rate(filt)[2], len(filt), f"{field}>={th}", filt))

    for field, val in [
        ("platoon_bucket", "platoon_adv"),
        ("platoon_bucket", "same_hand"),
        ("pitcher_hand", "R"),
        ("pitcher_hand", "L"),
        ("hitterish_park", "1"),
        ("pitcherish_park", "1"),
        ("wind_toward_pull_field", "1"),
    ]:
        filt = apply(rows, [eq(field, val)])
        if len(filt) >= args.min_n:
            rules.append((rate(filt)[2], len(filt), f"{field}={val}", filt))

    for rr, rn, label, filt in sorted(rules, key=lambda x: (x[0], x[1]), reverse=True)[:args.top]:
        print(fmt(label, filt, baseline))

    print("\nCOMBO RULE SEARCH")
    print("-----------------")
    combos = []
    for w in ("7d", "15d", "30d", "60d"):
        combos.extend([
            (f"{w}: maxEV>=105 + flyball>=.35", [ge(f"batter_max_ev_{w}", 105), ge(f"batter_fly_ball_rate_{w}", .35), min_sample(f"batter_bbe_{w}", args.min_bbe)]),
            (f"{w}: maxEV>=105 + pullair>=.30", [ge(f"batter_max_ev_{w}", 105), ge(f"batter_pull_air_rate_{w}", .30), min_sample(f"batter_bbe_{w}", args.min_bbe)]),
            (f"{w}: flyball>=.35 + pullair>=.30", [ge(f"batter_fly_ball_rate_{w}", .35), ge(f"batter_pull_air_rate_{w}", .30), min_sample(f"batter_bbe_{w}", args.min_bbe)]),
            (f"{w}: barrel>=.10 + LA20_35>=.25", [ge(f"batter_barrel_rate_{w}", .10), ge(f"batter_la_20_35_rate_{w}", .25), min_sample(f"batter_bbe_{w}", args.min_bbe)]),
            (f"{w}: hardhit>=.45 + pullair>=.30", [ge(f"batter_hard_hit_rate_{w}", .45), ge(f"batter_pull_air_rate_{w}", .30), min_sample(f"batter_bbe_{w}", args.min_bbe)]),
            (f"{w}: barrel>=.10 + pitcher_barrel_allowed>=.08", [ge(f"batter_barrel_rate_{w}", .10), ge(f"pitcher_barrel_allowed_rate_{w}", .08), min_sample(f"batter_bbe_{w}", args.min_bbe), min_sample(f"pitcher_bbe_allowed_{w}", args.min_bbe)]),
        ])

    combo_results = []
    for label, funcs in combos:
        filt = apply(rows, funcs)
        if len(filt) >= args.min_n:
            combo_results.append((rate(filt)[2], len(filt), label, filt))

    if not combo_results:
        print(f"No combo rules with n >= {args.min_n}. Build more batter_games / load more BBE lookback.")
    else:
        for rr, rn, label, filt in sorted(combo_results, key=lambda x: (x[0], x[1]), reverse=True)[:args.top]:
            print(fmt(label, filt, baseline))

    print("\nREAD")
    print("----")
    print("This is the correct analysis layer for the SQLite pipeline with pull-air included.")
    print("If pitcher features are low, load more prior BBE history before the target game dates.")
    print("Do not trust tiny samples. Move toward 2,000+ rows before feature locking.")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
