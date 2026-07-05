#!/usr/bin/env python3
"""
HR Dataset Analysis A — Prop Edge MLB API

Reads the real batter-game HR dataset produced by hr_dataset_builder_a.py.

It tests the exact HR ingredients:
- batter_barrel_rate
- recent_barrel_rate
- hard_hit_rate / EV
- launch_angle band
- pull_air_rate
- pitcher_barrel_allowed_rate
- pitcher_fly_ball_rate
- pitcher HR/FB
- park factor by batter handedness
- wind toward pull field
- FanDuel HR odds

Run:
    python hr_dataset_analysis_a.py

or:
    python hr_dataset_analysis_a.py --input /data/hr_model/hr_batter_game_dataset.csv --min-n 10
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def default_input() -> Path:
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model")) / "hr_batter_game_dataset.csv"
    if p.exists():
        return p
    return Path("./hr_model/hr_batter_game_dataset.csv")


def fnum(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "" or x == "None":
            return default
        return float(x)
    except Exception:
        return default


def bval(x: Any) -> Optional[bool]:
    s = str(x).lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def rate(rows: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    n = len(rows)
    h = sum(1 for r in rows if int(float(r.get("actual_hr") or 0)) == 1)
    return h, n, h / n if n else 0.0


def wilson_lower(h: int, n: int, z: float = 1.28155) -> float:
    if n <= 0:
        return 0
    phat = h / n
    denom = 1 + z*z/n
    center = phat + z*z/(2*n)
    margin = z * math.sqrt((phat*(1-phat) + z*z/(4*n)) / n)
    return max(0.0, (center - margin) / denom)


def fmt(label: str, rows: List[Dict[str, Any]], baseline: float) -> str:
    h, n, r = rate(rows)
    lift = r / baseline if baseline else None
    lb = wilson_lower(h, n)
    lift_s = f"{lift:.2f}x" if lift is not None else "n/a"
    return f"{r:6.2%} n={n:4d} hit={h:3d} lb80={lb:6.2%} lift={lift_s:>7s}  {label}"


def avg(rows: List[Dict[str, Any]], field: str) -> Optional[float]:
    vals = [fnum(r.get(field)) for r in rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def ge(field: str, threshold: float):
    return lambda r: (fnum(r.get(field), -999) or -999) >= threshold


def eq(field: str, value: Any):
    return lambda r: str(r.get(field)) == str(value)


def truth(field: str):
    return lambda r: bval(r.get(field)) is True


def not_empty(field: str):
    return lambda r: r.get(field) not in (None, "", "None")


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
    ap.add_argument("--input", default=str(default_input()))
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Deduplicate in case file was appended multiple times.
    seen = set()
    deduped = []
    for r in rows:
        key = (r.get("game_date"), r.get("game_id"), r.get("batter_id"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    rows = deduped

    h, n, baseline = rate(rows)

    print("HR DATASET ANALYSIS A")
    print("=====================")
    print(f"input: {path}")
    print(f"rows: {n}")
    print(f"hr_hits: {h}")
    print(f"baseline_hr_rate: {baseline:.3%}")
    print(f"min_n: {args.min_n}")

    print("\nFEATURE AVAILABILITY")
    print("--------------------")
    fields = [
        "batter_barrel_rate",
        "recent_barrel_rate",
        "batter_hard_hit_rate",
        "batter_avg_ev",
        "batter_ev90",
        "batter_la_20_35_rate",
        "batter_pull_air_rate",
        "pitcher_barrel_allowed_rate",
        "pitcher_fly_ball_rate",
        "pitcher_hrfb_rate",
        "park_factor_by_batter_hand",
        "wind_toward_pull_field",
        "fanduel_hr_odds",
    ]
    for field in fields:
        have = sum(1 for r in rows if r.get(field) not in (None, "", "None"))
        print(f"{field:34s} {have}/{n}")

    hit_rows = [r for r in rows if int(float(r.get("actual_hr") or 0)) == 1]
    miss_rows = [r for r in rows if int(float(r.get("actual_hr") or 0)) == 0]

    print("\nHIT VS MISS FEATURE AVERAGES")
    print("----------------------------")
    for field in fields:
        ah = avg(hit_rows, field)
        am = avg(miss_rows, field)
        delta = None if ah is None or am is None else ah - am
        print(f"{field:34s} hit_avg={ah} miss_avg={am} delta={delta}")

    print("\nSINGLE FEATURE RULE SEARCH")
    print("--------------------------")
    rules = []

    thresholds = {
        "batter_barrel_rate": [0.06, 0.08, 0.10, 0.12, 0.15],
        "recent_barrel_rate": [0.06, 0.08, 0.10, 0.12, 0.15],
        "batter_hard_hit_rate": [0.35, 0.40, 0.45, 0.50],
        "batter_ev90": [100, 102, 103, 104, 105, 106],
        "batter_la_20_35_rate": [0.20, 0.25, 0.30, 0.35],
        "batter_pull_air_rate": [0.20, 0.25, 0.30, 0.35, 0.40],
        "pitcher_barrel_allowed_rate": [0.06, 0.08, 0.10, 0.12],
        "pitcher_fly_ball_rate": [0.25, 0.30, 0.35, 0.40],
        "pitcher_hrfb_rate": [0.08, 0.10, 0.12, 0.15],
        "pitcher_season_hr9": [0.8, 1.0, 1.2, 1.5],
        "fanduel_hr_implied_prob": [0.10, 0.15, 0.20, 0.25],
    }

    for field, ths in thresholds.items():
        for t in ths:
            filt = apply(rows, [ge(field, t)])
            if len(filt) >= args.min_n:
                rules.append((rate(filt)[2], wilson_lower(rate(filt)[0], rate(filt)[1]), len(filt), f"{field}>={t}", filt))

    for field, val in [
        ("opposing_pitcher_hand", "R"),
        ("opposing_pitcher_hand", "L"),
        ("platoon_bucket", "same_hand"),
        ("platoon_bucket", "platoon_adv"),
        ("wind_toward_pull_field", "True"),
        ("hitterish_park", "True"),
        ("pitcherish_park", "True"),
    ]:
        filt = apply(rows, [eq(field, val)])
        if len(filt) >= args.min_n:
            rules.append((rate(filt)[2], wilson_lower(rate(filt)[0], rate(filt)[1]), len(filt), f"{field}={val}", filt))

    for rrate, lb, rn, label, filt in sorted(rules, key=lambda x: (x[0], x[1], x[2]), reverse=True)[:args.top]:
        print(fmt(label, filt, baseline))

    print("\nCOMBO RULE SEARCH")
    print("-----------------")
    combos = [
        ("barrel10 + EV90_103", [ge("batter_barrel_rate", 0.10), ge("batter_ev90", 103)]),
        ("barrel10 + pull_air30", [ge("batter_barrel_rate", 0.10), ge("batter_pull_air_rate", 0.30)]),
        ("recent_barrel10 + EV90_103", [ge("recent_barrel_rate", 0.10), ge("batter_ev90", 103)]),
        ("barrel10 + pitcher_barrel_allowed08", [ge("batter_barrel_rate", 0.10), ge("pitcher_barrel_allowed_rate", 0.08)]),
        ("EV90_103 + pitcher_hrfb12", [ge("batter_ev90", 103), ge("pitcher_hrfb_rate", 0.12)]),
        ("EV90_103 + LA_band25 + pull_air25", [ge("batter_ev90", 103), ge("batter_la_20_35_rate", 0.25), ge("batter_pull_air_rate", 0.25)]),
        ("barrel10 + vsRHP", [ge("batter_barrel_rate", 0.10), eq("opposing_pitcher_hand", "R")]),
        ("recent_barrel10 + vsRHP", [ge("recent_barrel_rate", 0.10), eq("opposing_pitcher_hand", "R")]),
        ("EV90_103 + wind_pull", [ge("batter_ev90", 103), eq("wind_toward_pull_field", "True")]),
        ("barrel10 + FD_implied_lt20", [ge("batter_barrel_rate", 0.10), lambda r: (fnum(r.get("fanduel_hr_implied_prob"), 999) or 999) <= 0.20]),
    ]

    combo_results = []
    for label, funcs in combos:
        filt = apply(rows, funcs)
        if len(filt) >= args.min_n:
            combo_results.append((rate(filt)[2], wilson_lower(rate(filt)[0], rate(filt)[1]), len(filt), label, filt))

    if not combo_results:
        print(f"No combo rules with n >= {args.min_n}. Build more rows.")
    else:
        for rrate, lb, rn, label, filt in sorted(combo_results, key=lambda x: (x[0], x[1], x[2]), reverse=True)[:args.top]:
            print(fmt(label, filt, baseline))

    print("\nREAD")
    print("----")
    print("This is the real HR model test: every starting batter, not just old HR candidates.")
    print("If availability is low, build more days or add the missing source/logger.")
    print("Do not trust any rule with tiny sample. Use min-n 50+ once the dataset grows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
