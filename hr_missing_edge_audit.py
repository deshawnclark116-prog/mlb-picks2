#!/usr/bin/env python3
"""
HR Missing Edge Audit for Prop Edge MLB API.

Diagnostic only. Does NOT modify api.py.

Goal:
- Stop re-testing the same 29 rows with only the same fields.
- Audit exactly what the current HR scorer is missing.
- Separate current available factors into:
    1) already tested
    2) useful but weak/sample-limited
    3) missing high-value features that should be logged next
- Print a recommended HR v2 feature plan.

Requires:
- hr_calibration_probe.py
- hr_context_factor_probe.py with corrected V2 content

Run:
    python hr_missing_edge_audit.py --days 180

Optional:
    python hr_missing_edge_audit.py --days 180 --write-csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import hr_calibration_probe as cal
except Exception as e:
    raise SystemExit(f"ERROR: missing hr_calibration_probe.py: {e}")

try:
    import hr_context_factor_probe as ctx
except Exception as e:
    raise SystemExit(f"ERROR: missing corrected hr_context_factor_probe.py: {e}")


def fnum(x: Any, default: Optional[float] = None) -> Optional[float]:
    return cal.safe_float(x, default)


def pct(x: float) -> str:
    return f"{x*100:.1f}%"


def load_enriched(api_mod: Any, days: int, exact_date: Optional[str]) -> List[Dict[str, Any]]:
    rows, _meta = ctx.load_all_rows(api_mod, days, exact_date, 0)
    out = []
    for r in rows:
        try:
            out.append(ctx.enrich_row(api_mod, r))
        except Exception:
            out.append(dict(r))
    return out


def is_base(row: Dict[str, Any]) -> bool:
    return (fnum(row.get("hr_score"), -999) or -999) >= 1.70 and (fnum(row.get("season_slg"), -999) or -999) >= 0.450


def has(row: Dict[str, Any], field: str) -> bool:
    v = row.get(field)
    return v not in (None, "", "unknown", "park_unknown", "temp_unknown", "wind_unknown", "platoon_unknown")


def result_rate(rows: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    n = len(rows)
    h = sum(1 for r in rows if r.get("result") == "hit")
    return h, n, h / n if n else 0.0


def factor_bool(row: Dict[str, Any], name: str) -> bool:
    temp = fnum(row.get("weather_temp_f"), -999) or -999
    p_hr9 = fnum(row.get("opp_pitcher_hr9"), -999) or -999
    iso = fnum(row.get("recent_iso"), -999) or -999
    h2h = fnum(row.get("h2h_slg"), -999) or -999
    score = fnum(row.get("hr_score"), -999) or -999
    slg = fnum(row.get("season_slg"), -999) or -999
    spot = fnum(row.get("lineup_spot"), 99) or 99
    opp = row.get("opp_pitcher_hand")
    platoon = row.get("platoon_bucket")
    wind = row.get("weather_wind_bucket")
    park = row.get("park_bucket")
    hitterish = park in getattr(ctx, "HITTERISH_PARK_BUCKETS", set())
    pitcherish = park in getattr(ctx, "BAD_PARK_BUCKETS", set())

    mapping = {
        "vs_rhp": opp == "R",
        "vs_lhp": opp == "L",
        "same_hand": platoon == "same_hand",
        "platoon_adv": platoon == "platoon_adv",
        "temp80": temp >= 80,
        "temp85": temp >= 85,
        "temp90": temp >= 90,
        "wind_out": wind == "wind_out",
        "wind_in": wind == "wind_in",
        "hitterish_park": hitterish,
        "pitcher_park": pitcherish,
        "not_pitcher_park": not pitcherish,
        "pHR9_1.0": p_hr9 >= 1.0,
        "pHR9_1.2": p_hr9 >= 1.2,
        "recent_iso_300": iso >= 0.300,
        "recent_iso_350": iso >= 0.350,
        "h2h_700": h2h >= 0.700,
        "h2h_900": h2h >= 0.900,
        "hr_score_180": score >= 1.80,
        "hr_score_190": score >= 1.90,
        "slg_500": slg >= 0.500,
        "top3": spot <= 3,
        "top5": spot <= 5,
    }
    return bool(mapping.get(name, False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--date", default=None)
    ap.add_argument("--write-csv", action="store_true")
    args = ap.parse_args()

    print("HR MISSING EDGE AUDIT")
    print("=====================")
    print("Diagnostic only. Audits what the HR model is missing instead of re-tuning the same small sample.")

    api_mod = importlib.import_module("api")
    print(f"api_version: {getattr(api_mod, 'VERSION', 'unknown')}")
    print(f"data_dir: {getattr(api_mod, 'DATA_DIR', '/data')}")
    print(f"filters: days={args.days} date={args.date}")

    rows = load_enriched(api_mod, args.days, args.date)
    all_hits, all_n, all_rate = result_rate(rows)
    base = [r for r in rows if is_base(r)]
    base_hits, base_n, base_rate = result_rate(base)

    print("\nCURRENT SAMPLE")
    print("--------------")
    print(f"all_graded_hr_rows: {all_n}")
    print(f"all_hr_hits: {all_hits}")
    print(f"all_hr_rate: {pct(all_rate)}")
    print(f"base_bucket_rows: {base_n}")
    print(f"base_bucket_hits: {base_hits}")
    print(f"base_bucket_rate: {pct(base_rate)}")
    print("base_rule: hr_score>=1.70 AND season_slg>=.450")

    print("\nCURRENT FIELD COMPLETENESS")
    print("--------------------------")
    fields = [
        "hr_score", "season_slg", "recent_iso", "h2h_slg", "lineup_spot",
        "venue", "park_bucket", "weather_temp_f", "weather_wind_bucket",
        "batter_side", "opp_pitcher", "opp_pitcher_hand", "platoon_bucket",
        "opp_pitcher_hr9", "opp_pitcher_hr_per_bf"
    ]
    for f in fields:
        n_has = sum(1 for r in rows if has(r, f))
        print(f"{f:24s} {n_has:3d}/{all_n}")

    print("\nCURRENT FACTORS THAT SEPARATE BASE HITS FROM BASE MISSES")
    print("--------------------------------------------------------")
    factors = [
        "vs_rhp", "vs_lhp", "same_hand", "platoon_adv", "temp80", "temp85", "temp90",
        "wind_out", "wind_in", "hitterish_park", "pitcher_park", "not_pitcher_park",
        "pHR9_1.0", "pHR9_1.2", "recent_iso_300", "recent_iso_350",
        "h2h_700", "h2h_900", "hr_score_180", "hr_score_190", "slg_500", "top3", "top5"
    ]
    base_hit_rows = [r for r in base if r.get("result") == "hit"]
    base_miss_rows = [r for r in base if r.get("result") != "hit"]
    sep = []
    for fac in factors:
        hy = sum(1 for r in base_hit_rows if factor_bool(r, fac))
        my = sum(1 for r in base_miss_rows if factor_bool(r, fac))
        hp = hy / len(base_hit_rows) if base_hit_rows else 0
        mp = my / len(base_miss_rows) if base_miss_rows else 0
        sep.append((hp - mp, hy, my, hp, mp, fac))
    for delta, hy, my, hp, mp, fac in sorted(sep, reverse=True):
        print(f"{fac:18s} hit_yes={hy}/{len(base_hit_rows)} ({pct(hp)}) miss_yes={my}/{len(base_miss_rows)} ({pct(mp)}) delta={delta:+.1%}")

    print("\nBASE BUCKET MISS AUTOPSY")
    print("------------------------")
    for r in base:
        if r.get("result") == "hit":
            continue
        likely_problem = []
        if factor_bool(r, "vs_lhp"):
            likely_problem.append("vs_lhp")
        if not factor_bool(r, "vs_rhp"):
            likely_problem.append("not_vs_rhp")
        if factor_bool(r, "pitcher_park"):
            likely_problem.append("pitcher_park")
        if not factor_bool(r, "temp80"):
            likely_problem.append("not_hot")
        if factor_bool(r, "wind_in"):
            likely_problem.append("wind_in")
        if not factor_bool(r, "h2h_900"):
            likely_problem.append("h2h_below_900")
        if not factor_bool(r, "recent_iso_350"):
            likely_problem.append("recent_iso_below_350")
        if (fnum(r.get("opp_pitcher_hr9"), 999) or 999) < 1.0:
            likely_problem.append("pitcher_hr9_below_1")
        print(
            f"MISS {str(r.get('player')):22s} score={fnum(r.get('hr_score'))} slg={fnum(r.get('season_slg'))} "
            f"iso={fnum(r.get('recent_iso'))} h2h={fnum(r.get('h2h_slg'))} "
            f"temp={r.get('weather_temp_f')} park={r.get('park_bucket')} vs={r.get('opp_pitcher_hand')} "
            f"platoon={r.get('platoon_bucket')} pHR9={None if r.get('opp_pitcher_hr9') is None else round(r.get('opp_pitcher_hr9'),2)} "
            f"likely_noise={','.join(likely_problem) or 'none'}"
        )

    print("\nHIGH-VALUE FEATURES STILL MISSING FROM THE MODEL")
    print("-----------------------------------------------")
    missing = [
        ("batter_barrel_rate", "Batter true HR contact quality; better than SLG/ISO."),
        ("batter_recent_barrel_or_hardhit", "Tells whether power is currently live, not just season-long."),
        ("batter_pull_air_rate", "HR upside depends on pulled fly balls, not just hard contact."),
        ("batter_launch_angle_band", "Separates line-drive doubles from HR-shaped contact."),
        ("pitcher_barrel_allowed_rate", "Better HR weakness signal than pitcher HR/9."),
        ("pitcher_flyball_or_hrfb_rate", "Pitcher can allow hard contact on ground; HR needs air contact."),
        ("pitcher_pitch_mix_vs_batter_power_zone", "Fastball/slider/changeup shape matters for HR matchup."),
        ("park_handedness_factor", "Yankee Stadium helps LHH differently than RHH; generic park is too crude."),
        ("wind_to_pull_field", "Wind out to center is not the same as wind to a hitter's pull field."),
        ("team_implied_total", "Run environment and PA quality; better than raw moneyline."),
        ("real_fanduel_hr_odds", "Without price, there is no true EV gate."),
        ("closing_line_hr_price", "Needed for calibration and market efficiency check."),
    ]
    for name, why in missing:
        print(f"{name:32s} {why}")

    print("\nWHAT WE ARE PROBABLY MISSING")
    print("----------------------------")
    print("1. Current HR scorer is mostly SLG/ISO/H2H. That is a power profile, not true HR probability.")
    print("2. The model lacks batted-ball shape: barrels, pull-air, launch angle, EV. That is probably the biggest gap.")
    print("3. Pitcher HR/9 helped a little, but pitcher barrel/fly-ball/pitch-mix would be much stronger.")
    print("4. Park/weather is too generic unless tied to batter handedness and pull field.")
    print("5. HR odds are mandatory before official picks, because longshots can be good predictions but bad bets.")

    print("\nRECOMMENDED NEXT BUILD")
    print("----------------------")
    print("Build HR v2 as an enriched logger first, not an official picker:")
    print("A. Keep current HR_CORE_LONGSHOT and HR_BOOSTED_LONGSHOT tiers.")
    print("B. Add daily logging fields for all missing high-value features above.")
    print("C. Pull real FanDuel HR odds if provider supports it; otherwise keep HR watchlist only.")
    print("D. After 100+ graded HR candidates, train/calibrate a true HR probability bucket.")
    print("E. Until then, use current best live filter: CORE + vs_rhp as boosted watchlist; downgrade vs_lhp.")

    if args.write_csv:
        out = Path("hr_missing_edge_audit_rows.csv")
        fields_out = sorted({k for r in rows for k in r.keys()})
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields_out, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nCSV_WRITTEN: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
