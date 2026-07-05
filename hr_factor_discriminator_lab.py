#!/usr/bin/env python3
"""
HR Factor Discriminator Lab for Prop Edge MLB API.

Purpose:
- Diagnostic only. Does NOT modify api.py.
- Fixes the problem found in hr_boost_score_lab:
    the boost score was too broad, so almost every base HR candidate qualified.
- This script looks only inside the current base HR bucket:
    hr_score >= 1.70 AND season_slg >= .450
- It compares HR hits vs HR misses and asks:
    Which factors actually separate winners from losers?
- Then it builds a narrow rule proposal using only the factors that discriminate.

Requires:
- hr_calibration_probe.py in same folder
- hr_context_factor_probe.py in same folder, using the corrected V2 content

Run:
    python hr_factor_discriminator_lab.py --days 180 --min-n 3

Conservative:
    python hr_factor_discriminator_lab.py --days 180 --min-n 5

Optional:
    python hr_factor_discriminator_lab.py --days 180 --min-n 3 --write-csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

try:
    import hr_calibration_probe as cal
except Exception as e:
    raise SystemExit(f"ERROR: missing hr_calibration_probe.py: {e}")

try:
    import hr_context_factor_probe as ctx
except Exception as e:
    raise SystemExit(
        "ERROR: missing hr_context_factor_probe.py with corrected V2 content. "
        f"Import failed: {e}"
    )


def fnum(x: Any, default: Optional[float] = None) -> Optional[float]:
    return cal.safe_float(x, default)


def wilson_lower(hits: int, n: int, z: float = 1.28155) -> float:
    if n <= 0:
        return 0.0
    phat = hits / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return max(0.0, (center - margin) / denom)


def summarize(rows: List[Dict[str, Any]], baseline: float) -> Dict[str, Any]:
    n = len(rows)
    hits = sum(1 for r in rows if r.get("result") == "hit")
    rate = hits / n if n else 0.0
    lower = wilson_lower(hits, n)
    lift = rate / baseline if baseline > 0 else None

    def avg(field: str) -> Optional[float]:
        vals = [fnum(r.get(field)) for r in rows]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n": n,
        "hits": hits,
        "misses": n - hits,
        "rate": rate,
        "wilson80_lower": lower,
        "lift": lift,
        "penalized": lower * math.log1p(n),
        "avg_hr_score": avg("hr_score"),
        "avg_season_slg": avg("season_slg"),
        "avg_recent_iso": avg("recent_iso"),
        "avg_h2h_slg": avg("h2h_slg"),
        "avg_temp": avg("weather_temp_f"),
        "avg_pHR9": avg("opp_pitcher_hr9"),
    }


def line(label: str, s: Dict[str, Any]) -> str:
    lift = f"{s['lift']:.2f}x" if s["lift"] is not None else "n/a"
    return (
        f"{s['rate']:6.1%} n={s['n']:3d} hit={s['hits']:3d} miss={s['misses']:3d} "
        f"lb80={s['wilson80_lower']:5.1%} lift={lift:>6s} "
        f"avg_score={round(s['avg_hr_score'],3) if s['avg_hr_score'] is not None else None} "
        f"avg_iso={round(s['avg_recent_iso'],3) if s['avg_recent_iso'] is not None else None} "
        f"avg_h2h={round(s['avg_h2h_slg'],3) if s['avg_h2h_slg'] is not None else None} "
        f"avg_temp={round(s['avg_temp'],1) if s['avg_temp'] is not None else None} "
        f"avg_pHR9={round(s['avg_pHR9'],2) if s['avg_pHR9'] is not None else None}  "
        f"{label}"
    )


def has_flag(row: Dict[str, Any], text: str) -> bool:
    hay = " ".join(str(row.get(k) or "") for k in ("bvp_flag", "reject_reason", "prediction_tier", "board_warnings"))
    return text.lower() in hay.lower()


def factor_map(row: Dict[str, Any]) -> Dict[str, bool]:
    temp = fnum(row.get("weather_temp_f"), -999) or -999
    p_hr9 = fnum(row.get("opp_pitcher_hr9"), -999) or -999
    iso = fnum(row.get("recent_iso"), -999) or -999
    h2h = fnum(row.get("h2h_slg"), -999) or -999
    score = fnum(row.get("hr_score"), -999) or -999
    slg = fnum(row.get("season_slg"), -999) or -999
    spot = fnum(row.get("lineup_spot"), 99) or 99
    park = str(row.get("park_bucket") or "")
    wind = str(row.get("weather_wind_bucket") or "")
    opp_hand = str(row.get("opp_pitcher_hand") or "")
    platoon = str(row.get("platoon_bucket") or "")
    bat_side = str(row.get("batter_side") or "")

    hitterish = park in getattr(ctx, "HITTERISH_PARK_BUCKETS", set())
    pitcher_park = park in getattr(ctx, "BAD_PARK_BUCKETS", set())

    return {
        "vs_rhp": opp_hand == "R",
        "vs_lhp": opp_hand == "L",
        "same_hand": platoon == "same_hand",
        "platoon_adv": platoon == "platoon_adv",
        "bat_r": bat_side == "R",
        "bat_l": bat_side == "L",
        "temp_ge_80": temp >= 80,
        "temp_ge_85": temp >= 85,
        "temp_ge_90": temp >= 90,
        "wind_out": wind == "wind_out",
        "wind_in": wind == "wind_in",
        "not_wind_in": wind != "wind_in",
        "hitterish_park": hitterish,
        "pitcher_park": pitcher_park,
        "not_pitcher_park": not pitcher_park,
        "pitcher_hr9_ge_1_0": p_hr9 >= 1.0,
        "pitcher_hr9_ge_1_2": p_hr9 >= 1.2,
        "pitcher_hr9_ge_1_5": p_hr9 >= 1.5,
        "recent_iso_ge_300": iso >= 0.300,
        "recent_iso_ge_350": iso >= 0.350,
        "h2h_slg_ge_700": h2h >= 0.700,
        "h2h_slg_ge_900": h2h >= 0.900,
        "hr_score_ge_180": score >= 1.80,
        "hr_score_ge_190": score >= 1.90,
        "season_slg_ge_500": slg >= 0.500,
        "lineup_top3": spot <= 3,
        "lineup_top5": spot <= 5,
        "quality_ok": bool(row.get("quality_ok")),
        "hr_elite": (row.get("tier_bucket") or cal.hr_tier(row)) == "hr_elite",
        "no_bad_flags": not (has_flag(row, "sum_avoid") or has_flag(row, "struggles") or has_flag(row, "sum_lean")),
    }


def load_enriched(api_mod: Any, days: int, exact_date: Optional[str], max_grade_rows: int) -> Tuple[List[Dict[str, Any]], Counter]:
    rows, meta = ctx.load_all_rows(api_mod, days, exact_date, max_grade_rows)
    out = []
    err = 0
    for r in rows:
        try:
            out.append(ctx.enrich_row(api_mod, r))
        except Exception:
            err += 1
            out.append(dict(r))
    meta["enrich_errors"] = err
    return out, meta


def is_base(row: Dict[str, Any], score: float, slg: float) -> bool:
    return (fnum(row.get("hr_score"), -999) or -999) >= score and (fnum(row.get("season_slg"), -999) or -999) >= slg


def apply_rule(rows: List[Dict[str, Any]], factors: Tuple[str, ...]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        fmap = factor_map(r)
        if all(fmap.get(f) for f in factors):
            out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--date", default=None)
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--base-score", type=float, default=1.70)
    ap.add_argument("--base-slg", type=float, default=0.450)
    ap.add_argument("--max-grade-rows", type=int, default=0)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--write-csv", action="store_true")
    args = ap.parse_args()

    print("HR FACTOR DISCRIMINATOR LAB")
    print("===========================")
    print("Diagnostic only. Finds which factors separate HR hits from misses inside the 30% base bucket.")

    api_mod = importlib.import_module("api")
    rows, meta = load_enriched(api_mod, args.days, args.date, args.max_grade_rows)

    n = len(rows)
    hits = sum(1 for r in rows if r.get("result") == "hit")
    baseline = hits / n if n else 0.0

    base = [r for r in rows if is_base(r, args.base_score, args.base_slg)]
    base_hits = [r for r in base if r.get("result") == "hit"]
    base_misses = [r for r in base if r.get("result") != "hit"]
    base_s = summarize(base, baseline)

    print("\nINPUT / BASELINE")
    print("----------------")
    for k, v in meta.items():
        print(f"{k}: {v}")
    print(f"all_graded_hr_rows: {n}")
    print(f"all_hits: {hits}")
    print(f"all_misses: {n - hits}")
    print(f"overall_baseline_hr_rate: {baseline:.1%}")

    print("\nBASE 30 MODEL")
    print("-------------")
    print(line(f"hr_score>={args.base_score:.2f} AND season_slg>={args.base_slg:.3f}", base_s))

    print("\nHIT VS MISS FACTOR PREVALENCE")
    print("-----------------------------")
    all_factor_names = sorted(factor_map(base[0]).keys()) if base else []
    prevalence_rows = []
    for f in all_factor_names:
        h_yes = sum(1 for r in base_hits if factor_map(r).get(f))
        m_yes = sum(1 for r in base_misses if factor_map(r).get(f))
        h_rate = h_yes / len(base_hits) if base_hits else 0.0
        m_rate = m_yes / len(base_misses) if base_misses else 0.0
        delta = h_rate - m_rate
        # add tiny smoothing for odds ratio
        odds_ratio = ((h_yes + 0.5) / (len(base_hits) - h_yes + 0.5)) / ((m_yes + 0.5) / (len(base_misses) - m_yes + 0.5)) if base_hits and base_misses else None
        prevalence_rows.append((delta, odds_ratio or 0, h_yes, m_yes, f, h_rate, m_rate))

    for delta, odds_ratio, h_yes, m_yes, f, h_rate, m_rate in sorted(prevalence_rows, key=lambda x: (x[0], x[1]), reverse=True):
        print(
            f"{f:24s} hit_yes={h_yes}/{len(base_hits)} ({h_rate:.0%}) "
            f"miss_yes={m_yes}/{len(base_misses)} ({m_rate:.0%}) "
            f"delta={delta:+.0%} smoothed_OR={odds_ratio:.2f}"
        )

    print("\nBASE MEMBERS")
    print("------------")
    for r in base:
        fmap = factor_map(r)
        active = [k for k, v in fmap.items() if v]
        print(
            f"{str(r.get('result')):4s} {str(r.get('player')):22s} "
            f"score={fnum(r.get('hr_score'))} slg={fnum(r.get('season_slg'))} iso={fnum(r.get('recent_iso'))} "
            f"temp={r.get('weather_temp_f')} park={r.get('park_bucket')} "
            f"bat={r.get('batter_side')} vs={r.get('opp_pitcher_hand')} platoon={r.get('platoon_bucket')} "
            f"pHR9={None if r.get('opp_pitcher_hr9') is None else round(r.get('opp_pitcher_hr9'),2)} "
            f"active={','.join(active)}"
        )

    # Choose candidate factors that appeared more in hits than misses and are not just trivial always-on.
    positive_factors = []
    for delta, odds_ratio, h_yes, m_yes, f, h_rate, m_rate in prevalence_rows:
        support = h_yes + m_yes
        if delta > 0 and support >= 2:
            positive_factors.append(f)

    # Always include factors we care about if present.
    priority = [
        "vs_rhp", "same_hand", "temp_ge_80", "temp_ge_85", "temp_ge_90",
        "pitcher_hr9_ge_1_0", "pitcher_hr9_ge_1_2",
        "recent_iso_ge_350", "h2h_slg_ge_900", "not_pitcher_park",
        "no_bad_flags", "lineup_top5"
    ]
    candidate_factors = []
    for f in priority + positive_factors:
        if f in all_factor_names and f not in candidate_factors:
            candidate_factors.append(f)

    print("\nRULE SEARCH USING DISCRIMINATING FACTORS")
    print("----------------------------------------")
    print("Candidate factors:", ", ".join(candidate_factors))

    results = []
    for k in range(1, min(4, len(candidate_factors)) + 1):
        for combo in itertools.combinations(candidate_factors, k):
            filt = apply_rule(base, combo)
            if len(filt) < args.min_n:
                continue
            s = summarize(filt, baseline)
            results.append((s["rate"], s["wilson80_lower"], s["penalized"], s["n"], combo, s))

    for rate, lb, pen, n2, combo, s in sorted(results, key=lambda x: (x[0], x[1], x[3]), reverse=True)[:args.top]:
        print(line("base + " + " AND ".join(combo), s))

    print("\nPENALIZED RANKING")
    print("-----------------")
    for rate, lb, pen, n2, combo, s in sorted(results, key=lambda x: (x[2], x[0], x[3]), reverse=True)[:args.top]:
        print(f"pen={pen:.4f}  {line('base + ' + ' AND '.join(combo), s)}")

    print("\nRECOMMENDED MODEL CHANGE")
    print("------------------------")
    usable = [x for x in results if x[5]["n"] >= max(args.min_n, 3)]
    if usable:
        best = max(usable, key=lambda x: (x[2], x[0], x[3]))
        _, _, _, _, combo, s = best
        print("Best discriminating rule:", "base + " + " AND ".join(combo))
        print(line("selected", s))
        if s["rate"] > base_s["rate"]:
            print("READ: This is the best current improvement candidate. Use as BOOSTED HR WATCHLIST only until larger sample/odds confirm.")
        else:
            print("READ: No improvement over base yet.")
    else:
        print("No usable discriminating rule found.")

    print("\nIMPLEMENTATION INTERPRETATION")
    print("-----------------------------")
    print("Do not make HR official yet.")
    print("Improve the model by changing HR from one loose watchlist to a two-stage selector:")
    print("  Stage 1: HR_CORE_LONGSHOT = hr_score>=1.70 AND season_slg>=.450")
    print("  Stage 2: HR_BOOSTED_LONGSHOT = CORE plus the best discriminating factor combo above")
    print("Then add HR odds and grade EV before official promotion.")

    if args.write_csv:
        out = Path("hr_factor_discriminator_rows.csv")
        fields = sorted({k for r in base for k in r.keys()})
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in base:
                w.writerow(r)
        print(f"\nCSV_WRITTEN: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
