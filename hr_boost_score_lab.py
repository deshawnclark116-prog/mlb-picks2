#!/usr/bin/env python3
"""
HR Boost Score Lab for Prop Edge MLB API.
Diagnostic only. Does NOT modify api.py.

It starts with the current proven HR base bucket:
    hr_score >= 1.70 AND season_slg >= .450

Then it builds a context boost score using park/weather/pitcher/platoon factors and
checks whether a boosted sub-bucket beats the current 30% bucket.

Requires these files beside api.py on Render:
    hr_calibration_probe.py
    hr_context_factor_probe.py   # the corrected V2 content is fine even if named without v2

Run:
    python hr_boost_score_lab.py --days 180 --min-n 3
    python hr_boost_score_lab.py --days 180 --min-n 10
"""
from __future__ import annotations

import argparse, csv, importlib, math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import hr_calibration_probe as cal
except Exception as e:
    raise SystemExit(f"ERROR: missing hr_calibration_probe.py: {e}")

try:
    import hr_context_factor_probe as ctx
except Exception as e:
    raise SystemExit("ERROR: missing hr_context_factor_probe.py with corrected V2 content: " + str(e))


def fnum(x: Any, default: Optional[float] = None) -> Optional[float]:
    return cal.safe_float(x, default)


def has_flag(row: Dict[str, Any], text: str) -> bool:
    hay = " ".join(str(row.get(k) or "") for k in ("bvp_flag", "reject_reason", "prediction_tier", "board_warnings"))
    return text.lower() in hay.lower()


def wilson_lower(hits: int, n: int, z: float = 1.28155) -> float:
    if n <= 0:
        return 0.0
    phat = hits / n
    denom = 1 + z*z/n
    center = phat + z*z/(2*n)
    margin = z * math.sqrt((phat*(1-phat) + z*z/(4*n)) / n)
    return max(0.0, (center - margin) / denom)


def summarize(rows: List[Dict[str, Any]], baseline: float) -> Dict[str, Any]:
    n = len(rows)
    hits = sum(1 for r in rows if r.get("result") == "hit")
    rate = hits / n if n else 0.0
    lower = wilson_lower(hits, n)

    def avg(field: str):
        vals = [fnum(r.get(field)) for r in rows]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n": n,
        "hits": hits,
        "misses": n - hits,
        "rate": rate,
        "lower": lower,
        "lift": rate / baseline if baseline else None,
        "pen": lower * math.log1p(n),
        "avg_boost": avg("hr_boost_score"),
        "avg_hr_score": avg("hr_score"),
        "avg_slg": avg("season_slg"),
        "avg_iso": avg("recent_iso"),
        "avg_temp": avg("weather_temp_f"),
        "avg_p_hr9": avg("opp_pitcher_hr9"),
    }


def fmt(label: str, s: Dict[str, Any]) -> str:
    lift = f"{s['lift']:.2f}x" if s["lift"] is not None else "n/a"
    return (
        f"{s['rate']:6.1%} n={s['n']:3d} hit={s['hits']:3d} miss={s['misses']:3d} "
        f"lb80={s['lower']:5.1%} lift={lift:>6s} "
        f"avg_boost={None if s['avg_boost'] is None else round(s['avg_boost'],2)} "
        f"avg_score={None if s['avg_hr_score'] is None else round(s['avg_hr_score'],3)} "
        f"avg_slg={None if s['avg_slg'] is None else round(s['avg_slg'],3)} "
        f"avg_iso={None if s['avg_iso'] is None else round(s['avg_iso'],3)} "
        f"avg_temp={None if s['avg_temp'] is None else round(s['avg_temp'],1)} "
        f"avg_pHR9={None if s['avg_p_hr9'] is None else round(s['avg_p_hr9'],2)}  {label}"
    )


def base_rule(row: Dict[str, Any], score: float, slg: float) -> bool:
    return (fnum(row.get("hr_score"), -999) or -999) >= score and (fnum(row.get("season_slg"), -999) or -999) >= slg


def factor_flags(row: Dict[str, Any]) -> Dict[str, bool]:
    temp = fnum(row.get("weather_temp_f"), -999) or -999
    p_hr9 = fnum(row.get("opp_pitcher_hr9"), -999) or -999
    iso = fnum(row.get("recent_iso"), -999) or -999
    h2h = fnum(row.get("h2h_slg"), -999) or -999
    score = fnum(row.get("hr_score"), -999) or -999
    spot = fnum(row.get("lineup_spot"), 99) or 99
    park = str(row.get("park_bucket") or "")
    wind = str(row.get("weather_wind_bucket") or "")
    opp_hand = str(row.get("opp_pitcher_hand") or "")
    platoon = str(row.get("platoon_bucket") or "")
    hitterish = getattr(ctx, "HITTERISH_PARK_BUCKETS", {"elite_hr_park", "hr_friendly", "slightly_hr_friendly", "weather_sensitive"})
    badparks = getattr(ctx, "BAD_PARK_BUCKETS", {"pitcher_friendly"})
    return {
        "vs_rhp": opp_hand == "R",
        "vs_lhp": opp_hand == "L",
        "same_hand": platoon == "same_hand",
        "platoon_adv": platoon == "platoon_adv",
        "hot_80": temp >= 80,
        "hot_85": temp >= 85,
        "hot_90": temp >= 90,
        "wind_out": wind == "wind_out",
        "not_wind_in": wind != "wind_in",
        "hitterish_park": park in hitterish,
        "not_pitcher_park": park not in badparks,
        "pitcher_hr9_1.0": p_hr9 >= 1.0,
        "pitcher_hr9_1.2": p_hr9 >= 1.2,
        "pitcher_hr9_1.5": p_hr9 >= 1.5,
        "recent_iso_300": iso >= 0.300,
        "recent_iso_350": iso >= 0.350,
        "h2h_slg_700": h2h >= 0.700,
        "h2h_slg_900": h2h >= 0.900,
        "hr_score_180": score >= 1.80,
        "hr_score_190": score >= 1.90,
        "lineup_top3": spot <= 3,
        "lineup_top5": spot <= 5,
        "quality_ok": bool(row.get("quality_ok")),
        "hr_elite": (row.get("tier_bucket") or cal.hr_tier(row)) == "hr_elite",
        "avoid_bad_flags": not (has_flag(row, "sum_avoid") or has_flag(row, "struggles") or has_flag(row, "sum_lean")),
    }


# Deliberately simple, transparent weights. This is a lab score, not a locked model.
WEIGHTS = {
    "vs_rhp": 1.00,
    "same_hand": 1.00,
    "hot_80": 0.50,
    "hot_85": 0.50,
    "hot_90": 0.50,
    "pitcher_hr9_1.0": 0.50,
    "pitcher_hr9_1.2": 0.50,
    "pitcher_hr9_1.5": 0.50,
    "recent_iso_300": 0.50,
    "recent_iso_350": 1.00,
    "h2h_slg_900": 0.50,
    "hr_score_180": 0.75,
    "hr_score_190": 0.75,
    "wind_out": 0.50,
    "not_wind_in": 0.25,
    "hitterish_park": 0.25,
    "not_pitcher_park": 0.25,
    "lineup_top5": 0.25,
    "quality_ok": 0.25,
    "avoid_bad_flags": 0.25,
}


def add_boost(row: Dict[str, Any]) -> Dict[str, Any]:
    rr = dict(row)
    flags = factor_flags(rr)
    score = 0.0
    tags = []
    for k, w in WEIGHTS.items():
        if flags.get(k):
            score += w
            tags.append(k)
    rr["hr_boost_score"] = score
    rr["hr_boost_tags"] = ",".join(tags)
    rr["high_signal_factor_count"] = sum(1 for k in ["vs_rhp", "same_hand", "hot_80", "pitcher_hr9_1.0", "recent_iso_350", "h2h_slg_900", "hr_score_180"] if flags.get(k))
    return rr


def load_rows(api_mod: Any, days: int, exact_date: Optional[str], max_grade_rows: int):
    rows, meta = ctx.load_all_rows(api_mod, days, exact_date, max_grade_rows)
    enriched = []
    for r in rows:
        enriched.append(ctx.enrich_row(api_mod, r))
    return enriched, meta


def build_results(rows: List[Dict[str, Any]], baseline: float, min_n: int):
    results = []
    rules = []

    for t in [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]:
        rules.append((f"base + boost_score>={t}", lambda r, t=t: (fnum(r.get("hr_boost_score"), -999) or -999) >= t))

    checks = [
        "vs_rhp", "same_hand", "hot_80", "hot_85", "hot_90", "pitcher_hr9_1.0", "pitcher_hr9_1.2",
        "recent_iso_350", "h2h_slg_900", "wind_out", "hitterish_park", "not_pitcher_park", "quality_ok", "avoid_bad_flags"
    ]
    for k in checks:
        rules.append((f"base + {k}", lambda r, k=k: factor_flags(r).get(k, False)))

    combos = [
        ("base + vs_rhp + hot_80", ["vs_rhp", "hot_80"]),
        ("base + vs_rhp + same_hand", ["vs_rhp", "same_hand"]),
        ("base + vs_rhp + pitcher_hr9_1.0", ["vs_rhp", "pitcher_hr9_1.0"]),
        ("base + hot_80 + recent_iso_350", ["hot_80", "recent_iso_350"]),
        ("base + same_hand + recent_iso_350", ["same_hand", "recent_iso_350"]),
        ("base + pitcher_hr9_1.0 + hot_80", ["pitcher_hr9_1.0", "hot_80"]),
        ("base + vs_rhp + h2h_slg_900", ["vs_rhp", "h2h_slg_900"]),
        ("base + vs_rhp + hot_80 + pitcher_hr9_1.0", ["vs_rhp", "hot_80", "pitcher_hr9_1.0"]),
    ]
    for label, keys in combos:
        rules.append((label, lambda r, keys=keys: all(factor_flags(r).get(k, False) for k in keys)))

    for k in range(1, 7):
        rules.append((f"base + high_signal_factor_count>={k}", lambda r, k=k: (fnum(r.get("high_signal_factor_count"), 0) or 0) >= k))

    for label, fn in rules:
        filt = [r for r in rows if fn(r)]
        if len(filt) >= min_n:
            s = summarize(filt, baseline)
            results.append((s["rate"], s["lower"], s["pen"], s["n"], label, s))
    return results


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

    print("HR BOOST SCORE LAB")
    print("==================")
    print("Diagnostic only. Tests if context boosts can improve the 30% HR base bucket.")
    api_mod = importlib.import_module("api")
    print(f"api_version: {getattr(api_mod, 'VERSION', 'unknown')}")
    print(f"filters: days={args.days} date={args.date} min_n={args.min_n}")

    all_rows, meta = load_rows(api_mod, args.days, args.date, args.max_grade_rows)
    n = len(all_rows)
    hits = sum(1 for r in all_rows if r.get("result") == "hit")
    baseline = hits / n if n else 0.0
    print("\nINPUT / BASELINE")
    print("----------------")
    for k, v in meta.items():
        print(f"{k}: {v}")
    print(f"combined_unique_graded_hr_rows: {n}")
    print(f"hits: {hits}")
    print(f"misses: {n-hits}")
    print(f"baseline_hr_rate: {baseline:.1%}")

    base = [add_boost(r) for r in all_rows if base_rule(r, args.base_score, args.base_slg)]
    base_s = summarize(base, baseline)
    print("\nBASE 30 MODEL")
    print("-------------")
    print(fmt(f"hr_score>={args.base_score:.2f} AND season_slg>={args.base_slg:.3f}", base_s))

    print("\nBASE MEMBERS WITH BOOST TAGS")
    print("----------------------------")
    for r in sorted(base, key=lambda x: (x.get("result") != "hit", -(x.get("hr_boost_score") or 0))):
        print(
            f"{str(r.get('result')):4s} {str(r.get('player')):22s} boost={r.get('hr_boost_score'):.2f} "
            f"factors={r.get('high_signal_factor_count')} tags={r.get('hr_boost_tags')} "
            f"score={fnum(r.get('hr_score'))} slg={fnum(r.get('season_slg'))} iso={fnum(r.get('recent_iso'))} "
            f"temp={r.get('weather_temp_f')} park={r.get('park_bucket')} bat={r.get('batter_side')} "
            f"vs={r.get('opp_pitcher_hand')} platoon={r.get('platoon_bucket')} "
            f"pHR9={None if r.get('opp_pitcher_hr9') is None else round(r.get('opp_pitcher_hr9'),2)}"
        )

    results = build_results(base, baseline, args.min_n)
    print("\nBOOST RULES BY HIT RATE")
    print("-----------------------")
    for _, _, _, _, label, s in sorted(results, key=lambda x: (x[0], x[1], x[3]), reverse=True)[:args.top]:
        print(fmt(label, s))

    print("\nBOOST RULES BY PENALIZED SCORE")
    print("------------------------------")
    print("Uses Wilson lower bound * log(sample), so tiny buckets are penalized.")
    for _, _, pen, _, label, s in sorted(results, key=lambda x: (x[2], x[0], x[3]), reverse=True)[:args.top]:
        print(f"pen={pen:.4f}  {fmt(label, s)}")

    print("\nRECOMMENDED BOOST READ")
    print("----------------------")
    min_reco = max(args.min_n, 5)
    candidates = [x for x in results if x[5]["n"] >= min_reco]
    if candidates:
        best = max(candidates, key=lambda x: (x[2], x[0], x[3]))
        _, _, _, _, label, s = best
        print(f"Best rule with n >= {min_reco}: {label}")
        print(fmt(label, s))
        if s["rate"] > base_s["rate"]:
            print("READ: This improves the 30% base in-sample. Treat as HR BOOST WATCHLIST only until more rows/odds confirm it.")
        else:
            print("READ: No improvement over the 30% base with this sample.")
    else:
        print(f"No boost rule had n >= {min_reco}.")

    print("\nLIVE RULE CANDIDATE")
    print("-------------------")
    print("HR_CORE_LONGSHOT = hr_score>=1.70 AND season_slg>=.450")
    print("HR_BOOSTED_LONGSHOT = HR_CORE_LONGSHOT AND hr_boost_score>=2.0")
    print("Still watchlist only. Real FanDuel HR odds are required before official EV picks.")

    if args.write_csv:
        out = Path("hr_boost_score_lab_rows.csv")
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
