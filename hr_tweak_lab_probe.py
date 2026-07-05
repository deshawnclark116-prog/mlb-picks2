#!/usr/bin/env python3
"""
HR Tweak Lab Probe for Prop Edge MLB API.

Purpose:
- Diagnostic only. Does NOT modify api.py.
- Starts from the best current HR bucket:
      hr_score >= 1.70 AND season_slg >= .450
- Tests "our tweaks" on top of that bucket:
      quality_ok, recent ISO, H2H SLG, lineup spot, avoid flags, model score, etc.
- Also searches challenger rules that may beat the current bucket.
- Uses Wilson lower bound / penalized score so tiny overfit buckets do not fool us.

Required:
- Put this file beside api.py
- Put hr_calibration_probe.py beside it too

Run:
    python hr_tweak_lab_probe.py --days 180 --min-n 3

More conservative:
    python hr_tweak_lab_probe.py --days 180 --min-n 10

Optional CSV:
    python hr_tweak_lab_probe.py --days 180 --min-n 3 --write-csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Callable, Tuple, Optional

try:
    import hr_calibration_probe as cal
except Exception as e:
    raise SystemExit(
        "ERROR: hr_calibration_probe.py must be in the same folder as this script. "
        f"Import failed: {e}"
    )


def fnum(x: Any, default: Optional[float] = None) -> Optional[float]:
    return cal.safe_float(x, default)


def norm_text(x: Any) -> str:
    return str(x or "").lower()


def has_text(row: Dict[str, Any], text: str) -> bool:
    hay = " ".join(str(row.get(k) or "") for k in (
        "bvp_flag", "hr_tier", "prediction_tier", "reject_reason", "board_warnings"
    ))
    return text.lower() in hay.lower()


def pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%"


def wilson_lower(hits: int, n: int, z: float = 1.28155) -> float:
    """
    One-sided-ish conservative lower bound.
    z=1.28155 approximates 80% confidence lower bound.
    z=1.96 would be stricter but too harsh for tiny current samples.
    """
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
        "penalized_score": lower * math.log1p(n),
        "avg_hr_score": avg("hr_score"),
        "avg_model_prob": avg("model_prob"),
        "avg_recent_iso": avg("recent_iso"),
        "avg_season_slg": avg("season_slg"),
        "avg_h2h_slg": avg("h2h_slg"),
        "avg_lineup_spot": avg("lineup_spot"),
    }


def label_summary(label: str, s: Dict[str, Any]) -> str:
    lift = f"{s['lift']:.2f}x" if s["lift"] is not None else "n/a"
    return (
        f"{s['rate']:6.1%}  n={s['n']:3d} hit={s['hits']:3d} "
        f"lb80={s['wilson80_lower']:5.1%} lift={lift:>6s} "
        f"avg_score={round(s['avg_hr_score'],3) if s['avg_hr_score'] is not None else None} "
        f"avg_slg={round(s['avg_season_slg'],3) if s['avg_season_slg'] is not None else None} "
        f"avg_iso={round(s['avg_recent_iso'],3) if s['avg_recent_iso'] is not None else None} "
        f"avg_spot={round(s['avg_lineup_spot'],2) if s['avg_lineup_spot'] is not None else None}  "
        f"{label}"
    )


def gate(rows: List[Dict[str, Any]], funcs: List[Callable[[Dict[str, Any]], bool]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        try:
            if all(fn(r) for fn in funcs):
                out.append(r)
        except Exception:
            pass
    return out


def load_all_rows(api_mod: Any, days: int, exact_date: Optional[str], max_grade_rows: int) -> Tuple[List[Dict[str, Any]], Counter]:
    candidate_rows = cal.load_candidate_rows(api_mod, days, exact_date)
    record_rows = cal.load_record_hr_rows(api_mod, days, exact_date)
    graded_candidates, reasons = cal.grade_candidates(api_mod, candidate_rows, max_rows=max_grade_rows)

    all_rows = []
    seen = set()

    for r in graded_candidates:
        rr = dict(r)
        rr["quality_ok"] = cal.quality_ok(rr)
        rr["tier_bucket"] = cal.hr_tier(rr)
        key = (
            rr.get("date"),
            str(rr.get("game_id")),
            str(rr.get("player_id") or ""),
            cal.norm(rr.get("player")),
            rr.get("pick"),
        )
        all_rows.append(rr)
        seen.add(key)

    for r in record_rows:
        if r.get("result") not in {"hit", "miss"}:
            continue
        key = (
            r.get("date"),
            str(r.get("game_id")),
            str(r.get("player_id") or ""),
            cal.norm(r.get("player")),
            r.get("pick"),
        )
        if key in seen:
            continue
        rr = dict(r)
        rr["quality_ok"] = cal.quality_ok(rr)
        rr["tier_bucket"] = cal.hr_tier(rr)
        all_rows.append(rr)
        seen.add(key)

    meta = Counter()
    meta["candidate_log_rows_loaded"] = len(candidate_rows)
    meta["record_rows_loaded"] = len(record_rows)
    meta["candidate_rows_graded"] = len(graded_candidates)
    for k, v in reasons.items():
        meta[f"skip_{k}"] = v
    return all_rows, meta


def make_basic_tweaks() -> List[Tuple[str, Callable[[Dict[str, Any]], bool]]]:
    tweaks: List[Tuple[str, Callable[[Dict[str, Any]], bool]]] = []

    tweaks.extend([
        ("quality_ok", lambda r: bool(r.get("quality_ok"))),
        ("tier=hr_elite", lambda r: (r.get("tier_bucket") or cal.hr_tier(r)) == "hr_elite"),
        ("lineup_spot<=3", lambda r: (fnum(r.get("lineup_spot"), 99) or 99) <= 3),
        ("lineup_spot<=5", lambda r: (fnum(r.get("lineup_spot"), 99) or 99) <= 5),
        ("not_sum_avoid", lambda r: not has_text(r, "sum_avoid")),
        ("not_struggles", lambda r: not has_text(r, "struggles")),
        ("not_sum_lean", lambda r: not has_text(r, "sum_lean")),
    ])

    for x in [0.150, 0.200, 0.250, 0.300, 0.350, 0.400]:
        tweaks.append((f"recent_iso>={x:.3f}", lambda r, x=x: (fnum(r.get("recent_iso"), -999) or -999) >= x))

    for x in [0.400, 0.450, 0.500, 0.550, 0.600]:
        tweaks.append((f"season_slg>={x:.3f}", lambda r, x=x: (fnum(r.get("season_slg"), -999) or -999) >= x))

    for x in [0.500, 0.700, 0.900, 1.000]:
        tweaks.append((f"h2h_slg>={x:.3f}", lambda r, x=x: (fnum(r.get("h2h_slg"), -999) or -999) >= x))

    for x in [0.35, 0.40, 0.42, 0.44, 0.46, 0.50]:
        tweaks.append((f"model_prob>={x:.2f}", lambda r, x=x: (fnum(r.get("model_prob"), -999) or -999) >= x))

    for x in [1.50, 1.60, 1.70, 1.80, 1.90, 2.00]:
        tweaks.append((f"hr_score>={x:.2f}", lambda r, x=x: (fnum(r.get("hr_score"), -999) or -999) >= x))

    return tweaks


def unique_rules(rules: List[Tuple[str, List[Callable[[Dict[str, Any]], bool]]]]) -> List[Tuple[str, List[Callable[[Dict[str, Any]], bool]]]]:
    seen = set()
    out = []
    for label, funcs in rules:
        if label in seen:
            continue
        seen.add(label)
        out.append((label, funcs))
    return out


def build_challenger_rules() -> List[Tuple[str, List[Callable[[Dict[str, Any]], bool]]]]:
    tweaks = make_basic_tweaks()
    rules: List[Tuple[str, List[Callable[[Dict[str, Any]], bool]]]] = []

    # Current base and close variants.
    for score in [1.50, 1.60, 1.70, 1.80, 1.90, 2.00]:
        for slg in [0.400, 0.450, 0.500, 0.550]:
            rules.append((
                f"hr_score>={score:.2f} AND season_slg>={slg:.3f}",
                [
                    lambda r, score=score: (fnum(r.get("hr_score"), -999) or -999) >= score,
                    lambda r, slg=slg: (fnum(r.get("season_slg"), -999) or -999) >= slg,
                ],
            ))

    # Base + one tweak, base + two tweaks.
    base_funcs = [
        lambda r: (fnum(r.get("hr_score"), -999) or -999) >= 1.70,
        lambda r: (fnum(r.get("season_slg"), -999) or -999) >= 0.450,
    ]
    base_label = "BASE: hr_score>=1.70 AND season_slg>=.450"

    for label, fn in tweaks:
        if label in {"hr_score>=1.70", "season_slg>=0.450"}:
            continue
        rules.append((f"{base_label} AND {label}", base_funcs + [fn]))

    # Limited two-tweak combos on base.
    preferred = [
        item for item in tweaks
        if item[0] in {
            "quality_ok", "tier=hr_elite", "lineup_spot<=3", "lineup_spot<=5",
            "not_sum_avoid", "not_struggles", "recent_iso>=0.250",
            "recent_iso>=0.300", "recent_iso>=0.350", "model_prob>=0.42", "model_prob>=0.44"
        }
    ]
    for (la, fa), (lb, fb) in itertools.combinations(preferred, 2):
        rules.append((f"{base_label} AND {la} AND {lb}", base_funcs + [fa, fb]))

    # Non-base alternatives: quality/tier + power thresholds.
    for score in [1.60, 1.70, 1.80]:
        for iso in [0.200, 0.250, 0.300, 0.350]:
            rules.append((
                f"hr_score>={score:.2f} AND recent_iso>={iso:.3f}",
                [
                    lambda r, score=score: (fnum(r.get("hr_score"), -999) or -999) >= score,
                    lambda r, iso=iso: (fnum(r.get("recent_iso"), -999) or -999) >= iso,
                ],
            ))
            rules.append((
                f"quality_ok AND hr_score>={score:.2f} AND recent_iso>={iso:.3f}",
                [
                    lambda r: bool(r.get("quality_ok")),
                    lambda r, score=score: (fnum(r.get("hr_score"), -999) or -999) >= score,
                    lambda r, iso=iso: (fnum(r.get("recent_iso"), -999) or -999) >= iso,
                ],
            ))

    return unique_rules(rules)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--date", default=None)
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--base-score", type=float, default=1.70)
    ap.add_argument("--base-slg", type=float, default=0.450)
    ap.add_argument("--max-grade-rows", type=int, default=0)
    ap.add_argument("--write-csv", action="store_true")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    print("HR TWEAK LAB / MODEL IMPROVEMENT PROBE")
    print("=======================================")
    print("No api.py changes. Diagnostic only.")

    api_mod = importlib.import_module("api")
    print(f"api_version: {getattr(api_mod, 'VERSION', 'unknown')}")
    print(f"data_dir: {getattr(api_mod, 'DATA_DIR', '/data')}")
    print(f"filters: days={args.days} date={args.date} min_n={args.min_n}")

    rows, meta = load_all_rows(api_mod, args.days, args.date, args.max_grade_rows)
    n = len(rows)
    hits = sum(1 for r in rows if r.get("result") == "hit")
    baseline = hits / n if n else 0.0

    print("\nINPUT / BASELINE")
    print("----------------")
    for k, v in meta.items():
        print(f"{k}: {v}")
    print(f"combined_unique_graded_hr_rows: {n}")
    print(f"hits: {hits}")
    print(f"misses: {n - hits}")
    print(f"baseline_hr_rate: {baseline:.1%}")

    if n == 0:
        print("NO DATA: no graded HR rows yet.")
        return 0

    base_funcs = [
        lambda r: (fnum(r.get("hr_score"), -999) or -999) >= args.base_score,
        lambda r: (fnum(r.get("season_slg"), -999) or -999) >= args.base_slg,
    ]
    base_label = f"BASE_30_MODEL: hr_score>={args.base_score:.2f} AND season_slg>={args.base_slg:.3f}"
    base_rows = gate(rows, base_funcs)
    base_s = summarize(base_rows, baseline)

    print("\nBASE 30 MODEL")
    print("-------------")
    print(label_summary(base_label, base_s))
    print("Base members:")
    for r in base_rows:
        print(
            f"  {r.get('result','?'):4s}  {r.get('player')}  "
            f"score={fnum(r.get('hr_score'))} slg={fnum(r.get('season_slg'))} "
            f"iso={fnum(r.get('recent_iso'))} spot={r.get('lineup_spot')} "
            f"tier={r.get('tier_bucket') or cal.hr_tier(r)} q={bool(r.get('quality_ok'))} "
            f"flag={r.get('bvp_flag')}"
        )

    print("\nTWEAKS ON TOP OF BASE")
    print("---------------------")
    tweak_results = []
    for label, fn in make_basic_tweaks():
        if label in {f"hr_score>={args.base_score:.2f}", f"season_slg>={args.base_slg:.3f}"}:
            continue
        filt = gate(base_rows, [fn])
        if len(filt) < args.min_n:
            continue
        s = summarize(filt, baseline)
        delta = s["rate"] - base_s["rate"]
        tweak_results.append((s["rate"], s["wilson80_lower"], s["n"], delta, label, s))

    for rate, lower, n2, delta, label, s in sorted(tweak_results, key=lambda x: (x[0], x[1], x[2]), reverse=True)[:args.top]:
        print(f"{label_summary('BASE + ' + label, s)}  delta_vs_base={delta:+.1%}")

    print("\nFULL SEARCH CHALLENGERS")
    print("-----------------------")
    challenge_results = []
    for label, funcs in build_challenger_rules():
        filt = gate(rows, funcs)
        if len(filt) < args.min_n:
            continue
        s = summarize(filt, baseline)
        challenge_results.append((s["rate"], s["wilson80_lower"], s["penalized_score"], s["n"], label, s))

    for rate, lower, pen, n2, label, s in sorted(challenge_results, key=lambda x: (x[0], x[1], x[3]), reverse=True)[:args.top]:
        print(label_summary(label, s))

    print("\nPENALIZED RANKING")
    print("-----------------")
    print("Uses Wilson lower bound * log(sample). This favors rules that are strong but not absurdly tiny.")
    for rate, lower, pen, n2, label, s in sorted(challenge_results, key=lambda x: (x[2], x[0], x[3]), reverse=True)[:args.top]:
        print(f"pen={pen:.4f}  {label_summary(label, s)}")

    print("\nRECOMMENDED HR WATCHLIST RULE")
    print("-----------------------------")
    conservative = [x for x in challenge_results if x[5]["n"] >= max(args.min_n, 10)]
    if conservative:
        best_pen = max(conservative, key=lambda x: (x[2], x[0], x[3]))
        rate, lower, pen, n2, label, s = best_pen
        print(f"Best conservative rule: {label}")
        print(label_summary(label, s))
        if s["rate"] >= 0.30 and s["n"] >= 10:
            print("READ: This is a viable HR WATCHLIST candidate rule, not official yet.")
        else:
            print("READ: Still experimental; keep logging before using live.")
    else:
        print("No conservative rule has enough rows yet.")

    print("\nWHAT TO DO NEXT")
    print("---------------")
    print("1. Do not make individual HR official from this sample yet.")
    print("2. Treat the best rule as HR Watchlist / Longshot Signal.")
    print("3. Add real FanDuel HR odds next, because 25-35% HR probability only matters if price is +EV.")
    print("4. Keep every HR candidate/reject logged so this probe grows beyond 100+ graded rows.")

    if args.write_csv:
        out = Path("hr_tweak_lab_results.csv")
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "label", "n", "hits", "misses", "rate", "wilson80_lower", "lift",
                "avg_hr_score", "avg_season_slg", "avg_recent_iso", "avg_h2h_slg", "avg_model_prob"
            ])
            for rate, lower, pen, n2, label, s in sorted(challenge_results, key=lambda x: (x[2], x[0], x[3]), reverse=True):
                writer.writerow([
                    label, s["n"], s["hits"], s["misses"], round(s["rate"], 6),
                    round(s["wilson80_lower"], 6),
                    round(s["lift"], 6) if s["lift"] is not None else "",
                    s["avg_hr_score"], s["avg_season_slg"], s["avg_recent_iso"],
                    s["avg_h2h_slg"], s["avg_model_prob"],
                ])
        print(f"\nCSV_WRITTEN: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
