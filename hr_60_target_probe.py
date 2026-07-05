#!/usr/bin/env python3
"""
HR 60% Target / Threshold Optimizer for Prop Edge MLB API.

Purpose:
- Diagnostic only. Does NOT modify api.py.
- Uses the same historical HR rows as hr_calibration_probe.py.
- Searches threshold/filter combinations to answer:
  "Can any HR bucket realistically reach a target hit rate, such as 60%?"
- Prints the best high-precision HR gates, target-reaching gates, and overfit warnings.

Required: place this file beside api.py AND hr_calibration_probe.py on Render.

Run:
    python hr_60_target_probe.py --days 120 --min-n 3 --target 0.60

More conservative sample:
    python hr_60_target_probe.py --days 180 --min-n 10 --target 0.30
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import itertools
import json
import math
from collections import Counter
from typing import Any, Dict, List, Callable, Tuple, Optional

# Reuse the grading/loading code already uploaded in hr_calibration_probe.py.
try:
    import hr_calibration_probe as cal
except Exception as e:
    raise SystemExit(
        "ERROR: hr_calibration_probe.py must be in the same folder as this script. "
        f"Import failed: {e}"
    )


def fnum(x: Any, default: Optional[float] = None) -> Optional[float]:
    return cal.safe_float(x, default)


def has_text(row: Dict[str, Any], text: str) -> bool:
    hay = " ".join(str(row.get(k) or "") for k in ("bvp_flag", "hr_tier", "prediction_tier", "reject_reason"))
    return text.lower() in hay.lower()


def pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%"


def summarize_gate(rows: List[Dict[str, Any]], baseline: float) -> Dict[str, Any]:
    n = len(rows)
    hits = sum(1 for r in rows if r.get("result") == "hit")
    rate = hits / n if n else 0.0
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
        "lift": lift,
        "avg_hr_score": avg("hr_score"),
        "avg_model_prob": avg("model_prob"),
        "avg_recent_iso": avg("recent_iso"),
        "avg_season_slg": avg("season_slg"),
        "avg_h2h_slg": avg("h2h_slg"),
    }


def gate_label(parts: List[str]) -> str:
    parts = [p for p in parts if p]
    return " AND ".join(parts) if parts else "ALL_HR_CANDIDATES"


def run_gate(rows: List[Dict[str, Any]], funcs: List[Callable[[Dict[str, Any]], bool]]) -> List[Dict[str, Any]]:
    if not funcs:
        return list(rows)
    out = []
    for r in rows:
        try:
            if all(fn(r) for fn in funcs):
                out.append(r)
        except Exception:
            continue
    return out


def load_all_rows(api_mod: Any, days: int, exact_date: Optional[str], max_grade_rows: int) -> Tuple[List[Dict[str, Any]], Counter]:
    candidate_rows = cal.load_candidate_rows(api_mod, days, exact_date)
    record_rows = cal.load_record_hr_rows(api_mod, days, exact_date)
    graded_candidates, reasons = cal.grade_candidates(api_mod, candidate_rows, max_rows=max_grade_rows)

    all_rows = list(graded_candidates)
    seen = {
        (r.get("date"), str(r.get("game_id")), str(r.get("player_id") or ""), cal.norm(r.get("player")), r.get("pick"))
        for r in all_rows
    }
    for r in record_rows:
        if r.get("result") not in {"hit", "miss"}:
            continue
        key = (r.get("date"), str(r.get("game_id")), str(r.get("player_id") or ""), cal.norm(r.get("player")), r.get("pick"))
        if key in seen:
            continue
        rr = dict(r)
        rr["quality_ok"] = cal.quality_ok(rr)
        rr["hr_score_bucket"] = cal.score_bucket(fnum(rr.get("hr_score")))
        rr["tier_bucket"] = cal.hr_tier(rr)
        rr["lineup_spot_bucket"] = cal.lineup_bucket(rr.get("lineup_spot"))
        rr["model_prob_bucket"] = cal.prob_bucket(fnum(rr.get("model_prob")))
        rr["mc_prob_bucket"] = cal.prob_bucket(fnum(rr.get("hitter_mc_prob")))
        all_rows.append(rr)
        seen.add(key)

    meta = Counter()
    meta["candidate_log_rows_loaded"] = len(candidate_rows)
    meta["record_rows_loaded"] = len(record_rows)
    meta["candidate_rows_graded"] = len(graded_candidates)
    for k, v in reasons.items():
        meta[f"skip_{k}"] = v
    return all_rows, meta


def build_gate_space(rows: List[Dict[str, Any]]) -> List[Tuple[str, List[Callable[[Dict[str, Any]], bool]]]]:
    """Generate threshold combos. This intentionally includes simple and compound gates."""
    gates: List[Tuple[str, List[Callable[[Dict[str, Any]], bool]]]] = []

    # Single filters.
    score_mins = [1.30, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00]
    season_slg_mins = [0.350, 0.400, 0.450, 0.500, 0.550]
    recent_iso_mins = [0.150, 0.200, 0.250, 0.300, 0.350, 0.400]
    h2h_slg_mins = [0.500, 0.700, 0.900, 1.000]
    model_prob_mins = [0.35, 0.40, 0.42, 0.44, 0.46, 0.50]

    bool_filters: List[Tuple[str, Callable[[Dict[str, Any]], bool]]] = [
        ("quality_ok", lambda r: bool(r.get("quality_ok"))),
        ("tier=hr_elite", lambda r: (r.get("tier_bucket") or cal.hr_tier(r)) == "hr_elite"),
        ("tier in {hr_elite,hr_strong}", lambda r: (r.get("tier_bucket") or cal.hr_tier(r)) in {"hr_elite", "hr_strong"}),
        ("lineup_spot<=3", lambda r: (fnum(r.get("lineup_spot"), 99) or 99) <= 3),
        ("lineup_spot<=5", lambda r: (fnum(r.get("lineup_spot"), 99) or 99) <= 5),
        ("not_sum_avoid", lambda r: not has_text(r, "sum_avoid")),
        ("not_struggles", lambda r: not has_text(r, "struggles")),
        ("not_sum_lean", lambda r: not has_text(r, "sum_lean")),
        ("odds_present", lambda r: r.get("odds") is not None and str(r.get("odds")).upper() not in {"", "N/A", "NA", "NONE"}),
    ]

    for s in score_mins:
        gates.append((f"hr_score>={s:.2f}", [lambda r, s=s: (fnum(r.get("hr_score"), -999) or -999) >= s]))
    for s in season_slg_mins:
        gates.append((f"season_slg>={s:.3f}", [lambda r, s=s: (fnum(r.get("season_slg"), -999) or -999) >= s]))
    for s in recent_iso_mins:
        gates.append((f"recent_iso>={s:.3f}", [lambda r, s=s: (fnum(r.get("recent_iso"), -999) or -999) >= s]))
    for s in h2h_slg_mins:
        gates.append((f"h2h_slg>={s:.3f}", [lambda r, s=s: (fnum(r.get("h2h_slg"), -999) or -999) >= s]))
    for s in model_prob_mins:
        gates.append((f"model_prob>={s:.2f}", [lambda r, s=s: (fnum(r.get("model_prob"), -999) or -999) >= s]))
    for label, fn in bool_filters:
        gates.append((label, [fn]))

    # Compound filters. Keep bounded to avoid nonsense overfitting explosion.
    primary: List[Tuple[str, Callable[[Dict[str, Any]], bool]]] = []
    for s in score_mins:
        primary.append((f"hr_score>={s:.2f}", lambda r, s=s: (fnum(r.get("hr_score"), -999) or -999) >= s))
    primary.extend(bool_filters[:5])  # quality/tier/lineup only.

    secondary: List[Tuple[str, Callable[[Dict[str, Any]], bool]]] = []
    for s in season_slg_mins:
        secondary.append((f"season_slg>={s:.3f}", lambda r, s=s: (fnum(r.get("season_slg"), -999) or -999) >= s))
    for s in recent_iso_mins:
        secondary.append((f"recent_iso>={s:.3f}", lambda r, s=s: (fnum(r.get("recent_iso"), -999) or -999) >= s))
    for s in model_prob_mins:
        secondary.append((f"model_prob>={s:.2f}", lambda r, s=s: (fnum(r.get("model_prob"), -999) or -999) >= s))
    secondary.extend(bool_filters[5:8])

    for (la, fa), (lb, fb) in itertools.product(primary, secondary):
        if la == lb:
            continue
        gates.append((gate_label([la, lb]), [fa, fb]))

    # Three-part gates aimed at realistic HR promotion criteria.
    three_part_specs = [
        ("tier=hr_elite", lambda r: (r.get("tier_bucket") or cal.hr_tier(r)) == "hr_elite"),
        ("quality_ok", lambda r: bool(r.get("quality_ok"))),
        ("not_sum_avoid", lambda r: not has_text(r, "sum_avoid")),
        ("not_struggles", lambda r: not has_text(r, "struggles")),
        ("lineup_spot<=5", lambda r: (fnum(r.get("lineup_spot"), 99) or 99) <= 5),
    ]
    for score_min in [1.50, 1.70, 1.90]:
        sf = (f"hr_score>={score_min:.2f}", lambda r, s=score_min: (fnum(r.get("hr_score"), -999) or -999) >= s)
        for a, b in itertools.combinations(three_part_specs, 2):
            gates.append((gate_label([sf[0], a[0], b[0]]), [sf[1], a[1], b[1]]))

    # Dedupe by label.
    seen = set()
    out = []
    for label, funcs in gates:
        if label in seen:
            continue
        seen.add(label)
        out.append((label, funcs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--date", default=None)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--target", type=float, default=0.60, help="Target hit rate as decimal, e.g. 0.60")
    ap.add_argument("--max-grade-rows", type=int, default=0)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    print("HR 60% TARGET / THRESHOLD OPTIMIZER")
    print("====================================")
    print("No api.py changes. Diagnostic only.")

    api_mod = importlib.import_module("api")
    print(f"api_version: {getattr(api_mod, 'VERSION', 'unknown')}")
    print(f"data_dir: {getattr(api_mod, 'DATA_DIR', '/data')}")
    print(f"filters: days={args.days} date={args.date} min_n={args.min_n} target={args.target:.1%}")

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
        print("\nNO DATA: no graded HR rows available yet.")
        return 0

    needed_for_target = math.ceil(args.target * n)
    print("\nTARGET REALITY CHECK")
    print("--------------------")
    print(f"Current sample would need {needed_for_target}/{n} HRs to show {args.target:.1%} overall.")
    print(f"Current actual: {hits}/{n} = {baseline:.1%}")
    if args.target >= 0.50:
        print("NOTE: A 50-60% true HR hit rate is not a realistic full-slate HR goal. Treat this test as a precision-bucket search, not a normal target.")

    candidates = []
    for label, funcs in build_gate_space(rows):
        filt = run_gate(rows, funcs)
        if len(filt) < args.min_n:
            continue
        s = summarize_gate(filt, baseline)
        candidates.append((s["rate"], s["n"], s["hits"], label, s))

    print("\nFILTERS THAT REACH TARGET")
    print("-------------------------")
    target_hits = [c for c in candidates if c[0] >= args.target]
    if not target_hits:
        print(f"None with n >= {args.min_n} reached {args.target:.1%}.")
    else:
        for rate, n2, h2, label, s in sorted(target_hits, key=lambda x: (x[0], x[1]), reverse=True)[:args.top]:
            lift = f"{s['lift']:.2f}x" if s["lift"] is not None else "n/a"
            print(f"{rate:6.1%}  n={n2:3d} hit={h2:3d} lift={lift:>6s}  {label}")

    print("\nBEST FILTERS BY HIT RATE")
    print("------------------------")
    for rate, n2, h2, label, s in sorted(candidates, key=lambda x: (x[0], x[1]), reverse=True)[:args.top]:
        lift = f"{s['lift']:.2f}x" if s["lift"] is not None else "n/a"
        print(
            f"{rate:6.1%}  n={n2:3d} hit={h2:3d} lift={lift:>6s}  "
            f"avg_score={s['avg_hr_score'] and round(s['avg_hr_score'],3)} "
            f"avg_slg={s['avg_season_slg'] and round(s['avg_season_slg'],3)} "
            f"avg_iso={s['avg_recent_iso'] and round(s['avg_recent_iso'],3)}  {label}"
        )

    # More conservative display: require at least 10 or 20 samples when possible.
    for conservative_min in (10, 20, 30):
        big = [c for c in candidates if c[1] >= conservative_min]
        print(f"\nBEST FILTERS WITH n >= {conservative_min}")
        print("-" * (24 + len(str(conservative_min))))
        if not big:
            print(f"No filters with n >= {conservative_min}.")
            continue
        for rate, n2, h2, label, s in sorted(big, key=lambda x: (x[0], x[1]), reverse=True)[:10]:
            lift = f"{s['lift']:.2f}x" if s["lift"] is not None else "n/a"
            print(f"{rate:6.1%}  n={n2:3d} hit={h2:3d} lift={lift:>6s}  {label}")

    print("\nPROMOTION READ")
    print("--------------")
    best = max(candidates, key=lambda x: (x[0], x[1])) if candidates else None
    if not best:
        print("NOT_READY: no filter has enough rows.")
    else:
        rate, n2, h2, label, s = best
        print(f"Best observed gate: {label}")
        print(f"Best observed rate: {h2}/{n2} = {rate:.1%}")
        if rate >= args.target and n2 >= max(args.min_n, 10):
            print("TARGET_REACHED_BUT_VERIFY: target reached with enough minimal rows; still needs out-of-sample validation before official HR.")
        elif rate >= args.target:
            print("TARGET_REACHED_TOO_SMALL: target reached only in a tiny bucket; likely overfit until more games accumulate.")
        elif rate >= 0.25:
            print("NO_60_BUCKET: no realistic 60% bucket yet, but there may be a useful longshot/EV watchlist bucket.")
        else:
            print("KEEP_EXPERIMENTAL: no high-precision HR bucket found yet.")

    print("\nNEXT TEST IF SAMPLE IS TOO SMALL")
    print("--------------------------------")
    print("Let more HR candidate logs accumulate, then rerun with: python hr_60_target_probe.py --days 180 --min-n 10 --target 0.30")
    print("For HR, the practical promotion target is usually calibrated EV with real odds, not 60% raw hit rate.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
