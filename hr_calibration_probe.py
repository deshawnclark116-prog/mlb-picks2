#!/usr/bin/env python3
"""
HR scorer / calibration probe for Prop Edge MLB API.

Purpose:
- Does NOT modify api.py.
- Reads saved candidate logs and record.json from Render persistent disk.
- Grades historical batter_home_runs candidates using MLB game logs.
- Measures whether hr_score / hr_tier has real lift above baseline.
- Checks whether current HR model probability is calibrated or only a heuristic.

Run from the Render Shell in the same directory as api.py:
    python hr_calibration_probe.py --days 60 --min-n 5

Useful options:
    python hr_calibration_probe.py --days 120 --min-n 10 --write-csv
    python hr_calibration_probe.py --date 2026-07-04 --min-n 3
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import json
import math
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, str):
            x = x.strip()
            if not x or x.upper() in {"N/A", "NA", "NONE", "NULL"}:
                return default
        return float(x)
    except Exception:
        return default


def norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def parse_date_from_name(path: Path) -> Optional[str]:
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
    return m.group(1) if m else None


def load_json(path: Path, fallback: Any = None) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        print(f"WARN: could not read {path}: {e}")
    return fallback


def flatten_candidate_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("candidates", "rows", "items", "predictions", "results"):
            val = payload.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def score_bucket(score: Optional[float]) -> str:
    if score is None:
        return "no_score"
    if score < 1.30:
        return "lt_1.30"
    if score < 1.50:
        return "1.30_1.49"
    if score < 1.70:
        return "1.50_1.69"
    if score < 1.90:
        return "1.70_1.89"
    return "1.90_plus"


def prob_bucket(prob: Optional[float]) -> str:
    if prob is None:
        return "no_prob"
    if prob < 0.10:
        return "lt_10pct"
    if prob < 0.20:
        return "10_19pct"
    if prob < 0.30:
        return "20_29pct"
    if prob < 0.40:
        return "30_39pct"
    if prob < 0.45:
        return "40_44pct"
    if prob < 0.50:
        return "45_49pct"
    return "50pct_plus"


def lineup_bucket(spot: Any) -> str:
    try:
        s = int(spot)
    except Exception:
        return "no_spot"
    if s <= 0:
        return "bad_spot"
    if s <= 3:
        return "spot_1_3"
    if s <= 6:
        return "spot_4_6"
    if s <= 9:
        return "spot_7_9"
    return "spot_10_plus"


def quality_ok(row: Dict[str, Any]) -> bool:
    score = safe_float(row.get("hr_score"), 0.0) or 0.0
    season_slg = safe_float(row.get("season_slg"), 0.0) or 0.0
    recent_iso = safe_float(row.get("recent_iso"), 0.0) or 0.0
    return score >= 1.70 and (season_slg >= 0.400 or recent_iso >= 0.350)


def hr_tier(row: Dict[str, Any]) -> str:
    tier = row.get("hr_tier")
    if tier:
        return str(tier)
    flag = str(row.get("bvp_flag") or "")
    for t in ("hr_elite", "hr_strong", "hr_lean"):
        if t in flag:
            return t
    return "no_tier"


def brier(prob: Optional[float], hit: int) -> Optional[float]:
    if prob is None:
        return None
    return (prob - hit) ** 2


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if not n:
        return {
            "n": 0,
            "hits": 0,
            "misses": 0,
            "hr_rate_pct": 0.0,
            "avg_hr_score": None,
            "avg_model_prob_pct": None,
            "avg_mc_prob_pct": None,
            "brier_model": None,
            "brier_mc": None,
        }

    hits = sum(1 for r in rows if r.get("result") == "hit")
    misses = sum(1 for r in rows if r.get("result") == "miss")

    def avg(vals: Iterable[Optional[float]]) -> Optional[float]:
        xs = [v for v in vals if v is not None]
        return sum(xs) / len(xs) if xs else None

    avg_score = avg(safe_float(r.get("hr_score")) for r in rows)
    avg_model = avg(safe_float(r.get("model_prob")) for r in rows)
    avg_mc = avg(safe_float(r.get("hitter_mc_prob")) for r in rows)
    model_brier = avg(brier(safe_float(r.get("model_prob")), 1 if r.get("result") == "hit" else 0) for r in rows)
    mc_brier = avg(brier(safe_float(r.get("hitter_mc_prob")), 1 if r.get("result") == "hit" else 0) for r in rows)

    return {
        "n": n,
        "hits": hits,
        "misses": misses,
        "hr_rate_pct": round(hits / n * 100, 1) if n else 0.0,
        "avg_hr_score": round(avg_score, 3) if avg_score is not None else None,
        "avg_model_prob_pct": round(avg_model * 100, 1) if avg_model is not None else None,
        "avg_mc_prob_pct": round(avg_mc * 100, 1) if avg_mc is not None else None,
        "brier_model": round(model_brier, 4) if model_brier is not None else None,
        "brier_mc": round(mc_brier, 4) if mc_brier is not None else None,
    }


def print_group(title: str, rows: List[Dict[str, Any]], key_func, min_n: int = 1) -> None:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        try:
            k = str(key_func(r))
        except Exception:
            k = "unknown"
        groups[k].append(r)

    print(f"\n{title}")
    print("-" * len(title))
    items = []
    for k, rs in groups.items():
        if len(rs) >= min_n:
            s = summarize(rs)
            items.append((s["hr_rate_pct"], s["n"], k, s))
    if not items:
        print(f"No groups with n >= {min_n}")
        return
    for _, _, k, s in sorted(items, reverse=True):
        print(
            f"{k:24s} n={s['n']:3d} hit={s['hits']:3d} miss={s['misses']:3d} "
            f"rate={s['hr_rate_pct']:5.1f}% avg_score={s['avg_hr_score']} "
            f"avg_model={s['avg_model_prob_pct']}% avg_mc={s['avg_mc_prob_pct']}%"
        )


def load_candidate_rows(api_mod: Any, days: Optional[int], exact_date: Optional[str]) -> List[Dict[str, Any]]:
    data_dir = Path(getattr(api_mod, "DATA_DIR", "/data"))
    log_dir = data_dir / "candidate_logs"
    paths: List[Path] = []
    for pattern in ("hitter_candidates_*.json", "all_candidates_*.json"):
        paths.extend(sorted(log_dir.glob(pattern)))

    today = getattr(api_mod, "today_et", lambda: dt.date.today())()
    if isinstance(today, str):
        today_date = dt.date.fromisoformat(today)
    else:
        today_date = today
    cutoff = today_date - dt.timedelta(days=int(days)) if days else None

    raw_rows: List[Dict[str, Any]] = []
    for p in paths:
        d = parse_date_from_name(p)
        if not d:
            continue
        if exact_date and d != exact_date:
            continue
        try:
            dd = dt.date.fromisoformat(d)
        except Exception:
            continue
        if cutoff and dd < cutoff:
            continue

        payload = load_json(p, [])
        for row in flatten_candidate_payload(payload):
            if row.get("prop_type") != "batter_home_runs":
                continue
            r = dict(row)
            r["date"] = r.get("date") or d
            r["source_file"] = str(p)
            raw_rows.append(r)

    # Dedupe because all_candidates and hitter_candidates may contain same rows.
    seen = set()
    out = []
    for r in raw_rows:
        key = (
            r.get("date"),
            str(r.get("game_id")),
            str(r.get("player_id") or ""),
            norm(r.get("player")),
            r.get("prop_type"),
            r.get("pick"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def load_record_hr_rows(api_mod: Any, days: Optional[int], exact_date: Optional[str]) -> List[Dict[str, Any]]:
    data_dir = Path(getattr(api_mod, "DATA_DIR", "/data"))
    doc = load_json(data_dir / "record.json", {})
    results = doc.get("results", []) if isinstance(doc, dict) else []
    if not isinstance(results, list):
        return []

    today = getattr(api_mod, "today_et", lambda: dt.date.today())()
    if isinstance(today, str):
        today_date = dt.date.fromisoformat(today)
    else:
        today_date = today
    cutoff = today_date - dt.timedelta(days=int(days)) if days else None

    out = []
    for r0 in results:
        if not isinstance(r0, dict) or r0.get("prop_type") != "batter_home_runs":
            continue
        d = r0.get("date")
        if exact_date and d != exact_date:
            continue
        try:
            dd = dt.date.fromisoformat(d)
        except Exception:
            continue
        if cutoff and dd < cutoff:
            continue
        r = dict(r0)
        r["record_source"] = "record_json"
        out.append(r)
    return out


def grade_candidates(api_mod: Any, rows: List[Dict[str, Any]], max_rows: int = 0) -> Tuple[List[Dict[str, Any]], Counter]:
    today = getattr(api_mod, "today_et", lambda: dt.date.today())()
    if isinstance(today, str):
        today_date = dt.date.fromisoformat(today)
    else:
        today_date = today

    get_actual_stat = getattr(api_mod, "get_actual_stat")
    cache: Dict[Tuple[str, str], Optional[float]] = {}
    graded: List[Dict[str, Any]] = []
    reasons = Counter()

    rows = sorted(rows, key=lambda r: str(r.get("date") or ""), reverse=True)
    if max_rows and max_rows > 0:
        rows = rows[:max_rows]

    for r0 in rows:
        r = dict(r0)
        d = r.get("date")
        if not d:
            reasons["missing_date"] += 1
            continue
        try:
            dd = dt.date.fromisoformat(d)
        except Exception:
            reasons["bad_date"] += 1
            continue
        if dd >= today_date:
            reasons["not_final_yet"] += 1
            continue

        pid = r.get("player_id")
        if not pid:
            reasons["missing_player_id"] += 1
            continue

        ck = (str(pid), d)
        if ck not in cache:
            try:
                cache[ck] = get_actual_stat(pid, "hitting", "homeRuns", d)
            except Exception as e:
                cache[ck] = None
                reasons[f"actual_fetch_error:{type(e).__name__}"] += 1

        actual = cache[ck]
        if actual is None:
            reasons["actual_missing"] += 1
            continue

        r["actual"] = actual
        r["result"] = "hit" if float(actual) > 0.5 else "miss"
        r["quality_ok"] = quality_ok(r)
        r["hr_score_bucket"] = score_bucket(safe_float(r.get("hr_score")))
        r["tier_bucket"] = hr_tier(r)
        r["lineup_spot_bucket"] = lineup_bucket(r.get("lineup_spot"))
        r["model_prob_bucket"] = prob_bucket(safe_float(r.get("model_prob")))
        r["mc_prob_bucket"] = prob_bucket(safe_float(r.get("hitter_mc_prob")))
        graded.append(r)

    return graded, reasons


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    fields = [
        "date", "api_version", "player", "player_id", "team", "opponent", "game_id",
        "pick", "actual", "result", "board_status", "prediction_tier", "reject_reason",
        "hr_score", "hr_tier", "season_slg", "h2h_slg", "recent_iso", "quality_ok",
        "model_prob", "hitter_mc_prob", "odds", "book", "odds_status", "lineup_spot",
        "hr_score_bucket", "tier_bucket", "lineup_spot_bucket", "model_prob_bucket", "mc_prob_bucket",
        "source_file", "record_source",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60, help="Lookback window for candidate logs / record rows")
    ap.add_argument("--date", default=None, help="Specific YYYY-MM-DD to audit")
    ap.add_argument("--min-n", type=int, default=5, help="Minimum sample for group printouts")
    ap.add_argument("--max-grade-rows", type=int, default=0, help="Cap rows graded from candidate logs; 0 means no cap")
    ap.add_argument("--write-csv", action="store_true", help="Write detailed graded HR rows to /data/hr_calibration_YYYYMMDD_HHMMSS.csv")
    args = ap.parse_args()

    print("HR SCORER / CALIBRATION PROBE")
    print("==============================")
    print("No api.py changes. Diagnostic only.")

    try:
        api_mod = importlib.import_module("api")
    except Exception as e:
        print(f"ERROR: Could not import api.py from current directory: {e}")
        return 2

    print(f"api_version: {getattr(api_mod, 'VERSION', 'unknown')}")
    print(f"data_dir: {getattr(api_mod, 'DATA_DIR', '/data')}")
    print(f"filters: days={args.days} date={args.date} min_n={args.min_n}")

    candidate_rows = load_candidate_rows(api_mod, args.days, args.date)
    record_rows = load_record_hr_rows(api_mod, args.days, args.date)

    print("\nINPUT COUNTS")
    print("------------")
    print(f"hr_candidate_log_rows_loaded: {len(candidate_rows)}")
    print(f"graded_record_hr_rows_loaded: {len(record_rows)}")

    graded_candidates, reasons = grade_candidates(api_mod, candidate_rows, max_rows=args.max_grade_rows)

    # Include record HR rows, but dedupe against graded candidate rows.
    all_rows = list(graded_candidates)
    seen = {
        (r.get("date"), str(r.get("game_id")), str(r.get("player_id") or ""), norm(r.get("player")), r.get("pick"))
        for r in all_rows
    }
    for r in record_rows:
        if r.get("result") not in {"hit", "miss"}:
            continue
        key = (r.get("date"), str(r.get("game_id")), str(r.get("player_id") or ""), norm(r.get("player")), r.get("pick"))
        if key in seen:
            continue
        rr = dict(r)
        rr["quality_ok"] = quality_ok(rr)
        rr["hr_score_bucket"] = score_bucket(safe_float(rr.get("hr_score")))
        rr["tier_bucket"] = hr_tier(rr)
        rr["lineup_spot_bucket"] = lineup_bucket(rr.get("lineup_spot"))
        rr["model_prob_bucket"] = prob_bucket(safe_float(rr.get("model_prob")))
        rr["mc_prob_bucket"] = prob_bucket(safe_float(rr.get("hitter_mc_prob")))
        all_rows.append(rr)
        seen.add(key)

    print("\nGRADING COUNTS")
    print("--------------")
    print(f"candidate_rows_graded: {len(graded_candidates)}")
    print(f"combined_unique_graded_hr_rows: {len(all_rows)}")
    if reasons:
        print("skip_reasons:")
        for k, v in reasons.most_common():
            print(f"  {k}: {v}")

    overall = summarize(all_rows)
    print("\nOVERALL HR SCORER RESULT")
    print("------------------------")
    print(json.dumps(overall, indent=2, sort_keys=True))

    if not all_rows:
        print("\nDIAGNOSIS")
        print("---------")
        print("No graded HR rows were available yet. Let candidate logs and completed games accumulate, then rerun.")
        print("This does not mean the HR scorer is broken; it means there is not enough completed HR-candidate history to calibrate.")
        return 0

    baseline = overall["hr_rate_pct"]

    print_group("BY BOARD STATUS", all_rows, lambda r: r.get("board_status") or "unknown", args.min_n)
    print_group("BY HR TIER", all_rows, lambda r: r.get("tier_bucket"), args.min_n)
    print_group("BY HR SCORE BUCKET", all_rows, lambda r: r.get("hr_score_bucket"), args.min_n)
    print_group("BY QUALITY OK", all_rows, lambda r: "quality_ok" if r.get("quality_ok") else "quality_not_ok", args.min_n)
    print_group("BY ODDS STATUS", all_rows, lambda r: r.get("odds_status") or ("priced" if r.get("odds") is not None else "missing"), args.min_n)
    print_group("BY LINEUP SPOT", all_rows, lambda r: r.get("lineup_spot_bucket"), args.min_n)
    print_group("BY MODEL PROB BUCKET", all_rows, lambda r: r.get("model_prob_bucket"), args.min_n)
    print_group("BY MC PROB BUCKET", all_rows, lambda r: r.get("mc_prob_bucket"), args.min_n)

    # Candidate threshold suggestions.
    combo_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        key = f"tier={r.get('tier_bucket')}|score={r.get('hr_score_bucket')}|quality={'Y' if r.get('quality_ok') else 'N'}"
        combo_groups[key].append(r)

    viable = []
    for key, rs in combo_groups.items():
        if len(rs) < args.min_n:
            continue
        s = summarize(rs)
        lift = (s["hr_rate_pct"] / baseline) if baseline else None
        viable.append((s["hr_rate_pct"], s["n"], lift, key, s))

    print("\nPOSSIBLE CALIBRATION GATES")
    print("--------------------------")
    if not viable:
        print(f"No combo groups with n >= {args.min_n}. HR sample is still too thin for a promotion decision.")
    else:
        for rate, n, lift, key, s in sorted(viable, reverse=True)[:15]:
            lift_txt = f"{lift:.2f}x" if lift is not None else "n/a"
            print(f"{key:58s} n={n:3d} rate={rate:5.1f}% lift={lift_txt} avg_score={s['avg_hr_score']}")

    # Explicit calibration warning: HR model currently uses heuristic probability-style numbers.
    model_avg = overall.get("avg_model_prob_pct")
    mc_avg = overall.get("avg_mc_prob_pct")
    observed = overall.get("hr_rate_pct")

    print("\nCALIBRATION READ")
    print("----------------")
    print(f"observed_hr_rate_pct: {observed}")
    print(f"avg_model_prob_pct: {model_avg}")
    print(f"avg_mc_prob_pct: {mc_avg}")

    if model_avg is not None and observed is not None and model_avg > observed + 10:
        print("MODEL_PROB_WARNING: HR model probability is overconfident. Treat it as a ranking score, not a true probability.")
    if mc_avg is not None and observed is not None and mc_avg > observed + 10:
        print("MC_PROB_WARNING: HR MC probability is overconfident because it is simulating a heuristic HR probability. It needs empirical calibration before official use.")

    print("\nPROMOTION DECISION")
    print("------------------")
    if len(all_rows) < max(args.min_n * 3, 30):
        print("NOT_READY: sample is too small to move HR out of experimental.")
    elif not viable:
        print("NOT_READY: no threshold bucket has enough sample.")
    else:
        best = max(viable, key=lambda x: (x[0], x[1]))
        best_rate, best_n, best_lift, best_key, _ = best
        if best_n >= args.min_n and best_rate >= max(12.0, baseline * 1.25):
            print("WATCHLIST_UPGRADE_POSSIBLE: scorer shows lift in at least one bucket.")
            print(f"Best current bucket: {best_key} n={best_n} rate={best_rate}% lift={(best_lift or 0):.2f}x")
            print("Do not make official HR picks yet unless real FanDuel HR odds/prices are added and EV is checked.")
        else:
            print("KEEP_EXPERIMENTAL: no bucket shows enough lift yet to justify promotion.")

    if args.write_csv:
        data_dir = Path(getattr(api_mod, "DATA_DIR", "/data"))
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = data_dir / f"hr_calibration_{stamp}.csv"
        write_csv(all_rows, out)
        print(f"\nCSV_WRITTEN: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
