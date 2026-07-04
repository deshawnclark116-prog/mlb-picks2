"""
projection_audit.py

Standalone diagnostic test for Prop Edge.

Purpose:
- Test saved projections against actual results.
- No API changes.
- No model changes.
- No prediction changes.
- No retraining.
- No future-data leakage.

It reads:
/data/record.json

It prints:
- overall projection accuracy
- by market
- by side
- by confidence
- by BVP bucket
- by lineup spot
- by lineup K flag
- worst projection misses
- closest projection hits

Usage examples:
python projection_audit.py
python projection_audit.py --days 7
python projection_audit.py --market batter_hits
python projection_audit.py --market pitcher_strikeouts
python projection_audit.py --market batter_total_bases
python projection_audit.py --api-version 8.16E
python projection_audit.py --date 2026-07-03
"""

import argparse
import datetime as dt
import json
import math
import re
from collections import defaultdict
from pathlib import Path


DATA_DIR = Path("/data")
RECORD_PATH = DATA_DIR / "record.json"


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def load_record():
    if not RECORD_PATH.exists():
        return {"summary": {}, "results": []}

    try:
        data = json.loads(RECORD_PATH.read_text())
        if not isinstance(data, dict):
            return {"summary": {}, "results": []}
        if not isinstance(data.get("results"), list):
            data["results"] = []
        return data
    except Exception as e:
        return {"summary": {"error": str(e)}, "results": []}


def pick_side(row):
    prop = row.get("prop_type")
    pick = str(row.get("pick", "")).upper().strip()

    if prop == "moneyline":
        return "ML"
    if pick.startswith("OVER"):
        return "OVER"
    if pick.startswith("UNDER"):
        return "UNDER"
    if "+" in pick:
        return "THRESHOLD"
    return "UNKNOWN"


def pick_line(row):
    pick = str(row.get("pick", "")).upper().strip()

    m = re.search(r"(?:OVER|UNDER)\s+(-?\d+(?:\.\d+)?)", pick)
    if m:
        return safe_float(m.group(1))

    m2 = re.search(r"^(\d+)\+", pick)
    if m2:
        return safe_float(m2.group(1))

    return None


def prob_bucket(row):
    p = safe_float(row.get("model_prob"))
    if p is None:
        return "no_prob"
    if p < 0.55:
        return "lt_55"
    if p < 0.60:
        return "55_59"
    if p < 0.65:
        return "60_64"
    if p < 0.70:
        return "65_69"
    if p < 0.75:
        return "70_74"
    if p < 0.80:
        return "75_79"
    return "80_plus"


def bvp_bucket(row):
    flag = row.get("bvp_flag")
    if not flag:
        return "none"

    flag = str(flag)
    tags = [
        "sum_premium",
        "sum_strong",
        "sum_good",
        "sum_lean",
        "sum_avoid",
        "hits",
        "struggles",
        "power",
        "weak",
        "hr_score",
        "lineup_kr",
    ]

    for tag in tags:
        if flag.startswith(tag):
            return tag

    return flag


def lineup_spot_bucket(row):
    spot = row.get("lineup_spot")
    try:
        spot = int(spot)
    except Exception:
        return "no_lineup_spot_recorded"

    if spot <= 0:
        return "bad_lineup_spot"
    if spot <= 3:
        return "spot_1_3"
    if spot <= 6:
        return "spot_4_6"
    if spot <= 9:
        return "spot_7_9"
    return "spot_10_plus"


def lineup_k_bucket(row):
    flag = row.get("bvp_flag")
    if isinstance(flag, str) and flag.startswith("lineup_kr"):
        return "with_lineup_kr"
    return "no_lineup_kr"


def line_bucket(row):
    line = pick_line(row)
    if line is None:
        return "no_line"

    if abs(line - round(line)) < 1e-9:
        return f"line_{int(line)}"

    return f"line_{line:.1f}"


def projection_gap_bucket(row):
    side = pick_side(row)
    line = pick_line(row)
    projected = safe_float(row.get("projected"))

    if side not in ("OVER", "UNDER") or line is None or projected is None:
        return "no_gap"

    if side == "OVER":
        gap = projected - line
    else:
        gap = line - projected

    if gap < 0:
        return "negative_gap"
    if gap < 0.25:
        return "gap_0_0.24"
    if gap < 0.50:
        return "gap_0.25_0.49"
    if gap < 0.75:
        return "gap_0.50_0.74"
    if gap < 1.00:
        return "gap_0.75_0.99"
    if gap < 1.50:
        return "gap_1.00_1.49"
    if gap < 2.00:
        return "gap_1.50_1.99"
    return "gap_2_plus"


def hr_score_bucket(row):
    score = safe_float(row.get("hr_score"))

    if score is None:
        flag = row.get("bvp_flag")
        if isinstance(flag, str):
            m = re.search(r"hr_score_([0-9]+(?:\.[0-9]+)?)", flag)
            if m:
                score = safe_float(m.group(1))

    if score is None:
        return "no_hr_score_recorded"

    if score < 1.30:
        return "lt_1.30"
    if score < 1.50:
        return "1.30_1.49"
    if score < 1.70:
        return "1.50_1.69"
    return "1.70_plus"


def filter_rows(rows, days=None, date=None, market=None, api_version=None):
    out = []

    today = dt.date.today()

    for row in rows:
        if not isinstance(row, dict):
            continue

        if date and row.get("date") != date:
            continue

        if market and row.get("prop_type") != market:
            continue

        if api_version and str(row.get("api_version")) != str(api_version):
            continue

        if days is not None:
            try:
                row_date = dt.date.fromisoformat(row.get("date"))
                cutoff = today - dt.timedelta(days=int(days))
                if row_date < cutoff:
                    continue
            except Exception:
                continue

        out.append(row)

    return out


def with_projection_error(row):
    projected = safe_float(row.get("projected"))
    actual = safe_float(row.get("actual"))

    if projected is None or actual is None:
        return None

    new = dict(row)
    new["_side"] = pick_side(row)
    new["_projection_error"] = projected - actual
    new["_abs_error"] = abs(projected - actual)
    new["_bvp_bucket"] = bvp_bucket(row)
    new["_prob_bucket"] = prob_bucket(row)
    new["_lineup_spot_bucket"] = lineup_spot_bucket(row)
    new["_lineup_k_bucket"] = lineup_k_bucket(row)
    new["_line_bucket"] = line_bucket(row)
    new["_projection_gap_bucket"] = projection_gap_bucket(row)
    new["_hr_score_bucket"] = hr_score_bucket(row)
    return new


def summarize(rows):
    rows = [r for r in rows if isinstance(r, dict)]

    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "hits": 0,
            "misses": 0,
            "hit_rate": 0,
            "avg_projected": None,
            "avg_actual": None,
            "projection_bias": None,
            "avg_abs_error": None,
            "bias_interpretation": "not_available",
        }

    hits = sum(1 for r in rows if r.get("result") == "hit")
    misses = sum(1 for r in rows if r.get("result") == "miss")

    projected = [safe_float(r.get("projected")) for r in rows]
    actual = [safe_float(r.get("actual")) for r in rows]

    projected = [x for x in projected if x is not None]
    actual = [x for x in actual if x is not None]

    error_rows = [with_projection_error(r) for r in rows]
    error_rows = [r for r in error_rows if r is not None]

    errors = [r["_projection_error"] for r in error_rows]
    abs_errors = [r["_abs_error"] for r in error_rows]

    avg_projected = sum(projected) / len(projected) if projected else None
    avg_actual = sum(actual) / len(actual) if actual else None
    projection_bias = sum(errors) / len(errors) if errors else None
    avg_abs_error = sum(abs_errors) / len(abs_errors) if abs_errors else None

    if projection_bias is None:
        bias_text = "not_available"
    elif projection_bias > 0.05:
        bias_text = "over_projecting"
    elif projection_bias < -0.05:
        bias_text = "under_projecting"
    else:
        bias_text = "roughly_balanced"

    return {
        "total": total,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total * 100, 1) if total else 0,
        "avg_projected": round(avg_projected, 3) if avg_projected is not None else None,
        "avg_actual": round(avg_actual, 3) if avg_actual is not None else None,
        "projection_bias": round(projection_bias, 3) if projection_bias is not None else None,
        "avg_abs_error": round(avg_abs_error, 3) if avg_abs_error is not None else None,
        "bias_interpretation": bias_text,
    }


def group_summary(rows, key_func):
    grouped = defaultdict(list)

    for row in rows:
        try:
            key = key_func(row)
        except Exception:
            key = "unknown"

        if key is None or key == "":
            key = "unknown"

        grouped[str(key)].append(row)

    out = {}
    for key in sorted(grouped.keys()):
        out[key] = summarize(grouped[key])

    return out


def compact_row(row):
    return {
        "date": row.get("date"),
        "api_version": row.get("api_version"),
        "player": row.get("player"),
        "team": row.get("team"),
        "opponent": row.get("opponent"),
        "game_id": row.get("game_id"),
        "prop_type": row.get("prop_type"),
        "pick": row.get("pick"),
        "projected": row.get("projected"),
        "actual": row.get("actual"),
        "projection_error": round(row.get("_projection_error"), 3),
        "abs_error": round(row.get("_abs_error"), 3),
        "result": row.get("result"),
        "model_prob": row.get("model_prob"),
        "confidence": row.get("confidence"),
        "bvp_flag": row.get("bvp_flag"),
        "lineup_spot": row.get("lineup_spot"),
        "odds": row.get("odds"),
        "book": row.get("book"),
        "line_source": row.get("line_source"),
        "raw_market": row.get("raw_market"),
        "k_gate_version": row.get("k_gate_version"),
        "k_projection_gap": row.get("k_projection_gap"),
        "k_has_lineup_kr": row.get("k_has_lineup_kr"),
        "hr_score": row.get("hr_score"),
        "hr_tier": row.get("hr_tier"),
    }


def build_audit(rows, limit=25):
    audit_rows = []
    missing_projection = []

    for row in rows:
        er = with_projection_error(row)
        if er is None:
            missing_projection.append(row)
        else:
            audit_rows.append(er)

    limit = max(1, min(int(limit or 25), 100))

    worst_projection_misses = sorted(
        [r for r in audit_rows if r.get("result") == "miss"],
        key=lambda r: r.get("_abs_error", 0),
        reverse=True,
    )[:limit]

    biggest_projection_hits = sorted(
        [r for r in audit_rows if r.get("result") == "hit"],
        key=lambda r: r.get("_abs_error", 0),
        reverse=True,
    )[:limit]

    closest_projection_hits = sorted(
        [r for r in audit_rows if r.get("result") == "hit"],
        key=lambda r: r.get("_abs_error", 999),
    )[:limit]

    return {
        "diagnostic_only": True,
        "leakage_protection": {
            "uses_saved_record_json": True,
            "does_not_rerun_old_games": True,
            "does_not_retrain_models": True,
            "does_not_change_picks": True,
        },
        "rows_seen": len(rows),
        "rows_with_projection_and_actual": len(audit_rows),
        "rows_missing_projection_or_actual": len(missing_projection),
        "summary": summarize(audit_rows),
        "by_market": group_summary(audit_rows, lambda r: r.get("prop_type")),
        "by_api_version": group_summary(audit_rows, lambda r: r.get("api_version") or "no_api_version_recorded"),
        "by_side": group_summary(audit_rows, pick_side),
        "by_market_side": group_summary(audit_rows, lambda r: f"{r.get('prop_type')}|{pick_side(r)}"),
        "by_confidence": group_summary(audit_rows, lambda r: r.get("confidence") or "unknown"),
        "by_probability_bucket": group_summary(audit_rows, prob_bucket),
        "by_bvp_bucket": group_summary(audit_rows, bvp_bucket),
        "by_lineup_spot": group_summary(audit_rows, lineup_spot_bucket),
        "by_lineup_k_flag": group_summary(audit_rows, lineup_k_bucket),
        "by_projection_gap_bucket": group_summary(audit_rows, projection_gap_bucket),
        "by_line_bucket": group_summary(audit_rows, line_bucket),
        "by_line_source": group_summary(audit_rows, lambda r: r.get("line_source") or "unknown"),
        "by_k_gate_version": group_summary(audit_rows, lambda r: r.get("k_gate_version") or "no_k_gate_version"),
        "by_hr_score_bucket": group_summary(audit_rows, hr_score_bucket),
        "worst_projection_misses": [compact_row(r) for r in worst_projection_misses],
        "biggest_projection_hits": [compact_row(r) for r in biggest_projection_hits],
        "closest_projection_hits": [compact_row(r) for r in closest_projection_hits],
        "training_read": {
            "projection_bias": "positive means projected stat was higher than actual; negative means projected stat was lower than actual",
            "avg_abs_error": "average size of projection miss regardless of direction",
            "best_use": "use by_market, by_side, by_bvp_bucket, by_lineup_k_flag, and worst_projection_misses to decide what needs model training work",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--market", type=str, default=None)
    parser.add_argument("--api-version", type=str, default=None)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    data = load_record()
    rows = data.get("results", [])
    rows = filter_rows(
        rows,
        days=args.days,
        date=args.date,
        market=args.market,
        api_version=args.api_version,
    )

    audit = build_audit(rows, limit=args.limit)

    audit["filters"] = {
        "days": args.days,
        "date": args.date,
        "market": args.market,
        "api_version": args.api_version,
        "limit": args.limit,
    }

    print(json.dumps(audit, indent=2))

    if args.save:
        out_dir = DATA_DIR / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"projection_audit_{stamp}.json"
        out_path.write_text(json.dumps(audit, indent=2))
        print(f"\nSaved audit to: {out_path}")


if __name__ == "__main__":
    main()
