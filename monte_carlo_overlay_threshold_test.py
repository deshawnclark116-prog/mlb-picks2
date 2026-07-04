#!/usr/bin/env python3
"""
monte_carlo_overlay_threshold_test.py

Standalone diagnostic for Prop Edge MLB.

Purpose:
- Read the latest local candidate log.
- Do NOT call PropLine, MLB, FanDuel, or any external API.
- Layer a 250,000-run Monte Carlo simulation over the system's existing candidates.
- Compare old/model probability vs Monte Carlo probability.
- Show which picks would stay, drop, or flip if MC became the final probability layer.

Default input:
  /data/candidate_logs/candidates_YYYY-MM-DD.json   newest file found

Default outputs:
  /data/candidate_logs/monte_carlo_overlay_latest.json
  /data/candidate_logs/monte_carlo_overlay_latest.csv

Run:
  python monte_carlo_overlay_test.py

Useful options:
  python monte_carlo_overlay_test.py --sim-n 250000 --top 30
  python monte_carlo_overlay_test.py --file /data/candidate_logs/candidates_2026-07-04.json
  python monte_carlo_overlay_test.py --include-rejected
  python monte_carlo_overlay_test.py --min-mc-prob 0.63 --hr-min-mc-prob 0.15
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    np = None
    NUMPY_IMPORT_ERROR = repr(e)
else:
    NUMPY_IMPORT_ERROR = None


DEFAULT_CANDIDATE_DIR = "/data/candidate_logs"
STANDARD_LINES = {
    "batter_hits": 0.5,
    "batter_total_bases": 1.5,
    "batter_home_runs": 0.5,
    "batter_rbi": 0.5,
    "batter_rbis": 0.5,
    "batter_runs": 0.5,
}

MARKET_ALIASES = {
    "hits": "batter_hits",
    "hit": "batter_hits",
    "batter_hit": "batter_hits",
    "batter_hits": "batter_hits",
    "total_bases": "batter_total_bases",
    "tb": "batter_total_bases",
    "batter_total_bases": "batter_total_bases",
    "home_runs": "batter_home_runs",
    "home_run": "batter_home_runs",
    "hr": "batter_home_runs",
    "homer": "batter_home_runs",
    "batter_home_runs": "batter_home_runs",
    "rbi": "batter_rbi",
    "rbis": "batter_rbi",
    "batter_rbi": "batter_rbi",
    "batter_rbis": "batter_rbi",
    "runs": "batter_runs",
    "run": "batter_runs",
    "batter_runs": "batter_runs",
    "strikeouts": "pitcher_strikeouts",
    "pitcher_k": "pitcher_strikeouts",
    "pitcher_ks": "pitcher_strikeouts",
    "ks": "pitcher_strikeouts",
    "pitcher_strikeouts": "pitcher_strikeouts",
}

PUBLISH_STATUSES = {
    "official",
    "official_prediction",
    "watchlist_prediction",
    "watchlist",
    "candidate_official",
}

# Diagnostic volatility parameters.  This is intentionally not a hard-coded pick model yet.
# It is a stress-test overlay for the existing system's candidate projections.
COUNT_SHAPE_BY_CONF = {
    "ELITE": 55.0,
    "HIGH": 42.0,
    "MEDIUM": 24.0,
    "MED": 24.0,
    "LOW": 12.0,
    "UNKNOWN": 18.0,
}

MARKET_SHAPE_MULTIPLIER = {
    "batter_hits": 1.00,
    "batter_total_bases": 0.70,   # TB is noisier than hits.
    "batter_rbi": 0.55,           # Opportunity-dependent.
    "batter_runs": 0.55,          # Opportunity-dependent.
    "pitcher_strikeouts": 0.90,
}

PROJECTION_KEYS_BY_MARKET = {
    "batter_hits": [
        "projection", "model_projection", "projected", "predicted", "expected", "mean",
        "hit_projection", "hits_projection", "projected_hits", "batter_hits_projection",
        "raw_projection", "stat_projection", "model_pred", "yhat",
    ],
    "batter_total_bases": [
        "projection", "model_projection", "projected", "predicted", "expected", "mean",
        "tb_projection", "total_bases_projection", "projected_total_bases", "projected_tb",
        "raw_projection", "stat_projection", "model_pred", "yhat",
    ],
    "batter_home_runs": [
        "hr_probability", "home_run_probability", "model_probability", "probability", "prob",
        "projection", "model_projection", "projected", "expected_hr", "hr_projection",
        "hr_rate", "expected", "mean", "model_pred", "yhat",
    ],
    "batter_rbi": [
        "projection", "model_projection", "projected", "predicted", "expected", "mean",
        "rbi_projection", "rbis_projection", "projected_rbi", "projected_rbis",
        "raw_projection", "stat_projection", "model_pred", "yhat",
    ],
    "batter_runs": [
        "projection", "model_projection", "projected", "predicted", "expected", "mean",
        "runs_projection", "run_projection", "projected_runs", "projected_run",
        "raw_projection", "stat_projection", "model_pred", "yhat",
    ],
    "pitcher_strikeouts": [
        "k_pitcher_projection", "pitcher_projection", "pitcher_proj", "k_projection",
        "projected_ks", "ks_projection", "strikeout_projection", "strikeouts_projection",
        "blended_projection", "k_blended_projection", "sim_mean", "projection",
        "model_projection", "projected", "predicted", "expected", "mean", "model_pred", "yhat",
    ],
}

LINE_KEYS_BY_MARKET = {
    "pitcher_strikeouts": [
        "fanduel_k_line", "k_line", "line", "line_used", "point", "threshold", "fanduel_line",
        "book_line", "market_line", "prop_line",
    ],
    "default": ["line", "line_used", "point", "threshold", "book_line", "market_line", "prop_line"],
}

PROB_KEYS = [
    "model_prob", "model_probability", "probability", "prob", "old_prob", "final_model_prob",
    "pick_probability", "edge_probability", "win_probability", "over_probability", "confidence_prob",
]

SIDE_KEYS = ["side", "pick_side", "final_side", "prediction", "lean", "bet_side"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    if isinstance(x, bool):
        return default
    if isinstance(x, (int, float)):
        if math.isfinite(float(x)):
            return float(x)
        return default
    if isinstance(x, str):
        s = x.strip().replace("%", "")
        if not s:
            return default
        try:
            v = float(s)
            if math.isfinite(v):
                return v
        except Exception:
            return default
    return default


def normalize_prob(x: Any) -> Optional[float]:
    v = safe_float(x)
    if v is None:
        return None
    # Treat 63 or 63.0 as percent, 0.63 as probability.
    if v > 1.0 and v <= 100.0:
        return v / 100.0
    if 0.0 <= v <= 1.0:
        return v
    return None


def dig_value(d: Dict[str, Any], keys: Iterable[str]) -> Tuple[Optional[Any], Optional[str]]:
    """Look for keys at top level and one level deep in common sub-objects."""
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k), k
    for parent in ("signals", "features", "debug", "projection_debug", "line_info", "market", "model", "result"):
        sub = d.get(parent)
        if isinstance(sub, dict):
            for k in keys:
                if k in sub and sub.get(k) is not None:
                    return sub.get(k), f"{parent}.{k}"
    return None, None


def canonical_market(row: Dict[str, Any]) -> str:
    for k in ("market", "prop", "prop_type", "type", "market_name", "stat", "category"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            key = v.strip().lower().replace(" ", "_").replace("-", "_")
            return MARKET_ALIASES.get(key, key)
    # Infer from available fields.
    keys = set(row.keys())
    if any(k in keys for k in ("k_pitcher_projection", "pitcher_proj", "fanduel_k_line")):
        return "pitcher_strikeouts"
    if any(k in keys for k in ("hr_score", "hr_tier", "recent_iso")):
        return "batter_home_runs"
    return "unknown"


def player_name(row: Dict[str, Any]) -> str:
    for k in ("player", "player_name", "name", "batter", "pitcher", "display_name"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # One-level nested fallback.
    for parent in ("player", "batter", "pitcher"):
        sub = row.get(parent)
        if isinstance(sub, dict):
            for k in ("name", "full_name", "display_name"):
                v = sub.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return "UNKNOWN"


def team_name(row: Dict[str, Any]) -> str:
    for k in ("team", "team_abbr", "player_team", "batter_team", "pitcher_team"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def status_of(row: Dict[str, Any]) -> str:
    for k in ("status", "candidate_status", "board_status"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    if row.get("official") is True:
        return "official"
    return "unknown"


def reject_reason(row: Dict[str, Any]) -> str:
    for k in ("reject_reason", "reason", "drop_reason", "blocked_reason"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def old_side(row: Dict[str, Any]) -> str:
    raw, _ = dig_value(row, SIDE_KEYS)
    if isinstance(raw, str):
        s = raw.strip().upper()
        if "UNDER" in s:
            return "UNDER"
        if "OVER" in s:
            return "OVER"
    return "UNKNOWN"


def old_probability(row: Dict[str, Any]) -> Optional[float]:
    raw, _ = dig_value(row, PROB_KEYS)
    return normalize_prob(raw)


def extract_projection(row: Dict[str, Any], market: str) -> Tuple[Optional[float], str]:
    keys = PROJECTION_KEYS_BY_MARKET.get(market, PROJECTION_KEYS_BY_MARKET["batter_hits"])
    raw, source = dig_value(row, keys)
    val = safe_float(raw)
    if val is not None:
        # If HR source is probability field, normalize later in simulate_hr.
        return val, source or "unknown"

    # HR special fallback: derive a rough probability proxy from score only.
    if market == "batter_home_runs":
        score = safe_float(row.get("hr_score"))
        if score is not None:
            # This is intentionally conservative. It is a diagnostic proxy, not a locked HR model.
            # Score 1.3 -> ~8%, 1.7 -> ~15%, 2.2 -> ~25%.
            p = 1.0 / (1.0 + math.exp(-3.25 * (score - 2.15)))
            p = max(0.03, min(0.32, p))
            return p, "score_proxy.hr_score"

    return None, "missing"


def extract_line(row: Dict[str, Any], market: str) -> Tuple[Optional[float], str]:
    line_keys = LINE_KEYS_BY_MARKET.get(market, LINE_KEYS_BY_MARKET["default"])
    raw, source = dig_value(row, line_keys)
    val = safe_float(raw)
    if val is not None:
        return val, source or "unknown"
    if market in STANDARD_LINES:
        return STANDARD_LINES[market], "standard_default"
    return None, "missing"


def confidence_label(row: Dict[str, Any]) -> str:
    for k in ("confidence", "tier", "pick_confidence", "final_confidence", "hr_tier"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            s = v.strip().upper()
            if "ELITE" in s:
                return "ELITE"
            if "HIGH" in s:
                return "HIGH"
            if "MED" in s:
                return "MEDIUM"
            if "LOW" in s:
                return "LOW"
    return "UNKNOWN"


def shape_for(row: Dict[str, Any], market: str) -> float:
    conf = confidence_label(row)
    base = COUNT_SHAPE_BY_CONF.get(conf, COUNT_SHAPE_BY_CONF["UNKNOWN"])
    mult = MARKET_SHAPE_MULTIPLIER.get(market, 0.8)

    # Record-known flags can tighten/loosen a little without overriding the system.
    bvp_flag = str(row.get("bvp_flag", row.get("bvp", ""))).lower()
    if any(x in bvp_flag for x in ("premium", "good")):
        base *= 1.08
    if any(x in bvp_flag for x in ("lean", "struggle")):
        base *= 0.82

    # Bad/rejected flags loosen uncertainty.
    rr = reject_reason(row).lower()
    if rr and rr != "none":
        base *= 0.75

    return max(4.0, base * mult)


def simulate_count_market(
    rng: Any,
    market: str,
    projection: float,
    line: float,
    sim_n: int,
    shape: float,
) -> Dict[str, Any]:
    """Gamma-Poisson Monte Carlo for count-like props.

    Mean = projection.
    Gamma mixing adds uncertainty around the candidate's projection.
    Counts are then simulated as Poisson(lambda_sample).
    """
    mean = max(0.0001, float(projection))
    shape = max(4.0, float(shape))
    scale = mean / shape
    lam = rng.gamma(shape=shape, scale=scale, size=sim_n)
    sims = rng.poisson(lam=lam)

    over = sims > line
    under = sims < line
    push = sims == line

    over_rate = float(over.mean())
    under_rate = float(under.mean())
    push_rate = float(push.mean())
    side = "OVER" if over_rate >= under_rate else "UNDER"
    mc_prob = max(over_rate, under_rate)

    return {
        "sim_mean": float(np.mean(sims)),
        "sim_median": float(np.median(sims)),
        "sim_p10": float(np.quantile(sims, 0.10)),
        "sim_p90": float(np.quantile(sims, 0.90)),
        "sim_prob_over": over_rate,
        "sim_prob_under": under_rate,
        "sim_prob_push": push_rate,
        "mc_side": side,
        "mc_prob": mc_prob,
    }


def simulate_hr_market(rng: Any, projection: float, line: float, sim_n: int, projection_source: str) -> Dict[str, Any]:
    """Bernoulli simulation for HR over 0.5.

    projection can be a probability or an expected HR rate.  For HR, current system is score/tier-like,
    so this remains diagnostic until calibrated.
    """
    p = normalize_prob(projection)
    if p is None:
        p = safe_float(projection, 0.0) or 0.0
        # If value looks like an expected HR count/lambda, convert to at-least-one probability.
        if p > 1.0:
            p = min(0.95, p / 100.0)
        elif p <= 0.35:
            p = 1.0 - math.exp(-max(0.0, p))
    p = max(0.0, min(0.95, float(p)))

    sims = rng.binomial(n=1, p=p, size=sim_n)
    over = sims > line
    under = sims < line
    push = sims == line

    over_rate = float(over.mean())
    under_rate = float(under.mean())
    push_rate = float(push.mean())
    side = "OVER" if over_rate >= under_rate else "UNDER"
    mc_prob = max(over_rate, under_rate)

    return {
        "sim_mean": float(np.mean(sims)),
        "sim_median": float(np.median(sims)),
        "sim_p10": float(np.quantile(sims, 0.10)),
        "sim_p90": float(np.quantile(sims, 0.90)),
        "sim_prob_over": over_rate,
        "sim_prob_under": under_rate,
        "sim_prob_push": push_rate,
        "mc_side": side,
        "mc_prob": mc_prob,
        "hr_probability_used": p,
        "hr_probability_source": projection_source,
    }


def flatten_candidates(obj: Any) -> List[Dict[str, Any]]:
    """Robustly extract candidate dicts from several possible log shapes."""
    found: List[Dict[str, Any]] = []

    def looks_like_candidate(d: Dict[str, Any]) -> bool:
        m = canonical_market(d)
        if m != "unknown":
            return True
        marker_keys = {
            "player", "player_name", "batter", "pitcher", "projection", "model_projection",
            "status", "reject_reason", "model_prob", "k_pitcher_projection", "hr_score",
        }
        return len(marker_keys.intersection(d.keys())) >= 3

    def walk(x: Any, path: str = "") -> None:
        if isinstance(x, list):
            if x and all(isinstance(i, dict) for i in x):
                # If this is a candidate-ish list, harvest it. Otherwise recurse.
                candidateish = sum(1 for i in x if looks_like_candidate(i))
                if candidateish >= max(1, len(x) // 3):
                    for i in x:
                        if isinstance(i, dict) and looks_like_candidate(i):
                            found.append(i)
                    return
            for i in x:
                walk(i, path + "[]")
        elif isinstance(x, dict):
            # Prefer common direct containers first.
            for key in (
                "candidates", "items", "official", "officials", "official_board", "board",
                "predictions", "rejected", "watchlist", "rows", "data",
            ):
                if key in x and isinstance(x[key], list):
                    walk(x[key], path + f".{key}")
            # If this dict itself is a single candidate, keep it.
            if looks_like_candidate(x):
                found.append(x)

    walk(obj)

    # Deduplicate by stable identity.
    dedup: List[Dict[str, Any]] = []
    seen = set()
    for d in found:
        ident = (
            player_name(d),
            team_name(d),
            canonical_market(d),
            status_of(d),
            reject_reason(d),
            str(d.get("line", d.get("point", d.get("fanduel_k_line", "")))),
        )
        if ident not in seen:
            seen.add(ident)
            dedup.append(d)
    return dedup


def find_latest_candidate_file(candidate_dir: str) -> Optional[str]:
    patterns = [
        os.path.join(candidate_dir, "candidates_*.json"),
        os.path.join(candidate_dir, "candidate_*.json"),
        os.path.join(candidate_dir, "*candidates*.json"),
    ]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files), key=lambda p: os.path.getmtime(p), reverse=True)
    for p in files:
        if os.path.basename(p).startswith("graded_"):
            continue
        if "monte_carlo_overlay" in os.path.basename(p):
            continue
        return p
    return None


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{x * 100:.1f}%"


def round_or_none(x: Optional[float], ndigits: int = 4) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), ndigits)
    except Exception:
        return None


def analyze_candidate(
    row: Dict[str, Any],
    rng: Any,
    sim_n: int,
    min_mc_prob: float,
    hr_min_mc_prob: float,
    market_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    market = canonical_market(row)
    player = player_name(row)
    team = team_name(row)
    status = status_of(row)
    reason = reject_reason(row)

    projection, projection_source = extract_projection(row, market)
    line, line_source = extract_line(row, market)
    old_prob = old_probability(row)
    old_pick_side = old_side(row)
    conf = confidence_label(row)

    out: Dict[str, Any] = {
        "player": player,
        "team": team,
        "market": market,
        "status": status,
        "reject_reason": reason,
        "old_side": old_pick_side,
        "old_prob": old_prob,
        "confidence": conf,
        "projection": projection,
        "projection_source": projection_source,
        "line": line,
        "line_source": line_source,
        "sim_n": sim_n,
        "sim_engine": "none",
        "mc_side": "UNKNOWN",
        "mc_prob": None,
        "sim_prob_over": None,
        "sim_prob_under": None,
        "sim_prob_push": None,
        "sim_mean": None,
        "sim_median": None,
        "sim_p10": None,
        "sim_p90": None,
        "projection_gap": None,
        "side_flip": False,
        "would_keep": False,
        "decision_reason": "not_simulated",
    }

    if market == "unknown":
        out["decision_reason"] = "unknown_market"
        return out
    if projection is None:
        out["decision_reason"] = "missing_projection"
        return out
    if line is None:
        out["decision_reason"] = "missing_line"
        return out

    out["projection_gap"] = projection - line

    if market == "batter_home_runs":
        sim = simulate_hr_market(rng, projection, line, sim_n, projection_source)
        out.update(sim)
        out["sim_engine"] = "bernoulli_hr"
        threshold = hr_min_mc_prob
        out["market_mc_threshold"] = threshold
        # For HR we judge the OVER probability specifically, not max(over/under), because UNDER is usually huge.
        out["mc_side"] = "OVER"
        out["mc_prob"] = out["sim_prob_over"]
        out["would_keep"] = bool((out["mc_prob"] or 0.0) >= threshold)
        out["decision_reason"] = "keep_hr_mc_threshold" if out["would_keep"] else "drop_below_hr_mc_threshold"
    else:
        shape = shape_for(row, market)
        sim = simulate_count_market(rng, market, projection, line, sim_n, shape)
        out.update(sim)
        out["sim_engine"] = "gamma_poisson_count"
        out["shape_used"] = shape
        threshold = (market_thresholds or {}).get(market, min_mc_prob)
        out["market_mc_threshold"] = threshold
        out["would_keep"] = bool((out["mc_prob"] or 0.0) >= threshold)
        out["decision_reason"] = "keep_mc_threshold" if out["would_keep"] else "drop_below_mc_threshold"

    if old_pick_side in ("OVER", "UNDER") and out["mc_side"] in ("OVER", "UNDER"):
        out["side_flip"] = old_pick_side != out["mc_side"]

    # Preserve compact raw flags useful for diagnosing deltas.
    for k in (
        "bvp_flag", "lineup_spot", "batting_order", "hr_score", "hr_tier", "k_has_lineup_kr",
        "k_lineup_exp_ks", "k_nudge", "avg_bf", "recent_k_avg", "outs_per_start", "starts",
        "model_prob", "probability", "score", "rank",
    ):
        if k in row:
            out[k] = row.get(k)

    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    preferred = [
        "player", "team", "market", "status", "reject_reason", "old_side", "old_prob",
        "projection", "line", "projection_gap", "mc_side", "mc_prob", "sim_prob_over",
        "sim_prob_under", "sim_prob_push", "sim_mean", "sim_median", "sim_p10", "sim_p90",
        "would_keep", "side_flip", "decision_reason", "confidence", "projection_source",
        "line_source", "sim_engine", "sim_n",
    ]
    extras = sorted({k for r in rows for k in r.keys()} - set(preferred))
    fields = preferred + extras
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_summary(rows: List[Dict[str, Any]], input_file: str, sim_n: int) -> Dict[str, Any]:
    by_market = Counter(r["market"] for r in rows)
    by_status = Counter(r["status"] for r in rows)
    by_decision = Counter(r["decision_reason"] for r in rows)
    kept = [r for r in rows if r.get("would_keep")]
    dropped = [r for r in rows if not r.get("would_keep") and r.get("mc_prob") is not None]
    flips = [r for r in rows if r.get("side_flip")]

    market_summary = {}
    for m in sorted(by_market):
        mrows = [r for r in rows if r["market"] == m]
        simmed = [r for r in mrows if r.get("mc_prob") is not None]
        market_summary[m] = {
            "total": len(mrows),
            "simulated": len(simmed),
            "would_keep": sum(1 for r in simmed if r.get("would_keep")),
            "would_drop": sum(1 for r in simmed if not r.get("would_keep")),
            "avg_mc_prob": round(float(statistics.mean([r["mc_prob"] for r in simmed])), 4) if simmed else None,
            "avg_old_prob": round(float(statistics.mean([r["old_prob"] for r in simmed if r.get("old_prob") is not None])), 4)
            if any(r.get("old_prob") is not None for r in simmed) else None,
            "side_flips": sum(1 for r in simmed if r.get("side_flip")),
        }

    return {
        "created_at_utc": now_iso(),
        "input_file": input_file,
        "sim_n": sim_n,
        "total_rows": len(rows),
        "simulated_rows": sum(1 for r in rows if r.get("mc_prob") is not None),
        "would_keep": len(kept),
        "would_drop": len(dropped),
        "side_flips": len(flips),
        "by_market": dict(by_market),
        "by_status": dict(by_status),
        "by_decision": dict(by_decision),
        "market_summary": market_summary,
    }


def print_report(summary: Dict[str, Any], rows: List[Dict[str, Any]], top: int) -> None:
    print("\n=== MONTE CARLO OVERLAY TEST ===")
    print(f"Input file: {summary['input_file']}")
    print(f"Simulations per candidate: {summary['sim_n']:,}")
    print(f"Rows found: {summary['total_rows']}")
    print(f"Rows simulated: {summary['simulated_rows']}")
    print(f"Would keep: {summary['would_keep']}")
    print(f"Would drop: {summary['would_drop']}")
    print(f"Side flips: {summary['side_flips']}")

    print("\n--- BY MARKET ---")
    for m, s in summary["market_summary"].items():
        print(
            f"{m}: total={s['total']} simulated={s['simulated']} keep={s['would_keep']} "
            f"drop={s['would_drop']} avg_mc={fmt_pct(s['avg_mc_prob'])} avg_old={fmt_pct(s['avg_old_prob'])} flips={s['side_flips']}"
        )

    print("\n--- DECISIONS ---")
    for k, v in sorted(summary["by_decision"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{k}: {v}")

    ranked = [r for r in rows if r.get("mc_prob") is not None]
    ranked.sort(key=lambda r: (r.get("would_keep", False), r.get("mc_prob") or -1), reverse=True)

    print(f"\n--- TOP {min(top, len(ranked))} BY MC PROB ---")
    for i, r in enumerate(ranked[:top], 1):
        oldp = fmt_pct(r.get("old_prob"))
        mcp = fmt_pct(r.get("mc_prob"))
        overp = fmt_pct(r.get("sim_prob_over"))
        underp = fmt_pct(r.get("sim_prob_under"))
        proj = r.get("projection")
        line = r.get("line")
        gap = r.get("projection_gap")
        print(
            f"{i:02d}. {r['player']} {r.get('team','')} | {r['market']} | "
            f"proj={proj:.3f} line={line:.1f} gap={gap:+.3f} | "
            f"MC {r['mc_side']}={mcp} (O={overp}, U={underp}) | old={oldp} | "
            f"keep={r['would_keep']} | status={r['status']} | reason={r['decision_reason']}"
        )

    flips = [r for r in ranked if r.get("side_flip")]
    if flips:
        print(f"\n--- SIDE FLIPS ({len(flips)}) ---")
        for r in flips[:top]:
            print(
                f"{r['player']} | {r['market']} | old={r['old_side']} -> mc={r['mc_side']} | "
                f"proj={r['projection']:.3f} line={r['line']:.1f} mc={fmt_pct(r['mc_prob'])}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone Monte Carlo overlay test for Prop Edge candidates.")
    ap.add_argument("--file", dest="candidate_file", default=None, help="Specific candidate JSON file to read.")
    ap.add_argument("--candidate-dir", default=DEFAULT_CANDIDATE_DIR, help="Candidate log directory.")
    ap.add_argument("--out-dir", default=DEFAULT_CANDIDATE_DIR, help="Output directory.")
    ap.add_argument("--sim-n", type=int, default=250000, help="Simulations per candidate. Default: 250000.")
    ap.add_argument("--seed", type=int, default=81718, help="Random seed.")
    ap.add_argument("--top", type=int, default=30, help="Rows to print in ranked table.")
    ap.add_argument("--include-rejected", action="store_true", help="Also simulate rejected candidates. Default simulates published/watchlist statuses plus anything not clearly rejected.")
    ap.add_argument("--min-mc-prob", type=float, default=0.63, help="Fallback keep threshold for non-HR markets. Accepts 0.63 or 63.")
    ap.add_argument("--hit-min-mc-prob", type=float, default=None, help="Override keep threshold for batter_hits. Example: 0.606")
    ap.add_argument("--tb-min-mc-prob", type=float, default=None, help="Override keep threshold for batter_total_bases. Example: 0.553")
    ap.add_argument("--k-min-mc-prob", type=float, default=None, help="Override keep threshold for pitcher_strikeouts.")
    ap.add_argument("--rbi-min-mc-prob", type=float, default=None, help="Override keep threshold for batter_rbi/batter_rbis.")
    ap.add_argument("--runs-min-mc-prob", type=float, default=None, help="Override keep threshold for batter_runs.")
    ap.add_argument("--hr-min-mc-prob", type=float, default=0.15, help="Keep threshold for HR over 0.5. Accepts 0.15 or 15.")
    args = ap.parse_args()

    if np is None:
        print("ERROR: numpy is required for the 250,000-run Monte Carlo test.", file=sys.stderr)
        print(f"numpy import error: {NUMPY_IMPORT_ERROR}", file=sys.stderr)
        return 2

    sim_n = max(1000, int(args.sim_n))
    min_mc_prob = normalize_prob(args.min_mc_prob)
    hr_min_mc_prob = normalize_prob(args.hr_min_mc_prob)
    if min_mc_prob is None:
        min_mc_prob = 0.63
    if hr_min_mc_prob is None:
        hr_min_mc_prob = 0.15

    market_thresholds: Dict[str, float] = {}
    threshold_arg_map = {
        "batter_hits": args.hit_min_mc_prob,
        "batter_total_bases": args.tb_min_mc_prob,
        "pitcher_strikeouts": args.k_min_mc_prob,
        "batter_rbi": args.rbi_min_mc_prob,
        "batter_runs": args.runs_min_mc_prob,
    }
    for market_name, raw_threshold in threshold_arg_map.items():
        if raw_threshold is not None:
            v = normalize_prob(raw_threshold)
            if v is not None:
                market_thresholds[market_name] = v

    candidate_file = args.candidate_file or find_latest_candidate_file(args.candidate_dir)
    if not candidate_file:
        print(f"ERROR: No candidate JSON file found in {args.candidate_dir}", file=sys.stderr)
        return 1

    data = load_json(candidate_file)
    candidates = flatten_candidates(data)
    if not candidates:
        print(f"ERROR: No candidate rows found in {candidate_file}", file=sys.stderr)
        return 1

    # Default: do not waste time simming clearly rejected rows unless requested.
    if not args.include_rejected:
        filtered = []
        for r in candidates:
            st = status_of(r).lower()
            rr = reject_reason(r).lower()
            if st in PUBLISH_STATUSES or (st in ("unknown", "") and rr in ("", "none")):
                filtered.append(r)
        # If filter accidentally removes everything due to unfamiliar status labels, fall back to all.
        if filtered:
            candidates = filtered

    rng = np.random.default_rng(args.seed)
    rows = [analyze_candidate(r, rng, sim_n, min_mc_prob, hr_min_mc_prob, market_thresholds) for r in candidates]

    # Round floats in output for readability but keep enough precision.
    cleaned_rows: List[Dict[str, Any]] = []
    for r in rows:
        cleaned = {}
        for k, v in r.items():
            if isinstance(v, float):
                if k.endswith("prob") or "prob_" in k or k in ("old_prob", "mc_prob"):
                    cleaned[k] = round(v, 6)
                else:
                    cleaned[k] = round(v, 4)
            else:
                cleaned[k] = v
        cleaned_rows.append(cleaned)

    summary = make_summary(cleaned_rows, candidate_file, sim_n)
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "monte_carlo_overlay_latest.json")
    out_csv = os.path.join(args.out_dir, "monte_carlo_overlay_latest.csv")

    payload = {
        "summary": summary,
        "settings": {
            "sim_n": sim_n,
            "seed": args.seed,
            "min_mc_prob": min_mc_prob,
            "hr_min_mc_prob": hr_min_mc_prob,
            "market_thresholds": market_thresholds,
            "include_rejected": bool(args.include_rejected),
            "note": "Standalone diagnostic only. No external API calls. Does not modify live picks.",
        },
        "rows": cleaned_rows,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_csv(out_csv, cleaned_rows)

    print_report(summary, cleaned_rows, args.top)
    print("\n--- OUTPUTS ---")
    print(out_json)
    print(out_csv)
    print("\nNOTE: This is an overlay test. It does not change api.py, predictions, record.json, or PropLine usage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
