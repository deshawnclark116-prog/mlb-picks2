#!/usr/bin/env python3
"""
k_mc_match_test.py

Standalone diagnostic only. Does NOT modify api.py, predictions, record.json, or live picks.

Purpose:
  1) Pull today's/tomorrow's FanDuel main MLB pitcher strikeout lines from Odds-API.io
     using the documented single-event player-props endpoint.
  2) Read the latest local candidate log and find pitcher_strikeouts candidates.
  3) Match provider K lines to system K candidates.
  4) Run 250,000 Monte Carlo simulations on matched K candidates.
  5) Print matched/unmatched lines and save JSON/CSV diagnostics.

Env key supported:
  ODDS_API_IO_KEY

Example:
  python k_mc_match_test.py --max-events 25
  python k_mc_match_test.py --max-events 25 --sim-n 250000 --min-mc-prob 0.63
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import sys
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

try:
    import requests
except Exception as exc:  # pragma: no cover
    print("ERROR: requests is required:", exc)
    sys.exit(1)

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

BASE = "https://api.odds-api.io/v3"
SPORT = "baseball"
LEAGUE = "usa-mlb"
BOOKMAKER = "FanDuel"
OUT_DIR = Path("/data/candidate_logs")
ALT_OUT_DIR = Path("/data/odds_provider_tests")

K_MARKET_PATTERNS = (
    "total strikeouts",
    "pitcher strikeouts",
    "strikeouts",
)

PROJECTION_KEYS = [
    "k_pitcher_projection",
    "pitcher_projection",
    "projected_ks",
    "projection_ks",
    "projected_k",
    "projection",
    "projected",
    "model_projection",
    "model_projected",
    "expected_ks",
    "mean",
]

PLAYER_KEYS = [
    "pitcher",
    "pitcher_name",
    "player",
    "player_name",
    "name",
    "athlete",
]

MARKET_KEYS = ["prop", "market", "market_name", "type", "pick_type", "stat_type"]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm_name(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    # remove common suffixes and team/side junk
    s = re.sub(r"\([^)]*total strikeouts[^)]*\)", " ", s, flags=re.I)
    s = re.sub(r"\([^)]*strikeouts[^)]*\)", " ", s, flags=re.I)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_k_suffix(label: Any) -> str:
    s = str(label or "").strip()
    s = re.sub(r"\s*\([^)]*Total Strikeouts[^)]*\)\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*\([^)]*Strikeouts[^)]*\)\s*$", "", s, flags=re.I)
    return s.strip()


def as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if math.isfinite(float(x)):
            return float(x)
        return None
    s = str(x).strip().replace("%", "")
    if not s or s.upper() in {"N/A", "NA", "NONE", "NULL"}:
        return None
    try:
        val = float(s)
        if math.isfinite(val):
            return val
    except Exception:
        return None
    return None


def get_nested(obj: Any, path: Iterable[str]) -> Any:
    cur = obj
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def safe_shape(x: Any) -> Dict[str, Any]:
    if isinstance(x, list):
        d = {"type": "list", "len": len(x)}
        if x:
            d["first_type"] = type(x[0]).__name__
            if isinstance(x[0], dict):
                d["first_keys"] = list(x[0].keys())[:12]
        return d
    if isinstance(x, dict):
        return {"type": "dict", "len": len(x), "first_keys": list(x.keys())[:12]}
    return {"type": type(x).__name__}


def request_json(url: str, timeout: int = 25) -> Tuple[int, Any, Optional[str]]:
    try:
        r = requests.get(url, timeout=timeout)
        try:
            payload = r.json()
        except Exception:
            payload = r.text[:1000]
        return r.status_code, payload, None
    except Exception as exc:
        return 0, None, str(exc)


def build_url(path: str, params: Dict[str, Any]) -> str:
    return f"{BASE}{path}?{urlencode(params)}"


def mask_url(url: str) -> str:
    return re.sub(r"apiKey=[^&]+", "apiKey=***", url)


def fetch_oddsapiio_k_lines(max_events: int = 25, include_started: bool = False) -> Dict[str, Any]:
    key = os.environ.get("ODDS_API_IO_KEY", "").strip()
    out: Dict[str, Any] = {
        "provider": "oddsapiio",
        "has_key": bool(key),
        "sport": SPORT,
        "league": LEAGUE,
        "bookmaker": BOOKMAKER,
        "request_count": 0,
        "requests": [],
        "events_total": 0,
        "events_used": 0,
        "k_lines": [],
        "error": None,
    }
    if not key:
        out["error"] = "Missing ODDS_API_IO_KEY"
        return out

    events_url = build_url("/events", {
        "sport": SPORT,
        "league": LEAGUE,
        "bookmaker": BOOKMAKER,
        "apiKey": key,
    })
    status, payload, err = request_json(events_url)
    out["request_count"] += 1
    out["requests"].append({"endpoint": "events", "status": status, "shape": safe_shape(payload), "url": mask_url(events_url), "error": err})
    if err or status != 200 or not isinstance(payload, list):
        out["error"] = f"events_failed status={status} error={err} shape={safe_shape(payload)}"
        return out

    events = payload
    out["events_total"] = len(events)
    filtered = []
    now = datetime.now(timezone.utc)
    for ev in events:
        if not isinstance(ev, dict):
            continue
        status_s = str(ev.get("status", "")).lower()
        date_s = ev.get("date") or ev.get("commence_time") or ev.get("startTime")
        is_future = True
        if date_s:
            try:
                ds = str(date_s).replace("Z", "+00:00")
                dt = datetime.fromisoformat(ds)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                is_future = dt >= now
            except Exception:
                is_future = True
        if include_started or status_s in {"pending", "scheduled", "pre-game", "pregame", "preview"} or is_future:
            filtered.append(ev)
    if not filtered:
        filtered = events
    filtered = filtered[:max_events]
    out["events_used"] = len(filtered)
    out["events_sample"] = [
        {
            "id": ev.get("id"),
            "away": ev.get("away"),
            "home": ev.get("home"),
            "date": ev.get("date"),
            "status": ev.get("status"),
        }
        for ev in filtered[:10]
    ]

    k_lines: List[Dict[str, Any]] = []
    market_names: Dict[str, int] = {}
    for ev in filtered:
        event_id = ev.get("id")
        if not event_id:
            continue
        odds_url = build_url("/odds", {
            "eventId": event_id,
            "bookmakers": BOOKMAKER,
            "apiKey": key,
        })
        status, odds_payload, err = request_json(odds_url)
        out["request_count"] += 1
        out["requests"].append({"endpoint": "odds", "event_id": event_id, "status": status, "shape": safe_shape(odds_payload), "url": mask_url(odds_url), "error": err})
        if err or status != 200 or not isinstance(odds_payload, dict):
            continue
        bms = odds_payload.get("bookmakers") or {}
        fd_markets = None
        if isinstance(bms, dict):
            fd_markets = bms.get(BOOKMAKER) or bms.get("fanduel") or bms.get("FanDuel")
        if not isinstance(fd_markets, list):
            continue
        for market in fd_markets:
            if not isinstance(market, dict):
                continue
            mname = str(market.get("name") or market.get("label") or market.get("key") or "").strip()
            if mname:
                market_names[mname] = market_names.get(mname, 0) + 1
            odds = market.get("odds") or market.get("outcomes") or []
            if not isinstance(odds, list):
                continue
            for prop in odds:
                if not isinstance(prop, dict):
                    continue
                label = str(prop.get("label") or prop.get("description") or prop.get("name") or "").strip()
                hay = f"{mname} {label}".lower()
                if not any(pat in hay for pat in K_MARKET_PATTERNS):
                    continue
                # Avoid game spread/totals handicap rows; require a player-looking label.
                if "total strikeouts" not in hay and "pitcher strikeouts" not in hay and not label:
                    continue
                player = strip_k_suffix(label)
                line = as_float(prop.get("hdp"))
                if line is None:
                    line = as_float(prop.get("point"))
                if line is None:
                    line = as_float(prop.get("line"))
                if line is None:
                    continue
                row = {
                    "provider": "oddsapiio",
                    "bookmaker": BOOKMAKER,
                    "event_id": event_id,
                    "event": f"{ev.get('away')} @ {ev.get('home')}",
                    "away": ev.get("away"),
                    "home": ev.get("home"),
                    "date": ev.get("date"),
                    "status": ev.get("status"),
                    "market": mname,
                    "raw_label": label,
                    "player": player,
                    "player_norm": norm_name(player),
                    "line": line,
                    "over": prop.get("over"),
                    "under": prop.get("under"),
                    "raw": prop,
                }
                k_lines.append(row)
    out["market_names_seen"] = dict(sorted(market_names.items(), key=lambda kv: (-kv[1], kv[0])))
    out["k_lines"] = k_lines
    out["k_lines_count"] = len(k_lines)
    return out


def latest_file(patterns: List[str]) -> Optional[str]:
    candidates: List[str] = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: os.path.getmtime(p))


def flatten_candidates(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ["candidates", "items", "rows", "data", "predictions", "picks", "latest"]:
        v = payload.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    # Some logs may be grouped by status/market. Recursively collect dicts with likely fields.
    found: List[Dict[str, Any]] = []
    def walk(x: Any):
        if isinstance(x, list):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            if any(k in x for k in PLAYER_KEYS) and any(k in x for k in PROJECTION_KEYS + MARKET_KEYS):
                found.append(x)
            else:
                for y in x.values():
                    walk(y)
    walk(payload)
    return found


def read_latest_candidates(path: Optional[str] = None) -> Dict[str, Any]:
    if not path:
        path = latest_file([
            "/data/candidate_logs/candidates_*.json",
            "/data/candidate_logs/*candidate*.json",
            "/data/predictions/*candidate*.json",
            "/data/predictions/predictions_*.json",
        ])
    out = {"path": path, "rows": [], "error": None}
    if not path:
        out["error"] = "No candidate/prediction json file found under /data"
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        out["rows"] = flatten_candidates(payload)
    except Exception as exc:
        out["error"] = str(exc)
    return out


def row_market(row: Dict[str, Any]) -> str:
    vals = []
    for k in MARKET_KEYS:
        v = row.get(k)
        if v is not None:
            vals.append(str(v))
    return " ".join(vals).lower()


def is_k_candidate(row: Dict[str, Any]) -> bool:
    m = row_market(row)
    if "pitcher_strikeouts" in m or "pitcher strikeouts" in m:
        return True
    if "strikeout" in m and "pitch" in m:
        return True
    # If a row has K-specific projection keys, count it.
    if any(k in row for k in ["k_pitcher_projection", "k_lineup_exp_ks", "recent_k_avg"]):
        return True
    return False


def get_player(row: Dict[str, Any]) -> str:
    for k in PLAYER_KEYS:
        v = row.get(k)
        if v:
            return str(v)
    # nested possibilities
    for path in [("pitcher", "name"), ("player", "name")]:
        v = get_nested(row, path)
        if v:
            return str(v)
    return ""


def get_projection(row: Dict[str, Any]) -> Optional[float]:
    for k in PROJECTION_KEYS:
        if k in row:
            v = as_float(row.get(k))
            if v is not None:
                return v
    # nested fallback
    for path in [("signals", "k_pitcher_projection"), ("features", "k_pitcher_projection")]:
        v = as_float(get_nested(row, path))
        if v is not None:
            return v
    return None


def match_line(player: str, k_lines: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str, float]:
    pn = norm_name(player)
    if not pn:
        return None, "no_player_name", 0.0
    # exact normalized
    exact = [ln for ln in k_lines if ln.get("player_norm") == pn]
    if len(exact) == 1:
        return exact[0], "exact_norm", 1.0
    if len(exact) > 1:
        return exact[0], "exact_norm_multiple", 1.0

    # unique last-name match
    parts = pn.split()
    if parts:
        last = parts[-1]
        last_matches = [ln for ln in k_lines if (ln.get("player_norm", "").split() or [""])[-1] == last]
        if len(last_matches) == 1:
            return last_matches[0], "unique_last_name", 0.86

    # fuzzy fallback
    best = None
    best_score = 0.0
    for ln in k_lines:
        score = SequenceMatcher(None, pn, ln.get("player_norm", "")).ratio()
        if score > best_score:
            best_score = score
            best = ln
    if best and best_score >= 0.84:
        return best, "fuzzy", best_score
    return None, "no_match", best_score


def simulate_k_mc(mu: float, line: float, sim_n: int, seed: int, dispersion: float = 1.15) -> Dict[str, Any]:
    mu = max(0.01, float(mu))
    line = float(line)
    if np is not None:
        rng = np.random.default_rng(seed)
        if dispersion and dispersion > 1.0001:
            var = max(mu + 1e-6, mu * dispersion)
            # Negative binomial parameterization: mean = n(1-p)/p, variance = n(1-p)/p^2
            p = min(0.999999, max(1e-6, mu / var))
            n = max(1e-6, (mu * p) / (1.0 - p))
            sims = rng.negative_binomial(n, p, size=sim_n)
        else:
            sims = rng.poisson(mu, size=sim_n)
        over = float((sims > line).mean())
        under = float((sims < line).mean())
        push = float((sims == line).mean())
        return {
            "sim_n": sim_n,
            "sim_mean": float(sims.mean()),
            "sim_median": float(np.median(sims)),
            "sim_p10": float(np.percentile(sims, 10)),
            "sim_p90": float(np.percentile(sims, 90)),
            "mc_prob_over": over,
            "mc_prob_under": under,
            "mc_prob_push": push,
        }
    # pure python fallback
    random.seed(seed)
    over_n = under_n = push_n = 0
    vals = []
    # Knuth Poisson, acceptable for this diagnostic fallback
    for _ in range(sim_n):
        L = math.exp(-mu)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= random.random()
        val = k - 1
        vals.append(val)
        if val > line:
            over_n += 1
        elif val < line:
            under_n += 1
        else:
            push_n += 1
    vals.sort()
    return {
        "sim_n": sim_n,
        "sim_mean": sum(vals) / len(vals),
        "sim_median": vals[len(vals)//2],
        "sim_p10": vals[int(0.10 * (len(vals)-1))],
        "sim_p90": vals[int(0.90 * (len(vals)-1))],
        "mc_prob_over": over_n / sim_n,
        "mc_prob_under": under_n / sim_n,
        "mc_prob_push": push_n / sim_n,
    }


def old_side(row: Dict[str, Any]) -> str:
    for k in ["side", "pick_side", "direction", "selection", "bet_side"]:
        v = row.get(k)
        if v:
            s = str(v).upper()
            if "OVER" in s:
                return "OVER"
            if "UNDER" in s:
                return "UNDER"
    return "UNKNOWN"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-events", type=int, default=25)
    ap.add_argument("--sim-n", type=int, default=250000)
    ap.add_argument("--min-mc-prob", type=float, default=0.63)
    ap.add_argument("--candidate-file", default=None)
    ap.add_argument("--include-started", action="store_true")
    ap.add_argument("--dispersion", type=float, default=1.15, help="K count overdispersion multiplier; 1.0 = Poisson")
    ap.add_argument("--seed", type=int, default=81718)
    args = ap.parse_args()

    provider = fetch_oddsapiio_k_lines(max_events=args.max_events, include_started=args.include_started)
    cand = read_latest_candidates(args.candidate_file)
    rows = cand.get("rows") or []
    k_rows = [r for r in rows if is_k_candidate(r)]
    k_lines = provider.get("k_lines") or []

    matched: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    used_line_ids = set()

    for idx, row in enumerate(k_rows):
        player = get_player(row)
        proj = get_projection(row)
        line_row, method, score = match_line(player, k_lines)
        base = {
            "candidate_index": idx,
            "player": player,
            "player_norm": norm_name(player),
            "projection": proj,
            "status": row.get("status"),
            "reject_reason": row.get("reject_reason") or row.get("reason"),
            "old_side": old_side(row),
            "match_method": method,
            "match_score": score,
        }
        if not line_row or proj is None:
            base["unmatched_reason"] = "missing_line" if not line_row else "missing_projection"
            unmatched.append(base)
            continue
        line = as_float(line_row.get("line"))
        if line is None:
            base["unmatched_reason"] = "line_not_numeric"
            unmatched.append(base)
            continue
        gap = float(proj) - float(line)
        sim = simulate_k_mc(float(proj), float(line), args.sim_n, args.seed + idx, args.dispersion)
        mc_over = sim["mc_prob_over"]
        mc_under = sim["mc_prob_under"]
        if gap > 0:
            final_side = "OVER"
            final_prob = mc_over
        elif gap < 0:
            final_side = "UNDER"
            final_prob = mc_under
        else:
            final_side = "OVER" if mc_over >= mc_under else "UNDER"
            final_prob = max(mc_over, mc_under)
        side_flip = base["old_side"] in {"OVER", "UNDER"} and base["old_side"] != final_side
        keep = final_prob >= args.min_mc_prob
        line_id = f"{line_row.get('event_id')}|{line_row.get('player_norm')}|{line_row.get('line')}"
        used_line_ids.add(line_id)
        rec = {
            **base,
            "line": line,
            "provider_player": line_row.get("player"),
            "provider_event": line_row.get("event"),
            "provider_date": line_row.get("date"),
            "provider_over": line_row.get("over"),
            "provider_under": line_row.get("under"),
            "provider_market": line_row.get("market"),
            "provider_raw_label": line_row.get("raw_label"),
            "projection_gap": gap,
            **sim,
            "final_side": final_side,
            "final_mc_prob": final_prob,
            "would_keep": bool(keep),
            "side_flip": bool(side_flip),
        }
        matched.append(rec)

    unused_lines = []
    for ln in k_lines:
        line_id = f"{ln.get('event_id')}|{ln.get('player_norm')}|{ln.get('line')}"
        if line_id not in used_line_ids:
            unused_lines.append({k: ln.get(k) for k in ["player", "line", "event", "date", "over", "under", "market"]})

    summary = {
        "generated_at_utc": now_utc(),
        "script": "k_mc_match_test.py",
        "sim_n": args.sim_n,
        "min_mc_prob": args.min_mc_prob,
        "dispersion": args.dispersion,
        "candidate_file": cand.get("path"),
        "candidate_rows_total": len(rows),
        "k_candidates_total": len(k_rows),
        "provider": {
            "name": provider.get("provider"),
            "has_key": provider.get("has_key"),
            "request_count": provider.get("request_count"),
            "events_total": provider.get("events_total"),
            "events_used": provider.get("events_used"),
            "k_lines_count": len(k_lines),
            "error": provider.get("error"),
            "market_names_seen": provider.get("market_names_seen"),
            "events_sample": provider.get("events_sample"),
            "requests": provider.get("requests"),
        },
        "match_summary": {
            "matched": len(matched),
            "unmatched": len(unmatched),
            "unused_provider_lines": len(unused_lines),
            "simulated": len(matched),
            "would_keep": sum(1 for r in matched if r.get("would_keep")),
            "would_drop": sum(1 for r in matched if not r.get("would_keep")),
            "side_flips": sum(1 for r in matched if r.get("side_flip")),
        },
        "matched": sorted(matched, key=lambda r: r.get("final_mc_prob", 0), reverse=True),
        "unmatched": unmatched,
        "unused_provider_lines": unused_lines,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ALT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "k_mc_match_test_latest.json"
    csv_path = OUT_DIR / "k_mc_match_test_latest.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(ALT_OUT_DIR / "k_mc_match_test_latest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_fields = [
        "player", "provider_player", "provider_event", "projection", "line", "projection_gap",
        "final_side", "final_mc_prob", "mc_prob_over", "mc_prob_under", "would_keep",
        "old_side", "side_flip", "match_method", "match_score", "provider_over", "provider_under",
        "status", "reject_reason",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in summary["matched"]:
            w.writerow({k: r.get(k) for k in csv_fields})

    print("=== K MC MATCH TEST ===")
    print(f"Generated UTC: {summary['generated_at_utc']}")
    print(f"Candidate file: {summary['candidate_file']}")
    print(f"Simulations per matched K candidate: {args.sim_n:,}")
    print(f"MC keep threshold: {args.min_mc_prob:.1%}")
    print()
    print("PROVIDER LINES")
    print(f"provider=Odds-API.io has_key={provider.get('has_key')} requests={provider.get('request_count')} events_total={provider.get('events_total')} events_used={provider.get('events_used')} k_lines={len(k_lines)} error={provider.get('error')}")
    for ln in k_lines[:12]:
        print(f"- {ln.get('player')} | line={ln.get('line')} | over={ln.get('over')} under={ln.get('under')} | {ln.get('event')} | {ln.get('date')}")
    if not k_lines:
        print("No provider K lines found.")
    print()
    print("MATCH SUMMARY")
    ms = summary["match_summary"]
    print(f"k_candidates={len(k_rows)} matched={ms['matched']} unmatched={ms['unmatched']} unused_provider_lines={ms['unused_provider_lines']}")
    print(f"simulated={ms['simulated']} would_keep={ms['would_keep']} would_drop={ms['would_drop']} side_flips={ms['side_flips']}")
    print()
    print("TOP MATCHED K MC")
    for i, r in enumerate(summary["matched"][:30], 1):
        print(
            f"{i:02d}. {r.get('player')} | proj={r.get('projection'):.3f} line={r.get('line')} gap={r.get('projection_gap'):+.3f} "
            f"MC {r.get('final_side')}={r.get('final_mc_prob'):.1%} (O={r.get('mc_prob_over'):.1%}, U={r.get('mc_prob_under'):.1%}) "
            f"keep={r.get('would_keep')} match={r.get('match_method')} event={r.get('provider_event')} old_side={r.get('old_side')} flip={r.get('side_flip')}"
        )
    if not summary["matched"]:
        print("No matched K candidates were simulated.")
    print()
    print("UNMATCHED K CANDIDATES")
    for r in unmatched[:30]:
        print(f"- {r.get('player')} | proj={r.get('projection')} | reason={r.get('unmatched_reason')} | match={r.get('match_method')} score={r.get('match_score'):.2f} | reject={r.get('reject_reason')}")
    if not unmatched:
        print("None")
    print()
    print("UNUSED PROVIDER LINES")
    for ln in unused_lines[:20]:
        print(f"- {ln.get('player')} | line={ln.get('line')} | {ln.get('event')} | {ln.get('date')}")
    if not unused_lines:
        print("None")
    print()
    print("OUTPUTS")
    print(str(json_path))
    print(str(csv_path))
    print()
    print("NOTE: Standalone diagnostic only. It does not change api.py, predictions, record.json, or PropLine usage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
