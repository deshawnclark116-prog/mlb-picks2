#!/usr/bin/env python3
"""
k_mc_match_test_v2.py
Standalone diagnostic only. Does not change api.py, predictions, record.json, or PropLine usage.

Purpose:
- Pull FanDuel MLB pitcher strikeout lines from Odds-API.io using the documented
  single-event player-props endpoint.
- Read the latest candidate log/snapshot from /data/candidate_logs.
- Match provider lines to pitcher_strikeouts candidates using robust player-name matching.
- Run 250,000 Monte Carlo simulations for matched K candidates.

Env required:
- ODDS_API_IO_KEY

Run:
  python k_mc_match_test_v2.py --max-events 25
  python k_mc_match_test_v2.py --candidate-file /data/candidate_logs/latest_candidate_snapshot.json --max-events 25
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except Exception as e:
    print("ERROR: requests is required:", e)
    sys.exit(1)

try:
    import numpy as np
except Exception:
    np = None

BASE = "https://api.odds-api.io/v3"
DATA_DIR = Path("/data/candidate_logs")
OUT_JSON = DATA_DIR / "k_mc_match_test_v2_latest.json"
OUT_CSV = DATA_DIR / "k_mc_match_test_v2_latest.csv"
SIM_N_DEFAULT = 250_000
MC_KEEP_THRESHOLD_DEFAULT = 0.63


# ------------------------- basic helpers -------------------------

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    try:
        if isinstance(x, str):
            x = x.strip().replace("+", "")
            if x.upper() in {"", "NA", "N/A", "NONE", "NULL"}:
                return default
        return float(x)
    except Exception:
        return default


def shape(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, list):
        d = {"type": "list", "len": len(obj)}
        if obj:
            d["first_type"] = type(obj[0]).__name__
            if isinstance(obj[0], dict):
                d["first_keys"] = list(obj[0].keys())[:12]
        return d
    if isinstance(obj, dict):
        return {"type": "dict", "len": len(obj), "first_keys": list(obj.keys())[:12]}
    return {"type": type(obj).__name__}


def http_get_json(url: str, params: Dict[str, Any], requests_log: List[Dict[str, Any]]) -> Any:
    redacted_params = dict(params)
    if "apiKey" in redacted_params:
        redacted_params["apiKey"] = "***"
    try:
        r = requests.get(url, params=params, timeout=25)
        ctype = r.headers.get("content-type", "")
        try:
            data = r.json()
        except Exception:
            data = {"_text_preview": r.text[:1000]}
        requests_log.append({
            "status": r.status_code,
            "endpoint": url.replace(BASE, ""),
            "params": redacted_params,
            "ok": r.ok,
            "content_type": ctype,
            "shape": shape(data),
            "error_preview": data.get("error") if isinstance(data, dict) else None,
        })
        return data
    except Exception as e:
        requests_log.append({
            "status": None,
            "endpoint": url.replace(BASE, ""),
            "params": redacted_params,
            "ok": False,
            "exception": repr(e),
        })
        return {"error": repr(e)}


# ------------------------- normalization/matching -------------------------

SUFFIX_RE = re.compile(r"\s*\((?:total\s+)?strikeouts?\)\s*$", re.I)
JUNK_RE = re.compile(r"\b(jr|sr|ii|iii|iv)\b\.?", re.I)


def normalize_name(name: Any) -> str:
    if name is None:
        return ""
    s = str(name)
    s = SUFFIX_RE.sub("", s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = JUNK_RE.sub("", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def name_tokens(name: str) -> List[str]:
    return [t for t in normalize_name(name).split() if t]


def match_score(candidate_name: str, provider_name: str) -> float:
    cn = normalize_name(candidate_name)
    pn = normalize_name(provider_name)
    if not cn or not pn:
        return 0.0
    if cn == pn:
        return 1.0

    ct = name_tokens(cn)
    pt = name_tokens(pn)
    if not ct or not pt:
        return 0.0

    # token set exact, e.g. "yamamoto yoshinobu" vs "yoshinobu yamamoto"
    if sorted(ct) == sorted(pt):
        return 0.97

    # Last name + first initial match. Good for most pitcher-name variations.
    if len(ct) >= 2 and len(pt) >= 2:
        if ct[-1] == pt[-1] and ct[0][0] == pt[0][0]:
            return 0.92

    # Last name exact and one side has shortened first name.
    if ct[-1] == pt[-1]:
        return max(0.78, SequenceMatcher(None, cn, pn).ratio())

    return SequenceMatcher(None, cn, pn).ratio()


def find_best_provider_line(candidate_name: str, provider_lines: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float, str]:
    best = None
    best_score = 0.0
    for line in provider_lines:
        score = match_score(candidate_name, line.get("player", ""))
        if score > best_score:
            best_score = score
            best = line
    if best is None:
        return None, 0.0, "no_provider_lines"
    if best_score >= 0.90:
        return best, best_score, "exact_or_high_name_match"
    if best_score >= 0.78:
        return best, best_score, "last_name_or_fuzzy_match"
    return None, best_score, "no_match"


# ------------------------- candidate loading -------------------------

def iter_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from iter_dicts(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_dicts(item)


def get_any(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d.get(k) not in (None, ""):
            return d.get(k)
    return None


def looks_like_k_candidate(d: Dict[str, Any]) -> bool:
    hay = " ".join(str(get_any(d, [k]) or "") for k in [
        "market", "prop", "prop_type", "type", "pick_type", "stat", "name", "reason", "reject_reason"
    ]).lower()
    if "pitcher_strikeouts" in hay or "pitcher strikeout" in hay:
        return True
    # If it has clear K projection fields, count it too.
    if any(k in d for k in ["k_pitcher_projection", "pitcher_projection", "projected_ks", "k_projection"]):
        return True
    return False


def extract_candidate(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not looks_like_k_candidate(d):
        return None

    player = get_any(d, [
        "player", "player_name", "pitcher", "pitcher_name", "name", "display_name", "athlete", "candidate_name"
    ])
    if not player:
        return None

    proj = safe_float(get_any(d, [
        "k_pitcher_projection", "projected_ks", "k_projection", "pitcher_projection", "projection", "proj", "model_projection", "raw_projection"
    ]))
    if proj is None:
        # Sometimes nested under debug/details.
        for subkey in ["debug", "details", "signals", "extra", "features"]:
            sub = d.get(subkey)
            if isinstance(sub, dict):
                proj = safe_float(get_any(sub, [
                    "k_pitcher_projection", "projected_ks", "k_projection", "pitcher_projection", "projection", "proj"
                ]))
                if proj is not None:
                    break

    return {
        "player": str(player),
        "norm_player": normalize_name(player),
        "projection": proj,
        "status": get_any(d, ["status", "candidate_status"]),
        "reason": get_any(d, ["reason", "reject_reason", "gate_reason"]),
        "raw": d,
    }


def latest_candidate_file() -> Optional[Path]:
    patterns = [
        "/data/candidate_logs/latest_candidate_snapshot.json",
        "/data/candidate_logs/candidates_*.json",
        "/data/candidate_logs/*candidate*.json",
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = [f for f in set(files) if not f.endswith("_latest.json") or "k_mc_match" not in f]
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(files[0])


def load_k_candidates(candidate_file: Optional[str]) -> Tuple[Path, List[Dict[str, Any]], Any]:
    p = Path(candidate_file) if candidate_file else latest_candidate_file()
    if not p:
        raise FileNotFoundError("No candidate file found under /data/candidate_logs")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    seen = set()
    candidates: List[Dict[str, Any]] = []
    for d in iter_dicts(data):
        c = extract_candidate(d)
        if not c:
            continue
        key = (c["norm_player"], c.get("projection"), str(c.get("reason")))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(c)
    return p, candidates, data


# ------------------------- Odds-API.io provider -------------------------

def strip_total_strikeouts(label: str) -> str:
    return SUFFIX_RE.sub("", str(label)).strip()


def fetch_oddsapiio_k_lines(max_events: int, api_key: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    requests_log: List[Dict[str, Any]] = []

    events = http_get_json(
        f"{BASE}/events",
        {"sport": "baseball", "league": "usa-mlb", "bookmaker": "FanDuel", "apiKey": api_key},
        requests_log,
    )
    if not isinstance(events, list):
        return [], {"error": "events_not_list", "events_shape": shape(events)}, requests_log

    now = dt.datetime.now(dt.timezone.utc)
    pending: List[Dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        status = str(ev.get("status", "")).lower()
        date_s = ev.get("date") or ev.get("commence_time")
        futureish = True
        if date_s:
            try:
                ev_dt = dt.datetime.fromisoformat(str(date_s).replace("Z", "+00:00"))
                futureish = ev_dt >= (now - dt.timedelta(minutes=30))
            except Exception:
                futureish = True
        if status in {"pending", "pre-game", "pregame", "scheduled", "preview", "not_started", "not started"} or futureish:
            pending.append(ev)

    used = pending[:max_events]
    lines: List[Dict[str, Any]] = []
    market_names: Dict[str, int] = {}
    k_hits = 0

    for ev in used:
        event_id = ev.get("id")
        if not event_id:
            continue
        odds = http_get_json(
            f"{BASE}/odds",
            {"eventId": event_id, "bookmakers": "FanDuel", "apiKey": api_key},
            requests_log,
        )
        if not isinstance(odds, dict):
            continue
        books = odds.get("bookmakers") or {}
        fd = None
        if isinstance(books, dict):
            fd = books.get("FanDuel") or books.get("fanduel")
        if not isinstance(fd, list):
            continue
        for market in fd:
            if not isinstance(market, dict):
                continue
            mname = str(market.get("name") or market.get("label") or "")
            if mname:
                market_names[mname] = market_names.get(mname, 0) + 1
            odds_rows = market.get("odds")
            if not isinstance(odds_rows, list):
                continue
            for row in odds_rows:
                if not isinstance(row, dict):
                    continue
                label = str(row.get("label") or row.get("name") or "")
                label_l = label.lower()
                mname_l = mname.lower()
                if "strikeout" not in label_l and "strikeout" not in mname_l:
                    continue
                # We only want pitcher total strikeouts labels, not team/game text.
                if "total strikeout" not in label_l and "strikeout" not in label_l:
                    continue
                line = safe_float(row.get("hdp") if "hdp" in row else row.get("point"))
                if line is None:
                    continue
                k_hits += 1
                lines.append({
                    "provider": "oddsapiio",
                    "book": "FanDuel",
                    "event_id": event_id,
                    "event": f"{ev.get('away')} @ {ev.get('home')}",
                    "home": ev.get("home"),
                    "away": ev.get("away"),
                    "date": ev.get("date"),
                    "status": ev.get("status"),
                    "market": mname,
                    "label": label,
                    "player": strip_total_strikeouts(label),
                    "line": line,
                    "over": safe_float(row.get("over"), None),
                    "under": safe_float(row.get("under"), None),
                    "raw": row,
                })

    # Deduplicate same provider/player/event/line.
    dedup = []
    seen = set()
    for line in lines:
        key = (normalize_name(line["player"]), line.get("event_id"), line.get("line"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(line)

    meta = {
        "events_total": len(events),
        "pending_or_future": len(pending),
        "events_used": len(used),
        "k_hits_raw": k_hits,
        "k_lines": len(dedup),
        "market_names_seen": market_names,
        "events_used_sample": [
            {"id": e.get("id"), "event": f"{e.get('away')} @ {e.get('home')}", "date": e.get("date"), "status": e.get("status")}
            for e in used[:12]
        ],
    }
    return dedup, meta, requests_log


# ------------------------- Monte Carlo -------------------------

def simulate_k(projection: float, line: float, n: int) -> Dict[str, Any]:
    # Count-stat simulation. Uses Poisson centered on projection. We keep this deliberately
    # conservative/simple for the standalone matcher; live v8.18 can use the richer ksim layer.
    lam = max(0.05, float(projection))
    if np is not None:
        rng = np.random.default_rng(20260705)
        sims = rng.poisson(lam, size=n)
        prob_over = float(np.mean(sims > line))
        prob_under = float(np.mean(sims < line))
        mean = float(np.mean(sims))
        p10 = float(np.percentile(sims, 10))
        p50 = float(np.percentile(sims, 50))
        p90 = float(np.percentile(sims, 90))
    else:
        # Knuth Poisson fallback. Slower, but okay for small matched counts.
        import random
        over = under = 0
        vals = []
        L = math.exp(-lam)
        for _ in range(n):
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= random.random()
            x = k - 1
            vals.append(x)
            if x > line:
                over += 1
            if x < line:
                under += 1
        vals.sort()
        prob_over = over / n
        prob_under = under / n
        mean = sum(vals) / n
        p10 = vals[int(0.10 * (n - 1))]
        p50 = vals[int(0.50 * (n - 1))]
        p90 = vals[int(0.90 * (n - 1))]

    side = "OVER" if projection > line else "UNDER"
    side_prob = prob_over if side == "OVER" else prob_under
    return {
        "sim_n": n,
        "sim_mean": round(mean, 4),
        "sim_p10": p10,
        "sim_p50": p50,
        "sim_p90": p90,
        "mc_prob_over": prob_over,
        "mc_prob_under": prob_under,
        "mc_side": side,
        "mc_side_prob": side_prob,
    }


# ------------------------- main -------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-events", type=int, default=25)
    ap.add_argument("--candidate-file", default=None)
    ap.add_argument("--sim-n", type=int, default=SIM_N_DEFAULT)
    ap.add_argument("--mc-keep-threshold", type=float, default=MC_KEEP_THRESHOLD_DEFAULT)
    ap.add_argument("--match-threshold", type=float, default=0.78)
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api_key = os.getenv("ODDS_API_IO_KEY") or os.getenv("ODDSAPIIO_KEY") or ""

    print("=== K MC MATCH TEST V2 ===")
    print(f"Generated UTC: {utc_now()}")
    print("Provider: Odds-API.io single-event FanDuel Player Props")
    print(f"Simulations per matched K candidate: {args.sim_n:,}")
    print(f"MC keep threshold: {args.mc_keep_threshold * 100:.1f}%")
    print("Matching: exact normalized full name first; game/date not required for standalone test")

    try:
        candidate_path, k_candidates, _raw_candidates = load_k_candidates(args.candidate_file)
    except Exception as e:
        print(f"ERROR loading candidates: {e}")
        return 2

    print(f"Candidate file: {candidate_path}")
    print(f"K candidates found: {len(k_candidates)}")

    provider_lines: List[Dict[str, Any]] = []
    provider_meta: Dict[str, Any] = {}
    requests_log: List[Dict[str, Any]] = []
    error = None
    if not api_key:
        error = "missing ODDS_API_IO_KEY"
    else:
        provider_lines, provider_meta, requests_log = fetch_oddsapiio_k_lines(args.max_events, api_key)

    matched: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    used_provider_keys = set()

    for cand in k_candidates:
        line, score, reason = find_best_provider_line(cand["player"], provider_lines)
        if line and score >= args.match_threshold:
            proj = cand.get("projection")
            if proj is None:
                row = {
                    "player": cand["player"],
                    "projection": None,
                    "match_player": line.get("player"),
                    "line": line.get("line"),
                    "match_score": score,
                    "match_reason": reason,
                    "status": "matched_no_projection",
                }
            else:
                sim = simulate_k(float(proj), float(line["line"]), args.sim_n)
                gap = float(proj) - float(line["line"])
                keep = sim["mc_side_prob"] >= args.mc_keep_threshold
                row = {
                    "player": cand["player"],
                    "projection": float(proj),
                    "provider_player": line.get("player"),
                    "line": float(line["line"]),
                    "event": line.get("event"),
                    "date": line.get("date"),
                    "over_price": line.get("over"),
                    "under_price": line.get("under"),
                    "gap": gap,
                    "match_score": score,
                    "match_reason": reason,
                    "candidate_status": cand.get("status"),
                    "candidate_reason": cand.get("reason"),
                    "would_keep": keep,
                    **sim,
                }
            matched.append(row)
            used_provider_keys.add((normalize_name(line.get("player")), line.get("event_id"), line.get("line")))
        else:
            unmatched.append({
                "player": cand["player"],
                "projection": cand.get("projection"),
                "best_score": score,
                "match_reason": reason,
                "candidate_status": cand.get("status"),
                "candidate_reason": cand.get("reason"),
            })

    unused = []
    for line in provider_lines:
        key = (normalize_name(line.get("player")), line.get("event_id"), line.get("line"))
        if key not in used_provider_keys:
            unused.append(line)

    simulated = [r for r in matched if r.get("projection") is not None and "mc_side_prob" in r]
    would_keep = [r for r in simulated if r.get("would_keep")]
    would_drop = [r for r in simulated if not r.get("would_keep")]
    side_flips = 0  # Standalone chooses side from projection-vs-line; no old side is reliably available.

    result = {
        "generated_at_utc": utc_now(),
        "candidate_file": str(candidate_path),
        "provider": "oddsapiio",
        "has_key": bool(api_key),
        "error": error,
        "sim_n": args.sim_n,
        "mc_keep_threshold": args.mc_keep_threshold,
        "provider_meta": provider_meta,
        "provider_lines_count": len(provider_lines),
        "provider_lines": provider_lines,
        "requests": requests_log,
        "summary": {
            "k_candidates": len(k_candidates),
            "matched": len(matched),
            "unmatched": len(unmatched),
            "simulated": len(simulated),
            "would_keep": len(would_keep),
            "would_drop": len(would_drop),
            "unused_provider_lines": len(unused),
            "side_flips": side_flips,
        },
        "matched": matched,
        "unmatched": unmatched,
        "unused_provider_lines": unused,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "player", "projection", "provider_player", "line", "gap", "mc_side", "mc_side_prob",
            "mc_prob_over", "mc_prob_under", "would_keep", "match_score", "event", "date",
            "over_price", "under_price", "candidate_status", "candidate_reason"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in matched:
            w.writerow({k: row.get(k) for k in fieldnames})

    # ------------------------- print report -------------------------
    print("\nPROVIDER LINES")
    print(f"provider=Odds-API.io has_key={bool(api_key)} requests={len(requests_log)} events_total={provider_meta.get('events_total')} events_used={provider_meta.get('events_used')} k_lines={len(provider_lines)} error={error}")
    for line in provider_lines[:20]:
        print(f"- {line['player']} | line={line['line']} | over={line.get('over')} under={line.get('under')} | {line.get('event')} | {line.get('date')}")

    print("\nMATCH SUMMARY")
    s = result["summary"]
    print(f"k_candidates={s['k_candidates']} matched={s['matched']} unmatched={s['unmatched']} unused_provider_lines={s['unused_provider_lines']}")
    print(f"simulated={s['simulated']} would_keep={s['would_keep']} would_drop={s['would_drop']} side_flips={s['side_flips']}")

    print("\nTOP MATCHED K MC")
    if not simulated:
        print("No matched K candidates were simulated.")
    else:
        top = sorted(simulated, key=lambda r: r.get("mc_side_prob", 0), reverse=True)[:30]
        for i, r in enumerate(top, 1):
            print(
                f"{i:02d}. {r['player']} -> {r.get('provider_player')} | proj={r['projection']:.3f} line={r['line']} gap={r['gap']:+.3f} "
                f"| MC {r['mc_side']}={r['mc_side_prob']*100:.1f}% "
                f"(O={r['mc_prob_over']*100:.1f}%, U={r['mc_prob_under']*100:.1f}%) "
                f"| keep={r['would_keep']} | match={r['match_score']:.2f} | {r.get('event')}"
            )

    print("\nUNMATCHED K CANDIDATES")
    for r in unmatched[:40]:
        proj = r.get("projection")
        proj_s = "NA" if proj is None else f"{float(proj):.3f}"
        print(f"- {r['player']} | proj={proj_s} | match={r['match_reason']} score={r['best_score']:.2f} | reason={r.get('candidate_reason')}")

    print("\nUNUSED PROVIDER LINES")
    for line in unused[:40]:
        print(f"- {line['player']} | line={line['line']} | {line.get('event')} | {line.get('date')}")

    print("\nREQUESTS")
    for req in requests_log[:40]:
        status = req.get("status")
        ep = req.get("endpoint")
        ok = req.get("ok")
        err = req.get("error_preview")
        print(f"- {status} {ep} ok={ok} shape={req.get('shape')} error={err}")

    print("\nOUTPUTS")
    print(str(OUT_JSON))
    print(str(OUT_CSV))
    print("\nNOTE: Standalone diagnostic only. It does not change api.py, predictions, record.json, or PropLine usage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
