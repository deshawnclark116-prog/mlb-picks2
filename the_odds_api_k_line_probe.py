#!/usr/bin/env python3
"""
Standalone The Odds API MLB FanDuel pitcher strikeout line probe.

Purpose:
  Test whether The Odds API can return FanDuel main MLB pitcher strikeout lines.

Does NOT touch api.py, predictions, records, or PropLine.

Env vars accepted:
  THE_ODDS_API_KEY
  ODDS_API_KEY
  THEODDSAPI_KEY

Example:
  python the_odds_api_k_line_probe.py --max-events 15
  python the_odds_api_k_line_probe.py --max-events 15 --dump-raw-sample
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

BASE = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
OUT_DIR = Path("/data/odds_provider_tests")
JSON_OUT = OUT_DIR / "the_odds_api_k_line_probe_latest.json"
CSV_OUT = OUT_DIR / "the_odds_api_k_line_probe_latest.csv"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def get_key() -> Tuple[Optional[str], Optional[str]]:
    for name in ("THE_ODDS_API_KEY", "ODDS_API_KEY", "THEODDSAPI_KEY"):
        val = os.getenv(name)
        if val:
            return val.strip(), name
    return None, None


def mask_url(url: str, key: Optional[str]) -> str:
    if key:
        url = url.replace(key, "***")
    return re.sub(r"apiKey=[^&]+", "apiKey=***", url)


def shape(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, list):
        out: Dict[str, Any] = {"type": "list", "len": len(obj)}
        if obj:
            out["first_type"] = type(obj[0]).__name__
            if isinstance(obj[0], dict):
                out["first_keys"] = list(obj[0].keys())[:12]
        return out
    if isinstance(obj, dict):
        return {"type": "dict", "len": len(obj), "first_keys": list(obj.keys())[:12]}
    return {"type": type(obj).__name__}


def safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"_non_json_text_preview": resp.text[:1000]}


def request_json(session: requests.Session, url: str, params: Dict[str, Any], key: Optional[str], requests_log: List[Dict[str, Any]], label: str, timeout: int = 25) -> Any:
    resp = session.get(url, params=params, timeout=timeout)
    payload = safe_json(resp)
    headers = {
        "x-requests-remaining": resp.headers.get("x-requests-remaining"),
        "x-requests-used": resp.headers.get("x-requests-used"),
        "x-requests-last": resp.headers.get("x-requests-last"),
    }
    requests_log.append({
        "label": label,
        "status_code": resp.status_code,
        "ok": resp.ok,
        "url": mask_url(resp.url, key),
        "shape": shape(payload),
        "headers": headers,
        "error_preview": payload if (not resp.ok and isinstance(payload, dict)) else None,
    })
    return payload


def parse_iso(dt: str) -> Optional[datetime]:
    if not dt:
        return None
    try:
        if dt.endswith("Z"):
            dt = dt[:-1] + "+00:00"
        return datetime.fromisoformat(dt)
    except Exception:
        return None


def is_future_event(event: Dict[str, Any], margin_hours: float = 0.25) -> bool:
    dt = parse_iso(str(event.get("commence_time") or ""))
    if not dt:
        return True
    return dt.timestamp() >= (now_utc().timestamp() - margin_hours * 3600)


def norm_book_key(book: Dict[str, Any]) -> str:
    return str(book.get("key") or book.get("title") or "").lower().strip()


def extract_k_lines(event_odds: Dict[str, Any], wanted_book: str = "fanduel") -> Tuple[List[Dict[str, Any]], Counter, Counter, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    book_counter: Counter = Counter()
    market_counter: Counter = Counter()
    raw_samples: List[Dict[str, Any]] = []

    event_id = event_odds.get("id")
    home_team = event_odds.get("home_team")
    away_team = event_odds.get("away_team")
    commence_time = event_odds.get("commence_time")

    for bm in event_odds.get("bookmakers") or []:
        book_key = norm_book_key(bm)
        book_title = bm.get("title") or bm.get("key")
        book_counter[book_key or str(book_title)] += 1
        if wanted_book and book_key != wanted_book.lower():
            continue

        for mkt in bm.get("markets") or []:
            market_key = str(mkt.get("key") or "")
            market_counter[market_key] += 1
            if market_key != "pitcher_strikeouts":
                continue

            raw_samples.append({
                "event_id": event_id,
                "bookmaker": book_title,
                "market_key": market_key,
                "market_last_update": mkt.get("last_update"),
                "sample_outcomes": (mkt.get("outcomes") or [])[:8],
            })

            grouped: Dict[Tuple[str, Any], Dict[str, Any]] = defaultdict(dict)
            for outcome in mkt.get("outcomes") or []:
                name = str(outcome.get("name") or "").strip()
                player = str(outcome.get("description") or outcome.get("player") or "").strip()
                point = outcome.get("point")
                if not player:
                    # Some providers place player in name; The Odds API usually uses description.
                    player = name
                key = (player, point)
                side = name.lower()
                entry = grouped[key]
                entry.update({
                    "provider": "the_odds_api",
                    "event_id": event_id,
                    "event": f"{away_team} @ {home_team}",
                    "away_team": away_team,
                    "home_team": home_team,
                    "commence_time": commence_time,
                    "bookmaker_key": bm.get("key"),
                    "bookmaker_title": book_title,
                    "market_key": market_key,
                    "market_last_update": mkt.get("last_update"),
                    "player": player,
                    "line": point,
                })
                if side == "over":
                    entry["over_price"] = outcome.get("price")
                    entry["over_raw"] = outcome
                elif side == "under":
                    entry["under_price"] = outcome.get("price")
                    entry["under_raw"] = outcome
                else:
                    entry.setdefault("other_outcomes", []).append(outcome)

            for row in grouped.values():
                # Need a player and a point to be useful as the main K line.
                if row.get("player") and row.get("line") is not None:
                    rows.append(row)

    return rows, book_counter, market_counter, raw_samples


def write_outputs(result: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = [
        "provider", "player", "line", "over_price", "under_price", "bookmaker_title", "market_key",
        "event", "commence_time", "event_id", "market_last_update",
    ]
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-events", type=int, default=15)
    ap.add_argument("--bookmaker", default="fanduel", help="The Odds API bookmaker key, default fanduel")
    ap.add_argument("--market", default="pitcher_strikeouts")
    ap.add_argument("--include-started", action="store_true", help="Do not filter out already-started events")
    ap.add_argument("--dump-raw-sample", action="store_true")
    args = ap.parse_args()

    key, key_name = get_key()
    generated = now_utc().isoformat()
    requests_log: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    book_counter_total: Counter = Counter()
    market_counter_total: Counter = Counter()
    raw_market_samples: List[Dict[str, Any]] = []

    if not key:
        result = {
            "probe_version": "the_odds_api_k_line_probe_v1",
            "generated_at_utc": generated,
            "has_key": False,
            "key_env_used": None,
            "error": "Missing API key. Set THE_ODDS_API_KEY or ODDS_API_KEY in Render env.",
        }
        write_outputs(result, [])
        print(json.dumps(result, indent=2))
        return 1

    session = requests.Session()
    events_url = f"{BASE}/sports/{SPORT}/events"
    events_payload = request_json(session, events_url, {"apiKey": key}, key, requests_log, "events")

    events: List[Dict[str, Any]] = []
    if isinstance(events_payload, list):
        events = [e for e in events_payload if isinstance(e, dict)]
    else:
        errors.append("Events response was not a list")

    future_events = events if args.include_started else [e for e in events if is_future_event(e)]
    selected_events = future_events[: max(args.max_events, 0)]

    for ev in selected_events:
        event_id = ev.get("id")
        if not event_id:
            continue
        odds_url = f"{BASE}/sports/{SPORT}/events/{event_id}/odds"
        # Use bookmaker-specific pull first. This should return only FanDuel if supported.
        params = {
            "apiKey": key,
            "bookmakers": args.bookmaker,
            "markets": args.market,
            "oddsFormat": "american",
        }
        payload = request_json(session, odds_url, params, key, requests_log, f"event_odds:{event_id}")
        if isinstance(payload, dict) and payload.get("bookmakers") is not None:
            event_rows, book_counter, market_counter, raw_samples = extract_k_lines(payload, args.bookmaker)
            rows.extend(event_rows)
            book_counter_total.update(book_counter)
            market_counter_total.update(market_counter)
            raw_market_samples.extend(raw_samples[:2])
        elif isinstance(payload, dict) and payload.get("message"):
            errors.append(f"{event_id}: {payload.get('message')}")
        elif isinstance(payload, dict) and payload.get("error"):
            errors.append(f"{event_id}: {payload.get('error')}")

    # De-dupe rows by event/player/line/book
    deduped: Dict[Tuple[Any, Any, Any, Any], Dict[str, Any]] = {}
    for r in rows:
        deduped[(r.get("event_id"), r.get("player"), r.get("line"), r.get("bookmaker_key"))] = r
    rows = list(deduped.values())
    rows.sort(key=lambda r: (str(r.get("commence_time") or ""), str(r.get("event") or ""), str(r.get("player") or "")))

    result = {
        "probe_version": "the_odds_api_k_line_probe_v1",
        "generated_at_utc": generated,
        "base": BASE,
        "sport": SPORT,
        "has_key": True,
        "key_env_used": key_name,
        "bookmaker_requested": args.bookmaker,
        "market_requested": args.market,
        "events_total": len(events),
        "future_events": len(future_events),
        "events_used": len(selected_events),
        "k_lines_count": len(rows),
        "bookmakers_seen": dict(book_counter_total),
        "markets_seen": dict(market_counter_total),
        "k_lines": rows,
        "events_sample": [
            {
                "id": e.get("id"),
                "away_team": e.get("away_team"),
                "home_team": e.get("home_team"),
                "commence_time": e.get("commence_time"),
            }
            for e in selected_events[:10]
        ],
        "requests": requests_log,
        "errors": errors[:30],
        "raw_market_samples": raw_market_samples[:5] if args.dump_raw_sample else [],
        "output_json": str(JSON_OUT),
        "output_csv": str(CSV_OUT),
    }
    write_outputs(result, rows)

    print("=== THE ODDS API K LINE PROBE ===")
    print(f"Generated UTC: {generated}")
    print("Target: FanDuel main MLB pitcher strikeouts line")
    print(f"has_key=True key_env={key_name} sport={SPORT} bookmaker={args.bookmaker} market={args.market}")
    print(f"request_count={len(requests_log)} events_total={len(events)} future_events={len(future_events)} events_used={len(selected_events)} k_lines_count={len(rows)}")
    print("\nEVENTS USED")
    for e in selected_events[:12]:
        print(f"- {e.get('id')} | {e.get('away_team')} @ {e.get('home_team')} | {e.get('commence_time')}")
    print("\nBOOKMAKERS SEEN")
    if book_counter_total:
        for k, v in book_counter_total.most_common(20):
            print(f"- {k}: {v}")
    else:
        print("None detected")
    print("\nMARKET KEYS SEEN")
    if market_counter_total:
        for k, v in market_counter_total.most_common(30):
            print(f"- {k}: {v}")
    else:
        print("None detected")
    print("\nSAMPLE K LINES")
    if rows:
        for r in rows[:30]:
            print(f"- {r.get('player')} | line={r.get('line')} | O={r.get('over_price')} U={r.get('under_price')} | {r.get('event')} | {r.get('commence_time')}")
    else:
        print("No K lines found.")
    if errors:
        print("\nERRORS")
        for err in errors[:10]:
            print(f"- {err}")
    print("\nREQUESTS")
    for req in requests_log:
        h = req.get("headers") or {}
        print(f"- {req.get('status_code')} {req.get('label')} shape={req.get('shape')} last={h.get('x-requests-last')} used={h.get('x-requests-used')} remaining={h.get('x-requests-remaining')} url={req.get('url')}")
    if args.dump_raw_sample and raw_market_samples:
        print("\nRAW MARKET SAMPLE")
        print(json.dumps(raw_market_samples[:2], indent=2)[:3000])
    print("\nOUTPUTS")
    print(JSON_OUT)
    print(CSV_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
