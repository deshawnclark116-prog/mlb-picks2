#!/usr/bin/env python3
"""
Standalone PropLine pitcher-K market probe.

Purpose:
- Does NOT modify api.py or any prediction files.
- Tests the PropLine events endpoint response shape.
- Tests FanDuel pitcher_strikeouts / pitcher_strikeouts_alternate market availability.
- Shows whether the problem is: auth/response shape/events/market coverage/parser format.

Run on Render Shell after adding this file to the repo:
    python propline_k_probe.py

Optional:
    python propline_k_probe.py --max-events 25
    python propline_k_probe.py --market-scan
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except Exception as exc:  # pragma: no cover
    print(json.dumps({"ok": False, "error": f"requests import failed: {exc}"}, indent=2))
    sys.exit(1)

API_KEY = os.environ.get("PROPLINE_API_KEY") or os.environ.get("PROPLINE_KEY") or ""
BASE = os.environ.get("PROPLINE_BASE", "https://api.prop-line.com/v1/sports/baseball_mlb").rstrip("/")
BOOKMAKER_TARGETS = {"fanduel", "fan duel", "fd"}
DEFAULT_MARKETS = ["pitcher_strikeouts", "pitcher_strikeouts_alternate"]
MARKET_SCAN = [
    "pitcher_strikeouts",
    "pitcher_strikeouts_alternate",
    "player_pitcher_strikeouts",
    "player_pitcher_strikeouts_alternate",
    "player_strikeouts",
    "player_strikeouts_alternate",
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "prop-edge-propline-k-probe/1.0"})


def redact(obj: Any) -> Any:
    """Redact secrets from strings recursively."""
    if isinstance(obj, str):
        s = obj
        if API_KEY:
            s = s.replace(API_KEY, "<REDACTED_API_KEY>")
        s = re.sub(r"(apiKey=)[^&\s]+", r"\1<REDACTED_API_KEY>", s, flags=re.I)
        s = re.sub(r"(api_key=)[^&\s]+", r"\1<REDACTED_API_KEY>", s, flags=re.I)
        s = re.sub(r"(key=)[^&\s]+", r"\1<REDACTED_API_KEY>", s, flags=re.I)
        return s
    if isinstance(obj, dict):
        return {k: ("<REDACTED>" if "key" in str(k).lower() else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_json_response(resp: requests.Response) -> Tuple[Optional[Any], str]:
    try:
        return resp.json(), "json"
    except Exception:
        return None, "text"


def response_shape(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list):
        return {"type": "list", "length": len(payload), "sample_item_keys": sorted(list(payload[0].keys()))[:50] if payload and isinstance(payload[0], dict) else None}
    if isinstance(payload, dict):
        out = {"type": "dict", "keys": sorted(list(payload.keys()))[:80]}
        for key in ("data", "events", "results", "response"):
            if key in payload:
                val = payload.get(key)
                out[f"{key}_type"] = type(val).__name__
                if isinstance(val, list):
                    out[f"{key}_length"] = len(val)
                    if val and isinstance(val[0], dict):
                        out[f"{key}_sample_item_keys"] = sorted(list(val[0].keys()))[:50]
                elif isinstance(val, dict):
                    out[f"{key}_keys"] = sorted(list(val.keys()))[:50]
        return out
    return {"type": type(payload).__name__}


def unwrap_events(payload: Any) -> Tuple[List[Dict[str, Any]], str]:
    """Return events list and the path used to unwrap it."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)], "root_list"
    if not isinstance(payload, dict):
        return [], "not_dict_or_list"

    candidates = [
        ("events", payload.get("events")),
        ("data", payload.get("data")),
        ("results", payload.get("results")),
        ("response", payload.get("response")),
    ]
    for path, val in candidates:
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)], path
        if isinstance(val, dict):
            for sub in ("events", "data", "results"):
                subval = val.get(sub)
                if isinstance(subval, list):
                    return [x for x in subval if isinstance(x, dict)], f"{path}.{sub}"
    return [], "no_event_list_found"


def unwrap_odds(payload: Any) -> Dict[str, Any]:
    """Return an odds event dict where bookmakers might live."""
    if isinstance(payload, dict):
        # Common Odds API shape is already a dict with bookmakers.
        if isinstance(payload.get("bookmakers"), list):
            return payload
        for key in ("data", "event", "result", "response"):
            val = payload.get(key)
            if isinstance(val, dict) and isinstance(val.get("bookmakers"), list):
                return val
        # Sometimes data is list with one event.
        for key in ("data", "events", "results"):
            val = payload.get(key)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                if isinstance(val[0].get("bookmakers"), list):
                    return val[0]
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        if isinstance(payload[0].get("bookmakers"), list):
            return payload[0]
    return {}


def event_id(event: Dict[str, Any]) -> Optional[str]:
    for key in ("id", "event_id", "eventId", "key"):
        val = event.get(key)
        if val:
            return str(val)
    return None


def event_summary(event: Dict[str, Any]) -> Dict[str, Any]:
    keep = {}
    for key in ("id", "event_id", "eventId", "key", "commence_time", "commenceTime", "home_team", "away_team", "homeTeam", "awayTeam", "sport_key", "sport_title"):
        if key in event:
            keep[key] = event.get(key)
    return keep


def is_fanduel(book: Dict[str, Any]) -> bool:
    vals = [book.get("key"), book.get("title"), book.get("name")]
    joined = " ".join(str(v).lower() for v in vals if v)
    return any(target in joined for target in BOOKMAKER_TARGETS)


def get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    safe_params = dict(params)
    resp = SESSION.get(url, params=params, timeout=25)
    payload, mode = parse_json_response(resp)
    result = {
        "url": redact(resp.url),
        "status_code": resp.status_code,
        "content_type": resp.headers.get("content-type"),
        "parse_mode": mode,
        "ok": resp.ok,
        "shape": response_shape(payload),
        "payload": payload,
    }
    if payload is None:
        result["text_preview"] = redact(resp.text[:1200])
    return result


def summarize_fanduel_odds(odds_payload: Any) -> Dict[str, Any]:
    event = unwrap_odds(odds_payload)
    books = event.get("bookmakers") if isinstance(event, dict) else None
    out = {
        "bookmakers_seen": [],
        "has_fanduel": False,
        "fanduel_market_keys": [],
        "k_market_keys": [],
        "k_outcome_count": 0,
        "k_outcome_samples": [],
    }
    if not isinstance(books, list):
        out["bookmakers_type"] = type(books).__name__
        return out

    for book in books:
        if not isinstance(book, dict):
            continue
        book_key = book.get("key") or book.get("title") or book.get("name")
        out["bookmakers_seen"].append(str(book_key))
        if not is_fanduel(book):
            continue
        out["has_fanduel"] = True
        markets = book.get("markets") or []
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict):
                continue
            mkey = str(market.get("key") or market.get("name") or market.get("title") or "")
            out["fanduel_market_keys"].append(mkey)
            if "strikeout" in mkey.lower() or "pitcher" in mkey.lower() and "k" in mkey.lower():
                out["k_market_keys"].append(mkey)
                outcomes = market.get("outcomes") or []
                if isinstance(outcomes, list):
                    out["k_outcome_count"] += len(outcomes)
                    for outcome in outcomes[:12]:
                        if not isinstance(outcome, dict):
                            continue
                        sample = {}
                        for key in ("name", "description", "point", "price", "odds", "handicap", "label"):
                            if key in outcome:
                                sample[key] = outcome.get(key)
                        sample["market_key"] = mkey
                        out["k_outcome_samples"].append(sample)
    out["bookmakers_seen"] = sorted(set(out["bookmakers_seen"]))
    out["fanduel_market_keys"] = sorted(set(out["fanduel_market_keys"]))
    out["k_market_keys"] = sorted(set(out["k_market_keys"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=15)
    parser.add_argument("--market-scan", action="store_true", help="Try possible PropLine/Odds API pitcher K aliases, not just current keys.")
    parser.add_argument("--save", default="/data/propline_k_probe_latest.json")
    args = parser.parse_args()

    report: Dict[str, Any] = {
        "probe_version": "standalone_propline_k_probe_v1",
        "generated_at_utc": now_iso(),
        "base": BASE,
        "has_api_key": bool(API_KEY),
        "markets_requested": MARKET_SCAN if args.market_scan else DEFAULT_MARKETS,
        "max_events": args.max_events,
    }

    if not API_KEY:
        report["fatal"] = "missing_PROPLINE_API_KEY"
        print(json.dumps(report, indent=2, default=str))
        return 2

    events_url = f"{BASE}/events"
    events_resp = get_json(events_url, {"apiKey": API_KEY})
    events_payload = events_resp.get("payload")
    events, unwrap_path = unwrap_events(events_payload)
    report["events_response"] = {
        "url": events_resp.get("url"),
        "status_code": events_resp.get("status_code"),
        "content_type": events_resp.get("content_type"),
        "parse_mode": events_resp.get("parse_mode"),
        "ok": events_resp.get("ok"),
        "shape": events_resp.get("shape"),
        "unwrap_path": unwrap_path,
        "events_unwrapped_count": len(events),
        "sample_events": [event_summary(e) for e in events[:5]],
    }
    if events_resp.get("text_preview"):
        report["events_response"]["text_preview"] = events_resp.get("text_preview")
    if not events:
        # Include a sanitized small payload preview if JSON so we can see error/wrapper shape.
        if isinstance(events_payload, dict):
            report["events_response"]["payload_preview"] = redact({k: events_payload.get(k) for k in list(events_payload.keys())[:20]})
        print(json.dumps(redact(report), indent=2, default=str))
        try:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save).write_text(json.dumps(redact(report), indent=2, default=str), encoding="utf-8")
        except Exception:
            pass
        return 0

    markets = report["markets_requested"]
    markets_param = ",".join(markets)
    event_results = []
    totals = {
        "events_checked": 0,
        "odds_http_ok": 0,
        "events_with_fanduel": 0,
        "events_with_k_markets": 0,
        "k_outcome_count": 0,
    }
    market_counter: Counter[str] = Counter()
    bookmaker_counter: Counter[str] = Counter()

    for event in events[: max(0, args.max_events)]:
        eid = event_id(event)
        if not eid:
            continue
        totals["events_checked"] += 1
        odds_url = f"{BASE}/events/{eid}/odds"
        params = {
            "apiKey": API_KEY,
            "bookmakers": "fanduel",
            "markets": markets_param,
            "oddsFormat": "american",
        }
        odds_resp = get_json(odds_url, params)
        odds_payload = odds_resp.get("payload")
        fd = summarize_fanduel_odds(odds_payload)
        if odds_resp.get("ok"):
            totals["odds_http_ok"] += 1
        if fd.get("has_fanduel"):
            totals["events_with_fanduel"] += 1
        if fd.get("k_market_keys"):
            totals["events_with_k_markets"] += 1
        totals["k_outcome_count"] += int(fd.get("k_outcome_count") or 0)
        for k in fd.get("fanduel_market_keys") or []:
            market_counter[k] += 1
        for b in fd.get("bookmakers_seen") or []:
            bookmaker_counter[b] += 1

        item = {
            "event": event_summary(event),
            "odds_response": {
                "url": odds_resp.get("url"),
                "status_code": odds_resp.get("status_code"),
                "ok": odds_resp.get("ok"),
                "shape": odds_resp.get("shape"),
            },
            "fan_duel_summary": fd,
        }
        if odds_resp.get("text_preview"):
            item["odds_response"]["text_preview"] = odds_resp.get("text_preview")
        if not fd.get("has_fanduel") and isinstance(odds_payload, dict):
            # This helps diagnose error objects without dumping huge odds payloads.
            preview = {k: odds_payload.get(k) for k in list(odds_payload.keys())[:20]}
            item["odds_payload_preview"] = redact(preview)
        event_results.append(item)

    report["totals"] = totals
    report["fanduel_market_keys_seen_counts"] = dict(sorted(market_counter.items()))
    report["bookmakers_seen_counts"] = dict(sorted(bookmaker_counter.items()))
    report["events"] = event_results

    safe = redact(report)
    print(json.dumps(safe, indent=2, default=str))
    try:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(safe, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved sanitized report to {args.save}")
    except Exception as exc:
        print(f"\nCould not save report to {args.save}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
