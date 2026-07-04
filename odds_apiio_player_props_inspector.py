#!/usr/bin/env python3
"""
Odds-API.io Player Props Inspector
Standalone diagnostic. Does NOT touch api.py or PropLine.

Goal:
- Pull Odds-API.io MLB events for FanDuel
- Prefer pending/upcoming events
- Pull /odds/multi in small batches
- Print exactly where 'Player Props' lives and what labels/keys it contains
- Find any strikeout/K strings even if nested under generic Player Props market

Env:
  ODDS_API_IO_KEY

Usage:
  python odds_apiio_player_props_inspector.py --max-events 10
  python odds_apiio_player_props_inspector.py --max-events 10 --dump-sample
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

BASE = "https://api.odds-api.io/v3"
OUTDIR = Path("/data/odds_provider_tests")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUTDIR / "odds_apiio_player_props_inspector_latest.json"

K_PAT = re.compile(r"(strikeout|strikeouts|\bk\b|pitcher)", re.I)
PLAYER_PROP_PAT = re.compile(r"player\s*props?", re.I)


def mask_url(url: str) -> str:
    return re.sub(r"(apiKey=)[^&]+", r"\1***", url)


def shape(x: Any) -> Dict[str, Any]:
    if isinstance(x, list):
        d = {"type": "list", "len": len(x)}
        if x:
            d["first_type"] = type(x[0]).__name__
            if isinstance(x[0], dict):
                d["first_keys"] = list(x[0].keys())[:20]
        return d
    if isinstance(x, dict):
        return {"type": "dict", "len": len(x), "first_keys": list(x.keys())[:20]}
    return {"type": type(x).__name__, "repr": repr(x)[:120]}


def get_json(path: str, params: Dict[str, Any], requests_log: List[Dict[str, Any]]) -> Any:
    url = BASE + path
    r = requests.get(url, params=params, timeout=30)
    try:
        payload = r.json()
    except Exception:
        payload = {"_non_json_text_preview": r.text[:1000]}
    requests_log.append({
        "status_code": r.status_code,
        "path": path,
        "url": mask_url(r.url),
        "shape": shape(payload),
        "ok": r.ok,
        "error_preview": payload if (isinstance(payload, dict) and "error" in payload) else None,
    })
    return payload


def event_status(e: Dict[str, Any]) -> str:
    return str(e.get("status") or e.get("state") or "").lower()


def event_dt(e: Dict[str, Any]):
    s = e.get("date") or e.get("commence_time") or e.get("startTime")
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def is_pending_or_future(e: Dict[str, Any]) -> bool:
    st = event_status(e)
    if st in {"pending", "scheduled", "pre-game", "pregame", "not_started", "upcoming"}:
        return True
    dt = event_dt(e)
    if dt and dt >= datetime.now(timezone.utc):
        return True
    return False


def chunks(xs: List[Any], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]


def walk(obj: Any, path: str = ""):
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield from walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            yield from walk(v, p)


def short(v: Any, limit: int = 240) -> str:
    try:
        s = json.dumps(v, ensure_ascii=False, default=str)
    except Exception:
        s = repr(v)
    return s[:limit]


def collect_markets(odds_payload: List[Any]) -> Tuple[Counter, Counter, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    bookmakers = Counter()
    markets = Counter()
    player_prop_samples = []
    k_hits = []
    market_paths = []

    for event_idx, ev in enumerate(odds_payload):
        if not isinstance(ev, dict):
            continue
        event_label = f"{ev.get('away')} @ {ev.get('home')}" if (ev.get('away') or ev.get('home')) else str(ev.get('id'))

        # Common Odds-API.io shape: event.bookmakers.{FanDuel}.{marketName}: ...
        # But this inspector is intentionally recursive.
        for p, v in walk(ev):
            if isinstance(v, dict):
                # bookmaker-like dict keys
                for key in v.keys():
                    if str(key).lower() == "fanduel":
                        bookmakers["FanDuel"] += 1
                # market-like names
                for key in v.keys():
                    ks = str(key)
                    if ks in {"ML", "Spread", "Totals", "Correct Score", "ML HT", "Team Total Away", "Team Total Home", "Totals HT", "Player Props"} or "prop" in ks.lower() or "strike" in ks.lower():
                        markets[ks] += 1
                        market_paths.append({"event": event_label, "path": f"{p}.{ks}" if p else ks, "market_name": ks})

            if isinstance(v, list):
                # sample lists under Player Props path or containing K-ish values
                if PLAYER_PROP_PAT.search(p):
                    player_prop_samples.append({"event": event_label, "path": p, "shape": shape(v), "sample": v[:3]})
                if any(K_PAT.search(str(item)) for item in v[:50]):
                    k_hits.append({"event": event_label, "path": p, "shape": shape(v), "sample": v[:5]})

            if isinstance(v, (str, int, float, bool)) or v is None:
                s = str(v)
                if K_PAT.search(s):
                    k_hits.append({"event": event_label, "path": p, "value": s})

    return bookmakers, markets, player_prop_samples, k_hits, market_paths[:100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-events", type=int, default=10)
    ap.add_argument("--include-settled", action="store_true", help="Do not filter to pending/future events")
    ap.add_argument("--dump-sample", action="store_true", help="Print first raw odds event sample, truncated")
    args = ap.parse_args()

    key = os.environ.get("ODDS_API_IO_KEY") or os.environ.get("ODDSAPIIO_KEY")
    if not key:
        print("Missing ODDS_API_IO_KEY")
        sys.exit(1)

    requests_log = []
    events = get_json("/events", {"sport": "baseball", "league": "usa-mlb", "bookmaker": "FanDuel", "apiKey": key}, requests_log)
    if not isinstance(events, list):
        report = {"error": "events_not_list", "events_shape": shape(events), "requests": requests_log}
        OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    pending = [e for e in events if isinstance(e, dict) and is_pending_or_future(e)]
    chosen = events if args.include_settled else pending
    chosen = chosen[: args.max_events]
    event_ids = [str(e.get("id")) for e in chosen if isinstance(e, dict) and e.get("id")]

    odds_payload = []
    for batch in chunks(event_ids, 10):
        data = get_json("/odds/multi", {"eventIds": ",".join(batch), "bookmakers": "FanDuel", "apiKey": key}, requests_log)
        if isinstance(data, list):
            odds_payload.extend(data)
        else:
            odds_payload.append({"_error_payload": data})

    bookmakers, markets, player_prop_samples, k_hits, market_paths = collect_markets(odds_payload)

    report = {
        "probe_version": "odds_apiio_player_props_inspector_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_count": len(requests_log),
        "events_total": len(events),
        "events_pending_or_future": len(pending),
        "events_used": len(chosen),
        "include_settled": args.include_settled,
        "requests": requests_log,
        "events_sample": [{"id": e.get("id"), "away": e.get("away"), "home": e.get("home"), "date": e.get("date"), "status": e.get("status")} for e in chosen[:10] if isinstance(e, dict)],
        "odds_payload_count": len(odds_payload),
        "bookmakers_seen": dict(bookmakers),
        "market_names_seen": dict(markets),
        "player_prop_samples": player_prop_samples[:20],
        "k_or_strikeout_hits": k_hits[:50],
        "market_sample_paths": market_paths[:50],
    }
    if args.dump_sample:
        report["raw_odds_first_event_sample"] = odds_payload[0] if odds_payload else None

    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("=== ODDS-API.IO PLAYER PROPS INSPECTOR ===")
    print(f"Generated UTC: {report['generated_at_utc']}")
    print(f"Requests: {len(requests_log)} | events_total={len(events)} pending_or_future={len(pending)} events_used={len(chosen)} odds_payload_count={len(odds_payload)}")
    print("\nEVENTS USED")
    for e in report["events_sample"]:
        print(f"- {e.get('id')} | {e.get('away')} @ {e.get('home')} | {e.get('date')} | {e.get('status')}")
    print("\nBOOKMAKERS SEEN")
    if bookmakers:
        for k, v in bookmakers.most_common():
            print(f"- {k}: {v}")
    else:
        print("None detected")
    print("\nMARKET NAMES SEEN")
    if markets:
        for k, v in markets.most_common(50):
            print(f"- {k}: {v}")
    else:
        print("None detected")
    print("\nPLAYER PROPS SAMPLES")
    if player_prop_samples:
        for s in player_prop_samples[:10]:
            print(f"- {s['event']} | {s['path']} | {s['shape']} | sample={short(s['sample'], 500)}")
    else:
        print("None detected")
    print("\nK / STRIKEOUT STRING HITS")
    print(f"count={len(k_hits)}")
    for h in k_hits[:20]:
        print(f"- {h.get('event')} | {h.get('path')} | {short(h.get('value', h.get('sample')), 500)}")
    print("\nREQUESTS")
    for r in requests_log:
        print(f"- {r['status_code']} {r['path']} shape={r['shape']} url={r['url']}")
    print("\nOUTPUT")
    print(str(OUT_JSON))


if __name__ == "__main__":
    main()
