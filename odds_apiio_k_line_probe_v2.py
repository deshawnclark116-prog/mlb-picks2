#!/usr/bin/env python3
"""
Odds-API.io MLB FanDuel main pitcher-strikeouts line probe v2.

Purpose:
  Test ONE thing without touching api.py:
  Can Odds-API.io return FanDuel's main MLB pitcher strikeout lines?

Why v2 exists:
  v1 used sport=mlb. Odds-API.io docs list the sport slug as baseball.
  v2 uses sport=baseball, discovers MLB league slugs, and uses /odds/multi
  to reduce request count.

Env var:
  ODDS_API_IO_KEY

Examples:
  python odds_apiio_k_line_probe_v2.py
  python odds_apiio_k_line_probe_v2.py --max-events 15
  python odds_apiio_k_line_probe_v2.py --league major-league-baseball
  python odds_apiio_k_line_probe_v2.py --raw

Outputs:
  /data/odds_provider_tests/odds_apiio_k_line_probe_v2_latest.json
  /data/odds_provider_tests/odds_apiio_k_line_probe_v2_latest.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

BASE = os.getenv("ODDS_API_IO_BASE_URL", "https://api.odds-api.io/v3").rstrip("/")
OUT_DIR = Path("/data/odds_provider_tests")
OUT_JSON = OUT_DIR / "odds_apiio_k_line_probe_v2_latest.json"
OUT_CSV = OUT_DIR / "odds_apiio_k_line_probe_v2_latest.csv"

K_MARKET_TERMS = (
    "pitcher strikeout",
    "pitcher strikeouts",
    "strikeout",
    "strikeouts",
    "total pitcher strikeouts",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def redact_url(url: str) -> str:
    return re.sub(r"([?&](?:apiKey|apikey|api_key|key|token)=)[^&]+", r"\1<REDACTED>", url, flags=re.I)


def shape(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, list):
        return {
            "type": "list",
            "len": len(obj),
            "first_type": type(obj[0]).__name__ if obj else None,
            "first_keys": list(obj[0].keys())[:30] if obj and isinstance(obj[0], dict) else None,
        }
    if isinstance(obj, dict):
        return {"type": "dict", "keys": list(obj.keys())[:50], "key_count": len(obj)}
    return {"type": type(obj).__name__}


def as_list(payload: Any) -> List[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("data", "events", "results", "items", "fixtures", "leagues", "sports", "bookmakers"):
            v = payload.get(k)
            if isinstance(v, list):
                return v
        data = payload.get("data")
        if isinstance(data, dict):
            for k in ("events", "results", "items", "fixtures", "leagues"):
                v = data.get(k)
                if isinstance(v, list):
                    return v
    return []


def get_json(session: requests.Session, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 25) -> Tuple[Dict[str, Any], Any]:
    url = f"{BASE}{endpoint}"
    params = params or {}
    started = time.time()
    try:
        r = session.get(url, params=params, timeout=timeout)
        meta = {
            "endpoint": endpoint,
            "url": redact_url(r.url),
            "status_code": r.status_code,
            "ok": bool(r.ok),
            "elapsed_ms": round((time.time() - started) * 1000, 1),
            "content_type": r.headers.get("content-type", ""),
        }
        try:
            payload = r.json()
            meta["parse"] = "json"
            meta["shape"] = shape(payload)
        except Exception:
            payload = None
            meta["parse"] = "text"
            meta["text_preview"] = r.text[:1200]
        return meta, payload
    except Exception as e:
        return {"endpoint": endpoint, "url": redact_url(url), "status_code": None, "ok": False, "error": repr(e)}, None


def norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def text(s: Any) -> str:
    return str(s or "").strip()


def lower(s: Any) -> str:
    return text(s).lower()


def contains_k_market(s: Any) -> bool:
    h = lower(s)
    return any(t in h for t in K_MARKET_TERMS)


def parse_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"-?\d+(?:\.\d+)?", str(x))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def event_id(e: Dict[str, Any]) -> str:
    return text(e.get("id") or e.get("eventId") or e.get("event_id") or e.get("fixtureId") or "")


def event_name(e: Dict[str, Any]) -> str:
    name = e.get("name") or e.get("event") or e.get("title")
    if name:
        return text(name)
    home = e.get("home") or e.get("homeTeam") or e.get("home_team") or e.get("homeName")
    away = e.get("away") or e.get("awayTeam") or e.get("away_team") or e.get("awayName")
    if home or away:
        return f"{away or '?'} @ {home or '?'}"
    return event_id(e)


def event_start(e: Dict[str, Any]) -> str:
    return text(e.get("date") or e.get("startTime") or e.get("start_time") or e.get("commence_time") or e.get("time") or "")


def find_mlb_leagues(leagues: List[Any]) -> List[Dict[str, Any]]:
    hits = []
    for l in leagues:
        if not isinstance(l, dict):
            continue
        hay = " ".join(text(l.get(k)) for k in ("slug", "league", "name", "title", "leagueName", "sport", "country"))
        h = lower(hay)
        if "mlb" in h or "major league baseball" in h or h.strip() == "major league baseball":
            hits.append(l)
    return hits


def league_slug(l: Dict[str, Any]) -> str:
    return text(l.get("slug") or l.get("league") or l.get("leagueSlug") or l.get("id") or l.get("name") or "")


def chunked(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def flatten_odds_payload(payload: Any) -> List[Dict[str, Any]]:
    """Return event odds objects from either /odds or /odds/multi payload."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        # Single event odds object
        if "bookmakers" in payload and isinstance(payload.get("bookmakers"), dict):
            return [payload]
        # Wrapped multi response
        lst = as_list(payload)
        return [x for x in lst if isinstance(x, dict)]
    return []


def parse_k_lines_from_event_odds(ev: Dict[str, Any], wanted_book: str = "FanDuel") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    eid = text(ev.get("id") or ev.get("eventId") or ev.get("event_id") or "")
    ename = event_name(ev)
    start = event_start(ev)
    bookmakers = ev.get("bookmakers")
    if not isinstance(bookmakers, dict):
        return rows
    wanted = norm(wanted_book)
    for book_name, markets in bookmakers.items():
        if wanted not in norm(book_name) and norm(book_name) not in wanted:
            continue
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict):
                continue
            mname = text(market.get("name") or market.get("marketName") or market.get("key") or market.get("market") or "")
            if not contains_k_market(mname):
                continue
            odds = market.get("odds") or market.get("outcomes") or market.get("selections") or []
            if not isinstance(odds, list):
                continue
            for o in odds:
                if not isinstance(o, dict):
                    continue
                player = text(o.get("label") or o.get("player") or o.get("description") or o.get("name") or o.get("participant") or "")
                line = parse_float(o.get("hdp") if "hdp" in o else o.get("point") if "point" in o else o.get("line"))
                over_price = o.get("over") if "over" in o else None
                under_price = o.get("under") if "under" in o else None
                if over_price is not None or under_price is not None:
                    rows.append({
                        "event_id": eid,
                        "event": ename,
                        "start_time": start,
                        "bookmaker": text(book_name),
                        "market_name": mname,
                        "player": player,
                        "line": line,
                        "over_price": over_price,
                        "under_price": under_price,
                        "main_line": True,
                        "raw_shape": shape(o),
                    })
                    continue
                # Fallback side-row shape
                side = lower(o.get("side") or o.get("name") or o.get("label"))
                price = o.get("price") or o.get("odds") or o.get("american") or o.get("decimal")
                rows.append({
                    "event_id": eid,
                    "event": ename,
                    "start_time": start,
                    "bookmaker": text(book_name),
                    "market_name": mname,
                    "player": player,
                    "line": line,
                    "side": "over" if "over" in side else "under" if "under" in side else None,
                    "price": price,
                    "main_line": True,
                    "raw_shape": shape(o),
                })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe Odds-API.io for FanDuel MLB pitcher K main lines using sport=baseball.")
    ap.add_argument("--sport", default="baseball", help="Odds-API.io sport slug. Docs list baseball, not mlb.")
    ap.add_argument("--league", default="", help="Optional league slug. If blank, script discovers MLB and tries that first.")
    ap.add_argument("--bookmaker", default="FanDuel")
    ap.add_argument("--max-events", type=int, default=15)
    ap.add_argument("--use-multi", action="store_true", default=True)
    ap.add_argument("--no-multi", action="store_false", dest="use_multi")
    ap.add_argument("--raw", action="store_true", help="Store small raw previews/shapes in JSON output.")
    args = ap.parse_args()

    key = os.getenv("ODDS_API_IO_KEY") or os.getenv("ODDSAPI_IO_KEY") or ""
    report: Dict[str, Any] = {
        "probe_version": "odds_apiio_k_line_probe_v2",
        "generated_at_utc": utc_now(),
        "base": BASE,
        "has_key": bool(key),
        "sport_requested": args.sport,
        "bookmaker_requested": args.bookmaker,
        "request_count": 0,
        "requests": [],
        "sports_baseball_found": False,
        "mlb_leagues_found": [],
        "events_count": 0,
        "events_sample": [],
        "k_lines_count": 0,
        "k_lines": [],
        "error": None,
        "notes": [
            "Uses sport=baseball because Odds-API.io docs list baseball as the sport slug.",
            "Uses /odds/multi in chunks of up to 10 event IDs to cut requests.",
            "This script does not touch api.py, PropLine, picks, grading, or record.json.",
        ],
    }

    if not key:
        report["error"] = "Missing ODDS_API_IO_KEY env var."
        print(json.dumps(report, indent=2))
        return 2

    s = requests.Session()
    s.headers.update({"User-Agent": "PropEdge-OddsAPIIO-KLineProbe/2.0"})

    # 1) Sports discovery, no auth needed but harmless without key.
    meta, sports_payload = get_json(s, "/sports")
    report["request_count"] += 1
    report["requests"].append(meta)
    sports = as_list(sports_payload)
    for sp in sports:
        if isinstance(sp, dict):
            h = " ".join(text(sp.get(k)) for k in ("slug", "sport", "name", "title"))
            if "baseball" in lower(h):
                report["sports_baseball_found"] = True
                report["baseball_sport_sample"] = sp if args.raw else {k: sp.get(k) for k in list(sp.keys())[:8]}
                break

    # 2) League discovery.
    league_to_try = args.league.strip()
    meta, leagues_payload = get_json(s, "/leagues", {"apiKey": key, "sport": args.sport})
    report["request_count"] += 1
    report["requests"].append(meta)
    leagues = as_list(leagues_payload)
    mlb_leagues = find_mlb_leagues(leagues)
    report["mlb_leagues_found"] = [
        {"slug": league_slug(l), "name": text(l.get("name") or l.get("leagueName") or l.get("title") or l.get("slug"))}
        for l in mlb_leagues[:10]
    ]
    if not league_to_try and mlb_leagues:
        league_to_try = league_slug(mlb_leagues[0])

    # 3) Events. Try discovered/provided MLB league first, then sport-only fallback if needed.
    event_attempts: List[Tuple[str, Dict[str, Any]]] = []
    base_params = {"apiKey": key, "sport": args.sport, "bookmaker": args.bookmaker, "status": "pending"}
    if league_to_try:
        p = dict(base_params)
        p["league"] = league_to_try
        event_attempts.append(("sport+league+bookmaker", p))
    event_attempts.append(("sport+bookmaker", dict(base_params)))
    p2 = {"apiKey": key, "sport": args.sport, "status": "pending"}
    if league_to_try:
        p2["league"] = league_to_try
        event_attempts.append(("sport+league", p2))
    event_attempts.append(("sport-only", {"apiKey": key, "sport": args.sport, "status": "pending"}))

    events: List[Any] = []
    for label, params in event_attempts:
        meta, events_payload = get_json(s, "/events", params)
        report["request_count"] += 1
        report["requests"].append(meta | {"attempt": label})
        events = as_list(events_payload)
        if events:
            report["events_attempt_used"] = label
            if args.raw:
                report["events_payload_shape"] = shape(events_payload)
            break

    report["league_used"] = league_to_try
    report["events_count"] = len(events)
    report["events_sample"] = [
        {"id": event_id(e), "event": event_name(e), "start_time": event_start(e), "raw_keys": list(e.keys())[:20]}
        for e in events[:8] if isinstance(e, dict)
    ]

    if not events:
        report["error"] = "No events found with sport=baseball attempts. Check league slug, selected bookmakers, or account permissions."
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print_summary(report)
        return 1

    event_ids = [event_id(e) for e in events if isinstance(e, dict) and event_id(e)][: args.max_events]
    k_lines: List[Dict[str, Any]] = []

    # 4) Odds: prefer /odds/multi. It can fetch up to 10 event IDs per request.
    if args.use_multi:
        for chunk in chunked(event_ids, 10):
            meta, odds_payload = get_json(s, "/odds/multi", {"apiKey": key, "eventIds": ",".join(chunk), "bookmakers": args.bookmaker})
            report["request_count"] += 1
            report["requests"].append(meta | {"event_ids_count": len(chunk)})
            for ev_odds in flatten_odds_payload(odds_payload):
                k_lines.extend(parse_k_lines_from_event_odds(ev_odds, args.bookmaker))
    else:
        for eid in event_ids:
            meta, odds_payload = get_json(s, "/odds", {"apiKey": key, "eventId": eid, "bookmakers": args.bookmaker})
            report["request_count"] += 1
            report["requests"].append(meta | {"event_id": eid})
            for ev_odds in flatten_odds_payload(odds_payload):
                k_lines.extend(parse_k_lines_from_event_odds(ev_odds, args.bookmaker))

    report["k_lines_count"] = len(k_lines)
    report["k_lines"] = k_lines[:300]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    if k_lines:
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            fields = ["event_id", "event", "start_time", "bookmaker", "market_name", "player", "line", "over_price", "under_price", "side", "price", "main_line"]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(k_lines)

    print_summary(report)
    return 0


def print_summary(report: Dict[str, Any]) -> None:
    print("=== ODDS-API.IO K LINE PROBE V2 ===")
    print(f"Generated UTC: {report.get('generated_at_utc')}")
    print("Target: FanDuel main MLB pitcher strikeouts line")
    print(f"has_key={report.get('has_key')} sport={report.get('sport_requested')} bookmaker={report.get('bookmaker_requested')}")
    print(f"request_count={report.get('request_count')} error={report.get('error')}")
    print(f"sports_baseball_found={report.get('sports_baseball_found')}")
    print(f"league_used={report.get('league_used')}")
    print(f"mlb_leagues_found={report.get('mlb_leagues_found')}")
    print(f"events_attempt_used={report.get('events_attempt_used')}")
    print(f"events_count={report.get('events_count')} k_lines_count={report.get('k_lines_count')}")
    print("")
    print("EVENTS SAMPLE")
    for e in report.get("events_sample", [])[:8]:
        print(f"- {e.get('id')} | {e.get('event')} | {e.get('start_time')}")
    print("")
    print("SAMPLE K LINES")
    lines = report.get("k_lines", []) or []
    if not lines:
        print("No K lines found.")
    for i, r in enumerate(lines[:25], 1):
        print(f"{i:02d}. {r.get('player')} | line={r.get('line')} | over={r.get('over_price') or r.get('price')} under={r.get('under_price')} | market={r.get('market_name')} | event={r.get('event')}")
    print("")
    print("REQUESTS")
    for r in report.get("requests", [])[:12]:
        print(f"- {r.get('status_code')} {r.get('endpoint')} attempt={r.get('attempt')} shape={r.get('shape')}")
    print("")
    print("OUTPUTS")
    print(str(OUT_JSON))
    if OUT_CSV.exists():
        print(str(OUT_CSV))


if __name__ == "__main__":
    raise SystemExit(main())
