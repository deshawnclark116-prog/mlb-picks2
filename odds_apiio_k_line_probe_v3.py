#!/usr/bin/env python3
"""
odds_apiio_k_line_probe_v3.py
Standalone Odds-API.io FanDuel MLB pitcher strikeout line probe.

Purpose:
- Tests Odds-API.io ONLY.
- Does not call PropLine.
- Does not modify api.py, predictions, records, or candidate logs.
- Uses the documented Odds-API.io params:
    /v3/events?sport=baseball&league=usa-mlb&bookmaker=FanDuel
    /v3/odds/multi?eventIds=1,2,3&bookmakers=FanDuel
  Note: events endpoint uses `bookmaker` singular; odds endpoints use `bookmakers` plural.

Env:
- ODDS_API_IO_KEY

Run:
  python odds_apiio_k_line_probe_v3.py --max-events 15
  python odds_apiio_k_line_probe_v3.py --max-events 3 --save-raw
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except Exception as e:  # pragma: no cover
    print("ERROR: requests is required. Install with: pip install requests")
    raise

BASE = "https://api.odds-api.io/v3"
OUT_DIR = Path("/data/odds_provider_tests")
OUT_JSON = OUT_DIR / "odds_apiio_k_line_probe_v3_latest.json"
OUT_CSV = OUT_DIR / "odds_apiio_k_line_probe_v3_latest.csv"
OUT_RAW = OUT_DIR / "odds_apiio_k_line_probe_v3_raw_latest.json"

K_TERMS = ("strikeout", "strikeouts", "pitcher strikeout", "pitcher strikeouts", "player props - strikeouts")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def mask_url(url: str, api_key: Optional[str]) -> str:
    if not url:
        return url
    if api_key:
        url = url.replace(api_key, "***")
    return re.sub(r"(apiKey=)[^&]+", r"\1***", url)


def shape(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, list):
        d = {"type": "list", "len": len(obj)}
        if obj:
            d["first_type"] = type(obj[0]).__name__
            if isinstance(obj[0], dict):
                d["first_keys"] = list(obj[0].keys())[:20]
        return d
    if isinstance(obj, dict):
        return {"type": "dict", "len": len(obj), "first_keys": list(obj.keys())[:30]}
    return {"type": type(obj).__name__, "repr": repr(obj)[:200]}


@dataclass
class ReqLog:
    status: int
    endpoint: str
    attempt: Optional[str]
    url: str
    shape: Dict[str, Any]
    error_preview: Optional[str] = None


class Client:
    def __init__(self, api_key: str, timeout: int = 20):
        self.api_key = api_key
        self.timeout = timeout
        self.logs: List[ReqLog] = []

    def get(self, path: str, params: Dict[str, Any], attempt: Optional[str] = None) -> Any:
        params = dict(params or {})
        params["apiKey"] = self.api_key
        url = f"{BASE}{path}"
        try:
            r = requests.get(url, params=params, timeout=self.timeout)
            text = r.text[:500]
            try:
                data = r.json()
            except Exception:
                data = {"_non_json_preview": text}
            err = None
            if r.status_code >= 400:
                err = text
            self.logs.append(ReqLog(
                status=r.status_code,
                endpoint=path,
                attempt=attempt,
                url=mask_url(r.url, self.api_key),
                shape=shape(data),
                error_preview=err,
            ))
            return data
        except Exception as e:
            self.logs.append(ReqLog(
                status=0,
                endpoint=path,
                attempt=attempt,
                url=mask_url(url, self.api_key),
                shape={"type": "exception"},
                error_preview=repr(e),
            ))
            return {"error": repr(e)}


def as_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        for key in ("data", "events", "results", "items"):
            v = x.get(key)
            if isinstance(v, list):
                return v
        data = x.get("data")
        if isinstance(data, dict):
            for key in ("events", "results", "items"):
                v = data.get(key)
                if isinstance(v, list):
                    return v
    return []


def get_mlb_league(client: Client) -> Tuple[str, List[Dict[str, Any]], Any, Any]:
    sports = client.get("/sports", {}, attempt="sports")
    leagues = client.get("/leagues", {"sport": "baseball"}, attempt="leagues_baseball")
    league_list = as_list(leagues)
    mlb = []
    for lg in league_list:
        if not isinstance(lg, dict):
            continue
        slug = str(lg.get("slug", ""))
        name = str(lg.get("name", ""))
        if "mlb" in slug.lower() or "mlb" in name.lower() or "major league baseball" in name.lower():
            mlb.append({"slug": slug, "name": name, "eventsCount": lg.get("eventsCount")})
    chosen = "usa-mlb"
    for lg in mlb:
        if lg.get("slug") == "usa-mlb":
            chosen = "usa-mlb"
            break
    return chosen, mlb, sports, leagues


def get_events(client: Client, league: str, bookmaker: str) -> List[Dict[str, Any]]:
    # Docs best-practice example uses singular `bookmaker` on events.
    attempts = [
        ("sport+league+bookmaker", {"sport": "baseball", "league": league, "bookmaker": bookmaker}),
        ("sport+league", {"sport": "baseball", "league": league}),
        ("sport+bookmaker", {"sport": "baseball", "bookmaker": bookmaker}),
        ("sport_only", {"sport": "baseball"}),
    ]
    for name, params in attempts:
        data = client.get("/events", params, attempt=name)
        events = [x for x in as_list(data) if isinstance(x, dict)]
        if events:
            return events
    return []


def chunked(seq: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def collect_bookmaker_market_names(odds_payloads: List[Any]) -> Tuple[Dict[str, int], Dict[str, int]]:
    books: Dict[str, int] = {}
    markets: Dict[str, int] = {}
    for ev in odds_payloads:
        if not isinstance(ev, dict):
            continue
        bks = ev.get("bookmakers")
        if isinstance(bks, dict):
            iterable = bks.items()
        elif isinstance(bks, list):
            iterable = []
            for b in bks:
                if isinstance(b, dict):
                    name = b.get("name") or b.get("title") or b.get("key") or b.get("bookmaker")
                    iterable.append((str(name), b.get("markets") or b.get("odds") or b))
        else:
            iterable = []
        for bname, bval in iterable:
            books[str(bname)] = books.get(str(bname), 0) + 1
            if isinstance(bval, list):
                for m in bval:
                    if isinstance(m, dict):
                        mn = str(m.get("name") or m.get("key") or m.get("market") or m.get("title") or "")
                        if mn:
                            markets[mn] = markets.get(mn, 0) + 1
            elif isinstance(bval, dict):
                for _, m in bval.items():
                    if isinstance(m, dict):
                        mn = str(m.get("name") or m.get("key") or m.get("market") or m.get("title") or "")
                        if mn:
                            markets[mn] = markets.get(mn, 0) + 1
                    elif isinstance(m, list):
                        for mm in m:
                            if isinstance(mm, dict):
                                mn = str(mm.get("name") or mm.get("key") or mm.get("market") or mm.get("title") or "")
                                if mn:
                                    markets[mn] = markets.get(mn, 0) + 1
    return books, markets


def market_is_k(market_name: str) -> bool:
    s = (market_name or "").lower()
    return "strikeout" in s and ("player" in s or "pitcher" in s or "prop" in s or s.strip() in ("strikeouts", "pitcher strikeouts"))


def parse_k_lines(odds_payloads: List[Any], bookmaker: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    market_names_seen: List[str] = []

    for ev in odds_payloads:
        if not isinstance(ev, dict):
            continue
        event_id = ev.get("id")
        home = ev.get("home")
        away = ev.get("away")
        date = ev.get("date")
        event_label = f"{away} @ {home}" if away and home else str(event_id)
        bks = ev.get("bookmakers")

        # Docs example: bookmakers is object keyed by bookmaker name: {"FanDuel": [market, ...]}
        markets_container = None
        if isinstance(bks, dict):
            if bookmaker in bks:
                markets_container = bks.get(bookmaker)
            else:
                # Case-insensitive fallback
                for k, v in bks.items():
                    if str(k).lower() == bookmaker.lower():
                        markets_container = v
                        break
        elif isinstance(bks, list):
            for b in bks:
                if not isinstance(b, dict):
                    continue
                bname = str(b.get("name") or b.get("title") or b.get("key") or b.get("bookmaker") or "")
                if bname.lower() == bookmaker.lower():
                    markets_container = b.get("markets") or b.get("odds") or b
                    break

        if not markets_container:
            continue

        markets = markets_container if isinstance(markets_container, list) else list(markets_container.values()) if isinstance(markets_container, dict) else []
        for market in markets:
            if not isinstance(market, dict):
                continue
            market_name = str(market.get("name") or market.get("key") or market.get("market") or market.get("title") or "")
            if market_name:
                market_names_seen.append(market_name)
            if not market_is_k(market_name):
                continue

            # Docs example: market["odds"] = [{label, hdp, over, under}, ...]
            odds_list = market.get("odds") or market.get("outcomes") or market.get("selections") or []
            if isinstance(odds_list, dict):
                odds_list = list(odds_list.values())
            if not isinstance(odds_list, list):
                continue
            for item in odds_list:
                if not isinstance(item, dict):
                    continue
                player = item.get("label") or item.get("player") or item.get("participant") or item.get("description") or item.get("name")
                line = item.get("hdp")
                if line is None:
                    line = item.get("point") or item.get("line") or item.get("handicap")
                over = item.get("over") or item.get("Over")
                under = item.get("under") or item.get("Under")
                # Some APIs represent over/under as separate outcomes; handle the simple pair form first.
                if player and line is not None:
                    try:
                        line_val = float(line)
                    except Exception:
                        line_val = line
                    rows.append({
                        "provider": "odds-api.io",
                        "bookmaker": bookmaker,
                        "market": market_name,
                        "event_id": event_id,
                        "event": event_label,
                        "date": date,
                        "player": str(player),
                        "line": line_val,
                        "over": over,
                        "under": under,
                    })
    return rows, sorted(set(market_names_seen))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-events", type=int, default=15)
    ap.add_argument("--bookmaker", default="FanDuel")
    ap.add_argument("--save-raw", action="store_true", help="Save raw odds payload sample to JSON for inspection")
    args = ap.parse_args()

    key = os.getenv("ODDS_API_IO_KEY", "").strip()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "probe_version": "v3",
        "generated_at_utc": now_utc(),
        "target": "FanDuel main MLB pitcher strikeouts line",
        "has_key": bool(key),
        "sport": "baseball",
        "bookmaker": args.bookmaker,
        "max_events": args.max_events,
        "requests": [],
        "events_count": 0,
        "events_sample": [],
        "odds_payload_count": 0,
        "bookmakers_seen": {},
        "market_names_seen": {},
        "k_lines_count": 0,
        "k_lines": [],
        "error": None,
        "notes": [],
    }

    if not key:
        result["error"] = "Missing ODDS_API_IO_KEY"
        OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("ERROR: Missing ODDS_API_IO_KEY")
        return 2

    client = Client(key)
    league, mlb_leagues, sports_raw, leagues_raw = get_mlb_league(client)
    result["league_used"] = league
    result["mlb_leagues_found"] = mlb_leagues

    events = get_events(client, league, args.bookmaker)
    result["events_count"] = len(events)
    result["events_sample"] = [
        {"id": e.get("id"), "away": e.get("away"), "home": e.get("home"), "date": e.get("date"), "status": e.get("status")}
        for e in events[:10]
    ]

    selected = events[: max(0, args.max_events)]
    odds_payloads: List[Any] = []

    # /odds/multi docs: eventIds and bookmakers are required. Up to 10 event IDs.
    for chunk in chunked(selected, 10):
        ids = ",".join(str(e.get("id")) for e in chunk if e.get("id") is not None)
        if not ids:
            continue
        data = client.get("/odds/multi", {"eventIds": ids, "bookmakers": args.bookmaker}, attempt="multi_eventIds_bookmakers")
        if isinstance(data, list):
            odds_payloads.extend(data)
        else:
            # Fallback to single-event endpoint only if multi fails.
            result["notes"].append(f"/odds/multi returned non-list for ids {ids}; trying /odds single-event fallback")
            for e in chunk:
                eid = e.get("id")
                if eid is None:
                    continue
                single = client.get("/odds", {"eventId": str(eid), "bookmakers": args.bookmaker}, attempt="single_eventId_bookmakers")
                if isinstance(single, dict) and not single.get("error"):
                    odds_payloads.append(single)

    result["odds_payload_count"] = len(odds_payloads)
    books_seen, markets_seen = collect_bookmaker_market_names(odds_payloads)
    k_rows, market_names_list = parse_k_lines(odds_payloads, args.bookmaker)
    result["bookmakers_seen"] = dict(sorted(books_seen.items(), key=lambda kv: (-kv[1], kv[0])))
    result["market_names_seen"] = dict(sorted(markets_seen.items(), key=lambda kv: (-kv[1], kv[0])))
    result["k_lines_count"] = len(k_rows)
    result["k_lines"] = k_rows
    result["requests"] = [asdict(x) for x in client.logs]
    result["request_count"] = len(client.logs)

    if args.save_raw:
        OUT_RAW.write_text(json.dumps({"events": selected, "odds_payloads": odds_payloads}, indent=2), encoding="utf-8")
        result["raw_saved_to"] = str(OUT_RAW)

    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["provider", "bookmaker", "market", "event_id", "event", "date", "player", "line", "over", "under"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in k_rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    print("=== ODDS-API.IO K LINE PROBE V3 ===")
    print(f"Generated UTC: {result['generated_at_utc']}")
    print("Target: FanDuel main MLB pitcher strikeouts line")
    print(f"has_key={result['has_key']} sport=baseball league={league} bookmaker={args.bookmaker}")
    print(f"request_count={result['request_count']} events_count={result['events_count']} odds_payload_count={result['odds_payload_count']} k_lines_count={result['k_lines_count']}")
    print()
    print("EVENTS SAMPLE")
    for e in result["events_sample"][:8]:
        print(f"- {e.get('id')} | {e.get('away')} @ {e.get('home')} | {e.get('date')} | {e.get('status')}")
    print()
    print("BOOKMAKERS SEEN")
    if result["bookmakers_seen"]:
        for k, v in list(result["bookmakers_seen"].items())[:20]:
            print(f"- {k}: {v}")
    else:
        print("None detected")
    print()
    print("MARKET NAMES SEEN")
    if result["market_names_seen"]:
        for k, v in list(result["market_names_seen"].items())[:40]:
            print(f"- {k}: {v}")
    else:
        print("None detected")
    print()
    print("SAMPLE K LINES")
    if k_rows:
        for r in k_rows[:25]:
            print(f"- {r.get('player')} | line={r.get('line')} | over={r.get('over')} | under={r.get('under')} | {r.get('event')} | {r.get('market')}")
    else:
        print("No K lines found.")
    print()
    print("REQUESTS")
    for log in client.logs:
        err = f" error={log.error_preview[:180]!r}" if log.error_preview else ""
        print(f"- {log.status} {log.endpoint} attempt={log.attempt} shape={log.shape} url={log.url}{err}")
    print()
    print("OUTPUTS")
    print(str(OUT_JSON))
    print(str(OUT_CSV))
    if args.save_raw:
        print(str(OUT_RAW))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
