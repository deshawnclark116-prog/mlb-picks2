#!/usr/bin/env python3
"""
NFL_PROPS_ODDS_PROBE

Standalone diagnostic: does The Odds API actually carry NFL player-prop
markets (rushing/receiving yards) on any real bookmaker, and at what
real per-player lines? Mirrors the_odds_api_k_line_probe.py's proven
two-step pattern (MLB pitcher strikeouts) -- The Odds API's player
props live on a PER-EVENT odds endpoint, not the general /odds endpoint
that only carries game-level h2h/spreads/totals:

  1. GET /v4/sports/{sport}/events                       (event list)
  2. GET /v4/sports/{sport}/events/{id}/odds?markets=...  (per event)

Does NOT touch api.py, predictions, records, or nfl_serving_builder_a.py.
Read-only against The Odds API. Writes only its own output file.

Run
---
python -u nfl_props_odds_probe.py --max-events 20
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
OUT_PATH = Path("nfl_props_odds_probe_latest.json")

# Real, documented-by-The-Odds-API candidate market keys for NFL player
# props -- checked directly rather than assumed, since this repo's rule
# is verify every data-source claim before trusting it.
CANDIDATE_MARKETS = [
    "player_rush_yds", "player_reception_yds", "player_receptions",
    "player_pass_yds", "player_pass_tds", "player_rush_attempts",
    "player_anytime_td",
]


def now_utc():
    return datetime.now(timezone.utc)


def get_key():
    for name in ("THE_ODDS_API_KEY", "ODDS_API_KEY", "THEODDSAPI_KEY"):
        val = os.getenv(name)
        if val:
            return val.strip(), name
    return None, None


def mask(url, key):
    if key:
        url = url.replace(key, "***")
    return re.sub(r"apiKey=[^&]+", "apiKey=***", url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-events", type=int, default=20)
    ap.add_argument("--bookmakers", default="fanduel,draftkings")
    args = ap.parse_args()

    key, key_name = get_key()
    result = {"generated_at_utc": now_utc().isoformat(), "sport": SPORT}
    if not key:
        result["error"] = "no api key found (THE_ODDS_API_KEY / ODDS_API_KEY / THEODDSAPI_KEY)"
        OUT_PATH.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 1
    result["key_env_used"] = key_name

    session = requests.Session()
    events_r = session.get(f"{BASE}/sports/{SPORT}/events", params={"apiKey": key}, timeout=25)
    result["events_status"] = events_r.status_code
    events = events_r.json() if events_r.ok else []
    if not events_r.ok:
        result["events_error"] = events_r.text[:500]
        OUT_PATH.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 1
    result["events_total"] = len(events)
    print(f"events: {len(events)} total")

    selected = events[: args.max_events]
    market_counter = Counter()
    bookmaker_counter = Counter()
    samples = defaultdict(list)
    per_event_market_errors = []

    for ev in selected:
        eid = ev.get("id")
        if not eid:
            continue
        r = session.get(f"{BASE}/sports/{SPORT}/events/{eid}/odds",
                         params={"apiKey": key, "bookmakers": args.bookmakers,
                                 "markets": ",".join(CANDIDATE_MARKETS),
                                 "oddsFormat": "american"}, timeout=25)
        if not r.ok:
            per_event_market_errors.append({"event_id": eid, "status": r.status_code,
                                             "body": r.text[:300]})
            continue
        payload = r.json()
        for bm in payload.get("bookmakers") or []:
            bookmaker_counter[bm.get("key")] += 1
            for mkt in bm.get("markets") or []:
                mk = mkt.get("key")
                market_counter[mk] += 1
                if len(samples[mk]) < 3:
                    samples[mk].append({
                        "event": f"{ev.get('away_team')} @ {ev.get('home_team')}",
                        "bookmaker": bm.get("key"),
                        "outcomes": (mkt.get("outcomes") or [])[:6],
                    })

    result["events_checked"] = len(selected)
    result["remaining_requests_header"] = events_r.headers.get("x-requests-remaining")
    result["bookmakers_seen"] = dict(bookmaker_counter)
    result["markets_seen"] = dict(market_counter)
    result["samples_by_market"] = dict(samples)
    result["per_event_errors"] = per_event_market_errors[:5]

    print(f"remaining API requests: {result['remaining_requests_header']}")
    print(f"bookmakers seen: {result['bookmakers_seen']}")
    print(f"markets seen: {result['markets_seen']}")
    for mk, ex in samples.items():
        print(f"\n--- {mk} sample ---")
        print(json.dumps(ex[0], indent=2)[:800])
    if per_event_market_errors:
        print(f"\n{len(per_event_market_errors)} events errored, e.g.: {per_event_market_errors[0]}")

    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
