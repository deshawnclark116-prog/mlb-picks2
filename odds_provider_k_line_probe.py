#!/usr/bin/env python3
"""
Standalone K-line provider probe for Prop Edge.

Goal:
  Compare Odds-API.io and OddsPapi for ONE job only:
  "Can this provider return today's FanDuel main MLB pitcher strikeout line?"

This file does NOT modify api.py, predictions, record.json, grading, or PropLine usage.
It only calls the provider(s) for which you have set API keys.

Recommended env vars:
  ODDS_API_IO_KEY   -> Odds-API.io key
  ODDSPAPI_KEY      -> OddsPapi key

Optional env vars:
  ODDSPAPI_MLB_TOURNAMENT_IDS  -> comma-separated OddsPapi MLB tournament IDs, if discovery fails
  ODDSPAPI_BASE_URL            -> default https://api.oddspapi.io/v4
  ODDS_API_IO_BASE_URL         -> default https://api.odds-api.io/v3

Example:
  python odds_provider_k_line_probe.py --provider both --max-events 15
  python odds_provider_k_line_probe.py --provider oddsapiio --max-events 15
  python odds_provider_k_line_probe.py --provider oddspapi --max-events 15
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

try:
    import requests
except Exception as exc:  # pragma: no cover
    print("ERROR: requests is required. Install with: pip install requests")
    raise

OUT_DIR = Path("/data/odds_provider_tests")
OUT_JSON = OUT_DIR / "odds_provider_k_line_probe_latest.json"
OUT_CSV = OUT_DIR / "odds_provider_k_line_probe_latest.csv"

DEFAULT_ODDS_API_IO_BASE = "https://api.odds-api.io/v3"
DEFAULT_ODDSPAPI_BASE = "https://api.oddspapi.io/v4"

K_TERMS = ("strikeout", "strikeouts", "pitcher strikeout", "pitcher strikeouts", "total pitcher strikeouts")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def redact_url(url: str) -> str:
    url = re.sub(r"([?&](?:apiKey|apikey|api_key|key|token)=)[^&]+", r"\1<REDACTED>", url, flags=re.I)
    return url


def compact_json_shape(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, list):
        return {"type": "list", "len": len(obj), "first_type": type(obj[0]).__name__ if obj else None}
    if isinstance(obj, dict):
        keys = list(obj.keys())
        return {"type": "dict", "keys": keys[:25], "key_count": len(keys)}
    return {"type": type(obj).__name__}


def http_get_json(session: requests.Session, url: str, params: Dict[str, Any], timeout: int = 20) -> Tuple[Dict[str, Any], Any]:
    started = time.time()
    try:
        resp = session.get(url, params=params, timeout=timeout)
        elapsed_ms = round((time.time() - started) * 1000, 1)
        meta = {
            "url": redact_url(resp.url),
            "status_code": resp.status_code,
            "ok": resp.ok,
            "content_type": resp.headers.get("content-type", ""),
            "elapsed_ms": elapsed_ms,
        }
        try:
            payload = resp.json()
            meta["parse_mode"] = "json"
            meta["shape"] = compact_json_shape(payload)
        except Exception:
            payload = None
            meta["parse_mode"] = "text"
            meta["text_preview"] = resp.text[:800]
        return meta, payload
    except Exception as exc:
        return {
            "url": redact_url(url),
            "status_code": None,
            "ok": False,
            "error": repr(exc),
        }, None


def as_list(payload: Any) -> List[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "events", "fixtures", "results", "items"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("events", "fixtures", "results", "items"):
                val = data.get(key)
                if isinstance(val, list):
                    return val
    return []


def lower_text(x: Any) -> str:
    return str(x or "").strip().lower()


def contains_k_market(text: Any) -> bool:
    s = lower_text(text)
    return any(term in s for term in K_TERMS)


def parse_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def side_from_text(x: Any) -> Optional[str]:
    s = lower_text(x)
    if "over" in s or s in {"o"}:
        return "over"
    if "under" in s or s in {"u"}:
        return "under"
    return None


def norm_book_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", lower_text(name))


def pair_lines(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pair over/under entries where possible, preserving singletons too."""
    groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    singles: List[Dict[str, Any]] = []

    for row in rows:
        key = (
            row.get("provider"),
            row.get("event_id") or row.get("fixture_id"),
            row.get("bookmaker"),
            row.get("player"),
            row.get("market_name"),
            row.get("line"),
        )
        side = lower_text(row.get("side"))
        if side not in ("over", "under"):
            singles.append(row)
            continue
        g = groups.setdefault(key, {
            "provider": row.get("provider"),
            "event_id": row.get("event_id"),
            "fixture_id": row.get("fixture_id"),
            "event": row.get("event"),
            "start_time": row.get("start_time"),
            "bookmaker": row.get("bookmaker"),
            "player": row.get("player"),
            "market_name": row.get("market_name"),
            "line": row.get("line"),
            "main_line": row.get("main_line"),
            "raw_market_id": row.get("raw_market_id"),
            "source": row.get("source"),
            "over_price": None,
            "under_price": None,
            "over_raw": None,
            "under_raw": None,
        })
        if side == "over":
            g["over_price"] = row.get("price")
            g["over_raw"] = row.get("raw")
        elif side == "under":
            g["under_price"] = row.get("price")
            g["under_raw"] = row.get("raw")
        # If any paired row says main_line true, keep true.
        if row.get("main_line") is True:
            g["main_line"] = True

    paired = list(groups.values())
    # Keep paired first, then singletons for debugging.
    return paired + singles


# ---------------------------- Odds-API.io ----------------------------


def parse_oddsapiio_event(event: Dict[str, Any]) -> Tuple[str, str, str]:
    eid = str(event.get("id") or event.get("eventId") or event.get("event_id") or "")
    home = event.get("home") or event.get("home_team") or event.get("homeTeam") or ""
    away = event.get("away") or event.get("away_team") or event.get("awayTeam") or ""
    name = event.get("name") or event.get("event") or (f"{away} @ {home}" if home or away else eid)
    start = event.get("date") or event.get("commence_time") or event.get("startTime") or event.get("start_time") or ""
    return eid, str(name), str(start)


def probe_oddsapiio(args: argparse.Namespace) -> Dict[str, Any]:
    key = os.getenv("ODDS_API_IO_KEY") or os.getenv("ODDSAPI_IO_KEY") or ""
    base = os.getenv("ODDS_API_IO_BASE_URL", DEFAULT_ODDS_API_IO_BASE).rstrip("/")
    result: Dict[str, Any] = {
        "provider": "odds-api.io",
        "has_key": bool(key),
        "base": base,
        "request_count": 0,
        "requests": [],
        "events_count": 0,
        "events_sample": [],
        "raw_k_rows_count": 0,
        "k_lines_count": 0,
        "k_lines": [],
        "error": None,
    }
    if not key:
        result["error"] = "Missing ODDS_API_IO_KEY env var."
        return result

    session = requests.Session()
    session.headers.update({"User-Agent": "PropEdge-KLineProbe/1.0"})

    # Docs show /v3/events with apiKey + sport=mlb.
    meta, payload = http_get_json(session, f"{base}/events", {"apiKey": key, "sport": args.oddsapiio_sport})
    result["request_count"] += 1
    result["requests"].append(meta)
    events = as_list(payload)
    result["events_count"] = len(events)
    result["events_sample"] = [dict(zip(("id", "event", "start_time"), parse_oddsapiio_event(e))) for e in events[:5] if isinstance(e, dict)]
    if not events:
        result["error"] = "No events found or response was not an event list."
        result["events_payload_shape"] = compact_json_shape(payload)
        return result

    rows: List[Dict[str, Any]] = []
    for event in events[: args.max_events]:
        if not isinstance(event, dict):
            continue
        eid, event_name, start_time = parse_oddsapiio_event(event)
        if not eid:
            continue
        if args.sleep:
            time.sleep(args.sleep)
        # Docs show /v3/odds with eventId + bookmakers=FanDuel.
        params = {"apiKey": key, "eventId": eid, "bookmakers": args.oddsapiio_bookmaker}
        meta, odds_payload = http_get_json(session, f"{base}/odds", params)
        result["request_count"] += 1
        result["requests"].append(meta)
        if not isinstance(odds_payload, dict):
            continue
        bookmakers = odds_payload.get("bookmakers") or {}
        if not isinstance(bookmakers, dict):
            continue
        wanted_book_norm = norm_book_name(args.oddsapiio_bookmaker)
        for book_name, markets in bookmakers.items():
            if wanted_book_norm not in norm_book_name(str(book_name)) and norm_book_name(str(book_name)) not in wanted_book_norm:
                continue
            if not isinstance(markets, list):
                continue
            for market in markets:
                if not isinstance(market, dict):
                    continue
                market_name = str(market.get("name") or market.get("marketName") or market.get("key") or "")
                if not contains_k_market(market_name):
                    continue
                odds_list = market.get("odds") or market.get("outcomes") or []
                if not isinstance(odds_list, list):
                    continue
                for prop in odds_list:
                    if not isinstance(prop, dict):
                        continue
                    player = prop.get("label") or prop.get("player") or prop.get("description") or prop.get("name")
                    line = parse_float(prop.get("hdp") if "hdp" in prop else prop.get("point"))
                    over = prop.get("over") if "over" in prop else None
                    under = prop.get("under") if "under" in prop else None
                    # Odds-API.io example format has one prop row containing both over + under.
                    if over is not None or under is not None:
                        if over is not None:
                            rows.append({
                                "provider": "odds-api.io",
                                "event_id": eid,
                                "event": event_name,
                                "start_time": start_time,
                                "bookmaker": str(book_name),
                                "player": str(player or ""),
                                "market_name": market_name,
                                "line": line,
                                "side": "over",
                                "price": over,
                                "main_line": True,
                                "source": "/v3/odds",
                                "raw": {"prop": prop, "market_name": market_name},
                            })
                        if under is not None:
                            rows.append({
                                "provider": "odds-api.io",
                                "event_id": eid,
                                "event": event_name,
                                "start_time": start_time,
                                "bookmaker": str(book_name),
                                "player": str(player or ""),
                                "market_name": market_name,
                                "line": line,
                                "side": "under",
                                "price": under,
                                "main_line": True,
                                "source": "/v3/odds",
                                "raw": {"prop": prop, "market_name": market_name},
                            })
                    else:
                        side = side_from_text(prop.get("side") or prop.get("name") or prop.get("label"))
                        price = prop.get("price") or prop.get("odds") or prop.get("decimal") or prop.get("american")
                        rows.append({
                            "provider": "odds-api.io",
                            "event_id": eid,
                            "event": event_name,
                            "start_time": start_time,
                            "bookmaker": str(book_name),
                            "player": str(player or ""),
                            "market_name": market_name,
                            "line": line,
                            "side": side,
                            "price": price,
                            "main_line": True,
                            "source": "/v3/odds",
                            "raw": {"prop": prop, "market_name": market_name},
                        })

    result["raw_k_rows_count"] = len(rows)
    paired = pair_lines(rows)
    result["k_lines"] = paired
    result["k_lines_count"] = len(paired)
    return result


# ---------------------------- OddsPapi ----------------------------


def oddspapi_get(session: requests.Session, base: str, endpoint: str, key: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    p = dict(params)
    p["apiKey"] = key
    return http_get_json(session, f"{base}{endpoint}", p)


def find_baseball_sport_id(sports: List[Any]) -> Optional[int]:
    for s in sports:
        if not isinstance(s, dict):
            continue
        slug = lower_text(s.get("slug"))
        name = lower_text(s.get("sportName") or s.get("name"))
        if slug == "baseball" or name == "baseball" or "baseball" in slug or "baseball" in name:
            sid = s.get("sportId") or s.get("id")
            try:
                return int(sid)
            except Exception:
                return None
    return None


def find_mlb_tournament_ids(tournaments: List[Any]) -> List[int]:
    found: List[int] = []
    for t in tournaments:
        if not isinstance(t, dict):
            continue
        hay = " ".join(str(t.get(k, "")) for k in ("tournamentSlug", "tournamentName", "categorySlug", "categoryName")).lower()
        if "mlb" in hay or "major league baseball" in hay or "regular season" in hay and "baseball" in hay:
            tid = t.get("tournamentId") or t.get("id")
            try:
                found.append(int(tid))
            except Exception:
                pass
    # Prefer exact MLB-like names first but preserve order.
    return found


def build_market_maps(markets_payload: Any, sport_id: Optional[int]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    markets = as_list(markets_payload)
    market_map: Dict[str, Dict[str, Any]] = {}
    k_markets: List[Dict[str, Any]] = []
    for m in markets:
        if not isinstance(m, dict):
            continue
        mid = m.get("marketId") or m.get("id")
        if mid is None:
            continue
        market_map[str(mid)] = m
        if sport_id is not None:
            try:
                if int(m.get("sportId")) != int(sport_id):
                    continue
            except Exception:
                pass
        hay = " ".join(str(m.get(k, "")) for k in ("marketName", "marketType", "period"))
        if bool(m.get("playerProp")) and contains_k_market(hay):
            k_markets.append(m)
    return market_map, k_markets


def parse_oddspapi_rows(fixtures_payload: Any, market_map: Dict[str, Dict[str, Any]], bookmaker_wanted: str) -> List[Dict[str, Any]]:
    fixtures = as_list(fixtures_payload)
    rows: List[Dict[str, Any]] = []
    wanted_norm = norm_book_name(bookmaker_wanted)

    for fx in fixtures:
        if not isinstance(fx, dict):
            continue
        fixture_id = str(fx.get("fixtureId") or fx.get("id") or "")
        event = fx.get("event") or f"{fx.get('participant1Name') or fx.get('participant1ShortName') or ''} vs {fx.get('participant2Name') or fx.get('participant2ShortName') or ''}".strip()
        start_time = fx.get("startTime") or fx.get("date") or ""
        bookmaker_odds = fx.get("bookmakerOdds") or {}
        if not isinstance(bookmaker_odds, dict):
            continue
        for book_key, book_obj in bookmaker_odds.items():
            if wanted_norm not in norm_book_name(str(book_key)) and norm_book_name(str(book_key)) not in wanted_norm:
                continue
            if not isinstance(book_obj, dict):
                continue
            markets = book_obj.get("markets") or {}
            if not isinstance(markets, dict):
                continue
            for raw_market_id, market_obj in markets.items():
                if not isinstance(market_obj, dict):
                    continue
                meta = market_map.get(str(raw_market_id), {})
                market_name = str(meta.get("marketName") or market_obj.get("marketName") or "")
                hay = " ".join([
                    market_name,
                    str(meta.get("marketType", "")),
                    str(market_obj.get("bookmakerMarketId", "")),
                    str(raw_market_id),
                ])
                # If we have market metadata, require a K-market name.
                # If no metadata, still allow rows whose raw strings mention strikeout.
                if not contains_k_market(hay):
                    continue
                outcomes = market_obj.get("outcomes") or {}
                if not isinstance(outcomes, dict):
                    continue
                for outcome_id, outcome_obj in outcomes.items():
                    if not isinstance(outcome_obj, dict):
                        continue
                    outcome_meta_name = ""
                    for out_meta in meta.get("outcomes", []) if isinstance(meta.get("outcomes"), list) else []:
                        if str(out_meta.get("outcomeId")) == str(outcome_id):
                            outcome_meta_name = str(out_meta.get("outcomeName") or "")
                            break
                    players = outcome_obj.get("players") or {}
                    if not isinstance(players, dict):
                        continue
                    for player_id, player_obj in players.items():
                        if not isinstance(player_obj, dict):
                            continue
                        player_name = player_obj.get("playerName") or player_obj.get("name") or ""
                        bookmaker_outcome_id = player_obj.get("bookmakerOutcomeId")
                        side = side_from_text(bookmaker_outcome_id) or side_from_text(outcome_meta_name)
                        line = parse_float(bookmaker_outcome_id)
                        rows.append({
                            "provider": "oddspapi",
                            "fixture_id": fixture_id,
                            "event": event,
                            "start_time": start_time,
                            "bookmaker": str(book_key),
                            "player": str(player_name or ""),
                            "market_name": market_name or f"market_{raw_market_id}",
                            "line": line,
                            "side": side,
                            "price": player_obj.get("priceAmerican") or player_obj.get("price") or player_obj.get("priceDecimal"),
                            "main_line": player_obj.get("mainLine"),
                            "raw_market_id": str(raw_market_id),
                            "source": "/v4/odds-by-tournaments",
                            "raw": {
                                "outcome_id": outcome_id,
                                "player_id": player_id,
                                "bookmakerOutcomeId": bookmaker_outcome_id,
                                "player": player_obj,
                                "market_meta": meta,
                            },
                        })
    return rows


def probe_oddspapi(args: argparse.Namespace) -> Dict[str, Any]:
    key = os.getenv("ODDSPAPI_KEY") or os.getenv("ODDS_PAPI_KEY") or ""
    base = os.getenv("ODDSPAPI_BASE_URL", DEFAULT_ODDSPAPI_BASE).rstrip("/")
    result: Dict[str, Any] = {
        "provider": "oddspapi",
        "has_key": bool(key),
        "base": base,
        "request_count": 0,
        "requests": [],
        "sport_id": args.oddspapi_sport_id,
        "tournament_ids": [],
        "k_market_ids_seen": [],
        "raw_k_rows_count": 0,
        "k_lines_count": 0,
        "k_lines": [],
        "error": None,
    }
    if not key:
        result["error"] = "Missing ODDSPAPI_KEY env var."
        return result

    session = requests.Session()
    session.headers.update({"User-Agent": "PropEdge-KLineProbe/1.0"})

    sport_id = args.oddspapi_sport_id
    tournament_ids: List[int] = []

    env_tids = os.getenv("ODDSPAPI_MLB_TOURNAMENT_IDS", "").strip()
    if args.oddspapi_tournament_ids:
        env_tids = args.oddspapi_tournament_ids
    if env_tids:
        for x in env_tids.split(","):
            try:
                tournament_ids.append(int(x.strip()))
            except Exception:
                pass

    if sport_id is None and not tournament_ids:
        meta, sports_payload = oddspapi_get(session, base, "/sports", key, {"language": "en"})
        result["request_count"] += 1
        result["requests"].append(meta)
        sports = as_list(sports_payload)
        sport_id = find_baseball_sport_id(sports)
        result["sport_id"] = sport_id
        result["sports_sample"] = sports[:10]
        if args.sleep:
            time.sleep(max(args.sleep, 1.0))

    if not tournament_ids and sport_id is not None:
        meta, tournaments_payload = oddspapi_get(session, base, "/tournaments", key, {"sportId": sport_id, "language": "en"})
        result["request_count"] += 1
        result["requests"].append(meta)
        tournaments = as_list(tournaments_payload)
        tournament_ids = find_mlb_tournament_ids(tournaments)
        result["tournaments_sample"] = tournaments[:20]
        if args.sleep:
            time.sleep(max(args.sleep, 1.0))

    result["tournament_ids"] = tournament_ids
    if not tournament_ids:
        result["error"] = "Could not discover MLB tournamentIds. Set ODDSPAPI_MLB_TOURNAMENT_IDS or use --oddspapi-tournament-ids."
        return result

    # Fetch markets once so numeric market IDs can be mapped to names.
    meta, markets_payload = oddspapi_get(session, base, "/markets", key, {"language": "en"})
    result["request_count"] += 1
    result["requests"].append(meta)
    market_map, k_markets = build_market_maps(markets_payload, sport_id)
    result["k_market_ids_seen"] = [{
        "marketId": m.get("marketId"),
        "marketName": m.get("marketName"),
        "sportId": m.get("sportId"),
        "playerProp": m.get("playerProp"),
        "handicap": m.get("handicap"),
        "marketType": m.get("marketType"),
    } for m in k_markets[:50]]
    if args.sleep:
        time.sleep(max(args.sleep, 1.0))

    # Query the tournament odds for FanDuel. OddsPapi docs use bookmakers param in endpoint detail.
    params = {
        "tournamentIds": ",".join(str(x) for x in tournament_ids[: args.max_tournaments]),
        "bookmakers": args.oddspapi_bookmaker,
        "language": "en",
        "verbosity": args.oddspapi_verbosity,
        "oddsFormat": "american",
    }
    meta, odds_payload = oddspapi_get(session, base, "/odds-by-tournaments", key, params)
    result["request_count"] += 1
    result["requests"].append(meta)

    rows = parse_oddspapi_rows(odds_payload, market_map, args.oddspapi_bookmaker)

    # If no rows and response looks empty, try singular bookmaker param because the overview page uses it.
    if not rows and args.oddspapi_retry_singular_bookmaker:
        if args.sleep:
            time.sleep(max(args.sleep, 1.0))
        params2 = dict(params)
        params2.pop("bookmakers", None)
        params2["bookmaker"] = args.oddspapi_bookmaker
        meta2, odds_payload2 = oddspapi_get(session, base, "/odds-by-tournaments", key, params2)
        result["request_count"] += 1
        result["requests"].append(meta2)
        rows = parse_oddspapi_rows(odds_payload2, market_map, args.oddspapi_bookmaker)
        result["singular_bookmaker_retry_used"] = True

    result["raw_k_rows_count"] = len(rows)
    paired = pair_lines(rows)

    # Main-line view: if a provider marks mainLine true anywhere, prefer those for the summary.
    main_rows = [r for r in paired if r.get("main_line") is True]
    if main_rows:
        result["main_k_lines"] = main_rows
        result["main_k_lines_count"] = len(main_rows)
    else:
        result["main_k_lines"] = []
        result["main_k_lines_count"] = 0

    result["k_lines"] = paired[: args.max_lines_output]
    result["k_lines_count"] = len(paired)
    return result


# ---------------------------- Output ----------------------------


def flatten_lines_for_csv(provider_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in provider_result.get("k_lines", []) or []:
        rows.append({
            "provider": r.get("provider") or provider_result.get("provider"),
            "event_id": r.get("event_id") or r.get("fixture_id"),
            "event": r.get("event"),
            "start_time": r.get("start_time"),
            "bookmaker": r.get("bookmaker"),
            "player": r.get("player"),
            "market_name": r.get("market_name"),
            "line": r.get("line"),
            "main_line": r.get("main_line"),
            "over_price": r.get("over_price") if "over_price" in r else (r.get("price") if r.get("side") == "over" else None),
            "under_price": r.get("under_price") if "under_price" in r else (r.get("price") if r.get("side") == "under" else None),
            "source": r.get("source"),
        })
    return rows


def save_outputs(report: Dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    csv_rows: List[Dict[str, Any]] = []
    for pr in report.get("providers", {}).values():
        if isinstance(pr, dict):
            csv_rows.extend(flatten_lines_for_csv(pr))
    if csv_rows:
        fields = ["provider", "event_id", "event", "start_time", "bookmaker", "player", "market_name", "line", "main_line", "over_price", "under_price", "source"]
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(csv_rows)


def print_summary(report: Dict[str, Any]) -> None:
    print("=== ODDS PROVIDER K LINE PROBE ===")
    print(f"Generated UTC: {report['generated_at_utc']}")
    print(f"Target: FanDuel main MLB pitcher strikeouts line")
    print(f"Total requests made: {report.get('total_request_count')}")
    print("")
    for name, pr in report.get("providers", {}).items():
        print(f"--- {name.upper()} ---")
        print(f"has_key={pr.get('has_key')} request_count={pr.get('request_count')} error={pr.get('error')}")
        if name == "oddsapiio":
            print(f"events_count={pr.get('events_count')} raw_k_rows={pr.get('raw_k_rows_count')} k_lines={pr.get('k_lines_count')}")
        if name == "oddspapi":
            print(f"sport_id={pr.get('sport_id')} tournament_ids={pr.get('tournament_ids')} raw_k_rows={pr.get('raw_k_rows_count')} k_lines={pr.get('k_lines_count')} main_k_lines={pr.get('main_k_lines_count')}")
            if not pr.get("k_market_ids_seen"):
                print("k_market_ids_seen=0 or not mapped")
            else:
                print("k_market_ids_seen sample:")
                for m in pr.get("k_market_ids_seen", [])[:8]:
                    print(f"  marketId={m.get('marketId')} name={m.get('marketName')} handicap={m.get('handicap')}")
        lines = pr.get("main_k_lines") or pr.get("k_lines") or []
        if not lines:
            print("No K lines found.")
        else:
            print("Sample K lines:")
            for i, r in enumerate(lines[:20], start=1):
                over = r.get("over_price") if "over_price" in r else None
                under = r.get("under_price") if "under_price" in r else None
                print(f"{i:02d}. {r.get('player')} | line={r.get('line')} | over={over} under={under} | main={r.get('main_line')} | market={r.get('market_name')} | event={r.get('event')}")
        print("")
    print("--- OUTPUTS ---")
    print(str(OUT_JSON))
    if OUT_CSV.exists():
        print(str(OUT_CSV))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare Odds-API.io and OddsPapi for FanDuel MLB pitcher strikeout lines.")
    p.add_argument("--provider", choices=["both", "oddsapiio", "oddspapi"], default="both")
    p.add_argument("--max-events", type=int, default=15, help="Max Odds-API.io events to query. Default 15.")
    p.add_argument("--max-tournaments", type=int, default=3, help="Max OddsPapi tournamentIds to query. Default 3.")
    p.add_argument("--max-lines-output", type=int, default=200, help="Max lines to store in provider summary. Full raw responses are not stored unless provider includes them in row raw snippets.")
    p.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between requests. Use 1.0 if OddsPapi cooldown complains.")

    p.add_argument("--oddsapiio-sport", default="mlb")
    p.add_argument("--oddsapiio-bookmaker", default="FanDuel")

    p.add_argument("--oddspapi-sport-id", type=int, default=None)
    p.add_argument("--oddspapi-tournament-ids", default="")
    p.add_argument("--oddspapi-bookmaker", default="fanduel")
    p.add_argument("--oddspapi-verbosity", type=int, default=3)
    p.add_argument("--oddspapi-retry-singular-bookmaker", action="store_true", default=True)
    p.add_argument("--no-oddspapi-retry-singular-bookmaker", action="store_false", dest="oddspapi_retry_singular_bookmaker")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    report: Dict[str, Any] = {
        "probe_version": "odds_provider_k_line_probe_v1",
        "generated_at_utc": utc_now(),
        "purpose": "Compare providers for FanDuel main MLB pitcher strikeouts line only.",
        "providers": {},
        "total_request_count": 0,
        "notes": [
            "This standalone probe does not modify api.py, predictions, record.json, or PropLine usage.",
            "Odds-API.io uses /v3/events then /v3/odds per event.",
            "OddsPapi uses sports/tournaments/markets discovery then odds-by-tournaments.",
        ],
    }

    if args.provider in ("both", "oddsapiio"):
        report["providers"]["oddsapiio"] = probe_oddsapiio(args)
    if args.provider in ("both", "oddspapi"):
        report["providers"]["oddspapi"] = probe_oddspapi(args)

    report["total_request_count"] = sum(int(p.get("request_count", 0) or 0) for p in report["providers"].values() if isinstance(p, dict))
    save_outputs(report)
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
