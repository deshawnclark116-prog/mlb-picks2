#!/usr/bin/env python3
"""
odds_apiio_single_event_props_probe.py

Standalone Odds-API.io player-props probe for MLB FanDuel pitcher strikeouts.

Purpose:
- Uses the documented player props endpoint style:
    /v3/odds?apiKey=...&eventId=...&bookmakers=FanDuel
- Does NOT touch api.py.
- Does NOT call PropLine or The Odds API.
- Tests whether Odds-API.io returns FanDuel MLB K props when using single-event odds,
  not /odds/multi.

Env key accepted:
- ODDS_API_IO_KEY
- ODDSAPIIO_KEY
- ODDS_APIIO_KEY

Output files:
- /data/odds_provider_tests/odds_apiio_single_event_props_probe_latest.json
- /data/odds_provider_tests/odds_apiio_single_event_props_probe_latest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

BASE = "https://api.odds-api.io/v3"
OUT_DIR = Path("/data/odds_provider_tests")
OUT_JSON = OUT_DIR / "odds_apiio_single_event_props_probe_latest.json"
OUT_CSV = OUT_DIR / "odds_apiio_single_event_props_probe_latest.csv"

KEY_ENV_NAMES = ["ODDS_API_IO_KEY", "ODDSAPIIO_KEY", "ODDS_APIIO_KEY"]


def get_key() -> Optional[str]:
    for name in KEY_ENV_NAMES:
        val = os.getenv(name)
        if val and val.strip():
            return val.strip()
    return None


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_shape(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, list):
        first = obj[0] if obj else None
        return {
            "type": "list",
            "len": len(obj),
            "first_type": type(first).__name__ if first is not None else None,
            "first_keys": list(first.keys())[:15] if isinstance(first, dict) else None,
        }
    if isinstance(obj, dict):
        return {"type": "dict", "len": len(obj), "first_keys": list(obj.keys())[:20]}
    return {"type": type(obj).__name__}


def req_json(path: str, params: Dict[str, Any], requests_log: List[Dict[str, Any]], label: str) -> Tuple[Optional[Any], Optional[str]]:
    url = f"{BASE}{path}"
    try:
        r = requests.get(url, params=params, timeout=25)
        masked_url = r.url
        key = params.get("apiKey")
        if key:
            masked_url = masked_url.replace(str(key), "***")
        try:
            data = r.json()
        except Exception:
            data = {"_non_json_text_preview": r.text[:800]}
        requests_log.append({
            "label": label,
            "status_code": r.status_code,
            "ok": r.ok,
            "url": masked_url,
            "shape": safe_shape(data),
            "error_preview": data.get("error") if isinstance(data, dict) else None,
            "message_preview": data.get("message") if isinstance(data, dict) else None,
        })
        if not r.ok:
            return data, f"HTTP {r.status_code}: {str(data)[:300]}"
        return data, None
    except Exception as e:
        requests_log.append({"label": label, "error": repr(e), "url": url})
        return None, repr(e)


def normalize_status(ev: Dict[str, Any]) -> str:
    return str(ev.get("status") or ev.get("state") or "").lower().strip()


def parse_dt(s: Any) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def is_future_or_pending(ev: Dict[str, Any]) -> bool:
    status = normalize_status(ev)
    if status in {"pending", "pregame", "pre-game", "scheduled", "preview", "not_started", "not started"}:
        return True
    dt = parse_dt(ev.get("date") or ev.get("commence_time") or ev.get("start_time"))
    return bool(dt and dt > datetime.now(timezone.utc))


def event_name(ev: Dict[str, Any]) -> str:
    away = ev.get("away") or ev.get("away_team") or ev.get("awayName") or ""
    home = ev.get("home") or ev.get("home_team") or ev.get("homeName") or ""
    if away or home:
        return f"{away} @ {home}".strip(" @")
    return str(ev.get("name") or ev.get("title") or ev.get("id") or "")


def pick_mlb_league(leagues: Any) -> Optional[str]:
    if not isinstance(leagues, list):
        return None
    # Prefer exact observed slug from previous successful probes.
    for lg in leagues:
        if not isinstance(lg, dict):
            continue
        slug = str(lg.get("slug") or "").lower()
        name = str(lg.get("name") or "").lower()
        if slug == "usa-mlb" or name == "usa - mlb":
            return str(lg.get("slug"))
    # Fallback: anything MLB but not all-star.
    for lg in leagues:
        if not isinstance(lg, dict):
            continue
        slug = str(lg.get("slug") or "").lower()
        name = str(lg.get("name") or "").lower()
        hay = slug + " " + name
        if "mlb" in hay and "all star" not in hay and "all-star" not in hay:
            return str(lg.get("slug"))
    return None


def iter_paths(obj: Any, path: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield p, v
            yield from iter_paths(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            yield p, v
            yield from iter_paths(v, p)


def extract_books(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict) and isinstance(data.get("bookmakers"), dict):
        return data["bookmakers"]
    if isinstance(data, dict) and isinstance(data.get("bookmakers"), list):
        # Rare alternate shape. Convert list to title/key map.
        out = {}
        for b in data["bookmakers"]:
            if isinstance(b, dict):
                name = b.get("title") or b.get("key") or b.get("name")
                if name:
                    out[str(name)] = b.get("markets") or b.get("odds") or b
        return out
    return {}


def line_from_prop(prop: Dict[str, Any]) -> Optional[float]:
    for key in ("hdp", "line", "point", "points", "total", "handicap"):
        val = prop.get(key)
        if val is None or val == "":
            continue
        try:
            return float(val)
        except Exception:
            pass
    return None


def extract_k_lines_from_odds_event(data: Any, event_meta: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[Dict[str, Any]], List[Dict[str, Any]]]:
    lines: List[Dict[str, Any]] = []
    market_counts: Dict[str, int] = {}
    props_samples: List[Dict[str, Any]] = []
    k_string_hits: List[Dict[str, Any]] = []

    books = extract_books(data)
    # Search all string hits for diagnostics.
    for p, v in iter_paths(data):
        if isinstance(v, str) and re.search(r"strikeout|\bK\b|Total Strikeouts|Pitcher", v, re.I):
            if len(k_string_hits) < 60:
                k_string_hits.append({"path": p, "value": v[:300]})

    for book_name, markets in books.items():
        if str(book_name).lower() != "fanduel":
            continue
        if isinstance(markets, dict):
            # Some shapes may put markets under a key.
            markets_list = markets.get("markets") or markets.get("odds") or []
            if isinstance(markets_list, dict):
                markets_list = list(markets_list.values())
        else:
            markets_list = markets
        if not isinstance(markets_list, list):
            continue

        for mi, market in enumerate(markets_list):
            if not isinstance(market, dict):
                continue
            mname = str(market.get("name") or market.get("label") or market.get("key") or market.get("market") or "")
            if mname:
                market_counts[mname] = market_counts.get(mname, 0) + 1
            odds = market.get("odds") or market.get("outcomes") or market.get("selections") or []
            if isinstance(odds, dict):
                odds = list(odds.values())
            if not isinstance(odds, list):
                continue

            is_props_market = bool(re.search(r"prop", mname, re.I))
            is_k_market_name = bool(re.search(r"strikeout|pitcher.*k|total strikeouts", mname, re.I))
            if is_props_market and len(props_samples) < 15:
                props_samples.append({
                    "market_index": mi,
                    "market_name": mname,
                    "odds_len": len(odds),
                    "sample_odds": odds[:3],
                })

            for oi, prop in enumerate(odds):
                if not isinstance(prop, dict):
                    continue
                label = str(prop.get("label") or prop.get("name") or prop.get("description") or prop.get("player") or "")
                combined = f"{mname} {label}"
                is_k = bool(re.search(r"strikeout|total strikeouts|pitcher strikeouts", combined, re.I))
                if not is_k:
                    continue

                # Extract player. Odds-API.io may use:
                # label: "Yoshinobu Yamamoto (Total Strikeouts)"
                # or market: Player Props - Strikeouts, label: "Yoshinobu Yamamoto"
                player = label
                m = re.match(r"^(.+?)\s*\((?:Total\s+)?Strikeouts\)\s*$", label, flags=re.I)
                if m:
                    player = m.group(1).strip()
                player = re.sub(r"\s+", " ", player).strip()

                line = line_from_prop(prop)
                lines.append({
                    "provider": "oddsapiio",
                    "bookmaker": "FanDuel",
                    "event_id": event_meta.get("id"),
                    "event": event_name(event_meta),
                    "start_time": event_meta.get("date"),
                    "event_status": event_meta.get("status"),
                    "market_name": mname,
                    "market_index": mi,
                    "odds_index": oi,
                    "player": player,
                    "raw_label": label,
                    "line": line,
                    "over": prop.get("over") or prop.get("overOdds") or prop.get("priceOver"),
                    "under": prop.get("under") or prop.get("underOdds") or prop.get("priceUnder"),
                    "raw_prop_keys": list(prop.keys()),
                })

    return lines, market_counts, props_samples, k_string_hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-events", type=int, default=15)
    ap.add_argument("--bookmaker", default="FanDuel")
    ap.add_argument("--sport", default="baseball")
    ap.add_argument("--league", default="usa-mlb")
    ap.add_argument("--include-live", action="store_true", help="Do not filter to future/pending events only.")
    ap.add_argument("--dump-raw-sample", action="store_true", help="Save first raw odds response sample in JSON output.")
    args = ap.parse_args()

    key = get_key()
    requests_log: List[Dict[str, Any]] = []
    notes: List[str] = []
    all_lines: List[Dict[str, Any]] = []
    all_props_samples: List[Dict[str, Any]] = []
    all_k_hits: List[Dict[str, Any]] = []
    market_counts: Dict[str, int] = {}
    raw_sample: Optional[Any] = None

    if not key:
        print("ERROR: Missing Odds-API.io key. Set ODDS_API_IO_KEY.")
        return 2

    # Verify sport/leagues lightly and use league arg.
    leagues, league_err = req_json("/leagues", {"sport": args.sport}, requests_log, "leagues")
    league_used = args.league
    if isinstance(leagues, list):
        picked = pick_mlb_league(leagues)
        if picked:
            league_used = picked
    if league_err:
        notes.append(f"leagues_error={league_err}")

    events, events_err = req_json(
        "/events",
        {"sport": args.sport, "league": league_used, "bookmaker": args.bookmaker, "apiKey": key},
        requests_log,
        "events",
    )
    if events_err:
        notes.append(f"events_error={events_err}")
    if not isinstance(events, list):
        events_list: List[Dict[str, Any]] = []
    else:
        events_list = [e for e in events if isinstance(e, dict)]

    candidates = events_list if args.include_live else [e for e in events_list if is_future_or_pending(e)]
    used_events = candidates[: max(0, args.max_events)]

    for ev in used_events:
        eid = ev.get("id")
        if not eid:
            continue
        odds, odds_err = req_json(
            "/odds",
            {"apiKey": key, "eventId": eid, "bookmakers": args.bookmaker},
            requests_log,
            f"odds:{eid}",
        )
        if odds_err:
            notes.append(f"odds_error_event_{eid}={odds_err}")
            continue
        if raw_sample is None and args.dump_raw_sample:
            raw_sample = odds
        lines, mcounts, psamples, khits = extract_k_lines_from_odds_event(odds, ev)
        all_lines.extend(lines)
        for k, v in mcounts.items():
            market_counts[k] = market_counts.get(k, 0) + v
        for sample in psamples:
            if len(all_props_samples) < 30:
                sample = dict(sample)
                sample["event_id"] = eid
                sample["event"] = event_name(ev)
                all_props_samples.append(sample)
        for hit in khits:
            if len(all_k_hits) < 80:
                hit = dict(hit)
                hit["event_id"] = eid
                hit["event"] = event_name(ev)
                all_k_hits.append(hit)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "probe_version": "oddsapiio-single-event-props-v1",
        "generated_at_utc": now_utc(),
        "base": BASE,
        "has_key": bool(key),
        "sport": args.sport,
        "league_used": league_used,
        "bookmaker": args.bookmaker,
        "request_count": len(requests_log),
        "events_total": len(events_list),
        "pending_or_future": len([e for e in events_list if is_future_or_pending(e)]),
        "events_used_count": len(used_events),
        "events_used": [
            {"id": e.get("id"), "event": event_name(e), "date": e.get("date"), "status": e.get("status")}
            for e in used_events
        ],
        "k_lines_count": len(all_lines),
        "k_lines": all_lines,
        "market_names_seen": dict(sorted(market_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "player_props_samples": all_props_samples,
        "k_string_hits": all_k_hits,
        "requests": requests_log,
        "notes": notes,
    }
    if args.dump_raw_sample:
        result["raw_odds_sample"] = raw_sample

    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        fields = ["provider", "bookmaker", "event_id", "event", "start_time", "event_status", "market_name", "player", "raw_label", "line", "over", "under"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in all_lines:
            w.writerow({k: row.get(k) for k in fields})

    print("=== ODDS-API.IO SINGLE-EVENT PLAYER PROPS PROBE ===")
    print(f"Generated UTC: {result['generated_at_utc']}")
    print("Target: FanDuel MLB pitcher strikeouts via documented /v3/odds?eventId=...&bookmakers=FanDuel")
    print()
    print(f"has_key={bool(key)} sport={args.sport} league={league_used} bookmaker={args.bookmaker}")
    print(f"request_count={len(requests_log)} events_total={len(events_list)} pending_or_future={result['pending_or_future']} events_used={len(used_events)} k_lines_count={len(all_lines)}")
    print()
    print("EVENTS USED")
    for e in result["events_used"][:20]:
        print(f"- {e['id']} | {e['event']} | {e['date']} | {e['status']}")
    print()
    print("MARKET NAMES SEEN")
    if market_counts:
        for name, count in sorted(market_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:40]:
            print(f"- {name}: {count}")
    else:
        print("None detected")
    print()
    print("PLAYER PROPS SAMPLES")
    if all_props_samples:
        for s in all_props_samples[:8]:
            print(f"- {s.get('event')} | {s.get('market_name')} | odds_len={s.get('odds_len')}")
            for odd in s.get("sample_odds") or []:
                print(f"    {json.dumps(odd, ensure_ascii=False)[:300]}")
    else:
        print("None detected")
    print()
    print("K / STRIKEOUT STRING HITS")
    print(f"count={len(all_k_hits)}")
    for hit in all_k_hits[:20]:
        print(f"- {hit.get('event')} | {hit['path']}: {hit['value']}")
    print()
    print("SAMPLE K LINES")
    if all_lines:
        for row in all_lines[:30]:
            print(f"- {row['player']} | line={row['line']} | over={row['over']} | under={row['under']} | {row['event']} | market={row['market_name']}")
    else:
        print("No K lines found.")
    print()
    print("REQUESTS")
    for r in requests_log[:40]:
        print(f"- {r.get('status_code')} {r.get('label')} shape={r.get('shape')} url={r.get('url')}")
        if r.get("error_preview") or r.get("message_preview"):
            print(f"    error={r.get('error_preview')} message={r.get('message_preview')}")
    if len(requests_log) > 40:
        print(f"... {len(requests_log)-40} more requests omitted")
    print()
    print("OUTPUTS")
    print(str(OUT_JSON))
    print(str(OUT_CSV))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
