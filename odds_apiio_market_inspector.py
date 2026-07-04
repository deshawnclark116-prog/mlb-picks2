#!/usr/bin/env python3
"""
odds_apiio_market_inspector.py

Standalone diagnostic for Odds-API.io MLB/FanDuel markets.

Purpose:
- Do NOT change api.py.
- Prefer reading the latest saved v2 probe output.
- If raw odds are not present, use a tiny live sample to inspect exactly how Odds-API.io names pitcher strikeout markets.

Env:
  ODDS_API_IO_KEY

Examples:
  python odds_apiio_market_inspector.py
  python odds_apiio_market_inspector.py --live --max-events 3
  python odds_apiio_market_inspector.py --live --max-events 3 --bookmaker FanDuel
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import requests
except Exception as e:  # pragma: no cover
    requests = None

BASE = "https://api.odds-api.io/v3"
OUT_DIR = Path("/data/odds_provider_tests")
LATEST_V2 = OUT_DIR / "odds_apiio_k_line_probe_v2_latest.json"
OUT_JSON = OUT_DIR / "odds_apiio_market_inspector_latest.json"

KEY_CANDIDATES = ("apiKey", "apikey", "api_key", "key")

K_TERMS = re.compile(r"(strike|strikeout|pitcher|k\b|\bks\b)", re.I)


def shape(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return {"type": "dict", "len": len(x), "first_keys": list(x.keys())[:20]}
    if isinstance(x, list):
        return {"type": "list", "len": len(x), "first_type": type(x[0]).__name__ if x else None, "first_shape": shape(x[0]) if x else None}
    return {"type": type(x).__name__, "repr": repr(x)[:200]}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def request_json(path: str, params: Dict[str, Any], key: str | None, requests_log: List[Dict[str, Any]]) -> Any:
    if requests is None:
        raise RuntimeError("requests is not installed")
    url = BASE + path
    # Odds-API.io examples commonly use apiKey; v2 used this successfully.
    if key:
        params = dict(params)
        params.setdefault("apiKey", key)
    r = requests.get(url, params=params, timeout=30)
    item = {
        "path": path,
        "status_code": r.status_code,
        "url_masked": mask_url(r.url),
        "text_preview": r.text[:500],
    }
    try:
        data = r.json()
        item["shape"] = shape(data)
    except Exception:
        data = {"_non_json_text": r.text[:2000]}
        item["shape"] = shape(data)
    requests_log.append(item)
    return data


def mask_url(url: str) -> str:
    # Mask API key-ish query params.
    for k in KEY_CANDIDATES:
        url = re.sub(rf"({k}=)[^&]+", rf"\1***", url, flags=re.I)
    return url


def deep_find_lists(obj: Any, path: str = "") -> Iterable[Tuple[str, List[Any]]]:
    if isinstance(obj, list):
        yield path or "$", obj
        for i, v in enumerate(obj[:5]):
            yield from deep_find_lists(v, f"{path}[{i}]")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield from deep_find_lists(v, p)


def deep_strings(obj: Any, path: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if isinstance(k, str):
                yield (p + "<key>", k)
            yield from deep_strings(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from deep_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def candidate_bookmaker_nodes(obj: Any) -> List[Tuple[str, Any]]:
    hits: List[Tuple[str, Any]] = []
    def walk(x: Any, path: str = "$"):
        if isinstance(x, dict):
            lower_keys = {str(k).lower(): k for k in x.keys()}
            # Looks like an event with bookmakers or a bookmaker object.
            if "bookmakers" in lower_keys:
                hits.append((f"{path}.{lower_keys['bookmakers']}", x[lower_keys["bookmakers"]]))
            if any(k in lower_keys for k in ("bookmaker", "bookmakerid", "bookmaker_id", "name", "id")) and any(k in lower_keys for k in ("markets", "odds", "bets")):
                hits.append((path, x))
            for k, v in x.items():
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
    walk(obj)
    return hits


def extract_market_like_nodes(obj: Any) -> List[Tuple[str, Dict[str, Any]]]:
    nodes: List[Tuple[str, Dict[str, Any]]] = []
    def walk(x: Any, path: str = "$"):
        if isinstance(x, dict):
            keys = {str(k).lower(): k for k in x.keys()}
            # Market-ish if it has a name/key/id and some outcome/odds/prices children.
            if any(k in keys for k in ("market", "marketname", "market_name", "marketid", "market_id", "key", "name", "label", "type")) and any(k in keys for k in ("outcomes", "odds", "prices", "selections", "bets")):
                nodes.append((path, x))
            for k, v in x.items():
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
    walk(obj)
    return nodes


def summarize_markets(obj: Any) -> Dict[str, Any]:
    market_nodes = extract_market_like_nodes(obj)
    market_names = Counter()
    market_samples = []

    for path, node in market_nodes:
        keys = {str(k).lower(): k for k in node.keys()}
        name = None
        for nk in ("market", "marketname", "market_name", "marketid", "market_id", "key", "name", "label", "type"):
            if nk in keys and isinstance(node.get(keys[nk]), (str, int, float)):
                val = str(node.get(keys[nk]))
                # Avoid generic labels like "FanDuel" if possible.
                if val and val.lower() not in ("fanduel", "fd"):
                    name = val
                    break
        if not name:
            # fallback: first short string in node
            for _, s in deep_strings(node):
                if 1 <= len(s) <= 80:
                    name = s
                    break
        if name:
            market_names[name] += 1
            if len(market_samples) < 30:
                market_samples.append({"path": path, "market_name_guess": name, "keys": list(node.keys())[:30], "sample": compact(node, max_chars=1500)})

    k_string_hits = []
    for path, s in deep_strings(obj):
        if K_TERMS.search(s):
            k_string_hits.append({"path": path, "value": s[:300]})
            if len(k_string_hits) >= 100:
                break

    bookmaker_names = Counter()
    for path, node in candidate_bookmaker_nodes(obj):
        if isinstance(node, list):
            for item in node:
                if isinstance(item, dict):
                    nm = pick_name(item)
                    if nm:
                        bookmaker_names[nm] += 1
        elif isinstance(node, dict):
            nm = pick_name(node)
            if nm:
                bookmaker_names[nm] += 1

    return {
        "market_node_count": len(market_nodes),
        "market_names_top": market_names.most_common(100),
        "market_samples": market_samples,
        "k_string_hits_count": len(k_string_hits),
        "k_string_hits_sample": k_string_hits[:50],
        "bookmaker_names_top": bookmaker_names.most_common(50),
    }


def pick_name(d: Dict[str, Any]) -> str | None:
    for k in ("name", "title", "bookmaker", "bookmakerName", "bookmaker_name", "bookmakerId", "id", "slug"):
        if k in d and isinstance(d[k], (str, int, float)):
            return str(d[k])
    return None


def compact(x: Any, max_chars: int = 2000) -> str:
    try:
        txt = json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        txt = repr(x)
    return txt[:max_chars]


def live_fetch(max_events: int, bookmaker: str, key: str | None) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]]]:
    log: List[Dict[str, Any]] = []
    events = request_json("/events", {"sport": "baseball", "league": "usa-mlb", "bookmaker": bookmaker}, key, log)
    if not isinstance(events, list):
        return events, log, []
    ids = [str(e.get("id")) for e in events if isinstance(e, dict) and e.get("id")][:max_events]
    odds_data: List[Dict[str, Any]] = []
    if ids:
        # Use odds/multi in batches of 10. Try likely param names, one should work.
        batch = ids[:max_events]
        data = request_json("/odds/multi", {"eventIds": ",".join(batch), "bookmaker": bookmaker}, key, log)
        if looks_empty_or_error(data):
            # Alternate param casing. Costs one extra request only if needed.
            data2 = request_json("/odds/multi", {"event_ids": ",".join(batch), "bookmaker": bookmaker}, key, log)
            data = data2
        odds_data = data if isinstance(data, list) else [data]
    return {"events": events, "odds": odds_data}, log, odds_data


def looks_empty_or_error(data: Any) -> bool:
    if data is None:
        return True
    if isinstance(data, dict) and any(k.lower() in ("error", "message", "success") for k in data.keys()):
        return True
    if isinstance(data, list) and len(data) == 0:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="Make a tiny live Odds-API.io sample request if saved raw data is not enough.")
    ap.add_argument("--max-events", type=int, default=3)
    ap.add_argument("--bookmaker", default="FanDuel")
    ap.add_argument("--input", default=str(LATEST_V2))
    args = ap.parse_args()

    key = os.environ.get("ODDS_API_IO_KEY") or os.environ.get("ODDSAPIIO_KEY")
    result: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "has_key": bool(key),
        "mode": "saved_file",
        "input": args.input,
        "live_requests": [],
        "raw_shape": None,
        "summary": None,
    }

    data: Any = None
    input_path = Path(args.input)
    if input_path.exists():
        try:
            data = read_json(input_path)
            result["raw_shape"] = shape(data)
        except Exception as e:
            result["saved_file_error"] = repr(e)

    # If saved file appears to have no raw odds, or user asked live, make tiny live request.
    if args.live or data is None:
        result["mode"] = "live"
        data, reqlog, odds_data = live_fetch(args.max_events, args.bookmaker, key)
        result["live_requests"] = reqlog
        result["raw_shape"] = shape(data)

    result["summary"] = summarize_markets(data)
    save_json(OUT_JSON, result)

    print("\n=== ODDS-API.IO MARKET INSPECTOR ===\n")
    print(f"Generated UTC: {result['generated_utc']}")
    print(f"Mode: {result['mode']} has_key={result['has_key']} bookmaker={args.bookmaker}")
    print(f"Input: {args.input}")
    print(f"Raw shape: {result['raw_shape']}")
    print(f"Live requests made: {len(result['live_requests'])}")
    for r in result["live_requests"]:
        print(f"- {r.get('status_code')} {r.get('path')} shape={r.get('shape')} url={r.get('url_masked')}")

    s = result["summary"] or {}
    print("\nBOOKMAKERS SEEN")
    if s.get("bookmaker_names_top"):
        for name, cnt in s["bookmaker_names_top"][:20]:
            print(f"{cnt:>4}  {name}")
    else:
        print("None detected")

    print("\nMARKET NAMES SEEN")
    if s.get("market_names_top"):
        for name, cnt in s["market_names_top"][:60]:
            print(f"{cnt:>4}  {name}")
    else:
        print("None detected")

    print("\nK / STRIKEOUT STRING HITS")
    print(f"count={s.get('k_string_hits_count', 0)}")
    for item in (s.get("k_string_hits_sample") or [])[:30]:
        print(f"- {item['path']}: {item['value']}")

    print("\nMARKET SAMPLE PATHS")
    for item in (s.get("market_samples") or [])[:10]:
        print(f"- {item['path']} | {item['market_name_guess']} | keys={item['keys']}")

    print("\nOUTPUT")
    print(str(OUT_JSON))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
