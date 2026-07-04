#!/usr/bin/env python3
"""
PropLine Quota Audit - standalone, NO NETWORK CALLS.

Purpose:
  Figure out what is most likely burning PropLine request quota by scanning the
  current repo code and estimating request cost for common app/debug actions.

Usage on Render shell:
  python propline_quota_audit.py
  python propline_quota_audit.py --events 15 --markets 14
  python propline_quota_audit.py --json

This script is safe: it does not call PropLine, MLB, FanDuel, or any external API.
It only reads local .py files and /data JSON files when present.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_DEFAULT = Path.cwd()
DATA_DEFAULT = Path("/data")

PROPLINE_TOKENS = (
    "prop-line",
    "PROPLINE",
    "propline",
    "api.prop-line.com",
)

REQUEST_TOKENS = (
    "requests.get",
    "requests.post",
    "requests.request",
    "httpx.get",
    "httpx.post",
    "httpx.request",
    ".get(",
    ".post(",
    ".request(",
)

HIGH_RISK_DEBUG_ENDPOINT_HINTS = (
    "fanduel-market-probe",
    "propline-fetch",
    "line-audit",
    "run/now",
    "run/daily",
)

@dataclass
class PyFileHit:
    file: str
    line_no: int
    kind: str
    text: str

@dataclass
class EndpointInfo:
    file: str
    route: str
    method: str
    function: str
    start_line: int
    end_line: int
    has_propline_token: bool
    request_call_count: int
    propline_token_count: int
    risk: str
    notes: List[str]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_read(path: Path, max_bytes: int = 2_500_000) -> str:
    try:
        data = path.read_bytes()
        if len(data) > max_bytes:
            data = data[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def iter_py_files(root: Path) -> Iterable[Path]:
    skip_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules"}
    for p in root.rglob("*.py"):
        if any(part in skip_dirs for part in p.parts):
            continue
        yield p


def line_hits(path: Path, text: str) -> List[PyFileHit]:
    hits: List[PyFileHit] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        low = line.lower()
        stripped = line.strip()
        if any(tok.lower() in low for tok in PROPLINE_TOKENS):
            hits.append(PyFileHit(str(path), idx, "propline_token", stripped[:240]))
        elif any(tok in stripped for tok in REQUEST_TOKENS) and not stripped.startswith("#"):
            hits.append(PyFileHit(str(path), idx, "request_call", stripped[:240]))
    return hits


def parse_fastapi_endpoints(path: Path, text: str) -> List[EndpointInfo]:
    """Best-effort endpoint scanner without importing app code."""
    endpoints: List[EndpointInfo] = []
    lines = text.splitlines()

    # Match decorators like @app.get("/debug/propline-fetch")
    dec_re = re.compile(r"^\s*@\w+\.(get|post|put|delete|patch)\(\s*['\"]([^'\"]+)['\"]")
    fn_re = re.compile(r"^\s*(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")

    pending: List[Tuple[int, str, str]] = []
    for i, line in enumerate(lines, start=1):
        m = dec_re.match(line)
        if m:
            method, route = m.group(1).upper(), m.group(2)
            pending.append((i, method, route))
            continue

        fm = fn_re.match(line)
        if fm and pending:
            func = fm.group(1)
            dec_line, method, route = pending[-1]
            pending.clear()

            start = i
            # Determine function body until next top-level def/decorator/class.
            end = len(lines)
            for j in range(i + 1, len(lines) + 1):
                ln = lines[j - 1]
                if j > i and re.match(r"^(def|async\s+def|class|@\w+\.)", ln):
                    end = j - 1
                    break
            body = "\n".join(lines[start - 1:end])
            low = body.lower()
            propline_count = sum(low.count(tok.lower()) for tok in PROPLINE_TOKENS)
            req_count = sum(body.count(tok) for tok in REQUEST_TOKENS)
            has_prop = propline_count > 0 or any(x in route for x in ("propline", "fanduel", "line", "run"))

            notes: List[str] = []
            risk = "low"
            route_low = route.lower()
            if any(h in route_low for h in HIGH_RISK_DEBUG_ENDPOINT_HINTS):
                risk = "high"
                notes.append("route name suggests live PropLine/line refresh risk")
            if "market-probe" in route_low:
                risk = "very_high"
                notes.append("market probe endpoints often multiply calls by events × markets")
            if "/run/now" in route_low or "/run/daily" in route_low:
                risk = "high"
                notes.append("manual/daily prediction run may refresh line data")
            if req_count > 0:
                notes.append(f"function body contains {req_count} request-like call(s)")
            if propline_count > 0:
                notes.append(f"function body contains {propline_count} PropLine token(s)")
            if has_prop and risk == "low":
                risk = "medium"

            endpoints.append(EndpointInfo(
                file=str(path), route=route, method=method, function=func,
                start_line=start, end_line=end, has_propline_token=has_prop,
                request_call_count=req_count, propline_token_count=propline_count,
                risk=risk, notes=notes,
            ))
    return endpoints


def extract_market_literals(text: str) -> List[str]:
    # Find common Odds API / PropLine market key strings.
    market_re = re.compile(r'[\'\"]([a-z0-9_]*(?:strikeouts|home_runs|total_bases|hits|rbis|runs_scored|runs)(?:_alternate)?)[\'\"]')
    found = []
    for m in market_re.finditer(text):
        s = m.group(1)
        if s not in found:
            found.append(s)
    return found


def load_recent_json_files(data_root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not data_root.exists():
        return out
    candidates = []
    for pattern in (
        "propline_k_probe_latest.json",
        "propline_quota_audit_latest.json",
        "predictions/pitcher_k_candidates_*.json",
        "predictions/hitter_candidates_*.json",
        "predictions/predictions_*.json",
    ):
        candidates.extend(data_root.glob(pattern))
    # newest first, max 10
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)[:10]
    for p in candidates:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            summary = obj.get("summary") if isinstance(obj, dict) else None
            out.append({
                "path": str(p),
                "modified_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
                "top_level_type": type(obj).__name__,
                "top_level_keys": list(obj.keys())[:20] if isinstance(obj, dict) else None,
                "summary": summary,
            })
        except Exception as e:
            out.append({"path": str(p), "error": str(e)})
    return out


def estimate_actions(events: int, markets: int, runs_now: int, propline_fetches: int, market_probes: int, k_probes: int, k_market_scan_probes: int) -> Dict[str, Any]:
    """
    Conservative request-cost estimates.

    Common Odds API event-odds flow:
      - 1 request to /events
      - 1 request per event to /events/{eventId}/odds when multiple markets are bundled
      - Some market-probe code may do 1 request per event per market to isolate market keys
    """
    per_normal_refresh_min = 1 + events  # events + one event-odds call per event, if markets bundled
    per_market_probe = 1 + (events * markets)  # worst/common probe pattern
    per_k_probe = 1 + (events * 2)  # pitcher_strikeouts + alternate per event
    per_k_market_scan = 1 + (events * 6)  # scanner uses 6 likely K keys

    totals = {
        "one_normal_line_refresh_estimate": per_normal_refresh_min,
        "one_fanduel_market_probe_estimate": per_market_probe,
        "one_k_probe_estimate": per_k_probe,
        "one_k_probe_market_scan_estimate": per_k_market_scan,
        "entered_run_now_total_estimate": runs_now * per_normal_refresh_min,
        "entered_propline_fetch_total_estimate": propline_fetches * per_normal_refresh_min,
        "entered_market_probe_total_estimate": market_probes * per_market_probe,
        "entered_k_probe_total_estimate": k_probes * per_k_probe,
        "entered_k_market_scan_probe_total_estimate": k_market_scan_probes * per_k_market_scan,
    }
    totals["entered_total_estimate"] = sum(v for k, v in totals.items() if k.startswith("entered_") and k != "entered_total_estimate")

    return {
        "assumptions": {
            "events": events,
            "markets": markets,
            "normal_refresh_formula": "1 events request + events event-odds requests",
            "market_probe_formula": "1 events request + events × markets requests",
            "k_probe_formula": "1 events request + events × 2 K-market requests",
            "k_market_scan_formula": "1 events request + events × 6 possible K-market-key requests",
        },
        "estimates": totals,
        "quick_examples": {
            "15_event_normal_refresh": 1 + 15,
            "15_event_14_market_probe": 1 + (15 * 14),
            "5_manual_run_now_at_15_events": 5 * (1 + 15),
            "3_market_probes_at_15_events_14_markets": 3 * (1 + (15 * 14)),
        },
    }


def classify_likely_hogs(endpoints: List[EndpointInfo]) -> List[Dict[str, Any]]:
    order = {"very_high": 0, "high": 1, "medium": 2, "low": 3}
    rows = []
    for e in endpoints:
        if e.risk in ("very_high", "high", "medium"):
            why = list(e.notes)
            if "fanduel-market-probe" in e.route:
                why.append("most likely quota hog: can call event odds once per event per market")
            if "propline-fetch" in e.route or "line-audit" in e.route:
                why.append("likely triggers a line refresh unless cache is used")
            if "run/now" in e.route:
                why.append("manual refreshes can repeat the same PropLine fetch many times")
            rows.append({
                "risk": e.risk,
                "route": e.route,
                "method": e.method,
                "function": e.function,
                "file": e.file,
                "lines": f"{e.start_line}-{e.end_line}",
                "why": why,
            })
    return sorted(rows, key=lambda r: order.get(r["risk"], 99))


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone PropLine quota audit. No network calls.")
    ap.add_argument("--repo", default=str(REPO_DEFAULT), help="Repo path to scan. Default: current directory")
    ap.add_argument("--data", default=str(DATA_DEFAULT), help="Data path to inspect. Default: /data")
    ap.add_argument("--events", type=int, default=15, help="Assumed MLB events to fetch. Default: 15")
    ap.add_argument("--markets", type=int, default=14, help="Assumed markets requested/probed. Default: 14")
    ap.add_argument("--run-now", type=int, default=0, help="How many /run/now calls to estimate")
    ap.add_argument("--propline-fetches", type=int, default=0, help="How many /debug/propline-fetch calls to estimate")
    ap.add_argument("--market-probes", type=int, default=0, help="How many /debug/fanduel-market-probe calls to estimate")
    ap.add_argument("--k-probes", type=int, default=0, help="How many propline_k_probe normal calls to estimate")
    ap.add_argument("--k-market-scan-probes", type=int, default=0, help="How many propline_k_probe --market-scan calls to estimate")
    ap.add_argument("--json", action="store_true", help="Print full JSON only")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    data_root = Path(args.data)

    all_hits: List[PyFileHit] = []
    endpoints: List[EndpointInfo] = []
    market_keys: List[str] = []
    scanned_files = 0

    for py in iter_py_files(repo):
        scanned_files += 1
        txt = safe_read(py)
        all_hits.extend(line_hits(py, txt))
        endpoints.extend(parse_fastapi_endpoints(py, txt))
        for m in extract_market_literals(txt):
            if m not in market_keys:
                market_keys.append(m)

    request_lines = [h for h in all_hits if h.kind == "request_call"]
    propline_lines = [h for h in all_hits if h.kind == "propline_token"]

    result: Dict[str, Any] = {
        "audit_version": "standalone_propline_quota_audit_v1",
        "generated_at_utc": now_utc(),
        "safe_no_network_calls": True,
        "repo": str(repo),
        "data_root": str(data_root),
        "scanned_py_files": scanned_files,
        "environment": {
            "has_PROPLINE_API_KEY": bool(os.getenv("PROPLINE_API_KEY")),
            "has_ODDS_API_KEY": bool(os.getenv("ODDS_API_KEY")),
        },
        "detected_market_keys": sorted(market_keys),
        "likely_quota_hogs": classify_likely_hogs(endpoints),
        "fastapi_endpoints_with_line_risk": [asdict(e) for e in endpoints if e.has_propline_token or e.risk != "low"],
        "propline_code_hits_sample": [asdict(h) for h in propline_lines[:80]],
        "request_call_hits_sample": [asdict(h) for h in request_lines[:80]],
        "local_data_files_sample": load_recent_json_files(data_root),
        "request_cost_estimates": estimate_actions(
            events=args.events,
            markets=args.markets,
            runs_now=args.run_now,
            propline_fetches=args.propline_fetches,
            market_probes=args.market_probes,
            k_probes=args.k_probes,
            k_market_scan_probes=args.k_market_scan_probes,
        ),
        "interpretation": {
            "most_likely_hogs_in_this_project": [
                "/debug/fanduel-market-probe or market-scan style probes, because they can multiply events × markets",
                "repeated /run/now calls if each run refreshes PropLine lines instead of using cache",
                "repeated /debug/propline-fetch or /debug/line-audit if they live-fetch rather than read cache",
            ],
            "quota_guard_needed": [
                "cache events once per day",
                "cache event odds 30-60 minutes",
                "make debug endpoints read cache by default",
                "stop all PropLine calls after first 429 daily_limit_exceeded until next reset",
                "only fetch K markets for probable starters/games that need K lines",
            ],
        },
    }

    # Save output if /data exists.
    try:
        data_root.mkdir(parents=True, exist_ok=True)
        out_path = data_root / "propline_quota_audit_latest.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        result["saved_to"] = str(out_path)
    except Exception as e:
        result["save_error"] = str(e)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    # Human-readable condensed output.
    print(json.dumps({
        "audit_version": result["audit_version"],
        "generated_at_utc": result["generated_at_utc"],
        "safe_no_network_calls": True,
        "scanned_py_files": scanned_files,
        "has_PROPLINE_API_KEY": result["environment"]["has_PROPLINE_API_KEY"],
        "detected_market_keys_count": len(result["detected_market_keys"]),
        "saved_to": result.get("saved_to"),
    }, indent=2))

    print("\nLIKELY QUOTA HOGS")
    hogs = result["likely_quota_hogs"][:12]
    if not hogs:
        print("  No obvious endpoint hogs found by static scan.")
    for h in hogs:
        print(f"- [{h['risk']}] {h['method']} {h['route']} -> {h['function']} ({h['file']}:{h['lines']})")
        for w in h.get("why", [])[:4]:
            print(f"    - {w}")

    print("\nREQUEST COST ESTIMATES")
    print(json.dumps(result["request_cost_estimates"], indent=2))

    print("\nRECENT LOCAL DATA FILES")
    for item in result["local_data_files_sample"][:8]:
        print(f"- {item.get('path')}")
        if item.get("summary"):
            print(f"    summary: {json.dumps(item.get('summary'), ensure_ascii=False)[:500]}")
        if item.get("top_level_keys"):
            print(f"    keys: {item.get('top_level_keys')}")

    print("\nNEXT SAFE STEP")
    print("  Paste the LIKELY QUOTA HOGS section and REQUEST COST ESTIMATES back into chat.")
    print("  No API quota was used by this audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
