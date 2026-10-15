#!/usr/bin/env python3
"""Bounded *real* FanDuel NHL SOG quote capture for locked B2 research forecasts.

Uses The Odds API existing repository secret ODDS_API_KEY. Never shops
bookmakers/alternate markets, never invents lines, never issues official
picks, and never backfills a quote from after puck drop. Raw normalized
book prices stay in the PRIVATE GitHub Actions artifact; the public report
contains only provider status/coverage, not redistributable quote data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nhl_sog_fanduel_mainline_reference import (
    TEAM_NAMES, NAME_TO_TEAM, compare_original_model_to_real_market,
    model_rows, timestamp,
)

API = "https://api.the-odds-api.com/v4/sports/icehockey_nhl"
SCHEMA = "NHL_V2_REAL_FANDUEL_SOG_BOUNDED_PROVIDER_CHECK_V1"
MAX_EVENTS = 3


class QuoteCaptureFailure(RuntimeError):
    pass


def fetch_json(url, params, *, timeout=20):
    # Secret values must never appear in errors, printed URLs or artifacts.
    uri = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(uri, headers={
        "User-Agent": "nhl-v2-bounded-pregame-provider",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read(2_500_001)
            if len(raw) > 2_500_000:
                raise QuoteCaptureFailure("PROVIDER_RESPONSE_TOO_LARGE")
            return json.loads(raw), {
                "remaining": response.headers.get("x-requests-remaining"),
                "used": response.headers.get("x-requests-used"),
                "response_sha256": hashlib.sha256(raw).hexdigest(),
            }
    except urllib.error.HTTPError as exc:
        # Never print exc; it can contain the API key in the request URL.
        raise QuoteCaptureFailure("PROVIDER_HTTP_STATUS_" + str(exc.code)) from None
    except (urllib.error.URLError, TimeoutError):
        raise QuoteCaptureFailure("PROVIDER_TRANSPORT_UNAVAILABLE") from None
    except (UnicodeError, ValueError):
        raise QuoteCaptureFailure("PROVIDER_INVALID_JSON") from None


def future_model_games(board, now):
    rows = model_rows(board)
    games = defaultdict(list)
    for r in rows:
        start = timestamp(r["scheduled_start"])
        issued = timestamp(r["generated_at"])
        cutoff = timestamp(r["cutoff_at"])
        if not issued <= cutoff < start or issued > now or start <= now:
            continue
        if start > now + timedelta(hours=30):
            continue
        if r["team"] not in TEAM_NAMES or r["opponent"] not in TEAM_NAMES:
            continue
        if r["team"] == r["opponent"]:
            continue
        games[(str(r["game_id"]), start.isoformat())].append(r)
    return games


def event_match(ev, games):
    try:
        start = timestamp(ev.get("commence_time"))
    except (ValueError, TypeError):
        return None
    pair = {NAME_TO_TEAM.get(str(ev.get(k, "")).casefold())
            for k in ("home_team", "away_team")}
    if None in pair or len(pair) != 2:
        return None
    matches = [(k, v) for k, v in games.items()
               if abs((timestamp(k[1])-start).total_seconds()) <= 600
               and {v[0]["team"], v[0]["opponent"]} == pair]
    return matches[0][0] if len(matches) == 1 else None


def collect(board, *, api_key, now, fetch=fetch_json, max_events=MAX_EVENTS):
    if now.tzinfo is None:
        raise ValueError("NAIVE_CAPTURE_CLOCK")
    if not 1 <= max_events <= MAX_EVENTS:
        raise ValueError("UNSAFE_PROVIDER_SPEND_CAP")
    base = {
        "schema": SCHEMA,
        "captured_at_utc": now.astimezone(timezone.utc).isoformat(),
        "status": "UNINITIALIZED",
        "provider": "the_odds_api",
        "sport": "icehockey_nhl",
        "bookmaker": "fanduel",
        "market": "player_shots_on_goal",
        "max_paid_event_calls": max_events,
        "actual_paid_event_calls": 0,
        "estimated_max_billed_market_regions": max_events,
        "model_sha256": board.get("source_ledger_sha256"),
        "model_games_eligible": 0,
        "matched_events": 0,
        "qualified_player_main_lines": 0,
        "provider_rejections": {},
        "model_eligibility_verified": False,
        "lineup_certified": False,
        "official_picks": 0,
        "model_probabilities_posthoc_not_originally_published": True,
        "data_provider_calls_completed": False,
        "records": [],
        "retrieval_hashes": [],
    }
    # Fail closed against fabricated/unlocked model board even without API key.
    games = future_model_games(board, now)
    base["model_games_eligible"] = len(games)
    base["model_eligibility_verified"] = True
    if not api_key:
        base["status"] = "MISSING_EXISTING_GITHUB_ODDS_API_KEY"
        return base
    if not games:
        base["status"] = "NO_FUTURE_VERIFIED_MODEL_GAMES_NO_PAID_API_CALLS"
        return base

    params = {"apiKey": api_key}
    try:
        events, header = fetch(API + "/events", params)
    except QuoteCaptureFailure as exc:
        base["status"] = str(exc)
        return base
    if not isinstance(events, list):
        base["status"] = "MALFORMED_EVENTS_LIST"
        return base
    candidates, seen = [], set()
    for ev in events:
        if not isinstance(ev, dict) or not ev.get("id"):
            continue
        identity = event_match(ev, games)
        if identity is None or identity in seen:
            continue
        seen.add(identity)
        candidates.append((identity, ev))
    candidates.sort(key=lambda x: timestamp(x[1]["commence_time"]))
    rejects = Counter()
    for identity, ev in candidates[:max_events]:
        endpoint = API + "/events/" + urllib.parse.quote(str(ev["id"]), safe="") + "/odds"
        args = {
            "apiKey": api_key, "bookmakers": "fanduel",
            "markets": "player_shots_on_goal", "oddsFormat": "american",
        }
        try:
            # Capture clock follows the paid request, not a user-supplied time.
            quote, meta = fetch(endpoint, args)
            base["actual_paid_event_calls"] += 1
        except QuoteCaptureFailure as exc:
            base["actual_paid_event_calls"] += 1
            rejects[str(exc)] += 1
            if "429" in str(exc):
                break
            continue
        if not isinstance(quote, dict):
            rejects["INVALID_ODDS_PAYLOAD"] += 1
            continue
        observed = datetime.now(timezone.utc)
        try:
            x = compare_original_model_to_real_market(
                board, [quote], captured_at_utc=observed.isoformat())
        except (ValueError, KeyError, TypeError) as exc:
            rejects["MODEL_PROVIDER_ADMISSION_FAILED_" + type(exc).__name__] += 1
            continue
        base["matched_events"] += 1
        base["records"].extend(x["records"])
        rejects.update(x.get("rejections") or {})
        base["retrieval_hashes"].append({
            "event_id": str(ev["id"]), "raw_response_sha256": meta["response_sha256"],
            "received_at_utc": observed.isoformat(),
            "requests_remaining": meta["remaining"],
        })
    base["qualified_player_main_lines"] = len(base["records"])
    base["provider_rejections"] = dict(sorted(rejects.items()))
    base["data_provider_calls_completed"] = base["actual_paid_event_calls"] > 0
    base["status"] = ("VERIFIED_REAL_FANDUEL_MAINLINES_RESEARCH_ONLY"
                      if base["records"] else
                      "NO_ADMISSIBLE_REAL_FANDUEL_MAINLINES")
    return base


def public_status(result):
    # No individual sportsbook prices, player-level selections, or raw bytes.
    return {k: v for k, v in result.items()
            if k not in {"records", "retrieval_hashes"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--private-out", type=Path, required=True)
    parser.add_argument("--public-out", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=MAX_EVENTS)
    args = parser.parse_args()
    board = json.loads(args.board.read_text())
    result = collect(
        board, api_key=os.environ.get("THE_ODDS_API_KEY") or
        os.environ.get("ODDS_API_KEY"), now=datetime.now(timezone.utc),
        max_events=args.max_events)
    args.private_out.parent.mkdir(parents=True, exist_ok=True)
    args.public_out.parent.mkdir(parents=True, exist_ok=True)
    args.private_out.write_text(json.dumps(result, sort_keys=True,
                                           allow_nan=False, indent=2) + "\n")
    args.public_out.write_text(json.dumps(public_status(result), sort_keys=True,
                                          allow_nan=False, indent=2) + "\n")
    print(json.dumps(public_status(result), sort_keys=True))
    return 0 if result["status"] in {
        "VERIFIED_REAL_FANDUEL_MAINLINES_RESEARCH_ONLY",
        "NO_FUTURE_VERIFIED_MODEL_GAMES_NO_PAID_API_CALLS",
        "NO_ADMISSIBLE_REAL_FANDUEL_MAINLINES",
    } else 1


if __name__ == "__main__":
    sys.exit(main())
