#!/usr/bin/env python3
"""
k_tomorrow_candidate_builder_test.py
Standalone diagnostic only. Does not change api.py, predictions, record.json, or PropLine usage.

Purpose:
- Build the NEXT/FUTURE pitcher strikeout candidate slate directly from MLB probable pitchers
  for the same date as available FanDuel K lines.
- Pull FanDuel main pitcher strikeout lines from Odds-API.io using the documented
  single-event player-props endpoint: /v3/odds?eventId=<id>&bookmakers=FanDuel
- Match Odds-API.io K lines to MLB probable pitchers by normalized pitcher name.
- Use the existing app's pitcher feature logic from api.py to create projections.
- Run 250,000 Monte Carlo simulations for matched K candidates.

Env required:
- ODDS_API_IO_KEY

Run:
  python k_tomorrow_candidate_builder_test.py
  python k_tomorrow_candidate_builder_test.py --target-date 2026-07-05 --max-events 25

Outputs:
  /data/candidate_logs/k_tomorrow_candidate_builder_test_latest.json
  /data/candidate_logs/k_tomorrow_candidate_builder_test_latest.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except Exception as e:
    print("ERROR: requests is required:", e)
    sys.exit(1)

try:
    import numpy as np
except Exception:
    np = None

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# Import the deployed app module from the Render working directory.
# This lets the diagnostic use the same pitcher feature logic/constants as api.py.
try:
    import api  # type: ignore
except Exception as e:
    print("ERROR: could not import local api.py. Run this from the Render project/src directory.")
    print(repr(e))
    sys.exit(2)

BASE = "https://api.odds-api.io/v3"
OUT_DIR = Path("/data/candidate_logs")
OUT_JSON = OUT_DIR / "k_tomorrow_candidate_builder_test_latest.json"
OUT_CSV = OUT_DIR / "k_tomorrow_candidate_builder_test_latest.csv"
SIM_N_DEFAULT = 250_000
MC_KEEP_THRESHOLD_DEFAULT = 0.63

SUFFIX_RE = re.compile(r"\s*\((?:total\s+)?strikeouts?\)\s*$", re.I)
JUNK_RE = re.compile(r"\b(jr|sr|ii|iii|iv)\b\.?", re.I)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def now_et() -> dt.datetime:
    if ZoneInfo is not None:
        return dt.datetime.now(ZoneInfo("America/New_York"))
    # fallback approximation; Render is UTC, use app's now_et if available
    try:
        return api.now_et()  # type: ignore[attr-defined]
    except Exception:
        return dt.datetime.utcnow() - dt.timedelta(hours=4)


def default_target_date() -> str:
    # The user's current need is tomorrow/future FanDuel lines. Default to tomorrow ET.
    return (now_et().date() + dt.timedelta(days=1)).isoformat()


def parse_dt(s: Any) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def et_date_of(s: Any) -> Optional[str]:
    d = parse_dt(s)
    if not d:
        return None
    if ZoneInfo is not None:
        return d.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    return d.date().isoformat()


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    try:
        if isinstance(x, str):
            x = x.strip().replace("+", "")
            if x.upper() in {"", "NA", "N/A", "NONE", "NULL"}:
                return default
        return float(x)
    except Exception:
        return default


def shape(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, list):
        d = {"type": "list", "len": len(obj)}
        if obj:
            d["first_type"] = type(obj[0]).__name__
            if isinstance(obj[0], dict):
                d["first_keys"] = list(obj[0].keys())[:12]
        return d
    if isinstance(obj, dict):
        return {"type": "dict", "len": len(obj), "first_keys": list(obj.keys())[:12]}
    return {"type": type(obj).__name__}


def http_get_json(url: str, params: Dict[str, Any], log: List[Dict[str, Any]]) -> Any:
    redacted = dict(params)
    if "apiKey" in redacted:
        redacted["apiKey"] = "***"
    try:
        r = requests.get(url, params=params, timeout=25)
        try:
            data = r.json()
        except Exception:
            data = {"_text_preview": r.text[:1000]}
        log.append({
            "status": r.status_code,
            "endpoint": url.replace(BASE, ""),
            "params": redacted,
            "ok": r.ok,
            "shape": shape(data),
            "error": data.get("error") if isinstance(data, dict) else None,
        })
        return data
    except Exception as e:
        log.append({"status": None, "endpoint": url.replace(BASE, ""), "params": redacted, "ok": False, "exception": repr(e)})
        return {"error": repr(e)}


def normalize_name(name: Any) -> str:
    if name is None:
        return ""
    s = SUFFIX_RE.sub("", str(name))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = JUNK_RE.sub("", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def name_tokens(name: Any) -> List[str]:
    return [t for t in normalize_name(name).split() if t]


def match_score(a: Any, b: Any) -> float:
    an = normalize_name(a)
    bn = normalize_name(b)
    if not an or not bn:
        return 0.0
    if an == bn:
        return 1.0
    at = name_tokens(an)
    bt = name_tokens(bn)
    if sorted(at) == sorted(bt):
        return 0.97
    if len(at) >= 2 and len(bt) >= 2 and at[-1] == bt[-1] and at[0][0] == bt[0][0]:
        return 0.92
    if at and bt and at[-1] == bt[-1]:
        return max(0.78, SequenceMatcher(None, an, bn).ratio())
    return SequenceMatcher(None, an, bn).ratio()


def strip_total_strikeouts(label: str) -> str:
    return SUFFIX_RE.sub("", str(label)).strip()


# ------------------------- provider lines -------------------------

def fetch_oddsapiio_k_lines(api_key: str, target_date: str, max_events: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    log: List[Dict[str, Any]] = []
    events = http_get_json(
        f"{BASE}/events",
        {"sport": "baseball", "league": "usa-mlb", "bookmaker": "FanDuel", "apiKey": api_key},
        log,
    )
    if not isinstance(events, list):
        return [], {"error": "events_not_list", "events_shape": shape(events)}, log

    now_utc = dt.datetime.now(dt.timezone.utc)
    used_events: List[Dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        ev_dt = parse_dt(ev.get("date"))
        ev_et_date = et_date_of(ev.get("date"))
        status = str(ev.get("status", "")).lower()
        if ev_et_date != target_date:
            continue
        if ev_dt and ev_dt < now_utc - dt.timedelta(minutes=45):
            continue
        if status in {"settled", "final", "complete", "completed", "cancelled", "canceled"}:
            continue
        used_events.append(ev)

    used_events = used_events[:max_events]
    lines: List[Dict[str, Any]] = []
    market_names: Dict[str, int] = {}
    k_hits_raw = 0

    for ev in used_events:
        event_id = ev.get("id")
        if not event_id:
            continue
        odds = http_get_json(
            f"{BASE}/odds",
            {"eventId": event_id, "bookmakers": "FanDuel", "apiKey": api_key},
            log,
        )
        if not isinstance(odds, dict):
            continue
        books = odds.get("bookmakers") or {}
        fd = books.get("FanDuel") if isinstance(books, dict) else None
        if not isinstance(fd, list):
            continue
        for market in fd:
            if not isinstance(market, dict):
                continue
            mname = str(market.get("name") or market.get("label") or "")
            if mname:
                market_names[mname] = market_names.get(mname, 0) + 1
            rows = market.get("odds")
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                label = str(row.get("label") or row.get("name") or "")
                hay = f"{mname} {label}".lower()
                if "strikeout" not in hay:
                    continue
                line = safe_float(row.get("hdp") if "hdp" in row else row.get("point"))
                if line is None:
                    continue
                k_hits_raw += 1
                lines.append({
                    "provider": "oddsapiio",
                    "book": "FanDuel",
                    "event_id": str(event_id),
                    "event": f"{ev.get('away')} @ {ev.get('home')}",
                    "home": ev.get("home"),
                    "away": ev.get("away"),
                    "date": ev.get("date"),
                    "target_date_et": et_date_of(ev.get("date")),
                    "status": ev.get("status"),
                    "market": mname,
                    "label": label,
                    "player": strip_total_strikeouts(label),
                    "line": line,
                    "over_odds": safe_float(row.get("over"), None),
                    "under_odds": safe_float(row.get("under"), None),
                    "raw": row,
                })

    dedup: List[Dict[str, Any]] = []
    seen = set()
    for line in lines:
        key = (normalize_name(line["player"]), line.get("event_id"), line.get("line"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(line)

    meta = {
        "events_total": len(events),
        "target_date": target_date,
        "events_used": len(used_events),
        "k_hits_raw": k_hits_raw,
        "k_lines": len(dedup),
        "market_names_seen": market_names,
        "events_used_sample": [
            {"id": e.get("id"), "event": f"{e.get('away')} @ {e.get('home')}", "date": e.get("date"), "status": e.get("status")}
            for e in used_events[:12]
        ],
    }
    return dedup, meta, log


# ------------------------- MLB probable pitchers -------------------------

def mlb_schedule_probables(target_date: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = api.get(f"{api.MLB}/schedule", sportId=1, date=target_date, hydrate="probablePitcher,team")
    games: List[Dict[str, Any]] = []
    pitchers: List[Dict[str, Any]] = []

    for day in data.get("dates", []):
        for g in day.get("games", []):
            teams = g.get("teams", {})
            hteam = (teams.get("home", {}) or {}).get("team", {}) or {}
            ateam = (teams.get("away", {}) or {}).get("team", {}) or {}
            home_name = hteam.get("name")
            away_name = ateam.get("name")
            gid = str(g.get("gamePk"))
            game_date = g.get("gameDate")
            status = (g.get("status", {}) or {}).get("abstractGameState", "")
            game_row = {
                "game_id": gid,
                "date": target_date,
                "game_time": game_date,
                "status": str(status).lower(),
                "home_team": home_name,
                "away_team": away_name,
            }
            games.append(game_row)

            for side in ("home", "away"):
                side_obj = teams.get(side, {}) or {}
                pp = side_obj.get("probablePitcher") or {}
                pid = pp.get("id")
                if not pid:
                    continue
                pname = pp.get("fullName")
                if not pname:
                    try:
                        pdata = api.get(f"{api.MLB}/people/{pid}")
                        pname = pdata.get("people", [{}])[0].get("fullName")
                    except Exception:
                        pname = None
                team = home_name if side == "home" else away_name
                opp = away_name if side == "home" else home_name
                pitchers.append({
                    "player": pname or str(pid),
                    "norm_player": normalize_name(pname or str(pid)),
                    "player_id": pid,
                    "side": side,
                    "team": team,
                    "opponent": opp,
                    "game_id": gid,
                    "game_time": game_date,
                    "target_date": target_date,
                    "status": str(status).lower(),
                    "home_team": home_name,
                    "away_team": away_name,
                })
    return games, pitchers


def best_line_for_pitcher(pitcher_name: str, provider_lines: List[Dict[str, Any]], threshold: float) -> Tuple[Optional[Dict[str, Any]], float, str]:
    best = None
    best_score = 0.0
    for line in provider_lines:
        score = match_score(pitcher_name, line.get("player"))
        if score > best_score:
            best_score = score
            best = line
    if best is None:
        return None, 0.0, "no_provider_lines"
    if best_score >= threshold:
        return best, best_score, "name_match"
    return None, best_score, "no_match"


# ------------------------- projection + Monte Carlo -------------------------

def compute_projection_for_pitcher(p: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    pid = p.get("player_id")
    try:
        feat = api.pitcher_feature_row(pid)
    except Exception as e:
        return None, f"pitcher_feature_error:{repr(e)}"
    if not feat:
        return None, "missing_pitcher_features"

    exp_bf = safe_float(feat.get("avg_bf"), 0.0) or 0.0
    k_per_bf = safe_float(feat.get("k_per_bf"), 0.0) or 0.0
    pitcher_proj = k_per_bf * exp_bf
    if pitcher_proj <= 0 or exp_bf <= 0:
        return None, "invalid_pitcher_projection"

    lineup_exp_ks = None
    k_nudge = 1.0
    bvp_flag = None
    opp_batters: List[Any] = []

    # If lineups are available, use the same optional K-context layer as api.py.
    try:
        lineup = api.get_confirmed_lineup(str(p.get("game_id")))
        opp_side = "away" if p.get("side") == "home" else "home"
        opp_batters = lineup.get(opp_side, []) if isinstance(lineup, dict) else []
    except Exception:
        opp_batters = []

    if opp_batters:
        try:
            throws = api.lineupk.get_pitcher_throws(pid)
            ek, avg_kr, n = api.lineupk.lineup_k_expectation(
                opp_batters, throws, api.SEASON, exp_bf, pitcher_id=pid
            )
            if ek and n >= 5:
                lineup_exp_ks = float(ek)
        except Exception:
            lineup_exp_ks = None

        try:
            if getattr(api, "BVP_ENABLED", False):
                agg = api.bvp.lineup_vs_pitcher(opp_batters, pid)
                if agg and agg.get("sample_pa", 0) >= 20:
                    k_nudge = float(agg.get("k_nudge") or 1.0)
                    bvp_flag = f"lineup_kr_{agg.get('lineup_k_rate')}_n{k_nudge}"
        except Exception:
            pass

    pitcher_weight = float(getattr(api, "PITCHER_WEIGHT", 0.55))
    lineup_weight = float(getattr(api, "LINEUP_WEIGHT", 0.45))
    if lineup_exp_ks is not None and lineup_exp_ks > 0:
        blended = pitcher_weight * pitcher_proj + lineup_weight * lineup_exp_ks
    else:
        blended = pitcher_proj
    final_projection = blended * k_nudge

    return {
        "feat": feat,
        "avg_bf": exp_bf,
        "k_per_bf": k_per_bf,
        "pitcher_projection": pitcher_proj,
        "lineup_exp_ks": lineup_exp_ks,
        "k_nudge": k_nudge,
        "bvp_flag": bvp_flag,
        "final_projection": final_projection,
        "recent_k_avg": safe_float(feat.get("recent_k_avg")),
        "outs_per_start": safe_float(feat.get("outs_per_start")),
        "starts": feat.get("starts"),
        "per_start_krate": feat.get("per_start_krate") or [],
    }, None


def simulate_k_mc(projection: float, line: float, exp_bf: float, per_start_krate: List[Any], sim_n: int, seed: int) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("numpy is required for 250,000-simulation K MC test")

    rng = np.random.default_rng(seed)
    proj = max(0.05, float(projection))
    exp_bf = max(10.0, float(exp_bf or 24.0))
    mean_rate = min(0.70, max(0.02, proj / exp_bf))

    rates = []
    for r in per_start_krate or []:
        rf = safe_float(r)
        if rf is not None and 0 <= rf <= 1:
            rates.append(float(rf))

    if len(rates) >= 3:
        # Preserve recent-start volatility but center around current projection.
        raw_sd = float(np.std(np.array(rates, dtype=float)))
        rate_sd = min(0.18, max(0.025, raw_sd))
    else:
        rate_sd = 0.055

    # Workload uncertainty: most starters vary by a few batters faced start-to-start.
    bf_sd = max(1.75, min(4.0, exp_bf * 0.11))
    bf_samples = rng.normal(loc=exp_bf, scale=bf_sd, size=sim_n)
    bf_samples = np.clip(bf_samples, 12, 34)

    rate_samples = rng.normal(loc=mean_rate, scale=rate_sd, size=sim_n)
    rate_samples = np.clip(rate_samples, 0.01, 0.70)

    lambdas = np.clip(rate_samples * bf_samples, 0.05, 18.0)
    sims = rng.poisson(lam=lambdas)

    prob_over = float(np.mean(sims > line))
    prob_under = float(np.mean(sims < line))
    side = "OVER" if projection > line else "UNDER"
    side_prob = prob_over if side == "OVER" else prob_under

    return {
        "sim_n": sim_n,
        "sim_mean": float(np.mean(sims)),
        "sim_median": float(np.median(sims)),
        "sim_p10": float(np.quantile(sims, 0.10)),
        "sim_p90": float(np.quantile(sims, 0.90)),
        "mc_prob_over": prob_over,
        "mc_prob_under": prob_under,
        "mc_side": side,
        "mc_side_prob": side_prob,
        "line": float(line),
    }


def pct(x: Any) -> str:
    f = safe_float(x)
    if f is None:
        return "NA"
    return f"{f*100:.1f}%"


# ------------------------- main -------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-date", default=None, help="ET slate date, YYYY-MM-DD. Default: tomorrow ET.")
    ap.add_argument("--max-events", type=int, default=25)
    ap.add_argument("--sim-n", type=int, default=SIM_N_DEFAULT)
    ap.add_argument("--mc-keep-threshold", type=float, default=MC_KEEP_THRESHOLD_DEFAULT)
    ap.add_argument("--match-threshold", type=float, default=0.78)
    ap.add_argument("--seed", type=int, default=20260705)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_date = args.target_date or default_target_date()
    api_key = os.getenv("ODDS_API_IO_KEY") or os.getenv("ODDSAPIIO_KEY") or ""

    print("=== K TOMORROW/FUTURE CANDIDATE BUILDER TEST ===")
    print(f"Generated UTC: {utc_now()}")
    print(f"Target date ET: {target_date}")
    print("Provider: Odds-API.io /v3/odds by eventId, FanDuel Player Props")
    print(f"Simulations per matched K candidate: {args.sim_n:,}")
    print(f"MC keep threshold: {args.mc_keep_threshold * 100:.1f}%")
    print("NOTE: standalone diagnostic only; does not change api.py, predictions, record.json, or PropLine usage.")

    if not api_key:
        print("ERROR: missing ODDS_API_IO_KEY")
        return 2

    provider_lines, provider_meta, req_log = fetch_oddsapiio_k_lines(api_key, target_date, args.max_events)
    games, probable_pitchers = mlb_schedule_probables(target_date)

    rows: List[Dict[str, Any]] = []
    unused_provider = []
    used_provider_keys = set()

    for p in probable_pitchers:
        line, score, match_reason = best_line_for_pitcher(p.get("player", ""), provider_lines, args.match_threshold)
        base = {
            **p,
            "match_score": round(score, 3),
            "match_reason": match_reason,
            "has_provider_line": bool(line),
        }
        if not line:
            rows.append({**base, "status": "unmatched_line", "reason": "no_provider_line"})
            continue

        used_provider_keys.add((normalize_name(line.get("player")), line.get("event_id"), line.get("line")))
        proj, proj_err = compute_projection_for_pitcher(p)
        if proj_err or not proj:
            rows.append({
                **base,
                "status": "matched_no_projection",
                "reason": proj_err or "projection_failed",
                "provider_player": line.get("player"),
                "line": line.get("line"),
                "provider_event": line.get("event"),
                "provider_date": line.get("date"),
            })
            continue

        projection = float(proj["final_projection"])
        kline = float(line["line"])
        mc = simulate_k_mc(
            projection=projection,
            line=kline,
            exp_bf=float(proj.get("avg_bf") or 24.0),
            per_start_krate=proj.get("per_start_krate") or [],
            sim_n=args.sim_n,
            seed=args.seed + int(p.get("player_id") or 0) % 100000,
        )
        projection_gap = projection - kline
        side = "OVER" if projection_gap > 0 else "UNDER"
        side_prob = mc["mc_prob_over"] if side == "OVER" else mc["mc_prob_under"]
        keep = bool(side_prob >= args.mc_keep_threshold)

        rows.append({
            **base,
            "status": "simulated",
            "reason": "keep_mc_threshold" if keep else "drop_below_mc_threshold",
            "provider_player": line.get("player"),
            "provider_event": line.get("event"),
            "provider_date": line.get("date"),
            "provider_market": line.get("market"),
            "line": kline,
            "over_odds": line.get("over_odds"),
            "under_odds": line.get("under_odds"),
            "pitcher_projection": round(float(proj["pitcher_projection"]), 3),
            "lineup_exp_ks": round(float(proj["lineup_exp_ks"]), 3) if proj.get("lineup_exp_ks") is not None else None,
            "k_nudge": round(float(proj["k_nudge"]), 3),
            "projection": round(projection, 3),
            "projection_gap": round(projection_gap, 3),
            "side": side,
            "mc_prob_over": mc["mc_prob_over"],
            "mc_prob_under": mc["mc_prob_under"],
            "mc_side_prob": side_prob,
            "sim_mean": round(float(mc["sim_mean"]), 3),
            "sim_p10": mc["sim_p10"],
            "sim_median": mc["sim_median"],
            "sim_p90": mc["sim_p90"],
            "keep": keep,
            "confidence": "HIGH" if side_prob >= 0.68 else "MEDIUM" if side_prob >= 0.63 else "LOW",
            "avg_bf": round(float(proj.get("avg_bf") or 0), 3),
            "recent_k_avg": proj.get("recent_k_avg"),
            "outs_per_start": proj.get("outs_per_start"),
            "starts": proj.get("starts"),
            "bvp_flag": proj.get("bvp_flag"),
        })

    for line in provider_lines:
        key = (normalize_name(line.get("player")), line.get("event_id"), line.get("line"))
        if key not in used_provider_keys:
            unused_provider.append(line)

    simulated = [r for r in rows if r.get("status") == "simulated"]
    kept = [r for r in simulated if r.get("keep")]
    dropped = [r for r in simulated if not r.get("keep")]
    no_line = [r for r in rows if r.get("status") == "unmatched_line"]
    no_proj = [r for r in rows if r.get("status") == "matched_no_projection"]

    summary = {
        "generated_at_utc": utc_now(),
        "target_date_et": target_date,
        "api_version": getattr(api, "VERSION", None),
        "provider": "oddsapiio_single_event_fanduel_player_props",
        "provider_meta": provider_meta,
        "request_count": len(req_log),
        "mlb_games_count": len(games),
        "probable_pitchers_count": len(probable_pitchers),
        "provider_k_lines_count": len(provider_lines),
        "matched_and_simulated": len(simulated),
        "matched_no_projection": len(no_proj),
        "unmatched_no_line": len(no_line),
        "unused_provider_lines": len(unused_provider),
        "would_keep": len(kept),
        "would_drop": len(dropped),
        "sim_n": args.sim_n,
        "mc_keep_threshold": args.mc_keep_threshold,
    }

    out = {
        "summary": summary,
        "provider_lines": provider_lines,
        "mlb_games": games,
        "probable_pitchers": probable_pitchers,
        "rows": rows,
        "unused_provider_lines": unused_provider,
        "requests": req_log,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    csv_cols = [
        "player", "team", "opponent", "game_time", "status", "line", "projection", "projection_gap", "side",
        "mc_side_prob", "mc_prob_over", "mc_prob_under", "keep", "confidence", "reason", "match_score",
        "provider_player", "provider_event", "over_odds", "under_odds", "avg_bf", "recent_k_avg", "outs_per_start", "starts",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\nPROVIDER LINES")
    print(f"provider=Odds-API.io has_key=True requests={len(req_log)} events_used={provider_meta.get('events_used')} k_lines={len(provider_lines)} error={provider_meta.get('error')}")
    for line in provider_lines[:20]:
        print(f"- {line.get('player')} | line={line.get('line')} | over={line.get('over_odds')} under={line.get('under_odds')} | {line.get('event')} | {line.get('date')}")

    print("\nMLB PROBABLE PITCHERS")
    print(f"games={len(games)} probable_pitchers={len(probable_pitchers)}")
    for p in probable_pitchers[:30]:
        print(f"- {p.get('player')} | {p.get('team')} vs {p.get('opponent')} | gid={p.get('game_id')} | {p.get('game_time')}")

    print("\nMATCH / MC SUMMARY")
    for k, v in summary.items():
        if k not in {"provider_meta"}:
            print(f"{k}: {v}")

    print("\nTOP K MC RESULTS")
    top = sorted(simulated, key=lambda r: safe_float(r.get("mc_side_prob"), 0.0) or 0.0, reverse=True)
    if not top:
        print("No matched K candidates were simulated.")
    for i, r in enumerate(top[:30], start=1):
        print(
            f"{i:02d}. {r.get('player')} | {r.get('side')} {r.get('line')} | "
            f"proj={r.get('projection')} gap={r.get('projection_gap')} | "
            f"MC={pct(r.get('mc_side_prob'))} O={pct(r.get('mc_prob_over'))} U={pct(r.get('mc_prob_under'))} | "
            f"keep={r.get('keep')} | {r.get('team')} vs {r.get('opponent')} | reason={r.get('reason')}"
        )

    print("\nMATCHED BUT NO PROJECTION")
    if not no_proj:
        print("None")
    for r in no_proj[:30]:
        print(f"- {r.get('player')} | line={r.get('line')} | reason={r.get('reason')} | provider={r.get('provider_player')}")

    print("\nPROBABLE PITCHERS WITHOUT PROVIDER LINE")
    if not no_line:
        print("None")
    for r in no_line[:30]:
        print(f"- {r.get('player')} | {r.get('team')} vs {r.get('opponent')} | best_score={r.get('match_score')} | reason={r.get('reason')}")

    print("\nUNUSED PROVIDER LINES")
    if not unused_provider:
        print("None")
    for line in unused_provider[:30]:
        print(f"- {line.get('player')} | line={line.get('line')} | {line.get('event')} | {line.get('date')}")

    print("\nREQUESTS")
    for q in req_log[:40]:
        print(f"- {q.get('status')} {q.get('endpoint')} ok={q.get('ok')} shape={q.get('shape')} error={q.get('error')}")

    print("\nOUTPUTS")
    print(str(OUT_JSON))
    print(str(OUT_CSV))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
