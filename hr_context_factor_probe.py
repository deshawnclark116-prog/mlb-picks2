#!/usr/bin/env python3
"""
HR Context Factor Probe for Prop Edge MLB API.

Purpose:
- Diagnostic only. Does NOT modify api.py.
- Starts from the current best HR bucket:
      hr_score >= 1.70 AND season_slg >= .450
- Adds context tests the previous HR probes did NOT cover:
      park factor / venue bucket
      weather temperature
      wind direction
      batter/pitcher handedness and platoon edge
      starter side
      lineup spot
      recent ISO / HR score combo
- Uses MLB StatsAPI game feed to enrich historical HR rows by game_id.

Important:
- This is not a final model. It is a context-factor lab.
- Small samples can lie. The script prints Wilson lower bound and sample counts.

Requirements:
- Put this file beside api.py
- Put hr_calibration_probe.py beside it too

Run:
    python hr_context_factor_probe.py --days 180 --min-n 3

Conservative:
    python hr_context_factor_probe.py --days 180 --min-n 10

Optional CSV:
    python hr_context_factor_probe.py --days 180 --min-n 3 --write-csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import itertools
import json
import math
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

try:
    import hr_calibration_probe as cal
except Exception as e:
    raise SystemExit(
        "ERROR: hr_calibration_probe.py must be in the same folder as this script. "
        f"Import failed: {e}"
    )


# Coarse HR park buckets. These are intentionally buckets, not fake precision.
# This should later be replaced by real multi-year park HR factors.
PARK_BUCKET_BY_VENUE = {
    "coors field": "elite_hr_park",
    "great american ball park": "hr_friendly",
    "citizens bank park": "hr_friendly",
    "yankee stadium": "hr_friendly",
    "guaranteed rate field": "hr_friendly",
    "american family field": "hr_friendly",
    "globe life field": "hr_friendly",
    "orangelike park at camden yards": "neutral",  # defensive typo fallback if venue text changes
    "oripe park at camden yards": "neutral",
    "oriole park at camden yards": "neutral",
    "truist park": "slightly_hr_friendly",
    "dodger stadium": "slightly_hr_friendly",
    "fenway park": "slightly_hr_friendly",
    "target field": "neutral",
    "minute maid park": "neutral",
    "rogers centre": "neutral",
    "loanDepot park".lower(): "pitcher_friendly",
    "t-mobile park": "pitcher_friendly",
    "petco park": "pitcher_friendly",
    "oracle park": "pitcher_friendly",
    "pnc park": "pitcher_friendly",
    "citi field": "pitcher_friendly",
    "busch stadium": "pitcher_friendly",
    "kauffman stadium": "pitcher_friendly",
    "progressive field": "neutral",
    "comerica park": "pitcher_friendly",
    "angel stadium": "neutral",
    "chase field": "neutral",
    "nationals park": "neutral",
    "wrigley field": "weather_sensitive",
    "tropicana field": "dome_neutral",
    "oakland coliseum": "pitcher_friendly",
    "sutter health park": "unknown",
}

HITTERISH_PARK_BUCKETS = {"elite_hr_park", "hr_friendly", "slightly_hr_friendly", "weather_sensitive"}
BAD_PARK_BUCKETS = {"pitcher_friendly"}


def fnum(x: Any, default: Optional[float] = None) -> Optional[float]:
    return cal.safe_float(x, default)


def norm(s: Any) -> str:
    return cal.norm(s)


def today_from_api(api_mod: Any) -> dt.date:
    today = getattr(api_mod, "today_et", lambda: dt.date.today())()
    if isinstance(today, str):
        return dt.date.fromisoformat(today)
    return today


def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def wilson_lower(hits: int, n: int, z: float = 1.28155) -> float:
    if n <= 0:
        return 0.0
    phat = hits / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return max(0.0, (center - margin) / denom)


def summarize(rows: List[Dict[str, Any]], baseline: float) -> Dict[str, Any]:
    n = len(rows)
    hits = sum(1 for r in rows if r.get("result") == "hit")
    rate = hits / n if n else 0.0
    lower = wilson_lower(hits, n)
    lift = rate / baseline if baseline else None

    def avg(field: str) -> Optional[float]:
        vals = [fnum(r.get(field)) for r in rows]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n": n,
        "hits": hits,
        "misses": n - hits,
        "rate": rate,
        "wilson80_lower": lower,
        "lift": lift,
        "penalized_score": lower * math.log1p(n),
        "avg_hr_score": avg("hr_score"),
        "avg_season_slg": avg("season_slg"),
        "avg_recent_iso": avg("recent_iso"),
        "avg_h2h_slg": avg("h2h_slg"),
        "avg_lineup_spot": avg("lineup_spot"),
        "avg_temp_f": avg("weather_temp_f"),
    }


def label_summary(label: str, s: Dict[str, Any]) -> str:
    lift = f"{s['lift']:.2f}x" if s["lift"] is not None else "n/a"
    return (
        f"{s['rate']:6.1%} n={s['n']:3d} hit={s['hits']:3d} "
        f"miss={s['misses']:3d} lb80={s['wilson80_lower']:5.1%} lift={lift:>6s} "
        f"avg_score={round(s['avg_hr_score'],3) if s['avg_hr_score'] is not None else None} "
        f"avg_slg={round(s['avg_season_slg'],3) if s['avg_season_slg'] is not None else None} "
        f"avg_iso={round(s['avg_recent_iso'],3) if s['avg_recent_iso'] is not None else None} "
        f"avg_temp={round(s['avg_temp_f'],1) if s['avg_temp_f'] is not None else None}  "
        f"{label}"
    )


def load_all_rows(api_mod: Any, days: int, exact_date: Optional[str], max_grade_rows: int = 0) -> Tuple[List[Dict[str, Any]], Counter]:
    candidate_rows = cal.load_candidate_rows(api_mod, days, exact_date)
    record_rows = cal.load_record_hr_rows(api_mod, days, exact_date)
    graded_candidates, reasons = cal.grade_candidates(api_mod, candidate_rows, max_rows=max_grade_rows)

    all_rows: List[Dict[str, Any]] = []
    seen = set()

    for r in graded_candidates:
        rr = dict(r)
        rr["quality_ok"] = cal.quality_ok(rr)
        rr["tier_bucket"] = cal.hr_tier(rr)
        key = (rr.get("date"), str(rr.get("game_id")), str(rr.get("player_id") or ""), norm(rr.get("player")), rr.get("pick"))
        all_rows.append(rr)
        seen.add(key)

    for r in record_rows:
        if r.get("result") not in {"hit", "miss"}:
            continue
        key = (r.get("date"), str(r.get("game_id")), str(r.get("player_id") or ""), norm(r.get("player")), r.get("pick"))
        if key in seen:
            continue
        rr = dict(r)
        rr["quality_ok"] = cal.quality_ok(rr)
        rr["tier_bucket"] = cal.hr_tier(rr)
        all_rows.append(rr)
        seen.add(key)

    meta = Counter()
    meta["candidate_log_rows_loaded"] = len(candidate_rows)
    meta["record_rows_loaded"] = len(record_rows)
    meta["candidate_rows_graded"] = len(graded_candidates)
    for k, v in reasons.items():
        meta[f"skip_{k}"] = v
    return all_rows, meta


def fetch_json(url: str, timeout: int = 12) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "PropEdgeHRContextProbe/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def cache_dir(api_mod: Any) -> Path:
    base = Path(getattr(api_mod, "DATA_DIR", "/data"))
    d = base / "hr_context_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_game_feed(api_mod: Any, game_id: Any) -> Optional[Dict[str, Any]]:
    if not game_id:
        return None
    gid = str(game_id)
    p = cache_dir(api_mod) / f"gamefeed_{gid}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass

    url = f"https://statsapi.mlb.com/api/v1.1/game/{gid}/feed/live"
    try:
        data = fetch_json(url)
        p.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(0.08)
        return data
    except Exception as e:
        # Store tiny error marker so repeated runs don't hammer failed calls.
        try:
            p.write_text(json.dumps({"_error": str(e)}), encoding="utf-8")
        except Exception:
            pass
        return None


def team_matches(a: Any, b: Any) -> bool:
    aa = norm(a)
    bb = norm(b)
    if not aa or not bb:
        return False
    return aa == bb or aa in bb or bb in aa


def first_number(text: Any) -> Optional[float]:
    m = re.search(r"-?\d+(?:\.\d+)?", str(text or ""))
    return float(m.group()) if m else None


def parse_wind_bucket(wind: Any) -> str:
    s = str(wind or "").lower()
    if not s or s in {"unknown", "none"}:
        return "wind_unknown"
    if "out" in s or "to " in s:
        return "wind_out"
    if "in" in s or "from " in s:
        return "wind_in"
    if "left" in s or "right" in s or "cross" in s:
        return "wind_cross"
    return "wind_other"


def parse_temp_bucket(temp: Optional[float]) -> str:
    if temp is None:
        return "temp_unknown"
    if temp < 60:
        return "temp_lt60"
    if temp < 70:
        return "temp_60_69"
    if temp < 80:
        return "temp_70_79"
    if temp < 90:
        return "temp_80_89"
    return "temp_90_plus"


def get_box_player(box_team: Dict[str, Any], pid: Any) -> Optional[Dict[str, Any]]:
    if not pid:
        return None
    players = box_team.get("players") or {}
    return players.get(f"ID{pid}") or players.get(str(pid))


def get_starting_pitcher_id(feed: Dict[str, Any], side: str) -> Optional[int]:
    gd = feed.get("gameData") or {}
    pp = gd.get("probablePitchers") or {}
    probable = pp.get(side) or {}
    if probable.get("id"):
        return int(probable["id"])

    box = ((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
    team = box.get(side) or {}
    for pid in team.get("pitchers") or []:
        pl = get_box_player(team, pid)
        stats = ((pl or {}).get("stats") or {}).get("pitching") or {}
        if stats.get("gamesStarted") == 1:
            try:
                return int(pid)
            except Exception:
                return None
    try:
        pitchers = team.get("pitchers") or []
        return int(pitchers[0]) if pitchers else None
    except Exception:
        return None


def get_batter_box_player(feed: Dict[str, Any], side: str, player_id: Any) -> Optional[Dict[str, Any]]:
    box = ((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
    team = box.get(side) or {}
    return get_box_player(team, player_id)


def extract_context(api_mod: Any, row: Dict[str, Any]) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "context_loaded": False,
        "context_error": None,
        "home_team": None,
        "away_team": None,
        "venue": None,
        "park_bucket": "park_unknown",
        "weather_temp_f": None,
        "weather_temp_bucket": "temp_unknown",
        "weather_condition": None,
        "weather_wind": None,
        "weather_wind_mph": None,
        "weather_wind_bucket": "wind_unknown",
        "batter_side": None,
        "opp_pitcher": None,
        "opp_pitcher_id": None,
        "opp_pitcher_hand": None,
        "platoon_bucket": "platoon_unknown",
        "is_home_batter": None,
    }

    feed = fetch_game_feed(api_mod, row.get("game_id"))
    if not feed or feed.get("_error"):
        ctx["context_error"] = (feed or {}).get("_error") or "feed_missing"
        return ctx

    try:
        gd = feed.get("gameData") or {}
        teams = gd.get("teams") or {}
        home = (teams.get("home") or {}).get("name")
        away = (teams.get("away") or {}).get("name")
        venue = (gd.get("venue") or {}).get("name")

        ctx["home_team"] = home
        ctx["away_team"] = away
        ctx["venue"] = venue
        ctx["park_bucket"] = PARK_BUCKET_BY_VENUE.get(str(venue or "").lower(), "park_unknown")

        w = gd.get("weather") or {}
        ctx["weather_condition"] = w.get("condition")
        temp = fnum(w.get("temp"))
        if temp is None:
            temp = first_number(w.get("temperature"))
        ctx["weather_temp_f"] = temp
        ctx["weather_temp_bucket"] = parse_temp_bucket(temp)
        wind = w.get("wind") or w.get("windSpeed")
        ctx["weather_wind"] = wind
        ctx["weather_wind_mph"] = first_number(wind)
        ctx["weather_wind_bucket"] = parse_wind_bucket(wind)

        side = None
        if team_matches(row.get("team"), home):
            side = "home"
        elif team_matches(row.get("team"), away):
            side = "away"

        ctx["is_home_batter"] = True if side == "home" else False if side == "away" else None

        if side:
            opp_side = "away" if side == "home" else "home"
            box = ((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
            batter = get_batter_box_player(feed, side, row.get("player_id"))
            if batter:
                bat_side = ((batter.get("person") or {}).get("batSide") or {}).get("code")
                if not bat_side:
                    bat_side = ((batter.get("batting") or {}).get("batSide") or {}).get("code")
                ctx["batter_side"] = bat_side

            spid = get_starting_pitcher_id(feed, opp_side)
            ctx["opp_pitcher_id"] = spid
            opp_box_team = box.get(opp_side) or {}
            sp = get_box_player(opp_box_team, spid)
            if sp:
                person = sp.get("person") or {}
                ctx["opp_pitcher"] = person.get("fullName")
                ph = (person.get("pitchHand") or {}).get("code")
                ctx["opp_pitcher_hand"] = ph

            b = str(ctx.get("batter_side") or "").upper()
            p = str(ctx.get("opp_pitcher_hand") or "").upper()
            if b and p:
                if b == "S":
                    ctx["platoon_bucket"] = "switch_hitter"
                elif (b == "L" and p == "R") or (b == "R" and p == "L"):
                    ctx["platoon_bucket"] = "platoon_adv"
                elif b in {"L", "R"} and p in {"L", "R"}:
                    ctx["platoon_bucket"] = "same_hand"
                else:
                    ctx["platoon_bucket"] = "platoon_unknown"

        ctx["context_loaded"] = True
        return ctx
    except Exception as e:
        ctx["context_error"] = f"{type(e).__name__}: {e}"
        return ctx


def enrich_rows(api_mod: Any, rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Counter]:
    out = []
    meta = Counter()
    for r in rows:
        rr = dict(r)
        ctx = extract_context(api_mod, rr)
        rr.update(ctx)
        meta["context_loaded" if ctx.get("context_loaded") else "context_missing"] += 1
        if ctx.get("context_error"):
            meta[f"context_error:{ctx.get('context_error')}"] += 1
        out.append(rr)
    return out, meta


def gate(rows: List[Dict[str, Any]], funcs: List[Callable[[Dict[str, Any]], bool]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        try:
            if all(fn(r) for fn in funcs):
                out.append(r)
        except Exception:
            pass
    return out


def has_bad_flag(row: Dict[str, Any], text: str) -> bool:
    hay = " ".join(str(row.get(k) or "") for k in ("bvp_flag", "reject_reason", "prediction_tier", "board_warnings"))
    return text.lower() in hay.lower()


def build_context_tweaks() -> List[Tuple[str, Callable[[Dict[str, Any]], bool]]]:
    tweaks: List[Tuple[str, Callable[[Dict[str, Any]], bool]]] = [
        ("park_hitterish", lambda r: r.get("park_bucket") in HITTERISH_PARK_BUCKETS),
        ("not_pitcher_park", lambda r: r.get("park_bucket") not in BAD_PARK_BUCKETS),
        ("park_hr_friendly_or_better", lambda r: r.get("park_bucket") in {"elite_hr_park", "hr_friendly"}),
        ("temp>=70", lambda r: (fnum(r.get("weather_temp_f"), -999) or -999) >= 70),
        ("temp>=75", lambda r: (fnum(r.get("weather_temp_f"), -999) or -999) >= 75),
        ("temp>=80", lambda r: (fnum(r.get("weather_temp_f"), -999) or -999) >= 80),
        ("wind_out", lambda r: r.get("weather_wind_bucket") == "wind_out"),
        ("not_wind_in", lambda r: r.get("weather_wind_bucket") != "wind_in"),
        ("platoon_adv", lambda r: r.get("platoon_bucket") == "platoon_adv"),
        ("not_same_hand", lambda r: r.get("platoon_bucket") != "same_hand"),
        ("vs_lefty", lambda r: r.get("opp_pitcher_hand") == "L"),
        ("vs_righty", lambda r: r.get("opp_pitcher_hand") == "R"),
        ("home_batter", lambda r: r.get("is_home_batter") is True),
        ("away_batter", lambda r: r.get("is_home_batter") is False),
        ("lineup_spot<=3", lambda r: (fnum(r.get("lineup_spot"), 99) or 99) <= 3),
        ("lineup_spot<=5", lambda r: (fnum(r.get("lineup_spot"), 99) or 99) <= 5),
        ("quality_ok", lambda r: bool(r.get("quality_ok"))),
        ("hr_elite", lambda r: (r.get("tier_bucket") or cal.hr_tier(r)) == "hr_elite"),
        ("recent_iso>=.300", lambda r: (fnum(r.get("recent_iso"), -999) or -999) >= 0.300),
        ("recent_iso>=.350", lambda r: (fnum(r.get("recent_iso"), -999) or -999) >= 0.350),
        ("h2h_slg>=.700", lambda r: (fnum(r.get("h2h_slg"), -999) or -999) >= 0.700),
        ("h2h_slg>=.900", lambda r: (fnum(r.get("h2h_slg"), -999) or -999) >= 0.900),
        ("not_sum_avoid", lambda r: not has_bad_flag(r, "sum_avoid")),
        ("not_struggles", lambda r: not has_bad_flag(r, "struggles")),
        ("not_sum_lean", lambda r: not has_bad_flag(r, "sum_lean")),
    ]
    return tweaks


def base_funcs(score: float, slg: float) -> List[Callable[[Dict[str, Any]], bool]]:
    return [
        lambda r, score=score: (fnum(r.get("hr_score"), -999) or -999) >= score,
        lambda r, slg=slg: (fnum(r.get("season_slg"), -999) or -999) >= slg,
    ]


def print_group(title: str, rows: List[Dict[str, Any]], key: str, baseline: float, min_n: int) -> None:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        k = str(r.get(key) if r.get(key) is not None else "unknown")
        groups.setdefault(k, []).append(r)

    print(f"\n{title}")
    print("-" * len(title))
    items = []
    for k, rs in groups.items():
        if len(rs) >= min_n:
            s = summarize(rs, baseline)
            items.append((s["rate"], s["n"], k, s))
    if not items:
        print(f"No groups with n >= {min_n}")
        return
    for _, _, k, s in sorted(items, reverse=True):
        print(label_summary(f"{key}={k}", s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--date", default=None)
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--base-score", type=float, default=1.70)
    ap.add_argument("--base-slg", type=float, default=0.450)
    ap.add_argument("--max-grade-rows", type=int, default=0)
    ap.add_argument("--write-csv", action="store_true")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    print("HR CONTEXT FACTOR PROBE")
    print("=======================")
    print("No api.py changes. Tests park/weather/pitcher/platoon context on HR bucket.")

    api_mod = importlib.import_module("api")
    print(f"api_version: {getattr(api_mod, 'VERSION', 'unknown')}")
    print(f"data_dir: {getattr(api_mod, 'DATA_DIR', '/data')}")
    print(f"filters: days={args.days} date={args.date} min_n={args.min_n}")

    rows, meta = load_all_rows(api_mod, args.days, args.date, args.max_grade_rows)
    n0 = len(rows)
    hits0 = sum(1 for r in rows if r.get("result") == "hit")
    baseline = hits0 / n0 if n0 else 0.0

    print("\nINPUT / BASELINE")
    print("----------------")
    for k, v in meta.items():
        print(f"{k}: {v}")
    print(f"combined_unique_graded_hr_rows: {n0}")
    print(f"hits: {hits0}")
    print(f"misses: {n0 - hits0}")
    print(f"baseline_hr_rate: {baseline:.1%}")

    if not rows:
        return 0

    rows, ctx_meta = enrich_rows(api_mod, rows)
    print("\nCONTEXT ENRICHMENT COUNTS")
    print("-------------------------")
    for k, v in ctx_meta.most_common():
        print(f"{k}: {v}")

    bfuncs = base_funcs(args.base_score, args.base_slg)
    base = gate(rows, bfuncs)
    base_s = summarize(base, baseline)
    base_label = f"BASE_30_MODEL: hr_score>={args.base_score:.2f} AND season_slg>={args.base_slg:.3f}"

    print("\nBASE 30 CONTEXT PROFILE")
    print("-----------------------")
    print(label_summary(base_label, base_s))
    print_group("BASE BY PARK BUCKET", base, "park_bucket", baseline, 1)
    print_group("BASE BY TEMP BUCKET", base, "weather_temp_bucket", baseline, 1)
    print_group("BASE BY WIND BUCKET", base, "weather_wind_bucket", baseline, 1)
    print_group("BASE BY PLATOON", base, "platoon_bucket", baseline, 1)
    print_group("BASE BY OPP PITCHER HAND", base, "opp_pitcher_hand", baseline, 1)

    print("\nBASE MEMBERS WITH CONTEXT")
    print("-------------------------")
    for r in base:
        print(
            f"{str(r.get('result','?')):4s} {str(r.get('player')):22s} "
            f"score={fnum(r.get('hr_score'))} slg={fnum(r.get('season_slg'))} iso={fnum(r.get('recent_iso'))} "
            f"park={r.get('park_bucket')} temp={r.get('weather_temp_f')} wind={r.get('weather_wind_bucket')} "
            f"bat={r.get('batter_side')} vs={r.get('opp_pitcher_hand')} platoon={r.get('platoon_bucket')} "
            f"oppSP={r.get('opp_pitcher')} venue={r.get('venue')}"
        )

    print("\nCONTEXT TWEAKS ON BASE")
    print("----------------------")
    results = []
    for label, fn in build_context_tweaks():
        filt = gate(base, [fn])
        if len(filt) < args.min_n:
            continue
        s = summarize(filt, baseline)
        results.append((s["rate"], s["wilson80_lower"], s["penalized_score"], s["n"], label, s))

    if not results:
        print(f"No context tweaks had n >= {args.min_n}.")
    else:
        for rate, lower, pen, n, label, s in sorted(results, key=lambda x: (x[0], x[1], x[3]), reverse=True)[:args.top]:
            print(f"{label_summary('BASE + ' + label, s)}  delta_vs_base={s['rate'] - base_s['rate']:+.1%}")

    print("\nFULL CONTEXT SEARCH")
    print("-------------------")
    full_rules: List[Tuple[str, List[Callable[[Dict[str, Any]], bool]]]] = []
    tweaks = build_context_tweaks()

    # Base + each one/two context tweak combos.
    for label, fn in tweaks:
        full_rules.append((f"{base_label} AND {label}", bfuncs + [fn]))
    for (la, fa), (lb, fb) in itertools.combinations(tweaks, 2):
        full_rules.append((f"{base_label} AND {la} AND {lb}", bfuncs + [fa, fb]))

    # Challenger rules that may identify context-first HR spots.
    for score in [1.60, 1.70, 1.80, 1.90]:
        for label, fn in tweaks:
            full_rules.append((
                f"hr_score>={score:.2f} AND {label}",
                [lambda r, score=score: (fnum(r.get('hr_score'), -999) or -999) >= score, fn]
            ))

    search = []
    seen = set()
    for label, funcs in full_rules:
        if label in seen:
            continue
        seen.add(label)
        filt = gate(rows, funcs)
        if len(filt) < args.min_n:
            continue
        s = summarize(filt, baseline)
        search.append((s["rate"], s["wilson80_lower"], s["penalized_score"], s["n"], label, s))

    for rate, lower, pen, n, label, s in sorted(search, key=lambda x: (x[0], x[1], x[3]), reverse=True)[:args.top]:
        print(label_summary(label, s))

    print("\nPENALIZED CONTEXT RANKING")
    print("-------------------------")
    print("Uses Wilson lower bound * log(sample), so tiny 2/3 traps do not automatically win.")
    for rate, lower, pen, n, label, s in sorted(search, key=lambda x: (x[2], x[0], x[3]), reverse=True)[:args.top]:
        print(f"pen={pen:.4f}  {label_summary(label, s)}")

    print("\nDATA GAPS")
    print("---------")
    def missing_count(field: str) -> int:
        return sum(1 for r in rows if r.get(field) in (None, "", "unknown", "park_unknown", "temp_unknown", "wind_unknown", "platoon_unknown"))
    for field in ["venue", "park_bucket", "weather_temp_f", "weather_wind_bucket", "opp_pitcher_hand", "platoon_bucket"]:
        print(f"{field}_missing_or_unknown: {missing_count(field)} / {len(rows)}")

    print("\nRECOMMENDED NEXT READ")
    print("---------------------")
    conservative = [x for x in search if x[5]["n"] >= max(args.min_n, 10)]
    if conservative:
        best = max(conservative, key=lambda x: (x[2], x[0], x[3]))
        rate, lower, pen, n, label, s = best
        print(f"Best conservative context rule: {label}")
        print(label_summary(label, s))
        if s["rate"] > base_s["rate"] and s["n"] >= 10:
            print("READ: Context improved the 30% bucket in-sample. Needs more rows/out-of-sample before patching official HR.")
        elif s["rate"] == base_s["rate"]:
            print("READ: Context did not beat the current 30% base with enough sample yet.")
        else:
            print("READ: Context did not improve the base rule yet.")
    else:
        print("No conservative context rule had enough rows. Use min-n 3 only as exploratory.")

    print("\nNEXT PATCH DIRECTION IF SIGNAL HOLDS")
    print("------------------------------------")
    print("1. Keep HR watchlist only.")
    print("2. Add context fields to HR logs permanently: venue, park_bucket, temp, wind, opp starter hand, platoon.")
    print("3. Do not promote HR until sample grows and real HR odds are available.")
    print("4. Use context only as a boost/filter, not as a fake 60% probability.")

    if args.write_csv:
        out = Path("hr_context_factor_rows.csv")
        fields = sorted({k for r in rows for k in r.keys()})
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nCSV_WRITTEN: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
