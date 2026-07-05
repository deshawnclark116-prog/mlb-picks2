#!/usr/bin/env python3
"""
HR Contact Quality Probe for Prop Edge MLB API.

Diagnostic only. Does NOT modify api.py.

This tests the exact missing HR ingredients when they can be sourced:
- batter_barrel_rate
- recent_barrel_rate
- hard_hit_rate / EV
- launch_angle band
- pull_air_rate
- pitcher_barrel_allowed_rate
- pitcher_fly_ball_rate
- pitcher HR/FB
- park factor by batter handedness
- wind toward pull field
- FanDuel HR odds

Important:
- Uses Baseball Savant Statcast CSV when possible.
- Uses only data available up to each candidate's generated_at date to reduce future leakage.
- Caches Statcast pulls under /data/hr_feature_cache when /data exists.
- FanDuel HR odds and park-by-handedness/wind-to-pull-field are audited as missing unless already present;
  they require a separate odds/park-orientation data source.

Requires existing files beside api.py:
- hr_calibration_probe.py
- hr_context_factor_probe.py with corrected V2 content

Run:
    python hr_contact_quality_probe.py --days 180 --min-n 3 --base-only

Then:
    python hr_contact_quality_probe.py --days 180 --min-n 5 --base-only

Full but slower:
    python hr_contact_quality_probe.py --days 180 --min-n 3
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

try:
    import hr_calibration_probe as cal
except Exception as e:
    raise SystemExit(f"ERROR: missing hr_calibration_probe.py: {e}")

try:
    import hr_context_factor_probe as ctx
except Exception as e:
    raise SystemExit(f"ERROR: missing corrected hr_context_factor_probe.py: {e}")

SAVANT_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"
MLB_SEARCH = "https://statsapi.mlb.com/api/v1/people/search"

CACHE_DIR = Path(os.environ.get("HR_FEATURE_CACHE_DIR", "/data/hr_feature_cache"))
if not CACHE_DIR.parent.exists():
    CACHE_DIR = Path("./hr_feature_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def pct(x: Optional[float]) -> str:
    if x is None:
        return "None"
    return f"{x*100:.1f}%"


def parse_date(s: Any) -> Optional[dt.date]:
    if not s:
        return None
    text = str(s)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if not m:
        return None
    try:
        return dt.date.fromisoformat(m.group(1))
    except Exception:
        return None


def start_of_mlb_season(d: dt.date) -> dt.date:
    return dt.date(d.year, 3, 1)


def cache_key(parts: List[Any]) -> Path:
    raw = "|".join(str(p) for p in parts)
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.json"


def read_cache(path: Path) -> Optional[Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def write_cache(path: Path, data: Any) -> None:
    try:
        path.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def http_get_json(url: str, params: Dict[str, Any], timeout: int = 20) -> Optional[Dict[str, Any]]:
    key = cache_key(["json", url, sorted(params.items())])
    cached = read_cache(key)
    if cached is not None:
        return cached
    try:
        r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "PropEdgeHRProbe/1.0"})
        if r.status_code != 200:
            return None
        data = r.json()
        write_cache(key, data)
        time.sleep(0.15)
        return data
    except Exception:
        return None


def lookup_player_id_by_name(name: str) -> Optional[int]:
    if not name:
        return None
    data = http_get_json(MLB_SEARCH, {"names": name})
    if not data:
        return None
    people = data.get("people") or []
    if not people:
        return None
    name_norm = re.sub(r"[^a-z]", "", name.lower())
    for p in people:
        full = p.get("fullName") or ""
        if re.sub(r"[^a-z]", "", full.lower()) == name_norm:
            return p.get("id")
    return people[0].get("id")


def savant_rows(player_id: int, player_type: str, start_date: dt.date, end_date: dt.date) -> List[Dict[str, Any]]:
    if not player_id:
        return []
    params = {
        "all": "true",
        "hfPT": "",
        "hfAB": "",
        "hfGT": "R|",
        "hfPR": "",
        "hfZ": "",
        "stadium": "",
        "hfBBL": "",
        "hfNewZones": "",
        "hfPull": "",
        "hfC": "",
        "hfSea": "",
        "hfSit": "",
        "player_type": player_type,
        "hfOuts": "",
        "opponent": "",
        "pitcher_throws": "",
        "batter_stands": "",
        "hfSA": "",
        "game_date_gt": start_date.isoformat(),
        "game_date_lt": end_date.isoformat(),
        "team": "",
        "position": "",
        "hfRO": "",
        "home_road": "",
        "hfFlag": "",
        "hfBBT": "",
        "metric_1": "",
        "group_by": "name",
        "sort_col": "pitches",
        "player_event_sort": "h_launch_speed",
        "sort_order": "desc",
        "min_pitches": "0",
        "min_results": "0",
        "type": "details",
        "player_id": str(player_id),
    }
    key = cache_key(["savant", player_id, player_type, start_date.isoformat(), end_date.isoformat()])
    cached = read_cache(key)
    if cached is not None:
        return cached
    try:
        r = requests.get(SAVANT_CSV, params=params, timeout=45, headers={"User-Agent": "PropEdgeHRProbe/1.0"})
        if r.status_code != 200 or not r.text.strip():
            return []
        rows = list(csv.DictReader(r.text.splitlines()))
        write_cache(key, rows)
        time.sleep(0.25)
        return rows
    except Exception:
        return []


def is_bbe(row: Dict[str, Any]) -> bool:
    ev = safe_float(row.get("launch_speed"))
    la = safe_float(row.get("launch_angle"))
    return ev is not None and la is not None


def is_barrel(row: Dict[str, Any]) -> bool:
    return str(row.get("launch_speed_angle") or "").strip() == "6"


def percentile(values: List[float], p: float) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    idx = (len(vals) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return vals[int(idx)]
    return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)


def batted_ball_features(rows: List[Dict[str, Any]], role: str, batter_side: Optional[str] = None) -> Dict[str, Any]:
    bbe = [r for r in rows if is_bbe(r)]
    n = len(bbe)
    evs = [safe_float(r.get("launch_speed")) for r in bbe if safe_float(r.get("launch_speed")) is not None]
    las = [safe_float(r.get("launch_angle")) for r in bbe if safe_float(r.get("launch_angle")) is not None]

    barrels = sum(1 for r in bbe if is_barrel(r))
    hard = sum(1 for r in bbe if (safe_float(r.get("launch_speed"), 0) or 0) >= 95)
    la_band = sum(1 for r in bbe if 20 <= (safe_float(r.get("launch_angle"), -999) or -999) <= 35)
    air = [r for r in bbe if str(r.get("bb_type") or "").lower() in ("fly_ball", "line_drive")]

    # Approximate pull-air using Savant hc_x. This is diagnostic only.
    pull_air = 0
    pull_air_known = 0
    side = (batter_side or "").upper()
    for r in air:
        x = safe_float(r.get("hc_x"))
        if x is None or side not in ("R", "L"):
            continue
        pull_air_known += 1
        if side == "R" and x < 125:
            pull_air += 1
        elif side == "L" and x > 125:
            pull_air += 1

    fly_balls = [r for r in bbe if str(r.get("bb_type") or "").lower() == "fly_ball"]
    hrs = [r for r in rows if str(r.get("events") or "").lower() == "home_run"]

    return {
        f"{role}_statcast_rows": len(rows),
        f"{role}_bbe": n,
        f"{role}_barrels": barrels,
        f"{role}_barrel_rate": barrels / n if n else None,
        f"{role}_hard_hit_rate": hard / n if n else None,
        f"{role}_avg_ev": sum(evs) / len(evs) if evs else None,
        f"{role}_ev90": percentile(evs, 0.90),
        f"{role}_avg_la": sum(las) / len(las) if las else None,
        f"{role}_la_20_35_rate": la_band / n if n else None,
        f"{role}_fly_ball_rate": len(fly_balls) / n if n else None,
        f"{role}_hr_per_fb": len(hrs) / len(fly_balls) if fly_balls else None,
        f"{role}_pull_air_rate": pull_air / pull_air_known if pull_air_known else None,
        f"{role}_pull_air_known": pull_air_known,
    }


def load_rows(api_mod: Any, days: int, exact_date: Optional[str]) -> List[Dict[str, Any]]:
    rows, _meta = ctx.load_all_rows(api_mod, days, exact_date, 0)
    out = []
    for r in rows:
        try:
            out.append(ctx.enrich_row(api_mod, r))
        except Exception:
            out.append(dict(r))
    return out


def is_base(row: Dict[str, Any]) -> bool:
    return (safe_float(row.get("hr_score"), -999) or -999) >= 1.70 and (safe_float(row.get("season_slg"), -999) or -999) >= 0.450


def get_batter_id(row: Dict[str, Any]) -> Optional[int]:
    for k in ("player_id", "batter_id", "mlb_id"):
        v = row.get(k)
        if v:
            try:
                return int(v)
            except Exception:
                pass
    return None


def get_pitcher_id(row: Dict[str, Any]) -> Optional[int]:
    for k in ("opp_pitcher_id", "opposing_pitcher_id", "pitcher_id", "starter_id", "opponent_pitcher_id"):
        v = row.get(k)
        if v:
            try:
                return int(v)
            except Exception:
                pass
    name = row.get("opp_pitcher") or row.get("opposing_pitcher")
    if name:
        return lookup_player_id_by_name(str(name))
    return None


def enrich_contact_features(row: Dict[str, Any], lookback_days: int, recent_days: int) -> Dict[str, Any]:
    rr = dict(row)
    end = parse_date(row.get("generated_at")) or dt.date.today()
    season_start = start_of_mlb_season(end)
    recent_start = max(season_start, end - dt.timedelta(days=recent_days))

    batter_id = get_batter_id(row)
    pitcher_id = get_pitcher_id(row)
    batter_side = row.get("batter_side")

    rr["contact_feature_end_date"] = end.isoformat()
    rr["batter_statcast_id"] = batter_id
    rr["pitcher_statcast_id"] = pitcher_id

    if batter_id:
        batter_season_rows = savant_rows(batter_id, "batter", season_start, end)
        batter_recent_rows = savant_rows(batter_id, "batter", recent_start, end)
        rr.update(batted_ball_features(batter_season_rows, "batter", batter_side=batter_side))
        recent_feat = batted_ball_features(batter_recent_rows, "recent_batter", batter_side=batter_side)
        rr.update(recent_feat)
        rr["recent_barrel_rate"] = recent_feat.get("recent_batter_barrel_rate")
    else:
        rr["batter_statcast_missing_reason"] = "missing_batter_id"

    if pitcher_id:
        pitcher_rows = savant_rows(pitcher_id, "pitcher", season_start, end)
        pf = batted_ball_features(pitcher_rows, "pitcher_allowed", batter_side=batter_side)
        rr.update(pf)
        rr["pitcher_barrel_allowed_rate"] = pf.get("pitcher_allowed_barrel_rate")
        rr["pitcher_fly_ball_rate"] = pf.get("pitcher_allowed_fly_ball_rate")
        rr["pitcher_hrfb_rate"] = pf.get("pitcher_allowed_hr_per_fb")
    else:
        rr["pitcher_statcast_missing_reason"] = "missing_pitcher_id"

    rr["park_factor_by_batter_hand_available"] = False
    rr["wind_toward_pull_field_available"] = False
    rr["fanduel_hr_odds_available"] = bool(row.get("odds") and row.get("prop_type") == "batter_home_runs")
    return rr


def feature_flags(row: Dict[str, Any]) -> Dict[str, bool]:
    def ge(k: str, t: float) -> bool:
        v = safe_float(row.get(k))
        return v is not None and v >= t

    return {
        "batter_barrel_10": ge("batter_barrel_rate", 0.10),
        "batter_barrel_12": ge("batter_barrel_rate", 0.12),
        "recent_barrel_10": ge("recent_barrel_rate", 0.10),
        "recent_barrel_12": ge("recent_barrel_rate", 0.12),
        "hard_hit_45": ge("batter_hard_hit_rate", 0.45),
        "hard_hit_50": ge("batter_hard_hit_rate", 0.50),
        "ev90_103": ge("batter_ev90", 103),
        "ev90_105": ge("batter_ev90", 105),
        "la_band_25": ge("batter_la_20_35_rate", 0.25),
        "la_band_30": ge("batter_la_20_35_rate", 0.30),
        "pull_air_25": ge("batter_pull_air_rate", 0.25),
        "pull_air_35": ge("batter_pull_air_rate", 0.35),
        "pitcher_barrel_allowed_08": ge("pitcher_barrel_allowed_rate", 0.08),
        "pitcher_barrel_allowed_10": ge("pitcher_barrel_allowed_rate", 0.10),
        "pitcher_fb_35": ge("pitcher_fly_ball_rate", 0.35),
        "pitcher_hrfb_12": ge("pitcher_hrfb_rate", 0.12),
        "pitcher_hrfb_15": ge("pitcher_hrfb_rate", 0.15),
    }


def summarize(rows: List[Dict[str, Any]], baseline: float) -> Dict[str, Any]:
    n = len(rows)
    h = sum(1 for r in rows if r.get("result") == "hit")
    rate = h / n if n else 0.0
    return {"n": n, "hits": h, "misses": n - h, "rate": rate, "lift": rate / baseline if baseline else None}


def fmt(label: str, s: Dict[str, Any]) -> str:
    lift = f"{s['lift']:.2f}x" if s["lift"] is not None else "n/a"
    return f"{s['rate']:.1%} n={s['n']} hit={s['hits']} miss={s['misses']} lift={lift}  {label}"


def avg(rows: List[Dict[str, Any]], field: str) -> Optional[float]:
    vals = [safe_float(r.get(field)) for r in rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--date", default=None)
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--recent-days", type=int, default=30)
    ap.add_argument("--base-only", action="store_true", help="Only fetch Statcast for base HR bucket rows.")
    ap.add_argument("--write-csv", action="store_true")
    args = ap.parse_args()

    print("HR CONTACT QUALITY PROBE")
    print("========================")
    print("Diagnostic only. Tests actual HR contact-quality ingredients when Statcast data is available.")

    api_mod = importlib.import_module("api")
    print(f"api_version: {getattr(api_mod, 'VERSION', 'unknown')}")
    print(f"data_dir: {getattr(api_mod, 'DATA_DIR', '/data')}")
    print(f"filters: days={args.days} date={args.date} min_n={args.min_n}")

    rows = load_rows(api_mod, args.days, args.date)
    all_hits = sum(1 for r in rows if r.get("result") == "hit")
    all_rate = all_hits / len(rows) if rows else 0.0

    target_rows = [r for r in rows if is_base(r)] if args.base_only else rows
    print("\nINPUT")
    print("-----")
    print(f"all_graded_hr_rows: {len(rows)}")
    print(f"all_hits: {all_hits}")
    print(f"all_hr_rate: {all_rate:.1%}")
    print(f"rows_to_fetch_contact_features: {len(target_rows)}")

    enriched_target = []
    for i, r in enumerate(target_rows, start=1):
        print(f"fetching_contact_features {i}/{len(target_rows)} {r.get('player')} {r.get('generated_at')}")
        enriched_target.append(enrich_contact_features(r, args.days, args.recent_days))

    if args.base_only:
        enriched = enriched_target
    else:
        enriched = enriched_target

    base = [r for r in enriched if is_base(r)]
    base_s = summarize(base, all_rate)

    print("\nBASE MODEL")
    print("----------")
    print(fmt("hr_score>=1.70 AND season_slg>=.450", base_s))

    print("\nDEEP FEATURE AVAILABILITY IN BASE BUCKET")
    print("----------------------------------------")
    exact_fields = {
        "batter_barrel_rate": "batter_barrel_rate",
        "recent_barrel_rate": "recent_barrel_rate",
        "hard_hit_rate / EV": "batter_hard_hit_rate",
        "EV90": "batter_ev90",
        "launch_angle band": "batter_la_20_35_rate",
        "pull_air_rate": "batter_pull_air_rate",
        "pitcher_barrel_allowed_rate": "pitcher_barrel_allowed_rate",
        "pitcher_fly_ball_rate": "pitcher_fly_ball_rate",
        "pitcher HR/FB": "pitcher_hrfb_rate",
        "park factor by batter handedness": "park_factor_by_batter_hand_available",
        "wind toward pull field": "wind_toward_pull_field_available",
        "FanDuel HR odds": "fanduel_hr_odds_available",
    }
    for label, field in exact_fields.items():
        if field.endswith("_available"):
            n_have = sum(1 for r in base if bool(r.get(field)))
        else:
            n_have = sum(1 for r in base if safe_float(r.get(field)) is not None)
        print(f"{label:34s} {n_have}/{len(base)}")

    print("\nBASE BUCKET WITH CONTACT FEATURES")
    print("---------------------------------")
    for r in base:
        print(
            f"{str(r.get('result')):4s} {str(r.get('player')):22s} "
            f"barrel={pct(safe_float(r.get('batter_barrel_rate')))} "
            f"recent_barrel={pct(safe_float(r.get('recent_barrel_rate')))} "
            f"hardhit={pct(safe_float(r.get('batter_hard_hit_rate')))} "
            f"ev90={None if safe_float(r.get('batter_ev90')) is None else round(safe_float(r.get('batter_ev90')),1)} "
            f"LA20_35={pct(safe_float(r.get('batter_la_20_35_rate')))} "
            f"pull_air={pct(safe_float(r.get('batter_pull_air_rate')))} "
            f"p_barrel={pct(safe_float(r.get('pitcher_barrel_allowed_rate')))} "
            f"p_fb={pct(safe_float(r.get('pitcher_fly_ball_rate')))} "
            f"p_hrfb={pct(safe_float(r.get('pitcher_hrfb_rate')))} "
            f"bbe={r.get('batter_bbe')} recent_bbe={r.get('recent_batter_bbe')} p_bbe={r.get('pitcher_allowed_bbe')}"
        )

    print("\nHIT VS MISS CONTACT FEATURE AVERAGES INSIDE BASE")
    print("------------------------------------------------")
    hit_rows = [r for r in base if r.get("result") == "hit"]
    miss_rows = [r for r in base if r.get("result") != "hit"]
    avg_fields = [
        "batter_barrel_rate", "recent_barrel_rate", "batter_hard_hit_rate", "batter_ev90",
        "batter_la_20_35_rate", "batter_pull_air_rate",
        "pitcher_barrel_allowed_rate", "pitcher_fly_ball_rate", "pitcher_hrfb_rate"
    ]
    for field in avg_fields:
        ah = avg(hit_rows, field)
        am = avg(miss_rows, field)
        delta = None if ah is None or am is None else ah - am
        print(f"{field:32s} hit_avg={ah} miss_avg={am} delta={delta}")

    print("\nCONTACT FEATURE RULE SEARCH")
    print("---------------------------")
    flags = sorted(feature_flags(base[0]).keys()) if base else []
    results = []
    for fac in flags:
        filt = [r for r in base if feature_flags(r).get(fac)]
        if len(filt) >= args.min_n:
            s = summarize(filt, all_rate)
            results.append((s["rate"], fac, s))
    for rate, fac, s in sorted(results, key=lambda x: (x[0], x[2]["n"]), reverse=True):
        print(fmt(f"base + {fac}", s))

    print("\nMISSING EXACT FEATURES READ")
    print("---------------------------")
    print("If a feature shows 0/N availability, it is not currently testable from stored candidate rows.")
    print("park factor by batter handedness requires a park L/R HR factor table.")
    print("wind toward pull field requires park orientation + batter handedness + wind direction, not just wind bucket.")
    print("FanDuel HR odds require adding HR odds provider calls; current minimal provider path only gets pitcher Ks.")

    print("\nRECOMMENDED NEXT STEP")
    print("---------------------")
    print("If Statcast features loaded, compare the hit/miss averages above.")
    print("If they did not load, the real patch is an enriched HR logger that stores these fields daily.")
    print("Do not promote HR official until FanDuel HR odds and at least 100 graded HR candidates exist.")

    if args.write_csv:
        out = Path("hr_contact_quality_probe_rows.csv")
        fields = sorted({k for r in enriched for k in r.keys()})
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in enriched:
                w.writerow(r)
        print(f"\nCSV_WRITTEN: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
