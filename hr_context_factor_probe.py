#!/usr/bin/env python3
"""
HR Context Factor Probe v2 for Prop Edge MLB API.

Diagnostic only. Does NOT modify api.py.

Why v2 exists:
- v1 tested park/weather, but pitcher hand/platoon came back mostly unknown.
- v2 fixes batter/pitcher handedness extraction from MLB gameData.players.
- v2 adds opposing starter season pitcher factors:
    pitcher HR/9
    pitcher HR per batter faced
    pitcher K/9
    pitcher BB/9
    pitcher WHIP/ERA when available

Required:
- Put this file beside api.py
- Put hr_calibration_probe.py beside it too

Run:
    python hr_context_factor_probe_v2.py --days 180 --min-n 3

Conservative:
    python hr_context_factor_probe_v2.py --days 180 --min-n 10

Optional CSV:
    python hr_context_factor_probe_v2.py --days 180 --min-n 3 --write-csv
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

PARK_BUCKET_BY_VENUE = {
    "coors field": "elite_hr_park",
    "great american ball park": "hr_friendly",
    "citizens bank park": "hr_friendly",
    "yankee stadium": "hr_friendly",
    "guaranteed rate field": "hr_friendly",
    "american family field": "hr_friendly",
    "globe life field": "hr_friendly",
    "truist park": "slightly_hr_friendly",
    "dodger stadium": "slightly_hr_friendly",
    "fenway park": "slightly_hr_friendly",
    "target field": "neutral",
    "minute maid park": "neutral",
    "rogers centre": "neutral",
    "progressive field": "neutral",
    "chase field": "neutral",
    "nationals park": "neutral",
    "angel stadium": "neutral",
    "loanDepot park".lower(): "pitcher_friendly",
    "loandepot park": "pitcher_friendly",
    "t-mobile park": "pitcher_friendly",
    "petco park": "pitcher_friendly",
    "oracle park": "pitcher_friendly",
    "pnc park": "pitcher_friendly",
    "citi field": "pitcher_friendly",
    "busch stadium": "pitcher_friendly",
    "kauffman stadium": "pitcher_friendly",
    "comerica park": "pitcher_friendly",
    "oriole park at camden yards": "neutral",
    "wrigley field": "weather_sensitive",
    "tropicana field": "dome_neutral",
    "sutter health park": "unknown",
}
HITTERISH_PARK_BUCKETS = {"elite_hr_park", "hr_friendly", "slightly_hr_friendly", "weather_sensitive"}
BAD_PARK_BUCKETS = {"pitcher_friendly"}


def fnum(x: Any, default: Optional[float] = None) -> Optional[float]:
    return cal.safe_float(x, default)


def norm(s: Any) -> str:
    return cal.norm(s)


def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def innings_to_float(ip: Any) -> Optional[float]:
    if ip is None:
        return None
    s = str(ip).strip()
    if not s:
        return None
    try:
        if "." not in s:
            return float(s)
        whole, frac = s.split(".", 1)
        whole_i = int(whole or "0")
        outs = int((frac or "0")[0])
        if outs not in (0, 1, 2):
            return float(s)
        return whole_i + outs / 3.0
    except Exception:
        return fnum(ip)


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
        "avg_pitcher_hr9": avg("opp_pitcher_hr9"),
        "avg_pitcher_hr_bf": avg("opp_pitcher_hr_per_bf"),
    }


def label_summary(label: str, s: Dict[str, Any]) -> str:
    lift = f"{s['lift']:.2f}x" if s["lift"] is not None else "n/a"
    return (
        f"{s['rate']:6.1%} n={s['n']:3d} hit={s['hits']:3d} miss={s['misses']:3d} "
        f"lb80={s['wilson80_lower']:5.1%} lift={lift:>6s} "
        f"avg_score={round(s['avg_hr_score'],3) if s['avg_hr_score'] is not None else None} "
        f"avg_slg={round(s['avg_season_slg'],3) if s['avg_season_slg'] is not None else None} "
        f"avg_iso={round(s['avg_recent_iso'],3) if s['avg_recent_iso'] is not None else None} "
        f"avg_temp={round(s['avg_temp_f'],1) if s['avg_temp_f'] is not None else None} "
        f"avg_pHR9={round(s['avg_pitcher_hr9'],2) if s['avg_pitcher_hr9'] is not None else None}  "
        f"{label}"
    )


def fetch_json(url: str, timeout: int = 12) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "PropEdgeHRContextProbeV2/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def data_cache_dir(api_mod: Any) -> Path:
    base = Path(getattr(api_mod, "DATA_DIR", "/data"))
    d = base / "hr_context_cache_v2"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_game_feed(api_mod: Any, game_id: Any) -> Optional[Dict[str, Any]]:
    if not game_id:
        return None
    gid = str(game_id)
    p = data_cache_dir(api_mod) / f"gamefeed_{gid}.json"
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
        try:
            p.write_text(json.dumps({"_error": str(e)}), encoding="utf-8")
        except Exception:
            pass
        return None


def fetch_pitcher_season_stats(api_mod: Any, pitcher_id: Any, season: Any) -> Dict[str, Any]:
    out = {
        "opp_pitcher_stats_loaded": False,
        "opp_pitcher_ip": None,
        "opp_pitcher_hr": None,
        "opp_pitcher_bf": None,
        "opp_pitcher_hr9": None,
        "opp_pitcher_hr_per_bf": None,
        "opp_pitcher_k9": None,
        "opp_pitcher_bb9": None,
        "opp_pitcher_era": None,
        "opp_pitcher_whip": None,
        "opp_pitcher_gs": None,
    }
    if not pitcher_id or not season:
        return out
    pid = str(pitcher_id)
    yr = str(season)
    p = data_cache_dir(api_mod) / f"pitcher_season_{pid}_{yr}.json"
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception:
            data = None
    else:
        url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=season&group=pitching&season={yr}"
        try:
            data = fetch_json(url)
            p.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(0.08)
        except Exception as e:
            data = {"_error": str(e)}
            try:
                p.write_text(json.dumps(data), encoding="utf-8")
            except Exception:
                pass

    try:
        splits = (((data or {}).get("stats") or [{}])[0].get("splits") or [])
        if not splits:
            return out
        stat = splits[0].get("stat") or {}
        ip = innings_to_float(stat.get("inningsPitched"))
        hr = fnum(stat.get("homeRuns"))
        bf = fnum(stat.get("battersFaced"))
        k = fnum(stat.get("strikeOuts"))
        bb = fnum(stat.get("baseOnBalls"))
        out["opp_pitcher_stats_loaded"] = True
        out["opp_pitcher_ip"] = ip
        out["opp_pitcher_hr"] = hr
        out["opp_pitcher_bf"] = bf
        out["opp_pitcher_gs"] = fnum(stat.get("gamesStarted"))
        out["opp_pitcher_era"] = fnum(stat.get("era"))
        out["opp_pitcher_whip"] = fnum(stat.get("whip"))
        if ip and ip > 0:
            out["opp_pitcher_hr9"] = (hr or 0) * 9.0 / ip
            out["opp_pitcher_k9"] = (k or 0) * 9.0 / ip
            out["opp_pitcher_bb9"] = (bb or 0) * 9.0 / ip
        if bf and bf > 0:
            out["opp_pitcher_hr_per_bf"] = (hr or 0) / bf
    except Exception:
        pass
    return out


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


def team_matches(a: Any, b: Any) -> bool:
    aa = norm(a)
    bb = norm(b)
    if not aa or not bb:
        return False
    return aa == bb or aa in bb or bb in aa


def game_player(feed: Dict[str, Any], pid: Any) -> Optional[Dict[str, Any]]:
    if not pid:
        return None
    players = ((feed.get("gameData") or {}).get("players") or {})
    return players.get(f"ID{pid}") or players.get(str(pid))


def box_player(feed: Dict[str, Any], side: str, pid: Any) -> Optional[Dict[str, Any]]:
    if not pid:
        return None
    box = ((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
    team = box.get(side) or {}
    players = team.get("players") or {}
    return players.get(f"ID{pid}") or players.get(str(pid))


def starting_pitcher_id(feed: Dict[str, Any], side: str) -> Optional[int]:
    gd = feed.get("gameData") or {}
    pp = gd.get("probablePitchers") or {}
    probable = pp.get(side) or {}
    if probable.get("id"):
        try:
            return int(probable.get("id"))
        except Exception:
            pass

    box = ((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
    team = box.get(side) or {}
    pitchers = team.get("pitchers") or []
    for pid in pitchers:
        bp = box_player(feed, side, pid)
        stats = ((bp or {}).get("stats") or {}).get("pitching") or {}
        if stats.get("gamesStarted") == 1 or str(stats.get("gamesStarted")) == "1":
            try:
                return int(pid)
            except Exception:
                pass
    try:
        return int(pitchers[0]) if pitchers else None
    except Exception:
        return None


def official_date_year(feed: Dict[str, Any]) -> Optional[int]:
    gd = feed.get("gameData") or {}
    d = gd.get("datetime") or {}
    od = d.get("officialDate") or d.get("originalDate")
    if od:
        try:
            return int(str(od)[:4])
        except Exception:
            pass
    return None


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


def enrich_row(api_mod: Any, row: Dict[str, Any]) -> Dict[str, Any]:
    rr = dict(row)
    rr.update({
        "context_loaded": False,
        "home_team": None,
        "away_team": None,
        "venue": None,
        "park_bucket": "park_unknown",
        "weather_temp_f": None,
        "weather_temp_bucket": "temp_unknown",
        "weather_wind": None,
        "weather_wind_mph": None,
        "weather_wind_bucket": "wind_unknown",
        "batter_side": None,
        "opp_pitcher": None,
        "opp_pitcher_id": None,
        "opp_pitcher_hand": None,
        "platoon_bucket": "platoon_unknown",
        "is_home_batter": None,
    })

    feed = fetch_game_feed(api_mod, rr.get("game_id"))
    if not feed or feed.get("_error"):
        rr["context_error"] = (feed or {}).get("_error") or "feed_missing"
        rr.update(fetch_pitcher_season_stats(api_mod, None, None))
        return rr

    gd = feed.get("gameData") or {}
    teams = gd.get("teams") or {}
    home = (teams.get("home") or {}).get("name")
    away = (teams.get("away") or {}).get("name")
    venue = (gd.get("venue") or {}).get("name")
    rr["home_team"] = home
    rr["away_team"] = away
    rr["venue"] = venue
    rr["park_bucket"] = PARK_BUCKET_BY_VENUE.get(str(venue or "").lower(), "park_unknown")

    weather = gd.get("weather") or {}
    temp = fnum(weather.get("temp"))
    if temp is None:
        temp = first_number(weather.get("temperature"))
    wind = weather.get("wind") or weather.get("windSpeed")
    rr["weather_temp_f"] = temp
    rr["weather_temp_bucket"] = parse_temp_bucket(temp)
    rr["weather_wind"] = wind
    rr["weather_wind_mph"] = first_number(wind)
    rr["weather_wind_bucket"] = parse_wind_bucket(wind)

    side = None
    if team_matches(rr.get("team"), home):
        side = "home"
    elif team_matches(rr.get("team"), away):
        side = "away"

    rr["is_home_batter"] = True if side == "home" else False if side == "away" else None

    if side:
        opp_side = "away" if side == "home" else "home"

        gp_batter = game_player(feed, rr.get("player_id"))
        bp_batter = box_player(feed, side, rr.get("player_id"))
        bat_side = None
        for obj in (gp_batter, bp_batter):
            if not obj:
                continue
            bat_side = (((obj.get("batSide") or {}) if "batSide" in obj else ((obj.get("person") or {}).get("batSide") or {})).get("code"))
            if bat_side:
                break
        rr["batter_side"] = bat_side

        spid = starting_pitcher_id(feed, opp_side)
        rr["opp_pitcher_id"] = spid
        gp_sp = game_player(feed, spid)
        bp_sp = box_player(feed, opp_side, spid)
        opp_name = None
        opp_hand = None
        for obj in (gp_sp, bp_sp):
            if not obj:
                continue
            opp_name = opp_name or obj.get("fullName") or ((obj.get("person") or {}).get("fullName"))
            # gameData.players has pitchHand at top level; boxscore usually under person
            hand_obj = obj.get("pitchHand") if isinstance(obj.get("pitchHand"), dict) else ((obj.get("person") or {}).get("pitchHand") or {})
            opp_hand = opp_hand or hand_obj.get("code")
        rr["opp_pitcher"] = opp_name
        rr["opp_pitcher_hand"] = opp_hand

        b = str(rr.get("batter_side") or "").upper()
        p = str(rr.get("opp_pitcher_hand") or "").upper()
        if b and p:
            if b == "S":
                rr["platoon_bucket"] = "switch_hitter"
            elif (b == "L" and p == "R") or (b == "R" and p == "L"):
                rr["platoon_bucket"] = "platoon_adv"
            elif b in {"L", "R"} and p in {"L", "R"}:
                rr["platoon_bucket"] = "same_hand"

        year = official_date_year(feed)
        rr.update(fetch_pitcher_season_stats(api_mod, spid, year))

    else:
        rr.update(fetch_pitcher_season_stats(api_mod, None, None))

    rr["context_loaded"] = True
    return rr


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


def base_funcs(score: float, slg: float) -> List[Callable[[Dict[str, Any]], bool]]:
    return [
        lambda r, score=score: (fnum(r.get("hr_score"), -999) or -999) >= score,
        lambda r, slg=slg: (fnum(r.get("season_slg"), -999) or -999) >= slg,
    ]


def build_context_rules(base_label: str, bfuncs: List[Callable[[Dict[str, Any]], bool]]) -> List[Tuple[str, List[Callable[[Dict[str, Any]], bool]]]]:
    tweaks: List[Tuple[str, Callable[[Dict[str, Any]], bool]]] = [
        ("park_hitterish", lambda r: r.get("park_bucket") in HITTERISH_PARK_BUCKETS),
        ("not_pitcher_park", lambda r: r.get("park_bucket") not in BAD_PARK_BUCKETS),
        ("park_hr_friendly_or_better", lambda r: r.get("park_bucket") in {"elite_hr_park", "hr_friendly"}),
        ("temp>=75", lambda r: (fnum(r.get("weather_temp_f"), -999) or -999) >= 75),
        ("temp>=80", lambda r: (fnum(r.get("weather_temp_f"), -999) or -999) >= 80),
        ("temp>=85", lambda r: (fnum(r.get("weather_temp_f"), -999) or -999) >= 85),
        ("temp>=90", lambda r: (fnum(r.get("weather_temp_f"), -999) or -999) >= 90),
        ("wind_out", lambda r: r.get("weather_wind_bucket") == "wind_out"),
        ("not_wind_in", lambda r: r.get("weather_wind_bucket") != "wind_in"),
        ("platoon_adv", lambda r: r.get("platoon_bucket") == "platoon_adv"),
        ("not_same_hand", lambda r: r.get("platoon_bucket") != "same_hand"),
        ("same_hand", lambda r: r.get("platoon_bucket") == "same_hand"),
        ("switch_hitter", lambda r: r.get("platoon_bucket") == "switch_hitter"),
        ("vs_lefty", lambda r: r.get("opp_pitcher_hand") == "L"),
        ("vs_righty", lambda r: r.get("opp_pitcher_hand") == "R"),
        ("opp_pitcher_hr9>=1.0", lambda r: (fnum(r.get("opp_pitcher_hr9"), -999) or -999) >= 1.0),
        ("opp_pitcher_hr9>=1.2", lambda r: (fnum(r.get("opp_pitcher_hr9"), -999) or -999) >= 1.2),
        ("opp_pitcher_hr9>=1.5", lambda r: (fnum(r.get("opp_pitcher_hr9"), -999) or -999) >= 1.5),
        ("opp_pitcher_hr9>=1.8", lambda r: (fnum(r.get("opp_pitcher_hr9"), -999) or -999) >= 1.8),
        ("opp_pitcher_hr_per_bf>=.030", lambda r: (fnum(r.get("opp_pitcher_hr_per_bf"), -999) or -999) >= 0.030),
        ("opp_pitcher_hr_per_bf>=.040", lambda r: (fnum(r.get("opp_pitcher_hr_per_bf"), -999) or -999) >= 0.040),
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

    rules: List[Tuple[str, List[Callable[[Dict[str, Any]], bool]]]] = []
    for label, fn in tweaks:
        rules.append((f"{base_label} AND {label}", bfuncs + [fn]))
    for (la, fa), (lb, fb) in itertools.combinations(tweaks, 2):
        rules.append((f"{base_label} AND {la} AND {lb}", bfuncs + [fa, fb]))

    # Important "our tweaks" combos.
    combos = [
        ("CONTEXT_CORE: base AND temp>=80 AND not_same_hand", ["temp>=80", "not_same_hand"]),
        ("CONTEXT_POWER: base AND recent_iso>=.350 AND temp>=80", ["recent_iso>=.350", "temp>=80"]),
        ("CONTEXT_PITCHER: base AND opp_pitcher_hr9>=1.2", ["opp_pitcher_hr9>=1.2"]),
        ("CONTEXT_PARK_WEATHER: base AND park_hitterish AND temp>=80", ["park_hitterish", "temp>=80"]),
        ("CONTEXT_PLATOON_POWER: base AND platoon_adv AND recent_iso>=.300", ["platoon_adv", "recent_iso>=.300"]),
    ]
    fn_by_label = {label: fn for label, fn in tweaks}
    for label, names in combos:
        funcs = bfuncs + [fn_by_label[n] for n in names if n in fn_by_label]
        rules.append((label, funcs))

    seen = set()
    out = []
    for label, funcs in rules:
        if label not in seen:
            seen.add(label)
            out.append((label, funcs))
    return out


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
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    print("HR CONTEXT FACTOR PROBE V2")
    print("==========================")
    print("No api.py changes. Tests park/weather/pitcher/platoon context with fixed pitcher-hand extraction.")

    api_mod = importlib.import_module("api")
    print(f"api_version: {getattr(api_mod, 'VERSION', 'unknown')}")
    print(f"data_dir: {getattr(api_mod, 'DATA_DIR', '/data')}")
    print(f"filters: days={args.days} date={args.date} min_n={args.min_n}")

    rows, meta = load_all_rows(api_mod, args.days, args.date, args.max_grade_rows)
    enriched = [enrich_row(api_mod, r) for r in rows]

    n0 = len(enriched)
    hits0 = sum(1 for r in enriched if r.get("result") == "hit")
    baseline = hits0 / n0 if n0 else 0.0

    print("\nINPUT / BASELINE")
    print("----------------")
    for k, v in meta.items():
        print(f"{k}: {v}")
    print(f"combined_unique_graded_hr_rows: {n0}")
    print(f"hits: {hits0}")
    print(f"misses: {n0 - hits0}")
    print(f"baseline_hr_rate: {baseline:.1%}")

    if n0 == 0:
        return 0

    print("\nDATA COMPLETENESS")
    print("-----------------")
    for field in ["venue", "park_bucket", "weather_temp_f", "weather_wind_bucket", "batter_side", "opp_pitcher", "opp_pitcher_hand", "platoon_bucket", "opp_pitcher_hr9"]:
        missing = sum(1 for r in enriched if r.get(field) in (None, "", "unknown", "park_unknown", "temp_unknown", "wind_unknown", "platoon_unknown"))
        print(f"{field}_missing_or_unknown: {missing} / {n0}")

    bfuncs = base_funcs(args.base_score, args.base_slg)
    base_label = f"BASE_30_MODEL: hr_score>={args.base_score:.2f} AND season_slg>={args.base_slg:.3f}"
    base = gate(enriched, bfuncs)
    base_s = summarize(base, baseline)

    print("\nBASE 30 CONTEXT PROFILE")
    print("-----------------------")
    print(label_summary(base_label, base_s))
    print_group("BASE BY PARK BUCKET", base, "park_bucket", baseline, 1)
    print_group("BASE BY TEMP BUCKET", base, "weather_temp_bucket", baseline, 1)
    print_group("BASE BY WIND BUCKET", base, "weather_wind_bucket", baseline, 1)
    print_group("BASE BY BATTER SIDE", base, "batter_side", baseline, 1)
    print_group("BASE BY OPP PITCHER HAND", base, "opp_pitcher_hand", baseline, 1)
    print_group("BASE BY PLATOON", base, "platoon_bucket", baseline, 1)

    print("\nBASE MEMBERS WITH PITCHER CONTEXT")
    print("---------------------------------")
    for r in base:
        print(
            f"{str(r.get('result','?')):4s} {str(r.get('player')):22s} "
            f"score={fnum(r.get('hr_score'))} slg={fnum(r.get('season_slg'))} iso={fnum(r.get('recent_iso'))} "
            f"park={r.get('park_bucket')} temp={r.get('weather_temp_f')} wind={r.get('weather_wind_bucket')} "
            f"bat={r.get('batter_side')} vs={r.get('opp_pitcher_hand')} platoon={r.get('platoon_bucket')} "
            f"oppSP={r.get('opp_pitcher')} pHR9={None if r.get('opp_pitcher_hr9') is None else round(r.get('opp_pitcher_hr9'),2)} "
            f"venue={r.get('venue')}"
        )

    rules = build_context_rules(base_label, bfuncs)
    results = []
    for label, funcs in rules:
        filt = gate(enriched, funcs)
        if len(filt) < args.min_n:
            continue
        s = summarize(filt, baseline)
        results.append((s["rate"], s["wilson80_lower"], s["penalized_score"], s["n"], label, s))

    print("\nCONTEXT SEARCH BY HIT RATE")
    print("--------------------------")
    for rate, lower, pen, n, label, s in sorted(results, key=lambda x: (x[0], x[1], x[3]), reverse=True)[:args.top]:
        print(label_summary(label, s))

    print("\nPENALIZED CONTEXT RANKING")
    print("-------------------------")
    print("Uses Wilson lower bound * log(sample), so small buckets are penalized.")
    for rate, lower, pen, n, label, s in sorted(results, key=lambda x: (x[2], x[0], x[3]), reverse=True)[:args.top]:
        print(f"pen={pen:.4f}  {label_summary(label, s)}")

    print("\nRECOMMENDED NEXT READ")
    print("---------------------")
    conservative = [x for x in results if x[5]["n"] >= max(args.min_n, 10)]
    if conservative:
        best = max(conservative, key=lambda x: (x[2], x[0], x[3]))
        _, _, _, _, label, s = best
        print(f"Best conservative context rule: {label}")
        print(label_summary(label, s))
        if s["rate"] > base_s["rate"]:
            print("READ: Context improved the 30% bucket in-sample. Needs more rows/out-of-sample validation before HR promotion.")
        elif s["rate"] == base_s["rate"]:
            print("READ: Context did not beat the current 30% base with enough sample yet.")
        else:
            print("READ: Context did not improve the base rule yet.")
    else:
        print("No conservative context rule had enough rows. Use min-n 3 as exploratory only.")

    print("\nNEXT PATCH DIRECTION IF SIGNAL HOLDS")
    print("------------------------------------")
    print("1. Keep HR watchlist only.")
    print("2. Permanently log park/weather/pitcher/platoon fields for every HR candidate and rejection.")
    print("3. Add real FanDuel HR odds next.")
    print("4. Only promote HR after larger graded sample confirms context + positive EV.")

    if args.write_csv:
        out = Path("hr_context_factor_v2_rows.csv")
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
