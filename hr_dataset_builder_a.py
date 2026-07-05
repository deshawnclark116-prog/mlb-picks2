#!/usr/bin/env python3
"""
HR Dataset Builder A — Prop Edge MLB API

This is the correct HR-model foundation.

It does NOT grade only old HR watchlist candidates.
It builds a batter-game dataset from historical MLB games:

One row = one confirmed starting batter in one completed game.

Label:
    actual_hr = 1 if the batter hit a HR in that game
    actual_hr = 0 otherwise

Features added before the game date:
    batter_barrel_rate
    recent_barrel_rate
    batter_hard_hit_rate
    batter_avg_ev
    batter_ev90
    launch-angle HR band rate
    pull-air rate
    pitcher_barrel_allowed_rate
    pitcher_fly_ball_rate
    pitcher_HR_FB
    pitcher HR/9
    pitcher hand
    batter hand
    platoon bucket
    park bucket
    weather temp/wind
    wind toward pull field, rough parser
    FanDuel HR odds fields, currently cache/forward-log only

Why this script exists:
    The old tests had selection bias because they only tested HR candidates the old scorer already liked.
    This builds the real training/evaluation table.

Run small first:
    python hr_dataset_builder_a.py --days 7 --max-games 10 --max-batter-rows 100

Then bigger:
    python hr_dataset_builder_a.py --days 30 --max-games 60 --max-batter-rows 600

Output:
    /data/hr_model/hr_batter_game_dataset.csv
or local:
    ./hr_model/hr_batter_game_dataset.csv

Notes:
    - Baseball Savant pulls can be slow. This script caches tiny aggregate features, not raw CSV.
    - Features are calculated using data up to the day BEFORE the game to reduce leakage.
    - Historical FanDuel HR odds are not magically available unless you have odds snapshots/logs.
      This script creates odds columns and can read a local odds cache, but official HR EV needs a live odds logger next.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
MLB_PEOPLE_STATS_URL = "https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
SAVANT_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

USER_AGENT = "PropEdgeHRDatasetBuilderA/1.0"

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
    "loandepot park": "pitcher_friendly",
    "loanDepot park".lower(): "pitcher_friendly",
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
PITCHERISH_PARK_BUCKETS = {"pitcher_friendly"}


def data_root() -> Path:
    p = Path(os.environ.get("HR_MODEL_DATA_DIR", "/data/hr_model"))
    if not p.parent.exists():
        p = Path("./hr_model")
    p.mkdir(parents=True, exist_ok=True)
    return p


DATA_DIR = data_root()
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def parse_date(s: Any) -> Optional[dt.date]:
    if not s:
        return None
    m = re.search(r"(\d{4}-\d{2}-\d{2})", str(s))
    if not m:
        return None
    try:
        return dt.date.fromisoformat(m.group(1))
    except Exception:
        return None


def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def season_start_for(game_date: dt.date) -> dt.date:
    return dt.date(game_date.year, 3, 1)


def md5_key(parts: List[Any]) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def cache_path(namespace: str, parts: List[Any]) -> Path:
    d = CACHE_DIR / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{md5_key(parts)}.json"


def read_json_cache(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json_cache(path: Path, data: Any) -> None:
    try:
        path.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def get_json_cached(namespace: str, url: str, params: Dict[str, Any], ttl_seconds: Optional[int] = None) -> Optional[Dict[str, Any]]:
    path = cache_path(namespace, [url, sorted(params.items())])
    cached = read_json_cache(path)
    if cached is not None:
        if ttl_seconds is None:
            return cached
        try:
            age = time.time() - path.stat().st_mtime
            if age <= ttl_seconds:
                return cached
        except Exception:
            return cached

    try:
        r = SESSION.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        write_json_cache(path, data)
        time.sleep(0.10)
        return data
    except Exception as e:
        return None


def get_game_feed(game_id: int) -> Optional[Dict[str, Any]]:
    url = MLB_FEED_URL.format(game_id=game_id)
    return get_json_cached("game_feed", url, {})


def get_schedule_for_date(day: dt.date) -> List[Dict[str, Any]]:
    data = get_json_cached("schedule", MLB_SCHEDULE_URL, {
        "sportId": 1,
        "date": day.isoformat(),
        "hydrate": "probablePitcher,team",
    })
    games = []
    for date_blob in (data or {}).get("dates", []):
        games.extend(date_blob.get("games") or [])
    return games


def is_final_game(game: Dict[str, Any]) -> bool:
    status = ((game.get("status") or {}).get("abstractGameState") or "").lower()
    detailed = ((game.get("status") or {}).get("detailedState") or "").lower()
    return status == "final" or "final" in detailed or "completed early" in detailed


def team_name(feed: Dict[str, Any], side: str) -> Optional[str]:
    return (((feed.get("gameData") or {}).get("teams") or {}).get(side) or {}).get("name")


def venue_name(feed: Dict[str, Any]) -> Optional[str]:
    return ((feed.get("gameData") or {}).get("venue") or {}).get("name")


def official_date(feed: Dict[str, Any]) -> Optional[dt.date]:
    d = ((feed.get("gameData") or {}).get("datetime") or {}).get("officialDate")
    return parse_date(d)


def weather_fields(feed: Dict[str, Any]) -> Dict[str, Any]:
    w = ((feed.get("gameData") or {}).get("weather") or {})
    temp = safe_float(w.get("temp"))
    if temp is None:
        m = re.search(r"-?\d+(?:\.\d+)?", str(w.get("temperature") or ""))
        temp = float(m.group()) if m else None
    wind = w.get("wind") or w.get("windSpeed")
    return {
        "weather_condition": w.get("condition"),
        "weather_temp_f": temp,
        "weather_wind": wind,
        "weather_wind_mph": first_number(wind),
        "weather_wind_bucket": wind_bucket(wind),
    }


def first_number(x: Any) -> Optional[float]:
    m = re.search(r"-?\d+(?:\.\d+)?", str(x or ""))
    return float(m.group()) if m else None


def wind_bucket(wind: Any) -> str:
    s = str(wind or "").lower()
    if not s:
        return "wind_unknown"
    if "out" in s or "to cf" in s or "to lf" in s or "to rf" in s:
        return "wind_out"
    if "in" in s or "from cf" in s or "from lf" in s or "from rf" in s:
        return "wind_in"
    if "left" in s or "right" in s or "cross" in s:
        return "wind_cross"
    return "wind_other"


def wind_toward_pull(wind: Any, batter_hand: Optional[str]) -> Optional[bool]:
    if not wind or not batter_hand:
        return None
    s = str(wind).lower()
    hand = batter_hand.upper()
    # Rough parser only. Real version needs park orientation and wind vector.
    if "out" in s:
        return True
    if hand == "R" and ("to lf" in s or "toward lf" in s or "left field" in s):
        return True
    if hand == "L" and ("to rf" in s or "toward rf" in s or "right field" in s):
        return True
    if "in" in s:
        return False
    return None


def park_bucket(venue: Optional[str]) -> str:
    return PARK_BUCKET_BY_VENUE.get(str(venue or "").lower(), "park_unknown")


def game_player(feed: Dict[str, Any], player_id: Any) -> Optional[Dict[str, Any]]:
    if not player_id:
        return None
    players = ((feed.get("gameData") or {}).get("players") or {})
    return players.get(f"ID{player_id}") or players.get(str(player_id))


def box_team(feed: Dict[str, Any], side: str) -> Dict[str, Any]:
    return (((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}).get(side) or {}


def box_player(feed: Dict[str, Any], side: str, player_id: Any) -> Optional[Dict[str, Any]]:
    if not player_id:
        return None
    players = (box_team(feed, side).get("players") or {})
    return players.get(f"ID{player_id}") or players.get(str(player_id))


def player_hand(feed: Dict[str, Any], player_id: Any, kind: str) -> Optional[str]:
    p = game_player(feed, player_id) or {}
    if kind == "bat":
        obj = p.get("batSide") or {}
    else:
        obj = p.get("pitchHand") or {}
    code = obj.get("code")
    if code:
        return code
    return None


def starting_pitcher_id(feed: Dict[str, Any], side: str) -> Optional[int]:
    probable = (((feed.get("gameData") or {}).get("probablePitchers") or {}).get(side) or {})
    if probable.get("id"):
        return safe_int(probable.get("id"))

    team = box_team(feed, side)
    for pid in team.get("pitchers") or []:
        bp = box_player(feed, side, pid) or {}
        stats = ((bp.get("stats") or {}).get("pitching") or {})
        if str(stats.get("gamesStarted")) == "1":
            return safe_int(pid)

    pitchers = team.get("pitchers") or []
    return safe_int(pitchers[0]) if pitchers else None


def batting_order_ids(feed: Dict[str, Any], side: str) -> List[int]:
    team = box_team(feed, side)
    order = team.get("battingOrder") or []
    out = []
    for pid in order:
        i = safe_int(pid)
        if i and i not in out:
            out.append(i)
    return out


def batter_game_stats(feed: Dict[str, Any], side: str, player_id: int) -> Dict[str, Any]:
    p = box_player(feed, side, player_id) or {}
    person = p.get("person") or {}
    stats = ((p.get("stats") or {}).get("batting") or {})
    return {
        "batter_name": person.get("fullName") or (game_player(feed, player_id) or {}).get("fullName"),
        "at_bats": safe_int(stats.get("atBats"), 0),
        "plate_appearances": safe_int(stats.get("plateAppearances"), None),
        "hits": safe_int(stats.get("hits"), 0),
        "doubles": safe_int(stats.get("doubles"), 0),
        "triples": safe_int(stats.get("triples"), 0),
        "home_runs": safe_int(stats.get("homeRuns"), 0),
        "rbi": safe_int(stats.get("rbi"), 0),
        "walks": safe_int(stats.get("baseOnBalls"), 0),
        "strikeouts": safe_int(stats.get("strikeOuts"), 0),
    }


def platoon_bucket(batter_hand: Optional[str], pitcher_hand: Optional[str]) -> str:
    b = str(batter_hand or "").upper()
    p = str(pitcher_hand or "").upper()
    if not b or not p:
        return "platoon_unknown"
    if b == "S":
        return "switch_hitter"
    if (b == "L" and p == "R") or (b == "R" and p == "L"):
        return "platoon_adv"
    if b in {"L", "R"} and p in {"L", "R"}:
        return "same_hand"
    return "platoon_unknown"


def innings_to_float(ip: Any) -> Optional[float]:
    if ip is None:
        return None
    s = str(ip)
    if "." not in s:
        return safe_float(s)
    try:
        whole, frac = s.split(".", 1)
        outs = int((frac or "0")[0])
        if outs not in (0, 1, 2):
            return safe_float(s)
        return int(whole or "0") + outs / 3.0
    except Exception:
        return safe_float(s)


def pitcher_season_stats(player_id: int, season: int) -> Dict[str, Any]:
    if not player_id:
        return {}
    data = get_json_cached("pitcher_season_stats", MLB_PEOPLE_STATS_URL.format(player_id=player_id), {
        "stats": "season",
        "group": "pitching",
        "season": season,
    })
    out = {
        "pitcher_season_ip": None,
        "pitcher_season_hr": None,
        "pitcher_season_bf": None,
        "pitcher_season_hr9": None,
        "pitcher_season_hr_per_bf": None,
        "pitcher_season_era": None,
        "pitcher_season_whip": None,
    }
    try:
        splits = (((data or {}).get("stats") or [{}])[0].get("splits") or [])
        if not splits:
            return out
        stat = splits[0].get("stat") or {}
        ip = innings_to_float(stat.get("inningsPitched"))
        hr = safe_float(stat.get("homeRuns"))
        bf = safe_float(stat.get("battersFaced"))
        out.update({
            "pitcher_season_ip": ip,
            "pitcher_season_hr": hr,
            "pitcher_season_bf": bf,
            "pitcher_season_era": safe_float(stat.get("era")),
            "pitcher_season_whip": safe_float(stat.get("whip")),
        })
        if ip and ip > 0:
            out["pitcher_season_hr9"] = (hr or 0) * 9 / ip
        if bf and bf > 0:
            out["pitcher_season_hr_per_bf"] = (hr or 0) / bf
    except Exception:
        pass
    return out


class StatcastAgg:
    def __init__(self, role: str, batter_hand: Optional[str] = None):
        self.role = role
        self.batter_hand = (batter_hand or "").upper()
        self.rows = 0
        self.filtered_rows = 0
        self.bbe = 0
        self.barrels = 0
        self.hard_hits = 0
        self.la_band = 0
        self.fly_balls = 0
        self.home_runs = 0
        self.air = 0
        self.pull_air = 0
        self.pull_air_known = 0
        self.evs: List[float] = []
        self.las: List[float] = []

    def add_row(self, row: Dict[str, Any]) -> None:
        self.rows += 1
        ev = safe_float(row.get("launch_speed"))
        la = safe_float(row.get("launch_angle"))
        event = str(row.get("events") or "").lower()
        bb_type = str(row.get("bb_type") or "").lower()

        if event == "home_run":
            self.home_runs += 1

        if ev is None or la is None:
            return

        self.bbe += 1
        self.evs.append(ev)
        self.las.append(la)

        if str(row.get("launch_speed_angle") or "").strip() == "6":
            self.barrels += 1
        if ev >= 95:
            self.hard_hits += 1
        if 20 <= la <= 35:
            self.la_band += 1
        if bb_type == "fly_ball":
            self.fly_balls += 1
        if bb_type in {"fly_ball", "line_drive"}:
            self.air += 1
            x = safe_float(row.get("hc_x"))
            if x is not None and self.batter_hand in {"R", "L"}:
                self.pull_air_known += 1
                if self.batter_hand == "R" and x < 125:
                    self.pull_air += 1
                elif self.batter_hand == "L" and x > 125:
                    self.pull_air += 1

    def percentile(self, p: float) -> Optional[float]:
        vals = sorted(self.evs)
        if not vals:
            return None
        if len(vals) == 1:
            return vals[0]
        idx = (len(vals) - 1) * p
        lo = math.floor(idx)
        hi = math.ceil(idx)
        return vals[lo] if lo == hi else vals[lo] * (hi - idx) + vals[hi] * (idx - lo)

    def result(self) -> Dict[str, Any]:
        r = self.role
        n = self.bbe
        return {
            f"{r}_statcast_rows": self.rows,
            f"{r}_bbe": self.bbe,
            f"{r}_barrels": self.barrels,
            f"{r}_barrel_rate": self.barrels / n if n else None,
            f"{r}_hard_hit_rate": self.hard_hits / n if n else None,
            f"{r}_avg_ev": sum(self.evs) / len(self.evs) if self.evs else None,
            f"{r}_ev90": self.percentile(0.90),
            f"{r}_avg_la": sum(self.las) / len(self.las) if self.las else None,
            f"{r}_la_20_35_rate": self.la_band / n if n else None,
            f"{r}_fly_ball_rate": self.fly_balls / n if n else None,
            f"{r}_hr_per_fb": self.home_runs / self.fly_balls if self.fly_balls else None,
            f"{r}_pull_air_rate": self.pull_air / self.pull_air_known if self.pull_air_known else None,
            f"{r}_pull_air_known": self.pull_air_known,
        }


def statcast_aggregate(player_id: int, player_type: str, start_date: dt.date, end_date: dt.date, role: str, batter_hand: Optional[str], max_rows: int) -> Dict[str, Any]:
    """
    Memory-safe Statcast aggregate.
    It streams the CSV and filters rows to the exact player id.
    """
    if not player_id or end_date < start_date:
        return {}

    path = cache_path("statcast_agg", [player_id, player_type, start_date, end_date, role, batter_hand, max_rows])
    cached = read_json_cache(path)
    if cached is not None:
        return cached

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
        # Savant accepts player_id in most CSV paths. We still filter below to guard against broad returns.
        "player_id": str(player_id),
    }

    agg = StatcastAgg(role, batter_hand=batter_hand)
    too_broad = False
    try:
        with SESSION.get(SAVANT_CSV_URL, params=params, timeout=60, stream=True) as r:
            if r.status_code != 200:
                return {f"{role}_statcast_error": f"status_{r.status_code}"}

            text_iter = (line.decode("utf-8", errors="ignore") for line in r.iter_lines() if line)
            reader = csv.DictReader(text_iter)
            for i, row in enumerate(reader, start=1):
                if i > max_rows:
                    too_broad = True
                    break

                if player_type == "batter":
                    row_pid = safe_int(row.get("batter"))
                else:
                    row_pid = safe_int(row.get("pitcher"))

                if row_pid != int(player_id):
                    # If first several hundred rows do not match, query likely got too broad.
                    if i > 300 and agg.rows == 0:
                        too_broad = True
                        break
                    continue

                agg.filtered_rows += 1
                agg.add_row(row)

        out = agg.result()
        out[f"{role}_query_too_broad"] = too_broad
        out[f"{role}_filtered_rows"] = agg.filtered_rows
        write_json_cache(path, out)
        time.sleep(0.15)
        return out
    except Exception as e:
        return {f"{role}_statcast_error": f"{type(e).__name__}: {e}"}


def odds_from_cache(game_date: dt.date, player_name: str, team: str, odds_cache_dir: Optional[Path]) -> Dict[str, Any]:
    """
    Historical HR odds require snapshots. This reads simple JSON cache files if you later add them.
    Supported loose formats:
      /data/hr_odds/YYYY-MM-DD.json
      list/dict with player/name, book, odds, line
    """
    out = {
        "fanduel_hr_odds": None,
        "fanduel_hr_implied_prob": None,
        "fanduel_hr_odds_status": "missing_no_odds_cache",
    }
    if not odds_cache_dir:
        return out

    p = odds_cache_dir / f"{game_date.isoformat()}.json"
    if not p.exists():
        return out

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("items") or data.get("odds") or []
        pname = norm(player_name)
        for item in rows:
            name = item.get("player") or item.get("name") or item.get("description")
            book = str(item.get("book") or item.get("bookmaker") or item.get("sportsbook") or "").lower()
            market = str(item.get("market") or item.get("prop_type") or "").lower()
            if pname and norm(name) == pname and ("fanduel" in book or book == "fd") and ("home" in market or "hr" in market):
                odds = item.get("odds") or item.get("price") or item.get("american_odds")
                out["fanduel_hr_odds"] = odds
                out["fanduel_hr_implied_prob"] = implied_prob_american_or_decimal(odds)
                out["fanduel_hr_odds_status"] = "found_cache"
                return out
        out["fanduel_hr_odds_status"] = "missing_player_not_in_cache"
        return out
    except Exception:
        out["fanduel_hr_odds_status"] = "cache_parse_error"
        return out


def implied_prob_american_or_decimal(odds: Any) -> Optional[float]:
    try:
        val = float(str(odds).replace("+", ""))
    except Exception:
        return None
    # decimal
    if 1.01 <= val <= 100:
        return 1 / val
    # American
    if val > 0:
        return 100 / (val + 100)
    if val < 0:
        return abs(val) / (abs(val) + 100)
    return None


def build_rows_for_game(feed: Dict[str, Any], args: argparse.Namespace, odds_cache_dir: Optional[Path]) -> List[Dict[str, Any]]:
    gdate = official_date(feed)
    if not gdate:
        return []
    if args.skip_today and gdate >= dt.date.today():
        return []

    game_id = safe_int(((feed.get("gamePk") or feed.get("game_id"))))
    venue = venue_name(feed)
    pbucket = park_bucket(venue)
    wf = weather_fields(feed)

    rows: List[Dict[str, Any]] = []
    for side in ("away", "home"):
        opp_side = "home" if side == "away" else "away"
        team = team_name(feed, side)
        opp = team_name(feed, opp_side)
        spid = starting_pitcher_id(feed, opp_side)
        sp = game_player(feed, spid) or {}
        pitcher_name = sp.get("fullName")
        pitcher_hand = player_hand(feed, spid, "pitch")
        p_season = pitcher_season_stats(spid, gdate.year) if spid else {}

        batting_ids = batting_order_ids(feed, side)
        for spot, batter_id in enumerate(batting_ids, start=1):
            bstats = batter_game_stats(feed, side, batter_id)
            batter_name = bstats.get("batter_name")
            if not batter_name:
                continue
            batter_hand = player_hand(feed, batter_id, "bat")
            pl_bucket = platoon_bucket(batter_hand, pitcher_hand)
            wt_pull = wind_toward_pull(wf.get("weather_wind"), batter_hand)

            # Features must stop at day before game to avoid future leakage.
            feature_end = gdate - dt.timedelta(days=1)
            season_start = season_start_for(gdate)
            recent_start = max(season_start, feature_end - dt.timedelta(days=args.recent_days - 1))

            batter_season = {}
            batter_recent = {}
            pitcher_allowed = {}
            if not args.no_statcast:
                batter_season = statcast_aggregate(
                    batter_id, "batter", season_start, feature_end, "batter", batter_hand, args.max_savant_rows
                )
                batter_recent = statcast_aggregate(
                    batter_id, "batter", recent_start, feature_end, "recent_batter", batter_hand, args.max_savant_rows
                )
                if spid and not args.skip_pitcher_statcast:
                    pitcher_allowed = statcast_aggregate(
                        spid, "pitcher", season_start, feature_end, "pitcher_allowed", batter_hand, args.max_savant_rows
                    )

            odds = odds_from_cache(gdate, batter_name, team or "", odds_cache_dir)

            row = {
                "game_date": gdate.isoformat(),
                "game_id": game_id,
                "side": side,
                "team": team,
                "opponent": opp,
                "venue": venue,
                "park_bucket": pbucket,
                "hitterish_park": pbucket in HITTERISH_PARK_BUCKETS,
                "pitcherish_park": pbucket in PITCHERISH_PARK_BUCKETS,
                **wf,
                "batter_id": batter_id,
                "batter_name": batter_name,
                "lineup_spot": spot,
                "batter_hand": batter_hand,
                "opposing_pitcher_id": spid,
                "opposing_pitcher_name": pitcher_name,
                "opposing_pitcher_hand": pitcher_hand,
                "platoon_bucket": pl_bucket,
                "wind_toward_pull_field": wt_pull,
                "park_factor_by_batter_hand": None,
                "park_factor_by_batter_hand_status": "missing_need_park_hand_table",
                **bstats,
                **p_season,
                **batter_season,
                "recent_barrel_rate": batter_recent.get("recent_batter_barrel_rate"),
                "recent_hard_hit_rate": batter_recent.get("recent_batter_hard_hit_rate"),
                "recent_ev90": batter_recent.get("recent_batter_ev90"),
                "recent_la_20_35_rate": batter_recent.get("recent_batter_la_20_35_rate"),
                "recent_pull_air_rate": batter_recent.get("recent_batter_pull_air_rate"),
                **batter_recent,
                "pitcher_barrel_allowed_rate": pitcher_allowed.get("pitcher_allowed_barrel_rate"),
                "pitcher_fly_ball_rate": pitcher_allowed.get("pitcher_allowed_fly_ball_rate"),
                "pitcher_hrfb_rate": pitcher_allowed.get("pitcher_allowed_hr_per_fb"),
                **pitcher_allowed,
                **odds,
                "actual_hr": 1 if (bstats.get("home_runs") or 0) > 0 else 0,
                "actual_hr_count": bstats.get("home_runs") or 0,
            }
            rows.append(row)

            if args.sleep_per_batter > 0:
                time.sleep(args.sleep_per_batter)

    return rows


def append_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    existing_fields: List[str] = []
    if path.exists():
        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_fields = next(reader, [])
        except Exception:
            existing_fields = []

    new_fields = sorted({k for r in rows for k in r.keys()})
    fields = list(dict.fromkeys(existing_fields + new_fields)) if existing_fields else new_fields

    # If fields changed, rewrite file preserving old rows.
    old_rows: List[Dict[str, Any]] = []
    if path.exists() and existing_fields and fields != existing_fields:
        with path.open("r", newline="", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))

    mode = "a" if path.exists() and not old_rows and existing_fields == fields else "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
            for r in old_rows:
                w.writerow(r)
        for r in rows:
            w.writerow(r)


def dedupe_csv(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = rows[0].keys() if rows else []
    seen = set()
    out = []
    for r in rows:
        key = (r.get("game_date"), r.get("game_id"), r.get("batter_id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow(r)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    n = len(rows)
    h = sum(1 for r in rows if safe_int(r.get("actual_hr"), 0) == 1)
    print("\nBUILD SUMMARY")
    print("-------------")
    print(f"rows_built_this_run: {n}")
    print(f"actual_hr_hits_this_run: {h}")
    print(f"actual_hr_rate_this_run: {h/n:.3%}" if n else "actual_hr_rate_this_run: n/a")

    checks = [
        ("batter_barrel_rate", "batter_barrel_rate"),
        ("recent_barrel_rate", "recent_barrel_rate"),
        ("hard_hit_rate / EV", "batter_hard_hit_rate"),
        ("EV90", "batter_ev90"),
        ("launch_angle_band", "batter_la_20_35_rate"),
        ("pull_air_rate", "batter_pull_air_rate"),
        ("pitcher_barrel_allowed_rate", "pitcher_barrel_allowed_rate"),
        ("pitcher_fly_ball_rate", "pitcher_fly_ball_rate"),
        ("pitcher HR/FB", "pitcher_hrfb_rate"),
        ("park factor by batter handedness", "park_factor_by_batter_hand"),
        ("wind toward pull field", "wind_toward_pull_field"),
        ("FanDuel HR odds", "fanduel_hr_odds"),
    ]
    print("\nFEATURE AVAILABILITY THIS RUN")
    print("-----------------------------")
    for label, field in checks:
        have = sum(1 for r in rows if r.get(field) not in (None, "", "None"))
        print(f"{label:36s} {have}/{n}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="Days back from yesterday to build.")
    ap.add_argument("--start-date", default=None, help="YYYY-MM-DD. Overrides --days.")
    ap.add_argument("--end-date", default=None, help="YYYY-MM-DD. Default yesterday.")
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-games", type=int, default=10)
    ap.add_argument("--max-batter-rows", type=int, default=100)
    ap.add_argument("--recent-days", type=int, default=30)
    ap.add_argument("--max-savant-rows", type=int, default=6000)
    ap.add_argument("--no-statcast", action="store_true")
    ap.add_argument("--skip-pitcher-statcast", action="store_true")
    ap.add_argument("--skip-today", action="store_true", default=True)
    ap.add_argument("--sleep-per-batter", type=float, default=0.0)
    ap.add_argument("--odds-cache-dir", default=None)
    ap.add_argument("--dedupe", action="store_true", default=True)
    args = ap.parse_args()

    out_path = Path(args.output) if args.output else DATA_DIR / "hr_batter_game_dataset.csv"
    odds_cache_dir = Path(args.odds_cache_dir) if args.odds_cache_dir else None

    if args.end_date:
        end = dt.date.fromisoformat(args.end_date)
    else:
        end = dt.date.today() - dt.timedelta(days=1)

    if args.start_date:
        start = dt.date.fromisoformat(args.start_date)
    else:
        start = end - dt.timedelta(days=max(0, args.days - 1))

    print("HR DATASET BUILDER A")
    print("====================")
    print("One row = one confirmed starting batter in one completed game.")
    print(f"date_range: {start} to {end}")
    print(f"output: {out_path}")
    print(f"max_games: {args.max_games}")
    print(f"max_batter_rows: {args.max_batter_rows}")
    print(f"statcast_enabled: {not args.no_statcast}")
    print(f"pitcher_statcast_enabled: {not args.skip_pitcher_statcast and not args.no_statcast}")

    all_rows: List[Dict[str, Any]] = []
    games_seen = 0
    games_used = 0
    errors = Counter()

    for day in daterange(start, end):
        games = get_schedule_for_date(day)
        for game in games:
            if args.max_games and games_used >= args.max_games:
                break
            if args.max_batter_rows and len(all_rows) >= args.max_batter_rows:
                break
            games_seen += 1
            if not is_final_game(game):
                continue

            gid = safe_int(game.get("gamePk"))
            if not gid:
                continue

            print(f"\nGAME {games_used+1}: {day} game_id={gid}")
            feed = get_game_feed(gid)
            if not feed:
                errors["missing_feed"] += 1
                continue

            try:
                rows = build_rows_for_game(feed, args, odds_cache_dir)
                if args.max_batter_rows:
                    remaining = max(0, args.max_batter_rows - len(all_rows))
                    rows = rows[:remaining]
                all_rows.extend(rows)
                games_used += 1
                print(f"  rows_added={len(rows)} total_rows={len(all_rows)}")
                append_csv(out_path, rows)
                if args.dedupe:
                    dedupe_csv(out_path)
            except Exception as e:
                errors[f"game_error:{type(e).__name__}"] += 1
                print(f"  ERROR: {type(e).__name__}: {e}")

        if args.max_games and games_used >= args.max_games:
            break
        if args.max_batter_rows and len(all_rows) >= args.max_batter_rows:
            break

    print_summary(all_rows)
    print("\nRUN COUNTS")
    print("----------")
    print(f"games_seen: {games_seen}")
    print(f"games_used: {games_used}")
    for k, v in errors.items():
        print(f"{k}: {v}")
    print(f"\nDONE: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
