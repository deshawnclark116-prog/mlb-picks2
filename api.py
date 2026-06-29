"""
api.py - Prop Edge v8.15B.

CRITICAL SAFETY FIX:
- run_predictions now generates new picks only for games still in pregame/preview state.
- Prevents /run/now from creating contaminated live-game picks after games have started.
- Current-version existing picks for already-started games are preserved on later reruns.

HITTER ENGINE:
- Hits/TB/RBI/runs scan all 9 confirmed lineup batters.
- HRs scan all 9 confirmed lineup batters.
- Actual batting order is passed into batter model features.
- Hitter candidates are separated from official board picks.
- Official hitter board uses a governor:
    * one official pick per player
    * max hitter picks per team/game
    * max hitter picks per game
    * max HR picks per game
    * max HR picks per slate
    * max 2 HR picks per team, but the 2nd HR must be elite
    * HR official quality gate to reduce h2h-only inflation
- Pitcher K model logic is intentionally unchanged in this patch.

Previous locked features remain:
- HR MODEL LIVE: 3-signal HR score season SLG + h2h_SLG + recent ISO.
- Totals removed v8.13.
- Run lines removed v8.11.
- Under gate v8.10.
- Sum-score hits.
- FanDuel lines.
- ksim.
- ET time.
"""

import os, json, math, time, threading, datetime as dt
from pathlib import Path
from collections import defaultdict
from zoneinfo import ZoneInfo

import requests
import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import backfill
import gamelines
import ksim
import marginsim
import bvp
import lineupk

VERSION = "8.15B"

ET = ZoneInfo("America/New_York")
def today_et(): return dt.datetime.now(ET).date()
def now_et(): return dt.datetime.now(ET)

app = FastAPI(title="Prop Edge ML API", version=VERSION)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

MLB = "https://statsapi.mlb.com/api/v1"
MLB11 = "https://statsapi.mlb.com/api/v1.1"
PROPLINE_KEY = os.environ.get("PROPLINE_API_KEY", "")
PROPLINE_BASE = "https://api.prop-line.com/v1/sports/baseball_mlb"
SEASON = today_et().year
DATA_DIR = Path("/data")
MODEL_DIR = DATA_DIR / "models"
PRED_DIR = DATA_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
GH_PAGES_BASE = "https://deshawnclark116-prog.github.io/mlb-picks2"

S = requests.Session()
S.headers["User-Agent"] = f"prop-edge/{VERSION}"

STANDARD_LINE = {
    "batter_hits": 0.5,
    "pitcher_strikeouts": 4.5,
    "batter_total_bases": 1.5,
    "batter_home_runs": 0.5,
}

PROB_FLOOR = 0.55
MIN_EDGE = 0.05
REGRADE_DAYS = 3
BVP_ENABLED = True
PITCHER_WEIGHT = 0.55
LINEUP_WEIGHT = 0.45
UNDER_PITCHER_WEIGHT = 0.35
UNDER_LINEUP_WEIGHT = 0.65
RECENCY_DECAY = 0.6
SEASON_ANCHOR = 0.15
PREFERRED_BOOK = "fanduel"
HR_SCORE_THRESHOLD = 1.30

# v8.15B board governor settings
MAX_HITTER_PICKS_PER_TEAM = 2
MAX_HITTER_PICKS_PER_GAME = 3
MAX_HR_PICKS_PER_TEAM = 2
MAX_HR_PICKS_PER_GAME = 2
MAX_HR_PICKS_PER_SLATE = 6
SECOND_TEAM_HR_MIN_SCORE = 1.50
SECOND_TEAM_HR_MIN_TIER = "hr_elite"

# HR official quality gate. This does not stop candidate tracking.
# It only stops weak/low-season-power profiles from becoming official HR board picks.
HR_OFFICIAL_MIN_SCORE = 1.50
HR_OFFICIAL_MIN_SEASON_SLG = 0.400
HR_OFFICIAL_LOW_SLG_RECENT_ISO = 0.350

# Moneyline sanity gate. Extreme lines are usually live/settled/badly mapped.
MAX_ABS_MONEYLINE_ODDS = 500

PREGAME_STATUSES = {"preview", "pre-game", "pregame", "scheduled"}

PROP_MODEL = {
    "batter_hits": "batter_hits",
    "pitcher_strikeouts": "pitcher_strikeouts",
    "batter_total_bases": "batter_total_bases",
    "batter_rbis": "batter_rbi",
    "batter_runs": "batter_runs",
}

HITTER_PROPS = {
    "batter_hits",
    "batter_total_bases",
    "batter_rbis",
    "batter_runs",
    "batter_home_runs",
}

_models = {}

def load_models():
    _models.clear()
    for name in ("batter_hits", "pitcher_strikeouts", "batter_total_bases",
                 "batter_rbi", "batter_runs"):
        mp = MODEL_DIR / f"{name}.json"
        cp = MODEL_DIR / f"{name}_columns.json"
        if mp.exists() and cp.exists():
            booster = xgb.Booster()
            booster.load_model(str(mp))
            _models[name] = (booster, json.loads(cp.read_text()))
            print(f"Loaded model {name}")
        else:
            print(f"Model {name} not found")

load_models()


def model_predict(name, feat_dict):
    if name not in _models:
        return None
    booster, cols = _models[name]
    x = np.array([[feat_dict.get(c, 0) for c in cols]], dtype=np.float32)
    return float(booster.predict(xgb.DMatrix(x))[0])


def get(url, **params):
    for attempt in range(3):
        try:
            r = S.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  retry {url}: {e}")
            time.sleep(1.5 * (attempt + 1))
    return {}


def _norm(s):
    return "".join(c for c in str(s).lower() if c.isalpha() or c == " ").strip()


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _pick_book(bookmakers):
    if not bookmakers:
        return None, None
    for b in bookmakers:
        if b.get("key") == PREFERRED_BOOK:
            return b, b.get("key")
    return bookmakers[0], bookmakers[0].get("key")


def ip_to_outs(ip):
    try:
        whole = int(float(ip))
        frac = round((float(ip) - whole) * 10)
        return whole * 3 + frac
    except Exception:
        return 0


def poisson_cdf(k, lam):
    if lam <= 0:
        return 1.0
    s, term = 0.0, math.exp(-lam)
    for i in range(k + 1):
        if i:
            term *= lam / i
        s += term
    return min(s, 1.0)


def prob_over(expected, line):
    return 1 - poisson_cdf(int(math.floor(line)), max(expected, 1e-6))


def prob_at_least(expected, threshold):
    return 1 - poisson_cdf(threshold - 1, max(expected, 1e-6))


def american_to_prob(odds):
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def no_vig_two_way(over_odds, under_odds):
    po = american_to_prob(over_odds)
    pu = american_to_prob(under_odds)
    tot = po + pu
    if tot == 0:
        return 0.5, 0.5
    return po / tot, pu / tot


def value_edge(model_p, fair_p):
    if fair_p is None or fair_p <= 0:
        return None
    return (model_p - fair_p) / fair_p


def kelly_fraction(model_p, american_odds, cap=0.25):
    b = (american_odds / 100) if american_odds > 0 else (100 / -american_odds)
    q = 1 - model_p
    f = (b * model_p - q) / b if b else 0
    return max(0.0, min(f, cap))


def _is_real_game(ev):
    h = ev.get("home_team", "")
    return "(" not in h and "Runs" not in h


def _parse_threshold(name):
    try:
        if "+" in name:
            return int(name.split("+")[0].strip())
    except Exception:
        pass
    return None


def _is_pregame_game(game):
    status = str(game.get("status", "")).lower().strip()
    return status in PREGAME_STATUSES


def _load_json_file(path, fallback=None):
    fallback = fallback if fallback is not None else []
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return fallback


def _is_hr_elite_pick(p):
    score = _safe_float(p.get("hr_score", p.get("projected")), 0.0)
    tier = p.get("hr_tier")
    return tier == SECOND_TEAM_HR_MIN_TIER or score >= SECOND_TEAM_HR_MIN_SCORE


def _hr_official_quality_ok(p):
    """
    HR candidate can exist below this gate, but it will not become an official board pick.

    Goal:
    - Keep real power bats.
    - Reduce h2h-only inflated candidates where h2h_slg=1.000 but season power profile is weak.
    """
    score = _safe_float(p.get("hr_score", p.get("projected")), 0.0)
    season_slg = _safe_float(p.get("season_slg"), 0.0)
    recent_iso = _safe_float(p.get("recent_iso"), 0.0)

    if score < HR_OFFICIAL_MIN_SCORE:
        return False

    if season_slg >= HR_OFFICIAL_MIN_SEASON_SLG:
        return True

    if recent_iso >= HR_OFFICIAL_LOW_SLG_RECENT_ISO:
        return True

    return False


def _candidate_rank(p):
    """
    Board ranking only. This does not change the underlying model probability.
    It is used to choose the best official market when a player qualifies for
    multiple markets.
    """
    prop = p.get("prop_type")
    mp = _safe_float(p.get("model_prob"), 0.0)
    edge = max(_safe_float(p.get("value_edge"), 0.0), 0.0)
    has_line = bool(p.get("has_line"))
    spot = int(p.get("lineup_spot") or 9)
    lineup_bonus = max(0.0, (10 - spot) * 0.005)
    line_bonus = 0.04 if has_line else 0.0

    if prop == "batter_home_runs":
        hr_score = _safe_float(p.get("hr_score", p.get("projected")), 0.0)
        tier_bonus = 0.25 if p.get("hr_tier") == "hr_elite" else 0.10
        odds_bonus = 0.25 if has_line else -0.18
        quality_bonus = 0.15 if _hr_official_quality_ok(p) else -0.60
        return 2.00 + hr_score + tier_bonus + odds_bonus + quality_bonus + (edge * 0.25) + lineup_bonus

    if prop == "batter_total_bases":
        return 2.25 + mp + (edge * 0.15) + line_bonus + lineup_bonus

    if prop == "batter_hits":
        flag = p.get("bvp_flag") or ""
        bvp_bonus = (
            0.08 if flag == "sum_premium" else
            0.05 if flag == "sum_strong" else
            0.02 if flag == "sum_good" else
            0.00
        )
        return 2.05 + mp + bvp_bonus + (edge * 0.10) + line_bonus + lineup_bonus

    if prop == "batter_rbis":
        return 1.60 + mp + (edge * 0.10) + line_bonus + lineup_bonus

    if prop == "batter_runs":
        return 1.55 + mp + (edge * 0.10) + line_bonus + lineup_bonus

    return mp


def govern_hitter_board(candidates):
    """
    Takes all hitter candidates and chooses official board picks.

    v8.15B changes:
    - No pre-deduping that kills a player completely if their HR candidate is rejected.
    - Candidates are considered in rank order.
    - If a candidate is rejected for HR/team/game cap, a lower-ranked hit/TB candidate for
      that same player can still become official later.
    """
    annotated = []
    for c in candidates:
        q = dict(c)
        q["board_score"] = round(_candidate_rank(q), 3)
        q["board_status"] = "candidate"
        annotated.append(q)

    ranked = sorted(annotated, key=lambda r: r.get("board_score", 0), reverse=True)

    official = []
    rejected = []

    official_player_keys = set()
    game_hitter_count = defaultdict(int)
    team_hitter_count = defaultdict(int)
    team_hr_count = defaultdict(int)
    game_hr_count = defaultdict(int)
    slate_hr_count = 0

    for q in ranked:
        gid = str(q.get("game_id", ""))
        team = q.get("team", "")
        prop = q.get("prop_type", "")
        team_key = (gid, team)
        player_key = (gid, _norm(q.get("player", "")))

        reason = None

        if player_key in official_player_keys:
            reason = "same_player_lower_ranked_market"

        elif game_hitter_count[gid] >= MAX_HITTER_PICKS_PER_GAME:
            reason = "game_hitter_cap"

        elif team_hitter_count[team_key] >= MAX_HITTER_PICKS_PER_TEAM:
            reason = "team_hitter_cap"

        elif prop == "batter_home_runs":
            if not _hr_official_quality_ok(q):
                reason = "hr_official_quality_gate"
            elif slate_hr_count >= MAX_HR_PICKS_PER_SLATE:
                reason = "slate_hr_cap"
            elif game_hr_count[gid] >= MAX_HR_PICKS_PER_GAME:
                reason = "game_hr_cap"
            elif team_hr_count[team_key] >= MAX_HR_PICKS_PER_TEAM:
                reason = "team_hr_cap"
            elif team_hr_count[team_key] >= 1 and not _is_hr_elite_pick(q):
                reason = "second_team_hr_not_elite"

        if reason:
            q2 = dict(q)
            q2["board_status"] = "rejected"
            q2["reject_reason"] = reason
            rejected.append(q2)
            continue

        q["board_status"] = "official"
        official.append(q)
        official_player_keys.add(player_key)
        game_hitter_count[gid] += 1
        team_hitter_count[team_key] += 1

        if prop == "batter_home_runs":
            team_hr_count[team_key] += 1
            game_hr_count[gid] += 1
            slate_hr_count += 1

    debug_rows = official + rejected
    debug_rows.sort(key=lambda r: r.get("board_score", 0), reverse=True)
    return official, debug_rows


def fetch_propline_props():
    over_under = {}
    thresholds = defaultdict(dict)
    book_of = {}
    if not PROPLINE_KEY:
        print("  No PROPLINE_API_KEY — projection-only")
        return over_under, thresholds, book_of
    try:
        events = get(f"{PROPLINE_BASE}/events", apiKey=PROPLINE_KEY)
        if not isinstance(events, list):
            return over_under, thresholds, book_of
    except Exception as e:
        print(f"  PropLine events failed: {e}")
        return over_under, thresholds, book_of

    today = today_et().isoformat()
    prop_keys = ("batter_hits,pitcher_strikeouts,batter_total_bases,"
                 "batter_rbis,batter_runs,batter_home_runs")
    pulled = 0

    for ev in events:
        if not _is_real_game(ev):
            continue
        if not str(ev.get("commence_time", "")).startswith(today):
            continue
        eid = ev.get("id")
        if not eid:
            continue
        try:
            data = get(f"{PROPLINE_BASE}/events/{eid}/odds", apiKey=PROPLINE_KEY,
                       markets=prop_keys, regions="us")
        except Exception:
            continue

        book, book_key = _pick_book(data.get("bookmakers") or [])
        if not book:
            continue

        for mkt in book.get("markets", []):
            mkey = mkt.get("key")
            ou = defaultdict(dict)
            for o in mkt.get("outcomes", []):
                player = _norm(o.get("description", ""))
                name = o.get("name", "")
                price = o.get("price")
                point = o.get("point")
                low = name.lower()

                if low in ("over", "under"):
                    ou[player][low] = {"price": price, "point": point}
                else:
                    thr = _parse_threshold(name)
                    if thr is not None:
                        key = (player, mkey)
                        if thr not in thresholds[key]:
                            thresholds[key][thr] = price
                            book_of[(player, mkey, f"thr{thr}")] = book_key

            for player, sides in ou.items():
                if "over" in sides and "under" in sides:
                    pt = sides["over"].get("point")
                    if mkey == "batter_total_bases" and pt is not None and pt < 1.5:
                        continue
                    key = (player, mkey)
                    if key not in over_under:
                        over_under[key] = {
                            "line": pt,
                            "over_odds": sides["over"]["price"],
                            "under_odds": sides["under"]["price"],
                        }
                        book_of[key] = book_key

        pulled += 1
        time.sleep(0.2)

    print(f"  PropLine ({PREFERRED_BOOK} pref): props for {pulled} games "
          f"({len(over_under)} O/U, {len(thresholds)} threshold sets)")
    return over_under, thresholds, book_of


def fetch_propline_gamelines():
    out = {}
    if not PROPLINE_KEY:
        return out
    try:
        events = get(f"{PROPLINE_BASE}/events", apiKey=PROPLINE_KEY)
        if not isinstance(events, list):
            return out
    except Exception as e:
        print(f"  PropLine GL events failed: {e}")
        return out

    today = today_et().isoformat()
    mlb_games = {(_norm(g["home_team"]), _norm(g["away_team"])): g["game_id"]
                 for g in todays_games()}

    for ev in events:
        if not _is_real_game(ev):
            continue
        if not str(ev.get("commence_time", "")).startswith(today):
            continue
        eid = ev.get("id")
        if not eid:
            continue

        home_name = ev.get("home_team", "")
        away_name = ev.get("away_team", "")
        gid = mlb_games.get((_norm(home_name), _norm(away_name)))
        if not gid:
            continue

        try:
            data = get(f"{PROPLINE_BASE}/events/{eid}/odds", apiKey=PROPLINE_KEY,
                       markets="h2h", regions="us")
        except Exception:
            continue

        book, book_key = _pick_book(data.get("bookmakers") or [])
        if not book:
            continue

        entry = {"home_team": home_name, "away_team": away_name, "book": book_key}
        for mkt in book.get("markets", []):
            k = mkt.get("key")
            outs = mkt.get("outcomes", [])
            if k == "h2h" and "h2h" not in entry:
                od = {_norm(o.get("name", "")): o.get("price") for o in outs}
                ho, ao = od.get(_norm(home_name)), od.get(_norm(away_name))
                if ho is not None and ao is not None:
                    entry["h2h"] = {"home_odds": ho, "away_odds": ao}

        if "h2h" in entry:
            out[gid] = entry
        time.sleep(0.2)

    print(f"  PropLine ({PREFERRED_BOOK} pref): game lines for {len(out)} games")
    return out


def player_index():
    data = get(f"{MLB}/sports/1/players", season=SEASON)
    return {_norm(p.get("fullName", "")): p.get("id") for p in data.get("people", [])}


def get_player_team(pid):
    data = get(f"{MLB}/people/{pid}", hydrate="currentTeam")
    try:
        return data["people"][0]["currentTeam"]["name"]
    except Exception:
        return ""


def todays_games():
    d = today_et().isoformat()
    data = get(f"{MLB}/schedule", sportId=1, date=d, hydrate="probablePitcher,team")
    out = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            h = g["teams"]["home"]["team"]
            a = g["teams"]["away"]["team"]
            out.append({
                "game_id": str(g.get("gamePk")),
                "date": d,
                "home_team": h.get("name"),
                "away_team": a.get("name"),
                "home_pitcher": (g["teams"]["home"].get("probablePitcher") or {}).get("id"),
                "away_pitcher": (g["teams"]["away"].get("probablePitcher") or {}).get("id"),
                "game_time": g.get("gameDate"),
                "status": g.get("status", {}).get("abstractGameState", "").lower(),
            })
    return out


def _final_game_pks(target_date):
    data = get(f"{MLB}/schedule", sportId=1, date=target_date)
    final = set()
    for day in data.get("dates", []):
        for g in day.get("games", []):
            state = g.get("status", {}).get("abstractGameState", "")
            detailed = g.get("status", {}).get("detailedState", "")
            if state == "Final" or detailed in ("Final", "Game Over", "Completed Early"):
                final.add(str(g.get("gamePk")))
    return final


def get_confirmed_lineup(game_pk):
    data = get(f"{MLB11}/game/{game_pk}/feed/live")
    out = {"home": [], "away": []}
    try:
        teams = data["liveData"]["boxscore"]["teams"]
        for side in ("home", "away"):
            order = teams.get(side, {}).get("battingOrder", [])
            out[side] = [int(pid) for pid in order]
    except Exception as e:
        print(f"  lineup fetch failed for {game_pk}: {e}")
    return out


def get_game_final_score(gpk):
    try:
        data = get(f"{MLB11}/game/{gpk}/feed/live")
        linescore = data.get("liveData", {}).get("linescore", {})
        teams = data.get("gameData", {}).get("teams", {})
        home_runs = linescore.get("teams", {}).get("home", {}).get("runs")
        away_runs = linescore.get("teams", {}).get("away", {}).get("runs")
        home_name = teams.get("home", {}).get("name", "")
        away_name = teams.get("away", {}).get("name", "")
        if home_runs is None or away_runs is None:
            return None
        return int(home_runs), int(away_runs), home_name, away_name
    except Exception:
        return None


def batter_feature_row(pid):
    g = get(f"{MLB}/people/{pid}/stats", stats="gameLog", group="hitting", season=SEASON)
    try:
        splits = g["stats"][0]["splits"]
    except Exception:
        return None

    cum_h = cum_ab = cum_pa = cum_hr = cum_bb = cum_so = cum_tb = cum_rbi = cum_runs = 0
    rec_h = []
    rec_tb = []
    rec_rbi = []
    rec_runs = []

    for sp in splits:
        st = sp["stat"]
        h = int(st.get("hits", 0) or 0)
        tb = int(st.get("totalBases", 0) or 0)
        rbi = int(st.get("rbi", 0) or 0)
        runs = int(st.get("runs", 0) or 0)

        cum_h += h
        cum_ab += int(st.get("atBats", 0) or 0)
        cum_pa += int(st.get("plateAppearances", 0) or 0)
        cum_hr += int(st.get("homeRuns", 0) or 0)
        cum_bb += int(st.get("baseOnBalls", 0) or 0)
        cum_so += int(st.get("strikeOuts", 0) or 0)
        cum_tb += tb
        cum_rbi += rbi
        cum_runs += runs

        rec_h.append(h)
        rec_tb.append(tb)
        rec_rbi.append(rbi)
        rec_runs.append(runs)

    if cum_ab < 20 or len(rec_h) < 5:
        return None

    return {
        "season_avg": cum_h / cum_ab if cum_ab else 0,
        "recent15_avg": sum(rec_h[-15:]) / len(rec_h[-15:]),
        "recent5_avg": sum(rec_h[-5:]) / len(rec_h[-5:]),
        "hr_rate": cum_hr / cum_pa if cum_pa else 0,
        "bb_rate": cum_bb / cum_pa if cum_pa else 0,
        "so_rate": cum_so / cum_pa if cum_pa else 0,
        "batting_order": 9,
        "games_played": len(rec_h),
        "tb_per_pa": cum_tb / cum_pa if cum_pa else 0,
        "rbi_per_pa": cum_rbi / cum_pa if cum_pa else 0,
        "runs_per_pa": cum_runs / cum_pa if cum_pa else 0,
        "recent5_target": 0,
        "recent15_target": 0,
        "_rec_tb": rec_tb,
        "_rec_rbi": rec_rbi,
        "_rec_runs": rec_runs,
    }


def _batter_feat_for(prop, base):
    f = dict(base)
    if prop == "batter_total_bases":
        rec = base["_rec_tb"]
    elif prop == "batter_rbis":
        rec = base["_rec_rbi"]
    elif prop == "batter_runs":
        rec = base["_rec_runs"]
    else:
        rec = None

    if rec is not None:
        f["recent5_target"] = sum(rec[-5:]) / len(rec[-5:]) if rec else 0
        f["recent15_target"] = sum(rec[-15:]) / len(rec[-15:]) if rec else 0

    f.pop("_rec_tb", None)
    f.pop("_rec_rbi", None)
    f.pop("_rec_runs", None)
    return f


def pitcher_feature_row(pid):
    g = get(f"{MLB}/people/{pid}/stats", stats="gameLog", group="pitching", season=SEASON)
    try:
        splits = g["stats"][0]["splits"]
    except Exception:
        return None

    sos = []
    bfs = []
    per_start_krate = []
    cum_bf = cum_so = cum_outs = cum_bb = 0
    n_starts = 0

    for sp in splits:
        st = sp["stat"]
        bf = int(st.get("battersFaced", 0) or 0)
        so = int(st.get("strikeOuts", 0) or 0)
        outs = int(st.get("outs", 0) or 0) or ip_to_outs(st.get("inningsPitched", "0.0"))

        if bf >= 12:
            sos.append(so)
            bfs.append(bf)
            per_start_krate.append(so / bf if bf else 0)
            cum_bf += bf
            cum_so += so
            cum_outs += outs
            cum_bb += int(st.get("baseOnBalls", 0) or 0)
            n_starts += 1

    if n_starts < 3:
        return None

    season_kbf = cum_so / cum_bf if cum_bf else 0
    n = len(sos)
    w = [math.exp(-RECENCY_DECAY * (n - 1 - i)) for i in range(n)]
    rec_kbf = (sum(wi * s for wi, s in zip(w, sos)) /
               sum(wi * b for wi, b in zip(w, bfs))) if sum(w) else season_kbf
    k_per_bf = (1 - SEASON_ANCHOR) * rec_kbf + SEASON_ANCHOR * season_kbf

    return {
        "k_per_bf": k_per_bf,
        "season_k_per_bf": season_kbf,
        "avg_bf": sum(bfs[-5:]) / len(bfs[-5:]),
        "recent_k_avg": sum(sos[-5:]) / len(sos[-5:]),
        "bb_rate": cum_bb / cum_bf if cum_bf else 0,
        "outs_per_start": cum_outs / n_starts if n_starts else 0,
        "starts": n_starts,
        "per_start_krate": per_start_krate[-12:],
    }


def conf_from_prob(p):
    return "HIGH" if p >= 0.72 else "MEDIUM" if p >= 0.65 else "LOW"


def fetch_predictions_for(date_str):
    local = PRED_DIR / f"predictions_{date_str}.json"
    if local.exists():
        try:
            return json.loads(local.read_text())
        except Exception:
            pass
    try:
        r = S.get(f"{GH_PAGES_BASE}/predictions_{date_str}.json", timeout=20)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def append_yesterday_to_season():
    year = today_et().year
    yesterday = (today_et() - dt.timedelta(days=1)).isoformat()
    season_file = DATA_DIR / f"season_{year}.jsonl"
    progress_file = DATA_DIR / f"season_{year}_progress.txt"

    done = set()
    if progress_file.exists():
        done = set(progress_file.read_text().splitlines())

    if yesterday in done:
        print(f"  {yesterday} already recorded, skipping")
        return

    games = backfill.get_schedule(yesterday)
    if not games:
        print(f"  No final games for {yesterday}")
        return

    rows_written = 0
    with open(season_file, "a") as fout:
        for gpk in games:
            box = backfill.get_boxscore(gpk)
            if not box:
                continue
            rows = backfill.extract_player_lines(gpk, yesterday, box)
            for r in rows:
                fout.write(json.dumps(r) + "\n")
            rows_written += len(rows)
            time.sleep(0.3)

    with open(progress_file, "a") as p:
        p.write(yesterday + "\n")

    print(f"  Appended {rows_written} lines for {yesterday}")


def _pick(name, team, opp, gid, prop, pick_str, proj, mp, odds, fair_p=None,
          conf=None, bvp_flag=None, book=None, player_id=None,
          lineup_spot=None, extra=None):
    edge = value_edge(mp, fair_p) if fair_p is not None else None
    row = {
        "api_version": VERSION,
        "player": name,
        "player_id": player_id,
        "team": team,
        "opponent": opp,
        "game_id": gid,
        "prop_type": prop,
        "pick": pick_str,
        "projected": round(proj, 2) if proj is not None else None,
        "model_prob": round(mp, 3),
        "prob_pct": round(mp * 100, 1),
        "odds": odds,
        "book": book,
        "fair_prob": round(fair_p, 3) if fair_p is not None else None,
        "value_edge": round(edge, 3) if edge is not None else None,
        "kelly": round(kelly_fraction(mp, odds), 4) if odds is not None else None,
        "has_line": odds is not None,
        "is_edge": (edge is not None and edge >= MIN_EDGE),
        "confidence": conf if conf else conf_from_prob(mp),
        "bvp_flag": bvp_flag,
        "lineup_spot": lineup_spot,
        "generated_at": now_et().isoformat(),
    }
    if extra and isinstance(extra, dict):
        row.update(extra)
    return row


def build_batter_prop_picks(name, team, opp, gid, base_feat, over_under, thresholds,
                            book_of, batter_id=None, pitcher_id=None,
                            lineup_spot=None):
    picks = []
    nrm = _norm(name)
    hits_flag = power_flag = None

    hit_tier = None
    if batter_id and pitcher_id:
        try:
            sig = lineupk.batter_hit_sum_score(batter_id, pitcher_id, SEASON)
            hit_tier = sig["tier"]
        except Exception:
            hit_tier = None

    if BVP_ENABLED and batter_id and pitcher_id:
        bv = bvp.batter_vs_pitcher(batter_id, pitcher_id)
        if bv:
            hits_flag = bvp.classify_batter(bv)
            power_flag = bvp.power_flag(bv)

    def _bvp_nudge(p, prop):
        if prop == "batter_hits":
            if hit_tier == "sum_premium":
                return min(0.99, p + 0.10), "sum_premium"
            if hit_tier == "sum_strong":
                return min(0.99, p + 0.08), "sum_strong"
            if hit_tier == "sum_good":
                return min(0.99, p + 0.05), "sum_good"
            if hit_tier == "sum_lean":
                return min(0.99, p + 0.02), "sum_lean"
            if hit_tier == "sum_avoid":
                return max(0.0, p - 0.06), "sum_avoid"

        if prop == "batter_total_bases" and power_flag:
            if power_flag == "power":
                return min(0.99, p + 0.04), "power"
            if power_flag == "weak":
                return max(0.0, p - 0.03), "weak"

        return p, (hits_flag if prop == "batter_hits" else power_flag)

    for prop in ("batter_hits", "batter_total_bases", "batter_rbis", "batter_runs"):
        model_name = PROP_MODEL[prop]
        feat = _batter_feat_for(prop, base_feat) if prop != "batter_hits" else dict(base_feat)
        feat.pop("_rec_tb", None)
        feat.pop("_rec_rbi", None)
        feat.pop("_rec_runs", None)

        proj = model_predict(model_name, feat)
        if proj is None:
            continue

        made_over_under = False
        ou = over_under.get((nrm, prop))
        bk = book_of.get((nrm, prop))

        if ou and ou.get("over_odds") is not None and ou.get("under_odds") is not None:
            line = ou["line"]
            p_over = prob_over(proj, line)
            p_over, flag = _bvp_nudge(p_over, prop)
            fo, _ = no_vig_two_way(ou["over_odds"], ou["under_odds"])

            if p_over >= PROB_FLOOR:
                picks.append(_pick(name, team, opp, gid, prop, f"OVER {line}",
                                   proj, p_over, ou["over_odds"], fo,
                                   bvp_flag=flag, book=bk,
                                   player_id=batter_id,
                                   lineup_spot=lineup_spot))
                made_over_under = True

        else:
            line = STANDARD_LINE.get(prop)
            if line is not None:
                p_over = prob_over(proj, line)
                p_over, flag = _bvp_nudge(p_over, prop)
                if p_over >= PROB_FLOOR:
                    picks.append(_pick(name, team, opp, gid, prop, f"OVER {line}",
                                       proj, p_over, None, None,
                                       bvp_flag=flag, book=None,
                                       player_id=batter_id,
                                       lineup_spot=lineup_spot))
                    made_over_under = True

        thr = thresholds.get((nrm, prop))
        if thr:
            for t in (1, 2):
                if t not in thr:
                    continue
                if prop == "batter_hits" and t == 1 and made_over_under:
                    continue

                price = thr[t]
                p_yes = prob_at_least(proj, t)
                if p_yes >= PROB_FLOOR:
                    fair = american_to_prob(price)
                    bk2 = book_of.get((nrm, prop, f"thr{t}"))
                    label = {
                        "batter_total_bases": "Total Bases",
                        "batter_rbis": "RBIs",
                        "batter_runs": "Runs",
                        "batter_hits": "Hits",
                    }[prop]

                    picks.append(_pick(name, team, opp, gid, prop, f"{t}+ {label}",
                                       proj, p_yes, price, fair, book=bk2,
                                       player_id=batter_id,
                                       lineup_spot=lineup_spot))

    return picks


def build_hr_pick(name, team, opp, gid, batter_id, pitcher_id,
                  over_under, book_of, lineup_spot=None):
    """
    HR model: 3-signal score season SLG + h2h_SLG + recent ISO.
    Fires when score >= 1.30.

    v8.15B:
    - Still records all HR candidates in debug.
    - Official board gate is handled by govern_hitter_board().
    """
    if not batter_id or not pitcher_id:
        return None

    try:
        sig = lineupk.batter_hr_score(batter_id, pitcher_id, SEASON)
    except Exception as e:
        print(f"  HR score error {name}: {e}")
        return None

    if not sig["fires"]:
        return None

    nrm = _norm(name)
    ou = over_under.get((nrm, "batter_home_runs"))
    bk = book_of.get((nrm, "batter_home_runs"))

    mp = 0.444 if sig["tier"] == "hr_elite" else 0.419

    if ou and ou.get("over_odds") is not None:
        odds = ou["over_odds"]
        line = ou.get("line", 0.5)
        fair = american_to_prob(odds) if odds else None
    else:
        odds = None
        line = 0.5
        fair = None
        bk = None

    print(f"  HR CANDIDATE: {name} score={sig['score']} tier={sig['tier']} "
          f"slg={sig['season_slg']:.3f} h2h={sig['h2h_slg']:.3f} "
          f"iso={sig['recent_iso']:.3f}")

    return _pick(
        name, team, opp, gid, "batter_home_runs", f"OVER {line}",
        sig["score"], mp, odds, fair,
        conf="HIGH" if sig["tier"] == "hr_elite" else "MEDIUM",
        bvp_flag=f"hr_score_{sig['score']}_{sig['tier']}",
        book=bk,
        player_id=batter_id,
        lineup_spot=lineup_spot,
        extra={
            "hr_score": round(_safe_float(sig.get("score")), 3),
            "hr_tier": sig.get("tier"),
            "season_slg": round(_safe_float(sig.get("season_slg")), 3),
            "h2h_slg": round(_safe_float(sig.get("h2h_slg")), 3),
            "recent_iso": round(_safe_float(sig.get("recent_iso")), 3),
            "odds_status": "priced" if odds is not None else "missing",
            "hr_official_quality_ok": None,
        },
    )


def build_strikeout_pick(name, team, opp, gid, feat, ou, book=None,
                         lineup_exp_ks=None, k_nudge=1.0, bvp_flag=None):
    exp_bf = feat["avg_bf"]
    pitcher_proj = feat["k_per_bf"] * exp_bf

    if pitcher_proj <= 0 or exp_bf <= 0:
        return None

    if lineup_exp_ks is not None and lineup_exp_ks > 0:
        blended = PITCHER_WEIGHT * pitcher_proj + LINEUP_WEIGHT * lineup_exp_ks
    else:
        blended = pitcher_proj

    blended_kbf = (blended / exp_bf) * k_nudge

    if ou and ou.get("over_odds") is not None and ou.get("under_odds") is not None:
        line = ou["line"]
        over_odds = ou["over_odds"]
        under_odds = ou["under_odds"]
    else:
        line = STANDARD_LINE["pitcher_strikeouts"]
        over_odds = under_odds = None
        book = None

    start_rates = feat.get("per_start_krate")
    if start_rates and k_nudge != 1.0:
        start_rates = [r * k_nudge for r in start_rates]

    sim = ksim.simulate(blended_kbf, exp_bf, line, start_k_rates=start_rates)
    if sim["no_bet"]:
        return None

    side = sim["side"]
    mp = sim["side_prob"]

    # UNDER CONFIRMATION GATE v8.10
    if side == "UNDER" and lineup_exp_ks is not None and lineup_exp_ks > 0:
        lineup_heavy = (UNDER_PITCHER_WEIGHT * pitcher_proj +
                        UNDER_LINEUP_WEIGHT * lineup_exp_ks)
        if lineup_heavy >= line:
            print(f"  GATE: skipping {name} UNDER {line} "
                  f"(lineup-heavy proj {lineup_heavy:.1f} >= line)")
            return None

    odds = over_odds if side == "OVER" else under_odds
    fair = None
    if over_odds is not None and under_odds is not None:
        fo, fu = no_vig_two_way(over_odds, under_odds)
        fair = fo if side == "OVER" else fu

    return _pick(name, team, opp, gid, "pitcher_strikeouts", f"{side} {line}",
                 sim["mean"], mp, odds, fair, conf=sim["confidence"],
                 bvp_flag=bvp_flag, book=book)


def build_gameline_picks(games, gl_market, run_table):
    picks = []
    for game in games:
        gid = game["game_id"]
        home, away = game["home_team"], game["away_team"]
        gl = gl_market.get(gid)
        if not gl:
            continue

        bk = gl.get("book")
        if "h2h" in gl:
            probs = gamelines.moneyline_prob(home, away, run_table)
            if probs:
                hp, ap = probs
                fh, fa = gamelines.no_vig_two_way(gl["h2h"]["home_odds"],
                                                   gl["h2h"]["away_odds"])
                if hp >= ap:
                    team, mp, fp, od = home, hp, fh, gl["h2h"]["home_odds"]
                else:
                    team, mp, fp, od = away, ap, fa, gl["h2h"]["away_odds"]

                if od is None:
                    continue

                if abs(int(od)) > MAX_ABS_MONEYLINE_ODDS:
                    print(f"  ML GATE: skipping {team} odds={od} sanity gate")
                    continue

                edge = value_edge(mp, fp)
                if edge is not None and edge < MIN_EDGE:
                    print(f"  ML GATE: skipping {team} edge={edge:.3f} below floor")
                    continue

                if mp >= PROB_FLOOR:
                    picks.append(_pick(team, team, away if team == home else home, gid,
                                       "moneyline", f"{team} ML", mp, mp, od, fp, book=bk))
    return picks


def run_predictions():
    print(f"Running predictions {now_et()}")
    idx = player_index()

    try:
        append_yesterday_to_season()
    except Exception as e:
        print(f"  append error (non-fatal): {e}")

    backfill_all_history(idx, days_back=20)

    today = today_et().isoformat()
    pred_path = PRED_DIR / f"predictions_{today}.json"

    all_games = todays_games()
    pregame_games = [g for g in all_games if _is_pregame_game(g)]
    started_game_ids = {str(g["game_id"]) for g in all_games if not _is_pregame_game(g)}

    existing_preds = _load_json_file(pred_path, [])
    locked_existing = []
    if existing_preds and any(p.get("api_version") == VERSION for p in existing_preds if isinstance(p, dict)):
        locked_existing = [
            p for p in existing_preds
            if isinstance(p, dict) and str(p.get("game_id", "")) in started_game_ids
        ]

    print(f"  Games today: {len(all_games)} | pregame: {len(pregame_games)} | started/final: {len(started_game_ids)}")
    if locked_existing:
        print(f"  Preserving {len(locked_existing)} existing current-version picks for already-started games")

    if not pregame_games:
        print("  No pregame games available for new prediction generation")
        final_preds = locked_existing if locked_existing else []
        final_preds.sort(key=lambda r: r.get("model_prob") or 0, reverse=True)
        pred_path.write_text(json.dumps(final_preds))
        (PRED_DIR / f"hitter_candidates_{today}.json").write_text(json.dumps([]))
        byt = {}
        for p in final_preds:
            byt[p["prop_type"]] = byt.get(p["prop_type"], 0) + 1
        print(f"  Generated {len(final_preds)} predictions {byt}")
        return final_preds, all_games

    over_under, thresholds, book_of = fetch_propline_props()
    gl_market = fetch_propline_gamelines()
    run_table = gamelines.team_run_table()

    preds = []
    hitter_candidates = []

    preds.extend(locked_existing)
    preds.extend(build_gameline_picks(pregame_games, gl_market, run_table))

    # Pitcher side unchanged in v8.15B.
    for game in pregame_games:
        gid = game["game_id"]
        lineup = get_confirmed_lineup(gid)

        for side in ("home_pitcher", "away_pitcher"):
            pid = game.get(side)
            if not pid:
                continue

            feat = pitcher_feature_row(pid)
            if not feat:
                continue

            pdata = get(f"{MLB}/people/{pid}")
            name = pdata.get("people", [{}])[0].get("fullName", "")
            team = get_player_team(pid)
            opp = game["away_team"] if side == "home_pitcher" else game["home_team"]
            ou = over_under.get((_norm(name), "pitcher_strikeouts"))
            bk = book_of.get((_norm(name), "pitcher_strikeouts"))

            opp_side = "away" if side == "home_pitcher" else "home"
            opp_batters = lineup.get(opp_side, [])

            lineup_exp_ks = None
            if opp_batters:
                throws = lineupk.get_pitcher_throws(pid)
                ek, avg_kr, n = lineupk.lineup_k_expectation(
                    opp_batters, throws, SEASON, feat["avg_bf"], pitcher_id=pid)
                if ek and n >= 5:
                    lineup_exp_ks = ek

            k_nudge = 1.0
            bvp_flag = None
            if BVP_ENABLED and opp_batters:
                agg = bvp.lineup_vs_pitcher(opp_batters, pid)
                if agg["sample_pa"] >= 20:
                    k_nudge = agg["k_nudge"]
                    bvp_flag = f"lineup_kr_{agg['lineup_k_rate']}_n{k_nudge}"

            pick = build_strikeout_pick(name, team, opp, gid, feat, ou, book=bk,
                                        lineup_exp_ks=lineup_exp_ks,
                                        k_nudge=k_nudge, bvp_flag=bvp_flag)
            if pick:
                preds.append(pick)

    # v8.15B: scan all 9 confirmed hitters for every hitter market, pregame only.
    for game in pregame_games:
        gid = game["game_id"]
        lineup = get_confirmed_lineup(gid)

        for tside in ("home", "away"):
            team_name = game["home_team"] if tside == "home" else game["away_team"]
            opp = game["away_team"] if tside == "home" else game["home_team"]
            opp_pitcher = game.get("away_pitcher") if tside == "home" else game.get("home_pitcher")

            for spot, pid in enumerate(lineup.get(tside, []), start=1):
                pdata = get(f"{MLB}/people/{pid}")
                name = pdata.get("people", [{}])[0].get("fullName", "")

                base = batter_feature_row(pid)
                if base:
                    base["batting_order"] = spot
                    pks = build_batter_prop_picks(
                        name, team_name, opp, gid, base,
                        over_under, thresholds, book_of,
                        batter_id=pid, pitcher_id=opp_pitcher,
                        lineup_spot=spot,
                    )
                    hitter_candidates.extend(pks)

                hr_pick = build_hr_pick(
                    name, team_name, opp, gid,
                    pid, opp_pitcher,
                    over_under, book_of,
                    lineup_spot=spot,
                )
                if hr_pick:
                    hr_pick["hr_official_quality_ok"] = _hr_official_quality_ok(hr_pick)
                    hitter_candidates.append(hr_pick)

    hitter_official, hitter_debug = govern_hitter_board(hitter_candidates)
    preds.extend(hitter_official)

    preds.sort(key=lambda r: r.get("model_prob") or 0, reverse=True)

    pred_path.write_text(json.dumps(preds))
    (PRED_DIR / f"hitter_candidates_{today}.json").write_text(json.dumps(hitter_debug))

    byt = {}
    for p in preds:
        byt[p["prop_type"]] = byt.get(p["prop_type"], 0) + 1

    cand_by_type = {}
    for p in hitter_candidates:
        cand_by_type[p["prop_type"]] = cand_by_type.get(p["prop_type"], 0) + 1

    official_hitter_by_type = {}
    for p in hitter_official:
        official_hitter_by_type[p["prop_type"]] = official_hitter_by_type.get(p["prop_type"], 0) + 1

    print(f"  Hitter candidates scanned: {len(hitter_candidates)} {cand_by_type}")
    print(f"  Hitter official after governor: {len(hitter_official)} {official_hitter_by_type}")
    print(f"  Generated {len(preds)} predictions {byt}")

    return preds, all_games


PROP_STAT = {
    "batter_hits": ("hitting", "hits"),
    "pitcher_strikeouts": ("pitching", "strikeOuts"),
    "batter_total_bases": ("hitting", "totalBases"),
    "batter_rbis": ("hitting", "rbi"),
    "batter_runs": ("hitting", "runs"),
    "batter_home_runs": ("hitting", "homeRuns"),
}


def get_actual_stat(pid, group, field, target_date):
    data = get(f"{MLB}/people/{pid}/stats", stats="gameLog", group=group, season=SEASON)
    try:
        for sp in reversed(data["stats"][0]["splits"]):
            if sp.get("date") == target_date:
                return float(sp["stat"].get(field, 0) or 0)
    except Exception:
        pass
    return None


def grade_picks(target_date, idx):
    if target_date > today_et().isoformat():
        return []

    final_pks = _final_game_pks(target_date)
    if not final_pks:
        return []

    preds = fetch_predictions_for(target_date)
    if not preds:
        return []

    results = []
    score_cache = {}

    for pred in preds:
        if not isinstance(pred, dict):
            continue

        gpk = str(pred.get("game_id", ""))
        if gpk not in final_pks:
            continue

        prop = pred.get("prop_type")
        if prop == "run_line":
            continue

        pick = pred.get("pick", "")
        result = None
        actual = None

        if prop == "moneyline":
            if gpk not in score_cache:
                score_cache[gpk] = get_game_final_score(gpk)
            score = score_cache[gpk]
            if score is None:
                continue

            home_r, away_r, home_name, away_name = score
            picked_team = _norm(pred.get("team", ""))
            home_norm = _norm(home_name)
            away_norm = _norm(away_name)

            if picked_team == home_norm:
                result = "hit" if home_r > away_r else "miss"
                actual = home_r
            elif picked_team == away_norm:
                result = "hit" if away_r > home_r else "miss"
                actual = away_r
            else:
                continue

        elif prop == "total":
            if gpk not in score_cache:
                score_cache[gpk] = get_game_final_score(gpk)
            score = score_cache[gpk]
            if score is None:
                continue

            home_r, away_r, _, _ = score
            actual = home_r + away_r

            if pick.upper().startswith("OVER"):
                try:
                    line = float(pick.split()[-1])
                    result = "hit" if actual > line else "miss"
                except Exception:
                    continue
            elif pick.upper().startswith("UNDER"):
                try:
                    line = float(pick.split()[-1])
                    result = "hit" if actual < line else "miss"
                except Exception:
                    continue

        elif prop in PROP_STAT:
            group, field = PROP_STAT[prop]
            pid = pred.get("player_id") or idx.get(_norm(pred.get("player", "")))
            if not pid:
                continue

            actual = get_actual_stat(pid, group, field, target_date)
            if actual is None:
                continue

            if "+" in pick:
                try:
                    thr = int(pick.split("+")[0].strip())
                    result = "hit" if actual >= thr else "miss"
                except Exception:
                    continue
            elif pick.upper().startswith("OVER"):
                try:
                    line = float(pick.split()[-1])
                    result = "hit" if actual > line else "miss"
                except Exception:
                    continue
            elif pick.upper().startswith("UNDER"):
                try:
                    line = float(pick.split()[-1])
                    result = "hit" if actual < line else "miss"
                except Exception:
                    continue

        else:
            continue

        if result is None:
            continue

        results.append({
            "date": target_date,
            "player": pred.get("player", ""),
            "team": pred.get("team", ""),
            "prop_type": prop,
            "pick": pick,
            "projected": pred.get("projected"),
            "actual": actual,
            "result": result,
            "model_prob": pred.get("model_prob"),
            "confidence": pred.get("confidence", ""),
            "bvp_flag": pred.get("bvp_flag"),
        })

    return results


def update_record(new_results, regrade_dates=None):
    path = DATA_DIR / "record.json"
    regrade_dates = set(regrade_dates or [])

    try:
        ed = json.loads(path.read_text()) if path.exists() else {}
        existing = ed.get("results", []) if isinstance(ed, dict) else []
        existing = [r for r in existing if isinstance(r, dict)]
    except Exception:
        existing = []

    if regrade_dates:
        existing = [r for r in existing if r.get("date") not in regrade_dates]

    keys = {(r.get("date", ""), r.get("player", ""), r.get("prop_type", ""),
             r.get("pick", "")) for r in existing}

    for r in new_results:
        k = (r.get("date", ""), r.get("player", ""), r.get("prop_type", ""), r.get("pick", ""))
        if k not in keys:
            existing.append(r)
            keys.add(k)

    existing.sort(key=lambda r: r.get("date", ""), reverse=True)

    total = len(existing)
    hits = sum(1 for r in existing if r.get("result") == "hit")
    hr = round(hits / total * 100, 1) if total else 0

    by_prop = {}
    by_conf = {}
    by_bvp = {}

    for r in existing:
        pt = r.get("prop_type", "?")
        by_prop.setdefault(pt, {"hits": 0, "total": 0})
        by_prop[pt]["total"] += 1
        by_prop[pt]["hits"] += 1 if r.get("result") == "hit" else 0

        c = r.get("confidence", "?")
        by_conf.setdefault(c, {"hits": 0, "total": 0})
        by_conf[c]["total"] += 1
        by_conf[c]["hits"] += 1 if r.get("result") == "hit" else 0

        bf = r.get("bvp_flag") or "none"
        bucket = "none"
        for tag in ("hr_score", "sum_premium", "sum_strong", "sum_good",
                    "sum_lean", "sum_avoid", "hits", "struggles", "power", "weak"):
            if isinstance(bf, str) and bf.startswith(tag):
                bucket = tag
                break

        by_bvp.setdefault(bucket, {"hits": 0, "total": 0})
        by_bvp[bucket]["total"] += 1
        by_bvp[bucket]["hits"] += 1 if r.get("result") == "hit" else 0

    record = {
        "summary": {
            "total": total,
            "hits": hits,
            "misses": total - hits,
            "hit_rate": hr,
        },
        "by_prop": {
            k: {**v, "hit_rate": round(v["hits"] / v["total"] * 100, 1) if v["total"] else 0}
            for k, v in by_prop.items()
        },
        "by_confidence": {
            k: {**v, "hit_rate": round(v["hits"] / v["total"] * 100, 1) if v["total"] else 0}
            for k, v in by_conf.items()
        },
        "by_bvp": {
            k: {**v, "hit_rate": round(v["hits"] / v["total"] * 100, 1) if v["total"] else 0}
            for k, v in by_bvp.items()
        },
        "results": existing,
        "last_updated": now_et().isoformat(),
    }

    path.write_text(json.dumps(record, indent=2))
    print(f"  Record: {hits}/{total} ({hr}%)")
    return record


def backfill_all_history(idx, days_back=20):
    today = today_et()
    regrade_dates = [(today - dt.timedelta(days=i)).isoformat()
                     for i in range(1, REGRADE_DAYS + 1)]

    all_new = []
    for i in range(1, days_back + 1):
        d = (today - dt.timedelta(days=i)).isoformat()
        all_new.extend(grade_picks(d, idx))

    if all_new:
        update_record(all_new, regrade_dates=regrade_dates)


_retrain_status = {"running": False, "last_run": None, "last_result": None}


def _do_weekly_update():
    _retrain_status["running"] = True
    _retrain_status["last_run"] = now_et().isoformat()

    try:
        import train
        train.main()
        load_models()
        _retrain_status["last_result"] = "success"
    except Exception as e:
        _retrain_status["last_result"] = f"error: {e}"
        print("Weekly update error:", e)
    finally:
        _retrain_status["running"] = False


@app.get("/health")
def health():
    all_games = todays_games()
    pregame_games = [g for g in all_games if _is_pregame_game(g)]
    return {
        "status": "ok",
        "version": VERSION,
        "models_loaded": list(_models.keys()),
        "propline": bool(PROPLINE_KEY),
        "bvp": BVP_ENABLED,
        "preferred_book": PREFERRED_BOOK,
        "hr_threshold": HR_SCORE_THRESHOLD,
        "pregame_gate": {
            "enabled": True,
            "pregame_statuses": sorted(PREGAME_STATUSES),
            "games_today": len(all_games),
            "pregame_games_now": len(pregame_games),
        },
        "hitter_governor": {
            "max_hitter_picks_per_team": MAX_HITTER_PICKS_PER_TEAM,
            "max_hitter_picks_per_game": MAX_HITTER_PICKS_PER_GAME,
            "max_hr_picks_per_team": MAX_HR_PICKS_PER_TEAM,
            "max_hr_picks_per_game": MAX_HR_PICKS_PER_GAME,
            "max_hr_picks_per_slate": MAX_HR_PICKS_PER_SLATE,
            "second_team_hr_min_score": SECOND_TEAM_HR_MIN_SCORE,
            "second_team_hr_min_tier": SECOND_TEAM_HR_MIN_TIER,
            "hr_official_min_score": HR_OFFICIAL_MIN_SCORE,
            "hr_official_min_season_slg": HR_OFFICIAL_MIN_SEASON_SLG,
            "hr_official_low_slg_recent_iso": HR_OFFICIAL_LOW_SLG_RECENT_ISO,
        },
        "moneyline_gate": {
            "max_abs_moneyline_odds": MAX_ABS_MONEYLINE_ODDS,
            "requires_positive_edge": True,
        },
        "server_date_et": today_et().isoformat(),
        "server_time_et": now_et().isoformat(),
    }


@app.get("/predictions")
def predictions():
    today = today_et().isoformat()
    path = PRED_DIR / f"predictions_{today}.json"

    if path.exists():
        data = json.loads(path.read_text())
        if data:
            return data

    preds, _ = run_predictions()
    return preds


@app.get("/games")
def games():
    return todays_games()


@app.get("/debug/hitter-candidates")
def debug_hitter_candidates():
    today = today_et().isoformat()
    path = PRED_DIR / f"hitter_candidates_{today}.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


@app.post("/run/daily")
def trigger_daily():
    preds, games_list = run_predictions()
    return {"status": "completed", "predictions": len(preds), "games": len(games_list)}


@app.get("/run/now")
def run_now():
    preds, games_list = run_predictions()
    byt = {}
    for p in preds:
        byt[p["prop_type"]] = byt.get(p["prop_type"], 0) + 1
    return {"status": "completed", "total": len(preds), "by_type": byt}


@app.get("/run/weekly")
def trigger_weekly():
    if _retrain_status["running"]:
        return {"status": "already_running", "last_run": _retrain_status["last_run"]}

    threading.Thread(target=_do_weekly_update, daemon=True).start()
    return {"status": "started", "message": "Retraining in background."}


@app.get("/run/weekly/status")
def weekly_status():
    return _retrain_status


@app.get("/record")
def record():
    path = DATA_DIR / "record.json"
    if path.exists():
        return json.loads(path.read_text())

    return {
        "summary": {"total": 0, "hits": 0, "misses": 0, "hit_rate": 0},
        "by_prop": {},
        "by_confidence": {},
        "by_bvp": {},
        "results": [],
   }
