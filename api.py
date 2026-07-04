"""
api.py - Prop Edge v8.16E.

v8.16E MODEL DISCIPLINE PATCH:
- Keeps batter hits active as the strongest hitter model.
- Rejects weak batter-hit BVP flags from official board: sum_lean, struggles.
- Demotes batter_total_bases to candidate-only until it earns back official status.
- Keeps pitcher Ks FanDuel-line locked from v8.16D.
- Hardens pitcher K UNDER picks:
  - require HIGH confidence
  - require projection gap >= 0.75
- Hardens pitcher K OVER picks:
  - allow if lineup_kr confirmation exists
  - if no lineup_kr, require model_prob >= 0.70 OR projection gap >= 1.50
- Raises official HR threshold to 1.70.
- Requires real FanDuel HR price for official HR picks.
- Keeps moneyline probationary.

v8.16D K LINE LOCK + RESTORE HITTER CORE:
- Keeps batter hits/TB active.
- Does NOT require FanDuel lines for official hitter picks.
- Requires real FanDuel lines for pitcher strikeouts.
- If pitcher K line is missing, skips the K pick instead of falling back to 4.5.
- Keeps HR FanDuel pricing from v8.16C.
- Keeps FanDuel-only line engine.

v8.16C FANDUEL LINE TRUTH FIX:
- FanDuel-only line engine.
- No fallback to other sportsbooks.
- Fixes HR one-way FanDuel parser for "to hit a home run" style markets.
- Adds /debug/fanduel-market-probe to test FanDuel markets one-by-one.
- Keeps v8.16A record intelligence endpoints.

v8.16B LINE MATCHING FIX:
- Fixes UTC/ET date mismatch from commence_time.
- Strengthens player-name normalization.
- Requests main + alternate PropLine/Odds-style markets.
- Maps batter_runs_scored back to internal batter_runs.
- Reads player names from multiple possible outcome fields.
- Adds /debug/propline-fetch.
- Adds /debug/line-audit.

v8.16A RECORD INTELLIGENCE PATCH:
- Adds /debug/record-splits.
- Adds /record/active.
- Separates mature core, probationary, experimental, and retired markets.

SAFETY FIX FROM v8.15C:
- If no pregame games remain, run_predictions returns existing board and does not overwrite it.

CRITICAL SAFETY FIX FROM v8.15B:
- run_predictions generates new picks only for pregame/preview/scheduled games.
"""

import os, json, math, time, threading, datetime as dt, re, unicodedata, zipfile
from pathlib import Path
from collections import defaultdict
from zoneinfo import ZoneInfo

import requests
import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import backfill
import gamelines
import ksim
import marginsim
import bvp
import lineupk

VERSION = "8.17D"

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
LOG_DIR = DATA_DIR / "candidate_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
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
USE_ONLY_PREFERRED_BOOK = True

# v8.16D/v8.16E line truth:
# Hits remain a model-core stat prediction market.
# Pitcher Ks must have real FanDuel lines.
REQUIRE_FANDUEL_LINE_FOR_OFFICIAL_HITTERS = False
REQUIRE_FANDUEL_LINE_FOR_PITCHER_KS = True

# v8.16E model discipline gates.
CANDIDATE_ONLY_HITTER_PROPS = {"batter_total_bases"}
BAD_OFFICIAL_HIT_FLAGS = {"sum_lean", "struggles"}

K_UNDER_MIN_CONFIDENCE = "HIGH"
K_UNDER_MIN_PROJECTION_GAP = 0.75

K_OVER_NO_LINEUP_MIN_PROB = 0.70
K_OVER_NO_LINEUP_MIN_PROJECTION_GAP = 1.50

HR_SCORE_THRESHOLD = 1.30
HR_OFFICIAL_MIN_SCORE = 1.70
HR_OFFICIAL_REQUIRE_FANDUEL_PRICE = True

MAX_HITTER_PICKS_PER_TEAM = 2
MAX_HITTER_PICKS_PER_GAME = 3
MAX_HR_PICKS_PER_TEAM = 2
MAX_HR_PICKS_PER_GAME = 2
MAX_HR_PICKS_PER_SLATE = 6
SECOND_TEAM_HR_MIN_SCORE = 1.70
SECOND_TEAM_HR_MIN_TIER = "hr_elite"

HR_OFFICIAL_MIN_SEASON_SLG = 0.400
HR_OFFICIAL_LOW_SLG_RECENT_ISO = 0.350

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

PROPLINE_MARKETS = [
    "batter_hits",
    "batter_hits_alternate",
    "pitcher_strikeouts",
    "pitcher_strikeouts_alternate",
    "batter_total_bases",
    "batter_total_bases_alternate",
    "batter_rbis",
    "batter_rbis_alternate",
    "batter_runs",
    "batter_runs_alternate",
    "batter_runs_scored",
    "batter_runs_scored_alternate",
    "batter_home_runs",
    "batter_home_runs_alternate",
]

PROPLINE_CANON_MARKET = {
    "batter_hits": "batter_hits",
    "batter_hits_alternate": "batter_hits",
    "pitcher_strikeouts": "pitcher_strikeouts",
    "pitcher_strikeouts_alternate": "pitcher_strikeouts",
    "batter_total_bases": "batter_total_bases",
    "batter_total_bases_alternate": "batter_total_bases",
    "batter_rbis": "batter_rbis",
    "batter_rbis_alternate": "batter_rbis",
    "batter_runs": "batter_runs",
    "batter_runs_alternate": "batter_runs",
    "batter_runs_scored": "batter_runs",
    "batter_runs_scored_alternate": "batter_runs",
    "batter_home_runs": "batter_home_runs",
    "batter_home_runs_alternate": "batter_home_runs",
}

LAST_LINE_AUDIT = {
    "last_updated": None,
    "status": "not_run_yet",
}

# v8.16E lifecycle:
# Total bases is demoted from active mature to probationary/candidate-watch.
ACTIVE_MATURE_MARKETS = {"pitcher_strikeouts", "batter_hits"}
PROBATIONARY_MARKETS = {"moneyline", "batter_total_bases"}
EXPERIMENTAL_MARKETS = {"batter_home_runs"}
RETIRED_MARKETS = {"total", "run_line"}

MARKET_LIFECYCLE = {}
for _m in ACTIVE_MATURE_MARKETS:
    MARKET_LIFECYCLE[_m] = "active_mature"
for _m in PROBATIONARY_MARKETS:
    MARKET_LIFECYCLE[_m] = "active_probationary"
for _m in EXPERIMENTAL_MARKETS:
    MARKET_LIFECYCLE[_m] = "experimental"
for _m in RETIRED_MARKETS:
    MARKET_LIFECYCLE[_m] = "retired"

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
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = s.replace(".", " ")
    s = s.replace("-", " ")
    s = s.replace("'", " ")
    s = s.replace("’", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    s = "".join(c for c in s if c.isalpha() or c == " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_commence_time_et(raw):
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        d = dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(ET)
    except Exception:
        return None


def _event_is_today_et(ev):
    d = _parse_commence_time_et(ev.get("commence_time"))
    if d is None:
        return False
    return d.date() == today_et()


def _canonical_market_key(raw_key):
    raw_key = str(raw_key or "").strip()
    return PROPLINE_CANON_MARKET.get(raw_key)


def _market_line_source(raw_key):
    raw_key = str(raw_key or "")
    return "alternate" if raw_key.endswith("_alternate") else "main"


def _player_name_from_outcome(outcome):
    for key in ("description", "player", "participant", "player_name", "name"):
        val = outcome.get(key)
        if not val:
            continue

        val = str(val).strip()
        low = val.lower().strip()

        if low in ("over", "under", "yes", "no"):
            continue
        if re.match(r"^\d+\+", val):
            continue

        n = _norm(val)
        if n:
            return n

    return ""


def _line_priority(rec, canon_market):
    source = rec.get("line_source")
    if source == "main":
        source_priority = 0
    elif source == "one_way_hr":
        source_priority = 0
    elif source == "one_way_hr_over":
        source_priority = 1
    elif source == "alternate":
        source_priority = 2
    else:
        source_priority = 3

    std = STANDARD_LINE.get(canon_market)
    line = rec.get("line")

    try:
        dist = abs(float(line) - float(std)) if std is not None and line is not None else 999.0
    except Exception:
        dist = 999.0

    return (source_priority, dist)


def _set_best_over_under(over_under, book_of, key, candidate, canon_market):
    existing = over_under.get(key)

    if existing is None:
        over_under[key] = candidate
        book_of[key] = candidate.get("book")
        return

    if _line_priority(candidate, canon_market) < _line_priority(existing, canon_market):
        over_under[key] = candidate
        book_of[key] = candidate.get("book")


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

    return None, None


def _bvp_gate_flag(flag):
    if not flag:
        return "none"
    flag = str(flag)
    for tag in ("sum_premium", "sum_strong", "sum_good", "sum_lean", "sum_avoid",
                "hits", "struggles", "power", "weak", "hr_score", "lineup_kr"):
        if flag.startswith(tag):
            return tag
    return flag


def _has_lineup_kr(flag):
    return isinstance(flag, str) and flag.startswith("lineup_kr")


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
        odds_bonus = 0.25 if has_line else -0.50
        quality_bonus = 0.20 if _hr_official_quality_ok(p) else -0.85
        return 2.00 + hr_score + tier_bonus + odds_bonus + quality_bonus + (edge * 0.25) + lineup_bonus

    if prop == "batter_total_bases":
        # Still rank it for candidate/debug, but keep lower than official hit core.
        return 1.30 + mp + (edge * 0.10) + line_bonus + lineup_bonus

    if prop == "batter_hits":
        flag = _bvp_gate_flag(p.get("bvp_flag"))
        bvp_bonus = (
            0.10 if flag == "sum_good" else
            0.08 if flag == "hits" else
            0.06 if flag == "sum_premium" else
            0.04 if flag == "sum_strong" else
            -0.35 if flag in BAD_OFFICIAL_HIT_FLAGS else
            0.00
        )
        spot_bonus = 0.06 if spot <= 3 else 0.03 if spot <= 6 else 0.0
        return 2.15 + mp + bvp_bonus + spot_bonus + (edge * 0.10) + line_bonus + lineup_bonus

    if prop == "batter_rbis":
        return 1.40 + mp + (edge * 0.10) + line_bonus + lineup_bonus

    if prop == "batter_runs":
        return 1.40 + mp + (edge * 0.10) + line_bonus + lineup_bonus

    return mp



def govern_hitter_board(candidates):
    """
    v8.17C Prediction-First Board.

    Core rule:
    - Odds do not block prediction visibility.
    - Hard game/team caps do not block qualified hitter winners.
    - Same-player conflicts do not hide alternate model signals.
    - Lineup spot is informational only, not a hard gate.

    Official/watched structure:
    - batter_hits >= 63% and not bad BVP flag => official_prediction
    - batter_total_bases >= 63% => watchlist_prediction
    - batter_home_runs with hr_score >= 1.70 or HR quality OK => watchlist_prediction
    - lower-quality rows remain rejected so the board does not become noise.
    """

    HIT_MIN_PROB = 0.63
    TB_WATCH_MIN_PROB = 0.63
    HR_WATCH_MIN_SCORE = 1.70

    def _player_key(q):
        return (str(q.get("game_id", "")), _norm(q.get("player", "")))

    def _market_priority_score(q):
        prop = q.get("prop_type")
        mp = _safe_float(q.get("model_prob"), 0.0)
        board_score = _safe_float(q.get("board_score"), 0.0)
        flag = _bvp_gate_flag(q.get("bvp_flag"))

        # Do not use lineup spot as a hard truth. It stays informational.
        if prop == "batter_hits":
            if flag in BAD_OFFICIAL_HIT_FLAGS:
                return -100.0 + board_score

            bonus = 0.0
            if flag == "sum_good":
                bonus += 0.25
            elif flag == "hits":
                bonus += 0.22
            elif flag == "sum_premium":
                bonus += 0.22
            elif flag == "sum_strong":
                bonus += 0.18

            return 100.0 + mp + bonus + (board_score * 0.05)

        if prop == "batter_home_runs":
            hr_score = _safe_float(q.get("hr_score", q.get("projected")), 0.0)
            quality_bonus = 0.25 if _hr_official_quality_ok(q) else 0.0
            return 70.0 + hr_score + quality_bonus + (mp * 0.10) + (board_score * 0.05)

        if prop == "batter_total_bases":
            return 50.0 + mp + (board_score * 0.05)

        if prop in ("batter_rbis", "batter_runs"):
            return 20.0 + mp + (board_score * 0.05)

        return mp + (board_score * 0.05)

    annotated = []
    for c in candidates:
        q = dict(c)
        q["board_score"] = round(_candidate_rank(q), 3)
        q["board_status"] = "candidate"
        q["market_priority_score"] = round(_market_priority_score(q), 3)
        q["market_priority_version"] = "8.17C"
        annotated.append(q)

    # Pre-count potential clusters so we can warn without hiding winners.
    raw_game_counts = defaultdict(int)
    raw_team_counts = defaultdict(int)
    raw_player_counts = defaultdict(int)
    for q in annotated:
        gid = str(q.get("game_id", ""))
        team = q.get("team", "")
        raw_game_counts[gid] += 1
        raw_team_counts[(gid, team)] += 1
        raw_player_counts[_player_key(q)] += 1

    official = []
    rejected = []

    ranked = sorted(
        annotated,
        key=lambda r: (
            _safe_float(r.get("market_priority_score"), 0.0),
            _safe_float(r.get("board_score"), 0.0),
            _safe_float(r.get("model_prob"), 0.0),
        ),
        reverse=True,
    )

    for q in ranked:
        gid = str(q.get("game_id", ""))
        team = q.get("team", "")
        prop = q.get("prop_type", "")
        team_key = (gid, team)
        player_key = _player_key(q)
        mp = _safe_float(q.get("model_prob"), 0.0)
        flag = _bvp_gate_flag(q.get("bvp_flag"))
        hr_score = _safe_float(q.get("hr_score", q.get("projected")), 0.0)

        warnings = []
        if raw_game_counts[gid] > MAX_HITTER_PICKS_PER_GAME:
            warnings.append("game_cluster_warning")
        if raw_team_counts[team_key] > MAX_HITTER_PICKS_PER_TEAM:
            warnings.append("team_cluster_warning")
        if raw_player_counts[player_key] > 1:
            warnings.append("same_player_multiple_markets")
        if not q.get("has_line") and q.get("odds") is None:
            warnings.append("unpriced_prediction")

        q["board_warnings"] = warnings
        q["cluster_warning"] = bool(warnings)

        reason = None
        status = None

        if prop == "batter_hits":
            if flag in BAD_OFFICIAL_HIT_FLAGS:
                reason = f"bad_hit_flag_{flag}"
            elif mp < HIT_MIN_PROB:
                reason = "below_hit_probability_threshold"
            else:
                status = "official_prediction"
                q["prediction_tier"] = "core_hit_prediction"

        elif prop == "batter_total_bases":
            # TB is visible for learning/grading, but still not treated as core mature.
            if mp >= TB_WATCH_MIN_PROB:
                status = "watchlist_prediction"
                q["prediction_tier"] = "tb_watchlist_prediction"
            else:
                reason = "below_tb_watchlist_threshold"

        elif prop == "batter_home_runs":
            # Odds no longer hide HR model signals. Price can be handled later.
            if hr_score >= HR_WATCH_MIN_SCORE or _hr_official_quality_ok(q):
                status = "watchlist_prediction"
                q["prediction_tier"] = "hr_watchlist_prediction"
                q["odds_required_for_visibility"] = False
            else:
                reason = "hr_quality_watchlist_gate"

        elif prop in ("batter_rbis", "batter_runs"):
            # Future markets. Keep candidates logged but do not surface unless explicitly enabled later.
            reason = "future_market_not_enabled"

        else:
            reason = "unsupported_hitter_market"

        if reason:
            q2 = dict(q)
            q2["board_status"] = "rejected"
            q2["reject_reason"] = reason
            rejected.append(q2)
            continue

        q["board_status"] = status
        q["reject_reason"] = None
        official.append(q)

    debug_rows = official + rejected
    debug_rows.sort(
        key=lambda r: (
            1 if r.get("board_status") in ("official_prediction", "watchlist_prediction") else 0,
            _safe_float(r.get("market_priority_score"), 0.0),
            _safe_float(r.get("board_score"), 0.0),
            _safe_float(r.get("model_prob"), 0.0),
        ),
        reverse=True,
    )

    return official, debug_rows


def fetch_propline_props():
    global LAST_LINE_AUDIT

    over_under = {}
    thresholds = defaultdict(dict)
    book_of = {}

    audit = {
        "last_updated": now_et().isoformat(),
        "status": "started",
        "preferred_book": PREFERRED_BOOK,
        "fan_duel_only": True,
        "markets_requested": PROPLINE_MARKETS,
        "events_seen": 0,
        "events_today_et": 0,
        "events_skipped_not_today_et": 0,
        "events_skipped_non_real_game": 0,
        "events_with_fanduel": 0,
        "events_without_fanduel": 0,
        "bookmakers_seen": {},
        "markets_seen_raw": {},
        "markets_seen_canonical": {},
        "outcomes_seen": 0,
        "player_name_missing": 0,
        "over_under_pairs_found": 0,
        "hr_one_way_prices_found": 0,
        "hr_one_way_over_prices_found": 0,
        "threshold_prices_found": 0,
        "sample_lines": [],
        "sample_hr_outcomes": [],
        "sample_missing_player_outcomes": [],
    }

    if not PROPLINE_KEY:
        audit["status"] = "missing_PROPLINE_API_KEY"
        LAST_LINE_AUDIT = audit
        print("  No PROPLINE_API_KEY — projection-only")
        return over_under, thresholds, book_of

    try:
        events = get(f"{PROPLINE_BASE}/events", apiKey=PROPLINE_KEY)
        if not isinstance(events, list):
            audit["status"] = "events_response_not_list"
            LAST_LINE_AUDIT = audit
            return over_under, thresholds, book_of
    except Exception as e:
        audit["status"] = f"events_failed: {e}"
        LAST_LINE_AUDIT = audit
        print(f"  PropLine events failed: {e}")
        return over_under, thresholds, book_of

    prop_keys = ",".join(PROPLINE_MARKETS)

    for ev in events:
        audit["events_seen"] += 1

        if not _is_real_game(ev):
            audit["events_skipped_non_real_game"] += 1
            continue

        if not _event_is_today_et(ev):
            audit["events_skipped_not_today_et"] += 1
            continue

        audit["events_today_et"] += 1

        eid = ev.get("id")
        if not eid:
            continue

        try:
            data = get(
                f"{PROPLINE_BASE}/events/{eid}/odds",
                apiKey=PROPLINE_KEY,
                markets=prop_keys,
                regions="us",
            )
        except Exception as e:
            print(f"  PropLine odds failed for event {eid}: {e}")
            continue

        bookmakers = data.get("bookmakers") or []
        for b in bookmakers:
            bk = b.get("key") or "unknown"
            audit["bookmakers_seen"][bk] = audit["bookmakers_seen"].get(bk, 0) + 1

        book, book_key = _pick_book(bookmakers)
        if not book:
            audit["events_without_fanduel"] += 1
            continue

        audit["events_with_fanduel"] += 1

        ou_pairs = defaultdict(lambda: {
            "over": None,
            "under": None,
            "line": None,
            "line_source": None,
            "raw_market": None,
            "book": book_key,
        })

        for mkt in book.get("markets", []):
            raw_mkey = mkt.get("key")
            canon = _canonical_market_key(raw_mkey)

            audit["markets_seen_raw"][str(raw_mkey)] = audit["markets_seen_raw"].get(str(raw_mkey), 0) + 1

            if not canon:
                continue

            audit["markets_seen_canonical"][canon] = audit["markets_seen_canonical"].get(canon, 0) + 1
            line_source = _market_line_source(raw_mkey)

            for o in mkt.get("outcomes", []):
                audit["outcomes_seen"] += 1

                name = str(o.get("name", "")).strip()
                low = name.lower()
                price = o.get("price")
                point = o.get("point")

                if canon == "batter_home_runs" and len(audit["sample_hr_outcomes"]) < 20:
                    audit["sample_hr_outcomes"].append({
                        "raw_market": raw_mkey,
                        "name": o.get("name"),
                        "description": o.get("description"),
                        "player": o.get("player"),
                        "participant": o.get("participant"),
                        "point": point,
                        "price": price,
                    })

                player = _player_name_from_outcome(o)
                if not player:
                    audit["player_name_missing"] += 1
                    if len(audit["sample_missing_player_outcomes"]) < 12:
                        audit["sample_missing_player_outcomes"].append({
                            "raw_market": raw_mkey,
                            "name": o.get("name"),
                            "description": o.get("description"),
                            "point": point,
                            "price": price,
                        })
                    continue

                if low in ("over", "under"):
                    if point is None:
                        point = STANDARD_LINE.get(canon)

                    try:
                        point_val = float(point) if point is not None else None
                    except Exception:
                        point_val = None

                    pair_key = (player, canon, point_val)
                    rec = ou_pairs[pair_key]
                    rec["line"] = point_val
                    rec["line_source"] = line_source
                    rec["raw_market"] = raw_mkey
                    rec["book"] = book_key
                    rec[low] = {"price": price, "point": point_val}
                    continue

                if canon == "batter_home_runs" and price is not None:
                    key = (player, canon)
                    candidate = {
                        "line": 0.5,
                        "over_odds": price,
                        "under_odds": None,
                        "book": book_key,
                        "line_source": "one_way_hr",
                        "raw_market": raw_mkey,
                    }
                    _set_best_over_under(over_under, book_of, key, candidate, canon)
                    audit["hr_one_way_prices_found"] += 1

                    if len(audit["sample_lines"]) < 30:
                        audit["sample_lines"].append({
                            "player": player,
                            "market": canon,
                            "line": 0.5,
                            "over_odds": price,
                            "under_odds": None,
                            "book": book_key,
                            "line_source": "one_way_hr",
                            "raw_market": raw_mkey,
                        })
                    continue

                thr = _parse_threshold(name)
                if thr is not None:
                    key = (player, canon)
                    if thr not in thresholds[key]:
                        thresholds[key][thr] = price
                        book_of[(player, canon, f"thr{thr}")] = book_key
                        audit["threshold_prices_found"] += 1

        for (player, canon, point_val), rec in ou_pairs.items():
            if rec.get("over") and rec.get("under"):
                key = (player, canon)

                candidate = {
                    "line": rec.get("line"),
                    "over_odds": rec["over"].get("price"),
                    "under_odds": rec["under"].get("price"),
                    "book": rec.get("book"),
                    "line_source": rec.get("line_source"),
                    "raw_market": rec.get("raw_market"),
                }

                _set_best_over_under(over_under, book_of, key, candidate, canon)
                audit["over_under_pairs_found"] += 1

                if len(audit["sample_lines"]) < 30:
                    audit["sample_lines"].append({
                        "player": player,
                        "market": canon,
                        "line": candidate["line"],
                        "over_odds": candidate["over_odds"],
                        "under_odds": candidate["under_odds"],
                        "book": candidate["book"],
                        "line_source": candidate["line_source"],
                        "raw_market": candidate["raw_market"],
                    })

            elif canon == "batter_home_runs" and rec.get("over"):
                key = (player, canon)
                line = rec.get("line")
                if line is None:
                    line = 0.5

                candidate = {
                    "line": line,
                    "over_odds": rec["over"].get("price"),
                    "under_odds": None,
                    "book": rec.get("book"),
                    "line_source": "one_way_hr_over",
                    "raw_market": rec.get("raw_market"),
                }

                _set_best_over_under(over_under, book_of, key, candidate, canon)
                audit["hr_one_way_over_prices_found"] += 1

                if len(audit["sample_lines"]) < 30:
                    audit["sample_lines"].append({
                        "player": player,
                        "market": canon,
                        "line": line,
                        "over_odds": candidate["over_odds"],
                        "under_odds": None,
                        "book": candidate["book"],
                        "line_source": "one_way_hr_over",
                        "raw_market": candidate["raw_market"],
                    })

        time.sleep(0.2)

    audit["status"] = "completed"
    audit["final_line_keys"] = len(over_under)
    audit["final_threshold_keys"] = len(thresholds)
    LAST_LINE_AUDIT = audit

    print(f"  PropLine FanDuel-only: props "
          f"events_today_et={audit['events_today_et']} "
          f"lines={len(over_under)} threshold_sets={len(thresholds)} "
          f"hr_one_way={audit['hr_one_way_prices_found']} "
          f"player_missing={audit['player_name_missing']}")

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

    mlb_games = {
        (_norm(g["home_team"]), _norm(g["away_team"])): g["game_id"]
        for g in todays_games()
    }

    for ev in events:
        if not _is_real_game(ev):
            continue

        if not _event_is_today_et(ev):
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
            data = get(
                f"{PROPLINE_BASE}/events/{eid}/odds",
                apiKey=PROPLINE_KEY,
                markets="h2h",
                regions="us",
            )
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
                    entry["h2h"] = {
                        "home_odds": ho,
                        "away_odds": ao,
                    }

        if "h2h" in entry:
            out[gid] = entry

        time.sleep(0.2)

    print(f"  PropLine FanDuel-only: game lines for {len(out)} games")
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

        if ou and ou.get("over_odds") is not None:
            line = ou["line"]
            p_over = prob_over(proj, line)
            p_over, flag = _bvp_nudge(p_over, prop)

            fair = None
            if ou.get("under_odds") is not None:
                fo, _ = no_vig_two_way(ou["over_odds"], ou["under_odds"])
                fair = fo
            else:
                fair = american_to_prob(ou["over_odds"])

            if p_over >= PROB_FLOOR:
                picks.append(_pick(name, team, opp, gid, prop, f"OVER {line}",
                                   proj, p_over, ou["over_odds"], fair,
                                   bvp_flag=flag, book=bk,
                                   player_id=batter_id,
                                   lineup_spot=lineup_spot,
                                   extra={
                                       "line_source": ou.get("line_source"),
                                       "raw_market": ou.get("raw_market"),
                                   }))
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
                                       lineup_spot=lineup_spot,
                                       extra={
                                           "line_source": "standard_fallback",
                                           "raw_market": None,
                                       }))
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
                                       lineup_spot=lineup_spot,
                                       extra={
                                           "line_source": "threshold",
                                           "raw_market": None,
                                       }))

    return picks


def build_hr_pick(name, team, opp, gid, batter_id, pitcher_id,
                  over_under, book_of, lineup_spot=None):
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
        line_source = ou.get("line_source")
        raw_market = ou.get("raw_market")
    else:
        odds = None
        line = 0.5
        fair = None
        bk = None
        line_source = "standard_fallback"
        raw_market = None

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
            "line_source": line_source,
            "raw_market": raw_market,
        },
    )


def build_strikeout_pick(name, team, opp, gid, feat, ou, book=None,
                         lineup_exp_ks=None, k_nudge=1.0, bvp_flag=None):
    exp_bf = feat["avg_bf"]
    pitcher_proj = feat["k_per_bf"] * exp_bf

    if pitcher_proj <= 0 or exp_bf <= 0:
        return None

    if REQUIRE_FANDUEL_LINE_FOR_PITCHER_KS:
        if not ou or ou.get("over_odds") is None or ou.get("under_odds") is None or ou.get("line") is None:
            print(f"  K LINE GATE: skipping {name} K prop — no real FanDuel K line")
            return None

    if lineup_exp_ks is not None and lineup_exp_ks > 0:
        blended = PITCHER_WEIGHT * pitcher_proj + LINEUP_WEIGHT * lineup_exp_ks
    else:
        blended = pitcher_proj

    blended_kbf = (blended / exp_bf) * k_nudge

    if ou and ou.get("over_odds") is not None and ou.get("under_odds") is not None:
        line = ou["line"]
        over_odds = ou["over_odds"]
        under_odds = ou.get("under_odds")
        line_source = ou.get("line_source")
        raw_market = ou.get("raw_market")
    else:
        line = STANDARD_LINE["pitcher_strikeouts"]
        over_odds = under_odds = None
        book = None
        line_source = "standard_fallback"
        raw_market = None

    start_rates = feat.get("per_start_krate")
    if start_rates and k_nudge != 1.0:
        start_rates = [r * k_nudge for r in start_rates]

    sim = ksim.simulate(blended_kbf, exp_bf, line, start_k_rates=start_rates)
    if sim["no_bet"]:
        return None

    side = sim["side"]
    mp = sim["side_prob"]

    if side == "OVER":
        projection_gap = sim["mean"] - line
    else:
        projection_gap = line - sim["mean"]

    has_lineup_kr = _has_lineup_kr(bvp_flag)

    if side == "UNDER" and lineup_exp_ks is not None and lineup_exp_ks > 0:
        lineup_heavy = (UNDER_PITCHER_WEIGHT * pitcher_proj +
                        UNDER_LINEUP_WEIGHT * lineup_exp_ks)
        if lineup_heavy >= line:
            print(f"  GATE: skipping {name} UNDER {line} "
                  f"(lineup-heavy proj {lineup_heavy:.1f} >= line)")
            return None

    # v8.16E: K under discipline.
    if side == "UNDER":
        if sim.get("confidence") != K_UNDER_MIN_CONFIDENCE:
            print(f"  K UNDER GATE: skipping {name} UNDER {line} "
                  f"confidence={sim.get('confidence')} < {K_UNDER_MIN_CONFIDENCE}")
            return None

        if projection_gap < K_UNDER_MIN_PROJECTION_GAP:
            print(f"  K UNDER GATE: skipping {name} UNDER {line} "
                  f"gap={projection_gap:.2f} < {K_UNDER_MIN_PROJECTION_GAP}")
            return None

    # v8.16E: K over discipline.
    if side == "OVER" and not has_lineup_kr:
        if mp < K_OVER_NO_LINEUP_MIN_PROB and projection_gap < K_OVER_NO_LINEUP_MIN_PROJECTION_GAP:
            print(f"  K OVER GATE: skipping {name} OVER {line} "
                  f"no lineup_kr, prob={mp:.3f}, gap={projection_gap:.2f}")
            return None

    odds = over_odds if side == "OVER" else under_odds
    fair = None
    if over_odds is not None and under_odds is not None:
        fo, fu = no_vig_two_way(over_odds, under_odds)
        fair = fo if side == "OVER" else fu

    if REQUIRE_FANDUEL_LINE_FOR_PITCHER_KS and odds is None:
        print(f"  K LINE GATE: skipping {name} {side} {line} — missing side odds")
        return None

    return _pick(name, team, opp, gid, "pitcher_strikeouts", f"{side} {line}",
                 sim["mean"], mp, odds, fair, conf=sim["confidence"],
                 bvp_flag=bvp_flag, book=book,
                 extra={
                     "line_source": line_source,
                     "raw_market": raw_market,
                     "k_line_locked": True,
                     "k_gate_version": "8.16E",
                     "k_projection_gap": round(projection_gap, 3),
                     "k_has_lineup_kr": has_lineup_kr,
                 })



def build_strikeout_pick_with_debug(name, team, opp, gid, feat, ou, book=None,
                                    lineup_exp_ks=None, k_nudge=1.0, bvp_flag=None,
                                    pitcher_id=None):
    """
    v8.17D K Candidate Logger.

    Prediction-first rule:
    - K line is required because the line changes the target.
    - K odds do NOT block visibility.
    - Every K attempt gets a debug row with the exact gate/reject reason.
    """

    def _base_row(status="candidate", reject_reason=None):
        return {
            "api_version": VERSION,
            "player": name,
            "player_id": pitcher_id,
            "team": team,
            "opponent": opp,
            "game_id": gid,
            "prop_type": "pitcher_strikeouts",
            "pick": None,
            "projected": None,
            "model_prob": None,
            "prob_pct": None,
            "odds": None,
            "book": book,
            "fair_prob": None,
            "value_edge": None,
            "kelly": None,
            "has_line": False,
            "has_k_line": False,
            "is_edge": False,
            "confidence": None,
            "bvp_flag": bvp_flag,
            "lineup_spot": None,
            "generated_at": now_et().isoformat(),
            "line_source": None,
            "raw_market": None,
            "board_score": None,
            "board_status": status,
            "candidate_status": status,
            "candidate_source": "pitcher_k_candidates",
            "market_priority_version": "8.17D",
            "k_candidate_logger_version": "8.17D",
            "k_line_required": True,
            "k_odds_block_prediction_visibility": False,
            "k_line_locked": True,
            "k_gate_version": "8.17D",
            "k_reject_reason": reject_reason,
            "reject_reason": reject_reason,
            "k_has_lineup_kr": _has_lineup_kr(bvp_flag),
            "k_lineup_exp_ks": round(_safe_float(lineup_exp_ks, 0.0), 3) if lineup_exp_ks is not None else None,
            "k_nudge": round(_safe_float(k_nudge, 1.0), 3),
            "avg_bf": round(_safe_float(feat.get("avg_bf")), 3) if isinstance(feat, dict) else None,
            "recent_k_avg": round(_safe_float(feat.get("recent_k_avg")), 3) if isinstance(feat, dict) else None,
            "outs_per_start": round(_safe_float(feat.get("outs_per_start")), 3) if isinstance(feat, dict) else None,
            "starts": feat.get("starts") if isinstance(feat, dict) else None,
        }

    if not feat:
        return None, _base_row("rejected", "missing_pitcher_features")

    exp_bf = _safe_float(feat.get("avg_bf"), 0.0)
    k_per_bf = _safe_float(feat.get("k_per_bf"), 0.0)
    pitcher_proj = k_per_bf * exp_bf

    dbg = _base_row("candidate", None)
    dbg["k_pitcher_projection"] = round(pitcher_proj, 3)

    if pitcher_proj <= 0 or exp_bf <= 0:
        dbg["board_status"] = "rejected"
        dbg["candidate_status"] = "rejected"
        dbg["reject_reason"] = "invalid_pitcher_projection"
        dbg["k_reject_reason"] = "invalid_pitcher_projection"
        return None, dbg

    if REQUIRE_FANDUEL_LINE_FOR_PITCHER_KS:
        if not ou or ou.get("line") is None:
            dbg["board_status"] = "rejected"
            dbg["candidate_status"] = "rejected"
            dbg["reject_reason"] = "missing_fanduel_k_line"
            dbg["k_reject_reason"] = "missing_fanduel_k_line"
            print(f"  K LINE GATE: skipping {name} K prop — no real FanDuel K line")
            return None, dbg

    if lineup_exp_ks is not None and lineup_exp_ks > 0:
        blended = PITCHER_WEIGHT * pitcher_proj + LINEUP_WEIGHT * lineup_exp_ks
    else:
        blended = pitcher_proj

    blended_kbf = (blended / exp_bf) * k_nudge

    if ou and ou.get("line") is not None:
        line = ou["line"]
        over_odds = ou.get("over_odds")
        under_odds = ou.get("under_odds")
        line_source = ou.get("line_source")
        raw_market = ou.get("raw_market")
        has_k_line = True
    else:
        line = STANDARD_LINE["pitcher_strikeouts"]
        over_odds = under_odds = None
        book = None
        line_source = "standard_fallback"
        raw_market = None
        has_k_line = False

    dbg.update({
        "pick_line": line,
        "k_line": line,
        "line_source": line_source,
        "raw_market": raw_market,
        "has_line": has_k_line,
        "has_k_line": has_k_line,
        "over_odds": over_odds,
        "under_odds": under_odds,
        "priced": over_odds is not None and under_odds is not None,
    })

    start_rates = feat.get("per_start_krate")
    if start_rates and k_nudge != 1.0:
        start_rates = [r * k_nudge for r in start_rates]

    sim = ksim.simulate(blended_kbf, exp_bf, line, start_k_rates=start_rates)
    dbg["ksim"] = sim
    dbg["projected"] = round(_safe_float(sim.get("mean"), 0.0), 2)

    if sim.get("no_bet"):
        dbg["board_status"] = "rejected"
        dbg["candidate_status"] = "rejected"
        dbg["reject_reason"] = "ksim_no_bet"
        dbg["k_reject_reason"] = "ksim_no_bet"
        return None, dbg

    side = sim["side"]
    mp = sim["side_prob"]

    if side == "OVER":
        projection_gap = sim["mean"] - line
    else:
        projection_gap = line - sim["mean"]

    has_lineup_kr = _has_lineup_kr(bvp_flag)

    dbg.update({
        "pick": f"{side} {line}",
        "pick_side": side,
        "model_prob": round(mp, 3),
        "prob_pct": round(mp * 100, 1),
        "confidence": sim.get("confidence"),
        "k_projection_gap": round(projection_gap, 3),
        "k_has_lineup_kr": has_lineup_kr,
    })

    if side == "UNDER" and lineup_exp_ks is not None and lineup_exp_ks > 0:
        lineup_heavy = (UNDER_PITCHER_WEIGHT * pitcher_proj +
                        UNDER_LINEUP_WEIGHT * lineup_exp_ks)
        dbg["k_lineup_heavy_projection"] = round(lineup_heavy, 3)
        if lineup_heavy >= line:
            reason = "under_lineup_heavy_conflict"
            dbg["board_status"] = "rejected"
            dbg["candidate_status"] = "rejected"
            dbg["reject_reason"] = reason
            dbg["k_reject_reason"] = reason
            print(f"  GATE: skipping {name} UNDER {line} (lineup-heavy proj {lineup_heavy:.1f} >= line)")
            return None, dbg

    if side == "UNDER":
        if sim.get("confidence") != K_UNDER_MIN_CONFIDENCE:
            reason = "under_confidence_gate"
            dbg["board_status"] = "rejected"
            dbg["candidate_status"] = "rejected"
            dbg["reject_reason"] = reason
            dbg["k_reject_reason"] = reason
            print(f"  K UNDER GATE: skipping {name} UNDER {line} confidence={sim.get('confidence')} < {K_UNDER_MIN_CONFIDENCE}")
            return None, dbg

        if projection_gap < K_UNDER_MIN_PROJECTION_GAP:
            reason = "under_projection_gap_gate"
            dbg["board_status"] = "rejected"
            dbg["candidate_status"] = "rejected"
            dbg["reject_reason"] = reason
            dbg["k_reject_reason"] = reason
            print(f"  K UNDER GATE: skipping {name} UNDER {line} gap={projection_gap:.2f} < {K_UNDER_MIN_PROJECTION_GAP}")
            return None, dbg

    if side == "OVER" and not has_lineup_kr:
        if mp < K_OVER_NO_LINEUP_MIN_PROB and projection_gap < K_OVER_NO_LINEUP_MIN_PROJECTION_GAP:
            reason = "over_no_lineup_gate"
            dbg["board_status"] = "rejected"
            dbg["candidate_status"] = "rejected"
            dbg["reject_reason"] = reason
            dbg["k_reject_reason"] = reason
            print(f"  K OVER GATE: skipping {name} OVER {line} no lineup_kr, prob={mp:.3f}, gap={projection_gap:.2f}")
            return None, dbg

    odds = over_odds if side == "OVER" else under_odds
    fair = None
    if over_odds is not None and under_odds is not None:
        fo, fu = no_vig_two_way(over_odds, under_odds)
        fair = fo if side == "OVER" else fu

    pick = _pick(
        name, team, opp, gid, "pitcher_strikeouts", f"{side} {line}",
        sim["mean"], mp, odds, fair, conf=sim.get("confidence"),
        bvp_flag=bvp_flag, book=book, player_id=pitcher_id,
        extra={
            "line_source": line_source,
            "raw_market": raw_market,
            "has_line": has_k_line,
            "has_k_line": has_k_line,
            "priced": odds is not None,
            "k_line_locked": True,
            "k_gate_version": "8.17D",
            "k_projection_gap": round(projection_gap, 3),
            "k_has_lineup_kr": has_lineup_kr,
            "k_line_required": True,
            "k_odds_block_prediction_visibility": False,
            "avg_bf": round(exp_bf, 3),
            "recent_k_avg": round(_safe_float(feat.get("recent_k_avg")), 3),
            "outs_per_start": round(_safe_float(feat.get("outs_per_start")), 3),
            "starts": feat.get("starts"),
        },
    )

    pick["board_status"] = "official_prediction"
    pick["candidate_status"] = "official_prediction"
    pick["candidate_source"] = "pitcher_k_candidates"
    pick["prediction_tier"] = "core_k_prediction"
    pick["reject_reason"] = None

    dbg.update(pick)
    dbg["candidate_source"] = "pitcher_k_candidates"
    dbg["candidate_status"] = "official_prediction"
    dbg["board_status"] = "official_prediction"
    dbg["reject_reason"] = None
    dbg["k_reject_reason"] = None

    return pick, dbg

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
                                       "moneyline", f"{team} ML", mp, mp, od, fp, book=bk,
                                       extra={
                                           "line_source": "h2h",
                                           "raw_market": "h2h",
                                       }))
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
    existing_preds = [p for p in existing_preds if isinstance(p, dict)]

    locked_existing = []
    if existing_preds:
        locked_existing = [
            p for p in existing_preds
            if str(p.get("game_id", "")) in started_game_ids
        ]

    print(f"  Games today: {len(all_games)} | pregame: {len(pregame_games)} | started/final: {len(started_game_ids)}")
    if locked_existing:
        print(f"  Preserving {len(locked_existing)} existing picks for already-started games")

    if not pregame_games:
        print("  No pregame games available for new prediction generation")

        if existing_preds:
            existing_preds.sort(key=lambda r: r.get("model_prob") or 0, reverse=True)

            byt = {}
            for p in existing_preds:
                byt[p["prop_type"]] = byt.get(p["prop_type"], 0) + 1

            print(f"  Returning existing board with {len(existing_preds)} picks; not overwriting {pred_path}")
            print(f"  Existing board by type {byt}")
            return existing_preds, all_games

        print("  No existing board found; returning empty without overwriting")
        return [], all_games

    over_under, thresholds, book_of = fetch_propline_props()
    gl_market = fetch_propline_gamelines()
    run_table = gamelines.team_run_table()

    preds = []
    hitter_candidates = []
    k_candidates = []

    preds.extend(locked_existing)
    preds.extend(build_gameline_picks(pregame_games, gl_market, run_table))

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

            pick, k_debug = build_strikeout_pick_with_debug(
                name, team, opp, gid, feat, ou, book=bk,
                lineup_exp_ks=lineup_exp_ks,
                k_nudge=k_nudge, bvp_flag=bvp_flag,
                pitcher_id=pid,
            )
            if k_debug:
                k_candidates.append(k_debug)
            if pick:
                preds.append(pick)

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

    prediction_debug = hitter_debug + k_candidates

    preds.sort(key=lambda r: r.get("model_prob") or 0, reverse=True)

    pred_path.write_text(json.dumps(preds))
    (PRED_DIR / f"hitter_candidates_{today}.json").write_text(json.dumps(prediction_debug))
    (PRED_DIR / f"pitcher_k_candidates_{today}.json").write_text(json.dumps(k_candidates))
    snapshot_candidate_log(today, preds, prediction_debug)

    byt = {}
    for p in preds:
        byt[p["prop_type"]] = byt.get(p["prop_type"], 0) + 1

    cand_by_type = {}
    for p in hitter_candidates:
        cand_by_type[p["prop_type"]] = cand_by_type.get(p["prop_type"], 0) + 1

    k_by_status = {}
    k_reject_reasons = {}
    for p in k_candidates:
        st = p.get("board_status") or p.get("candidate_status") or "unknown"
        k_by_status[st] = k_by_status.get(st, 0) + 1
        rr = p.get("reject_reason") or "none"
        k_reject_reasons[rr] = k_reject_reasons.get(rr, 0) + 1

    official_hitter_by_type = {}
    rejected_by_reason = {}
    for p in hitter_debug:
        if p.get("board_status") in ("official", "official_prediction", "watchlist_prediction"):
            key = f"{p.get('prop_type')}|{p.get('board_status')}"
            official_hitter_by_type[key] = official_hitter_by_type.get(key, 0) + 1
        else:
            rr = p.get("reject_reason", "unknown")
            rejected_by_reason[rr] = rejected_by_reason.get(rr, 0) + 1

    line_by_type = {}
    for p in preds:
        prop = p.get("prop_type", "unknown")
        line_by_type.setdefault(prop, {"total": 0, "with_line": 0, "fallback": 0})
        line_by_type[prop]["total"] += 1
        if p.get("has_line") or p.get("odds") is not None:
            line_by_type[prop]["with_line"] += 1
        if p.get("line_source") == "standard_fallback":
            line_by_type[prop]["fallback"] += 1

    print(f"  Hitter candidates scanned: {len(hitter_candidates)} {cand_by_type}")
    print(f"  K candidates scanned: {len(k_candidates)} status={k_by_status} reasons={k_reject_reasons}")
    print(f"  Hitter official after governor: {len(hitter_official)} {official_hitter_by_type}")
    print(f"  Hitter rejected reasons: {rejected_by_reason}")
    print(f"  Line coverage by official type: {line_by_type}")
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
            "api_version": pred.get("api_version"),
            "player": pred.get("player", ""),
            "player_id": pred.get("player_id"),
            "team": pred.get("team", ""),
            "opponent": pred.get("opponent"),
            "game_id": pred.get("game_id"),
            "prop_type": prop,
            "pick": pick,
            "projected": pred.get("projected"),
            "actual": actual,
            "result": result,
            "model_prob": pred.get("model_prob"),
            "confidence": pred.get("confidence", ""),
            "bvp_flag": pred.get("bvp_flag"),
            "odds": pred.get("odds"),
            "book": pred.get("book"),
            "fair_prob": pred.get("fair_prob"),
            "value_edge": pred.get("value_edge"),
            "has_line": pred.get("has_line"),
            "line_source": pred.get("line_source"),
            "raw_market": pred.get("raw_market"),
            "k_line_locked": pred.get("k_line_locked"),
            "k_gate_version": pred.get("k_gate_version"),
            "k_projection_gap": pred.get("k_projection_gap"),
            "k_has_lineup_kr": pred.get("k_has_lineup_kr"),
            "lineup_spot": pred.get("lineup_spot"),
            "board_score": pred.get("board_score"),
            "hr_score": pred.get("hr_score"),
            "hr_tier": pred.get("hr_tier"),
            "season_slg": pred.get("season_slg"),
            "h2h_slg": pred.get("h2h_slg"),
            "recent_iso": pred.get("recent_iso"),
            "odds_status": pred.get("odds_status"),
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
        "record_intelligence_version": VERSION,
        "results": existing,
        "last_updated": now_et().isoformat(),
    }

    path.write_text(json.dumps(record, indent=2))
    print(f"  Record: {hits}/{total} ({hr}%)")
    return record


def _empty_stat():
    return {"hits": 0, "misses": 0, "total": 0}


def _add_stat(bucket, key, result):
    if key is None or key == "":
        key = "unknown"
    key = str(key)
    bucket.setdefault(key, _empty_stat())
    bucket[key]["total"] += 1
    if result == "hit":
        bucket[key]["hits"] += 1
    elif result == "miss":
        bucket[key]["misses"] += 1


def _finish_stats(bucket):
    out = {}
    for k in sorted(bucket.keys()):
        v = bucket[k]
        total = v.get("total", 0)
        hits = v.get("hits", 0)
        misses = v.get("misses", max(0, total - hits))
        out[k] = {
            "hits": hits,
            "misses": misses,
            "total": total,
            "hit_rate": round(hits / total * 100, 1) if total else 0,
        }
    return out


def _summarize_rows(rows):
    total = len(rows)
    hits = sum(1 for r in rows if r.get("result") == "hit")
    return {
        "hits": hits,
        "misses": total - hits,
        "total": total,
        "hit_rate": round(hits / total * 100, 1) if total else 0,
    }


def _pick_side(row):
    prop = row.get("prop_type")
    pick = str(row.get("pick", "")).upper().strip()

    if prop == "moneyline":
        return "ML"
    if pick.startswith("OVER"):
        return "OVER"
    if pick.startswith("UNDER"):
        return "UNDER"
    if "+" in pick:
        return "THRESHOLD"
    return "UNKNOWN"


def _pick_line(row):
    pick = str(row.get("pick", "")).upper().strip()
    m = re.search(r"(?:OVER|UNDER)\s+(-?\d+(?:\.\d+)?)", pick)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None

    m2 = re.search(r"^(\d+)\+", pick)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            return None

    return None


def _line_bucket(row):
    line = _pick_line(row)
    if line is None:
        return "no_line"
    if abs(line - round(line)) < 1e-9:
        return f"line_{int(line)}"
    return f"line_{line:.1f}"


def _prob_bucket(row):
    p = _safe_float(row.get("model_prob"), None)
    if p is None:
        return "no_prob"
    if p < 0.55:
        return "lt_55"
    if p < 0.60:
        return "55_59"
    if p < 0.65:
        return "60_64"
    if p < 0.70:
        return "65_69"
    if p < 0.75:
        return "70_74"
    if p < 0.80:
        return "75_79"
    return "80_plus"


def _edge_bucket(row):
    e = _safe_float(row.get("value_edge"), None)
    if e is None:
        return "no_edge_recorded"
    if e < 0:
        return "negative_edge"
    if e < 0.05:
        return "0_4_edge"
    if e < 0.10:
        return "5_9_edge"
    if e < 0.20:
        return "10_19_edge"
    return "20_plus_edge"


def _odds_bucket(row):
    odds = row.get("odds")
    if odds is None:
        return "no_odds_recorded"
    try:
        odds = int(odds)
    except Exception:
        return "bad_odds"

    if odds < -300:
        return "favorite_lt_minus300"
    if odds < -200:
        return "favorite_minus300_to_minus201"
    if odds < -150:
        return "favorite_minus200_to_minus151"
    if odds < -110:
        return "favorite_minus150_to_minus111"
    if odds <= 110:
        return "near_pickem"
    if odds <= 150:
        return "plus111_to_plus150"
    if odds <= 250:
        return "plus151_to_plus250"
    return "plus251_plus"


def _projection_gap_bucket(row):
    side = _pick_side(row)
    line = _pick_line(row)
    proj = _safe_float(row.get("projected"), None)

    if side not in ("OVER", "UNDER") or line is None or proj is None:
        return "no_gap"

    if side == "OVER":
        gap = proj - line
    else:
        gap = line - proj

    if gap < 0:
        return "negative_gap"
    if gap < 0.25:
        return "gap_0_0.24"
    if gap < 0.50:
        return "gap_0.25_0.49"
    if gap < 0.75:
        return "gap_0.50_0.74"
    if gap < 1.00:
        return "gap_0.75_0.99"
    if gap < 1.50:
        return "gap_1.00_1.49"
    if gap < 2.00:
        return "gap_1.50_1.99"
    return "gap_2_plus"


def _lineup_flag_bucket(row):
    flag = row.get("bvp_flag")
    if isinstance(flag, str) and flag.startswith("lineup_kr"):
        return "with_lineup_kr"
    return "no_lineup_kr"


def _bvp_bucket(row):
    bf = row.get("bvp_flag")
    if not bf:
        return "none"
    bf = str(bf)
    for tag in ("sum_premium", "sum_strong", "sum_good", "sum_lean", "sum_avoid",
                "hits", "struggles", "power", "weak", "hr_score", "lineup_kr"):
        if bf.startswith(tag):
            return tag
    return bf


def _lineup_spot_bucket(row):
    spot = row.get("lineup_spot")
    try:
        spot = int(spot)
    except Exception:
        return "no_lineup_spot_recorded"

    if spot <= 0:
        return "bad_lineup_spot"
    if spot <= 3:
        return "spot_1_3"
    if spot <= 6:
        return "spot_4_6"
    if spot <= 9:
        return "spot_7_9"
    return "spot_10_plus"


def _hr_tier_bucket(row):
    tier = row.get("hr_tier")
    if tier:
        return str(tier)

    flag = row.get("bvp_flag")
    if isinstance(flag, str):
        if "hr_elite" in flag:
            return "hr_elite"
        if "hr_strong" in flag:
            return "hr_strong"
        if "hr_lean" in flag:
            return "hr_lean"

    return "no_hr_tier_recorded"


def _hr_score_bucket(row):
    score = _safe_float(row.get("hr_score"), None)

    if score is None:
        flag = row.get("bvp_flag")
        if isinstance(flag, str):
            m = re.search(r"hr_score_([0-9]+(?:\.[0-9]+)?)", flag)
            if m:
                score = _safe_float(m.group(1), None)

    if score is None:
        return "no_hr_score_recorded"

    if score < 1.30:
        return "lt_1.30"
    if score < 1.50:
        return "1.30_1.49"
    if score < 1.70:
        return "1.50_1.69"
    return "1.70_plus"


def _odds_status_bucket(row):
    status = row.get("odds_status")
    if status:
        return str(status)
    if row.get("odds") is not None:
        return "priced"
    return "missing_or_not_recorded"


def _api_version_bucket(row):
    v = row.get("api_version")
    return str(v) if v else "no_api_version_recorded"


def _market_class(prop):
    return MARKET_LIFECYCLE.get(prop, "unclassified")


def _rows_by_market(results, market):
    return [r for r in results if r.get("prop_type") == market]


def _generic_split(rows, func):
    bucket = {}
    for r in rows:
        _add_stat(bucket, func(r), r.get("result"))
    return _finish_stats(bucket)


def _generic_split_two(rows, func1, func2):
    bucket = {}
    for r in rows:
        key = f"{func1(r)}|{func2(r)}"
        _add_stat(bucket, key, r.get("result"))
    return _finish_stats(bucket)


def _market_lifecycle_summary(results):
    bucket = {}
    market_bucket = {}

    for r in results:
        prop = r.get("prop_type", "unknown")
        klass = _market_class(prop)
        _add_stat(bucket, klass, r.get("result"))
        _add_stat(market_bucket, prop, r.get("result"))

    return {
        "by_class": _finish_stats(bucket),
        "by_market": _finish_stats(market_bucket),
        "definitions": {
            "active_mature": sorted(ACTIVE_MATURE_MARKETS),
            "active_probationary": sorted(PROBATIONARY_MARKETS),
            "experimental": sorted(EXPERIMENTAL_MARKETS),
            "retired": sorted(RETIRED_MARKETS),
        },
    }


def _record_active_view(results):
    mature = [r for r in results if r.get("prop_type") in ACTIVE_MATURE_MARKETS]
    probationary = [r for r in results if r.get("prop_type") in PROBATIONARY_MARKETS]
    experimental = [r for r in results if r.get("prop_type") in EXPERIMENTAL_MARKETS]
    retired = [r for r in results if r.get("prop_type") in RETIRED_MARKETS]

    def _market_table(rows, markets):
        d = {}
        for m in markets:
            mr = [r for r in rows if r.get("prop_type") == m]
            d[m] = _summarize_rows(mr)
        return d

    return {
        "version": VERSION,
        "generated_at": now_et().isoformat(),
        "active_mature": {
            "markets": sorted(ACTIVE_MATURE_MARKETS),
            "summary": _summarize_rows(mature),
            "by_market": _market_table(mature, ACTIVE_MATURE_MARKETS),
        },
        "active_probationary": {
            "markets": sorted(PROBATIONARY_MARKETS),
            "summary": _summarize_rows(probationary),
            "by_market": _market_table(probationary, PROBATIONARY_MARKETS),
        },
        "experimental": {
            "markets": sorted(EXPERIMENTAL_MARKETS),
            "summary": _summarize_rows(experimental),
        },
        "retired": {
            "markets": sorted(RETIRED_MARKETS),
            "summary": _summarize_rows(retired),
        },
        "all_recorded": _summarize_rows(results),
        "note": "v8.16E active mature is tightened to batter_hits + pitcher_strikeouts. Total bases is probationary/candidate-only.",
    }


def _build_record_splits(results):
    pitcher = _rows_by_market(results, "pitcher_strikeouts")
    pitcher_over = [r for r in pitcher if _pick_side(r) == "OVER"]
    pitcher_under = [r for r in pitcher if _pick_side(r) == "UNDER"]

    batter_hits_rows = _rows_by_market(results, "batter_hits")
    tb_rows = _rows_by_market(results, "batter_total_bases")
    moneyline_rows = _rows_by_market(results, "moneyline")
    hr_rows = _rows_by_market(results, "batter_home_runs")

    return {
        "version": VERSION,
        "generated_at": now_et().isoformat(),
        "source_total": len(results),
        "active_view": _record_active_view(results),
        "market_lifecycle": _market_lifecycle_summary(results),

        "pitcher_strikeouts": {
            "summary": _summarize_rows(pitcher),
            "overs": {
                "summary": _summarize_rows(pitcher_over),
                "by_confidence": _generic_split(pitcher_over, lambda r: r.get("confidence") or "unknown"),
                "by_lineup_flag": _generic_split(pitcher_over, _lineup_flag_bucket),
                "by_line": _generic_split(pitcher_over, _line_bucket),
                "by_projection_gap": _generic_split(pitcher_over, _projection_gap_bucket),
                "by_prob_bucket": _generic_split(pitcher_over, _prob_bucket),
            },
            "unders": {
                "summary": _summarize_rows(pitcher_under),
                "by_confidence": _generic_split(pitcher_under, lambda r: r.get("confidence") or "unknown"),
                "by_lineup_flag": _generic_split(pitcher_under, _lineup_flag_bucket),
                "by_line": _generic_split(pitcher_under, _line_bucket),
                "by_projection_gap": _generic_split(pitcher_under, _projection_gap_bucket),
                "by_prob_bucket": _generic_split(pitcher_under, _prob_bucket),
            },
            "by_side": _generic_split(pitcher, _pick_side),
            "by_side_confidence": _generic_split_two(pitcher, _pick_side, lambda r: r.get("confidence") or "unknown"),
            "by_side_lineup_flag": _generic_split_two(pitcher, _pick_side, _lineup_flag_bucket),
            "by_line": _generic_split(pitcher, _line_bucket),
            "by_projection_gap": _generic_split(pitcher, _projection_gap_bucket),
            "by_api_version": _generic_split(pitcher, _api_version_bucket),
        },

        "batter_hits": {
            "summary": _summarize_rows(batter_hits_rows),
            "by_bvp_flag": _generic_split(batter_hits_rows, _bvp_bucket),
            "by_confidence": _generic_split(batter_hits_rows, lambda r: r.get("confidence") or "unknown"),
            "by_prob_bucket": _generic_split(batter_hits_rows, _prob_bucket),
            "by_lineup_spot": _generic_split(batter_hits_rows, _lineup_spot_bucket),
            "by_api_version": _generic_split(batter_hits_rows, _api_version_bucket),
        },

        "batter_total_bases": {
            "summary": _summarize_rows(tb_rows),
            "by_bvp_flag": _generic_split(tb_rows, _bvp_bucket),
            "by_confidence": _generic_split(tb_rows, lambda r: r.get("confidence") or "unknown"),
            "by_prob_bucket": _generic_split(tb_rows, _prob_bucket),
            "by_lineup_spot": _generic_split(tb_rows, _lineup_spot_bucket),
            "by_api_version": _generic_split(tb_rows, _api_version_bucket),
            "note": "v8.16E demotes total bases to candidate-only/probationary due weak recent record.",
        },

        "moneyline": {
            "summary": _summarize_rows(moneyline_rows),
            "by_confidence": _generic_split(moneyline_rows, lambda r: r.get("confidence") or "unknown"),
            "by_prob_bucket": _generic_split(moneyline_rows, _prob_bucket),
            "by_edge_bucket": _generic_split(moneyline_rows, _edge_bucket),
            "by_odds_bucket": _generic_split(moneyline_rows, _odds_bucket),
            "by_api_version": _generic_split(moneyline_rows, _api_version_bucket),
            "note": "Older graded moneyline records may not include odds/value metadata yet.",
        },

        "batter_home_runs": {
            "summary": _summarize_rows(hr_rows),
            "by_hr_tier": _generic_split(hr_rows, _hr_tier_bucket),
            "by_hr_score_bucket": _generic_split(hr_rows, _hr_score_bucket),
            "by_odds_status": _generic_split(hr_rows, _odds_status_bucket),
            "by_prob_bucket": _generic_split(hr_rows, _prob_bucket),
            "by_lineup_spot": _generic_split(hr_rows, _lineup_spot_bucket),
            "by_api_version": _generic_split(hr_rows, _api_version_bucket),
            "note": "HR model is experimental; v8.16E requires hr_score >= 1.70 and real FanDuel price for official HR picks.",
        },

        "retired": {
            "total": {
                "summary": _summarize_rows(_rows_by_market(results, "total")),
                "by_side": _generic_split(_rows_by_market(results, "total"), _pick_side),
            },
            "run_line": {
                "summary": _summarize_rows(_rows_by_market(results, "run_line")),
            },
        },
    }


def _load_record_doc():
    path = DATA_DIR / "record.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return {
        "summary": {"total": 0, "hits": 0, "misses": 0, "hit_rate": 0},
        "by_prop": {},
        "by_confidence": {},
        "by_bvp": {},
        "results": [],
    }


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
        "fan_duel_only": USE_ONLY_PREFERRED_BOOK,
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
            "hr_official_require_fanduel_price": HR_OFFICIAL_REQUIRE_FANDUEL_PRICE,
            "require_fanduel_line_for_official_hitters": REQUIRE_FANDUEL_LINE_FOR_OFFICIAL_HITTERS,
            "candidate_only_hitter_props": sorted(CANDIDATE_ONLY_HITTER_PROPS),
            "bad_official_hit_flags": sorted(BAD_OFFICIAL_HIT_FLAGS),
            "prediction_first_board": True,
            "odds_block_prediction_visibility": False,
            "game_team_caps_block_predictions": False,
            "caps_are_warnings_only": True,
            "hit_prediction_min_prob": 0.63,
            "tb_watchlist_min_prob": 0.63,
            "hr_watchlist_min_score": 1.70,
        },
        "pitcher_k_governor": {
            "require_fanduel_line_for_pitcher_ks": REQUIRE_FANDUEL_LINE_FOR_PITCHER_KS,
            "k_candidate_logging": True,
            "k_line_required": True,
            "k_odds_block_prediction_visibility": False,
            "under_min_confidence": K_UNDER_MIN_CONFIDENCE,
            "under_min_projection_gap": K_UNDER_MIN_PROJECTION_GAP,
            "over_no_lineup_min_prob": K_OVER_NO_LINEUP_MIN_PROB,
            "over_no_lineup_min_projection_gap": K_OVER_NO_LINEUP_MIN_PROJECTION_GAP,
        },
        "moneyline_gate": {
            "max_abs_moneyline_odds": MAX_ABS_MONEYLINE_ODDS,
            "requires_positive_edge": True,
            "fan_duel_only": True,
        },
        "line_matching": {
            "enabled": True,
            "fan_duel_only": True,
            "requested_markets_count": len(PROPLINE_MARKETS),
            "requested_markets": PROPLINE_MARKETS,
            "et_date_matching": True,
            "accent_normalization": True,
            "main_and_alternate_markets": True,
            "hr_one_way_parser": True,
            "require_fanduel_line_for_pitcher_ks": REQUIRE_FANDUEL_LINE_FOR_PITCHER_KS,
            "debug_endpoints": ["/debug/propline-fetch", "/debug/line-audit", "/debug/fanduel-market-probe", "/debug/k-candidate-log/latest"],
        },
        "record_intelligence": {
            "enabled": True,
            "active_mature": sorted(ACTIVE_MATURE_MARKETS),
            "active_probationary": sorted(PROBATIONARY_MARKETS),
            "experimental": sorted(EXPERIMENTAL_MARKETS),
            "retired": sorted(RETIRED_MARKETS),
            "endpoints": ["/record/active", "/debug/record-splits"],
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


@app.get("/debug/propline-fetch")
def debug_propline_fetch():
    over_under, thresholds, book_of = fetch_propline_props()

    by_market = {}
    by_line_source = {}
    for player, market in over_under.keys():
        by_market.setdefault(market, 0)
        by_market[market] += 1

        src = over_under[(player, market)].get("line_source") or "unknown"
        by_line_source.setdefault(src, 0)
        by_line_source[src] += 1

    threshold_by_market = {}
    for player, market in thresholds.keys():
        threshold_by_market.setdefault(market, 0)
        threshold_by_market[market] += 1

    return {
        "version": VERSION,
        "generated_at": now_et().isoformat(),
        "fan_duel_only": True,
        "line_count": len(over_under),
        "threshold_count": len(thresholds),
        "lines_by_market": by_market,
        "lines_by_source": by_line_source,
        "threshold_by_market": threshold_by_market,
        "audit": LAST_LINE_AUDIT,
    }


@app.get("/debug/fanduel-market-probe")
def debug_fanduel_market_probe(max_events: int = 3):
    if not PROPLINE_KEY:
        return {
            "version": VERSION,
            "status": "missing_PROPLINE_API_KEY",
        }

    events = get(f"{PROPLINE_BASE}/events", apiKey=PROPLINE_KEY)
    if not isinstance(events, list):
        return {
            "version": VERSION,
            "status": "events_response_not_list",
        }

    today_events = []
    for ev in events:
        if not _is_real_game(ev):
            continue
        if not _event_is_today_et(ev):
            continue
        today_events.append(ev)

    if max_events <= 0:
        max_events = 3

    checked_events = today_events[:max_events]
    report = {}

    for raw_market in PROPLINE_MARKETS:
        row = {
            "canonical_market": _canonical_market_key(raw_market),
            "events_checked": 0,
            "events_with_fanduel": 0,
            "events_with_requested_market": 0,
            "markets_seen": {},
            "outcomes_seen": 0,
            "sample_outcomes": [],
        }

        for ev in checked_events:
            eid = ev.get("id")
            if not eid:
                continue

            row["events_checked"] += 1

            data = get(
                f"{PROPLINE_BASE}/events/{eid}/odds",
                apiKey=PROPLINE_KEY,
                markets=raw_market,
                regions="us",
            )

            book, book_key = _pick_book(data.get("bookmakers") or [])
            if not book:
                continue

            row["events_with_fanduel"] += 1

            markets = book.get("markets", [])
            if markets:
                row["events_with_requested_market"] += 1

            for mkt in markets:
                mk = mkt.get("key")
                row["markets_seen"][str(mk)] = row["markets_seen"].get(str(mk), 0) + 1

                for o in mkt.get("outcomes", []):
                    row["outcomes_seen"] += 1
                    if len(row["sample_outcomes"]) < 10:
                        row["sample_outcomes"].append({
                            "market": mk,
                            "name": o.get("name"),
                            "description": o.get("description"),
                            "player": o.get("player"),
                            "participant": o.get("participant"),
                            "point": o.get("point"),
                            "price": o.get("price"),
                            "normalized_player": _player_name_from_outcome(o),
                        })

            time.sleep(0.05)

        report[raw_market] = row

    return {
        "version": VERSION,
        "generated_at": now_et().isoformat(),
        "fan_duel_only": True,
        "total_today_events": len(today_events),
        "events_checked": len(checked_events),
        "note": "Use ?max_events=15 if you want the whole slate.",
        "markets": report,
    }


@app.get("/debug/line-audit")
def debug_line_audit():
    today = today_et().isoformat()
    preds = _load_json_file(PRED_DIR / f"predictions_{today}.json", [])
    hitter_candidates = _load_json_file(PRED_DIR / f"hitter_candidates_{today}.json", [])

    if not isinstance(preds, list):
        preds = []
    if not isinstance(hitter_candidates, list):
        hitter_candidates = []

    def summarize_rows(rows):
        by_prop = {}
        missing = []

        for p in rows:
            if not isinstance(p, dict):
                continue

            prop = p.get("prop_type", "unknown")
            by_prop.setdefault(prop, {
                "total": 0,
                "with_line": 0,
                "missing_line": 0,
                "fallback_line": 0,
                "k_line_locked": 0,
                "k_8_16e_gate": 0,
                "books": {},
                "line_sources": {},
                "raw_markets": {},
                "reject_reasons": {},
            })

            by_prop[prop]["total"] += 1

            source = p.get("line_source") or "unknown"
            raw_market = p.get("raw_market") or "unknown"
            by_prop[prop]["line_sources"][source] = by_prop[prop]["line_sources"].get(source, 0) + 1
            by_prop[prop]["raw_markets"][raw_market] = by_prop[prop]["raw_markets"].get(raw_market, 0) + 1

            if p.get("k_line_locked"):
                by_prop[prop]["k_line_locked"] += 1

            if p.get("k_gate_version") == "8.16E":
                by_prop[prop]["k_8_16e_gate"] += 1

            rr = p.get("reject_reason")
            if rr:
                by_prop[prop]["reject_reasons"][rr] = by_prop[prop]["reject_reasons"].get(rr, 0) + 1

            if p.get("line_source") == "standard_fallback":
                by_prop[prop]["fallback_line"] += 1

            if p.get("has_line") or p.get("odds") is not None:
                by_prop[prop]["with_line"] += 1
                book = p.get("book") or "unknown_book"
                by_prop[prop]["books"][book] = by_prop[prop]["books"].get(book, 0) + 1
            else:
                by_prop[prop]["missing_line"] += 1
                if len(missing) < 40:
                    missing.append({
                        "player": p.get("player"),
                        "normalized_player": _norm(p.get("player")),
                        "team": p.get("team"),
                        "prop_type": prop,
                        "pick": p.get("pick"),
                        "projected": p.get("projected"),
                        "model_prob": p.get("model_prob"),
                        "bvp_flag": p.get("bvp_flag"),
                        "line_source": p.get("line_source"),
                        "raw_market": p.get("raw_market"),
                        "k_line_locked": p.get("k_line_locked"),
                        "k_gate_version": p.get("k_gate_version"),
                        "k_projection_gap": p.get("k_projection_gap"),
                        "k_has_lineup_kr": p.get("k_has_lineup_kr"),
                        "board_status": p.get("board_status"),
                        "reject_reason": p.get("reject_reason"),
                    })

        for prop, row in by_prop.items():
            total = row["total"]
            row["line_coverage_pct"] = round(row["with_line"] / total * 100, 1) if total else 0
            row["fallback_pct"] = round(row["fallback_line"] / total * 100, 1) if total else 0

        return by_prop, missing

    official_by_prop, official_missing = summarize_rows(preds)
    candidate_by_prop, candidate_missing = summarize_rows(hitter_candidates)

    return {
        "version": VERSION,
        "generated_at": now_et().isoformat(),
        "today": today,
        "fan_duel_only": True,
        "official_board_line_coverage": official_by_prop,
        "official_sample_missing_line_picks": official_missing,
        "hitter_candidate_line_coverage": candidate_by_prop,
        "hitter_candidate_sample_missing_line_picks": candidate_missing,
        "last_propline_audit": LAST_LINE_AUDIT,
    }


@app.get("/debug/record-splits")
def debug_record_splits():
    data = _load_record_doc()
    results = data.get("results", [])
    if not isinstance(results, list):
        results = []
    results = [r for r in results if isinstance(r, dict)]
    return _build_record_splits(results)


@app.get("/record/active")
def record_active():
    data = _load_record_doc()
    results = data.get("results", [])
    if not isinstance(results, list):
        results = []
    results = [r for r in results if isinstance(r, dict)]
    return _record_active_view(results)


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




# =========================
# v8.17 CANDIDATE LOGGING + DOWNLOADS
# =========================

def _candidate_pick_side(row):
    pick = str(row.get("pick", "")).upper().strip()
    if row.get("prop_type") == "moneyline":
        return "ML"
    if pick.startswith("OVER"):
        return "OVER"
    if pick.startswith("UNDER"):
        return "UNDER"
    if "+" in pick:
        return "THRESHOLD"
    return "UNKNOWN"


def _candidate_pick_line(row):
    pick = str(row.get("pick", "")).upper().strip()

    m = re.search(r"(?:OVER|UNDER)\s+(-?\d+(?:\.\d+)?)", pick)
    if m:
        return _safe_float(m.group(1), None)

    m2 = re.search(r"^(\d+)\+", pick)
    if m2:
        return _safe_float(m2.group(1), None)

    return None


def _candidate_key(row):
    return "|".join([
        str(row.get("date", "")),
        str(row.get("game_id", "")),
        str(row.get("player_id") or _norm(row.get("player", ""))),
        str(row.get("prop_type", "")),
        str(row.get("pick", "")),
    ])


def _clean_candidate_log_row(row, date_str, source):
    r = dict(row)
    status = r.get("board_status")

    if not status:
        status = "official" if source == "official_board" else "candidate"

    r["date"] = r.get("date") or date_str
    r["candidate_logger_version"] = VERSION
    r["candidate_logged_at"] = now_et().isoformat()
    r["candidate_source"] = r.get("candidate_source") or source
    r["candidate_status"] = status
    r["official_board"] = source == "official_board" or status in ("official", "official_prediction", "watchlist_prediction")
    r["candidate_key"] = _candidate_key(r)
    r["pick_side"] = _candidate_pick_side(r)
    r["pick_line"] = _candidate_pick_line(r)

    if r.get("projected") is not None:
        r["raw_projected"] = r.get("projected")

    if r.get("model_prob") is not None:
        r["raw_model_prob"] = r.get("model_prob")

    return r


def _candidate_snapshot_summary(rows):
    by_status = defaultdict(int)
    by_prop = defaultdict(int)
    by_prop_status = defaultdict(int)
    by_reject_reason = defaultdict(int)

    for r in rows:
        prop = r.get("prop_type", "unknown")
        status = r.get("candidate_status", "unknown")
        reject = r.get("reject_reason") or "none"

        by_status[status] += 1
        by_prop[prop] += 1
        by_prop_status[f"{prop}|{status}"] += 1
        by_reject_reason[reject] += 1

    return {
        "total_candidates": len(rows),
        "official_board_candidates": sum(1 for r in rows if r.get("official_board")),
        "by_status": dict(sorted(by_status.items())),
        "by_prop": dict(sorted(by_prop.items())),
        "by_prop_status": dict(sorted(by_prop_status.items())),
        "by_reject_reason": dict(sorted(by_reject_reason.items())),
    }


def snapshot_candidate_log(date_str, official_rows, hitter_debug_rows):
    try:
        merged = {}

        if not isinstance(official_rows, list):
            official_rows = []
        if not isinstance(hitter_debug_rows, list):
            hitter_debug_rows = []

        for row in hitter_debug_rows:
            if not isinstance(row, dict):
                continue
            c = _clean_candidate_log_row(row, date_str, "hitter_candidates")
            merged[c["candidate_key"]] = c

        for row in official_rows:
            if not isinstance(row, dict):
                continue

            c = _clean_candidate_log_row(row, date_str, "official_board")
            key = c["candidate_key"]

            if key in merged:
                old = merged[key]
                old["official_board"] = True
                old["candidate_status"] = c.get("candidate_status") or "official_prediction"
                old["candidate_source"] = "hitter_candidates+official_board"
                old["official_api_version"] = c.get("api_version")
                old["official_model_prob"] = c.get("model_prob")
                old["official_projected"] = c.get("projected")
                merged[key] = old
            else:
                merged[key] = c

        candidates = list(merged.values())
        candidates.sort(
            key=lambda r: (
                str(r.get("game_id", "")),
                str(r.get("team", "")),
                str(r.get("prop_type", "")),
                str(r.get("player", "")),
            )
        )

        out_path = LOG_DIR / f"candidates_{date_str}.json"
        out_path.write_text(json.dumps(candidates, indent=2))

        latest_path = LOG_DIR / "latest_candidate_snapshot.json"
        latest_path.write_text(json.dumps({
            "version": VERSION,
            "date": date_str,
            "generated_at": now_et().isoformat(),
            "summary": _candidate_snapshot_summary(candidates),
            "candidates": candidates,
        }, indent=2))

        print(f"  Candidate log saved: {out_path} ({len(candidates)} rows)")
    except Exception as e:
        print(f"  Candidate log failed: {e}")


def _load_candidate_snapshot(date_str=None):
    if date_str is None:
        path = LOG_DIR / "latest_candidate_snapshot.json"
    else:
        path = LOG_DIR / f"candidates_{date_str}.json"

    if not path.exists():
        return {
            "version": VERSION,
            "status": "missing",
            "path": str(path),
        }

    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return {
                "version": VERSION,
                "date": date_str,
                "summary": _candidate_snapshot_summary(data),
                "candidates": data,
            }
        return data
    except Exception as e:
        return {
            "version": VERSION,
            "status": "error",
            "path": str(path),
            "error": str(e),
        }


def _zip_folder(folder, zip_path):
    folder = Path(folder)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        if folder.exists():
            for p in folder.glob("*.json"):
                z.write(p, arcname=p.name)

    return zip_path


@app.get("/debug/candidate-log/latest")
def debug_candidate_log_latest():
    return _load_candidate_snapshot()


@app.get("/debug/candidate-log")
def debug_candidate_log(date: str = None):
    return _load_candidate_snapshot(date)


@app.get("/debug/candidate-log/report")
def debug_candidate_log_report():
    record_path = LOG_DIR / "candidate_record.json"

    out = {
        "version": VERSION,
        "generated_at": now_et().isoformat(),
        "candidate_logging_enabled": True,
        "latest_snapshot": _load_candidate_snapshot(),
        "candidate_record_available": record_path.exists(),
    }

    if record_path.exists():
        try:
            record = json.loads(record_path.read_text())
            out["candidate_record_summary"] = {
                "version": record.get("version"),
                "last_updated": record.get("last_updated"),
                "summary": record.get("summary"),
                "by_prop": record.get("by_prop"),
                "by_status": record.get("by_status"),
                "by_official_board": record.get("by_official_board"),
                "by_prop_status": record.get("by_prop_status"),
                "by_reject_reason": record.get("by_reject_reason"),
            }
        except Exception as e:
            out["candidate_record_error"] = str(e)

    return out


@app.get("/debug/download-candidate-logs")
def debug_download_candidate_logs():
    zip_path = _zip_folder(LOG_DIR, DATA_DIR / "candidate_logs.zip")
    return FileResponse(
        path=str(zip_path),
        filename="candidate_logs.zip",
        media_type="application/zip",
    )


@app.get("/debug/download-projection-audits")
def debug_download_projection_audits():
    folder = DATA_DIR / "projection_audits"
    zip_path = _zip_folder(folder, DATA_DIR / "projection_audits.zip")
    return FileResponse(
        path=str(zip_path),
        filename="projection_audits.zip",
        media_type="application/zip",
    )


@app.get("/debug/k-candidate-log/latest")
def debug_k_candidate_log_latest():
    path = PRED_DIR / f"pitcher_k_candidates_{today_et().isoformat()}.json"
    rows = _load_json_file(path, [])
    if not isinstance(rows, list):
        rows = []

    by_status = {}
    by_reason = {}
    by_side = {}
    with_line = 0

    for r in rows:
        st = r.get("board_status") or r.get("candidate_status") or "unknown"
        by_status[st] = by_status.get(st, 0) + 1
        rr = r.get("reject_reason") or "none"
        by_reason[rr] = by_reason.get(rr, 0) + 1
        side = r.get("pick_side") or _pick_side(r)
        by_side[side] = by_side.get(side, 0) + 1
        if r.get("has_k_line") or r.get("has_line"):
            with_line += 1

    return {
        "version": VERSION,
        "date": today_et().isoformat(),
        "generated_at": now_et().isoformat(),
        "k_candidate_logging": True,
        "k_line_required": True,
        "k_odds_block_prediction_visibility": False,
        "path": str(path),
        "summary": {
            "total_k_candidates": len(rows),
            "with_k_line": with_line,
            "without_k_line": len(rows) - with_line,
            "by_status": by_status,
            "by_reject_reason": by_reason,
            "by_side": by_side,
        },
        "candidates": rows,
    }


# =========================
# v8.17 PROJECTION AUDIT ENDPOINTS
# Diagnostic only. No model changes.
# =========================

def _v817_audit_date_filter(row, days=None, date=None, api_version=None, market=None):
    if not isinstance(row, dict):
        return False

    if date and row.get("date") != date:
        return False

    if api_version and str(row.get("api_version")) != str(api_version):
        return False

    if market and row.get("prop_type") != market:
        return False

    if days is not None:
        try:
            d = dt.date.fromisoformat(row.get("date"))
            cutoff = today_et() - dt.timedelta(days=int(days))
            if d < cutoff:
                return False
        except Exception:
            return False

    return True


def _v817_audit_error_row(row):
    projected = _safe_float(row.get("projected"), None)
    actual = _safe_float(row.get("actual"), None)

    if projected is None or actual is None:
        return None

    out = dict(row)
    out["_projection_error"] = projected - actual
    out["_projection_abs_error"] = abs(projected - actual)
    return out


def _v817_audit_empty_summary():
    return {
        "total": 0,
        "hits": 0,
        "misses": 0,
        "hit_rate": 0,
        "avg_projected": None,
        "avg_actual": None,
        "projection_bias": None,
        "avg_abs_error": None,
        "bias_interpretation": "not_available",
    }


def _v817_audit_summarize(rows):
    rows = [r for r in rows if isinstance(r, dict)]
    total = len(rows)

    if total == 0:
        return _v817_audit_empty_summary()

    hits = sum(1 for r in rows if r.get("result") == "hit")
    misses = sum(1 for r in rows if r.get("result") == "miss")

    projected_vals = []
    actual_vals = []
    errors = []
    abs_errors = []

    for r in rows:
        projected = _safe_float(r.get("projected"), None)
        actual = _safe_float(r.get("actual"), None)

        if projected is None or actual is None:
            continue

        projected_vals.append(projected)
        actual_vals.append(actual)
        errors.append(projected - actual)
        abs_errors.append(abs(projected - actual))

    avg_projected = sum(projected_vals) / len(projected_vals) if projected_vals else None
    avg_actual = sum(actual_vals) / len(actual_vals) if actual_vals else None
    bias = sum(errors) / len(errors) if errors else None
    avg_abs = sum(abs_errors) / len(abs_errors) if abs_errors else None

    if bias is None:
        bias_text = "not_available"
    elif bias > 0.05:
        bias_text = "over_projecting"
    elif bias < -0.05:
        bias_text = "under_projecting"
    else:
        bias_text = "roughly_balanced"

    return {
        "total": total,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total * 100, 1) if total else 0,
        "avg_projected": round(avg_projected, 3) if avg_projected is not None else None,
        "avg_actual": round(avg_actual, 3) if avg_actual is not None else None,
        "projection_bias": round(bias, 3) if bias is not None else None,
        "avg_abs_error": round(avg_abs, 3) if avg_abs is not None else None,
        "bias_interpretation": bias_text,
    }


def _v817_audit_group(rows, key_func):
    grouped = defaultdict(list)

    for r in rows:
        try:
            key = key_func(r)
        except Exception:
            key = "unknown"

        if key is None or key == "":
            key = "unknown"

        grouped[str(key)].append(r)

    return {k: _v817_audit_summarize(grouped[k]) for k in sorted(grouped.keys())}


def _v817_audit_compact_row(row):
    return {
        "date": row.get("date"),
        "api_version": row.get("api_version"),
        "player": row.get("player"),
        "team": row.get("team"),
        "opponent": row.get("opponent"),
        "game_id": row.get("game_id"),
        "prop_type": row.get("prop_type"),
        "pick": row.get("pick"),
        "projected": row.get("projected"),
        "actual": row.get("actual"),
        "projection_error": round(row.get("_projection_error"), 3) if row.get("_projection_error") is not None else None,
        "abs_error": round(row.get("_projection_abs_error"), 3) if row.get("_projection_abs_error") is not None else None,
        "result": row.get("result"),
        "model_prob": row.get("model_prob"),
        "confidence": row.get("confidence"),
        "bvp_flag": row.get("bvp_flag"),
        "lineup_spot": row.get("lineup_spot"),
        "odds": row.get("odds"),
        "book": row.get("book"),
        "line_source": row.get("line_source"),
        "raw_market": row.get("raw_market"),
        "k_gate_version": row.get("k_gate_version"),
        "k_projection_gap": row.get("k_projection_gap"),
        "k_has_lineup_kr": row.get("k_has_lineup_kr"),
        "hr_score": row.get("hr_score"),
        "hr_tier": row.get("hr_tier"),
    }


@app.get("/debug/projection-audit")
def debug_projection_audit(days: int = None, date: str = None, market: str = None,
                           api_version: str = None, limit: int = 25):
    data = _load_record_doc()
    results = data.get("results", [])
    if not isinstance(results, list):
        results = []

    filtered = [
        r for r in results
        if _v817_audit_date_filter(r, days=days, date=date, api_version=api_version, market=market)
    ]

    audit_rows = []
    missing_rows = []

    for r in filtered:
        er = _v817_audit_error_row(r)
        if er is None:
            missing_rows.append(r)
        else:
            audit_rows.append(er)

    limit = max(1, min(int(limit or 25), 100))

    worst_misses = sorted(
        [r for r in audit_rows if r.get("result") == "miss"],
        key=lambda r: r.get("_projection_abs_error", 0),
        reverse=True,
    )[:limit]

    closest_hits = sorted(
        [r for r in audit_rows if r.get("result") == "hit"],
        key=lambda r: r.get("_projection_abs_error", 999),
    )[:limit]

    biggest_hits = sorted(
        [r for r in audit_rows if r.get("result") == "hit"],
        key=lambda r: r.get("_projection_abs_error", 0),
        reverse=True,
    )[:limit]

    return {
        "version": VERSION,
        "generated_at": now_et().isoformat(),
        "diagnostic_only": True,
        "leakage_protection": {
            "uses_saved_record_json": True,
            "does_not_rerun_old_games": True,
            "does_not_retrain_models": True,
            "does_not_change_picks": True,
        },
        "filters": {
            "days": days,
            "date": date,
            "market": market,
            "api_version": api_version,
            "limit": limit,
        },
        "record_rows_seen": len(results),
        "filtered_rows": len(filtered),
        "rows_with_projection_and_actual": len(audit_rows),
        "rows_missing_projection_or_actual": len(missing_rows),
        "summary": _v817_audit_summarize(audit_rows),
        "by_market": _v817_audit_group(audit_rows, lambda r: r.get("prop_type")),
        "by_api_version": _v817_audit_group(audit_rows, lambda r: r.get("api_version") or "no_api_version_recorded"),
        "by_side": _v817_audit_group(audit_rows, _pick_side),
        "by_market_side": _v817_audit_group(audit_rows, lambda r: f"{r.get('prop_type')}|{_pick_side(r)}"),
        "by_confidence": _v817_audit_group(audit_rows, lambda r: r.get("confidence") or "unknown"),
        "by_probability_bucket": _v817_audit_group(audit_rows, _prob_bucket),
        "by_bvp_bucket": _v817_audit_group(audit_rows, _bvp_bucket),
        "by_lineup_spot": _v817_audit_group(audit_rows, _lineup_spot_bucket),
        "by_lineup_k_flag": _v817_audit_group(audit_rows, _lineup_flag_bucket),
        "by_projection_gap_bucket": _v817_audit_group(audit_rows, _projection_gap_bucket),
        "by_line_bucket": _v817_audit_group(audit_rows, _line_bucket),
        "by_line_source": _v817_audit_group(audit_rows, lambda r: r.get("line_source") or "unknown"),
        "by_k_gate_version": _v817_audit_group(audit_rows, lambda r: r.get("k_gate_version") or "no_k_gate_version"),
        "by_hr_score_bucket": _v817_audit_group(audit_rows, _hr_score_bucket),
        "worst_projection_misses": [_v817_audit_compact_row(r) for r in worst_misses],
        "biggest_projection_hits": [_v817_audit_compact_row(r) for r in biggest_hits],
        "closest_projection_hits": [_v817_audit_compact_row(r) for r in closest_hits],
        "training_read": {
            "projection_bias": "positive means projected stat was higher than actual; negative means projected stat was lower than actual",
            "avg_abs_error": "average size of projection miss regardless of direction",
            "best_use": "use this to decide projection upgrades, not to prove the app works",
        },
    }


@app.get("/debug/projection-audit/recent")
def debug_projection_audit_recent(days: int = 7, limit: int = 25):
    return debug_projection_audit(days=days, limit=limit)


@app.get("/debug/projection-audit/market/{market_name}")
def debug_projection_audit_market(market_name: str, days: int = None, limit: int = 25):
    return debug_projection_audit(days=days, market=market_name, limit=limit)


@app.get("/record")
def record():
    return _load_record_doc()
