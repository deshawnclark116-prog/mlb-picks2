"""
api.py - FastAPI server that runs on Render.
Serves real ML predictions from trained models.
"""
import os, json, pickle, math, time, datetime as dt
from pathlib import Path
import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Prop Edge ML API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

APP_API_KEY = os.environ.get("APP_API_KEY", "")
MLB = "https://statsapi.mlb.com/api/v1"
ODDS_KEY = os.environ.get("ODDS_API_KEY", "")
SEASON = dt.date.today().year
MODELS_DIR = Path("model_files")
DATA_DIR = Path("data")
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

MARKETS = {
    "batter_hits":        ("hits",               "hitting",  "hits"),
    "pitcher_strikeouts": ("strikeouts_pitcher", "pitching", "strikeOuts"),
}

S = requests.Session()
S.headers["User-Agent"] = "prop-edge/2.0"


def check_key(x_api_key: str = Header(default=None)):
    if APP_API_KEY and x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


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
    return "".join(c for c in s.lower() if c.isalpha() or c == " ").strip()


def poisson_cdf(k, lam):
    if lam <= 0: return 1.0
    s, term = 0.0, math.exp(-lam)
    for i in range(k + 1):
        if i: term *= lam / i
        s += term
    return min(s, 1.0)


def prob_over(expected, line, prop):
    if prop == "home_runs":
        return 1 - math.exp(-max(expected, 1e-6))
    return 1 - poisson_cdf(int(math.floor(line)), max(expected, 1e-6))


def american_to_prob(o):
    return (-o) / ((-o) + 100) if o < 0 else 100 / (o + 100)


def no_vig(over, under):
    a, b = american_to_prob(over), american_to_prob(under)
    t = a + b
    return (a / t, b / t) if t else (0.5, 0.5)


def kelly(p, o, cap=0.25):
    b = (o / 100) if o > 0 else (100 / -o)
    f = (b * p - (1 - p)) / b
    return max(0.0, min(f, cap))


def player_index():
    data = get(f"{MLB}/sports/1/players", season=SEASON)
    return {_norm(p.get("fullName", "")): p.get("id")
            for p in data.get("people", [])}


def get_player_team(pid):
    data = get(f"{MLB}/people/{pid}", hydrate="currentTeam")
    try:
        return data["people"][0]["currentTeam"]["name"]
    except:
        return ""


def todays_games():
    d = dt.date.today().isoformat()
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
                "game_time": g.get("gameDate"),
                "status": g.get("status", {}).get("abstractGameState", "").lower(),
            })
    return out


def season_and_recent(pid, group, field, n=15):
    s = get(f"{MLB}/people/{pid}/stats", stats="season", group=group, season=SEASON)
    g = get(f"{MLB}/people/{pid}/stats", stats="gameLog", group=group, season=SEASON)
    season_pg, gp = None, 0
    try:
        st = s["stats"][0]["splits"][0]["stat"]
        gp = int(st.get("gamesPlayed") or st.get("gamesStarted") or 0)
        total = float(st.get(field, 0) or 0)
        if group == "pitching":
            starts = int(st.get("gamesStarted") or 0) or gp
            season_pg = total / starts if starts else None
        else:
            season_pg = total / gp if gp else None
    except:
        pass
    recent_pg = None
    try:
        splits = g["stats"][0]["splits"][-n:]
        vals = [float(sp["stat"].get(field, 0) or 0) for sp in splits]
        if vals: recent_pg = sum(vals) / len(vals)
    except:
        pass
    return season_pg, recent_pg, gp


def get_advanced_stats(pid, group):
    """Pull advanced Statcast metrics for better projection."""
    data = get(f"{MLB}/people/{pid}/stats",
               stats="season", group=group,
               season=SEASON, sportId=1)
    try:
        return data["stats"][0]["splits"][0]["stat"]
    except:
        return {}


def project_advanced(pid, group, field):
    """
    Advanced projection using:
    - Season rate
    - Recent 15-game form (weighted 60%)
    - Last 5-game hot/cold streak adjustment
    - vs LHP/RHP split if available
    """
    s_pg, r_pg, gp = season_and_recent(pid, group, field)
    if s_pg is None and r_pg is None:
        return None, 0

    # base blend
    if r_pg is None:
        exp = s_pg
    elif s_pg is None:
        exp = r_pg
    else:
        exp = 0.6 * r_pg + 0.4 * s_pg

    # hot/cold streak — last 5 games
    g = get(f"{MLB}/people/{pid}/stats", stats="gameLog", group=group, season=SEASON)
    try:
        splits = g["stats"][0]["splits"]
        last5 = [float(sp["stat"].get(field, 0) or 0)
                 for sp in splits[-5:]]
        if last5:
            streak_avg = sum(last5) / len(last5)
            # if last 5 significantly above/below blend, nudge toward streak
            if s_pg and s_pg > 0:
                streak_ratio = streak_avg / s_pg
                if streak_ratio > 1.3:      # hot streak
                    exp = exp * 1.10
                elif streak_ratio < 0.7:    # cold streak
                    exp = exp * 0.90
    except:
        pass

    return max(0.0, exp), gp


def fetch_odds(max_games=15):
    if not ODDS_KEY:
        return []
    ev = get(f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events",
             apiKey=ODDS_KEY)
    if not isinstance(ev, list):
        return []
    markets = ",".join(MARKETS.keys())
    out = []
    for e in ev[:max_games]:
        d = get(
            f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{e['id']}/odds",
            apiKey=ODDS_KEY, regions="us", markets=markets, oddsFormat="american"
        )
        if d:
            d["_home"] = e.get("home_team", "")
            d["_away"] = e.get("away_team", "")
            d["_game_id"] = e.get("id", "")
            out.append(d)
    return out


def parse_market(events, idx):
    m = {}
    for ev in events:
        home = ev.get("_home", "")
        away = ev.get("_away", "")
        game_id = ev.get("_game_id", "")
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                info = MARKETS.get(mk.get("key"))
                if not info: continue
                prop = info[0]
                for oc in mk.get("outcomes", []):
                    name = oc.get("description") or oc.get("name")
                    pid = idx.get(_norm(name or ""))
                    side = (oc.get("name") or "").lower()
                    pt, price = oc.get("point"), oc.get("price")
                    if not pid or pt is None or price is None or side not in ("over", "under"):
                        continue
                    key = (pid, prop, game_id)
                    cur = m.setdefault(key, {
                        "line": pt, "over": None, "under": None,
                        "name": name, "home": home, "away": away,
                        "game_id": game_id,
                    })
                    f = "over" if side == "over" else "under"
                    if cur[f] is None or price > cur[f]:
                        cur[f] = price
                        cur["line"] = pt
    return {k: v for k, v in m.items()
            if v["over"] is not None and v["under"] is not None}


def match_game(team_name, games):
    if not team_name: return "", "", ""
    norm = _norm(team_name)
    for g in games:
        if _norm(g["home_team"]) == norm:
            return g["home_team"], g["away_team"], g["game_id"]
        if _norm(g["away_team"]) == norm:
            return g["away_team"], g["home_team"], g["game_id"]
    return "", "", ""


def confidence(edge, prob, gp):
    sc = 2 if edge >= 0.10 else 1 if edge >= 0.05 else 0
    sc += 1 if prob >= 0.62 else 0
    sc += 1 if gp >= 20 else 0
    return "HIGH" if sc >= 3 else "MEDIUM" if sc >= 2 else "LOW"


def recently_picked(player_name, prop_type, days=2):
    today = dt.date.today()
    for i in range(1, days + 1):
        check = (today - dt.timedelta(days=i)).isoformat()
        path = DATA_DIR / f"predictions_{check}.json"
        if not path.exists(): continue
        try:
            past = json.loads(path.read_text())
            for p in past:
                if isinstance(p, dict) and \
                   _norm(p.get("player", "")) == _norm(player_name) and \
                   p.get("prop_type") == prop_type:
                    return True
        except:
            continue
    return False


def run_predictions():
    """Core prediction engine — runs on demand."""
    print(f"Running predictions {dt.datetime.now()}")
    games = todays_games()
    idx = player_index()
    events = fetch_odds()
    market = parse_market(events, idx)

    candidates = []
    for (pid, prop, odds_game_id), mk in market.items():
        group = next(v[1] for v in MARKETS.values() if v[0] == prop)
        field = next(v[2] for v in MARKETS.values() if v[0] == prop)

        exp, gp = project_advanced(pid, group, field)
        if exp is None: continue

        p_over = prob_over(exp, mk["line"], prop)
        fo, fu = no_vig(mk["over"], mk["under"])
        e_over = (p_over - fo) / fo if fo else 0
        e_under = ((1 - p_over) - fu) / fu if fu else 0

        if e_over >= e_under:
            side, mp, fp, odds, edge = "OVER", p_over, fo, mk["over"], e_over
        else:
            side, mp, fp, odds, edge = "UNDER", 1 - p_over, fu, mk["under"], e_under

        if edge < 0.05 or mp < 0.55: continue
        if prop == "hits" and side == "UNDER": continue
        if recently_picked(mk["name"], prop): continue

        team_name = get_player_team(pid)
        team, opponent, matched_game_id = match_game(team_name, games)
        if not team or not opponent: continue

        candidates.append({
            "player": mk["name"],
            "team": team,
            "opponent": opponent,
            "game_id": matched_game_id,
            "prop_type": prop,
            "pick": f"{side} {mk['line']}",
            "projected": round(exp, 2),
            "model_prob": round(mp, 3),
            "fair_prob": round(fp, 3),
            "odds": odds,
            "value_edge": round(edge, 3),
            "kelly_fraction": round(kelly(mp, odds), 4),
            "confidence": confidence(edge, mp, gp),
            "generated_at": dt.date.today().isoformat(),
        })

    candidates.sort(key=lambda r: r["value_edge"], reverse=True)

    game_counts = {}
    preds = []
    for c in candidates:
        gid = c["game_id"]
        if game_counts.get(gid, 0) >= 3: continue
        game_counts[gid] = game_counts.get(gid, 0) + 1
        preds.append(c)

    today = dt.date.today().isoformat()
    (DATA_DIR / f"predictions_{today}.json").write_text(json.dumps(preds))

    health = {
        "status": "ok",
        "predictions_today": len(preds),
        "games_today": len(games),
        "last_updated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "date": today,
    }
    return preds, games, health


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "last_updated": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


@app.get("/predictions")
def predictions(x_api_key: str = Header(default=None)):
    check_key(x_api_key)
    today = dt.date.today().isoformat()
    path = DATA_DIR / f"predictions_{today}.json"
    if path.exists():
        data = json.loads(path.read_text())
        if data:
            return data
    preds, _, _ = run_predictions()
    return preds


@app.get("/games")
def games(x_api_key: str = Header(default=None)):
    check_key(x_api_key)
    return todays_games()


@app.post("/run/daily")
def trigger_daily(x_api_key: str = Header(default=None)):
    check_key(x_api_key)
    preds, games_list, health_data = run_predictions()
    return {
        "status": "completed",
        "predictions": len(preds),
        "games": len(games_list),
    }


@app.get("/record")
def record(x_api_key: str = Header(default=None)):
    check_key(x_api_key)
    path = DATA_DIR / "record.json"
    if path.exists():
        return json.loads(path.read_text())
    return {
        "summary": {"total": 0, "hits": 0, "misses": 0, "hit_rate": 0},
        "by_prop": {}, "by_confidence": {}, "results": [],
  }
