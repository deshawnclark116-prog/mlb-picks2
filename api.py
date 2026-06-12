"""
api.py - FastAPI server on Render.
Self-sustaining ML prop system WITH PropLine market layer.
batter_hits is OVER-only (never pick a player to go hitless); picks that don't
clear the edge/probability bar are dropped, not forced.
"""
import os, json, math, time, threading, datetime as dt
from pathlib import Path
from collections import defaultdict

import requests
import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import backfill

app = FastAPI(title="Prop Edge ML API", version="4.1")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

MLB = "https://statsapi.mlb.com/api/v1"
MLB11 = "https://statsapi.mlb.com/api/v1.1"
PROPLINE_KEY = os.environ.get("PROPLINE_API_KEY", "")
PROPLINE_BASE = "https://api.prop-line.com/v1/sports/baseball_mlb"
SEASON = dt.date.today().year
DATA_DIR = Path("/data")
MODEL_DIR = DATA_DIR / "models"
PRED_DIR = DATA_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
GH_PAGES_BASE = "https://deshawnclark116-prog.github.io/mlb-picks2"

S = requests.Session()
S.headers["User-Agent"] = "prop-edge/4.1"

STANDARD_LINE = {"batter_hits": 0.5, "pitcher_strikeouts": 4.5}
MIN_EDGE = 0.05

_models = {}

def load_models():
    _models.clear()
    for name in ("batter_hits", "pitcher_strikeouts"):
        mp = MODEL_DIR / f"{name}.json"
        cp = MODEL_DIR / f"{name}_columns.json"
        if mp.exists() and cp.exists():
            booster = xgb.Booster(); booster.load_model(str(mp))
            _models[name] = (booster, json.loads(cp.read_text()))
            print(f"Loaded model {name}")
        else:
            print(f"Model {name} not found")

load_models()


def model_predict(name, feat_dict):
    if name not in _models: return None
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
    return "".join(c for c in s.lower() if c.isalpha() or c == " ").strip()


def ip_to_outs(ip):
    try:
        whole = int(float(ip)); frac = round((float(ip) - whole) * 10)
        return whole * 3 + frac
    except: return 0


def poisson_cdf(k, lam):
    if lam <= 0: return 1.0
    s, term = 0.0, math.exp(-lam)
    for i in range(k + 1):
        if i: term *= lam / i
        s += term
    return min(s, 1.0)


def prob_over(expected, line):
    return 1 - poisson_cdf(int(math.floor(line)), max(expected, 1e-6))


# ── odds math ─────────────────────────────────────────────────────────────────

def american_to_prob(odds):
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def no_vig_two_way(over_odds, under_odds):
    po = american_to_prob(over_odds); pu = american_to_prob(under_odds)
    tot = po + pu
    if tot == 0: return 0.5, 0.5
    return po / tot, pu / tot


def value_edge(model_p, fair_p):
    if fair_p <= 0: return 0.0
    return (model_p - fair_p) / fair_p


def kelly_fraction(model_p, american_odds, cap=0.25):
    b = (american_odds / 100) if american_odds > 0 else (100 / -american_odds)
    q = 1 - model_p
    f = (b * model_p - q) / b if b else 0
    return max(0.0, min(f, cap))


# ── PropLine market layer ─────────────────────────────────────────────────────

def _is_real_game(ev):
    h = ev.get("home_team", "")
    return "(" not in h and "Runs" not in h


def fetch_propline_odds():
    market_index = {}
    if not PROPLINE_KEY:
        print("  No PROPLINE_API_KEY — projection-only mode")
        return market_index
    try:
        events = get(f"{PROPLINE_BASE}/events", apiKey=PROPLINE_KEY)
        if not isinstance(events, list):
            return market_index
    except Exception as e:
        print(f"  PropLine events failed: {e}")
        return market_index

    today = dt.date.today().isoformat()
    pulled = 0
    for ev in events:
        if not _is_real_game(ev): continue
        if not str(ev.get("commence_time", "")).startswith(today): continue
        eid = ev.get("id")
        if not eid: continue
        try:
            data = get(f"{PROPLINE_BASE}/events/{eid}/odds",
                       apiKey=PROPLINE_KEY,
                       markets="pitcher_strikeouts,batter_hits",
                       regions="us")
        except Exception as e:
            print(f"  PropLine odds {eid} failed: {e}")
            continue
        for book in (data.get("bookmakers") or []):
            for mkt in book.get("markets", []):
                mkey = mkt.get("key")
                per_player = defaultdict(dict)
                for o in mkt.get("outcomes", []):
                    player = _norm(o.get("description", ""))
                    side = o.get("name", "").lower()
                    per_player[player][side] = {
                        "price": o.get("price"), "point": o.get("point"),
                    }
                for player, sides in per_player.items():
                    if "over" in sides and "under" in sides:
                        key = (player, mkey)
                        if key not in market_index:
                            market_index[key] = {
                                "line": sides["over"]["point"],
                                "over_odds": sides["over"]["price"],
                                "under_odds": sides["under"]["price"],
                            }
        pulled += 1
        time.sleep(0.2)
    print(f"  PropLine: pulled odds for {pulled} games, "
          f"{len(market_index)} player-markets")
    return market_index


# ── MLB data + features ───────────────────────────────────────────────────────

def player_index():
    data = get(f"{MLB}/sports/1/players", season=SEASON)
    return {_norm(p.get("fullName", "")): p.get("id") for p in data.get("people", [])}


def get_player_team(pid):
    data = get(f"{MLB}/people/{pid}", hydrate="currentTeam")
    try: return data["people"][0]["currentTeam"]["name"]
    except: return ""


def todays_games():
    d = dt.date.today().isoformat()
    data = get(f"{MLB}/schedule", sportId=1, date=d, hydrate="probablePitcher,team")
    out = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            h = g["teams"]["home"]["team"]; a = g["teams"]["away"]["team"]
            out.append({
                "game_id": str(g.get("gamePk")), "date": d,
                "home_team": h.get("name"), "away_team": a.get("name"),
                "home_pitcher": (g["teams"]["home"].get("probablePitcher") or {}).get("id"),
                "away_pitcher": (g["teams"]["away"].get("probablePitcher") or {}).get("id"),
                "game_time": g.get("gameDate"),
                "status": g.get("status", {}).get("abstractGameState", "").lower(),
            })
    return out


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


def batter_feature_row(pid):
    g = get(f"{MLB}/people/{pid}/stats", stats="gameLog", group="hitting", season=SEASON)
    try: splits = g["stats"][0]["splits"]
    except: return None
    cum_h = cum_ab = cum_pa = cum_hr = cum_bb = cum_so = 0
    recent = []
    for sp in splits:
        st = sp["stat"]
        cum_h += int(st.get("hits", 0) or 0); cum_ab += int(st.get("atBats", 0) or 0)
        cum_pa += int(st.get("plateAppearances", 0) or 0)
        cum_hr += int(st.get("homeRuns", 0) or 0); cum_bb += int(st.get("baseOnBalls", 0) or 0)
        cum_so += int(st.get("strikeOuts", 0) or 0)
        recent.append(int(st.get("hits", 0) or 0))
    if cum_ab < 20 or len(recent) < 5: return None
    return {
        "season_avg": cum_h / cum_ab if cum_ab else 0,
        "recent15_avg": sum(recent[-15:]) / len(recent[-15:]),
        "recent5_avg": sum(recent[-5:]) / len(recent[-5:]),
        "hr_rate": cum_hr / cum_pa if cum_pa else 0,
        "bb_rate": cum_bb / cum_pa if cum_pa else 0,
        "so_rate": cum_so / cum_pa if cum_pa else 0,
        "batting_order": 9, "games_played": len(recent),
    }


def pitcher_feature_row(pid):
    g = get(f"{MLB}/people/{pid}/stats", stats="gameLog", group="pitching", season=SEASON)
    try: splits = g["stats"][0]["splits"]
    except: return None
    cum_bf = cum_so = cum_outs = cum_bb = 0
    recent_k = []; recent_bf = []; n_starts = 0
    for sp in splits:
        st = sp["stat"]
        bf = int(st.get("battersFaced", 0) or 0); so = int(st.get("strikeOuts", 0) or 0)
        outs = int(st.get("outs", 0) or 0) or ip_to_outs(st.get("inningsPitched", "0.0"))
        if bf >= 12:
            cum_bf += bf; cum_so += so; cum_outs += outs
            cum_bb += int(st.get("baseOnBalls", 0) or 0)
            recent_k.append(so); recent_bf.append(bf); n_starts += 1
    if n_starts < 3: return None
    return {
        "k_per_bf": cum_so / cum_bf if cum_bf else 0,
        "avg_bf": sum(recent_bf[-10:]) / len(recent_bf[-10:]),
        "recent_k_avg": sum(recent_k[-5:]) / len(recent_k[-5:]),
        "bb_rate": cum_bb / cum_bf if cum_bf else 0,
        "outs_per_start": cum_outs / n_starts if n_starts else 0,
        "starts": n_starts,
    }


def confidence(distance, gp, edge=None):
    sc = 2 if distance >= 0.4 else 1 if distance >= 0.2 else 0
    sc += 1 if gp >= 20 else 0
    if edge is not None and edge >= 0.10: sc += 1
    return "HIGH" if sc >= 3 else "MEDIUM" if sc >= 2 else "LOW"


def fetch_predictions_for(date_str):
    local = PRED_DIR / f"predictions_{date_str}.json"
    if local.exists():
        try: return json.loads(local.read_text())
        except: pass
    try:
        r = S.get(f"{GH_PAGES_BASE}/predictions_{date_str}.json", timeout=20)
        if r.status_code == 200: return r.json()
    except: pass
    return None


def append_yesterday_to_season():
    year = dt.date.today().year
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    season_file = DATA_DIR / f"season_{year}.jsonl"
    progress_file = DATA_DIR / f"season_{year}_progress.txt"
    done = set()
    if progress_file.exists():
        done = set(progress_file.read_text().splitlines())
    if yesterday in done:
        print(f"  {yesterday} already recorded, skipping append")
        return
    games = backfill.get_schedule(yesterday)
    if not games:
        print(f"  No final games for {yesterday}")
        return
    rows_written = 0
    with open(season_file, "a") as fout:
        for gpk in games:
            box = backfill.get_boxscore(gpk)
            if not box: continue
            rows = backfill.extract_player_lines(gpk, yesterday, box)
            for r in rows:
                fout.write(json.dumps(r) + "\n")
            rows_written += len(rows)
            time.sleep(0.3)
    with open(progress_file, "a") as p:
        p.write(yesterday + "\n")
    print(f"  Appended {rows_written} lines for {yesterday}")


def _make_pick(name, team, opp, gid, prop, proj, gp, market):
    """Build a pick. batter_hits is OVER-only; picks that don't clear the
    edge/probability bar are dropped (return None), never forced."""
    key = (_norm(name), prop)
    m = market.get(key)

    if m and m.get("over_odds") is not None and m.get("under_odds") is not None:
        line = m["line"]
        p_over = prob_over(proj, line)
        fair_over, fair_under = no_vig_two_way(m["over_odds"], m["under_odds"])
        edge_over = value_edge(p_over, fair_over)
        edge_under = value_edge(1 - p_over, fair_under)

        if prop == "batter_hits":
            if p_over < 0.55 and edge_over < MIN_EDGE:
                return None
            side, mp, fair_p, odds, edge = "OVER", p_over, fair_over, m["over_odds"], edge_over
        else:
            if edge_over >= edge_under:
                side, mp, fair_p, odds, edge = "OVER", p_over, fair_over, m["over_odds"], edge_over
            else:
                side, mp, fair_p, odds, edge = "UNDER", 1 - p_over, fair_under, m["under_odds"], edge_under
            if mp < 0.55 and edge < MIN_EDGE:
                return None

        return {
            "player": name, "team": team, "opponent": opp, "game_id": gid,
            "prop_type": prop, "pick": f"{side} {line}",
            "projected": round(proj, 2),
            "model_prob": round(mp, 3),
            "fair_prob": round(fair_p, 3),
            "odds": odds,
            "value_edge": round(edge, 3),
            "kelly": round(kelly_fraction(mp, odds), 4),
            "has_line": True,
            "is_edge": edge >= MIN_EDGE,
            "confidence": confidence(abs(proj - line), gp, edge),
            "generated_at": dt.date.today().isoformat(),
        }

    # no market line -> projection-only fallback
    line = STANDARD_LINE[prop]
    p_over = prob_over(proj, line)
    if prop == "batter_hits":
        if p_over < 0.55:
            return None
        side, mp = "OVER", p_over
    else:
        side = "OVER" if proj > line else "UNDER"
        mp = p_over if side == "OVER" else 1 - p_over
        if mp < 0.55:
            return None

    return {
        "player": name, "team": team, "opponent": opp, "game_id": gid,
        "prop_type": prop, "pick": f"{side} {line}",
        "projected": round(proj, 2), "model_prob": round(mp, 3),
        "fair_prob": None, "odds": None, "value_edge": None, "kelly": None,
        "has_line": False, "is_edge": False,
        "confidence": confidence(abs(proj - line), gp),
        "generated_at": dt.date.today().isoformat(),
    }


def run_predictions():
    print(f"Running predictions {dt.datetime.now()}")
    idx = player_index()

    try:
        append_yesterday_to_season()
    except Exception as e:
        print(f"  append step error (non-fatal): {e}")

    backfill_all_history(idx, days_back=20)

    market = fetch_propline_odds()
    games = todays_games()
    preds = []

    for game in games:
        gid = game["game_id"]
        for side in ("home_pitcher", "away_pitcher"):
            pid = game.get(side)
            if not pid: continue
            feat = pitcher_feature_row(pid)
            if not feat: continue
            proj = model_predict("pitcher_strikeouts", feat)
            if proj is None: continue
            pdata = get(f"{MLB}/people/{pid}")
            name = pdata.get("people", [{}])[0].get("fullName", "")
            team = get_player_team(pid)
            opp = game["away_team"] if side == "home_pitcher" else game["home_team"]
            pick = _make_pick(name, team, opp, gid, "pitcher_strikeouts",
                              proj, feat["starts"] * 5, market)
            if pick: preds.append(pick)

    for game in games:
        gid = game["game_id"]
        lineup = get_confirmed_lineup(gid)
        for tside in ("home", "away"):
            team_name = game["home_team"] if tside == "home" else game["away_team"]
            opp = game["away_team"] if tside == "home" else game["home_team"]
            count = 0
            for pid in lineup.get(tside, []):
                if count >= 4: break
                feat = batter_feature_row(pid)
                if not feat: continue
                proj = model_predict("batter_hits", feat)
                if proj is None: continue
                pdata = get(f"{MLB}/people/{pid}")
                name = pdata.get("people", [{}])[0].get("fullName", "")
                pick = _make_pick(name, team_name, opp, gid, "batter_hits",
                                  proj, feat["games_played"], market)
                if pick:
                    preds.append(pick); count += 1

    preds.sort(key=lambda r: (r.get("is_edge", False),
                              r.get("value_edge") or 0,
                              r.get("model_prob") or 0), reverse=True)
    today = dt.date.today().isoformat()
    (PRED_DIR / f"predictions_{today}.json").write_text(json.dumps(preds))
    n_edge = sum(1 for p in preds if p.get("is_edge"))
    print(f"  Generated {len(preds)} predictions ({n_edge} with edge)")
    return preds, games


PROP_MAP = {"batter_hits": ("hitting", "hits"),
            "pitcher_strikeouts": ("pitching", "strikeOuts")}


def get_actual_stat(pid, group, field, target_date):
    data = get(f"{MLB}/people/{pid}/stats", stats="gameLog", group=group, season=SEASON)
    try:
        for sp in reversed(data["stats"][0]["splits"]):
            if sp.get("date") == target_date:
                return float(sp["stat"].get(field, 0) or 0)
    except: pass
    return None


def grade_picks(target_date, idx):
    preds = fetch_predictions_for(target_date)
    if not preds: return []
    results = []
    for pred in preds:
        if not isinstance(pred, dict): continue
        prop = pred.get("prop_type")
        if prop not in PROP_MAP: continue
        try: line = float(pred.get("pick", "").split()[-1])
        except: continue
        side = "OVER" if "OVER" in pred.get("pick", "").upper() else "UNDER"
        group, field = PROP_MAP[prop]
        pid = idx.get(_norm(pred.get("player", "")))
        if not pid: continue
        actual = get_actual_stat(pid, group, field, target_date)
        if actual is None: continue
        result = "hit" if (side == "OVER" and actual > line) or \
                          (side == "UNDER" and actual < line) else "miss"
        results.append({
            "date": target_date, "player": pred.get("player", ""),
            "team": pred.get("team", ""), "prop_type": prop,
            "pick": pred.get("pick"), "projected": pred.get("projected"),
            "actual": actual, "result": result,
            "had_line": pred.get("has_line", False),
            "value_edge": pred.get("value_edge"),
            "confidence": pred.get("confidence", ""),
        })
    return results


def update_record(new_results):
    path = DATA_DIR / "record.json"
    try:
        ed = json.loads(path.read_text()) if path.exists() else {}
        existing = ed.get("results", []) if isinstance(ed, dict) else []
        existing = [r for r in existing if isinstance(r, dict)]
    except: existing = []
    keys = {(r.get("date",""), r.get("player",""), r.get("prop_type","")) for r in existing}
    for r in new_results:
        k = (r.get("date",""), r.get("player",""), r.get("prop_type",""))
        if k not in keys: existing.append(r); keys.add(k)
    existing.sort(key=lambda r: r.get("date",""), reverse=True)
    total = len(existing); hits = sum(1 for r in existing if r.get("result") == "hit")
    hr = round(hits/total*100, 1) if total else 0
    by_prop = {}; by_conf = {}
    for r in existing:
        pt = r.get("prop_type","?"); by_prop.setdefault(pt, {"hits":0,"total":0})
        by_prop[pt]["total"] += 1; by_prop[pt]["hits"] += 1 if r.get("result")=="hit" else 0
        c = r.get("confidence","?"); by_conf.setdefault(c, {"hits":0,"total":0})
        by_conf[c]["total"] += 1; by_conf[c]["hits"] += 1 if r.get("result")=="hit" else 0
    record = {
        "summary": {"total":total,"hits":hits,"misses":total-hits,"hit_rate":hr},
        "by_prop": {k:{**v,"hit_rate":round(v["hits"]/v["total"]*100,1) if v["total"] else 0} for k,v in by_prop.items()},
        "by_confidence": {k:{**v,"hit_rate":round(v["hits"]/v["total"]*100,1) if v["total"] else 0} for k,v in by_conf.items()},
        "results": existing,
        "last_updated": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(record, indent=2))
    print(f"  Record: {hits}/{total} ({hr}%)")
    return record


def backfill_all_history(idx, days_back=20):
    today = dt.date.today(); all_new = []
    for i in range(1, days_back + 1):
        d = (today - dt.timedelta(days=i)).isoformat()
        all_new.extend(grade_picks(d, idx))
    if all_new: update_record(all_new)


_retrain_status = {"running": False, "last_run": None, "last_result": None}


def _do_weekly_update():
    _retrain_status["running"] = True
    _retrain_status["last_run"] = dt.datetime.now(dt.timezone.utc).isoformat()
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
    return {"status": "ok", "models_loaded": list(_models.keys()),
            "propline": bool(PROPLINE_KEY),
            "last_updated": dt.datetime.now(dt.timezone.utc).isoformat()}


@app.get("/predictions")
def predictions():
    today = dt.date.today().isoformat()
    path = PRED_DIR / f"predictions_{today}.json"
    if path.exists():
        data = json.loads(path.read_text())
        if data: return data
    preds, _ = run_predictions()
    return preds


@app.get("/games")
def games():
    return todays_games()


@app.post("/run/daily")
def trigger_daily():
    preds, games_list = run_predictions()
    return {"status": "completed", "predictions": len(preds), "games": len(games_list)}


@app.get("/run/now")
def run_now():
    preds, games_list = run_predictions()
    n_hits = sum(1 for p in preds if p["prop_type"] == "batter_hits")
    n_k = sum(1 for p in preds if p["prop_type"] == "pitcher_strikeouts")
    n_edge = sum(1 for p in preds if p.get("is_edge"))
    n_line = sum(1 for p in preds if p.get("has_line"))
    return {"status": "completed", "total": len(preds),
            "batter_hits": n_hits, "pitcher_strikeouts": n_k,
            "with_line": n_line, "with_edge": n_edge}


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
    return {"summary": {"total":0,"hits":0,"misses":0,"hit_rate":0},
            "by_prop": {}, "by_confidence": {}, "results": []}
