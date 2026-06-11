"""
api.py - FastAPI server on Render.
Self-sustaining ML prop system. Batter lineups now pulled from the live feed's
confirmed battingOrder (posts earlier/more reliably than the boxscore batters list).
"""
import os, json, math, time, glob, threading, datetime as dt
from pathlib import Path
from collections import defaultdict

import requests
import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import backfill

app = FastAPI(title="Prop Edge ML API", version="3.3")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

MLB = "https://statsapi.mlb.com/api/v1"
MLB11 = "https://statsapi.mlb.com/api/v1.1"
ODDS_KEY = os.environ.get("ODDS_API_KEY", "")
SEASON = dt.date.today().year
DATA_DIR = Path("/data")
MODEL_DIR = DATA_DIR / "models"
PRED_DIR = DATA_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
GH_PAGES_BASE = "https://deshawnclark116-prog.github.io/mlb-picks2"

S = requests.Session()
S.headers["User-Agent"] = "prop-edge/3.3"

STANDARD_LINE = {"batter_hits": 0.5, "pitcher_strikeouts": 4.5}

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
    """Pull confirmed batting order from the live feed. Returns
    {'home': [pid,...], 'away': [pid,...]} in batting-order sequence.
    This posts earlier and more reliably than the boxscore 'batters' list."""
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


def confidence(distance, gp):
    sc = 2 if distance >= 0.4 else 1 if distance >= 0.2 else 0
    sc += 1 if gp >= 20 else 0
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


def run_predictions():
    print(f"Running predictions {dt.datetime.now()}")
    idx = player_index()

    try:
        append_yesterday_to_season()
    except Exception as e:
        print(f"  append step error (non-fatal): {e}")

    backfill_all_history(idx, days_back=20)

    games = todays_games()
    preds = []

    # pitchers (probable starters - available early)
    for game in games:
        gid = game["game_id"]
        for side in ("home_pitcher", "away_pitcher"):
            pid = game.get(side)
            if not pid: continue
            feat = pitcher_feature_row(pid)
            if not feat: continue
            proj = model_predict("pitcher_strikeouts", feat)
            if proj is None: continue
            line = STANDARD_LINE["pitcher_strikeouts"]
            p_over = prob_over(proj, line)
            side_pick = "OVER" if proj > line else "UNDER"
            mp = p_over if side_pick == "OVER" else 1 - p_over
            if mp < 0.55: continue
            pdata = get(f"{MLB}/people/{pid}")
            name = pdata.get("people", [{}])[0].get("fullName", "")
            team = get_player_team(pid)
            opp = game["away_team"] if side == "home_pitcher" else game["home_team"]
            preds.append({
                "player": name, "team": team, "opponent": opp, "game_id": gid,
                "prop_type": "pitcher_strikeouts", "pick": f"{side_pick} {line}",
                "projected": round(proj, 2), "model_prob": round(mp, 3),
                "confidence": confidence(abs(proj - line), feat["starts"] * 5),
                "generated_at": dt.date.today().isoformat(),
            })

    # batters (confirmed lineup from live feed - posts ~3-4h pre-game)
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
                line = STANDARD_LINE["batter_hits"]
                p_over = prob_over(proj, line)
                if p_over < 0.55: continue
                pdata = get(f"{MLB}/people/{pid}")
                name = pdata.get("people", [{}])[0].get("fullName", "")
                preds.append({
                    "player": name, "team": team_name, "opponent": opp,
                    "game_id": gid, "prop_type": "batter_hits",
                    "pick": f"OVER {line}", "projected": round(proj, 2),
                    "model_prob": round(p_over, 3),
                    "confidence": confidence(abs(proj - line), feat["games_played"]),
                    "generated_at": dt.date.today().isoformat(),
                })
                count += 1

    preds.sort(key=lambda r: r["model_prob"], reverse=True)
    today = dt.date.today().isoformat()
    (PRED_DIR / f"predictions_{today}.json").write_text(json.dumps(preds))
    print(f"  Generated {len(preds)} predictions")
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
    """Same as daily but GET, so you can trigger from a browser on your phone."""
    preds, games_list = run_predictions()
    n_hits = sum(1 for p in preds if p["prop_type"] == "batter_hits")
    n_k = sum(1 for p in preds if p["prop_type"] == "pitcher_strikeouts")
    return {"status": "completed", "total": len(preds),
            "batter_hits": n_hits, "pitcher_strikeouts": n_k}


@app.get("/run/weekly")
def trigger_weekly():
    if _retrain_status["running"]:
        return {"status": "already_running", "last_run": _retrain_status["last_run"]}
    threading.Thread(target=_do_weekly_update, daemon=True).start()
    return {"status": "started", "message": "Retraining in background. Check /run/weekly/status."}


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
