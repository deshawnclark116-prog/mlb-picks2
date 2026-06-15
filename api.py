"""
api.py - Prop Edge full system.
GRADING FIX: only grades games >=1 day old (final), and re-grades the last 3
days every run so any pick caught mid-game gets corrected with final stats.
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
import gamelines

app = FastAPI(title="Prop Edge ML API", version="6.2")
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
S.headers["User-Agent"] = "prop-edge/6.2"

STANDARD_LINE = {"batter_hits": 0.5, "pitcher_strikeouts": 4.5, "batter_total_bases": 1.5}
MIN_EDGE = 0.05
REGRADE_DAYS = 3  # re-grade the last N days each run to fix stale grades

PROP_MODEL = {
    "batter_hits": "batter_hits",
    "pitcher_strikeouts": "pitcher_strikeouts",
    "batter_total_bases": "batter_total_bases",
    "batter_rbis": "batter_rbi",
    "batter_runs": "batter_runs",
}

_models = {}

def load_models():
    _models.clear()
    for name in ("batter_hits", "pitcher_strikeouts", "batter_total_bases",
                 "batter_rbi", "batter_runs"):
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


def prob_at_least(expected, threshold):
    return 1 - poisson_cdf(threshold - 1, max(expected, 1e-6))


def american_to_prob(odds):
    if odds < 0: return -odds / (-odds + 100)
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


def _is_real_game(ev):
    h = ev.get("home_team", "")
    return "(" not in h and "Runs" not in h


def _parse_threshold(name):
    try:
        if "+" in name:
            return int(name.split("+")[0].strip())
    except:
        pass
    return None


def fetch_propline_props():
    over_under = {}
    thresholds = defaultdict(dict)
    if not PROPLINE_KEY:
        print("  No PROPLINE_API_KEY — projection-only")
        return over_under, thresholds
    try:
        events = get(f"{PROPLINE_BASE}/events", apiKey=PROPLINE_KEY)
        if not isinstance(events, list):
            return over_under, thresholds
    except Exception as e:
        print(f"  PropLine events failed: {e}")
        return over_under, thresholds
    today = dt.date.today().isoformat()
    prop_keys = "batter_hits,pitcher_strikeouts,batter_total_bases,batter_rbis,batter_runs"
    pulled = 0
    for ev in events:
        if not _is_real_game(ev): continue
        if not str(ev.get("commence_time", "")).startswith(today): continue
        eid = ev.get("id")
        if not eid: continue
        try:
            data = get(f"{PROPLINE_BASE}/events/{eid}/odds", apiKey=PROPLINE_KEY,
                       markets=prop_keys, regions="us")
        except Exception:
            continue
        for book in (data.get("bookmakers") or []):
            for mkt in book.get("markets", []):
                mkey = mkt.get("key")
                ou = defaultdict(dict)
                for o in mkt.get("outcomes", []):
                    player = _norm(o.get("description", ""))
                    name = o.get("name", "")
                    price = o.get("price"); point = o.get("point")
                    low = name.lower()
                    if low in ("over", "under"):
                        ou[player][low] = {"price": price, "point": point}
                    else:
                        thr = _parse_threshold(name)
                        if thr is not None:
                            key = (player, mkey)
                            if thr not in thresholds[key]:
                                thresholds[key][thr] = price
                for player, sides in ou.items():
                    if "over" in sides and "under" in sides:
                        key = (player, mkey)
                        if key not in over_under:
                            over_under[key] = {
                                "line": sides["over"]["point"],
                                "over_odds": sides["over"]["price"],
                                "under_odds": sides["under"]["price"],
                            }
        pulled += 1
        time.sleep(0.2)
    print(f"  PropLine: props for {pulled} games "
          f"({len(over_under)} O/U, {len(thresholds)} threshold sets)")
    return over_under, thresholds


def fetch_propline_gamelines():
    out = {}
    if not PROPLINE_KEY: return out
    try:
        events = get(f"{PROPLINE_BASE}/events", apiKey=PROPLINE_KEY)
        if not isinstance(events, list): return out
    except Exception as e:
        print(f"  PropLine GL events failed: {e}")
        return out
    today = dt.date.today().isoformat()
    mlb_games = {(_norm(g["home_team"]), _norm(g["away_team"])): g["game_id"]
                 for g in todays_games()}
    for ev in events:
        if not _is_real_game(ev): continue
        if not str(ev.get("commence_time", "")).startswith(today): continue
        eid = ev.get("id")
        if not eid: continue
        home_name = ev.get("home_team", ""); away_name = ev.get("away_team", "")
        gid = mlb_games.get((_norm(home_name), _norm(away_name)))
        if not gid: continue
        try:
            data = get(f"{PROPLINE_BASE}/events/{eid}/odds", apiKey=PROPLINE_KEY,
                       markets="h2h,totals,spreads", regions="us")
        except Exception:
            continue
        entry = {"home_team": home_name, "away_team": away_name}
        for book in (data.get("bookmakers") or []):
            for mkt in book.get("markets", []):
                k = mkt.get("key"); outs = mkt.get("outcomes", [])
                if k == "h2h" and "h2h" not in entry:
                    od = {_norm(o.get("name","")): o.get("price") for o in outs}
                    ho, ao = od.get(_norm(home_name)), od.get(_norm(away_name))
                    if ho is not None and ao is not None:
                        entry["h2h"] = {"home_odds": ho, "away_odds": ao}
                elif k == "totals" and "totals" not in entry:
                    over = next((o for o in outs if o.get("name","").lower()=="over"), None)
                    under = next((o for o in outs if o.get("name","").lower()=="under"), None)
                    if over and under:
                        entry["totals"] = {"line": over.get("point"),
                                           "over_odds": over.get("price"),
                                           "under_odds": under.get("price")}
                elif k == "spreads" and "spreads" not in entry:
                    sp = {}
                    for o in outs:
                        sp[_norm(o.get("name",""))] = {"point": o.get("point"), "price": o.get("price")}
                    h = sp.get(_norm(home_name)); a = sp.get(_norm(away_name))
                    if h and a:
                        entry["spreads"] = {"home": h, "away": a}
        if any(x in entry for x in ("h2h","totals","spreads")):
            out[gid] = entry
        time.sleep(0.2)
    print(f"  PropLine: game lines for {len(out)} games")
    return out


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
    cum_h = cum_ab = cum_pa = cum_hr = cum_bb = cum_so = cum_tb = cum_rbi = cum_runs = 0
    rec_h = []; rec_tb = []; rec_rbi = []; rec_runs = []
    for sp in splits:
        st = sp["stat"]
        h = int(st.get("hits", 0) or 0); tb = int(st.get("totalBases", 0) or 0)
        rbi = int(st.get("rbi", 0) or 0); runs = int(st.get("runs", 0) or 0)
        cum_h += h; cum_ab += int(st.get("atBats", 0) or 0)
        cum_pa += int(st.get("plateAppearances", 0) or 0)
        cum_hr += int(st.get("homeRuns", 0) or 0); cum_bb += int(st.get("baseOnBalls", 0) or 0)
        cum_so += int(st.get("strikeOuts", 0) or 0)
        cum_tb += tb; cum_rbi += rbi; cum_runs += runs
        rec_h.append(h); rec_tb.append(tb); rec_rbi.append(rbi); rec_runs.append(runs)
    if cum_ab < 20 or len(rec_h) < 5: return None
    return {
        "season_avg": cum_h / cum_ab if cum_ab else 0,
        "recent15_avg": sum(rec_h[-15:]) / len(rec_h[-15:]),
        "recent5_avg": sum(rec_h[-5:]) / len(rec_h[-5:]),
        "hr_rate": cum_hr / cum_pa if cum_pa else 0,
        "bb_rate": cum_bb / cum_pa if cum_pa else 0,
        "so_rate": cum_so / cum_pa if cum_pa else 0,
        "batting_order": 9, "games_played": len(rec_h),
        "tb_per_pa": cum_tb / cum_pa if cum_pa else 0,
        "rbi_per_pa": cum_rbi / cum_pa if cum_pa else 0,
        "runs_per_pa": cum_runs / cum_pa if cum_pa else 0,
        "recent5_target": 0, "recent15_target": 0,
        "_rec_tb": rec_tb, "_rec_rbi": rec_rbi, "_rec_runs": rec_runs,
    }


def _batter_feat_for(prop, base):
    f = dict(base)
    if prop == "batter_total_bases": rec = base["_rec_tb"]
    elif prop == "batter_rbis": rec = base["_rec_rbi"]
    elif prop == "batter_runs": rec = base["_rec_runs"]
    else: rec = None
    if rec is not None:
        f["recent5_target"] = sum(rec[-5:]) / len(rec[-5:]) if rec else 0
        f["recent15_target"] = sum(rec[-15:]) / len(rec[-15:]) if rec else 0
    f.pop("_rec_tb", None); f.pop("_rec_rbi", None); f.pop("_rec_runs", None)
    return f


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


def conf_from_edge(edge):
    if edge is None: return "LOW"
    return "HIGH" if edge >= 0.08 else "MEDIUM" if edge >= 0.04 else "LOW"


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
        print(f"  {yesterday} already recorded, skipping"); return
    games = backfill.get_schedule(yesterday)
    if not games:
        print(f"  No final games for {yesterday}"); return
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


def _pick(name, team, opp, gid, prop, pick_str, proj, mp, fair_p, odds, edge):
    has_line = odds is not None
    return {
        "player": name, "team": team, "opponent": opp, "game_id": gid,
        "prop_type": prop, "pick": pick_str,
        "projected": round(proj, 2) if proj is not None else None,
        "model_prob": round(mp, 3) if mp is not None else None,
        "fair_prob": round(fair_p, 3) if fair_p is not None else None,
        "odds": odds,
        "value_edge": round(edge, 3) if edge is not None else None,
        "kelly": round(kelly_fraction(mp, odds), 4) if (mp is not None and odds is not None) else None,
        "has_line": has_line,
        "is_edge": (edge is not None and edge >= MIN_EDGE),
        "confidence": conf_from_edge(edge),
        "generated_at": dt.date.today().isoformat(),
    }


def build_batter_prop_picks(name, team, opp, gid, base_feat, over_under, thresholds):
    picks = []
    nrm = _norm(name)
    for prop in ("batter_hits", "batter_total_bases", "batter_rbis", "batter_runs"):
        model_name = PROP_MODEL[prop]
        feat = _batter_feat_for(prop, base_feat) if prop != "batter_hits" else base_feat
        proj = model_predict(model_name, feat)
        if proj is None: continue
        ou = over_under.get((nrm, prop))
        if ou and ou.get("over_odds") is not None and ou.get("under_odds") is not None:
            line = ou["line"]; p_over = prob_over(proj, line)
            fo, fu = no_vig_two_way(ou["over_odds"], ou["under_odds"])
            edge = value_edge(p_over, fo)
            if p_over >= 0.55 or edge >= MIN_EDGE:
                picks.append(_pick(name, team, opp, gid, prop, f"OVER {line}",
                                   proj, p_over, fo, ou["over_odds"], edge))
        thr = thresholds.get((nrm, prop))
        if thr:
            for t in (1, 2):
                if t not in thr: continue
                price = thr[t]; p_yes = prob_at_least(proj, t)
                fair = american_to_prob(price)
                edge = value_edge(p_yes, fair)
                if p_yes >= 0.55 or edge >= MIN_EDGE:
                    label = {"batter_total_bases": "Total Bases",
                             "batter_rbis": "RBIs", "batter_runs": "Runs"}[prop]
                    picks.append(_pick(name, team, opp, gid, prop, f"{t}+ {label}",
                                       proj, p_yes, fair, price, edge))
    return picks


def build_gameline_picks(games, gl_market, run_table):
    picks = []
    for game in games:
        gid = game["game_id"]; home, away = game["home_team"], game["away_team"]
        gl = gl_market.get(gid)
        if not gl: continue
        if "h2h" in gl:
            probs = gamelines.moneyline_prob(home, away, run_table)
            if probs:
                hp, ap = probs
                fh, fa = gamelines.no_vig_two_way(gl["h2h"]["home_odds"], gl["h2h"]["away_odds"])
                eh, ea = gamelines.value_edge(hp, fh), gamelines.value_edge(ap, fa)
                if eh >= ea: team, mp, fp, od, ed = home, hp, fh, gl["h2h"]["home_odds"], eh
                else: team, mp, fp, od, ed = away, ap, fa, gl["h2h"]["away_odds"], ea
                if mp >= 0.5 or ed >= MIN_EDGE:
                    picks.append({"player": team, "team": team,
                        "opponent": away if team==home else home, "game_id": gid,
                        "prop_type": "moneyline", "pick": f"{team} ML",
                        "projected": round(mp,3), "model_prob": round(mp,3),
                        "fair_prob": round(fp,3), "odds": od, "value_edge": round(ed,3),
                        "kelly": round(gamelines.kelly_fraction(mp,od),4),
                        "has_line": True, "is_edge": ed >= MIN_EDGE,
                        "confidence": conf_from_edge(ed),
                        "generated_at": dt.date.today().isoformat()})
        if "totals" in gl:
            et = gamelines.total_runs(home, away, run_table)
            if et:
                line = gl["totals"]["line"]; po = gamelines.prob_total_over(et, line)
                fo, fu = gamelines.no_vig_two_way(gl["totals"]["over_odds"], gl["totals"]["under_odds"])
                eo, eu = gamelines.value_edge(po, fo), gamelines.value_edge(1-po, fu)
                if eo >= eu: side, mp, fp, od, ed = "OVER", po, fo, gl["totals"]["over_odds"], eo
                else: side, mp, fp, od, ed = "UNDER", 1-po, fu, gl["totals"]["under_odds"], eu
                if mp >= 0.5 or ed >= MIN_EDGE:
                    picks.append({"player": f"{away} @ {home}", "team": f"{away} @ {home}",
                        "opponent": "", "game_id": gid, "prop_type": "total",
                        "pick": f"{side} {line}", "projected": round(et,2),
                        "model_prob": round(mp,3), "fair_prob": round(fp,3), "odds": od,
                        "value_edge": round(ed,3), "kelly": round(gamelines.kelly_fraction(mp,od),4),
                        "has_line": True, "is_edge": ed >= MIN_EDGE,
                        "confidence": conf_from_edge(ed),
                        "generated_at": dt.date.today().isoformat()})
        if "spreads" in gl:
            rl = gamelines.run_line_prob(home, away, run_table)
            if rl:
                hc, ac = rl; hsp, asp = gl["spreads"]["home"], gl["spreads"]["away"]
                fh, fa = gamelines.no_vig_two_way(hsp["price"], asp["price"])
                eh, ea = gamelines.value_edge(hc, fh), gamelines.value_edge(ac, fa)
                if eh >= ea: team, mp, fp, od, ed, pt = home, hc, fh, hsp["price"], eh, hsp["point"]
                else: team, mp, fp, od, ed, pt = away, ac, fa, asp["price"], ea, asp["point"]
                if ed >= MIN_EDGE:
                    picks.append({"player": team, "team": team,
                        "opponent": away if team==home else home, "game_id": gid,
                        "prop_type": "run_line", "pick": f"{team} {pt:+g}",
                        "projected": round(mp,3), "model_prob": round(mp,3),
                        "fair_prob": round(fp,3), "odds": od, "value_edge": round(ed,3),
                        "kelly": round(gamelines.kelly_fraction(mp,od),4),
                        "has_line": True, "is_edge": ed >= MIN_EDGE,
                        "confidence": conf_from_edge(ed),
                        "generated_at": dt.date.today().isoformat()})
    return picks


def run_predictions():
    print(f"Running predictions {dt.datetime.now()}")
    idx = player_index()
    try:
        append_yesterday_to_season()
    except Exception as e:
        print(f"  append error (non-fatal): {e}")
    backfill_all_history(idx, days_back=20)

    over_under, thresholds = fetch_propline_props()
    gl_market = fetch_propline_gamelines()
    run_table = gamelines.team_run_table()
    games = todays_games()
    preds = []
    preds.extend(build_gameline_picks(games, gl_market, run_table))

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
            ou = over_under.get((_norm(name), "pitcher_strikeouts"))
            if ou and ou.get("over_odds") is not None and ou.get("under_odds") is not None:
                line = ou["line"]; p_over = prob_over(proj, line)
                fo, fu = no_vig_two_way(ou["over_odds"], ou["under_odds"])
                eo, eu = value_edge(p_over, fo), value_edge(1-p_over, fu)
                if eo >= eu: side2, mp, fp, od, ed = "OVER", p_over, fo, ou["over_odds"], eo
                else: side2, mp, fp, od, ed = "UNDER", 1-p_over, fu, ou["under_odds"], eu
                if mp >= 0.55 or ed >= MIN_EDGE:
                    preds.append(_pick(name, team, opp, gid, "pitcher_strikeouts",
                                       f"{side2} {line}", proj, mp, fp, od, ed))
            else:
                line = STANDARD_LINE["pitcher_strikeouts"]; p_over = prob_over(proj, line)
                side2 = "OVER" if proj > line else "UNDER"
                mp = p_over if side2 == "OVER" else 1 - p_over
                if mp >= 0.55:
                    preds.append(_pick(name, team, opp, gid, "pitcher_strikeouts",
                                       f"{side2} {line}", proj, mp, None, None, None))

    for game in games:
        gid = game["game_id"]
        lineup = get_confirmed_lineup(gid)
        for tside in ("home", "away"):
            team_name = game["home_team"] if tside == "home" else game["away_team"]
            opp = game["away_team"] if tside == "home" else game["home_team"]
            count = 0
            for pid in lineup.get(tside, []):
                if count >= 5: break
                base = batter_feature_row(pid)
                if not base: continue
                pdata = get(f"{MLB}/people/{pid}")
                name = pdata.get("people", [{}])[0].get("fullName", "")
                pks = build_batter_prop_picks(name, team_name, opp, gid, base,
                                              over_under, thresholds)
                if pks:
                    preds.extend(pks); count += 1

    preds.sort(key=lambda r: (r.get("is_edge", False),
                              r.get("value_edge") or 0,
                              r.get("model_prob") or 0), reverse=True)
    today = dt.date.today().isoformat()
    (PRED_DIR / f"predictions_{today}.json").write_text(json.dumps(preds))
    byt = {}
    for p in preds: byt[p["prop_type"]] = byt.get(p["prop_type"], 0) + 1
    print(f"  Generated {len(preds)} predictions {byt}")
    return preds, games


PROP_STAT = {
    "batter_hits": ("hitting", "hits"),
    "pitcher_strikeouts": ("pitching", "strikeOuts"),
    "batter_total_bases": ("hitting", "totalBases"),
    "batter_rbis": ("hitting", "rbi"),
    "batter_runs": ("hitting", "runs"),
}


def get_actual_stat(pid, group, field, target_date):
    data = get(f"{MLB}/people/{pid}/stats", stats="gameLog", group=group, season=SEASON)
    try:
        for sp in reversed(data["stats"][0]["splits"]):
            if sp.get("date") == target_date:
                return float(sp["stat"].get(field, 0) or 0)
    except: pass
    return None


def grade_picks(target_date, idx):
    # Only grade games that are definitely final: target date must be in the past.
    if target_date >= dt.date.today().isoformat():
        return []
    preds = fetch_predictions_for(target_date)
    if not preds: return []
    results = []
    for pred in preds:
        if not isinstance(pred, dict): continue
        prop = pred.get("prop_type")
        if prop in ("moneyline", "total", "run_line"):
            continue
        if prop not in PROP_STAT: continue
        group, field = PROP_STAT[prop]
        pid = idx.get(_norm(pred.get("player", "")))
        if not pid: continue
        actual = get_actual_stat(pid, group, field, target_date)
        if actual is None: continue
        pick = pred.get("pick", "")
        result = None
        if "+" in pick:
            try:
                thr = int(pick.split("+")[0].strip())
                result = "hit" if actual >= thr else "miss"
            except: pass
        elif pick.upper().startswith("OVER"):
            try:
                line = float(pick.split()[-1])
                result = "hit" if actual > line else "miss"
            except: pass
        elif pick.upper().startswith("UNDER"):
            try:
                line = float(pick.split()[-1])
                result = "hit" if actual < line else "miss"
            except: pass
        if result is None: continue
        results.append({
            "date": target_date, "player": pred.get("player", ""),
            "team": pred.get("team", ""), "prop_type": prop,
            "pick": pick, "projected": pred.get("projected"),
            "actual": actual, "result": result,
            "had_line": pred.get("has_line", False),
            "value_edge": pred.get("value_edge"),
            "confidence": pred.get("confidence", ""),
        })
    return results


def update_record(new_results, regrade_dates=None):
    """Add new results. For dates in regrade_dates, REPLACE existing entries
    (fixes stale in-progress grades). Older entries stay frozen."""
    path = DATA_DIR / "record.json"
    regrade_dates = set(regrade_dates or [])
    try:
        ed = json.loads(path.read_text()) if path.exists() else {}
        existing = ed.get("results", []) if isinstance(ed, dict) else []
        existing = [r for r in existing if isinstance(r, dict)]
    except: existing = []

    # drop any existing entries on regrade dates — they'll be re-added fresh
    if regrade_dates:
        existing = [r for r in existing if r.get("date") not in regrade_dates]

    keys = {(r.get("date",""), r.get("player",""), r.get("prop_type",""), r.get("pick","")) for r in existing}
    for r in new_results:
        k = (r.get("date",""), r.get("player",""), r.get("prop_type",""), r.get("pick",""))
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
    today = dt.date.today()
    # recent days get re-graded (replace stale); older days only fill gaps
    regrade_dates = [(today - dt.timedelta(days=i)).isoformat() for i in range(1, REGRADE_DAYS + 1)]
    all_new = []
    for i in range(1, days_back + 1):
        d = (today - dt.timedelta(days=i)).isoformat()
        all_new.extend(grade_picks(d, idx))
    if all_new:
        update_record(all_new, regrade_dates=regrade_dates)


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
    byt = {}
    for p in preds: byt[p["prop_type"]] = byt.get(p["prop_type"], 0) + 1
    n_edge = sum(1 for p in preds if p.get("is_edge"))
    return {"status": "completed", "total": len(preds), "by_type": byt, "with_edge": n_edge}


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
