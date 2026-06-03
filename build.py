#!/usr/bin/env python3
"""MLB prop value finder with automatic results tracking."""
import os, json, math, time, datetime as dt
import requests

MLB = "https://statsapi.mlb.com/api/v1"
ODDS = "https://api.the-odds-api.com/v4"
ODDS_KEY = os.environ.get("ODDS_API_KEY", "").strip()
SEASON = dt.date.today().year
MAX_GAMES = int(os.environ.get("MAX_GAMES", "15"))
MIN_EDGE = float(os.environ.get("MIN_EDGE", "0.05"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.55"))
MAX_PROPS_PER_GAME = 3
S = requests.Session()
S.headers["User-Agent"] = "prop-edge/1.0"

MARKETS = {
    "batter_hits":        ("hits",               "hitting",  "hits"),
    "pitcher_strikeouts": ("strikeouts_pitcher", "pitching", "strikeOuts"),
}

def poisson_cdf(k, lam):
    if lam <= 0: return 1.0
    s, term = 0.0, math.exp(-lam)
    for i in range(0, k + 1):
        if i: term *= lam / i
        s += term
    return min(s, 1.0)

def prob_over(expected, line, prop):
    if prop == "home_runs":
        return 1 - math.exp(-max(expected, 1e-6))
    return 1 - poisson_cdf(int(math.floor(line)), max(expected, 1e-6))

def american_to_prob(o):
    return (-o)/((-o)+100) if o < 0 else 100/(o+100)

def no_vig(over, under):
    a, b = american_to_prob(over), american_to_prob(under)
    t = a + b
    return (a/t, b/t) if t else (0.5, 0.5)

def kelly(p, o, cap=0.25):
    b = (o/100) if o > 0 else (100/-o)
    f = (b*p - (1-p))/b
    return max(0.0, min(f, cap))

def get(url, **params):
    for attempt in range(3):
        try:
            r = S.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print("  retry", url, e)
            time.sleep(1.5*(attempt+1))
    return {}

def _norm(s):
    return "".join(c for c in s.lower() if c.isalpha() or c == " ").strip()

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

def player_index():
    data = get(f"{MLB}/sports/1/players", season=SEASON)
    return {
        _norm(p.get("fullName", "")): p.get("id")
        for p in data.get("people", [])
    }

def get_player_team(pid):
    data = get(f"{MLB}/people/{pid}", hydrate="currentTeam")
    try:
        return data["people"][0]["currentTeam"]["name"]
    except Exception:
        return ""

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
            season_pg = total/starts if starts else None
        else:
            season_pg = total/gp if gp else None
    except Exception:
        pass
    recent_pg = None
    try:
        splits = g["stats"][0]["splits"][-n:]
        vals = [float(sp["stat"].get(field, 0) or 0) for sp in splits]
        if vals: recent_pg = sum(vals)/len(vals)
    except Exception:
        pass
    return season_pg, recent_pg, gp

def project(pid, group, field):
    s_pg, r_pg, gp = season_and_recent(pid, group, field)
    if s_pg is None and r_pg is None: return None, 0
    if r_pg is None: exp = s_pg
    elif s_pg is None: exp = r_pg
    else: exp = 0.6*r_pg + 0.4*s_pg
    return exp, gp

def fetch_odds():
    if not ODDS_KEY:
        print("No ODDS_API_KEY — games only.")
        return []
    ev = get(f"{ODDS}/sports/baseball_mlb/events", apiKey=ODDS_KEY)
    if not isinstance(ev, list): return []
    markets = ",".join(MARKETS.keys())
    out = []
    for e in ev[:MAX_GAMES]:
        d = get(
            f"{ODDS}/sports/baseball_mlb/events/{e['id']}/odds",
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
                    if not pid or pt is None or price is None or side not in ("over","under"):
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
    if not team_name:
        return "", "", ""
    norm = _norm(team_name)
    for g in games:
        if _norm(g["home_team"]) == norm:
            return g["home_team"], g["away_team"], g["game_id"]
        if _norm(g["away_team"]) == norm:
            return g["away_team"], g["home_team"], g["game_id"]
    return "", "", ""

def confidence(edge, prob, gp):
    sc = (2 if edge >= 0.10 else 1 if edge >= 0.05 else 0)
    sc += 1 if prob >= 0.62 else 0
    sc += 1 if gp >= 20 else 0
    return "HIGH" if sc >= 3 else "MEDIUM" if sc >= 2 else "LOW"

def get_yesterdays_predictions():
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    path = f"docs/predictions_{yesterday}.json"
    if not os.path.exists(path):
        path = "docs/predictions.json"
    try:
        data = json.load(open(path))
        if isinstance(data, list):
            return [p for p in data
                    if isinstance(p, dict) and p.get("generated_at") == yesterday]
    except Exception:
        pass
    return []

def get_player_stat_yesterday(pid, group, field):
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    data = get(f"{MLB}/people/{pid}/stats",
               stats="gameLog", group=group, season=SEASON)
    try:
        splits = data["stats"][0]["splits"]
        for sp in reversed(splits):
            if sp.get("date") == yesterday:
                return float(sp["stat"].get(field, 0) or 0)
    except Exception:
        pass
    return None

def grade_prediction(pred, idx):
    prop = pred.get("prop_type")
    pick = pred.get("pick", "")
    try:
        line = float(pick.split()[-1])
    except Exception:
        return None
    side = "OVER" if "OVER" in pick.upper() else "UNDER"
    prop_map = {
        "hits":               ("hitting",  "hits"),
        "strikeouts_pitcher": ("pitching", "strikeOuts"),
        "total_bases":        ("hitting",  "totalBases"),
        "home_runs":          ("hitting",  "homeRuns"),
    }
    if prop not in prop_map:
        return None
    group, field = prop_map[prop]
    pid = idx.get(_norm(pred.get("player", "")))
    if not pid:
        return None
    actual = get_player_stat_yesterday(pid, group, field)
    if actual is None:
        return None
    result = "hit" if (side == "OVER" and actual > line) or \
                      (side == "UNDER" and actual < line) else "miss"
    return {"result": result, "actual": actual, "line": line, "side": side}

def grade_yesterdays_picks(idx):
    preds = get_yesterdays_predictions()
    if not preds:
        print("  No yesterday predictions to grade.")
        return []
    print(f"  Grading {len(preds)} yesterday predictions...")
    results = []
    for pred in preds:
        if not isinstance(pred, dict):
            continue
        graded = grade_prediction(pred, idx)
        if graded is None:
            continue
        results.append({
            "date":       pred.get("generated_at", ""),
            "player":     pred.get("player", ""),
            "team":       pred.get("team", ""),
            "prop_type":  pred.get("prop_type", ""),
            "pick":       pred.get("pick", ""),
            "projected":  pred.get("projected"),
            "actual":     graded["actual"],
            "result":     graded["result"],
            "confidence": pred.get("confidence", ""),
            "value_edge": pred.get("value_edge"),
        })
    hits = sum(1 for r in results if r["result"] == "hit")
    total = len(results)
    if total:
        print(f"  Graded: {hits}/{total} hits ({round(hits/total*100)}%)")
    return results

def update_record(new_results):
    path = "docs/record.json"
    try:
        existing_data = json.load(open(path)) if os.path.exists(path) else {}
        existing = existing_data.get("results", []) if isinstance(existing_data, dict) else existing_data
        existing = [r for r in existing if isinstance(r, dict)]
    except Exception:
        existing = []

    existing_keys = {
        (r.get("date",""), r.get("player",""), r.get("prop_type",""))
        for r in existing
    }
    added = 0
    for r in new_results:
        if not isinstance(r, dict): continue
        key = (r.get("date",""), r.get("player",""), r.get("prop_type",""))
        if key not in existing_keys:
            existing.append(r)
            added += 1

    existing.sort(key=lambda r: r.get("date",""), reverse=True)
    total = len(existing)
    hits  = sum(1 for r in existing if r.get("result") == "hit")
    hit_rate = round(hits/total*100, 1) if total else 0

    by_prop = {}
    for r in existing:
        pt = r.get("prop_type","unknown")
        by_prop.setdefault(pt, {"hits":0,"total":0})
        by_prop[pt]["total"] += 1
        if r.get("result") == "hit":
            by_prop[pt]["hits"] += 1
    prop_breakdown = {
        pt: {"hits":v["hits"],"total":v["total"],
             "hit_rate":round(v["hits"]/v["total"]*100,1) if v["total"] else 0}
        for pt,v in by_prop.items()
    }

    by_conf = {}
    for r in existing:
        c = r.get("confidence","UNKNOWN")
        by_conf.setdefault(c, {"hits":0,"total":0})
        by_conf[c]["total"] += 1
        if r.get("result") == "hit":
            by_conf[c]["hits"] += 1
    conf_breakdown = {
        c: {"hits":v["hits"],"total":v["total"],
            "hit_rate":round(v["hits"]/v["total"]*100,1) if v["total"] else 0}
        for c,v in by_conf.items()
    }

    record = {
        "summary":      {"total":total,"hits":hits,
                         "misses":total-hits,"hit_rate":hit_rate},
        "by_prop":       prop_breakdown,
        "by_confidence": conf_breakdown,
        "results":       existing,
        "last_updated":  dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    json.dump(record, open(path,"w"), indent=2)
    print(f"  Record: {hits}/{total} ({hit_rate}%) — added {added} new")
    return record

def main():
    os.makedirs("docs", exist_ok=True)

    print("Grading yesterday's picks...")
    idx = player_index()
    new_results = grade_yesterdays_picks(idx)
    if new_results:
        update_record(new_results)
    elif not os.path.exists("docs/record.json"):
        json.dump({
            "summary":      {"total":0,"hits":0,"misses":0,"hit_rate":0},
            "by_prop":      {},"by_confidence":{},"results":[],
            "last_updated": dt.datetime.now(dt.timezone.utc).isoformat(),
        }, open("docs/record.json","w"), indent=2)

    print("Fetching games...")
    games = todays_games()
    print(f"  {len(games)} games")
    print("Fetching odds...")
    events = fetch_odds()
    market = parse_market(events, idx) if events else {}
    print(f"  {len(market)} prop markets")

    candidates = []
    for (pid, prop, odds_game_id), mk in market.items():
        group = next(v[1] for v in MARKETS.values() if v[0]==prop)
        field = next(v[2] for v in MARKETS.values() if v[0]==prop)
        exp, gp = project(pid, group, field)
        if exp is None: continue

        p_over = prob_over(exp, mk["line"], prop)
        fo, fu = no_vig(mk["over"], mk["under"])
        e_over  = (p_over-fo)/fo       if fo else 0
        e_under = ((1-p_over)-fu)/fu   if fu else 0
        if e_over >= e_under:
            side,mp,fp,odds,edge = "OVER", p_over,   fo, mk["over"],  e_over
        else:
            side,mp,fp,odds,edge = "UNDER",1-p_over, fu, mk["under"], e_under

        if edge < MIN_EDGE or mp < MIN_PROB: continue

        # Only allow OVER on hits — unders not available at most books
        if prop == "hits" and side == "UNDER": continue

        team_name = get_player_team(pid)
        team, opponent, matched_game_id = match_game(team_name, games)
        if not team or not opponent:
            print(f"  Skipping {mk['name']} — could not match to a game")
            continue

        candidates.append({
            "player":         mk["name"],
            "team":           team,
            "opponent":       opponent,
            "game_id":        matched_game_id,
            "prop_type":      prop,
            "pick":           f"{side} {mk['line']}",
            "projected":      round(exp,2),
            "model_prob":     round(mp,3),
            "fair_prob":      round(fp,3),
            "odds":           odds,
            "value_edge":     round(edge,3),
            "kelly_fraction": round(kelly(mp,odds),4),
            "confidence":     confidence(edge,mp,gp),
            "generated_at":   dt.date.today().isoformat(),
        })

    candidates.sort(key=lambda r: r["value_edge"], reverse=True)

    game_counts = {}
    preds = []
    for c in candidates:
        gid = c["game_id"]
        if game_counts.get(gid,0) >= MAX_PROPS_PER_GAME: continue
        game_counts[gid] = game_counts.get(gid,0) + 1
        preds.append(c)

    today = dt.date.today().isoformat()
    json.dump(preds, open(f"docs/predictions_{today}.json","w"), indent=2)
    json.dump(preds, open("docs/predictions.json","w"),          indent=2)
    json.dump(games, open("docs/games.json","w"),                indent=2)
    json.dump({
        "status":           "ok",
        "predictions_today": len(preds),
        "games_today":       len(games),
        "last_updated":      dt.datetime.now(dt.timezone.utc).isoformat(),
        "date":              today,
    }, open("docs/health.json","w"), indent=2)

    print(f"Wrote {len(preds)} predictions across {len(game_counts)} games.")

if __name__ == "__main__":
    main()
