#!/usr/bin/env python3
"""MLB prop value finder. Real data, no training step, no synthetic data."""
import os, json, math, time, datetime as dt
import requests

MLB = "https://statsapi.mlb.com/api/v1"
ODDS = "https://api.the-odds-api.com/v4"
ODDS_KEY = os.environ.get("ODDS_API_KEY", "").strip()
SEASON = dt.date.today().year
MAX_GAMES = int(os.environ.get("MAX_GAMES", "5"))
MIN_EDGE = float(os.environ.get("MIN_EDGE", "0.05"))
MIN_PROB = float(os.environ.get("MIN_PROB", "0.55"))
S = requests.Session()
S.headers["User-Agent"] = "prop-edge/1.0"

MARKETS = {
    "batter_hits":        ("hits",               "hitting",  "hits"),
    "batter_total_bases": ("total_bases",        "hitting",  "totalBases"),
    "batter_home_runs":   ("home_runs",          "hitting",  "homeRuns"),
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
            r = S.get(url, params=params, timeout=20); r.raise_for_status()
            return r.json()
        except Exception as e:
            print("  retry", url, e); time.sleep(1.5*(attempt+1))
    return {}

def _norm(s):
    return "".join(c for c in s.lower() if c.isalpha() or c == " ").strip()

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
                "game_time": g.get("gameDate"),
                "status": g.get("status", {}).get("abstractGameState", "").lower(),
            })
    return out

def player_index():
    data = get(f"{MLB}/sports/1/players", season=SEASON)
    return {_norm(p.get("fullName", "")): p.get("id") for p in data.get("people", [])}

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
        print("No ODDS_API_KEY — games only."); return []
    ev = get(f"{ODDS}/sports/baseball_mlb/events", apiKey=ODDS_KEY)
    if not isinstance(ev, list): return []
    markets = ",".join(MARKETS.keys())
    out = []
    for e in ev[:MAX_GAMES]:
        d = get(f"{ODDS}/sports/baseball_mlb/events/{e['id']}/odds",
                apiKey=ODDS_KEY, regions="us", markets=markets, oddsFormat="american")
        if d: out.append(d)
    return out

def parse_market(events, idx):
    m = {}
    for ev in events:
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
                    cur = m.setdefault((pid, prop), {"line":pt,"over":None,"under":None,"name":name})
                    f = "over" if side == "over" else "under"
                    if cur[f] is None or price > cur[f]:
                        cur[f] = price; cur["line"] = pt
    return {k:v for k,v in m.items() if v["over"] is not None and v["under"] is not None}

def confidence(edge, prob, gp):
    sc = (2 if edge>=0.10 else 1 if edge>=0.05 else 0)
    sc += 1 if prob>=0.62 else 0
    sc += 1 if gp>=20 else 0
    return "HIGH" if sc>=3 else "MEDIUM" if sc>=2 else "LOW"

def main():
    os.makedirs("docs", exist_ok=True)
    print("Fetching games..."); games = todays_games(); print(f"  {len(games)} games")
    print("Fetching odds..."); events = fetch_odds()
    idx = player_index() if events else {}
    market = parse_market(events, idx) if events else {}
    print(f"  {len(market)} prop markets")
    preds = []
    for (pid, prop), mk in market.items():
        group = next(v[1] for v in MARKETS.values() if v[0]==prop)
        field = next(v[2] for v in MARKETS.values() if v[0]==prop)
        exp, gp = project(pid, group, field)
        if exp is None: continue
        p_over = prob_over(exp, mk["line"], prop)
        fo, fu = no_vig(mk["over"], mk["under"])
        e_over = (p_over-fo)/fo if fo else 0
        e_under = ((1-p_over)-fu)/fu if fu else 0
        if e_over >= e_under:
            side, mp, fp, odds, edge = "OVER", p_over, fo, mk["over"], e_over
        else:
            side, mp, fp, odds, edge = "UNDER", 1-p_over, fu, mk["under"], e_under
        if edge < MIN_EDGE or mp < MIN_PROB: continue
        preds.append({
            "player": mk["name"], "prop_type": prop, "pick": f"{side} {mk['line']}",
            "projected": round(exp,2), "model_prob": round(mp,3), "fair_prob": round(fp,3),
            "odds": odds, "value_edge": round(edge,3),
            "kelly_fraction": round(kelly(mp,odds),4),
            "confidence": confidence(edge,mp,gp),
            "generated_at": dt.date.today().isoformat(),
        })
    preds.sort(key=lambda r:r["value_edge"], reverse=True)
    health = {"status":"ok","predictions_today":len(preds),"games_today":len(games),
              "last_updated":dt.datetime.now(dt.timezone.utc).isoformat(),
              "date":dt.date.today().isoformat()}
    json.dump(preds, open("docs/predictions.json","w"), indent=2)
    json.dump(games, open("docs/games.json","w"), indent=2)
    json.dump(health, open("docs/health.json","w"), indent=2)
    print(f"Wrote {len(preds)} predictions, {len(games)} games.")

if __name__ == "__main__":
    main()
