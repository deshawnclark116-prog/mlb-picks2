import urllib.request, json, datetime as dt
from zoneinfo import ZoneInfo

def get(u):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(u, headers={'User-Agent': 'x'}), timeout=30).read())

def rates(bid, pid):
    sp = get(f'https://statsapi.mlb.com/api/v1/people/{bid}/stats?stats=season&group=hitting&season=2026')
    gen = None
    try:
        st = sp['stats'][0]['splits'][0]['stat']
        ab = int(st.get('atBats', 0) or 0)
        h = int(st.get('hits', 0) or 0)
        if ab >= 20:
            gen = h / ab
    except Exception:
        pass
    d = get(f'https://statsapi.mlb.com/api/v1/people/{bid}/stats?stats=vsPlayer&opposingPlayerId={pid}&group=hitting')
    hab = 0; hh = 0
    try:
        for s in d['stats'][0]['splits']:
            if s.get('stat'):
                st = s['stat']
                hab += int(st.get('atBats', 0) or 0)
                hh += int(st.get('hits', 0) or 0)
    except Exception:
        pass
    h2h = hh / hab if hab >= 4 else None
    return gen, h2h

def games_in_window(start_day, end_day):
    today = dt.datetime.now(ZoneInfo('America/New_York')).date()
    gpks = []
    for i in range(start_day, end_day):
        d = (today - dt.timedelta(days=i)).isoformat()
        s = get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}')
        for day in s.get('dates', []):
            for g in day.get('games', []):
                if g.get('status', {}).get('abstractGameState') == 'Final':
                    gpks.append(g['gamePk'])
    return gpks

def collect(gpks):
    rows = []
    for gpk in gpks:
        f = get(f'https://statsapi.mlb.com/api/v1.1/game/{gpk}/feed/live')
        try:
            for ts, os_ in [('home', 'away'), ('away', 'home')]:
                op = f['gameData'].get('probablePitchers', {}).get(os_, {}).get('id')
                if not op:
                    continue
                order = f['liveData']['boxscore']['teams'][ts].get('battingOrder', [])[:9]
                box = f['liveData']['boxscore']['teams'][ts].get('players', {})
                for bid in order:
                    gen, h2h = rates(bid, op)
                    if gen is None or h2h is None:
                        continue
                    pd = box.get(f'ID{bid}', {})
                    ah = pd.get('stats', {}).get('batting', {}).get('hits', None)
                    if ah is None:
                        continue
                    got = 1 if ah >= 1 else 0
                    rows.append((gen, h2h, got))
        except Exception:
            pass
    return rows

def separation(rows, scorer):
    scored = [(scorer(gen, h2h), got) for gen, h2h, got in rows]
    hit = [s for s, g in scored if g == 1]
    miss = [s for s, g in scored if g == 0]
    hs = sum(hit)/len(hit) if hit else 0
    ms = sum(miss)/len(miss) if miss else 0
    return hs - ms

def main():
    # 4 non-overlapping windows: days 2-7, 7-12, 12-17, 17-22 back
    windows = [(2,7),(7,12),(12,17),(17,22)]
    add_score = lambda g,h: g + h
    blend_score = lambda g,h: 0.4*h + 0.6*g
    gen_score = lambda g,h: g

    print('Testing ADDED vs BLEND vs GENERAL across 4 separate windows')
    print('(separation = gap between hitters and non-hitters; bigger = better predictor)')
    print()
    add_wins = 0
    for w in windows:
        gpks = games_in_window(w[0], w[1])
        rows = collect(gpks)
        if len(rows) < 30:
            print(f'Window days {w[0]}-{w[1]}: only {len(rows)} batters, skipping')
            continue
        sa = separation(rows, add_score)
        sb = separation(rows, blend_score)
        sg = separation(rows, gen_score)
        winner = 'ADDED' if sa > sb and sa > sg else ('BLEND' if sb > sg else 'GENERAL')
        if winner == 'ADDED':
            add_wins += 1
        print(f'Window days {w[0]}-{w[1]} ({len(gpks)} games, {len(rows)} batters):')
        print(f'  ADDED   separation: {sa:+.3f}')
        print(f'  BLEND   separation: {sb:+.3f}')
        print(f'  GENERAL separation: {sg:+.3f}')
        print(f'  >>> WINNER: {winner}')
        print()
    print(f'ADDED won {add_wins} of {len(windows)} windows')

if __name__ == '__main__':
    main()
