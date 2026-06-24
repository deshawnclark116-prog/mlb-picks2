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

def main():
    today = dt.datetime.now(ZoneInfo('America/New_York')).date()
    gpks = []
    for i in range(2, 12):  # 10 days
        d = (today - dt.timedelta(days=i)).isoformat()
        s = get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}')
        for day in s.get('dates', []):
            for g in day.get('games', []):
                if g.get('status', {}).get('abstractGameState') == 'Final':
                    gpks.append(g['gamePk'])
    print(f'Pulled {len(gpks)} completed games')

    # Strategy buckets — each batter classified by what the two signals say:
    # general "likes" = season avg >= .270 ; h2h "likes" = h2h avg >= .300
    both_h = both_t = 0       # BOTH like him
    genonly_h = genonly_t = 0 # only general likes
    h2honly_h = h2honly_t = 0 # only h2h likes
    neither_h = neither_t = 0 # neither

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
                    gen_likes = gen >= 0.270
                    h2h_likes = h2h >= 0.300
                    if gen_likes and h2h_likes:
                        both_t += 1; both_h += got
                    elif gen_likes and not h2h_likes:
                        genonly_t += 1; genonly_h += got
                    elif h2h_likes and not gen_likes:
                        h2honly_t += 1; h2honly_h += got
                    else:
                        neither_t += 1; neither_h += got
        except Exception:
            pass

    def pct(h, t):
        return f'{h}/{t} = {round(h/t*100) if t else 0}%'

    print()
    print('Hit rate by what the two signals say:')
    print(f'  BOTH like him:        {pct(both_h, both_t)}')
    print(f'  ONLY general likes:   {pct(genonly_h, genonly_t)}')
    print(f'  ONLY head-to-head:    {pct(h2honly_h, h2honly_t)}')
    print(f'  NEITHER likes:        {pct(neither_h, neither_t)}')

if __name__ == '__main__':
    main()
