import urllib.request, json, datetime as dt
from zoneinfo import ZoneInfo

def get(u):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(u, headers={'User-Agent': 'x'}), timeout=30).read())

H2H_W = 0.40
GEN_W = 0.60

def proj_batter(bid, pid):
    sp = get(f'https://statsapi.mlb.com/api/v1/people/{bid}/stats?stats=season&group=hitting&season=2026')
    gen = 0
    try:
        st = sp['stats'][0]['splits'][0]['stat']
        ab = int(st.get('atBats', 0) or 0)
        h = int(st.get('hits', 0) or 0)
        if ab >= 20:
            gen = h / ab
    except Exception:
        pass
    d = get(f'https://statsapi.mlb.com/api/v1/people/{bid}/stats?stats=vsPlayer&opposingPlayerId={pid}&group=hitting')
    hab = 0
    hh = 0
    try:
        for s in d['stats'][0]['splits']:
            if s.get('stat'):
                st = s['stat']
                hab += int(st.get('atBats', 0) or 0)
                hh += int(st.get('hits', 0) or 0)
    except Exception:
        pass
    if hab >= 4:
        h2h = hh / hab
        return H2H_W * h2h + GEN_W * gen, h2h
    return None, None

def main():
    today = dt.datetime.now(ZoneInfo('America/New_York')).date()
    gpks = []
    for i in range(2, 9):
        d = (today - dt.timedelta(days=i)).isoformat()
        s = get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}')
        for day in s.get('dates', []):
            for g in day.get('games', []):
                if g.get('status', {}).get('abstractGameState') == 'Final':
                    gpks.append(g['gamePk'])
    print(f'Pulled {len(gpks)} completed games')

    fh = ft = nh = nt = 0
    buckets = {}
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
                    pr, h2h = proj_batter(bid, op)
                    if pr is None:
                        continue
                    pd = box.get(f'ID{bid}', {})
                    ah = pd.get('stats', {}).get('batting', {}).get('hits', None)
                    if ah is None:
                        continue
                    got = 1 if ah >= 1 else 0
                    if h2h >= 0.300:
                        ft += 1
                        fh += got
                    else:
                        nt += 1
                        nh += got
                    pb = round(pr, 1)
                    buckets.setdefault(pb, [0, 0])
                    buckets[pb][0] += got
                    buckets[pb][1] += 1
        except Exception:
            pass

    print()
    print(f'Batters with h2h history (4+ AB): {ft + nt}')
    fr = round(fh / ft * 100) if ft else 0
    nr = round(nh / nt * 100) if nt else 0
    print(f'FLAGGED (h2h .300+): {fh}/{ft} = {fr}% got a hit')
    print(f'NOT flagged:         {nh}/{nt} = {nr}% got a hit')
    print(f'EDGE: {fr - nr:+d} points')
    print()
    print('By projection bucket:')
    for pb in sorted(buckets):
        h, t = buckets[pb]
        if t >= 3:
            print(f'  proj {pb}: {h}/{t} = {round(h / t * 100)}%')

if __name__ == '__main__':
    main()
