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
    for i in range(2, 12):
        d = (today - dt.timedelta(days=i)).isoformat()
        s = get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}')
        for day in s.get('dates', []):
            for g in day.get('games', []):
                if g.get('status', {}).get('abstractGameState') == 'Final':
                    gpks.append(g['gamePk'])
    print(f'Pulled {len(gpks)} completed games')

    # collect every batter with both rates + whether they got a hit
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

    print(f'Total batters with both rates: {len(rows)}')
    print()

    # THREE methods, compare hit rate by bucketing each method's score:
    # 1. ADDED: gen + h2h
    # 2. BLEND 40/60: 0.4*h2h + 0.6*gen
    # 3. GENERAL only
    def bucketize(scores_got, label, edges):
        print(f'{label}:')
        for lo, hi in edges:
            sub = [g for s, g in scores_got if lo <= s < hi]
            if len(sub) >= 5:
                print(f'  score {lo:.2f}-{hi:.2f}: {sum(sub)}/{len(sub)} = {round(sum(sub)/len(sub)*100)}% ({len(sub)} batters)')
        print()

    added = [(gen + h2h, got) for gen, h2h, got in rows]
    blend = [(0.4*h2h + 0.6*gen, got) for gen, h2h, got in rows]
    genonly = [(gen, got) for gen, h2h, got in rows]

    # ADDED ranges 0 to ~1.3, use wider buckets
    bucketize(added, 'METHOD 1 - ADDED (gen + h2h)',
              [(0.0,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,1.5)])
    bucketize(blend, 'METHOD 2 - BLEND 40/60',
              [(0.0,0.2),(0.2,0.25),(0.25,0.3),(0.3,0.35),(0.35,1.0)])
    bucketize(genonly, 'METHOD 3 - GENERAL only',
              [(0.0,0.2),(0.2,0.25),(0.25,0.3),(0.3,0.35),(0.35,1.0)])

    # correlation check: which method's score best separates hits from non-hits?
    def avg_score_split(scores_got):
        hit_scores = [s for s, g in scores_got if g == 1]
        miss_scores = [s for s, g in scores_got if g == 0]
        return (sum(hit_scores)/len(hit_scores) if hit_scores else 0,
                sum(miss_scores)/len(miss_scores) if miss_scores else 0)

    print('SEPARATION (gap between avg score of hitters vs non-hitters - bigger=better):')
    for label, data in [('ADDED', added), ('BLEND', blend), ('GENERAL', genonly)]:
        hs, ms = avg_score_split(data)
        print(f'  {label}: hitters avg {hs:.3f}, non-hitters avg {ms:.3f}, gap {hs-ms:+.3f}')

if __name__ == '__main__':
    main()
