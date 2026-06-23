import urllib.request, json, datetime as dt
from zoneinfo import ZoneInfo

def get(u):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(u, headers={'User-Agent': 'x'}), timeout=30).read())

H2H_W = 0.40
GEN_W = 0.60

def rates(bid, pid):
    # general season avg
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
    # head-to-head
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
    return gen, h2h, hab

def main():
    today = dt.datetime.now(ZoneInfo('America/New_York')).date()
    # use a DIFFERENT window: days 9-16 back (not the same games as bt.py)
    gpks = []
    for i in range(9, 17):
        d = (today - dt.timedelta(days=i)).isoformat()
        s = get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}')
        for day in s.get('dates', []):
            for g in day.get('games', []):
                if g.get('status', {}).get('abstractGameState') == 'Final':
                    gpks.append(g['gamePk'])
    print(f'Pulled {len(gpks)} completed games (days 9-16 back, DIFFERENT window)')

    # For batters who have BOTH a general rate and h2h history, compare:
    # which model better predicts whether they got a hit?
    gen_correct = 0
    h2h_correct = 0
    both_n = 0
    # also track: when h2h disagrees with gen (h2h high, gen low or vice versa), who wins?
    disagree_h2h_right = 0
    disagree_gen_right = 0
    disagree_n = 0

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
                    gen, h2h, hab = rates(bid, op)
                    if gen is None or h2h is None:
                        continue
                    pd = box.get(f'ID{bid}', {})
                    ah = pd.get('stats', {}).get('batting', {}).get('hits', None)
                    if ah is None:
                        continue
                    got = 1 if ah >= 1 else 0
                    both_n += 1
                    # general model predicts hit if proj P(hit in ~4 AB) > 0.5
                    # P(>=1 hit in 4 AB) = 1-(1-avg)^4
                    gen_p = 1 - (1 - gen) ** 4
                    blended = H2H_W * h2h + GEN_W * gen
                    bl_p = 1 - (1 - blended) ** 4
                    gen_pred = 1 if gen_p > 0.5 else 0
                    bl_pred = 1 if bl_p > 0.5 else 0
                    if gen_pred == got:
                        gen_correct += 1
                    if bl_pred == got:
                        h2h_correct += 1
                    # disagreement cases
                    if gen_pred != bl_pred:
                        disagree_n += 1
                        if bl_pred == got:
                            disagree_h2h_right += 1
                        if gen_pred == got:
                            disagree_gen_right += 1
        except Exception:
            pass

    print()
    print(f'Batters with BOTH general + h2h: {both_n}')
    print()
    print(f'GENERAL model alone correct:  {gen_correct}/{both_n} = {round(gen_correct/both_n*100) if both_n else 0}%')
    print(f'BLENDED (h2h+general) correct: {h2h_correct}/{both_n} = {round(h2h_correct/both_n*100) if both_n else 0}%')
    print()
    print(f'When they DISAGREED ({disagree_n} times):')
    print(f'  blended/h2h was right: {disagree_h2h_right}/{disagree_n} = {round(disagree_h2h_right/disagree_n*100) if disagree_n else 0}%')
    print(f'  general was right:     {disagree_gen_right}/{disagree_n} = {round(disagree_gen_right/disagree_n*100) if disagree_n else 0}%')

if __name__ == '__main__':
    main()
