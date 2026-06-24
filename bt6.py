import urllib.request, json, datetime as dt
from zoneinfo import ZoneInfo

def get(u):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(u, headers={'User-Agent': 'x'}), timeout=30).read())

def batter_krates(bid, pid, throws, season):
    # general K-rate vs handedness
    code = "vl" if throws == "L" else "vr"
    sp = get(f'https://statsapi.mlb.com/api/v1/people/{bid}/stats?stats=statSplits&sitCodes={code}&group=hitting&season={season}')
    gen = None
    try:
        st = sp['stats'][0]['splits'][0]['stat']
        pa = int(st.get('plateAppearances', 0) or 0)
        so = int(st.get('strikeOuts', 0) or 0)
        if pa >= 15:
            gen = so / pa
    except Exception:
        pass
    if gen is None:
        return None, None
    # head-to-head K-rate vs this pitcher
    d = get(f'https://statsapi.mlb.com/api/v1/people/{bid}/stats?stats=vsPlayer&opposingPlayerId={pid}&group=hitting')
    hpa = 0; hk = 0
    try:
        for s in d['stats'][0]['splits']:
            if s.get('stat'):
                st = s['stat']
                hpa += int(st.get('plateAppearances', 0) or 0)
                hk += int(st.get('strikeOuts', 0) or 0)
    except Exception:
        pass
    h2h = hk / hpa if hpa >= 2 else None
    return gen, h2h

def pitcher_throws(pid):
    d = get(f'https://statsapi.mlb.com/api/v1/people/{pid}')
    try:
        return d['people'][0]['pitchHand']['code']
    except Exception:
        return 'R'

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

def collect(gpks, season):
    # for each starting pitcher: compute lineup K-rate sums (3 methods) + his actual Ks
    rows = []
    for gpk in gpks:
        f = get(f'https://statsapi.mlb.com/api/v1.1/game/{gpk}/feed/live')
        try:
            for pside, oside in [('home', 'away'), ('away', 'home')]:
                pid = f['gameData'].get('probablePitchers', {}).get(pside, {}).get('id')
                if not pid:
                    continue
                # his actual Ks this game
                pbox = f['liveData']['boxscore']['teams'][pside].get('players', {})
                pstats = pbox.get(f'ID{pid}', {}).get('stats', {}).get('pitching', {})
                actual_k = pstats.get('strikeOuts', None)
                if actual_k is None:
                    continue
                throws = pitcher_throws(pid)
                order = f['liveData']['boxscore']['teams'][oside].get('battingOrder', [])[:9]
                if len(order) < 9:
                    continue
                add_sum = 0.0; blend_sum = 0.0; gen_sum = 0.0; valid = 0
                for bid in order:
                    gen, h2h = batter_krates(bid, pid, throws, season)
                    if gen is None:
                        continue
                    valid += 1
                    if h2h is not None:
                        add_sum += (gen + h2h)
                        blend_sum += (0.4 * h2h + 0.6 * gen)
                    else:
                        add_sum += gen
                        blend_sum += gen
                    gen_sum += gen
                if valid >= 7:
                    rows.append((add_sum, blend_sum, gen_sum, actual_k))
        except Exception:
            pass
    return rows

def separation(rows, idx):
    # split pitchers into high-K (8+) vs low-K (<5) games, see gap in each method's score
    high = [r[idx] for r in rows if r[3] >= 8]
    low = [r[idx] for r in rows if r[3] < 5]
    hi = sum(high)/len(high) if high else 0
    lo = sum(low)/len(low) if low else 0
    return hi - lo, len(high), len(low)

def main():
    season = dt.datetime.now(ZoneInfo('America/New_York')).date().year
    windows = [(2, 8), (8, 14), (14, 20)]
    print('PITCHER strikeout test: ADDED vs BLEND vs GENERAL lineup K-rate sums')
    print('(separation = gap between high-K games (8+) and low-K games (<5); bigger=better)')
    print()
    add_wins = 0
    for w in windows:
        gpks = games_in_window(w[0], w[1])
        rows = collect(gpks, season)
        if len(rows) < 15:
            print(f'Window {w[0]}-{w[1]}: only {len(rows)} pitchers, skipping')
            continue
        sa, nh, nl = separation(rows, 0)
        sb, _, _ = separation(rows, 1)
        sg, _, _ = separation(rows, 2)
        winner = 'ADDED' if sa > sb and sa > sg else ('BLEND' if sb > sg else 'GENERAL')
        if winner == 'ADDED':
            add_wins += 1
        print(f'Window {w[0]}-{w[1]} ({len(gpks)} games, {len(rows)} pitchers, {nh} high-K / {nl} low-K):')
        print(f'  ADDED   separation: {sa:+.3f}')
        print(f'  BLEND   separation: {sb:+.3f}')
        print(f'  GENERAL separation: {sg:+.3f}')
        print(f'  >>> WINNER: {winner}')
        print()
    print(f'ADDED won {add_wins} of {len(windows)} windows')

if __name__ == '__main__':
    main()
