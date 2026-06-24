import urllib.request, json, datetime as dt
from zoneinfo import ZoneInfo

def get(u):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(u, headers={'User-Agent': 'x'}), timeout=30).read())

def batter_krates(bid, pid, throws, season):
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

def main():
    season = dt.datetime.now(ZoneInfo('America/New_York')).date().year
    today = dt.datetime.now(ZoneInfo('America/New_York')).date()
    gpks = []
    for i in range(2, 20):  # 18 days, all of it
        d = (today - dt.timedelta(days=i)).isoformat()
        s = get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}')
        for day in s.get('dates', []):
            for g in day.get('games', []):
                if g.get('status', {}).get('abstractGameState') == 'Final':
                    gpks.append(g['gamePk'])
    print(f'Pulled {len(gpks)} completed games')

    # for each starting pitcher: ADDED lineup K-score + his actual Ks
    rows = []
    for gpk in gpks:
        f = get(f'https://statsapi.mlb.com/api/v1.1/game/{gpk}/feed/live')
        try:
            for pside, oside in [('home', 'away'), ('away', 'home')]:
                pid = f['gameData'].get('probablePitchers', {}).get(pside, {}).get('id')
                if not pid:
                    continue
                pbox = f['liveData']['boxscore']['teams'][pside].get('players', {})
                pstats = pbox.get(f'ID{pid}', {}).get('stats', {}).get('pitching', {})
                actual_k = pstats.get('strikeOuts', None)
                if actual_k is None:
                    continue
                throws = pitcher_throws(pid)
                order = f['liveData']['boxscore']['teams'][oside].get('battingOrder', [])[:9]
                if len(order) < 9:
                    continue
                add_sum = 0.0; valid = 0
                for bid in order:
                    gen, h2h = batter_krates(bid, pid, throws, season)
                    if gen is None:
                        continue
                    valid += 1
                    add_sum += (gen + h2h) if h2h is not None else gen
                if valid >= 7:
                    rows.append((add_sum, actual_k))
        except Exception:
            pass

    print(f'Pitchers with lineup data: {len(rows)}')
    print()

    # BUCKET TEST: group pitchers by their ADDED lineup K-score,
    # show AVERAGE actual strikeouts in each bucket (the real-scale proof)
    print('Avg ACTUAL strikeouts by ADDED lineup-K-score bucket:')
    print('(if higher score = more actual Ks, the signal is real, not scale)')
    edges = [(0.0, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 10.0)]
    for lo, hi in edges:
        sub = [k for s, k in rows if lo <= s < hi]
        if len(sub) >= 4:
            avg_k = sum(sub) / len(sub)
            # also % that went 6+ Ks
            pct6 = sum(1 for k in sub if k >= 6) / len(sub) * 100
            print(f'  score {lo:.1f}-{hi:.1f}: avg {avg_k:.2f} Ks, {round(pct6)}% threw 6+ ({len(sub)} pitchers)')

    print()
    # correlation: does the score track actual Ks?
    import statistics
    if len(rows) >= 10:
        scores = [s for s, k in rows]
        ks = [k for s, k in rows]
        n = len(rows)
        mean_s = sum(scores)/n; mean_k = sum(ks)/n
        cov = sum((s-mean_s)*(k-mean_k) for s, k in rows)/n
        sd_s = statistics.pstdev(scores); sd_k = statistics.pstdev(ks)
        corr = cov/(sd_s*sd_k) if sd_s and sd_k else 0
        print(f'Correlation between ADDED score and actual Ks: {corr:+.3f}')
        print('(closer to +1.0 = score strongly predicts strikeouts)')

if __name__ == '__main__':
    main()
