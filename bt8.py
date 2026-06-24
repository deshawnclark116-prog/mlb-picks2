import urllib.request, json, datetime as dt, statistics
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
    for i in range(2, 20):
        d = (today - dt.timedelta(days=i)).isoformat()
        s = get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}')
        for day in s.get('dates', []):
            for g in day.get('games', []):
                if g.get('status', {}).get('abstractGameState') == 'Final':
                    gpks.append(g['gamePk'])
    print(f'Pulled {len(gpks)} completed games')

    # collect: added_score, blend_score, actual_ks
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
                add_s = 0.0; bl_s = 0.0; valid = 0
                for bid in order:
                    gen, h2h = batter_krates(bid, pid, throws, season)
                    if gen is None:
                        continue
                    valid += 1
                    if h2h is not None:
                        add_s += (gen + h2h); bl_s += (0.4*h2h + 0.6*gen)
                    else:
                        add_s += gen; bl_s += gen
                if valid >= 7:
                    rows.append((add_s, bl_s, actual_k))
        except Exception:
            pass
    print(f'Pitchers: {len(rows)}')
    n = len(rows)
    if n < 30:
        print('Not enough data'); return

    # Build conversion: linear fit from added_score -> actual Ks (on first half)
    half = n // 2
    train = rows[:half]; test = rows[half:]
    # linear regression added -> ks on train
    xs = [r[0] for r in train]; ys = [r[2] for r in train]
    mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
    slope = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
    intercept = my - slope*mx
    # same for blend
    xb = [r[1] for r in train]
    mxb = sum(xb)/len(xb)
    slope_b = sum((x-mxb)*(y-my) for x, y in zip(xb, ys)) / sum((x-mxb)**2 for x in xb)
    intercept_b = my - slope_b*mxb

    def add_proj(s): return slope*s + intercept
    def bl_proj(s): return slope_b*s + intercept_b

    print()
    print('='*55)
    print('TEST 1 - HEAD-TO-HEAD: which projection lands closer to actual?')
    add_closer = 0; bl_closer = 0
    for a, b, k in test:
        ea = abs(add_proj(a) - k); eb = abs(bl_proj(b) - k)
        if ea < eb: add_closer += 1
        elif eb < ea: bl_closer += 1
    print(f'  ADDED closer: {add_closer}  |  BLEND closer: {bl_closer}  (of {len(test)} test pitchers)')

    print()
    print('TEST 2 - BEAT THE LINE: pick over/under vs a proxy line (lineup median)')
    # proxy "line" = the rounded median actual K (no FanDuel history available);
    # test: does each method pick the correct side relative to that line?
    med = statistics.median([k for _, _, k in test])
    line = round(med) + 0.5
    add_right = 0; bl_right = 0; gradable = 0
    for a, b, k in test:
        actual_side = 'over' if k > line else 'under'
        add_side = 'over' if add_proj(a) > line else 'under'
        bl_side = 'over' if bl_proj(b) > line else 'under'
        gradable += 1
        if add_side == actual_side: add_right += 1
        if bl_side == actual_side: bl_right += 1
    print(f'  proxy line {line}: ADDED {add_right}/{gradable} = {round(add_right/gradable*100)}% | BLEND {bl_right}/{gradable} = {round(bl_right/gradable*100)}%')

    print()
    print('TEST 3 - ERROR SIZE (avg miss + bias, lower/closer-to-0 better):')
    add_err = [add_proj(a) - k for a, b, k in test]
    bl_err = [bl_proj(b) - k for a, b, k in test]
    print(f'  ADDED: avg miss {sum(abs(e) for e in add_err)/len(add_err):.2f}, bias {sum(add_err)/len(add_err):+.2f}')
    print(f'  BLEND: avg miss {sum(abs(e) for e in bl_err)/len(bl_err):.2f}, bias {sum(bl_err)/len(bl_err):+.2f}')

    print()
    print('TEST 4 - EXTREMES: does ADDED flag the blowups (8+ K) and duds (<3 K)?')
    blowups = [(a, k) for a, b, k in rows if k >= 8]
    duds = [(a, k) for a, b, k in rows if k < 3]
    allscores = sorted([a for a, b, k in rows])
    p70 = allscores[int(len(allscores)*0.7)]
    p30 = allscores[int(len(allscores)*0.3)]
    blow_flagged = sum(1 for a, k in blowups if a >= p70)
    dud_flagged = sum(1 for a, k in duds if a <= p30)
    print(f'  blowups (8+K) with high score (top 30%): {blow_flagged}/{len(blowups)} = {round(blow_flagged/len(blowups)*100) if blowups else 0}%')
    print(f'  duds (<3K) with low score (bottom 30%):  {dud_flagged}/{len(duds)} = {round(dud_flagged/len(duds)*100) if duds else 0}%')

    print()
    print('TEST 5 - OUT-OF-SAMPLE: conversion built on 1st half, tested on 2nd')
    oos_err = [abs(add_proj(a) - k) for a, b, k in test]
    in_err = [abs(add_proj(a) - k) for a, b, k in train]
    print(f'  in-sample avg miss:     {sum(in_err)/len(in_err):.2f}')
    print(f'  out-of-sample avg miss: {sum(oos_err)/len(oos_err):.2f}')
    print(f'  (close = generalizes, not overfit)')
    # correlation on test set
    ts = [a for a, b, k in test]; tk = [k for a, b, k in test]
    mts = sum(ts)/len(ts); mtk = sum(tk)/len(tk)
    cov = sum((s-mts)*(k-mtk) for s, k in zip(ts, tk))/len(ts)
    corr = cov/(statistics.pstdev(ts)*statistics.pstdev(tk)) if statistics.pstdev(ts) else 0
    print(f'  out-of-sample correlation: {corr:+.3f}')

if __name__ == '__main__':
    main()
