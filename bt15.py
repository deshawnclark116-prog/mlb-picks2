import urllib.request, json, datetime as dt, math
from zoneinfo import ZoneInfo
import sys
sys.path.insert(0, '/home/render/project/src')

MLB = 'https://statsapi.mlb.com/api/v1'
MLB11 = 'https://statsapi.mlb.com/api/v1.1'
GH = 'https://deshawnclark116-prog.github.io/mlb-picks2'
RECENCY_DECAY = 0.6

def get(u):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(u, headers={'User-Agent': 'x'}), timeout=30).read())

def norm(s):
    return ''.join(c for c in s.lower() if c.isalpha() or c == ' ').strip()

def batter_stats(pid):
    g = get(f'{MLB}/people/{pid}/stats?stats=season&group=hitting&season=2026')
    try:
        st = g['stats'][0]['splits'][0]['stat']
        pa = int(st.get('plateAppearances', 0) or 0)
        if pa < 20: return None
        slg = float(st.get('slg', 0) or 0)
        avg = float(st.get('avg', 0) or 0)
        hr = int(st.get('homeRuns', 0) or 0)
        return {'slg': slg, 'avg': avg, 'iso': slg-avg, 'hr_per_pa': hr/pa, 'pa': pa}
    except Exception:
        return None

def h2h_vs_pitcher(batter_id, pitcher_id):
    d = get(f'{MLB}/people/{batter_id}/stats?stats=vsPlayer'
            f'&opposingPlayerId={pitcher_id}&group=hitting')
    ab=0; h=0; hr=0; tb=0
    try:
        for s in d['stats'][0]['splits']:
            if s.get('stat'):
                st=s['stat']
                ab+=int(st.get('atBats',0) or 0); h+=int(st.get('hits',0) or 0)
                hr+=int(st.get('homeRuns',0) or 0); tb+=int(st.get('totalBases',0) or 0)
    except Exception:
        pass
    if ab>=4:
        return {'h2h_slg':tb/ab, 'h2h_avg':h/ab, 'h2h_hr':hr, 'h2h_ab':ab, 'has_h2h':True}
    return {'h2h_slg':0, 'h2h_avg':0, 'h2h_hr':0, 'h2h_ab':ab, 'has_h2h':False}

def pitcher_hr_rate(pid):
    """Pitcher's HR-allowed rate per batter faced (recency-weighted)."""
    g = get(f'{MLB}/people/{pid}/stats?stats=gameLog&group=pitching&season=2026')
    try:
        splits = g['stats'][0]['splits']
    except Exception:
        return 0, 0
    hrs=[]; bfs=[]
    cum_hr=0; cum_bf=0
    for sp in splits:
        st=sp['stat']
        bf=int(st.get('battersFaced',0) or 0)
        hr=int(st.get('homeRuns',0) or 0)
        if bf>=12:
            hrs.append(hr); bfs.append(bf)
            cum_hr+=hr; cum_bf+=bf
    if not hrs: return 0, 0
    season_rate=cum_hr/cum_bf if cum_bf else 0
    n=len(hrs)
    w=[math.exp(-RECENCY_DECAY*(n-1-i)) for i in range(n)]
    rec_rate=(sum(wi*h for wi,h in zip(w,hrs))/
              sum(wi*b for wi,b in zip(w,bfs))) if sum(w) else season_rate
    # blend recency + season
    rate=0.7*rec_rate+0.3*season_rate
    return rate, cum_bf

def main():
    today = dt.datetime.now(ZoneInfo('America/New_York')).date()
    all_dates = [(today-dt.timedelta(days=i)).isoformat() for i in range(1,22)]

    idx_data = get(f'{MLB}/sports/1/players?season=2026')
    idx = {norm(p.get('fullName','')): p.get('id')
           for p in idx_data.get('people',[])}

    rows = []
    seen = set()
    files_found = 0

    for date in all_dates:
        try:
            preds = get(f'{GH}/predictions_{date}.json')
            files_found += 1
        except Exception:
            continue

        for pred in preds:
            if pred.get('prop_type') not in ('batter_hits','batter_total_bases'):
                continue
            nm = norm(pred.get('player',''))
            gpk = pred.get('game_id')
            if not nm or not gpk: continue
            key = (date, nm)
            if key in seen: continue
            seen.add(key)

            pid = idx.get(nm)
            if not pid: continue

            stats = batter_stats(pid)
            if not stats: continue

            # find opposing starter from boxscore
            pitcher_id = None
            try:
                f = get(f'{MLB11}/game/{gpk}/feed/live')
                batter_side = None
                for side in ('home','away'):
                    order = f['liveData']['boxscore']['teams'][side].get('battingOrder',[])
                    if pid in order or pid in [int(x) for x in order
                                               if str(x).isdigit()]:
                        batter_side=side; break
                if batter_side:
                    oside='away' if batter_side=='home' else 'home'
                    pitchers=f['liveData']['boxscore']['teams'][oside].get('pitchers',[])
                    if pitchers: pitcher_id=pitchers[0]
            except Exception:
                pass

            h2h = h2h_vs_pitcher(pid, pitcher_id) if pitcher_id else \
                  {'h2h_slg':0,'h2h_avg':0,'h2h_hr':0,'h2h_ab':0,'has_h2h':False}

            p_hr_rate, p_bf = pitcher_hr_rate(pitcher_id) if pitcher_id else (0,0)

            # actual HR
            actual_hr = 0
            try:
                g = get(f'{MLB}/people/{pid}/stats'
                        f'?stats=gameLog&group=hitting&season=2026')
                for sp in g['stats'][0]['splits']:
                    if sp.get('date')==date:
                        actual_hr=int(sp['stat'].get('homeRuns',0) or 0); break
            except Exception:
                pass

            # SUM SCORES — three versions
            two_signal = stats['slg'] + h2h['h2h_slg']
            three_signal = stats['slg'] + h2h['h2h_slg'] + p_hr_rate
            # also test with pitcher rate weighted heavier
            three_w = stats['slg'] + h2h['h2h_slg'] + (p_hr_rate * 5)

            rows.append({
                'date': date, 'name': nm,
                **stats, **h2h,
                'p_hr_rate': p_hr_rate, 'p_bf': p_bf,
                'two_signal': two_signal,
                'three_signal': three_signal,
                'three_w': three_w,
                'actual_hr': actual_hr,
            })

    print(f'Files: {files_found} | Entries: {len(rows)}')
    print()

    # SEPARATION TEST — the honest tiebreaker
    print('SEPARATION TEST (bigger gap = better predictor):')
    for key, label in [
        ('slg',          'SLG alone'),
        ('two_signal',   'SLG + h2h_SLG (2-signal)'),
        ('three_signal', 'SLG + h2h_SLG + pitcher_HR_rate (3-signal)'),
        ('three_w',      'SLG + h2h_SLG + pitcher_HR_rate*5 (weighted)'),
        ('p_hr_rate',    'Pitcher HR rate alone'),
    ]:
        hr_s  = [r[key] for r in rows if r['actual_hr']>=1]
        no_s  = [r[key] for r in rows if r['actual_hr']==0]
        hs = sum(hr_s)/len(hr_s) if hr_s else 0
        ns = sum(no_s)/len(no_s) if no_s else 0
        print(f'  {label:45} gap {hs-ns:+.3f}  '
              f'(hitters {hs:.3f} vs non {ns:.3f})')

    print()

    # BUCKET TEST on 3-signal score
    print('3-SIGNAL SCORE BUCKETS (SLG + h2h_SLG + pitcher_HR_rate):')
    edges = [(0,.3),(.3,.4),(.4,.5),(.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,99)]
    for lo,hi in edges:
        group=[r for r in rows if lo<=r['three_signal']<hi]
        if len(group)<3: continue
        hr_g=sum(1 for r in group if r['actual_hr']>=1)
        t=len(group)
        be=round((t/hr_g-1)*100) if hr_g else 0
        print(f'  score {lo:.1f}-{hi:.1f}: {hr_g}/{t} = '
              f'{round(hr_g/t*100,1)}%  '
              f'breakeven +{be}' if hr_g else
              f'  score {lo:.1f}-{hi:.1f}: {hr_g}/{t} = 0%')

    print()

    # MULTI-WINDOW — does the 3-signal hold?
    print('MULTI-WINDOW (3-signal score >= 0.80):')
    for i in range(0, 20, 5):
        w_dates = all_dates[i:i+5]
        above = [r for r in rows if r['date'] in w_dates
                 and r['three_signal']>=0.80]
        below = [r for r in rows if r['date'] in w_dates
                 and r['three_signal']<0.80]
        if len(above)<3: continue
        ah=sum(1 for r in above if r['actual_hr']>=1)
        bh=sum(1 for r in below if r['actual_hr']>=1)
        print(f'  Days {i+1}-{i+5}: '
              f'score>=0.80 {ah}/{len(above)}='
              f'{round(ah/len(above)*100,1) if above else 0}%  '
              f'below {bh}/{len(below)}='
              f'{round(bh/len(below)*100,1) if below else 0}%')

    print()

    # pitcher HR rate alone — does it add anything?
    print('DOES PITCHER HR RATE ADD SIGNAL? (vs 2-signal):')
    two_hi = [r for r in rows if r['two_signal']>=0.80]
    three_hi = [r for r in rows if r['three_signal']>=0.80]
    if two_hi:
        th=sum(1 for r in two_hi if r['actual_hr']>=1)
        print(f'  2-signal >= 0.80: {th}/{len(two_hi)} = '
              f'{round(th/len(two_hi)*100,1)}%')
    if three_hi:
        th=sum(1 for r in three_hi if r['actual_hr']>=1)
        print(f'  3-signal >= 0.80: {th}/{len(three_hi)} = '
              f'{round(th/len(three_hi)*100,1)}%')
    print()
    print('If 3-signal hits higher % than 2-signal at the same threshold,')
    print('the pitcher rate is adding real value. If similar, stick with 2.')

if __name__ == '__main__':
    main()
