import urllib.request, json, datetime as dt, math
from zoneinfo import ZoneInfo
import sys
sys.path.insert(0, '/home/render/project/src')

MLB = 'https://statsapi.mlb.com/api/v1'
MLB11 = 'https://statsapi.mlb.com/api/v1.1'
GH = 'https://deshawnclark116-prog.github.io/mlb-picks2'

def get(u):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(u, headers={'User-Agent': 'x'}), timeout=30).read())

def norm(s):
    return ''.join(c for c in s.lower() if c.isalpha() or c == ' ').strip()

def batter_season_stats(pid):
    g = get(f'{MLB}/people/{pid}/stats?stats=season&group=hitting&season=2026')
    try:
        st = g['stats'][0]['splits'][0]['stat']
        pa = int(st.get('plateAppearances', 0) or 0)
        if pa < 20: return None
        return {
            'slg': float(st.get('slg', 0) or 0),
            'avg': float(st.get('avg', 0) or 0),
            'iso': float(st.get('slg', 0) or 0) - float(st.get('avg', 0) or 0),
            'obp': float(st.get('obp', 0) or 0),
            'ops': float(st.get('ops', 0) or 0),
            'hr_per_pa': int(st.get('homeRuns',0) or 0)/pa,
            'bb_rate': int(st.get('baseOnBalls',0) or 0)/pa,
            'so_rate': int(st.get('strikeOuts',0) or 0)/pa,
            'pa': pa,
        }
    except Exception:
        return None

def batter_recent_form(pid, before_date, n_games=7):
    """Stats from the last n games before a given date."""
    g = get(f'{MLB}/people/{pid}/stats?stats=gameLog&group=hitting&season=2026')
    try:
        splits = g['stats'][0]['splits']
    except Exception:
        return None
    # filter games before this date, take last n
    prior = [s for s in splits if s.get('date','') < before_date]
    if len(prior) < 3: return None
    recent = prior[-n_games:]
    h=tb=hr=ab=pa=bb=so=0
    for s in recent:
        st=s['stat']
        h+=int(st.get('hits',0) or 0)
        tb+=int(st.get('totalBases',0) or 0)
        hr+=int(st.get('homeRuns',0) or 0)
        ab+=int(st.get('atBats',0) or 0)
        pa+=int(st.get('plateAppearances',0) or 0)
        bb+=int(st.get('baseOnBalls',0) or 0)
        so+=int(st.get('strikeOuts',0) or 0)
    if ab==0: return None
    return {
        'recent_avg': h/ab,
        'recent_slg': tb/ab,
        'recent_iso': (tb-h)/ab,
        'recent_hr': hr,
        'recent_hr_rate': hr/pa if pa else 0,
        'recent_bb_rate': bb/pa if pa else 0,
        'recent_so_rate': so/pa if pa else 0,
        'recent_games': len(recent),
    }

def h2h_vs_pitcher(batter_id, pitcher_id):
    d = get(f'{MLB}/people/{batter_id}/stats?stats=vsPlayer'
            f'&opposingPlayerId={pitcher_id}&group=hitting')
    ab=h=hr=tb=0
    try:
        for s in d['stats'][0]['splits']:
            if s.get('stat'):
                st=s['stat']
                ab+=int(st.get('atBats',0) or 0)
                h+=int(st.get('hits',0) or 0)
                hr+=int(st.get('homeRuns',0) or 0)
                tb+=int(st.get('totalBases',0) or 0)
    except Exception:
        pass
    if ab>=4:
        return {'h2h_slg':tb/ab,'h2h_avg':h/ab,'h2h_hr':hr,'h2h_ab':ab,'has_h2h':True}
    return {'h2h_slg':0,'h2h_avg':0,'h2h_hr':0,'h2h_ab':ab,'has_h2h':False}

def get_game_context(gpk, pid):
    """Get batting order position, handedness, park."""
    try:
        f = get(f'{MLB11}/game/{gpk}/feed/live')
        batter_side = None
        order_pos = None
        for side in ('home','away'):
            order=f['liveData']['boxscore']['teams'][side].get('battingOrder',[])
            if pid in order or pid in [int(x) for x in order if str(x).isdigit()]:
                batter_side=side
                try:
                    order_pos = [int(x) for x in order].index(pid)+1
                except Exception:
                    order_pos = None
                break
        pitcher_id = None
        pitcher_throws = None
        if batter_side:
            oside='away' if batter_side=='home' else 'home'
            pitchers=f['liveData']['boxscore']['teams'][oside].get('pitchers',[])
            if pitchers:
                pitcher_id=pitchers[0]
                pdata=f['liveData']['boxscore']['teams'][oside]['players'].get(f'ID{pitcher_id}',{})
                pitcher_throws=pdata.get('position',{}).get('code')
        # batter handedness
        batter_bats = None
        pinfo=get(f'{MLB}/people/{pid}')
        try:
            batter_bats=pinfo['people'][0]['batSide']['code']
        except Exception:
            pass
        # is home team
        home_team=f['gameData']['teams']['home']['name']
        venue=f['gameData']['venue']['name']
        return {
            'pitcher_id': pitcher_id,
            'pitcher_throws': pitcher_throws,
            'batter_bats': batter_bats,
            'order_pos': order_pos,
            'venue': venue,
            'is_home': batter_side=='home',
        }
    except Exception as e:
        return {}

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

            season = batter_season_stats(pid)
            if not season: continue

            recent = batter_recent_form(pid, date, n_games=7)
            if not recent: continue

            ctx = get_game_context(gpk, pid)
            h2h = h2h_vs_pitcher(pid, ctx.get('pitcher_id')) \
                  if ctx.get('pitcher_id') else \
                  {'h2h_slg':0,'h2h_avg':0,'h2h_hr':0,'h2h_ab':0,'has_h2h':False}

            # actual HR
            actual_hr = 0
            try:
                g = get(f'{MLB}/people/{pid}/stats'
                        f'?stats=gameLog&group=hitting&season=2026')
                for sp in g['stats'][0]['splits']:
                    if sp.get('date')==date:
                        actual_hr=int(sp['stat'].get('homeRuns',0) or 0)
                        break
            except Exception:
                pass

            # platoon advantage
            b=ctx.get('batter_bats','?'); t=ctx.get('pitcher_throws','?')
            platoon = 'advantage' if (b=='L' and t=='R') or (b=='R' and t=='L') else \
                      'disadvantage' if (b=='L' and t=='L') or (b=='R' and t=='R') else 'unknown'

            rows.append({
                'date':date,'name':nm,'actual_hr':actual_hr,
                **season, **recent, **h2h,
                'two_signal': season['slg']+h2h['h2h_slg'],
                'platoon': platoon,
                'order_pos': ctx.get('order_pos'),
                'venue': ctx.get('venue',''),
                'is_home': ctx.get('is_home'),
                'batter_bats': ctx.get('batter_bats'),
            })

    hr_batters = [r for r in rows if r['actual_hr']>=1]
    no_batters  = [r for r in rows if r['actual_hr']==0]

    print(f'Files: {files_found} | Entries: {len(rows)}')
    print(f'HR games: {len(hr_batters)} | Non-HR games: {len(no_batters)}')
    print()

    def avg(lst, key):
        vals = [x[key] for x in lst if x.get(key) is not None]
        return round(sum(vals)/len(vals),3) if vals else 0

    # DEEP COMPARISON — every signal, HR batters vs non-HR batters
    print('SIGNAL COMPARISON (HR batters vs non-HR batters):')
    print(f'  {"signal":30} {"HR batters":12} {"non-HR":12} {"gap":8}')
    print('-'*65)
    signals = [
        ('slg',             'Season SLG'),
        ('iso',             'Season ISO'),
        ('ops',             'Season OPS'),
        ('hr_per_pa',       'Season HR/PA'),
        ('recent_slg',      'Recent 7-game SLG'),
        ('recent_iso',      'Recent 7-game ISO'),
        ('recent_avg',      'Recent 7-game AVG'),
        ('recent_hr_rate',  'Recent 7-game HR/PA'),
        ('recent_so_rate',  'Recent K-rate (lower=better contact)'),
        ('h2h_slg',         'H2H SLG vs pitcher'),
        ('h2h_avg',         'H2H AVG vs pitcher'),
        ('two_signal',      'SLG + H2H SLG (sum)'),
    ]
    for key, label in signals:
        h_avg = avg(hr_batters, key)
        n_avg = avg(no_batters, key)
        gap = round(h_avg-n_avg, 3)
        marker = ' <<<' if abs(gap) > 0.05 else ''
        print(f'  {label:30} {h_avg:10}   {n_avg:10}   {gap:+.3f}{marker}')

    print()

    # PLATOON breakdown
    print('PLATOON ADVANTAGE:')
    for p_type in ('advantage','disadvantage','unknown'):
        group=[r for r in rows if r['platoon']==p_type]
        if not group: continue
        hr_g=sum(1 for r in group if r['actual_hr']>=1)
        print(f'  {p_type:15}: {hr_g}/{len(group)} = '
              f'{round(hr_g/len(group)*100,1)}% HR rate')

    print()

    # BATTING ORDER position
    print('BATTING ORDER POSITION:')
    for pos in range(1,10):
        group=[r for r in rows if r.get('order_pos')==pos]
        if len(group)<3: continue
        hr_g=sum(1 for r in group if r['actual_hr']>=1)
        print(f'  Position {pos}: {hr_g}/{len(group)} = '
              f'{round(hr_g/len(group)*100,1)}% HR rate')

    print()

    # HOME vs AWAY
    print('HOME vs AWAY:')
    for is_home, label in [(True,'Home'),(False,'Away')]:
        group=[r for r in rows if r.get('is_home')==is_home]
        if not group: continue
        hr_g=sum(1 for r in group if r['actual_hr']>=1)
        print(f'  {label}: {hr_g}/{len(group)} = '
              f'{round(hr_g/len(group)*100,1)}% HR rate')

    print()

    # RECENT FORM — were they hot?
    print('RECENT 7-GAME ISO BUCKETS:')
    for lo,hi in [(0,.05),(.05,.10),(.10,.15),(.15,.20),(.20,.30),(.30,99)]:
        group=[r for r in rows if lo<=r['recent_iso']<hi]
        if len(group)<3: continue
        hr_g=sum(1 for r in group if r['actual_hr']>=1)
        print(f'  recent ISO {lo:.2f}-{hi:.2f}: {hr_g}/{len(group)} = '
              f'{round(hr_g/len(group)*100,1)}%')

    print()

    # SUM SCORE with recent form
    print('WHAT IF WE ADD RECENT ISO TO THE SUM SCORE?')
    print('(SLG + h2h_SLG + recent_ISO — same additive principle)')
    for lo,hi in [(0,.4),(.4,.5),(.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,99)]:
        group=[r for r in rows
               if lo<=(r['slg']+r['h2h_slg']+r['recent_iso'])<hi]
        if len(group)<3: continue
        hr_g=sum(1 for r in group if r['actual_hr']>=1)
        t=len(group)
        score_key=lo
        be=round((t/hr_g-1)*100) if hr_g else 0
        print(f'  score {lo:.1f}-{hi:.1f}: {hr_g}/{t} = '
              f'{round(hr_g/t*100,1)}%'
              + (f'  breakeven +{be}' if hr_g else ''))

    print()

    # final separation including recent form
    print('SEPARATION — does adding recent ISO help the sum?')
    for key, label in [
        ('two_signal', 'SLG + h2h_SLG'),
    ]:
        three_key = [(r['slg']+r['h2h_slg']+r['recent_iso'],r['actual_hr'])
                     for r in rows]
        hr_s=[s for s,h in three_key if h>=1]
        no_s=[s for s,h in three_key if h==0]
        hs=sum(hr_s)/len(hr_s) if hr_s else 0
        ns=sum(no_s)/len(no_s) if no_s else 0
        print(f'  SLG + h2h_SLG + recent_ISO: '
              f'hitters {hs:.3f} non-hitters {ns:.3f} gap {hs-ns:+.3f}')
        hr_s2=[r['two_signal'] for r in rows if r['actual_hr']>=1]
        no_s2=[r['two_signal'] for r in rows if r['actual_hr']==0]
        hs2=sum(hr_s2)/len(hr_s2) if hr_s2 else 0
        ns2=sum(no_s2)/len(no_s2) if no_s2 else 0
        print(f'  SLG + h2h_SLG only:         '
              f'hitters {hs2:.3f} non-hitters {ns2:.3f} gap {hs2-ns2:+.3f}')

if __name__ == '__main__':
    main()
