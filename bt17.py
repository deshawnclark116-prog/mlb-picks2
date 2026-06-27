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
        slg = float(st.get('slg', 0) or 0)
        avg = float(st.get('avg', 0) or 0)
        return {
            'slg': slg,
            'avg': avg,
            'iso': slg - avg,
            'ops': float(st.get('ops', 0) or 0),
            'hr_per_pa': int(st.get('homeRuns', 0) or 0) / pa,
            'pa': pa,
        }
    except Exception:
        return None

def batter_recent_form(pid, before_date, n_games=7):
    g = get(f'{MLB}/people/{pid}/stats?stats=gameLog&group=hitting&season=2026')
    try:
        splits = g['stats'][0]['splits']
    except Exception:
        return None
    prior = [s for s in splits if s.get('date', '') < before_date]
    if len(prior) < 3: return None
    recent = prior[-n_games:]
    h=tb=hr=ab=pa=0
    for s in recent:
        st = s['stat']
        h  += int(st.get('hits', 0) or 0)
        tb += int(st.get('totalBases', 0) or 0)
        hr += int(st.get('homeRuns', 0) or 0)
        ab += int(st.get('atBats', 0) or 0)
        pa += int(st.get('plateAppearances', 0) or 0)
    if ab == 0: return None
    return {
        'recent_slg': tb / ab,
        'recent_iso': (tb - h) / ab,
        'recent_hr':  hr,
        'recent_hr_rate': hr / pa if pa else 0,
        'recent_games': len(recent),
    }

def h2h_vs_pitcher(batter_id, pitcher_id):
    d = get(f'{MLB}/people/{batter_id}/stats?stats=vsPlayer'
            f'&opposingPlayerId={pitcher_id}&group=hitting')
    ab=h=hr=tb=0
    try:
        for s in d['stats'][0]['splits']:
            if s.get('stat'):
                st = s['stat']
                ab += int(st.get('atBats', 0) or 0)
                h  += int(st.get('hits', 0) or 0)
                hr += int(st.get('homeRuns', 0) or 0)
                tb += int(st.get('totalBases', 0) or 0)
    except Exception:
        pass
    if ab >= 4:
        return {'h2h_slg': tb/ab, 'h2h_avg': h/ab,
                'h2h_hr': hr, 'h2h_ab': ab, 'has_h2h': True}
    return {'h2h_slg': 0, 'h2h_avg': 0, 'h2h_hr': 0,
            'h2h_ab': ab, 'has_h2h': False}

def get_context(gpk, pid):
    try:
        f = get(f'{MLB11}/game/{gpk}/feed/live')
        batter_side = None; order_pos = None
        for side in ('home', 'away'):
            order = f['liveData']['boxscore']['teams'][side].get('battingOrder', [])
            int_order = [int(x) for x in order if str(x).isdigit()]
            if pid in int_order:
                batter_side = side
                try: order_pos = int_order.index(pid) + 1
                except: order_pos = None
                break
        pitcher_id = None; pitcher_throws = None; batter_bats = None
        if batter_side:
            oside = 'away' if batter_side == 'home' else 'home'
            pitchers = f['liveData']['boxscore']['teams'][oside].get('pitchers', [])
            if pitchers:
                pitcher_id = pitchers[0]
                pp = f['liveData']['boxscore']['teams'][oside]['players'].get(
                    f'ID{pitcher_id}', {})
                pitcher_throws = pp.get('position', {}).get('code')
        try:
            pinfo = get(f'{MLB}/people/{pid}')
            batter_bats = pinfo['people'][0]['batSide']['code']
        except Exception:
            pass
        venue = f['gameData']['venue']['name']
        return {
            'pitcher_id': pitcher_id,
            'pitcher_throws': pitcher_throws,
            'batter_bats': batter_bats,
            'order_pos': order_pos,
            'venue': venue,
            'is_home': batter_side == 'home',
        }
    except Exception:
        return {}

def main():
    today = dt.datetime.now(ZoneInfo('America/New_York')).date()
    all_dates = [(today - dt.timedelta(days=i)).isoformat() for i in range(1, 22)]

    idx_data = get(f'{MLB}/sports/1/players?season=2026')
    idx = {norm(p.get('fullName', '')): p.get('id')
           for p in idx_data.get('people', [])}

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
            if pred.get('prop_type') not in ('batter_hits', 'batter_total_bases'):
                continue
            nm = norm(pred.get('player', ''))
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
            ctx = get_context(gpk, pid)
            h2h = h2h_vs_pitcher(pid, ctx.get('pitcher_id')) \
                  if ctx.get('pitcher_id') else \
                  {'h2h_slg':0,'h2h_avg':0,'h2h_hr':0,'h2h_ab':0,'has_h2h':False}

            # actual HR
            actual_hr = 0
            try:
                g = get(f'{MLB}/people/{pid}/stats'
                        f'?stats=gameLog&group=hitting&season=2026')
                for sp in g['stats'][0]['splits']:
                    if sp.get('date') == date:
                        actual_hr = int(sp['stat'].get('homeRuns', 0) or 0)
                        break
            except Exception:
                pass

            # platoon advantage bonus
            b = ctx.get('batter_bats', '?'); t = ctx.get('pitcher_throws', '?')
            platoon_bonus = 0.05 if (
                (b=='L' and t=='R') or (b=='R' and t=='L')) else 0.0

            # batting order bonus — positions 1,2,5,6 showed higher HR rates
            pos = ctx.get('order_pos')
            order_bonus = 0.05 if pos in (1, 2, 5, 6) else 0.0

            # home bonus
            home_bonus = 0.03 if ctx.get('is_home') else 0.0

            # THE FULL SUM SCORE — every signal added together
            full_score = (
                season['slg']          # general power
                + h2h['h2h_slg']       # hits THIS pitcher for power
                + recent['recent_iso'] # hot right now
                + platoon_bonus        # favorable matchup
                + order_bonus          # lineup spot
                + home_bonus           # home field
            )

            # also track sub-combinations for comparison
            two_sig   = season['slg'] + h2h['h2h_slg']
            three_sig = season['slg'] + h2h['h2h_slg'] + recent['recent_iso']

            rows.append({
                'date': date, 'name': nm, 'actual_hr': actual_hr,
                **season, **recent, **h2h,
                'platoon_bonus': platoon_bonus,
                'order_bonus': order_bonus,
                'home_bonus': home_bonus,
                'two_sig': two_sig,
                'three_sig': three_sig,
                'full_score': full_score,
                'order_pos': pos,
                'is_home': ctx.get('is_home'),
            })

    print(f'Files: {files_found} | Entries: {len(rows)}')
    print(f'HR games: {sum(1 for r in rows if r["actual_hr"]>=1)} | '
          f'Non-HR: {sum(1 for r in rows if r["actual_hr"]==0)}')
    print()

    # SEPARATION — all versions head to head
    print('SEPARATION TEST — all versions:')
    for key, label in [
        ('two_sig',    'SLG + h2h_SLG (2-signal)'),
        ('three_sig',  'SLG + h2h_SLG + recent_ISO (3-signal)'),
        ('full_score', 'ALL signals combined (full score)'),
    ]:
        hr_s  = [r[key] for r in rows if r['actual_hr'] >= 1]
        no_s  = [r[key] for r in rows if r['actual_hr'] == 0]
        hs = sum(hr_s)/len(hr_s) if hr_s else 0
        ns = sum(no_s)/len(no_s) if no_s else 0
        print(f'  {label:45} gap {hs-ns:+.3f}  '
              f'(hitters {hs:.3f} vs non {ns:.3f})')

    print()

    # BUCKET TEST — full score
    print('FULL SCORE BUCKETS:')
    edges = [(0,.4),(.4,.5),(.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.0),(1.0,99)]
    for lo, hi in edges:
        group = [r for r in rows if lo <= r['full_score'] < hi]
        if len(group) < 3: continue
        hr_g = sum(1 for r in group if r['actual_hr'] >= 1)
        t = len(group)
        be = round((t/hr_g-1)*100) if hr_g else 0
        print(f'  score {lo:.1f}-{hi:.1f}: {hr_g}/{t} = '
              f'{round(hr_g/t*100,1)}%'
              + (f'  breakeven +{be}' if hr_g else ''))

    print()

    # MULTI-WINDOW — full score vs 2-signal at same threshold
    print('MULTI-WINDOW (full score >= 0.85 vs 2-signal >= 0.80):')
    for i in range(0, 20, 5):
        w_dates = all_dates[i:i+5]
        f_above = [r for r in rows if r['date'] in w_dates
                   and r['full_score'] >= 0.85]
        t_above = [r for r in rows if r['date'] in w_dates
                   and r['two_sig'] >= 0.80]
        f_below = [r for r in rows if r['date'] in w_dates
                   and r['full_score'] < 0.85]
        if not f_above and not t_above: continue
        fh = sum(1 for r in f_above if r['actual_hr']>=1)
        th = sum(1 for r in t_above if r['actual_hr']>=1)
        fb = sum(1 for r in f_below if r['actual_hr']>=1)
        print(f'  Days {i+1}-{i+5}:  '
              f'full>=0.85 {fh}/{len(f_above)}='
              f'{round(fh/len(f_above)*100,1) if f_above else 0}%  '
              f'| 2-sig>=0.80 {th}/{len(t_above)}='
              f'{round(th/len(t_above)*100,1) if t_above else 0}%  '
              f'| below {fb}/{len(f_below)}='
              f'{round(fb/len(f_below)*100,1) if f_below else 0}%')

    print()

    # WHO DOES THE FULL SCORE PICK? — see exactly who gets flagged
    flagged = sorted([r for r in rows if r['full_score']>=0.85],
                     key=lambda x: -x['full_score'])
    print(f'FLAGGED by full score >= 0.85 ({len(flagged)} picks):')
    print(f'  {"name":22} {"score":6} {"HR?":5} '
          f'{"slg":5} {"h2h":5} {"rec_iso":7} '
          f'{"plat":5} {"ord":4} {"home":5}')
    for r in flagged:
        hr_marker = 'HR ✓' if r['actual_hr']>=1 else 'no'
        print(f'  {r["name"][:20]:20}  '
              f'{r["full_score"]:.3f}  {hr_marker:5}  '
              f'{r["slg"]:.3f} {r["h2h_slg"]:.3f} '
              f'{r["recent_iso"]:.3f}    '
              f'{r["platoon_bonus"]:.2f}  '
              f'{r["order_bonus"]:.2f}  '
              f'{"Y" if r["is_home"] else "N"}')

    print()
    print('HONEST VERDICT:')
    full_flag = [r for r in rows if r['full_score']>=0.85]
    two_flag  = [r for r in rows if r['two_sig']>=0.80]
    fh = sum(1 for r in full_flag if r['actual_hr']>=1)
    th = sum(1 for r in two_flag  if r['actual_hr']>=1)
    print(f'  2-signal >= 0.80:    {th}/{len(two_flag)} = '
          f'{round(th/len(two_flag)*100,1) if two_flag else 0}%')
    print(f'  Full score >= 0.85:  {fh}/{len(full_flag)} = '
          f'{round(fh/len(full_flag)*100,1) if full_flag else 0}%')
    print()
    print('If full score hits HIGHER % than 2-signal: everything adds value')
    print('If similar or lower: the bonus signals are noise, stick with 3')

if __name__ == '__main__':
    main()
