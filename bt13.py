import urllib.request, json, datetime as dt
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

def get_batter_stats(pid):
    """Season SLG, HR rate, AVG."""
    g = get(f'{MLB}/people/{pid}/stats?stats=season&group=hitting&season=2026')
    try:
        st = g['stats'][0]['splits'][0]['stat']
        pa = int(st.get('plateAppearances', 0) or 0)
        hr = int(st.get('homeRuns', 0) or 0)
        slg = float(st.get('slg', 0) or 0)
        avg = float(st.get('avg', 0) or 0)
        if pa >= 20:
            return hr/pa, slg, avg, pa
    except Exception:
        pass
    return 0, 0, 0, 0

def main():
    # go back further — use all available prediction files
    today = dt.datetime.now(ZoneInfo('America/New_York')).date()
    all_dates = [(today - dt.timedelta(days=i)).isoformat() for i in range(1, 22)]

    print(f'Testing SLG as HR signal across up to {len(all_dates)} days')
    print('Pulling all available prediction files...')
    print()

    # build player index once
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

            hr_per_pa, slg, avg, pa = get_batter_stats(pid)
            if pa < 20: continue

            # actual HR that game
            actual_hr = 0
            try:
                g = get(f'{MLB}/people/{pid}/stats'
                        f'?stats=gameLog&group=hitting&season=2026')
                for sp in g['stats'][0]['splits']:
                    if sp.get('date') == date:
                        actual_hr = int(sp['stat'].get('homeRuns',0) or 0)
                        break
            except Exception:
                pass

            rows.append({
                'date': date, 'name': nm,
                'slg': slg, 'hr_per_pa': hr_per_pa,
                'avg': avg, 'pa': pa,
                'actual_hr': actual_hr,
            })

    print(f'Files found: {files_found}')
    print(f'Total batter-game entries: {len(rows)}')
    print()

    # SLG tiers
    tiers = [
        ('SLG .500+',     .50, 99.0),
        ('SLG .475-.499', .475, .50),
        ('SLG .450-.474', .45, .475),
        ('SLG .425-.449', .425, .45),
        ('SLG .400-.424', .40, .425),
        ('SLG .375-.399', .375, .40),
        ('SLG <.375',     0,   .375),
    ]

    print('HR RATE BY SLG TIER (our picked batters):')
    print(f'  {"tier":20} {"hr_games":10} {"total":8} {"hr_rate":10} {"avg_odds_needed"}')
    print('-' * 70)
    for label, lo, hi in tiers:
        group = [r for r in rows if lo <= r['slg'] < hi]
        if len(group) < 3: continue
        hr_games = sum(1 for r in group if r['actual_hr'] >= 1)
        t = len(group)
        hr_pct = round(hr_games/t*100, 1)
        # what american odds would break even at this hit rate
        if hr_games > 0:
            breakeven = round((t/hr_games - 1) * 100)
            be_str = f'+{breakeven}'
        else:
            be_str = 'n/a'
        print(f'  {label:20} {hr_games:5}/{t:<6} = {hr_pct:5}%    '
              f'breakeven: {be_str}')

    print()
    # the key cutoff test
    above = [r for r in rows if r['slg'] >= 0.450]
    below = [r for r in rows if r['slg'] < 0.450]
    a_hr = sum(1 for r in above if r['actual_hr'] >= 1)
    b_hr = sum(1 for r in below if r['actual_hr'] >= 1)
    print(f'KEY CUTOFF — SLG .450+:')
    print(f'  Above .450: {a_hr}/{len(above)} = '
          f'{round(a_hr/len(above)*100,1) if above else 0}% HR rate')
    print(f'  Below .450: {b_hr}/{len(below)} = '
          f'{round(b_hr/len(below)*100,1) if below else 0}% HR rate')
    print(f'  Gap: +{round(a_hr/len(above)*100,1 if above else 0) - round(b_hr/len(below)*100,1 if below else 0):.1f} points')
    print()

    # multiple non-overlapping windows — the discipline check
    print('WINDOW BREAKDOWN (does it hold across different weeks?):')
    window_size = 5
    for i in range(0, min(len(all_dates), 20), window_size):
        window_dates = all_dates[i:i+window_size]
        w_above = [r for r in rows if r['date'] in window_dates and r['slg']>=0.450]
        w_below = [r for r in rows if r['date'] in window_dates and r['slg']<0.450]
        if len(w_above) < 3: continue
        wah = sum(1 for r in w_above if r['actual_hr']>=1)
        wbh = sum(1 for r in w_below if r['actual_hr']>=1)
        print(f'  Days {i+1}-{i+window_size}: '
              f'above .450 {wah}/{len(w_above)}='
              f'{round(wah/len(w_above)*100,1) if w_above else 0}%  '
              f'below .450 {wbh}/{len(w_below)}='
              f'{round(wbh/len(w_below)*100,1) if w_below else 0}%')

    print()
    # false positive / selectivity
    print(f'SELECTIVITY:')
    print(f'  System picks SLG .450+ batters '
          f'{len(above)}/{len(rows)} = '
          f'{round(len(above)/len(rows)*100,1) if rows else 0}% of the time')
    print(f'  So roughly {round(len(above)/files_found,1) if files_found else 0} '
          f'SLG .450+ batter picks per day on average')
    print()
    print('CONTEXT: Typical FanDuel HR prop is +300 to +500')
    print('Breakeven at +300 = 25.0%,  +350 = 22.2%,  +400 = 20.0%')
    if above:
        hr_pct = a_hr/len(above)*100
        print(f'Our SLG .450+ HR rate: {hr_pct:.1f}%')
        if hr_pct >= 25:
            print('→ PROFITABLE at +300 or better odds')
        elif hr_pct >= 20:
            print('→ PROFITABLE at +400 or better odds')
        else:
            print('→ Need better odds or tighter filter to be profitable')

if __name__ == '__main__':
    main()
