import urllib.request, json, datetime as dt
from zoneinfo import ZoneInfo
import sys
sys.path.insert(0, '/home/render/project/src')
import bvp

MLB = 'https://statsapi.mlb.com/api/v1'
MLB11 = 'https://statsapi.mlb.com/api/v1.1'
GH = 'https://deshawnclark116-prog.github.io/mlb-picks2'

def get(u):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(u, headers={'User-Agent': 'x'}), timeout=30).read())

def norm(s):
    return ''.join(c for c in s.lower() if c.isalpha() or c == ' ').strip()

def get_hr_rate(pid):
    g = get(f'{MLB}/people/{pid}/stats?stats=season&group=hitting&season=2026')
    try:
        st = g['stats'][0]['splits'][0]['stat']
        pa = int(st.get('plateAppearances', 0) or 0)
        hr = int(st.get('homeRuns', 0) or 0)
        slg = float(st.get('slg', 0) or 0)
        if pa >= 20:
            return hr/pa, slg
    except Exception:
        pass
    return 0, 0

def main():
    valid = ['2026-06-19','2026-06-20','2026-06-21','2026-06-22',
             '2026-06-23','2026-06-24','2026-06-25','2026-06-26']

    # build player index once
    idx_data = get(f'{MLB}/sports/1/players?season=2026')
    idx = {norm(p.get('fullName','')): p.get('id')
           for p in idx_data.get('people',[])}

    # load record for actual HRs
    rec = get('https://prop-edge-api.onrender.com/record')
    actual_lookup = {}
    for x in rec.get('results',[]):
        if x.get('date','') in valid:
            actual_lookup[(x['date'], norm(x.get('player','')))] = x

    rows = []
    seen = set()

    for date in valid:
        try:
            preds = get(f'{GH}/predictions_{date}.json')
        except Exception:
            continue

        for pred in preds:
            if pred.get('prop_type') not in ('batter_hits','batter_total_bases'):
                continue
            nm = norm(pred.get('player',''))
            gpk = pred.get('game_id')
            if not nm or not gpk: continue
            if (date, nm) in seen: continue
            seen.add((date, nm))

            pid = idx.get(nm)
            if not pid: continue

            # get season hr_rate + slg
            hr_per_pa, slg = get_hr_rate(pid)

            # get power_flag vs opposing starter
            pflag = None
            try:
                f = get(f'{MLB11}/game/{gpk}/feed/live')
                batter_side = None
                for side in ('home','away'):
                    order = f['liveData']['boxscore']['teams'][side].get('battingOrder',[])
                    if pid in order or pid in [int(x) for x in order if str(x).isdigit()]:
                        batter_side = side; break
                if batter_side:
                    oside = 'away' if batter_side == 'home' else 'home'
                    pitchers = f['liveData']['boxscore']['teams'][oside].get('pitchers',[])
                    if pitchers:
                        bv = bvp.batter_vs_pitcher(pid, pitchers[0])
                        if bv:
                            pflag = bvp.power_flag(bv)
            except Exception:
                pass

            # get actual HR from game log
            actual_hr = 0
            try:
                g = get(f'{MLB}/people/{pid}/stats?stats=gameLog&group=hitting&season=2026')
                for sp in g['stats'][0]['splits']:
                    if sp.get('date') == date:
                        actual_hr = int(sp['stat'].get('homeRuns',0) or 0)
                        break
            except Exception:
                pass

            # combined flag
            flagged = (hr_per_pa >= 0.04) or (pflag == 'power')

            rows.append({
                'date': date, 'name': nm,
                'hr_per_pa': hr_per_pa,
                'slg': slg,
                'pflag': pflag,
                'flagged': flagged,
                'actual_hr': actual_hr,
            })

    print(f'Total unique batter-game entries: {len(rows)}')
    print()

    flagged = [r for r in rows if r['flagged']]
    not_flagged = [r for r in rows if not r['flagged']]

    def summary(group, label):
        if not group: return
        total = len(group)
        hr_games = sum(1 for r in group if r['actual_hr'] >= 1)
        total_hr = sum(r['actual_hr'] for r in group)
        print(f'{label} ({total} batter-games):')
        print(f'  {hr_games}/{total} = {round(hr_games/total*100,1)}% '
              f'had a HR game')
        print(f'  {total_hr} total HRs')
        print()

    summary(flagged, 'FLAGGED (hr_rate>=0.04 OR power_flag=power)')
    summary(not_flagged, 'NOT FLAGGED')

    # break flagged down by which signal fired
    hr_only = [r for r in rows if r['hr_per_pa']>=0.04 and r['pflag']!='power']
    power_only = [r for r in rows if r['hr_per_pa']<0.04 and r['pflag']=='power']
    both = [r for r in rows if r['hr_per_pa']>=0.04 and r['pflag']=='power']
    neither = not_flagged

    print('BREAKDOWN BY SIGNAL:')
    for group, label in [
        (both,       'BOTH signals (hr_rate + power_flag)'),
        (hr_only,    'hr_rate >= 0.04 only'),
        (power_only, 'power_flag only'),
        (neither,    'NEITHER signal'),
    ]:
        if not group: continue
        hr_games = sum(1 for r in group if r['actual_hr']>=1)
        t = len(group)
        print(f'  {label}: {hr_games}/{t} = '
              f'{round(hr_games/t*100,1) if t else 0}% HR rate')

    print()
    # SLG as an additional signal
    print('SLG BREAKDOWN (all picked batters):')
    buckets = [('.500+', .5, 99), ('.450-.499', .45, .5),
               ('.400-.449', .4, .45), ('<.400', 0, .4)]
    for label, lo, hi in buckets:
        group = [r for r in rows if lo <= r['slg'] < hi]
        if not group: continue
        hr_games = sum(1 for r in group if r['actual_hr']>=1)
        t = len(group)
        print(f'  SLG {label}: {hr_games}/{t} = {round(hr_games/t*100,1)}%')

    print()
    print('FALSE POSITIVE RATE:')
    print(f'  System flags {len(flagged)}/{len(rows)} = '
          f'{round(len(flagged)/len(rows)*100,1) if rows else 0}% of picked batters')
    print(f'  Of those flagged, {sum(1 for r in flagged if r["actual_hr"]>=1)}'
          f'/{len(flagged)} actually hit a HR')

if __name__ == '__main__':
    main()
