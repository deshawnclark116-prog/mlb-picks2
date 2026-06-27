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

def hr_rate(pid):
    g = get(f'{MLB}/people/{pid}/stats?stats=season&group=hitting&season=2026')
    try:
        st = g['stats'][0]['splits'][0]['stat']
        pa = int(st.get('plateAppearances', 0) or 0)
        hr = int(st.get('homeRuns', 0) or 0)
        ab = int(st.get('atBats', 0) or 0)
        avg = float(st.get('avg', 0) or 0)
        slg = float(st.get('slg', 0) or 0)
        return hr/pa if pa >= 20 else 0, pa, hr, avg, slg
    except Exception:
        return 0, 0, 0, 0, 0

def get_pitcher_for_game(gpk, batter_side):
    """Find opposing pitcher from boxscore."""
    f = get(f'{MLB11}/game/{gpk}/feed/live')
    oside = 'away' if batter_side == 'home' else 'home'
    pitchers = f['liveData']['boxscore']['teams'][oside].get('pitchers', [])
    if pitchers:
        return pitchers[0]  # starter
    return None

def find_batter_side(gpk, pid):
    """Find which side the batter was on."""
    f = get(f'{MLB11}/game/{gpk}/feed/live')
    for side in ('home', 'away'):
        order = f['liveData']['boxscore']['teams'][side].get('battingOrder', [])
        if pid in order or str(pid) in [str(x) for x in order]:
            return side, f
    return None, f

def main():
    valid = ['2026-06-19','2026-06-20','2026-06-21','2026-06-22',
             '2026-06-23','2026-06-24','2026-06-25','2026-06-26']

    # the 9 batters who went yard from bt10 output
    # we'll find them from the predictions files + record
    rec = get('https://prop-edge-api.onrender.com/record')

    # find all picked batters who hit an HR
    hr_batters = []
    for x in rec.get('results', []):
        if (x['prop_type'] in ('batter_hits', 'batter_total_bases')
                and x.get('date', '') in valid
                and float(x.get('actual', 0)) > 0):
            # check if they hit an HR that game
            nm = norm(x.get('player', ''))
            date = x['date']
            hr_batters.append((date, nm, x.get('prop_type')))

    # also grab from predictions to get game_id
    pred_lookup = {}  # (date, norm_name) -> pred
    for date in valid:
        try:
            preds = get(f'{GH}/predictions_{date}.json')
            for p in preds:
                if p.get('prop_type') in ('batter_hits', 'batter_total_bases'):
                    pred_lookup[(date, norm(p.get('player', '')))] = p
        except Exception:
            pass

    # build player index
    idx_data = get(f'{MLB}/sports/1/players?season=2026')
    idx = {norm(p.get('fullName', '')): p.get('id')
           for p in idx_data.get('people', [])}

    # the HR hitters we found in bt10
    hr_names = [
        ('2026-06-19', 'nolan schanuel'),
        ('2026-06-19', 'caleb durbin'),
        ('2026-06-19', 'denzer guzman'),
        ('2026-06-19', 'jacob wilson'),
        ('2026-06-20', 'pete crow armstrong'),
        ('2026-06-21', 'masyn winn'),
        ('2026-06-24', 'otto lopez'),
        ('2026-06-24', 'wyatt langford'),
        ('2026-06-25', 'wyatt langford'),
    ]

    print('Checking existing signals on the 9 HR batters:')
    print()
    print(f'{"name":22} {"hr_rate":8} {"PA":5} {"HR":4} {"AVG":5} {"SLG":5} '
          f'{"power_flag":12} {"h2h_hr":8} {"would_flag"}')
    print('-' * 90)

    # for comparison - also check non-HR batters
    non_hr_hr_rates = []
    hr_batter_rates = []

    for date, nm in hr_names:
        pid = idx.get(nm)
        if not pid:
            # try partial match
            for k, v in idx.items():
                if nm in k or k in nm:
                    pid = v; break
        if not pid:
            print(f'  {nm[:22]:22} NO PLAYER ID FOUND')
            continue

        rate, pa, hr, avg, slg = hr_rate(pid)
        hr_batter_rates.append(rate)

        # get game_id from predictions
        pred = pred_lookup.get((date, nm))
        gpk = pred.get('game_id') if pred else None

        # power_flag from bvp — need opposing pitcher
        pflag = None
        h2h_hr = 0
        if gpk:
            try:
                side, f = find_batter_side(gpk, pid)
                if side:
                    oside = 'away' if side == 'home' else 'home'
                    pitchers = f['liveData']['boxscore']['teams'][oside].get('pitchers', [])
                    if pitchers:
                        starter_pid = pitchers[0]
                        bv = bvp.batter_vs_pitcher(pid, starter_pid)
                        if bv:
                            pflag = bvp.power_flag(bv)
                            h2h_hr = bv.get('home_runs', 0)
            except Exception as e:
                pflag = f'err: {str(e)[:20]}'

        # would the system have flagged this guy?
        # flag = hr_rate >= 0.04 (roughly 1 HR per 25 PA) OR power_flag == 'power'
        would_flag = (rate >= 0.04) or (pflag == 'power')

        print(f'  {nm[:20]:20} {rate:.4f}   {pa:4}  {hr:3}  '
              f'{avg:.3f} {slg:.3f} '
              f'{str(pflag):12} {h2h_hr:6}HR   '
              f'{"YES ✓" if would_flag else "no"}')

    print()
    print('=' * 60)

    # now check non-HR batters for comparison
    print()
    print('Checking non-HR batters from same dates (comparison group):')
    non_hr_rates = []
    checked = 0
    for (date, nm), pred in pred_lookup.items():
        if date not in valid: continue
        if (date, nm) in [(d, n) for d, n in hr_names]: continue
        if checked >= 20: break  # sample of 20 non-HR batters
        pid = idx.get(nm)
        if not pid: continue
        rate, pa, hr, avg, slg = hr_rate(pid)
        if pa < 20: continue
        non_hr_rates.append(rate)
        checked += 1

    if hr_batter_rates and non_hr_rates:
        print()
        print('COMPARISON:')
        print(f'  HR batters avg hr_rate:     {sum(hr_batter_rates)/len(hr_batter_rates):.4f} '
              f'({round(sum(hr_batter_rates)/len(hr_batter_rates)*100,1)}% per PA)')
        print(f'  Non-HR batters avg hr_rate: {sum(non_hr_rates)/len(non_hr_rates):.4f} '
              f'({round(sum(non_hr_rates)/len(non_hr_rates)*100,1)}% per PA)')
        print()
        flagged = sum(1 for r in hr_batter_rates if r >= 0.04)
        print(f'  Of the 9 HR batters, {flagged} had hr_rate >= 0.04 '
              f'(the power threshold)')
        print(f'  That means {flagged}/9 would have been flagged by hr_rate alone')
        print()
        print('KEY QUESTION: is the hr_rate signal already separating')
        print('these guys from the non-HR batters?')

if __name__ == '__main__':
    main()
