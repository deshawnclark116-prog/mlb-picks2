import urllib.request, json, datetime as dt, os
from zoneinfo import ZoneInfo
import sys
sys.path.insert(0, '/home/render/project/src')

MLB = 'https://statsapi.mlb.com/api/v1'
MLB11 = 'https://statsapi.mlb.com/api/v1.1'
GH = 'https://deshawnclark116-prog.github.io/mlb-picks2'
DATA_DIR = '/data/predictions'

def get(u):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(u, headers={'User-Agent': 'x'}), timeout=30).read())

def norm(s):
    return ''.join(c for c in s.lower() if c.isalpha() or c == ' ').strip()

def batter_season_slg(pid):
    g = get(f'{MLB}/people/{pid}/stats?stats=season&group=hitting&season=2026')
    try:
        st = g['stats'][0]['splits'][0]['stat']
        pa = int(st.get('plateAppearances', 0) or 0)
        if pa < 20: return None
        return float(st.get('slg', 0) or 0)
    except Exception:
        return None

def recent_iso(pid, before_date, n_games=7):
    g = get(f'{MLB}/people/{pid}/stats?stats=gameLog&group=hitting&season=2026')
    try:
        splits = g['stats'][0]['splits']
    except Exception:
        return None
    prior = [s for s in splits if s.get('date', '') < before_date]
    if len(prior) < 3: return None
    recent = prior[-n_games:]
    h=tb=ab=0
    for s in recent:
        st = s['stat']
        h  += int(st.get('hits', 0) or 0)
        tb += int(st.get('totalBases', 0) or 0)
        ab += int(st.get('atBats', 0) or 0)
    if ab == 0: return None
    return (tb - h) / ab

def h2h_slg(batter_id, pitcher_id):
    d = get(f'{MLB}/people/{batter_id}/stats?stats=vsPlayer'
            f'&opposingPlayerId={pitcher_id}&group=hitting')
    ab=tb=0
    try:
        for s in d['stats'][0]['splits']:
            if s.get('stat'):
                st = s['stat']
                ab += int(st.get('atBats', 0) or 0)
                tb += int(st.get('totalBases', 0) or 0)
    except Exception:
        pass
    return tb/ab if ab >= 4 else 0

def get_starter(gpk, pid):
    try:
        f = get(f'{MLB11}/game/{gpk}/feed/live')
        for side in ('home', 'away'):
            order = f['liveData']['boxscore']['teams'][side].get('battingOrder', [])
            if pid in [int(x) for x in order if str(x).isdigit()]:
                oside = 'away' if side == 'home' else 'home'
                pitchers = f['liveData']['boxscore']['teams'][oside].get('pitchers', [])
                return pitchers[0] if pitchers else None
    except Exception:
        pass
    return None

def get_actual_hr(pid, date):
    try:
        g = get(f'{MLB}/people/{pid}/stats?stats=gameLog&group=hitting&season=2026')
        for sp in g['stats'][0]['splits']:
            if sp.get('date') == date:
                return int(sp['stat'].get('homeRuns', 0) or 0)
    except Exception:
        pass
    return 0

def load_preds():
    all_preds = {}
    if os.path.exists(DATA_DIR):
        for fname in sorted(os.listdir(DATA_DIR)):
            if fname.startswith('predictions_') and fname.endswith('.json'):
                date = fname.replace('predictions_','').replace('.json','')
                try:
                    with open(os.path.join(DATA_DIR, fname)) as f:
                        all_preds[date] = json.load(f)
                except Exception:
                    pass
    today = dt.datetime.now(ZoneInfo('America/New_York')).date()
    for i in range(1, 60):
        date = (today - dt.timedelta(days=i)).isoformat()
        if date in all_preds: continue
        try:
            all_preds[date] = get(f'{GH}/predictions_{date}.json')
        except Exception:
            pass
    return all_preds

def main():
    print('Loading predictions...')
    all_preds = load_preds()
    dates = sorted(all_preds.keys())
    print(f'Dates: {len(dates)} ({dates[0]} through {dates[-1]})')
    print()

    idx_data = get(f'{MLB}/sports/1/players?season=2026')
    idx = {norm(p.get('fullName','')): p.get('id')
           for p in idx_data.get('people', [])}

    rows = []
    seen = set()
    for date, preds in sorted(all_preds.items()):
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
            slg = batter_season_slg(pid)
            if slg is None: continue
            r_iso = recent_iso(pid, date)
            if r_iso is None: continue
            starter = get_starter(gpk, pid)
            h2h = h2h_slg(pid, starter) if starter else 0
            actual_hr = get_actual_hr(pid, date)
            score = slg + h2h + r_iso
            rows.append({
                'date': date, 'name': nm,
                'slg': slg, 'h2h': h2h, 'r_iso': r_iso,
                'score': score, 'actual_hr': actual_hr,
            })

    total = len(rows)
    hr_total = sum(1 for r in rows if r['actual_hr'] >= 1)
    print(f'Entries: {total} | HR games: {hr_total} '
          f'({round(hr_total/total*100,1) if total else 0}%)')
    print()

    # OVERALL at both thresholds
    for thresh, label in [(1.30,'1.30'),(1.35,'1.35')]:
        above = [r for r in rows if r['score'] >= thresh]
        below = [r for r in rows if r['score'] < thresh]
        ah = sum(1 for r in above if r['actual_hr'] >= 1)
        bh = sum(1 for r in below if r['actual_hr'] >= 1)
        a_pct = round(ah/len(above)*100,1) if above else 0
        b_pct = round(bh/len(below)*100,1) if below else 0
        be = round((len(above)/ah-1)*100) if ah else 0
        ppd = round(len(above)/len(dates),1) if dates else 0
        print(f'OVERALL score >= {label}:')
        print(f'  Above: {ah}/{len(above)} = {a_pct}%  '
              f'breakeven +{be}  {ppd} picks/day')
        print(f'  Below: {bh}/{len(below)} = {b_pct}%')
        print()

    # MULTI-WINDOW — 4 non-overlapping windows, both thresholds
    print('='*60)
    print('MULTI-WINDOW: 4 non-overlapping windows')
    print('='*60)
    window_size = len(dates)//4
    for thresh, label in [(1.30,'1.30'),(1.35,'1.35')]:
        print(f'\nThreshold >= {label}:')
        wins = 0; tested = 0
        for i in range(4):
            w_dates = set(dates[i*window_size:(i+1)*window_size])
            if not w_dates: continue
            above = [r for r in rows if r['date'] in w_dates
                     and r['score'] >= thresh]
            below = [r for r in rows if r['date'] in w_dates
                     and r['score'] < thresh]
            if len(above) < 3: continue
            tested += 1
            ah = sum(1 for r in above if r['actual_hr'] >= 1)
            bh = sum(1 for r in below if r['actual_hr'] >= 1)
            a_pct = round(ah/len(above)*100,1)
            b_pct = round(bh/len(below)*100,1) if below else 0
            win = a_pct > b_pct
            if win: wins += 1
            print(f'  Window {i+1} ({min(w_dates)}-{max(w_dates)}): '
                  f'above {ah}/{len(above)}={a_pct}%  '
                  f'below {bh}/{len(below)}={b_pct}%  '
                  f'{"WIN ✓" if win else "loss X"}')
        print(f'  → {wins}/{tested} windows won')

    print()

    # SIDE BY SIDE comparison at the threshold line
    print('SIDE BY SIDE — what does each threshold pick?')
    print(f'  {"threshold":8} {"picks":8} {"HRs":6} {"rate":8} '
          f'{"per_day":10} {"breakeven"}')
    for thresh in [1.20, 1.25, 1.30, 1.35, 1.40, 1.50]:
        group = [r for r in rows if r['score'] >= thresh]
        if len(group) < 3: continue
        hr_g = sum(1 for r in group if r['actual_hr'] >= 1)
        t = len(group)
        pct = round(hr_g/t*100,1)
        be = round((t/hr_g-1)*100) if hr_g else 0
        ppd = round(t/len(dates),1)
        profit = '← PROFIT' if pct >= 25 else ''
        print(f'  >={thresh:.2f}:  {t:5}  {hr_g:4}   {pct:5}%   '
              f'{ppd}/day     +{be}  {profit}')

    print()

    # WHO are the 1.30+ batters — see the actual picks
    flagged_130 = sorted([r for r in rows if r['score']>=1.30],
                         key=lambda x:-x['score'])
    flagged_135 = [r for r in flagged_130 if r['score']>=1.35]
    print(f'PICKS at 1.30+ ({len(flagged_130)} total, '
          f'{len(flagged_135)} also clear 1.35):')
    print(f'  {"name":22} {"score":7} {"HR?":6} '
          f'{"slg":6} {"h2h":6} {"r_iso":6}')
    for r in flagged_130:
        marker = 'HR ✓' if r['actual_hr']>=1 else 'no'
        border = ' ← 1.35+' if r['score']>=1.35 else ''
        print(f'  {r["name"][:20]:20}  '
              f'{r["score"]:.3f}  {marker:5}  '
              f'{r["slg"]:.3f}  {r["h2h"]:.3f}  '
              f'{r["r_iso"]:.3f}{border}')

if __name__ == '__main__':
    main()
    
