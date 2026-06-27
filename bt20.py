import urllib.request, json, datetime as dt, os, math
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
            'pa': pa,
        }
    except Exception:
        return None

def recent_stats(pid, before_date, n_games=7):
    g = get(f'{MLB}/people/{pid}/stats?stats=gameLog&group=hitting&season=2026')
    try:
        splits = g['stats'][0]['splits']
    except Exception:
        return None
    prior = [s for s in splits if s.get('date', '') < before_date]
    if len(prior) < 3: return None
    recent = prior[-n_games:]
    h=tb=ab=hr=0
    for s in recent:
        st = s['stat']
        h  += int(st.get('hits', 0) or 0)
        tb += int(st.get('totalBases', 0) or 0)
        ab += int(st.get('atBats', 0) or 0)
        hr += int(st.get('homeRuns', 0) or 0)
    if ab == 0: return None
    return {
        'recent_avg': h/ab,
        'recent_slg': tb/ab,
        'recent_iso': (tb-h)/ab,
        'recent_hr': hr,
    }

def h2h_stats(batter_id, pitcher_id):
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
        return {
            'h2h_avg': h/ab,
            'h2h_slg': tb/ab,
            'h2h_hr': hr,
            'h2h_ab': ab,
        }
    return {'h2h_avg': 0, 'h2h_slg': 0, 'h2h_hr': 0, 'h2h_ab': 0}

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

def get_actual_stats(pid, date):
    try:
        g = get(f'{MLB}/people/{pid}/stats?stats=gameLog&group=hitting&season=2026')
        for sp in g['stats'][0]['splits']:
            if sp.get('date') == date:
                st = sp['stat']
                return {
                    'hr': int(st.get('homeRuns', 0) or 0),
                    'hits': int(st.get('hits', 0) or 0),
                    'tb': int(st.get('totalBases', 0) or 0),
                }
    except Exception:
        pass
    return None

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

            season = batter_season_stats(pid)
            if not season: continue

            recent = recent_stats(pid, date)
            if not recent: continue

            starter = get_starter(gpk, pid)
            h2h = h2h_stats(pid, starter) if starter else \
                  {'h2h_avg':0,'h2h_slg':0,'h2h_hr':0,'h2h_ab':0}

            actual = get_actual_stats(pid, date)
            if not actual: continue

            # the three scores we're testing
            hr_score    = season['slg'] + h2h['h2h_slg'] + recent['recent_iso']
            hit_score   = season['avg'] + h2h['h2h_avg']   # existing sum-score
            hit_score_3 = season['avg'] + h2h['h2h_avg'] + recent['recent_avg']  # + recent form
            tb_score    = season['slg'] + h2h['h2h_slg']   # existing 2-signal
            tb_score_3  = season['slg'] + h2h['h2h_slg'] + recent['recent_iso']  # + recent ISO

            rows.append({
                'date': date, 'name': nm,
                **season, **recent, **h2h,
                'hr_score': hr_score,
                'hit_score': hit_score,
                'hit_score_3': hit_score_3,
                'tb_score': tb_score,
                'tb_score_3': tb_score_3,
                'actual_hr': actual['hr'],
                'actual_hits': actual['hits'],
                'actual_tb': actual['tb'],
                'got_hit': 1 if actual['hits'] >= 1 else 0,
                'got_tb': 1 if actual['tb'] >= 2 else 0,  # 2+ TB
                'got_hr': 1 if actual['hr'] >= 1 else 0,
            })

    total = len(rows)
    print(f'Total entries: {total}')
    print(f'  Got a hit:   {sum(r["got_hit"] for r in rows)}/{total} = '
          f'{round(sum(r["got_hit"] for r in rows)/total*100,1)}%')
    print(f'  Got 2+ TB:   {sum(r["got_tb"] for r in rows)}/{total} = '
          f'{round(sum(r["got_tb"] for r in rows)/total*100,1)}%')
    print(f'  Got a HR:    {sum(r["got_hr"] for r in rows)}/{total} = '
          f'{round(sum(r["got_hr"] for r in rows)/total*100,1)}%')
    print()

    def separation(rows, score_key, outcome_key):
        hr_s  = [r[score_key] for r in rows if r[outcome_key]==1]
        no_s  = [r[score_key] for r in rows if r[outcome_key]==0]
        hs = sum(hr_s)/len(hr_s) if hr_s else 0
        ns = sum(no_s)/len(no_s) if no_s else 0
        return hs-ns, hs, ns

    def bucket_test(rows, score_key, outcome_key, edges, label):
        print(f'{label}:')
        for lo, hi in edges:
            group = [r for r in rows if lo <= r[score_key] < hi]
            if len(group) < 3: continue
            hits = sum(r[outcome_key] for r in group)
            t = len(group)
            print(f'  score {lo:.2f}-{hi:.2f}: '
                  f'{hits}/{t} = {round(hits/t*100,1)}%')
        print()

    # ================================================================
    # TEST 1: Does HR score also predict HITS?
    # ================================================================
    print('='*60)
    print('TEST 1: HR SCORE AS A HIT PREDICTOR')
    print('='*60)
    gap, hs, ns = separation(rows, 'hr_score', 'got_hit')
    gap2, hs2, ns2 = separation(rows, 'hit_score', 'got_hit')
    gap3, hs3, ns3 = separation(rows, 'hit_score_3', 'got_hit')
    print(f'Separation on HITS:')
    print(f'  Existing hit sum-score (avg+h2h_avg):        gap {gap2:+.3f}')
    print(f'  Hit sum-score + recent avg (3-signal):       gap {gap3:+.3f}')
    print(f'  HR score (slg+h2h_slg+recent_iso):           gap {gap:+.3f}')
    print()
    bucket_test(rows, 'hr_score', 'got_hit',
                [(0,.6),(.6,.8),(.8,1.0),(1.0,1.2),(1.2,1.4),(1.4,99)],
                'Hit rate by HR score tier')

    # ================================================================
    # TEST 2: Does HR score predict TOTAL BASES (2+)?
    # ================================================================
    print('='*60)
    print('TEST 2: HR SCORE AS A TOTAL BASES PREDICTOR (2+ TB)')
    print('='*60)
    gap, hs, ns = separation(rows, 'tb_score_3', 'got_tb')
    gap2, hs2, ns2 = separation(rows, 'tb_score', 'got_tb')
    print(f'Separation on 2+ TB:')
    print(f'  2-signal TB score (slg+h2h_slg):             gap {gap2:+.3f}')
    print(f'  3-signal TB score (slg+h2h_slg+recent_iso):  gap {gap:+.3f}')
    print()
    bucket_test(rows, 'tb_score_3', 'got_tb',
                [(0,.4),(.4,.6),(.6,.8),(.8,1.0),(1.0,1.2),(1.2,99)],
                '2+ TB rate by 3-signal TB score')

    # ================================================================
    # TEST 3: Does recent AVG improve the existing hit sum-score?
    # ================================================================
    print('='*60)
    print('TEST 3: DOES ADDING RECENT AVG IMPROVE HIT SUM-SCORE?')
    print('='*60)
    print(f'Separation on HITS:')
    print(f'  2-signal (avg+h2h_avg):                      gap {gap2:+.3f}')
    print(f'  3-signal (avg+h2h_avg+recent_avg):           gap {gap3:+.3f}')
    print()
    bucket_test(rows, 'hit_score_3', 'got_hit',
                [(0,.4),(.4,.5),(.5,.6),(.6,.7),(.7,.8),(.8,99)],
                'Hit rate by 3-signal hit score')

    # ================================================================
    # TEST 4: THE GOLDEN BUCKET — high HR score + high hit score
    # ================================================================
    print('='*60)
    print('TEST 4: GOLDEN BUCKET — HR score 1.40+ AND hit score 0.60+')
    print('(both signals agree: power matchup AND hit matchup)')
    print('='*60)
    both   = [r for r in rows if r['hr_score']>=1.40 and r['hit_score']>=0.60]
    hr_only= [r for r in rows if r['hr_score']>=1.40 and r['hit_score']<0.60]
    hit_only=[r for r in rows if r['hr_score']<1.40 and r['hit_score']>=0.60]
    neither= [r for r in rows if r['hr_score']<1.40 and r['hit_score']<0.60]

    def show(group, label, outcomes):
        if not group:
            print(f'  {label}: no entries')
            return
        for ok, olabel in outcomes:
            hits = sum(r[ok] for r in group)
            t = len(group)
            print(f'  {label} ({t} picks): '
                  f'{olabel} {hits}/{t} = {round(hits/t*100,1)}%')

    for group, label in [
        (both,    'BOTH hr_score>=1.40 AND hit_score>=0.60'),
        (hr_only, 'HR score>=1.40 only'),
        (hit_only,'Hit score>=0.60 only'),
        (neither, 'NEITHER signal'),
    ]:
        if not group: continue
        print(f'  {label} ({len(group)} picks):')
        for ok, olabel in [('got_hit','hits'), ('got_tb','2+TB'), ('got_hr','HRs')]:
            hits = sum(r[ok] for r in group)
            t = len(group)
            print(f'    {olabel}: {hits}/{t} = {round(hits/t*100,1)}%')
        print()

if __name__ == '__main__':
    main()
