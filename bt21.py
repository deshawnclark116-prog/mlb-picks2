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

def batter_season_stats(pid):
    g = get(f'{MLB}/people/{pid}/stats?stats=season&group=hitting&season=2026')
    try:
        st = g['stats'][0]['splits'][0]['stat']
        pa = int(st.get('plateAppearances', 0) or 0)
        if pa < 20: return None
        slg = float(st.get('slg', 0) or 0)
        avg = float(st.get('avg', 0) or 0)
        return {'slg': slg, 'avg': avg, 'iso': slg-avg, 'pa': pa}
    except Exception:
        return None

def recent_iso(pid, before_date, n_games=7):
    g = get(f'{MLB}/people/{pid}/stats?stats=gameLog&group=hitting&season=2026')
    try:
        splits = g['stats'][0]['splits']
    except Exception:
        return None
    prior = [s for s in splits if s.get('date','') < before_date]
    if len(prior) < 3: return None
    recent = prior[-n_games:]
    h=tb=ab=0
    for s in recent:
        st=s['stat']
        h+=int(st.get('hits',0) or 0)
        tb+=int(st.get('totalBases',0) or 0)
        ab+=int(st.get('atBats',0) or 0)
    if ab==0: return None
    return (tb-h)/ab

def h2h_stats(batter_id, pitcher_id):
    d = get(f'{MLB}/people/{batter_id}/stats?stats=vsPlayer'
            f'&opposingPlayerId={pitcher_id}&group=hitting')
    ab=h=tb=0
    try:
        for s in d['stats'][0]['splits']:
            if s.get('stat'):
                st=s['stat']
                ab+=int(st.get('atBats',0) or 0)
                h+=int(st.get('hits',0) or 0)
                tb+=int(st.get('totalBases',0) or 0)
    except Exception:
        pass
    if ab>=4:
        return {'h2h_avg':h/ab,'h2h_slg':tb/ab,'h2h_ab':ab}
    return {'h2h_avg':0,'h2h_slg':0,'h2h_ab':0}

def get_starter(gpk, pid):
    try:
        f = get(f'{MLB11}/game/{gpk}/feed/live')
        for side in ('home','away'):
            order=f['liveData']['boxscore']['teams'][side].get('battingOrder',[])
            if pid in [int(x) for x in order if str(x).isdigit()]:
                oside='away' if side=='home' else 'home'
                pitchers=f['liveData']['boxscore']['teams'][oside].get('pitchers',[])
                return pitchers[0] if pitchers else None
    except Exception:
        pass
    return None

def get_actual(pid, date):
    try:
        g=get(f'{MLB}/people/{pid}/stats?stats=gameLog&group=hitting&season=2026')
        for sp in g['stats'][0]['splits']:
            if sp.get('date')==date:
                st=sp['stat']
                return {
                    'hr':int(st.get('homeRuns',0) or 0),
                    'hits':int(st.get('hits',0) or 0),
                    'tb':int(st.get('totalBases',0) or 0),
                }
    except Exception:
        pass
    return None

def load_preds():
    all_preds={}
    if os.path.exists(DATA_DIR):
        for fname in sorted(os.listdir(DATA_DIR)):
            if fname.startswith('predictions_') and fname.endswith('.json'):
                date=fname.replace('predictions_','').replace('.json','')
                try:
                    with open(os.path.join(DATA_DIR,fname)) as f:
                        all_preds[date]=json.load(f)
                except Exception:
                    pass
    today=dt.datetime.now(ZoneInfo('America/New_York')).date()
    for i in range(1,60):
        date=(today-dt.timedelta(days=i)).isoformat()
        if date in all_preds: continue
        try:
            all_preds[date]=get(f'{GH}/predictions_{date}.json')
        except Exception:
            pass
    return all_preds

def main():
    print('Loading predictions...')
    all_preds=load_preds()
    dates=sorted(all_preds.keys())
    print(f'Dates: {len(dates)} ({dates[0]} through {dates[-1]})')
    print()

    idx_data=get(f'{MLB}/sports/1/players?season=2026')
    idx={norm(p.get('fullName','')): p.get('id')
         for p in idx_data.get('people',[])}

    rows=[]
    seen=set()
    for date,preds in sorted(all_preds.items()):
        for pred in preds:
            if pred.get('prop_type') not in ('batter_hits','batter_total_bases'):
                continue
            nm=norm(pred.get('player',''))
            gpk=pred.get('game_id')
            if not nm or not gpk: continue
            key=(date,nm)
            if key in seen: continue
            seen.add(key)
            pid=idx.get(nm)
            if not pid: continue
            season=batter_season_stats(pid)
            if not season: continue
            r_iso=recent_iso(pid,date)
            if r_iso is None: continue
            starter=get_starter(gpk,pid)
            h2h=h2h_stats(pid,starter) if starter else \
                {'h2h_avg':0,'h2h_slg':0,'h2h_ab':0}
            actual=get_actual(pid,date)
            if not actual: continue

            hr_score  = season['slg'] + h2h['h2h_slg'] + r_iso
            hit_score = season['avg'] + h2h['h2h_avg']

            rows.append({
                'date':date,'name':nm,
                'hr_score':hr_score,'hit_score':hit_score,
                'actual_hr':actual['hr'],
                'actual_hits':actual['hits'],
                'actual_tb':actual['tb'],
                'got_hit':1 if actual['hits']>=1 else 0,
                'got_tb':1 if actual['tb']>=2 else 0,
                'got_hr':1 if actual['hr']>=1 else 0,
            })

    total=len(rows)
    print(f'Total entries: {total}')
    print()

    # OVERALL GOLDEN BUCKET
    golden=[r for r in rows if r['hr_score']>=1.40 and r['hit_score']>=0.80]
    rest  =[r for r in rows if not (r['hr_score']>=1.40 and r['hit_score']>=0.80)]
    print('OVERALL GOLDEN BUCKET (hr_score>=1.40 AND hit_score>=0.80):')
    print(f'  Golden: {len(golden)} picks, '
          f'rest: {len(rest)} picks')
    for ok,label in [('got_hr','HRs'),('got_hit','Hits'),('got_tb','2+TB')]:
        gh=sum(r[ok] for r in golden)
        rh=sum(r[ok] for r in rest)
        print(f'  {label:6}: golden {gh}/{len(golden)}='
              f'{round(gh/len(golden)*100,1) if golden else 0}%  '
              f'rest {rh}/{len(rest)}='
              f'{round(rh/len(rest)*100,1) if rest else 0}%')
    print()

    # MULTI-WINDOW — 4 non-overlapping windows
    print('='*60)
    print('MULTI-WINDOW: 4 non-overlapping windows')
    print('='*60)
    window_size=len(dates)//4
    hr_wins=0; hit_wins=0; tb_wins=0; tested=0

    for i in range(4):
        w_dates=set(dates[i*window_size:(i+1)*window_size])
        if not w_dates: continue
        gold=[r for r in rows if r['date'] in w_dates
              and r['hr_score']>=1.40 and r['hit_score']>=0.80]
        other=[r for r in rows if r['date'] in w_dates
               and not (r['hr_score']>=1.40 and r['hit_score']>=0.80)]
        if len(gold)<3: continue
        tested+=1
        print(f'Window {i+1} ({min(w_dates)}-{max(w_dates)}): '
              f'{len(gold)} golden picks')
        for ok,label in [('got_hr','HRs'),('got_hit','Hits'),('got_tb','2+TB')]:
            gh=sum(r[ok] for r in gold)
            oh=sum(r[ok] for r in other)
            g_pct=round(gh/len(gold)*100,1) if gold else 0
            o_pct=round(oh/len(other)*100,1) if other else 0
            win=g_pct>o_pct
            if ok=='got_hr' and win: hr_wins+=1
            if ok=='got_hit' and win: hit_wins+=1
            if ok=='got_tb' and win: tb_wins+=1
            print(f'  {label:6}: golden {gh}/{len(gold)}={g_pct}%  '
                  f'rest {oh}/{len(other)}={o_pct}%  '
                  f'{"WIN ✓" if win else "loss X"}')
        print()

    print(f'Windows tested: {tested}')
    print(f'Golden bucket beat rest:')
    print(f'  HRs:  {hr_wins}/{tested} windows')
    print(f'  Hits: {hit_wins}/{tested} windows')
    print(f'  2+TB: {tb_wins}/{tested} windows')
    print()

    # THRESHOLD SENSITIVITY — does 1.40/0.80 hold vs nearby thresholds?
    print('='*60)
    print('THRESHOLD SENSITIVITY (HR rate only):')
    print('='*60)
    for hr_t, hit_t in [
        (1.30, 0.70),(1.30, 0.80),(1.35, 0.80),
        (1.40, 0.70),(1.40, 0.80),(1.40, 0.90),
        (1.50, 0.80),(1.50, 0.90),
    ]:
        group=[r for r in rows if r['hr_score']>=hr_t and r['hit_score']>=hit_t]
        if len(group)<3: continue
        gh=sum(r['got_hr'] for r in group)
        hh=sum(r['got_hit'] for r in group)
        t=len(group)
        ppd=round(t/len(dates),1)
        print(f'  hr>={hr_t} & hit>={hit_t}: '
              f'{gh}/{t}={round(gh/t*100,1)}% HR  '
              f'{hh}/{t}={round(hh/t*100,1)}% hits  '
              f'{ppd}/day')

    print()

    # WHAT ARE THESE PICKS? — see the actual players
    print('GOLDEN BUCKET PICKS (all 27 days):')
    print(f'  {"name":22} {"hr_sc":6} {"hit_sc":6} '
          f'{"HR":4} {"hits":5} {"TB":4}')
    for r in sorted(golden, key=lambda x:-x['hr_score']):
        print(f'  {r["name"][:20]:20}  '
              f'{r["hr_score"]:.3f}  {r["hit_score"]:.3f}  '
              f'{r["actual_hr"]:3}HR  {r["actual_hits"]:3}H  '
              f'{r["actual_tb"]:3}TB  '
              f'{"✓" if r["got_hr"] else ""}')

if __name__ == '__main__':
    main()
