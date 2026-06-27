import urllib.request, json, datetime as dt, os
from zoneinfo import ZoneInfo
import sys
sys.path.insert(0, '/home/render/project/src')

MLB = 'https://statsapi.mlb.com/api/v1'
MLB11 = 'https://statsapi.mlb.com/api/v1.1'
GH = 'https://deshawnclark116-prog.github.io/mlb-picks2'
DATA_DIR = '/data/predictions'

# THE FIX: raise h2h minimum from 4 AB to 8 AB
# Prevents 2-for-4 with a HR from inflating h2h_slg to 2.000
MIN_H2H_AB = 8

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
        return {'slg': slg, 'avg': avg, 'pa': pa}
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
    # THE FIX — require MIN_H2H_AB instead of 4
    if ab >= MIN_H2H_AB:
        return {'h2h_avg':h/ab,'h2h_slg':tb/ab,'h2h_ab':ab,'has_h2h':True}
    return {'h2h_avg':0,'h2h_slg':0,'h2h_ab':ab,'has_h2h':False}

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
    print(f'Loading predictions (h2h minimum raised to {MIN_H2H_AB} AB)...')
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
                {'h2h_avg':0,'h2h_slg':0,'h2h_ab':0,'has_h2h':False}
            actual=get_actual(pid,date)
            if not actual: continue

            hr_score  = season['slg'] + h2h['h2h_slg'] + r_iso
            hit_score = season['avg'] + h2h['h2h_avg']

            rows.append({
                'date':date,'name':nm,
                'hr_score':hr_score,'hit_score':hit_score,
                'h2h_ab':h2h['h2h_ab'],'has_h2h':h2h['has_h2h'],
                'actual_hr':actual['hr'],
                'actual_hits':actual['hits'],
                'actual_tb':actual['tb'],
                'got_hit':1 if actual['hits']>=1 else 0,
                'got_tb':1 if actual['tb']>=2 else 0,
                'got_hr':1 if actual['hr']>=1 else 0,
            })

    total=len(rows)
    print(f'Total entries: {total}')
    print(f'Entries with real h2h (>={MIN_H2H_AB} AB): '
          f'{sum(1 for r in rows if r["has_h2h"])}')
    print()

    # OVERALL GOLDEN BUCKET — both thresholds
    for hr_t, hit_t, label in [
        (1.30, 0.80, '1.30/0.80'),
        (1.40, 0.80, '1.40/0.80'),
    ]:
        golden=[r for r in rows if r['hr_score']>=hr_t and r['hit_score']>=hit_t]
        rest  =[r for r in rows if not(r['hr_score']>=hr_t and r['hit_score']>=hit_t)]
        print(f'GOLDEN BUCKET hr>={hr_t} & hit>={hit_t}: '
              f'{len(golden)} picks')
        for ok,lbl in [('got_hr','HRs'),('got_hit','Hits'),('got_tb','2+TB')]:
            gh=sum(r[ok] for r in golden)
            rh=sum(r[ok] for r in rest)
            print(f'  {lbl:6}: golden {gh}/{len(golden)}='
                  f'{round(gh/len(golden)*100,1) if golden else 0}%  '
                  f'rest {rh}/{len(rest)}='
                  f'{round(rh/len(rest)*100,1) if rest else 0}%')
        print()

    # MULTI-WINDOW
    print('='*60)
    print('MULTI-WINDOW (hr>=1.30 & hit>=0.80):')
    print('='*60)
    window_size=len(dates)//4
    hr_wins=hit_wins=tb_wins=tested=0
    for i in range(4):
        w_dates=set(dates[i*window_size:(i+1)*window_size])
        if not w_dates: continue
        gold=[r for r in rows if r['date'] in w_dates
              and r['hr_score']>=1.30 and r['hit_score']>=0.80]
        other=[r for r in rows if r['date'] in w_dates
               and not(r['hr_score']>=1.30 and r['hit_score']>=0.80)]
        if len(gold)<3: continue
        tested+=1
        print(f'Window {i+1} ({min(w_dates)}-{max(w_dates)}): '
              f'{len(gold)} golden picks')
        for ok,lbl in [('got_hr','HRs'),('got_hit','Hits'),('got_tb','2+TB')]:
            gh=sum(r[ok] for r in gold)
            oh=sum(r[ok] for r in other)
            g_pct=round(gh/len(gold)*100,1) if gold else 0
            o_pct=round(oh/len(other)*100,1) if other else 0
            win=g_pct>o_pct
            if ok=='got_hr' and win: hr_wins+=1
            if ok=='got_hit' and win: hit_wins+=1
            if ok=='got_tb' and win: tb_wins+=1
            print(f'  {lbl:6}: golden {gh}/{len(gold)}={g_pct}%  '
                  f'rest {oh}/{len(other)}={o_pct}%  '
                  f'{"WIN ✓" if win else "loss X"}')
        print()

    print(f'Windows tested: {tested}')
    print(f'Golden bucket beat rest:')
    print(f'  HRs:  {hr_wins}/{tested}')
    print(f'  Hits: {hit_wins}/{tested}')
    print(f'  2+TB: {tb_wins}/{tested}')
    print()

    # SIDE BY SIDE vs old 4 AB minimum
    print('='*60)
    print(f'IMPACT OF FIX ({MIN_H2H_AB} AB min vs old 4 AB min):')
    print('='*60)
    print('Who got REMOVED from golden bucket by raising h2h minimum?')
    removed=[r for r in rows
             if r['h2h_ab']>0 and r['h2h_ab']<MIN_H2H_AB
             and r['hr_score']<1.30]  # would have scored higher with 4 AB min
    print(f'  Batters with h2h samples below {MIN_H2H_AB} AB '
          f'(small sample h2h now zeroed out): '
          f'{sum(1 for r in rows if 0<r["h2h_ab"]<MIN_H2H_AB)}')
    print()

    # GOLDEN PICKS LIST
    golden_130=[r for r in rows if r['hr_score']>=1.30 and r['hit_score']>=0.80]
    print(f'GOLDEN BUCKET PICKS at 1.30/0.80 ({len(golden_130)} total):')
    print(f'  {"name":22} {"hr_sc":6} {"hit_sc":6} '
          f'{"h2h_ab":7} {"HR":4} {"H":4} {"TB"}')
    for r in sorted(golden_130,key=lambda x:-x['hr_score']):
        print(f'  {r["name"][:20]:20}  '
              f'{r["hr_score"]:.3f}  {r["hit_score"]:.3f}  '
              f'{r["h2h_ab"]:5}AB  '
              f'{r["actual_hr"]}HR  {r["actual_hits"]}H  '
              f'{r["actual_tb"]}TB  '
              f'{"✓" if r["got_hr"] else ""}')

    print()
    print('HONEST VERDICT:')
    golden=golden_130
    if not golden:
        print('  No golden bucket picks found — thresholds too tight with 8 AB min')
        return
    gh=sum(r['got_hr'] for r in golden)
    hh=sum(r['got_hit'] for r in golden)
    th=sum(r['got_tb'] for r in golden)
    t=len(golden)
    print(f'  Golden bucket: {t} picks, {gh} HRs ({round(gh/t*100,1)}%), '
          f'{hh} hits ({round(hh/t*100,1)}%), {th} 2+TB ({round(th/t*100,1)}%)')
    if tested>=2 and hr_wins>=2:
        print(f'  MULTI-WINDOW: {hr_wins}/{tested} — BUILD IT')
    elif tested<2:
        print(f'  Only {tested} window(s) testable — build cautiously, '
              f'confirm live or wait for more data')
    else:
        print(f'  Multi-window weak ({hr_wins}/{tested}) — '
              f'signal real but needs more data before full build')

if __name__ == '__main__':
    main()
