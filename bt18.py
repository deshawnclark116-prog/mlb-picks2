import urllib.request, json, datetime as dt, math, os
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

def batter_recent_iso(pid, before_date, n_games=7):
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
    return (tb-h)/ab  # recent ISO

def h2h_slg_vs_pitcher(batter_id, pitcher_id):
    d = get(f'{MLB}/people/{batter_id}/stats?stats=vsPlayer'
            f'&opposingPlayerId={pitcher_id}&group=hitting')
    ab=tb=0
    try:
        for s in d['stats'][0]['splits']:
            if s.get('stat'):
                st=s['stat']
                ab+=int(st.get('atBats',0) or 0)
                tb+=int(st.get('totalBases',0) or 0)
    except Exception:
        pass
    return tb/ab if ab>=4 else 0

def get_starter(gpk, pid):
    """Find opposing starter and batter side from boxscore."""
    try:
        f = get(f'{MLB11}/game/{gpk}/feed/live')
        batter_side = None
        for side in ('home','away'):
            order = f['liveData']['boxscore']['teams'][side].get('battingOrder',[])
            int_order=[int(x) for x in order if str(x).isdigit()]
            if pid in int_order:
                batter_side=side; break
        if not batter_side: return None
        oside='away' if batter_side=='home' else 'home'
        pitchers=f['liveData']['boxscore']['teams'][oside].get('pitchers',[])
        return pitchers[0] if pitchers else None
    except Exception:
        return None

def get_actual_hr(pid, date):
    try:
        g = get(f'{MLB}/people/{pid}/stats?stats=gameLog&group=hitting&season=2026')
        for sp in g['stats'][0]['splits']:
            if sp.get('date')==date:
                return int(sp['stat'].get('homeRuns',0) or 0)
    except Exception:
        pass
    return 0

def load_all_predictions():
    """Load from BOTH GitHub Pages AND local /data/predictions/ disk."""
    today = dt.datetime.now(ZoneInfo('America/New_York')).date()
    all_preds = {}  # date -> list of preds

    # try local disk first (has everything Render has saved)
    if os.path.exists(DATA_DIR):
        for fname in sorted(os.listdir(DATA_DIR)):
            if fname.startswith('predictions_') and fname.endswith('.json'):
                date = fname.replace('predictions_','').replace('.json','')
                try:
                    with open(os.path.join(DATA_DIR, fname)) as f:
                        all_preds[date] = json.load(f)
                except Exception:
                    pass
        print(f'  Loaded {len(all_preds)} files from local disk')

    # supplement with GitHub Pages for any missing dates
    gh_found = 0
    for i in range(1, 60):  # try up to 60 days back
        date = (today-dt.timedelta(days=i)).isoformat()
        if date in all_preds: continue
        try:
            preds = get(f'{GH}/predictions_{date}.json')
            all_preds[date] = preds
            gh_found += 1
        except Exception:
            pass
    print(f'  Supplemented with {gh_found} files from GitHub Pages')
    return all_preds

def main():
    print('Loading all available prediction files...')
    all_preds = load_all_predictions()
    dates = sorted(all_preds.keys())
    print(f'Total dates available: {len(dates)} '
          f'({dates[0] if dates else "none"} through {dates[-1] if dates else "none"})')
    print()

    # build player index once
    idx_data = get(f'{MLB}/sports/1/players?season=2026')
    idx = {norm(p.get('fullName','')): p.get('id')
           for p in idx_data.get('people',[])}

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

            recent_iso = batter_recent_iso(pid, date, n_games=7)
            if recent_iso is None: continue

            starter = get_starter(gpk, pid)
            h2h = h2h_slg_vs_pitcher(pid, starter) if starter else 0

            actual_hr = get_actual_hr(pid, date)

            three_sig = season['slg'] + h2h + recent_iso

            rows.append({
                'date': date, 'name': nm,
                'slg': season['slg'],
                'h2h_slg': h2h,
                'recent_iso': recent_iso,
                'three_sig': three_sig,
                'actual_hr': actual_hr,
            })

    total = len(rows)
    hr_total = sum(1 for r in rows if r['actual_hr']>=1)
    print(f'Total batter-game entries: {total}')
    print(f'HR games: {hr_total} ({round(hr_total/total*100,1) if total else 0}%)')
    print()

    # SEPARATION
    hr_s = [r['three_sig'] for r in rows if r['actual_hr']>=1]
    no_s = [r['three_sig'] for r in rows if r['actual_hr']==0]
    hs = sum(hr_s)/len(hr_s) if hr_s else 0
    ns = sum(no_s)/len(no_s) if no_s else 0
    print(f'SEPARATION: hitters avg {hs:.3f} vs non-hitters {ns:.3f} '
          f'gap {hs-ns:+.3f}')
    print()

    # THRESHOLD TEST — multiple thresholds
    print('THRESHOLD TEST (3-signal: SLG + h2h_SLG + recent_ISO):')
    print(f'  {"threshold":12} {"hr/total":12} {"hit_rate":10} '
          f'{"breakeven":12} {"picks_per_day"}')
    print('-'*65)
    for thresh in [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20]:
        group = [r for r in rows if r['three_sig']>=thresh]
        if len(group) < 3: continue
        hr_g = sum(1 for r in group if r['actual_hr']>=1)
        t = len(group)
        hr_pct = round(hr_g/t*100,1)
        be = round((t/hr_g-1)*100) if hr_g else 0
        ppd = round(t/len(dates),1) if dates else 0
        be_str = f'+{be}' if hr_g else 'n/a'
        profit = '✓ PROFIT' if hr_g and hr_pct >= 25 else ''
        print(f'  score>={thresh:.2f}:  {hr_g:4}/{t:<6} = {hr_pct:5}%  '
              f'breakeven {be_str:8}  {ppd}/day  {profit}')

    print()

    # MULTI-WINDOW — 4 non-overlapping windows
    print('MULTI-WINDOW VALIDATION (3-signal >= 0.90):')
    window_size = max(len(dates)//4, 5)
    date_list = sorted(all_preds.keys())
    add_wins = 0
    windows_tested = 0
    for i in range(0, len(date_list), window_size):
        w_dates = set(date_list[i:i+window_size])
        if not w_dates: continue
        above = [r for r in rows if r['date'] in w_dates and r['three_sig']>=0.90]
        below = [r for r in rows if r['date'] in w_dates and r['three_sig']<0.90]
        if len(above)<3: continue
        windows_tested+=1
        ah = sum(1 for r in above if r['actual_hr']>=1)
        bh = sum(1 for r in below if r['actual_hr']>=1)
        a_pct = round(ah/len(above)*100,1)
        b_pct = round(bh/len(below)*100,1) if below else 0
        win = a_pct > b_pct
        if win: add_wins+=1
        print(f'  Window {windows_tested} ({min(w_dates)}-{max(w_dates)}): '
              f'above 0.90: {ah}/{len(above)}={a_pct}%  '
              f'below: {bh}/{len(below)}={b_pct}%  '
              f'{"WIN ✓" if win else "loss"}')

    print(f'\n  Score >= 0.90 beat "below" in {add_wins}/{windows_tested} windows')
    print()

    # STAIRCASE — does it climb cleanly?
    print('STAIRCASE (does higher score = more HRs?):')
    edges=[(0,.3),(.3,.5),(.5,.7),(.7,.9),(.9,1.1),(1.1,1.3),(1.3,99)]
    clean = True
    prev_pct = -1
    for lo,hi in edges:
        group=[r for r in rows if lo<=r['three_sig']<hi]
        if len(group)<3: continue
        hr_g=sum(1 for r in group if r['actual_hr']>=1)
        t=len(group)
        pct=round(hr_g/t*100,1)
        marker='↑' if pct>prev_pct else '↓ broken'
        if pct<prev_pct: clean=False
        prev_pct=pct
        print(f'  {lo:.1f}-{hi:.1f}: {hr_g}/{t} = {pct}%  {marker}')
    print(f'  Staircase {"CLEAN ✓" if clean else "BROKEN — noise present"}')
    print()

    # FINAL VERDICT
    best_thresh = None; best_pct = 0; best_n = 0
    for thresh in [0.80, 0.85, 0.90, 0.95, 1.00, 1.10]:
        group=[r for r in rows if r['three_sig']>=thresh]
        if len(group)<5: continue
        hr_g=sum(1 for r in group if r['actual_hr']>=1)
        pct=hr_g/len(group)*100
        if pct>best_pct:
            best_pct=pct; best_thresh=thresh; best_n=len(group)
    print('FINAL VERDICT:')
    if best_thresh and best_pct>=25:
        print(f'  BUILD IT — 3-signal >= {best_thresh} hits {round(best_pct,1)}% '
              f'across {best_n} picks ({round(best_n/len(dates),1)}/day)')
        print(f'  Profitable at any HR prop odds of '
              f'+{round((100/best_pct-1)*100)} or better')
        print(f'  (typical FanDuel HR props: +300 to +500)')
    else:
        print(f'  NOT READY — best threshold hits {round(best_pct,1) if best_pct else 0}%')
        print(f'  Need >= 25% to be profitable at +300. Keep testing.')

if __name__ == '__main__':
    main()
