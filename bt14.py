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
    g = get(f'{MLB}/people/{pid}/stats?stats=season&group=hitting&season=2026')
    try:
        st = g['stats'][0]['splits'][0]['stat']
        pa = int(st.get('plateAppearances', 0) or 0)
        if pa < 20: return None
        slg = float(st.get('slg', 0) or 0)
        avg = float(st.get('avg', 0) or 0)
        hr = int(st.get('homeRuns', 0) or 0)
        # recent form — last 14 days
        return {
            'slg': slg,
            'avg': avg,
            'iso': slg - avg,  # isolated power = pure extra base hit power
            'hr_per_pa': hr/pa if pa else 0,
            'pa': pa,
        }
    except Exception:
        return None

def get_h2h_vs_pitcher(batter_id, pitcher_id):
    """Head-to-head: batter vs this specific pitcher."""
    d = get(f'{MLB}/people/{batter_id}/stats?stats=vsPlayer'
            f'&opposingPlayerId={pitcher_id}&group=hitting')
    ab=0; h=0; hr=0; tb=0
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
            'has_h2h': True,
        }
    return {'h2h_avg': 0, 'h2h_slg': 0, 'h2h_hr': 0, 'h2h_ab': ab, 'has_h2h': False}

def main():
    today = dt.datetime.now(ZoneInfo('America/New_York')).date()
    all_dates = [(today - dt.timedelta(days=i)).isoformat() for i in range(1, 22)]

    idx_data = get(f'{MLB}/sports/1/players?season=2026')
    idx = {norm(p.get('fullName','')): p.get('id')
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

            stats = get_batter_stats(pid)
            if not stats: continue

            # find opposing starter from boxscore
            pitcher_id = None
            try:
                f = get(f'{MLB11}/game/{gpk}/feed/live')
                batter_side = None
                for side in ('home','away'):
                    order = f['liveData']['boxscore']['teams'][side].get('battingOrder',[])
                    if pid in order or pid in [int(x) for x in order if str(x).isdigit()]:
                        batter_side = side; break
                if batter_side:
                    oside = 'away' if batter_side=='home' else 'home'
                    pitchers = f['liveData']['boxscore']['teams'][oside].get('pitchers',[])
                    if pitchers:
                        pitcher_id = pitchers[0]
            except Exception:
                pass

            # head-to-head vs this pitcher
            h2h = {'h2h_avg':0,'h2h_slg':0,'h2h_hr':0,'h2h_ab':0,'has_h2h':False}
            if pitcher_id:
                h2h = get_h2h_vs_pitcher(pid, pitcher_id)

            # SUM SCORES (DeShawn's instinct — add them together)
            slg_plus_h2h_slg = stats['slg'] + h2h['h2h_slg']
            iso_plus_h2h_avg = stats['iso'] + h2h['h2h_avg']

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
                **stats, **h2h,
                'slg_plus_h2h_slg': slg_plus_h2h_slg,
                'iso_plus_h2h_avg': iso_plus_h2h_avg,
                'actual_hr': actual_hr,
            })

    print(f'Files: {files_found} | Batter-game entries: {len(rows)}')
    print()

    # First — what did the .375-.399 HR batters actually have?
    low_slg_hrs = [r for r in rows if 0.375<=r['slg']<0.450 and r['actual_hr']>=1]
    print(f'THE .375-.449 HR BATTERS ({len(low_slg_hrs)} of them):')
    print(f'  {"name":22} slg    iso    h2h_avg h2h_slg h2h_ab sum_slg sum_iso')
    for r in low_slg_hrs:
        print(f'  {r["name"][:20]:20} '
              f'{r["slg"]:.3f}  {r["iso"]:.3f}  '
              f'{r["h2h_avg"]:.3f}   {r["h2h_slg"]:.3f}  '
              f'{r["h2h_ab"]:4}   '
              f'{r["slg_plus_h2h_slg"]:.3f}  {r["iso_plus_h2h_avg"]:.3f}')

    print()

    # Test sum scores as separators
    def bucket_test(score_key, label, edges):
        print(f'SUM SCORE TEST: {label}')
        for lo, hi in edges:
            group = [r for r in rows if lo <= r[score_key] < hi]
            if len(group) < 3: continue
            hr_g = sum(1 for r in group if r['actual_hr']>=1)
            t = len(group)
            be = round((t/hr_g-1)*100) if hr_g else 0
            be_str = f'+{be}' if hr_g else 'n/a'
            print(f'  {score_key} {lo:.2f}-{hi:.2f}: '
                  f'{hr_g}/{t} = {round(hr_g/t*100,1)}%  '
                  f'breakeven {be_str}')
        print()

    # SLG + h2h_slg sum score
    bucket_test('slg_plus_h2h_slg', 'SLG + head-to-head SLG vs pitcher',
                [(0,.4),(.4,.5),(.5,.6),(.6,.7),(.7,.8),(.8,99)])

    # ISO + h2h_avg sum score
    bucket_test('iso_plus_h2h_avg', 'ISO + head-to-head AVG vs pitcher',
                [(0,.1),(.1,.2),(.2,.3),(.3,.4),(.4,.5),(.5,99)])

    # ISO alone
    bucket_test('iso', 'ISO alone (SLG - AVG = pure power)',
                [(0,.1),(.1,.15),(.15,.2),(.2,.25),(.25,99)])

    # separation test — same as batter hits
    print('SEPARATION TEST (which score best separates HR from non-HR):')
    for score_key, label in [
        ('slg', 'SLG alone'),
        ('iso', 'ISO alone'),
        ('slg_plus_h2h_slg', 'SLG + h2h_SLG (sum)'),
        ('iso_plus_h2h_avg', 'ISO + h2h_AVG (sum)'),
    ]:
        hr_scores = [r[score_key] for r in rows if r['actual_hr']>=1]
        no_scores = [r[score_key] for r in rows if r['actual_hr']==0]
        hs = sum(hr_scores)/len(hr_scores) if hr_scores else 0
        ns = sum(no_scores)/len(no_scores) if no_scores else 0
        print(f'  {label:30} hitters avg {hs:.3f} '
              f'non-hitters avg {ns:.3f} '
              f'gap {hs-ns:+.3f}')

if __name__ == '__main__':
    main()
