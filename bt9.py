import urllib.request, json, sys, math
from zoneinfo import ZoneInfo
import datetime as dt
sys.path.insert(0, '/home/render/project/src')
import lineupk

MLB = 'https://statsapi.mlb.com/api/v1'
MLB11 = 'https://statsapi.mlb.com/api/v1.1'
RECENCY_DECAY = 0.6
SEASON_ANCHOR = 0.15

def get(u):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(u, headers={'User-Agent': 'x'}), timeout=30).read())

def get_pitcher_id(name):
    d = get(f'{MLB}/sports/1/players?season=2026')
    nm = name.lower().strip()
    for p in d.get('people', []):
        if p.get('fullName', '').lower().strip() == nm:
            return p.get('id')
    return None

def pitcher_k_per_bf(pid):
    g = get(f'{MLB}/people/{pid}/stats?stats=gameLog&group=pitching&season=2026')
    try: splits = g['stats'][0]['splits']
    except: return None, None
    sos=[]; bfs=[]
    for sp in splits:
        st=sp['stat']
        bf=int(st.get('battersFaced',0) or 0)
        so=int(st.get('strikeOuts',0) or 0)
        if bf>=12: sos.append(so); bfs.append(bf)
    if len(sos)<3: return None, None
    cum_bf=sum(bfs); cum_so=sum(sos)
    season_kbf=cum_so/cum_bf if cum_bf else 0
    n=len(sos)
    w=[math.exp(-RECENCY_DECAY*(n-1-i)) for i in range(n)]
    rec_kbf=(sum(wi*s for wi,s in zip(w,sos))/sum(wi*b for wi,b in zip(w,bfs))) if sum(w) else season_kbf
    kpbf=(1-SEASON_ANCHOR)*rec_kbf+SEASON_ANCHOR*season_kbf
    avg_bf=sum(bfs[-5:])/len(bfs[-5:])
    return kpbf, avg_bf

def main():
    rec=get('https://prop-edge-api.onrender.com/record')
    valid=['2026-06-19','2026-06-20','2026-06-21','2026-06-22','2026-06-23','2026-06-24','2026-06-25']
    unders=[x for x in rec.get('results',[])
            if x['prop_type']=='pitcher_strikeouts'
            and 'UNDER' in x['pick']
            and x.get('date','') in valid
            and x.get('actual') is not None]
    print(f'Under picks to retest: {len(unders)}')
    print()

    rows=[]
    for u in unders:
        pname=u.get('player','')
        date=u.get('date','')
        actual_k=float(u.get('actual',0))
        try: line=float(u['pick'].split()[-1])
        except: continue

        pid=get_pitcher_id(pname)
        if not pid: print(f'  no ID: {pname}'); continue

        # find game on that date
        s=get(f'{MLB}/schedule?sportId=1&date={date}')
        gpk=None; pside=None
        for day in s.get('dates',[]):
            for g in day.get('games',[]):
                for side in ('home','away'):
                    if g['teams'][side].get('probablePitcher',{}).get('id')==pid:
                        gpk=g['gamePk']; pside=side; break
                if gpk: break
            if gpk: break
        if not gpk: print(f'  no game: {pname} {date}'); continue

        k_per_bf, avg_bf=pitcher_k_per_bf(pid)
        if not k_per_bf: continue

        f=get(f'{MLB11}/game/{gpk}/feed/live')
        oside='away' if pside=='home' else 'home'
        order=f['liveData']['boxscore']['teams'][oside].get('battingOrder',[])[:9]
        if len(order)<7: continue

        throws=lineupk.get_pitcher_throws(pid)
        ek,avg_kr,n=lineupk.lineup_k_expectation(order,throws,2026,avg_bf,pitcher_id=pid)
        if not ek or n<5: print(f'  low lineup data ({n}): {pname}'); continue

        pitcher_proj=k_per_bf*avg_bf

        # current: 55% pitcher / 45% lineup
        blend_current=0.55*pitcher_proj+0.45*ek
        # lineup-heavy: 35% pitcher / 65% lineup
        blend_lineup=0.35*pitcher_proj+0.65*ek

        actual_side='UNDER' if actual_k<line else 'OVER'
        side_current='UNDER' if blend_current<line else 'OVER'
        side_lineup='UNDER' if blend_lineup<line else 'OVER'

        rows.append({
            'name':pname,'date':date,'actual':actual_k,'line':line,
            'pitcher_proj':pitcher_proj,'lineup_proj':ek,
            'blend_current':blend_current,'blend_lineup':blend_lineup,
            'actual_side':actual_side,
            'current_right':side_current==actual_side,
            'lineup_right':side_lineup==actual_side,
            'lineup_says_under':side_lineup=='UNDER',
            'current_err':abs(blend_current-actual_k),
            'lineup_err':abs(blend_lineup-actual_k),
            'result':u.get('result'),
        })
        cur='+' if side_current==actual_side else 'X'
        lup='+' if side_lineup==actual_side else 'X'
        print(f'  {date} {pname[:15]:15} act={actual_k} line={line} p={pitcher_proj:.1f} lu={ek:.1f} '
              f'curr={blend_current:.1f}{cur} luhvy={blend_lineup:.1f}{lup}')

    print()
    n=len(rows)
    if n==0: print('No data'); return

    cr=sum(1 for r in rows if r['current_right'])
    lr=sum(1 for r in rows if r['lineup_right'])
    cc=sum(1 for r in rows if r['current_err']<r['lineup_err'])
    lc=sum(1 for r in rows if r['lineup_err']<r['current_err'])
    avg_ce=sum(r['current_err'] for r in rows)/n
    avg_le=sum(r['lineup_err'] for r in rows)/n

    print(f'=== RESULTS ({n} under picks retested) ===')
    print()
    print(f'TEST 1 - Which picks the right side more often:')
    print(f'  Current (55/45 pitcher-heavy): {cr}/{n} = {round(cr/n*100)}%')
    print(f'  Lineup-heavy (35/65):          {lr}/{n} = {round(lr/n*100)}%')
    print()
    print(f'TEST 2 - Which projection is closer to actual Ks:')
    print(f'  Current closer:      {cc}/{n}')
    print(f'  Lineup-heavy closer: {lc}/{n}')
    print(f'  Current avg miss:    {avg_ce:.2f} Ks')
    print(f'  Lineup-heavy miss:   {avg_le:.2f} Ks')
    print()

    # KEY TEST: only bet under when BOTH signals agree (lineup-heavy also says under)
    both_under=[r for r in rows if r['lineup_says_under']]
    would_skip=[r for r in rows if not r['lineup_says_under']]
    both_hit=sum(1 for r in both_under if r['result']=='hit')
    skip_hit=sum(1 for r in would_skip if r['result']=='hit')
    print(f'TEST 3 - CONFIRMATION GATE (only bet when lineup-heavy ALSO says under):')
    print(f'  Would BET  ({len(both_under)} picks): {both_hit}/{len(both_under)} = {round(both_hit/len(both_under)*100) if both_under else 0}% hit rate')
    print(f'  Would SKIP ({len(would_skip)} picks): {skip_hit}/{len(would_skip)} = {round(skip_hit/len(would_skip)*100) if would_skip else 0}% hit rate')
    print(f'  (if skip rate < bet rate, the gate is correctly filtering bad picks)')

if __name__ == '__main__':
    main()
