import urllib.request, json, sys, math
import datetime as dt
from zoneinfo import ZoneInfo
sys.path.insert(0, '/home/render/project/src')
import lineupk

MLB = 'https://statsapi.mlb.com/api/v1'
MLB11 = 'https://statsapi.mlb.com/api/v1.1'
RECENCY_DECAY = 0.6
SEASON_ANCHOR = 0.15
GH = 'https://deshawnclark116-prog.github.io/mlb-picks2'

def get(u):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(u, headers={'User-Agent': 'x'}), timeout=30).read())

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
    rec_kbf=(sum(wi*s for wi,s in zip(w,sos))/
             sum(wi*b for wi,b in zip(w,bfs))) if sum(w) else season_kbf
    kpbf=(1-SEASON_ANCHOR)*rec_kbf+SEASON_ANCHOR*season_kbf
    avg_bf=sum(bfs[-5:])/len(bfs[-5:])
    return kpbf, avg_bf

def norm(s):
    return ''.join(c for c in s.lower() if c.isalpha() or c==' ').strip()

def main():
    valid=['2026-06-19','2026-06-20','2026-06-21','2026-06-22',
           '2026-06-23','2026-06-24','2026-06-25']

    # load record for actual results
    rec=get('https://prop-edge-api.onrender.com/record')
    record_lookup={}
    for x in rec.get('results',[]):
        if x['prop_type']=='pitcher_strikeouts' and 'UNDER' in x.get('pick',''):
            key=(x['date'], norm(x.get('player','')))
            record_lookup[key]=x

    # load predictions per date to get game_id + player_id
    pred_lookup={}  # (date, norm_name) -> prediction dict
    for date in valid:
        try:
            preds=get(f'{GH}/predictions_{date}.json')
            for p in preds:
                if p.get('prop_type')=='pitcher_strikeouts' and 'UNDER' in p.get('pick',''):
                    key=(date, norm(p.get('player','')))
                    pred_lookup[key]=p
        except Exception as e:
            print(f'  no predictions file for {date}: {str(e)[:30]}')

    print(f'Prediction entries found: {len(pred_lookup)}')
    print(f'Record entries found: {len(record_lookup)}')
    print()

    rows=[]
    for key, pred in pred_lookup.items():
        date, _ = key
        rec_entry = record_lookup.get(key)
        if not rec_entry:
            continue  # not graded yet

        actual_k=float(rec_entry.get('actual',0))
        try: line=float(pred['pick'].split()[-1])
        except: continue

        pid=pred.get('player_id') or pred.get('pitcher_id')
        gpk=pred.get('game_id')
        pname=pred.get('player','')

        if not pid or not gpk:
            print(f'  missing pid/gpk: {pname} {date}')
            continue

        # pitcher projection
        k_per_bf, avg_bf = pitcher_k_per_bf(pid)
        if not k_per_bf:
            print(f'  no pitcher stats: {pname}')
            continue

        # find which side pitcher was on + opposing lineup
        try:
            f=get(f'{MLB11}/game/{gpk}/feed/live')
            pside=None
            for side in ('home','away'):
                players=f['liveData']['boxscore']['teams'][side].get('players',{})
                if f'ID{pid}' in players:
                    pside=side; break
            if not pside:
                print(f'  pitcher not in boxscore: {pname}')
                continue
            oside='away' if pside=='home' else 'home'
            order=f['liveData']['boxscore']['teams'][oside].get('battingOrder',[])[:9]
        except Exception as e:
            print(f'  game fetch err {pname}: {str(e)[:30]}')
            continue

        if len(order)<7:
            print(f'  short lineup ({len(order)}): {pname}')
            continue

        # lineup K expectation
        throws=lineupk.get_pitcher_throws(pid)
        ek,avg_kr,n=lineupk.lineup_k_expectation(order,throws,2026,avg_bf,pitcher_id=pid)
        if not ek or n<5:
            print(f'  low lineup data ({n}): {pname}')
            continue

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
            'result':rec_entry.get('result'),
        })
        cur='+' if side_current==actual_side else 'X'
        lup='+' if side_lineup==actual_side else 'X'
        print(f'  {date} {pname[:15]:15} act={actual_k} line={line} '
              f'p={pitcher_proj:.1f} lu={ek:.1f} '
              f'curr={blend_current:.1f}{cur} luhvy={blend_lineup:.1f}{lup}')

    print()
    n=len(rows)
    if n==0:
        print('No rows - predictions files may be missing player_id field')
        print('Check: does your predictions JSON have player_id on strikeout picks?')
        return

    cr=sum(1 for r in rows if r['current_right'])
    lr=sum(1 for r in rows if r['lineup_right'])
    cc=sum(1 for r in rows if r['current_err']<r['lineup_err'])
    lc=sum(1 for r in rows if r['lineup_err']<r['current_err'])
    avg_ce=sum(r['current_err'] for r in rows)/n
    avg_le=sum(r['lineup_err'] for r in rows)/n

    print(f'=== RESULTS ({n} under picks retested) ===')
    print()
    print('TEST 1 - Which picks the right side more often:')
    print(f'  Current (55/45 pitcher-heavy): {cr}/{n} = {round(cr/n*100)}%')
    print(f'  Lineup-heavy (35/65):          {lr}/{n} = {round(lr/n*100)}%')
    print()
    print('TEST 2 - Which projection is closer to actual Ks:')
    print(f'  Current closer:      {cc}/{n}')
    print(f'  Lineup-heavy closer: {lc}/{n}')
    print(f'  Current avg miss:    {avg_ce:.2f} Ks')
    print(f'  Lineup-heavy miss:   {avg_le:.2f} Ks')
    print()
    both_under=[r for r in rows if r['lineup_says_under']]
    would_skip=[r for r in rows if not r['lineup_says_under']]
    both_hit=sum(1 for r in both_under if r['result']=='hit')
    skip_hit=sum(1 for r in would_skip if r['result']=='hit')
    print('TEST 3 - CONFIRMATION GATE:')
    print(f'  BET  (lineup-heavy also says under) {len(both_under)} picks: '
          f'{both_hit}/{len(both_under)} = {round(both_hit/len(both_under)*100) if both_under else 0}%')
    print(f'  SKIP (lineup-heavy disagrees)       {len(would_skip)} picks: '
          f'{skip_hit}/{len(would_skip)} = {round(skip_hit/len(would_skip)*100) if would_skip else 0}%')

if __name__ == '__main__':
    main()
