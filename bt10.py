import urllib.request, json, datetime as dt
from collections import defaultdict
from zoneinfo import ZoneInfo

GH = 'https://deshawnclark116-prog.github.io/mlb-picks2'
MLB = 'https://statsapi.mlb.com/api/v1'

def get(u):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(u, headers={'User-Agent': 'x'}), timeout=30).read())

def norm(s):
    return ''.join(c for c in s.lower() if c.isalpha() or c == ' ').strip()

def main():
    valid = ['2026-06-19','2026-06-20','2026-06-21','2026-06-22',
             '2026-06-23','2026-06-24','2026-06-25','2026-06-26']

    # load record for actual results
    rec = get('https://prop-edge-api.onrender.com/record')
    record_lookup = {}
    for x in rec.get('results', []):
        if x['prop_type'] in ('batter_hits', 'batter_total_bases'):
            key = (x['date'], norm(x.get('player', '')))
            record_lookup[key] = x

    print(f'Graded hit/TB picks in record: {len(record_lookup)}')
    print()

    # for each picked batter, find their actual HR in that game
    hit_batters = []    # picked for hits prop
    tb_batters = []     # picked for total bases prop
    both_batters = []   # picked for both

    for date in valid:
        try:
            preds = get(f'{GH}/predictions_{date}.json')
        except Exception as e:
            print(f'  no file {date}: {str(e)[:30]}')
            continue

        # find who was picked for hits and/or TB on this date
        hit_picks = {}   # norm_name -> pred
        tb_picks = {}
        for p in preds:
            ptype = p.get('prop_type','')
            nm = norm(p.get('player',''))
            if not nm: continue
            if ptype == 'batter_hits':
                hit_picks[nm] = p
            elif ptype == 'batter_total_bases':
                tb_picks[nm] = p

        all_names = set(hit_picks) | set(tb_picks)
        if not all_names: continue

        # build player index for this lookup
        idx_data = get(f'{MLB}/sports/1/players?season=2026')
        idx = {norm(p.get('fullName','')): p.get('id')
               for p in idx_data.get('people', [])}

        for nm in all_names:
            pid = idx.get(nm)
            if not pid: continue

            # get their game log for this date and find actual HR
            try:
                g = get(f'{MLB}/people/{pid}/stats?stats=gameLog&group=hitting&season=2026')
                actual_hr = None
                actual_h = None
                actual_tb = None
                for sp in g['stats'][0]['splits']:
                    if sp.get('date') == date:
                        actual_hr = int(sp['stat'].get('homeRuns', 0) or 0)
                        actual_h = int(sp['stat'].get('hits', 0) or 0)
                        actual_tb = int(sp['stat'].get('totalBases', 0) or 0)
                        break
                if actual_hr is None:
                    continue
            except Exception:
                continue

            in_hit = nm in hit_picks
            in_tb = nm in tb_picks
            entry = {
                'date': date, 'name': nm,
                'actual_hr': actual_hr,
                'actual_h': actual_h,
                'actual_tb': actual_tb,
                'in_hit': in_hit,
                'in_tb': in_tb,
                'hit_proj': hit_picks[nm].get('projected') if in_hit else None,
                'tb_proj': tb_picks[nm].get('projected') if in_tb else None,
            }
            if in_hit and in_tb:
                both_batters.append(entry)
            elif in_hit:
                hit_batters.append(entry)
            elif in_tb:
                tb_batters.append(entry)

    all_picked = hit_batters + tb_batters + both_batters

    print(f'Total batter-game entries found: {len(all_picked)}')
    print(f'  hit-only picks:  {len(hit_batters)}')
    print(f'  TB-only picks:   {len(tb_batters)}')
    print(f'  both picks:      {len(both_batters)}')
    print()

    def hr_rate(rows):
        if not rows: return 0, 0, 0
        hrs = sum(r['actual_hr'] for r in rows)
        games = len(rows)
        return hrs, games, round(hrs/games*100, 1)

    h, g, r = hr_rate(all_picked)
    print(f'ALL PICKED BATTERS: {h} HRs in {g} games = {r}% HR rate per game')

    h, g, r = hr_rate(hit_batters)
    print(f'HIT PROP picks only: {h} HRs in {g} games = {r}% HR rate per game')

    h, g, r = hr_rate(tb_batters)
    print(f'TOTAL BASES picks only: {h} HRs in {g} games = {r}% HR rate per game')

    h, g, r = hr_rate(both_batters)
    print(f'BOTH hit + TB picks: {h} HRs in {g} games = {r}% HR rate per game')

    print()
    print('The guys who actually went yard:')
    hr_games = [(r['date'], r['name'], r['actual_hr'], r['actual_tb'],
                 r['in_hit'], r['in_tb'])
                for r in all_picked if r['actual_hr'] >= 1]
    hr_games.sort(key=lambda x: -x[2])
    for d, nm, hr, tb, ih, it in hr_games:
        tags = []
        if ih: tags.append('HIT-pick')
        if it: tags.append('TB-pick')
        print(f'  {d} {nm[:20]:20} {hr}HR {tb}TB  [{" + ".join(tags)}]')

    print()
    # league average HR rate for context
    print('Context: MLB league avg HR rate is roughly 3% per player per game')
    print('If our picked batters are hitting above that, the system is already')
    print('identifying power bats - we just need to exploit it with HR props')
    print()

    # TB projection correlation with HRs
    # does a higher TB projection correlate with HRs?
    tb_with_hr = [(r['tb_proj'], r['actual_hr'])
                  for r in all_picked if r['tb_proj'] is not None]
    if len(tb_with_hr) >= 10:
        print('TB PROJECTION vs ACTUAL HRs:')
        buckets = defaultdict(lambda: [0, 0])
        for proj, hr in tb_with_hr:
            if proj >= 2.5:   b = '2.5+'
            elif proj >= 2.0: b = '2.0-2.5'
            elif proj >= 1.5: b = '1.5-2.0'
            else:             b = '<1.5'
            buckets[b][0] += hr
            buckets[b][1] += 1
        for b in ['2.5+', '2.0-2.5', '1.5-2.0', '<1.5']:
            if b in buckets:
                hrs, games = buckets[b]
                print(f'  TB proj {b}: {hrs} HRs in {games} games = {round(hrs/games*100,1)}%')

if __name__ == '__main__':
    main()
