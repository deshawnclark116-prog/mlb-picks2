#!/usr/bin/env python3
"""Timestamp-safe pregame depth-chart challenger; every used snapshot < game date.

2025-26 depth charts have timestamped daily observations; missing historical
2023-24 coverage is explicit NaN, never silently backfilled from the future.
Research only, fixed historical benchmark with no sportsbook inputs.
"""
from collections import defaultdict
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path
import csv,sqlite3,numpy as np
import challenger_tenure_v5 as ten

base=ten.base

DEPTH_FEATURES=['depth_pregame_listed','depth_rank','depth_slot',
                'depth_is_first','depth_is_top_two','depth_is_lower',
                'depth_rank_change_14d','depth_rank_change_30d',
                'depth_chart_age_days','depth_very_recent_move','depth_seen_pregame_n']

original=base.build_pbp_asof

def build_depth(rows,dir):
    paired={(m['team'],m['player_id']) for m,f,y in rows}
    dates=defaultdict(dict)
    valid_days={}
    for season in (2025,2026):
        p=Path(dir)/f'depth_charts_{season}.csv'
        if not p.exists():raise FileNotFoundError(p)
        n=0
        with p.open(newline='') as inp:
            for r in csv.DictReader(inp):
                pair=(r.get('team',''),r.get('gsis_id',''))
                if pair not in paired:continue
                date=(r.get('dt') or '')[:10]
                if len(date)!=10 or date[4]!='-':continue
                try:
                    rank=float(r.get('pos_rank'))
                    slot=float(r.get('pos_slot'))
                except (TypeError,ValueError):continue
                if not np.isfinite(rank):continue
                # Across multiple listed slots on the same day, use the
                # smallest rank. This is a pregame snapshot choice only.
                d=dates[pair]
                prior=d.get(date)
                if prior is None or (rank,slot)<prior:d[date]=(rank,slot)
                n+=1
        valid_days[season]=n
    sorted_depth={pair:sorted(entries.items()) for pair,entries in dates.items()}
    con=sqlite3.connect(f"file:{Path('nfl_models/nfl_model.sqlite').resolve()}?mode=ro",uri=True)
    gm={}
    for season,week,home,away,date in con.execute('''SELECT season,week,home_team,away_team,game_date
           FROM games WHERE season_type='REG' '''):
        if not date:continue
        date=str(date)[:10]
        gm[(season,week,home)]=date;gm[(season,week,away)]=date
    con.close()
    coverage=defaultdict(lambda:[0,0])
    for m,f,y in rows:
        season,week,team=m['season'],m['week'],m['team']
        target=gm.get((season,week,team))
        f.update({k:float('nan') for k in DEPTH_FEATURES})
        coverage[(season,week)][1]+=1
        if not target:continue
        seq=sorted_depth.get((team,m['player_id']),[])
        if not seq:continue
        dx=[x[0] for x in seq]
        idx=bisect_left(dx,target)-1 # strictly PREVIOUS calendar day
        if idx<0:continue
        last_date,(rank,slot)=seq[idx]
        age=(datetime.fromisoformat(target)-datetime.fromisoformat(last_date)).days
        if age>120:continue # never carry ancient season-roster snapshot
        def old_rank(days):
            cutoff=(datetime.fromisoformat(target)-timedelta(days=days)).date().isoformat()
            j=bisect_left(dx,cutoff)-1
            return seq[j][1][0] if j>=0 else float('nan')
        old14=old_rank(14);old30=old_rank(30)
        f.update({
            'depth_pregame_listed':1.,'depth_rank':rank,'depth_slot':slot,
            'depth_is_first':float(rank<=1),'depth_is_top_two':float(rank<=2),
            'depth_is_lower':float(rank>2),
            'depth_rank_change_14d':old14-rank,
            'depth_rank_change_30d':old30-rank,
            'depth_chart_age_days':age,
            'depth_very_recent_move':float(age<=3 and np.isfinite(old14) and abs(rank-old14)>=1),
            'depth_seen_pregame_n':idx+1,
        })
        coverage[(season,week)][0]+=1
    return {'source_records':valid_days,'coverage_tail':{str(k):v for k,v in list(sorted(coverage.items()))[-4:]}}

def build_depth_augmented(rows,pbpdir,ftndir):
    cov=original(rows,pbpdir,ftndir)
    return {'charting':cov, 'depth':build_depth(rows,ftndir)}

base.build_pbp_asof=build_depth_augmented
X=ten.POS
D=DEPTH_FEATURES
OL=ten.OL
P=base.PROPS+base.FTN
B=base.BASE+base.ROLE+base.TEAM+base.DEPTH+base.INJ
base.CFG={
 'depth_role_xgb':ten.xcfg(B+X+D),
 'depth_oline_xgb':ten.xcfg(B+X+OL+D),
 'depth_ftn_xgb':ten.xcfg(B+X+P+D),
 'depth_full_xgb':ten.xcfg(B+X+OL+P+D),
 'depth_role_lgb':ten.lcfg(B+X+D),
 'depth_oline_lgb':ten.lcfg(B+X+OL+D),
 'depth_full_lgb':ten.lcfg(B+X+OL+P+D),
 'depth_full_cat':ten.ccfg(B+X+OL+P+D),
}
assert all(not any(s in f for s in ('spread','implied_team','total_line','odds','price'))
           for spec in base.CFG.values() for f in spec['cols'])

if __name__=='__main__':base.main()