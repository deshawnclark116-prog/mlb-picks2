#!/usr/bin/env python3
"""Add strictly pregame line/injury/roster disruption features to FTN challenger.

Time ordering: for each week, read current week's PRE-GAME injury designations,
combine them with OL/DL/QB snap histories through the PREVIOUS completed week;
then update those histories after forecasting the current week.
"""
import csv,math,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collections import defaultdict
import numpy as np
import nfl_yardage_v2 as v2
import challenger_ftn_v2 as base

OL=['off_ol_start_out','off_ol_start_doubtful','off_ol_start_questionable',
    'off_ol_out_weight','off_ol_questionable_weight','off_ol_practice_limited',
    'off_ol_continuity','off_ol_recent_starters','off_ol_recent_usage',
    'opp_dl_start_out','opp_dl_questionable','opp_dl_out_weight',
    'qb_out','qb_questionable','qb_practice_limited',
    'off_ol_snap_change','opp_dl_snap_change','OL_recent_injury_pressure']

# Keep market-free rows. Compare lines alone, lines + recent role, FTN lines,
# and conservative regularized variants. Do not retune to the target window.
FEATURES = {
 'ol_core': base.BASE+base.TEAM+OL,
 'ol_role': base.BASE+base.ROLE+base.TEAM+base.DEPTH+OL,
 'ol_role_inj': base.BASE+base.ROLE+base.TEAM+base.DEPTH+base.INJ+OL,
 'ol_full_ftn':base.BASE+base.ROLE+base.TEAM+base.DEPTH+base.PROPS+base.FTN+base.INJ+OL,
 'ol_quality':base.BASE+base.TEAM+base.PROPS+base.FTN+OL,
}
base.CFG = {name:{'cols':cols,'params':dict(v2.PARAMS,nthread=3),'rounds':140}
            for name,cols in FEATURES.items()}
base.CFG.update({
 'ol_role_depth3':{'cols':FEATURES['ol_role'],'params':dict(v2.PARAMS,nthread=3,max_depth=3,min_child_weight=12),'rounds':175},
 'ol_role_recent':{'cols':FEATURES['ol_role'],'params':dict(v2.PARAMS,nthread=3),'rounds':140,'half_life':45},
 'ol_full_depth3':{'cols':FEATURES['ol_full_ftn'],'params':dict(v2.PARAMS,nthread=3,max_depth=3,min_child_weight=12),'rounds':175},
 'ol_full_recent':{'cols':FEATURES['ol_full_ftn'],'params':dict(v2.PARAMS,nthread=3),'rounds':140,'half_life':45},
})
assert all(not any(s in col for s in ('spread','implied_team','total_line','odds','price'))
           for spec in base.CFG.values() for col in spec['cols'])

_OLD = base.build_injuries

def name_key(x):return v2.norm_name(x or '')

def posgroup(pos):
    pos=(pos or '').upper()
    if pos in ('T','LT','RT','G','LG','RG','C','OT','OG','OL'):return 'OL'
    if pos in ('DE','DT','NT','DL','EDGE','ED','OLB','LB','ILB','MLB'):return 'DL'
    if pos=='QB':return 'QB'
    return None

def build_ol(con,rows,featdir):
    injury_week=defaultdict(lambda:defaultdict(dict))
    snap_week=defaultdict(list)
    check={}
    for season in (2023,2024,2025,2026):
        inf=Path(featdir)/f'injuries_{season}.csv'
        with inf.open(newline='') as inp:
            for r in csv.DictReader(inp):
                if r.get('game_type')!='REG':continue
                week=round(base.num(r.get('week')))
                tm=r.get('team');nm=name_key(r.get('full_name'))
                if not tm or not nm or week<=0:continue
                k=(season,week)
                stat={'Out':3,'Doubtful':2,'Questionable':1}.get(r.get('report_status'),0)
                practice={'did not participate in practice':2,'limited participation in practice':1,'full participation in practice':0}.get((r.get('practice_status') or '').lower(),-1)
                # Rows are repeated for separate practice days. Final pregame designation
                # is a max over pregame statuses (the original injury csv has no timestamp).
                prev=injury_week[k][tm].get(nm,(0,-1,''))
                injury_week[k][tm][nm]=(max(stat,prev[0]),max(practice,prev[1]),r.get('position') or prev[2])
        sf=Path(featdir)/f'snap_counts_{season}.csv'
        with sf.open(newline='') as inp:
            for r in csv.DictReader(inp):
                if r.get('game_type')!='REG':continue
                kind=posgroup(r.get('position'))
                tm=r.get('team');nm=name_key(r.get('player'))
                week=round(base.num(r.get('week')))
                if not kind or not tm or not nm or week<=0:continue
                pct=base.num(r.get('offense_pct') if kind in ('OL','QB') else r.get('defense_pct'))
                snap_week[(season,week)].append((tm,nm,kind,pct))
    byweek=defaultdict(list)
    for i,r in enumerate(rows):byweek[(r[0]['season'],r[0]['week'])].append(i)
    keys=sorted(set(byweek)|set(snap_week))
    hist=defaultdict(list);last_team={};last_index={};span={}
    for step,key in enumerate(keys):
        active=defaultdict(list)
        for (tm,nm,kind),h in hist.items():
            if step-last_index[(tm,nm,kind)]>4 or not h:continue
            recent=[x[1] for x in h[-3:]]
            av=float(np.mean(recent))
            if av>=.30 or (kind=='QB' and av>=.15):
                active[(tm,kind)].append((nm,av,recent[-1],len(recent)))
        for i in byweek.get(key,[]):
            m,f,y=rows[i];tm=m['team'];opp=m['opp']
            injuries=injury_week[key]
            ol=active.get((tm,'OL'),[])
            dl=active.get((opp,'DL'),[])
            qbs=active.get((tm,'QB'),[])
            # LT/RT/G/C with substantial past snaps are probable starters.
            # No same-game snap information can enter active or these flags.
            j=injuries.get(tm,{})
            olout=sum(1 for nm,av,last,n in ol if j.get(nm,(0,))[0]>=3)
            oldoubt=sum(1 for nm,av,last,n in ol if j.get(nm,(0,))[0]==2)
            olquest=sum(1 for nm,av,last,n in ol if j.get(nm,(0,))[0]==1)
            olweight=sum(av for nm,av,last,n in ol if j.get(nm,(0,))[0]>=2)
            olqweight=sum(av for nm,av,last,n in ol if j.get(nm,(0,))[0]==1)
            olprac=sum(av for nm,av,last,n in ol if j.get(nm,(0,-1))[1]==2 and j.get(nm,(0,-1))[0]==0)
            continuity=sum(n>=3 and av>=.60 for nm,av,last,n in ol)
            oluse=sum(av for nm,av,last,n in ol)
            olchange=sum(last-av for nm,av,last,n in ol)
            o=injuries.get(opp,{})
            dlout=sum(1 for nm,av,last,n in dl if o.get(nm,(0,))[0]>=2)
            dlquest=sum(1 for nm,av,last,n in dl if o.get(nm,(0,))[0]==1)
            dlweight=sum(av for nm,av,last,n in dl if o.get(nm,(0,))[0]>=2)
            dlchange=sum(last-av for nm,av,last,n in dl)
            qbout=sum(1 for nm,av,last,n in qbs if j.get(nm,(0,))[0]>=2)
            qbquest=sum(1 for nm,av,last,n in qbs if j.get(nm,(0,))[0]==1)
            qbprac=sum(1 for nm,av,last,n in qbs if j.get(nm,(0,-1))[1]>=1)
            f.update(dict(
              off_ol_start_out=olout,off_ol_start_doubtful=oldoubt,
              off_ol_start_questionable=olquest,
              off_ol_out_weight=olweight,off_ol_questionable_weight=olqweight,
              off_ol_practice_limited=olprac,
              off_ol_continuity=continuity,off_ol_recent_starters=len(ol),
              off_ol_recent_usage=oluse,
              opp_dl_start_out=dlout,opp_dl_questionable=dlquest,
              opp_dl_out_weight=dlweight,
              qb_out=qbout,qb_questionable=qbquest,
              qb_practice_limited=qbprac,
              off_ol_snap_change=olchange,opp_dl_snap_change=dlchange,
              OL_recent_injury_pressure=olweight-.30*dlweight,
            ))
        # Update after features only. Cannot see current-game snaps at prediction time.
        for tm,nm,kind,pct in snap_week.get(key,[]):
            h=hist[(tm,nm,kind)];h.append((key,pct));last_index[(tm,nm,kind)]=step
    return {'OL_week_counts':{str(k):len(v) for k,v in list(snap_week.items())[-3:]}}

def patched_build_injuries(con,market,rows,injdir):
    first=_OLD(con,market,rows,injdir)
    second=build_ol(con,rows,injdir)
    return {'generic_injuries':first,'oline_injuries':second}

base.build_injuries=patched_build_injuries

if __name__=='__main__':
    base.main()