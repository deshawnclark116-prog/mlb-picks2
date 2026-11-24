#!/usr/bin/env python3
"""Player-position, role-interruption and positional-specialist challenger.

Past played-game appearances are replayed chronologically. Target-week
participation outcomes are NEVER used until after the forecast features are
constructed. Builds on observed charting + offensive line history.
"""
from collections import defaultdict
import numpy as np
import challenger_gbdt_v4 as fam
import challenger_oline_v3 as ol

base=fam.base
base.xgb=fam.GBDTShim
base.build_injuries=ol.patched_build_injuries
v2=ol.v2

POS=['is_wr','is_te','is_rb','is_fb','games_since_last_active',
    'career_games_before','team_games_before','last8_active_week_count',
    'player_team_changed','last_same_team_streak','games_since_first_active',
    'is_first_nfl_season','recent_team_share_3','tenure_first_month']

prev=base.build_pbp_asof

def replay_tenure(con, market, rows):
    allp=con.execute('''SELECT player_id,season,week,team,position,
          coalesce(carries,0),coalesce(targets,0)
          FROM player_games WHERE season_type='REG' ORDER BY season,week''').fetchall()
    byweek=defaultdict(list)
    for pid,s,w,tm,pos,carry,targ in allp:byweek[(s,w)].append((pid,tm,pos,carry,targ))
    byeval=defaultdict(list)
    for i,r in enumerate(rows):byeval[(r[0]['season'],r[0]['week'])].append(i)
    keys=sorted(set(byweek)|set(byeval))
    order={k:i for i,k in enumerate(keys)}
    last={};birth={};history=defaultdict(list)
    for key in keys:
        current=order[key]
        for i in byeval.get(key,[]):
            meta,f,y=rows[i]
            pid=meta['player_id'];tm=meta['team']
            h=history[pid]
            recent8=[x for x in h if current-x[0]<9]
            teamhist=[x for x in h if x[1]==tm]
            tlast=teamhist[-1] if teamhist else None
            # Player position as of latest *prior* game. Same-week row's position
            # is intrinsically pregame roster info but we avoid using it here.
            pos=h[-1][2] if h else ''
            streak=0
            for x in reversed(h):
                if x[1]!=tm:break
                streak+=1
            gap=current-h[-1][0] if h else float('nan')
            share_recent=base.mdiv(sum(x[3] for x in recent8[-3:]),
                                  sum(x[3]+x[4] for x in recent8[-3:])) if recent8 else float('nan')
            f.update(dict(
                is_wr=float(pos=='WR'),is_te=float(pos=='TE'),
                is_rb=float(pos=='RB'),is_fb=float(pos=='FB'),
                games_since_last_active=gap,
                career_games_before=len(h),team_games_before=len(teamhist),
                last8_active_week_count=len(recent8),
                player_team_changed=float(bool(h) and h[-1][1]!=tm),
                last_same_team_streak=streak,
                games_since_first_active=current-birth[pid] if pid in birth else float('nan'),
                is_first_nfl_season=float(bool(h) and key[0]==h[0][5]),
                recent_team_share_3=share_recent,
                tenure_first_month=float(len(teamhist)<=4)
            ))
        # The target week's realized activity can enter future history only.
        for pid,tm,pos,carry,targ in byweek.get(key,[]):
            if pid not in birth:birth[pid]=current
            history[pid].append((current,tm,pos,float(carry),float(targ),key[0]))


def augmented_asof(rows,pbpdir,ftndir):
    report=prev(rows,pbpdir,ftndir)
    # Reopen read-only database only to replay historical appearances.  The
    # build is deterministic and has no target-week outcome dependencies.
    import sqlite3
    from pathlib import Path
    con=sqlite3.connect(f"file:{Path('nfl_models/nfl_model.sqlite').resolve()}?mode=ro",uri=True)
    replay_tenure(con,'both',rows)
    con.close()
    return report

base.build_pbp_asof=augmented_asof
ROLE=base.BASE+base.ROLE+base.TEAM+base.DEPTH+POS
FTN=ROLE+base.PROPS+base.FTN+base.INJ
OL=ROLE+ol.OL+base.INJ
FULL=FTN+ol.OL

def xcfg(cols,depth=4,rounds=140,halflife=None):
    p=dict(v2.PARAMS,nthread=2,max_depth=depth)
    d={'cols':cols,'params':p,'rounds':rounds}
    if halflife:d['half_life']=halflife
    return d

def lcfg(cols,leaf=15,minleaf=24,rounds=185,halflife=None):
    d={'cols':cols,'params':{'engine':'lgb','leaves':leaf,'min_leaf':minleaf},'rounds':rounds}
    if halflife:d['half_life']=halflife
    return d

def ccfg(cols,depth=5,rounds=240):
    return {'cols':cols,'params':{'engine':'cat','depth':depth},'rounds':rounds}

base.CFG={
 'tenure_xgb':xcfg(ROLE),
 'tenure_ftn_xgb':xcfg(FTN),
 'tenure_oline_xgb':xcfg(OL),
 'tenure_full_xgb':xcfg(FULL),
 'tenure_xgb_shallow':xcfg(ROLE,depth=3,rounds=185),
 'tenure_oline_lgb':lcfg(OL),
 'tenure_lgb':lcfg(ROLE),
 'tenure_ftn_lgb':lcfg(FTN),
 'tenure_full_lgb':lcfg(FULL),
 'tenure_lgb_recent':lcfg(FULL,halflife=45),
 'tenure_cb_role':ccfg(ROLE),
 'tenure_cb_full':ccfg(FULL),
}
assert all(not any(s in f for s in ('spread','implied_team','total_line','odds','price'))
           for spec in base.CFG.values() for f in spec['cols'])

if __name__=='__main__':base.main()