import pandas as pd, numpy as np, csv, time
D='/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/d/'; OUT='/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/intel/'
t0=time.time()
pl=pd.read_csv(D+'players.csv',low_memory=False)[['gsis_id','pfr_id']].dropna().drop_duplicates('pfr_id'); pf2g=dict(zip(pl.pfr_id,pl.gsis_id))
# defender coverage results per game (PFR charting)
pd_=pd.concat([pd.read_csv(D+'pfr_def_%d.csv'%y) for y in range(2018,2027) if __import__('os').path.exists(D+'pfr_def_%d.csv'%y)],ignore_index=True)
pd_=pd_[pd_.game_type=='REG'].copy(); pd_['key']=pd_.season*100+pd_.week
pd_=pd_.rename(columns={'pfr_player_id':'pfr'})[['pfr','key','team','def_targets','def_yards_allowed','def_completions_allowed']].fillna(0)
# defence snaps by game (all seasons; includes the current one)
sn=pd.concat([pd.read_csv(D+'snap_counts_%d.csv'%y,low_memory=False) for y in range(2018,2027)],ignore_index=True)
sn=sn[(sn.game_type=='REG')&(sn.defense_snaps>0)].copy(); sn['key']=sn.season*100+sn.week
def grp(p):
    return 'CB' if p=='CB' else 'S' if p in ('S','FS','SS','DB') else 'LB' if p in ('LB','ILB','OLB','MLB') else 'DL' if p in ('DE','DT','NT','DL') else None
sn['grp']=sn.position.map(grp); sn=sn[sn.grp.notna()]
# rolling shrunk yards/target allowed per defender (previous up-to-16 charted games)
LGY={g:None for g in ('CB','S','LB','DL')}
m=pd_.merge(sn[['pfr_player_id','key','grp']].rename(columns={'pfr_player_id':'pfr'}),on=['pfr','key'],how='left')
lg=m.groupby('grp').apply(lambda x:x.def_yards_allowed.sum()/max(x.def_targets.sum(),1)).to_dict(); print('league yards per target by position group:',{k:round(v,2) for k,v in lg.items()})
pd_=pd_.sort_values('key'); rows=[]
for pfr,x in pd_.groupby('pfr'):
    x=x.reset_index(drop=True)
    for j in range(len(x)):
        h=x.iloc[max(0,j-16):j]
        rows.append((pfr,x.key[j],h.def_targets.sum(),h.def_yards_allowed.sum()))
dq=pd.DataFrame(rows,columns=['pfr','key','h_t','h_y'])
# quality at each (defender, game): what his previous games say (shrunk), only meaningful when he had earlier charted games
sn=sn.rename(columns={'pfr_player_id':'pfr'})
sn=sn.sort_values('key')
# for the NEXT game we need quality as of the end of the latest game: take the row for (pfr,key) with history BEFORE it, then add that game's result later
dq=dq.merge(sn[['pfr','key','grp']],on=['pfr','key'],how='left')
K=25.0
dq['lg']=dq.grp.map(lg); dq['q']=(dq.h_y+K*dq.lg)/(dq.h_t+K)/dq.lg-1.0      # >0 = allows more yards per target than an average player at his position
dq.to_pickle(OUT+'defender_q.pkl')
# snap-weighted recent usage per defender and team (previous 3 team games) -> expected lineup for the coming game
sn['team']=sn.team; usage=sn[['pfr','key','team','grp','defense_pct']]
usage.to_pickle(OUT+'defender_usage.pkl'); print('defender-game rows',len(usage),'quality rows',len(dq),round(time.time()-t0),'s',flush=True)
