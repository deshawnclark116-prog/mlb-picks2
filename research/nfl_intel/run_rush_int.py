import sys, json, numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0,'/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl'); sys.path.insert(0,'/home/user/mlb-picks2')
import exp5 as exp, nfl_yardage_v2 as v2
OUT='/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/intel/'
LG=json.load(open(OUT+'league_consts.json'))
rows=exp.build('rushing_yards')
BASE=v2.FEATURES+["own_status","own_practice","tm_out_vol","tm_quest_vol","snap3","snap1"]
df=pd.DataFrame([{**{'pid':m['pid'],'season':m['season'],'week':m['week'],'team':m['team'],'y':y},**{c:f.get(c) for c in BASE}} for m,f,y in rows])
df['key']=df.season*100+df.week
_tg=pd.read_pickle(OUT+'team_games.pkl')[['key','posteam','defteam']].rename(columns={'posteam':'team','defteam':'opp'}); df=df.merge(_tg,on=['key','team'],how='left')

import sqlite3, pickle
LGR=json.load(open(OUT+'rush_league.json'))
rp=pd.read_pickle(OUT+'rusher_profile.pkl'); drf=pd.read_pickle(OUT+'rush_def.pkl').rename(columns={'defteam':'opp'}); orf=pd.read_pickle(OUT+'rush_off.pkl').rename(columns={'posteam':'team'})
dcols=[c for c in drf.columns if c.startswith('dr_')]; ocols=[c for c in orf.columns if c.startswith('or_')]
drf=drf.rename(columns={c:c.replace('dr_','dd_') for c in dcols}); dcols=[c.replace('dr_','dd_') for c in dcols]
df=df.merge(rp,on=['pid','key'],how='left').merge(drf[['key','opp']+dcols],on=['key','opp'],how='left').merge(orf[['key','team']+ocols],on=['key','team'],how='left')
print('rows',len(df),'rusher profile',round(df.u_ypc.notna().mean(),3),'defence run profile',round(df.dd_ypc.notna().mean(),3),'offence run profile',round(df.or_ypc.notna().mean(),3),flush=True)
du=pd.read_pickle(OUT+'defender_usage.pkl'); pl=pd.read_csv('/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/d/players.csv',low_memory=False)[['gsis_id','pfr_id']].dropna().drop_duplicates('pfr_id'); pf2g=dict(zip(pl.pfr_id,pl.gsis_id))
du['gsis']=du.pfr.map(pf2g); du=du.sort_values('key'); feat=[]
for team,x in du.groupby('team'):
    keys=sorted(x.key.unique())
    for i,k in enumerate(keys):
        prev=keys[max(0,i-3):i]
        if len(prev)<2: continue
        h=x[x.key.isin(prev)]; us=h.groupby(['pfr','grp']).defense_pct.mean().reset_index(); season=k//100; week=k%100; row={'opp':team,'key':k}
        for g,n in (('DL',4),('LB',3)):
            t=us[us.grp==g].nlargest(n,'defense_pct'); row[g.lower()+'_out']=sum(1 for pfr in t.pfr if (pf2g.get(pfr) and exp.INJ.get((pf2g[pfr],season,week),(0,-1))[0]>=2))
        feat.append(row)
df=df.merge(pd.DataFrame(feat),on=['opp','key'],how='left')
# offensive line: who started last game vs who is listed out now (snap counts positions T,G,C)
sn=pd.concat([pd.read_csv('/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/d/snap_counts_%d.csv'%y,low_memory=False) for y in range(2018,2027)],ignore_index=True)
sn=sn[(sn.game_type=='REG')&sn.position.isin(['T','G','C','OL','OT','OG'])&(sn.offense_pct>=0.5)].copy(); sn['key']=sn.season*100+sn.week; sn['gsis']=sn.pfr_player_id.map(pf2g)
olrows=[]
for team,x in sn.groupby('team'):
    ks=sorted(x.key.unique())
    for i,k in enumerate(ks):
        if i==0: continue
        last=set(x[x.key==ks[i-1]].gsis.dropna()); season=k//100; week=k%100
        olrows.append({'team':team,'key':k,'ol_out':sum(1 for g in last if exp.INJ.get((g,season,week),(0,-1))[0]>=2),'ol_q':sum(1 for g in last if exp.INJ.get((g,season,week),(0,-1))[0]==1)})
# a team's key must match the NEXT game week; line-up of previous game + this week's injury statuses
df=df.merge(pd.DataFrame(olrows),on=['team','key'],how='left')
df['gap_matchup']=df.u_inside_share*df.dd_ypc_in/LGR['ypc_in']+(1-df.u_inside_share)*df.dd_ypc_out/LGR['ypc_out']
df['p_heavy']=0.5*(df.dd_heavy+df.or_heavy); df['p_light']=0.5*(df.dd_light+df.or_light)
df['box_matchup']=df.p_heavy*df.u_ypc_heavy+df.p_light*df.u_ypc_light+(1-df.p_heavy-df.p_light)*df.u_ypc
df['ypc_matchup']=df.u_ypc*df.dd_ypc/LGR['ypc']
df['blocking_x_front']=df.u_ybc*df.dd_ypc/LGR['ypc']
U=['u_ypc','u_expl','u_stuff','u_ypc_light','u_ypc_heavy','u_ypc_in','u_ypc_out','u_inside_share','u_ybc','u_yac','u_bt']
DR=[c for c in dcols]; OR=ocols
FR=['dl_out','lb_out']; OL=['ol_out','ol_q']
INT=['gap_matchup','box_matchup','ypc_matchup','blocking_x_front','p_heavy']
VAR={'base (live v2x inputs)':BASE,'+ ALL five interactions':BASE+INT}
for f in INT: VAR['+ only '+f]=BASE+[f]
for f in INT: VAR['five interactions minus '+f]=BASE+[g for g in INT if g!=f]
VAR['+ interactions + rusher profile']=BASE+INT+U
VAR['+ interactions + rusher profile + offence/defence run profiles']=BASE+INT+U+DR+OR
P=dict(v2.PARAMS); P['nthread']=4
def wf(lo,hi):
    keys=sorted(df.key.unique()); pred={n:pd.Series(np.nan,index=df.index) for n in VAR}
    for k in keys:
        if not (lo<=k<=hi): continue
        tr=df[(df.key<k)&(df.season>=2016)]; te=df[df.key==k]
        if len(tr)<300 or not len(te): continue
        for n,cols in VAR.items():
            b=xgb.train(P,xgb.DMatrix(tr[cols].astype(float),label=tr.y),num_boost_round=140)
            pred[n].loc[te.index]=b.predict(xgb.DMatrix(te[cols].astype(float)))
    return pred
res={}
for wn,(lo,hi) in (('DEVELOPMENT 2024 wk6-18',(202406,202418)),('CONFIRMATION 2025 wk6 - 2026 wk11',(202506,202611))):
    pred=wf(lo,hi); m=pred[list(VAR)[0]].notna(); y=df.y[m].to_numpy(float); rng=np.random.RandomState(1)
    base=np.abs(pred[list(VAR)[0]][m].to_numpy()-y); r={'n':int(m.sum())}
    for n in VAR:
        e=np.abs(pred[n][m].to_numpy()-y); r[n]=float(e.mean())
        if n!=list(VAR)[0]: d=base-e; r['P(better) '+n]=float(np.mean([d[rng.randint(0,len(d),len(d))].mean()>0 for _ in range(1500)]))
    res[wn]=r; print(wn,json.dumps({k:round(v,3) for k,v in r.items()}),flush=True)
json.dump(res,open(OUT+'rush_int_result.json','w'),indent=1)
