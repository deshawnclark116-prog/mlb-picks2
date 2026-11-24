import sys, json, numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0,'/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl'); sys.path.insert(0,'/home/user/mlb-picks2')
import exp5 as exp, nfl_yardage_v2 as v2
OUT='/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/intel/'
LG=json.load(open(OUT+'league_consts.json'))
rows=exp.build('receiving_yards')
BASE=v2.FEATURES+["own_status","own_practice","tm_out_vol","tm_quest_vol","snap3","snap1"]
df=pd.DataFrame([{**{'pid':m['pid'],'season':m['season'],'week':m['week'],'team':m['team'],'y':y},**{c:f.get(c) for c in BASE}} for m,f,y in rows])
df['key']=df.season*100+df.week
_tg=pd.read_pickle(OUT+'team_games.pkl')[['key','posteam','defteam']].rename(columns={'posteam':'team','defteam':'opp'}); df=df.merge(_tg,on=['key','team'],how='left')
rp=pd.read_pickle(OUT+'recv_profile.pkl').drop(columns='game_id'); dr=pd.read_pickle(OUT+'def_profile.pkl'); orr=pd.read_pickle(OUT+'off_profile.pkl')
dr=dr.rename(columns={'defteam':'opp'}); orr=orr.rename(columns={'posteam':'team'})
df=df.merge(rp,left_on=['pid','key'],right_on=['pid','key'],how='left')
dcols=[c for c in dr.columns if c.startswith('r_')]; ocols=[c for c in orr.columns if c.startswith('r_')]
df=df.merge(dr[['key','opp']+dcols],on=['key','opp'],how='left').merge(orr[['key','team']+ocols],on=['key','team'],how='left',suffixes=('','_o'))
print('rows',len(df),'with receiver profile',round(df.p_ypt.notna().mean(),3),'defense profile',round(df.r_ypt_all.notna().mean(),3),'coverage mix known',round(df.r_d_man.notna().mean(),3),'pressure known',round(df.r_d_pressure.notna().mean(),3),flush=True)

import pickle
du=pd.read_pickle(OUT+'defender_usage.pkl'); dq=pd.read_pickle(OUT+'defender_q.pkl')[['pfr','key','q']]
pl=pd.read_csv('/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/d/players.csv',low_memory=False)[['gsis_id','pfr_id']].dropna().drop_duplicates('pfr_id'); pf2g=dict(zip(pl.pfr_id,pl.gsis_id))
du=du.merge(dq,on=['pfr','key'],how='left')
du['gsis']=du.pfr.map(pf2g); du=du.sort_values('key')
TOP={'CB':3,'S':2,'LB':3,'DL':4}
feat=[]
for team,x in du.groupby('team'):
    keys=sorted(x.key.unique())
    for i,k in enumerate(keys):
        prev=keys[max(0,i-3):i]
        if len(prev)<2: continue
        h=x[x.key.isin(prev)]; lastq=h.sort_values('key').groupby('pfr').q.last()
        us=h.groupby(['pfr','grp']).defense_pct.mean().reset_index()
        row={'team':team,'key':k}
        season=k//100; week=k%100
        for g,n in TOP.items():
            t=us[us.grp==g].nlargest(n,'defense_pct'); qs=[];out=0
            for pfr,pct in zip(t.pfr,t.defense_pct):
                gs=pf2g.get(pfr); st=exp.INJ.get((gs,season,week),(0,-1))[0] if gs else 0
                if st>=2: out+=1; qs.append(0.10)                    # a listed-out starter is replaced by a below-average body
                else: qs.append(lastq.get(pfr,np.nan))
            row[g.lower()+'_q']=np.nanmean(qs) if len(qs) else np.nan; row[g.lower()+'_out']=out
        feat.append(row)
lf=pd.DataFrame(feat).rename(columns={'team':'opp'})
df=df.merge(lf,on=['opp','key'],how='left'); print('lineup features join rate',round(df.cb_q.notna().mean(),3),flush=True)
pos=pd.read_pickle(OUT+'pos_map.pkl') if False else None
import sqlite3
con=sqlite3.connect('file:/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/nfl_ext.sqlite?mode=ro',uri=True)
pm=dict(con.execute("select player_id,position from player_games order by season,week").fetchall()); df['pos']=df.pid.map(pm).map({'WR':0,'TE':1,'RB':2,'FB':2}).fillna(0)
df['cov_q_pos']=np.where(df.pos==0,df.cb_q,np.where(df.pos==1,0.5*df.lb_q+0.5*df.s_q,df.lb_q))          # who would cover him: corners for WR, safeties/LB for TE, LB for RB
df['cov_out_pos']=np.where(df.pos==0,df.cb_out,np.where(df.pos==1,df.lb_out+df.s_out,df.lb_out))
PP=['p_ypt','p_adot','p_deep','p_short','p_catch','p_ypt_man','p_ypt_zone']
LU=['cb_q','s_q','lb_q','dl_q','cb_out','s_out','lb_out','dl_out','cov_q_pos','cov_out_pos','pos']
VAR={'BASE + player profile (current best)':BASE+PP,'+ defence lineup quality (who is actually in the secondary / front)':BASE+PP+LU,'+ lineup, only the role-matched coverage score':BASE+PP+['cov_q_pos','cov_out_pos','pos']}
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
        if n!=list(VAR)[0]: d=base-e; r['P(better) '+n[:30]]=float(np.mean([d[rng.randint(0,len(d),len(d))].mean()>0 for _ in range(1500)]))
    res[wn]=r; print(wn,json.dumps({k:round(v,3) for k,v in r.items()}),flush=True)
json.dump(res,open(OUT+'lineup_result.json','w'),indent=1)
