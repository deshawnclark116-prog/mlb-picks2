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
# ---- explicit interactions
mid=(1-df.p_short-df.p_deep).clip(lower=0)
df['mu_depth']=(df.p_short*df.r_ypt_short/LG['short']+mid*df.r_ypt_mid/LG['mid']+df.p_deep*df.r_ypt_deep/LG['deep'])        # defence vs this receiver's target depth mix
df['ypt_matchup']=df.p_ypt*df.mu_depth
df['ypt_cov_mix']=df.r_d_man*df.p_ypt_man+(1-df.r_d_man)*df.p_ypt_zone                                                 # defence man/zone mix x his results vs each
df['exp_pressure']=df.r_o_pressure+df.r_d_pressure-df[['r_o_pressure']].mean().iloc[0]
df['press_x_deep']=df.exp_pressure*df.p_deep                                                                           # protection hurts deep targets most
df['shell_deep']=(df.r_d_COVER_2.fillna(0)+df.r_d_COVER_4.fillna(0)+df.r_d_COVER_6.fillna(0))*df.p_deep               # two-high shells x deep share
df['blitz_x_short']=df.r_d_blitz5*df.p_short
df['ttt_x_adot']=df.r_o_ttt*df.p_adot
RAW=['p_ypt','p_adot','p_deep','p_short','p_ypt_man','p_ypt_zone','p_catch','r_ypt_all','r_ypt_short','r_ypt_mid','r_ypt_deep','r_d_pressure','r_d_blitz5','r_d_man','r_d_COVER_2','r_d_COVER_3','r_d_COVER_4','r_o_pressure','r_o_ttt','r_o_pa','r_o_motion','r_o_p11']
INT=['mu_depth','ypt_matchup','ypt_cov_mix','exp_pressure','press_x_deep','shell_deep','blitz_x_short','ttt_x_adot']
VAR={'base (live v2x inputs)':BASE,'+ raw profiles (no explicit interactions)':BASE+RAW,'+ profiles + explicit interactions':BASE+RAW+INT}
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
        if n!=list(VAR)[0]: d=base-e; r['P(better than base) '+n]=float(np.mean([d[rng.randint(0,len(d),len(d))].mean()>0 for _ in range(1500)]))
    res[wn]=r; print(wn,json.dumps({k:round(v,3) for k,v in r.items()}),flush=True)
json.dump(res,open(OUT+'recv_result.json','w'),indent=1)
b=xgb.train(P,xgb.DMatrix(df[(df.season>=2016)][BASE+RAW+INT].astype(float),label=df[df.season>=2016].y),num_boost_round=140)
g=b.get_score(importance_type='total_gain'); tot=sum(g.values()); print('\nwhat the interaction model leans on (top 15, share of total gain):'); 
for k,v in sorted(g.items(),key=lambda a:-a[1])[:15]: print('  %-18s %.1f%%'%(k,100*v/tot))
