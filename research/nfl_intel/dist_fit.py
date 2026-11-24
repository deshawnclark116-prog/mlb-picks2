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

PP_LIVE=['p_ypt','p_adot','p_deep','p_short','p_catch']
recv=df.copy(); recv['market']='receiving_yards'; RC_FEATS=BASE+PP_LIVE
# rushing frame: live model inputs only (no promoted rushing features)
rows_r=exp.build('rushing_yards')
rush=pd.DataFrame([{**{'pid':m['pid'],'season':m['season'],'week':m['week'],'team':m['team'],'y':y},**{c:f.get(c) for c in BASE}} for m,f,y in rows_r])
rush['key']=rush.season*100+rush.week; rush=rush.merge(_tg,on=['key','team'],how='left'); rush['market']='rushing_yards'
QS=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95]
P=dict(objective='reg:quantileerror',quantile_alpha=np.array(QS),max_depth=4,eta=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=10,reg_lambda=5.0,nthread=4,seed=20260927)
out=[]
for mkt,frame,feats in (('receiving_yards',recv,RC_FEATS),('rushing_yards',rush,BASE)):
    keys=sorted(frame.key.unique())
    for k in keys:
        if not (202401<=k<=202611): continue
        if not (202406<=k<=202418 or 202506<=k<=202611): continue
        tr=frame[(frame.key<k)&(frame.season>=2016)]; te=frame[frame.key==k]
        if len(tr)<300 or not len(te): continue
        b=xgb.train(P,xgb.DMatrix(tr[feats].astype(float),label=tr.y),num_boost_round=120)
        q=b.predict(xgb.DMatrix(te[feats].astype(float)))
        o=te[['pid','season','week','key','team','opp','y','market']].copy()
        for i,qq in enumerate(QS): o['q%02d'%int(round(qq*100))]=q[:,i]
        out.append(o)
    pd.concat([o for o in out if o.market.iloc[0]==mkt],ignore_index=True).to_pickle(OUT+'quantile_preds_%s.pkl'%mkt); print(mkt,'done',flush=True)
res=pd.concat(out,ignore_index=True); res.to_pickle(OUT+'quantile_preds.pkl'); print(res.shape)
