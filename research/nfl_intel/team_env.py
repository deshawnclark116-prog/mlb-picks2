import pandas as pd, numpy as np, xgboost as xgb, json, sys
OUT='/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/intel/'
p=pd.read_pickle(OUT+'plays.pkl')
p['key']=p.season*100+p.week
p['is_pass']=(p.play_type=='pass').astype(int)
p['spread_team']=np.where(p.posteam_type=='home',p.spread_line,-p.spread_line)
p['neutral']=((p.qtr<=3)&(p.score_differential.abs()<=7)&(p.down.isin([1,2]))&(p.wp.between(0.2,0.8))).astype(int)
g=p.groupby(['game_id','season','week','key','posteam','defteam'])
tg=g.agg(plays=('play_type','size'),passes=('is_pass','sum'),spread=('spread_team','first'),total=('total_line','first'),xpass=('xpass','mean'),pass_oe=('pass_oe','mean'),
         n_plays_neutral=('neutral','sum'),yds=('yards_gained','sum')).reset_index()
nt=p[p.neutral==1].groupby(['game_id','posteam']).agg(neutral_pass=('is_pass','mean')).reset_index()
tg=tg.merge(nt,on=['game_id','posteam'],how='left')
tg['rushes']=tg.plays-tg.passes; tg['pass_rate']=tg.passes/tg.plays
home=p.groupby(['game_id','posteam']).posteam_type.first().reset_index().rename(columns={'posteam_type':'ptype'})
tg=tg.merge(home,on=['game_id','posteam']); tg['is_home']=(tg.ptype=='home').astype(float)
tg=tg.sort_values(['key','game_id']).reset_index(drop=True)
# as-of rolling features per team (decayed, strictly earlier games) and for the defense faced
def asof(df,team_col,val,n=8,hl=4.0,prior=None):
    out=np.full(len(df),np.nan); 
    for t,idx in df.groupby(team_col).groups.items():
        idx=list(idx); vals=df.loc[idx,val].to_numpy(float)
        for j,i in enumerate(idx):
            h=vals[max(0,j-n):j]; 
            if len(h)==0: continue
            w=0.5**((len(h)-1-np.arange(len(h)))/hl); m=~np.isnan(h)
            out[i]=(h[m]*w[m]).sum()/w[m].sum() if m.any() else np.nan
    return out
for c in ('plays','passes','rushes','pass_rate','pass_oe','neutral_pass','xpass'):
    tg['own_'+c]=asof(tg,'posteam',c)
    tg['opp_faced_'+c]=asof(tg,'defteam',c)            # what offenses did against this defense
tg['spread_abs']=tg.spread.abs(); tg['implied']=tg.total/2+tg.spread/2
tg['opp_def_own_plays']=np.nan
# opponent's own pace (the opponent offense's plays) and the matchup pace
opp_plays=tg[['game_id','posteam','own_plays']].rename(columns={'posteam':'defteam','own_plays':'opp_own_plays'})
tg=tg.merge(opp_plays,on=['game_id','defteam'],how='left')
tg.to_pickle(OUT+'team_games.pkl')
F_BASE=['spread','total','implied','is_home']
F_OWN=['own_plays','own_passes','own_rushes','own_pass_rate']
F_WHY=['own_pass_oe','own_neutral_pass','own_xpass','opp_faced_plays','opp_faced_passes','opp_faced_rushes','opp_faced_pass_rate','opp_faced_pass_oe','opp_own_plays','spread_abs']
P=dict(objective='reg:squarederror',max_depth=3,eta=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=20,reg_lambda=5.0,nthread=4,seed=7)
def wf(target,feats,start_keys,rounds=200,train_from=2016):
    out={}
    d=tg.dropna(subset=['own_plays','opp_faced_plays'])
    keys=sorted(d.key.unique()); pred=pd.Series(np.nan,index=d.index)
    for k in keys:
        if k<start_keys[0] or k>start_keys[1]: continue
        tr=d[(d.key<k)&(d.season>=train_from)]; te=d[d.key==k]
        if len(tr)<300: continue
        b=xgb.train(P,xgb.DMatrix(tr[feats],label=tr[target]),num_boost_round=rounds)
        pred.loc[te.index]=b.predict(xgb.DMatrix(te[feats]))
    m=pred.notna(); return d.loc[m,target].to_numpy(float),pred[m].to_numpy(float),d.loc[m]
res={}
windows={'development_2024_w6+':(202406,202418),'confirmation_2025w6_to_2026w11':(202506,202611)}
for target,naive in (('passes','own_passes'),('rushes','own_rushes'),('plays','own_plays')):
    for wn,wk in windows.items():
        a,p1,sub=wf(target,F_BASE+F_OWN,wk)
        _,p2,_=wf(target,F_BASE+F_OWN+F_WHY,wk)
        nv=sub[naive].to_numpy(float)
        r={'n':len(a),'mean':float(a.mean()),'naive_last8_avg':float(np.abs(nv-a).mean()),'script_model(spread,total,own volume)':float(np.abs(p1-a).mean()),'plus_tendency_and_opponent(PROE,neutral rate,opp faced,pace)':float(np.abs(p2-a).mean())}
        rng=np.random.RandomState(1); d=np.abs(p1-a)-np.abs(p2-a); r['P(tendencies help)']=float(np.mean([d[rng.randint(0,len(d),len(d))].mean()>0 for _ in range(1500)]))
        res[(target,wn)]=r; print(target,wn,json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()}),flush=True)
