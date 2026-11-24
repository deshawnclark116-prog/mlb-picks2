import pandas as pd, numpy as np, time
OUT='/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/intel/'
t0=time.time()
p=pd.read_pickle(OUT+'plays.pkl'); p['key']=p.season*100+p.week
pas=p[p.play_type=='pass'].copy()
pas['tgt']=(pas.receiver_player_id.notna()).astype(int)
pas['air']=pas.air_yards
pas['deep']=((pas.air>=20)&pas.air.notna()).astype(int); pas['short']=((pas.air<=5)&pas.air.notna()).astype(int); pas['mid']=((pas.air>5)&(pas.air<20)).astype(int)
pas['pressure']=pas.was_pressure.map({True:1.0,False:0.0,'True':1.0,'False':0.0}).astype(float)
pas['blitz5']=(pas.number_of_pass_rushers>=5).astype(float).where(pas.number_of_pass_rushers.notna())
pas['man']=(pas.defense_man_zone_type=='MAN_COVERAGE').astype(float).where(pas.defense_man_zone_type.notna())
for c in ('COVER_0','COVER_1','COVER_2','COVER_3','COVER_4','COVER_6'): pas['sh_'+c]=(pas.defense_coverage_type==c).astype(float).where(pas.defense_coverage_type.notna())
pas['pa']=pas.is_play_action.astype(float) if 'is_play_action' in pas else np.nan
pas['motion']=pas.is_motion.astype(float) if 'is_motion' in pas else np.nan
pas['ttt']=pas.time_to_throw
pas['p11']=(pas.offense_personnel.astype(str).str.contains('1 RB, 1 TE, 3 WR')).astype(float).where(pas.offense_personnel.notna())
# ---------- defense per-game profile (what this defense allows / does on pass plays)
tp=pas[pas.tgt==1]
dg=pas.groupby(['game_id','key','defteam']).agg(d_pressure=('pressure','mean'),d_blitz5=('blitz5','mean'),d_man=('man','mean'),d_ttt=('ttt','mean'),d_sack=('sack','mean'),
   **{'d_'+c:('sh_'+c,'mean') for c in ('COVER_0','COVER_1','COVER_2','COVER_3','COVER_4','COVER_6')}).reset_index()
t_allowed=tp.groupby(['game_id','defteam']).agg(t_n=('tgt','size'),t_yds=('yards_gained','sum'),t_epa=('epa','sum'),
   s_n=('short','sum'),m_n=('mid','sum'),d_n=('deep','sum')).reset_index()
for b,bn in (('short','s'),('mid','m'),('deep','d')):
    t_allowed[bn+'_yds']=tp[tp[b]==1].groupby(['game_id','defteam']).yards_gained.sum().reindex(pd.MultiIndex.from_frame(t_allowed[['game_id','defteam']])).fillna(0).to_numpy()
dg=dg.merge(t_allowed,on=['game_id','defteam'],how='left')
# offense per-game profile
og=pas.groupby(['game_id','key','posteam']).agg(o_pressure=('pressure','mean'),o_ttt=('ttt','mean'),o_pa=('pa','mean'),o_motion=('motion','mean'),o_p11=('p11','mean'),o_sack=('sack','mean'),o_deep=('deep','mean')).reset_index()
def rolling_team(df,team,cols,n=8,hl=4.0):
    df=df.sort_values('key').reset_index(drop=True); res={c:np.full(len(df),np.nan) for c in cols}
    for t,idx in df.groupby(team).groups.items():
        idx=list(idx)
        for c in cols:
            v=df.loc[idx,c].to_numpy(float)
            for j,i in enumerate(idx):
                h=v[max(0,j-n):j]
                if len(h)==0: continue
                w=0.5**((len(h)-1-np.arange(len(h)))/hl); m=~np.isnan(h)
                res[c][i]=(h[m]*w[m]).sum()/w[m].sum() if m.any() else np.nan
    for c in cols: df['r_'+c]=res[c]
    return df
dcols=['d_pressure','d_blitz5','d_man','d_ttt','d_sack','d_COVER_0','d_COVER_1','d_COVER_2','d_COVER_3','d_COVER_4','d_COVER_6']
dr=rolling_team(dg.assign(**{'s_rate':dg.s_n/dg.t_n.clip(lower=1)}),'defteam',dcols)
# defence yards-per-target allowed by depth bin, as rolling SUM ratios (more stable than ratios of means)
def rolling_sum_ratio(df,team,num,den,n=8,hl=4.0,k=6.0,league=None):
    df=df.sort_values('key').reset_index(drop=True); out=np.full(len(df),np.nan)
    for t,idx in df.groupby(team).groups.items():
        idx=list(idx); a=df.loc[idx,num].to_numpy(float); b=df.loc[idx,den].to_numpy(float)
        for j,i in enumerate(idx):
            aa=a[max(0,j-n):j]; bb=b[max(0,j-n):j]
            if len(aa)==0: continue
            w=0.5**((len(aa)-1-np.arange(len(aa)))/hl)
            out[i]=((aa*w).sum()+k*league)/((bb*w).sum()+k)
    return out
dr=dr.sort_values('key').reset_index(drop=True)
lg=lambda n,d:(dg[n].sum()/dg[d].sum())
dr['r_ypt_all']=rolling_sum_ratio(dr,'defteam','t_yds','t_n',league=lg('t_yds','t_n'))
dr['r_ypt_short']=rolling_sum_ratio(dr,'defteam','s_yds','s_n',league=lg('s_yds','s_n'))
dr['r_ypt_mid']=rolling_sum_ratio(dr,'defteam','m_yds','m_n',league=lg('m_yds','m_n'))
dr['r_ypt_deep']=rolling_sum_ratio(dr,'defteam','d_yds','d_n',league=lg('d_yds','d_n'))
LG={'short':lg('s_yds','s_n'),'mid':lg('m_yds','m_n'),'deep':lg('d_yds','d_n'),'all':lg('t_yds','t_n')}
dr.to_pickle(OUT+'def_profile.pkl')
orr=rolling_team(og,'posteam',['o_pressure','o_ttt','o_pa','o_motion','o_p11','o_sack','o_deep']); orr.to_pickle(OUT+'off_profile.pkl')
print('team profiles',len(dr),len(orr),round(time.time()-t0),'s',LG,flush=True)
# ---------- receiver profile (his own target type and results, rolling over his previous games)
rg=tp.groupby(['receiver_player_id','game_id','key']).agg(n=('tgt','size'),yds=('yards_gained','sum'),air=('air','sum'),air_n=('air',lambda s:s.notna().sum()),deep=('deep','sum'),short=('short','sum'),
    rec=('complete_pass','sum'),man_n=('man',lambda s:s.fillna(-1).eq(1).sum()),zone_n=('man',lambda s:s.fillna(-1).eq(0).sum())).reset_index()
tp2=tp.assign(man_y=np.where(tp.man==1,tp.yards_gained,0.0),zone_y=np.where(tp.man==0,tp.yards_gained,0.0))
rg=rg.merge(tp2.groupby(['receiver_player_id','game_id']).agg(man_y=('man_y','sum'),zone_y=('zone_y','sum')).reset_index(),on=['receiver_player_id','game_id'])
rg=rg.sort_values('key').reset_index(drop=True)
L={'ypt':rg.yds.sum()/rg.n.sum(),'adot':rg.air.sum()/rg.air_n.sum(),'deep':rg.deep.sum()/rg.air_n.sum(),'short':rg.short.sum()/rg.air_n.sum(),'man_ypt':rg.man_y.sum()/rg.man_n.sum(),'zone_ypt':rg.zone_y.sum()/rg.zone_n.sum(),'catch':rg.rec.sum()/rg.n.sum()}
print('league receiver baselines',{k:round(v,3) for k,v in L.items()},flush=True)
rows=[]
for pid,x in rg.groupby('receiver_player_id'):
    x=x.reset_index(drop=True)
    for j in range(len(x)):
        h=x.iloc[max(0,j-12):j]                  # his previous (up to) 12 games with targets
        if len(h)==0: continue
        w=0.5**((len(h)-1-np.arange(len(h)))/5.0)
        S=lambda c:(h[c].to_numpy(float)*w).sum()
        n=S('n'); an=S('air_n'); K=25.0
        rows.append((pid,x.game_id[j],x.key[j],(S('yds')+K*L['ypt'])/(n+K),(S('air')+K*L['adot'])/(an+K),(S('deep')+K*L['deep'])/(an+K),(S('short')+K*L['short'])/(an+K),
                     (S('man_y')+K*L['man_ypt'])/(S('man_n')+K),(S('zone_y')+K*L['zone_ypt'])/(S('zone_n')+K),(S('rec')+K*L['catch'])/(n+K),n))
rp=pd.DataFrame(rows,columns=['pid','game_id','key','p_ypt','p_adot','p_deep','p_short','p_ypt_man','p_ypt_zone','p_catch','p_n'])
rp.to_pickle(OUT+'recv_profile.pkl'); print('receiver profiles',len(rp),round(time.time()-t0),'s',flush=True)
import json; json.dump(LG|{'recv':L},open(OUT+'league_consts.json','w'))
