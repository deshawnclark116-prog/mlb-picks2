import pandas as pd, numpy as np, time, json
D='/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/d/'; OUT='/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/intel/'
t0=time.time(); p=pd.read_pickle(OUT+'plays.pkl'); p['key']=p.season*100+p.week
r=p[(p.play_type=='run')&p.rusher_player_id.notna()].copy()
r['box']=r.defenders_in_box.where(r.defenders_in_box.notna(),r.n_defense_box)
r['light']=(r.box<=6).astype(float).where(r.box.notna()); r['heavy']=(r.box>=8).astype(float).where(r.box.notna())
r['expl']=(r.yards_gained>=10).astype(float); r['stuff']=(r.yards_gained<=0).astype(float)
r['gap']=(r.run_location.fillna('na')+'_'+r.run_gap.fillna('na')).where(r.run_gap.notna()&r.run_location.notna())
r['inside']=r.run_gap.isin(['guard']).astype(float).where(r.run_gap.notna()); r['outside']=r.run_gap.isin(['end']).astype(float).where(r.run_gap.notna())
r['y_light']=np.where(r.light==1,r.yards_gained,np.nan); r['y_heavy']=np.where(r.heavy==1,r.yards_gained,np.nan)
r['y_in']=np.where(r.inside==1,r.yards_gained,np.nan); r['y_out']=np.where(r.outside==1,r.yards_gained,np.nan)
r['p11']=r.offense_personnel.astype(str).str.contains('1 RB, 1 TE, 3 WR').astype(float).where(r.offense_personnel.notna())
LG={'ypc':r.yards_gained.mean(),'expl':r.expl.mean(),'stuff':r.stuff.mean(),'heavy':r.heavy.mean(),'light':r.light.mean(),'ypc_light':r.y_light.mean(),'ypc_heavy':r.y_heavy.mean(),'ypc_in':r.y_in.mean(),'ypc_out':r.y_out.mean(),'inside':r.inside.mean(),'outside':r.outside.mean()}
print({k:round(float(v),3) for k,v in LG.items()},flush=True)
def roll(df,team,cols,n=8,hl=4.0,prior=None,k=5.0):
    df=df.sort_values('key').reset_index(drop=True); res={c:np.full(len(df),np.nan) for c in cols}
    for t,idx in df.groupby(team).groups.items():
        idx=list(idx)
        for c in cols:
            v=df.loc[idx,c].to_numpy(float); 
            for j,i in enumerate(idx):
                h=v[max(0,j-n):j]; 
                if len(h)==0: continue
                m=~np.isnan(h); w=0.5**((len(h)-1-np.arange(len(h)))/hl)
                pr=prior[c] if prior and c in prior else np.nanmean(v)
                res[c][i]=((h[m]*w[m]).sum()+k*pr)/(w[m].sum()+k) if m.any() else np.nan
    for c in cols: df['r_'+c]=res[c]
    return df
agg=dict(n=('yards_gained','size'),ypc=('yards_gained','mean'),expl=('expl','mean'),stuff=('stuff','mean'),heavy=('heavy','mean'),light=('light','mean'),ypc_light=('y_light','mean'),ypc_heavy=('y_heavy','mean'),ypc_in=('y_in','mean'),ypc_out=('y_out','mean'),inside=('inside','mean'),outside=('outside','mean'),p11=('p11','mean'))
dg=r.groupby(['game_id','key','defteam']).agg(**agg).reset_index()
dr=roll(dg,'defteam',['ypc','expl','stuff','heavy','light','ypc_light','ypc_heavy','ypc_in','ypc_out','inside','outside'],prior=LG); dr=dr.rename(columns={c:'d'+c for c in dr.columns if c.startswith('r_')})
dr.to_pickle(OUT+'rush_def.pkl')
og=r.groupby(['game_id','key','posteam']).agg(**agg).reset_index()
orr=roll(og,'posteam',['ypc','expl','stuff','heavy','light','ypc_light','ypc_heavy','ypc_in','ypc_out','inside','outside','p11'],prior=LG); orr=orr.rename(columns={c:'o'+c for c in orr.columns if c.startswith('r_')})
orr.to_pickle(OUT+'rush_off.pkl')
# individual rusher profile (previous 12 games with carries), shrunk to league
rg=r.groupby(['rusher_player_id','game_id','key']).agg(n=('yards_gained','size'),ys=('yards_gained','sum'),ex=('expl','sum'),st=('stuff','sum'),
     nl=('light','sum'),nh=('heavy','sum'),yl=('y_light',lambda s:s.sum(skipna=True)),yh=('y_heavy',lambda s:s.sum(skipna=True)),
     ni=('inside','sum'),no=('outside','sum'),yi=('y_in',lambda s:s.sum(skipna=True)),yo=('y_out',lambda s:s.sum(skipna=True)),ng=('inside',lambda s:s.notna().sum())).reset_index().sort_values('key')
# PFR: yards before / after contact per carry (player-level split between blocking and the runner)
pr=pd.concat([pd.read_csv(D+'pfr_rush_%d.csv'%y) for y in range(2018,2027)],ignore_index=True); pr=pr[pr.game_type=='REG']; pr['key']=pr.season*100+pr.week
pl=pd.read_csv(D+'players.csv',low_memory=False)[['gsis_id','pfr_id']].dropna().drop_duplicates('pfr_id'); pf2g=dict(zip(pl.pfr_id,pl.gsis_id)); pr['gsis']=pr.pfr_player_id.map(pf2g)
pr=pr.dropna(subset=['gsis'])[['gsis','key','carries','rushing_yards_before_contact','rushing_yards_after_contact','rushing_broken_tackles']].fillna(0)
rows=[]
pri={g:x.sort_values('key').reset_index(drop=True) for g,x in pr.groupby('gsis')}
LGP={'ybc':pr.rushing_yards_before_contact.sum()/pr.carries.sum(),'yac':pr.rushing_yards_after_contact.sum()/pr.carries.sum(),'bt':pr.rushing_broken_tackles.sum()/pr.carries.sum()}
for pid,x in rg.groupby('rusher_player_id'):
    x=x.reset_index(drop=True); xp=pri.get(pid)
    for j in range(len(x)):
        h=x.iloc[max(0,j-12):j]
        if len(h)==0: continue
        w=0.5**((len(h)-1-np.arange(len(h)))/5.0); S=lambda c:(h[c].to_numpy(float)*w).sum()
        n=S('n'); K=30.0
        def sh(num,den,prior,Kk=K): return (S(num)+Kk*prior)/(S(den)+Kk)
        ybc=yac=bt=np.nan
        if xp is not None:
            hp=xp[xp.key<x.key[j]].tail(12)
            if len(hp):
                wp=0.5**((len(hp)-1-np.arange(len(hp)))/5.0); cs=(hp.carries.to_numpy()*wp).sum()
                ybc=((hp.rushing_yards_before_contact.to_numpy()*wp).sum()+K*LGP['ybc'])/(cs+K); yac=((hp.rushing_yards_after_contact.to_numpy()*wp).sum()+K*LGP['yac'])/(cs+K); bt=((hp.rushing_broken_tackles.to_numpy()*wp).sum()+K*LGP['bt'])/(cs+K)
        rows.append((pid,x.key[j],sh('ys','n',LG['ypc']),sh('ex','n',LG['expl']),sh('st','n',LG['stuff']),sh('yl','nl',LG['ypc_light']),sh('yh','nh',LG['ypc_heavy']),sh('yi','ni',LG['ypc_in']),sh('yo','no',LG['ypc_out']),S('ni')/max(S('ng'),1e-6) if S('ng')>3 else np.nan,ybc,yac,bt))
rp=pd.DataFrame(rows,columns=['pid','key','u_ypc','u_expl','u_stuff','u_ypc_light','u_ypc_heavy','u_ypc_in','u_ypc_out','u_inside_share','u_ybc','u_yac','u_bt'])
rp.to_pickle(OUT+'rusher_profile.pkl'); json.dump({k:float(v) for k,v in LG.items()},open(OUT+'rush_league.json','w')); print('rusher profiles',len(rp),round(time.time()-t0),'s',flush=True)
