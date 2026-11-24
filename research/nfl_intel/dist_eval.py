import pandas as pd, numpy as np, json
from sklearn.isotonic import IsotonicRegression
OUT='/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/intel/'
d=pd.read_pickle(OUT+'quantile_preds.pkl').reset_index(drop=True)
QS=[5,10,20,30,40,50,60,70,80,90,95]; qc=['q%02d'%q for q in QS]; P=np.array(QS)/100.0
d[qc]=np.maximum.accumulate(np.maximum(d[qc].to_numpy(float),0),axis=1)      # monotone, non-negative quantiles
d['window']=np.where(d.key<=202418,'DEVELOPMENT 2024 wk6-18','CONFIRMATION 2025 wk6-2026 wk11')
d['game']=d.apply(lambda r:r.key*1000+hash(tuple(sorted((r.team,r.opp))))%1000,axis=1)
Q=d[qc].to_numpy(float)
def p_ge(q,T):                                    # P(Y >= T) from the 11 predicted quantiles (piecewise-linear CDF through them)
    xs=np.concatenate([[0.0],q+np.arange(len(q))*1e-6,[q[-1]*1.6+10]]); ps=np.concatenate([[0.0],P,[1.0]])
    return 1.0-np.interp(T,xs,ps)
# ---------------- numerical accuracy (separate from probabilities)
print('NUMERICAL ACCURACY (median projection, average miss in yards) and range coverage')
for mk in ('receiving_yards','rushing_yards'):
    for w in sorted(d.window.unique()):
        s=d[(d.market==mk)&(d.window==w)]; y=s.y.to_numpy(); med=s.q50.to_numpy()
        pin=np.mean([np.mean(np.maximum(p*(y-s['q%02d'%q]),(p-1)*(y-s['q%02d'%q]))) for q,p in zip(QS,P)])
        cov80=np.mean((y>=s.q10)&(y<=s.q90)); cov90=np.mean((y>=s.q05)&(y<=s.q95))
        print('  %-16s %-34s n=%4d  median miss %5.2f | mean pinball %.3f | 80%% range holds %.0f%% | 90%% range holds %.0f%%'%(mk,w,len(s),np.mean(np.abs(y-med)),pin,100*cov80,100*cov90))
# ---------------- threshold probabilities: candidate thresholds on a 5-yard grid
GRID=np.arange(10,155,5)
rec=[]
for i in range(len(d)):
    q=Q[i]; 
    for T in GRID:
        p=p_ge(q,T)
        if 0.03<=p<=0.97: rec.append((i,T,p,float(d.y.iat[i]>=T)))
c=pd.DataFrame(rec,columns=['row','T','p_raw','hit']); c=c.join(d[['market','window','key','game','pid','q50']],on='row')
# walk-forward isotonic calibration, per market: fit on all strictly earlier weeks, apply to this week
c['p_cal']=np.nan
for mk in ('receiving_yards','rushing_yards'):
    s=c[c.market==mk]
    for k in sorted(s.key.unique()):
        tr=s[s.key<k]; te=s.key==k
        if len(tr)<1500: c.loc[te[te].index,'p_cal']=np.nan; continue
        iso=IsotonicRegression(out_of_bounds='clip',y_min=0.01,y_max=0.99).fit(tr.p_raw,tr.hit)
        idx=s.index[s.key==k]; c.loc[idx,'p_cal']=iso.predict(s.loc[idx,'p_raw'])
cc=c.dropna(subset=['p_cal']).copy()
def ece(p,h,bins=10):
    b=np.minimum((p*bins).astype(int),bins-1); return float(sum(abs(p[b==i].mean()-h[b==i].mean())*np.mean(b==i) for i in range(bins) if (b==i).any()))
print('\nTHRESHOLD PROBABILITIES (all candidate thresholds; chance Y >= T), before and after calibration fitted only on EARLIER weeks')
for mk in ('receiving_yards','rushing_yards'):
    for w in sorted(cc.window.unique()):
        s=cc[(cc.market==mk)&(cc.window==w)]; h=s.hit.to_numpy()
        print('  %-16s %-34s n=%6d | Brier raw %.4f -> calibrated %.4f (always-average %.4f) | calibration error raw %.3f -> %.3f'%(mk,w,len(s),np.mean((s.p_raw-h)**2),np.mean((s.p_cal-h)**2),np.mean((h.mean()-h)**2),ece(s.p_raw.to_numpy(),h),ece(s.p_cal.to_numpy(),h)))
print('\nRELIABILITY (calibrated): stated chance bucket -> what really happened (all thresholds, both windows)')
bk=np.minimum((cc.p_cal*10).astype(int),9)
for i in range(10):
    m=bk==i
    if m.sum()>200: print('   stated %2d-%2d%%: n=%6d  stated avg %.3f  actual %.3f'%(10*i,10*i+10,m.sum(),cc.p_cal[m].mean(),cc.hit[m].mean()))
# ---------------- top-four selector per game
MIN_T={'receiving_yards':20,'rushing_yards':25}
def select(cdf,rankcol,probcol,thresh_p=0.70):
    rows=[]
    for (g,pid,mk),x in cdf.groupby(['game','pid','market']):
        x=x[(x['T']>=MIN_T[mk])&(x['T']>=0.55*x.q50)&(x[probcol]>=thresh_p)]
        if len(x): r=x.loc[x['T'].idxmax()]; rows.append(r)
    if not rows: return pd.DataFrame()
    r=pd.DataFrame(rows); r['rank']=r.groupby('game')[rankcol].rank(ascending=False,method='first'); return r[r['rank']<=4]
print('\nTOP-FOUR SELECTOR per game: for each player the highest 5-yard threshold whose chance is >= 70%; best four per game by the ranking shown')
for w in sorted(cc.window.unique()):
    s=cc[cc.window==w]; rng=np.random.RandomState(5)
    res={}
    res['A. ranked by CALIBRATED chance (the design)']=select(s,'p_cal','p_cal')
    res['B. ranked by raw uncalibrated chance']=select(s,'p_raw','p_raw')
    t=select(s,'p_cal','p_cal'); allc=s.copy(); allc['rnd']=rng.random(len(allc)); res['C. random four of the same eligible players']=select(allc,'rnd','p_cal')
    allc['sz']=allc.q50; res['D. ranked by biggest projection']=select(allc,'sz','p_cal')
    print('  '+w)
    for k,r in res.items():
        if len(r): print('     %-48s picks %4d | stated chance %.3f | actually hit %.3f | average threshold %.0f yds'%(k,len(r),r.p_cal.mean(),r.hit.mean(),r['T'].mean()))
c.to_pickle(OUT+'threshold_probs.pkl')
