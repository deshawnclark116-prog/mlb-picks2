import sys, csv, json, sqlite3, time
from collections import defaultdict
sys.path.insert(0,'/home/user/mlb-picks2'); sys.path.insert(0,'/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl')
import numpy as np
import nfl_v8_sim as sim, nfl_yardage_v2 as v2, exp
D=exp.D
S=int(sys.argv[1]) if len(sys.argv)>1 else 1500
con=sqlite3.connect('file:/home/user/mlb-picks2/nfl_models/nfl_model.sqlite?mode=ro',uri=True)
pos_of={}
for pid,pos in con.execute("select player_id,position from player_games order by season,week"): pos_of[pid]=pos
t0=time.time()
plays=sim.Plays([D+'pbp_%d.csv.gz'%y for y in (2023,2024,2025,2026)],pos_of); print('plays loaded',time.time()-t0)
# injuries status + played
STAT={'Out':'OUT','Doubtful':'DOUBTFUL','Questionable':'QUESTIONABLE'}
status={}; played=set(); inj_rows=[]
for y in (2023,2024,2025,2026):
    for r in csv.DictReader(open(D+'snap_counts_%d.csv'%y)):
        if r['game_type']=='REG': played.add((v2.norm_name(r['player']),int(r['season']),int(r['week'])))
    for r in csv.DictReader(open(D+'injuries_%d.csv'%y)):
        if r['game_type']!='REG' or not r['report_status']: continue
        k=(r['gsis_id'],int(r['season']),int(r['week'])); st=STAT.get(r['report_status'])
        if st: status[k]=st; inj_rows.append((int(r['season'])*100+int(r['week']),st,(v2.norm_name(r['full_name']),int(r['season']),int(r['week']))))
PRI={'OUT':0.02,'DOUBTFUL':0.12,'QUESTIONABLE':0.60,'UNLISTED':0.965}
def p_table(key):
    c=defaultdict(lambda:[0,0])
    for k,st,nk in inj_rows:
        if k<key: c[st][0]+=nk in played; c[st][1]+=1
    out=dict(PRI)
    for st,(a,n) in c.items(): out[st]=(a+PRI[st]*8)/(n+8)
    return out
games=defaultdict(list)
for gid,season,week,home,away,spread,total in con.execute("select game_id,season,week,home_team,away_team,spread_line,total_line from games where season_type='REG'"):
    games[(season,week)].append((home,away,spread,total))
MK={'rushing_yards':'rush_yards','receiving_yards':'rec_yards'}
result={}
for mkt in MK:
    rows=exp.build(mkt)
    keys=sorted({(r[0]['season'],r[0]['week']) for r in rows})
    test=[k for k in keys if (k[0]==2025 and k[1]>=6) or k[0]==2026]
    sel=[r for r in rows if (r[0]['season'],r[0]['week']) in set(test)]
    pr=np.load(D+'../preds2_%s.npy'%mkt)
    assert len(sel)==pr.shape[1],(len(sel),pr.shape)
    result[mkt]=(sel,pr)
cache={}; league_cache={}; summ={}   # (season,week,team)->{pid:row}
t0=time.time(); ntg=0
for (season,week) in sorted({(r[0]['season'],r[0]['week']) for m in result for r in result[m][0]}):
    key=sim.keyof(season,week); pt=p_table(key); pool_cache={}
    for home,away,spread,total in games.get((season,week),[]):
        for team,opp,sp in ((home,away,spread),(away,home,-spread if spread is not None else None)):
            st={pid:v for (pid,s,w),v in status.items() if s==season and w==week} if False else None
            sts={}
            ti=sim.build_team_input(plays,team,opp,season,week,sp or 0.0,total,{p:status.get((p,season,week),'UNLISTED') for p in plays.carry.keys()|plays.targets.keys()},pt,pos_of,league_cache)
            if ti is None: continue
            res=sim.simulate_team(plays,ti,S,hash((season,week,team))%(2**31),pool_cache,pos_of)
            summ[(season,week,team)]=res; ntg+=1
    print(season,week,'done',round(time.time()-t0),flush=True)
out={}
for mkt,(sel,pr) in result.items():
    f=MK[mkt]; act=pr[0]; p2=pr[1]; p3=pr[2]
    A=[];M=[];Md=[];V2=[];V3=[];W=[];R=[];N=[]
    for i,(m,fe,y) in enumerate(sel):
        r=summ.get((m['season'],m['week'],m['team']),{}).get(m['pid'])
        if r is None or f not in r: continue
        s=sim.summarize(r[f],r['plays'])
        A.append(y);M.append(s['mean']);Md.append(s['median']);V2.append(p2[i]);V3.append(p3[i]);W.append(s['whole']);R.append(s['reach'])
    A,M,Md,V2,V3,W,R=map(np.array,(A,M,Md,V2,V3,W,R))
    hit=(A>=W).astype(float)
    e=lambda p:float(np.abs(p-A).mean())
    out[mkt]={'rows_total':len(sel),'rows_covered':len(A),'MAE_sim_mean':e(M),'MAE_sim_median':e(Md),'MAE_v2_current':e(V2),'MAE_v2_plus_injury_snaps':e(V3),
      'blend_sim_v2_MAE':e(0.5*M+0.5*V2),'blend_sim_median_v2':e(0.5*Md+0.5*V2),
      'bias_sim_mean':float(np.median(M-A)),'within10_sim':float((np.abs(Md-A)<=10).mean()),'within10_v2':float((np.abs(V2-A)<=10).mean()),
      'reach_stated_mean':float(R.mean()),'reach_actual':float(hit.mean()),'brier_reach':float(((R-hit)**2).mean()),'brier_always_half':float(((0.5-hit)**2).mean())}
    # bins of stated reach
    b={}
    for lo,hi in ((0,.4),(.4,.5),(.5,.6),(.6,1.01)):
        i=(R>=lo)&(R<hi)
        if i.sum()>30: b['%.1f-%.1f'%(lo,hi)]=[int(i.sum()),float(R[i].mean()),float(hit[i].mean())]
    out[mkt]['reach_bins']=b
print(json.dumps(out,indent=1))
json.dump(out,open('/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/bt8_result.json','w'),indent=1)
