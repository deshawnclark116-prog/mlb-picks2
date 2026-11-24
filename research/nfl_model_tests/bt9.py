import sys, csv, json, sqlite3, time, multiprocessing as mp
from collections import defaultdict
sys.path.insert(0,'/home/user/mlb-picks2'); sys.path.insert(0,'/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl')
import numpy as np
import nfl_v8_sim as sim, nfl_yardage_v2 as v2, exp
D=exp.D; OUT='/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/'
S=int(sys.argv[1]) if len(sys.argv)>1 else 800
con=sqlite3.connect('file:/home/user/mlb-picks2/nfl_models/nfl_model.sqlite?mode=ro',uri=True)
pos_of={}
for pid,pos in con.execute("select player_id,position from player_games order by season,week"): pos_of[pid]=pos
plays=sim.Plays([D+'pbp_%d.csv.gz'%y for y in (2023,2024,2025,2026)],pos_of)
STAT={'Out':'OUT','Doubtful':'DOUBTFUL','Questionable':'QUESTIONABLE'}
status={}; played=set(); inj_rows=[]
for y in (2023,2024,2025,2026):
    for r in csv.DictReader(open(D+'snap_counts_%d.csv'%y)):
        if r['game_type']=='REG': played.add((v2.norm_name(r['player']),int(r['season']),int(r['week'])))
    for r in csv.DictReader(open(D+'injuries_%d.csv'%y)):
        if r['game_type']!='REG' or not r['report_status']: continue
        st=STAT.get(r['report_status'])
        if st: status[(r['gsis_id'],int(r['season']),int(r['week']))]=st; inj_rows.append((int(r['season'])*100+int(r['week']),st,(v2.norm_name(r['full_name']),int(r['season']),int(r['week']))))
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
DATA={}
for mkt in MK:
    rows=exp.build(mkt)
    keys=sorted({(r[0]['season'],r[0]['week']) for r in rows}); test=set(k for k in keys if (k[0]==2025 and k[1]>=6) or k[0]==2026)
    sel=[r for r in rows if (r[0]['season'],r[0]['week']) in test]
    pr=np.load(OUT+'preds2_%s.npy'%mkt); assert len(sel)==pr.shape[1]
    DATA[mkt]=(sel,pr)
weeks=sorted({(r[0]['season'],r[0]['week']) for m in DATA for r in DATA[m][0]})
allp=set(plays.carry.keys())|set(plays.targets.keys())

def crps(samples,a):
    s=np.sort(samples); n=len(s); c=2.0*np.arange(1,n+1)-n-1.0
    return float(np.mean(np.abs(s-a))-np.sum(c*s)/(n*n))

def run(cfg):
    for k,v in cfg.items():
        if k!='name': setattr(sim,k,v)
    summ={}; cache={}; lcache={}
    for (season,week) in weeks:
        key=sim.keyof(season,week); pt=p_table(key)
        sts={p:status.get((p,season,week),'UNLISTED') for p in allp}
        for home,away,spread,total in games.get((season,week),[]):
            for team,opp,sp in ((home,away,spread),(away,home,-spread if spread is not None else None)):
                ti=sim.build_team_input(plays,team,opp,season,week,sp or 0.0,total,sts,pt,pos_of,lcache)
                if ti is None: continue
                summ[(season,week,team)]=sim.simulate_team(plays,ti,S,hash((season,week,team))%(2**31),cache,pos_of)
    res={'name':cfg['name']}
    for mkt,(sel,pr) in DATA.items():
        f=MK[mkt]; act=pr[0]; p2=pr[1]
        ratios=act/np.maximum(p2,1.0); bands=[(0,20),(20,35),(35,50),(50,70),(70,1e9)]
        bidx=lambda p: next(i for i,(lo,hi) in enumerate(bands) if lo<=p<hi)
        bi=np.array([bidx(p) for p in p2]); bratios=[ratios[bi==i] for i in range(len(bands))]
        cs=[];cb=[];md=[];ps=[];pb=[];hit=[];cov=[];ab=[]
        for i,(m,fe,y) in enumerate(sel):
            r=summ.get((m['season'],m['week'],m['team']),{}).get(m['pid'])
            if r is None or f not in r: continue
            x=r[f][r['plays']] if r['plays'].any() else r[f]
            cs.append(crps(x,y)); b=bratios[bi[i]]; cb.append(crps(p2[i]*b,y))
            md.append(abs(np.median(x)-y)); ab.append(abs(p2[i]-y))
            t=0.7*p2[i]; ps.append(float((x>=t).mean())); pb.append(float((b>=0.7).mean())); hit.append(float(y>=t))
            cov.append(float(np.percentile(x,10)<=y<=np.percentile(x,90)))
        ps,pb,hit=map(np.array,(ps,pb,hit))
        q=np.argsort(ps); n=len(q); top=q[int(.8*n):]; bot=q[:int(.2*n)]
        res[mkt]={'n':len(cs),'CRPS_sim':float(np.mean(cs)),'CRPS_base':float(np.mean(cb)),'MAE_median_sim':float(np.mean(md)),'MAE_v2':float(np.mean(ab)),
          'cover80':float(np.mean(cov)),'brier_sim_0.7':float(((ps-hit)**2).mean()),'brier_base_0.7':float(((pb-hit)**2).mean()),
          'hit_top20':float(hit[top].mean()),'stated_top20':float(ps[top].mean()),'hit_bottom20':float(hit[bot].mean()),'stated_bottom20':float(ps[bot].mean())}
    print(json.dumps(res),flush=True)
    return res

if __name__=='__main__':
    cfgs=json.loads(sys.argv[2]) if len(sys.argv)>2 else [{'name':'base'}]
    base={'SHARE_CONC':14.0,'OWN_K_CARRY':40.0,'OWN_K_TARGET':60.0,'PLAYER_POOL_MAX':250,'LUCK_RUSH_SD':0.0,'LUCK_PASS_SD':0.0,'SLOPE_SCALE':1.0,'PLAYS_COEF':0.40,'PLAYS_NOISE':4.0,'MARGIN_SD':13.2,'TOTAL_SD':9.5}
    cfgs=[dict(base,**c) for c in cfgs]
    with mp.get_context('fork').Pool(4) as pool: out=pool.map(run,cfgs,chunksize=1)
    json.dump(out,open(OUT+(sys.argv[3] if len(sys.argv)>3 else 'bt9_out.json'),'w'),indent=1)
