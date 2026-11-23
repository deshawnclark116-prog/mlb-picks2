import sys, sqlite3, json, numpy as np, xgboost as xgb
sys.path.insert(0,'/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl'); sys.path.insert(0,'/home/user/mlb-picks2')
import exp6 as exp, nfl_yardage_v2 as v2
db=sqlite3.connect('file:/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/nfl_ext.sqlite?mode=ro',uri=True)
BASE=v2.FEATURES+["own_status","own_practice","tm_out_vol","tm_quest_vol","snap3","snap1"]
WL=["share5","share8","share_trend","vol_std8","vol_max3","vol_min3","team_plays5","team_pass_rate5","team_car5","game_total","vacated_share","games_missed5","opp_faced_vol5"]
VARS={"A_base_features":(BASE,'reg:squarederror'),"A_poisson":(BASE,'count:poisson'),"B_plus_workload":(BASE+WL,'reg:squarederror'),"B_poisson":(BASE+WL,'count:poisson')}
mat=lambda rows,cols:np.array([[r[1].get(c) if r[1].get(c) is not None else np.nan for c in cols] for r in rows],np.float32)
out={}
for mkt,vc in (('rushing_yards','carries'),('receiving_yards','targets')):
    rows=exp.build(mkt)
    vol={(p,s,w):v for p,s,w,v in db.execute(f"select player_id,season,week,coalesce({vc},0) from player_games")}
    rows=[(m,f,y,vol.get((m['pid'],m['season'],m['week']))) for (m,f,y) in rows]; rows=[r for r in rows if r[3] is not None]
    keys=sorted({(r[0]['season'],r[0]['week']) for r in rows}); test=[k for k in keys if (k[0]==2025 and k[1]>=6) or k[0]==2026]
    V=[];N3=[];PR={n:[] for n in VARS}
    for k in test:
        tr=[r for r in rows if r[0]['season']>=2016 and (r[0]['season'],r[0]['week'])<k]; te=[r for r in rows if (r[0]['season'],r[0]['week'])==k]
        if len(tr)<300 or not te: continue
        for n,(cols,obj) in VARS.items():
            P=dict(v2.PARAMS); P['nthread']=4; P['objective']=obj
            if obj=='count:poisson': P['eval_metric']='poisson-nloglik'
            b=xgb.train(P,xgb.DMatrix(mat(tr,cols),label=[r[3] for r in tr],feature_names=cols),num_boost_round=140)
            PR[n]+=list(b.predict(xgb.DMatrix(mat(te,cols),feature_names=cols)))
        V+=[r[3] for r in te]; N3+=[r[1]['last3_avg_vol'] for r in te]
    V,N3=np.array(V,float),np.array(N3,float); res={'n':len(V),'last3_avg':float(np.abs(N3-V).mean())}
    rng=np.random.RandomState(1); base=np.abs(np.array(PR['A_base_features'])-V)
    for n in VARS:
        e=np.abs(np.array(PR[n])-V); res[n]=float(e.mean())
        if n!='A_base_features':
            d=base-e; res['P_better_than_A_'+n]=float(np.mean([d[rng.randint(0,len(d),len(d))].mean()>0 for _ in range(2000)]))
    out[mkt]=res; print(mkt,json.dumps(res),flush=True)
    np.save('/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/wl_%s.npy'%mkt,np.vstack([V]+[np.array(PR[n]) for n in VARS]))
json.dump(out,open('/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/wl_result.json','w'),indent=1)
