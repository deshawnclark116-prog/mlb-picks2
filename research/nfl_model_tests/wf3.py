import sys, json
sys.path.insert(0,'/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl'); sys.path.insert(0,'/home/user/mlb-picks2')
import numpy as np, xgboost as xgb, exp2 as exp
import nfl_yardage_v2 as v2
OWN=["own_status","own_practice"]; TM=["tm_out_vol","tm_quest_vol"]; SN=["snap3","snap1"]
BASE=v2.FEATURES+OWN+TM+SN
VARS={"v3_all":BASE,"plus_weather":BASE+["wind","temp","dome"],"plus_rest":BASE+["rest_days"],"plus_pos_defense":BASE+["opp_pos_ypt","opp_pos_tshare","opp_ypc"],
 "plus_all_three":BASE+["wind","temp","dome","rest_days","opp_pos_ypt","opp_pos_tshare","opp_ypc"]}
P=dict(v2.PARAMS); P["nthread"]=4
def mat(rows,cols): return np.array([[r[1][c] if r[1][c] is not None else np.nan for c in cols] for r in rows],np.float32)
res={}
p_dir='/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl'
for mkt in ('rushing_yards','receiving_yards'):
    rows=exp.build(mkt)
    keys=sorted({(r[0]['season'],r[0]['week']) for r in rows})
    test=[k for k in keys if (k[0]==2025 and k[1]>=6) or k[0]==2026]
    out={n:[] for n in VARS}; act=[]; naive=[]; meta=[]
    for k in test:
        tr=[r for r in rows if (r[0]['season'],r[0]['week'])<k]; te=[r for r in rows if (r[0]['season'],r[0]['week'])==k]
        if len(tr)<300 or not te: continue
        y=[r[2] for r in tr]
        for n,cols in VARS.items():
            b=xgb.train(P,xgb.DMatrix(mat(tr,cols),label=y,feature_names=cols),num_boost_round=140)
            out[n]+=list(b.predict(xgb.DMatrix(mat(te,cols),feature_names=cols)))
        act+=[r[2] for r in te]; naive+=[v2.naive_preds(r[1])[1] for r in te]; meta+=[k for _ in te]
    act=np.array(act); naive=np.array(naive); meta=np.array(meta)
    res[mkt]={'n':len(act)}
    def rep(name,pred):
        e=np.abs(pred-act); return float(e.mean())
    r={'naive_average':rep('n',naive)}
    for n in VARS: r[n]=rep(n,np.array(out[n]))
    # paired bootstrap v3 vs v2
    rng=np.random.RandomState(1)
    for n in ('plus_weather','plus_rest','plus_pos_defense','plus_all_three'):
        d=np.abs(np.array(out['v3_all'])-act)-np.abs(np.array(out[n])-act)
        bs=[d[rng.randint(0,len(d),len(d))].mean() for _ in range(2000)]
        r['P_better_than_v2_'+n]=float(np.mean(np.array(bs)>0))
    for n in ('v3_all','plus_all_three'):
        e=np.abs(np.array(out[n])-act); r['within10_'+n]=float((e<=10).mean()); r['bias_'+n]=float(np.median(np.array(out[n])-act))
    # 2026 only
    m26=meta[:,0]==2026
    r['mae_2026_v2']=float(np.abs(np.array(out['v3_all'])-act)[m26].mean()); r['mae_2026_v3_all']=float(np.abs(np.array(out['plus_all_three'])-act)[m26].mean()); r['mae_2026_naive']=float(np.abs(naive-act)[m26].mean()); r['n_2026']=int(m26.sum())
    # rows where an absence exists
    vac=np.array([r_[1]['vacated_vol'] for r_ in rows if (r_[0]['season'],r_[0]['week']) in set(map(tuple,meta))])
    res[mkt]|=r
    np.save(p_dir+'/preds3_%s.npy'%mkt,np.vstack([act,np.array(out['v3_all']),np.array(out['plus_all_three']),meta[:,0],meta[:,1]]))
print(json.dumps(res,indent=1))
json.dump(res,open('/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/wf_result.json','w'),indent=1)
