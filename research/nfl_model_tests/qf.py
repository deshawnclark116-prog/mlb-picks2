import sys, json
sys.path.insert(0,'/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl'); sys.path.insert(0,'/home/user/mlb-picks2')
import numpy as np, xgboost as xgb, exp4 as exp
import nfl_yardage_v2 as v2
OWN=["own_status","own_practice"]; TM=["tm_out_vol","tm_quest_vol"]; SN=["snap3","snap1"]
BASE=v2.FEATURES+OWN+TM+SN
P=dict(v2.PARAMS); P['nthread']=4
def mat(rows,cols): return np.array([[r[1][c] if r[1][c] is not None else np.nan for c in cols] for r in rows],np.float32)
Q=0.25   # floor = the number reached 75% of the time
res={}
for mkt in ('rushing_yards','receiving_yards'):
    rows=exp.build(mkt)
    keys=sorted({(r[0]['season'],r[0]['week']) for r in rows})
    test=[k for k in keys if (k[0]==2025 and k[1]>=6) or k[0]==2026]
    act=[];pm=[];pq=[];meta=[]
    for k in test:
        tr=[r for r in rows if r[0]['season']>=2016 and (r[0]['season'],r[0]['week'])<k]; te=[r for r in rows if (r[0]['season'],r[0]['week'])==k]
        if len(tr)<300 or not te: continue
        y=[r[2] for r in tr]
        b=xgb.train(P,xgb.DMatrix(mat(tr,BASE),label=y,feature_names=BASE),num_boost_round=140)
        pm+=list(b.predict(xgb.DMatrix(mat(te,BASE),feature_names=BASE)))
        Pq=dict(P); Pq.update(objective='reg:quantileerror',quantile_alpha=Q)
        bq=xgb.train(Pq,xgb.DMatrix(mat(tr,BASE),label=y,feature_names=BASE),num_boost_round=140)
        pq+=list(bq.predict(xgb.DMatrix(mat(te,BASE),feature_names=BASE)))
        act+=[r[2] for r in te]
    act,pm,pq=map(np.array,(act,pm,pq)); pq=np.maximum(pq,0)
    # baseline: band-constant ratio floor (fitted on all test rows -> favours the baseline)
    bands=[(0,20),(20,35),(35,50),(50,70),(70,1e9)]; base=np.zeros(len(act))
    for lo,hi in bands:
        i=(pm>=lo)&(pm<hi)
        if i.sum(): base[i]=np.floor(np.quantile(act[i]/np.maximum(pm[i],1),Q)*pm[i])
    pin=lambda f:float(np.mean(np.maximum(Q*(act-f),(Q-1)*(act-f))))
    ratio_q=pq/np.maximum(pm,1); ratio_b=base/np.maximum(pm,1)
    hit_q=(act>=np.floor(pq)); hit_b=(act>=base)
    # discrimination: among rows with the model's floor/projection ratio highest vs lowest quintile, does the hit rate of a FIXED rule differ?
    order=np.argsort(ratio_q); n=len(order); lo5=order[:n//5]; hi5=order[-n//5:]
    fixed=np.floor(0.6*pm)  # same fixed 60%-of-projection floor for everyone
    res[mkt]={'n':len(act),'pinball_individual':pin(np.floor(pq)),'pinball_band':pin(base),
      'hit_individual':float(hit_q.mean()),'hit_band':float(hit_b.mean()),
      'ratio_sd_individual':float(ratio_q.std()),'ratio_sd_band':float(ratio_b.std()),
      'fixed60_hit_rate_in_steadiest_fifth':float((act[hi5]>=fixed[hi5]).mean()),'fixed60_hit_rate_in_boomorbust_fifth':float((act[lo5]>=fixed[lo5]).mean()),
      'avg_floor_ratio_steadiest':float(ratio_q[hi5].mean()),'avg_floor_ratio_boomorbust':float(ratio_q[lo5].mean())}
    print(mkt,json.dumps(res[mkt]),flush=True)
json.dump(res,open('qf_result.json','w'),indent=1)
