import sys, sqlite3, numpy as np, xgboost as xgb
sys.path.insert(0,'/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl'); sys.path.insert(0,'/home/user/mlb-picks2')
import exp4 as exp, nfl_yardage_v2 as v2
db=sqlite3.connect('file:/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/nfl_ext.sqlite?mode=ro',uri=True)
BASE=v2.FEATURES+["own_status","own_practice","tm_out_vol","tm_quest_vol","snap3","snap1"]
P=dict(v2.PARAMS); P['nthread']=4
mat=lambda rows:np.array([[r[1][c] if r[1][c] is not None else np.nan for c in BASE] for r in rows],np.float32)
D='/tmp/claude-0/-home-user-mlb-picks2/6555c7a5-a0fd-5c25-9610-d60f79dc8208/scratchpad/nfl/'
for mkt,vc in (('rushing_yards','carries'),('receiving_yards','targets')):
    rows=exp.build(mkt)
    vol={(p,s,w):v for p,s,w,v in db.execute(f"select player_id,season,week,coalesce({vc},0) from player_games")}
    rows=[(m,f,y,vol.get((m['pid'],m['season'],m['week']))) for (m,f,y) in rows]; rows=[r for r in rows if r[3] is not None]
    keys=sorted({(r[0]['season'],r[0]['week']) for r in rows}); test=[k for k in keys if (k[0]==2025 and k[1]>=6) or k[0]==2026]
    V=[];N3=[];NS=[];PRED=[]
    for k in test:
        tr=[r for r in rows if r[0]['season']>=2016 and (r[0]['season'],r[0]['week'])<k]; te=[r for r in rows if (r[0]['season'],r[0]['week'])==k]
        if len(tr)<300 or not te: continue
        b=xgb.train(P,xgb.DMatrix(mat(tr),label=[r[3] for r in tr],feature_names=BASE),num_boost_round=140)
        PRED+=list(b.predict(xgb.DMatrix(mat(te),feature_names=BASE))); V+=[r[3] for r in te]
        N3+=[r[1]['last3_avg_vol'] for r in te]; NS+=[r[1]['cur_avg_vol'] if r[1]['cur_avg_vol'] is not None else r[1]['last3_avg_vol'] for r in te]
    V,N3,NS,PRED=map(lambda a:np.array(a,float),(V,N3,NS,PRED))
    print('\n%s: predicting the number of %s (avg %.1f), n=%d'%(mkt,vc,V.mean(),len(V)))
    print('   last-3-games average:        miss %.2f'%np.abs(N3-V).mean())
    print('   season-to-date average:      miss %.2f'%np.abs(NS-V).mean())
    print('   trained model (all inputs):  miss %.2f   (%.1f%% better than last-3 average)'%(np.abs(PRED-V).mean(),100*(1-np.abs(PRED-V).mean()/np.abs(N3-V).mean())))
