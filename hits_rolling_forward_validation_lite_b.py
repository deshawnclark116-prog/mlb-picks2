#!/usr/bin/env python3
import argparse,csv,glob,json,math,os,re,sqlite3,subprocess,sys
from collections import deque
from pathlib import Path

FEATURES=["season_avg","recent15_avg","recent5_avg","hr_rate","bb_rate","so_rate","batting_order","games_played"]
OFFICIAL_MIN=0.630
WATCHLIST_MIN=0.606
WORK_DIR=Path("/data/hr_model/hits_rfv_lite_b")
OUT_JSON=Path("/data/hr_model/hits_rolling_forward_validation_lite_b_results.json")
TRAIN_PARAMS={"objective":"count:poisson","learning_rate":0.05,"max_depth":6,"subsample":0.8,"colsample_bytree":0.8,"min_child_weight":5,"tree_method":"hist","nthread":2}
ALIASES={
"player_id":["player_id","batter_id","person_id","id"],"date":["date","game_date"],
"hits":["hits","h"],"ab":["at_bats","atBats","ab"],"pa":["plate_appearances","plateAppearances","pa"],
"hr":["home_runs","homeRuns","hr"],"bb":["walks","base_on_balls","baseOnBalls","bb"],
"so":["strikeouts","strike_outs","strikeOuts","so"],"batting_order":["batting_order","battingOrder","lineup_spot","order"],
"player_type":["player_type","type"],"game_id":["game_id","game_pk","gamePk"]}

def first_value(row,names,default=None):
    for n in names:
        if n in row and row[n] is not None:return row[n]
    for c in ("stats","hitting","batting"):
        d=row.get(c)
        if isinstance(d,dict):
            for n in names:
                if n in d and d[n] is not None:return d[n]
    return default

def to_float(v,d=0.0):
    try:return float(d if v in (None,"") else v)
    except:return float(d)

def year_from_path(p):
    m=re.search(r"season_(20\d{2})\.(?:jsonl|json)$",Path(p).name)
    return int(m.group(1)) if m else None

def discover():
    out={}
    for pat in ("/data/season_*.jsonl","/data/season_*.json"):
        for p in glob.glob(pat):
            y=year_from_path(p)
            if y is not None and (y not in out or p.endswith(".jsonl")):out[y]=p
    return dict(sorted(out.items()))

def iter_json_records(path):
    with open(path,"r",encoding="utf-8",errors="replace") as f:
        first=""
        pos=f.tell()
        for line in f:
            if line.strip():
                first=line.lstrip()[0];break
        f.seek(pos)
        if first=="[":
            raise RuntimeError(f"{path} is a giant JSON array; this lite script refuses to load it under 512 MB. Convert to JSONL first.")
        for i,line in enumerate(f,1):
            line=line.strip()
            if not line:continue
            yield json.loads(line)

def normalize(row,seq):
    ptype=str(first_value(row,ALIASES["player_type"],"") or "").lower()
    if ptype and "batter" not in ptype and "hitter" not in ptype:return None
    pid=first_value(row,ALIASES["player_id"]); date=first_value(row,ALIASES["date"])
    hits=first_value(row,ALIASES["hits"]); ab=first_value(row,ALIASES["ab"])
    if pid is None or date is None or hits is None or ab is None:return None
    bb=to_float(first_value(row,ALIASES["bb"],0))
    pa_raw=first_value(row,ALIASES["pa"],None); pa=to_float(pa_raw,to_float(ab)+bb)
    return (str(pid),str(date),str(first_value(row,ALIASES["game_id"],seq)),seq,to_float(hits),to_float(ab),pa,to_float(first_value(row,ALIASES["hr"],0)),bb,to_float(first_value(row,ALIASES["so"],0)),to_float(first_value(row,ALIASES["batting_order"],0)))

def build_feature_files(path,year):
    WORK_DIR.mkdir(parents=True,exist_ok=True)
    db=WORK_DIR/f"spool_{year}.sqlite"; lib=WORK_DIR/f"features_{year}.libsvm"; meta=WORK_DIR/f"features_{year}_meta.csv"; summ=WORK_DIR/f"features_{year}_summary.json"
    for p in (db,lib,meta,summ):
        if p.exists():p.unlink()
    conn=sqlite3.connect(str(db)); conn.execute("PRAGMA journal_mode=OFF"); conn.execute("PRAGMA synchronous=OFF"); conn.execute("PRAGMA temp_store=FILE")
    conn.execute("CREATE TABLE games(player_id TEXT,game_date TEXT,game_id TEXT,seq INTEGER,hits REAL,ab REAL,pa REAL,hr REAL,bb REAL,so REAL,batting_order REAL)")
    batch=[]; seen=accepted=0
    for seq,row in enumerate(iter_json_records(path),1):
        seen+=1; n=normalize(row,seq)
        if n is None:continue
        batch.append(n); accepted+=1
        if len(batch)>=2000:
            conn.executemany("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?)",batch); batch.clear()
    if batch:conn.executemany("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?)",batch)
    conn.commit(); conn.execute("CREATE INDEX idx_games ON games(player_id,game_date,game_id,seq)"); conn.commit()
    cur=conn.execute("SELECT player_id,game_date,game_id,seq,hits,ab,pa,hr,bb,so,batting_order FROM games ORDER BY player_id,game_date,game_id,seq")
    eligible=players=0; current=None
    cum_hits=cum_ab=cum_pa=cum_hr=cum_bb=cum_so=0.0; games=0; recent=deque(maxlen=15)
    with open(lib,"w",encoding="utf-8") as fout,open(meta,"w",encoding="utf-8",newline="") as mf:
        w=csv.writer(mf); w.writerow(["game_date","actual_hits"])
        for r in cur:
            pid,date,gid,seq,hits,ab,pa,hr,bb,so,order=r
            if pid!=current:
                current=pid; players+=1; cum_hits=cum_ab=cum_pa=cum_hr=cum_bb=cum_so=0.0; games=0; recent=deque(maxlen=15)
            if cum_ab>=20 and games>=5:
                last5=list(recent)[-5:]
                feats=[cum_hits/cum_ab if cum_ab else 0.0,sum(recent)/len(recent) if recent else 0.0,sum(last5)/len(last5) if last5 else 0.0,cum_hr/cum_pa if cum_pa else 0.0,cum_bb/cum_pa if cum_pa else 0.0,cum_so/cum_pa if cum_pa else 0.0,float(order),float(games)]
                fout.write(f"{float(hits):.8g} "+" ".join(f"{i}:{v:.10g}" for i,v in enumerate(feats,1))+"\n")
                w.writerow([date,f"{float(hits):.8g}"]); eligible+=1
            cum_hits+=hits; cum_ab+=ab; cum_pa+=pa; cum_hr+=hr; cum_bb+=bb; cum_so+=so; games+=1; recent.append(hits)
    conn.close()
    try:db.unlink()
    except:pass
    out={"year":year,"source_path":path,"raw_json_rows_seen":seen,"normalized_batter_rows":accepted,"distinct_players_processed":players,"eligible_training_rows":eligible}
    summ.write_text(json.dumps(out,indent=2),encoding="utf-8")
    return out

def run_child(args,label):
    env=dict(os.environ); env.update({"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
    print(f"\n{label}\n"+"-"*len(label),flush=True)
    cp=subprocess.run([sys.executable,"-u",str(Path(__file__).resolve())]+args,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(cp.stdout or "",end="" if (cp.stdout or "").endswith("\n") else "\n",flush=True)
    if cp.returncode:raise RuntimeError(f"{label} failed with exit code {cp.returncode}")

def worker_train(a):
    import xgboost as xgb
    d=xgb.DMatrix(a.train_file); prev=None; prior=0
    if a.prev_model and Path(a.prev_model).exists():
        prev=xgb.Booster(); prev.load_model(a.prev_model); prior=int(prev.num_boosted_rounds())
    b=xgb.train(TRAIN_PARAMS,d,num_boost_round=60,xgb_model=prev); b.save_model(a.out_model)
    print(f"prior_rounds={prior}\nadded_rounds=60\ntotal_rounds={b.num_boosted_rounds()}\nsaved_model={a.out_model}",flush=True)

def worker_score(a):
    import xgboost as xgb
    b=xgb.Booster(); b.load_model(a.model_path); d=xgb.DMatrix(a.score_file); pred=b.predict(d)
    dates=[]; hits=[]
    with open(a.meta_file,encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):dates.append(r["game_date"]); hits.append(float(r["actual_hits"]))
    probs=[min(max(1.0-math.exp(-max(float(x),1e-8)),1e-8),1-1e-8) for x in pred]
    y=[1 if h>=1 else 0 for h in hits]
    Path(a.out_score_json).write_text(json.dumps({"model_rounds":int(b.num_boosted_rounds()),"dates":dates,"y_binary":y,"probs":probs}),encoding="utf-8")
    print(f"scored_rows={len(y):,}\nmodel_rounds={b.num_boosted_rounds()}\nsaved_scores={a.out_score_json}",flush=True)

def auc_rank(y,p):
    pos=sum(y); neg=len(y)-pos
    if not pos or not neg:return float("nan")
    pairs=sorted(enumerate(p),key=lambda x:x[1]); ranks=[0.0]*len(y); i=0
    while i<len(pairs):
        j=i+1
        while j<len(pairs) and pairs[j][1]==pairs[i][1]:j+=1
        r=((i+1)+j)/2
        for k in range(i,j):ranks[pairs[k][0]]=r
        i=j
    s=sum(ranks[i] for i,v in enumerate(y) if v==1)
    return (s-pos*(pos+1)/2)/(pos*neg)

def metrics(y,p):
    n=len(y); order=sorted(range(n),key=lambda i:p[i],reverse=True)
    out={"rows":n,"actual_hit_rate":sum(y)/n,"mean_probability":sum(p)/n,"brier":sum((pi-yi)**2 for yi,pi in zip(y,p))/n,"logloss":sum(-(yi*math.log(min(max(pi,1e-8),1-1e-8))+(1-yi)*math.log(1-min(max(pi,1e-8),1-1e-8))) for yi,pi in zip(y,p))/n,"auc":auc_rank(y,p)}
    for frac,name in ((.05,"top5"),(.10,"top10")):
        k=max(1,math.ceil(n*frac)); idx=order[:k]
        out[f"{name}_rows"]=k; out[f"{name}_hits"]=sum(y[i] for i in idx); out[f"{name}_actual"]=sum(y[i] for i in idx)/k; out[f"{name}_mean_prob"]=sum(p[i] for i in idx)/k
    return out

def thresholds(y,p):
    def one(idx):
        return {"rows":len(idx),"hits":sum(y[i] for i in idx),"actual_hit_rate":(sum(y[i] for i in idx)/len(idx) if idx else None),"mean_probability":(sum(p[i] for i in idx)/len(idx) if idx else None)}
    return {"official_ge_0.630":one([i for i,x in enumerate(p) if x>=OFFICIAL_MIN]),"watchlist_0.606_to_0.630":one([i for i,x in enumerate(p) if WATCHLIST_MIN<=x<OFFICIAL_MIN]),"official_plus_watchlist_ge_0.606":one([i for i,x in enumerate(p) if x>=WATCHLIST_MIN])}

def line(label,m):
    print(f"{label}: n={m['rows']:,} actual={m['actual_hit_rate']:.4f} pred={m['mean_probability']:.4f} brier={m['brier']:.8f} logloss={m['logloss']:.8f} auc={m['auc']:.6f} top5={m['top5_actual']:.4f} top10={m['top10_actual']:.4f}",flush=True)

def parent():
    WORK_DIR.mkdir(parents=True,exist_ok=True); seasons=discover()
    print("HITS_ROLLING_FORWARD_VALIDATION_LITE_B\n======================================")
    print("\nPREFLIGHT\n---------")
    if not seasons:
        print("No /data/season_*.jsonl or /data/season_*.json files found.\nHistorical files must be restored before validation can run.");return 2
    for y,p in seasons.items():print(f"{y}: {p} | {Path(p).stat().st_size:,} bytes")
    missing=[y for y in range(2019,2026) if y not in seasons]
    if missing:print(f"Missing required years: {missing}");return 2
    print("2026 intentionally excluded from clean historical folds.")
    print("\nFEATURE BUILD SUMMARY\n---------------------")
    summaries={}
    for y in range(2019,2026):
        lib=WORK_DIR/f"features_{y}.libsvm"; meta=WORK_DIR/f"features_{y}_meta.csv"; summ=WORK_DIR/f"features_{y}_summary.json"
        if lib.exists() and meta.exists() and summ.exists():
            s=json.loads(summ.read_text()); print(f"{y}: REUSED | eligible_rows={s['eligible_training_rows']:,}")
        else:
            print(f"{y}: building streamingly...",flush=True); s=build_feature_files(seasons[y],y); print(f"{y}: raw={s['raw_json_rows_seen']:,} normalized={s['normalized_batter_rows']:,} eligible={s['eligible_training_rows']:,}")
        summaries[y]=s
    current=None; folds=[]; all_y=[]; all_p=[]
    for train_year in range(2019,2025):
        model=WORK_DIR/f"booster_through_{train_year}.json"; args=["--worker-train","--train-file",str(WORK_DIR/f"features_{train_year}.libsvm"),"--out-model",str(model)]
        if current:args+=["--prev-model",str(current)]
        run_child(args,f"TRAIN THROUGH {train_year}"); current=model
        test=train_year+1
        if 2022<=test<=2025:
            score=WORK_DIR/f"scores_{test}.json"
            run_child(["--worker-score","--model-path",str(current),"--score-file",str(WORK_DIR/f"features_{test}.libsvm"),"--meta-file",str(WORK_DIR/f"features_{test}_meta.csv"),"--out-score-json",str(score)],f"SCORE FORWARD HOLDOUT {test}")
            z=json.loads(score.read_text()); y=z["y_binary"]; p=z["probs"]; m=metrics(y,p); t=thresholds(y,p); folds.append({"train_start":2019,"train_end":train_year,"test_year":test,"model_rounds":z["model_rounds"],"metrics":m,"thresholds":t}); all_y+=y; all_p+=p
    agg=metrics(all_y,all_p); agg_t=thresholds(all_y,all_p)
    print("\nROLLING FORWARD RESULTS\n-----------------------")
    for f in folds:line(f"TRAIN 2019-{f['train_end']} -> TEST {f['test_year']}",f["metrics"])
    print("\nAGGREGATE BASELINE\n------------------");line("ALL FORWARD FOLDS",agg)
    print("\nPRODUCTION THRESHOLD READ\n-------------------------")
    for name,b in agg_t.items():
        if not b["rows"]:print(f"{name}: n=0")
        else:print(f"{name}: n={b['rows']:,} hits={b['hits']:,} actual_hit_rate={b['actual_hit_rate']:.4f} mean_probability={b['mean_probability']:.4f}")
    print("\nBASELINE STATUS\n---------------\nincumbent_reconstruction: COMPLETE\nrolling_forward_baseline: COMPLETE\n2026_clean_holdout: NO — burned by original incumbent training\nchallenger_features_authorized: NOT YET — review and lock hits-specific promotion gate first.")
    OUT_JSON.write_text(json.dumps({"script":"HITS_ROLLING_FORWARD_VALIDATION_LITE_B","market":"batter_hits_over_0.5","incumbent":{"features":FEATURES,"params":TRAIN_PARAMS,"rounds_per_season_file":60,"official_min_prob":OFFICIAL_MIN,"watchlist_min_prob":WATCHLIST_MIN,"2026_holdout_status":"burned"},"feature_summaries":summaries,"fold_results":folds,"aggregate":agg,"aggregate_thresholds":agg_t},indent=2),encoding="utf-8")
    print(f"\nresults_json={OUT_JSON}");return 0

def parser():
    a=argparse.ArgumentParser(); a.add_argument("--worker-train",action="store_true"); a.add_argument("--worker-score",action="store_true")
    for x in ("train-file","prev-model","out-model","model-path","score-file","meta-file","out-score-json"):a.add_argument("--"+x)
    return a

def main():
    a=parser().parse_args()
    if a.worker_train:worker_train(a);return 0
    if a.worker_score:worker_score(a);return 0
    return parent()

if __name__=="__main__":raise SystemExit(main())
