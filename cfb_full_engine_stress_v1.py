"""CFB full-engine research: week-locked features and preweek candidate population.

No production changes. Historical final stats are outcomes only; missing players
are never silently zeroed. Numeric and probability heads are tested separately.
All 2024-26 comparisons are retrospective, not untouched research holdouts.
"""
import argparse,json,sqlite3,math
from collections import defaultdict,deque
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from cfb_numerical_opportunity_shadow import project,MARKETS as OLD_MARKETS

MARKETS={
 "rushing_yards":("carries","rushing_yards",69.5,2,{"RB","QB","FB","WR"}),
 "receiving_yards":("receptions","receiving_yards",49.5,1,{"WR","TE","RB","FB"}),
 "passing_yards":("pass_attempts","passing_yards",224.5,8,{"QB"}),
 "passing_touchdowns":("pass_attempts","passing_touchdowns",1.5,8,{"QB"}),
 "anytime_touchdowns":("touch_opportunities","touchdowns",.5,2,{"RB","QB","WR","TE","FB"})}
def avg(a,default=0.):
 return float(np.mean(a)) if len(a) else default
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))
def values(r,market):
 a,b,*_=MARKETS[market]
 if market=="anytime_touchdowns":
  keys=["carries","receptions","rushing_touchdowns","receiving_touchdowns"]
  if any(r[k] is None for k in keys): return None
  return float(r["carries"]+r["receptions"]),float(r["rushing_touchdowns"]+r["receiving_touchdowns"])
 if r[a] is None or r[b] is None:return None
 return float(r[a]),float(r[b])
def team_features(h):
 v=list(h)
 if not v:return [0,0,0,0,0,0,0]
 return [len(v),avg([r[0] for r in v]),avg([r[1] for r in v]),avg([r[0] for r in v[-3:]]),avg([r[1] for r in v[-3:]]),avg([r[2] for r in v]),avg([r[2] for r in v[-3:]])]
def build(con):
 con.row_factory=sqlite3.Row
 games=[dict(r) for r in con.execute("SELECT * FROM games WHERE home_points IS NOT NULL AND away_points IS NOT NULL ORDER BY season,week,game_id")]
 by_week=defaultdict(list);players=defaultdict(list)
 for g in games:by_week[(g["season"],g["week"])].append(g)
 for r in con.execute("SELECT p.* FROM player_games p JOIN games g ON g.game_id=p.game_id WHERE g.home_points IS NOT NULL AND g.away_points IS NOT NULL"):
  players[str(r["game_id"])].append(dict(r))
 teamhist=defaultdict(lambda:deque(maxlen=14));ratings={};hist=defaultdict(list)
 teamvol=defaultdict(list);oppvol=defaultdict(list);league=defaultdict(lambda:[0.,0.])
 roster=defaultdict(dict);teamweeks=defaultdict(list);out=defaultdict(list);missing=defaultdict(lambda:[0,0]);prev=None
 for (season,week),gs in sorted(by_week.items()):
  if prev!=season:
   ratings={k:.5*v for k,v in ratings.items()};prev=season
  # Build candidates exclusively from prior games on this team and in this season.
  for g in gs:
   gid=str(g["game_id"]);h,a=g["home_team"],g["away_team"]
   rh,ra=ratings.get(h,0.),ratings.get(a,0.);adv=0. if g["neutral_site"] else 3.
   feat=[rh,ra,rh-ra,adv,*team_features(teamhist[(season,h)]),*team_features(teamhist[(season,a)])]
   margin=g["home_points"]-g["away_points"];total=g["home_points"]+g["away_points"]
   out["moneyline"].append(dict(season=season,week=week,gid=gid,home=h,away=a,x=feat,y=float(margin>0),baseline=float(sigmoid((rh-ra+adv)/12)),margin=margin,total=total))
   actual={(str(r["player_id"]),r["team"]):r for r in players[gid]}
   for team,opp in [(h,a),(a,h)]:
    for pid,pos in roster[(season,team)].items():
     for market,(_,_,line,minatt,positions) in MARKETS.items():
      if pos not in positions:continue
      ph=hist[(season,team,pid,market)]
      if len(ph)<3 or len(teamweeks[(season,team)])<3:continue
      recent=ph[-3:]
      # A player must have an observed row in each of the team's last three games.
      if [r[1] for r in recent]!=teamweeks[(season,team)][-3:]:continue
      att=avg([r[3] for r in recent])
      if att<minatt:continue
      v=np.array([[r[3],r[4],r[5]] for r in ph],dtype=float)
      vr=v[-3:];v8=v[-8:];lgv,lgy=league[market]
      le=lgy/max(lgv,1)
      opph=oppvol[(season,opp,market)]
      oe=sum(r[1] for r in opph)/max(sum(r[0] for r in opph),1) if opph else le
      volume=teamvol[(season,team,market)]
      tv=avg([r[3] for r in volume[-3:]])
      f=[len(ph),att,avg(v[:,0]),avg(vr[:,1]),avg(v[:,1]),avg(v[-1:,1]),float(np.std(v8[:,1])),
         float(np.std(v8[:,0])),float(v8[:,1].sum()/max(v8[:,0].sum(),1)),le,oe,tv,
         avg(vr[:,0]/np.maximum(vr[:,2],1)),float(v[-1,0]),float(v[-1,1]),
         ratings.get(team,0.)-ratings.get(opp,0.),float(team==h and not g["neutral_site"]),*team_features(teamhist[(season,team)]),*team_features(teamhist[(season,opp)])]
      old=project(ph,volume,opph,league[market],market) if market in OLD_MARKETS else None
      base=avg(vr[:,1])
      r=actual.get((pid,team));pair=values(r,market) if r else None
      key=(season,market);missing[key][0]+=1
      if pair is None:
       missing[key][1]+=1;continue
      out[market].append(dict(season=season,week=week,gid=gid,pid=pid,team=team,x=f,y=pair[1],baseline=base,old_numeric=old["expected_stat"] if old else None,line=line,actual_opportunities=pair[0]))
  # Only after every game in the week has been forecast: update stats and ratings.
  changes=defaultdict(float)
  for g in gs:
   gid=str(g["game_id"]);h,a=g["home_team"],g["away_team"];hp,ap=g["home_points"],g["away_points"]
   diff=hp-ap;adv=0 if g["neutral_site"] else 3.
   expected=ratings.get(h,0.)-ratings.get(a,0.)+adv
   err=np.clip(diff-expected,-28,28)
   changes[h]+=.12*err;changes[a]-=.12*err
   for t,o,p,q in [(h,a,hp,ap),(a,h,ap,hp)]:
    teamhist[(season,t)].append((p,q,p-q-(ratings.get(t,0.)-ratings.get(o,0.))))
    teamweeks[(season,t)].append(week)
   sums=defaultdict(lambda:[0.,0.])
   for r in players[gid]:
    t=r["team"];pid=str(r["player_id"]);pos=str(r["position"])
    if t not in (h,a):continue
    roster[(season,t)][pid]=pos
    for m in MARKETS:
     pair=values(r,m)
     if pair is None:continue
     sums[(t,m)][0]+=pair[0];sums[(t,m)][1]+=pair[1]
   for r in players[gid]:
    t=r["team"];pid=str(r["player_id"])
    if t not in (h,a):continue
    for m in MARKETS:
     pair=values(r,m)
     if pair is not None:hist[(season,t,pid,m)].append((season,week,t,*pair,sums[(t,m)][0]))
   for (t,m),(v,y) in sums.items():
    if v>0:
     teamvol[(season,t,m)].append((season,week,gid,v));oppvol[(season,a if t==h else h,m)].append((v,y))
     league[m][0]+=v;league[m][1]+=y
  for t,c in changes.items():ratings[t]=ratings.get(t,0.)+c
 return out,missing
def summarize(y,p):
 p=np.clip(p,1e-6,1-1e-6)
 return {"n":len(y),"brier":float(np.mean((p-y)**2)),"accuracy":float(np.mean((p>=.5)==y)),
         "logloss":float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))}
def numeric(y,p):
 return {"n":len(y),"mae":float(np.mean(abs(p-y))),"bias":float(np.mean(p-y))}
def paired_ci(rows,y,p,q,prob=False):
 groups=defaultdict(list)
 for r,a,b,c in zip(rows,y,p,q):
  groups[r["gid"]].append((b-a)**2-(c-a)**2 if prob else abs(b-a)-abs(c-a))
 sums=np.array([sum(v) for v in groups.values()]);ns=np.array([len(v) for v in groups.values()])
 rng=np.random.default_rng(2711);idx=rng.integers(0,len(ns),(1000,len(ns)))
 v=sums[idx].sum(axis=1)/ns[idx].sum(axis=1)
 return [float(x) for x in np.quantile(v,[.025,.975])]
def original_coverage(con):
 from cfb_grade_record_a import load_ledger,load_games,load_player_rows,select_canonical,Schedule,is_moneyline,base_market
 ledger=load_ledger(Path("docs/cfb_picks_log.jsonl"))
 con.row_factory=sqlite3.Row
 games,_=load_games(con);actual=load_player_rows(con,ledger)
 canonical,integrity=select_canonical(ledger,Schedule(games))
 team_rows=defaultdict(int);player_other=defaultdict(set)
 for (pid,gid),r in actual.items():
  team_rows[(gid,r["team"])]+=1;player_other[(pid,int(r["season"]),int(r["week"]))].add(r["team"])
 out=defaultdict(lambda:defaultdict(int));details=[]
 for k,record in canonical.items():
  e,g=record["entry"],record["game"];m=base_market(e["market"])
  if is_moneyline(m) or not g or g.get("home_points") is None or g.get("away_points") is None:continue
  gid=str(g["game_id"]);pid=str(e.get("player_id"));team=e.get("team");w=int(e["week"]);s=int(e["season"])
  row=actual.get((pid,gid));d=out[m];d["final_canonical"]+=1
  if row is not None and row["team"]==team:
   d["observed_matching_team_row"]+=1;continue
  if row is not None:
   reason="player_row_wrong_team"
  elif not team_rows[(gid,team)]:
   reason="no_team_stat_feed_or_team_identity_mismatch"
  elif player_other[(pid,s,w)]-{team}:
   reason="other_team_observed_same_week"
  else:
   reason="player_not_in_team_stat_feed_participation_unknown"
  d[reason]+=1
  details.append({"key":k,"game":gid,"market":m,"week":w,"player":e.get("player"),"team":team,"reason":reason})
 return {"integrity":integrity,"by_market":out,"missing_or_mismatched_details":details,
         "rule":"No missing player classified as zero or DNP without independent participation evidence."}

def compare_archived(rows,market,truth,forecast,archive):
 """Original receipt vs retrospective challenger on identical graded targets.
    Neither comparison nor historical reconstruction is new forward evidence."""
 matched=[];old=[];new=[];ys=[]
 for r,y,p in zip(rows,truth,forecast):
  key=(r["gid"],None if market=="moneyline" else r["pid"],market)
  a=archive.get(key)
  if not a:continue
  if market=="moneyline":
   if a.get("team") not in (r["home"],r["away"]):continue
   o=float(a["original_model_prob"])
   o=o if a["team"]==r["home"] else 1-o
   actual=1. if a["actual"]==r["home"] else 0.
  else:
   if a.get("team")!=r.get("team") or a.get("line")!=r["line"]:continue
   if not isinstance(a.get("actual"),(int,float)) or abs(float(a["actual"])-r["y"])>1e-6:continue
   pick=a.get("pick","").upper()
   if not pick.startswith(("OVER","UNDER")):continue
   o=float(a["original_model_prob"])
   if pick.startswith("UNDER"):o=1-o
   actual=float(a["actual"]>r["line"])
  if actual!=y:continue
  matched.append(r);old.append(o);new.append(p);ys.append(y)
 if not matched:return {"n":0}
 yy=np.array(ys);op=np.array(old);np_=np.array(new)
 return {"n":len(matched),"original":summarize(yy,op),"challenger":summarize(yy,np_),
         "brier_gain_ci95":paired_ci(matched,yy,op,np_,True),
         "note":"Original archived probabilities vs reconstructed as-of predictions; not two prospectively captured forecasts."}

def run(con):
 rows,missing=build(con);result={"schema":"CFB_FULL_ENGINE_V1","promotion_allowed":False,"missing_stats":{},"markets":{}}
 result["original_missing_audit"]=original_coverage(con)
 archived=json.loads(Path("docs/cfb_record.json").read_text())["results"]
 archive={(str(r["game_id"]),None if r["market"]=="moneyline" else str(r["player_id"]),r["market"]):r
          for r in archived if r.get("original_model_prob") is not None and r.get("market") in (*MARKETS,"moneyline")}
 for (s,m),(n,k) in missing.items():result["missing_stats"][f"{s}_{m}"]={"pregame_candidates":n,"no_confirmed_stat_line":k}
 for m,rs in rows.items():
  result["markets"][m]={}
  for year in (2024,2025,2026):
   tr=[r for r in rs if r["season"]<year];te=[r for r in rs if r["season"]==year]
   if min(len(tr),len(te))<50:continue
   x=np.array([r["x"] for r in tr]);xt=np.array([r["x"] for r in te]);y=np.array([r["y"] for r in tr]);yt=np.array([r["y"] for r in te])
   assert max(r["season"] for r in tr)<year
   z={"train":len(tr),"test":len(te)}
   if m=="moneyline":
    by=y;byt=yt;base=np.array([r["baseline"] for r in te])
    z["probability"]={"rating_reference":summarize(byt,base)}
   else:
    line=MARKETS[m][2];by=(y>line).astype(float);byt=(yt>line).astype(float)
    base=np.array([r["baseline"] for r in te]);base_tr=np.array([r["baseline"] for r in tr])
    z["numeric"]={"last3":numeric(yt,base)}
    # Causal univariate calibration for the previous-3 mean baseline.
    simple=LogisticRegression(C=1,max_iter=500).fit(base_tr[:,None],by)
    base=simple.predict_proba(base[:,None])[:,1]
    z["probability"]={"last3_calibrated":summarize(byt,base)}
    for name,model in [("ridge",make_pipeline(StandardScaler(),Ridge(alpha=100.))),
                       ("nonlinear_mean",HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,min_samples_leaf=70,l2_regularization=10,random_state=17)),
                       ("nonlinear_median",HistGradientBoostingRegressor(loss="absolute_error",max_iter=120,max_leaf_nodes=15,min_samples_leaf=70,l2_regularization=10,random_state=17))]:
     model.fit(x,y);p=model.predict(xt);last=np.array([r["baseline"] for r in te])
     z["numeric"][name]={**numeric(yt,p),"mae_gain_vs_last3_ci95":paired_ci(te,yt,last,p)}
     idx=np.array([r["old_numeric"] is not None for r in te])
     if idx.any():
      old=np.array([r["old_numeric"] for r in te if r["old_numeric"] is not None])
      z["numeric"][name]["same_cohort_vs_old_numeric"]={"old":numeric(yt[idx],old),"new":numeric(yt[idx],p[idx])}
   for name,model in [("logistic",make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500))),
                       ("nonlinear_probability",HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,min_samples_leaf=70,l2_regularization=10,random_state=17))]:
    model.fit(x,by);p=model.predict_proba(xt)[:,1]
    z["probability"][name]={**summarize(byt,p),"brier_gain_ci95":paired_ci(te,byt,base,p,True)}
    if year==2026:
     z["probability"][name]["paired_archived_original"]=compare_archived(te,m,byt,p,archive)
    if m=="moneyline":
     confident=np.maximum(p,1-p)>=.7
     z["probability"][name]["70plus"]={"n":int(confident.sum()),"accuracy":float(np.mean((p[confident]>=.5)==byt[confident])) if confident.any() else None}
   if m=="moneyline":
    for target in ("margin","total"):
     t=np.array([r[target] for r in tr]);tt=np.array([r[target] for r in te])
     z[target]={}
     for name,model in [("ridge",make_pipeline(StandardScaler(),Ridge(alpha=100.))),("nonlinear",HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,min_samples_leaf=70,l2_regularization=10,random_state=17))]:
      model.fit(x,t);z[target][name]=numeric(tt,model.predict(xt))
   result["markets"][m][str(year)]=z
   print("CFB_STRESS",m,year,json.dumps(z,sort_keys=True),flush=True)
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument("--db",required=True);p.add_argument("--out",required=True);a=p.parse_args()
 con=sqlite3.connect("file:"+a.db+"?mode=ro",uri=True)
 result=run(con);con.close()
 Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
