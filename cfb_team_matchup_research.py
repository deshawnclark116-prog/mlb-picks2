"""Research only: expanding-season comparison, never publishes official picks."""
import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

def features(rows):
    history=[]
    result=[]
    for key in sorted({(r[0],r[1]) for r in rows}):
        season,week=key
        past=[r for r in history if r[0]>=season-2]
        teams=sorted({t for r in past for t in r[2:4]})
        idx={t:i for i,t in enumerate(teams)}
        n=len(teams)
        if past:
            X=[]; y=[]; weights=[]
            for s,w,h,a,hp,ap,neutral in past:
                for t,o,p,home in [(h,a,hp,1-neutral),(a,h,ap,0)]:
                    x=np.zeros(2*n+1); x[idx[t]]=1; x[n+idx[o]]=1; x[-1]=home
                    X.append(x); y.append(p); weights.append(0.5**(season-s))
            fit=Ridge(alpha=20).fit(X,y,sample_weight=weights)
            coef=fit.coef_
        else:
            coef=np.zeros(2*n+1)
        recent=defaultdict(list)
        for s,w,h,a,hp,ap,neutral in past:
            if s==season:
                recent[h].append(hp-ap); recent[a].append(ap-hp)
        for r in rows_by_week(rows,key):
            s,w,h,a,hp,ap,neutral=r
            def rating(t,k):
                return coef[k*n+idx[t]] if t in idx else 0.
            def form(t):
                z=recent[t][-5:]; return sum(z)/(len(z)+3)
            f=[rating(h,0)-rating(a,0),rating(h,1)-rating(a,1),form(h)-form(a),1-neutral,len(recent[h]),len(recent[a])]
            result.append((s,w,f,int(hp>ap)))
        history.extend(rows_by_week(rows,key))
    return result

def rows_by_week(rows,key):
    return [r for r in rows if tuple(r[:2])==key]

def evaluate(data):
    output=[]
    for season in sorted({r[0] for r in data}):
        train=[r for r in data if r[0]<season]
        test=[r for r in data if r[0]==season]
        if len(train)<500 or len(test)<100: continue
        X=np.array([r[2] for r in train]); y=np.array([r[3] for r in train])
        T=np.array([r[2] for r in test]); Y=np.array([r[3] for r in test])
        for name,model,cols in [
            ("recent_margin_baseline",LogisticRegression(max_iter=1000),[2,3]),
            ("adjusted_offense_defense",LogisticRegression(max_iter=1000),list(range(6))),
            ("nonlinear_matchup",HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,l2_regularization=10,random_state=42),list(range(6)))]:
            model.fit(X[:,cols],y)
            p=model.predict_proba(T[:,cols])[:,1]
            output.append(dict(season=season,model=name,n=len(Y),accuracy=accuracy_score(Y,p>=.5),brier=brier_score_loss(Y,p),log_loss=log_loss(Y,p)))
    return output

def main():
    p=argparse.ArgumentParser(); p.add_argument("--db",default="cfb_models/cfb_model.sqlite"); p.add_argument("--out",required=True); a=p.parse_args()
    con=sqlite3.connect(f"file:{Path(a.db).resolve()}?mode=ro",uri=True)
    rows=con.execute("SELECT season,week,home_team,away_team,home_points,away_points,COALESCE(neutral_site,0) FROM games WHERE season<=2025 AND home_points IS NOT NULL AND away_points IS NOT NULL AND home_points!=away_points ORDER BY season,week,game_id").fetchall()
    con.close()
    result={"research_only":True,"historical_development_only":True,"incumbent_comparison":False,"rows":len(rows),"metrics":evaluate(features(rows))}
    Path(a.out).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not result["metrics"]: raise SystemExit("Insufficient historical coverage")
if __name__=="__main__": main()
