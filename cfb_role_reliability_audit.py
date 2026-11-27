"""CFB personnel reliability: audit original pregame evidence, never infer DNP.
The target is exact archived grade availability after a game with verified
moneyline final, not participation. Rolling fits use earlier completed weeks.
No production change, no new official prediction, no automatic promotion.
"""
import argparse,json,math
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from statistics import mean
def stamp(s):return datetime.fromisoformat(s.replace("Z","+00:00"))
def identity(r):return (str(r["game_id"]),str(r["player_id"]),r["market"],r["team"])
def load(root):
 root=Path(root)
 grades=json.loads((root/"cfb_record.json").read_text())["results"]
 matches={identity(g):g for g in grades}
 finals={str(g["game_id"]) for g in grades if g["market"]=="moneyline"}
 receipts={}
 for line in (root/"cfb_personnel_first_seen.jsonl").read_text().splitlines():
  if not line.strip():continue
  r=json.loads(line)
  if stamp(r["first_source_capture_utc"])>=stamp(r["kickoff_utc"]):continue
  k=identity(r)
  if k not in receipts or stamp(r["first_source_capture_utc"])<stamp(receipts[k]["first_source_capture_utc"]):receipts[k]=r
 out=[]
 for k,r in receipts.items():
  if r["week"] not in range(7,13) or k[0] not in finals:continue
  g=matches.get(k)
  if g and (g["pick"]!=r["original_model_pick"] or abs(g["original_model_prob"]-r["original_model_probability"])>.00011):continue
  u=r["prior_team_game_usage_evidence"]
  out.append({"week":r["week"],"market":r["market"],"game":k[0],"player":r["player"],
    "role_rows":u["observed_player_boxscore_rows"],"mean_share":u.get("mean_share_when_observed"),
    "side":"over" if r["original_model_pick"].startswith("OVER") else "under",
    "p":r["original_model_probability"],"matched":int(g is not None),
    "correct":int(g["result"]=="hit") if g else None})
 return out
def metrics(rows):
 n=len(rows);g=[r for r in rows if r["matched"]];h=sum(r["correct"] for r in g)
 return {"n":n,"graded":len(g),"unmatched":n-len(g),"hit":h,"miss":len(g)-h,
         "observed_hit_rate":h/len(g) if g else None,"grade_coverage":len(g)/n if n else None}
def risk_fit(train,row,kind):
 prior=(sum(r["matched"] for r in train)+2)/(len(train)+4)
 if kind=="global":return prior
 # Shrink role/market estimates to prior, with fixed pseudo-counts.
 pool=[r for r in train if r["role_rows"]==row["role_rows"]]
 role=(sum(r["matched"] for r in pool)+12*prior)/(len(pool)+12)
 if kind=="role":return role
 pool=[r for r in pool if r["market"]==row["market"]]
 return (sum(r["matched"] for r in pool)+16*role)/(len(pool)+16)
def probability_metrics(rows):
 if not rows:return {}
 return {"n":len(rows),"brier":mean((p-y)**2 for p,y in rows),
         "logloss":mean(-y*math.log(max(p,1e-9))-(1-y)*math.log(max(1-p,1e-9)) for p,y in rows),
         "mean_probability":mean(p for p,y in rows),"observed_grade_rate":mean(y for p,y in rows)}
def audit(rows):
 report={"schema":"CFB_PERSONNEL_GRADE_RELIABILITY_V1","target":"exact grade available; not participation or betting settlement",
  "historical_diagnostic_only":True,"promotion_allowed":False,"cohorts":{},"rolling":{}}
 for market in ["all","passing_touchdowns","rushing_yards","anytime_touchdowns"]:
  q=[r for r in rows if market=="all" or r["market"]==market]
  report["cohorts"][market]={name:metrics([r for r in q if pred(r)]) for name,pred in [
   ("all",lambda r:True),("role_three",lambda r:r["role_rows"]==3),
   ("role_incomplete",lambda r:r["role_rows"]<3),("role_zero",lambda r:r["role_rows"]==0),
   ("overs_role_three",lambda r:r["side"]=="over" and r["role_rows"]==3),
   ("overs_role_incomplete",lambda r:r["side"]=="over" and r["role_rows"]<3)]}
 for week in range(9,13):
  train=[r for r in rows if r["week"]<week];test=[r for r in rows if r["week"]==week]
  if not train or not test:continue
  for r in test:
   for k in ["global","role","role_market"]:r["coverage_"+k]=risk_fit(train,r,k)
  report["rolling"][str(week)]={"n":len(test),"probabilities":{k:probability_metrics([(r["coverage_"+k],r["matched"]) for r in test]) for k in ["global","role","role_market"]},
   "all":metrics(test),"role_three":metrics([r for r in test if r["role_rows"]==3]),
   "coverage_80plus":metrics([r for r in test if r["coverage_role_market"]>=.8])}
 test=[r for r in rows if "coverage_global" in r]
 report["rolling_all"]={"probabilities":{k:probability_metrics([(r["coverage_"+k],r["matched"]) for r in test]) for k in ["global","role","role_market"]},
  "all":metrics(test),"role_three":metrics([r for r in test if r["role_rows"]==3]),
  "coverage_80plus":metrics([r for r in test if r["coverage_role_market"]>=.8])}
 return report
def main():
 p=argparse.ArgumentParser();p.add_argument("--docs",default="docs");p.add_argument("--out",required=True);a=p.parse_args()
 out=audit(load(a.docs));Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print("CFB_ROLE_AUDIT",json.dumps(out,sort_keys=True))
if __name__=="__main__":main()
