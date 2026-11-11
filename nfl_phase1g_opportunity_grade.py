#!/usr/bin/env python3
"""Grade actual original-pregame player OPPORTUNITIES, not just yardage.

Phase1G's target/carry engine is research only. Strict same-game/player/market
matching, real prekickoff receipts, no result-driven filtering, and missing
observations never become zero. Compare Phase1F+G challenger to Phase1E
original on identical rows. Does not assert the challenger is superior.
"""
from __future__ import annotations
import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import nfl_phase1_store as ST

MARKETS={"rec_yds":("targets","targets"),"rush_yds":("carries","rush_att"),
         "pass_yds":("attempts","pass_att")}

def _utc(v):
    d=datetime.fromisoformat(str(v).replace("Z","+00:00"))
    if d.tzinfo is None:raise ValueError("NAIVE_TIME")
    return d.astimezone(timezone.utc)

def _original(path,kind):
    h,rows=ST.Store(path.parent.parent,kind).read_batch(path)
    if _utc(h["generated_at"])>=_utc(rows[0]["kickoff"]):
        raise ValueError("NOT_ORIGINAL_PREGAME")
    return h,rows

def grade(root,db):
    root=Path(root)
    baseline=ST.Store(root,"forecasts")
    challenger=ST.Store(root,"challenger_forecasts")
    bfiles={p.name:p for p in baseline.batch_files()}
    cfiles={p.name:p for p in challenger.batch_files()}
    details=[];exclusions=defaultdict(int)
    with sqlite3.connect("file:"+str(Path(db).absolute())+"?mode=ro",uri=True) as con:
        for name in sorted(set(bfiles)&set(cfiles)):
            bh,br=_original(bfiles[name],"forecasts")
            ch,cr=_original(cfiles[name],"challenger_forecasts")
            b={(r["game_id"],r["team"],str(r["player_id"]),r["outcome"]):r
               for r in br if r.get("outcome") in MARKETS}
            c={(r["game_id"],r["team"],str(r["player_id"]),r["outcome"]):r
               for r in cr if r.get("outcome") in MARKETS}
            for key in sorted(set(b)&set(c)):
                old,new=b[key],c[key]
                if old["horizon"]!=new["horizon"] or old["kickoff"]!=new["kickoff"]:
                    exclusions["schedule_or_horizon_mismatch"]+=1;continue
                bp=old["input_snapshots"];cp=new["input_snapshots"]
                if bp.get("content_id")!=cp.get("content_id"):
                    exclusions["not_same_original_source"]+=1;continue
                evidence=new.get("phase1f_evidence") or {}
                if evidence.get("individual_opportunity_model")=="NOT_ENABLED":
                    exclusions["role_model_not_enabled"]+=1;continue
                if not evidence.get("player_opportunity_receipts"):
                    exclusions["no_role_receipts"]+=1;continue
                actual_col,opportunity=MARKETS[key[3]]
                oldp=old.get("expected_opportunities")
                newp=new.get("expected_opportunities")
                if oldp is None or newp is None or not all(
                        math.isfinite(float(v)) and float(v)>=0 for v in (oldp,newp)):
                    exclusions["missing_or_invalid_expected_opportunity"]+=1;continue
                actuals=con.execute(
                    f"SELECT {actual_col} FROM player_games WHERE game_id=? AND team=? "
                    "AND player_id=? AND season_type='REG'",
                    (key[0],key[1],key[2])).fetchall()
                if len(actuals)!=1 or actuals[0][0] is None:
                    exclusions["unresolved_actual_not_zero"]+=1;continue
                actual=float(actuals[0][0])
                if not math.isfinite(actual) or actual<0:
                    exclusions["invalid_actual"]+=1;continue
                details.append({"game_id":key[0],"horizon":old["horizon"],
                    "team":key[1],"player_id":key[2],
                    "player":old.get("player_name"),"market":key[3],
                    "component":actual_col,"actual":actual,
                    "phase1e_opportunities":float(oldp),
                    "phase1g_opportunities":float(newp),
                    "phase1e_abs_error":abs(float(oldp)-actual),
                    "phase1g_abs_error":abs(float(newp)-actual),
                    "same_snapshot_content_id":bp["content_id"]})
    def metrics(rows):
        n=len(rows)
        if not n:return {"n":0}
        return {"n":n,
          "phase1e_mae":round(sum(r["phase1e_abs_error"] for r in rows)/n,4),
          "phase1g_mae":round(sum(r["phase1g_abs_error"] for r in rows)/n,4),
          "phase1g_better":sum(r["phase1g_abs_error"]<r["phase1e_abs_error"] for r in rows),
          "phase1g_worse":sum(r["phase1g_abs_error"]>r["phase1e_abs_error"] for r in rows),
          "phase1g_equal":sum(r["phase1g_abs_error"]==r["phase1e_abs_error"] for r in rows)}
    return {"schema":"NFL_PHASE1G_ORIGINAL_PREGAME_OPPORTUNITY_COMPONENT_GRADE_A",
      "research_only":True,"promotion_eligible":False,
      "matched_originals":metrics(details),
      "by_market":{m:metrics([r for r in details if r["market"]==m]) for m in MARKETS},
      "by_game":{g:metrics([r for r in details if r["game_id"]==g])
                  for g in sorted({r["game_id"] for r in details})},
      "exclusions":dict(exclusions),"player_rows":details}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--state",required=True);p.add_argument("--db",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    result=grade(a.state,a.db)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="player_rows"},indent=2))

if __name__=="__main__":main()
