#!/usr/bin/env python3
"""Immutable CFB numerical shadow: first-prekickoff predictions + actual grading.

Shadow only. Does NOT substitute for validated model or sportsbook lines.
- Publish only if source model is explicitly FROZEN_SHADOW_NO_PROMOTION.
- Require exact ESPN scheduled game/id and strictly before kickoff.
- The earliest ledger receipt is never rewritten after line/model refresh.
- Historical yard/TD actual measured ONLY from confirmed final games with
  observed player-game boxscore records, never treating missing as zero.
- Do not invent betting odds, hit probabilities, starters or trades.
"""
import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean

MODEL="cfb_numerical_opportunity_v01_PRECOMMITTED_SHADOW"
MARKET_COLUMN={"rushing_yards":"rushing_yards",
               "receiving_yards":"receiving_yards",
               "passing_touchdowns":"passing_touchdowns"}

def utc(x):
 if not isinstance(x,str) or not x:return None
 try:
  y=datetime.fromisoformat(x.replace("Z","+00:00"))
  return y.astimezone(timezone.utc) if y.tzinfo else None
 except ValueError:return None

def identity(r):
 return (str(r["season"]),str(r["week"]),str(r["game_id"]),
         str(r["player_id"]),str(r["market"]))

def load_ledger(path):
 out=[]
 if not Path(path).exists():return out
 for line in Path(path).read_text().splitlines():
  if not line.strip():continue
  out.append(json.loads(line))
 return out

def publish(con,snapshot,board,ledger,now):
 """returns NEW immutable receipts and audit. Does not rewrite old ledger."""
 if snapshot.get("research_model")!=MODEL or snapshot.get("status")!="FROZEN_SHADOW_NO_PROMOTION":
  raise ValueError("not the frozen numerical shadow")
 if snapshot.get("upcoming_asof")!=board.get("generated_at_utc"):
  raise ValueError("model snapshot is not the current board generation")
 if utc(board.get("generated_at_utc")) is None or utc(board["generated_at_utc"])>now:
  raise ValueError("model board timestamp is missing or future")
 rows=snapshot.get("upcoming_numeric_projections_RESEARCH_ONLY")
 if not isinstance(rows,list):raise ValueError("missing numeric candidate cohort")
 season,week=int(board["season"]),int(board["week"])
 known={}
 for gid,home,away,kick,state,hp,ap in con.execute("""
    SELECT g.game_id,g.home_team,g.away_team,g.kickoff_utc,
      COALESCE(s.espn_state,'unknown'),g.home_points,g.away_points
    FROM games g LEFT JOIN schedule_snapshot s ON s.game_id=g.game_id
    WHERE g.season=? AND g.week=?""",(season,week)):
  ko=utc(kick)
  if ko and ko>now and hp is None and ap is None:
   known[str(gid)]=(home,away,ko,state)
 original={identity(x) for x in ledger}
 new=[];reasons=defaultdict(int);seen=set()
 for r in rows:
  gid=str(r.get("game_id"));mkt=r.get("market")
  rec=known.get(gid)
  if mkt not in MARKET_COLUMN or not rec:
   reasons["invalid_or_started_game"]+=1;continue
  home,away,ko,espn=rec
  if espn not in ("pre","unknown"):
   reasons["espn_not_confirmed_pregame"]+=1;continue
  if {r.get("team"),r.get("opponent")}!={home,away}:
   reasons["team_or_opponent_mismatch"]+=1;continue
  if utc(r.get("kickoff_utc"))!=ko:
   reasons["kickoff_mismatch"]+=1;continue
  projection=r.get("expected_stat")
  if not isinstance(projection,(int,float)) or isinstance(projection,bool) or not math.isfinite(projection) or projection<0:
   reasons["invalid_numerical_stat"]+=1;continue
  if not r.get("player_id") or not r.get("player"):
   reasons["player_identity_missing"]+=1;continue
  fresh={"season":season,"week":week,"game_id":gid,
        "team":r["team"],"opponent":r["opponent"],
        "player_id":str(r["player_id"]),"player":r["player"],
        "market":mkt,"expected_stat":round(float(projection),5),
        "published_at_utc":now.isoformat().replace("+00:00","Z"),
        "original_board_generated_at_utc":board["generated_at_utc"],
        "model_source":MODEL,"kickoff_utc":ko.isoformat().replace("+00:00","Z"),
        "research_only":True,"not_a_betting_pick":True,
        "role_is_confirmed_active_starter":False}
  key=identity(fresh)
  if key in seen or key in original:
   reasons["prior_immutable_prediction_exists"]+=1;continue
  seen.add(key);new.append(fresh)
 return new,{"new_first_pregame":len(new),
             "unchanged_first_pregame":len(original),
             "omitted_reasons":dict(reasons),
             "source_candidates":len(rows),"model_changed":False}

def grade(con,ledger):
 """Grade earliest valid immutable numerical receipt; preserve omitted rows."""
 groups=defaultdict(list)
 for i,e in enumerate(ledger):
  try:key=identity(e)
  except (TypeError,KeyError):continue
  groups[key].append((i,e))
 actuals=[];pending=defaultdict(int);overwritten=0
 for key,pairs in groups.items():
  if len(pairs)>1:overwritten+=len(pairs)-1
  valid=[(i,e) for i,e in pairs if utc(e.get("published_at_utc")) and
         utc(e.get("kickoff_utc")) and utc(e["published_at_utc"])<utc(e["kickoff_utc"])
         and e.get("model_source")==MODEL]
  if not valid:
   pending["no_authenticated_pregame_receipt"]+=1;continue
  first=min(valid,key=lambda z:(utc(z[1]["published_at_utc"]),z[0]))[1]
  gid=str(first["game_id"])
  g=con.execute("""SELECT home_points,away_points FROM games WHERE game_id=?""",(gid,)).fetchone()
  if not g:
   pending["game_unresolved"]+=1;continue
  if g[0] is None or g[1] is None:
   pending["game_not_final"]+=1;continue
  market=first["market"]
  if market not in MARKET_COLUMN:
   pending["unsupported_market"]+=1;continue
  x=con.execute(f"""SELECT {MARKET_COLUMN[market]}
     FROM player_games WHERE game_id=? AND player_id=? AND team=?""",
     (gid,first["player_id"],first["team"])).fetchall()
  if not x or x[0][0] is None:
   pending["player_no_stat_line_after_final_NOT_ZERO"]+=1;continue
  if len(x)!=1:
   pending["ambiguous_player_game"]+=1;continue
  actual=float(x[0][0])
  if not math.isfinite(actual):
   pending["invalid_actual"]+=1;continue
  predicted=float(first["expected_stat"])
  actuals.append({"season":first["season"],"week":first["week"],
    "game_id":gid,"market":market,"player":first["player"],
    "team":first["team"],"predicted_mean":predicted,
    "observed_actual":actual,"error":round(predicted-actual,4),
    "absolute_error":round(abs(predicted-actual),4),
    "first_published_at_utc":first["published_at_utc"]})
 market_summary={}
 for market in MARKET_COLUMN:
  sample=[r for r in actuals if r["market"]==market]
  market_summary[market]={"graded":len(sample),
      "mae":round(mean(r["absolute_error"] for r in sample),3) if sample else None,
      "signed_bias":round(mean(r["error"] for r in sample),3) if sample else None}
 return {"status":"OBSERVATIONAL_FIRST_PREGAME_NUMERIC_GRADER",
   "model_source":MODEL,"graded_count":len(actuals),
   "duplicate_or_later_ledger_rows_never_used":overwritten,
   "ungraded":dict(pending),
   "by_market":market_summary,
   "evaluations":actuals,
   "betting_edge_calibrated":False,
   "promotion":"NOT_AUTHORIZED"}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--db",default="cfb_models/cfb_model.sqlite")
 p.add_argument("--board",default="docs/cfb_predictions.json")
 p.add_argument("--numeric",default="docs/cfb_numeric_shadow_live.json")
 p.add_argument("--ledger",default="docs/cfb_numeric_first_seen.jsonl")
 p.add_argument("--out",default="docs/cfb_numeric_forward_grade.json")
 a=p.parse_args()
 db=sqlite3.connect(f"file:{a.db}?mode=ro",uri=True)
 prior=load_ledger(a.ledger)
 snapshot_path=Path(a.numeric)
 if snapshot_path.exists():
  snapshot=json.loads(snapshot_path.read_text())
  board=json.loads(Path(a.board).read_text())
  added,audit=publish(db,snapshot,board,prior,datetime.now(timezone.utc))
  if added:
   Path(a.ledger).parent.mkdir(parents=True,exist_ok=True)
   with Path(a.ledger).open("a") as out:
    for row in added:out.write(json.dumps(row,sort_keys=True)+"\n")
   prior+=added
  print("CFB first-pregame numerical shadow receipt: "+json.dumps(audit,sort_keys=True))
 result=grade(db,prior)
 db.close()
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps({k:result[k] for k in ("graded_count","ungraded","by_market","promotion")},sort_keys=True))
if __name__=="__main__":main()
