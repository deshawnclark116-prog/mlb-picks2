#!/usr/bin/env python3
"""CFB numerical player-stat predictions, FROZEN research shadow v0.1.

This is deliberately DIFFERENT from the CFB production classifiers that
answer P(over a FIXED 69.5 rush / 1.5 pass TD line). Here we estimate actual
yards/TD counts using observed player opportunities, team share, teammate
volume and opponent prior observed efficiency, with as-of uncertainty.

No bookmaker availability, injury resolution, starting lineup, targets or
snap counts are inferred. An unavailable player's missing game is NOT zero.
2025/2026 outcomes have already been consulted in earlier research and
must NOT be called untouched holdouts. No model weights are tuned after
these results are viewed. Promotion requires independent future evidence.
"""
import argparse
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean

MODEL="cfb_numerical_opportunity_v01_PRECOMMITTED_SHADOW"
CFG={
 "recent_three_weight":0.65,"season_weight":0.35,
 "direct_attempt_weight":0.60,"team_share_attempt_weight":0.40,
 "opponent_adjustment_clamp":0.12,"opp_pseudovol":120.0,
 "yards_efficiency_pseudovol":{"rushing_yards":30.0,"receiving_yards":15.0,
                               "passing_touchdowns":90.0},
 "min_avg_attempts":{"rushing_yards":2.0,"receiving_yards":1.0,
                     "passing_touchdowns":8.0},
}
MARKETS={
 "rushing_yards":("carries","rushing_yards",{"RB","QB","FB","WR"}),
 "receiving_yards":("receptions","receiving_yards",{"WR","TE","RB","FB"}),
 "passing_touchdowns":("pass_attempts","passing_touchdowns",{"QB"}),
}

def avg(x):
 return mean(x) if x else None

def clip(v,a,b):
 return max(a,min(b,v))

def project(history,team_hist,opp_hist,league_sum,market):
 """History records are all completed BEFORE the target week. Last three
 entries must be observed on the same team in the same current season."""
 if market not in MARKETS or len(history)<3:return None
 recent=history[-3:]
 season,team=recent[-1][0],recent[-1][2]
 if len({(r[0],r[2]) for r in recent})!=1:return None
 season_history=[r for r in history if r[0]==season and r[2]==team]
 if len(season_history)<3:return None
 attempts_recent=avg([r[3] for r in recent])
 if attempts_recent<CFG["min_avg_attempts"][market]:return None
 attempts_season=avg([r[3] for r in season_history])
 share=[r[3]/r[5] for r in recent if r[5]>0]
 team_recent=[r[3] for r in team_hist[-3:] if r[0]==season]
 if len(share)!=3 or len(team_recent)!=3:return None
 expected_team_volume=avg(team_recent)
 expected_share=avg(share)
 direct=(CFG["recent_three_weight"]*attempts_recent+
         CFG["season_weight"]*attempts_season)
 expected=(CFG["direct_attempt_weight"]*direct+
           CFG["team_share_attempt_weight"]*expected_share*expected_team_volume)
 league_vol,league_yds=league_sum
 if league_vol<=0:return None
 league_eff=league_yds/league_vol
 last8=history[-8:]
 vol=sum(r[3] for r in last8)
 yds=sum(r[4] for r in last8)
 pseudo=CFG["yards_efficiency_pseudovol"][market]
 eff=(yds+pseudo*league_eff)/(vol+pseudo)
 opposing_volume=sum(r[0] for r in opp_hist)
 opposing_outcomes=sum(r[1] for r in opp_hist)
 delta=0.
 if opposing_volume>=10 and league_eff>0:
  observed=opposing_outcomes/opposing_volume/league_eff-1
  delta=clip(observed*opposing_volume/(opposing_volume+CFG["opp_pseudovol"]),
             -CFG["opponent_adjustment_clamp"],CFG["opponent_adjustment_clamp"])
 older=[r[3] for r in history[-6:-3] if (r[0],r[2])==(season,team)]
 ratio=attempts_recent/avg(older) if len(older)==3 and avg(older)>0 else None
 return {
  "statistic":"estimated_conditional_mean_NOT_median_or_bet",
  "expected_stat":max(0,expected*eff*(1+delta)),
  "expected_opportunities":round(expected,3),
  "observed_team_share_recent3":round(expected_share,4),
  "past_three_opportunities":round(attempts_recent,3),
  "team_expected_total_opportunities":round(expected_team_volume,3),
  "estimated_efficiency_per_opportunity":round(eff,4),
  "opponent_prior_efficiency_adjustment":round(delta,4),
  "role_change_ratio_observed":round(ratio,3) if ratio is not None else None,
  "history_observed_games":len(history),
  "certainty":"UNVALIDATED_RESEARCH_NO_BOOK_EDGE",
 }

def rows_for(con):
 p=[r[1] for r in con.execute("PRAGMA table_info(player_games)")]
 needed={"player_id","player_name","position","team","opponent","season","week",
         "game_id","carries","rushing_yards","receptions","receiving_yards",
         "pass_attempts","passing_touchdowns"}
 if not needed<=set(p):raise ValueError("incorrect player_games schema")
 return con.execute("""SELECT p.player_id,p.player_name,p.position,p.team,p.opponent,
      p.season,p.week,p.game_id,p.carries,p.rushing_yards,p.receptions,p.receiving_yards,
      p.pass_attempts,p.passing_touchdowns
      FROM player_games p JOIN games g ON g.game_id=p.game_id
      WHERE g.home_points IS NOT NULL AND g.away_points IS NOT NULL
      ORDER BY p.season,p.week,p.game_id,p.player_id""").fetchall()

def market_stats(r,market):
 a,b,pos=MARKETS[market]
 if str(r[2]) not in pos:return None
 ix_a={"carries":8,"receptions":10,"pass_attempts":12}[a]
 ix_b={"rushing_yards":9,"receiving_yards":11,"passing_touchdowns":13}[b]
 v=r[ix_a];y=r[ix_b]
 if v is None or y is None:return None
 return float(v),float(y)

def walkforward(con,target=None):
 """Simultaneously hold out every game of a given week before ingesting it."""
 weeks=defaultdict(list)
 for r in rows_for(con):
  key=(int(r[5]),int(r[6]))
  if target is None or key<target:weeks[key].append(r)
 history=defaultdict(list)
 team_history=defaultdict(list)
 opponent_history=defaultdict(list)
 league=[0.0,0.0]
 league_by_market=defaultdict(lambda:[0.,0.])
 observed=defaultdict(int)
 forecasts=[]
 for (season,week),playing in sorted(weeks.items()):
  team_sum=defaultdict(lambda:[0.,0.])
  all_game_teams=set()
  for r in playing:
   gid,team=r[7],r[3];all_game_teams.add((gid,team))
   for market in MARKETS:
    pair=market_stats(r,market)
    if pair:
     a,y=pair
     team_sum[(gid,team,market)][0]+=a
     team_sum[(gid,team,market)][1]+=y
  # No current-week data are inserted before the COMPLETE week is scored.
  for r in playing:
   pid,name,pos,team,opp,s,w,gid=r[:8]
   for market in MARKETS:
    pair=market_stats(r,market)
    if not pair:continue
    observed[(season,market)]+=1
    hist=history[(pid,market)]
    result=project(hist,team_history[(season,team,market)],
        opponent_history[(season,opp,market)],
        league_by_market[market],market)
    if not result:continue
    forecasts.append({
      "season":season,"week":week,"game_id":gid,"player_id":pid,
      "player":name,"team":team,"opponent":opp,"market":market,
      "actual_stat":pair[1],"baseline_last3":avg([h[4] for h in hist[-3:]]),
      **result
    })
  # Now and ONLY now can this game's actual data become future prior data.
  for r in playing:
   pid,name,pos,team,opp,s,w,gid=r[:8]
   for market in MARKETS:
    pair=market_stats(r,market)
    if pair:
     a,y=pair
     tv=team_sum[(gid,team,market)][0]
     history[(pid,market)].append((season,week,team,a,y,tv))
  for (gid,team,mkt),(vol,outcome) in team_sum.items():
   if vol>0:
    team_history[(season,team,mkt)].append((season,week,gid,vol))
    league_by_market[mkt][0]+=vol
    league_by_market[mkt][1]+=outcome
   # Opponent is known only from the completed game, but its allowed stats
   # contribute strictly to NEXT week, never the game itself.
   participants=[r[4] for r in playing if r[7]==gid and r[3]==team and r[4]]
   opponent=participants[0] if participants else None
   if opponent and vol>0:
    opponent_history[(season,opponent,mkt)].append((vol,outcome))
 return forecasts,observed,(history,team_history,opponent_history,league_by_market)

def paired_metrics(rows):
 if not rows:return {"n":0}
 e=[abs(r["expected_stat"]-r["actual_stat"]) for r in rows]
 baseline=[abs(r["baseline_last3"]-r["actual_stat"]) for r in rows]
 return {"n":len(rows),"mae_numeric":round(mean(e),3),
         "mae_last3_baseline":round(mean(baseline),3),
         "lift_vs_naive":round(mean(baseline)-mean(e),3),
         "bias":round(mean(r["expected_stat"]-r["actual_stat"] for r in rows),3),
         "within_10_yards_or_1td":round(sum(abs(r["expected_stat"]-r["actual_stat"])<=(1 if
               r["market"]=="passing_touchdowns" else 10) for r in rows)/len(rows),4)}

def cluster_ci(rows,n=900):
 by_game=defaultdict(list)
 for r in rows:
  by_game[r["game_id"]].append(
   abs(r["baseline_last3"]-r["actual_stat"])-abs(r["expected_stat"]-r["actual_stat"]))
 if len(by_game)<3:return None
 data=list(by_game.values());rng=random.Random(20261011)
 lifts=[]
 for _ in range(n):
  blocks=[data[rng.randrange(len(data))] for __ in data]
  lifts.append(mean(v for block in blocks for v in block))
 lifts.sort()
 return [round(lifts[22],3),round(lifts[877],3)]

def report(forecasts,observed):
 cohorts={}
 for season in (2025,2026):
  for market in MARKETS:
   r=[p for p in forecasts if p["season"]==season and p["market"]==market]
   d=paired_metrics(r)
   d["observed_player_game_rows"]=observed[(season,market)]
   d["sample_coverage"]=round(len(r)/observed[(season,market)],4) if observed[(season,market)] else None
   d["paired_game_bootstrap_ci95_lift"]=cluster_ci(r)
   cohorts[f"{season}_{market}"]=d
 return {"research_model":MODEL,"parameter_lock":CFG,"status":"FROZEN_SHADOW_NO_PROMOTION",
         "cohorts":cohorts,"total_scored":len(forecasts),
         "limitations":["2025 and 2026 outcomes were examined previously, not untouched holdouts",
           "Evaluation includes only recorded player-game rows; absent players ungraded, not zero",
           "No pregame injury statuses or confirmed starter snaps in historical dataset",
           "CFB production predictions are binary fixed-line classifiers; no equivalent numerical incumbent for paired MAE",
           "Passing TD expected count is NOT a calibrated distribution or a sportsbook P(over 1.5)",
           "Receiving opportunities use receptions, not targets; no invented route/target shares",
           "No line-specific odds or profitability is inferred; no bet or model promotion authorized"]}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--db",default="cfb_models/cfb_model.sqlite")
 p.add_argument("--board",default=None)
 p.add_argument("--out",default="docs/cfb_numeric_research_report.json")
 args=p.parse_args()
 conn=sqlite3.connect(f"file:{args.db}?mode=ro",uri=True)
 if args.board:
  board=json.loads(Path(args.board).read_text())
  target=(int(board["season"]),int(board["week"]))
 else:board=None;target=None
 rows,observed,state=walkforward(conn,target=target)
 doc=report(rows,observed)
 if board:
  h,th,oh,lg=state
  out=[]
  for b in board.get("picks",[]):
   market=b.get("market")
   if market not in MARKETS or not b.get("player_id"):continue
   player_id=str(b["player_id"])
   data=project(h[(player_id,market)],th[(board["season"],b["team"],market)],
        oh[(board["season"],b["opponent"],market)],lg[market],market)
   if data:
    out.append({"player":b.get("player"),"player_id":player_id,
      "team":b["team"],"opponent":b["opponent"],"game_id":b.get("game_id"),
      "market":market,"kickoff_utc":b.get("kickoff_utc"),**data})
  doc["upcoming_asof"]=board["generated_at_utc"]
  doc["upcoming_numeric_projections_RESEARCH_ONLY"]=out
  doc["upcoming_coverage"]={"shadow":len(out),"live_market_candidates":sum(
    b.get("market") in MARKETS for b in board.get("picks",[]))}
 conn.close()
 path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(doc,indent=2)+"\n")
 print(json.dumps({"model":MODEL,"status":doc["status"],"cohorts":doc["cohorts"],
                  "upcoming_coverage":doc.get("upcoming_coverage")},sort_keys=True))
if __name__=="__main__":main()
