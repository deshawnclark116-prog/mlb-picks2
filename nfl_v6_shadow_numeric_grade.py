#!/usr/bin/env python3
"""Forward-safe NFL V6 shadow numerical grader: exact game/team/player match.

Grading reads frozen V6 predictions, never rebuilds them from final stats.
All unmatched and unplayed identities are explicit. This is NOT a champion
comparison and must not be construed as an official betting track record.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

OUTCOME_COL = {"receiving_yards":"receiving_yards",
               "rushing_yards":"rushing_yards",
               "passing_yards":"passing_yards"}


def grade(forecasts, database):
    if forecasts.get("version")!="NFL_V6_JOINT_OPPORTUNITY_SHADOW_A" or \
       forecasts.get("status")!="RESEARCH_SHADOW_NOT_PROMOTED":
        raise ValueError("not a frozen, labeled NFL V6 shadow artifact")
    connection=sqlite3.connect("file:"+str(Path(database).absolute())+"?mode=ro",uri=True)
    connection.row_factory=sqlite3.Row
    graded,unresolved,by_market=[],[],defaultdict(list)
    for game in forecasts["games"]:
        result=connection.execute(
          "SELECT home_score,away_score FROM games WHERE game_id=?",
          (game["game_id"],)).fetchone()
        final=result is not None and result["home_score"] is not None \
             and result["away_score"] is not None
        for p in game["projections"]:
            identity={"game_id":game["game_id"],"team":p["team"],
              "player_id":p["player_id"],"player":p["player"],
              "market":p["market"]}
            if not final:
                unresolved.append({**identity,"reason":"GAME_NOT_FINAL"})
                continue
            col=OUTCOME_COL.get(p["market"])
            if col is None:
                raise ValueError("unexpected NFL V6 market")
            actual=connection.execute(
              "SELECT "+col+" FROM player_games WHERE game_id=? AND team=? AND player_id=? AND season_type='REG'",
              (game["game_id"],p["team"],p["player_id"])).fetchall()
            if len(actual)!=1 or actual[0][0] is None:
                unresolved.append({**identity,"reason":"NO_VERIFIED_PARTICIPATION_OR_IDENTITY"})
                continue
            x=float(actual[0][0]);pred=float(p["projected_mean"])
            base=p.get("naive_recent3_mean")
            if base is None or not math.isfinite(float(base)):
                raise ValueError("missing as-of baseline")
            baseline=float(base)
            record={**identity,"actual":x,"v6_prediction":pred,
              "naive_prior3_prediction":baseline,
              "v6_absolute_error":round(abs(x-pred),2),
              "naive_absolute_error":round(abs(x-baseline),2),
              "v6_within_10":abs(x-pred)<=10,
              "prob_reach_mean":p["prob_reach_rounded_mean"],
              "reached_mean":x>=p["rounded_mean"],
              "prob_within_15":p["prob_within_15_yards_of_rounded_mean"],
              "within_15":abs(x-p["rounded_mean"])<=15}
            graded.append(record)
            by_market[p["market"]].append(record)
    connection.close()
    def stats(rows):
        if not rows:
            return {"n":0}
        n=len(rows)
        return {"n":n,
            "v6_mae":round(sum(r["v6_absolute_error"] for r in rows)/n,3),
            "naive_prior3_mae":round(sum(r["naive_absolute_error"] for r in rows)/n,3),
            "v6_within_10_rate":round(sum(r["v6_within_10"] for r in rows)/n,4),
            "v6_bias":round(sum(r["v6_prediction"]-r["actual"] for r in rows)/n,3),
            "reach_mean_brier":round(sum((r["prob_reach_mean"]-r["reached_mean"])**2
                                    for r in rows)/n,4),
            "within_15_brier":round(sum((r["prob_within_15"]-r["within_15"])**2
                                    for r in rows)/n,4)}
    return {"schema":"NFL_V6_SHADOW_GRADE_A",
      "status":"NOT_PROMOTABLE_NO_SAME_COHORT_CHAMPION_COMPARISON",
      "source_version":forecasts["version"],
      "season":forecasts["season"],"week":forecasts["week"],
      "predicted_rows":len(graded)+len(unresolved),
      "matched":len(graded),"unmatched_or_pending":len(unresolved),
      "all_matched":stats(graded),
      "markets":{k:stats(v) for k,v in sorted(by_market.items())},
      "worst_numeric_misses":sorted(graded,key=lambda r:-r["v6_absolute_error"])[:25],
      "unresolved":unresolved}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--forecast",required=True)
    p.add_argument("--db",default="nfl_models/nfl_model.sqlite")
    p.add_argument("--out",default="/tmp/nfl_v6_shadow_grade.json")
    args=p.parse_args()
    report=grade(json.loads(Path(args.forecast).read_text()),args.db)
    Path(args.out).write_text(json.dumps(report,indent=2)+"\n")
    print("NFL_V6_SHADOW_GRADE",json.dumps({k:report[k] for k in
      ("status","predicted_rows","matched","unmatched_or_pending","all_matched","markets")}))
