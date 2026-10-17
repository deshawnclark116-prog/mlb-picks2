"""Automatic every-game NFL V2 QUARTERBACK ROLE contradiction audit.

A red flag is NOT a verified starter and not an improved prediction.
The immutable forecast has conditional/split QB workload failure risks:
* no quarterback receives even 50% model play probability, or
* 2+ quarterbacks each receive a meaningful nonzero passing-yard median.
Do not silently promote the largest number to starter. Only verified
game/team role evidence can license a separate full-event simulation.
"""
from __future__ import annotations
import argparse,json,math
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

SCHEMA="NFL_V2_ALL_GAMES_QB_ROLE_CONTRADICTIONS_V1"


def parse_time(s):
    x=datetime.fromisoformat(str(s).replace("Z","+00:00"))
    if x.tzinfo is None:raise ValueError("TIMESTAMP_NO_UTC")
    return x.astimezone(timezone.utc)


def inspect(board,*,now):
    if board.get("schema")!="nfl-new-engine-shadow-v1" or board.get(
            "status",{}).get("publication_state")!="PUBLISHED":
        raise ValueError("UNVERIFIED_OR_UNPUBLISHED_V2_BOARD")
    fields=board["row_fields"]
    if len(fields)!=len(set(fields)):
        raise ValueError("DUPLICATE_PLAYER_STAT_FIELD")
    needed=["game_id","player","team","pos","outcome","p_active","median"]
    if not set(needed).issubset(fields):raise ValueError("MISSING_PLAYER_FIELDS")
    col={k:fields.index(k) for k in needed}
    games={g["game_id"]:g for g in board["games"]}
    reports=[];n_checked=0
    for horizon in ("T24","T90"):
        grouped=defaultdict(list)
        for r in board["forecasts"].get(horizon,[]):
            gid=r[col["game_id"]]
            if gid not in games or horizon not in games[gid].get("horizons",{}):continue
            if parse_time(games[gid]["kickoff_utc"])<=now:continue
            if r[col["pos"]]!="QB" or r[col["outcome"]]!="pass_yds":continue
            p=r[col["p_active"]];med=r[col["median"]]
            if not all(isinstance(x,(float,int)) and math.isfinite(x) for x in (p,med)):
                raise ValueError("NONFINITE_OR_NONNUMERIC_QB_FORECAST")
            if not 0<=p<=1 or med<0:raise ValueError("INVALID_QB_AVAILABILITY_OR_YARDS")
            grouped[(gid,r[col["team"]])].append({
                "player":r[col["player"]],"p_active":p,"pass_yds_median":med})
        for (gid,team),qbs in sorted(grouped.items()):
            n_checked+=1
            qbs.sort(key=lambda x:x["p_active"],reverse=True)
            flags=[]
            if max(x["p_active"] for x in qbs)<0.5:
                flags.append("NO_QB_WITH_MAJORITY_PLAY_PROBABILITY")
            high=[x for x in qbs if x["pass_yds_median"]>=10]
            if len(high)>1:
                flags.append("MULTIPLE_QBS_PROJECTED_FOR_SUBSTANTIVE_PASSING")
            if flags:
                reports.append({
                    "game_id":gid,"team":team,"horizon":horizon,
                    "kickoff_utc":games[gid]["kickoff_utc"],
                    "flags":flags,"quarterbacks":qbs,
                    "automatic_starter_selection":False,
                    "reason":"Source-certified starter role required before treating player medians as actionable",
                    "source_evidence_type":"PUBLISHED_ORIGINAL_FORECAST_ONLY_NOT_CONFIRMED_DEPTH_CHART"
                })
    return {"schema":SCHEMA,"generated_at_utc":now.isoformat(),
            "board_source":"docs/nfl_phase1_shadow.json",
            "research_only":True,"original_forecasts_not_changed":True,
            "all_nfl_games_scanned":True,"official_bets_created":0,
            "teams_horizons_with_qb_projections":n_checked,
            "flagged_team_horizons":len(reports),
            "detection_rule":"No QB with p_active >=.50 OR two QB passing medians >=10 yards; operational heuristic, not calibrated probabilities",
            "role_conflicts":reports}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--board",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=inspect(json.loads(a.board.read_text()),now=datetime.now(timezone.utc))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("NFL_V2_QB_ROLE_CONTRADICTIONS",
          json.dumps({"checked":result["teams_horizons_with_qb_projections"],
                      "flagged":result["flagged_team_horizons"],
                      "details":[{"game":r["game_id"],"team":r["team"],
                                  "h":r["horizon"],"flags":r["flags"]}
                                 for r in result["role_conflicts"]]},sort_keys=True))


if __name__=="__main__":main()
