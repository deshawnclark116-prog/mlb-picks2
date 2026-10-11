#!/usr/bin/env python3
"""Read-only NHL SOG board safety gate.

No arbitrary balancing: do not publish research-only 2.5 lines as sportsbook
picks. Preserve raw forecasts for audit, expose actionable picks only with
verified per-player bookmaker lines and non-conflicting model provenance.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

def audit(board):
    picks=board.get("picks",[])
    groups=defaultdict(list)
    for p in picks:
        if p.get("market") not in ("shots_on_goal","shots_on_goal_early_season"):
            continue
        key=(str(p.get("game_id")),str(p.get("player_id")))
        groups[key].append(p)
    findings=Counter()
    rows=[]
    for (game,player),candidates in groups.items():
        sides={"UNDER" if str(x.get("pick","")).startswith("UNDER") else "OVER"
               if str(x.get("pick","")).startswith("OVER") else "UNKNOWN" for x in candidates}
        if len(candidates)>1: findings["duplicate_player_game"]+=1
        if len(sides)>1: findings["conflicting_side"]+=1
        verified=[p for p in candidates if p.get("sportsbook_line_verified") is True
                  and p.get("bookmaker") and p.get("line_source_timestamp_utc")
                  and p.get("line") is not None and p.get("active_roster_verified") is True
                  and p.get("player_status_verified") is True]
        if not verified: findings["unverified_line_or_availability"]+=1
        rows.append({"game_id":game,"player_id":player,"candidate_count":len(candidates),
                     "sides":sorted(sides),"verified_candidates":len(verified),
                     "status":"RESEARCH_ONLY" if not verified or len(sides)>1 else "NEEDS_MODEL_VALIDATION"})
    return {"schema":"NHL_SOG_SERVING_SAFETY_AUDIT_V1","total_raw_sog_entries":sum(map(len,groups.values())),
            "unique_player_games":len(groups),"findings":dict(findings),
            "actionable_verified_picks":0,
            "reason":"This audit never certifies model calibration, odds or betting value.",
            "players":rows}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--board",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    data=audit(json.loads(args.board.read_text()))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in data.items() if k!="players"},sort_keys=True))

if __name__=="__main__":main()
