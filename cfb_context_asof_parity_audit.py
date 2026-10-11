#!/usr/bin/env python3
"""CFB research: falsify target-game hindsight in the CURRENT context model.

Precommitted research-only feature protocol:
- ALL player volume/role inputs originate in games from weeks strictly
  before the target week. A multi-game week is held out atomically.
- Never infer a teammate is absent by inspecting the target game's actual
  boxscore; "vacated_*" and "qb_change" remain UNKNOWN (None) without a
  timestamped pregame roster/depth-chart source.
- Do not reuse finalized-season ratings as though they were pregame ratings.
  Team ratings for (season,week) come from the rating map indexed as of
  *before* that week. Current league strength cannot see final scores later
  in its own target week.
- No odds, calibrated probabilities, official picks, selection thresholds,
  confidence labels or deployed model weights are changed here.
"""
from __future__ import annotations
import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MODEL = "CFB_CONTEXT_ASOF_PARITY_AUDIT_V1"
FEATURES = ("carry_share_last3","rec_share_last3","pass_share_last3",
            "vacated_car","vacated_rec","qb_change","team_plays_last3")
UNAVAILABLE_WITHOUT_PREGAME_EVIDENCE = ("vacated_car","vacated_rec","qb_change")
NO_PROMOTION = "RESEARCH_ONLY_NO_MODEL_PROMOTION"


def roster_role_unknown():
    return {"vacated_car":None,"vacated_rec":None,"qb_change":None}


class PriorWeekUsage:
    """Snapshot of real player/team observations; target week NEVER a feature."""
    def __init__(self, con):
        data=con.execute("""
            SELECT season,week,game_id,game_date,team,player_id,
                   COALESCE(carries,0),COALESCE(receptions,0),COALESCE(pass_attempts,0)
            FROM player_games
            WHERE season IS NOT NULL AND week IS NOT NULL AND
                  team IS NOT NULL AND game_id IS NOT NULL
            ORDER BY season,week,game_date,game_id,team,player_id
        """).fetchall()
        teams=defaultdict(dict)
        for season,week,gid,gdate,team,pid,car,rec,pas in data:
            if pid is None:continue
            entry=teams[(int(season),team)].setdefault((int(week),str(gid)), {})
            entry[str(pid)]=(float(car or 0),float(rec or 0),float(pas or 0))
        self.teams={key:sorted(games.items(),key=lambda z:z[0])
                    for key,games in teams.items()}

    def features(self,season,week,team,player_id):
        prior=[entries for (w,gid),entries in self.teams.get((season,team),())
               if w<week][-3:]
        if not prior:
            return {**roster_role_unknown(),
                    "carry_share_last3":None,"rec_share_last3":None,
                    "pass_share_last3":None,"team_plays_last3":None}
        totals=[sum(x[k] for entries in prior for x in entries.values())
                for k in range(3)]
        player=[sum(entries.get(str(player_id),(0.,0.,0.))[k]
                    for entries in prior) for k in range(3)]
        return {
            "carry_share_last3":player[0]/totals[0] if totals[0]>0 else None,
            "rec_share_last3":player[1]/totals[1] if totals[1]>0 else None,
            "pass_share_last3":player[2]/totals[2] if totals[2]>0 else None,
            "team_plays_last3":(totals[0]+totals[2])/len(prior),
            **roster_role_unknown(),
        }


def checked_asof_rating(ratings,season,week,team,opp):
    """Do not use build_ratings.latest: that includes post-target-week games."""
    own=ratings.get((season,team,week))
    away=ratings.get((season,opp,week))
    return {"team_rating":own,"opp_rating":away,
            "rating_diff":own-away if own is not None and away is not None else None}


def audit(con, *, season=2025, max_examples=12):
    import cfb_context_v2 as legacy
    old=legacy.build_context(con)
    usage=PriorWeekUsage(con)
    ratings=legacy.build_ratings(con)
    rows=con.execute("""
        SELECT season,week,game_id,team,player_id FROM player_games
        WHERE season=? AND player_id IS NOT NULL AND team IS NOT NULL
        ORDER BY week,game_id,team,player_id
    """,(season,)).fetchall()
    seen=set();total=0;old_known=0;future_used=Counter()
    current_mismatch=Counter();examples=[];first_weeks=0
    for ss,ww,gid,team,pid in rows:
        key=(ss,pid,ww)
        if key in seen:continue
        seen.add(key)
        past=usage.features(ss,ww,team,pid)
        original=old.get(key)
        if original is None:continue
        total+=1
        if past["team_plays_last3"] is None:first_weeks+=1
        for flag in UNAVAILABLE_WITHOUT_PREGAME_EVIDENCE:
            value=original.get(flag)
            if isinstance(value,(int,float)) and math.isfinite(value) and value!=0:
                future_used[flag]+=1
                if len(examples)<max_examples:
                    examples.append({"week":ww,"team":team,"player_id":str(pid),
                        "leaked_feature":flag,"observed_target_game_value":round(float(value),4),
                        "asof_value":"UNKNOWN_PRE_GAME"})
        for f in ("carry_share_last3","rec_share_last3","pass_share_last3","team_plays_last3"):
            old_v,new_v=original.get(f),past.get(f)
            if old_v is not None and new_v is not None and abs(old_v-new_v)>1e-7:
                current_mismatch[f]+=1
        old_known+=int(all(original.get(k) is not None for k in FEATURES))
    leak_any=set((e["week"],e["team"],e["player_id"]) for e in examples)
    return {
        "audit":MODEL,"season":season,
        "status":NO_PROMOTION,"number_of_evaluated_historical_player_weeks":total,
        "future_game_absence_data_may_enter_training":True,
        "number_of_nonzero_target_game_derived_features":dict(future_used),
        "asof_safe_values_for_unverified_pregame_absence":roster_role_unknown(),
        "other_feature_parity_mismatches":dict(current_mismatch),
        "historical_observations_without_prior_team_game":first_weeks,
        "examples_first_n":examples,
        "disclosure":[
            "Current cfb_context_v2.build_context sees target-game player identities and usage to build vacated_car/vacated_rec/qb_change during historical training",
            "Live cfb_context_v2.ServingContext sets these same variables to 0.0 without verifying an injury, inactive or actual role change",
            "Historical target-game values are NOT available in real pregame forecasts and must not contribute to a validated training feature",
            "This audit reports structural train/serve contamination; it does not establish that this defect alone caused Saturday's moneyline losses",
            "No source-certified pregame injury/active roster or player-specific sportsbook lines in the current context model",
            "2025 and 2026 results have already been consulted; a model trained or tuned in response cannot claim an untouched 2026 holdout",
            "The LOCKED production model is unchanged; only a new leak-free candidate trained and evaluated before future contests can replace it"
        ],
        "zero_predictive_code_changes":True,
        "rated_teams_found_asof":len(ratings),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--db",required=True)
    p.add_argument("--season",type=int,default=2025)
    p.add_argument("--out")
    opts=p.parse_args()
    con=sqlite3.connect(f"file:{opts.db}?mode=ro",uri=True)
    try:doc=audit(con,season=opts.season)
    finally:con.close()
    if opts.out:
        target=Path(opts.out);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(doc,indent=2)+"\n")
    print(json.dumps({k:v for k,v in doc.items() if k in
        ("season","status","number_of_evaluated_historical_player_weeks",
         "number_of_nonzero_target_game_derived_features",
         "other_feature_parity_mismatches","historical_observations_without_prior_team_game")}))
if __name__=="__main__":
    main()
