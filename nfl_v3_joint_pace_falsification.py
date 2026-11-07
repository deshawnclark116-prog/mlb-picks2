"""Falsify (rather than presume) the NFL V3 shared game-pace hypothesis.

As-of baseline: each team's average of its prior 4 *finished same-season*
offensive play PROXY totals, derived from nflverse player-game carries plus
passing attempts. Current game's final stats are OUTCOMES only. QB sacks
aren't included because this sqlite source lacks offensive sack/dropback
labels: this is NOT a validated exact dropback or team-play reconstruction.

The coupled research simulator proposes positive residual cross-team pace
correlation. Test the sign and estimated 95% game-cluster CI in 2024 (design)
then 2025 (development validation); reserve 2026 as a separate inspection
cohort and never retrofit a failed 2025 result to it.

No production model inputs, source lines, weather, odds or forecasts changed.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA="NFL_V3_EMPIRICAL_JOINT_PACE_HYPOTHESIS_AUDIT_V1"
LOOKBACK=4
MIN_GAMES=20


def _t(value):
    dt=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if dt.tzinfo is None:raise ValueError("NFL_V3_PACE_SOURCE_NAIVE_KICKOFF")
    return dt.astimezone(timezone.utc)


def _corr(pairs):
    if len(pairs)<3:return None
    xs=[p[0] for p in pairs];ys=[p[1] for p in pairs]
    xm=sum(xs)/len(xs);ym=sum(ys)/len(ys)
    den=math.sqrt(sum((x-xm)**2 for x in xs)*sum((y-ym)**2 for y in ys))
    return sum((x-xm)*(y-ym) for x,y in pairs)/den if den>1e-12 else None


def _ci(vals, seed=20261107, bootstrap=700):
    if len(vals)<MIN_GAMES:return None
    gen=random.Random(seed)
    n=len(vals); out=[]
    for _ in range(bootstrap):
        sampled=[vals[gen.randrange(n)] for _ in range(n)]
        val=_corr(sampled)
        if val is not None:out.append(val)
    if len(out)<bootstrap*.95:return None
    out.sort()
    return [round(out[int(.025*(len(out)-1))],4),
            round(out[int(.975*(len(out)-1))],4)]


def _games(con, asof):
    # This DB has exactly one player row per player/season/week/season_type.
    # The season/season_type and finished-game identity are checked
    # separately; current-week outcomes cannot be looked up prematurely.
    rows=con.execute("""
        SELECT g.game_id,g.season,g.week,g.home_team,g.away_team,
               g.kickoff_utc,g.home_score,g.away_score,
               p.team,
               SUM(COALESCE(p.carries,0)) as runs,
               SUM(COALESCE(p.attempts,0)) as passes,
               COUNT(DISTINCT p.player_id) as player_rows
        FROM games g JOIN player_games p ON p.game_id=g.game_id
            AND p.season=g.season AND p.week=g.week
        WHERE g.season_type='REG' AND p.season_type='REG'
          AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        GROUP BY g.game_id,p.team
        ORDER BY g.season,g.week,g.game_id,p.team
    """).fetchall()
    groups=defaultdict(dict)
    for gid,s,w,home,away,kick,home_score,away_score,team,car,att,n in rows:
        ts=_t(kick)
        if ts>=asof:continue
        if team not in (home,away):
            # A reused game ID or stale player-team join cannot prove volume.
            raise ValueError("NFL_V3_PACE_PLAYER_TEAM_NOT_IN_SCHEDULE")
        if not 15<=float(car or 0)+float(att or 0)<=110 or n<8:
            continue
        if float(car or 0)<0 or float(att or 0)<0:
            raise ValueError("NFL_V3_PACE_INVALID_ACTUAL_OPPORTUNITY")
        key=(int(s),int(w),gid,ts,home,away,int(home_score),int(away_score))
        if team in groups[key]:
            raise ValueError("NFL_V3_PACE_DUPLICATE_TEAM_GAME")
        groups[key][team]=(float(car),float(att))
    result=[]
    for k,by in sorted(groups.items(),key=lambda x:(x[0][3],x[0][2])):
        s,w,gid,kick,h,a,hs,as_=k
        if h in by and a in by:
            result.append((s,w,gid,kick,h,a,hs,as_,by[h],by[a]))
    return result


def empirical(con, asof):
    now=_t(asof)
    historic=defaultdict(list)
    buckets=defaultdict(list)
    inspected=defaultdict(int)
    for season,week,gid,kick,home,away,hs,as_,h,a in _games(con,now):
        inspected[season]+=1
        priors={}
        for tm in (home,away):
            prev=[p for p in historic[(season,tm)] if p[0]<kick]
            if len(prev)>=LOOKBACK:
                priors[tm]=sum(p[1] for p in prev[-LOOKBACK:])/LOOKBACK
        # A final score is used only for SECONDARY validation labels.
        # It never enters the model's pregame volume or game-script input.
        if len(priors)==2:
            htot=sum(h);atot=sum(a)
            hr=math.log(htot/priors[home]);ar=math.log(atot/priors[away])
            rush_delta=h[0]/htot-a[0]/atot
            winner_delta=(rush_delta if hs>as_ else
                          -rush_delta if as_>hs else None)
            buckets[season].append({
                "game_id":gid,"home":home,"away":away,
                "home_log_pace_residual":hr,
                "away_log_pace_residual":ar,
                "winner_minus_loser_rush_share":winner_delta,
            })
        # Append THIS outcome only AFTER calculating both pregame priors.
        for tm,actual in ((home,h),(away,a)):
            historic[(season,tm)].append((kick,sum(actual)))
    result={}
    for season,items in sorted(buckets.items()):
        points=[(x["home_log_pace_residual"],x["away_log_pace_residual"])
                for x in items]
        samples=[x["winner_minus_loser_rush_share"] for x in items
                 if x["winner_minus_loser_rush_share"] is not None]
        raw=_corr(points)
        ci=_ci(points,20261107+season)
        result[str(season)]={
            "all_completed_source_games":inspected[season],
            "paired_games_with_four_prior_team_games":len(items),
            "pace_residual_correlation":round(raw,4) if raw is not None else None,
            "game_cluster_bootstrap_95pct_interval":ci,
            "mean_winner_minus_loser_rush_share":(
                round(sum(samples)/len(samples),4) if samples else None),
            "score_used_only_for_postgame_label":True,
            "hypothesis_positive_common_tempo_supported":(
                bool(ci and ci[0]>0)),
            "team_game_experience_after_play_used_only_subsequently":True,
        }
    return {
        "schema":SCHEMA,
        "method":"trailing 4 completed same-season team carries+pass attempts versus current game final; equal game unit",
        "source_limitations":[
            "Not exact plays or dropbacks: no offensive sack/scramble disambiguation in this player_games table",
            "Current completion data may have been revised retrospectively; not a frozen original pretip raw-source archive",
            "Same-game script is a statistical hypothesis, not known winner or real score margin before kickoff"
        ],
        "asof_utc":now.isoformat(),
        "fixed_prior_team_games":LOOKBACK,
        "development_seasons":[2024,2025],
        "inspection_cohort_2026_not_promotion_evidence":True,
        "positive_common_pace_claim_requires_2025_CI_lower_above_zero":True,
        "by_season":result,
        "production_promotion_allowed":False,
        "model_changed":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--db",required=True)
    p.add_argument("--asof",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    con=sqlite3.connect(f"file:{args.db}?mode=ro",uri=True)
    try:r=empirical(con,args.asof)
    finally:con.close()
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    print("NFL_V3_PACE_HYPOTHESIS_AUDIT",json.dumps(r["by_season"],sort_keys=True))


if __name__=="__main__":
    main()
