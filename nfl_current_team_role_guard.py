"""NFL game-day CURRENT TEAM participation proof — frozen model left intact.

Never use a decades-old player's final team to infer that player is active
for the current scheduled NFL matchup. Used after all source models and
before any official board, firstseen receipt or frontend publication.

Read ONLY completed PRIOR weeks from player_games. A prior game's actual
source row is participation evidence, NOT a verified starter, injury-active
or sportsbook hit probability source.

The frozen nfl_yardage_v2.py model and research manifest must remain
byte-for-byte untouched. This is a separate serving/publishing contract.
"""
from __future__ import annotations

from collections import Counter,defaultdict

from nfl_yardage_v2 import norm_name

MODEL_YARDS={"receiving_yards","rushing_yards","passing_yards"}
PLAYER_EVENTS=MODEL_YARDS|{"anytime_touchdowns","sacks"}
SOURCE_VERSION="NFL_PREGAME_CURRENT_TEAM_PARTICIPATION_V1"

def recent_team_supported(history,team,season,latest_team_week,min_team_games=2):
    if not history or latest_team_week is None:
        return False
    current=[g for g in history if
        g["season"]==season and g["team"]==team]
    recent=history[-1]
    return len(current)>=min_team_games and recent["season"]==season \
       and recent["team"]==team and recent["week"]==latest_team_week

def filter_current_team_picks(con,picks,season,week):
    """Return (copy of safely served picks, audit), never rewrite numeric centers.

    For names/IDs positively matched to nflverse player_games, known stale
    histories are rejected. Unknown identities are retained unverified,
    not presented as active or automatically assigned a probability.
    """
    if not isinstance(season,int) or not isinstance(week,int):
        raise ValueError("MODEL_SEASON_WEEK_REQUIRED")
    latest_team_week={}
    by_id=defaultdict(list)
    by_name_team=defaultdict(set)
    sql="""
        SELECT player_id,player_name,team,season,week
        FROM player_games
        WHERE season_type='REG' AND player_id IS NOT NULL
          AND (season<? OR (season=? AND week<?))
        ORDER BY season,week
    """
    for pid,name,team,s,w in con.execute(sql,(season,season,week)):
        r={"season":int(s),"week":int(w),"team":str(team)}
        by_id[str(pid)].append(r)
        by_name_team[(norm_name(name),str(team))].add(str(pid))
        if s==season:
            latest_team_week[str(team)]=max(latest_team_week.get(str(team),0),int(w))
    good=[]
    reasons=Counter()
    samples=[]
    verified=0
    unresolved=0
    for p in picks:
        market=str(p.get("market") or "")
        if market not in PLAYER_EVENTS:
            good.append(p)
            continue
        team=str(p.get("team") or "")
        pname=str(p.get("player") or "")
        pid=str(p.get("player_id") or "")
        history=by_id.get(pid)
        if not history:
            ids=by_name_team.get((norm_name(pname),team),set())
            if len(ids)==1:
                history=by_id[next(iter(ids))]
        if not history:
            unresolved+=1
            good.append({**p,"current_team_role_source":"UNVERIFIED_SOURCE_IDENTITY"})
            continue
        min_games=2 if market in MODEL_YARDS else 1
        supported=recent_team_supported(history,team,season,
                     latest_team_week.get(team),min_games)
        if supported:
            verified+=1
            good.append({**p,
                "current_team_role_source":"PRIOR_COMPLETED_TEAM_GAME",
                "starter_verified":False})
            continue
        last=history[-1]
        if last["season"]<season:
            reason="STALE_PRIOR_SEASON_PLAYER"
        elif last["team"]!=team:
            reason="FORMER_TEAM_PLAYER"
        elif last["week"]!=latest_team_week.get(team):
            reason="MISSED_LATEST_COMPLETED_TEAM_GAME"
        else:
            reason="INSUFFICIENT_CURRENT_TEAM_GAMES"
        reasons[(market,reason)]+=1
        if len(samples)<30:
            samples.append({"player":pname,"team":team,"market":market,
                "reason":reason,"last_source":last,
                "latest_team_week":latest_team_week.get(team)})
    audit={
        "version":SOURCE_VERSION,
        "season":season,"week":week,
        "input_rows":len(picks),"published_rows":len(good),
        "source_current_team_verified":verified,
        "source_identity_unresolved_retained_unverified":unresolved,
        "removed_count":len(picks)-len(good),
        "removed_by_market_reason":{
            f"{market}|{reason}":n for (market,reason),n in reasons.items()
        },
        "samples":samples,
        "no_model_centers_or_probabilities_changed":True,
        "no_current_game_result_used":True,
    }
    return good,audit
