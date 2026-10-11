#!/usr/bin/env python3
"""Pregame NFL analyst evidence: verified roster signals + observed usage.

NOT a probabilistic model, depth-chart source, starter inference or betting
recommendation. Every timestamp, roster source outage and sampling unknown is
explicit. Same week stats are NEVER read for a future forecast; player-game
absence must NOT be silently treated as a zero-opportunity performance.
"""
import argparse
import json
import re
import sqlite3
import unicodedata
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

import requests

ESPN_ROSTER = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team}/roster"
TEAM_ESPN = {"LA":"lar","WAS":"wsh"}
UNAVAILABLE={"out","injured reserve","injuredreserve","suspended","suspension",
             "physically unable to perform","pup","non-football injury",
             "reserve/retired","did not play"}
CAUTION={"questionable","doubtful","limited","day-to-day"}

def normalize(s):
    s=unicodedata.normalize("NFKD",str(s or ""))
    s="".join(x for x in s if not unicodedata.combining(x))
    s=re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b","",s.lower())
    return " ".join(re.sub("[^a-z ]"," ",s).split())


def roster_evidence(team, checked, http_get=requests.get):
    """Read official athlete roster listing, not an inferred depth chart."""
    try:
        r=http_get(ESPN_ROSTER.format(team=TEAM_ESPN.get(team,team.lower())),timeout=15)
        r.raise_for_status()
        data=r.json()
        athletes=defaultdict(list)
        for group in data.get("athletes",[]):
            grouping=str(group.get("position") or "")
            for item in group.get("items") or []:
                fullname=str(item.get("displayName") or item.get("fullName") or "")
                if not normalize(fullname):
                    continue
                statuses={str(i.get("status") or "").strip().lower()
                          for i in item.get("injuries",[]) if isinstance(i,dict)}
                # Athlete primary status can be nested. Unknown strings never
                # become a positive active/starting certification.
                state=str(item.get("status") or "").strip().lower()
                if state:
                    statuses.add(state)
                if (statuses & UNAVAILABLE or
                    grouping.lower() in ("injuredreserveorout","suspended")):
                    availability="LISTED_UNAVAILABLE"
                elif statuses & CAUTION:
                    availability="INJURY_CAUTION_NOT_CONFIRMED_ACTIVE"
                else:
                    availability="ROSTER_LISTED_NOT_STARTER_VERIFIED"
                athletes[normalize(fullname)].append({
                    "player":fullname,"team":team,
                    "status":availability,
                    "raw_reported_statuses":sorted(statuses),
                    "source_group":grouping,
                    "checked_at_utc":checked,
                    "provider_effective_at_utc":None,
                    "has_verified_depth_chart_starter_designation":False,
                })
        return {"status":"ROSTER_CAPTURED","source":ESPN_ROSTER.format(
            team=TEAM_ESPN.get(team,team.lower())),
            "checked_at_utc":checked,"players":dict(athletes)}
    except (requests.RequestException,ValueError,TypeError,KeyError,AttributeError) as exc:
        return {"status":"ROSTER_SOURCE_UNAVAILABLE","source":ESPN_ROSTER.format(
            team=TEAM_ESPN.get(team,team.lower())),"checked_at_utc":checked,
            "players":{},"error_type":type(exc).__name__}


def roster_match(snapshot, player):
    options=snapshot.get("players",{}).get(normalize(player),[])
    if len(options)==1:
        return options[0]
    if len(options)>1:
        return {"status":"AMBIGUOUS_ROSTER_IDENTITY","has_verified_depth_chart_starter_designation":False}
    return {"status":"ROSTER_IDENTITY_NOT_VERIFIED","has_verified_depth_chart_starter_designation":False}


def observed_usage(con,*,season,week,team,player,market):
    """Return explicit known games/unknown appearances and conditional shares.

    An absent row is NOT a confirmed DNP, not zero attempts and not a zero
    yardage outcome; roster/snap identity must be independently verified.
    """
    metric={"rushing_yards":"carries","receiving_yards":"targets"}[market]
    fixtures=con.execute("""
        SELECT game_id,season,week FROM games
        WHERE home_team=? OR away_team=?
        ORDER BY season,week
    """,(team,team)).fetchall()
    prior=[g for g in fixtures if (g[1],g[2])<(season,week)][-6:]
    records=con.execute("""
        SELECT player_id,player_name,game_id,season,week,
               carries,targets,rushing_yards,receiving_yards
        FROM player_games
        WHERE team=? AND season_type='REG'
          AND (season<? OR (season=? AND week<?))
    """,(team,season,season,week)).fetchall()
    aliases=defaultdict(list)
    for r in records:
        if normalize(r[1])==normalize(player):
            aliases[str(r[0])].append(r)
    if len(aliases)>1:
        return {"status":"AMBIGUOUS_NFLVERSE_PLAYER_ID","recent_games":[],
                "has_verified_starter_role":False}
    by_game={r[2]:r for rr in aliases.values() for r in rr}
    totals={}
    for gid,s,w in prior:
        t=con.execute(f"""SELECT COALESCE(SUM({metric}),0)
                           FROM player_games
                           WHERE game_id=? AND team=? AND season_type='REG'""",
                      (gid,team)).fetchone()[0]
        totals[gid]=float(t or 0)
    recent=[]
    for gid,s,w in prior[-3:]:
        r=by_game.get(gid)
        att=float(r[5 if market=="rushing_yards" else 6] or 0) if r else None
        team_att=totals.get(gid,0)
        recent.append({"season":s,"week":w,"game_id":gid,
                       "stat_row_observed":r is not None,
                       "opportunities_when_observed":att,
                       "team_observed_opportunities":team_att,
                       "share_when_observed":(att/team_att if att is not None and team_att>0 else None)})
    observed=[r for r in recent if r["stat_row_observed"]]
    shares=[r["share_when_observed"] for r in observed if r["share_when_observed"] is not None]
    prior3=[]
    for gid,s,w in prior[-6:-3]:
        r=by_game.get(gid)
        if r:
            prior3.append(float(r[5 if market=="rushing_yards" else 6] or 0))
    opportunities=sum(r["opportunities_when_observed"] for r in observed)
    current_avg=opportunities/len(observed) if observed else None
    old_avg=sum(prior3)/len(prior3) if len(prior3)==3 else None
    trend=(current_avg/old_avg if current_avg is not None and old_avg and
           len(observed)==3 and len(prior3)==3 else None)
    if len(prior)<3 or len(observed)<3:
        verdict="ROLE_PARTICIPATION_NOT_ESTABLISHED"
    elif current_avg is None or current_avg<=0:
        verdict="NO_RECENT_OPPORTUNITY"
    elif current_avg < (5.0 if market=="rushing_yards" else 3.0):
        # Mirrors the V2 volume eligibility floors. This is merely an
        # evidence warning, never a calibrated confidence or bet gate.
        verdict="LOW_RECENT_OPPORTUNITY_NO_ROLE_CERTAINTY"
    elif trend and trend>=1.35:
        verdict="RECENT_OPPORTUNITIES_EXPANDING"
    elif trend and trend<=0.75:
        verdict="RECENT_OPPORTUNITIES_DECLINING"
    else:
        verdict="RECENT_OPPORTUNITIES_OBSERVED_NO_STARTER_PROOF"
    return {
        "status":verdict,
        "recent_games":recent,
        "observed_stat_rows":len(observed),
        "scheduled_prior_games":len(recent),
        "unknown_stat_appearances":len(recent)-len(observed),
        "opportunities_per_observed_game":round(current_avg,2) if current_avg is not None else None,
        "prior_three_opportunities_per_observed_game":round(old_avg,2) if old_avg is not None else None,
        "share_per_observed_game":round(sum(shares)/len(shares),4) if shares else None,
        "recent_volume_ratio":round(trend,3) if trend is not None else None,
        "has_verified_starter_role":False,
        "source":"nflverse completed player_games and games; lagged season/week",
    }


def build(con,forecast,home,away,clock,fetch=requests.get):
    schedule=[g for g in forecast.get("scheduled_games",[]) if
              g.get("home_team")==home and g.get("away_team")==away]
    if len(schedule)!=1:
        raise ValueError("EXACT_SCHEDULED_MATCH_REQUIRED")
    kickoff=datetime.fromisoformat(schedule[0]["kickoff_utc"].replace("Z","+00:00"))
    generated=datetime.fromisoformat(forecast["generated_at_utc"].replace("Z","+00:00"))
    if generated>=kickoff:
        raise ValueError("PREDICTIONS_NOT_PREGAME")
    if clock>=kickoff:
        raise ValueError("GAME_ALREADY_STARTED_NO_NEW_PREGAME_CAPTURE")
    checked=clock.isoformat().replace("+00:00","Z")
    roster={team:roster_evidence(team,checked,fetch) for team in (home,away)}
    selections=[]
    for pick in forecast.get("picks",[]):
        if pick.get("team") not in (home,away) or pick.get("opponent") not in (home,away):
            continue
        market=pick.get("market")
        if market not in ("rushing_yards","receiving_yards"):
            continue
        if pick.get("projected_median") is None:
            continue
        team=pick["team"]; player=pick["player"]
        usage=observed_usage(con,season=forecast["season"],week=forecast["week"],
                             team=team,player=player,market=market)
        rs=roster_match(roster[team],player)
        if roster[team]["status"]!="ROSTER_CAPTURED":
            rs={"status":"ROSTER_SOURCE_UNAVAILABLE",
                "has_verified_depth_chart_starter_designation":False}
        flags=[]
        if rs["status"]!="ROSTER_LISTED_NOT_STARTER_VERIFIED":
            flags.append("CURRENT_AVAILABILITY_NOT_VERIFIED_OR_RISK")
        if usage["status"] in ("ROLE_PARTICIPATION_NOT_ESTABLISHED",
                               "AMBIGUOUS_NFLVERSE_PLAYER_ID","NO_RECENT_OPPORTUNITY",
                               "LOW_RECENT_OPPORTUNITY_NO_ROLE_CERTAINTY"):
            flags.append("ROLE_OPPORTUNITY_NOT_ESTABLISHED")
        if (pick.get("confidence")=="HIGH" and
            (usage.get("observed_stat_rows",0)<3 or
             pick.get("games_played",0)<4)):
            flags.append("HIGH_STABILITY_IS_NOT_PROVEN_ROLE")
        selections.append({
            "player":player,"team":team,"opponent":pick["opponent"],"market":market,
            "published_model_median":pick["projected_median"],
            "published_stability_grade":pick.get("confidence"),
            "published_history_n":pick.get("games_played"),
            "roster_evidence":rs,"usage_evidence":usage,
            "warnings":flags,
            "eligible_for_automatic_official_bet":False,
        })
    return {"status":"PREGAME_ROLE_EVIDENCE_RESEARCH_ONLY",
            "season":forecast["season"],"week":forecast["week"],
            "matchup":{"away":away,"home":home,"kickoff_utc":schedule[0]["kickoff_utc"]},
            "checked_at_utc":checked,"forecast_generated_at":forecast["generated_at_utc"],
            "roster_source_status":{k:v["status"] for k,v in roster.items()},
            "sources_have_verified_starting_lineups":False,
            "no_role_status_is_sportsbook_probability":True,
            "player_market_evidence":selections,
            "warnings_total":sum(bool(x["warnings"]) for x in selections)}


def main():
    ap=argparse.ArgumentParser()
    for key in ("db","forecast","home","away","out"):
        ap.add_argument("--"+key,required=True)
    args=ap.parse_args()
    doc=json.loads(Path(args.forecast).read_text())
    con=sqlite3.connect(f"file:{args.db}?mode=ro",uri=True)
    res=build(con,doc,args.home,args.away,datetime.now(timezone.utc))
    con.close()
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(res,indent=2)+"\n")
    print(json.dumps({"status":res["status"],"matchup":res["matchup"],
          "rosters":res["roster_source_status"],
          "players":len(res["player_market_evidence"]),
          "players_with_risks":res["warnings_total"],
          "detail":[{"player":r["player"],"market":r["market"],
                     "availability":r["roster_evidence"]["status"],
                     "history":r["usage_evidence"]["status"],
                     "recent_opportunities":r["usage_evidence"].get("opportunities_per_observed_game"),
                     "recent_team_share":r["usage_evidence"].get("share_per_observed_game"),
                     "warnings":r["warnings"]} for r in res["player_market_evidence"]]},
       sort_keys=True))
if __name__=="__main__":
    main()
