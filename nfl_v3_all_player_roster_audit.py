"""Audit ALL named NFL Phase1 player projections against independently captured
current ESPN roster membership, including defenders and quarterbacks.

This is a READ-ONLY, pregame evidence receipt. A missing roster listing is
UNKNOWN, not verified OUT. A roster listing is NOT confirmed starting lineup.
Never rewrite original T24/T90, rank a bet, or silently discard a player.

Use after published shadow evidence and before actual kickoff. Different
horizon snapshots remain distinct, with source evidence timestamped at
capture time rather than falsely attributed to their older cutoff.
"""
import argparse
import json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import nfl_analyst_role_preflight as R

SCHEMA="NFL_V3_ALL_PLAYER_PRETIP_ROSTER_IDENTITY_AUDIT_V1"
FIELDS={"game_id","player","team","opp","pos","outcome"}
MARKETS={"rush_yds","rush_td","rec_yds","rec","rec_td","pass_yds",
         "pass_td","int","atd","tackles","sacks","def_int"}


def time_iso(v):
    d=datetime.fromisoformat(str(v).replace("Z","+00:00"))
    if d.tzinfo is None:raise ValueError("NFL_IDENTITY_NAIVE_TIMESTAMP")
    return d.astimezone(timezone.utc)


def build(shadow,*,now=None,roster_fetch=R.roster_evidence):
    clock=time_iso(now or datetime.now(timezone.utc))
    if (not isinstance(shadow,dict) or not isinstance(shadow.get("games"),list)
        or not isinstance(shadow.get("row_fields"),list)
        or not isinstance(shadow.get("forecasts"),dict)):
        raise ValueError("NFL_IDENTITY_SOURCE_MISSING_REQUIRED_STRUCTURE")
    ix={k:i for i,k in enumerate(shadow["row_fields"])}
    if not FIELDS.issubset(ix) or len(ix)!=len(shadow["row_fields"]):
        raise ValueError("NFL_IDENTITY_MISSING_OR_DUPLICATE_FORECAST_FIELDS")
    games={};teams=set();rows=defaultdict(list);unnamed=defaultdict(list)
    for g in shadow["games"]:
        gid=g.get("game_id")
        if not gid or gid in games:
            raise ValueError("NFL_IDENTITY_DUPLICATE_GAME_ID")
        kick=time_iso(g["kickoff_utc"])
        if kick<=clock:continue
        home,away=g["home"],g["away"]
        if not home or not away or home==away:
            raise ValueError("NFL_IDENTITY_INVALID_SCHEDULE")
        games[gid]=(home,away,kick)
        teams.update((home,away))
    for h in ("T24","T90"):
        ledger=shadow["forecasts"].get(h)
        if ledger is None:continue
        if not isinstance(ledger,list):
            raise ValueError("NFL_IDENTITY_FORECAST_ROWS_NOT_LIST")
        for row in ledger:
            if not isinstance(row,list) or len(row)!=len(ix):
                raise ValueError("NFL_IDENTITY_BAD_FORECAST_ROW_SHAPE")
            r={k:row[i] for k,i in ix.items()}
            gid=r["game_id"]
            if gid not in games:continue
            home,away,kick=games[gid]
            if r["team"] not in (home,away) or r["opp"] != (home if r["team"]==away else away):
                raise ValueError("NFL_IDENTITY_PLAYER_TEAM_DOES_NOT_MATCH_GAME")
            if r["outcome"] not in MARKETS:
                raise ValueError("NFL_IDENTITY_UNKNOWN_MODEL_MARKET")
            if not R.normalize(r["player"]):
                # Real shadow rows sometimes lack a player display name.
                # Those cannot be roster-certified; record every offending
                # original player MARKET as an explicit upstream defect,
                # without hiding the rest of the usable roster evidence.
                unnamed[(gid,h,r["team"])].append(r["outcome"])
                continue
            rows[(gid,h,r["team"],R.normalize(r["player"]))].append(r)
    timestamp=clock.isoformat().replace("+00:00","Z")
    rosters={}
    if teams:
        with ThreadPoolExecutor(max_workers=min(8,len(teams))) as pool:
            jobs={t:pool.submit(roster_fetch,t,timestamp) for t in sorted(teams)}
            for tm,future in jobs.items():
                try:state=future.result()
                except Exception:state={"status":"ROSTER_SOURCE_UNAVAILABLE","players":{}}
                if (not isinstance(state,dict) or
                    state.get("status") not in ("ROSTER_CAPTURED","ROSTER_SOURCE_UNAVAILABLE")):
                    raise ValueError("NFL_IDENTITY_INVALID_ROSTER_SOURCE_STATE")
                rosters[tm]=state
    summary=Counter();games_report=defaultdict(lambda:{
        "named_players":0,"source_unavailable":0,"unverified_roster_identity":0,
        "listed_nonstarter_proof":0,"explicit_injury_cautions":0,
        "listed_unavailable":0,"ambiguous_names":0,
        "sample_flags":[]
    })
    named=[]
    for gid,h,team,norm in sorted(rows):
        sources=rows[(gid,h,team,norm)]
        name=sources[0]["player"]
        if len({r["player"] for r in sources})>1:
            raise ValueError("NFL_IDENTITY_AMBIGUOUS_INTERNAL_PLAYER_NAMES")
        source=rosters[team]
        status=(R.roster_match(source,name)["status"] if source["status"]=="ROSTER_CAPTURED"
                else "ROSTER_SOURCE_UNAVAILABLE")
        bucket=games_report[(gid,h,team)]
        bucket["named_players"]+=1;summary[status]+=1
        if status=="ROSTER_SOURCE_UNAVAILABLE":
            bucket["source_unavailable"]+=1
        elif status=="ROSTER_IDENTITY_NOT_VERIFIED":
            bucket["unverified_roster_identity"]+=1
        elif status=="AMBIGUOUS_ROSTER_IDENTITY":
            bucket["ambiguous_names"]+=1
        elif status=="ROSTER_LISTED_NOT_STARTER_VERIFIED":
            bucket["listed_nonstarter_proof"]+=1
        elif status=="INJURY_CAUTION_NOT_CONFIRMED_ACTIVE":
            bucket["explicit_injury_cautions"]+=1
        elif status=="LISTED_UNAVAILABLE":
            bucket["listed_unavailable"]+=1
        else:
            raise ValueError("NFL_IDENTITY_UNRECOGNIZED_SOURCE_MATCH")
        if status!="ROSTER_LISTED_NOT_STARTER_VERIFIED":
            if len(bucket["sample_flags"])<15:
                bucket["sample_flags"].append({
                    "player":name,"status":status,
                    "markets":sorted(set(r["outcome"] for r in sources))
                })
        named.append((gid,h,team,norm,name,status))
    for (gid,h,team),markets in sorted(unnamed.items()):
        bucket=games_report[(gid,h,team)]
        bucket["unnamed_model_projection_rows"]=len(markets)
        bucket["unnamed_markets"]=dict(sorted(Counter(markets).items()))
    # A single player name may appear for both teams: flag but cannot
    # resolve shared names to one identity absent original GSIS IDs.
    collisions=[]
    by_game=defaultdict(lambda:defaultdict(set))
    for gid,h,tm,norm,name,status in named:
        by_game[(gid,h)][norm].add(tm)
    for (gid,h),by in by_game.items():
        for norm,ts in by.items():
            if len(ts)>1:
                collisions.append({"game_id":gid,"horizon":h,"name":norm,
                                   "teams":sorted(ts),"identity_verified":False})
    return {
        "schema":SCHEMA,
        "observed_at_utc":timestamp,
        "original_immutable_forecasts_unchanged":True,
        "snapshot_source_is_later_than_original_T24_when_applicable":True,
        "roster_presence_is_not_game_day_starter_certification":True,
        "roster_absence_is_not_verified_player_inactivity":True,
        "no_numerical_forecast_or_frontend_changes":True,
        "games_covered":len(games),
        "distinct_players_by_game_horizon":len(named),
        "unidentifiable_model_market_rows":sum(len(v) for v in unnamed.values()),
        "unidentifiable_model_rows_by_market":dict(sorted(Counter(
            m for markets in unnamed.values() for m in markets).items())),
        "source_status_by_team":{t:rosters[t]["status"] for t in sorted(rosters)},
        "roster_identity_status_counts":dict(sorted(summary.items())),
        "per_game_team_horizon":[
            {"game_id":g,"horizon":h,"team":t,**data}
            for (g,h,t),data in sorted(games_report.items())
        ],
        "ambiguous_same_name_across_opponents":collisions,
        "warning":"Only current ESPN roster evidence, not an original T24 source; cannot retrospectively edit prior forecast or certify Sunday starter/active status",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shadow",default="docs/nfl_phase1_shadow.json")
    p.add_argument("--out",required=True)
    a=p.parse_args()
    res=build(json.loads(Path(a.shadow).read_text()))
    out=Path(a.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(res,indent=2,sort_keys=True)+"\n")
    print("NFL_V3_ALL_PLAYER_ROSTER_VALIDATION",json.dumps({
        "games":res["games_covered"],
        "named_player_game_horizon_rows":res["distinct_players_by_game_horizon"],
        "status":res["roster_identity_status_counts"],
        "collisions":len(res["ambiguous_same_name_across_opponents"]),
        "unnamed_player_market_rows":res["unidentifiable_model_market_rows"],
        "no_model_change":True,
    },sort_keys=True))

if __name__=="__main__":
    main()
