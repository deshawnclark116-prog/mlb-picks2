#!/usr/bin/env python3
"""Experimental NUMERICAL grade of V7 on real 2026 W10 games.

Important: NOT a genuine prospective V7 capture or promotion test.
Uses an archived production-selected pregame roster evidence subset, not a
complete team roster. The subset is necessarily selection-biased. It cannot
establish that V7 beats the production model. All comparison populations,
exclusions, and non-evaluable cases are reported.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import nfl_v6_joint_opportunity_shadow as v6
import nfl_v7_roster_returner as v7
from nfl_v6_snap_role_evidence import read_snap_roles

SEASON, WEEK = 2026, 10
STAMP="2026-11-15T09:15:55Z"
SOURCE_COMMIT="03d5032be7abfb7900d608401d05b96948a0a7cf"
MARKETS={"rushing_yards","receiving_yards"}


def key(row):
    return (row["team"],str(row["player_id"]),row["market"])


def archived_rosters(snapshot, game):
    """Accept only observed, CURRENT-TEAM, pre-kickoff listed players."""
    if (snapshot.get("schema")!="nfl-role-evidence-coverage-v1"
        or snapshot.get("season")!=SEASON or snapshot.get("week")!=WEEK
        or v6.utc(snapshot["captured_at_utc"])>=v6.utc(game["kickoff_utc"])):
        raise ValueError("UNUSABLE_PRETIP_ROSTER_EVIDENCE")
    if snapshot.get("starter_status_verified") is not False:
        raise ValueError("UNEXPECTED_STARTER_CERTIFICATION")
    d={}
    for team, opp in ((game["home_team"],game["away_team"]),
                      (game["away_team"],game["home_team"])):
        if snapshot.get("roster_source_status",{}).get(team)!="ROSTER_CAPTURED":
            raise ValueError("ARCHIVED_ROSTER_SOURCE_NOT_CAPTURED")
        selected=defaultdict(list)
        for row in snapshot.get("player_market_evidence",[]):
            if row.get("team")!=team or row.get("opponent")!=opp or \
                row.get("kickoff_utc")!=game["kickoff_utc"]:
                continue
            # Reject missing IDs: name similarity alone is not identity proof.
            name=v7.normalized(row.get("player"))
            if not name or not row.get("player_id"):
                continue
            status=row.get("roster_status")
            if status not in ("ROSTER_LISTED_NOT_STARTER_VERIFIED",
                    "INJURY_CAUTION_NOT_CONFIRMED_ACTIVE"):
                continue
            selected[name].append({"name":row["player"],"status":(
                "ROSTER_LISTED_CAUTION_NOT_CONFIRMED_ACTIVE"
                if status=="INJURY_CAUTION_NOT_CONFIRMED_ACTIVE"
                else "ROSTER_LISTED_NOT_CONFIRMED_ACTIVE")})
        # A player can be in both rushing and receiving markets:
        # dedup identical same-name evidence, reject conflicting flags.
        unique={}
        for name,rec in selected.items():
            statuses={v["status"] for v in rec}
            if len(statuses)==1:
                unique[name]=[rec[0]]
        d[team]={"team":team,"status":"ROSTER_CAPTURED",
                 "items":len(unique),"players":unique,
                 "source":"ARCHIVED_PRETIP_PUBLISHED_PICK_ROSTER_SUBSET",
                 "observed_at_utc":snapshot["captured_at_utc"],
                 "no_game_day_active_certification":True}
        if len(unique)<2:
            raise ValueError("INSUFFICIENT_PARTIAL_ROSTER")
    return d


def stat_truth(con, gid):
    rows=defaultdict(list)
    for p in con.execute("""SELECT player_id,team,receiving_yards,rushing_yards
      FROM player_games WHERE game_id=? AND season_type='REG'""",(gid,)):
        rows[(str(p[0]),p[1])].append({
            "receiving_yards":p[2],"rushing_yards":p[3]})
    return rows


def grades(rows):
    if not rows:
        return {"n":0,"v7_mae":None,"v6b_mae":None,"incumbent_mae":None}
    n=len(rows)
    d={"n":n}
    for m in ("v7","v6b","incumbent"):
        d[m+"_mae"]=round(sum(abs(x[m]-x["actual"]) for x in rows)/n,3)
        d[m+"_within_10"]=round(sum(abs(x[m]-x["actual"])<=10 for x in rows)/n,4)
    d["by_market"]={}
    for market in sorted(MARKETS):
        part=[x for x in rows if x["market"]==market]
        if part:
            d["by_market"][market]={m+"_mae":round(
              sum(abs(x[m]-x["actual"]) for x in part)/len(part),3)
              for m in ("v7","v6b","incumbent")}
            d["by_market"][market]["n"]=len(part)
    return d


def run(db, roster_path, production_path, v6dir, snap_files,
        *, sims=10000, max_games=0):
    if sims<10000:
        raise ValueError("TEN_THOUSAND_DRAWS_MINIMUM")
    raw=Path(roster_path).read_bytes()
    snapshot=json.loads(raw)
    published=json.loads(Path(production_path).read_text())
    if snapshot["captured_at_utc"]>"2026-11-15T09:15:55Z":
        raise ValueError("SOURCE_AFTER_FROZEN_GIT_COMMIT")
    if v6.utc(published["generated_at_utc"])>v6.utc(snapshot["captured_at_utc"]):
        raise ValueError("PUBLISHED_PRODUCTION_AFTER_ARCHIVED_ROSTER")
    if (published["season"],published["week"])!=(SEASON,WEEK):
        raise ValueError("INCUMBENT_SEASON_WEEK_MISMATCH")
    forecasts=defaultdict(list)
    for row in published["picks"]:
        if row.get("market") in MARKETS and row.get("projected_median") is not None:
            forecasts[key(row)].append(row)
    con=sqlite3.connect("file:"+str(Path(db).absolute())+"?mode=ro",uri=True)
    con.row_factory=sqlite3.Row
    excluded=Counter()
    paired_ab=[];triple=[]
    all_model_rows=[]
    games_out=[]
    prior_games=[p for p in sorted(Path(v6dir).glob("2026_10_*.json"))
        if p.is_file()]
    if not prior_games:
        raise ValueError("NO_GENUINE_FROZEN_V6B_FIRSTSEEN")
    for path in prior_games:
        original=json.loads(path.read_text())
        gid=original["game_id"]
        if original["status"]!="RESEARCH_ONLY_NEVER_AN_OFFICIAL_BET" or (
             v6.utc(original["source_snapshot_utc"]) >=
             v6.utc(original["kickoff_utc"])):
            raise ValueError("INVALID_FROZEN_ORIGINAL_V6B")
        row=con.execute("""SELECT game_id,season,week,home_team,away_team,
                        kickoff_utc,home_score,away_score FROM games
                        WHERE game_id=?""",(gid,)).fetchone()
        if row is None:
            excluded["game_not_in_actual_db"]+=1
            continue
        g=dict(row)
        if g["home_score"] is None or g["away_score"] is None:
            excluded["game_not_final"]+=1
            continue
        if original["kickoff_utc"]!=g["kickoff_utc"]:
            excluded["frozen_kickoff_differs_from_db"]+=1
            continue
        try:
            rost=archived_rosters(snapshot,g)
        except ValueError as e:
            excluded["roster_"+str(e)]+=1
            continue
        cutoff=min(v6.utc(snapshot["captured_at_utc"]),
                   v6.utc(g["kickoff_utc"])-__import__("datetime").timedelta(hours=2))
        old=v6.asof_rows(con,cutoff)
        by_team, hist=v6.game_history(old)
        league=v6.league_rates(by_team)
        source=read_snap_roles(snap_files,db,cutoff)
        seed=v6.SEED+int(hashlib.sha256(gid.encode()).hexdigest()[:8],16)
        sim=v7.simulate_game(g,by_team,hist,league,rost,
                            sims=sims,seed=seed,role_source=source)
        orig={(r["team"],str(r["player_id"]),r["market"]):r
              for r in original["projections"]}
        truth=stat_truth(con,gid)
        case={"game_id":gid,"v7_count":0,"actual_verified":0,
            "paired_v6b":0,"paired_all":0,
            "partial_roster_items":{t:rost[t]["items"] for t in rost},
            "returners":sum(r.get("returner_added_to_player_universe",False)
                for r in sim["projections"])}
        for r in sim["projections"]:
            if r["market"] not in MARKETS:
                continue
            case["v7_count"]+=1
            k=key(r); t=truth.get((str(r["player_id"]),r["team"]),[])
            if len(t)!=1 or t[0][r["market"]] is None:
                excluded["missing_or_ambiguous_stat_actual"]+=1
                continue
            actual=float(t[0][r["market"]])
            if not math.isfinite(actual):
                excluded["nonfinite_stat_actual"]+=1
                continue
            case["actual_verified"]+=1
            val=float(r["projected_median"])
            prior=orig.get(k)
            live=forecasts.get(k,[])
            rec={"game_id":gid,"team":r["team"],"player_id":str(r["player_id"]),
                "player":r["player"],"market":r["market"],
                "actual":actual,"v7":val,
                "v7_absolute_error":round(abs(val-actual),2),
                "is_returner":bool(r.get("returner_added_to_player_universe")),
                "archived_roster_evidence_is_production_selected":True}
            all_model_rows.append(rec)
            if prior is None:
                excluded["v6b_original_not_on_new_cohort"]+=1
                continue
            rec["v6b"]=float(prior["projected_median"])
            paired_ab.append(rec)
            case["paired_v6b"]+=1
            if len(live)!=1:
                excluded["production_numeric_no_exact_player_id"]+=1
                continue
            if (live[0].get("kickoff_utc")!=g["kickoff_utc"] or
                live[0].get("season")!=SEASON or
                live[0].get("week")!=WEEK):
                excluded["production_wrong_game"]+=1
                continue
            rec["incumbent"]=float(live[0]["projected_median"])
            triple.append(rec)
            case["paired_all"]+=1
        games_out.append(case)
        if max_games and len(games_out)>=max_games:
            break
    con.close()
    def ab_summary(x):
        if not x:return {"n":0}
        n=len(x)
        return {"n":n,"v7_mae":round(sum(abs(a["v7"]-a["actual"]) for a in x)/n,3),
                "v6b_mae":round(sum(abs(a["v6b"]-a["actual"]) for a in x)/n,3),
                "v7_within_10":round(sum(abs(a["v7"]-a["actual"])<=10 for a in x)/n,3),
                "v6b_within_10":round(sum(abs(a["v6b"]-a["actual"])<=10 for a in x)/n,3)}
    return {"schema":"NFL_V7_ARCHIVED_PARTIAL_ROSTER_RETROSPECTIVE_ONLY_A",
        "validation_status":"EXPLORATORY_BIASED_ROSTER_DIAGNOSTIC_NOT_PROSPECTIVE",
        "eligible_to_promote":False,
        "firstseen_v7":False,
        "archived_source_commit":SOURCE_COMMIT,
        "roster_capture":snapshot["captured_at_utc"],
        "roster_sha256":hashlib.sha256(raw).hexdigest(),
        "production_capture":published["generated_at_utc"],
        "historical_source_publication_asof_verified":False,
        "v7_roster_incomplete":True,
        "same_information_timestamp_between_models":False,
        "full_unselected_player_universe_not_available":True,
        "game_count":len(games_out),"games":games_out,
        "original_v7_numeric_count":len(all_model_rows),
        "original_v7_rows":all_model_rows,
        "paired_v6b_count":len(paired_ab),
        "paired_v6b_metrics":ab_summary(paired_ab),
        "paired_three_way_metrics":grades(triple),
        "paired_three_way_rows":triple,
        "exclusions":dict(excluded)}


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--db",required=True)
    ap.add_argument("--roster",required=True)
    ap.add_argument("--production",required=True)
    ap.add_argument("--v6b-dir",default="docs/nfl_v6b_firstseen")
    ap.add_argument("--snap-files",nargs="+",required=True)
    ap.add_argument("--sims",type=int,default=10000)
    ap.add_argument("--max-games",type=int,default=0)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    report=run(args.db,args.roster,args.production,args.v6b_dir,
               args.snap_files,sims=args.sims,max_games=args.max_games)
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+"\n")
    print("NFL_V7_REAL_WEEK10_PARTIAL_ROSTER_DIAGNOSTIC",json.dumps({
        "games":report["game_count"],
        "original":report["original_v7_numeric_count"],
        "paired_V6B":report["paired_v6b_metrics"],
        "paired_3way":report["paired_three_way_metrics"],
        "exclusions":report["exclusions"],
        "nonpromotion":report["validation_status"]}))
