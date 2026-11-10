#!/usr/bin/env python3
"""Strict numerical grade of ORIGINAL pregame Phase 1E vs frozen V2 forecasts.

Scientific diagnostic only; does not train, tune, promote, alter stored
predictions or infer missing players' actual yards to be zero.
Both predictions must refer to the exact game, team, player ID, outcome and
horizon and have the same immutable source snapshot content ID.

The report includes *every* matched observation and also counts every
unmatched/no-box-score forecast. A player absent from the box score is
UNRESOLVED, not silently counted as a win or a zero-yard performance.
"""
from __future__ import annotations
import argparse
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA="NFL_PHASE1E_ORIGINAL_LIVE_VS_FROZEN_V2_NUMERIC_DIAGNOSTIC_A"
MARKETS={"rec_yds":"receiving_yards","rush_yds":"rushing_yards"}
OPPORTUNITIES={"rec_yds":"targets","rush_yds":"carries"}
HOURS={"T24":24,"T90":1.5}


def utc(s):
    d=datetime.fromisoformat(str(s).replace("Z","+00:00"))
    if d.tzinfo is None:
        raise ValueError("MISSING_TIMEZONE")
    return d.astimezone(timezone.utc)


def rows(path):
    with Path(path).open(encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                yield json.loads(line)


def find_batches(root, season, weeks, horizon):
    folder=Path(root)
    return sorted(p for p in (folder/"forecasts"/"batches").glob(
        f"{season}_wk*_{horizon}_*.jsonl") if
        int(p.name.split("_")[1][2:]) in weeks)


def read_pair(first, second, horizon):
    a=list(rows(first));b=list(rows(second))
    p1=[r for r in a if r.get("outcome") in MARKETS]
    v2=[r for r in b if r.get("outcome") in MARKETS and
        r.get("eligibility")=="eligible"]
    if not p1 or not v2:
        raise ValueError("NO_PAIRABLE_FORECASTS")
    meta=next((r for r in a if "_batch" in r),None)
    if meta is None:
        raise ValueError("MISSING_ORIGINAL_BATCH_PROVENANCE")
    if meta["_batch"] != first.stem or first.name!=second.name:
        raise ValueError("BATCH_IDENTITY_MISMATCH")
    game=meta["game_id"]
    seen_a=set();seen_b=set()
    for source,items,seen in (("P1",p1,seen_a),("V2",v2,seen_b)):
        for r in items:
            if r.get("game_id")!=game or r.get("horizon")!=horizon:
                raise ValueError("FORECAST_GAME_HORIZON_MISMATCH_"+source)
            kick=utc(r["kickoff"]);cut=utc(r["cutoff"])
            expected=kick-timedelta(hours=HOURS[horizon])
            if abs((cut-expected).total_seconds())>1.1:
                raise ValueError("INVALID_ORIGINAL_CUTOFF_"+source)
            if source=="P1":
                snap=r["input_snapshots"]
                retrieved=utc(snap["retrieval_ts"])
                if snap.get("time_travel") is not False:
                    raise ValueError("NOT_A_GENUINE_LIVE_CAPTURE")
            else:
                retrieved=utc(r["as_of_retrieval_ts"])
            if retrieved>cut or retrieved>=kick:
                raise ValueError("SOURCE_RETRIEVED_AFTER_ORIGINAL_CUTOFF_"+source)
            if source=="P1" and utc(meta["generated_at"])>=kick:
                raise ValueError("PHASE1_FORECAST_GENERATED_AFTER_KICKOFF")
            if source=="V2" and r.get("kind")!="v2_forward_forecast":
                raise ValueError("NOT_FROZEN_V2_FORWARD_FORECAST")
            key=(r["team"],str(r["player_id"]),r["outcome"])
            if key in seen:
                raise ValueError("DUPLICATE_ORIGINAL_ID_"+source)
            seen.add(key)
    a_map={(x["team"],str(x["player_id"]),x["outcome"]):x for x in p1}
    b_map={(x["team"],str(x["player_id"]),x["outcome"]):x for x in v2}
    return game,a_map,b_map


def summarize(items):
    if not items:
        return {"n":0}
    n=len(items)
    def mae(key):
        return round(statistics.mean(abs(r[key]-r["actual"]) for r in items),3)
    def within(key,tol):
        return round(sum(abs(r[key]-r["actual"])<=tol for r in items)/n,4)
    s={"n":n,"games":len({r["game_id"] for r in items}),
       "phase1_median_mae":mae("phase1_median"),
       "phase1_mean_mae":mae("phase1_mean"),
       "frozen_v2_point_mae":mae("frozen_v2_point"),
       "phase1_vs_v2_median_mae_lift_yards":round(
           mae("frozen_v2_point")-mae("phase1_median"),3),
       "phase1_median_within_10":within("phase1_median",10),
       "frozen_v2_within_10":within("frozen_v2_point",10),
       "phase1_median_within_20":within("phase1_median",20),
       "frozen_v2_within_20":within("frozen_v2_point",20),
       "phase1_median_bias":round(statistics.mean(
           r["phase1_median"]-r["actual"] for r in items),3),
       "frozen_v2_bias":round(statistics.mean(
           r["frozen_v2_point"]-r["actual"] for r in items),3),
       "phase1_paired_strict_wins":sum(
           abs(r["phase1_median"]-r["actual"])<
           abs(r["frozen_v2_point"]-r["actual"]) for r in items),
       "v2_paired_strict_wins":sum(
           abs(r["phase1_median"]-r["actual"])>
           abs(r["frozen_v2_point"]-r["actual"]) for r in items),
       "tied_absolute_error":sum(
           abs(r["phase1_median"]-r["actual"])==
           abs(r["frozen_v2_point"]-r["actual"]) for r in items),
       "phase1_empirical_p10_p90_coverage":round(sum(
           r["phase1_p10"]<=r["actual"]<=r["phase1_p90"]
           for r in items)/n,4)}
    opp=[r for r in items if r.get("opportunity_error") is not None]
    if opp:
        s["opportunity_diagnostic"]={
            "n":len(opp),
            "projected_mean_opportunities":round(statistics.mean(
                r["phase1_expected_opportunities"] for r in opp),3),
            "actual_mean_opportunities":round(statistics.mean(
                r["actual_opportunities"] for r in opp),3),
            "actual_minus_projected_opportunity_bias":round(statistics.mean(
                r["opportunity_error"] for r in opp),3),
            "opportunity_mae":round(statistics.mean(
                abs(r["opportunity_error"]) for r in opp),3),
            "avg_yards_error_from_opportunity_component":round(statistics.mean(
                r["opportunity_yard_error_component"] for r in opp),3),
            "avg_yards_error_from_efficiency_component":round(statistics.mean(
                r["efficiency_yard_error_component"] for r in opp),3),
            "mean_abs_opportunity_yard_component":round(statistics.mean(
                abs(r["opportunity_yard_error_component"]) for r in opp),3),
            "mean_abs_efficiency_yard_component":round(statistics.mean(
                abs(r["efficiency_yard_error_component"]) for r in opp),3),
            "opportunity_component_dominant_rows":sum(abs(
                r["opportunity_yard_error_component"])>abs(
                r["efficiency_yard_error_component"]) for r in opp),
            "efficiency_component_dominant_rows":sum(abs(
                r["opportunity_yard_error_component"])<abs(
                r["efficiency_yard_error_component"]) for r in opp),
            "method":"Actual_yards - predicted_mean = "
                "(actual_events - expected_events) * predicted_mean_per_event "
                "+ (actual_yards - actual_events * predicted_mean_per_event). "
                "Descriptive accounting identity, NOT causal attribution; "
                "do not tune on these forward outcomes."
        }
    return s


def grade(state, db_path, *,season=2026,weeks=range(4,10),horizon="T90"):
    if horizon not in HOURS:
        raise ValueError("INVALID_FORECAST_HORIZON")
    con=sqlite3.connect(f"file:{Path(db_path).absolute()}?mode=ro",uri=True)
    con.row_factory=sqlite3.Row
    scored=[];excluded=Counter();detail=[]
    files=find_batches(state,season,set(weeks),horizon)
    if not files:
        raise ValueError("NO_FROZEN_FORWARD_GAMES_FOUND")
    try:
        for first in files:
            second=Path(state)/"v2forecasts"/"batches"/first.name
            if not second.exists():
                excluded["missing_v2_original_game_file"]+=1
                continue
            gid,a,b=read_pair(first,second,horizon)
            final=con.execute("SELECT home_team,away_team,home_score,away_score "
                "FROM games WHERE game_id=?",(gid,)).fetchall()
            if len(final)!=1 or final[0]["home_score"] is None or final[0]["away_score"] is None:
                excluded["game_not_final_with_verified_source"]+=1
                continue
            teams={final[0]["home_team"],final[0]["away_team"]}
            if len(teams)!=2:
                raise ValueError("BAD_FINAL_GAME_TEAMS")
            for key in sorted(a.keys()|b.keys()):
                p=a.get(key);q=b.get(key)
                if p is None or q is None:
                    excluded["not_in_same_predeclared_model_cohort"]+=1
                    continue
                team,pid,market=key
                if team not in teams or q["opponent"] not in teams or p["opponent"]!=q["opponent"]:
                    raise ValueError("WRONG_TEAM_MATCHED_FORECAST")
                if p["input_snapshots"]["content_id"]!=q["input_hashes"]["content_id"]:
                    raise ValueError("PHASE1_V2_USED_DIFFERENT_PREGAME_SNAPSHOT")
                if utc(p["kickoff"])!=utc(q["kickoff"]):
                    raise ValueError("PAIRED_KICKOFF_MISMATCH")
                col=MARKETS[market]
                opp_col=OPPORTUNITIES[market]
                actual=con.execute(f"SELECT {col}, {opp_col} FROM player_games WHERE "
                    "game_id=? AND team=? AND player_id=? AND season_type='REG'",
                    (gid,team,pid)).fetchall()
                if len(actual)!=1 or actual[0][0] is None:
                    excluded["unverified_player_participation_or_identity"]+=1
                    continue
                observed=float(actual[0][0])
                pred=float(p["median"]);mean=float(p["mean"])
                v2=float(q["v2_frozen_comparator"]["point"])
                p10=float(p["quantiles"]["p10"]);p90=float(p["quantiles"]["p90"])
                if not all(math.isfinite(v) for v in (observed,pred,mean,v2,p10,p90)):
                    raise ValueError("NONFINITE_FORECAST_OR_ACTUAL")
                expected=p.get("expected_opportunities")
                observed_events=actual[0][1]
                expected=(float(expected) if expected is not None else None)
                observed_events=(float(observed_events) if observed_events is not None else None)
                can_decompose=(expected is not None and expected>0 and
                    math.isfinite(expected) and observed_events is not None and
                    math.isfinite(observed_events) and observed_events>=0)
                implied_rate=(mean/expected if can_decompose else None)
                opp_error=(observed_events-expected if can_decompose else None)
                opp_yards=(opp_error*implied_rate if can_decompose else None)
                eff_yards=(observed-observed_events*implied_rate if can_decompose else None)
                if can_decompose and abs((opp_yards+eff_yards)-(observed-mean))>1e-6:
                    raise ValueError("INVALID_YARDAGE_DECOMPOSITION")
                result={"game_id":gid,"horizon":horizon,"team":team,
                    "opponent":p["opponent"],"player_id":pid,
                    "player":p["player_name"],"market":market,
                    "phase1_median":pred,"phase1_mean":mean,
                    "phase1_expected_opportunities":p.get("expected_opportunities"),
                    "phase1_p_active":p.get("p_active"),
                    "actual_opportunities":observed_events,
                    "opportunity_type":opp_col,
                    "opportunity_error":opp_error,
                    "implied_model_yards_per_opportunity":implied_rate,
                    "opportunity_yard_error_component":opp_yards,
                    "efficiency_yard_error_component":eff_yards,
                    "phase1_p10":p10,"phase1_p90":p90,
                    "frozen_v2_point":v2,"actual":observed,
                    "phase1_median_abs_error":round(abs(pred-observed),3),
                    "frozen_v2_abs_error":round(abs(v2-observed),3),
                    "original_snapshot_content_id":p["input_snapshots"]["content_id"],
                    "pregame_capture_utc":p["input_snapshots"]["retrieval_ts"]}
                scored.append(result)
            game_rows=[r for r in scored if r["game_id"]==gid]
            detail.append({"game_id":gid,"matched":len(game_rows),
                "metrics":summarize(game_rows)})
    finally:
        con.close()
    # Focus cohort is forecast-defined, not selected from observed outcomes.
    # Show all rows and both "star-level" and complete-board error.
    focus=[r for r in scored if max(r["phase1_median"],r["frozen_v2_point"])>=50]
    return {"schema":SCHEMA,"status":"RESEARCH_DIAGNOSTIC_NO_PROMOTION",
        "time_fidelity":"REAL_PRE_CUTOFF_RETRIEVAL_AND_PRE_KICKOFF_GENERATION",
        "season":season,"weeks":sorted(set(weeks)),"horizon":horizon,
        "raw_phase1_game_files":len(files),"matched_rows":len(scored),
        "excluded_counts":dict(excluded),
        "entire_paired_cohort":summarize(scored),
        "pregame_projected_50plus_cohort":summarize(focus),
        "by_market":{market:summarize([r for r in scored if r["market"]==market])
            for market in MARKETS},
        "top_opportunity_misses":sorted([r for r in scored if
            r.get("opportunity_error") is not None],
            key=lambda x:abs(x["opportunity_yard_error_component"]),reverse=True)[:30],
        "top_efficiency_misses":sorted([r for r in scored if
            r.get("opportunity_error") is not None],
            key=lambda x:abs(x["efficiency_yard_error_component"]),reverse=True)[:30],
        "game_by_game":detail,
        "complete_player_by_player":scored,
        "caveat":"Only uniquely verified player-game box-score outcomes are scored. "
          "Players absent from player_games are UNRESOLVED rather than zero; "
          "DNP accuracy and complete-population performance are NOT claimed. "
          "V2 is a frozen scientific comparator, not necessarily the changing live V2 artifact. "
          "Outcome-conditioned subsets are not a new pregame selection gate."}


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--shadow",required=True)
    a.add_argument("--db",required=True)
    a.add_argument("--season",type=int,default=2026)
    a.add_argument("--from-week",type=int,default=4)
    a.add_argument("--through-week",type=int,default=9)
    a.add_argument("--horizon",choices=tuple(HOURS),default="T90")
    a.add_argument("--out",required=True)
    p=a.parse_args()
    report=grade(p.shadow,p.db,season=p.season,
        weeks=range(p.from_week,p.through_week+1),horizon=p.horizon)
    Path(p.out).parent.mkdir(parents=True,exist_ok=True)
    Path(p.out).write_text(json.dumps(report,indent=2)+"\n")
    print("NFL_PHASE1_FROZEN_FORWARDS",json.dumps({k:report[k] for k in (
        "status","raw_phase1_game_files","matched_rows",
        "excluded_counts","entire_paired_cohort",
        "pregame_projected_50plus_cohort","by_market")}))
    for game in report["game_by_game"]:
        print("NFL_PHASE1_GAME",json.dumps(game))
    if not report["matched_rows"]:
        raise SystemExit("NO_COMPLETE_VERIFIED_PAIRED_OUTCOMES")


if __name__=="__main__":
    main()
