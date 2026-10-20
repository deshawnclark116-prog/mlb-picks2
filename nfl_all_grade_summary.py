#!/usr/bin/env python3
"""Small public NFL grade index. Never calculate grades from hindsight.

Merges independently recorded source assessments:
* first-seen production predictions (numeric stat accuracy, any original side)
* immutable experimental V2 T24/T90 and separate late-but-pregame forecasts
* independent completed-game and original-forecast coverage alarm

No model-derived 65% milestone is a real sportsbook wager. The raw per-player
documents stay available separately, rather than being hidden behind a
one-number hit rate.
"""
import argparse
import json
from pathlib import Path

SCHEMA="NFL_ALL_ORIGINAL_PREDICTIONS_PER_GAME_SUMMARY_V1"

def build(production,v2,coverage):
    if production.get("schema")!="NFL_FIRST_SEEN_ALL_SERVED_PREDICTIONS_GRADE_V1":
        raise ValueError("Unknown production original grade source")
    if v2.get("schema")!="NFL_V2_FIRST_SEEN_REAL_FINAL_NUMERIC_SCORECARD_V1":
        raise ValueError("Unknown V2 original grade source")
    if coverage.get("schema")!="NFL_V2_POSTGAME_GRADING_COVERAGE_V1":
        raise ValueError("Unknown completed-game coverage source")
    if not v2.get("every_genuine_original_forecast_accounted") or not production.get(
        "production_vs_research_separate"):
        raise ValueError("Prediction identity or source integrity not certified")
    if production["original_count"]!=production["accounted_for"] or (
        production["verified_outcome_count"]+production["unverified_count"]!=
        production["original_count"]):
        raise ValueError("Production first-seen predictions omitted")
    if (v2["total_original_pregame_forecasts"]!=
        v2["graded_player_outcomes"]+v2["unverified_original_forecasts"]):
        raise ValueError("V2 original forecast missing grade or unknown")

    bygame={}
    for row in production["by_game"]:
        gid=row["game_id"]
        if gid in bygame and "serving" in bygame[gid]:
            raise ValueError("Duplicate production game")
        bygame.setdefault(gid,{"game_id":gid})["serving"]={
            "original_predictions":row["original_predictions"],
            "verified_actuals":row["graded"],"unverified_actuals":row["unverified"],
            "by_market":row["by_market"],
            "note":"An originally numeric prediction has STAT ERROR but no artificial win or loss"}
    for row in v2["by_game_horizon"]:
        gid=row["game_id"]
        bygame.setdefault(gid,{"game_id":gid})
        cohorts=bygame[gid].setdefault("v2_cohorts",{})
        h=row["horizon"]
        if h in cohorts:raise ValueError("Duplicate scientific game-horizon")
        cohorts[h]={
            "original_model_stat_predictions":row["original_forecasts"],
            "actuals_verified":row["graded"],"actuals_unverified":row["unverified"],
            "markets":row["markets"],
            "original_horizon":row.get("original_horizon",h.removesuffix("_LATE")),
            "strict_horizon_qualified":not h.endswith("_LATE"),
            "note":("Original immutable forecasts made before kickoff, but AFTER strict horizon "
                    "cutoff; NOT eligible for certified on-time T24/T90 results")
                   if h.endswith("_LATE") else
                   "Original immutable on-time model forecast; NOT a bookmaker bet"}

    finals={}
    for game in coverage.get("recent_completed_games") or []:
        gid=game["game_id"]
        finals[gid]={
            "home":game["home"],"away":game["away"],
            "final_home_score":game["home_score"],"final_away_score":game["away_score"],
            "independently_completed":True}
        bygame.setdefault(gid,{"game_id":gid})["final"]=finals[gid]

    for game in bygame.values():
        game["season"]=int(game["game_id"].split("_")[0]) if game["game_id"].startswith(
            "20") else None
        game["week"]=int(game["game_id"].split("_")[1]) if game["game_id"].count("_")>=2 and (
            game["game_id"].split("_")[1]).isdigit() else None

    production_count=sum(x["serving"]["original_predictions"] for x in bygame.values()
                         if "serving" in x)
    scientific_count=sum(r["original_model_stat_predictions"] for x in bygame.values()
                         for r in x.get("v2_cohorts",{}).values())
    if production_count!=production["original_count"] or (
        scientific_count!=v2["total_original_pregame_forecasts"]):
        raise ValueError("Per-game original forecast counts do not equal source totals")
    return {
        "schema":SCHEMA,"created_from_source_utc":coverage["asof_utc"],
        "scope":"EVERY_SAVED_ORIGINAL_NFL_GAME_PREDICTION_NOT_JUST_BET_CARDS",
        "game_count":len(bygame),"recent_completed_game_count":len(finals),
        "production":{
            "original_saved":production["original_count"],
            "actuals_verified":production["verified_outcome_count"],
            "actuals_unverified":production["unverified_count"],
            "original_recorded_direction_wins":production["directional_wins"],
            "original_recorded_direction_losses":production["directional_losses"],
            "original_recorded_direction_pushes":production["directional_pushes"],
            "unverified_reasons":production["status_counts"]},
        "original_v2":{
            "original_saved":v2["total_original_pregame_forecasts"],
            "actuals_verified":v2["graded_player_outcomes"],
            "actuals_unverified":v2["unverified_original_forecasts"],
            "late_but_pregame_originals_not_on_time_T24_T90":
                v2.get("late_original_pregame_forecasts",0),
            "on_time_horizon_projection_errors":v2.get("by_horizon"),
            "late_horizon_projection_errors":v2.get("by_late_horizon"),
            "actual_missing_reasons":v2.get("exclusions")},
        "completed_games_audit":{
            "status":coverage["status"],
            "completed_games_with_published_source_checks":
                coverage["recent_completed_game_count"],
            "archived_game_horizons_missing_from_grade":
                coverage["recent_frozen_game_horizons_ungraded"],
            "late_cutoff_original_pregame_rows":
                coverage.get("recent_late_cutoff_pregame_original_predictions",0),
            "original_forecast_batches_missing":
                coverage["recent_game_horizons_missing_original_forecast"],
            "player_outcomes_unverified":
                coverage.get("recent_unverified_original_predictions",0)},
        "games":[bygame[k] for k in sorted(bygame)],
        "interpretation":"Numeric stat forecasts: compare projected yards/count with actual, not arbitrary win/loss. Original directional forecasts: W/L/push at originally saved thresholds. T24/T90: research only; late snapshot cohort is never falsely counted as on-time. No certified gambling ROI.",
        "raw_sources":{
            "production":"nfl_all_original_predictions_grade.json",
            "v2":"nfl_v2_firstseen_real_stat_backtest.json",
            "coverage":"nfl_v2_postgame_grade_status.json"}}

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--production",required=True,type=Path)
    a.add_argument("--v2",required=True,type=Path)
    a.add_argument("--coverage",required=True,type=Path)
    a.add_argument("--out",required=True,type=Path)
    x=a.parse_args()
    doc=build(json.loads(x.production.read_text()),
              json.loads(x.v2.read_text()),json.loads(x.coverage.read_text()))
    x.out.parent.mkdir(parents=True,exist_ok=True)
    x.out.write_text(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("NFL_ALL_GAME_GRADE_SUMMARY",json.dumps({
        "game_count":doc["game_count"],
        "serving":doc["production"],
        "v2_original_saved":doc["original_v2"]["original_saved"],
        "v2_verified":doc["original_v2"]["actuals_verified"],
        "v2_late_original":doc["original_v2"]["late_but_pregame_originals_not_on_time_T24_T90"],
        "grade_gaps":doc["completed_games_audit"]["archived_game_horizons_missing_from_grade"]
    },sort_keys=True))

if __name__=="__main__":main()
