#!/usr/bin/env python3
"""Read-only, forward-only NHL V2 shot-count miss investigation.

Uses ORIGINAL immutable pregame forecasts and official boxscore-reconciled
outcomes. No retraining, side selection, retrospective lineup features,
selection of optimal thresholds, or promotion. All diagnostics after the
fact are explicitly separated from prospective predictor inputs.

This computes within-game uncertainty and paired T30 vs T90 for the exact
same players; naive player-row bootstrap would seriously overstate certainty.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from nhl_v2_firstseen_forward_score import first_pregame_pairs

SCHEMA = "NHL_V2_FIRST_PREGAME_COUNT_MISS_FORENSICS_V1"
HORIZONS = ("T24H", "T90", "T30")


def safe_num(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _game_ci(rows, *, k, seed=20261011, simulations=2000):
    """Non-parametric game-cluster bootstrap; row-level resampling prohibited."""
    by_game = defaultdict(list)
    for p in rows:
        by_game[str(p["game_id"])].append(p[k])
    games = sorted(by_game)
    if len(games) < 8:
        return {"available": False, "games": len(games),
                "reason": "fewer than 8 distinct games; no numeric confidence interval"}
    by_game_mean = [mean(by_game[game]) for game in games]
    gen = random.Random(seed)
    estimates = sorted(mean(by_game_mean[gen.randrange(len(games))]
                            for _ in games) for _ in range(simulations))
    return {"available": True, "games": len(games), "unit": "game",
            "ci95": [round(estimates[49], 5), round(estimates[1949], 5)]}


def _summary(rows):
    if not rows:
        return {"n": 0, "status": "NO_MATCHED_OR_CERTIFIED_ROWS"}
    diffs = [x["forecast_mean"] - x["actual_shots"] for x in rows]
    return {
        "n": len(rows),
        "distinct_games": len({x["game_id"] for x in rows}),
        "played_count": sum(x["played"] for x in rows),
        "predicted_mean_shots": round(mean(x["forecast_mean"] for x in rows), 5),
        "actual_mean_shots": round(mean(x["actual_shots"] for x in rows), 5),
        "bias_predicted_minus_actual": round(mean(diffs), 5),
        "mae": round(mean(abs(d) for d in diffs), 5),
        "within_one_shot": round(mean(abs(d) <= 1 for d in diffs), 5),
        "large_miss_more_than_three": sum(abs(d) > 3 for d in diffs),
    }


def _obs(f, g):
    receipt = f.get("receipt") or {}
    usage = receipt.get("recent_participation") or {}
    toi = receipt.get("toi_seconds") or {}
    observed = g.get("diagnostics_postgame_only") or {}
    recent = toi.get("recent3")
    actual = observed.get("actual_toi_seconds")
    if recent is not None and not safe_num(recent):
        raise ValueError("non-numeric original pregame recent TOI")
    if actual is not None and not safe_num(actual):
        raise ValueError("non-numeric postgame actual TOI")
    ratio = (float(actual)/float(recent) if g["played"]
             and safe_num(actual) and safe_num(recent) and recent > 0
             else None)
    days = usage.get("team_games_since_appearance")
    if days is not None and not safe_num(days):
        raise ValueError("invalid as-of games-since-appearance")
    if g["played"] is False:
        deployment = "DID_NOT_PLAY_POSTGAME"
    elif ratio is None:
        deployment = "PLAYED_TOI_COMPARISON_UNAVAILABLE"
    elif ratio < .75:
        deployment = "ACTUAL_TOI_LESS_THAN_75PCT_OF_RECENT3"
    elif ratio > 1.25:
        deployment = "ACTUAL_TOI_MORE_THAN_125PCT_OF_RECENT3"
    else:
        deployment = "ACTUAL_TOI_75_TO_125PCT_OF_RECENT3"
    return {
        "game_id": str(f["game_id"]),
        "game_start_utc": f["scheduled_start"],
        "forecast_horizon": f["forecast_horizon"],
        "player_id": str(f["player_id"]),
        "player_name": receipt.get("player_name"),
        "team": f.get("team"),
        "position": receipt.get("position") or "UNKNOWN",
        "model_version": f["engine_version"],
        "forecast_mean": float(f["expected_sog"]),
        "actual_shots": int(g["actual_sog"]),
        "played": bool(g["played"]),
        "expected_meaningful_participant": f.get("meaningful_expected_participant") is True,
        "original_availability_state": f.get("availability_state") or "NOT_CAPTURED",
        "original_availability_certified": f.get("availability_confidence") == "CERTIFIED",
        "recent3_toi_seconds_pregame": recent,
        "actual_toi_seconds_POSTGAME_ONLY": actual,
        "actual_vs_recent_toi_ratio_POSTGAME_ONLY": ratio,
        "postgame_deployment_bucket_DIAGNOSTIC_ONLY": deployment,
        "pregame_team_games_since_appearance": days,
        "pregame_recent_shots_rate_per60": receipt.get("sog_per60_app10"),
        "preexisting_baseline_prior10": (f.get("comparators") or {}).get("simple_prior10_mean"),
        "preexisting_human_comparator": (f.get("comparators") or {}).get("human_frozen_mean"),
        "original_row_hash": f["row_hash"],
    }


def _paired_baseline(rows, field):
    valid = [r for r in rows if safe_num(r.get(field)) and r[field] >= 0]
    if not valid:
        return {"n": 0, "status": "COMPARATOR_MISSING"}
    by_game = [{
        "game_id": r["game_id"],
        "delta": abs(r[field] - r["actual_shots"]) -
                 abs(r["forecast_mean"] - r["actual_shots"])}
        for r in valid]
    gains = [r["delta"] for r in by_game]
    return {
        "n": len(valid),
        "games": len({r["game_id"] for r in valid}),
        "model_mae_same_rows": round(mean(abs(r["forecast_mean"]-r["actual_shots"])
                                          for r in valid), 5),
        "baseline_mae_same_rows": round(mean(abs(r[field]-r["actual_shots"])
                                             for r in valid), 5),
        "mae_gain_positive_favors_model": round(mean(gains), 5),
        "game_block_bootstrap_gain": _game_ci(by_game, k="delta"),
    }


def _reconcile_horizons(rows):
    by_identity = defaultdict(dict)
    for r in rows:
        key = (r["game_id"], r["game_start_utc"], r["player_id"])
        if r["forecast_horizon"] in by_identity[key]:
            raise ValueError("duplicated frozen player/game/horizon in paired cohort")
        by_identity[key][r["forecast_horizon"]] = r
    out = {}
    for late, early in (("T30", "T90"), ("T90", "T24H"), ("T30", "T24H")):
        matching = []
        for bucket in by_identity.values():
            if early not in bucket or late not in bucket:
                continue
            a, b = bucket[early], bucket[late]
            if a["actual_shots"] != b["actual_shots"] or a["played"] != b["played"]:
                raise ValueError("official actual shots disagree across frozen horizons")
            matching.append({
                "game_id": a["game_id"],
                "early": abs(a["forecast_mean"]-a["actual_shots"]),
                "late": abs(b["forecast_mean"]-b["actual_shots"]),
                "model_mean_change": b["forecast_mean"]-a["forecast_mean"],
                "availability_state_changed": (
                    a["original_availability_state"] != b["original_availability_state"]),
                "gain": abs(a["forecast_mean"]-a["actual_shots"]) -
                        abs(b["forecast_mean"]-b["actual_shots"]),
            })
        k = f"{late}_vs_{early}"
        out[k] = {
            "matched_player_games": len(matching),
            "games": len({r["game_id"] for r in matching}),
            "earlier_mae_same_players": (round(mean(r["early"] for r in matching),5)
                                         if matching else None),
            "later_mae_same_players": (round(mean(r["late"] for r in matching),5)
                                       if matching else None),
            "mae_gain_positive_means_later_improved":
                (round(mean(r["gain"] for r in matching), 5) if matching else None),
            "mean_absolute_model_shot_change":
                (round(mean(abs(r["model_mean_change"]) for r in matching), 6)
                 if matching else None),
            "max_absolute_model_shot_change":
                (round(max(abs(r["model_mean_change"]) for r in matching), 6)
                 if matching else None),
            "unchanged_count_predictions_within_0_001_shots":
                sum(abs(r["model_mean_change"]) < .001 for r in matching),
            "pregame_availability_state_changes":
                sum(r["availability_state_changed"] for r in matching),
            "game_block_bootstrap_gain": _game_ci(matching, k="gain"),
        }
    return out


def investigate(forecasts, official_grades):
    pairs, exclusions = first_pregame_pairs(forecasts, official_grades)
    rows = [_obs(f,g) for f,g in pairs]
    versions = sorted({r["model_version"] for r in rows})
    if len(versions)>1:
        raise ValueError("models of different frozen versions cannot share forensics")
    analysis={}
    for h in HORIZONS:
        total=[r for r in rows if r["forecast_horizon"]==h]
        meaningful=[r for r in total if r["expected_meaningful_participant"]]
        av_groups = defaultdict(list)
        usage_groups = defaultdict(list)
        position_groups = defaultdict(list)
        deploy_groups = defaultdict(list)
        for r in meaningful:
            av_groups[r["original_availability_state"]].append(r)
            position_groups[r["position"]].append(r)
            deploy_groups[r["postgame_deployment_bucket_DIAGNOSTIC_ONLY"]].append(r)
            gap=r["pregame_team_games_since_appearance"]
            bucket=("ASOF_GAMES_SINCE_APPEARANCE_UNKNOWN" if gap is None else
                    "ASOF_0_GAMES" if gap==0 else "ASOF_1_TO_2_GAMES"
                    if gap<=2 else "ASOF_3_PLUS_GAMES")
            usage_groups[bucket].append(r)
        large=sorted((r for r in meaningful
                      if abs(r["forecast_mean"]-r["actual_shots"])>3),
                     key=lambda r:(-abs(r["forecast_mean"]-r["actual_shots"]),
                                   r["game_id"],r["player_id"]))
        analysis[h]={
            "entire_unconditional_population":_summary(total),
            "meaningful_expected_population":_summary(meaningful),
            "sensitivity_actual_played_only_POSTGAME_DIAGNOSTIC":
                _summary([r for r in meaningful if r["played"]]),
            "original_availability_state_BEFORE_GAME":{
                k:_summary(v) for k,v in sorted(av_groups.items())},
            "pregame_games_since_appearance":{
                k:_summary(v) for k,v in sorted(usage_groups.items())},
            "position":{k:_summary(v) for k,v in sorted(position_groups.items())},
            "postgame_deployment_DIAGNOSTIC_ONLY":{
                k:_summary(v) for k,v in sorted(deploy_groups.items())},
            "prior10_matched_baseline":_paired_baseline(
                meaningful, "preexisting_baseline_prior10"),
            "human_frozen_matched_baseline":_paired_baseline(
                meaningful, "preexisting_human_comparator"),
            "large_misses": [{
                "game_id":r["game_id"],"player_id":r["player_id"],
                "player":r["player_name"],"team":r["team"],
                "projected":round(r["forecast_mean"],3),
                "actual":r["actual_shots"],
                "played":r["played"],
                "availability_asof":r["original_availability_state"],
                "toi_recent3_asof_seconds":r["recent3_toi_seconds_pregame"],
                "toi_actual_postgame_seconds":r["actual_toi_seconds_POSTGAME_ONLY"],
                "toi_ratio_postgame_DIAGNOSTIC":
                    r["actual_vs_recent_toi_ratio_POSTGAME_ONLY"],
                "postgame_deployment_DIAGNOSTIC":
                    r["postgame_deployment_bucket_DIAGNOSTIC_ONLY"],
                "original_row_hash":r["original_row_hash"]}
                for r in large[:25]],
        }
    rows_meaningful=[r for r in rows if r["expected_meaningful_participant"]]
    return {
        "schema":SCHEMA,
        "scope":"ORIGINAL_V2_B2_PRE_KICKOFF_FROZEN_FORECASTS_OFFICIAL_FINAL",
        "status":"READONLY_MISS_INVESTIGATION_NOT_A_CALIBRATION_OR_PROMOTION",
        "original_model_versions":versions,
        "graded_matched":len(rows),"meaningful_matched":len(rows_meaningful),
        "source_exclusions":exclusions,
        "model_vs_true_prior10_game_block_inference":{
            h:analysis[h]["prior10_matched_baseline"] for h in HORIZONS},
        "same_player_same_game_across_horizons":_reconcile_horizons(rows_meaningful),
        "by_horizon":analysis,
        "methodology":["Never select winner by hindsight actual participation; played-only group is diagnostic.",
                       "Opponent, availability, ice-time and deployment strata are exploratory; cannot reweight model based on these results.",
                       "Game-cluster bootstrap on the genuine matched game population, not independent player forecasts.",
                       "T30 has few games; no blanket fixed 0.26 correction based on aggregate bias.",
                       "No genuine main FanDuel line or lineup verification implied."],
        "betting_picks":0,"thresholds_tuned":False,"model_predictions_rewritten":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--forecast-ledger",type=Path,required=True)
    p.add_argument("--official-grade-ledger",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    fc=[json.loads(line) for line in args.forecast_ledger.read_text().splitlines()
        if line.strip()]
    gr=[json.loads(line) for line in args.official_grade_ledger.read_text().splitlines()
        if line.strip()]
    res=investigate(fc,gr)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(res,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({
        "matched":res["graded_matched"],
        "meaningful":res["meaningful_matched"],
        "T30":res["by_horizon"]["T30"]["meaningful_expected_population"],
        "T30_deployment":res["by_horizon"]["T30"]["postgame_deployment_DIAGNOSTIC_ONLY"],
        "T30_vs_T90":res["same_player_same_game_across_horizons"]["T30_vs_T90"],
        "T90_prior10":res["model_vs_true_prior10_game_block_inference"]["T90"],
    },sort_keys=True))
if __name__=="__main__":
    main()
