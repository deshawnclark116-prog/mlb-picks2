#!/usr/bin/env python3
"""Read-only publisher for ORIGINAL, frozen, pre-kickoff CFB player-stat research.

Builds a small public *research* contract for the Google AI Studio app from
the immutable original GH Actions run, never from the current/hindsight DB.

- Does not create, recompute or modify any actual player-stat projection.
- Preserves each original 3-game baseline as the baseline (NOT a champion).
- Overlay 2026 per-carry MC only on exact player, game, market identity.
- Refuses timestamp, run, game, original source or identity inconsistencies.
- No sportsbook-line fields, confidence badges or 81%-style classifier claims.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "cfb-real-stat-research-board-v1"
BASE = "CFB_2026_BOX_SCORE_LAST3_PROSPECTIVE_BASELINE_V1"
MC = "CFB_2026_PER_EVENT_SHADOW_V1"
ORIGIN_NAME = "CFB 2026 event data qualification (research, no serving)"
RUN_DEFAULT = 38053736857
ALLOWED = frozenset({"rushing_yards", "passing_touchdowns", "anytime_touchdowns"})


class PublicationError(ValueError):
    pass


def utc(value):
    if not isinstance(value, str):
        raise PublicationError("MISSING_UTC_TIMESTAMP")
    try:
        d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise PublicationError("INVALID_UTC_TIMESTAMP") from e
    if d.tzinfo is None:
        raise PublicationError("NAIVE_TIMESTAMP")
    return d.astimezone(timezone.utc)


def numeric(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PublicationError("INVALID_" + label.upper())
    return float(value)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(baseline, shadow, origin, *, original_artifact_sha256, shadow_sha256):
    if baseline.get("schema") != BASE or baseline.get("status") != "FROZEN_PREGAME_NAIVE_POINT_BASELINE_ONLY":
        raise PublicationError("INVALID_ORIGINAL_BASELINE_SCHEMA")
    if shadow.get("schema") != MC or shadow.get("status") != "NEW_PREKICKOFF_RESEARCH_ONLY_NOT_PRODUCTION":
        raise PublicationError("INVALID_ORIGINAL_SHADOW_SCHEMA")
    if baseline.get("historical_forecasts_backfilled") != 0 or baseline.get("production_model_modified") is not False:
        raise PublicationError("RETROACTIVE_OR_PRODUCTION_BASELINE")
    if shadow.get("historical_backfills") != 0 or shadow.get("forward_original_ledger_modified") is not False:
        raise PublicationError("RETROACTIVE_OR_PRODUCTION_SIMULATION")
    if not (origin.get("conclusion") == "success" and origin.get("status") == "completed"
            and origin.get("name") == ORIGIN_NAME and origin.get("id") == RUN_DEFAULT):
        raise PublicationError("ORIGINAL_GITHUB_ACTIONS_RUN_NOT_VERIFIED")
    sha=origin.get("head_sha")
    if not isinstance(sha,str) or len(sha)!=40 or any(c not in "0123456789abcdef" for c in sha):
        raise PublicationError("ORIGINAL_COMMIT_SHA_MISSING")
    start, created, ended = map(utc, (origin.get("run_started_at"),origin.get("created_at"),origin.get("updated_at")))
    btime, stime = utc(baseline.get("generated_at_utc")), utc(shadow.get("generated_at_utc"))
    if not created<=start<=btime<=ended or not created<=start<=stime<=ended:
        raise PublicationError("ORIGINAL_RESEARCH_ARTIFACT_NOT_DURING_ACTIONS_RUN")
    if not (isinstance(original_artifact_sha256,str) and len(original_artifact_sha256)==64
            and isinstance(shadow_sha256,str) and len(shadow_sha256)==64):
        raise PublicationError("SOURCE_ARTIFACT_SHA256_REQUIRED")
    brows, srows=baseline.get("forecasts"),shadow.get("shadow")
    if not isinstance(brows,list) or len(brows)!=baseline.get("baseline_rows"):
        raise PublicationError("ORIGINAL_BASELINE_COUNT_MISMATCH")
    if not isinstance(srows,list) or len(srows)!=shadow.get("shadow_predictions"):
        raise PublicationError("ORIGINAL_SIMULATION_COUNT_MISMATCH")
    if not brows:
        raise PublicationError("NO_REAL_PREGAME_BASELINE_RECORDS")
    rows, lookup = [], {}
    for b in brows:
        if b.get("market") not in ALLOWED or b.get("season")!=2026 or b.get("week")!=6:
            raise PublicationError("OUT_OF_SCOPE_BASELINE")
        generated=utc(b["research_generated_at_utc"]); kickoff=utc(b["kickoff_utc"])
        if not (generated==btime and created<kickoff and start<kickoff and generated<kickoff):
            raise PublicationError("BASELINE_NOT_GENERATED_BEFORE_KICKOFF")
        if utc(b["board_generated_at_utc"])>generated:
            raise PublicationError("LEGACY_SOURCE_BOARD_PUBLISHED_AFTER_BASELINE")
        vals=b.get("last3_values"); game_ids=b.get("last3_game_ids")
        if (not isinstance(vals,list) or len(vals)!=3 or
            not isinstance(game_ids,list) or len(game_ids)!=3 or len(set(game_ids))!=3):
            raise PublicationError("MISSING_ORIGINAL_THREE_PRIOR_GAMES")
        avg=sum(numeric(x,"prior_stat") for x in vals)/3
        mu=numeric(b.get("point_mean_last3"),"baseline_mean")
        if abs(avg-mu)>0.00006:
            raise PublicationError("BASELINE_MEAN_CHANGED_FROM_ORIGINAL_PRIOR_GAMES")
        med=numeric(b.get("point_median_last3"),"baseline_median")
        if med!=sorted(vals)[1]:
            raise PublicationError("BASELINE_MEDIAN_CHANGED_FROM_PRIOR_GAMES")
        key=(str(b["game_id"]),str(b["player_id"]),b["market"])
        if key in lookup:
            raise PublicationError("DUPLICATE_ORIGINAL_BASELINE_PLAYER_GAME_MARKET")
        if not b.get("player") or not b.get("team") or not b.get("opponent"):
            raise PublicationError("MISSING_PLAYER_TEAM_OR_OPPONENT")
        row={
            "id":"|".join(key),"game_id":key[0],"player_id":key[1],
            "player":b["player"],"team":b["team"],"opponent":b["opponent"],
            "market":b["market"],"kickoff_utc":b["kickoff_utc"],
            "season":2026,"week":6,
            "mean":round(mu,3),"median":round(med,3),
            "baseline_mean":round(mu,3),"baseline_median":round(med,3),
            "p10":None,"p90":None,
            "simulated_mean":None,"simulated_median":None,
            "simulated_p10":None,"simulated_p90":None,
            "simulated_prob_over_original_classifier_line":None,
            "simulations":None,
            "original_prior_game_ids":game_ids,
            "model_type":"NAIVE_LAST3_BOX_SCORE_BASELINE",
            "projection_status":"RESEARCH_BASELINE_UNVALIDATED",
            "sportsbook_line_verified":False,"betting_recommendation":False,
            "roster_verified":False,
            "generated_at_utc":b["research_generated_at_utc"],
        }
        lookup[key]=row;rows.append(row)
    seen_shadow=set()
    for m in srows:
        key=(str(m.get("game_id")),str(m.get("player_id")),m.get("market"))
        if key in seen_shadow:
            raise PublicationError("DUPLICATE_ORIGINAL_SIM_PLAYER_GAME_MARKET")
        seen_shadow.add(key)
        if key not in lookup:
            # A valid simulation absent from the frozen baseline must not
            # silently create a new player or alter the independently
            # captured 170-row denominator.
            raise PublicationError("SHADOW_PLAYER_GAME_NOT_IN_PREKICKOFF_BASELINE")
        b=lookup[key]
        if key[2]!="rushing_yards" or m.get("team")!=b["team"] or m.get("kickoff_utc")!=b["kickoff_utc"]:
            raise PublicationError("SIMULATION_PLAYER_TEAM_GAME_MISMATCH")
        if utc(m.get("forecast_generated_at_utc"))!=stime or stime>=utc(m["kickoff_utc"]):
            raise PublicationError("SIMULATION_NOT_GENUINELY_PREGAME")
        if not m.get("research_only") or m.get("historical_original_prediction") is not False:
            raise PublicationError("SIMULATION_IMPROPERLY_SERVING")
        for f in ("projected_mean","projected_median","p10","p90","n_simulations"):
            numeric(m.get(f),f)
        if m["p10"]>m["projected_median"] or m["projected_median"]>m["p90"]:
            raise PublicationError("SIMULATION_INVALID_QUANTILES")
        if not 100<=m["n_simulations"]<=250000:
            raise PublicationError("INVALID_SIMULATION_COUNT")
        hist=m.get("empirical_outcome_histogram")
        if not isinstance(hist,list) or not hist or m.get("empirical_outcome_histogram_schema")!="INTEGER_STAT_TOTAL_AND_FREQUENCY_V1":
            raise PublicationError("SIMULATED_DISTRIBUTION_NOT_FROZEN")
        if sum(x["count"] for x in hist)!=m["n_simulations"]:
            raise PublicationError("SIMULATED_DISTRIBUTION_MISSING_DRAWS")
        if abs(sum(x["stat_total"]*x["count"] for x in hist)/m["n_simulations"]-m["projected_mean"])>0.051:
            raise PublicationError("SIMULATED_MEAN_NOT_MATCHING_FROZEN_DRAWS")
        b.update({
            "mean":round(m["projected_mean"],3),
            "median":round(m["projected_median"],3),
            "p10":round(m["p10"],3),"p90":round(m["p90"],3),
            "simulated_mean":round(m["projected_mean"],3),
            "simulated_median":round(m["projected_median"],3),
            "simulated_p10":round(m["p10"],3),
            "simulated_p90":round(m["p90"],3),
            "simulated_prob_over_original_classifier_line":None,
            "simulations":m["n_simulations"],
            "model_type":"EMPIRICAL_PER_PLAY_MC_RESEARCH",
            "projection_status":"RESEARCH_SIMULATION_UNCALIBRATED",
            "generated_at_utc":m["forecast_generated_at_utc"],
        })
    rows.sort(key=lambda x:(x["kickoff_utc"],x["game_id"],x["market"],x["player_id"]))
    count=Counter(x["market"] for x in rows)
    simcount=sum(r["model_type"]=="EMPIRICAL_PER_PLAY_MC_RESEARCH" for r in rows)
    if count!={"rushing_yards":58,"passing_touchdowns":76,"anytime_touchdowns":36} or simcount!=15:
        raise PublicationError("FROZEN_OCT10_POPULATION_MISMATCH")
    return {
        "schema":SCHEMA,"status":"ORIGINAL_PREGAME_CFB_RESEARCH_NOT_PROMOTED",
        "research_only":True,"official_betting_picks":0,
        "season":2026,"week":6,"original_pregame_run_id":RUN_DEFAULT,
        "original_run_sha":sha,
        "original_run_url":origin.get("html_url"),
        "generated_at_utc":btime.isoformat().replace("+00:00","Z"),
        "first_original_pregame_run_created_at":origin["created_at"],
        "source_baseline_file_sha256":original_artifact_sha256,
        "source_simulation_file_sha256":shadow_sha256,
        "n_projections":len(rows),"by_market":dict(count),
        "n_empirical_simulations":simcount,
        "no_historical_backfills":True,"no_original_forecasts_modified":True,
        "not_certified_for_sportsbook_bets":True,
        "description":"Real original 2026 Week 6 pregame player-stat values. 170 three-game baselines, including 15 enriched with uncalibrated 4,000-draw rushing-yard distributions. Old classifier lines are not sportsbook prices.",
        "projections":rows,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--original-baseline",required=True,type=Path)
    ap.add_argument("--original-shadow",required=True,type=Path)
    ap.add_argument("--original-run-metadata",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    raw_b=a.original_baseline.read_bytes()
    raw_s=a.original_shadow.read_bytes()
    result=build(json.loads(raw_b),json.loads(raw_s),json.loads(a.original_run_metadata.read_bytes()),
                 original_artifact_sha256=hashlib.sha256(raw_b).hexdigest(),
                 shadow_sha256=hashlib.sha256(raw_s).hexdigest())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="projections"},sort_keys=True))

if __name__=="__main__":
    main()
