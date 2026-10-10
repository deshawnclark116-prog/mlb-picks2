#!/usr/bin/env python3
"""CFB paper-only 2026 player-stat outcome grader; independent of the live CFB grader.

Operates on an ALREADY SAVED, immutable pregame research artifact and a
separately refreshed ESPN game+player outcome SQLite. Refuses post-kickoff
forecasts, missing identities, mismatched kickoff, nonfinal games, and
duplicates. No historical forecast creation, no fitting, no promotion.
Do not claim as-of baseline values if the original research record lacked them.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "CFB_2026_SHADOW_OUTCOME_AUDIT_V1"
INPUT_SCHEMA = "CFB_2026_PER_EVENT_SHADOW_V1"
VALID_STAT = {"rushing_yards": "rushing_yards", "passing_touchdowns": "passing_touchdowns"}
FINAL_STATUSES = {"STATUS_FINAL", "STATUS_FINAL_OVERTIME"}
SCIENTIFIC_STATUS = "PAPER_ONLY_FUTURE_OUTCOME_OBSERVATIONAL_NO_TUNING"


class ShadowOutcomeError(ValueError):
    pass


def utc(raw):
    try:
        value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if value.tzinfo is None:
            raise ValueError("naive datetime")
        return value.astimezone(timezone.utc)
    except (ValueError, TypeError) as exc:
        raise ShadowOutcomeError("INVALID_OR_NAIVE_UTC_TIMESTAMP") from exc


def number(raw, label):
    if isinstance(raw, bool) or not isinstance(raw, (float, int)) or not math.isfinite(raw):
        raise ShadowOutcomeError("INVALID_" + label.upper())
    return float(raw)


def stats(rows):
    if not rows:
        return {"n": 0, "reason": "NO_ELIGIBLE_COMPLETED_SHADOW_FORECASTS"}
    n = len(rows)
    baseline = [r for r in rows if r["baseline_last3_mean"] is not None]
    quantiles = [r for r in rows if r["p10"] is not None and r["p90"] is not None]
    return {
        "n": n,
        "mean_projection_mae": round(sum(r["mean_absolute_error"] for r in rows)/n, 4),
        "median_projection_mae": round(sum(r["median_absolute_error"] for r in rows)/n, 4),
        "mean_projection_rmse": round(math.sqrt(sum(r["mean_error"]**2 for r in rows)/n), 4),
        "mean_projection_bias_pred_minus_actual": round(sum(r["mean_error"] for r in rows)/n, 4),
        "original_pregame_baseline_n": len(baseline),
        "baseline_last3_mean_mae_same_eligible_rows":
            round(sum(abs(r["baseline_last3_mean"] - r["actual"]) for r in baseline)/len(baseline), 4)
            if baseline else None,
        "challenger_mean_mae_on_baseline_comparable_rows":
            round(sum(r["mean_absolute_error"] for r in baseline)/len(baseline), 4)
            if baseline else None,
        "p10_p90_n": len(quantiles),
        "p10_p90_empirical_coverage":
            round(sum(r["p10"] <= r["actual"] <= r["p90"] for r in quantiles)/len(quantiles), 5)
            if quantiles else None,
        "p10_p90_mean_width":
            round(sum(r["p90"] - r["p10"] for r in quantiles)/len(quantiles), 4)
            if quantiles else None,
        "prob_over_line_brier":
            round(sum((r["p_over_fixed_line"] -
                       float(r["actual"] > r["original_classifier_line"]))**2 for r in rows)/n, 5),
        "statistical_note": "Observational paper model; no odds, profit, release, or superiority claim.",
    }


def audit(paper, conn, *, artifact_sha256=None, source_run_metadata=None):
    if paper.get("schema") != INPUT_SCHEMA or paper.get("status") != "NEW_PREKICKOFF_RESEARCH_ONLY_NOT_PRODUCTION":
        raise ShadowOutcomeError("UNVERIFIED_PAPER_REPORT_SCHEMA_OR_STATUS")
    if paper.get("historical_backfills") != 0 or paper.get("forward_original_ledger_modified") is not False:
        raise ShadowOutcomeError("FROZEN_EVIDENCE_CONTRACT_VIOLATION")
    original_generated = utc(paper.get("generated_at_utc"))
    forecasts = paper.get("shadow")
    if not isinstance(forecasts, list) or paper.get("shadow_predictions") != len(forecasts):
        raise ShadowOutcomeError("SHADOW_COUNT_MISMATCH")
    verified_action = False
    source_action = None
    if source_run_metadata is not None:
        meta = source_run_metadata
        if (meta.get("conclusion") != "success" or meta.get("status") != "completed"
            or meta.get("name") != "CFB 2026 event data qualification (research, no serving)"
            or not isinstance(meta.get("head_sha"), str)
            or len(meta["head_sha"]) != 40):
            raise ShadowOutcomeError("UNVERIFIED_OR_UNSUCCESSFUL_ORIGIN_ACTIONS_RUN")
        start=utc(meta.get("run_started_at"))
        created=utc(meta.get("created_at"))
        completed=utc(meta.get("updated_at"))
        if not created <= start <= original_generated <= completed:
            raise ShadowOutcomeError("ORIGINAL_ACTIONS_RUN_TIMELINE_INVALID")
        if any(original_generated >= utc(x.get("kickoff_utc")) or
               created >= utc(x.get("kickoff_utc")) or
               start >= utc(x.get("kickoff_utc")) for x in forecasts):
            raise ShadowOutcomeError("ACTIONS_RUN_DID_NOT_START_BEFORE_KICKOFF")
        verified_action = True
        source_action = {
            "run_id": meta.get("id"), "head_sha": meta["head_sha"],
            "run_created_at": meta["created_at"],
            "run_started_at": meta["run_started_at"],
            "run_completed_at": meta["updated_at"],
            "workflow_name": meta["name"],
            "html_url": meta.get("html_url"),
        }
    games = {}
    for g in conn.execute("SELECT game_id,season,week,kickoff_utc,home_team,away_team,home_points,away_points FROM games WHERE season=2026"):
        if str(g["game_id"]) in games:
            raise ShadowOutcomeError("DUPLICATE_OUTCOME_GAME_ID")
        games[str(g["game_id"])] = g
    states = {str(r["game_id"]):r["espn_status"] for r in conn.execute(
        "SELECT game_id,espn_status FROM schedule_snapshot WHERE season=2026")}
    seen, graded, ungraded = set(), [], defaultdict(int)
    for rec in forecasts:
        if not isinstance(rec, dict):
            raise ShadowOutcomeError("INVALID_FORECAST_ROW")
        gid, pid, market = str(rec.get("game_id")), str(rec.get("player_id")), rec.get("market")
        key=(gid,pid,market)
        if key in seen:
            raise ShadowOutcomeError("DUPLICATE_FORECAST_ID")
        seen.add(key)
        if market not in VALID_STAT:
            raise ShadowOutcomeError("UNKNOWN_SHADOW_MARKET")
        if rec.get("research_only") is not True or rec.get("historical_original_prediction") is not False:
            raise ShadowOutcomeError("LIVE_OR_RETROACTIVE_RECORD_NOT_ALLOWED")
        kickoff=utc(rec.get("kickoff_utc"))
        made=utc(rec.get("forecast_generated_at_utc"))
        if made != original_generated or made >= kickoff:
            raise ShadowOutcomeError("POSTKICKOFF_OR_INCONSISTENT_FORECAST_TIME")
        if rec.get("week") is None or not isinstance(rec["week"], int):
            raise ShadowOutcomeError("INVALID_FORECAST_WEEK")
        for col in ("projected_mean", "projected_median", "p10", "p90",
                    "original_classifier_line", "p_over_fixed_line"):
            number(rec.get(col), col)
        if rec["p10"] > rec["projected_median"] or rec["projected_median"] > rec["p90"]:
            raise ShadowOutcomeError("INVALID_DISTRIBUTION_QUANTILES")
        if not 0 <= rec["p_over_fixed_line"] <= 1:
            raise ShadowOutcomeError("INVALID_MODEL_PROBABILITY")
        baseline = rec.get("baseline_last3_mean")
        if baseline is not None:
            number(baseline, "original_baseline")
            if len(rec.get("baseline_last3_game_ids", [])) != 3:
                raise ShadowOutcomeError("BASELINE_GAME_IDS_MISSING")
            if not set(rec["baseline_last3_game_ids"]).issubset(set(rec.get("prior_verified_same_team_game_ids", []))):
                raise ShadowOutcomeError("BASELINE_NOT_IN_ORIGINAL_RECONCILED_HISTORY")
        game=games.get(gid)
        if game is None:
            ungraded["NO_MATCHING_GAME"] += 1
            continue
        gkick=utc(game["kickoff_utc"])
        if abs((gkick-kickoff).total_seconds()) > 120:
            raise ShadowOutcomeError("ORIGINAL_AND_OUTCOME_KICKOFF_DISAGREE")
        if rec.get("week") != game["week"] or rec.get("team") not in (game["home_team"], game["away_team"]):
            raise ShadowOutcomeError("GAME_AND_TEAM_MISMATCH")
        if states.get(gid) not in FINAL_STATUSES or game["home_points"] is None or game["away_points"] is None:
            ungraded["GAME_NOT_OFFICIALLY_FINAL"] += 1
            continue
        player_rows=conn.execute(
            "SELECT player_id,team,game_id,season,week,position,rushing_yards,passing_touchdowns "
            "FROM player_games WHERE player_id=? AND game_id=? AND season=2026",
            (pid, gid)).fetchall()
        if len(player_rows)>1:
            raise ShadowOutcomeError("AMBIGUOUS_ACTUAL_PLAYER_ID")
        if not player_rows:
            ungraded["MISSING_FINISHED_PLAYER_STAT_LINE"] += 1
            continue
        actual_row=player_rows[0]
        if (actual_row["team"]!=rec["team"] or actual_row["week"]!=rec["week"] or
            actual_row["position"]!=("RB" if market=="rushing_yards" else "QB")):
            raise ShadowOutcomeError("ACTUAL_PLAYER_TEAM_POSITION_MISMATCH")
        if actual_row[VALID_STAT[market]] is None:
            ungraded["MISSING_ACTUAL_STAT_NOT_ZERO"] += 1
            continue
        actual=number(actual_row[VALID_STAT[market]], "actual_stat")
        result={
            "game_id":gid,"player_id":pid,"team":rec["team"],"market":market,
            "actual":actual,
            "projected_mean":float(rec["projected_mean"]),
            "projected_median":float(rec["projected_median"]),
            "mean_error":round(float(rec["projected_mean"])-actual,5),
            "mean_absolute_error":round(abs(float(rec["projected_mean"])-actual),5),
            "median_absolute_error":round(abs(float(rec["projected_median"])-actual),5),
            "p10":rec["p10"],"p90":rec["p90"],
            "original_classifier_line":rec["original_classifier_line"],
            "p_over_fixed_line":rec["p_over_fixed_line"],
            "baseline_last3_mean":baseline,
            "original_forecast_generated_at_utc":rec["forecast_generated_at_utc"],
        }
        graded.append(result)
    grouped=defaultdict(list)
    for row in graded:
        grouped[row["market"]].append(row)
    return {
        "schema":SCHEMA,"scientific_status":SCIENTIFIC_STATUS,
        "input_artifact_sha256":artifact_sha256,
        "actions_run_temporal_provenance_verified":verified_action,
        "source_actions_run":source_action,
        "original_generated_at_utc":paper["generated_at_utc"],
        "total_original_shadow_forecasts":len(forecasts),
        "graded_completed":len(graded),"ungraded_count":len(forecasts)-len(graded),
        "ungraded_reasons":dict(sorted(ungraded.items())),
        "overall":stats(graded),
        "by_market":{m:stats(v) for m,v in sorted(grouped.items())},
        "graded_rows":graded,
        "no_model_fitting_or_backfill":True,
        "existing_live_predictions_unchanged":True,
        "no_stat_acc_score_for_incomplete_games":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--paper",type=Path,required=True)
    p.add_argument("--db",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source-run-metadata",type=Path,required=True,
                   help="Unchanged original GitHub Actions run REST metadata; not a forged filename timestamp")
    a=p.parse_args()
    raw=a.paper.read_bytes()
    paper=json.loads(raw)
    with sqlite3.connect(f"file:{a.db}?mode=ro",uri=True) as con:
        con.row_factory=sqlite3.Row
        result=audit(paper,con,artifact_sha256=hashlib.sha256(raw).hexdigest(),
                     source_run_metadata=json.loads(a.source_run_metadata.read_text()))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":SCIENTIFIC_STATUS,"graded":result["graded_completed"],
                      "ungraded":result["ungraded_count"],"overall":result["overall"]},sort_keys=True))


if __name__=="__main__":
    main()
