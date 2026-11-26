#!/usr/bin/env python3
"""Standalone next-day grading for frozen CBB neutral expert forecasts.

No dependency on model-fitting code or betting odds. Source receipts remain
untouched on cbb-neutral-forward-state. Official ESPN game_id + two team IDs
must match, and only completed, non-tied final scores may be graded.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA="CBB_NEUTRAL_FORWARD_ORIGINAL_V1"
GRADE_SCHEMA="CBB_NEUTRAL_FORWARD_GRADE_V1"


def grade_one(rec, official):
    if rec.get("schema") != SCHEMA:
        raise ValueError("not a frozen original")
    if not official or not official.get("completed"):
        return {"status":"PENDING"}
    for field in ("game_id","home_team_id","away_team_id"):
        if str(official.get(field)) != str(rec.get(field)):
            return {"status":"IDENTITY_MISMATCH"}
    try:
        home=float(official["home_score"])
        away=float(official["away_score"])
    except (ValueError,TypeError,KeyError):
        return {"status":"INVALID_FINAL"}
    if not 0<=home<=220 or not 0<=away<=220 or home==away:
        return {"status":"INVALID_FINAL"}
    margin,combined=home-away,home+away
    actual_win=(margin>0)
    scores={}
    for method,pred in rec["four_model_predictions"].items():
        p=float(pred["home_pwin"])
        if not 0<=p<=1:
            raise ValueError("corrupted pregame probability")
        scores[method]={
            "winner_correct":bool((p>=.5)==actual_win),
            "home_win_brier":round((p-float(actual_win))**2,8),
            "margin_mae":round(abs(float(pred["margin"])-margin),4),
            "total_mae":round(abs(float(pred["total"])-combined),4),
        }
    return {
        "status":"GRADED","actual_home_score":home,"actual_away_score":away,
        "selected_using_2025_only":rec["selected_with_2025_development_only"],
        "actual_margin":margin,"actual_total":combined,
        "model_grades":scores,
        "sportsbook_roi_calculated":False,
    }


def grade_state(root, grades_root, schedule_file, now=None):
    import pandas as pd
    if not Path(root).exists():
        raise ValueError("frozen original state branch missing")
    schedule=(pd.read_parquet(schedule_file)
              if str(schedule_file).endswith(".parquet") else pd.read_csv(schedule_file))
    official={}
    for row in schedule.to_dict("records"):
        if str(row.get("status_type_completed")).lower() not in ("true","1"):
            continue
        gid=str(row["game_id"])
        event={
            "completed":True,"game_id":gid,
            "home_team_id":str(row["home_id"]),
            "away_team_id":str(row["away_id"]),
            "home_score":row.get("home_score"),
            "away_score":row.get("away_score"),
        }
        if gid in official:
            raise ValueError("duplicate official source game ID")
        official[gid]=event
    captures=0
    counts={"read":0,"new_grades":0,"already_graded":0,"pending":0,
            "identity_mismatch":0,"invalid":0}
    for p in sorted(Path(root).glob("*/*.json")):
        r=json.loads(p.read_text())
        if r.get("schema")!=SCHEMA:
            continue
        counts["read"]+=1
        tip=datetime.fromisoformat(r["event_tip_utc"])
        first=datetime.fromisoformat(r["first_seen_capture_utc"])
        if first.tzinfo is None or tip.tzinfo is None or first>=tip:
            raise ValueError("CORRUPTED_OR_NON_PRETIP_ORIGINAL")
        dest=Path(grades_root)/p.parent.name/p.name
        if dest.exists():
            counts["already_graded"]+=1
            continue
        graded=grade_one(r,official.get(r["game_id"]))
        if graded["status"]=="PENDING":
            counts["pending"]+=1
            continue
        if graded["status"]=="IDENTITY_MISMATCH":
            counts["identity_mismatch"]+=1
            continue
        if graded["status"]!="GRADED":
            counts["invalid"]+=1
            continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("x") as f:
            json.dump({
                "schema":GRADE_SCHEMA,
                "source_receipt_sha256":r["sha256"],
                "graded_utc":(now or datetime.now(timezone.utc)).isoformat(),
                "grading_source":"ESPN_HOOPR_OFFICIAL_COMPLETED_SCHEDULE",
                "result":graded,
            },f,indent=2,sort_keys=True)
            f.write("\n")
        counts["new_grades"]+=1
    return counts


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--originals",required=True)
    ap.add_argument("--grades",required=True)
    ap.add_argument("--schedule",required=True)
    a=ap.parse_args()
    print(json.dumps(grade_state(a.originals,a.grades,a.schedule),
                     sort_keys=True))


if __name__=="__main__":
    main()
