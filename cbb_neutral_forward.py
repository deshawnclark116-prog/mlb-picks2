#!/usr/bin/env python3
"""Durable original, pretip CBB neutral-challenger receipts and later final grading.

Research only, never modifies main sports predictions. Files are write-once;
source forecast must have been generated before the original scheduled tip.
Actual final scores can only be joined later on ESPN event/team identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from cbb_neutral_challenger import normcdf

SCHEMA="CBB_NEUTRAL_FORWARD_ORIGINAL_V1"
PROJECTIONS={
    "incumbent": ("incumbent_home_margin","incumbent_total"),
    "swap_invariant": ("swap_home_margin","swap_total"),
    "neutral_specialist": ("specialist_home_margin","specialist_total"),
    "swap_plus_fixed_elo": ("elo_blend_home_margin","elo_blend_total"),
}


def utc(value):
    dt=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if dt.tzinfo is None:
        raise ValueError("UTC offset required")
    return dt.astimezone(timezone.utc)


def freeze(audit, source, root, now):
    capture=utc(now)
    if audit["schema"] != "CBB_NEUTRAL_COURT_CHALLENGER_V1":
        raise ValueError("unknown expert audit")
    if source.get("schema") != "CBB_GAME_INTELLIGENCE_V0_RESEARCH":
        raise ValueError("unknown underlying forecast")
    model_timestamp=utc(source["generated_utc"])
    if model_timestamp > capture or source.get("status") != "RESEARCH_ONLY_NO_PRETIP_FIRST_SEEN_ARCHIVE":
        raise ValueError("only genuine pretip generated forecasts")
    selected=audit["selection_decided_2025_only"]["chosen_using_2025_only"]
    if selected not in PROJECTIONS:
        raise ValueError("invalid selected challenger")
    forecast_by_id={str(x["game_id"]):x for x in source["game_predictions"]}
    created,existing,past,unknown=0,0,0,0
    for row in audit.get("upcoming_neutral_research_shadow",[]):
        tip=utc(row["tip_utc"])
        if tip <= capture:
            past+=1
            continue
        if model_timestamp >= tip:
            raise ValueError("future outcome leaked into as-of score model")
        original=forecast_by_id.get(str(row["game_id"]))
        if original is None or not original.get("neutral_site"):
            unknown+=1
            continue
        if utc(original["tip_utc"]) != tip:
            raise ValueError("event tip mismatch")
        sd=float(row["model_uncertainty_sd"])
        vals={}
        for name,(margin_key,total_key) in PROJECTIONS.items():
            if margin_key not in row or total_key not in row:
                if name == selected:
                    raise ValueError("selected candidate has no forecast")
                continue
            margin=float(row[margin_key]); total=float(row[total_key])
            vals[name]={
                "margin":margin,"total":total,
                "home_pwin":round(float(normcdf(margin/sd)),6),
                "home_score":round((total+margin)/2,3),
                "away_score":round((total-margin)/2,3),
            }
        rec={
            "schema":SCHEMA,
            "first_seen_capture_utc":capture.isoformat(),
            "model_generated_utc":model_timestamp.isoformat(),
            "event_tip_utc":tip.isoformat(),
            "game_id":str(row["game_id"]),
            "home_team_id":str(original["home_team_id"]),
            "away_team_id":str(original["away_team_id"]),
            "home":original["home"],"away":original["away"],
            "neutral_site":True,
            "selected_with_2025_development_only":selected,
            "selected_not_production_approved":True,
            "sigma_margin_from_2025":sd,
            "independent_elo_home_win":row.get("fixed_elo_home_win"),
            "four_model_predictions":vals,
            "odds_captured":False,
            "sportsbook_ats_ou_roi_not_graded":True,
        }
        rec["sha256"]=hashlib.sha256(
            json.dumps(rec,sort_keys=True,separators=(",",":")).encode()
        ).hexdigest()
        day=tip.strftime("%Y-%m-%d")
        destination=Path(root)/day/(str(row["game_id"])+".json")
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            older=json.loads(destination.read_text())
            if older["game_id"]!=rec["game_id"] or older["home_team_id"]!=rec["home_team_id"] or older["away_team_id"]!=rec["away_team_id"]:
                raise ValueError("FIRST_SEEN_IDENTITY_CONFLICT")
            existing+=1
            continue
        try:
            with destination.open("x") as file:
                json.dump(rec,file,sort_keys=True,indent=2)
                file.write("\n")
        except FileExistsError:
            existing+=1
        else:
            created+=1
    return {"new":created,"immutable_existing":existing,"excluded_already_started":past,
            "missing_source_game":unknown,"selected_challenger":selected}


def grade_one(rec, official):
    if rec.get("schema")!=SCHEMA:
        raise ValueError("unknown original source")
    if not official or not official.get("completed"):
        return {"status":"PENDING"}
    for field in ("game_id","home_team_id","away_team_id"):
        if str(official.get(field))!=str(rec.get(field)):
            return {"status":"IDENTITY_MISMATCH"}
    home,away=float(official["home_score"]),float(official["away_score"])
    if home<0 or away<0 or home==away:
        return {"status":"INVALID_FINAL"}
    margin=home-away;total=home+away
    truth=float(margin>0)
    grades={}
    for name,p in rec["four_model_predictions"].items():
        prob=float(p["home_pwin"])
        grades[name]={
            "winner_correct":bool((prob>=.5)==(truth>0)),
            "moneyline_brier":round((prob-truth)**2,8),
            "margin_absolute_error":round(abs(margin-float(p["margin"])),4),
            "total_absolute_error":round(abs(total-float(p["total"])),4),
        }
    return {
        "status":"GRADED",
        "actual_home_score":home,"actual_away_score":away,
        "actual_margin":margin,"actual_total":total,
        "selected_candidate":rec["selected_with_2025_development_only"],
        "per_method":grades,
        "no_real_market_odds_captured":True,
    }


def grade_frozen(root, grades_root, schedule_path):
    import pandas as pd
    data=(pd.read_parquet(schedule_path)
          if str(schedule_path).endswith(".parquet") else pd.read_csv(schedule_path))
    finals={}
    for r in data.to_dict("records"):
        if str(r.get("status_type_completed")).lower() not in ("1","true"):
            continue
        gid=str(r["game_id"])
        finals[gid]={
            "completed":True,"game_id":gid,
            "home_team_id":str(r["home_id"]),"away_team_id":str(r["away_id"]),
            "home_score":r.get("home_score"),"away_score":r.get("away_score"),
        }
    seen=0;new=0;pending=0;existing=0;invalid=0
    for p in Path(root).glob("*/*.json"):
        rec=json.loads(p.read_text())
        if rec.get("schema")!=SCHEMA:
            continue
        seen+=1
        dest=Path(grades_root)/p.parent.name/p.name
        if dest.exists():
            existing+=1
            continue
        result=grade_one(rec,finals.get(rec["game_id"]))
        if result["status"]=="PENDING":
            pending+=1
            continue
        if result["status"]!="GRADED":
            invalid+=1
            continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("x") as f:
            json.dump({"schema":"CBB_NEUTRAL_FORWARD_GRADE_V1",
                       "source_receipt_sha256":rec["sha256"],
                       "graded_at_utc":datetime.now(timezone.utc).isoformat(),
                       "result":result},f,indent=2,sort_keys=True)
            f.write("\n")
        new+=1
    return {"read":seen,"new_grades":new,"previously_graded":existing,
            "pending":pending,"identity_mismatch_or_invalid":invalid}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="cmd",required=True)
    f=sub.add_parser("freeze")
    f.add_argument("--audit",required=True);f.add_argument("--forecast",required=True)
    f.add_argument("--out-root",required=True);f.add_argument("--capture-utc")
    g=sub.add_parser("grade")
    g.add_argument("--originals",required=True);g.add_argument("--grades",required=True)
    g.add_argument("--schedule",required=True)
    a=ap.parse_args()
    if a.cmd=="freeze":
        out=freeze(json.loads(Path(a.audit).read_text()),
                   json.loads(Path(a.forecast).read_text()),
                   a.out_root,a.capture_utc or datetime.now(timezone.utc).isoformat())
    else:
        out=grade_frozen(a.originals,a.grades,a.schedule)
    print(json.dumps(out,sort_keys=True))


if __name__=="__main__":
    main()
