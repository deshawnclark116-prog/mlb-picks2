#!/usr/bin/env python3
"""Prospective, original-file NFL Phase1F vs Phase1E vs frozen V2.

Only forecasts originally saved before kickoff, against the identical source
snapshot and game/roster/player ID, are graded. Do NOT regenerate after game.
New unvalidated Phase1F challenger is never declared promoted by this script.
"""
from __future__ import annotations
import argparse
import json
import math
import sqlite3
from collections import Counter
from datetime import timezone
from pathlib import Path

from nfl_phase1e_forward_numeric_grade import MARKETS, HOURS, summarize, read_pair, utc

SCHEMA="NFL_PHASE1F_VS_PHASE1E_V2_ORIGINAL_PROSPECTIVE_A"


def _read(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compare(root, db, *,season=2026, weeks=range(11,19), horizon="T90"):
    if horizon not in HOURS: raise ValueError("BAD_HORIZON")
    root=Path(root)
    files=sorted(p for p in (root/"challenger_forecasts"/"batches").glob(
        f"{season}_wk*_{horizon}_*.jsonl")
        if int(p.name.split("_")[1][2:]) in set(weeks))
    excluded=Counter()
    out=[]
    if not files: raise ValueError("NO_ORIGINAL_PHASE1F_FORECASTS")
    con=sqlite3.connect(f"file:{Path(db).absolute()}?mode=ro",uri=True)
    con.row_factory=sqlite3.Row
    try:
        for f in files:
            o=root/"forecasts"/"batches"/f.name
            v=root/"v2forecasts"/"batches"/f.name
            if not o.exists() or not v.exists():
                excluded["missing_original_phase1e_or_frozen_v2_file"]+=1
                continue
            game,phase1,v2=read_pair(o,v,horizon)
            raw=_read(f)
            hdr=next((x for x in raw if x.get("_batch")),None)
            if hdr is None or hdr.get("_batch")!=f.stem:
                raise ValueError("CHALLENGER_BATCH_HEADER_MISMATCH")
            rows=[x for x in raw if x.get("outcome") in MARKETS]
            ktime=utc(rows[0]["kickoff"]) if rows else None
            if ktime is None or utc(hdr["generated_at"])>=ktime:
                raise ValueError("PHASE1F_GENERATED_AFTER_KICKOFF")
            by={}
            for x in rows:
                if not x["model_version"].startswith("p1f-"):
                    raise ValueError("NOT_ACTUAL_PHASE1F_VERSION")
                if x["game_id"]!=game or x["horizon"]!=horizon:
                    raise ValueError("CHALLENGER_GAME_HORIZON_MISMATCH")
                if utc(x["cutoff"])>=utc(x["kickoff"]):
                    raise ValueError("CHALLENGER_INVALID_CUTOFF")
                if utc(x["input_snapshots"]["retrieval_ts"])>utc(x["cutoff"]):
                    raise ValueError("CHALLENGER_POST_CUTOFF_SOURCE")
                if x["input_snapshots"].get("time_travel") is not False:
                    raise ValueError("CHALLENGER_TIME_TRAVEL_DISALLOWED")
                if x.get("phase1f_evidence",{}).get("forecast_promoted") is not False:
                    raise ValueError("CHALLENGER_NOT_LABELED_RESEARCH")
                key=(x["team"],str(x["player_id"]),x["outcome"])
                if key in by:raise ValueError("CHALLENGER_DUPLICATE_PLAYER_MARKET")
                by[key]=x
            games=con.execute("SELECT home_team,away_team,home_score,away_score "
                "FROM games WHERE game_id=?",(game,)).fetchall()
            if len(games)!=1 or games[0]["home_score"] is None or games[0]["away_score"] is None:
                excluded["game_unfinished"]+=1
                continue
            for key in sorted(set(by)|set(phase1)|set(v2)):
                a=by.get(key);b=phase1.get(key);c=v2.get(key)
                if not(a and b and c):
                    excluded["not_matched_in_all_three_pregame_models"]+=1
                    continue
                if not (a["input_snapshots"]["content_id"]==
                        b["input_snapshots"]["content_id"]==
                        c["input_hashes"]["content_id"]):
                    raise ValueError("NOT_SAME_PREGAME_SNAPSHOT")
                if utc(a["kickoff"])!=utc(b["kickoff"]) or utc(a["cutoff"])!=utc(b["cutoff"]):
                    raise ValueError("NOT_SAME_SCHEDULE_AND_HORIZON")
                team,pid,market=key
                if team not in {games[0]["home_team"],games[0]["away_team"]}:
                    raise ValueError("UNKNOWN_TEAM")
                actual=con.execute(
                    f"SELECT {MARKETS[market]} FROM player_games WHERE "
                    "game_id=? AND team=? AND player_id=? AND season_type='REG'",
                    (game,team,pid)).fetchall()
                if len(actual)!=1 or actual[0][0] is None:
                    excluded["unverified_player_outcome_not_zero"]+=1
                    continue
                y=float(actual[0][0])
                # Use medians for all three to compare exact numerical point
                # predictions, not a sportsbook threshold or cherry-picked pick.
                p1=float(b["median"])
                p1f=float(a["median"])
                p2=float(c["v2_frozen_comparator"]["point"])
                if not all(math.isfinite(z) for z in (y,p1,p1f,p2)):
                    raise ValueError("BAD_NUMERIC_FORECAST")
                out.append({"game_id":game,"horizon":horizon,"team":team,
                    "player_id":pid,"player":a["player_name"],"market":market,
                    "actual":y,"phase1f_median":p1f,"phase1e_median":p1,
                    "frozen_v2_point":p2,
                    "phase1f_abs_error":abs(p1f-y),
                    "phase1e_abs_error":abs(p1-y),"frozen_v2_abs_error":abs(p2-y),
                    "phase1f_p10":float(a["quantiles"]["p10"]),
                    "phase1f_p90":float(a["quantiles"]["p90"]),
                    "phase1e_p10":float(b["quantiles"]["p10"]),
                    "phase1e_p90":float(b["quantiles"]["p90"]),
                    "frozen_source_content_id":a["input_snapshots"]["content_id"],
                    "time_of_original_capture":a["input_snapshots"]["retrieval_ts"]})
    finally: con.close()
    def m(rows):
        if not rows:return {"n":0}
        def mean(k):return sum(x[k] for x in rows)/len(rows)
        return {"n":len(rows),"games":len({x["game_id"] for x in rows}),
          "phase1f_mae":round(mean("phase1f_abs_error"),3),
          "phase1e_mae":round(mean("phase1e_abs_error"),3),
          "frozen_v2_mae":round(mean("frozen_v2_abs_error"),3),
          "phase1f_vs_phase1e_mae_lift":round(
             mean("phase1e_abs_error")-mean("phase1f_abs_error"),3),
          "phase1f_within10":round(sum(x["phase1f_abs_error"]<=10 for x in rows)/len(rows),4),
          "phase1e_within10":round(sum(x["phase1e_abs_error"]<=10 for x in rows)/len(rows),4),
          "phase1f_p10_p90_empirical":round(sum(x["phase1f_p10"]<=x["actual"]<=x["phase1f_p90"]
                            for x in rows)/len(rows),4),
          "phase1e_p10_p90_empirical":round(sum(x["phase1e_p10"]<=x["actual"]<=x["phase1e_p90"]
                            for x in rows)/len(rows),4),
          "phase1f_closer":sum(x["phase1f_abs_error"]<x["phase1e_abs_error"] for x in rows),
          "phase1e_closer":sum(x["phase1f_abs_error"]>x["phase1e_abs_error"] for x in rows)}
    stars=[x for x in out if max(x["phase1f_median"],x["phase1e_median"],x["frozen_v2_point"])>=50]
    return {"schema":SCHEMA,"status":"RESEARCH_NO_PROMOTION",
        "horizon":horizon,"season":season,"weeks":sorted(set(weeks)),
        "raw_challenger_forecast_files":len(files),
        "matched":len(out),"excluded":dict(excluded),
        "all_verified_paired":m(out),"pregame_50plus_cohort":m(stars),
        "by_market":{k:m([x for x in out if x["market"]==k]) for k in MARKETS},
        "by_game":[{"game_id":g,"summary":m([x for x in out if x["game_id"]==g])}
                   for g in sorted({x["game_id"] for x in out})],
        "complete_player_by_player":out,
        "limitations":"Unobserved participants remain unresolved, not assigned zero. "
            "Only originally captured forecasts are eligible. Real prospective "
            "lift and calibration, not retrospective tuning, determine promotion."}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--db",required=True)
    ap.add_argument("--from-week",type=int,default=11)
    ap.add_argument("--through-week",type=int,default=18)
    ap.add_argument("--horizon",choices=tuple(HOURS),default="T90")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    d=compare(a.root,a.db,weeks=range(a.from_week,a.through_week+1),horizon=a.horizon)
    Path(a.out).write_text(json.dumps(d,indent=2)+"\n")
    print("PHASE1F_PROSPECTIVE_METRICS",json.dumps({k:d[k] for k in (
        "status","horizon","raw_challenger_forecast_files","matched","excluded",
        "all_verified_paired","pregame_50plus_cohort","by_market")}))
    if not d["matched"]:
        raise SystemExit("NO_MATCHED_COMPLETED_PREGAME_CHALLENGER_OUTCOMES")


if __name__=="__main__":main()
