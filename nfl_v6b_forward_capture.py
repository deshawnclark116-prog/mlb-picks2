#!/usr/bin/env python3
"""NFL V6B first-seen SHADOW capture: immutable game-level prekickoff receipts.

The validator never writes production pick files or edits a frozen game. Only
authenticated scheduled games, within a defined 30h pregame capture window,
can be recorded. The runner must have rebuilt the public nflverse foundation
and fetched snap source bytes before invoking this code. Snapshot source
timestamps are captured and hashed. No model-based sportsbook betting claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sqlite3
from datetime import datetime,timedelta,timezone
from pathlib import Path
import nfl_v6_joint_opportunity_shadow as v6

SCHEMA="NFL_V6B_REAL_PREGAME_FIRSTSEEN_RESEARCH_A"
WINDOW_HOURS=30


def capture(db,snap_files,output_root,*,now=None,sims=20000):
    at=v6.utc(now) if now else datetime.now(timezone.utc)
    if sims<10000:
        raise ValueError("real 10k+ complete joint simulations required")
    files=[Path(x) for x in snap_files]
    if len(files)<1 or any(not p.is_file() or p.stat().st_size<1000 for p in files):
        raise ValueError("missing actual public snap source files")
    con=sqlite3.connect("file:"+str(Path(db).absolute())+"?mode=ro",uri=True)
    try:
        matches=con.execute("""
          SELECT season,week,game_id,kickoff_utc
          FROM games WHERE kickoff_utc IS NOT NULL
          ORDER BY kickoff_utc
        """).fetchall()
    finally:
        con.close()
    opportunities=[]
    for season,week,gid,kick in matches:
        start=v6.utc(kick)
        remaining=(start-at).total_seconds()
        if 0<remaining<=WINDOW_HOURS*3600:
            opportunities.append((season,week,gid,start))
    statuses={"captured":[],"already_frozen":[],"source_no_projections":[],
              "outside_capture_window":max(0,len(matches)-len(opportunities))}
    groups={}
    for season,week,gid,kickoff in opportunities:
        groups.setdefault((season,week),[]).append(gid)
    root=Path(output_root)
    root.mkdir(parents=True,exist_ok=True)
    db_sha=hashlib.sha256(Path(db).read_bytes()).hexdigest()
    src_hashes=[{"file":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
                 "bytes":p.stat().st_size} for p in files]
    src_sha=hashlib.sha256(Path(v6.__file__).read_bytes()).hexdigest()
    for (season,week),ids in sorted(groups.items()):
        todo=[gid for gid in ids if not (root/(gid+".json")).exists()]
        for gid in ids:
            if gid not in todo:
                statuses["already_frozen"].append(gid)
        if not todo:
            continue
        with __import__("tempfile").TemporaryDirectory() as tmp:
            result=v6.run(db,season,week,str(Path(tmp)/"shadow.json"),
                 sims=sims,snap_files=files,
                 captured_at_utc=at.isoformat(),only_future=True)
        if result["version"]!="NFL_V6B_SNAP_ROLE_SHADOW_A" or \
           not result["forward_source_cutoff_enforced"] or \
           not result["only_future_game_snapshots"]:
            raise ValueError("V6B source cutoff and game safety contract failed")
        by_id={g["game_id"]:g for g in result["games"]}
        for gid in todo:
            game=by_id.get(gid)
            if game is None:
                raise ValueError("scheduled game omitted from research simulation")
            kickoff=v6.utc(game["kickoff_utc"])
            if not at<kickoff:
                raise ValueError("game started; refusing retroactive firstseen")
            if now is None and datetime.now(timezone.utc)>=kickoff:
                statuses["source_no_projections"].append(gid)
                continue  # the real kickoff occurred during long simulation
            if not game["projections"]:
                statuses["source_no_projections"].append(gid)
                continue
            rows=[x for x in result["snap_role_source_receipts"] if x["game_id"]==gid]
            if len(rows)!=1:
                raise ValueError("missing uniquely matched role-source receipt")
            receipt={"schema":SCHEMA,"model_version":result["version"],
              "status":"RESEARCH_ONLY_NEVER_AN_OFFICIAL_BET",
              "source_snapshot_utc":at.isoformat(),
              "kickoff_utc":game["kickoff_utc"],
              "season":season,"week":week,"game_id":gid,
              "time_before_kickoff_hours":round((kickoff-at).total_seconds()/3600,3),
              "asof_policy":result["asof_policy"],
              "simulations_per_game":sims,
              "team_roster_active_verified":False,
              "injury_inclusive_probability":False,
              "route_data_included":False,
              "forward_probability_calibrated":False,
              "historical_snap_publication_timing_verified":False,
              "source_db_sha256":db_sha,
              "source_python_sha256":src_sha,
              "snap_source_receipts":src_hashes,
              "source_role_game_receipt":rows[0],
              "projections":game["projections"],
              "team_metadata":game["teams"]}
            # Exclusive create: a later run cannot rewrite this first-seen
            # observation after scores, injuries, source revisions, or tuning.
            try:
                with (root/(gid+".json")).open("x") as fp:
                    json.dump(receipt,fp,indent=2)
                    fp.write("\n")
            except FileExistsError:
                statuses["already_frozen"].append(gid)
            else:
                statuses["captured"].append(gid)
    return {"schema":SCHEMA,"source_utc":at.isoformat(),
       "capture_window_hours":WINDOW_HOURS,
       "source_files_sha256":src_hashes,
       "source_db_sha256":db_sha,"summary":statuses}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--db",default="nfl_models/nfl_model.sqlite")
    p.add_argument("--snap-files",nargs="+",required=True)
    p.add_argument("--out-dir",default="docs/nfl_v6b_firstseen")
    p.add_argument("--simulations",type=int,default=20000)
    args=p.parse_args()
    report=capture(args.db,args.snap_files,args.out_dir,
                   sims=args.simulations)
    print("NFL_V6B_FIRSTSEEN_CAPTURE",json.dumps(report))
