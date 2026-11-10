#!/usr/bin/env python3
"""NFL V6 role evidence from public PFR offensive snaps via nflverse.

Source: nflverse-data release snap_counts_{season}.csv. The game-by-game
snapshot is postgame information. Historical as-of reconstruction is allowed
only for games finished 24h before cutoff. Live capture must have its own
retrieval timestamp; never label historic release contents as a proven
contemporaneous pregame snapshot. These are **snap shares, not routes run**.

Identity: exact official game_id/team and uniquely normalized player name.
No cross-team name matching or treating absent snap rows as zero involvement.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter
from datetime import datetime,timedelta,timezone
from pathlib import Path

SCHEMA="NFL_V6_PFR_SNAP_ROLE_EVIDENCE_A"
MIN_GAMES=2


def utc(value):
    dt=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if dt.tzinfo is None:
        raise ValueError("source timestamp must specify timezone")
    return dt.astimezone(timezone.utc)


def norm(value):
    s=unicodedata.normalize("NFKD",str(value or "").lower())
    s="".join(c for c in s if not unicodedata.combining(c))
    s=re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b","",s)
    return " ".join(re.sub(r"[^a-z ]"," ",s).split())


def float_stat(value):
    try:
        n=float(str(value).strip())
    except (ValueError,TypeError):
        return None
    return n if math.isfinite(n) else None


def read_snap_roles(paths,db,cutoff,*,retrieved_at=None):
    """Only source-game snapshots safely older than cutoff enter the map.

    Historical files do NOT prove what data were available on that date;
    the 24h publication lag is an explicit conservative assumption.
    """
    cutoff=utc(cutoff)
    if retrieved_at is not None and utc(retrieved_at)<cutoff-timedelta(days=3650):
        raise ValueError("implausibly old source receipt")
    con=sqlite3.connect("file:"+str(Path(db).absolute())+"?mode=ro",uri=True)
    try:
        games={gid:{"season":int(season),"week":int(week),
          "kickoff":utc(kickoff),"teams":(home,away)}
          for gid,season,week,kickoff,home,away in con.execute("""
            SELECT game_id,season,week,kickoff_utc,home_team,away_team FROM games
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
              AND kickoff_utc IS NOT NULL
          """)}
    finally:
        con.close()
    available={gid:meta for gid,meta in games.items()
      if meta["kickoff"]+timedelta(hours=24)<=cutoff}
    evidence={}
    duplicates=set()
    counts=Counter()
    sources=[]
    for path in paths:
        file=Path(path)
        raw=file.read_bytes()
        sources.append({"filename":file.name,"sha256":hashlib.sha256(raw).hexdigest(),
                        "bytes":len(raw)})
        with file.open(newline="",encoding="utf-8") as stream:
            reader=csv.DictReader(stream)
            required={"game_id","season","week","game_type","player","team",
              "offense_snaps","offense_pct"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("nflverse snap-count schema incomplete")
            for row in reader:
                counts["source_rows"]+=1
                if row.get("game_type")!="REG":
                    counts["not_regular_season"]+=1
                    continue
                gid=str(row["game_id"])
                g=available.get(gid)
                if not g:
                    counts["not_available_by_cutoff"]+=1
                    continue
                if (str(row["team"]) not in g["teams"] or
                    int(row["season"])!=g["season"] or
                    int(row["week"])!=g["week"]):
                    counts["unmatched_source_game"]+=1
                    continue
                name=norm(row["player"])
                pct=float_stat(row["offense_pct"])
                snaps=float_stat(row["offense_snaps"])
                if not name or pct is None or snaps is None or not (
                        0<=pct<=1 and snaps>=0 and int(snaps)==snaps):
                    counts["invalid_or_missing_offense_share"]+=1
                    continue
                # Not playing on offense is observed ONLY for players with
                # explicit records, but these rows aren't role predictions.
                if snaps==0:
                    counts["explicit_zero_offense_snaps"]+=1
                    continue
                if pct==0:
                    counts["contradictory_snap_share"]+=1
                    continue
                key=(gid,str(row["team"]),name)
                if key in evidence:
                    duplicates.add(key)
                    continue
                evidence[key]={"game_id":gid,"team":str(row["team"]),
                  "player":row["player"],"offense_snaps":int(snaps),
                  "offense_pct":pct}
                counts["matched_offense_rows"]+=1
    for key in duplicates:
        evidence.pop(key,None)
    counts["ambiguous_player_game_names"]=len(duplicates)
    return {"schema":SCHEMA,"cutoff_utc":cutoff.isoformat(),
      "captured_at_utc":utc(retrieved_at).isoformat() if retrieved_at else None,
      "historic_source_retrieval_was_before_target_cutoff":(
        retrieved_at is not None and utc(retrieved_at)<=cutoff),
      "historic_asof_snap_publication_timing_verified":False,
      "routes_run_available":False,"snapshot_sources":sources,
      "counts":dict(counts),"by_player_game":evidence}


def observed_role(player_history,player_name,team,source):
    """Role as-of last 3 *observed player games*, refusing missing data.

    Target rate = actual targets / observed offensive snaps. This is NOT a
    per-route target rate. Snaps may include blocking or run plays.
    """
    matches=[]
    for g in player_history[-3:]:
        if g.get("team")!=team:
            continue
        key=(g["game_id"],team,norm(player_name))
        row=source["by_player_game"].get(key)
        if row is None:
            continue
        matches.append((g,row))
    if len(matches)<MIN_GAMES:
        return {"status":"INSUFFICIENT_SNAP_SOURCE",
          "matched":len(matches),"required":MIN_GAMES,
          "no_inference_from_missing_snap_rows":True}
    snaps=sum(r["offense_snaps"] for _,r in matches)
    if snaps<=0:
        return {"status":"INSUFFICIENT_SNAP_SOURCE","matched":len(matches)}
    weights=[.80**(len(matches)-1-i) for i in range(len(matches))]
    share=sum(w*r["offense_pct"] for w,(_,r) in zip(weights,matches))/sum(weights)
    targets=sum(float(g["targets"]) for g,_ in matches)
    carries=sum(float(g["carries"]) for g,_ in matches)
    return {"status":"OBSERVED_ASOF_OFFENSE_SNAP_ROLE_NOT_ROUTE_DATA",
      "matched":len(matches),"offense_snap_share":round(share,5),
      "target_per_offensive_snap":round(targets/snaps,5),
      "carries_per_offensive_snap":round(carries/snaps,5),
      "recent_target_count":round(targets,2),
      "recent_carry_count":round(carries,2),
      "recent_offense_snap_count":int(snaps),
      "active_status_verified":False,
      "route_participation_verified":False}


def role_coverage(game,team,player_histories,source):
    from nfl_v6_joint_opportunity_shadow import participants
    import nfl_v6_joint_opportunity_shadow as v6
    # For previews only, the team player rows should be supplied by caller;
    # no inference of active status from the snapshots.
    return {}


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--db",default="nfl_models/nfl_model.sqlite")
    p.add_argument("--cutoff",required=True)
    p.add_argument("--files",nargs="+",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    role=read_snap_roles(a.files,a.db,a.cutoff,
        retrieved_at=datetime.now(timezone.utc))
    # JSON tuple-key source map is not persisted. Retain source audit only.
    out={k:v for k,v in role.items() if k!="by_player_game"}
    out["source_player_game_keys"]=len(role["by_player_game"])
    Path(a.out).write_text(json.dumps(out,indent=2)+"\n")
    print("NFL_V6_SNAP_ROLE_AUDIT",json.dumps(out))
