#!/usr/bin/env python3
"""Paired NFL V6 vs FIRST-SEEN incumbent forecasts on exactly the same outcomes.

This is an exploratory retrospective/as-of diagnostic, never a promotion trial:
V6 generated after these games, although its inputs are filtered to pregame.
No reshuffling the incumbent cohort, changing observed missing outcomes to 0,
or treating historical V6 better-than-naive as evidence it beats production.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from nfl_v6_shadow_numeric_grade import grade

MARKETS = frozenset(("rushing_yards","receiving_yards"))


def norm(name):
    s=unicodedata.normalize("NFKD",str(name or "").lower())
    s="".join(c for c in s if not unicodedata.combining(c))
    s=re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b","",s)
    return " ".join(re.sub(r"[^a-z ]"," ",s).split())


def utc(value):
    from datetime import datetime, timezone
    return datetime.fromisoformat(str(value).replace("Z","+00:00")).astimezone(timezone.utc)


def summarize(rows):
    if not rows:
        return {"n":0}
    n=len(rows)
    fields=("v6_mean","incumbent_mean","incumbent_median","naive_prior3")
    out={"n":n,"games":len({r["game_id"] for r in rows})}
    for f in fields:
        out[f+"_mae"]=round(statistics.mean(abs(r["actual"]-r[f]) for r in rows),3)
        out[f+"_within10"]=round(statistics.mean(abs(r["actual"]-r[f])<=10 for r in rows),4)
        out[f+"_within20"]=round(statistics.mean(abs(r["actual"]-r[f])<=20 for r in rows),4)
    out["mae_lift_v6_vs_incumbent_mean"]=round(
        out["incumbent_mean_mae"]-out["v6_mean_mae"],3)
    out["paired_winner_v6_vs_incumbent_mean"]=sum(
        abs(r["actual"]-r["v6_mean"])<abs(r["actual"]-r["incumbent_mean"])
        for r in rows)
    if out["games"]>=3:
        # Cluster by *game* to avoid treating highly correlated teammates as IID.
        clusters=list(defaultdict(list).items())
        bygame=defaultdict(list)
        for row in rows:
            bygame[row["game_id"]].append(
                abs(row["actual"]-row["incumbent_mean"])-
                abs(row["actual"]-row["v6_mean"]))
        blocks=list(bygame.values())
        rng=random.Random(20261109)
        diffs=[]
        for _ in range(1000):
            picked=[blocks[rng.randrange(len(blocks))] for _ in blocks]
            diffs.append(statistics.mean(v for b in picked for v in b))
        diffs.sort()
        out["game_cluster_paired_lift_ci95"]=[
            round(diffs[24],3),round(diffs[974],3)]
    return out


def compare(forecasts, frozen_rows, db):
    if forecasts.get("version")!="NFL_V6_JOINT_OPPORTUNITY_SHADOW_A" or \
       forecasts.get("status")!="RESEARCH_SHADOW_NOT_PROMOTED":
        raise ValueError("expected frozen research-only NFL V6 format")
    v6_grade=grade(forecasts,db)
    # Game & market must have distinct model player identity. Do not choose a
    # random same-name row or match based solely on player name.
    game_pair={}
    kickoff_by_game={}
    for g in forecasts["games"]:
        pair=frozenset((g["home"],g["away"]))
        key=(str(g["kickoff_utc"]),pair)
        if key in game_pair:
            raise ValueError("ambiguous kickoff-game identity")
        game_pair[key]=g["game_id"]
        kickoff_by_game[g["game_id"]]=str(g["kickoff_utc"])
    unique_id=defaultdict(list);unique_name=defaultdict(list)
    for r in v6_grade["worst_numeric_misses"][:0]:
        pass
    # The grader's worst-25 summary is intentionally not a complete cohort;
    # inspect all player outcomes directly from the exact frozen game/team IDs.
    # grade() acts as a strict consistency check, including missing actuals.
    import sqlite3
    con=sqlite3.connect("file:"+str(Path(db).absolute())+"?mode=ro",uri=True)
    try:
        for g in forecasts["games"]:
            for p in g["projections"]:
                if p["market"] not in MARKETS:
                    continue
                col=p["market"]
                actual=con.execute("SELECT "+col+" FROM player_games WHERE "
                  "game_id=? AND team=? AND player_id=? AND season_type='REG'",
                  (g["game_id"],p["team"],p["player_id"])).fetchall()
                if len(actual)!=1 or actual[0][0] is None:
                    continue
                d={"game_id":g["game_id"],"kickoff_utc":g["kickoff_utc"],
                   "player_id":str(p["player_id"]), "player":p["player"],
                   "team":p["team"],"opponent":p["opponent"],"market":col,
                   "actual":float(actual[0][0]),"v6_mean":float(p["projected_mean"]),
                   "naive_prior3":float(p["naive_recent3_mean"])}
                unique_id[(g["game_id"],p["team"],col,str(p["player_id"]))].append(d)
                unique_name[(g["game_id"],p["team"],col,norm(p["player"]))].append(d)
    finally:
        con.close()
    seen=set();paired=[];excluded=Counter()
    for f in frozen_rows:
        if f.get("schema")!="NFL_NUMERIC_PRECISION_FIRSTSEEN_V1" or \
           f.get("market") not in MARKETS:
            excluded["invalid_firstseen_source"]+=1
            continue
        k=(f.get("season"),f.get("week"),f.get("player_id"),
           f.get("team"),f.get("market"))
        if k in seen:
            excluded["duplicate_firstseen"]+=1
            continue
        seen.add(k)
        key=(f.get("kickoff_utc"),
             frozenset((f.get("team"),f.get("opponent"))))
        gid=game_pair.get(key)
        if gid is None:
            excluded["game_not_in_v6_frozen_slate"]+=1
            continue
        try:
            if utc(f["recorded_at_utc"]) >= utc(f["kickoff_utc"]) or \
               utc(f["source_generated_at_utc"]) >= utc(f["kickoff_utc"]):
                excluded["not_recorded_pregame"]+=1
                continue
        except (ValueError,KeyError,TypeError):
            excluded["unverified_firstseen_time"]+=1
            continue
        result=unique_id.get((gid,f["team"],f["market"],str(f["player_id"])),[])
        match="EXACT_SAME_PLAYER_ID"
        if len(result)!=1:
            result=unique_name.get((gid,f["team"],f["market"],norm(f.get("player"))),[])
            match="UNIQUE_SAME_GAME_TEAM_PLAYER_NAME"
        if len(result)!=1:
            excluded["no_unique_v6_actual_match"]+=1
            continue
        row=result[0]
        if row["opponent"]!=f["opponent"] or \
           not all(isinstance(f.get(x),(int,float)) and
             math.isfinite(f[x]) and f[x]>=0 for x in
             ("model_mean","model_median")):
            excluded["mismatch_or_invalid_frozen_incumbent"]+=1
            continue
        paired.append({**row,"incumbent_mean":float(f["model_mean"]),
          "incumbent_median":float(f["model_median"]),
          "incumbent_source":f.get("model_source"),
          "identity_method":match,"original_recorded_at":f["recorded_at_utc"],
          "original_source_sha256":f.get("source_sha256")})
    return {"schema":"NFL_V6_VS_FROZEN_FIRSTSEEN_SAME_COHORT_DIAGNOSTIC_A",
      "promotion_eligible":False,"reason":"BACKTEST_WEEK_ALREADY_KNOWN_AND_NO_CLEAN_HOLDOUT",
      "firstseen_total_rows":len(frozen_rows),"matched":len(paired),
      "excluded":dict(excluded),
      "v6_grade_coverage":{"predicted":v6_grade["predicted_rows"],
        "matched":v6_grade["matched"],"unresolved":v6_grade["unmatched_or_pending"]},
      "overall":summarize(paired),
      "markets":{m:summarize([r for r in paired if r["market"]==m])
                 for m in sorted(MARKETS)},
      "featured_incumbent_mean_ge_60":summarize(
        [r for r in paired if r["incumbent_mean"]>=60]),
      "all_paired_game_by_game":{
        gid:summarize([r for r in paired if r["game_id"]==gid])
        for gid in sorted({r["game_id"] for r in paired})},
      "identity_methods":dict(Counter(r["identity_method"] for r in paired)),
      "rows":paired}


def load_ledger(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--v6",required=True)
    p.add_argument("--firstseen",default="docs/nfl_precision_firstseen.jsonl")
    p.add_argument("--db",default="nfl_models/nfl_model.sqlite")
    p.add_argument("--out",default="/tmp/nfl_v6_same_cohort_diagnostic.json")
    args=p.parse_args()
    result=compare(json.loads(Path(args.v6).read_text()),
                   load_ledger(args.firstseen),args.db)
    Path(args.out).write_text(json.dumps(result,indent=2)+"\n")
    print("NFL_V6_SAME_COHORT_DIAGNOSTIC",json.dumps({
        "firstseen":result["firstseen_total_rows"],"matched":result["matched"],
        "excluded":result["excluded"],"overall":result["overall"],
        "markets":result["markets"],"featured":result["featured_incumbent_mean_ge_60"]}))
