#!/usr/bin/env python3
"""Backtest the exact Week 4 T90 SYSTEM PICKS using model projections, not alt lines.

Selection reproduces the frontend's SYSTEM PICKS eligibility/ranking:
- offensive outcomes only;
- p_active >= 0.75;
- a model-derived 65%+ milestone must exist;
- rank by milestone probability, then p_active, player, outcome, game/id;
- display cap: top 10 per game.

The 65% milestone is used ONLY to reconstruct which rows were shown as SYSTEM
PICKS. Accuracy is graded against each row's frozen model median, because that
is the UI's MODEL PROJECTION. Mean is retained as a secondary diagnostic.

No sportsbook line or odds enter selection or grading.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

OFFENSIVE_OUTCOMES=("pass_yds","pass_td","int","rush_yds","rush_td","rec_yds","rec","rec_td","atd")
YARD_OUTCOMES={"pass_yds","rush_yds","rec_yds"}
STAT_ALIASES={
    "pass_yds":("passing_yards",),
    "pass_td":("passing_tds",),
    "int":("passing_interceptions","interceptions"),
    "rush_yds":("rushing_yards",),
    "rush_td":("rushing_tds",),
    "rec_yds":("receiving_yards",),
    "rec":("receptions",),
    "rec_td":("receiving_tds",),
}
PLAYER_COLS=("player_id","gsis_id")
TEAM_COLS=("team","recent_team")


def fnum(v):
    try:
        x=float(v)
        return None if math.isnan(x) else x
    except (TypeError,ValueError):
        return None


def first_text(r,names):
    for n in names:
        v=r.get(n)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def first_num(r,names):
    if r is None:
        return None
    for n in names:
        if n in r:
            x=fnum(r.get(n))
            if x is not None:
                return x
    return None


def expand(values):
    out=[]
    for v in values:
        hits=sorted(glob.glob(v))
        out.extend(hits if hits else [v])
    seen=set()
    return [x for x in out if not (x in seen or seen.add(x))]


def load_records(paths):
    out=[]
    for p0 in expand(paths):
        p=Path(p0)
        lines=[x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(lines)<3:
            raise RuntimeError(f"Invalid batch {p}: too short")
        header=json.loads(lines[0]); footer=json.loads(lines[-1])
        body=[json.loads(x) for x in lines[1:-1]]
        if not header.get("_batch") or footer.get("_footer") is not True:
            raise RuntimeError(f"Invalid batch framing {p}")
        if int(footer.get("n",-1))!=len(body):
            raise RuntimeError(f"Invalid batch count {p}")
        for r in body:
            if r.get("horizon")!="T90":
                continue
            if r.get("outcome") in OFFENSIVE_OUTCOMES:
                out.append(r)
    return out


def alt65(r):
    oc=r.get("outcome")
    if oc in YARD_OUTCOMES:
        q=r.get("quantile_grid_99")
        if not isinstance(q,list) or len(q)!=99:
            return None
        vals=[]
        for x in q:
            fx=fnum(x)
            if fx is None:
                return None
            vals.append(fx)
        start,step=(100,25) if oc=="pass_yds" else (10,10)
        upper=math.floor(vals[34]/step)*step
        best=None
        m=start
        while m<=upper:
            first=next((i for i,x in enumerate(vals) if x>=m),-1)
            prob=0.0 if first<0 else (99-first)/100.0
            if prob<0.65:
                break
            best={"milestone":m,"probability":prob}
            m+=step
        return best

    lat=r.get("cdf_lattice")
    if not isinstance(lat,dict) or not isinstance(lat.get("cdf"),list):
        return None
    step=0.5 if oc=="sacks" else 1
    if fnum(lat.get("step"))!=step:
        return None
    cdf=[fnum(x) for x in lat["cdf"]]
    if any(x is None for x in cdf):
        return None
    max_m=math.floor(len(cdf)*step)
    if max_m<=0:
        return None
    idx=max_m/step-1
    if int(idx)!=idx or idx<0 or idx>=len(cdf):
        return None
    if 1-cdf[int(idx)]>=0.65:
        return None
    best=None
    for m in range(1,max_m+1):
        i=m/step-1
        if int(i)!=i or i<0 or i>=len(cdf):
            continue
        prob=1-cdf[int(i)]
        if prob<0.65:
            break
        best={"milestone":m,"probability":prob}
    return best


def system_picks(records):
    games=defaultdict(list)
    for r in records:
        pa=fnum(r.get("p_active"))
        if pa is None or pa<0.75 or not str(r.get("player_name") or "").strip():
            continue
        a=alt65(r)
        if a is None or a["probability"]<0.65:
            continue
        rr=dict(r)
        rr["_system_target"]=a
        games[r["game_id"]].append(rr)

    chosen=[]
    for game_id,rows in sorted(games.items()):
        rows.sort(key=lambda r:(
            -r["_system_target"]["probability"],
            -float(r["p_active"]),
            str(r.get("player_name") or ""),
            str(r.get("outcome") or ""),
            str(r.get("game_id") or ""),
            str(r.get("id") or ""),
        ))
        chosen.extend(rows[:10])
    return chosen


def load_stats(path):
    official={}
    teams=set()
    with open(path,newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("season_type") or "REG")!="REG":
                continue
            try:
                s=int(r["season"]); w=int(float(r["week"]))
            except (KeyError,TypeError,ValueError):
                continue
            pid=first_text(r,PLAYER_COLS)
            team=first_text(r,TEAM_COLS)
            if pid:
                official[(s,w,pid)]=r
            if team:
                teams.add((s,w,team))
    return official,teams


def load_censors(path):
    if not path:
        return {}
    obj=json.loads(Path(path).read_text())
    out={}
    for e in obj.get("events",[]):
        out[(int(e["season"]),int(e["week"]),str(e["player_id"]))]=e
    return out


def actual_value(outcome,row):
    if row is None:
        return 0.0
    if outcome=="atd":
        return (first_num(row,("rushing_tds",)) or 0.0)+(first_num(row,("receiving_tds",)) or 0.0)
    return first_num(row,STAT_ALIASES[outcome]) or 0.0


def actual_opportunities(outcome,row):
    if row is None:
        return 0.0
    if outcome in ("pass_yds","pass_td","int"):
        return first_num(row,("attempts",)) or 0.0
    if outcome in ("rush_yds","rush_td"):
        return first_num(row,("carries",)) or 0.0
    if outcome in ("rec_yds","rec","rec_td"):
        return first_num(row,("targets",)) or 0.0
    return None


def avg(xs):
    xs=[x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None


def median(xs):
    xs=[x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def summarize(rows):
    out={}
    for oc in OFFENSIVE_OUTCOMES:
        rs=[r for r in rows if r["outcome"]==oc]
        if not rs:
            continue
        aes=[r["abs_error_median"] for r in rs]
        signed=[r["error_median"] for r in rs]
        rec={
            "n":len(rs),
            "median_projection_mae":avg(aes),
            "median_absolute_error":median(aes),
            "median_projection_bias":avg(signed),
            "mean_projection_mae":avg(r["abs_error_mean"] for r in rs),
        }
        if oc in YARD_OUTCOMES:
            rec["within_5"]=sum(e<=5 for e in aes)/len(aes)
            rec["within_10"]=sum(e<=10 for e in aes)/len(aes)
            rec["within_15"]=sum(e<=15 for e in aes)/len(aes)
            rec["within_20"]=sum(e<=20 for e in aes)/len(aes)
            rec["within_25"]=sum(e<=25 for e in aes)/len(aes)
        else:
            rec["exact"]=sum(e==0 for e in aes)/len(aes)
            rec["within_1"]=sum(e<=1 for e in aes)/len(aes)
            rec["within_2"]=sum(e<=2 for e in aes)/len(aes)
        out[oc]=rec
    return out


def grade(picks,official,teams,censors):
    rows=[]
    for r in picks:
        s=int(r["season"]); w=int(r["week"]); pid=str(r["player_id"])
        if (s,w,r["team"]) not in teams:
            continue
        stat=official.get((s,w,pid))
        actual=actual_value(r["outcome"],stat)
        med=fnum(r.get("median"))
        mean_=fnum(r.get("mean"))
        censor=censors.get((s,w,pid))
        rows.append({
            "game_id":r["game_id"],"player_id":pid,"player":r.get("player_name"),
            "position":r.get("position"),"team":r.get("team"),"opponent":r.get("opponent"),
            "outcome":r["outcome"],"p_active":fnum(r.get("p_active")),
            "system_target":r["_system_target"]["milestone"],
            "system_target_probability":r["_system_target"]["probability"],
            "model_median":med,"model_mean":mean_,"actual":actual,
            "error_median":None if med is None else med-actual,
            "abs_error_median":None if med is None else abs(med-actual),
            "error_mean":None if mean_ is None else mean_-actual,
            "abs_error_mean":None if mean_ is None else abs(mean_-actual),
            "expected_opportunities":fnum(r.get("expected_opportunities")),
            "actual_opportunities":actual_opportunities(r["outcome"],stat),
            "censored":censor is not None,
            "censor_reason":None if censor is None else censor.get("reason"),
        })
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--forecast",action="append",required=True)
    ap.add_argument("--stats",required=True)
    ap.add_argument("--censors")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    records=load_records(a.forecast)
    picks=system_picks(records)
    official,teams=load_stats(a.stats)
    censors=load_censors(a.censors)
    rows=grade(picks,official,teams,censors)
    clean=[r for r in rows if not r["censored"]]

    by_game={}
    for gid in sorted({r["game_id"] for r in rows}):
        rs=[r for r in rows if r["game_id"]==gid]
        by_game[gid]={
            "n":len(rs),
            "clean_n":sum(not r["censored"] for r in rs),
            "yardage_median_mae":avg(r["abs_error_median"] for r in rs if r["outcome"] in YARD_OUTCOMES and not r["censored"]),
            "count_median_mae":avg(r["abs_error_median"] for r in rs if r["outcome"] not in YARD_OUTCOMES and not r["censored"]),
        }

    report={
        "schema":"nfl-week4-system-pick-projection-backtest-v1",
        "selection":"exact frontend SYSTEM PICKS logic; T90; top 10 per game",
        "grading_target":"frozen model median projection; alt target only identifies which rows were SYSTEM PICKS",
        "sportsbook_inputs_used":False,
        "raw_system_pick_rows":len(rows),
        "censored_rows":sum(r["censored"] for r in rows),
        "clean_system_pick_rows":len(clean),
        "raw_by_outcome":summarize(rows),
        "clean_by_outcome":summarize(clean),
        "by_game":by_game,
        "largest_clean_projection_misses":sorted(clean,key=lambda r:(-(r["abs_error_median"] or -1),r["game_id"],r["player"]))[:30],
        "closest_clean_projection_hits":sorted(clean,key=lambda r:((r["abs_error_median"] if r["abs_error_median"] is not None else 1e9),r["game_id"],r["player"]))[:30],
        "rows":rows,
    }
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps({
        "raw_system_pick_rows":report["raw_system_pick_rows"],
        "censored_rows":report["censored_rows"],
        "clean_system_pick_rows":report["clean_system_pick_rows"],
        "clean_by_outcome":report["clean_by_outcome"],
        "by_game":report["by_game"],
        "largest_clean_projection_misses":report["largest_clean_projection_misses"][:20],
    },indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
