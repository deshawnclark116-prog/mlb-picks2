#!/usr/bin/env python3
"""Reproducible, frozen first-seen CFB outcome audit. Observational only.

Does not change any model, feature, calibration, eligibility, or threshold.
Scores only canonical first valid pregame selections; estimates calibration
against actual completed games without presenting a hit-rate as ROI.
"""
import argparse
import json
import math
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
from pathlib import Path

def utc(v):
    dt=datetime.fromisoformat(v.replace("Z","+00:00"))
    if dt.tzinfo is None:
        raise ValueError("missing timezone")
    return dt.astimezone(timezone.utc)

def canonical(row):
    market=str(row["market"])
    return "|".join([str(row["season"]),str(row["week"]),str(row["game_id"]),market,
        "GAME" if market.startswith("moneyline") else str(row.get("player_id")),
        "ML" if market.startswith("moneyline") else str(row.get("line"))])

def summary(rows):
    if not rows:return {"n":0}
    n=len(rows);wins=sum(x["result"]=="hit" for x in rows)
    ps=[float(x["model_prob"]) for x in rows if x.get("model_prob") is not None]
    if any(not 0<=p<=1 for p in ps):
        raise ValueError("invalid logged probability")
    exp=sum(ps); actual=[1 if x["result"]=="hit" else 0 for x in rows if x.get("model_prob") is not None]
    return {"n":n,"wins":wins,"losses":n-wins,"hit_rate":round(wins/n,4),
            "mean_reported_confidence":round(exp/len(ps),4) if ps else None,
            "expected_successes_from_probabilities":round(exp,2),
            "actual_minus_expected":round(sum(actual)-exp,2) if ps else None,
            "brier":round(sum((p-y)**2 for p,y in zip(ps,actual))/len(ps),5) if ps else None}

def audit(ledger,results,season,week,start,end):
    # First pregame by canonical key, no refresh averaging or retroactive choice.
    earlier={};invalid=0;duplicates=0
    for row in ledger:
        if row.get("season")!=season or row.get("week")!=week:
            continue
        if not row.get("game_id") or not row.get("kickoff_utc") or not row.get("logged_at"):
            invalid+=1;continue
        kickoff=utc(row["kickoff_utc"])
        if not start<=kickoff<end:continue
        when=utc(row["logged_at"])
        if when>=kickoff:
            invalid+=1;continue
        key=canonical(row)
        if key in earlier:
            duplicates+=1
            if when<utc(earlier[key]["logged_at"]): earlier[key]=row
        else:
            earlier[key]=row
    scored={}
    for row in results:
        if row.get("season")!=season or row.get("week")!=week:continue
        ko=utc(row["kickoff_utc"])
        if start<=ko<end and row.get("result") in ("hit","miss"):
            key=row["canonical_key"]
            if key in scored: raise ValueError("duplicate grader result")
            scored[key]=row
    if not set(scored).issubset(set(earlier)):
        raise ValueError("grader results not a subset of canonical pregame ledger; cannot score")
    badprob=[]
    for key,r in scored.items():
        first=earlier[key]
        if (r["pick"]!=first["pick"] or
            abs(float(r["model_prob"])-float(first["model_prob"]))>0.000001):
            badprob.append(key)
    if badprob:
        raise ValueError("record differs from first pregame ledger: "+repr(badprob[:3]))
    ungraded=[v for k,v in earlier.items() if k not in scored]
    by_market={}
    for market in sorted({r["market"] for r in scored.values()}):
        rows=[r for r in scored.values() if r["market"]==market]
        by_market[market]=summary(rows)
        if market in ("rushing_yards","passing_touchdowns","anytime_touchdowns"):
            by_market[market]["offered_line_is_not_verified"]=True
            by_market[market]["unique_thresholds"]=sorted({float(r["line"]) for r in rows})
            by_market[market]["sides"]={side:summary([r for r in rows if r["pick"].startswith(side)])
                                         for side in ("OVER","UNDER") if any(r["pick"].startswith(side) for r in rows)}
        if market in ("rushing_yards","receiving_yards","passing_yards"):
            by_market[market]["numerical_point_forecasts_logged"]=sum(
                any(isinstance(earlier[r["canonical_key"]].get(k), (int,float))
                    for k in ("projected","projected_median","projected_mean"))
                for r in rows)
    ml=[r for r in scored.values() if r["market"]=="moneyline"]
    strong=[r for r in ml if float(r["model_prob"])>=0.75]
    def miss(x):
        return {"player":x["player"],"opponent":x["opponent"],"market":x["market"],
                "model_prob":x["model_prob"],"pick":x["pick"],"actual":x["actual"],
                "kickoff_utc":x["kickoff_utc"],"logged_at":x["logged_at"]}
    return {"season":season,"week":week,"window_utc":[start.isoformat(),end.isoformat()],
      "method":"first valid pregame immutable ledger entry; match exact canonical grader key",
      "distinct_pregame_ledger":len(earlier),
      "graded_predictions":len(scored),
      "not_graded_predictions":len(ungraded),
      "ungraded_by_market":dict(Counter(x["market"] for x in ungraded)),
      "invalid_or_postkickoff_ledger_rows":invalid,"duplicate_valid_ledger_rows":duplicates,
      "overall":summary(list(scored.values())),"by_market":by_market,
      "moneyline_75_plus":summary(strong),
      "moneyline_75_to_85":summary([r for r in ml if .75<=r["model_prob"]<.85]),
      "moneyline_85_plus":summary([r for r in ml if r["model_prob"]>=.85]),
      "highest_probability_moneyline_misses":[miss(r) for r in sorted(
          (r for r in ml if r["result"]=="miss"),key=lambda x:-x["model_prob"])[:20]],
      "grading_limitations":[
          "Outcomes measure OVER/UNDER static thresholds, NOT error in continuous rushing or receiving yards",
          "No FanDuel main prop, price, vig or closing line verified in this feed; hit rate is NOT profitability",
          "No-stat-line after final not counted as loss; missing outcomes can bias the graded cohort",
          "Retrospective results are diagnostic and already consulted; cannot tune or promote model on this sample",
          "CFB context historical validation uses current-game participant visibility for vacated_car/vacated_rec/qb_change; prospective serving sets these to zero",
          "Historical versus serving context inputs must be independently audited before claiming leak-safe 2025 validation",
      ],
      "decision":"FORENSIC_ONLY_NO_MODEL_OR_CALIBRATION_CHANGES"}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ledger",default="docs/cfb_picks_log.jsonl")
    p.add_argument("--record",default="docs/cfb_record.json")
    p.add_argument("--season",type=int,default=2026)
    p.add_argument("--week",type=int,default=6)
    p.add_argument("--date-et",default="2026-10-10")
    p.add_argument("--out",default=None)
    a=p.parse_args()
    # October 10, 2026 is daylight saving in US/Eastern (UTC-4).
    # Force explicit date+window, never infer user's timezone.
    from zoneinfo import ZoneInfo
    d=datetime.fromisoformat(a.date_et)
    local=datetime(d.year,d.month,d.day,tzinfo=ZoneInfo("America/New_York"))
    start=local.astimezone(timezone.utc)
    end=(local+timedelta(days=1)).astimezone(timezone.utc)
    ledger=[json.loads(x) for x in Path(a.ledger).read_text().splitlines() if x.strip()]
    results=json.loads(Path(a.record).read_text())["results"]
    r=audit(ledger,results,a.season,a.week,start,end)
    if a.out:
        dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(r,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
