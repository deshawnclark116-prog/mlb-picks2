#!/usr/bin/env python3
"""CFB moneyline: ORIGINAL archived live probabilities versus raw-model witness.

This is *observational* after Saturday 2026-10-10, not validation of a
new side or betting strategy. The raw XGBoost score may even oppose the
chosen calibrated side, so compare probability fidelity only, never
post-hoc selection success or profitability.
"""
import argparse
import json
import math
import random
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean

SOURCE="FIRST_PREGAME_LEDGER_ENTRY"
STATUS="OBSERVATIONAL_ALREADY_CONSULTED_NO_PROMOTION"


def utc(value):
    if not isinstance(value,str):return None
    try:
        x=datetime.fromisoformat(value.replace("Z","+00:00"))
        return x.astimezone(timezone.utc) if x.tzinfo is not None else None
    except ValueError:return None


def analyze(record,ledger):
    earliest={}
    for idx,r in enumerate(ledger):
        if r.get("market")!="moneyline":continue
        logged=utc(r.get("logged_at"))
        kickoff=utc(r.get("kickoff_utc"))
        if not logged or not kickoff or logged>=kickoff:continue
        key=(r.get("season"),r.get("week"),str(r.get("game_id")))
        prev=earliest.get(key)
        if prev is None or (logged,idx)<(prev[0],prev[1]):
            earliest[key]=(logged,idx,r)
    samples=[];skips=defaultdict(int)
    for r in record.get("results",[]):
        if r.get("market")!="moneyline" or r.get("generation")!="in_season_context_v2":
            continue
        key=(r.get("season"),r.get("week"),str(r.get("game_id")))
        prior=earliest.get(key)
        if prior is None:
            skips["missing_first_pregame_ledger"]+=1;continue
        orig=prior[2]
        if (r.get("evaluation_probability_source")!=SOURCE or
            orig.get("team")!=r.get("team") or
            utc(orig.get("logged_at"))!=utc(r.get("logged_at")) or
            orig.get("model_source")!="context_v2"):
            skips["original_source_mismatch"]+=1;continue
        pred=r.get("model_prob");raw=orig.get("raw_prob")
        if not isinstance(pred,(int,float)) or not isinstance(raw,(int,float)) or (
            isinstance(pred,bool) or isinstance(raw,bool)) or not (
            math.isfinite(pred) and math.isfinite(raw) and
            0<=pred<=1 and 0<=raw<=1):
            skips["bad_probability"]+=1;continue
        if r.get("result") not in ("hit","miss"):
            skips["not_final"]+=1;continue
        y=int(r["result"]=="hit")
        samples.append({
            "game_id":str(r["game_id"]),"week":r["week"],
            "team":r["team"],"outcome":y,
            "served_context_probability":pred,"original_raw_champion_probability":raw,
            "served_brier":(pred-y)**2,"raw_brier":(raw-y)**2,
            "raw_disagreed_with_selected_side":raw<.5,
            "logged_at":r["logged_at"]
        })
    def summary(ss):
        if not ss:return {"n":0}
        n=len(ss)
        return {"n":n,
            "actual_win_rate":round(mean(x["outcome"] for x in ss),4),
            "served_mean_confidence":round(mean(x["served_context_probability"] for x in ss),4),
            "raw_mean_probability_of_served_side":round(mean(x["original_raw_champion_probability"] for x in ss),4),
            "served_brier":round(mean(x["served_brier"] for x in ss),5),
            "raw_brier_reference_not_proven":round(mean(x["raw_brier"] for x in ss),5),
            "served_minus_raw_brier":round(mean(x["served_brier"]-x["raw_brier"] for x in ss),5),
            "raw_probability_below_half_for_served_side":sum(x["raw_disagreed_with_selected_side"] for x in ss),
        }
    ci=None
    if len(samples)>=8:
        # Only one selected moneyline side per actual NFL/CFB game; game is
        # the resampling unit. No fake independently sampled player props.
        rng=random.Random(20261011)
        lifts=[]
        for _ in range(1500):
            draw=[samples[rng.randrange(len(samples))] for _ in samples]
            lifts.append(mean(x["served_brier"]-x["raw_brier"] for x in draw))
        lifts.sort()
        ci=[round(lifts[37],5),round(lifts[1462],5)]
    return {"status":STATUS,"n_2026_context_original_first_pregame":len(samples),
            "by_week":{str(w):summary([x for x in samples if x["week"]==w])
                       for w in sorted({x["week"] for x in samples})},
            "all":summary(samples),
            "paired_game_bootstrap95_for_served_minus_raw_brier":ci,
            "missing_or_invalid":dict(skips),
            "large_context_vs_raw_disagreements":[{
               "game_id":x["game_id"],"team":x["team"],"week":x["week"],
               "served":x["served_context_probability"],"raw":x["original_raw_champion_probability"],
               "won":bool(x["outcome"])} for x in samples if
               abs(x["served_context_probability"]-x["original_raw_champion_probability"])>=.15][:20],
            "limits":[
               "Original raw champion scores were NOT separately calibrated for the served side in this analysis",
               "The raw model can oppose the served side; raw scores are a diagnostic, not an alternate profitable strategy",
               "Both model outputs and 2026 outcomes have already been examined: not a blind holdout",
               "No sportsbook prices or lineup confirmation; no real-world betting EV or ROI",
               "Pairwise normalization after Platt on each team can alter calibration; pre-normalization context inputs were not archived before this instrumentation",
               "The served probabilities and real original pregame ledger are never edited or recomputed"
            ]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ledger",default="docs/cfb_picks_log.jsonl")
    p.add_argument("--record",default="docs/cfb_record.json")
    p.add_argument("--out",default="docs/cfb_moneyline_probability_audit.json")
    args=p.parse_args()
    ledger=[json.loads(l) for l in Path(args.ledger).read_text().splitlines() if l.strip()]
    result=analyze(json.loads(Path(args.record).read_text()),ledger)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","all","paired_game_bootstrap95_for_served_minus_raw_brier","by_week")},sort_keys=True))


if __name__=="__main__":
    main()
