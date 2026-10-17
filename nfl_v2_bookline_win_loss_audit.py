#!/usr/bin/env python3
"""NFL V2/legacy bookmaker W-L truth table from ORIGINAL saved predictions.

Never claim shadow V2 had official wager selections. Conditional V2 sides
are derived only after joining: immutable first-precutoff model median,
a bookmaker quote recorded before that specific V2 forecast cutoff,
an unambiguous player/team/game, and independently graded real outcome.

Archived LEGACY picks, if separately requested, score exactly their original
published side. Do not use alt thresholds or point-projection correctness as
a 'bet'. No unsupported market, no fabricated spread, no hindsight bet.
"""
import argparse
import json
import math
import re
import unicodedata
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

SCHEMA="NFL_V2_REAL_ARCHIVED_BOOKLINE_WIN_LOSS_RESEARCH_V1"
MAPPING={"rushing_yards":"rush_yds","receiving_yards":"rec_yds"}


def norm(x):
    s=unicodedata.normalize("NFKD",str(x or ""))
    s="".join(c for c in s if not unicodedata.combining(c)).lower()
    s=re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b","",s)
    return " ".join(re.sub(r"[^a-z ]"," ",s).split())


def when(s):
    dt=datetime.fromisoformat(str(s).replace("Z","+00:00"))
    if dt.tzinfo is None:raise ValueError("timestamp lacks timezone")
    return dt.astimezone(timezone.utc)


def grade(side,line,actual):
    if side=="OVER":return "WIN" if actual>line else "LOSS" if actual<line else "PUSH"
    if side=="UNDER":return "WIN" if actual<line else "LOSS" if actual>line else "PUSH"
    return "NO_BET"


def summary(rows):
    c=Counter(r["result"] for r in rows)
    n=c["WIN"]+c["LOSS"]
    return {"book_comparisons":len(rows),"wins":c["WIN"],"losses":c["LOSS"],
            "pushes":c["PUSH"],"win_rate_excluding_pushes_pct":
            round(100*c["WIN"]/n,1) if n else None}


def build(v2,logged,record,schedules):
    # For the counterfactual, require the same player in the same opponent
    # matchup AND an original V2 median. A quote can only be used if its
    # capture preceded the V2 cutoff (not merely kickoff).
    source_rows=defaultdict(list)
    for r in v2["all_graded_rows"]:
        market=r.get("market")
        if market not in MAPPING.values():continue
        key=(r["week"],r["team"],norm(r["player"]),market,r["horizon"])
        source_rows[key].append(r)
    schedule_idx=defaultdict(list)
    for s in schedules:
        try:
            if s["kickoff_utc"] and s["home_team"] and s["away_team"]:
                schedule_idx[frozenset((s["home_team"],s["away_team"]))].append(s["kickoff_utc"])
        except (KeyError,TypeError):continue

    books=[];excluded=Counter()
    for q in logged:
        if q.get("season")!=2026 or q.get("week") not in (4,5,6):
            continue
        if q.get("market") not in MAPPING:continue
        if str(q.get("book","")).lower() not in ("fanduel","draftkings"):
            excluded["NO_ARCHIVED_NAMED_BOOK"]+=1;continue
        try:
            line=float(q["line"]);t=when(q["logged_at"])
            if not math.isfinite(line) or line<0 or line>5000:raise ValueError()
        except (KeyError,TypeError,ValueError):
            excluded["NO_VALID_ARCHIVED_PRICE_LINE_OR_TIME"]+=1;continue
        for h in ("T24","T90"):
            key=(q["week"],q["team"],norm(q["player"]),MAPPING[q["market"]],h)
            candidates=[r for r in source_rows.get(key,[]) if
                q["opponent"] in r["game_id"].split("_")[2:]]
            if len(candidates)!=1:
                excluded["NO_UNIQUE_REAL_V2_PLAYER_GAME_"+h]+=1;continue
            p=candidates[0]
            try:
                if t>when(p["cutoff_utc"]):
                    excluded["BOOK_QUOTE_AFTER_V2_CUTOFF"]+=1;continue
                if when(p["cutoff_utc"])>=when(p["kickoff_utc"]):
                    excluded["INVALID_V2_FORECAST_CUTOFF"]+=1;continue
            except (KeyError,ValueError):
                excluded["MISSING_V2_FORECAST_CUTOFF"]+=1;continue
            model=float(p["median"]);actual=float(p["actual"])
            side="OVER" if model>line else "UNDER" if model<line else "NO_BET"
            if side=="NO_BET":
                excluded["NO_MODEL_VS_LINE_DIFFERENCE"]+=1;continue
            books.append({
                "week":p["week"],"game_id":p["game_id"],"horizon":h,
                "player":q["player"],"team":p["team"],
                "market":q["market"],"book":q["book"],"line":line,
                "original_book_quote_captured_at":q["logged_at"],
                "v2_cutoff_utc":p["cutoff_utc"],
                "v2_predicted_median":model,"actual":actual,
                "hypothetical_side_from_v2_median":side,
                "result":grade(side,line,actual),
                "research_only_not_original_v2_bet":True
            })
    uniq={};dupes=0
    for b in books:
        k=(b["week"],b["game_id"],b["horizon"],b["player"].lower(),b["market"],b["book"])
        if k in uniq:
            dupes+=1
            # Never cherry pick the better outcome from multiple market lines
            # on the same player/game. Ambiguous quotes are excluded entirely.
            uniq[k]=None
        else:uniq[k]=b
    if dupes:excluded["MULTIPLE_BOOK_QUOTES_QUARANTINED"]+=dupes
    books=[b for b in uniq.values() if b is not None]
    books.sort(key=lambda x:(x["week"],x["game_id"],x["horizon"],x["player"],x["market"]))

    # Separate actual ORIGINAL served bookmaker bets (NOT the V2 shadow).
    # Require explicit book, line, a clean saved OVER/UNDER choice, real price
    # and a published pre-kickoff timestamp. No derived side here.
    outcomes=defaultdict(list)
    for r in record["results"]:
        k=(r["season"],r["week"],r["team"],r["opponent"],
           norm(r["player"]),r["market"],str(r["pick"]))
        outcomes[k].append(r)
    legacy=[]
    for q in logged:
        if q.get("season")!=2026 or q.get("week")!=5:
            continue
        if str(q.get("book","")).lower() not in ("fanduel","draftkings"):
            continue
        match=re.fullmatch(r"(OVER|UNDER)\s+(\d+(?:\.\d+)?)",str(q.get("pick","")),re.I)
        if match is None or q.get("odds") is None:continue
        try:
            line=float(q["line"]);price=int(q["odds"]);t=when(q["logged_at"])
            if abs(float(match.group(2))-line)>1e-6 or abs(price)<100:
                continue
        except (KeyError,TypeError,ValueError):continue
        key=(q["season"],q["week"],q["team"],q["opponent"],
             norm(q["player"]),q["market"],q["pick"])
        actual=outcomes.get(key,[])
        if len(actual)!=1:
            excluded["LEGACY_NO_UNIQUE_GRADED_PLAYER"]+=1;continue
        item=actual[0]
        if item.get("actual") is None:continue
        # If schedule is absent, never count the bet as pregame.
        games=schedule_idx.get(frozenset((q["team"],q["opponent"])),[])
        if len(games)!=1 or t>=when(games[0]):
            excluded["LEGACY_NO_VERIFIED_PREGAME_QUOTE"]+=1;continue
        result=grade(match.group(1).upper(),line,float(item["actual"]))
        legacy.append({"week":q["week"],"player":q["player"],
            "team":q["team"],"opponent":q["opponent"],"market":q["market"],
            "predicted_median_from_old_serving_model":q.get("projected_median"),
            "actual":item["actual"],"book":q["book"],"line":line,
            "odds":price,"original_published_pick":q["pick"],"result":result,
            "not_v2_shadow_engine":True})
    legacy.sort(key=lambda x:(x["player"],x["market"]))

    return {
        "schema":SCHEMA,"status":"RESEARCH_ONLY_CONDITIONAL_V2_SIDES_AND_SEPARATE_REAL_LEGACY_BETS",
        "v2_official_bet_record_count":0,
        "certified_executable_bookmaker_wagers":0,
        "archived_book_line_pairing_provenance":"UNVERIFIED: legacy odds parser may have joined over and under prices at mismatched point thresholds. These are ONLY directional results against archived values, not certified main lines or executable wagers.",
        "v2_model_not_original_wager_publisher":True,
        "v2_condition":"Original V2 median relative to an archived named-book line, not an original V2 wager. Source line capture must precede V2 cutoff; quote availability at V2 cutoff UNVERIFIED.",
        "v2_hypothetical_bookline_grading":{
            "T24":summary([r for r in books if r["horizon"]=="T24"]),
            "T90":summary([r for r in books if r["horizon"]=="T90"])},
        "v2_hypothetical_player_results":books,
        "original_legacy_serving_booked_picks_not_v2":{
            "summary":summary(legacy),"results":legacy},
        "no_line_no_grading_count":dict(sorted(excluded.items())),
        "source_versions":{"model_v2":v2.get("schema"),
                           "legacy_player_results":"nfl_record.json",
                           "original_quotes":"nfl_picks_log.jsonl"},
        "caveats":["Do not conflate legacy wagers with the V2 research-only shadow",
                   "Legacy sportsbook parser did not certify paired main lines: NO EXECUTABLE-BET WIN RATE IS CLAIMED",
                   "Don't count T24 and T90 bets separately as independent evidence for the same matchup",
                   "Old book quotes are not certified still available at the later V2 generation time",
                   "Missing odds/lines and inactive players are not fabricated as wins or losses",
                   "No odds/ROI claim from V2 because it did not publish actual wagers"]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--v2",type=Path,required=True)
    ap.add_argument("--ledger",type=Path,required=True)
    ap.add_argument("--legacy-grade",type=Path,required=True)
    ap.add_argument("--schedule",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    v2=json.loads(a.v2.read_text())
    logged=[json.loads(s) for s in a.ledger.read_text().splitlines() if s.strip()]
    legacy=json.loads(a.legacy_grade.read_text())
    schedule=json.loads(a.schedule.read_text()).get("scheduled_games",[])
    out=build(v2,logged,legacy,schedule)
    a.out.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
    print("NFL_V2_BOOKLINE_WIN_LOSS",json.dumps({
        "v2":out["v2_hypothetical_bookline_grading"],
        "legacy":out["original_legacy_serving_booked_picks_not_v2"]["summary"],
        "exclusions":out["no_line_no_grading_count"]},sort_keys=True))


if __name__=="__main__":main()
