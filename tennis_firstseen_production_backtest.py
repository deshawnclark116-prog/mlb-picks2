#!/usr/bin/env python3
"""Truthful, FIRST-PUBLISHED tennis forward backtest (source-only ESPN finals).

Separate match moneyline accuracy, non-bet exact-score forecasts, MODEL-created
total-game thresholds, and originally documented BOOK totals thresholds.
Do not include a non-book projection in a betting ROI. Never infer score or
change the logged side/line from a later/current publication.
"""
import argparse
import collections
import datetime as dt
import json
import math
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

SCHEMA="TENNIS_FIRSTSEEN_PRODUCTION_TRUTH_AUDIT_V1"
GROUPING={"atp":"men's singles","wta":"women's singles"}
RESULT_BASE="https://site.api.espn.com/apis/site/v2/sports/tennis"

def norm(name):
    t=unicodedata.normalize("NFKD",str(name or "")).casefold()
    return "".join(ch for ch in t if ch.isascii() and ch.isalnum())

def moment(value):
    try:
        d=dt.datetime.fromisoformat(str(value).replace("Z","+00:00"))
        if d.tzinfo is None:raise ValueError()
        return d.astimezone(dt.timezone.utc)
    except (ValueError,TypeError,AttributeError):
        return None

def fetch(tour,date):
    url=f"{RESULT_BASE}/{tour}/scoreboard?"+urllib.parse.urlencode({"dates":date})
    last=None
    for k in range(3):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"PropEdgeFirstSeenAudit/1.0","Accept":"application/json"})
            with urllib.request.urlopen(req,timeout=25) as response:
                return json.load(response)
        except Exception as e:
            last=e
            if k<2:time.sleep(1+k*2)
    raise RuntimeError(f"{tour}/{date}: {last}")

def finalized_by_id(data,tour):
    result={}
    for event in data.get("events",[]):
        for grouping in event.get("groupings",[]):
            if str(grouping.get("grouping",{}).get("displayName","")).strip().casefold()!=GROUPING[tour]:
                continue
            for comp in grouping.get("competitions",[]):
                state=comp.get("status",{}).get("type",{})
                if state.get("state")!="post" or not state.get("completed"):continue
                cid=str(comp.get("id") or "")
                if not cid:continue
                # If a competition identifier appears more than once, skip it.
                if cid in result:result[cid]=None
                else:result[cid]=comp
    return result

def final_result(comp):
    participants=comp.get("competitors") or []
    if len(participants)!=2:return None,"FINAL_INVALID_COMPETITOR_COUNT"
    status=comp.get("status") or {}
    detail=" ".join(str(x or "") for x in (
        status.get("displayClock"),status.get("detail"),status.get("shortDetail"),
        status.get("type",{}).get("description"),status.get("type",{}).get("detail")))
    if re.search(r"\b(retired|retirement|walkover|abandoned|defaulted|cancelled|suspended)\b",detail,re.I):
        return None,"FINAL_INCOMPLETE_RETIREMENT_OR_WALKOVER"
    winners=[c for c in participants if c.get("winner") is True]
    if len(winners)!=1:return None,"FINAL_INVALID_WINNER"
    score=[]
    for c in participants:
        sets=c.get("linescores") or []
        if not sets:return None,"FINAL_SET_SCORES_ABSENT"
        vals=[]
        for v in sets:
            value=v.get("value")
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                return None,"FINAL_INVALID_GAMES"
            vals.append(value)
        score.append({"name":c.get("athlete",{}).get("displayName"),"id":c.get("athlete",{}).get("id"),
                      "sets":sum(1 for v in sets if v.get("winner") is True),
                      "games":sum(vals),"is_winner":c.get("winner") is True})
    if any(not s["name"] for s in score):return None,"FINAL_PLAYER_NAME_ABSENT"
    if len({norm(x["name"]) for x in score})!=2:return None,"FINAL_PLAYER_NAMES_COLLIDE"
    a=next(x for x in score if x["is_winner"])
    b=next(x for x in score if not x["is_winner"])
    if a["sets"] not in (2,3) or b["sets"]>=a["sets"]:
        return None,"FINAL_INCOMPLETE_OR_BAD_SET_SCORE"
    return {"winner":a["name"],"winner_sets":a["sets"],"loser_sets":b["sets"],
            "total_games":a["games"]+b["games"],"players":[x["name"] for x in score],
            "competition_start":comp.get("date")},None

def classify_line(p):
    if p.get("market")!="total_games":return "not_applicable"
    if p.get("model_source")=="model_projected_line" or p.get("unagraded") is True:
        return "model_generated_not_a_book_bet"
    if p.get("model_source")=="real_book_line" and p.get("book") and isinstance(p.get("odds"),(int,float)):
        return "archived_book_quote_not_executable_certified"
    if not p.get("model_source") and p.get("book") and isinstance(p.get("odds"),(int,float)):
        return "legacy_book_quote_provenance_incomplete"
    return "unverified_line_not_bettable"

def grade(p, comp):
    """Original saved prediction + original event ID + FINAL verified players."""
    start=moment(p.get("start_time_utc"));logged=moment(p.get("logged_at"))
    if start is None or logged is None or logged>=start:
        return None,"NO_PROOF_PICK_PUBLISHED_BEFORE_SCHEDULED_START"
    fin,reason=final_result(comp)
    if reason:return None,reason
    names={norm(p.get("player1")),norm(p.get("player2"))}
    if "" in names or len(names)!=2 or names!={norm(s) for s in fin["players"]}:
        return None,"ESPN_FINAL_PLAYERS_NOT_ORIGINAL_PICK_PLAYERS"
    provider_start=moment(fin["competition_start"])
    if provider_start is not None and logged>=provider_start:
        return None,"PICK_NOT_BEFORE_FINAL_PROVIDER_START"
    market=p.get("market");pick=str(p.get("pick") or "").strip()
    classification=classify_line(p)
    probability=p.get("model_prob")
    if not isinstance(probability,(float,int)) or not math.isfinite(probability) or not 0<=probability<=1:
        return None,"INVALID_FROZEN_PROBABILITY"
    actual=fin["winner"]+" "+str(fin["winner_sets"])+"-"+str(fin["loser_sets"])+" ("+str(fin["total_games"])+" games)"
    row={"competition_id":str(p["competition_id"]),"tour":p["tour"],"tourney":p.get("tourney"),
         "player1":p["player1"],"player2":p["player2"],"pick":pick,
         "market":market,"model_prob":probability,"logged_at":p["logged_at"],
         "scheduled_start_original":p["start_time_utc"],
         "provider_start_now":fin["competition_start"],"actual":actual,
         "winner":fin["winner"],"actual_games":fin["total_games"],
         "source":"original_firstseen_ledger_and_final_espn_singles_event_identity",
         "bet_evidence":classification,
         "surface_confirmed_original":p.get("surface_confirmed"),
         "surface_evidence_original":p.get("surface_evidence")}
    if market=="moneyline":
        m=re.fullmatch(r"(.+?)\s+ML",pick,re.I)
        if m is None or norm(m.group(1)) not in names:return None,"INVALID_ORIGINAL_MONEYLINE"
        hit=norm(m.group(1))==norm(fin["winner"])
        row["result"]="WIN" if hit else "LOSS"
    elif market=="set_betting":
        m=re.fullmatch(r"(.+?)\s+([23])-([012])",pick,re.I)
        if m is None or norm(m.group(1)) not in names:return None,"INVALID_ORIGINAL_SET_SCORE"
        hit=(norm(m.group(1))==norm(fin["winner"]) and int(m.group(2))==fin["winner_sets"]
             and int(m.group(3))==fin["loser_sets"])
        row["result"]="WIN" if hit else "LOSS"
    elif market=="total_games":
        m=re.fullmatch(r"(OVER|UNDER)\s+(\d+(?:\.\d+)?)",pick,re.I)
        if m is None:return None,"INVALID_ORIGINAL_TOTAL_GAMES"
        side,line=m.group(1).upper(),float(m.group(2))
        if p.get("line") is None or abs(float(p["line"])-line)>1e-7:
            return None,"ORIGINAL_TOTALS_LINE_MISMATCH"
        actual_games=fin["total_games"]
        row.update({"line":line,"side":side,"book":p.get("book"),
                    "decimal_odds":p.get("odds"),"predicted_mean":p.get("predicted_mean")})
        row["result"]=("PUSH" if actual_games==line else
                       ("WIN" if actual_games>line else "LOSS") if side=="OVER" else
                       ("WIN" if actual_games<line else "LOSS"))
    else:return None,"UNSUPPORTED_ORIGINAL_MARKET"
    return row,None

def stats(rows):
    c=collections.Counter(x["result"] for x in rows)
    n=c["WIN"]+c["LOSS"]
    return {"graded":len(rows),"wins":c["WIN"],"losses":c["LOSS"],"pushes":c["PUSH"],
            "hit_rate_ex_pushes_pct":round(100*c["WIN"]/n,1) if n else None,
            "mean_model_probability_pct":round(100*sum(x["model_prob"] for x in rows)/len(rows),1) if rows else None,
            "brier_selected_outcome":round(sum((x["model_prob"]-(x["result"]=="WIN"))**2 for x in rows if x["result"]!="PUSH")/n,4) if n else None}

def audit(ledger,scoreboards,*,failed_sources=None):
    results=[];excluded=collections.Counter();ids=set()
    for p in ledger:
        k=(p.get("tour"),str(p.get("competition_id")),p.get("market"))
        if k in ids:
            excluded["DUPLICATE_FIRST_SEEN_PICK_KEY"]+=1;continue
        ids.add(k)
        group=(p.get("tour"),str(p.get("espn_date")))
        if failed_sources and group in failed_sources:
            excluded["SCOREBOARD_FETCH_FAILED"]+=1;continue
        comps=scoreboards.get(group,{})
        comp=comps.get(str(p.get("competition_id")))
        if comp is None:
            excluded["NO_UNAMBIGUOUS_FINAL_ESPN_COMPETITION"]+=1;continue
        row,reason=grade(p,comp)
        if reason:excluded[reason]+=1
        else:results.append(row)
    by_market={m:stats([x for x in results if x["market"]==m])
               for m in ("moneyline","total_games","set_betting")}
    by_line={cat:stats([x for x in results if x["market"]=="total_games" and x["bet_evidence"]==cat])
             for cat in ("archived_book_quote_not_executable_certified","legacy_book_quote_provenance_incomplete",
                         "model_generated_not_a_book_bet","unverified_line_not_bettable")}
    by_tour={t:{m:stats([x for x in results if x["tour"]==t and x["market"]==m])
                for m in ("moneyline","total_games","set_betting")} for t in ("atp","wta")}
    return {"schema":SCHEMA,"status":"RETROSPECTIVE_AUDIT_OF_ORIGINAL_PREGAME_LEDGER_RESEARCH_NOT_VALIDATED_BETS",
            "generated_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
            "ledger_firstseen_entries":len(ledger),"scored":len(results),
            "excluded_by_reason":dict(sorted(excluded.items())),
            "by_market":by_market,"by_total_games_line_origin":by_line,
            "by_tour_and_market":by_tour,
            "actual_verified_book_bets":0,
            "method":"ESPN FINAL competition ID + strict ORIGINAL two-player identity, original pre-kickoff timestamp + saved original line and side. Outcomes only if final and complete. Unknown results excluded.",
            "caveat":"A published moneyline outcome without a matched historical bookmaker quote is not a certified bet. Model-projected total-games lines and exact-score model selections are NOT sportsbook wagers. Archived book quote does not establish real-time availability or settled profit.",
            "results":sorted(results,key=lambda r:(r["scheduled_start_original"],r["competition_id"],r["market"]),reverse=True)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ledger",type=Path,default=Path("docs/tennis_picks_log.jsonl"))
    ap.add_argument("--out",type=Path,default=Path("docs/tennis_firstseen_production_backtest.json"))
    a=ap.parse_args()
    ledger=[json.loads(x) for x in a.ledger.read_text().splitlines() if x.strip()]
    groups=sorted({(p["tour"],str(p["espn_date"])) for p in ledger
                   if p.get("tour") in GROUPING and p.get("espn_date")})
    scoreboards={};failed=set()
    for tour,day in groups:
        try:
            scoreboards[(tour,day)]=finalized_by_id(fetch(tour,day),tour)
            print("ESPN_FINAL_BATCH",tour,day,len(scoreboards[(tour,day)]),flush=True)
        except Exception as e:
            failed.add((tour,day))
            print("ESPN_SOURCE_FAILED",tour,day,str(e)[:170],flush=True)
    report=audit(ledger,scoreboards,failed_sources=failed)
    report["unavailable_scoreboard_groups"]=[{"tour":t,"date":d} for t,d in sorted(failed)]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print("TENNIS_FIRSTSEEN_TRUTH_SUMMARY",json.dumps({
        "ledger":report["ledger_firstseen_entries"],"graded":report["scored"],
        "by_market":report["by_market"],"by_line":report["by_total_games_line_origin"],
        "by_tour":report["by_tour_and_market"],"excluded":report["excluded_by_reason"],
        "failed_scoreboards":len(failed)},sort_keys=True),flush=True)

if __name__=="__main__":main()
