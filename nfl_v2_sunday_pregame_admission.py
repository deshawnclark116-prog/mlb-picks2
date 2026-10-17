#!/usr/bin/env python3
"""Sunday NFL V2 analyst-readiness audit on genuine archived pregame output.

Do not mutate the frozen Phase1E forecaster; veto unsupported projections
using independent, timestamped roster/role evidence and team QB coherence.
This is NOT a calibrated win-probability estimator or a bookmaker product.
ESPN roster listings do not prove starter, availability at kickoff, snaps,
routes, or sportsbook market availability. No original picks are backfilled.
"""
from __future__ import annotations
import argparse
import json
import math
import unicodedata
import re
from collections import Counter,defaultdict
from datetime import datetime,timedelta,timezone
from pathlib import Path

SCHEMA="NFL_V2_SUNDAY_PREGAME_ROLE_AND_QB_READINESS_V1"
YARDS={"rush_yds":"rushing_yards","rec_yds":"receiving_yards"}
QB_PLAUSIBLE_ACTIVE=0.60
QB_MIN_MEDIAN_FOR_COMPETING_ROLE=20.0
QB_PASS_VOLUME_CAUTION=120.0
SCHEDULE_HOURS=48
ROLE_MAX_AGE_HOURS=30
MKT_MAX_AGE_MIN=30


def utc(value):
    date=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if date.tzinfo is None:raise ValueError("NAIVE_TIME")
    return date.astimezone(timezone.utc)


def namekey(value):
    value=unicodedata.normalize("NFKD",str(value or ""))
    value="".join(v for v in value if not unicodedata.combining(v))
    value=re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b","",value.lower())
    return " ".join(re.sub("[^a-z ]"," ",value).split())


def optional_number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        return None
    return float(value)


def admitted_source(board):
    s=board.get("status") or {}
    return bool(s.get("readiness_ready") is True and
                s.get("publication_state")=="PUBLISHED" and
                not s.get("verification_errors") and
                board.get("row_fields") and board.get("forecasts") and board.get("games"))


def compact(x,ix):
    return {field:x[ix[field]] for field in
            ("game_id","player","team","opp","pos","outcome","p_active",
             "mean","median","p10","p90","unc","unc_reasons")
            if field in ix}


def qb_team_certification(rows,team):
    raw=[x for x in rows if x.get("team")==team and x.get("pos")=="QB"
         and x.get("outcome")=="pass_yds"]
    plausible=sorted((x for x in raw if
                 optional_number(x.get("p_active")) is not None and
                 x["p_active"]>=QB_PLAUSIBLE_ACTIVE and
                 optional_number(x.get("median")) is not None and
                 x["median"]>=QB_MIN_MEDIAN_FOR_COMPETING_ROLE),
                 key=lambda x:(x["p_active"],x["median"]),reverse=True)
    candidates=[{"player":x["player"],"model_p_active":x["p_active"],
                 "model_pass_yards_median":round(x["median"],1)} for x in plausible]
    if not plausible:
        state="BLOCK_NO_PLAUSIBLE_MODEL_QUARTERBACK"
    elif len(plausible)>1:
        state="BLOCK_COMPETING_MODEL_QB_ROLES"
    elif plausible[0]["median"]<QB_PASS_VOLUME_CAUTION:
        state="BLOCK_IMPLAUSIBLY_LOW_MODEL_QB_PASS_VOLUME"
    else:
        state="UNIQUE_MODEL_QB_CANDIDATE_NOT_VERIFIED_STARTER"
    return {"team":team,"state":state,"model_qb_candidates":candidates,
            "all_model_qb_rows":len(raw),
            "starting_qb_independently_certified":False,
            "supports_automatic_offensive_bets":False}


def audit(board,role,*,now):
    current=utc(now.isoformat()) if isinstance(now,datetime) else utc(now)
    if not admitted_source(board):
        raise ValueError("UNVERIFIED_OR_UNPUBLISHED_V2_SHADOW")
    ix={v:i for i,v in enumerate(board["row_fields"])}
    required={"game_id","player","team","opp","pos","outcome",
              "p_active","median","mean"}
    if not required<=set(ix):
        raise ValueError("MISSING_REAL_MODEL_FIELDS")
    snapshots=role.get("player_market_evidence") or []
    index=defaultdict(list)
    for rec in snapshots:
        if isinstance(rec,dict):
            index[(rec.get("team"),namekey(rec.get("player")),rec.get("market"))].append(rec)
    try:
        role_time=utc(role["captured_at_utc"])
        role_fresh=timedelta(0)<=current-role_time<=timedelta(hours=ROLE_MAX_AGE_HOURS)
    except (ValueError,TypeError,KeyError):
        role_fresh=False
    games=[];shortlist=[];warnings=Counter();rejections=Counter()
    for game in board["games"]:
        kickoff=utc(game["kickoff_utc"])
        if not current<kickoff<=current+timedelta(hours=SCHEDULE_HOURS):
            continue
        g=game["game_id"];teams=(game["home"],game["away"])
        # Prefit-derived original forecasts remain immutable. Always prefer
        # already published T90 over T24; neither may be created after cutoff.
        available=game.get("horizons") or {}
        legal=[h for h in ("T90","T24") if h in available and
               utc(available[h]["cutoff"])<=current and
               utc(available[h]["cutoff"])<kickoff]
        choice=legal[0] if legal else None
        matches=[compact(x,ix) for x in board["forecasts"].get(choice,[])
                 if choice and x[ix["game_id"]]==g]
        if choice and len(matches)!=int(available[choice].get("n_records",0)):
            # Duplicate/count mismatch may indicate torn publication.
            rejections["MISMATCHED_V2_GAME_RECORD_COUNT"]+=1
            matches=[];choice=None
        team_qa={t:qb_team_certification(matches,t) for t in teams}
        team_risks=Counter()
        valid_role_n=0
        for t in teams:
            if team_qa[t]["state"].startswith("BLOCK_"):
                team_risks[team_qa[t]["state"]]+=1
        for x in matches:
            market=YARDS.get(x["outcome"])
            if not market:continue
            this_role=index.get((x["team"],namekey(x["player"]),market),[])
            flags=[]
            if not role_fresh:
                flags.append("CURRENT_ROSTER_ROLE_CAPTURE_MISSING_OR_STALE")
            if len(this_role)!=1:
                flags.append("PLAYER_ROLE_IDENTITY_NOT_UNIQUELY_MATCHED")
            else:
                obj=this_role[0]
                if obj.get("team") not in teams or obj.get("opponent") not in teams:
                    flags.append("ROLE_SNAPSHOT_OPPONENT_MISMATCH")
                if obj.get("roster_status")!="ROSTER_LISTED_NOT_STARTER_VERIFIED":
                    flags.append("PLAYER_AVAILABILITY_RISK_OR_NOT_LISTED")
                if not obj.get("role_evidence_complete_for_showcase",False):
                    flags.append("RECENT_OPPORTUNITY_NOT_ESTABLISHED")
                if not flags:valid_role_n+=1
            p=optional_number(x.get("p_active"))
            if p is None or p<0.85:
                flags.append("MODEL_PARTICIPATION_UNCERTAIN")
            if not optional_number(x.get("median")) or x["median"]<=0:
                flags.append("NO_POSITIVE_MODEL_OPPORTUNITY_MEDIAN")
            if team_qa[x["team"]]["state"].startswith("BLOCK_"):
                flags.append("TEAM_QB_MODEL_ROLE_UNRESOLVED")
            if flags:
                warnings.update(flags)
                continue
            shortlist.append({
                "game_id":g,"team":x["team"],"opponent":x["opp"],
                "player":x["player"],"market":market,
                "horizon":choice,"model_median":round(x["median"],2),
                "model_participation_probability":p,
                "role_from_live_roster_and_prior_games_only":True,
                "starter_certified":False,"calibrated_book_probability":False,
                "verified_current_main_bookline":False,
                "research_only_role_screen_passed":True,
                "eligible_as_official_bet":False})
        games.append({
            "game_id":g,"home":teams[0],"away":teams[1],
            "kickoff_utc":kickoff.isoformat(),
            "best_already_published_horizon":choice,
            "player_outcomes_from_genuine_forecast":len(matches),
            "qb_safety":team_qa,
            "qb_blocked_team_count":sum(v["state"].startswith("BLOCK_")
                                        for v in team_qa.values()),
            "roster_role_records_uniquely_supported":valid_role_n,
            "starting_roles_confirmed":False,
            "bookmaker_line_verified_for_betting":False,
            "status":"NO_VALID_PREGAME_FORECAST" if not choice
                     else "QB_COHERENCE_RISK" if team_risks
                     else "MODEL_QB_COHERENT_STARTER_NOT_VERIFIED"})
    # Old static sportsbook lines are not certifiably available at present.
    # A recent publication of the JSON is NOT a provider-issued market quote.
    # Even if old book lines happen to be numerically present, no side is
    # generated without matching current time, both prices, provider market
    # last_update, unique exact event and a calibrated P(over) model.
    quotes=board.get("lines") or []
    linesmeta=board.get("lines_meta") or {}
    try:
        quoted_time=utc(linesmeta["generated_at_utc"])
        book_age_min=round((current-quoted_time).total_seconds()/60,1)
    except (KeyError,ValueError,TypeError):
        book_age_min=None
    return {
        "schema":SCHEMA,
        "asof_utc":current.isoformat(),
        "season":board["status"].get("season"),
        "week":board["status"].get("week"),
        "original_forecast_version":board["status"].get("model_versions"),
        "source_model_mutated":False,
        "v2_pre_cutoff_delivery_reported_ready":True,
        "qb_role_evidence_rule":"QB>=0.60 activity and >=20 median; >1 competing, <120 starting-QB median caution; provisional engineering safety filter, not calibrated injury/starting model",
        "roster_source":role.get("schema"),
        "roster_capture_at_utc":role.get("captured_at_utc"),
        "roster_fresh":role_fresh,
        "starting_lineup_certification_available":False,
        "games":games,
        "research_only_role_screened_players":shortlist,
        "risk_counts":dict(sorted(warnings.items())),
        "unverified_or_missing_forecast_reasons":dict(sorted(rejections.items())),
        "archived_book_lines_visible":len(quotes),
        "archived_book_snapshot_minutes_old":book_age_min,
        "archived_lines_verified_still_offered":False,
        "fresh_provider_issued_book_lines_count":0,
        "official_bet_candidates":[],
        "official_bets_emitted":0,
        "betting_gate_reason":"NO_FRESH_UNIQUE_MAIN_LINE_PLUS_NO_VALIDATED_SIDE_CALIBRATION_OR_VERIFIED_STARTING_ROLES",
        "summary":{
            "future_games":len(games),
            "future_games_with_pregame_model":sum(bool(g["best_already_published_horizon"])
                                                  for g in games),
            "games_with_qb_coherence_blockers":sum(g["qb_blocked_team_count"]>0 for g in games),
            "offenses_with_qb_coherence_blockers":sum(g["qb_blocked_team_count"] for g in games),
            "role_screen_only_count":len(shortlist),
            "official_bet_count":0},
        "research_only":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shadow",type=Path,required=True)
    p.add_argument("--role",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=audit(json.loads(args.shadow.read_text()),
                 json.loads(args.role.read_text()),now=datetime.now(timezone.utc))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("NFL_V2_SUNDAY_ANALYST_QB_ROLE_GATE",json.dumps({
        "summary":result["summary"],"risk_counts":result["risk_counts"],
        "games":[{"game":g["game_id"],"horizon":g["best_already_published_horizon"],
                  "qb":{t:qa["state"] for t,qa in g["qb_safety"].items()}}
                 for g in result["games"]]},sort_keys=True))


if __name__=="__main__":main()
