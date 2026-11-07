"""Source-verified NFL pregame intelligence and integrity auditor.

Separates published serving forecasts, immutable T24/T90, recent workload/
roster evidence and unresolved QB rooms. Never pretends a listed roster
player is an NFL game-day starter. Does not change or hide any projection.
Unknown/unavailable information is explicitly marked unknown, not imputed.

Usage:
 python nfl_v3_pregame_intelligence.py \
   --serving docs/nfl_predictions.json \
   --shadow docs/nfl_phase1_shadow.json \
   --roles docs/nfl_role_snapshot.json \
   --qb-conflicts docs/nfl_v2_qb_role_conflicts.json \
   --out /tmp/nfl_v3_pregame_intelligence.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

SCHEMA="NFL_V3_PRETIP_INTELLIGENCE_EVIDENCE_AUDIT_V1"
YARDS={"rushing_yards","receiving_yards"}


def utc(value):
    t=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if t.tzinfo is None:
        raise ValueError("NFL_INTELLIGENCE_NAIVE_TIMESTAMP")
    return t.astimezone(timezone.utc)


def validate_document(doc, expected):
    if not isinstance(doc,dict) or not isinstance(doc.get(expected),list):
        raise ValueError("NFL_INTELLIGENCE_SOURCE_SCHEMA_INVALID")
    return doc


def build(serving,shadow,roles,conflicts,now=None,weather=None):
    now=utc(now or datetime.now(timezone.utc))
    validate_document(serving,"scheduled_games")
    validate_document(serving,"picks")
    validate_document(roles,"player_market_evidence")
    validate_document(conflicts,"role_conflicts")
    if not isinstance(shadow,dict) or not isinstance(shadow.get("games"),list):
        raise ValueError("NFL_INTELLIGENCE_SHADOW_SCHEMA_INVALID")
    if roles.get("schema")!="nfl-role-evidence-coverage-v1" or shadow.get("schema") is None:
        raise ValueError("NFL_INTELLIGENCE_UNKNOWN_UPSTREAM_SCHEMA")
    generated=utc(serving["generated_at_utc"])
    capture=utc(roles["captured_at_utc"])
    if generated>now or capture>now:
        raise ValueError("NFL_INTELLIGENCE_UPSTREAM_TIMESTAMP_IN_FUTURE")
    if utc(roles["forecast_generated_at_utc"])!=generated:
        raise ValueError("NFL_INTELLIGENCE_ROLE_EVIDENCE_WRONG_FORECAST")
    if serving["season"]!=roles["season"] or serving["week"]!=roles["week"]:
        raise ValueError("NFL_INTELLIGENCE_ROLE_EVIDENCE_WRONG_WEEK")
    seen=set()
    upcoming={}
    for g in serving["scheduled_games"]:
        home,away=g["home_team"],g["away_team"]
        kickoff=utc(g["kickoff_utc"])
        key=frozenset((home,away))
        if not home or not away or home==away or key in seen:
            raise ValueError("NFL_INTELLIGENCE_DUPLICATE_OR_INVALID_GAME")
        seen.add(key)
        if kickoff<=now:continue
        upcoming[key]={"home":home,"away":away,"kickoff_utc":kickoff.isoformat()}
    serving_by=defaultdict(list)
    for p in serving["picks"]:
        key=frozenset((p.get("team"),p.get("opponent")))
        if key not in seen or utc(p["kickoff_utc"])<=generated:
            raise ValueError("NFL_INTELLIGENCE_UNKNOWN_OR_POSTKICKOFF_SERVING_PICK")
        if key in upcoming:
            serving_by[key].append(p)
    role_by=defaultdict(list)
    for p in roles["player_market_evidence"]:
        key=frozenset((p.get("team"),p.get("opponent")))
        if key in upcoming and utc(p["kickoff_utc"])==utc(upcoming[key]["kickoff_utc"]):
            role_by[key].append(p)
    shadow_by=defaultdict(list)
    for g in shadow["games"]:
        key=frozenset((g.get("away"),g.get("home")))
        if key in upcoming and utc(g["kickoff_utc"])==utc(upcoming[key]["kickoff_utc"]):
            shadow_by[key].append(g)
    conflicts_by=defaultdict(list)
    for c in conflicts["role_conflicts"]:
        key=frozenset((c.get("team"),c.get("opponent"))) if c.get("opponent") else None
        # The published conflict audit may list team + game_id, without an
        # opponent property. The source game ID is matched explicitly below.
        for gamekey in upcoming:
            sg=shadow_by.get(gamekey,[])
            if c.get("team") in gamekey and any(g.get("game_id")==c.get("game_id") for g in sg):
                if c.get("horizon") in {"T24","T90"}:
                    conflicts_by[gamekey].append({
                        "team":c["team"],"horizon":c["horizon"],
                        "flags":c.get("flags") or [],
                        "type":"PUBLISHED_UNVERIFIED_QB_ROOM_CONFLICT",
                    })
                break
    weather_by={}
    if weather is not None:
        if weather.get("schema")!="NFL_V3_PRETIP_WEATHER_SOURCE_RECEIPT_V1" or not isinstance(weather.get("games"),list):
            raise ValueError("NFL_INTELLIGENCE_INVALID_WEATHER_SOURCE")
        if utc(weather["observed_at_utc"])>now:
            raise ValueError("NFL_INTELLIGENCE_WEATHER_RECEIVED_IN_THE_FUTURE")
        for wr in weather["games"]:
            key=frozenset((wr.get("home"),wr.get("away")))
            if key in upcoming and utc(wr["kickoff_utc"])==utc(upcoming[key]["kickoff_utc"]):
                if key in weather_by:
                    raise ValueError("NFL_INTELLIGENCE_DUPLICATE_WEATHER_GAME")
                weather_by[key]=wr
    game_rows=[]
    for key,meta in sorted(upcoming.items(),key=lambda item:(item[1]["kickoff_utc"],item[1]["home"],item[1]["away"])):
        picks=serving_by[key]
        evidence=role_by[key]
        shadow_games=shadow_by[key]
        if len(shadow_games)>1:
            raise ValueError("NFL_INTELLIGENCE_DUPLICATE_SHADOW_GAME")
        sgame=shadow_games[0] if shadow_games else {}
        if sgame and utc(sgame["kickoff_utc"])<=generated:
            raise ValueError("NFL_INTELLIGENCE_POSTKICKOFF_SHADOW_GAME")
        rcounts=Counter(p.get("roster_status","UNLISTED") for p in evidence)
        cautions=[{
            "player":p["player"],"team":p["team"],
            "market":p["market"],"roster_status":p["roster_status"],
            "warnings":p.get("warnings") or []
        } for p in evidence if p.get("roster_status")!="ROSTER_LISTED_NOT_STARTER_VERIFIED"]
        complete=sum(p.get("role_evidence_complete_for_showcase") is True and
                     not p.get("warnings") and
                     p.get("roster_status")=="ROSTER_LISTED_NOT_STARTER_VERIFIED"
                     for p in evidence)
        evidence_yards=Counter((p.get("player"),p.get("team"),p.get("market"))
                               for p in evidence)
        if any(n>1 for n in evidence_yards.values()):
            raise ValueError("NFL_INTELLIGENCE_DUPLICATE_ROLE_EVIDENCE")
        served_yards=[p for p in picks if p.get("market") in YARDS]
        matched=[p for p in served_yards if
                 evidence_yards[(p.get("player"),p.get("team"),p.get("market"))]==1]
        all_horizons={h:{
            "forecast_present":h in sgame.get("horizons",{}),
            "forecast_record_count":sgame.get("horizons",{}).get(h,{}).get("n_records",0),
            "game_start_verified":False,
            "conditioned_on_confirmed_starter":False,
        } for h in ("T24","T90")}
        # T90 inactivity does not prove that the remaining players start.
        weather_row=weather_by.get(key)
        weather_summary=None
        if weather_row is not None:
            state=weather_row.get("status")
            if state=="CITY_CENTRE_WEATHER_CAPTURED_NOT_STADIUM_SENSOR":
                received=utc(weather_row["last_provider_response_received_at_utc"])
                if received>=utc(meta["kickoff_utc"]) or received>now:
                    raise ValueError("NFL_INTELLIGENCE_WEATHER_RECEIPT_NOT_PRETIP")
                if weather_row.get("location_precision")!="CITY_CENTRE_APPROXIMATION_NOT_VENUE_COORDINATE":
                    raise ValueError("NFL_INTELLIGENCE_WEATHER_SOURCE_PRECISION_MISLABELED")
                hours=weather_row.get("hourly_forecast") or []
                if not hours:
                    raise ValueError("NFL_INTELLIGENCE_WEATHER_HOURLY_SOURCE_EMPTY")
                weather_summary={
                    "status":state,
                    "source_provenance":"CITY_CENTRE_NOT_STADIUM",
                    "venue":weather_row.get("venue"),
                    "wind_speed_mph_max":max(float(v["wind_speed_10m"]) for v in hours),
                    "wind_gust_mph_max":max(float(v["wind_gusts_10m"]) for v in hours),
                    "precip_probability_pct_max":max(float(v["precipitation_probability"]) for v in hours),
                    "provider_retrieved_at_utc":received.isoformat(),
                    "model_impact":"NONE_SHADOW_WEATHER_RESEARCH",
                }
            else:
                weather_summary={"status":"WEATHER_EVIDENCE_UNAVAILABLE",
                                 "reason":str(weather_row.get("reason","SOURCE_UNAVAILABLE"))[:150],
                                 "model_impact":"NONE_SHADOW_WEATHER_RESEARCH"}
        flags=[]
        if weather_row is None or weather_summary is None or weather_summary["status"]!="CITY_CENTRE_WEATHER_CAPTURED_NOT_STADIUM_SENSOR":
            flags.append("NO_VALID_PRETIP_CITY_WEATHER_SOURCE")
        if not all_horizons["T24"]["forecast_present"]:flags.append("NO_IMMUTABLE_T24_FORECAST")
        if not all_horizons["T90"]["forecast_present"]:flags.append("NO_IMMUTABLE_T90_FORECAST_YET")
        if conflicts_by[key]:flags.append("UNRESOLVED_QB_ROLE_CONTRADICTION")
        if cautions:flags.append("INJURY_OR_ROSTER_STATUS_CAUTION")
        if len(matched)<len(served_yards):flags.append("PUBLISHED_YARDS_WITHOUT_MATCHED_ROLE_EVIDENCE")
        if any(p.get("has_verified_starting_role") is True for p in evidence):
            # Future provider improvements must prove the actual source, not
            # merely set true by assumption.
            flags.append("STARTER_CLAIM_NOT_INDEPENDENTLY_VERIFIED")
        game_rows.append({
            **meta,
            "serving_player_outcomes":len(picks),
            "serving_yardage_outcomes":len(served_yards),
            "served_yards_with_matched_role_evidence":len(matched),
            "role_evidence_rows":len(evidence),
            "role_evidence_complete_no_warnings":complete,
            "source_roster_status_counts":dict(sorted(rcounts.items())),
            "roster_injury_cautions":cautions,
            "published_qb_conflicts":conflicts_by[key],
            "weather_city_approximation":weather_summary,
            "shadow":all_horizons,
            "unresolved_conditions":flags,
        })
    summary={
        "games_remaining":len(game_rows),
        "games_with_serving_player_outcomes":sum(g["serving_player_outcomes"]>0 for g in game_rows),
        "serving_player_outcomes":sum(g["serving_player_outcomes"] for g in game_rows),
        "games_with_immutable_T24":sum(g["shadow"]["T24"]["forecast_present"] for g in game_rows),
        "games_with_immutable_T90":sum(g["shadow"]["T90"]["forecast_present"] for g in game_rows),
        "games_with_QB_conflicts":sum(bool(g["published_qb_conflicts"]) for g in game_rows),
        "roster_injury_cautions":sum(len(g["roster_injury_cautions"]) for g in game_rows),
        "role_evidence_complete_no_warnings":sum(g["role_evidence_complete_no_warnings"] for g in game_rows),
        "role_evidence_rows":sum(g["role_evidence_rows"] for g in game_rows),
        "games_with_pretip_city_weather":sum(
            bool(g["weather_city_approximation"] and
                 g["weather_city_approximation"]["status"]=="CITY_CENTRE_WEATHER_CAPTURED_NOT_STADIUM_SENSOR")
            for g in game_rows),
        "serving_yardage_rows":sum(g["serving_yardage_outcomes"] for g in game_rows),
    }
    return {
        "schema":SCHEMA,
        "observed_at_utc":now.isoformat(),
        "serving_generated_at_utc":generated.isoformat(),
        "roster_evidence_captured_at_utc":capture.isoformat(),
        "serving_age_hours":round((now-generated).total_seconds()/3600,3),
        "roster_evidence_age_hours":round((now-capture).total_seconds()/3600,3),
        "snapshot_is_only_model_versioned_source":True,
        "weather_venue_wind_rain_verified":False,
        "weather_city_source_included":weather is not None,
        "game_state_model_forward_calibrated":False,
        "new_model_prediction_changes":False,
        "sportsbook_inputs_used":False,
        "unverified_starters_are_not_treated_as_confirmed":True,
        "summary":summary,
        "games":game_rows,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--serving",required=True)
    ap.add_argument("--shadow",required=True)
    ap.add_argument("--roles",required=True)
    ap.add_argument("--qb-conflicts",required=True)
    ap.add_argument("--weather",help="Optional pretip city-centre weather receipt, not a stadium forecast")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    args=[json.loads(Path(p).read_text()) for p in
          (a.serving,a.shadow,a.roles,a.qb_conflicts)]
    weather=json.loads(Path(a.weather).read_text()) if a.weather else None
    result=build(*args,weather=weather)
    target=Path(a.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("NFL_V3_PREGAME_INTELLIGENCE_AUDIT",json.dumps(result["summary"],sort_keys=True))


if __name__=="__main__":
    main()
