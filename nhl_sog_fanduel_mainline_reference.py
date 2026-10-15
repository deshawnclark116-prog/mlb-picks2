#!/usr/bin/env python3
"""Source-neutral, fail-closed NHL player SOG main-line admission.

Accept only independently captured The Odds API single-event FanDuel
'player_shots_on_goal' payloads. This is a *research* bridge, NOT a
betting model. It never manufactures a line, a dressed-player
confirmation, a price, an over/under selection or a profitable wager.

The pinned V2 count-model feed stores P(SOG>=1..5) derived AFTER its
original immutable model forecast; these are explicitly labeled posthoc,
never retroactively passed off as published historical probabilities.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone

SCHEMA = "NHL_REAL_FANDUEL_MAIN_SOG_SOURCE_RECEIPT_RESEARCH_V1"
TEAM_NAMES = {
    "ANA":"Anaheim Ducks","BOS":"Boston Bruins","BUF":"Buffalo Sabres",
    "CGY":"Calgary Flames","CAR":"Carolina Hurricanes",
    "CHI":"Chicago Blackhawks","COL":"Colorado Avalanche",
    "CBJ":"Columbus Blue Jackets","DAL":"Dallas Stars",
    "DET":"Detroit Red Wings","EDM":"Edmonton Oilers",
    "FLA":"Florida Panthers","LAK":"Los Angeles Kings",
    "MIN":"Minnesota Wild","MTL":"Montreal Canadiens",
    "NSH":"Nashville Predators","NJD":"New Jersey Devils",
    "NYI":"New York Islanders","NYR":"New York Rangers",
    "OTT":"Ottawa Senators","PHI":"Philadelphia Flyers",
    "PIT":"Pittsburgh Penguins","SEA":"Seattle Kraken",
    "SJS":"San Jose Sharks","STL":"St. Louis Blues",
    "TBL":"Tampa Bay Lightning","TOR":"Toronto Maple Leafs",
    "UTA":"Utah Mammoth","VAN":"Vancouver Canucks",
    "VGK":"Vegas Golden Knights","WPG":"Winnipeg Jets",
    "WSH":"Washington Capitals",
}
NAME_TO_TEAM = {v.casefold():k for k,v in TEAM_NAMES.items()}


class UnqualifiedSOGSource(ValueError):
    pass


def timestamp(s):
    if not isinstance(s,str):
        raise UnqualifiedSOGSource("MISSING_TIMESTAMP")
    try:
        t=datetime.fromisoformat(s.replace("Z","+00:00"))
    except ValueError as exc:
        raise UnqualifiedSOGSource("INVALID_TIMESTAMP") from exc
    if t.tzinfo is None:
        raise UnqualifiedSOGSource("NAIVE_TIMESTAMP")
    return t.astimezone(timezone.utc)


def namekey(v):
    if not isinstance(v,str):
        return ""
    s=unicodedata.normalize("NFKD",v).encode("ascii","ignore").decode().casefold()
    s=re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?","",s)
    return re.sub(r"[^a-z0-9]+"," ",s).strip()


def model_rows(board):
    if (board.get("schema") != "nhl-v2-sog-nfl-style-shadow-v1"
        or board.get("status") != "RESEARCH_ONLY_NOT_VALIDATED_FOR_BETTING"
        or board.get("not_a_betting_board") is not True
        or board.get("verified_original_hash_chain") is not True
        or board.get("verified_frozen_code_and_model_lock") is not True
        or board.get("actual_player_book_lines_verified") is not False
        or board.get("dressed_lineups_verified") is not False
        or board.get("historical_corrected_probabilities_were_not_published_pregame") is not True):
        raise UnqualifiedSOGSource("NOT_FROZEN_UNPRICED_V2_MODEL")
    names=board.get("row_fields")
    if not isinstance(names,list) or len(names)!=len(set(names)):
        raise UnqualifiedSOGSource("BAD_FORECAST_COLUMNS")
    required={"game_id","player_id","player","team","opponent","scheduled_start",
              "forecast_horizon","generated_at","cutoff_at","mean",
              "p_ge1","p_ge2","p_ge3","p_ge4","p_ge5"}
    if not required.issubset(set(names)) or "pick" in names or "line" in names:
        raise UnqualifiedSOGSource("MISSING_COUNT_MODEL_FEATURES")
    out=[]
    for horizon,rs in (board.get("forecasts") or {}).items():
        if horizon not in {"T24H","T90","T30"} or not isinstance(rs,list):
            raise UnqualifiedSOGSource("INVALID_HORIZON")
        for values in rs:
            if not isinstance(values,list) or len(values)!=len(names):
                raise UnqualifiedSOGSource("MALFORMED_MODEL_ROW")
            r=dict(zip(names,values))
            if r.get("forecast_horizon")!=horizon:
                raise UnqualifiedSOGSource("MIXED_FORECAST_HORIZON")
            out.append(r)
    return out


def _price(p):
    return (isinstance(p,int) and not isinstance(p,bool)
            and (100 <= p <= 10000 or -10000 <= p <= -100))


def _prob(p):
    return (isinstance(p,(int,float)) and not isinstance(p,bool)
            and math.isfinite(p) and 0 <= p <= 1)


def _nonnegative_count_mean(x):
    # A shot-count mean is NOT a probability. The previous gate incorrectly
    # sent any player forecast above one expected shot to the empty cohort.
    return (isinstance(x,(int,float)) and not isinstance(x,bool)
            and math.isfinite(x) and x >= 0)


def extract_main_lines(event, *, at):
    """One quote per named player. Ambiguous multiple lines are QUARANTINED."""
    if (not isinstance(event,dict) or event.get("sport_key")!="icehockey_nhl"
        or not isinstance(event.get("id"),str) or not event["id"]):
        raise UnqualifiedSOGSource("INVALID_REAL_NHL_EVENT")
    if timestamp(event.get("commence_time")) <= at:
        raise UnqualifiedSOGSource("SPORTSBOOK_EVENT_ALREADY_STARTED")
    pair={NAME_TO_TEAM.get(str(event.get(k,"")).casefold()) for k in ("home_team","away_team")}
    if None in pair or len(pair)!=2:
        raise UnqualifiedSOGSource("UNKNOWN_OR_AMBIGUOUS_NHL_TEAM")
    bookmakers=[b for b in event.get("bookmakers",[])
                if isinstance(b,dict) and b.get("key")=="fanduel"]
    if len(bookmakers)!=1:
        raise UnqualifiedSOGSource("NO_UNIQUE_FANDUEL_EVENT")
    markets=[m for m in bookmakers[0].get("markets",[])
             if isinstance(m,dict) and m.get("key")=="player_shots_on_goal"]
    if len(markets)!=1:
        raise UnqualifiedSOGSource("NO_UNIQUE_FANDUEL_MAIN_SOG_MARKET")
    market=markets[0]
    updated=timestamp(market.get("last_update"))
    if updated>at or updated>=timestamp(event["commence_time"]):
        raise UnqualifiedSOGSource("FUTURE_OR_AFTER_GAME_MARKET_TIMESTAMP")
    if (at-updated).total_seconds()>7200:
        raise UnqualifiedSOGSource("STALE_BOOK_MAIN_MARKET")
    by_name=defaultdict(list)
    for o in market.get("outcomes",[]):
        if not isinstance(o,dict) or o.get("name") not in ("Over","Under"):
            continue
        player=namekey(o.get("description"))
        point=o.get("point")
        if (not player or not isinstance(point,(int,float))
            or isinstance(point,bool) or not math.isfinite(point)
            or point<0 or point>12.5 or abs((point%1)-.5)>1e-8
            or not _price(o.get("price"))):
            continue
        by_name[player].append((o["name"],float(point),o["price"]))
    qualified={}
    excluded=Counter()
    for nm, offers in by_name.items():
        unique_points={x[1] for x in offers}
        sides={x[0] for x in offers}
        if len(offers)!=2 or len(unique_points)!=1 or sides!={"Over","Under"}:
            excluded["MULTIPLE_ALTERNATE_OR_UNPAIRED_MAIN_LINES"]+=1
            continue
        qualified[nm]={"line":offers[0][1],
                       "over_american":next(v[2] for v in offers if v[0]=="Over"),
                       "under_american":next(v[2] for v in offers if v[0]=="Under"),
                       "quote_market_updated_utc":updated.isoformat()}
    return pair,qualified,dict(excluded)


def compare_original_model_to_real_market(board, provider_events, *, captured_at_utc):
    captured=timestamp(captured_at_utc)
    forecasts=model_rows(board)
    groups=defaultdict(list)
    for r in forecasts:
        start=timestamp(r["scheduled_start"])
        issued=timestamp(r["generated_at"])
        cutoff=timestamp(r["cutoff_at"])
        if not issued<=cutoff<start or issued>captured or captured>=start:
            continue
        if r["team"] not in TEAM_NAMES or r["opponent"] not in TEAM_NAMES:
            continue
        if not _nonnegative_count_mean(r.get("mean")) or r.get("player_id") is None:
            continue
        groups[(str(r["game_id"]),r["scheduled_start"])].append(r)
    rows=[]
    rejects=Counter()
    seen_game_events=set()
    for event in provider_events:
        try:
            pair,lines,ignored=extract_main_lines(event,at=captured)
        except UnqualifiedSOGSource as exc:
            rejects[str(exc)]+=1
            continue
        rejects.update(ignored)
        start=timestamp(event["commence_time"])
        candidates=[
            (key,rr) for key,rr in groups.items()
            if abs((timestamp(key[1])-start).total_seconds())<=600
            and {rr[0]["team"],rr[0]["opponent"]}==pair
        ]
        if len(candidates)!=1:
            rejects["NO_UNIQUE_MODEL_GAME_WITH_MATCHED_REAL_START"]+=1
            continue
        (gid,scheduled),rr=candidates[0]
        if gid in seen_game_events:
            rejects["DUPLICATED_PROVIDER_EVENT_FOR_NHL_GAME"]+=1
            continue
        seen_game_events.add(gid)
        # A player may appear at T24H, T90 and T30. That is ONE player,
        # not three independent betting opportunities. Keep only the most
        # recent legal pregame forecast. Conflicting NHL IDs or teams are
        # quarantined, never resolved using postgame boxscores.
        by_name=defaultdict(list)
        for f in rr:
            by_name[namekey(f.get("player"))].append(f)
        selected=[]
        for nm,entries in by_name.items():
            owners={(str(e["player_id"]),e["team"]) for e in entries}
            horizons=[e["forecast_horizon"] for e in entries]
            if len(owners)!=1:
                rejects["AMBIGUOUS_PLAYER_NHL_ID"]+=len(entries)
                continue
            if len(horizons)!=len(set(horizons)):
                rejects["DUPLICATE_FORECAST_HORIZON_FOR_PLAYER"]+=len(entries)
                continue
            ranks={"T24H":0,"T90":1,"T30":2}
            best=max(entries,key=lambda e:(ranks[e["forecast_horizon"]],
                                           timestamp(e["generated_at"])))
            selected.append(best)
        for f in selected:
            nm=namekey(f.get("player"))
            q=lines.get(nm)
            if q is None:
                continue
            threshold=int(q["line"]+.5)
            p=f.get(f"p_ge{threshold}") if 1<=threshold<=5 else None
            if p is not None and not _prob(p):
                rejects["INVALID_POSTHOC_NB2_PROBABILITY"]+=1
                continue
            rows.append({
                "schema":SCHEMA,
                "nhl_game_id":gid,"the_odds_api_event_id":event["id"],
                "player_id":str(f["player_id"]),"player":f["player"],
                "team":f["team"],"opponent":f["opponent"],
                "model_forecast_horizon":f["forecast_horizon"],
                "original_model_generated_at":f["generated_at"],
                "original_model_source_sha256":board.get("source_ledger_sha256"),
                "original_model_expected_sog":f["mean"],
                "fan_duel_main_shots_line":q["line"],
                "fan_duel_over_american":q["over_american"],
                "fan_duel_under_american":q["under_american"],
                "model_posthoc_p_over_this_line":p,
                "model_posthoc_p_under_this_line":None if p is None else round(1-p,6),
                "original_historical_probability_labels_corrected_posthoc":True,
                "provider":"the_odds_api","bookmaker":"fanduel",
                "provider_market":"player_shots_on_goal",
                "quote_market_updated_at_utc":q["quote_market_updated_utc"],
                "observed_at_utc":captured.isoformat(),
                "scheduled_start_utc":scheduled,
                "verified_dressed_player":False,
                "official_prediction":False,
                "betting_edge_validated":False,
                "pick":None,
            })
    rows.sort(key=lambda r:(r["scheduled_start_utc"],r["nhl_game_id"],r["player_id"],r["model_forecast_horizon"]))
    return {
        "schema":SCHEMA,"status":"PROVIDER_RESPONSE_ADMISSION_RESEARCH_ONLY_NO_LIVE_CALL",
        "first_observed_at_utc":captured.isoformat(),
        "strictly_pregame_model_game_count":len(groups),
        "strictly_pregame_model_team_pairs":[{
            "game_id":key[0],"teams":sorted({rr[0]["team"],rr[0]["opponent"]}),
            "scheduled_start":key[1]}
            for key,rr in groups.items()][:5],
        "fanduel_player_main_lines_matched_to_real_model":len(rows),
        "lineup_certified":False,"official_betting_picks":0,
        "do_not_call_posthoc_thresholds_original_published_probabilities":True,
        "records":rows,
        "rejections":dict(sorted(rejects.items())),
        "no_provider_call_made_by_this_adapter":True,
    }
