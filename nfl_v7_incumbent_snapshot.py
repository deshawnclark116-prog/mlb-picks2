"""Freeze the ACTUAL published NFL numeric champion beside V7 pretip research.

This helper never rebuilds V2 using outcomes and never substitutes a model
fallback. The exact already-published GitHub production document, its SHA256,
source-generated timestamp, capture timestamp and player-game numerical
projections are stored inside the immutable V7 receipt. Source age and
nonmatching game identities are explicit reasons to withhold a comparison.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from nfl_v6_joint_opportunity_shadow import utc

MARKETS = frozenset(("rushing_yards", "receiving_yards", "passing_yards"))
MAX_SOURCE_AGE_HOURS = 36


def no_source(reason):
    return {"status": reason, "matched_eligible": False, "rows": [],
            "official_betting_predictions": False}


def capture_incumbent(path, game, *, observed_at_utc, max_source_age_hours=MAX_SOURCE_AGE_HOURS):
    """Read one public model artifact as bytes and freeze only relevant rows."""
    observed = utc(observed_at_utc)
    kickoff = utc(game["kickoff_utc"])
    if observed >= kickoff:
        raise ValueError("INCUMBENT_CAPTURE_AFTER_KICKOFF")
    source = Path(path)
    if not source.is_file():
        return no_source("NO_PRODUCTION_SOURCE_FILE")
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    try:
        doc = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return no_source("INVALID_PRODUCTION_JSON")
    if not isinstance(doc,dict):
        return no_source("INVALID_PRODUCTION_OBJECT")
    try:
        generated = utc(doc["generated_at_utc"])
    except (KeyError,TypeError,ValueError):
        return no_source("MISSING_OR_INVALID_PRODUCTION_TIMESTAMP")
    if generated > observed or generated >= kickoff:
        return no_source("PRODUCTION_SOURCE_AFTER_CAPTURE")
    age_hours = (observed-generated).total_seconds()/3600
    if age_hours > max_source_age_hours or age_hours < 0:
        return no_source("STALE_PRODUCTION_SOURCE")
    if doc.get("season") != game.get("season") or doc.get("week") != game.get("week"):
        return no_source("PRODUCTION_WRONG_SEASON_WEEK")
    schedules = [x for x in doc.get("scheduled_games",[]) if (
        x.get("home_team")==game.get("home_team")
        and x.get("away_team")==game.get("away_team")
        and x.get("kickoff_utc")==game.get("kickoff_utc"))]
    if len(schedules) != 1:
        return no_source("PRODUCTION_SCHEDULE_MISMATCH")
    rows,seen = [],set()
    for x in doc.get("picks",[]):
        if not isinstance(x,dict) or x.get("market") not in MARKETS:
            continue
        if not (x.get("team") in (game["home_team"],game["away_team"])
                and x.get("opponent") in (game["home_team"],game["away_team"])
                and x.get("team") != x.get("opponent")):
            continue
        if x.get("kickoff_utc") != game["kickoff_utc"]:
            continue
        if x.get("season") != game["season"] or x.get("week") != game["week"]:
            continue
        candidate = x.get("projected_median")
        if not isinstance(candidate,(int,float)) or isinstance(candidate,bool):
            continue
        if not math.isfinite(candidate) or candidate < 0:
            continue
        pid=str(x.get("player_id") or "").strip()
        if not pid:
            continue
        key=(x["team"],pid,x["market"])
        if key in seen:
            return no_source("DUPLICATE_PRODUCTION_PLAYER_MARKET")
        seen.add(key)
        rows.append({"team":x["team"],"opponent":x["opponent"],
             "player_id":pid,"player":x.get("player"),
             "market":x["market"],"projected_median":float(candidate),
             "model_source":str(x.get("model_source") or "UNSPECIFIED"),
             "production_generated_at_utc":generated.isoformat()})
    rows.sort(key=lambda x:(x["team"],x["player_id"],x["market"]))
    return {"status":"ORIGINAL_SAME_GAME_PUBLISHED_PRODUCTION",
            "matched_eligible":True,"source_path":str(source),
            "source_sha256":source_hash,
            "generated_at_utc":generated.isoformat(),
            "observed_at_utc":observed.isoformat(),
            "age_hours":round(age_hours,3),
            "game_id":game["game_id"],
            "season":game["season"],"week":game["week"],
            "kickoff_utc":game["kickoff_utc"],
            "rows":rows,"numeric_row_count":len(rows),
            "same_lead_time_as_v7":False,
            "bookmaker_thresholds_used_for_numeric_score":False,
            "official_betting_predictions":False}
