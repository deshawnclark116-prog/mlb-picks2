"""Fail-closed NFL *game-day* publication contract, independent of T24/T90.

No forecast engine imports, mutations, bookmaker inputs or silent schedule fills.
Validates generated JSON against actual NFL foundation game rows before any
shared All Sports workflow is permitted to publish these NFL-specific files.
"""
import argparse
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from nfl_pregame_delivery import as_utc, schedule_manifest


def validate(doc, db=None, now=None):
    if doc.get("season") is None and doc.get("week") is None:
        if doc.get("picks"):
            raise ValueError("OUT_OF_SEASON_NOT_EMPTY")
        return {"status":"OUT_OF_SEASON","scheduled":0,"future":0,"predictions":0}
    if not isinstance(doc.get("season"),int) or not isinstance(doc.get("week"),int):
        raise ValueError("SEASON_WEEK_MISSING")
    generated=as_utc(doc.get("generated_at_utc"))
    current=as_utc(now or datetime.now(timezone.utc))
    if generated>current or (current-generated).total_seconds()>7200:
        raise ValueError("PUBLICATION_FUTURE_OR_STALE")
    raw=doc.get("scheduled_games")
    if not isinstance(raw,list) or not raw:
        raise ValueError("SCHEDULE_MANIFEST_MISSING")
    games=schedule_manifest([(g["home_team"],g["away_team"],g["kickoff_utc"]) for g in raw])
    by_pair={frozenset((g["home_team"],g["away_team"])):as_utc(g["kickoff_utc"]) for g in games}
    if db:
        with sqlite3.connect(str(db)) as con:
            rows=con.execute("SELECT home_team,away_team,kickoff_utc FROM games WHERE season=? AND week=?",
                             (doc["season"],doc["week"])).fetchall()
        expected=schedule_manifest(rows)
        if games!=expected:
            raise ValueError("PUBLISHED_SCHEDULE_DIFFERS_FROM_FOUNDATION")
    seen=set();by_game=Counter();numeric=0;td=0
    picks=doc.get("picks")
    if not isinstance(picks,list):
        raise ValueError("PICKS_SCHEMA_INVALID")
    for p in picks:
        if not isinstance(p,dict):
            raise ValueError("PICK_NOT_OBJECT")
        team,opp=p.get("team"),p.get("opponent")
        pair=frozenset((team,opp))
        if not isinstance(team,str) or not isinstance(opp,str) or pair not in by_pair:
            raise ValueError("PICK_GAME_NOT_SCHEDULED")
        kickoff=by_pair[pair]
        if as_utc(p.get("kickoff_utc"))!=kickoff:
            raise ValueError("PICK_KICKOFF_MISMATCH")
        # An output written at or after kickoff cannot claim to be pregame,
        # even if an earlier run once projected the same player.
        if kickoff<=generated:
            raise ValueError("LATE_PREDICTION_OR_BACKFILL")
        identity=(p.get("market"),p.get("player_id") or p.get("player"),p.get("team"))
        if not all(identity) or identity in seen:
            raise ValueError("INVALID_OR_DUPLICATED_PLAYER_MARKET")
        seen.add(identity)
        by_game[pair]+=1
        if p.get("market") in ("rushing_yards","receiving_yards"):
            lo,med,hi=(p.get(k) for k in ("projected_low","projected_median","projected_high"))
            if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (lo,med,hi)) or not (0<=lo<=med<=hi):
                raise ValueError("INVALID_NUMERICAL_PROJECTION")
            numeric+=1
        if p.get("market")=="anytime_touchdowns":
            prob=p.get("model_prob")
            if not isinstance(prob,(int,float)) or not math.isfinite(prob) or not (0<=prob<=1):
                raise ValueError("INVALID_TD_PROBABILITY")
            td+=1
    future=[pair for pair,kick in by_pair.items() if kick>generated]
    empty=[sorted(pair) for pair in future if by_game[pair]==0]
    return {"status":"VALID_PREGAME_PUBLICATION","season":doc["season"],"week":doc["week"],
            "generated_at_utc":generated.isoformat(),"scheduled":len(games),"future":len(future),
            "predictions":len(picks),"numeric":numeric,"anytime_td":td,"games_without_eligible_picks":empty}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--file",required=True)
    ap.add_argument("--db")
    args=ap.parse_args()
    doc=json.loads(Path(args.file).read_text())
    try:
        result=validate(doc,args.db)
    except (ValueError,KeyError,TypeError,sqlite3.Error) as err:
        print(json.dumps({"status":"FAIL","reason":str(err)},sort_keys=True),flush=True)
        raise SystemExit(1) from err
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
