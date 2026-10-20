#!/usr/bin/env python3
"""Independent, final-only ESPN player stats for recent NFL games.

Used as a current-week supplement to the nflverse weekly data, which can lag
behind the game. All source events must have ESPN completed=true and be
at least six hours after kickoff. One verified ESPN game/athlete/stat is
reused by both production projection grading and immutable V2 grading.
Unknown players are NEVER inferred to have recorded zero.
"""
import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from nfl_v2_postgame_grade_coverage import fetch_scoreboard, finals_from_scoreboards
from nfl_week6_first_seen_grade import read_boxscore, read_espn, namekey

SCHEMA = "NFL_RECENT_FINAL_ESPN_PLAYER_TRUTH_V1"

def build(recent_finals, fetch_summary=read_espn, now=None):
    now = now or datetime.now(timezone.utc)
    games = {}
    missing = {}
    for gid, g in sorted(recent_finals.items()):
        try:
            payload = fetch_summary("summary?event=" + str(g["espn_id"]))
            observed = read_boxscore(payload, g)
            records = []
            for (team, name), rec in sorted(observed.items()):
                if team not in (g["away"], g["home"]) or name != namekey(rec["player"]):
                    raise ValueError("ESPN player/team identity mismatch")
                records.append({
                    "team":team, "name_key":name, "player":rec["player"],
                    "espn_athlete_id":rec["espn_athlete_id"],
                    "stats":rec["stats"], "offense_appeared":rec["offense_appeared"],
                    "defense_appeared":rec["defense_appeared"],"groups":rec["groups"]})
            if not records:
                raise ValueError("ESPN completed boxscore has zero verified athletes")
            games[gid] = {
                **g, "season":int(gid[:4]),"week":g["week"],
                "source":"ESPN_FINAL_GAME_BOXSCORE","retrieved_at_utc":now.isoformat(),
                "players":records}
        except (KeyError, TypeError, ValueError, OSError) as exc:
            # Explicit source error, not an empty boxscore interpreted as zero.
            missing[gid] = {"espn_id":g["espn_id"], "reason":type(exc).__name__,
                            "detail":str(exc)[:200]}
    return {
        "schema":SCHEMA,"asof_utc":now.isoformat(),
        "game_final_gate":"ESPN COMPLETED and 6h after kickoff",
        "scoreboards_verified_games":len(recent_finals),
        "games_with_verified_boxscore":len(games),
        "games_missing_verifiable_boxscore":missing,
        "games":games}

def recent_days(now, n=5):
    local=now.astimezone(ZoneInfo("America/New_York")).date()
    return [local-timedelta(days=i) for i in range(n)]

def collect(now=None, days=5, get_scoreboard=fetch_scoreboard,
            fetch_summary=read_espn):
    now = now or datetime.now(timezone.utc)
    responses=[(d.isoformat(),get_scoreboard(d)) for d in recent_days(now,days)]
    finals=finals_from_scoreboards(responses,now)
    return build(finals,fetch_summary=fetch_summary,now=now)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--days",type=int,default=5)
    a=parser.parse_args()
    if a.days<1 or a.days>14:raise ValueError("Recent ESPN days must be 1..14")
    doc=collect(days=a.days)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("NFL_RECENT_FINAL_TRUTH",json.dumps({
        "verified_final_games":doc["scoreboards_verified_games"],
        "verified_boxscores":doc["games_with_verified_boxscore"],
        "source_errors":doc["games_missing_verifiable_boxscore"]},sort_keys=True))

if __name__=="__main__":main()
