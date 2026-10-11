#!/usr/bin/env python3
"""CFB pregame analyst evidence: real role participation + roster-source metadata.

Non-predictive SHADOW: no change to CFB probabilities, market line, board
qualification or sportsbook picks. Opponent/game matchup is carried through,
but no future stats or injury information are manufactured.

Roster injuries are *source-reported fields only*. An empty injuries array
or an "active" roster entry NEVER certifies the future game availability.
No targets/snap/routes exist in the current CFB box-score database; past
receptions are explicitly only a lower-bound involvement proxy.
"""
import argparse
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

SCHEMA="cfb-analyst-evidence-v1"
MARKETS={"rushing_yards":"carries","rushing_yards_early_season":"carries",
         "passing_touchdowns":"pass_attempts",
         "passing_touchdowns_early_season":"pass_attempts",
         "anytime_touchdowns":None,"anytime_touchdowns_early_season":None}
SAFE_STATUS_PATTERN=re.compile(r"\b(out|doubtful|questionable|injured|injury|suspended|ir|reserve|pup|limited)\b",re.I)

def utc(s):
    if not isinstance(s,str): return None
    try:
        x=datetime.fromisoformat(s.replace("Z","+00:00"))
        return x.astimezone(timezone.utc) if x.tzinfo else None
    except ValueError:return None

def normalize(name):
    s=str(name or "").lower()
    return " ".join(re.sub(r"[^a-z0-9 ]"," ",s).split())

def roster_for(con,season):
    cols={row[1] for row in con.execute("PRAGMA table_info(current_roster)")} if con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='current_roster'").fetchone() else set()
    needed={"team","player_name","position","season","espn_athlete_id",
            "roster_status_raw","roster_injuries_json","roster_captured_at_utc"}
    if not needed<=cols: return {},False
    evidence={}
    for row in con.execute("""SELECT team,player_name,position,espn_athlete_id,
                roster_status_raw,roster_injuries_json,roster_captured_at_utc
                FROM current_roster WHERE season=?""",(season,)):
        team,name,pos,pid,status,injuries,checked=row
        try: injury=json.loads(injuries) if injuries else []
        except (TypeError,ValueError):injury=[]
        if not isinstance(injury,list):injury=[]
        key=(team,normalize(name))
        evidence.setdefault(key,[]).append({
            "espn_athlete_id":str(pid) if pid else None,
            "name":name,"position":pos,
            "source_roster_status":status,
            "source_injuries":injury,
            "roster_checked_utc":checked,
        })
    return evidence,True

def team_game_history(con,season,team,target_week,asof):
    """Only games final before BOTH the target week and the snapshot timestamp."""
    games=con.execute("""SELECT game_id,season,week,kickoff_utc
        FROM games
        WHERE season=? AND week<? AND home_points IS NOT NULL AND away_points IS NOT NULL
          AND (home_team=? OR away_team=?)
        ORDER BY week,game_id""",(season,target_week,team,team)).fetchall()
    return [g for g in games if utc(g[3]) and utc(g[3])<asof][-6:]

def observations(con,season,week,team,player_id,metric,asof):
    games=team_game_history(con,season,team,week,asof)
    recent=games[-3:]
    game_receipts=[]
    values=[]
    for gid,s,w,kickoff in recent:
        row=con.execute(f"""SELECT {metric} FROM player_games
            WHERE game_id=? AND player_id=? AND team=?""",(gid,str(player_id),team)).fetchall()
        team_rows=con.execute(f"""SELECT {metric} FROM player_games
            WHERE game_id=? AND team=?""",(gid,team)).fetchall()
        # Defensive: multiple athlete rows for a game mean non-unique identity.
        if len(row)>1:
            stat=None; source="AMBIGUOUS_PLAYER_ID"
        elif row and row[0][0] is not None:
            stat=float(row[0][0]);source="OBSERVED_BOXSCORE"
            values.append(stat)
        else:
            stat=None;source="NOT_IN_BOXSCORE_UNKNOWN_NOT_ZERO"
        total=sum(float(v[0] or 0) for v in team_rows)
        game_receipts.append({"game_id":gid,"week":w,"player_observed":source,
            "player_opportunities":stat,"team_recorded_opportunities":total,
            "share_of_recorded_team":round(stat/total,4) if stat is not None and total>0 else None})
    observed=[g for g in game_receipts if g["player_observed"]=="OBSERVED_BOXSCORE"]
    shares=[g["share_of_recorded_team"] for g in observed if g["share_of_recorded_team"] is not None]
    return {
        "metric":metric,"last_three_completed_team_games":game_receipts,
        "team_games_available":len(recent),"observed_player_boxscore_rows":len(observed),
        "unknown_or_ambiguous_appearances":len(recent)-len(observed),
        "mean_opportunities_in_observed_games":round(sum(values)/len(values),3) if values else None,
        "mean_share_when_observed":round(sum(shares)/len(shares),4) if shares else None,
        "role_ready_from_prior_games":len(recent)==3 and len(observed)==3,
        "snapshot_is_not_starting_lineup_certification":True,
    }

def injury_view(roster_row):
    if roster_row is None:
        return {"status":"ROSTER_IDENTITY_NOT_VERIFIED","has_explicit_injury_concern":False,
                "availability_certified":False}
    raw=roster_row.get("source_roster_status")
    injuries=roster_row.get("source_injuries") or []
    issue=bool(raw and SAFE_STATUS_PATTERN.search(str(raw)))
    issue|=any(SAFE_STATUS_PATTERN.search(" ".join(str(i.get(k) or "") for k in
                   ("status","type","detail"))) for i in injuries if isinstance(i,dict))
    return {"status":"EXPLICIT_SOURCE_STATUS_CONCERN" if issue
            else "ROSTER_LISTED_AVAILABILITY_UNKNOWN",
            "source_status":raw,"source_injury_details":injuries,
            "has_explicit_injury_concern":bool(issue),
            "availability_certified":False,
            "roster_checked_utc":roster_row.get("roster_checked_utc")}

def make_snapshot(con,board,clock):
    season=board["season"];week=board["week"]
    generated=utc(board["generated_at_utc"])
    if not generated or generated>clock:
        raise ValueError("missing or future board timestamp")
    roster,index_valid=roster_for(con,season)
    lookup={}
    for gid,home,away,ko in con.execute("""SELECT game_id,home_team,away_team,kickoff_utc
                                          FROM games WHERE season=? AND week=?""",(season,week)):
        lookup[(home,away)]=(gid,utc(ko))
        lookup[(away,home)]=(gid,utc(ko))
    rows=[]
    for p in board["picks"]:
        market=p.get("market")
        if market not in MARKETS:
            continue
        team,opp=p.get("team"),p.get("opponent")
        match=lookup.get((team,opp))
        if not match or not match[1] or not generated<match[1] or not clock<match[1]:
            continue
        gid,kickoff=match
        names=roster.get((team,normalize(p.get("player"))),[])
        named=names[0] if len(names)==1 else None
        pid=str(p.get("player_id") or "")
        identity=bool(named and named["espn_athlete_id"]==pid)
        position=named.get("position") if identity else None
        metric=MARKETS[market]
        if metric is None:
            # Anytime TD is RB carries or WR receptions. "receptions" is
            # NOT targets: never write target share from this source.
            metric=("carries" if position in ("RB","FB") else
                    "receptions" if position in ("WR","TE") else None)
        receipt=observations(con,season,week,team,pid,metric,clock) if metric and pid else None
        if not index_valid:
            roster_state={"status":"ROSTER_ID_SOURCE_UNAVAILABLE","availability_certified":False}
        elif len(names)!=1:
            roster_state={"status":"ROSTER_MISSING_OR_AMBIGUOUS","availability_certified":False}
        elif not identity:
            roster_state={"status":"PLAYER_ID_MISMATCH","availability_certified":False}
        else:
            roster_state=injury_view(named)
        flags=[]
        if roster_state["status"] not in ("ROSTER_LISTED_AVAILABILITY_UNKNOWN",):
            flags.append("ROSTER_IDENTITY_OR_AVAILABILITY_UNCERTAIN")
        if not receipt or not receipt["role_ready_from_prior_games"]:
            flags.append("THREE_GAME_ROLE_COVERAGE_INCOMPLETE")
        if receipt and receipt["mean_opportunities_in_observed_games"] is not None:
            mean=receipt["mean_opportunities_in_observed_games"]
            floor={"carries":5,"pass_attempts":12,"receptions":2}.get(metric)
            if floor and mean<floor:
                flags.append("LOW_RECENT_INVOLVEMENT_OBSERVATION")
        if receipt and receipt["unknown_or_ambiguous_appearances"]:
            flags.append("MISSING_BOXSCORE_IS_NOT_A_CONFIRMED_DNP")
        rows.append({
          "game_id":gid,"season":season,"week":week,"kickoff_utc":p["kickoff_utc"],
          "team":team,"opponent":opp,"player":p.get("player"),"player_id":pid,
          "market":market,"published_model_prob":p.get("model_prob"),
          "published_model_side":p.get("pick"),"published_line_is_sportsbook_verified":False,
          "roster":roster_state,"recent_usage":receipt,"evidence_warnings":flags,
          "fully_verified_active_starter":False,"calibrated_market_edge_established":False})
    return {
       "schema":SCHEMA,"season":season,"week":week,
       "board_generated_at_utc":board["generated_at_utc"],
       "evidence_captured_at_utc":clock.isoformat().replace("+00:00","Z"),
       "status":"RESEARCH_ONLY_DO_NOT_AUTO_PROMOTE",
       "has_source_roster_athlete_ids":index_valid,
       "is_current_injury_and_starter_feed_complete":False,
       "model_changed":False,
       "markets_book_verified":False,
       "player_evidence":rows,
       "evidence_count":len(rows),
       "warnings_count":sum(bool(r["evidence_warnings"]) for r in rows),
       "limitations":[
          "Historical player_games are not confirmed snap counts, routes or targets",
          "ESPN roster listing does not certify active, starter or game-time injury resolution",
          "No first-pregame numerical yard projection or calibrated line-specific betting edge exists here",
          "Missing player stat rows cannot be inferred as zero yards or a confirmed DNP",
          "No model or threshold was retrained using 2026 forward outcomes",
       ]
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--db",default="cfb_models/cfb_model.sqlite")
    ap.add_argument("--board",default="docs/cfb_predictions.json")
    ap.add_argument("--out",default="docs/cfb_pregame_evidence.json")
    args=ap.parse_args()
    board=json.loads(Path(args.board).read_text())
    db=sqlite3.connect(f"file:{args.db}?mode=ro",uri=True)
    try:
        result=make_snapshot(db,board,datetime.now(timezone.utc))
    finally:db.close()
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("season","week","evidence_count",
               "warnings_count","has_source_roster_athlete_ids","model_changed")}))
if __name__=="__main__":
    main()
