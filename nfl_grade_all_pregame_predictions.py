#!/usr/bin/env python3
"""Grade EVERY first-seen pregame NFL *serving* prediction, not only bet cards.

Input: append-only docs/nfl_picks_log.jsonl first-seen record, genuine final
nflverse player_games AND recent independently-final ESPN boxscores.

- Numeric rush/receive projections receive actual value + signed/absolute
  model error. They do not become fabricated OVER/UNDER bets.
- Explicit OVER/UNDER decisions (including anytime TD) receive W/L/push,
  actual stat and optional probability Brier.
- Unknown identity/participation/actual or missing pregame proof remains a
  visible, per-player UNVERIFIED result; NEVER silently skipped or zero.
- This is the production serving stream. Genuine frozen T24/T90 V2
  research projections are scored separately by nfl_v2_firstseen_*.
"""
import argparse
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime,timezone,timedelta
from pathlib import Path

from nfl_grade_record_a import norm_player_name
from nfl_v2_firstseen_true_stats_backtest import utc,team

SCHEMA="NFL_FIRST_SEEN_ALL_SERVED_PREDICTIONS_GRADE_V1"
MARKETS={
    "rushing_yards":"rushing_yards",
    "receiving_yards":"receiving_yards",
    "sacks":"def_sacks",
    "anytime_touchdowns":"anytime_touchdowns",
}
SIDE=re.compile(r"(OVER|UNDER)\s+(-?\d+(?:\.\d+)?)",re.I)

def market_base(m):
    base=str(m or "").removesuffix("_early_season")
    return base if base in MARKETS else None

def real_number(x):
    try:
        n=float(x)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None

def actual_nflverse(row, base):
    if row is None:return None
    if base=="anytime_touchdowns":
        a,b=real_number(row.get("rushing_tds")),real_number(row.get("receiving_tds"))
        if a is None or b is None:return None
        return a+b
    return real_number(row.get(MARKETS[base]))

def actual_espn(player,base):
    if player is None:return None
    defense=base=="sacks"
    if defense and not player["defense_appeared"]:return None
    if not defense and not player["offense_appeared"]:return None
    d=player["stats"]
    if base=="anytime_touchdowns":
        a,b=d.get("rush_td"),d.get("rec_td")
        return float(a+b) if a is not None and b is not None else None
    return real_number(d.get({"rushing_yards":"rush_yds",
                              "receiving_yards":"rec_yds",
                              "sacks":"sacks"}[base]))

def load_db(path):
    con=sqlite3.connect("file:"+str(Path(path).resolve())+"?mode=ro",uri=True)
    con.row_factory=sqlite3.Row
    try:
        games=[dict(x) for x in con.execute(
            "SELECT game_id,season,week,season_type,kickoff_utc,home_team,away_team,"
            "home_score,away_score FROM games")]
        players=[dict(x) for x in con.execute(
            "SELECT * FROM player_games WHERE season>=2025")]
    finally:con.close()
    return games,players

def load_ledger(path):
    if not Path(path).is_file():raise ValueError("First-seen NFL pick log absent")
    rows=[]
    for i,line in enumerate(Path(path).read_text().splitlines(),1):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except json.JSONDecodeError as exc:raise ValueError(f"Corrupt first-seen ledger at {i}") from exc
    return rows

def index_source(games,players,espn):
    schedules=defaultdict(list)
    known_games={g["game_id"] for g in games}
    for g in games:
        if g["season_type"]!="REG":continue
        for t,o in ((g["away_team"],g["home_team"]), (g["home_team"],g["away_team"])):
            schedules[(g["season"],g["week"],team(t),team(o))].append(g)
    # Recent final ESPN events are independent of weekly nflverse ingestion.
    espn_games=(espn or {}).get("games") or {}
    for gid,e in espn_games.items():
        if gid != e["game_id"]:raise ValueError("Conflicting ESPN final game ID")
        if e.get("source")!="ESPN_FINAL_GAME_BOXSCORE":
            raise ValueError("Unverified ESPN truth source")
        if gid not in known_games:
            year=int(gid.split("_")[0])
            for t,o in ((e["away"],e["home"]),(e["home"],e["away"])):
                schedules[(year,e["week"],team(t),team(o))].append({
                    "game_id":gid,"season":year,"week":e["week"],
                    "season_type":"REG","kickoff_utc":e["kickoff_utc"],
                    "away_team":e["away"],"home_team":e["home"],
                    "home_score":e["home_score"],"away_score":e["away_score"]})
    # Fix as-of weekly team identity: never match a name on another team.
    byid=defaultdict(list);byname=defaultdict(list)
    for p in players:
        if p.get("season_type")!="REG" or not p.get("game_id"):continue
        k=(p["game_id"],team(p["team"]))
        byid[(k,p["player_id"])].append(p)
        byname[(k,norm_player_name(p["player_name"]))].append(p)
    ename=defaultdict(list)
    for gid,g in espn_games.items():
        for p in g["players"]:
            ename[(gid,team(p["team"]),p["name_key"])].append(p)
    return schedules,byid,byname,ename,espn_games

def grade_explicit(side,line,actual):
    if abs(actual-line)<1e-9:return "push"
    return "win" if (actual>line if side=="OVER" else actual<line) else "loss"

def score_predictions(ledger,games,players,espn=None,now=None):
    now=now or datetime.now(timezone.utc)
    schedules,byid,byname,ename,espn_games=index_source(games,players,espn)
    rows=[]; seen=set()
    for r in ledger:
        player=str(r.get("player") or "")
        base=market_base(r.get("market"))
        item={"season":r.get("season"),"week":r.get("week"),
            "game_id":None,"market":r.get("market"),"player":player,
            "player_id":r.get("player_id"),"team":r.get("team"),
            "opponent":r.get("opponent"),"model_source":r.get("model_source"),
            "original_prediction":r.get("pick"),"original_logged_at":r.get("logged_at"),
            "model_prob":r.get("model_prob"),"predicted_median":r.get("projected_median"),
            "predicted_mean":r.get("projected_mean"),
            "predicted_low":r.get("projected_low"),"predicted_high":r.get("projected_high"),
            "original_line":r.get("line"),"actual":None,
            "actual_source":None,"absolute_stat_error":None,
            "direction_grade":None,"brier_of_explicit_side":None,
            "status":None}
        def stop(reason):
            item["status"]=reason;rows.append(item)
        try:
            identity=(r["season"],r["week"],r["market"],str(r["player_id"]))
            if identity in seen:
                stop("DUPLICATE_FIRST_SEEN_KEY_QUARANTINED");continue
            seen.add(identity)
            if base is None:stop("UNSUPPORTED_OR_UNMODELED_MARKET");continue
            k=(r["season"],r["week"],team(r["team"]),team(r["opponent"]))
            selected=schedules.get(k,[])
            if len(selected)!=1:
                stop("NO_UNIQUE_INDEPENDENT_GAME_SCHEDULE");continue
            g=selected[0];gid=g["game_id"];item["game_id"]=gid
            kickoff=utc(g["kickoff_utc"])
            if not r.get("logged_at") or utc(r["logged_at"])>=kickoff:
                stop("NOT_PROVABLY_SAVED_BEFORE_KICKOFF");continue
            # The original per-card saved kickoff must match the true game
            # exactly, otherwise user-visible game/pick identity is suspect.
            if not r.get("kickoff_utc") or abs(
                 (utc(r["kickoff_utc"])-kickoff).total_seconds())>1200:
                stop("ORIGINAL_GAME_KICKOFF_MISMATCH");continue
            if now-kickoff < timedelta(hours=2):
                stop("GAME_NOT_YET_MATURE");continue
            eg=espn_games.get(gid)
            isfinal=(g.get("home_score") is not None and
                     g.get("away_score") is not None) or eg is not None
            if not isfinal:
                stop("GAME_FINAL_NOT_IN_SOURCE");continue
            # ESPN final includes the current week before nflverse weekly
            # publishes. An independently verified *player* is still required.
            name=norm_player_name(player)
            ep=ename.get((gid,team(r["team"]),name),[])
            candidate_id=byid.get(((gid,team(r["team"])),str(r["player_id"])),[])
            candidate=candidate_id if candidate_id else byname.get(
                ((gid,team(r["team"])),name),[])
            if len(ep)>1 or len(candidate)>1:
                stop("AMBIGUOUS_PLAYER_IDENTITY");continue
            ev=actual_espn(ep[0],base) if ep else None
            nv=actual_nflverse(candidate[0],base) if candidate else None
            if ev is not None and nv is not None and abs(ev-nv)>1e-8:
                stop("CONFLICTING_FINAL_STAT_SOURCES");continue
            actual=ev if ev is not None else nv
            if actual is None:
                stop("NO_VERIFIED_PLAYER_MARKET_ACTUAL_NOT_ZERO");continue
            item["actual"]=actual
            item["actual_source"]="ESPN_FINAL_BOX" if ev is not None else "NFLVERSE_PLAYER_GAMES"
            item["status"]="GRADED"
            median=real_number(r.get("projected_median"))
            mean=real_number(r.get("projected_mean"))
            if median is not None:
                item["predicted_median"]=median
                item["absolute_stat_error"]=round(abs(median-actual),4)
                item["signed_stat_error"]=round(median-actual,4)
                if mean is not None:item["absolute_mean_error"]=round(abs(mean-actual),4)
                p10,p90=real_number(r.get("projected_low")),real_number(r.get("projected_high"))
                if p10 is not None and p90 is not None:
                    item["actual_inside_p10_p90"]=p10<=actual<=p90
            elif str(r.get("pick","")).startswith("Projected "):
                item["status"]="MISSING_NUMERIC_PROJECTION";rows.append(item);continue
            m=SIDE.search(str(r.get("pick") or ""))
            if m:
                side,line=m.group(1).upper(),float(m.group(2))
                if real_number(r.get("line")) != line:
                    item["status"]="EXPLICIT_SIDE_LINE_MISMATCH";rows.append(item);continue
                item["direction_grade"]=grade_explicit(side,line,actual)
                item["graded_side"]=side;item["graded_line"]=line
                prob=real_number(r.get("model_prob"))
                if prob is not None and 0<=prob<=1:
                    y=1.0 if item["direction_grade"]=="win" else 0.0
                    if item["direction_grade"]!="push":
                        item["brier_of_explicit_side"]=round((prob-y)**2,6)
            else:
                # Pure numeric projection. Not a bet, not 0.5 +/- guess.
                item["direction_grade"]=None
            rows.append(item)
        except (KeyError,TypeError,ValueError) as exc:
            item["status"]="INVALID_FIRST_SEEN_RECORD_OR_TIME"
            item["validation_detail"]=str(exc)[:140]
            rows.append(item)
    bygame=defaultdict(list)
    for r in rows:
        bygame[(r["season"],r["week"],r["game_id"] or
             f'UNRESOLVED_{r["team"]}_{r["opponent"]}')].append(r)
    bygameout=[]
    for (season,week,gid), group in sorted(bygame.items()):
        markets={}
        for market in sorted(set(r["market"] for r in group)):
            rr=[r for r in group if r["market"]==market]
            errors=[r["absolute_stat_error"] for r in rr if r["absolute_stat_error"] is not None]
            grades=Counter(r["direction_grade"] for r in rr)
            markets[market]={
                "original_predictions":len(rr),
                "actual_verified":sum(r["status"]=="GRADED" for r in rr),
                "status_counts":dict(Counter(r["status"] for r in rr)),
                "absolute_error_mean":round(sum(errors)/len(errors),4) if errors else None,
                "numeric_predictions_scored":len(errors),
                "explicit_direction_win":grades["win"],"explicit_direction_loss":grades["loss"],
                "explicit_direction_push":grades["push"],
                "not_a_bet_if_only_numeric":True}
        bygameout.append({"game_id":gid,"season":season,"week":week,
           "original_predictions":len(group),
           "graded":sum(r["status"]=="GRADED" for r in group),
           "unverified":sum(r["status"]!="GRADED" for r in group),
           "by_market":markets})
    graded=[r for r in rows if r["status"]=="GRADED"]
    return {"schema":SCHEMA,"asof_utc":now.isoformat(),
        "scope":"ALL_FIRST_SEEN_NFL_SERVING_PREDICTIONS_NOT_JUST_PRICED_BETS",
        "source":"First-seen logged model cards + verified final nflverse player_games / recent ESPN game-box",
        "production_vs_research_separate":True,
        "original_count":len(ledger),"accounted_for":len(rows),
        "verified_outcome_count":len(graded),
        "unverified_count":len(rows)-len(graded),
        "status_counts":dict(Counter(r["status"] for r in rows)),
        "directional_wins":sum(r["direction_grade"]=="win" for r in graded),
        "directional_losses":sum(r["direction_grade"]=="loss" for r in graded),
        "directional_pushes":sum(r["direction_grade"]=="push" for r in graded),
        "by_game":bygameout,"all_predictions":rows,
        "note":"Numeric projections graded against actual stat, with no invented win/loss. Explicit originally recorded sides graded separately; neither is a certified sportsbook bet."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--db",type=Path,required=True)
    p.add_argument("--log",type=Path,required=True)
    p.add_argument("--espn-recent",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    games,players=load_db(a.db)
    espn=json.loads(a.espn_recent.read_text()) if a.espn_recent else None
    if espn and espn.get("schema")!="NFL_RECENT_FINAL_ESPN_PLAYER_TRUTH_V1":
        raise ValueError("Unverified ESPN final truth document")
    d=score_predictions(load_ledger(a.log),games,players,espn=espn)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("NFL_ALL_ORIGINAL_SERVED_PREDICTIONS",json.dumps({
        k:d[k] for k in ("original_count","accounted_for","verified_outcome_count",
        "unverified_count","directional_wins","directional_losses",
        "directional_pushes","status_counts")},sort_keys=True))
    if d["accounted_for"]!=d["original_count"]:raise SystemExit("First-seen accounting loss")
    if not d["original_count"]:raise SystemExit("No original first-seen serving predictions")

if __name__=="__main__":main()
