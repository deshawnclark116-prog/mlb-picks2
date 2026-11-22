#!/usr/bin/env python3
"""NFL player-by-player ORIGINAL numeric forecast accuracy ledger.

Input is ONLY actual first-seen frozen NFL picks and completed-game stats.
This corrects a core feedback gap: the existing binary OVER/UNDER grader
cannot grade "Projected 30 yards" and was never measuring those residuals.

A model predicted a mean/median BEFORE the game -> grade actual AFTER game.
We do not replace / modify first-seen projections, rewrite historical picks,
infer missing players as zeros, invent probability, or silently match names
across different NFL teams. No automatic formula tuning or bet promotion.

Run:
python nfl_numeric_forward_grade.py --db nfl_models/nfl_model.sqlite \
 --log docs/nfl_picks_log.jsonl --out docs/nfl_numeric_forward_grade.json
"""
from __future__ import annotations
from collections import Counter,defaultdict
from datetime import datetime,timezone
import argparse,json,math,re,sqlite3,unicodedata
from pathlib import Path

SCHEMA="NFL_NUMERICAL_FIRST_SEEN_FORWARD_GRADE_V1"
MARKET_COLUMNS={
    "rushing_yards":"rushing_yards",
    "receiving_yards":"receiving_yards",
    "passing_yards":"passing_yards",
    "receptions":"receptions",
    "rushing_attempts":"carries",
    "passing_completions":"completions",
}
UTC=timezone.utc

def utc(value):
    if not isinstance(value,str) or not value.strip():raise ValueError("MISSING_SOURCE_TIME")
    t=datetime.fromisoformat(value.replace("Z","+00:00"))
    if t.tzinfo is None:raise ValueError("NAIVE_FORECAST_OR_KICKOFF")
    return t.astimezone(UTC)

def name_key(s):
    s=unicodedata.normalize("NFKD",str(s or ""))
    s="".join(c for c in s if not unicodedata.combining(c)).lower()
    s=re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b","",s)
    return re.sub(r"\s+"," ",re.sub(r"[^a-z ]","",s)).strip()

def first_seen_records(log_path,now):
    original=[]
    errors=Counter()
    seen=set()
    if not log_path.exists():
        return original,{"MISSING_SOURCE_LEDGER":1}
    for line in log_path.read_text().splitlines():
        if not line.strip():continue
        try:p=json.loads(line)
        except json.JSONDecodeError:
            errors["INVALID_SOURCE_JSON"]+=1
            continue
        if not isinstance(p,dict):
            errors["NON_OBJECT"]+=1;continue
        market=p.get("market")
        if market not in MARKET_COLUMNS:continue
        if p.get("season") is None or p.get("week") is None or not p.get("team") or not p.get("player_id"):
            errors["INCOMPLETE_IDENTITY"]+=1;continue
        # A first-seen pregame event must contain its ORIGINAL logged timestamp.
        try:
            freeze=utc(p.get("logged_at"))
            kickoff=utc(p.get("kickoff_utc"))
        except (ValueError,TypeError):
            errors["UNVERIFIED_OR_MISSING_FIRST_SEEN_TIME"]+=1
            continue
        if freeze>=kickoff or freeze>now:
            errors["AFTER_KICKOFF_OR_FUTURE_FIRST_SEEN"]+=1
            continue
        med=p.get("projected_median")
        mean=p.get("projected_mean")
        if not isinstance(med,(int,float)) or isinstance(med,bool) or not math.isfinite(med) or med<0:
            errors["MISSING_NUMERIC_MEDIAN"]+=1;continue
        if not isinstance(mean,(int,float)) or isinstance(mean,bool) or not math.isfinite(mean) or mean<0:
            errors["MISSING_NUMERIC_MEAN"]+=1;continue
        key=(int(p["season"]),int(p["week"]),market,str(p["player_id"]),str(p["team"]))
        if key in seen:
            errors["DUPLICATE_FIRST_SEEN_SKIPPED"]+=1;continue
        seen.add(key)
        original.append({
            "season":key[0],"week":key[1],"market":key[2],
            "player_id":key[3],"team":key[4],
            "player":p.get("player"),"opponent":p.get("opponent"),
            "kickoff_utc":kickoff.isoformat().replace("+00:00","Z"),
            "first_seen_at_utc":freeze.isoformat().replace("+00:00","Z"),
            "model_source":p.get("model_source"),
            "median_original":float(med),"mean_original":float(mean),
            "source_probability_tag":p.get("projection_probability_source"),
            "first_seen_has_no_actual":True,
        })
    return original,dict(errors)

def actual_index(con,keys):
    """One week's real stats, identity-first, with unique SAME TEAM name
    fallback only for documented ESPN-vs-GSIS identifier differences."""
    exact={}
    same_team_name=defaultdict(list)
    table_cols={r[1] for r in con.execute("PRAGMA table_info(player_games)")}
    required={"player_id","player_name","team","season","week"}|set(MARKET_COLUMNS.values())
    if not required.issubset(table_cols):
        raise RuntimeError("NFL_NUMERIC_GRADER_REQUIRED_PLAYER_GAMES_COLUMNS_MISSING")
    for season,week in sorted({(r["season"],r["week"]) for r in keys}):
        for p in con.execute("""
            SELECT player_id,player_name,team,season,week,
                   rushing_yards,receiving_yards,passing_yards,
                   receptions,carries,completions
            FROM player_games WHERE season=? AND week=? AND season_type='REG'
        """,(season,week)):
            aid,pname,team,_,_,*values=p
            stats=dict(zip(MARKET_COLUMNS.keys(),values))
            observed={
                "player_id":str(aid),"player":str(pname or ""),
                "team":str(team),"season":season,"week":week,
                "actual":stats,
            }
            exact[(season,week,str(aid),str(team))]=observed
            same_team_name[(season,week,name_key(pname),str(team))].append(observed)
    return exact,same_team_name

def grade(log_path,con,now=None):
    current=now or datetime.now(UTC)
    originals,reasons=first_seen_records(Path(log_path),current)
    exact,names=actual_index(con,originals)
    outcomes=[]
    pending=Counter()
    for p in originals:
        key=(p["season"],p["week"],p["player_id"],p["team"])
        real=exact.get(key)
        basis="EXACT_GSIS_ID"
        if real is None:
            candidates=names.get((p["season"],p["week"],name_key(p["player"]),p["team"]),[])
            if len(candidates)==1:real=candidates[0];basis="UNIQUE_SAME_TEAM_NAME_FALLBACK"
            elif len(candidates)>1:
                pending["AMBIGUOUS_NAME_MATCH_NOT_SCORED"]+=1;continue
        if real is None:
            pending["NO_COMPLETED_PLAYER_GAME_OR_UNRESOLVED_ID"]+=1
            continue
        value=real["actual"][p["market"]]
        if value is None or not isinstance(value,(int,float)) or not math.isfinite(value):
            pending["ACTUAL_STAT_UNAVAILABLE"]+=1
            continue
        if float(value)<0:
            pending["INVALID_NEGATIVE_ACTUAL"]+=1;continue
        # The firstseen event's time is immutable and must always precede
        # its kickoff. Never score a postgame retrospective as a prediction.
        over=float(p["median_original"])-float(value)
        outcomes.append({
            **p,"actual":float(value),"match_basis":basis,
            "median_error_signed":round(over,3),
            "absolute_error":round(abs(over),3),
            "mean_error_signed":round(float(p["mean_original"])-float(value),3),
            "overprojected_by_10_plus":over>=10,
            "overprojected_by_20_plus":over>=20,
            "original_snapshot_immutable":True,
        })
    by_market={}
    for market in MARKET_COLUMNS:
        rows=[r for r in outcomes if r["market"]==market]
        if not rows:continue
        by_market[market]={
            "n":len(rows),
            "median_mae":round(sum(r["absolute_error"] for r in rows)/len(rows),3),
            "median_bias":round(sum(r["median_error_signed"] for r in rows)/len(rows),3),
            "overproject_10_rate":round(sum(r["overprojected_by_10_plus"] for r in rows)/len(rows),4),
            "overproject_20_rate":round(sum(r["overprojected_by_20_plus"] for r in rows)/len(rows),4),
        }
    # Repeated-miss warnings ONLY on previously completed, original
    # predictions. Same-player and same-market, chronological ordering.
    # This is diagnostic; no sample-specific penalty or auto suppression.
    by_player=defaultdict(list)
    for r in outcomes:
        by_player[(r["player_id"],r["market"])].append(r)
    at_risk=[]
    for (pid,market),rows in by_player.items():
        rows.sort(key=lambda r:(r["season"],r["week"],r["kickoff_utc"]))
        tail=rows[-3:]
        if len(tail)<3:continue
        over=sum(r["median_error_signed"]>=10 for r in tail)
        if over>=2:
            at_risk.append({
                "player_id":pid,"player":tail[-1]["player"],
                "market":market,"team":tail[-1]["team"],
                "observed_prior_games":3,"overprojected_by_10_count":over,
                "last3_median_errors":[r["median_error_signed"] for r in tail],
                "last_completed_week":tail[-1]["week"],
                "diagnostic_only":True,
            })
    at_risk.sort(key=lambda r:(-r["overprojected_by_10_count"],r["market"],r["player_id"]))
    return {
        "schema":SCHEMA,
        "status":"NUMERIC_OUTCOME_TRACKING_NO_AUTO_MODEL_CHANGE",
        "graded_at_utc":current.isoformat().replace("+00:00","Z"),
        "first_seen_count":len(originals),
        "graded_count":len(outcomes),
        "skipped_unverifiable_first_seen":reasons,
        "ungraded":dict(pending),
        "metrics_by_market":by_market,
        "recent_repeated_overprojections":at_risk,
        "results":sorted(outcomes,key=lambda r:(r["season"],r["week"],r["market"],r["player_id"])),
        "model_automatic_learning":False,
        "requires_strict_new_forward_validation_before_reweight":True,
        "no_sportsbook_prices":True,
    }

def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("--db",default="nfl_models/nfl_model.sqlite")
    arg.add_argument("--log",default="docs/nfl_picks_log.jsonl")
    arg.add_argument("--out",default="docs/nfl_numeric_forward_grade.json")
    a=arg.parse_args()
    with sqlite3.connect(f"file:{a.db}?mode=ro",uri=True) as con:
        r=grade(a.log,con)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:r[k] for k in ("schema","status","first_seen_count",
           "graded_count","metrics_by_market","ungraded")},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
