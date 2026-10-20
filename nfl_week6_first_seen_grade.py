#!/usr/bin/env python3
"""NFL Week 6: frozen T24 shadow predictions vs independently verified ESPN finals.

Only sports research: NO available pregame sportsbook prices; accuracy for
thresholds derived from a model distribution is NOT executable bet ROI.

Source: immutable pre-kickoff batches, cryptographic footer, separate ESPN
final scoreboard+per-game player boxscores. Absent/ambiguous player identities
are reported UNGRADED, not silently scored as zero.
"""
from __future__ import annotations
import argparse, collections, datetime as dt, hashlib, json, math, re, statistics
import unicodedata, urllib.request
from pathlib import Path

SCHEMA="NFL_2026_W06_SUN_MON_FROZEN_T24_ACTUAL_COMPARISON_V1"
BASE="https://site.api.espn.com/apis/site/v2/sports/football/nfl/"
OUTCOMES=("rush_yds","rec_yds","pass_yds","rec","pass_td","int",
          "rush_td","rec_td","atd","tackles","sacks","def_int")
YARDS={"rush_yds","rec_yds","pass_yds"}
COUNTED=set(OUTCOMES)-YARDS
EXPECTED={
    ("HOU","JAX"),("CHI","ATL"),("BAL","CLE"),("TEN","IND"),
    ("NYJ","NE"),("NO","NYG"),("CAR","PHI"),("PIT","TB"),
    ("ARI","LA"),("LAC","KC"),("BUF","LV"),("DAL","GB"),("WAS","SF")}
NORMALIZE_TEAM={"LAR":"LA","WSH":"WAS","JAC":"JAX"}
def club(s):return NORMALIZE_TEAM.get(str(s).upper(),str(s).upper())

def namekey(s):
    s=unicodedata.normalize("NFKD",str(s or ""))
    s="".join(c for c in s if not unicodedata.combining(c)).lower()
    s=re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b","",s)
    return re.sub(r"\s+"," ",re.sub(r"[^a-z ]"," ",s)).strip()

def iso(s):
    x=dt.datetime.fromisoformat(str(s).replace("Z","+00:00"))
    if x.tzinfo is None:raise ValueError("missing timezone")
    return x.astimezone(dt.timezone.utc)

def read_espn(path):
    req=urllib.request.Request(BASE+path,headers={
        "Accept":"application/json","User-Agent":"PropEdgeNFLOutcomeIntegrity/1.0"})
    with urllib.request.urlopen(req,timeout=24) as res:
        return json.load(res)

def get_schedule(get=read_espn):
    games={}
    for date in ("20261018","20261019"):
        obj=get("scoreboard?dates="+date+"&limit=50")
        events=obj.get("events")
        if not isinstance(events,list):raise ValueError("ESPN scoreboard missing events")
        for e in events:
            contest=(e.get("competitions") or [None])[0] or {}
            sides=contest.get("competitors") or []
            if len(sides)!=2:raise ValueError("ESPN non-two-sided game")
            home=next((x for x in sides if x.get("homeAway")=="home"),None)
            away=next((x for x in sides if x.get("homeAway")=="away"),None)
            if not home or not away:raise ValueError("unknown home/away")
            h=club(home["team"]["abbreviation"]);a=club(away["team"]["abbreviation"])
            if not bool((e.get("status") or {}).get("type",{}).get("completed")):
                raise ValueError("NFL_WEEK6_GAME_NOT_FINAL_"+str(e.get("id")))
            if (a,h) not in EXPECTED:raise ValueError("unexpected 2026 week-6 team/date "+a+"/"+h)
            gid="2026_06_"+a+"_"+h
            kick=iso(e["date"])
            label="Monday" if date=="20261019" else "Sunday"
            if gid in games:raise ValueError("duplicate game in official scoreboard")
            games[gid]={"game_id":gid,"espn_event_id":str(e["id"]),
                "kickoff_utc":kick.isoformat(),"day":label,
                "away":a,"home":h,"away_score":int(away["score"]),"home_score":int(home["score"])}
    if len(games)!=13 or { (g["away"],g["home"]) for g in games.values()}!=EXPECTED:
        raise ValueError("INCOMPLETE_OR_MISMATCHED_OCT18_19_SCHEDULE")
    return games

def batch_file(state,game_id):
    return Path(state)/"forecasts"/"batches"/f"2026_wk06_T24_{game_id}.jsonl"

def read_verified_batch(path,game):
    raw=path.read_text()
    lines=raw.splitlines()
    if len(lines)<3:raise ValueError("truncated T24 archive "+str(path))
    head=json.loads(lines[0]);foot=json.loads(lines[-1])
    body="\n".join(lines[:-1])+"\n"
    expected=hashlib.sha256(body.encode()).hexdigest()
    if foot.get("_footer") is not True or foot.get("sha256")!=expected or foot.get("n")!=len(lines)-2:
        raise ValueError("T24 archive footer SHA or count failed "+str(path))
    if (head.get("game_id")!=game["game_id"] or head.get("horizon")!="T24" or
        head.get("n")!=len(lines)-2):
        raise ValueError("T24 archive identity/header mismatch")
    kick=iso(game["kickoff_utc"])
    gen=iso(head["generated_at"])
    if gen>=kick:raise ValueError("RECONSTRUCTED_AFTER_KICKOFF")
    rows=[]
    for line in lines[1:-1]:
        r=json.loads(line)
        if (r.get("game_id")!=game["game_id"] or r.get("horizon")!="T24"
            or r.get("outcome") not in OUTCOMES or r.get("team") not in (game["home"],game["away"])
            or r.get("opponent") not in (game["home"],game["away"])
            or r.get("team")==r.get("opponent")):
            raise ValueError("FORECAST_EVENT_IDENTITY_BAD")
        if iso(r["cutoff"])>=kick or iso(r["kickoff"])!=kick:
            raise ValueError("T24_POST_KICKOFF_OR_MISMATCHED_TIME")
        snapshot=r.get("input_snapshots") or {}
        if snapshot.get("time_travel") is True or iso(snapshot["retrieval_ts"])>iso(r["cutoff"]):
            raise ValueError("T24_FUTURE_DATA_LEAK")
        if snapshot.get("cutoff")!=r["cutoff"]:
            raise ValueError("T24_SOURCE_CUTOFF_DRIFT")
        if not r.get("id") or not r.get("player_name"):raise ValueError("UNIDENTIFIED_FORECAST_ROW")
        if not all(isinstance(r.get(k),(float,int)) and math.isfinite(r[k]) for k in ("mean","median")):
            raise ValueError("T24_NONNUMERIC_FORECAST")
        rows.append(r)
    if len(rows)!=len({r["id"] for r in rows}):raise ValueError("T24_DUPLICATE_FORECAST_ID")
    return rows,{"game_id":game["game_id"],"batch_generated_utc":gen.isoformat(),
      "cutoff_utc":min((r["cutoff"] for r in rows),default=None),
      "n":len(rows),"footer_sha256":expected,
      "model_versions":sorted({r["model_version"] for r in rows})}

def read_boxscore(payload,game):
    groups=(payload.get("boxscore") or {}).get("players")
    if not isinstance(groups,list) or len(groups)!=2:
        raise ValueError("MISSING_TWO_TEAM_REAL_BOXSCORE "+game["game_id"])
    observed={}
    by_team=collections.defaultdict(set)
    verified=set()
    for team in groups:
        t=club((team.get("team") or {}).get("abbreviation"))
        if t not in (game["home"],game["away"]) or t in verified:
            raise ValueError("REAL_ESPN_TEAM_DUPLICATE_OR_WRONG "+game["game_id"])
        verified.add(t)
        sections=team.get("statistics") or []
        if not isinstance(sections,list):raise ValueError("REAL_ESPN_STATS_NOT_ARRAY")
        for section in sections:
            category=section.get("name")
            if category not in ("passing","rushing","receiving","defensive","interceptions",
                                "kickReturns","puntReturns","kicking","punting","fumbles"):
                continue
            labels=section.get("labels") or []
            if len(labels)!=len(set(labels)):raise ValueError("ESPN_STAT_LABEL_DUPLICATION")
            for row in (section.get("athletes") or []):
                values=row.get("stats")
                if not isinstance(values,list) or not values:continue
                if len(values)!=len(labels):raise ValueError("ESPN_PLAYER_STAT_MISALIGNMENT")
                athlete=row.get("athlete") or {}
                pid=str(athlete.get("id") or "")
                display=athlete.get("displayName") or athlete.get("fullName")
                if not pid or not display:raise ValueError("ESPN_PLAYED_STAT_ID_MISSING")
                key=(t,namekey(display))
                if not key[1]:raise ValueError("EMPTY_PLAYER_NAME")
                prev=observed.setdefault(key,{"team":t,"player":display,
                    "espn_athlete_ids":set(),"groups":set(), "stats":{}})
                prev["espn_athlete_ids"].add(pid);prev["groups"].add(category)
                stat=dict(zip(labels,values))
                def v(label):
                    if label not in stat:return None
                    raw=str(stat[label]).strip()
                    if raw in ("--","-",""):return None
                    x=float(raw)
                    if not math.isfinite(x):raise ValueError("ESPN_NON_FINITE_PLAYER_STAT")
                    return x
                spec={
                   "passing":{"YDS":"pass_yds","TD":"pass_td","INT":"int"},
                   "rushing":{"YDS":"rush_yds","TD":"rush_td"},
                   "receiving":{"YDS":"rec_yds","REC":"rec","TD":"rec_td"},
                   "defensive":{"TOT":"tackles","SACKS":"sacks"},
                   "interceptions":{"INT":"def_int"}
                }.get(category,{})
                for label,field in spec.items():
                    z=v(label)
                    if z is not None:
                        if field in prev["stats"] and prev["stats"][field]!=z:
                            raise ValueError("CONFLICTED_INDEPENDENT_GAME_STAT "+field)
                        prev["stats"][field]=z
    if verified!={game["home"],game["away"]}:raise ValueError("BOXSCORE_TEAM_COVERAGE_INCOMPLETE")
    for record in observed.values():
        if len(record["espn_athlete_ids"])!=1:
            raise ValueError("AMBIGUOUS_PLAYER_SAME_NAME_DIFFERENT_IDS")
        record["espn_athlete_id"]=next(iter(record["espn_athlete_ids"]))
        del record["espn_athlete_ids"]
        record["groups"]=sorted(record["groups"])
        # A real roster participant with no rushing attempts has 0 RUSH yards
        # iff ESPN recorded them in passing/rushing/receiving group.
        offense=bool(set(record["groups"])&{"passing","rushing","receiving"})
        defense=bool(set(record["groups"])&{"defensive","interceptions"})
        record["offense_appeared"]=offense
        record["defense_appeared"]=defense
        if offense:
            for f in ("rush_yds","rec_yds","pass_yds","rec","pass_td","int","rush_td","rec_td"):
                record["stats"].setdefault(f,0.0)
            record["stats"]["atd"]=record["stats"]["rush_td"]+record["stats"]["rec_td"]
        if defense:
            for f in ("tackles","sacks","def_int"):
                record["stats"].setdefault(f,0.0)
    return observed

def alt65(r):
    oc=r["outcome"]
    if oc in YARDS:
        grid=r.get("quantile_grid_99")
        if not isinstance(grid,list) or len(grid)!=99:return None
        if not all(isinstance(x,(float,int)) and math.isfinite(x) for x in grid):return None
        if any(x<y for x,y in zip(grid[1:],grid[:-1])):return None
        start,step=(100,25) if oc=="pass_yds" else (10,10)
        best=None
        for m in range(start,int(math.floor(grid[34]/step)*step)+1,step):
            first=next((i for i,x in enumerate(grid) if x>=m),-1)
            probability=(99-first)/100 if first>=0 else 0
            if probability<.65:break
            best=(m,probability)
        return best
    lattice=r.get("cdf_lattice")
    step=.5 if oc=="sacks" else 1
    if not isinstance(lattice,dict) or lattice.get("step")!=step:return None
    cdf=lattice.get("cdf")
    if not isinstance(cdf,list) or not cdf:return None
    if not all(isinstance(x,(int,float)) and 0<=x<=1 and math.isfinite(x) for x in cdf):return None
    if any(a>b for a,b in zip(cdf,cdf[1:])):return None
    max_m=int(math.floor(len(cdf)*step))
    if max_m<=0:return None
    if 1-cdf[int(max_m/step)-1]>=.65:return None
    best=None
    for milestone in range(1,max_m+1):
        p=1-cdf[int(milestone/step)-1]
        if p<.65:break
        best=(milestone,p)
    return best

def evaluate(games,rows,box):
    scored=[];quarantine=[];alt=[]
    for r in rows:
        game=games[r["game_id"]]
        key=(club(r["team"]),namekey(r["player_name"]))
        rec=box[r["game_id"]].get(key)
        field=r["outcome"]
        mkt_off=field not in ("tackles","sacks","def_int")
        actual=None
        reason=None
        if rec is None:reason="PLAYER_NOT_IN_ESPN_COMPLETED_BOX"
        elif mkt_off and not rec["offense_appeared"]:reason="NO_VERIFIABLE_OFFENSIVE_PARTICIPATION"
        elif not mkt_off and not rec["defense_appeared"]:reason="NO_VERIFIABLE_DEFENSIVE_PARTICIPATION"
        else:actual=rec["stats"].get(field)
        if actual is None:
            quarantine.append({"game_id":r["game_id"],"day":game["day"],
             "player":r["player_name"],"team":r["team"],"market":field,
             "reason":reason or "STAT_MISSING","forecast_id":r["id"]})
            continue
        result={"game_id":r["game_id"],"day":game["day"],
            "team":r["team"],"opponent":r["opponent"],"player":r["player_name"],
            "market":field,"forecast_id":r["id"],
            "model_version":r["model_version"],"cutoff_utc":r["cutoff"],
            "pred_mean":round(float(r["mean"]),4),
            "pred_median":round(float(r["median"]),4),
            "p_active":round(float(r["p_active"]),4),
            "actual":actual,
            "abs_err_mean":round(abs(float(r["mean"])-actual),4),
            "abs_err_median":round(abs(float(r["median"])-actual),4)}
        scored.append(result)
        proposed=alt65(r)
        if proposed is not None:
            target,prob=proposed
            alt.append({**result,"threshold":target,"model_prob":round(prob,4),
               "hit":bool(actual>=target),
               "note":"MODEL_DERIVED_65P_MILESTONE_NOT_VERIFIED_SPORTSBOOK_LINE"})
    return scored,quarantine,alt

def groups_summary(scored,quarantine,alt):
    result={}
    for day in ("Sunday","Monday","combined"):
        subset=[r for r in scored if day=="combined" or r["day"]==day]
        missing=[r for r in quarantine if day=="combined" or r["day"]==day]
        altrows=[r for r in alt if day=="combined" or r["day"]==day]
        breakdown={}
        for market in OUTCOMES:
            rr=[r for r in subset if r["market"]==market]
            invalid=sum(r["market"]==market for r in missing)
            aa=[r for r in altrows if r["market"]==market]
            breakdown[market]={
              "n_scored":len(rr),"n_missing_independent_player_truth":invalid,
              "mae_pred_mean":round(statistics.mean(x["abs_err_mean"] for x in rr),3) if rr else None,
              "mae_pred_median":round(statistics.mean(x["abs_err_median"] for x in rr),3) if rr else None,
              "mae_trivial_zero":round(statistics.mean(abs(x["actual"]) for x in rr),3) if rr else None,
              "n_model_65pct_alt_thresholds":len(aa),
              "model_65pct_alt_hits":sum(x["hit"] for x in aa),
              "model_65pct_alt_misses":sum(not x["hit"] for x in aa),
              "model_65pct_alt_empirical_rate":round(sum(x["hit"] for x in aa)/len(aa),4) if aa else None,
              "model_65pct_mean_implied_prob":round(statistics.mean(x["model_prob"] for x in aa),4) if aa else None,
            }
        result[day]={
          "n_scored_stat_forecasts":len(subset),
          "n_stat_forecasts_without_independent_truth":len(missing),
          "n_derived_65pct_milestones_scored":len(altrows),
          "milestone_hits":sum(x["hit"] for x in altrows),
          "milestone_misses":sum(not x["hit"] for x in altrows),
          "milestone_hit_rate":round(sum(x["hit"] for x in altrows)/len(altrows),4) if altrows else None,
          "markets":breakdown}
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shadow-state",type=Path,default=Path("nfl-shadow-state"))
    p.add_argument("--out",type=Path,default=Path("docs/nfl_2026_w06_sun_mon_first_seen_audit.json"))
    p.add_argument("--graded-csv",type=Path,default=Path("docs/nfl_2026_w06_sun_mon_forecast_rows.csv"))
    a=p.parse_args()
    games=get_schedule()
    raw_forecasts=[];receipts=[];boxes={}
    for gid,game in sorted(games.items()):
        path=batch_file(a.shadow_state,gid)
        if not path.is_file():
            raise ValueError("MISSING_IMMUTABLE_T24_GAME_BATCH "+str(path))
        f,rec=read_verified_batch(path,game)
        raw_forecasts.extend(f);receipts.append(rec)
        final=read_espn("summary?event="+game["espn_event_id"])
        boxes[gid]=read_boxscore(final,game)
        print("NFL_W06_GAME",json.dumps({"game_id":gid,"day":game["day"],
          "source_rows":len(f),"matched_box_players":len(boxes[gid]),
          "actual_final":f'{game["away"]} {game["away_score"]} - {game["home"]} {game["home_score"]}',
          "first_frozen_at":rec["batch_generated_utc"]}),flush=True)
    if len(raw_forecasts)<3900:raise ValueError("NFL_WEEK6_MISSING_PREGAME_T24_ROWS")
    scored,quarantine,alt=evaluate(games,raw_forecasts,boxes)
    if len(scored)<1700:raise ValueError("NFL_W06_UNACCEPTABLE_PLAYER_MATCH_COVERAGE")
    result={"schema":SCHEMA,"period":"2026-10-18 and 2026-10-19 Eastern",
      "source_policy":"IMMUTABLE_T24_PRE_KICKOFF_SHADOW_BATCHES_MATCHED_TO_ESPN_FINAL_BOXES",
      "shadow_unpromoted_research_only":True,
      "n_games":len(games),"n_sunday_games":12,"n_monday_games":1,
      "source_batches_sha_receipts":receipts,
      "actual_games":list(games.values()),
      "original_t24_stat_forecast_rows":len(raw_forecasts),
      "t90_only_thursday_no_sun_mon_complete_evidence":True,
      "last_pre_game_live_book_lines_verified_for_model_milestones":0,
      "model_milestones_not_sportsbook_bets":True,
      "real_betting_wins":None,"real_betting_losses":None,
      "promotion_status":"RESEARCH_ONLY_NO_FIRST_SEEN_REAL_PRICED_BET_PERFORMANCE",
      "summary":groups_summary(scored,quarantine,alt),
      "scored_player_forecast_rows":scored,
      "ungraded_identity_and_market_rows":quarantine,
      "derived_model_65pct_thresholds":alt}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    import csv
    if scored:
        a.graded_csv.parent.mkdir(parents=True,exist_ok=True)
        with a.graded_csv.open("w",newline="") as dest:
            writer=csv.DictWriter(dest,fieldnames=list(scored[0]))
            writer.writeheader()
            writer.writerows(scored)
    print("NFL_W06_COMPLETED_BACKTEST",json.dumps({
      "n_games":len(games),"sources":len(raw_forecasts),
      "scored":len(scored),"ungraded":len(quarantine),
      "Sunday":result["summary"]["Sunday"],
      "Monday":result["summary"]["Monday"],
      "combined":result["summary"]["combined"],
      "no_real_book_bet_pnl":True},sort_keys=True),flush=True)
if __name__=="__main__":main()
