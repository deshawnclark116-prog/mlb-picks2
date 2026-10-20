#!/usr/bin/env python3
"""Strict read-only grade of original NFL Phase1E player STAT projections.

No sportsbook lines. No hindsight backfills. Uses verified NFL shadow
forecast batches and independent final nflverse player_games/schedules.
An unknown player-game actual is NOT zero. Results separated T24 and T90.
"""
import argparse
import json
import math
import re
import sqlite3
import statistics
from collections import defaultdict,Counter
from datetime import datetime,timezone,timedelta
from pathlib import Path

STATS={'rush_yds':'rushing_yards','rec_yds':'receiving_yards',
       'rec':'receptions','pass_yds':'passing_yards','pass_td':'passing_tds',
       'int':'interceptions','rush_td':'rushing_tds','rec_td':'receiving_tds',
       'sacks':'def_sacks',
       'def_int':'def_interceptions'}
SCHEMA='NFL_V2_FIRST_SEEN_REAL_FINAL_NUMERIC_SCORECARD_V1'


def utc(s):
    v=datetime.fromisoformat(str(s).replace('Z','+00:00'))
    if v.tzinfo is None:raise ValueError('timezone missing')
    return v.astimezone(timezone.utc)


def team(s):
    return {'LAR':'LA','WSH':'WAS','JAC':'JAX','OAK':'LV'}.get(s,s)


def actual(row,outcome):
    if outcome=='atd':
        a,b=row.get('rushing_tds'),row.get('receiving_tds')
        return float(a>0 or b>0) if a is not None and b is not None else None
    v=row.get(STATS.get(outcome,''))
    try:return float(v) if v is not None and math.isfinite(float(v)) else None
    except (TypeError,ValueError):return None


def summarize(rows):
    n=len(rows)
    if not n:return {'n':0}
    errors=[abs(r['median']-r['actual']) for r in rows]
    paired=[r for r in rows if r['last3'] is not None]
    with_bounds=[r for r in rows if r['p10'] is not None and r['p90'] is not None]
    return {'n':n,'median_mae':round(statistics.mean(errors),2),
            'mean_mae':round(statistics.mean(abs(r['mean']-r['actual']) for r in rows),2),
            'signed_bias_median_minus_actual':round(statistics.mean(
                r['median']-r['actual'] for r in rows),2),
            'median_absolute_error':round(statistics.median(errors),2),
            'within_10_pct':round(100*sum(e<=10 for e in errors)/n,1),
            'within_20_pct':round(100*sum(e<=20 for e in errors)/n,1),
            'p10_p90_coverage_pct':round(100*sum(
                r['p10']<=r['actual']<=r['p90'] for r in with_bounds)/len(with_bounds),1)
                if with_bounds else None,
            'paired_last3_n':len(paired),
            'paired_model_mae':round(statistics.mean(
                abs(r['median']-r['actual']) for r in paired),2) if paired else None,
            'paired_last3_mae':round(statistics.mean(
                abs(r['last3']-r['actual']) for r in paired),2) if paired else None}


def load_truth(db):
    con=sqlite3.connect('file:'+str(Path(db).resolve())+'?mode=ro',uri=True)
    con.row_factory=sqlite3.Row
    try:
        games={r['game_id']:dict(r) for r in con.execute(
            "SELECT game_id,season,week,season_type,kickoff_utc,home_team,away_team,"
            "home_score,away_score FROM games WHERE season=2026")}
        cols={r['name'] for r in con.execute('PRAGMA table_info(player_games)')}
        needed=set(STATS.values())|{'rushing_tds','receiving_tds'}
        if not needed <=cols:raise ValueError('missing official stat columns: '+repr(needed-cols))
        stats=defaultdict(list);history=defaultdict(list)
        for row in con.execute("SELECT * FROM player_games WHERE season IN (2025,2026)"):
            r=dict(row)
            stats[(r['season'],r['week'],team(r['team']),str(r['player_id']))].append(r)
            history[str(r['player_id'])].append(r)
        for rows in history.values():rows.sort(key=lambda r:(r['season'],r['week']))
    finally:con.close()
    return games,stats,history


def last3(history,pid,season,week,oc):
    candidates=[actual(r,oc) for r in history.get(pid,[]) if
                (r['season'],r['week'])<(season,week) and r.get('season_type')=='REG']
    values=[x for x in candidates if x is not None]
    return statistics.mean(values[-3:]) if len(values)>=3 else None


def pregame_timing_tier(header,r):
    """Classify immutable, SHA-verified source ROWS without inventing T24 cutoffs.

    The original snapshot input must have been retrieved by cutoff. A real
    batch generated after the 24-hour / 90-minute horizon deadline but before
    kickoff belongs to a SEPARATE late-pregame accuracy cohort, NEVER the
    on-time T24/T90 headline.
    """
    try:
        from datetime import timedelta
        if header.get("adhoc_label") or header.get("game_id")!=r["game_id"]:
            return None
        if header.get("horizon")!=r["horizon"] or r["horizon"] not in ("T24","T90"):
            return None
        kick,cut,gen=(utc(r["kickoff"]),utc(r["cutoff"]),utc(header["generated_at"]))
        snap=r["input_snapshots"]
        retrieved=utc(snap["retrieval_ts"])
        delta=timedelta(hours=24) if r["horizon"]=="T24" else timedelta(minutes=90)
        if (cut!=kick-delta or not retrieved<=cut or not gen<kick or
            snap.get("cutoff")!=r["cutoff"] or snap.get("time_travel")):
            return None
        return "ON_CUTOFF" if gen<=cut else "LATE_PREGAME"
    except (KeyError,TypeError,ValueError):
        return None


def load_original(state,weeks):
    """Strict published frozen T24/T90 + separate authentic late pregame files."""
    from nfl_phase1_publisher import genuine_records,ledger_last_state,read_batch_verified
    from nfl_shadow_publish_evidence import Evidence
    root=Path(state);last=ledger_last_state(root);ev=Evidence(root)
    certified={'T24':{},'T90':{}}
    for w in weeks:
        batches=genuine_records(root,last,(2026,w),evidence=ev)
        for h in certified:certified[h].update(batches[h])
    out={'T24':{},'T90':{},'LATE_T24':{},'LATE_T90':{}}
    approved={r["id"] for games in certified.values() for rows in games.values() for r in rows}
    seen={}
    # The previously minted saved batch itself is what is being graded;
    # cannot substitute a newly simulated forecast. Original content is
    # validated with the sha256 footer by read_batch_verified.
    for path in sorted((root/'forecasts/batches').glob('2026_wk*.jsonl')):
        result=read_batch_verified(path)
        if result is None:continue
        header,records=result
        hz=header.get("horizon")
        if hz not in ("T24","T90") or header.get("adhoc_label"):continue
        gid=header.get("game_id")
        if not isinstance(gid,str) or not re.fullmatch(r"2026_\d{2}_[A-Z]+_[A-Z]+",gid):
            continue
        week=int(gid.split("_")[1])
        if week not in weeks:continue
        filename=f"2026_wk{week:02d}_{hz}_{gid}.jsonl"
        if path.name!=filename or header.get("n")!=len(records):continue
        for r in records:
            if (r.get("game_id")!=gid or r.get("horizon")!=hz or
                r.get("season")!=2026 or r.get("week")!=week or
                not isinstance(r.get("id"),str) or not r["id"]):
                continue
            tier=pregame_timing_tier(header,r)
            if tier is None:continue
            identity=json.dumps(r,sort_keys=True,allow_nan=False)
            if r["id"] in seen:
                if seen[r["id"]]!=identity:
                    raise ValueError("Conflicting immutable original forecast ID across batches")
                continue  # Same source bytes replayed, not two distinct predictions.
            seen[r["id"]]=identity
            if tier=="ON_CUTOFF" and r["id"] in approved:
                selection=hz
                provenance="FULL_ORIGINAL_PUBLISHER_EVIDENCE_CERTIFIED"
            elif tier=="LATE_PREGAME":
                selection="LATE_"+hz
                provenance="HASH_VERIFIED_PRE_KICKOFF_BUT_GENERATED_AFTER_NAMED_HORIZON_CUTOFF"
            else:
                # A source can have a SHA-valid file while failing the
                # original publisher's source/provenance certification.
                # Do NOT silently promote it to an authentic T24/T90 grade.
                continue
            item={**r,"grading_provenance":provenance,
                  "original_horizon":hz,"graded_as_horizon":selection}
            out[selection].setdefault(gid,[]).append(item)
    return out

def available_weeks(state):
    """Read weeks from original verified-batch filename shape, not a stale hardcode."""
    folder=Path(state)/"forecasts"/"batches"
    matches=set()
    for path in folder.glob("2026_wk*.jsonl"):
        m=re.fullmatch(r"2026_wk(\d{2})_(?:T24|T90)_2026_\d{2}_[A-Z]+_[A-Z]+\.jsonl",path.name)
        if m:
            w=int(m.group(1))
            if 1<=w<=18:matches.add(w)
    return sorted(matches)


def evaluate(forecasts,games,stats,history,now,espn=None):
    """Account for every genuine original T24/T90 forecast, scored or unresolved."""
    from nfl_week6_first_seen_grade import namekey
    rows=[];ungraded=[];excluded=Counter();completed=set();planned=set()
    egames=(espn or {}).get("games") or {}
    espn_lookup={}
    for gid,game in egames.items():
        if gid!=game.get("game_id") or game.get("source")!="ESPN_FINAL_GAME_BOXSCORE":
            raise ValueError("Unsupported/unverified ESPN final-game player source")
        seen=set()
        for p in game.get("players") or []:
            key=(team(p["team"]),p["name_key"])
            if key in seen:raise ValueError("Duplicate ESPN player/team in final game")
            seen.add(key)
            espn_lookup[(gid,*key)]=p

    def report_ungraded(reason,horizon,gid,r):
        excluded[reason]+=1
        ungraded.append({
            "forecast_id":r.get("id"),"game_id":gid,"horizon":horizon,
            "original_horizon":r.get("original_horizon",horizon),
            "grading_provenance":r.get("grading_provenance"),
            "week":r.get("week"),"market":r.get("outcome"),
            "player":r.get("player_name"),"player_id":r.get("player_id"),
            "team":r.get("team"),"opponent":r.get("opponent"),
            "cutoff_utc":r.get("cutoff"),"kickoff_utc":r.get("kickoff"),
            "original_predicted_mean":r.get("mean"),
            "original_predicted_median":r.get("median"),
            "original_p_active":r.get("p_active"),
            "actual":None,"status":"UNVERIFIED","reason":reason})

    for horizon,gmap in forecasts.items():
        for gid,records in gmap.items():
            g=games.get(gid)
            eg=egames.get(gid)
            if not g and eg:
                g={"game_id":gid,"season":eg["season"],"week":eg["week"],
                   "season_type":"REG","kickoff_utc":eg["kickoff_utc"],
                   "home_team":eg["home"],"away_team":eg["away"],
                   "home_score":eg["home_score"],"away_score":eg["away_score"]}
            if not g:
                for r in records:report_ungraded("NO_OFFICIAL_GAME_ROW",horizon,gid,r)
                continue
            isfinal=(g.get("home_score") is not None and g.get("away_score") is not None) or eg is not None
            if (not isfinal or not g.get("kickoff_utc") or
                utc(g["kickoff_utc"])+timedelta(hours=2)>now):
                for r in records:report_ungraded("NOT_FINAL_OR_NOT_INGESTED",horizon,gid,r)
                continue
            completed.add(gid);planned.add((gid,horizon))
            for r in records:
                oc=r.get("outcome")
                if oc not in STATS and oc not in ("atd","tackles"):
                    report_ungraded("UNSUPPORTED_OUTCOME",horizon,gid,r);continue
                try:
                    kick=utc(r["kickoff"]);cut=utc(r["cutoff"])
                    if not cut<kick or abs((kick-utc(g["kickoff_utc"])).total_seconds())>1200:
                        raise ValueError("kickoff mismatch")
                except (KeyError,TypeError,ValueError):
                    report_ungraded("INVALID_TIMESTAMP_OR_KICKOFF",horizon,gid,r);continue
                if g["season_type"]!="REG" or team(r["team"]) not in (
                        team(g["home_team"]),team(g["away_team"])):
                    report_ungraded("TEAM_OR_STAGE_MISMATCH",horizon,gid,r);continue
                pid=str(r.get("player_id"))
                matched=stats.get((g["season"],g["week"],team(r["team"]),pid),[])
                if len(matched)>1:
                    report_ungraded("AMBIGUOUS_PLAYER_ACTUAL",horizon,gid,r);continue
                nflverse=matched[0] if matched else None
                if nflverse and nflverse.get("game_id") not in (gid,None):
                    report_ungraded("ACTUAL_WRONG_GAME",horizon,gid,r);continue
                ep=espn_lookup.get((gid,team(r["team"]),namekey(r.get("player_name"))))
                ev=None
                if ep:
                    offensive=oc not in ("tackles","sacks","def_int")
                    if (offensive and ep.get("offense_appeared")) or (
                            not offensive and ep.get("defense_appeared")):
                        if oc=="atd":
                            a,b=ep["stats"].get("rush_td"),ep["stats"].get("rec_td")
                            ev=float(a>0 or b>0) if a is not None and b is not None else None
                        else:
                            z=ep["stats"].get(oc)
                            ev=float(z) if z is not None else None
                nv=actual(nflverse,oc) if nflverse and oc!="tackles" else None
                if ev is not None and nv is not None and abs(ev-nv)>1e-8:
                    report_ungraded("CONFLICTING_INDEPENDENT_FINAL_STAT_SOURCES",horizon,gid,r)
                    continue
                val=ev if ev is not None else nv
                if val is None:
                    reason=("TACKLES_TOTAL_ACTUAL_NOT_AVAILABLE_IN_THIS_SOURCE" if oc=="tackles" else
                            "NO_UNIQUE_PLAYER_ACTUAL_NOT_ZERO" if not nflverse and not ep else
                            "NO_STAT_AVAILABLE_FOR_MARKET")
                    report_ungraded(reason,horizon,gid,r);continue
                try:
                    median=float(r["median"]);mean=float(r["mean"])
                    if not math.isfinite(median) or not math.isfinite(mean):
                        raise ValueError()
                except (KeyError,TypeError,ValueError):
                    report_ungraded("NO_NUMERIC_MODEL_STAT",horizon,gid,r);continue
                q=r.get("quantiles") or {}
                obs={"forecast_id":r.get("id"),"game_id":gid,"horizon":horizon,
                     "original_horizon":r.get("original_horizon",horizon),
                     "grading_provenance":r.get("grading_provenance"),
                     "week":g["week"],"market":oc,"player":r["player_name"],"team":r["team"],
                     "player_id":pid,"position":r.get("position"),
                     "median":median,"mean":mean,"actual":val,
                     "truth_source":"ESPN_FINAL_BOX" if ev is not None else "NFLVERSE_PLAYER_GAMES",
                     "cutoff_utc":r["cutoff"],"kickoff_utc":r["kickoff"],
                     "error":round(median-val,2),"p_active":r.get("p_active"),
                     "p10":q.get("p10"),"p90":q.get("p90"),
                     "last3":last3(history,pid,g["season"],g["week"],oc)}
                rows.append(obs)

    # A model observed at two horizons is the SAME player/game. Pair
    # them exactly rather than comparing differently selected cohorts.
    earlier={(r['game_id'],r['player_id'],r['market']):r
             for r in rows if r['horizon']=='T90'}
    pair=[]
    for r in rows:
        if r['horizon']!='T24':continue
        past=earlier.get((r['game_id'],r['player_id'],r['market']))
        if past is None:continue
        pair.append({
            'market':r['market'],
            't24_absolute_error':abs(r['median']-r['actual']),
            't90_absolute_error':abs(past['median']-r['actual']),
            'absolute_forecast_change':abs(r['median']-past['median'])
        })
    def paired_metrics(sample):
        if not sample:return {'n':0}
        return {
            'n':len(sample),
            't24_mae':round(statistics.mean(x['t24_absolute_error'] for x in sample),2),
            't90_mae_on_same_players':round(statistics.mean(x['t90_absolute_error'] for x in sample),2),
            't24_improvement_positive_is_better':round(statistics.mean(
                x['t90_absolute_error']-x['t24_absolute_error'] for x in sample),2),
            'unchanged_both_horizons_within_001':sum(
                x['absolute_forecast_change']<.001 for x in sample)
        }
    paired={'all_markets':paired_metrics(pair),'by_market':{
        m:paired_metrics([x for x in pair if x['market']==m])
        for m in sorted(set(x['market'] for x in pair))}}
    # Every first-seen original forecast must appear exactly once either
    # graded against a real outcome or explicitly unverified, per game.
    accounting=defaultdict(list)
    for r in rows:
        accounting[(r["game_id"],r["horizon"])].append(("GRADED",r))
    for r in ungraded:
        accounting[(r["game_id"],r["horizon"])].append((r["reason"],r))
    game_rows=[]
    for (gid,h),items in sorted(accounting.items()):
        markets={}
        for m in sorted(set(x["market"] for _,x in items)):
            mm=[(status,x) for status,x in items if x["market"]==m]
            graded_market=[x for status,x in mm if status=="GRADED"]
            markets[m]={
                "original_forecasts":len(mm),
                "graded":len(graded_market),
                "unverified":len(mm)-len(graded_market),
                "unverified_reasons":dict(sorted(Counter(status for status,_ in mm
                    if status!="GRADED").items())),
                "median_mae":summarize(graded_market).get("median_mae"),
                "mean_mae":summarize(graded_market).get("mean_mae")}
        game_rows.append({
            "game_id":gid,"horizon":h,
            "original_horizon":items[0][1].get("original_horizon",h),
            "grading_provenance":items[0][1].get("grading_provenance"),
            "week":items[0][1].get("week"),
            "original_forecasts":len(items),
            "graded":sum(status=="GRADED" for status,_ in items),
            "unverified":sum(status!="GRADED" for status,_ in items),
            "markets":markets})
    if sum(x["original_forecasts"] for x in game_rows)!=len(rows)+len(ungraded):
        raise AssertionError("All-original-prediction row accounting failed")
    return {
        'total_original_pregame_forecasts':len(rows)+len(ungraded),
        'late_horizon_but_before_kickoff_forecasts':sum(
            r['horizon'].startswith('LATE_') for r in rows+ungraded),
        'late_forecasts_are_NOT_valid_original_T24_T90_cutoff':True,
        'unverified_original_forecasts':len(ungraded),
        'by_game_horizon':game_rows,
        'all_unverified_rows':ungraded,
        'every_genuine_original_forecast_accounted':True,
        'schema':SCHEMA,'asof_utc':now.isoformat(),'first_seen_pre_cutoff_only':True,
        'source_contract_warning':'nflverse assisted tackle field is NOT total tackles; tackles quarantined when independent ESPN TOT unavailable',
        'paired_t24_vs_t90_same_player':paired,
        'sources':'Original SHA-verified stored pregame source, certified T24/T90 only when all original publisher evidence passed; separate late horizon source + independently verified final player stats',
        'grade_type':'Actual player-stat median/mean vs actual, never sportsbook alternate line',
        'research_only':True,'automatic_model_promotion':False,
        'completed_games_with_verified_forecasts':len(completed),
        'completed_game_horizon_pairs':len(planned),
        'graded_player_outcomes':len(rows),'exclusions':dict(sorted(excluded.items())),
        'by_horizon':{h:summarize([r for r in rows if r['horizon']==h])
                      for h in ('T24','T90','LATE_T24','LATE_T90')},
        'by_market_horizon':{h:{m:summarize([r for r in rows
                                         if r['horizon']==h and r['market']==m])
                                for m in sorted(set(r['market'] for r in rows))}
                             for h in ('T24','T90','LATE_T24','LATE_T90')},
        'by_week':{str(w):summarize([r for r in rows if r['week']==w])
                   for w in sorted(set(r['week'] for r in rows))},
        'high_activity_075':{h:summarize([r for r in rows if r['horizon']==h and
                                          r['p_active'] is not None and r['p_active']>=0.75])
                             for h in ('T24','T90')},
        'largest_misses':sorted(rows,key=lambda r:abs(r['error']),reverse=True)[:35],
        'all_graded_rows':rows}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--state',type=Path,required=True)
    ap.add_argument('--db',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--weeks',nargs='+',type=int)
    ap.add_argument('--espn-actuals',type=Path)
    args=ap.parse_args()
    weeks=args.weeks if args.weeks is not None else available_weeks(args.state)
    if not weeks:raise SystemExit('No genuine frozen 2026 NFL weeks available to grade')
    fc=load_original(args.state,weeks)
    games,stats,history=load_truth(args.db)
    espn=json.loads(args.espn_actuals.read_text()) if args.espn_actuals else None
    if espn and espn.get('schema')!='NFL_RECENT_FINAL_ESPN_PLAYER_TRUTH_V1':
        raise ValueError('Unverified recent ESPN truth input')
    result=evaluate(fc,games,stats,history,datetime.now(timezone.utc),espn=espn)
    result['weeks_requested']=weeks
    result['schedule_finals_in_database']=sum(1 for g in games.values()
                       if g.get('home_score') is not None and g.get('away_score') is not None)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print('NFL_V2_REAL_PLAYER_STAT_AUDIT',json.dumps({k:result[k] for k in
          ('completed_games_with_verified_forecasts','completed_game_horizon_pairs',
           'graded_player_outcomes','by_horizon','by_week','exclusions')},sort_keys=True))
    if not result['graded_player_outcomes']:raise SystemExit('No verifiable NFL V2 numeric scorecard: fail closed')


if __name__=='__main__':main()
