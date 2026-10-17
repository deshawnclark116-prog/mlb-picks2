#!/usr/bin/env python3
"""Strict read-only grade of original NFL Phase1E player STAT projections.

No sportsbook lines. No hindsight backfills. Uses verified NFL shadow
forecast batches and independent final nflverse player_games/schedules.
An unknown player-game actual is NOT zero. Results separated T24 and T90.
"""
import argparse
import json
import math
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


def load_original(state,weeks):
    from nfl_phase1_publisher import genuine_records,ledger_last_state,read_batch_verified
    from nfl_shadow_publish_evidence import Evidence
    root=Path(state);last=ledger_last_state(root);ev=Evidence(root)
    out={'T24':{},'T90':{}}
    for w in weeks:
        batches=genuine_records(root,last,(2026,w),evidence=ev)
        for h in out:out[h].update(batches[h])
    # Independent extra safeguard: not only retrieved by cutoff, the
    # actual GENERATED forecast batch must be timestamped by cutoff.
    legal=set()
    for file in (root/'forecasts/batches').glob('2026_wk*.jsonl'):
        result=read_batch_verified(file)
        if result is None:continue
        header,rows=result
        if header.get('adhoc_label') or not header.get('generated_at'):continue
        for r in rows:
            try:
                if utc(header['generated_at'])<=utc(r['cutoff']):
                    legal.add(r['id'])
            except (KeyError,TypeError,ValueError):pass
    return {h:{gid:[r for r in rows if r['id'] in legal]
               for gid,rows in games.items() if any(r['id'] in legal for r in rows)}
            for h,games in out.items()}


def evaluate(forecasts,games,stats,history,now):
    rows=[];excluded=Counter();completed=set();planned=set()
    for horizon,gmap in forecasts.items():
        for gid,records in gmap.items():
            g=games.get(gid)
            if not g:
                excluded['NO_OFFICIAL_GAME_ROW']+=len(records);continue
            if g.get('home_score') is None or g.get('away_score') is None or not g.get('kickoff_utc') or (
                    utc(g['kickoff_utc'])+timedelta(hours=2)>now):
                excluded['NOT_FINAL_OR_NOT_INGESTED']+=len(records);continue
            completed.add(gid);planned.add((gid,horizon))
            for r in records:
                oc=r.get('outcome')
                if oc=='tackles':
                    # nflverse def_tackles_with_assist means 'tackles this
                    # player made with an assist', NOT total/combined tackles.
                    # Do not pretend it measures the modeled tackles count.
                    excluded['TACKLES_TOTAL_ACTUAL_NOT_AVAILABLE_IN_THIS_SOURCE']+=1
                    continue
                if oc not in STATS and oc!='atd':
                    excluded['UNSUPPORTED_OUTCOME']+=1;continue
                try:
                    kick=utc(r['kickoff']);cut=utc(r['cutoff'])
                    if not cut<kick or abs((kick-utc(g['kickoff_utc'])).total_seconds())>1200:
                        raise ValueError('kickoff mismatch')
                except (KeyError,TypeError,ValueError):
                    excluded['INVALID_TIMESTAMP_OR_KICKOFF']+=1;continue
                if g['season_type']!='REG' or team(r['team']) not in (
                        team(g['home_team']),team(g['away_team'])):
                    excluded['TEAM_OR_STAGE_MISMATCH']+=1;continue
                pid=str(r.get('player_id'))
                found=stats.get((g['season'],g['week'],team(r['team']),pid),[])
                if len(found)!=1:
                    excluded['NO_UNIQUE_PLAYER_ACTUAL_NOT_ZERO']+=1;continue
                val=actual(found[0],oc)
                if val is None:
                    excluded['NO_STAT_AVAILABLE_FOR_MARKET']+=1;continue
                try:
                    median=float(r['median']);mean=float(r['mean'])
                    if not math.isfinite(median) or not math.isfinite(mean):raise ValueError()
                except (KeyError,TypeError,ValueError):
                    excluded['NO_NUMERIC_MODEL_STAT']+=1;continue
                q=r.get('quantiles') or {}
                obs={'game_id':gid,'horizon':horizon,'week':g['week'],
                     'market':oc,'player':r['player_name'],'team':r['team'],
                     'player_id':pid,'position':r.get('position'),
                     'median':median,'mean':mean,'actual':val,
                     'cutoff_utc':r['cutoff'],'kickoff_utc':r['kickoff'],
                     'error':round(median-val,2),'p_active':r.get('p_active'),
                     'p10':q.get('p10'),'p90':q.get('p90'),
                     'last3':last3(history,pid,g['season'],g['week'],oc)}
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
    return {
        'schema':SCHEMA,'asof_utc':now.isoformat(),'first_seen_pre_cutoff_only':True,
        'source_contract_warning':'nflverse def_tackles_with_assist is not total tackles; tackle market quarantined',
        'paired_t24_vs_t90_same_player':paired,
        'sources':'Original NFL Phase1E verified first-seen state + NFLverse official final stats',
        'grade_type':'Actual player-stat median/mean vs actual, never sportsbook alternate line',
        'research_only':True,'automatic_model_promotion':False,
        'completed_games_with_verified_forecasts':len(completed),
        'completed_game_horizon_pairs':len(planned),
        'graded_player_outcomes':len(rows),'exclusions':dict(sorted(excluded.items())),
        'by_horizon':{h:summarize([r for r in rows if r['horizon']==h])
                      for h in ('T24','T90')},
        'by_market_horizon':{h:{m:summarize([r for r in rows
                                         if r['horizon']==h and r['market']==m])
                                for m in sorted(set(r['market'] for r in rows))}
                             for h in ('T24','T90')},
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
    ap.add_argument('--weeks',nargs='+',type=int,default=[4,5,6])
    args=ap.parse_args()
    fc=load_original(args.state,args.weeks)
    games,stats,history=load_truth(args.db)
    result=evaluate(fc,games,stats,history,datetime.now(timezone.utc))
    result['weeks_requested']=args.weeks
    result['schedule_finals_in_database']=sum(1 for g in games.values()
                       if g.get('home_score') is not None and g.get('away_score') is not None)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print('NFL_V2_REAL_PLAYER_STAT_AUDIT',json.dumps({k:result[k] for k in
          ('completed_games_with_verified_forecasts','completed_game_horizon_pairs',
           'graded_player_outcomes','by_horizon','by_week','exclusions')},sort_keys=True))
    if not result['graded_player_outcomes']:raise SystemExit('No verifiable NFL V2 numeric scorecard: fail closed')


if __name__=='__main__':main()
